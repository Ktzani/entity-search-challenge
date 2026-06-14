"""
01_extract_features_v3.py - Versao v3: features ricas + negativos balanceados.

Mudancas vs v1/v2:
1. Negativos = 50 hard (top-50 BM25) + 50 random (corpus inteiro)
   - Random negatives sao quase certamente irrelevantes
   - Hard negatives ainda ensinam o modelo a distinguir candidatos plausiveis
2. Features novas (de 5 -> 11):
   - f_title_exact_match (binario)
   - f_title_token_overlap (% termos no titulo)
   - f_keyword_match_count (# keywords do doc que casam termos)
   - f_doc_length (tokens no doc)
   - f_bm25_rank (posicao no ranking BM25)
   - f_bm25_norm (score BM25 normalizado pelo top-1 da query)
3. Le conteudo do doc via searcher.doc(docid).raw() para extrair
   title/text/keywords originais

Uso:
    python submissions/s5_ltr/01_extract_features_v3.py \\
        -q data/kaggle/train_queries.csv \\
        --qrels data/kaggle/train_qrels.csv \\
        --i-bm25 data/indexes/pyserini_bm25/ \\
        --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \\
        -o submissions/s5_ltr/train_features_v3.csv
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

K_HARD_NEG = 50  # top-K do BM25 como hard negatives
K_RAND_NEG = 50  # random negatives
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai features ricas v3 para LTR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-q", "--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--i-bm25", required=True)
    parser.add_argument("--i-fw", required=True)
    parser.add_argument("-o", "--output", required=True)
    return parser.parse_args()


def read_qrels(path: str) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            qid, docid, rel = row[0].strip(), row[1].strip(), int(row[2])
            qrels[qid][docid] = rel
    return dict(qrels)


def read_queries(path: str) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            qid, qtext = row[0].strip(), row[1].strip()
            if qid and qtext:
                queries.append((qid, qtext))
    return queries


def mean_df(index_reader: LuceneIndexReader, query: str) -> float:
    try:
        analyzed = index_reader.analyze(query)
    except Exception:
        return 0.0
    if not analyzed:
        return 0.0
    total = 0
    n = 0
    for term in analyzed:
        try:
            df, _ = index_reader.get_term_counts(term, analyzer=None)
            total += df
            n += 1
        except Exception:
            continue
    return total / n if n > 0 else 0.0


def get_doc_fields(searcher: LuceneSearcher, docid: str) -> dict:
    """Retorna {title, text, keywords_str} do doc, ou vazio se nao encontrar."""
    try:
        doc = searcher.doc(docid)
        if doc is None:
            return {"title": "", "text": "", "keywords": ""}
        raw = doc.raw()
        if not raw:
            return {"title": "", "text": "", "keywords": ""}
        parsed = json.loads(raw)
        # Pode estar no formato Pyserini (id+contents) ou no formato original
        title = parsed.get("title", "")
        text = parsed.get("text", "")
        keywords = parsed.get("keywords", [])
        if isinstance(keywords, list):
            keywords_str = " ".join(str(k) for k in keywords)
        else:
            keywords_str = str(keywords)
        # Se nao tiver title/text mas tiver "contents" (formato Pyserini), usa
        if not title and not text and "contents" in parsed:
            contents = parsed["contents"]
            return {"title": "", "text": contents, "keywords": ""}
        return {"title": title, "text": text, "keywords": keywords_str}
    except Exception:
        return {"title": "", "text": "", "keywords": ""}


def title_exact_match(query: str, title: str) -> int:
    """1 se algum termo da query bate exato (case-insensitive) no titulo."""
    if not title:
        return 0
    title_lower = title.lower()
    q_tokens = query.lower().split()
    return int(any(t in title_lower for t in q_tokens))


def title_token_overlap(query: str, title: str) -> float:
    """% de tokens da query que aparecem no titulo (case-insensitive)."""
    if not title:
        return 0.0
    title_tokens = set(title.lower().split())
    q_tokens = query.lower().split()
    if not q_tokens:
        return 0.0
    matches = sum(1 for t in q_tokens if t in title_tokens)
    return matches / len(q_tokens)


def keyword_match_count(query: str, keywords_str: str) -> int:
    """# tokens da query que aparecem em keywords (case-insensitive)."""
    if not keywords_str:
        return 0
    keywords_lower = keywords_str.lower()
    q_tokens = query.lower().split()
    return sum(1 for t in q_tokens if t in keywords_lower)


def doc_length(text: str) -> int:
    """Tamanho do doc em tokens (word-split simples)."""
    return len(text.split()) if text else 0


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)

    # Validacoes
    for p in (args.queries, args.qrels):
        if not os.path.isfile(p):
            print(f"erro: arquivo nao encontrado: {p}", file=sys.stderr)
            sys.exit(1)
    for p in (args.i_bm25, args.i_fw):
        if not os.path.isdir(p):
            print(f"erro: indice nao encontrado: {p}", file=sys.stderr)
            sys.exit(1)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[s5-v3] carregando qrels...", file=sys.stderr)
    qrels = read_qrels(args.qrels)
    print(f"[s5-v3] {len(qrels)} queries, "
          f"{sum(len(d) for d in qrels.values())} julgamentos",
          file=sys.stderr)

    queries = read_queries(args.queries)
    queries = [(q, t) for q, t in queries if q in qrels]
    print(f"[s5-v3] {len(queries)} queries com qrels", file=sys.stderr)

    print(f"[s5-v3] carregando searchers...", file=sys.stderr)
    searcher_bm25 = LuceneSearcher(args.i_bm25)
    searcher_bm25.set_bm25(1.2, 0.75)
    searcher_fw = LuceneSearcher(args.i_fw)
    searcher_fw.set_bm25(1.2, 0.75)
    searcher_rm3 = LuceneSearcher(args.i_fw)
    searcher_rm3.set_bm25(1.2, 0.75)
    searcher_rm3.set_rm3(10, 10, 0.8)

    index_reader = LuceneIndexReader(args.i_fw)

    # Constroi lista de TODOS os docids do corpus (para random sampling)
    # Em vez de listar todos (caro), vamos amostrar via top-K de queries genericas
    # e cachear. Mais simples: usamos doc IDs sequenciais do corpus.
    # O corpus tem 4641784 docs com IDs 0000001 a 4641784 (zero-padded 7).
    print(f"[s5-v3] preparando random sampling de IDs...", file=sys.stderr)

    def sample_random_docids(n: int, exclude: set[str]) -> list[str]:
        """Amostra n IDs aleatorios do corpus, excluindo conjunto dado."""
        result = []
        attempts = 0
        while len(result) < n and attempts < n * 10:
            i = random.randint(1, 4_641_784)
            did = f"{i:07d}"
            if did not in exclude:
                result.append(did)
            attempts += 1
        return result

    print(f"[s5-v3] extraindo features de {len(queries)} queries...", file=sys.stderr)
    t_start = time.perf_counter()

    rows: list[dict] = []

    for i, (qid, qtext) in enumerate(queries, start=1):
        t_q = time.perf_counter()

        positives = qrels[qid]
        positive_docs = set(positives.keys())

        # Hard negatives: top-K BM25 que nao sao positivos
        hits_bm25 = searcher_bm25.search(qtext, k=K_HARD_NEG + len(positive_docs))
        hard_neg_docs = [h.docid for h in hits_bm25 if h.docid not in positive_docs][:K_HARD_NEG]

        # Random negatives
        excluded = positive_docs | set(hard_neg_docs)
        rand_neg_docs = sample_random_docids(K_RAND_NEG, excluded)

        # Conjunto total de candidatos
        candidate_docs = positive_docs | set(hard_neg_docs) | set(rand_neg_docs)

        if not candidate_docs:
            print(f"[s5-v3] q{i}/{len(queries)} ({qid}): SEM CANDIDATOS",
                  file=sys.stderr)
            continue

        # Scores dos 3 searchers (busca top-1000 ou mais p/ cobrir candidatos)
        K_LOOKUP = max(1000, len(candidate_docs) * 3)
        bm25_hits = searcher_bm25.search(qtext, k=K_LOOKUP)
        fw_hits = searcher_fw.search(qtext, k=K_LOOKUP)
        rm3_hits = searcher_rm3.search(qtext, k=K_LOOKUP)

        bm25_scores = {h.docid: h.score for h in bm25_hits}
        fw_scores = {h.docid: h.score for h in fw_hits}
        rm3_scores = {h.docid: h.score for h in rm3_hits}

        # bm25 rank: posicao 1, 2, 3 no ranking BM25
        bm25_rank = {h.docid: rank for rank, h in enumerate(bm25_hits, start=1)}
        # max scores para normalizacao
        max_bm25 = max((h.score for h in bm25_hits), default=1.0) or 1.0

        # Features que dependem so da query
        query_length = len(qtext.split())
        q_mean_df = mean_df(index_reader, qtext)

        # Gera uma row por candidato
        for docid in candidate_docs:
            rel = positives.get(docid, 0)
            bm25_s = bm25_scores.get(docid, 0.0)
            fw_s = fw_scores.get(docid, 0.0)
            rm3_s = rm3_scores.get(docid, 0.0)
            rank = bm25_rank.get(docid, K_LOOKUP + 1)  # se nao no top-K, rank alto

            # Features baseadas em conteudo do doc
            fields = get_doc_fields(searcher_fw, docid)
            t_match = title_exact_match(qtext, fields["title"])
            t_overlap = title_token_overlap(qtext, fields["title"])
            kw_match = keyword_match_count(qtext, fields["keywords"])
            dlen = doc_length(fields["text"])

            row = {
                "qid": qid,
                "docid": docid,
                "relevance": rel,
                "f_bm25": bm25_s,
                "f_field_weights": fw_s,
                "f_rm3": rm3_s,
                "f_query_length": query_length,
                "f_mean_df": q_mean_df,
                "f_title_exact_match": t_match,
                "f_title_overlap": t_overlap,
                "f_keyword_match": kw_match,
                "f_doc_length": dlen,
                "f_bm25_rank": rank,
                "f_bm25_norm": bm25_s / max_bm25,
            }
            rows.append(row)

        t_q_elapsed = time.perf_counter() - t_q
        print(f"[s5-v3] q{i}/{len(queries)} ({qid}): {qtext!r} "
              f"-> {len(candidate_docs)} cands ({len(positive_docs)} pos, "
              f"{len(hard_neg_docs)} hard, {len(rand_neg_docs)} rand) "
              f"in {t_q_elapsed*1000:.0f}ms",
              file=sys.stderr)

    elapsed = time.perf_counter() - t_start

    print(f"[s5-v3] salvando {len(rows)} rows em {args.output}", file=sys.stderr)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        if not rows:
            print(f"[s5-v3] ERRO: nenhuma row extraida", file=sys.stderr)
            sys.exit(1)
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["relevance"] > 0)
    n_neg = len(rows) - n_pos
    print(f"[s5-v3] CONCLUIDO em {elapsed:.1f}s: {len(rows)} rows "
          f"({n_pos} pos, {n_neg} neg)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
