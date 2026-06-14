"""
01_extract_features.py - Extrai features de pares (query, doc) para treinar
o modelo Learning-to-Rank.

Pipeline:
1. Le train_queries.csv (234 queries) e train_qrels.csv (8202 julgamentos)
2. Para cada query, monta o conjunto de docs candidatos:
   - Positivos: docs com rel >= 1 do qrels
   - Negativos: top-100 do BM25 que NAO estao no qrels (hard negatives)
3. Para cada par (query, doc), extrai features:
   - bm25_score: score BM25 puro (indice s2)
   - field_weights_score: score BM25 do indice s3 v4
   - rm3_score: score BM25+RM3 (s4 v2: orig_weight=0.8)
   - query_length: numero de tokens na query
   - mean_df: df medio dos termos da query
4. Salva tudo em CSV pronto para LightGBM

Uso:
    python submissions/s5_ltr/01_extract_features.py \\
        -q data/kaggle/train_queries.csv \\
        --qrels data/kaggle/train_qrels.csv \\
        -i-bm25 data/indexes/pyserini_bm25/ \\
        -i-fw data/indexes/pyserini_field_weights_v4_rm3/ \\
        -o submissions/s5_ltr/train_features.csv
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

K_NEGATIVES = 100  # top-K do BM25 para gerar negativos hard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai features para LTR training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-q", "--queries", required=True,
                        help="train_queries.csv")
    parser.add_argument("--qrels", required=True,
                        help="train_qrels.csv")
    parser.add_argument("--i-bm25", required=True,
                        help="indice s2 (BM25 puro)")
    parser.add_argument("--i-fw", required=True,
                        help="indice s3 v4 (field weights, com docvectors)")
    parser.add_argument("-o", "--output", required=True,
                        help="features.csv de saida")
    return parser.parse_args()


def read_qrels(path: str) -> dict[str, dict[str, int]]:
    """Le qrels: {qid -> {docid -> relevance}}"""
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
    """Le queries: lista de (qid, texto)."""
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
    """df medio dos termos da query (depois de analise do Lucene)."""
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


def main() -> None:
    args = parse_args()

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

    print(f"[s5-features] carregando qrels...", file=sys.stderr)
    qrels = read_qrels(args.qrels)
    print(
        f"[s5-features] {len(qrels)} queries no qrels, "
        f"{sum(len(d) for d in qrels.values())} julgamentos",
        file=sys.stderr,
    )

    queries = read_queries(args.queries)
    print(f"[s5-features] {len(queries)} queries no train_queries.csv", file=sys.stderr)

    # Filtra queries que tem qrels (para nao desperdicar trabalho)
    queries = [(q, t) for q, t in queries if q in qrels]
    print(f"[s5-features] {len(queries)} queries com qrels (utilizaveis)", file=sys.stderr)

    # Inicializa searchers
    print(f"[s5-features] carregando searcher BM25 puro...", file=sys.stderr)
    searcher_bm25 = LuceneSearcher(args.i_bm25)
    searcher_bm25.set_bm25(1.2, 0.75)

    print(f"[s5-features] carregando searcher field weights...", file=sys.stderr)
    searcher_fw = LuceneSearcher(args.i_fw)
    searcher_fw.set_bm25(1.2, 0.75)

    print(f"[s5-features] carregando searcher RM3 v2...", file=sys.stderr)
    searcher_rm3 = LuceneSearcher(args.i_fw)
    searcher_rm3.set_bm25(1.2, 0.75)
    searcher_rm3.set_rm3(10, 10, 0.8)

    # IndexReader para metricas como df
    print(f"[s5-features] carregando IndexReader (field weights)...", file=sys.stderr)
    index_reader = LuceneIndexReader(args.i_fw)

    # Para cada query, extrai features dos docs candidatos
    print(f"[s5-features] extraindo features de {len(queries)} queries...",
          file=sys.stderr)
    t_start = time.perf_counter()

    rows: list[dict] = []

    for i, (qid, qtext) in enumerate(queries, start=1):
        t_q = time.perf_counter()

        # Conjunto de docs candidatos para esta query
        candidate_docs: set[str] = set()

        # 1. Positivos (do qrels)
        positives = qrels[qid]
        candidate_docs.update(positives.keys())

        # 2. Negativos (top-K do BM25 que nao estao no qrels)
        hits_bm25 = searcher_bm25.search(qtext, k=K_NEGATIVES)
        for hit in hits_bm25:
            candidate_docs.add(hit.docid)

        if not candidate_docs:
            print(f"[s5-features] q{i}/{len(queries)} ({qid}): SEM CANDIDATOS",
                  file=sys.stderr)
            continue

        # Mapas de score por searcher (usando candidate docs)
        # Para cada searcher, busca top-1000 e extrai score dos candidatos
        K_LOOKUP = max(1000, len(candidate_docs) * 2)
        bm25_scores = {h.docid: h.score for h in searcher_bm25.search(qtext, k=K_LOOKUP)}
        fw_scores = {h.docid: h.score for h in searcher_fw.search(qtext, k=K_LOOKUP)}
        rm3_scores = {h.docid: h.score for h in searcher_rm3.search(qtext, k=K_LOOKUP)}

        # Features que dependem so da query
        query_length = len(qtext.split())
        q_mean_df = mean_df(index_reader, qtext)

        # Gera uma row por candidato
        for docid in candidate_docs:
            rel = positives.get(docid, 0)  # 0 = nao relevante
            row = {
                "qid": qid,
                "docid": docid,
                "relevance": rel,
                "f_bm25": bm25_scores.get(docid, 0.0),
                "f_field_weights": fw_scores.get(docid, 0.0),
                "f_rm3": rm3_scores.get(docid, 0.0),
                "f_query_length": query_length,
                "f_mean_df": q_mean_df,
            }
            rows.append(row)

        t_q_elapsed = time.perf_counter() - t_q
        print(
            f"[s5-features] q{i}/{len(queries)} ({qid}): {qtext!r} "
            f"-> {len(candidate_docs)} candidatos in {t_q_elapsed*1000:.0f}ms",
            file=sys.stderr,
        )

    elapsed = time.perf_counter() - t_start

    # Salva CSV
    print(f"[s5-features] salvando {len(rows)} rows em {args.output}", file=sys.stderr)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        if not rows:
            print(f"[s5-features] ERRO: nenhuma row extraida", file=sys.stderr)
            sys.exit(1)
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["relevance"] > 0)
    n_neg = len(rows) - n_pos
    print(
        f"[s5-features] CONCLUIDO em {elapsed:.1f}s: "
        f"{len(rows)} rows ({n_pos} positivos, {n_neg} negativos)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
