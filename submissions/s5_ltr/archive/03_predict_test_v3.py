"""
03_predict_test_v3.py - Aplica modelo LTR v3 (com features ricas) nas test queries.

Mantem a mesma estrategia da v2: candidatos = top-100 do BM25 puro (mesma
distribuicao do training). Tie-break por BM25 score.

Calcula as MESMAS 11 features que o 01_extract_features_v3.py.

Uso:
    python submissions/s5_ltr/03_predict_test_v3.py \\
        -q data/kaggle/test_queries.csv \\
        --i-bm25 data/indexes/pyserini_bm25/ \\
        --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \\
        --model submissions/s5_ltr/model_v3.txt \\
        -o submissions/s5_ltr/submission_ltr_v3.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

TOP_K = 100
K_LOOKUP = 100  # mesma distribuicao do training v3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica modelo LTR v3 nas test queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-q", "--queries", required=True)
    parser.add_argument("--i-bm25", required=True)
    parser.add_argument("--i-fw", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("-o", "--output", required=True)
    return parser.parse_args()


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
    try:
        doc = searcher.doc(docid)
        if doc is None:
            return {"title": "", "text": "", "keywords": ""}
        raw = doc.raw()
        if not raw:
            return {"title": "", "text": "", "keywords": ""}
        parsed = json.loads(raw)
        title = parsed.get("title", "")
        text = parsed.get("text", "")
        keywords = parsed.get("keywords", [])
        if isinstance(keywords, list):
            keywords_str = " ".join(str(k) for k in keywords)
        else:
            keywords_str = str(keywords)
        if not title and not text and "contents" in parsed:
            contents = parsed["contents"]
            return {"title": "", "text": contents, "keywords": ""}
        return {"title": title, "text": text, "keywords": keywords_str}
    except Exception:
        return {"title": "", "text": "", "keywords": ""}


def title_exact_match(query: str, title: str) -> int:
    if not title:
        return 0
    title_lower = title.lower()
    q_tokens = query.lower().split()
    return int(any(t in title_lower for t in q_tokens))


def title_token_overlap(query: str, title: str) -> float:
    if not title:
        return 0.0
    title_tokens = set(title.lower().split())
    q_tokens = query.lower().split()
    if not q_tokens:
        return 0.0
    matches = sum(1 for t in q_tokens if t in title_tokens)
    return matches / len(q_tokens)


def keyword_match_count(query: str, keywords_str: str) -> int:
    if not keywords_str:
        return 0
    keywords_lower = keywords_str.lower()
    q_tokens = query.lower().split()
    return sum(1 for t in q_tokens if t in keywords_lower)


def doc_length(text: str) -> int:
    return len(text.split()) if text else 0


def main() -> None:
    args = parse_args()

    for p in (args.queries, args.model):
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

    print(f"[s5-v3-predict] carregando modelo de {args.model}", file=sys.stderr)
    model = lgb.Booster(model_file=args.model)
    feature_names = model.feature_name()
    print(f"[s5-v3-predict] features ({len(feature_names)}): {feature_names}",
          file=sys.stderr)

    print(f"[s5-v3-predict] carregando searchers...", file=sys.stderr)
    searcher_bm25 = LuceneSearcher(args.i_bm25)
    searcher_bm25.set_bm25(1.2, 0.75)
    searcher_fw = LuceneSearcher(args.i_fw)
    searcher_fw.set_bm25(1.2, 0.75)
    searcher_rm3 = LuceneSearcher(args.i_fw)
    searcher_rm3.set_bm25(1.2, 0.75)
    searcher_rm3.set_rm3(10, 10, 0.8)

    index_reader = LuceneIndexReader(args.i_fw)

    queries = read_queries(args.queries)
    print(f"[s5-v3-predict] {len(queries)} queries lidas", file=sys.stderr)

    t_start = time.perf_counter()
    total_rows = 0
    with open(args.output, "w", encoding="utf-8", newline="") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["QueryId", "EntityId"])

        for i, (qid, qtext) in enumerate(queries, start=1):
            t_q = time.perf_counter()

            hits_bm25 = searcher_bm25.search(qtext, k=K_LOOKUP)
            if not hits_bm25:
                print(f"[s5-v3-predict] q{i}/{len(queries)} ({qid}): SEM CANDIDATOS",
                      file=sys.stderr)
                continue

            K_LOOKUP_OTHERS = 1000
            fw_scores = {h.docid: h.score for h in searcher_fw.search(qtext, k=K_LOOKUP_OTHERS)}
            rm3_scores = {h.docid: h.score for h in searcher_rm3.search(qtext, k=K_LOOKUP_OTHERS)}

            # bm25 rank e max score
            bm25_rank = {h.docid: rank for rank, h in enumerate(hits_bm25, start=1)}
            max_bm25 = max((h.score for h in hits_bm25), default=1.0) or 1.0

            # Candidatos = top-K BM25 (com score)
            candidates = [(h.docid, h.score) for h in hits_bm25]

            query_length = len(qtext.split())
            q_mean_df = mean_df(index_reader, qtext)

            # Matriz de features (11 colunas, mesma ordem do training)
            X_rows = []
            for docid, bm25_s in candidates:
                fields = get_doc_fields(searcher_fw, docid)
                t_match = title_exact_match(qtext, fields["title"])
                t_overlap = title_token_overlap(qtext, fields["title"])
                kw_match = keyword_match_count(qtext, fields["keywords"])
                dlen = doc_length(fields["text"])
                rank = bm25_rank.get(docid, K_LOOKUP + 1)
                X_rows.append([
                    bm25_s,
                    fw_scores.get(docid, 0.0),
                    rm3_scores.get(docid, 0.0),
                    query_length,
                    q_mean_df,
                    t_match,
                    t_overlap,
                    kw_match,
                    dlen,
                    rank,
                    bm25_s / max_bm25,
                ])
            X = np.array(X_rows, dtype=np.float32)

            ltr_scores = model.predict(X)

            # Ordena por LTR DESC, fallback BM25 DESC
            ranked = sorted(
                zip([c[0] for c in candidates], ltr_scores, [c[1] for c in candidates]),
                key=lambda x: (-x[1], -x[2])
            )[:TOP_K]

            for docid, _, _ in ranked:
                writer.writerow([qid, docid])
            total_rows += len(ranked)

            print(f"[s5-v3-predict] q{i}/{len(queries)} ({qid}): {qtext!r} "
                  f"-> {len(candidates)} cands, {len(ranked)} top in "
                  f"{(time.perf_counter()-t_q)*1000:.0f}ms",
                  file=sys.stderr)

    elapsed = time.perf_counter() - t_start
    print(f"[s5-v3-predict] concluido em {elapsed:.1f}s "
          f"({elapsed/max(len(queries),1)*1000:.0f}ms/query)",
          file=sys.stderr)
    print(f"[s5-v3-predict] CSV gerado: {args.output} "
          f"({total_rows} linhas + header)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
