"""
03_predict_test_v4.py - Aplica modelo LTR v4 nas test queries (re-ranking).

Candidatos = top-100 BM25 (mesma distribuicao do treino v4). Calcula as MESMAS
features que 01_extract_features_v4.py via o modulo compartilhado _features_v4.
Content features vem do corpus original (entities.jsonl).

Ordena por score LTR DESC, com desempate por BM25 score DESC.

Uso:
    python submissions/s5_ltr/03_predict_test_v4.py \\
        -q data/kaggle/test_queries.csv \\
        --i-bm25 data/indexes/pyserini_bm25/ \\
        --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \\
        --corpus data/corpus/entities.jsonl \\
        --model submissions/s5_ltr/models/model_v4.txt \\
        -o submissions/s5_ltr/submission_ltr_v4.csv
"""

import argparse
import csv
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

from _features_v4 import (
    FEATURE_ORDER,
    compute_features,
    feature_vector,
    load_corpus_fields,
)

TOP_K = 100        # candidatos re-rankeados por query
K_LOOKUP = 1000    # busca mais fundo p/ scores/ranks (consistente com o treino)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica modelo LTR v4 nas test queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-q", "--queries", required=True)
    parser.add_argument("--i-bm25", required=True)
    parser.add_argument("--i-fw", required=True)
    parser.add_argument("--corpus", default="data/corpus/entities.jsonl")
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


def main() -> None:
    args = parse_args()

    for p in (args.queries, args.model, args.corpus):
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

    print(f"[s5-v4-predict] carregando modelo de {args.model}", file=sys.stderr)
    model = lgb.Booster(model_file=args.model)
    model_features = model.feature_name()
    if list(model_features) != FEATURE_ORDER:
        print(f"[s5-v4-predict] AVISO: ordem de features do modelo difere!\n"
              f"  modelo: {model_features}\n  esperado: {FEATURE_ORDER}",
              file=sys.stderr)
    print(f"[s5-v4-predict] {len(model_features)} features", file=sys.stderr)

    print("[s5-v4-predict] carregando searchers...", file=sys.stderr)
    searcher_bm25 = LuceneSearcher(args.i_bm25)
    searcher_bm25.set_bm25(1.2, 0.75)
    searcher_fw = LuceneSearcher(args.i_fw)
    searcher_fw.set_bm25(1.2, 0.75)
    searcher_rm3 = LuceneSearcher(args.i_fw)
    searcher_rm3.set_bm25(1.2, 0.75)
    searcher_rm3.set_rm3(10, 10, 0.8)
    index_reader = LuceneIndexReader(args.i_fw)

    queries = read_queries(args.queries)
    print(f"[s5-v4-predict] {len(queries)} queries lidas", file=sys.stderr)

    # ---- PASSE A: busca candidatos e cacheia ----
    print("[s5-v4-predict] passe A: buscando candidatos...", file=sys.stderr)
    t_start = time.perf_counter()
    per_query = []
    all_docids: set[str] = set()

    for i, (qid, qtext) in enumerate(queries, start=1):
        bm25_hits = searcher_bm25.search(qtext, k=K_LOOKUP)
        if not bm25_hits:
            print(f"[s5-v4-predict] q{i} ({qid}): SEM CANDIDATOS", file=sys.stderr)
            per_query.append(None)
            continue

        fw_scores = {h.docid: h.score for h in searcher_fw.search(qtext, k=K_LOOKUP)}
        rm3_scores = {h.docid: h.score for h in searcher_rm3.search(qtext, k=K_LOOKUP)}
        bm25_rank = {h.docid: rank for rank, h in enumerate(bm25_hits, start=1)}

        max_bm25 = max((h.score for h in bm25_hits), default=1.0) or 1.0
        max_fw = max(fw_scores.values(), default=1.0) or 1.0
        max_rm3 = max(rm3_scores.values(), default=1.0) or 1.0

        candidates = [(h.docid, h.score) for h in bm25_hits[:TOP_K]]

        per_query.append({
            "qid": qid,
            "qtext": qtext,
            "candidates": candidates,
            "bm25_scores": {h.docid: h.score for h in bm25_hits},
            "fw_scores": fw_scores,
            "rm3_scores": rm3_scores,
            "bm25_rank": bm25_rank,
            "max_bm25": max_bm25,
            "max_fw": max_fw,
            "max_rm3": max_rm3,
            "query_length": len(qtext.split()),
            "mean_df": mean_df(index_reader, qtext),
        })
        all_docids.update(d for d, _ in candidates)

        if i % 20 == 0 or i == len(queries):
            print(f"[s5-v4-predict] passe A {i}/{len(queries)} "
                  f"({len(all_docids)} docids)", file=sys.stderr)

    print("[s5-v4-predict] carregando campos do corpus...", file=sys.stderr)
    docs = load_corpus_fields(args.corpus, all_docids)

    # ---- PASSE B: features + predicao + ranking ----
    print("[s5-v4-predict] passe B: predizendo...", file=sys.stderr)
    total_rows = 0
    with open(args.output, "w", encoding="utf-8", newline="") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["QueryId", "EntityId"])

        for q in per_query:
            if q is None:
                continue
            X_rows = []
            for docid, bm25_s in q["candidates"]:
                feat = compute_features(
                    q["qtext"],
                    bm25_s=q["bm25_scores"].get(docid, 0.0),
                    fw_s=q["fw_scores"].get(docid, 0.0),
                    rm3_s=q["rm3_scores"].get(docid, 0.0),
                    bm25_rank=q["bm25_rank"].get(docid, K_LOOKUP + 1),
                    max_bm25=q["max_bm25"],
                    max_fw=q["max_fw"],
                    max_rm3=q["max_rm3"],
                    query_length=q["query_length"],
                    q_mean_df=q["mean_df"],
                    doc=docs.get(docid),
                )
                X_rows.append(feature_vector(feat))

            X = np.array(X_rows, dtype=np.float32)
            ltr_scores = model.predict(X)

            # Ordena por LTR DESC, desempate por BM25 score DESC
            ranked = sorted(
                zip((d for d, _ in q["candidates"]),
                    ltr_scores,
                    (s for _, s in q["candidates"])),
                key=lambda x: (-x[1], -x[2]),
            )[:TOP_K]

            for docid, _, _ in ranked:
                writer.writerow([q["qid"], docid])
            total_rows += len(ranked)

    elapsed = time.perf_counter() - t_start
    print(f"[s5-v4-predict] concluido em {elapsed:.1f}s", file=sys.stderr)
    print(f"[s5-v4-predict] CSV gerado: {args.output} "
          f"({total_rows} linhas + header)", file=sys.stderr)


if __name__ == "__main__":
    main()
