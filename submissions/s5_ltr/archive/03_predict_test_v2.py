"""
03_predict_test_v2.py - VERSAO CORRIGIDA.

Mudancas vs v1:
1. Candidatos do test EXATAMENTE como no training: top-100 do BM25 puro
   (a uniao dos 3 searchers no v1 introduzia candidatos que o modelo
   nunca viu, causando descalibracao)
2. Fallback no tie-break: se 2 docs tem score LTR identico, ordena pelo
   BM25 score (em vez de cair na ordem alfanumerica do ID)

Uso:
    python submissions/s5_ltr/03_predict_test_v2.py \\
        -q data/kaggle/test_queries.csv \\
        --i-bm25 data/indexes/pyserini_bm25/ \\
        --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \\
        --model submissions/s5_ltr/model.txt \\
        -o submissions/s5_ltr/submission_ltr_v2.csv
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

TOP_K = 100
K_LOOKUP = 100  # mesmo do training (K_NEGATIVES=100 no 01_extract_features.py)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica modelo LTR nas test queries (v2 corrigido).",
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

    print(f"[s5-predict-v2] carregando modelo de {args.model}", file=sys.stderr)
    model = lgb.Booster(model_file=args.model)
    feature_names = model.feature_name()
    print(f"[s5-predict-v2] features: {feature_names}", file=sys.stderr)

    print(f"[s5-predict-v2] carregando searchers...", file=sys.stderr)
    searcher_bm25 = LuceneSearcher(args.i_bm25)
    searcher_bm25.set_bm25(1.2, 0.75)

    searcher_fw = LuceneSearcher(args.i_fw)
    searcher_fw.set_bm25(1.2, 0.75)

    searcher_rm3 = LuceneSearcher(args.i_fw)
    searcher_rm3.set_bm25(1.2, 0.75)
    searcher_rm3.set_rm3(10, 10, 0.8)

    index_reader = LuceneIndexReader(args.i_fw)

    queries = read_queries(args.queries)
    print(f"[s5-predict-v2] {len(queries)} queries lidas", file=sys.stderr)

    t_start = time.perf_counter()
    total_rows = 0
    with open(args.output, "w", encoding="utf-8", newline="") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["QueryId", "EntityId"])

        for i, (qid, qtext) in enumerate(queries, start=1):
            t_q = time.perf_counter()

            # MUDANCA CRITICA: candidatos = top-100 do BM25 puro
            # (mesma distribuicao do training, onde K_NEGATIVES=100)
            hits_bm25 = searcher_bm25.search(qtext, k=K_LOOKUP)

            if not hits_bm25:
                print(
                    f"[s5-predict-v2] q{i}/{len(queries)} ({qid}): SEM CANDIDATOS",
                    file=sys.stderr,
                )
                continue

            # Para os candidatos do BM25, pega scores dos outros 2 searchers
            # (lookups em top-1000 pra cobrir a maioria)
            K_LOOKUP_OTHERS = 1000
            fw_scores = {h.docid: h.score for h in searcher_fw.search(qtext, k=K_LOOKUP_OTHERS)}
            rm3_scores = {h.docid: h.score for h in searcher_rm3.search(qtext, k=K_LOOKUP_OTHERS)}

            # Lista de docids candidatos (ja ordenada pelo BM25 - importante!)
            candidates = [(h.docid, h.score) for h in hits_bm25]

            # Features compartilhadas
            query_length = len(qtext.split())
            q_mean_df = mean_df(index_reader, qtext)

            # Matriz de features (mesma ordem dos candidates)
            X = np.array([
                [
                    bm25_score,
                    fw_scores.get(docid, 0.0),
                    rm3_scores.get(docid, 0.0),
                    query_length,
                    q_mean_df,
                ]
                for docid, bm25_score in candidates
            ], dtype=np.float32)

            # Predict
            ltr_scores = model.predict(X)

            # Ordena por (LTR score DESC, BM25 score DESC) - fallback no BM25
            # se LTR der ties
            ranked = sorted(
                zip([c[0] for c in candidates], ltr_scores, [c[1] for c in candidates]),
                key=lambda x: (-x[1], -x[2])
            )[:TOP_K]

            # Escreve no CSV
            for docid, _, _ in ranked:
                writer.writerow([qid, docid])
            total_rows += len(ranked)

            print(
                f"[s5-predict-v2] q{i}/{len(queries)} ({qid}): {qtext!r} "
                f"-> {len(candidates)} cands, {len(ranked)} top in "
                f"{(time.perf_counter()-t_q)*1000:.0f}ms",
                file=sys.stderr,
            )

    elapsed = time.perf_counter() - t_start
    print(
        f"[s5-predict-v2] concluido em {elapsed:.1f}s "
        f"({elapsed/max(len(queries),1)*1000:.0f}ms/query)",
        file=sys.stderr,
    )
    print(
        f"[s5-predict-v2] CSV gerado: {args.output} ({total_rows} linhas + header)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
