"""
03_predict_test.py - Aplica o modelo LTR treinado nas test_queries
para gerar submission.csv.

Pipeline:
1. Le test_queries.csv (233 queries)
2. Para cada query, extrai as MESMAS features do training:
   - Roda 3 searchers (BM25 puro, field weights, RM3 conservador)
   - Coleta top-K candidatos da uniao dos 3 searchers
   - Para cada candidato, calcula features
3. Aplica o modelo para reordenar os top-100
4. Salva submission.csv

Uso:
    python submissions/s5_ltr/03_predict_test.py \\
        -q data/kaggle/test_queries.csv \\
        --i-bm25 data/indexes/pyserini_bm25/ \\
        --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \\
        --model submissions/s5_ltr/model.txt \\
        -o submissions/s5_ltr/submission_ltr_v1.csv
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
K_LOOKUP = 200  # quantos candidatos coletar por searcher antes de reordenar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica modelo LTR nas test queries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-q", "--queries", required=True,
                        help="test_queries.csv")
    parser.add_argument("--i-bm25", required=True,
                        help="indice s2 (BM25 puro)")
    parser.add_argument("--i-fw", required=True,
                        help="indice s3 v4 (field weights, com docvectors)")
    parser.add_argument("--model", required=True,
                        help="model.txt do LightGBM")
    parser.add_argument("-o", "--output", required=True,
                        help="submission.csv de saida")
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

    # Validacoes
    if not os.path.isfile(args.queries):
        print(f"erro: queries nao encontradas: {args.queries}", file=sys.stderr)
        sys.exit(1)
    for p in (args.i_bm25, args.i_fw):
        if not os.path.isdir(p):
            print(f"erro: indice nao encontrado: {p}", file=sys.stderr)
            sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"erro: modelo nao encontrado: {args.model}", file=sys.stderr)
        sys.exit(1)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Carrega modelo
    print(f"[s5-predict] carregando modelo de {args.model}", file=sys.stderr)
    model = lgb.Booster(model_file=args.model)
    feature_names = model.feature_name()
    print(f"[s5-predict] features do modelo: {feature_names}", file=sys.stderr)

    # Carrega searchers (mesmos do training)
    print(f"[s5-predict] carregando searchers...", file=sys.stderr)
    searcher_bm25 = LuceneSearcher(args.i_bm25)
    searcher_bm25.set_bm25(1.2, 0.75)

    searcher_fw = LuceneSearcher(args.i_fw)
    searcher_fw.set_bm25(1.2, 0.75)

    searcher_rm3 = LuceneSearcher(args.i_fw)
    searcher_rm3.set_bm25(1.2, 0.75)
    searcher_rm3.set_rm3(10, 10, 0.8)

    index_reader = LuceneIndexReader(args.i_fw)

    queries = read_queries(args.queries)
    print(f"[s5-predict] {len(queries)} queries lidas", file=sys.stderr)

    # Processa
    t_start = time.perf_counter()
    total_rows = 0
    with open(args.output, "w", encoding="utf-8", newline="") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["QueryId", "EntityId"])

        for i, (qid, qtext) in enumerate(queries, start=1):
            t_q = time.perf_counter()

            # 1. Coleta candidatos: uniao dos top-K dos 3 searchers
            hits_bm25 = searcher_bm25.search(qtext, k=K_LOOKUP)
            hits_fw = searcher_fw.search(qtext, k=K_LOOKUP)
            hits_rm3 = searcher_rm3.search(qtext, k=K_LOOKUP)

            bm25_scores = {h.docid: h.score for h in hits_bm25}
            fw_scores = {h.docid: h.score for h in hits_fw}
            rm3_scores = {h.docid: h.score for h in hits_rm3}

            candidate_docs = set(bm25_scores.keys()) | set(fw_scores.keys()) | set(rm3_scores.keys())

            if not candidate_docs:
                print(
                    f"[s5-predict] q{i}/{len(queries)} ({qid}): SEM CANDIDATOS",
                    file=sys.stderr,
                )
                continue

            # 2. Features compartilhadas
            query_length = len(qtext.split())
            q_mean_df = mean_df(index_reader, qtext)

            # 3. Monta matriz de features
            docids = sorted(candidate_docs)
            X = np.array([
                [
                    bm25_scores.get(d, 0.0),
                    fw_scores.get(d, 0.0),
                    rm3_scores.get(d, 0.0),
                    query_length,
                    q_mean_df,
                ]
                for d in docids
            ], dtype=np.float32)

            # 4. Predict
            scores = model.predict(X)

            # 5. Ordena por score decrescente, pega top-K
            ranked = sorted(zip(docids, scores), key=lambda x: -x[1])[:TOP_K]

            # 6. Escreve no CSV
            for docid, _ in ranked:
                writer.writerow([qid, docid])
            total_rows += len(ranked)

            print(
                f"[s5-predict] q{i}/{len(queries)} ({qid}): {qtext!r} "
                f"-> {len(candidate_docs)} cands, {len(ranked)} top in "
                f"{(time.perf_counter()-t_q)*1000:.0f}ms",
                file=sys.stderr,
            )

    elapsed = time.perf_counter() - t_start
    print(
        f"[s5-predict] concluido em {elapsed:.1f}s "
        f"({elapsed/max(len(queries),1)*1000:.0f}ms/query)",
        file=sys.stderr,
    )
    print(
        f"[s5-predict] CSV gerado: {args.output} ({total_rows} linhas + header)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
