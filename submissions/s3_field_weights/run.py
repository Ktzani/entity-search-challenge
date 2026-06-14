"""
run.py - Submissao 3: BM25 com FIELD WEIGHTING via field expansion.

Identico ao run.py do s2, mas usa o indice construido com convert_corpus_weighted.py
(field weighting embutido nos contents). Aceita k1 e b customizaveis.

Uso:
    python submissions/s3_field_weights/run.py \\
        -i data/indexes/pyserini_v1/ \\
        -q data/kaggle/test_queries.csv \\
        -o submissions/s3_field_weights/submission_v1.csv
"""

import argparse
import csv
import os
import sys
import time

from pyserini.search.lucene import LuceneSearcher

TOP_K = 100
BM25_K1 = 1.2
BM25_B = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submissao 3: BM25 com field weighting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--index", required=True,
                        help="Path do indice Lucene")
    parser.add_argument("-q", "--queries", required=True,
                        help="Path do test_queries.csv")
    parser.add_argument("-o", "--output", required=True,
                        help="Path de saida do submission.csv")
    parser.add_argument("--k1", type=float, default=BM25_K1)
    parser.add_argument("--b", type=float, default=BM25_B)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.index):
        print(f"erro: indice nao encontrado: {args.index}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.queries):
        print(f"erro: queries nao encontradas: {args.queries}", file=sys.stderr)
        sys.exit(1)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def read_queries_csv(queries_path: str) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    with open(queries_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if not row or len(row) < 2:
                continue
            qid, qtext = row[0].strip(), row[1].strip()
            if qid and qtext:
                queries.append((qid, qtext))
    return queries


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"[s3] carregando indice de {args.index}", file=sys.stderr)
    t0 = time.perf_counter()
    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(args.k1, args.b)
    print(
        f"[s3] indice carregado em {time.perf_counter()-t0:.2f}s "
        f"(BM25: k1={args.k1}, b={args.b})",
        file=sys.stderr,
    )

    queries = read_queries_csv(args.queries)
    print(f"[s3] {len(queries)} queries lidas", file=sys.stderr)

    t_start = time.perf_counter()
    total_rows = 0
    with open(args.output, "w", encoding="utf-8", newline="") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["QueryId", "EntityId"])

        for i, (qid, raw_query) in enumerate(queries, start=1):
            t_q = time.perf_counter()
            hits = searcher.search(raw_query, k=TOP_K)
            for hit in hits:
                writer.writerow([qid, hit.docid])
            total_rows += len(hits)
            print(
                f"[s3] q{i}/{len(queries)} ({qid}): {raw_query!r} "
                f"-> {len(hits)} hits in {(time.perf_counter()-t_q)*1000:.1f}ms",
                file=sys.stderr,
            )

    elapsed = time.perf_counter() - t_start
    print(
        f"[s3] concluido em {elapsed:.2f}s "
        f"({elapsed/max(len(queries),1)*1000:.1f}ms/query)",
        file=sys.stderr,
    )
    print(
        f"[s3] CSV gerado: {args.output} ({total_rows} linhas + header)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
