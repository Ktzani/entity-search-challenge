"""
run_rm3.py - Submissao 4: BM25 + RM3 Query Expansion via Pyserini.

RM3 (Relevance Model 3) eh pseudo-relevance feedback: roda a query
original, extrai termos importantes dos top-K documentos, e adiciona
esses termos a query original com peso ajustavel. Reformula a query e
roda de novo, melhorando recall.

Requer indice construido com --storePositions --storeDocvectors --storeRaw,
caso contrario o RM3 falha.

Uso:
    python submissions/s4_rm3/run_rm3.py \\
        -i data/indexes/pyserini_field_weights_v4_rm3/ \\
        -q data/kaggle/test_queries.csv \\
        -o submissions/s4_rm3/submission_rm3_v1.csv \\
        --fb-terms 10 --fb-docs 10 --orig-weight 0.5
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
        description="Submissao 4: BM25 + RM3 via Pyserini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--index", required=True,
                        help="Path do indice Lucene (com docvectors)")
    parser.add_argument("-q", "--queries", required=True,
                        help="Path do test_queries.csv")
    parser.add_argument("-o", "--output", required=True,
                        help="Path de saida do submission.csv")
    # BM25 parameters
    parser.add_argument("--k1", type=float, default=BM25_K1,
                        help="BM25 k1")
    parser.add_argument("--b", type=float, default=BM25_B,
                        help="BM25 b")
    # RM3 parameters
    parser.add_argument("--fb-terms", type=int, default=10,
                        help="Numero de termos para expandir a query")
    parser.add_argument("--fb-docs", type=int, default=10,
                        help="Numero de docs usados como feedback")
    parser.add_argument("--orig-weight", type=float, default=0.5,
                        help="Peso da query original (0=so expansion, 1=so original)")
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
        next(reader, None)
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

    print(f"[s4] carregando indice de {args.index}", file=sys.stderr)
    t0 = time.perf_counter()
    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(args.k1, args.b)
    searcher.set_rm3(args.fb_terms, args.fb_docs, args.orig_weight)
    print(
        f"[s4] indice carregado em {time.perf_counter()-t0:.2f}s",
        file=sys.stderr,
    )
    print(
        f"[s4] BM25: k1={args.k1}, b={args.b}",
        file=sys.stderr,
    )
    print(
        f"[s4] RM3: fb_terms={args.fb_terms}, fb_docs={args.fb_docs}, "
        f"orig_weight={args.orig_weight}",
        file=sys.stderr,
    )

    queries = read_queries_csv(args.queries)
    print(f"[s4] {len(queries)} queries lidas", file=sys.stderr)

    t_start = time.perf_counter()
    total_rows = 0
    n_empty = 0
    with open(args.output, "w", encoding="utf-8", newline="") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["QueryId", "EntityId"])

        for i, (qid, raw_query) in enumerate(queries, start=1):
            t_q = time.perf_counter()
            hits = searcher.search(raw_query, k=TOP_K)
            for hit in hits:
                writer.writerow([qid, hit.docid])
            total_rows += len(hits)
            if not hits:
                n_empty += 1
            print(
                f"[s4] q{i}/{len(queries)} ({qid}): {raw_query!r} "
                f"-> {len(hits)} hits in {(time.perf_counter()-t_q)*1000:.1f}ms",
                file=sys.stderr,
            )

    elapsed = time.perf_counter() - t_start
    print(
        f"[s4] concluido em {elapsed:.2f}s "
        f"({elapsed/max(len(queries),1)*1000:.1f}ms/query)",
        file=sys.stderr,
    )
    print(
        f"[s4] CSV gerado: {args.output} ({total_rows} linhas + header, "
        f"{n_empty} queries vazias)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
