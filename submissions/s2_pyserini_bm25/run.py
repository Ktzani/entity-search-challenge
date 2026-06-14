"""
run.py - Submissao 2: BM25 via Pyserini (Lucene).

Pre-requisito: indice Lucene ja construido em data/indexes/pyserini_bm25/
via:
    python -m pyserini.index.lucene \\
        --collection JsonCollection \\
        --generator DefaultLuceneDocumentGenerator \\
        --threads 8 \\
        --input data/corpus_pyserini/ \\
        --index data/indexes/pyserini_bm25/

Uso:
    python submissions/s2_pyserini_bm25/run.py \\
        -i data/indexes/pyserini_bm25/ \\
        -q data/kaggle/test_queries.csv \\
        -o submissions/s2_pyserini_bm25/submission.csv

Formato de entrada das queries (test_queries.csv):
    QueryId,Query
    002,roman architecture
    ...

Formato de saida (submission.csv):
    QueryId,EntityId
    002,0878002
    002,3056323
    ...
"""

import argparse
import csv
import os
import sys
import time

from pyserini.search.lucene import LuceneSearcher

TOP_K = 100  # Kaggle pede top-100 por query
BM25_K1 = 1.2  # mesmos parametros do PA2 para comparacao justa
BM25_B = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submissao 2: BM25 via Pyserini (Lucene).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--index", type=str, required=True,
        help="Path do indice Lucene construido pelo Pyserini",
    )
    parser.add_argument(
        "-q", "--queries", type=str, required=True,
        help="Path do test_queries.csv (formato: QueryId,Query)",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path de saida do submission.csv",
    )
    parser.add_argument(
        "--k1", type=float, default=BM25_K1,
        help="BM25 k1 parameter",
    )
    parser.add_argument(
        "--b", type=float, default=BM25_B,
        help="BM25 b parameter",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.index):
        print(f"erro: diretorio do indice nao encontrado: {args.index}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.queries):
        print(f"erro: arquivo de queries nao encontrado: {args.queries}", file=sys.stderr)
        sys.exit(1)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def read_queries_csv(queries_path: str) -> list[tuple[str, str]]:
    """Le CSV no formato 'QueryId,Query'. Retorna lista de (qid, query)."""
    queries: list[tuple[str, str]] = []
    with open(queries_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or header[0].strip().lower() != "queryid":
            print(
                f"aviso: header inesperado: {header}",
                file=sys.stderr,
            )
        for row in reader:
            if not row or len(row) < 2:
                continue
            qid = row[0].strip()
            qtext = row[1].strip()
            if qid and qtext:
                queries.append((qid, qtext))
    return queries


def main() -> None:
    args = parse_args()
    validate_args(args)

    # === Carrega indice ===
    print(f"[s2] carregando indice Lucene de {args.index}", file=sys.stderr)
    t0 = time.perf_counter()
    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(args.k1, args.b)
    elapsed_load = time.perf_counter() - t0
    print(
        f"[s2] indice carregado em {elapsed_load:.2f}s "
        f"(BM25: k1={args.k1}, b={args.b})",
        file=sys.stderr,
    )

    # === Le queries ===
    queries = read_queries_csv(args.queries)
    print(f"[s2] {len(queries)} queries lidas de {args.queries}", file=sys.stderr)

    # === Processa todas as queries ===
    t_start = time.perf_counter()
    total_rows = 0
    n_empty = 0
    with open(args.output, "w", encoding="utf-8", newline="") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["QueryId", "EntityId"])

        for i, (qid, raw_query) in enumerate(queries, start=1):
            t_query_start = time.perf_counter()
            hits = searcher.search(raw_query, k=TOP_K)
            t_query_elapsed = time.perf_counter() - t_query_start

            # Escreve cada (qid, docid) como linha no CSV
            for hit in hits:
                writer.writerow([qid, hit.docid])
            total_rows += len(hits)

            if not hits:
                n_empty += 1

            print(
                f"[s2] q{i}/{len(queries)} ({qid}): {raw_query!r} "
                f"-> {len(hits)} results in {t_query_elapsed*1000:.1f}ms",
                file=sys.stderr,
            )

    elapsed_total = time.perf_counter() - t_start
    print(
        f"[s2] concluido: {len(queries)} queries em "
        f"{elapsed_total:.2f}s ({elapsed_total/max(len(queries),1)*1000:.1f}ms/query)",
        file=sys.stderr,
    )
    print(
        f"[s2] CSV gerado: {args.output} ({total_rows} linhas + header, "
        f"{n_empty} queries vazias)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
