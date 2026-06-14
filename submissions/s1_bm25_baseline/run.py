"""
run.py - Submissao 1: BM25 disjuntivo (baseline) para o Research Challenge.

Reusa o indice gerado no PA2 e o pipeline de preprocessing/scoring. A
unica diferenca em relacao ao processor.py do PA2 eh o uso do
DisjunctiveDAAT em vez do ConjunctiveDAAT, e o output em formato CSV
(QueryId,EntityId) compativel com o Kaggle.

Uso:
    python submissions/s1_bm25_baseline/run.py \\
        -i <INDEX> -q <QUERIES_CSV> -o <SUBMISSION_CSV>

Argumentos:
    -i <INDEX>: caminho do diretorio com o indice do PA2
    -q <QUERIES_CSV>: data/kaggle/test_queries.csv
    -o <SUBMISSION_CSV>: caminho de saida (ex: submissions/s1_bm25_baseline/submission.csv)

Formato de entrada (test_queries.csv):
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
from pathlib import Path

# Adiciona raiz do projeto ao path para imports `src.*`'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.processor import TEXT_ENCODING, TOP_K
from src.index_store.document_index import DocumentIndex
from src.index_store.inverted_index import InvertedIndex
from src.index_store.term_lexicon import TermLexicon
from src.preprocessing.nltk_setup import ensure_nltk_data
from src.preprocessing.normalizer import Normalizer
from src.preprocessing.tokenizer import Tokenizer
from src.retrieval.daat import DisjunctiveDAAT
from src.retrieval.query import Query
from src.retrieval.ranker import Ranker
from src.retrieval.scorer import get_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submissao 1: BM25 disjuntivo (baseline) para Kaggle RC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", type=str, required=True, metavar="INDEX",
        help="Path to the index directory (reusa o do PA2)",
    )
    parser.add_argument(
        "-q", type=str, required=True, metavar="QUERIES_CSV",
        help="Path to test_queries.csv (formato: QueryId,Query)",
    )
    parser.add_argument(
        "-o", type=str, required=True, metavar="SUBMISSION_CSV",
        help="Path to output submission.csv (QueryId,EntityId)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    if not os.path.isdir(args.i):
        print(f"erro: diretorio de indice nao encontrado: {args.i}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.q):
        print(f"erro: arquivo de queries nao encontrado: {args.q}", file=sys.stderr)
        sys.exit(1)
    # Cria diretorio do output se nao existir
    out_dir = os.path.dirname(args.o)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def read_queries_csv(queries_path: str) -> list[tuple[str, str]]:
    """
    Le o CSV de queries do Kaggle.

    Formato esperado:
        QueryId,Query
        002,roman architecture
        ...

    Retorna lista de tuplas (query_id, raw_query). Ignora linhas vazias.
    """
    queries: list[tuple[str, str]] = []
    with open(queries_path, "r", encoding=TEXT_ENCODING, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or header[0].strip().lower() != "queryid":
            print(
                f"aviso: header inesperado em {queries_path}: {header}",
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


def process_query(
    raw_query: str,
    tokenizer: Tokenizer,
    normalizer: Normalizer,
    daat: DisjunctiveDAAT,
    scorer,
    ranker: Ranker,
    doc_index: DocumentIndex,
) -> list[str]:
    """
    Processa uma query e retorna lista de original_ids ordenada por
    score decrescente (ate TOP_K=100 elementos).
    """
    q = Query(raw_query, tokenizer, normalizer)

    if q.is_empty():
        return []

    daat_result = daat.intersect(q.terms)

    if not daat_result.matched_doc_ids:
        return []

    # Score de cada candidato
    candidates: list[tuple[int, float]] = []
    for doc_id in daat_result.matched_doc_ids:
        postings = daat_result.matched_postings[doc_id]
        score = scorer.score(doc_id, postings)
        candidates.append((doc_id, score))

    # Top-K via min-heap
    top = ranker.top_k(candidates)

    # Traduz internal_id -> original_id
    return [doc_index.get_original_id(doc_id) for doc_id, _ in top]


def main():
    args = parse_args()
    validate_args(args)

    ensure_nltk_data()

    print(f"[s1] carregando indice de {args.i}", file=sys.stderr)
    t0 = time.perf_counter()
    lexicon = TermLexicon(args.i)
    doc_index = DocumentIndex(args.i)
    elapsed_load = time.perf_counter() - t0
    print(
        f"[s1] indice carregado em {elapsed_load:.2f}s: "
        f"{lexicon.num_terms()} termos, {doc_index.num_docs()} docs, "
        f"avgdl={doc_index.avg_doc_length():.2f}",
        file=sys.stderr,
    )

    tokenizer = Tokenizer()
    normalizer = Normalizer()
    ranker = Ranker(k=TOP_K)  # TOP_K=100 (config alterado para o RC)
    scorer = get_scorer("BM25", doc_index, lexicon)

    queries = read_queries_csv(args.q)
    print(f"[s1] {len(queries)} queries lidas de {args.q}", file=sys.stderr)

    # Processa todas as queries e escreve CSV
    t_start = time.perf_counter()
    total_rows = 0
    with InvertedIndex(args.i) as ii, \
         open(args.o, "w", encoding="utf-8", newline="") as out_csv:
        daat = DisjunctiveDAAT(lexicon, ii)
        writer = csv.writer(out_csv)
        writer.writerow(["QueryId", "EntityId"])

        for i, (qid, raw_query) in enumerate(queries, start=1):
            t_query_start = time.perf_counter()
            entity_ids = process_query(
                raw_query, tokenizer, normalizer, daat, scorer, ranker, doc_index,
            )
            t_query_elapsed = time.perf_counter() - t_query_start

            # Escreve cada (qid, eid) como linha no CSV
            for eid in entity_ids:
                writer.writerow([qid, eid])
            total_rows += len(entity_ids)

            print(
                f"[s1] q{i}/{len(queries)} ({qid}): {raw_query!r} "
                f"-> {len(entity_ids)} results in {t_query_elapsed*1000:.1f}ms",
                file=sys.stderr,
            )

    elapsed_total = time.perf_counter() - t_start
    print(
        f"[s1] concluido: {len(queries)} queries processadas em "
        f"{elapsed_total:.2f}s ({elapsed_total/max(len(queries),1)*1000:.1f}ms/query)",
        file=sys.stderr,
    )
    print(
        f"[s1] CSV gerado: {args.o} ({total_rows} linhas + header)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
