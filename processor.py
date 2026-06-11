"""
processor.py - Entry point do query processor.

Uso:
    python3 processor.py -i <INDEX> -q <QUERIES> -r <RANKER>

Argumentos:
    -i <INDEX>: caminho do diretorio com o indice
    -q <QUERIES>: arquivo com queries (uma por linha)
    -r <RANKER>: TFIDF ou BM25

Output:
    Um JSON por linha no stdout, no formato:
        {"Query": "...", "Results": [{"ID": "...", "Score": ...}, ...]}

Fluxo por query:
    Query(raw) -> Tokenizer -> Normalizer -> terms
    terms -> ConjunctiveDAAT -> DAATResult (matched_doc_ids + postings)
    matched docs -> Scorer (TFIDF | BM25) -> [(doc_id, score)]
    candidates -> Ranker -> top-K
    top-K -> JSON output
"""

import argparse
import json
import os
import sys
import time

from src.config.processor import TEXT_ENCODING, TOP_K
from src.index_store.document_index import DocumentIndex
from src.index_store.inverted_index import InvertedIndex
from src.index_store.term_lexicon import TermLexicon
from src.preprocessing.nltk_setup import ensure_nltk_data
from src.preprocessing.normalizer import Normalizer
from src.preprocessing.tokenizer import Tokenizer
from src.retrieval.daat import ConjunctiveDAAT
from src.retrieval.query import Query
from src.retrieval.ranker import Ranker
from src.retrieval.scorer import get_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query processor with conjunctive DAAT matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        metavar="INDEX",
        help="Path to the index directory",
    )
    parser.add_argument(
        "-q",
        type=str,
        required=True,
        metavar="QUERIES",
        help="Path to the queries file (one query per line)",
    )
    parser.add_argument(
        "-r",
        type=str,
        required=True,
        metavar="RANKER",
        choices=["TFIDF", "BM25", "tfidf", "bm25"],
        help="Ranking function: TFIDF or BM25",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    if not os.path.isdir(args.i):
        print(f"erro: diretorio de indice nao encontrado: {args.i}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.q):
        print(f"erro: arquivo de queries nao encontrado: {args.q}", file=sys.stderr)
        sys.exit(1)


def read_queries(queries_path: str) -> list[str]:
    """Le queries do arquivo, uma por linha. Ignora linhas vazias."""
    queries = []
    with open(queries_path, "r", encoding=TEXT_ENCODING) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    return queries


def process_query(
    raw_query: str,
    tokenizer: Tokenizer,
    normalizer: Normalizer,
    daat: ConjunctiveDAAT,
    scorer,
    ranker: Ranker,
    doc_index: DocumentIndex,
) -> tuple[dict, dict]:
    """
    Processa uma query end-to-end. Retorna (output_json, metrics)
    onde metrics inclui skip_ratio, postings_scanned, postings_in_lists.
    """
    q = Query(raw_query, tokenizer, normalizer)

    if q.is_empty():
        return ({"Query": q.raw_text, "Results": []},
                {"skip_ratio": None, "scanned": 0, "in_lists": 0})

    daat_result = daat.intersect(q.terms)

    metrics = {
        "skip_ratio": daat_result.skip_ratio,
        "scanned": daat_result.postings_scanned,
        "in_lists": daat_result.postings_in_lists,
    }

    if not daat_result.matched_doc_ids:
        return ({"Query": q.raw_text, "Results": []}, metrics)

    candidates: list[tuple[int, float]] = []
    for doc_id in daat_result.matched_doc_ids:
        postings = daat_result.matched_postings[doc_id]
        score = scorer.score(doc_id, postings)
        candidates.append((doc_id, score))

    top = ranker.top_k(candidates)

    results = [
        {"ID": doc_index.get_original_id(doc_id), "Score": round(score, 4)}
        for doc_id, score in top
    ]

    return ({"Query": q.raw_text, "Results": results}, metrics)


def main():
    args = parse_args()
    validate_args(args)
    ranker_name = args.r.upper()

    ensure_nltk_data()

    print(f"[processor] carregando indice de {args.i}", file=sys.stderr)
    t0 = time.perf_counter()
    lexicon = TermLexicon(args.i)
    doc_index = DocumentIndex(args.i)
    elapsed_load = time.perf_counter() - t0
    print(
        f"[processor] indice carregado em {elapsed_load:.2f}s: "
        f"{lexicon.num_terms()} termos, {doc_index.num_docs()} docs, "
        f"avgdl={doc_index.avg_doc_length():.2f}",
        file=sys.stderr,
    )

    tokenizer = Tokenizer()
    normalizer = Normalizer()
    ranker = Ranker(k=TOP_K)
    scorer = get_scorer(ranker_name, doc_index, lexicon)

    queries = read_queries(args.q)
    print(f"[processor] {len(queries)} queries lidas", file=sys.stderr)

    t_start = time.perf_counter()
    skip_ratios = [] 
    with InvertedIndex(args.i) as ii:
        daat = ConjunctiveDAAT(lexicon, ii)

        for i, raw_query in enumerate(queries, start=1):
            t_query_start = time.perf_counter()
            output, metrics = process_query(
                raw_query, tokenizer, normalizer, daat, scorer, ranker, doc_index,
            )
            t_query_elapsed = time.perf_counter() - t_query_start

            print(json.dumps(output, ensure_ascii=False))

            # Log de progresso no stderr, incluindo skip_ratio
            if metrics["skip_ratio"] is not None:
                skip_ratios.append(metrics["skip_ratio"])
                skip_str = (
                    f"skip_ratio={metrics['skip_ratio']:.4f} "
                    f"(scanned={metrics['scanned']}/in_lists={metrics['in_lists']})"
                )
            else:
                skip_str = "skip_ratio=N/A"

            print(
                f"[processor] q{i}/{len(queries)}: {raw_query!r} "
                f"-> {len(output['Results'])} results in {t_query_elapsed*1000:.1f}ms; "
                f"{skip_str}",
                file=sys.stderr,
            )

    elapsed_total = time.perf_counter() - t_start
    print(
        f"[processor] {len(queries)} queries processadas em "
        f"{elapsed_total:.2f}s ({elapsed_total/max(len(queries),1)*1000:.1f}ms/query)",
        file=sys.stderr,
    )
    if skip_ratios:
        avg_skip = sum(skip_ratios) / len(skip_ratios)
        print(
            f"[processor] skip_ratio medio: {avg_skip:.4f} "
            f"(min={min(skip_ratios):.4f}, max={max(skip_ratios):.4f})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()