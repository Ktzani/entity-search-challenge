"""
01_extract_features_v4.py - Extracao de features para LTR (re-ranking setup).

Correcoes vs v3:
1. CONTENT FEATURES CORRIGIDAS: title/text/keywords vem do corpus original
   (data/corpus/entities.jsonl) via features_base.load_corpus_fields(), nao de
   um indice que nao guarda esses campos separados.
2. DISTRIBUICAO TREINO==TESTE: candidatos = top-100 BM25 (uniao com positivos
   do qrels), label por qrels. SEM random negatives. Isso alinha a distribuicao
   de treino com a de predicao (que tambem re-rankeia top-100 BM25) e elimina o
   descasamento de f_bm25_rank/f_doc_length que derrubou o v3 para 0.288.
3. Features novas (ver features_base.FEATURE_ORDER).

Uso:
    python submissions/s5_ltr/01_extract_features_v4.py \\
        -q data/kaggle/train_queries.csv \\
        --qrels data/kaggle/train_qrels.csv \\
        --i-bm25 data/indexes/pyserini_bm25/ \\
        --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \\
        --corpus data/corpus/entities.jsonl \\
        -o submissions/s5_ltr/features/train_features_v4.csv
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _c in (_os.path.join(_HERE, '..', '03_features'), _os.path.join(_HERE, '..', 'src', '03_features')):
    if _os.path.isdir(_c):
        _sys.path.insert(0, _os.path.abspath(_c)); break

from features_base import (
    FEATURE_ORDER,
    compute_features,
    load_corpus_fields,
)

TOP_K = 100          # candidatos = top-100 BM25 (mesma distribuicao do teste)
K_LOOKUP = 1000      # busca mais fundo p/ scores/ranks de fw, rm3 e positivos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai features v4 (re-ranking) para LTR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-q", "--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--i-bm25", required=True)
    parser.add_argument("--i-fw", required=True)
    parser.add_argument("--corpus", default="data/corpus/entities.jsonl")
    parser.add_argument("-o", "--output", required=True)
    return parser.parse_args()


def read_qrels(path: str) -> dict[str, dict[str, int]]:
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

    for p in (args.queries, args.qrels, args.corpus):
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

    print("[s5-v4] carregando qrels...", file=sys.stderr)
    qrels = read_qrels(args.qrels)
    print(f"[s5-v4] {len(qrels)} queries, "
          f"{sum(len(d) for d in qrels.values())} julgamentos", file=sys.stderr)

    queries = read_queries(args.queries)
    queries = [(q, t) for q, t in queries if q in qrels]
    print(f"[s5-v4] {len(queries)} queries com qrels", file=sys.stderr)

    print("[s5-v4] carregando searchers...", file=sys.stderr)
    searcher_bm25 = LuceneSearcher(args.i_bm25)
    searcher_bm25.set_bm25(1.2, 0.75)
    searcher_fw = LuceneSearcher(args.i_fw)
    searcher_fw.set_bm25(1.2, 0.75)
    searcher_rm3 = LuceneSearcher(args.i_fw)
    searcher_rm3.set_bm25(1.2, 0.75)
    searcher_rm3.set_rm3(10, 10, 0.8)
    index_reader = LuceneIndexReader(args.i_fw)

    print("[s5-v4] passe A: buscando candidatos...", file=sys.stderr)
    t_start = time.perf_counter()
    per_query = []        
    all_docids: set[str] = set()

    for i, (qid, qtext) in enumerate(queries, start=1):
        positives = qrels[qid]
        positive_docs = set(positives.keys())

        bm25_hits = searcher_bm25.search(qtext, k=K_LOOKUP)
        fw_hits = searcher_fw.search(qtext, k=K_LOOKUP)
        rm3_hits = searcher_rm3.search(qtext, k=K_LOOKUP)

        bm25_scores = {h.docid: h.score for h in bm25_hits}
        fw_scores = {h.docid: h.score for h in fw_hits}
        rm3_scores = {h.docid: h.score for h in rm3_hits}
        bm25_rank = {h.docid: rank for rank, h in enumerate(bm25_hits, start=1)}

        max_bm25 = max((h.score for h in bm25_hits), default=1.0) or 1.0
        max_fw = max((h.score for h in fw_hits), default=1.0) or 1.0
        max_rm3 = max((h.score for h in rm3_hits), default=1.0) or 1.0

        # Candidatos = top-100 BM25 UNIAO positivos (mesma distribuicao do teste,
        # mas garantindo que os positivos rotulados entrem no treino).
        top100 = [h.docid for h in bm25_hits[:TOP_K]]
        candidate_docs = set(top100) | positive_docs

        per_query.append({
            "qid": qid,
            "qtext": qtext,
            "candidates": candidate_docs,
            "positives": positives,
            "bm25_scores": bm25_scores,
            "fw_scores": fw_scores,
            "rm3_scores": rm3_scores,
            "bm25_rank": bm25_rank,
            "max_bm25": max_bm25,
            "max_fw": max_fw,
            "max_rm3": max_rm3,
            "query_length": len(qtext.split()),
            "mean_df": mean_df(index_reader, qtext),
        })
        all_docids |= candidate_docs

        if i % 20 == 0 or i == len(queries):
            print(f"[s5-v4] passe A {i}/{len(queries)} "
                  f"({len(all_docids)} docids unicos)", file=sys.stderr)

    print(f"[s5-v4] passe A concluido em {time.perf_counter()-t_start:.1f}s, "
          f"{len(all_docids)} docids unicos", file=sys.stderr)

    print("[s5-v4] carregando campos do corpus...", file=sys.stderr)
    t_corpus = time.perf_counter()
    docs = load_corpus_fields(args.corpus, all_docids)
    print(f"[s5-v4] corpus lido em {time.perf_counter()-t_corpus:.1f}s",
          file=sys.stderr)

    print("[s5-v4] passe B: computando features...", file=sys.stderr)
    rows: list[dict] = []
    for q in per_query:
        for docid in q["candidates"]:
            rel = q["positives"].get(docid, 0)
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
            row = {"qid": q["qid"], "docid": docid, "relevance": rel}
            row.update(feat)
            rows.append(row)

    if not rows:
        print("[s5-v4] ERRO: nenhuma row extraida", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["qid", "docid", "relevance"] + FEATURE_ORDER
    print(f"[s5-v4] salvando {len(rows)} rows em {args.output}", file=sys.stderr)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_pos = sum(1 for r in rows if r["relevance"] > 0)
    elapsed = time.perf_counter() - t_start
    print(f"[s5-v4] CONCLUIDO em {elapsed:.1f}s: {len(rows)} rows "
          f"({n_pos} pos, {len(rows)-n_pos} neg) sobre {len(per_query)} queries",
          file=sys.stderr)


if __name__ == "__main__":
    main()
