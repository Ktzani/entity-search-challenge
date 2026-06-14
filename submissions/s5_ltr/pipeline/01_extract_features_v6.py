"""
01_extract_features_v6.py - Como v4, mas com UNIAO DE CANDIDATOS de 3 retrievers
para elevar o teto de recall.

Mudanca vs v4:
- Pool de candidatos (= o que existira no teste) = top-100 BM25 UNIAO top-100 RM3
  UNIAO top-100 field_weights. No treino, adiciona-se tambem os positivos do qrels
  (so para rotular), marcados com is_pool=0 se nao estiverem no pool.
- Coluna extra `is_pool` (1 = candidato real de teste; 0 = positivo injetado so p/
  o rotulo). A avaliacao CV e a predicao devem rankear apenas is_pool==1.

Motivacao: o LTR v4/v5 estava limitado ao recall do top-100 BM25. RM3 e
field_weights recuperam entidades relevantes que o BM25 perde -> mais teto.

USO:
  python submissions/s5_ltr/01_extract_features_v6.py \\
      -q data/kaggle/train_queries.csv --qrels data/kaggle/train_qrels.csv \\
      --i-bm25 data/indexes/pyserini_bm25/ \\
      --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \\
      --corpus data/corpus/entities.jsonl \\
      -o submissions/s5_ltr/features/train_features_v6.csv
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

from _features_v4 import FEATURE_ORDER, compute_features, load_corpus_fields

TOP_K = 100
K_LOOKUP = 1000


def parse_args():
    ap = argparse.ArgumentParser(description="Extrai features v6 (uniao de candidatos).")
    ap.add_argument("-q", "--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--i-bm25", required=True)
    ap.add_argument("--i-fw", required=True)
    ap.add_argument("--corpus", default="data/corpus/entities.jsonl")
    ap.add_argument("-o", "--output", required=True)
    return ap.parse_args()


def read_qrels(path):
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if len(row) >= 3:
                try:
                    qrels[row[0].strip()][row[1].strip()] = int(row[2])
                except ValueError:
                    continue
    return dict(qrels)


def read_queries(path):
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                out.append((row[0].strip(), row[1].strip()))
    return out


def mean_df(index_reader, query):
    try:
        analyzed = index_reader.analyze(query)
    except Exception:
        return 0.0
    if not analyzed:
        return 0.0
    total = n = 0
    for term in analyzed:
        try:
            df, _ = index_reader.get_term_counts(term, analyzer=None)
            total += df; n += 1
        except Exception:
            continue
    return total / n if n > 0 else 0.0


def main():
    args = parse_args()
    for p in (args.queries, args.qrels, args.corpus):
        if not os.path.isfile(p):
            print(f"erro: arquivo nao encontrado: {p}", file=sys.stderr); sys.exit(1)
    for p in (args.i_bm25, args.i_fw):
        if not os.path.isdir(p):
            print(f"erro: indice nao encontrado: {p}", file=sys.stderr); sys.exit(1)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    qrels = read_qrels(args.qrels)
    queries = [(q, t) for q, t in read_queries(args.queries) if q in qrels]
    print(f"[v6] {len(queries)} queries com qrels", file=sys.stderr)

    sb = LuceneSearcher(args.i_bm25); sb.set_bm25(1.2, 0.75)
    sf = LuceneSearcher(args.i_fw); sf.set_bm25(1.2, 0.75)
    sr = LuceneSearcher(args.i_fw); sr.set_bm25(1.2, 0.75); sr.set_rm3(10, 10, 0.8)
    ir = LuceneIndexReader(args.i_fw)

    print("[v6] passe A: buscando candidatos (uniao bm25/rm3/fields)...", file=sys.stderr)
    t0 = time.perf_counter()
    per_query = []
    all_docids = set()
    pool_sizes = []
    for i, (qid, qtext) in enumerate(queries, 1):
        positives = qrels[qid]
        pos_docs = set(positives.keys())

        bm = sb.search(qtext, k=K_LOOKUP)
        fw = sf.search(qtext, k=K_LOOKUP)
        rm = sr.search(qtext, k=K_LOOKUP)

        bm25_scores = {h.docid: h.score for h in bm}
        fw_scores = {h.docid: h.score for h in fw}
        rm3_scores = {h.docid: h.score for h in rm}
        bm25_rank = {h.docid: r for r, h in enumerate(bm, 1)}

        # POOL = uniao dos top-100 dos 3 retrievers
        pool = set(h.docid for h in bm[:TOP_K])
        pool |= set(h.docid for h in fw[:TOP_K])
        pool |= set(h.docid for h in rm[:TOP_K])
        pool_sizes.append(len(pool))

        # candidatos do treino = pool UNIAO positivos (para rotular)
        candidates = pool | pos_docs

        per_query.append({
            "qid": qid, "qtext": qtext, "candidates": candidates, "pool": pool,
            "positives": positives, "bm25_scores": bm25_scores,
            "fw_scores": fw_scores, "rm3_scores": rm3_scores, "bm25_rank": bm25_rank,
            "max_bm25": max((h.score for h in bm), default=1.0) or 1.0,
            "max_fw": max((h.score for h in fw), default=1.0) or 1.0,
            "max_rm3": max((h.score for h in rm), default=1.0) or 1.0,
            "query_length": len(qtext.split()), "mean_df": mean_df(ir, qtext),
        })
        all_docids |= candidates
        if i % 20 == 0 or i == len(queries):
            print(f"[v6] passe A {i}/{len(queries)} ({len(all_docids)} docids)",
                  file=sys.stderr)

    avg_pool = sum(pool_sizes) / len(pool_sizes)
    print(f"[v6] passe A em {time.perf_counter()-t0:.1f}s; pool medio={avg_pool:.0f} "
          f"(vs 100 do v4)", file=sys.stderr)

    print("[v6] carregando campos do corpus...", file=sys.stderr)
    docs = load_corpus_fields(args.corpus, all_docids)

    print("[v6] passe B: computando features...", file=sys.stderr)
    rows = []
    for q in per_query:
        for docid in q["candidates"]:
            feat = compute_features(
                q["qtext"],
                bm25_s=q["bm25_scores"].get(docid, 0.0),
                fw_s=q["fw_scores"].get(docid, 0.0),
                rm3_s=q["rm3_scores"].get(docid, 0.0),
                bm25_rank=q["bm25_rank"].get(docid, K_LOOKUP + 1),
                max_bm25=q["max_bm25"], max_fw=q["max_fw"], max_rm3=q["max_rm3"],
                query_length=q["query_length"], q_mean_df=q["mean_df"],
                doc=docs.get(docid))
            row = {"qid": q["qid"], "docid": docid,
                   "relevance": q["positives"].get(docid, 0),
                   "is_pool": 1 if docid in q["pool"] else 0}
            row.update(feat)
            rows.append(row)

    if not rows:
        print("[v6] ERRO: nenhuma row", file=sys.stderr); sys.exit(1)

    fieldnames = ["qid", "docid", "relevance", "is_pool"] + FEATURE_ORDER
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    n_pos = sum(1 for r in rows if r["relevance"] > 0)
    n_pool = sum(1 for r in rows if r["is_pool"] == 1)
    print(f"[v6] CONCLUIDO em {time.perf_counter()-t0:.1f}s: {len(rows)} rows "
          f"({n_pos} pos, {n_pool} no pool) sobre {len(per_query)} queries",
          file=sys.stderr)


if __name__ == "__main__":
    main()
