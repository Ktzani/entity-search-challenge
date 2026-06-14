"""
01_extract_features_hybrid.py - Features para re-ranking sobre pool HIBRIDO:
candidatos = BM25 top-100 UNIAO denso top-K (faiss) [UNIAO positivos no treino].

Eleva o teto de recall (0.52 -> 0.73) trazendo os relevantes de lexical-mismatch
que so o denso acha. Feature densa f_dense_sim = cosseno query<->entidade do
bge-small (disponivel p/ TODO o corpus via corpus_index), entao funciona p/ os
candidatos novos tambem.

Colunas: qid,docid,relevance,is_pool + 16 features lexicais (FEATURE_ORDER) +
f_dense_sim + f_dense_rrank (1/rank do denso; 0 se nao recuperado).

USO (treino):
  python submissions/s5_ltr/pipeline/01_extract_features_hybrid.py \\
      -q data/kaggle/train_queries.csv --qrels data/kaggle/train_qrels.csv \\
      --i-bm25 data/indexes/pyserini_bm25/ \\
      --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \\
      --corpus data/corpus/entities.jsonl \\
      --dense-cands submissions/s5_ltr/features/dense_cands_train.csv \\
      --query-emb submissions/s5_ltr/dense/query_emb_train.npz \\
      --corpus-index submissions/s5_ltr/corpus_index/ \\
      --bm25-topk 100 --dense-topk 200 \\
      -o submissions/s5_ltr/features/train_features_hybrid.csv
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _c in (_os.path.join(_HERE, '..', '03_features'), _os.path.join(_HERE, '..', 'src', '03_features')):
    if _os.path.isdir(_c):
        _sys.path.insert(0, _os.path.abspath(_c)); break

from features_base import FEATURE_ORDER, compute_features, load_corpus_fields

K_LOOKUP = 1000


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--queries", required=True)
    ap.add_argument("--qrels", default=None, help="se ausente, modo teste (rel=0)")
    ap.add_argument("--i-bm25", required=True)
    ap.add_argument("--i-fw", required=True)
    ap.add_argument("--corpus", default="data/corpus/entities.jsonl")
    ap.add_argument("--dense-cands", required=True)
    ap.add_argument("--query-emb", required=True)
    ap.add_argument("--corpus-index", required=True)
    ap.add_argument("--bm25-topk", type=int, default=100)
    ap.add_argument("--dense-topk", type=int, default=200)
    ap.add_argument("-o", "--output", required=True)
    return ap.parse_args()


def read_qrels(path):
    q = defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if len(row) >= 3:
                try:
                    q[row[0].strip()][row[1].strip()] = int(row[2])
                except ValueError:
                    pass
    return dict(q)


def read_queries(path):
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                out.append((row[0].strip(), row[1].strip()))
    return out


def read_dense_cands(path, topk):
    d = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["dense_rank"]) <= topk:
                d[r["qid"]].append((r["docid"], int(r["dense_rank"])))
    return d


def mean_df(ir, query):
    try:
        analyzed = ir.analyze(query)
    except Exception:
        return 0.0
    if not analyzed:
        return 0.0
    tot = n = 0
    for t in analyzed:
        try:
            df, _ = ir.get_term_counts(t, analyzer=None); tot += df; n += 1
        except Exception:
            continue
    return tot / n if n else 0.0


def main():
    args = parse_args()
    qrels = read_qrels(args.qrels) if args.qrels else {}
    queries = read_queries(args.queries)
    if qrels:
        queries = [(q, t) for q, t in queries if q in qrels]
    dense_cands = read_dense_cands(args.dense_cands, args.dense_topk)
    print(f"[hyb] {len(queries)} queries", file=sys.stderr)

    sb = LuceneSearcher(args.i_bm25); sb.set_bm25(1.2, 0.75)
    sf = LuceneSearcher(args.i_fw); sf.set_bm25(1.2, 0.75)
    sr = LuceneSearcher(args.i_fw); sr.set_bm25(1.2, 0.75); sr.set_rm3(10, 10, 0.8)
    ir = LuceneIndexReader(args.i_fw)

    # embeddings densas (bge-small): queries + corpus inteiro
    qe = np.load(args.query_emb, allow_pickle=True)
    q_emb = {str(i): qe["embs"][k].astype(np.float32) for k, i in enumerate(qe["ids"])}
    print("[hyb] carregando docids do corpus_index...", file=sys.stderr)
    cdocids = np.load(os.path.join(args.corpus_index, "docids.npy"), allow_pickle=True)
    docid2idx = {str(d): k for k, d in enumerate(cdocids)}
    corpus_embs = np.load(os.path.join(args.corpus_index, "embs.f16.npy"), mmap_mode="r")
    print(f"[hyb] corpus_index: {len(docid2idx)} docs", file=sys.stderr)

    # pass A
    t0 = time.perf_counter()
    per_query = []
    all_docids = set()
    pool_sizes = []
    for i, (qid, qtext) in enumerate(queries, 1):
        positives = qrels.get(qid, {})
        bm = sb.search(qtext, k=K_LOOKUP)
        fw = sf.search(qtext, k=K_LOOKUP)
        rm = sr.search(qtext, k=K_LOOKUP)
        bm25_scores = {h.docid: h.score for h in bm}
        fw_scores = {h.docid: h.score for h in fw}
        rm3_scores = {h.docid: h.score for h in rm}
        bm25_rank = {h.docid: r for r, h in enumerate(bm, 1)}
        dcands = dense_cands.get(qid, [])
        drank = {d: r for d, r in dcands}
        pool = set(h.docid for h in bm[:args.bm25_topk]) | set(d for d, _ in dcands)
        pool_sizes.append(len(pool))
        candidates = pool | set(positives.keys())
        per_query.append({
            "qid": qid, "qtext": qtext, "pool": pool, "candidates": candidates,
            "positives": positives, "bm25_scores": bm25_scores,
            "fw_scores": fw_scores, "rm3_scores": rm3_scores, "bm25_rank": bm25_rank,
            "drank": drank,
            "max_bm25": max((h.score for h in bm), default=1.0) or 1.0,
            "max_fw": max((h.score for h in fw), default=1.0) or 1.0,
            "max_rm3": max((h.score for h in rm), default=1.0) or 1.0,
            "query_length": len(qtext.split()), "mean_df": mean_df(ir, qtext),
        })
        all_docids |= candidates
        if i % 20 == 0 or i == len(queries):
            print(f"[hyb] passe A {i}/{len(queries)} ({len(all_docids)} docids)",
                  file=sys.stderr)
    print(f"[hyb] pool medio={np.mean(pool_sizes):.0f} (vs 100); passe A "
          f"{time.perf_counter()-t0:.0f}s", file=sys.stderr)

    print("[hyb] carregando campos do corpus...", file=sys.stderr)
    docs = load_corpus_fields(args.corpus, all_docids)

    # cache de embeddings densas dos candidatos
    print("[hyb] coletando embeddings densas dos candidatos...", file=sys.stderr)
    doc_emb = {}
    miss = 0
    for did in all_docids:
        idx = docid2idx.get(did)
        if idx is None:
            miss += 1; continue
        doc_emb[did] = corpus_embs[idx].astype(np.float32)
    if miss:
        print(f"[hyb] {miss} candidatos sem embedding no corpus_index", file=sys.stderr)

    # pass B
    print("[hyb] passe B: features...", file=sys.stderr)
    fieldnames = ["qid", "docid", "relevance", "is_pool"] + FEATURE_ORDER + \
                 ["f_dense_sim", "f_dense_rrank"]
    rows = []
    for q in per_query:
        qv = q_emb.get(q["qid"])
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
            dv = doc_emb.get(docid)
            feat["f_dense_sim"] = float(np.dot(qv, dv)) if (qv is not None and dv is not None) else 0.0
            dr = q["drank"].get(docid, 0)
            feat["f_dense_rrank"] = 1.0 / dr if dr > 0 else 0.0
            row = {"qid": q["qid"], "docid": docid,
                   "relevance": q["positives"].get(docid, 0),
                   "is_pool": 1 if docid in q["pool"] else 0}
            row.update(feat)
            rows.append(row)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    n_pos = sum(1 for r in rows if r["relevance"] > 0)
    print(f"[hyb] CONCLUIDO: {len(rows)} rows ({n_pos} pos) -> {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
