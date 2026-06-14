"""
eval_sparse_recall.py - Recall@k dos retrievers ESPARSOS (BM25 plano,
field-weighted, RM3) e da UNIAO deles, nas train queries vs qrels.

Responde: o indice com pesos de campo (title boost) recupera melhor que o BM25
plano? A uniao esparsa eleva o teto de recall (sem precisar de denso)?
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
from pyserini.search.lucene import LuceneSearcher


def read_qrels(path):
    q = defaultdict(set)
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if len(row) >= 3:
                try:
                    if int(row[2]) > 0:
                        q[row[0].strip()].add(row[1].strip())
                except ValueError:
                    pass
    return q


def read_queries(path):
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                out.append((row[0].strip(), row[1].strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/kaggle/train_queries.csv")
    ap.add_argument("--qrels", default="data/kaggle/train_qrels.csv")
    ap.add_argument("--i-bm25", default="data/indexes/pyserini_bm25/")
    ap.add_argument("--i-fw", default="data/indexes/pyserini_field_weights_v4_rm3/")
    ap.add_argument("--depth", type=int, default=1000)
    args = ap.parse_args()

    qrels = read_qrels(args.qrels)
    queries = [(q, t) for q, t in read_queries(args.queries) if q in qrels]

    sb = LuceneSearcher(args.i_bm25); sb.set_bm25(1.2, 0.75)
    sf = LuceneSearcher(args.i_fw); sf.set_bm25(1.2, 0.75)
    sr = LuceneSearcher(args.i_fw); sr.set_bm25(1.2, 0.75); sr.set_rm3(10, 10, 0.8)

    cuts = [100, 200, 500, 1000]
    rec = {m: {k: [] for k in cuts} for m in ("BM25", "fields", "RM3", "uniao")}
    for qid, qtext in queries:
        pos = qrels[qid]
        if not pos:
            continue
        hb = [h.docid for h in sb.search(qtext, k=args.depth)]
        hf = [h.docid for h in sf.search(qtext, k=args.depth)]
        hr = [h.docid for h in sr.search(qtext, k=args.depth)]
        for k in cuts:
            sb_k, sf_k, sr_k = set(hb[:k]), set(hf[:k]), set(hr[:k])
            rec["BM25"][k].append(len(pos & sb_k) / len(pos))
            rec["fields"][k].append(len(pos & sf_k) / len(pos))
            rec["RM3"][k].append(len(pos & sr_k) / len(pos))
            rec["uniao"][k].append(len(pos & (sb_k | sf_k | sr_k)) / len(pos))

    print(f"recall@k medio ({len(rec['BM25'][100])} queries):")
    print(f"  {'metodo':10s}" + "".join(f"  @{k:<5}" for k in cuts))
    for m in ("BM25", "fields", "RM3", "uniao"):
        print(f"  {m:10s}" + "".join(f"  {np.mean(rec[m][k]):.3f}" for k in cuts))


if __name__ == "__main__":
    main()
