"""
eval_recall.py - Mede recall dos positivos (qrels) no pool de candidatos:
BM25 top-100 (do train_features) vs denso top-k vs HIBRIDO (uniao).

Recall por query = |positivos no pool| / |positivos|. Reporta media e mediana,
e o ganho do hibrido sobre o BM25 (teto que o LTR pode alcancar).

USO:
  python submissions/s5_ltr/eval/eval_recall.py \\
      --features submissions/s5_ltr/features/train_features_v4.csv \\
      --dense-cands submissions/s5_ltr/features/dense_cands_train.csv \\
      --qrels data/kaggle/train_qrels.csv [--topk 100]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dense-cands", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--topk", type=int, default=100, help="top-k denso a considerar")
    args = ap.parse_args()

    qrels = read_qrels(args.qrels)

    # BM25 top-100 por query (do CSV de features: f_bm25_rank<=100)
    bm25 = defaultdict(set)
    with open(args.features, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["f_bm25_rank"]) <= 100:
                bm25[r["qid"]].add(r["docid"])

    # denso top-k por query
    dense = defaultdict(set)
    with open(args.dense_cands, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["dense_rank"]) <= args.topk:
                dense[r["qid"]].add(r["docid"])

    rb, rd, rh, pool = [], [], [], []
    for qid, pos in qrels.items():
        if not pos:
            continue
        b = bm25.get(qid, set()); d = dense.get(qid, set())
        rb.append(len(pos & b) / len(pos))
        rd.append(len(pos & d) / len(pos))
        rh.append(len(pos & (b | d)) / len(pos))
        pool.append(len(b | d))

    print(f"recall dos positivos ({len(rb)} queries):")
    print(f"  BM25 top-100         media={np.mean(rb):.3f}  mediana={np.median(rb):.3f}")
    print(f"  denso top-{args.topk:<4}        media={np.mean(rd):.3f}  mediana={np.median(rd):.3f}")
    print(f"  HIBRIDO (uniao)      media={np.mean(rh):.3f}  mediana={np.median(rh):.3f}")
    print(f"  ganho hibrido vs BM25: +{np.mean(rh)-np.mean(rb):.3f}")
    print(f"  tamanho medio do pool hibrido: {np.mean(pool):.0f} (vs 100 do BM25)")


if __name__ == "__main__":
    main()
