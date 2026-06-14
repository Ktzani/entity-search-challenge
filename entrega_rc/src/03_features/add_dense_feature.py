"""
add_dense_feature.py - Adiciona a coluna f_dense_sim (cosseno query<->entidade)
a um CSV de features LTR, usando os caches de build_dense_cache.py.

Embeddings sao L2-normalizados -> cosseno = produto interno. Linhas cujo docid
ou qid nao estejam no cache recebem f_dense_sim=0.0 (fallback).

USO:
  python submissions/s5_ltr/add_dense_feature.py \\
      -i submissions/s5_ltr/features/train_features_v4.csv \\
      --dense-dir submissions/s5_ltr/dense/ --split train \\
      -o submissions/s5_ltr/features/train_features_v7.csv
"""
import argparse
import csv
import os
import sys

import numpy as np


def load_emb(path):
    d = np.load(path, allow_pickle=True)
    ids = [str(x) for x in d["ids"]]
    embs = d["embs"]
    return {i: embs[k] for k, i in enumerate(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--dense-dir", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--col-name", default="f_dense_sim",
                    help="nome da coluna de saida (use outro p/ um 2o embedding)")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    col = args.col_name

    doc_emb = load_emb(os.path.join(args.dense_dir, "doc_emb.npz"))
    q_emb = load_emb(os.path.join(args.dense_dir, f"query_emb_{args.split}.npz"))
    print(f"[dense+] {len(doc_emb)} docs, {len(q_emb)} queries no cache",
          file=sys.stderr)

    with open(args.input, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        fieldnames = list(rd.fieldnames)
        rows = list(rd)

    if col not in fieldnames:
        fieldnames = fieldnames + [col]

    n_miss = 0
    for r in rows:
        qe = q_emb.get(r["qid"])
        de = doc_emb.get(r["docid"])
        if qe is None or de is None:
            r[col] = 0.0
            n_miss += 1
        else:
            r[col] = float(np.dot(qe, de))

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    sims = [float(r[col]) for r in rows]
    print(f"[dense+] {len(rows)} rows, {col}: "
          f"min={min(sims):.3f} max={max(sims):.3f} mean={np.mean(sims):.3f} "
          f"(misses={n_miss}) -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
