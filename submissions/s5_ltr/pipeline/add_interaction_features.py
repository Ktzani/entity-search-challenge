"""
add_interaction_features.py - Adiciona features de INTERACAO derivadas de
f_dense_sim a um CSV de features (ex.: train_features_v7.csv -> v8).

Motivacao: o nivel absoluto de f_dense_sim varia entre queries, mas o LambdaMART
usa thresholds GLOBAIS por split. Features normalizadas POR QUERY tornam a ordem
intra-query explicita; produtos capturam sinergia semântico×léxico.

Features novas:
- f_dense_rank   : posição (1=mais similar) do candidato por f_dense_sim na query
- f_dense_norm   : f_dense_sim / max(f_dense_sim) da query
- f_dense_z      : (f_dense_sim - média da query) / std da query
- f_dense_x_bm25norm : f_dense_sim * f_bm25_norm
- f_dense_x_rm3norm  : f_dense_sim * f_rm3_norm

USO:
  python submissions/s5_ltr/pipeline/add_interaction_features.py \\
      -i submissions/s5_ltr/features/train_features_v7.csv \\
      -o submissions/s5_ltr/features/train_features_v8.csv
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

NEW = ["f_dense_rank", "f_dense_norm", "f_dense_z",
       "f_dense_x_bm25norm", "f_dense_x_rm3norm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        fieldnames = list(rd.fieldnames)
        rows = list(rd)

    if "f_dense_sim" not in fieldnames:
        print("erro: f_dense_sim ausente no input", file=sys.stderr); sys.exit(1)

    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)

    for qid, qrows in by_q.items():
        dense = np.array([float(r["f_dense_sim"]) for r in qrows])
        order = np.argsort(-dense)          # índices do maior p/ menor
        rank = np.empty(len(dense), dtype=int)
        for pos, idx in enumerate(order, start=1):
            rank[idx] = pos
        dmax = dense.max() if dense.size else 1.0
        dmean = dense.mean() if dense.size else 0.0
        dstd = dense.std() if dense.size else 1.0
        for i, r in enumerate(qrows):
            r["f_dense_rank"] = float(rank[i])
            r["f_dense_norm"] = float(dense[i] / dmax) if dmax else 0.0
            r["f_dense_z"] = float((dense[i] - dmean) / dstd) if dstd > 1e-9 else 0.0
            r["f_dense_x_bm25norm"] = float(dense[i]) * float(r.get("f_bm25_norm", 0.0))
            r["f_dense_x_rm3norm"] = float(dense[i]) * float(r.get("f_rm3_norm", 0.0))

    out_fields = fieldnames + [c for c in NEW if c not in fieldnames]
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader(); w.writerows(rows)
    print(f"[interact] {len(rows)} rows, +{len(NEW)} features -> {args.output}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
