"""
dense_retrieve.py - Top-k denso por query usando o indice FAISS do corpus
(build_faiss_from_memmap.py). Usa query-embeddings do MESMO modelo (cache dense/
do bge-small). Escreve qid,docid,dense_rank,dense_score.

USO:
  python submissions/s5_ltr/pipeline/dense_retrieve.py \\
      --index-dir submissions/s5_ltr/corpus_index/ \\
      --query-emb submissions/s5_ltr/dense/query_emb_train.npz \\
      --topk 200 -o submissions/s5_ltr/features/dense_cands_train.csv
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--query-emb", required=True)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    import faiss
    docids = np.load(os.path.join(args.index_dir, "docids.npy"), allow_pickle=True)
    docids = np.array([str(x) for x in docids])
    index = faiss.read_index(os.path.join(args.index_dir, "corpus.faiss"))
    print(f"[dret] faiss: {index.ntotal} vetores, dim={index.d}", file=sys.stderr)

    q = np.load(args.query_emb, allow_pickle=True)
    qids = [str(x) for x in q["ids"]]
    qembs = np.ascontiguousarray(q["embs"], dtype=np.float32)
    print(f"[dret] {len(qids)} queries; busca top-{args.topk}...", file=sys.stderr)

    t0 = time.perf_counter()
    D, I = index.search(qembs, args.topk)
    print(f"[dret] busca em {time.perf_counter()-t0:.1f}s", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "docid", "dense_rank", "dense_score"])
        for qi, qid in enumerate(qids):
            for rank, (di, sc) in enumerate(zip(I[qi], D[qi]), start=1):
                if di < 0:
                    continue
                w.writerow([qid, docids[di], rank, f"{sc:.5f}"])
    print(f"[dret] CONCLUIDO -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
