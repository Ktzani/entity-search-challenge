"""
build_faiss_from_memmap.py - Constroi um indice FAISS (IndexFlatIP = cosseno, pois
os vetores estao L2-normalizados) a partir do memmap fp16 gerado por
build_corpus_index.py.

Roda em processo SEPARADO do embedding (sem modelo/torch carregados), entao o
unico consumidor de RAM e o indice (~7GB p/ 4.6M x 384 float32), que cabe nos
~12GB livres. Adiciona em chunks p/ pico de RAM controlado.

USO:
  python submissions/s5_ltr/pipeline/build_faiss_from_memmap.py \\
      --index-dir submissions/s5_ltr/corpus_index/ --chunk 500000
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--chunk", type=int, default=500000)
    args = ap.parse_args()

    import faiss
    meta = json.load(open(os.path.join(args.index_dir, "meta.json"), encoding="utf-8"))
    n_docs, dim = meta["n_docs"], meta["dim"]
    embs = np.load(os.path.join(args.index_dir, "embs.f16.npy"), mmap_mode="r")
    print(f"[faiss] {n_docs} docs, dim={dim}; construindo IndexFlatIP...", file=sys.stderr)

    index = faiss.IndexFlatIP(dim)
    t0 = time.perf_counter()
    for start in range(0, n_docs, args.chunk):
        end = min(start + args.chunk, n_docs)
        # faiss exige float32 contiguo
        block = np.ascontiguousarray(embs[start:end], dtype=np.float32)
        index.add(block)
        del block
        print(f"[faiss] add {end}/{n_docs} ({time.perf_counter()-t0:.0f}s)", file=sys.stderr)

    out = os.path.join(args.index_dir, "corpus.faiss")
    faiss.write_index(index, out)
    print(f"[faiss] CONCLUIDO: {index.ntotal} vetores -> {out} "
          f"em {(time.perf_counter()-t0):.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
