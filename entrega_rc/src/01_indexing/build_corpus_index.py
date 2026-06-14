"""
build_corpus_index.py - Indice DENSO FAISS do corpus INTEIRO (4.6M). Embeda tudo
e adiciona direto num faiss IndexFlatIP (cosseno, pois normalizamos L2). Salva o
indice + a ordem dos docids.

NOTA RAM: o IndexFlatIP segura ~7GB (4.6M x 384 float32) em RAM durante o build,
somado ao modelo/torch (~2-3GB). Garanta ~10GB+ livres antes de rodar.

USO (GPU; ~85 min p/ bge-small em RTX 3060):
  python submissions/s5_ltr/pipeline/build_corpus_index.py \\
      --corpus data/corpus/entities.jsonl \\
      --model BAAI/bge-small-en-v1.5 \\
      --out-dir submissions/s5_ltr/corpus_index/
"""
import argparse
import json
import os
import sys
import time

import numpy as np

MAX_CHARS = 2000


def iter_corpus(path, doc_prefix=""):
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            did = str(obj.get("id", ""))
            if not did:
                continue
            title = obj.get("title", "") or ""
            text = obj.get("text", "") or ""
            kw = obj.get("keywords", [])
            kw_str = " ".join(str(k) for k in kw) if isinstance(kw, list) else str(kw)
            combined = (doc_prefix + f"{title}. {text} {kw_str}").strip()[:MAX_CHARS]
            yield did, combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/entities.jsonl")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--doc-prefix", default="")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--flush-every", type=int, default=8192)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    import faiss
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[idx] modelo {args.model} em {device}", file=sys.stderr)
    model = SentenceTransformer(args.model, device=device)
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    docids = []
    buf_ids, buf_txt = [], []
    n = 0
    t0 = time.perf_counter()

    def flush():
        nonlocal buf_ids, buf_txt, n
        if not buf_txt:
            return
        emb = model.encode(buf_txt, batch_size=args.batch_size,
                          normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False).astype(np.float32)
        index.add(emb)
        docids.extend(buf_ids)
        n += len(buf_ids)
        buf_ids, buf_txt = [], []

    for did, txt in iter_corpus(args.corpus, args.doc_prefix):
        buf_ids.append(did); buf_txt.append(txt)
        if len(buf_txt) >= args.flush_every:
            flush()
            if n % 102400 < args.flush_every:
                el = time.perf_counter() - t0
                print(f"[idx] {n} docs ({n/el:.0f}/s, {el/60:.1f}min)", file=sys.stderr)
    flush()

    faiss.write_index(index, os.path.join(args.out_dir, "corpus.faiss"))
    np.save(os.path.join(args.out_dir, "docids.npy"), np.array(docids))
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "doc_prefix": args.doc_prefix,
                   "dim": dim, "n_docs": n, "max_chars": MAX_CHARS}, f, indent=2)
    print(f"[idx] CONCLUIDO: {n} docs, dim={dim} em "
          f"{(time.perf_counter()-t0)/60:.1f}min -> {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
