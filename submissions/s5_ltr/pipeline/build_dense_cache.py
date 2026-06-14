"""
build_dense_cache.py - Pre-computa embeddings densos (uma vez) para a feature
f_dense_sim do LTR. Salva caches .npz que add_dense_feature.py e o predict v7
consomem, sem reembedar.

Entidades candidatas (treino: do CSV de features; teste: dos docids da submissao
v5) sao embedadas com title+text+keywords. Queries sao embedadas com prefixo de
instrucao (default = BGE). Tudo normalizado L2 -> cosseno = produto interno.

USO (apos instalar sentence-transformers + torch-CUDA):
  python submissions/s5_ltr/build_dense_cache.py \\
      --features submissions/s5_ltr/features/train_features_v4.csv \\
      --test-sub submissions/s5_ltr/submission_ltr_v5.csv \\
      --train-queries data/kaggle/train_queries.csv \\
      --test-queries data/kaggle/test_queries.csv \\
      --corpus data/corpus/entities.jsonl \\
      --out-dir submissions/s5_ltr/dense/
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ID_PREFIX = '{"id": "'
ID_LEN = 7
# instrucao recomendada do BGE para o lado da QUERY (passages sem prefixo)
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_CHARS = 2000  # trunca o texto da entidade antes de tokenizar


def read_ids_from_csv(path, col):
    ids = set()
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            ids.add(r[col].strip())
    return ids


def read_queries(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                out[row[0].strip()] = row[1].strip()
    return out


def load_entity_texts(corpus_path, docids):
    """Stream do corpus; retorna {docid: 'title. text keywords' truncado}."""
    need = set(docids)
    out = {}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if not need:
                break
            if line.startswith(ID_PREFIX):
                did = line[len(ID_PREFIX):len(ID_PREFIX) + ID_LEN]
                if did not in need:
                    continue
            else:
                try:
                    did = str(json.loads(line).get("id", ""))
                except Exception:
                    continue
                if did not in need:
                    continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            title = obj.get("title", "") or ""
            text = obj.get("text", "") or ""
            kw = obj.get("keywords", [])
            kw_str = " ".join(str(k) for k in kw) if isinstance(kw, list) else str(kw)
            combined = f"{title}. {text} {kw_str}".strip()[:MAX_CHARS]
            out[did] = combined
            need.discard(did)
    if need:
        print(f"[dense] AVISO: {len(need)} docids sem texto no corpus", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--test-sub", required=True)
    ap.add_argument("--train-queries", required=True)
    ap.add_argument("--test-queries", required=True)
    ap.add_argument("--corpus", default="data/corpus/entities.jsonl")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--query-prefix", default=DEFAULT_QUERY_PREFIX)
    ap.add_argument("--doc-prefix", default="",
                    help='prefixo do lado do documento (ex.: "passage: " p/ e5)')
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) coletar docids candidatos (treino + teste) e queries
    print("[dense] coletando docids...", file=sys.stderr)
    docids = read_ids_from_csv(args.features, "docid")
    docids |= read_ids_from_csv(args.test_sub, "EntityId")
    print(f"[dense] {len(docids)} entidades unicas a embedar", file=sys.stderr)

    q_train = read_queries(args.train_queries)
    q_test = read_queries(args.test_queries)

    # 2) textos das entidades
    print("[dense] lendo textos do corpus...", file=sys.stderr)
    t0 = time.perf_counter()
    texts = load_entity_texts(args.corpus, docids)
    print(f"[dense] {len(texts)} textos em {time.perf_counter()-t0:.1f}s", file=sys.stderr)

    # 3) modelo
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dense] carregando {args.model} em {device}...", file=sys.stderr)
    model = SentenceTransformer(args.model, device=device)

    def encode(items, prefix=""):
        keys = list(items.keys())
        vals = [prefix + items[k] for k in keys]
        embs = model.encode(vals, batch_size=args.batch_size,
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=True)
        return keys, embs.astype(np.float32)

    # 4) embeddings de entidades
    print("[dense] embedando entidades...", file=sys.stderr)
    t0 = time.perf_counter()
    dkeys, dembs = encode(texts, prefix=args.doc_prefix)
    np.savez(os.path.join(args.out_dir, "doc_emb.npz"),
             ids=np.array(dkeys), embs=dembs)
    print(f"[dense] {len(dkeys)} entidades em {time.perf_counter()-t0:.1f}s "
          f"(dim={dembs.shape[1]})", file=sys.stderr)

    # 5) embeddings de queries (com prefixo)
    for name, q in (("train", q_train), ("test", q_test)):
        qkeys, qembs = encode(q, prefix=args.query_prefix)
        np.savez(os.path.join(args.out_dir, f"query_emb_{name}.npz"),
                 ids=np.array(qkeys), embs=qembs)
        print(f"[dense] {len(qkeys)} queries {name}", file=sys.stderr)

    # registra metadados
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "query_prefix": args.query_prefix,
                   "doc_prefix": args.doc_prefix,
                   "dim": int(dembs.shape[1]), "max_chars": MAX_CHARS}, f, indent=2)
    print(f"[dense] CONCLUIDO -> {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
