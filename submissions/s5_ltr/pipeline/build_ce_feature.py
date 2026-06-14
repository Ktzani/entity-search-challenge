"""
build_ce_feature.py - Adiciona f_ce (score de CROSS-ENCODER / reranker) a um CSV
de features. O cross-encoder processa o par (query, entidade) JUNTO -> sinal de
relevancia muito mais forte que o cosseno de bi-encoder.

Modelo default: BAAI/bge-reranker-large (GPU). Score = logit bruto (bom p/ LTR).

USO:
  python submissions/s5_ltr/pipeline/build_ce_feature.py \\
      -i submissions/s5_ltr/features/train_features_hybrid.csv \\
      --queries data/kaggle/train_queries.csv \\
      --corpus data/corpus/entities.jsonl \\
      -o submissions/s5_ltr/features/train_features_hybrid_ce.csv
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
MAX_CHARS = 1500


def read_queries(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if len(row) >= 2 and row[0].strip():
                out[row[0].strip()] = row[1].strip()
    return out


def load_entity_texts(corpus_path, docids):
    need = set(docids); out = {}
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
            out[did] = f"{title}. {text} {kw_str}".strip()[:MAX_CHARS]
            need.discard(did)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--corpus", default="data/corpus/entities.jsonl")
    ap.add_argument("--model", default="BAAI/bge-reranker-large")
    ap.add_argument("--max-length", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--col-name", default="f_ce")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    qtext = read_queries(args.queries)
    with open(args.input, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        fieldnames = list(rd.fieldnames)
        rows = list(rd)
    docids = {r["docid"] for r in rows}
    print(f"[ce] {len(rows)} rows, {len(docids)} docids unicos", file=sys.stderr)

    print("[ce] lendo textos das entidades...", file=sys.stderr)
    t0 = time.perf_counter()
    etexts = load_entity_texts(args.corpus, docids)
    print(f"[ce] {len(etexts)} textos em {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    import torch
    from sentence_transformers import CrossEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ce] carregando {args.model} em {dev}...", file=sys.stderr)
    model = CrossEncoder(args.model, max_length=args.max_length, device=dev)

    pairs = [(qtext.get(r["qid"], ""), etexts.get(r["docid"], "")) for r in rows]
    print(f"[ce] scoring {len(pairs)} pares (query, entidade)...", file=sys.stderr)
    t0 = time.perf_counter()
    scores = model.predict(pairs, batch_size=args.batch_size,
                          show_progress_bar=True, convert_to_numpy=True)
    print(f"[ce] scoring em {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    col = args.col_name
    if col not in fieldnames:
        fieldnames = fieldnames + [col]
    for r, s in zip(rows, scores):
        r[col] = float(s)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    s = np.array(scores)
    print(f"[ce] {col}: min={s.min():.2f} max={s.max():.2f} mean={s.mean():.2f} "
          f"-> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
