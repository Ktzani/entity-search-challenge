"""
05_predict_ensemble_v7.py - Submissao v7: v5 (ensemble 6 seeds) + feature DENSA
(f_dense_sim = cosseno query<->entidade via embeddings BGE).

Ganho em CV (8 splits, _compare_two.py): v4=0.42807 -> v7=0.46852 (+0.04044,
8/8 splits). A feature densa separa relevantes DENTRO de cada query (o que o
LambdaMART otimiza), mesmo com niveis absolutos variando entre queries.

Treina o ensemble em train_features_v7.csv (17 features = 16 do v4 + f_dense_sim)
e na predicao calcula f_dense_sim dos candidatos de teste a partir do cache denso
(build_dense_cache.py): doc_emb.npz + query_emb_test.npz. Embeddings L2-norm ->
cosseno = produto interno.

USO (apos build_dense_cache.py + add_dense_feature.py --split train):
  python submissions/s5_ltr/pipeline/05_predict_ensemble_v7.py \\
      -q data/kaggle/test_queries.csv \\
      --features submissions/s5_ltr/features/train_features_v7.csv \\
      --dense-dir submissions/s5_ltr/dense/ \\
      --i-bm25 data/indexes/pyserini_bm25/ \\
      --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \\
      --corpus data/corpus/entities.jsonl \\
      -o submissions/s5_ltr/submission_ltr_v7.csv
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict

import lightgbm as lgb
import numpy as np
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader

from _features_v4 import compute_features, load_corpus_fields
from add_dense_feature import load_emb

TOP_K = 100
K_LOOKUP = 1000
N_SEEDS = 6
BEST_CONFIG = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [100],
    "verbose": -1,
    "learning_rate": 0.03, "num_leaves": 31, "max_depth": 5,
    "min_data_in_leaf": 10, "lambda_l2": 0.0, "lambda_l1": 0.0,
    "feature_fraction": 0.9, "bagging_fraction": 0.6, "bagging_freq": 1,
}
NUM_ROUNDS = 400
SEED = 42


def parse_args():
    ap = argparse.ArgumentParser(description="Submissao v7 ensemble + dense.")
    ap.add_argument("-q", "--queries", required=True)
    ap.add_argument("--features", required=True, help="train_features_v7.csv")
    ap.add_argument("--dense-dir", required=True)
    ap.add_argument("--i-bm25", required=True)
    ap.add_argument("--i-fw", required=True)
    ap.add_argument("--corpus", default="data/corpus/entities.jsonl")
    ap.add_argument("-o", "--output", required=True)
    return ap.parse_args()


def read_queries(path):
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f); next(rd, None)
        for row in rd:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                out.append((row[0].strip(), row[1].strip()))
    return out


def mean_df(index_reader, query):
    try:
        analyzed = index_reader.analyze(query)
    except Exception:
        return 0.0
    if not analyzed:
        return 0.0
    total = n = 0
    for term in analyzed:
        try:
            df, _ = index_reader.get_term_counts(term, analyzer=None)
            total += df; n += 1
        except Exception:
            continue
    return total / n if n > 0 else 0.0


def train_ensemble(features_csv):
    by_q = defaultdict(list)
    with open(features_csv, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        feats = [c for c in rd.fieldnames if c.startswith("f_")]
        for r in rd:
            by_q[r["qid"]].append(r)

    X, y, g = [], [], []
    for q in sorted(by_q.keys()):
        for r in by_q[q]:
            X.append([float(r[fc]) for fc in feats]); y.append(int(r["relevance"]))
        g.append(len(by_q[q]))
    X = np.array(X, np.float32); y = np.array(y, np.int32); g = np.array(g)
    dtrain = lgb.Dataset(X, label=y, group=g, feature_name=feats)

    models = []
    for s in range(N_SEEDS):
        p = dict(BEST_CONFIG)
        p["seed"] = SEED + s
        p["bagging_seed"] = SEED + s
        p["feature_fraction_seed"] = SEED + s
        models.append(lgb.train(p, dtrain, num_boost_round=NUM_ROUNDS))
        print(f"[v7] modelo {s+1}/{N_SEEDS} treinado", file=sys.stderr)
    return models, feats


def main():
    args = parse_args()
    for p in (args.queries, args.features, args.corpus):
        if not os.path.isfile(p):
            print(f"erro: arquivo nao encontrado: {p}", file=sys.stderr); sys.exit(1)
    for p in (args.i_bm25, args.i_fw, args.dense_dir):
        if not os.path.isdir(p):
            print(f"erro: dir nao encontrado: {p}", file=sys.stderr); sys.exit(1)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[v7] treinando ensemble de {N_SEEDS} modelos...", file=sys.stderr)
    models, feats = train_ensemble(args.features)
    print(f"[v7] {len(feats)} features: {feats}", file=sys.stderr)
    if "f_dense_sim" not in feats:
        print("[v7] ERRO: f_dense_sim ausente no CSV de treino", file=sys.stderr)
        sys.exit(1)

    print("[v7] carregando cache denso...", file=sys.stderr)
    doc_emb = load_emb(os.path.join(args.dense_dir, "doc_emb.npz"))
    q_emb = load_emb(os.path.join(args.dense_dir, "query_emb_test.npz"))
    print(f"[v7] {len(doc_emb)} docs, {len(q_emb)} queries no cache", file=sys.stderr)

    print("[v7] carregando searchers...", file=sys.stderr)
    sb = LuceneSearcher(args.i_bm25); sb.set_bm25(1.2, 0.75)
    sf = LuceneSearcher(args.i_fw); sf.set_bm25(1.2, 0.75)
    sr = LuceneSearcher(args.i_fw); sr.set_bm25(1.2, 0.75); sr.set_rm3(10, 10, 0.8)
    ir = LuceneIndexReader(args.i_fw)

    queries = read_queries(args.queries)
    print(f"[v7] {len(queries)} queries", file=sys.stderr)

    t0 = time.perf_counter()
    per_query = []
    all_docids = set()
    for i, (qid, qtext) in enumerate(queries, 1):
        bm = sb.search(qtext, k=K_LOOKUP)
        if not bm:
            per_query.append(None); continue
        fw_scores = {h.docid: h.score for h in sf.search(qtext, k=K_LOOKUP)}
        rm3_scores = {h.docid: h.score for h in sr.search(qtext, k=K_LOOKUP)}
        rank = {h.docid: r for r, h in enumerate(bm, 1)}
        per_query.append({
            "qid": qid, "qtext": qtext,
            "candidates": [(h.docid, h.score) for h in bm[:TOP_K]],
            "bm25_scores": {h.docid: h.score for h in bm},
            "fw_scores": fw_scores, "rm3_scores": rm3_scores, "bm25_rank": rank,
            "max_bm25": max((h.score for h in bm), default=1.0) or 1.0,
            "max_fw": max(fw_scores.values(), default=1.0) or 1.0,
            "max_rm3": max(rm3_scores.values(), default=1.0) or 1.0,
            "query_length": len(qtext.split()), "mean_df": mean_df(ir, qtext),
        })
        all_docids.update(d for d, _ in per_query[-1]["candidates"])
        if i % 50 == 0 or i == len(queries):
            print(f"[v7] busca {i}/{len(queries)}", file=sys.stderr)

    print("[v7] carregando campos do corpus...", file=sys.stderr)
    docs = load_corpus_fields(args.corpus, all_docids)

    print("[v7] predizendo (ensemble + dense)...", file=sys.stderr)
    total = 0
    n_dense_miss = 0
    with open(args.output, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out); w.writerow(["QueryId", "EntityId"])
        for q in per_query:
            if q is None:
                continue
            qe = q_emb.get(q["qid"])
            X = []
            for docid, _ in q["candidates"]:
                feat = compute_features(
                    q["qtext"],
                    bm25_s=q["bm25_scores"].get(docid, 0.0),
                    fw_s=q["fw_scores"].get(docid, 0.0),
                    rm3_s=q["rm3_scores"].get(docid, 0.0),
                    bm25_rank=q["bm25_rank"].get(docid, K_LOOKUP + 1),
                    max_bm25=q["max_bm25"], max_fw=q["max_fw"], max_rm3=q["max_rm3"],
                    query_length=q["query_length"], q_mean_df=q["mean_df"],
                    doc=docs.get(docid))
                de = doc_emb.get(docid)
                if qe is None or de is None:
                    feat["f_dense_sim"] = 0.0
                    n_dense_miss += 1
                else:
                    feat["f_dense_sim"] = float(np.dot(qe, de))
                # vetor na ordem EXATA das features do CSV de treino
                X.append([feat[fc] for fc in feats])
            X = np.array(X, np.float32)
            preds = np.mean([m.predict(X) for m in models], axis=0)
            ranked = sorted(
                zip((d for d, _ in q["candidates"]), preds,
                    (s for _, s in q["candidates"])),
                key=lambda x: (-x[1], -x[2]))[:TOP_K]
            for docid, _, _ in ranked:
                w.writerow([q["qid"], docid])
            total += len(ranked)

    print(f"[v7] concluido em {time.perf_counter()-t0:.1f}s; "
          f"CSV: {args.output} ({total} linhas + header, dense_miss={n_dense_miss})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
