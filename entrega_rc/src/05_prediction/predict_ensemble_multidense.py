"""
06_predict_ensemble_multi.py - Como o 05, mas com MULTIPLAS features densas
(ex.: e5-large + bge-large), passadas via --dense "DIR:COLUNA" (repetivel).

Cada coluna densa = cosseno query<->entidade do cache correspondente. O CSV de
treino (ex.: train_features_v9dual.csv) ja deve conter essas colunas (geradas por
add_dense_feature.py --col-name). A ordem das features e lida do CSV.

Ganho (CV 8 splits): e5-large=0.48414 -> e5+bge-large=0.48726 (+0.00312, 8/8).

USO:
  python submissions/s5_ltr/pipeline/06_predict_ensemble_multi.py \\
      -q data/kaggle/test_queries.csv \\
      --features submissions/s5_ltr/features/train_features_v9dual.csv \\
      --dense submissions/s5_ltr/dense_e5:f_dense_sim \\
      --dense submissions/s5_ltr/dense_large:f_dense_bge \\
      --i-bm25 data/indexes/pyserini_bm25/ \\
      --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \\
      --corpus data/corpus/entities.jsonl \\
      -o submissions/s5_ltr/submission_ltr_v9dual.csv
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

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _c in (_os.path.join(_HERE, '..', '03_features'), _os.path.join(_HERE, '..', 'src', '03_features')):
    if _os.path.isdir(_c):
        _sys.path.insert(0, _os.path.abspath(_c)); break

from features_base import compute_features, load_corpus_fields
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
    ap = argparse.ArgumentParser(description="Submissao ensemble + N features densas.")
    ap.add_argument("-q", "--queries", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--dense", action="append", required=True,
                    help='cache denso no formato "DIR:COLUNA" (repetivel)')
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
        p["seed"] = SEED + s; p["bagging_seed"] = SEED + s
        p["feature_fraction_seed"] = SEED + s
        models.append(lgb.train(p, dtrain, num_boost_round=NUM_ROUNDS))
        print(f"[multi] modelo {s+1}/{N_SEEDS} treinado", file=sys.stderr)
    return models, feats


def main():
    args = parse_args()
    for p in (args.queries, args.features, args.corpus):
        if not os.path.isfile(p):
            print(f"erro: arquivo nao encontrado: {p}", file=sys.stderr); sys.exit(1)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # parse specs "DIR:COL"
    dense_specs = []
    for spec in args.dense:
        d, _, col = spec.rpartition(":")
        if not d or not col:
            print(f"erro: --dense invalido (use DIR:COL): {spec}", file=sys.stderr)
            sys.exit(1)
        dense_specs.append((col, d))

    print(f"[multi] treinando ensemble de {N_SEEDS} modelos...", file=sys.stderr)
    models, feats = train_ensemble(args.features)
    print(f"[multi] {len(feats)} features: {feats}", file=sys.stderr)
    for col, _ in dense_specs:
        if col not in feats:
            print(f"[multi] ERRO: coluna densa {col} ausente no CSV de treino",
                  file=sys.stderr); sys.exit(1)

    print("[multi] carregando caches densos...", file=sys.stderr)
    caches = {}
    for col, d in dense_specs:
        doc_emb = load_emb(os.path.join(d, "doc_emb.npz"))
        q_emb = load_emb(os.path.join(d, "query_emb_test.npz"))
        caches[col] = (doc_emb, q_emb)
        print(f"[multi]   {col}: {len(doc_emb)} docs, {len(q_emb)} queries", file=sys.stderr)

    print("[multi] carregando searchers...", file=sys.stderr)
    sb = LuceneSearcher(args.i_bm25); sb.set_bm25(1.2, 0.75)
    sf = LuceneSearcher(args.i_fw); sf.set_bm25(1.2, 0.75)
    sr = LuceneSearcher(args.i_fw); sr.set_bm25(1.2, 0.75); sr.set_rm3(10, 10, 0.8)
    ir = LuceneIndexReader(args.i_fw)

    queries = read_queries(args.queries)
    print(f"[multi] {len(queries)} queries", file=sys.stderr)

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
            print(f"[multi] busca {i}/{len(queries)}", file=sys.stderr)

    print("[multi] carregando campos do corpus...", file=sys.stderr)
    docs = load_corpus_fields(args.corpus, all_docids)

    print("[multi] predizendo (ensemble + dense multi)...", file=sys.stderr)
    total = 0
    with open(args.output, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out); w.writerow(["QueryId", "EntityId"])
        for q in per_query:
            if q is None:
                continue
            qe = {col: emb[1].get(q["qid"]) for col, emb in caches.items()}
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
                for col, (doc_emb, _) in caches.items():
                    de = doc_emb.get(docid); qv = qe[col]
                    feat[col] = float(np.dot(qv, de)) if (qv is not None and de is not None) else 0.0
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

    print(f"[multi] concluido em {time.perf_counter()-t0:.1f}s; "
          f"CSV: {args.output} ({total} linhas + header)", file=sys.stderr)


if __name__ == "__main__":
    main()
