"""
predict_from_features.py - Treina o ensemble LambdaMART num CSV de features de
treino e gera a submissao rankeando um CSV de features de TESTE (ambos ja com
as colunas f_*). Generico: serve p/ qualquer conjunto de features (ex.: hibrido).

Ordena por score do ensemble DESC, desempate por f_bm25 DESC. Top-100 por query.

USO:
  python submissions/s5_ltr/pipeline/predict_from_features.py \\
      --train submissions/s5_ltr/features/train_features_hybrid.csv \\
      --test  submissions/s5_ltr/features/test_features_hybrid.csv \\
      -o submissions/s5_ltr/submission_ltr_hybrid.csv
"""
import argparse
import csv
import sys
from collections import defaultdict, OrderedDict

import lightgbm as lgb
import numpy as np

TOP_K = 100
N_SEEDS = 6
SEED = 42
BEST_CONFIG = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [100], "verbose": -1,
    "learning_rate": 0.03, "num_leaves": 31, "max_depth": 5, "min_data_in_leaf": 10,
    "lambda_l2": 0.0, "lambda_l1": 0.0, "feature_fraction": 0.9,
    "bagging_fraction": 0.6, "bagging_freq": 1,
}
NUM_ROUNDS = 400


def load(path):
    by_q = OrderedDict()
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        feats = [c for c in rd.fieldnames if c.startswith("f_")]
        for r in rd:
            by_q.setdefault(r["qid"], []).append(r)
    return by_q, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    by_q, feats = load(args.train)
    tby_q, tfeats = load(args.test)
    if feats != tfeats:
        print(f"[pred] AVISO: features treino != teste\n train={feats}\n test={tfeats}",
              file=sys.stderr)

    X, y, g = [], [], []
    for q in sorted(by_q):
        for r in by_q[q]:
            X.append([float(r[c]) for c in feats]); y.append(int(r["relevance"]))
        g.append(len(by_q[q]))
    X = np.array(X, np.float32); y = np.array(y, np.int32); g = np.array(g)
    dtrain = lgb.Dataset(X, label=y, group=g, feature_name=feats)

    models = []
    for s in range(N_SEEDS):
        p = dict(BEST_CONFIG)
        p["seed"] = SEED + s; p["bagging_seed"] = SEED + s
        p["feature_fraction_seed"] = SEED + s
        models.append(lgb.train(p, dtrain, num_boost_round=NUM_ROUNDS))
        print(f"[pred] modelo {s+1}/{N_SEEDS}", file=sys.stderr)

    total = 0
    with open(args.output, "w", encoding="utf-8", newline="") as out:
        w = csv.writer(out); w.writerow(["QueryId", "EntityId"])
        for qid, rows in tby_q.items():
            Xt = np.array([[float(r[c]) for c in feats] for r in rows], np.float32)
            preds = np.mean([m.predict(Xt) for m in models], axis=0)
            bm = [float(r.get("f_bm25", 0.0)) for r in rows]
            order = sorted(range(len(rows)), key=lambda i: (preds[i], bm[i]), reverse=True)
            for i in order[:TOP_K]:
                w.writerow([qid, rows[i]["docid"]])
            total += min(len(rows), TOP_K)
    print(f"[pred] CONCLUIDO -> {args.output} ({total} linhas + header)", file=sys.stderr)


if __name__ == "__main__":
    main()
