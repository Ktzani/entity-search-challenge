"""
eval_ltr_cv.py - nDCG@100 SEM leakage de um CSV de features LTR, via k-fold CV.

Treina K modelos (cada um sem ver o fold avaliado), gera predicoes out-of-fold,
re-rankeia o top-100 BM25 de cada query (reconstruido do proprio CSV via
f_bm25_rank<=100) e calcula o nDCG@100 com a MESMA formula do eval_ndcg.py.
Opcionalmente escreve a submissao oof (que pode ser auditada com eval_ndcg.py).

Auto-contido: precisa so do CSV de features e dos qrels (nao re-roda buscas).

USO:
  python submissions/s5_ltr/eval_ltr_cv.py \\
      -i submissions/s5_ltr/features/train_features_v4.csv \\
      --qrels data/kaggle/train_qrels.csv \\
      [--oof-out submissions/s5_ltr/features/oof_v4.csv] [--folds 5]
"""
import argparse
import csv
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from eval_ndcg import read_qrels, ndcg_at_k, K

LGB_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [100],
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
    "min_data_in_leaf": 20, "lambda_l2": 1.0, "verbose": -1,
}
NUM_ROUNDS = 150
SEED = 42


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        feats = [c for c in rd.fieldnames if c.startswith("f_")]
        for r in rd:
            rows.append(r)
    return rows, feats


def main():
    ap = argparse.ArgumentParser(description="nDCG@100 CV de um CSV de features LTR.")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--oof-out", help="escreve a submissao out-of-fold aqui")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--params", help="overrides JSON de LGB_PARAMS", default=None)
    args = ap.parse_args()

    qrels = read_qrels(args.qrels)
    rows, feats = load_csv(args.input)
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["qid"]].append(r)
    qids = sorted(by_q.keys())

    params = dict(LGB_PARAMS)
    if args.params:
        import json
        params.update(json.loads(args.params))

    rng = np.random.RandomState(SEED)
    order = list(qids)
    rng.shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]

    def build(qset):
        X, y, g = [], [], []
        for q in qset:
            for r in by_q[q]:
                X.append([float(r[fc]) for fc in feats])
                y.append(int(r["relevance"]))
            g.append(len(by_q[q]))
        return np.array(X, np.float32), np.array(y, np.int32), np.array(g)

    oof = {}
    for fi in range(args.folds):
        test_qs = set(folds[fi])
        train_qs = [q for q in qids if q not in test_qs]
        Xtr, ytr, gtr = build(train_qs)
        model = lgb.train(params, lgb.Dataset(Xtr, label=ytr, group=gtr,
                          feature_name=feats), num_boost_round=NUM_ROUNDS)
        for q in test_qs:
            qr = by_q[q]
            Xte = np.array([[float(r[fc]) for fc in feats] for r in qr], np.float32)
            for r, p in zip(qr, model.predict(Xte)):
                oof[(q, r["docid"])] = p

    # re-rankeia top-100 BM25 (reconstruido do CSV) por score oof; desempate bm25
    ranking = {}
    for q in qids:
        cands = [r for r in by_q[q] if float(r["f_bm25_rank"]) <= K]
        cands.sort(key=lambda r: (oof[(q, r["docid"])], float(r["f_bm25"])),
                   reverse=True)
        ranking[q] = [r["docid"] for r in cands[:K]]

    per_q = {q: ndcg_at_k(ranking.get(q, []), qrels[q], K) for q in qrels}
    mean = sum(per_q.values()) / len(per_q) if per_q else 0.0
    print(f"LTR CV ({args.folds}-fold) nDCG@{K} = {mean:.5f}  "
          f"({len(feats)} features, {len(qids)} queries)")

    if args.oof_out:
        with open(args.oof_out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["QueryId", "EntityId"])
            for q in qids:
                for d in ranking[q]:
                    w.writerow([q, d])
        print(f"submissao oof escrita em {args.oof_out}")


if __name__ == "__main__":
    main()
