"""
tune_ltr.py - Busca de hiperparametros + ensemble, avaliados via 5-fold CV
nDCG@100 (mesma metrica calibrada do eval_ndcg.py). Nao sobe nada ao Kaggle.

Estrategia:
1. Random search sobre hiperparametros do LightGBM lambdarank.
2. Para os melhores configs, testa ensemble de N seeds (so ajuda se houver
   subsampling: feature_fraction<1 ou bagging_fraction<1).

Tudo reusa o setup re-ranking: candidatos = top-100 BM25 (do CSV de features),
re-rankeados pela predicao out-of-fold.

USO:
  python submissions/s5_ltr/tune_ltr.py \\
      -i submissions/s5_ltr/features/train_features_v4.csv \\
      --qrels data/kaggle/train_qrels.csv \\
      --n-configs 50 --folds 5
"""
import argparse
import csv
import random
from collections import defaultdict

import lightgbm as lgb
import numpy as np

from eval_ndcg import read_qrels, ndcg_at_k, K

SEED = 42
BASE_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [100],
    "verbose": -1,
}
V4_CONFIG = {  # config atual do 02_train_ltr.py (baseline: CV 0.42671)
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
    "min_data_in_leaf": 20, "lambda_l2": 1.0,
    "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0,
    "num_rounds": 150,
}

SEARCH_SPACE = {
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "num_leaves": [15, 31, 47, 63, 95],
    "max_depth": [4, 5, 6, 8, -1],
    "min_data_in_leaf": [10, 20, 50, 100, 200],
    "lambda_l2": [0.0, 0.5, 1.0, 2.0, 5.0],
    "lambda_l1": [0.0, 0.0, 0.5, 1.0],
    "feature_fraction": [0.6, 0.7, 0.8, 0.9, 1.0],
    "bagging_fraction": [0.6, 0.7, 0.8, 1.0],
    "num_rounds": [100, 150, 250, 400, 600],
}


def load(path):
    by_q = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        rd = csv.DictReader(f)
        feats = [c for c in rd.fieldnames if c.startswith("f_")]
        for r in rd:
            by_q[r["qid"]].append(r)
    return by_q, feats


def precompute(by_q, feats, qrels, folds):
    """Arrays por query + arrays de treino por fold (feito uma vez)."""
    qids = sorted(by_q.keys())
    perq = {}
    for q in qids:
        rows = by_q[q]
        X = np.array([[float(r[fc]) for fc in feats] for r in rows], np.float32)
        y = np.array([int(r["relevance"]) for r in rows], np.int32)
        docids = [r["docid"] for r in rows]
        rank = np.array([float(r["f_bm25_rank"]) for r in rows])
        bm25 = np.array([float(r["f_bm25"]) for r in rows])
        # Pool = candidatos que existirao no teste. Se houver coluna is_pool
        # (v6, uniao de retrievers), usa-a; senao, top-100 BM25 (v4/v5).
        if "is_pool" in rows[0]:
            pool = np.array([int(r["is_pool"]) == 1 for r in rows])
        else:
            pool = rank <= K
        perq[q] = {"X": X, "y": y, "docids": docids, "rank": rank,
                   "bm25": bm25, "pool": pool}

    fold_train = []
    for fi in range(len(folds)):
        test_qs = set(folds[fi])
        tr = [q for q in qids if q not in test_qs]
        X = np.concatenate([perq[q]["X"] for q in tr])
        y = np.concatenate([perq[q]["y"] for q in tr])
        g = np.array([len(perq[q]["y"]) for q in tr])
        fold_train.append((X, y, g, test_qs))
    return qids, perq, fold_train


def run_cv(params, num_rounds, feats, qids, perq, fold_train, qrels, seeds=(0,)):
    """nDCG@100 out-of-fold; com >1 seed faz ensemble (media das predicoes)."""
    oof = {}
    for (Xtr, ytr, gtr, test_qs) in fold_train:
        dtr = lgb.Dataset(Xtr, label=ytr, group=gtr, feature_name=feats)
        preds_per_q = {q: np.zeros(len(perq[q]["y"])) for q in test_qs}
        for s in seeds:
            p = dict(params)
            p["seed"] = SEED + s
            p["bagging_seed"] = SEED + s
            p["feature_fraction_seed"] = SEED + s
            model = lgb.train(p, dtr, num_boost_round=num_rounds)
            for q in test_qs:
                preds_per_q[q] += model.predict(perq[q]["X"])
        for q in test_qs:
            oof[q] = preds_per_q[q] / len(seeds)

    total = 0.0
    for q in qids:
        pr = oof[q]
        bm25 = perq[q]["bm25"]
        docids = perq[q]["docids"]
        pool = perq[q]["pool"]
        # candidatos do pool, ordenados por pred desc (desempate bm25)
        idx = [i for i in range(len(docids)) if pool[i]]
        idx.sort(key=lambda i: (pr[i], bm25[i]), reverse=True)
        ranked = [docids[i] for i in idx[:K]]
        total += ndcg_at_k(ranked, qrels[q], K)
    return total / len(qids)


def sample_config(rng):
    cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
    # bagging so vale com bagging_freq>0
    cfg["bagging_freq"] = 0 if cfg["bagging_fraction"] >= 1.0 else 1
    return cfg


def cfg_to_params(cfg):
    p = dict(BASE_PARAMS)
    for k, v in cfg.items():
        if k == "num_rounds":
            continue
        p[k] = float(v) if isinstance(v, (int, float, np.floating)) and k not in (
            "num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq") else v
    # ints
    for k in ("num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq"):
        p[k] = int(cfg[k])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--n-configs", type=int, default=50)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ensemble-seeds", type=int, default=5)
    args = ap.parse_args()

    qrels = read_qrels(args.qrels)
    by_q, feats = load(args.input)
    qids_all = sorted(by_q.keys())
    rng = np.random.RandomState(SEED)
    order = list(qids_all); rng.shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]
    qids, perq, fold_train = precompute(by_q, feats, qrels, folds)

    print(f"[tune] {len(qids)} queries, {len(feats)} features, "
          f"{args.folds}-fold CV, {args.n_configs} configs", flush=True)

    # baseline v4
    base_score = run_cv(cfg_to_params(V4_CONFIG), V4_CONFIG["num_rounds"],
                        feats, qids, perq, fold_train, qrels)
    print(f"[tune] BASELINE v4: {base_score:.5f}", flush=True)

    crng = random.Random(SEED)
    results = []
    for c in range(args.n_configs):
        cfg = {k: crng.choice(v) for k, v in SEARCH_SPACE.items()}
        cfg["bagging_freq"] = 0 if cfg["bagging_fraction"] >= 1.0 else 1
        score = run_cv(cfg_to_params(cfg), cfg["num_rounds"],
                       feats, qids, perq, fold_train, qrels)
        results.append((score, cfg))
        tag = " *BEAT*" if score > base_score else ""
        print(f"[tune] cfg {c+1}/{args.n_configs}: {score:.5f}{tag}  "
              f"lr={cfg['learning_rate']} leaves={cfg['num_leaves']} "
              f"depth={cfg['max_depth']} mdl={cfg['min_data_in_leaf']} "
              f"l2={cfg['lambda_l2']} ff={cfg['feature_fraction']} "
              f"bf={cfg['bagging_fraction']} nr={cfg['num_rounds']}", flush=True)

    results.sort(key=lambda x: -x[0])
    print("\n[tune] === TOP 5 CONFIGS ===", flush=True)
    for score, cfg in results[:5]:
        print(f"  {score:.5f}  {cfg}", flush=True)

    # ensemble dos melhores
    print(f"\n[tune] === ENSEMBLE ({args.ensemble_seeds} seeds) ===", flush=True)
    for score, cfg in results[:3]:
        if cfg["feature_fraction"] >= 1.0 and cfg["bagging_fraction"] >= 1.0:
            print(f"  (config sem subsampling, ensemble = single) base={score:.5f}",
                  flush=True)
            continue
        ens = run_cv(cfg_to_params(cfg), cfg["num_rounds"], feats, qids, perq,
                     fold_train, qrels, seeds=tuple(range(args.ensemble_seeds)))
        print(f"  single={score:.5f} -> ensemble={ens:.5f}  ({cfg})", flush=True)

    best_score, best_cfg = results[0]
    print(f"\n[tune] MELHOR single: {best_score:.5f} (baseline {base_score:.5f}, "
          f"delta {best_score-base_score:+.5f})", flush=True)
    import json
    print("[tune] BEST_CONFIG_JSON " + json.dumps(best_cfg), flush=True)


if __name__ == "__main__":
    main()
