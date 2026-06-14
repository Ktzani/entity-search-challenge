"""
Teste de robustez: o ganho do tuning/ensemble sobrevive a DIFERENTES splits de
fold? Roda cada config sob varios seeds de fold e reporta media +/- std.
Se o delta do best vs baseline e << std entre splits, e ruido (overfit de CV).
"""
import numpy as np
from collections import defaultdict

from eval_ndcg import read_qrels, ndcg_at_k, K
import tune_ltr as T

CSV = "submissions/s5_ltr/features/train_features_v4.csv"
QRELS = "data/kaggle/train_qrels.csv"
FOLD_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
N_FOLDS = 5

CONFIGS = {
    "v4_baseline": (dict(T.V4_CONFIG), 1),
    "best_single": ({"learning_rate": 0.03, "num_leaves": 31, "max_depth": 5,
                     "min_data_in_leaf": 10, "lambda_l2": 0.0, "lambda_l1": 0.0,
                     "feature_fraction": 0.9, "bagging_fraction": 0.6,
                     "num_rounds": 400, "bagging_freq": 1}, 1),
    "best_ensemble6": ({"learning_rate": 0.03, "num_leaves": 31, "max_depth": 5,
                        "min_data_in_leaf": 10, "lambda_l2": 0.0, "lambda_l1": 0.0,
                        "feature_fraction": 0.9, "bagging_fraction": 0.6,
                        "num_rounds": 400, "bagging_freq": 1}, 6),
}

qrels = read_qrels(QRELS)
by_q, feats = T.load(CSV)
qids_all = sorted(by_q.keys())

results = defaultdict(list)
for fs in FOLD_SEEDS:
    rng = np.random.RandomState(1000 + fs)
    order = list(qids_all); rng.shuffle(order)
    folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]
    qids, perq, fold_train = T.precompute(by_q, feats, qrels, folds)
    for name, (cfg, nseeds) in CONFIGS.items():
        nr = cfg["num_rounds"]
        score = T.run_cv(T.cfg_to_params(cfg), nr, feats, qids, perq,
                         fold_train, qrels, seeds=tuple(range(nseeds)))
        results[name].append(score)
    print(f"fold_seed {fs}: " +
          "  ".join(f"{n}={results[n][-1]:.5f}" for n in CONFIGS), flush=True)

print("\n=== media +/- std sobre", len(FOLD_SEEDS), "splits ===")
base = np.array(results["v4_baseline"])
for name in CONFIGS:
    arr = np.array(results[name])
    delta = arr - base
    print(f"  {name:16s} {arr.mean():.5f} +/- {arr.std():.5f}   "
          f"delta_vs_base={delta.mean():+.5f} +/- {delta.std():.5f}")
