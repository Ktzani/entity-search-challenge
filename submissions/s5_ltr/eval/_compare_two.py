"""
Compara dois CSVs de features LTR (A vs B) com o config de ENSEMBLE (6 seeds),
pareado sobre varios splits de fold. Generico (usado p/ v4 vs v7 etc.).

USO: python submissions/s5_ltr/_compare_two.py A.csv B.csv [nomeA nomeB]
"""
import sys
import numpy as np
from collections import defaultdict

from eval_ndcg import read_qrels
import tune_ltr as T

QRELS = "data/kaggle/train_qrels.csv"
A = sys.argv[1]
B = sys.argv[2]
NA = sys.argv[3] if len(sys.argv) > 3 else "A"
NB = sys.argv[4] if len(sys.argv) > 4 else "B"
FOLD_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
N_FOLDS = 5
N_SEEDS = 6
CFG = {"learning_rate": 0.03, "num_leaves": 31, "max_depth": 5,
       "min_data_in_leaf": 10, "lambda_l2": 0.0, "lambda_l1": 0.0,
       "feature_fraction": 0.9, "bagging_fraction": 0.6, "num_rounds": 400,
       "bagging_freq": 1}

qrels = read_qrels(QRELS)
params = T.cfg_to_params(CFG)
nr = CFG["num_rounds"]
seeds = tuple(range(N_SEEDS))

datasets = {}
for name, path in ((NA, A), (NB, B)):
    by_q, feats = T.load(path)
    datasets[name] = (by_q, feats, sorted(by_q.keys()))
    print(f"[{name}] {len(by_q)} queries, {len(feats)} features", flush=True)

results = defaultdict(list)
for fs in FOLD_SEEDS:
    line = f"fold_seed {fs}:"
    for name in (NA, NB):
        by_q, feats, qids_all = datasets[name]
        rng = np.random.RandomState(1000 + fs)
        order = list(qids_all); rng.shuffle(order)
        folds = [order[i::N_FOLDS] for i in range(N_FOLDS)]
        qids, perq, fold_train = T.precompute(by_q, feats, qrels, folds)
        score = T.run_cv(params, nr, feats, qids, perq, fold_train, qrels, seeds=seeds)
        results[name].append(score)
        line += f"  {name}={score:.5f}"
    print(line, flush=True)

a = np.array(results[NA]); b = np.array(results[NB])
print(f"\n=== media +/- std sobre {len(FOLD_SEEDS)} splits (ensemble {N_SEEDS} seeds) ===")
print(f"  {NA:18s} {a.mean():.5f} +/- {a.std():.5f}")
print(f"  {NB:18s} {b.mean():.5f} +/- {b.std():.5f}")
d = b - a
print(f"  delta {NB}-{NA} = {d.mean():+.5f} +/- {d.std():.5f}   "
      f"{NB} vence em {int((d>0).sum())}/{len(d)} splits")
