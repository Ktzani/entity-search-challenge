"""
02_train_ltr.py - Treina LambdaMART (LightGBM) otimizando nDCG@100.

Le o CSV de features produzido por 01_extract_features.py e treina um
modelo de Learning-to-Rank. O modelo aprende uma combinacao otima dos
scores de BM25, field weights e RM3 mais features da query.

Uso:
    python submissions/s5_ltr/02_train_ltr.py \\
        -i submissions/s5_ltr/train_features.csv \\
        -o submissions/s5_ltr/model.txt
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import lightgbm as lgb
import numpy as np

# Hiperparametros do LambdaMART
LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10, 100],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "lambda_l2": 1.0,
    "verbose": -1,
}
NUM_ROUNDS = 500
EARLY_STOPPING = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina LambdaMART para LTR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True,
                        help="train_features.csv")
    parser.add_argument("-o", "--output", required=True,
                        help="model.txt de saida")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fracao de queries para validacao")
    return parser.parse_args()


def load_features(csv_path: str) -> tuple[list[dict], list[str]]:
    """Le o CSV, retorna lista de rows e lista de feature columns."""
    rows: list[dict] = []
    feature_cols: list[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        feature_cols = [c for c in reader.fieldnames if c.startswith("f_")]
        for row in reader:
            row["relevance"] = int(row["relevance"])
            for fc in feature_cols:
                row[fc] = float(row[fc])
            rows.append(row)
    return rows, feature_cols


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"erro: arquivo nao encontrado: {args.input}", file=sys.stderr)
        sys.exit(1)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Le features
    print(f"[s5-train] carregando features...", file=sys.stderr)
    rows, feature_cols = load_features(args.input)
    print(
        f"[s5-train] {len(rows)} rows, {len(feature_cols)} features: {feature_cols}",
        file=sys.stderr,
    )

    # Agrupa por qid (LightGBM precisa de groups)
    qid_to_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        qid_to_rows[r["qid"]].append(r)
    qids = sorted(qid_to_rows.keys())
    print(f"[s5-train] {len(qids)} queries unicas", file=sys.stderr)

    # Split train/val por QUERY (nao por row)
    np.random.seed(42)
    qids_shuffled = list(qids)
    np.random.shuffle(qids_shuffled)
    n_val = max(1, int(len(qids_shuffled) * args.val_fraction))
    val_qids = set(qids_shuffled[:n_val])
    train_qids = set(qids_shuffled[n_val:])
    print(
        f"[s5-train] split: {len(train_qids)} train, {len(val_qids)} val",
        file=sys.stderr,
    )

    # Monta arrays para LightGBM
    def build_arrays(qid_set: set[str]):
        X: list[list[float]] = []
        y: list[int] = []
        groups: list[int] = []
        for qid in sorted(qid_set):
            qrows = qid_to_rows[qid]
            for r in qrows:
                X.append([r[fc] for fc in feature_cols])
                y.append(r["relevance"])
            groups.append(len(qrows))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), np.array(groups)

    X_train, y_train, g_train = build_arrays(train_qids)
    X_val, y_val, g_val = build_arrays(val_qids)
    print(
        f"[s5-train] train: X={X_train.shape}, y={y_train.shape}, groups={len(g_train)}",
        file=sys.stderr,
    )
    print(
        f"[s5-train] val: X={X_val.shape}, y={y_val.shape}, groups={len(g_val)}",
        file=sys.stderr,
    )

    # Treina LambdaMART
    print(f"[s5-train] treinando LambdaMART...", file=sys.stderr)
    train_data = lgb.Dataset(X_train, label=y_train, group=g_train,
                              feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, group=g_val,
                            feature_name=feature_cols, reference=train_data)

    model = lgb.train(
        LGB_PARAMS,
        train_data,
        num_boost_round=NUM_ROUNDS,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING),
            lgb.log_evaluation(period=20),
        ],
    )

    # Salva
    model.save_model(args.output)
    print(f"[s5-train] modelo salvo em {args.output}", file=sys.stderr)

    # Feature importance
    importances = model.feature_importance(importance_type="gain")
    print(f"[s5-train] feature importance (gain):", file=sys.stderr)
    for fc, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
        print(f"  {fc:25s} {imp:>10.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
