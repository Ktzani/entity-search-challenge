"""
eval_ndcg.py - Avaliacao LOCAL de nDCG@100, calibrada para reproduzir o Kaggle.

Validado: a submissao v4 deu nDCG@100=0.42671 aqui (5-fold CV nas train queries)
vs 0.42696 no Kaggle -> diferenca de 0.00025. Logo, da pra iterar localmente e
so subir o que promete.

FORMULA (a que casa com o Kaggle):
  - ganho LINEAR: gain(rel) = rel  (qrels sao graduados: rel in {1, 2})
  - desconto: 1/log2(rank+1), rank começando em 1
  - IDCG: ordenacao ideal de TODOS os qrels da query (penaliza recall baixo)
  - media simples sobre todas as queries que tem qrels (ausente na submissao = 0)

USO:
  # Avaliar uma submissao (QueryId,EntityId em ordem de ranking) contra qrels:
  python submissions/s5_ltr/eval_ndcg.py \\
      --submission submissions/s5_ltr/submission_ltr_v4.csv \\
      --qrels data/kaggle/train_qrels.csv

  # Comparar contra um baseline:
  python submissions/s5_ltr/eval_ndcg.py --submission A.csv --baseline B.csv \\
      --qrels data/kaggle/train_qrels.csv
"""
import argparse
import csv
import math
import sys
from collections import defaultdict, OrderedDict

K = 100


def read_qrels(path: str) -> dict[str, dict[str, int]]:
    """{qid: {docid: rel}} ignorando rel<=0 e o header."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            if len(row) < 3:
                continue
            try:
                rel = int(row[2])
            except ValueError:
                continue  # header residual ou linha invalida
            if rel > 0:
                qrels[row[0].strip()][row[1].strip()] = rel
    return dict(qrels)


def read_submission(path: str) -> "OrderedDict[str, list[str]]":
    """{qid: [docid, ...]} na ordem do arquivo (= ordem de ranking)."""
    ranking: "OrderedDict[str, list[str]]" = OrderedDict()
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        next(rd, None)  # header QueryId,EntityId
        for row in rd:
            if len(row) < 2:
                continue
            ranking.setdefault(row[0].strip(), []).append(row[1].strip())
    return ranking


def ndcg_at_k(ranked_docids: list[str], rel_map: dict[str, int], k: int = K) -> float:
    """nDCG@k com ganho linear (gain=rel). rel_map = {docid: rel} da query."""
    dcg = 0.0
    for i, d in enumerate(ranked_docids[:k], start=1):
        rel = rel_map.get(d, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    ideal = sorted(rel_map.values(), reverse=True)
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal[:k], 1) if r > 0)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(ranking: dict[str, list[str]], qrels: dict[str, dict[str, int]],
             k: int = K) -> tuple[float, dict[str, float]]:
    """Retorna (nDCG@k medio, {qid: ndcg}) sobre TODAS as queries dos qrels."""
    per_q: dict[str, float] = {}
    for qid, rel_map in qrels.items():
        per_q[qid] = ndcg_at_k(ranking.get(qid, []), rel_map, k)
    mean = sum(per_q.values()) / len(per_q) if per_q else 0.0
    return mean, per_q


def main() -> None:
    ap = argparse.ArgumentParser(description="nDCG@100 local calibrado p/ Kaggle.")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--baseline", help="submissao de baseline p/ comparacao pareada")
    ap.add_argument("-k", type=int, default=K)
    args = ap.parse_args()

    qrels = read_qrels(args.qrels)
    sub = read_submission(args.submission)
    mean, per_q = evaluate(sub, qrels, args.k)
    matched = sum(1 for q in qrels if q in sub)
    print(f"submission: {args.submission}")
    print(f"  nDCG@{args.k} = {mean:.5f}  ({matched}/{len(qrels)} queries cobertas)")

    if args.baseline:
        base = read_submission(args.baseline)
        bmean, bper_q = evaluate(base, qrels, args.k)
        print(f"baseline:   {args.baseline}")
        print(f"  nDCG@{args.k} = {bmean:.5f}")
        diffs = [per_q[q] - bper_q[q] for q in qrels]
        n = len(diffs)
        md = sum(diffs) / n
        var = sum((d - md) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n) if var > 0 else 0.0
        t = md / se if se > 0 else float("nan")
        wins = sum(1 for d in diffs if d > 1e-9)
        losses = sum(1 for d in diffs if d < -1e-9)
        print(f"delta (sub - baseline): {md:+.5f}  t={t:+.2f}  "
              f"wins/ties/losses={wins}/{n-wins-losses}/{losses}")


if __name__ == "__main__":
    main()
