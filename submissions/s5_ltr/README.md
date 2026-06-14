# Submissão 5 — Learning to Rank (LambdaMART)

Re-ranking supervisionado dos `train_qrels.csv` combinando os sinais das
submissões anteriores (BM25, field weights, RM3) + features de conteúdo e
densas, via **LambdaMART** (LightGBM). Métrica: **nDCG@100**.

## Resultados no Kaggle

| Variante | Descrição | nDCG@100 |
| -------- | --------- | -------- |
| s2 BM25 | baseline engine | 0.397 |
| s3 fields | field weighting (title=5, text=1, kw=3) | 0.405 |
| s4 RM3 | query expansion | 0.40648 |
| v3 LTR | features ricas + random negatives | 0.288 (regrediu) |
| **v4 LTR** | bug de content corrigido + setup re-ranking | 0.42696 |
| **v5 LTR** | ensemble de 6 seeds (melhor config) | 0.43397 |
| v7 LTR | v5 + feature densa (BGE-small 384d) | CV 0.4685 |
| v7-base / large / e5 | dense BGE-base / BGE-large / e5-large-v2 | CV 0.4755 / 0.4800 / 0.4841 |
| **v9-dual LTR** | **v5 + dense e5-large + bge-large (2 features)** | **CV 0.4873 — a subir** |

v6 (união de candidatos) **descartado** — negativo em CV: recall não era o
gargalo. A **feature densa** deu o maior salto: **+0.040 em CV (8/8)**, via
ranking semântico fino dentro de cada query. Varredura de embeddings (todos 8/8):
bge-small→base→large→e5-large, ganhos decrescentes mas robustos; **combinar e5 +
bge-large** (famílias complementares) deu mais +0.0031. Features de **interação**
densas (rank/norm/produtos) = **nulo** (+0.0007, 4/8): o GBDT já captura via splits.

## Estrutura da pasta

```
s5_ltr/
├── README.md
├── submission_ltr_v4.csv          # deliverables Kaggle
├── submission_ltr_v5.csv          # <- melhor (0.43397)
│
├── pipeline/                      # extração → treino → predição
│   ├── _features_v4.py            #   features compartilhadas (treino==predição)
│   ├── 01_extract_features_v4.py  #   extrai features (re-ranking, top-100 BM25)
│   ├── 01_extract_features_v6.py  #   variante união de candidatos (experimento -)
│   ├── 02_train_ltr.py            #   treina LambdaMART (config padrão)
│   ├── 03_predict_test_v4.py      #   predição single-model
│   ├── 04_predict_ensemble_v5.py  #   predição ENSEMBLE (melhor submissão)
│   ├── build_dense_cache.py       #   embeddings densos (f_dense_sim) [WIP]
│   └── add_dense_feature.py       #   adiciona f_dense_sim a um CSV [WIP]
│
├── eval/                          # avaliação local (calibrada) + tuning
│   ├── eval_ndcg.py               #   nDCG@100 de uma submissão CSV vs qrels
│   ├── eval_ltr_cv.py             #   nDCG@100 sem leakage (k-fold CV)
│   ├── tune_ltr.py                #   random search de hiperparâmetros + ensemble
│   ├── _robust_check.py           #   robustez multi-split: v4 vs best vs ensemble
│   └── _compare_two.py            #   compara 2 CSVs de features (ex: v4 vs v7)
│
├── features/   # CSVs de features + oof (intermediários)
├── models/     # modelos treinados (model_v4.txt)
├── logs/       # logs de execução
└── archive/    # versões antigas v1/v2/v3 (scripts, models, submissões)
```

> Imports são autocontidos por grupo: `pipeline/` só importa `_features_v4`;
> `eval/` só importa `eval_ndcg`/`tune_ltr`. Rode sempre com a CWD na raiz do
> repositório (caminhos de dados são relativos a ela).

## Avaliação local (sem subir ao Kaggle)

O CV local é **calibrado**: v4 deu CV 0.42671 vs Kaggle 0.42696 (Δ 0.0003).
nDCG@100 com **ganho linear** (gain=rel), IDCG sobre todos os qrels.

```powershell
# nDCG de uma submissão CSV (treino) vs qrels:
.venv\Scripts\python.exe submissions\s5_ltr\eval\eval_ndcg.py `
    --submission X.csv --qrels data\kaggle\train_qrels.csv [--baseline Y.csv]

# nDCG sem leakage de um conjunto de features (k-fold CV):
.venv\Scripts\python.exe submissions\s5_ltr\eval\eval_ltr_cv.py `
    -i submissions\s5_ltr\features\train_features_v4.csv `
    --qrels data\kaggle\train_qrels.csv
```

## Pipeline da melhor submissão (v5)

```powershell
# 1) Extrair features (~25s)
.venv\Scripts\python.exe submissions\s5_ltr\pipeline\01_extract_features_v4.py `
    -q data\kaggle\train_queries.csv --qrels data\kaggle\train_qrels.csv `
    --i-bm25 data\indexes\pyserini_bm25\ `
    --i-fw data\indexes\pyserini_field_weights_v4_rm3\ `
    --corpus data\corpus\entities.jsonl `
    -o submissions\s5_ltr\features\train_features_v4.csv

# 2) Treinar + prever ENSEMBLE -> submissão (~26s)
.venv\Scripts\python.exe submissions\s5_ltr\pipeline\04_predict_ensemble_v5.py `
    -q data\kaggle\test_queries.csv `
    --features submissions\s5_ltr\features\train_features_v4.csv `
    --i-bm25 data\indexes\pyserini_bm25\ `
    --i-fw data\indexes\pyserini_field_weights_v4_rm3\ `
    --corpus data\corpus\entities.jsonl `
    -o submissions\s5_ltr\submission_ltr_v5.csv
```

## Notas técnicas

- **Fonte das features de conteúdo**: corpus original `data/corpus/entities.jsonl`
  (`{id,title,text,keywords}`), lido por streaming. Os índices Pyserini não
  guardam title/text/keywords separados (bug que zerava as features no v3).
- **Setup re-ranking**: candidatos = top-100 BM25 (∪ positivos no treino, p/
  rótulo). Alinha distribuição treino↔teste — corrigir isso (não só o bug de
  content) foi o que recuperou a regressão do v3.
- **16 features**, todas com gain > 0. Binários grosseiros (title_exact_match,
  title_full_match) foram removidos por redundância com as versões contínuas.
- **Ensemble**: 6 seeds do config tunado (lr=0.03, leaves=31, depth=5, ff=0.9,
  bf=0.6, nr=400). Ganho robusto: +0.0032 em CV, positivo em 8/8 splits.

## Feature densa (v7) — pipeline

Adiciona `f_dense_sim` (cosseno query↔entidade via embeddings BGE
`bge-small-en-v1.5`, GPU). Requer `sentence-transformers` + torch-CUDA.

```powershell
# 1) cache de embeddings (uma vez, ~1min GPU)
.venv\Scripts\python.exe submissions\s5_ltr\pipeline\build_dense_cache.py `
    --features submissions\s5_ltr\features\train_features_v4.csv `
    --test-sub submissions\s5_ltr\submission_ltr_v5.csv `
    --train-queries data\kaggle\train_queries.csv `
    --test-queries data\kaggle\test_queries.csv `
    --corpus data\corpus\entities.jsonl --out-dir submissions\s5_ltr\dense\

# 2) adiciona f_dense_sim ao CSV de features (treino)
.venv\Scripts\python.exe submissions\s5_ltr\pipeline\add_dense_feature.py `
    -i submissions\s5_ltr\features\train_features_v4.csv `
    --dense-dir submissions\s5_ltr\dense\ --split train `
    -o submissions\s5_ltr\features\train_features_v7.csv

# 3) valida o ganho em CV antes de subir
.venv\Scripts\python.exe submissions\s5_ltr\eval\_compare_two.py `
    submissions\s5_ltr\features\train_features_v4.csv `
    submissions\s5_ltr\features\train_features_v7.csv v4 v7

# 4) gera a submissão (ensemble + dense)
.venv\Scripts\python.exe submissions\s5_ltr\pipeline\05_predict_ensemble_v7.py `
    -q data\kaggle\test_queries.csv `
    --features submissions\s5_ltr\features\train_features_v7.csv `
    --dense-dir submissions\s5_ltr\dense\ `
    --i-bm25 data\indexes\pyserini_bm25\ --i-fw data\indexes\pyserini_field_weights_v4_rm3\ `
    --corpus data\corpus\entities.jsonl -o submissions\s5_ltr\submission_ltr_v7.csv
```
