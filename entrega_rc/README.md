# Entrega — IR Research Challenge (Entity Search, Kaggle ir-20261-rc)

Pacote com (2) os CSVs das 5 submissões mais efetivas e (3) o código-fonte que as gera.
O relatório PDF (item 1) é entregue à parte.

## As 5 submissões mais efetivas

| # | Arquivo (csv/) | ID no Kaggle | nDCG@100 |
|---|----------------|--------------|----------|
| 1 | `submission_ltr_hybrid_ce.csv` | LambdaMART LTR hybrid+CE | **0.54427** |
| 2 | `submission_ltr_hybrid.csv`    | LambdaMART LTR hybrid | 0.50981 |
| 3 | `submission_ltr_v9dual.csv`    | LambdaMART LTR v9-dual | 0.48087 |
| 4 | `submission_ltr_v7e5.csv`      | LambdaMART LTR v7-e5 | 0.47610 |
| 5 | `submission_ltr_v7base.csv`    | LambdaMART LTR v7-base | 0.47376 |

## Estrutura

```
entrega_rc/
├── README.md
├── csv/                 # item 2: os 5 CSVs (nomes originais das submissões)
└── src/                 # item 3: código-fonte que gera as 5 melhores, em estágios
    ├── 01_indexing/
    │   ├── build_corpus_index.py     # índice denso faiss do corpus inteiro (4.6M)
    │   └── build_dense_cache.py      # embeddings de queries + candidatos (bi-encoder)
    ├── 02_retrieval/
    │   ├── dense_retrieve.py         # recuperação densa top-k (faiss)
    │   └── hybrid_retrieve.py        # pool híbrido BM25∪denso + features
    ├── 03_features/
    │   ├── features_base.py          # módulo de features lexicais/conteúdo
    │   ├── extract_features.py       # features esparsas + content-aware (top-100 BM25)
    │   ├── add_dense_feature.py      # cosseno bi-encoder como feature
    │   └── add_ce_feature.py         # cross-encoder (reranker) como feature
    ├── 04_training/
    │   └── train_ltr.py              # treino LambdaMART (referência; predictors treinam inline)
    ├── 05_prediction/
    │   ├── predict_ensemble_dense.py      # ensemble + 1 feature densa  → v7-e5, v7-base
    │   ├── predict_ensemble_multidense.py # ensemble + N densas         → v9-dual
    │   └── predict_from_features.py       # ensemble genérico           → hybrid, hybrid+CE
    └── 06_evaluation/
        ├── eval_ndcg.py              # nDCG@100 de uma submissão vs qrels (calibrado p/ Kaggle)
        └── eval_ltr_cv.py            # nDCG@100 sem leakage (k-fold CV)
```

> As outras submissões (BM25, field weighting, RM3, LTR v3/v4/v5 e experimentos
> negativos) ficam apenas comentadas no relatório — só o código das **5 melhores**
> está incluído aqui.

**Mapa submissão → scripts usados** (tudo presente em `src/`):

| Submissão | scripts (na ordem de execução) |
|---|---|
| v7-e5 / v7-base | `03_features/extract_features` → `01_indexing/build_dense_cache` → `03_features/add_dense_feature` → `05_prediction/predict_ensemble_dense` |
| v9-dual | idem + 2º `build_dense_cache` (bge-large) + 2º `add_dense_feature` → `05_prediction/predict_ensemble_multidense` |
| hybrid | `01_indexing/build_corpus_index` → `02_retrieval/dense_retrieve` → `02_retrieval/hybrid_retrieve` → `05_prediction/predict_from_features` |
| hybrid+CE (melhor) | idem hybrid + `03_features/add_ce_feature` → `05_prediction/predict_from_features` |

Módulos compartilhados (`features_base.py`, `add_dense_feature.py`) vivem em
`03_features/`; scripts em outras pastas os importam via um shim de `sys.path`.

## Pré-requisitos
- Python (venv): `pyserini`, `lightgbm`, `numpy`, `sentence-transformers`,
  `torch` (CUDA), `faiss-cpu`.
- Dados esperados na raiz do repositório (baixar da página do Kaggle `ir-20261-rc`):
  `data/corpus/entities.jsonl`, `data/kaggle/{train_queries,test_queries,train_qrels}.csv`,
  e os índices `data/indexes/pyserini_bm25/` e `data/indexes/pyserini_field_weights_v4_rm3/`.
- Rodar com a **CWD na raiz do repositório**.

---

## Como gerar cada submissão

### Comum (features esparsas + content-aware) — usado por v7-e5, v7-base, v9-dual
```
python src/03_features/extract_features.py -q data/kaggle/train_queries.csv \
  --qrels data/kaggle/train_qrels.csv --i-bm25 data/indexes/pyserini_bm25/ \
  --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \
  --corpus data/corpus/entities.jsonl -o features/train_features_v4.csv
```

### #4 v7-e5 (0.47610) / #5 v7-base (0.47376) — dense bi-encoder como feature
```
# e5-large-v2 (prefixos query:/passage:); v7-base: --model BAAI/bge-base-en-v1.5 sem --doc-prefix
python src/01_indexing/build_dense_cache.py --features features/train_features_v4.csv \
  --test-sub csv/submission_ltr_v7base.csv \
  --train-queries data/kaggle/train_queries.csv --test-queries data/kaggle/test_queries.csv \
  --corpus data/corpus/entities.jsonl --model intfloat/e5-large-v2 \
  --query-prefix "query: " --doc-prefix "passage: " --out-dir dense_e5/
python src/03_features/add_dense_feature.py -i features/train_features_v4.csv \
  --dense-dir dense_e5/ --split train -o features/train_features_v7e5.csv
python src/05_prediction/predict_ensemble_dense.py -q data/kaggle/test_queries.csv \
  --features features/train_features_v7e5.csv --dense-dir dense_e5/ \
  --i-bm25 data/indexes/pyserini_bm25/ --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \
  --corpus data/corpus/entities.jsonl -o csv/submission_ltr_v7e5.csv
```

### #3 v9-dual (0.48087) — dois encoders complementares (e5 + bge-large)
```
# gerar dense_large/ (build_dense_cache com BAAI/bge-large-en-v1.5); depois:
python src/03_features/add_dense_feature.py -i features/train_features_v7e5.csv \
  --dense-dir dense_large/ --split train --col-name f_dense_bge \
  -o features/train_features_v9dual.csv
python src/05_prediction/predict_ensemble_multidense.py -q data/kaggle/test_queries.csv \
  --features features/train_features_v9dual.csv \
  --dense dense_e5/:f_dense_sim --dense dense_large/:f_dense_bge \
  --i-bm25 data/indexes/pyserini_bm25/ --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \
  --corpus data/corpus/entities.jsonl -o csv/submission_ltr_v9dual.csv
```

### #2 hybrid (0.50981) — RETRIEVAL híbrido (denso full-corpus + BM25)
```
python src/01_indexing/build_corpus_index.py --corpus data/corpus/entities.jsonl \
  --model BAAI/bge-small-en-v1.5 --out-dir corpus_index/
python src/02_retrieval/dense_retrieve.py --index-dir corpus_index/ \
  --query-emb dense/query_emb_train.npz --topk 200 -o features/dense_cands_train.csv
python src/02_retrieval/dense_retrieve.py --index-dir corpus_index/ \
  --query-emb dense/query_emb_test.npz  --topk 200 -o features/dense_cands_test.csv
python src/02_retrieval/hybrid_retrieve.py -q data/kaggle/train_queries.csv \
  --qrels data/kaggle/train_qrels.csv --i-bm25 data/indexes/pyserini_bm25/ \
  --i-fw data/indexes/pyserini_field_weights_v4_rm3/ --corpus data/corpus/entities.jsonl \
  --dense-cands features/dense_cands_train.csv --query-emb dense/query_emb_train.npz \
  --corpus-index corpus_index/ --bm25-topk 100 --dense-topk 200 \
  -o features/train_features_hybrid.csv
python src/02_retrieval/hybrid_retrieve.py -q data/kaggle/test_queries.csv \
  --i-bm25 data/indexes/pyserini_bm25/ --i-fw data/indexes/pyserini_field_weights_v4_rm3/ \
  --corpus data/corpus/entities.jsonl --dense-cands features/dense_cands_test.csv \
  --query-emb dense/query_emb_test.npz --corpus-index corpus_index/ \
  --bm25-topk 100 --dense-topk 200 -o features/test_features_hybrid.csv
python src/05_prediction/predict_from_features.py --train features/train_features_hybrid.csv \
  --test features/test_features_hybrid.csv -o csv/submission_ltr_hybrid.csv
```

### #1 hybrid + cross-encoder (0.54427) — MELHOR
```
python src/03_features/add_ce_feature.py -i features/train_features_hybrid.csv \
  --queries data/kaggle/train_queries.csv --corpus data/corpus/entities.jsonl \
  -o features/train_features_hybrid_ce.csv
python src/03_features/add_ce_feature.py -i features/test_features_hybrid.csv \
  --queries data/kaggle/test_queries.csv --corpus data/corpus/entities.jsonl \
  -o features/test_features_hybrid_ce.csv
python src/05_prediction/predict_from_features.py --train features/train_features_hybrid_ce.csv \
  --test features/test_features_hybrid_ce.csv -o csv/submission_ltr_hybrid_ce.csv
```

## Avaliação local (06_evaluation)
nDCG@100 calibrado p/ Kaggle (ganho linear, IDCG sobre todos os qrels):
```
python src/06_evaluation/eval_ltr_cv.py -i features/train_features_hybrid_ce.csv \
  --qrels data/kaggle/train_qrels.csv          # CV sem leakage
python src/06_evaluation/eval_ndcg.py --submission <sub.csv> --qrels data/kaggle/train_qrels.csv
```

## Notas
- Re-ranking: **LambdaMART** (LightGBM, `lambdarank`), ensemble de 6 seeds.
- `dense/query_emb_{train,test}.npz` (usados no híbrido) são as query-embeddings
  bge-small geradas por `build_dense_cache.py`.


# Dados (não incluídos no pacote por tamanho)

Os scripts esperam estes dados na raiz do repositório (caminhos relativos à CWD).
Baixe-os da página de dados da competição no Kaggle (`ir-20261-rc`).

## Esperado
```
data/
├── corpus/
│   └── entities.jsonl                 # 4.641.784 entidades {id,title,text,keywords} (~2 GB)
├── kaggle/
│   ├── train_queries.csv              # 234 queries de treino
│   ├── test_queries.csv               # 233 queries de teste
│   └── train_qrels.csv                # 8.202 julgamentos (rel ∈ {1,2})
└── indexes/
    ├── pyserini_bm25/                 # índice BM25 (Lucene/Pyserini)
    └── pyserini_field_weights_v4_rm3/ # índice com pesos de campo (title=5,text=1,kw=3) p/ RM3
```

## Artefatos intermediários gerados pelo pipeline (também grandes, não incluídos)
- `corpus_index/` — índice denso faiss do corpus inteiro (bge-small) + `embs`/`docids`.
- `dense/`, `dense_e5/`, `dense_large/` — caches de embeddings (queries + candidatos).
- `features/` — CSVs de features (lexicais, densas, híbridas, +CE).

Esses diretórios são recriados pelos scripts em `src/` (ver `../README.md`).
