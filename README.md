# Research Challenge - Entity Search

Information Retrieval - UFMG
Gabriel Catizani Faria Oliveira

## Descricao

Implementacao para o Research Challenge de Entity Search (Kaggle:
`ir-20261-rc`). O objetivo eh produzir, para cada uma das 233 queries de
teste, um ranking dos 100 documentos mais relevantes do corpus de 4{,}6M
entidades enciclopedicas, otimizando nDCG@100.

A base de codigo eh derivada do Programming Assignment #2 (indexer SPIMI
+ DAAT + BM25), com as seguintes adaptacoes:

- `daat.py` renomeado para `conjunctive_daat.py`
- `disjunctive_daat.py`: novo modulo para matching disjuntivo
  (necessario para preencher top-100; conjunctive AND retornaria poucos
  resultados)
- `config/processor.py`: `TOP_K` ajustado para 100
- `submissions/<id>/run.py`: scripts de geracao de CSV por submissao

## Setup

```bash
python3 -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## Estrutura

```
rc-entity-search/
├── src/
│   ├── config/             # constantes (TOP_K=100)
│   ├── preprocessing/      # tokenizer, normalizer (idem PA2)
│   ├── index_store/        # leitura de inverted_index, lexicon, doc_index
│   ├── retrieval/
│   │   ├── conjunctive_daat.py    # renomeado do PA2 (preservado p/ comparacao)
│   │   ├── disjunctive_daat.py    # NOVO: matching disjuntivo
│   │   ├── query.py, scorer.py, ranker.py
│   └── utils/
├── submissions/
│   └── s1_bm25_baseline/
│       ├── run.py          # gera submission.csv
│       ├── README.md       # hipotese + resultado
│       └── submission.csv  # arquivo enviado ao Kaggle
└── data/                   # NAO versionado
    ├── kaggle/
    │   ├── test_queries.csv
    │   ├── train_queries.csv
    │   └── train_qrels.csv
    └── (indice eh reutilizado do repo do PA2 via path absoluto em -i)
```

## Reutilizando o indice do PA2

O indice gerado no PA2 (4,6M docs, 3,9M termos, 1,14 GB) eh reutilizado
diretamente para evitar reindexacao. O argumento `-i` aponta para o
diretorio do indice em outro repositorio.

## Submissoes

Cada submissao tem sua propria pasta `submissions/sN_<nome>/` contendo:

- `run.py`: script que produz `submission.csv` a partir do indice + queries
- `submission.csv`: arquivo enviado ao Kaggle
- `README.md`: hipotese, resultado nDCG@100, e justificativa das escolhas

### Submissoes planejadas

| Id | Nome                | Hipotese                                            |
| -- | ------------------- | --------------------------------------------------- |
| s1 | bm25_baseline       | BM25 disjuntivo com indexer do PA2                  |
| s2 | pyserini_bm25       | BM25 via Lucene (Pyserini) supera implementacao pura |
| s3 | field_weights       | Pesos diferentes em title/text/keywords             |
| s4 | dense_retrieval     | Embeddings densos (sentence-transformers)           |
| s5 | hybrid_rerank       | BM25 candidatos + reranker neural top-100           |

## Como rodar uma submissao

Exemplo s1:

```powershell
python submissions\s1_bm25_baseline\run.py `
    -i <CAMINHO_ABSOLUTO_DO_INDICE_DO_PA2> `
    -q data\kaggle\test_queries.csv `
    -o submissions\s1_bm25_baseline\submission.csv `
    2>&1 | Tee-Object -FilePath submissions\s1_bm25_baseline\run.log
```

O `submission.csv` resultante eh o arquivo a ser enviado para o Kaggle.

## Formato da submissao Kaggle

CSV com header `QueryId,EntityId`, contendo ate 100 entidades por query
(233 queries x 100 = 23.300 linhas no maximo), em ordem decrescente de
relevancia:

```csv
QueryId,EntityId
002,0878002
002,3056323
...
465,3411664
```
