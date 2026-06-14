# Submissao 1 - BM25 Baseline (Disjunctive DAAT)

## Hipotese

Quao longe a implementacao "from scratch" do PA2 vai sobre o corpus do
RC, sem qualquer otimizacao adicional? Esta submissao estabelece o
**baseline** contra o qual todas as proximas (Pyserini, field weighting,
densos, hybrid) serao comparadas.

## Estrategias

### Indexacao

Reutiliza diretamente o indice gerado no PA2 (`data/indexes/index_full3/`):

- **Tokenizacao**: `nltk.word_tokenize` + lowercase
- **Normalizacao**: remocao de stopwords NLTK, filtro de tamanho
  (2-40 chars), descarte de tokens puramente numericos, stemming com
  Snowball
- **Estrutura**: indice invertido + lexicon + document_index (PA2)

Sem reindexacao — economia de ~36 minutos de CPU.

### Query Processing

- **Pre-processamento**: mesmo pipeline do indexador (simetria)
- **Matching**: **DisjunctiveDAAT** com min-heap sobre cursors
  (mudanca em relacao ao PA2, que usava ConjunctiveDAAT). Necessario
  porque o Kaggle pede top-100 por query: matching AND strict retornaria
  poucos resultados, deixando slots em branco.
- **Scoring**: BM25 (Robertson) com IDF BM25+ do Lucene, `k1=1.2`,
  `b=0.75`
- **Ranqueamento**: top-100 via min-heap; tie-break por `doc_id` menor

## Como executar

```powershell
# Da raiz do repo `rc-entity-search/`
python submissions\s1_bm25_baseline\run.py `
    -i C:\Users\gabri\Documents\GitHub\indexes-query-processor-tp\data\indexes\index_full3\ `
    -q data\kaggle\test_queries.csv `
    -o submissions\s1_bm25_baseline\submission.csv `
    2>&1 | Tee-Object -FilePath submissions\s1_bm25_baseline\run.log
```

## Observacoes

- Disjunctive matching eh significativamente mais caro que conjunctive
  (ordem de magnitude). Tempo total esperado: 2-5 minutos.
- BM25 padrao trata o documento como bag-of-words plana — title,
  keywords e text recebem peso igual. Submissao 3 vai explorar
  field weighting.
- Termos da query ausentes do lexicon sao silenciosamente pulados
  (em vez do curto-circuito do conjunctive).
