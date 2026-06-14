# Submissao 2 - Pyserini BM25 (Lucene)

## Hipotese

A implementacao "from scratch" do PA2 atingiu nDCG@100 = 0.39923. Qual o
ganho de migrar para uma engine madura e otimizada (Lucene via Pyserini),
mantendo **todos os outros parametros constantes**?

Comparacao direta:
- **Submissao 1**: BM25 disjuntivo em Python puro (implementacao PA2)
- **Submissao 2**: BM25 via Lucene (Pyserini), mesmos campos concatenados,
  mesmos parametros (k1=1.2, b=0.75)

Diferenca atribuivel a:
- Tokenizer/analyzer mais sofisticado do Lucene (StandardAnalyzer com
  rules unicode, stemming Porter mais robusto)
- Otimizacoes de query processing (WAND, MaxScore implicito)
- Provavel diferenca no tratamento de stopwords e normalizacao

## Estrategias

### Conversao do corpus

- Le `data/corpus/entities.jsonl` (formato PA2: id, title, text, keywords)
- Concatena `title + text + " ".join(keywords)` em campo unico `contents`
- Escreve `data/corpus/entities_pyserini.jsonl` no formato JsonCollection
  do Pyserini: `{"id": "...", "contents": "..."}`

### Indexacao

Comando CLI do Pyserini:
```
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --generator DefaultLuceneDocumentGenerator \
    --threads 8 \
    --input data/corpus_pyserini/ \
    --index data/indexes/pyserini_bm25/
```

(Note: o input eh um *diretorio* contendo o .jsonl, nao o arquivo direto.)

### Query Processing

- `LuceneSearcher.search(query, k=100)` - tokenizacao e BM25 do Lucene
- Sem field weighting (subm. 3)
- Sem query expansion (subm. 4)
- Sem reranker (subm. 5)

## Como executar

### Passo 1: Converter corpus (~3-5 min)

```powershell
python submissions\s2_pyserini_bm25\convert_corpus.py `
    -i data\corpus\entities.jsonl `
    -o data\corpus\entities_pyserini.jsonl
```

### Passo 2: Preparar diretorio do input para Pyserini

O Pyserini espera o JSONL dentro de um *diretorio* (le todos `.jsonl` ali).
Sugiro:

```powershell
# Move o arquivo para um subdiretorio
mkdir data\corpus\corpus_pyserini -Force
Move-Item data\corpus\entities_pyserini.jsonl data\corpus\corpus_pyserini\
```

### Passo 3: Indexar (~10-20 min)

```powershell
python -m pyserini.index.lucene `
    --collection JsonCollection `
    --generator DefaultLuceneDocumentGenerator `
    --threads 8 `
    --input data\corpus\corpus_pyserini\ `
    --index data\indexes\pyserini_bm25\ `
    2>&1 | Tee-Object -FilePath data\indexes\pyserini_bm25_log.txt
```

### Passo 4: Gerar submissao (~1-3 min)

```powershell
python submissions\s2_pyserini_bm25\run.py `
    -i data\indexes\pyserini_bm25\ `
    -q data\kaggle\test_queries.csv `
    -o submissions\s2_pyserini_bm25\submission.csv `
    2>&1 | Tee-Object -FilePath submissions\s2_pyserini_bm25\run.log
```

### Passo 5: Subir no Kaggle

Upload de `submissions\s2_pyserini_bm25\submission.csv`.

## Observacoes

- Lucene faz tokenizacao/analise diferente do nosso pipeline NLTK +
  Snowball. Em particular, o StandardAnalyzer remove menos stopwords e
  mantem mais tokens significativos
- BM25 mathematics eh identico (mesma formula), mas IDF computation pode
  ter pequenas diferencas (offset por documento, etc)
- Tempo de busca esperado: ~10-50ms/query (muito mais rapido que nossos
  300ms+ no PA2)
