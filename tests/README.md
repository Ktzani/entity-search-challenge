# Testes

Suite de testes unitarios e de integracao para o projeto.

## Como rodar

**Rodar todos os testes:**

```bash
# Da raiz do projeto
python -m unittest discover tests
```

**Rodar um arquivo especifico:**

```bash
python -m unittest tests.test_preprocessing
python -m unittest tests.test_partial_index
python -m unittest tests.test_daat
```

**Rodar com verbose (mostra cada teste):**

```bash
python -m unittest discover tests -v
```

**Rodar um teste especifico:**

```bash
python -m unittest tests.test_daat.TestConjunctiveDAAT.test_simple_intersection
```

## Estrutura

| Arquivo | O que testa |
|---|---|
| `test_preprocessing.py` | Tokenizer + Normalizer (stem, stopwords, filtros) |
| `test_io_utils.py` | Leitura/escrita binaria de postings |
| `test_memory.py` | MemoryMonitor (psutil wrapper) |
| `test_corpus_reader.py` | Leitura JSONL com tolerancia a erros |
| `test_partial_index.py` | Indexacao em memoria + dump |
| `test_merger.py` | K-way merge externo de blocks |
| `test_query.py` | Preprocessing de queries |
| `test_daat.py` | Conjunctive matching com galloping |
| `test_scorer.py` | TF-IDF e BM25 |
| `test_ranker.py` | Top-K via min-heap |
| `test_end_to_end.py` | Pipeline completo (indexer + processor) |

## Saida esperada

Quando tudo passa:

```
......................
----------------------------------------------------------------------
Ran 22 tests in 12.345s

OK
```

Quando algo falha, mostra o teste e o stack trace.

## Notas

- Os testes criam arquivos temporarios em `/tmp/` (Linux/Mac) ou
  `%TEMP%` (Windows) e limpam ao final.
- O download dos dados do NLTK acontece apenas na primeira execucao.
- Os testes sao deterministicos e independentes entre si.
