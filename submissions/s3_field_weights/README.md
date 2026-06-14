# Submissao 3 - Field Weighting via Field Expansion

## Hipotese

A s2 (Lucene BM25 com title+text+keywords concatenados) deu nDCG=0.39774,
quase identico ao s1 (Python puro). Isso sugere que **mudar engine nao foi
a alavanca**.

Hipotese desta submissao: **em entity search, o titulo e as keywords
carregam sinal de relevancia muito mais forte que o text descritivo**.
Dar pesos diferentes por campo deve melhorar o ranking.

## Estrategia: Field Expansion

Pyserini nao tem suporte nativo bom para multifield em Python (exige
custom generator Java). Solucao alternativa: **field expansion via
repeticao**.

Em vez de campo unico `contents = title + " " + text + " " + keywords`,
construo `contents = (title * W_t) + " " + (text * W_text) + " " + (keywords * W_k)`,
onde `W_t, W_text, W_k` sao os pesos desejados.

Como BM25 e' dominado por term frequency (tf), repetir um termo no
documento aumenta seu tf e equivale (na pratica) a boost multifield
Lucene.

**Vantagens:**
- Sem codigo Java
- Mesma infraestrutura Pyserini do s2
- Multiplas variantes = multiplos JSONLs + indices (paralelo)

**Custo:**
- Cada variante reindexada do zero (~15-20 min)
- ~3-4 GB de disco por variante

## Como executar

Ver `variants.md` para os comandos completos de cada variante (v1 a v5).

### Etapas (por variante)

1. **Convert** corpus com pesos especificos:
```powershell
python submissions\s3_field_weights\convert_corpus_weighted.py `
    -i data\corpus\entities.jsonl `
    -o data\corpus\v1\entities_weighted.jsonl `
    --w-title 3 --w-text 1 --w-keywords 2
```

2. **Index** com Pyserini (o input ja' eh um diretorio):
```powershell
python -m pyserini.index.lucene `
    --collection JsonCollection `
    --generator DefaultLuceneDocumentGenerator `
    --threads 8 `
    --input data\corpus\v1\ `
    --index data\indexes\pyserini_v1\
```

3. **Search e gera submission.csv**:
```powershell
python submissions\s3_field_weights\run.py `
    -i data\indexes\pyserini_v1\ `
    -q data\kaggle\test_queries.csv `
    -o submissions\s3_field_weights\submission_v1.csv
```

4. **Sobe no Kaggle** e registra o nDCG na tabela de `variants.md`.

## Variantes

Ver `variants.md` para a lista completa com comandos.

Resumo:
- **v1**: (3, 1, 2) — moderado
- **v2**: (5, 1, 3) — titulo dominante
- **v3**: (2, 1, 4) — keywords dominante
- **v4**: (4, 0, 3) — sem text
- **v5**: (5, 1, 5) — agressivo
- **v6** *(opcional)*: (1, 1, 1) — controle (deve dar identico ao s2)

A melhor variante eh a que vai como "s3" no relatorio final.

## Observacoes

- BM25 satura tf via `k1*(1-b+b*|d|/avgdl)`, entao repetir 10x nao
  multiplica score por 10 — ha diminishing returns. Por isso variantes
  com pesos altos (5+) podem nao ajudar tanto quanto se esperaria
  linearmente.
- O comprimento do documento (`|d|`) aumenta com a repeticao, o que
  **diminui** o BM25 do termo. Mas como TODOS os termos do campo
  repetido tambem aumentam o tf, o efeito liquido ainda eh positivo.
- Se v4 (sem text) der bem, isso indica que o `text` esta mais atrapalhando
  que ajudando (ruido > sinal em corpus enciclopedico curto).
