# Submissao 4 - BM25 + RM3 Query Expansion

## Hipotese

A s3 v4 (field weighting com pesos 5/1/3) chegou a nDCG=0.40518, batendo
em um plateau. A hipotese desta submissao eh que **expandir as queries
com termos relacionados dos documentos top-K iniciais** pode aumentar
recall e melhorar o ranking, especialmente para queries curtas (a media
das queries do dataset tem 3-5 termos).

Especificamente, testamos **RM3 (Relevance Model 3)**, um modelo
classico de pseudo-relevance feedback (PRF). RM3 funciona em 2 etapas:

1. Roda a query original via BM25 → pega top-K documentos
2. Extrai termos importantes desses docs (tf-idf based)
3. Adiciona os termos a query com peso ajustavel
4. Roda a query expandida → ranking final

## Por que faz sentido para entity search

Queries do Kaggle como "vietnam war movie" sao curtas e podem se
beneficiar da expansao. Por exemplo, RM3 pode adicionar termos como
"film", "cinema", "documentary" aos top-docs, melhorando recall sobre
documentos que usam essas variacoes.

**Risco:** entity queries podem sofrer **drift semantico** se o feedback
incluir termos genericos demais ("history", "country", etc).

## Setup

RM3 **requer indice com docvectors**, que NAO foi gerado nas submissoes
anteriores. Por isso, antes de rodar qualquer variante, precisa
reindexar o JSONL v4 (melhor field weighting) com as flags:
`--storePositions --storeDocvectors --storeRaw`.

Ver `variants.md` para os comandos completos.

### Etapas (uma vez)

1. **Regerar JSONL** com pesos (5, 1, 3) → ~5 min
2. **Reindexar** com docvectors → ~25-30 min

Total de setup: **~30 min**. Depois, cada variante RM3 leva **~3-5 min**.

## Variantes

Ver `variants.md`. Resumo:

- **v1**: (fb_terms=10, fb_docs=10, orig_weight=0.5) — padrao Anserini
- **v2**: (10, 5, 0.5) — menos docs feedback
- **v3**: (20, 10, 0.5) — mais termos expansao
- **v4**: (10, 10, 0.7) — mais peso na query original (conservador)
- **v5**: (10, 10, 0.3) — mais peso na expansao (agressivo)

## Como executar

Ver `variants.md` para os comandos prontos copy-paste de cada variante.

## Observacoes

- Entity search com queries curtas eh um caso dificil para PRF — o
  feedback eh pequeno e pode introduzir ruido. Esperamos ganho modesto
  (~+0.01 a +0.03 sobre s3 v4) ou ate piora se RM3 sofrer drift.
- A submissao "s4" no relatorio final eh a melhor variante de RM3.
