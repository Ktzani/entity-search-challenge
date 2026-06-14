# Relatório — Learning to Rank para Entity Search (Kaggle ir-20261-rc)

## Contexto
- **Tarefa**: Entity Search, corpus de 4.641.784 entidades (`{id, title, text, keywords}`).
  234 train queries / 8202 qrels graduados (rel ∈ {1,2}); 233 test queries.
  Métrica: **nDCG@100**. Deadline 15/jun/2026.
- **Plateau inicial** (não-supervisionado): BM25=0.398, field weighting (5,1,3)=0.405,
  RM3 (10,10,0.8)=0.40648 — todos tunados, mas presos em ~0.40.
- **Aposta s5**: re-ranking supervisionado com **LambdaMART** + features densas.

## Resultados no Kaggle (confirmados)

| Submissão | Descrição | nDCG@100 |
| --------- | --------- | -------- |
| s4 RM3 (a bater) | query expansion | 0.40648 |
| v3 LTR (quebrado) | features ricas + random negatives | 0.288 ❌ |
| v4 LTR | bug content corrigido + setup re-ranking | 0.42696 |
| v5 LTR | ensemble 6 seeds | 0.43397 |
| v7 (dense BGE-small) | +feature densa | 0.46366 |
| v7-large (BGE-large) | dense 1024d | 0.46966 |
| v7-base (BGE-base) | dense 768d | 0.47376 |
| v7-e5 (e5-large-v2) | dense 1024d | 0.47610 |
| v9-dual (e5 + BGE-large) | 2 features densas (re-rank top-100) | 0.48087 |
| híbrido (retrieval denso) | BM25@100 ∪ denso@200 → re-rank | 0.50981 |
| **híbrido + cross-encoder** | **+ reranker bge-reranker-large** | **0.54427** ✅ |

**Progressão: 0.40648 → 0.54427 no Kaggle (+34% sobre o plateau).**

## 1. Diagnóstico do v3 (0.288)
Duas falhas; a premissa inicial de correção estava errada:
- **(a)** Content features zeradas — liam `title/text/keywords` de índices que **não os guardam separados** (BM25 só tem `id`; field-weighted só tem `contents` concatenado). Fonte real: corpus original `entities.jsonl` (lido por streaming).
- **(b) Causa principal**: random negatives criaram **descasamento treino/teste** — `f_bm25_rank`/`f_doc_length` dominaram com sinais ausentes no teste. Overfit (val 0.733 vs Kaggle 0.288).

## 2. v4 (→ 0.42696) e v5 (→ 0.43397)
- v4: content features do corpus; **setup re-ranking** (candidatos = top-100 BM25, alinha treino↔teste — *foi isso, não só o bug, que recuperou a regressão*); 16 features, todas gain>0 (removidos binários redundantes); features novas (`title_jaccard` virou 2ª mais importante, `bigram_overlap`, `title_match_pos`, `query_coverage`).
- v5: **ensemble de 6 seeds** do melhor config tunado. Ganho robusto (+0.0032 em CV, 8/8 splits).

## 3. Avaliação local calibrada (metodologia central)
Harness para medir nDCG@100 **localmente** (ganho linear, IDCG sobre todos os qrels),
evitando gastar submissões. **k-fold CV sem leakage** + **teste de robustez sobre 8
splits de fold** (distingue ganho real de ruído).

**Calibração**: v4 CV 0.42671 vs Kaggle 0.42696 (Δ 0.0003). O CV nas 230 train
queries é proxy quase perfeito para **saltos grandes**. (Ver §6 a ressalva fina.)

## 4. Feature densa — o maior salto (→ 0.46–0.48)
`f_dense_sim` = cosseno query↔entidade (embeddings, GPU). **+0.040 em CV (8/8 splits)**.
- **Insight**: a separação GLOBAL relevante/não-rel era fraca (+0.029 nas médias), mas
  o ganho vem da separação **dentro de cada query** (o que o LambdaMART pairwise otimiza).
- **Varredura de modelos** (CV → Kaggle): bge-small 0.46366 → bge-base 0.47376 →
  bge-large 0.46966 → e5-large 0.47610. Modelos maiores ajudam com retornos decrescentes.
- **Embeddings complementares**: usar **e5-large + bge-large** como DUAS features densas
  (não escolher uma) deu o melhor resultado: **0.48087**. Famílias diferentes capturam
  semânticas distintas.

## 5. Resultados NEGATIVOS (enriquecem a discussão)
- **União de candidatos (v6)**: BM25∪RM3∪fields top-100 → delta CV −0.00005, variância
  dobrou. **Recall não era o gargalo no top-100**; só adicionou distratores.
- **Features de interação densas** (rank/norm/produtos por query): +0.0007 (4/8) = nulo.
  O GBDT já captura interações via splits.

## 6. Calibração CV↔Kaggle — limites (descoberta sofisticada)
Com os 5 densos no Kaggle, a CV ficou **otimista** (gap −0.002 a −0.010) e **errou o
ranking fino**: CV dizia bge-large(0.4800) > bge-base(0.4755), mas no Kaggle
**base(0.47376) > large(0.46966)**. bge-large foi o pior overfit de CV (−0.010).
**Conclusão**: a CV é confiável para **saltos grandes** (dense >> sem-dense; híbrido
CV 0.539 → Kaggle 0.510, gap −0.029 mas ganho real enorme), mas diferenças <0.005
entre variantes estão **dentro do ruído de transferência (~0.006–0.029)** — só o
upload resolve. Por isso uploads de triangulação importam.

## 7. O verdadeiro teto: RECALL do 1º estágio → retrieval híbrido (o maior salto)
Medições no treino:
- **recall@100 do BM25 = 0.52** (mediana 0.50): metade dos relevantes nem está no pool
  de candidatos → o re-ranker nunca os alcança.
- recall@1000 = 0.75; **25% dos relevantes ficam fora até do top-1000** (lexical-mismatch puro).
- **Índice não é o problema**: analyzer saudável (Porter stemming + stopwords). O índice
  **field-weighted recupera ≈ BM25** (0.539 vs 0.521 @100; empatados @1000) — bom como
  *feature* de re-ranking, mas não destrava recall. Todo método **esparso** satura em
  ~0.78 (união dos 3 @1000): compartilham os mesmos pontos cegos lexicais.
- **Retrieval híbrido (o maior salto — CONFIRMADO no Kaggle)**: índice **denso faiss do
  corpus inteiro** (4.6M, bge-small). Pool híbrido = BM25@100 ∪ denso@200 eleva o recall
  de **0.52 → 0.73** (denso e BM25 acham relevantes DIFERENTES). nDCG@100 CV:
  v7 top-100 = 0.4685 → **híbrido = 0.5387 (+0.070, 8/8 splits)**; **Kaggle = 0.50981**
  (vs v9dual 0.48087, +0.029). É a técnica "dense encoder" do enunciado atacando o
  gargalo medido — e funcionou. *Upgrades não explorados*: embedding mais forte
  (e5/bge-large) no híbrido e denso top-500.

## Notas metodológicas (o que está dando certo)
- **Iterar em CV local antes de subir** — viabilizou a varredura de embeddings e os
  experimentos negativos sem queimar submissões.
- **Robustez multi-split** distingue ganho real de ruído (ensemble 8/8; interação 4/8).
- **Resultados negativos informam**: união esparsa e features de interação descartadas
  com fundamento; recall (não re-ranking) identificado como o teto restante.

## Stack
Pyserini (Lucene/BM25/RM3), LightGBM (LambdaMART), sentence-transformers (BGE, e5),
faiss, torch-CUDA (RTX 3060). Pipeline em `submissions/s5_ltr/{pipeline,eval,...}`.

## Seleção para o relatório formal — as 5 MAIS EFETIVAS (enunciado)
1. **v7-e5** (0.47610) — bi-encoder denso como feature de re-ranking; a FAMÍLIA do
   encoder importa (e5 contrastivo > bge no mesmo tamanho). Introduz o sinal semântico.
2. **v9-dual** (0.48087) — encoders complementares (e5 + bge-large) como DUAS features
   densas; famílias diferentes capturam semânticas distintas.
3. **Retrieval híbrido** (0.50981) — denso full-corpus destrava o **recall** do 1º
   estágio (0.52→0.73): denso e BM25 acham relevantes diferentes (lexical-mismatch).
4. **Híbrido + cross-encoder** (0.54427, MELHOR) — reranker bge-reranker-large pontua
   o par (query, entidade) junto; sinal de re-ranking muito mais forte que o cosseno.
5. **(complemento)** v7-base (0.47376) ou v7-large (0.46966) — varredura de tamanho do
   encoder; achado: "maior é melhor" refutado no Kaggle (base 0.474 > large 0.470) e a
   CV errou o ranking fino (overfit de CV; ver §6).

As 5 ilustram os DOIS eixos: melhor **re-ranking** (#1,2,4) e melhor **recuperação**
(#3). Hipótese transversal: a **semântica densa** ataca o lexical-mismatch — como
feature de re-ranking (bi-encoder #1,2; cross-encoder #4) e no estágio de recuperação (#3).
Discussão (profundidade): a base supervisionada (LTR v4/v5), os negativos (§5: união
esparsa, interação), os limites da CV calibrada (§6) e o diagnóstico de recall (§7).

## Apêndice — histórico completo de submissões (nDCG@100 Kaggle)

| Família | Variante | Config | Score |
|---|---|---|---|
| BM25 | Pyserini Lucene | default | 0.39774 |
| BM25 | disjunctive (PA2) | — | 0.39923 |
| Field weighting | (4,0,3) sem text | title=4, kw=3 | 0.29267 |
| Field weighting | (3,2,2) | | 0.39589 |
| Field weighting | (10,1,5) | | 0.40385 |
| Field weighting | (3,1,2) | | 0.40490 |
| Field weighting | **(5,1,3)** | melhor esparso de campo | **0.40518** |
| RM3 | (10,10,0.5) | origW baixo | 0.37647 |
| RM3 | (10,10,0.9) | | 0.40539 |
| RM3 | **(10,10,0.8)** | melhor não-supervisionado | **0.40648** |
| LTR | v1 | bug de calibração | 0.10785 |
| LTR | v2 | calibração corrigida | 0.36857 |
| LTR | v3 | hard+random negatives | 0.28844 |
| LTR | v4 | content-aware, re-ranking | 0.42696 |
| LTR | v5 | ensemble 6 seeds | 0.43397 |
| LTR+dense | v7 (bge-small) | | 0.46366 |
| LTR+dense | v7-large (bge-large) | | 0.46966 |
| LTR+dense | v7-base (bge-base) | | 0.47376 |
| LTR+dense | v7-e5 (e5-large) | | 0.47610 |
| LTR+dense | v9-dual (e5+bge-large) | | 0.48087 |
| LTR+híbrido | retrieval denso | BM25@100 ∪ denso@200 | 0.50981 |
| **LTR+híbrido+CE** | **+ cross-encoder reranker** | bge-reranker-large | **0.54427** |

Observações da exploração: remover o campo `text` do field weighting despenca o score
(0.293), confirmando que a descrição importa além do título; RM3 precisa de peso alto
na query original (origW 0.8 > 0.5); e o LTR só superou os baselines após corrigir
calibração (v1→v2) E o setup de candidatos (v3→v4).

## Conclusão
De **0.40648 → 0.54427 (+34%)**. O ganho veio em duas frentes ortogonais: (i) melhor
**re-ranking** (LTR supervisionado → ensemble → features densas de bi-encoder →
cross-encoder reranker), e (ii) melhor **recuperação** (retrieval híbrido denso, que
destravou o recall — o teto real, de 0.52 para 0.73). A metodologia de **CV calibrada
+ robustez multi-split** guiou cada decisão e evitou queimar submissões em ganhos
ilusórios. O maior aprendizado: o gargalo migrou do re-ranking para a recuperação, e
atacá-lo com encoder denso (técnica do enunciado) foi o que destravou o salto final.
