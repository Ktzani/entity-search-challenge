# Conteúdo dos 3 slides — RC Entity Search

---

## SLIDE 1 — Título
**Posição: 7º · Score (privado): 0.53671 · Membros: Gabriel Catizani**

Entity Search (4,6M entidades, nDCG@100)
Abordagem: **recuperação híbrida (BM25 + densa) → re-ranking supervisionado (LambdaMART)**

---

## SLIDE 2 — A estratégia
**Pipeline em 2 estágios:**

1. **Recuperar candidatos (híbrido):**
   - BM25 (Pyserini/Lucene) — top 100
   - Denso: embeddings do corpus (BGE) + faiss — top 200
   - Pool = união dos dois

2. **Re-rankear (LambdaMART / LightGBM):**
   - Treinado nos 8.202 qrels de relevância
   - Features: scores BM25/RM3 + match de conteúdo + **similaridade densa** + **cross-encoder (reranker)**
   - Ensemble de 6 modelos

*(diagrama: query → [BM25 ∪ denso] → LambdaMART → ranking)*

---

## SLIDE 3 — Por que funcionou + resultado
- **Usamos os qrels** (aprendizado supervisionado) — muitos só usaram métodos não-supervisionados.
- O gargalo era o **recall**: metade dos relevantes não estava no top-100 do BM25.
  A **busca densa** recupera o que o léxico perde → recall 0.52 → 0.73.
- O **cross-encoder** deu o sinal de relevância mais forte no re-ranking.

**Evolução:** BM25 0.40 → +LTR 0.43 → +denso 0.48 → +híbrido 0.51 → **+cross-encoder 0.54**

