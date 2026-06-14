"""
features_base.py - Logica COMPARTILHADA de features entre extracao (treino) e
predicao (teste). Centralizar aqui garante que as features sejam computadas
de forma IDENTICA nos dois lados (requisito critico de LTR).

Correcoes vs v3:
1. As content features (title/keywords) vinham vazias porque o codigo lia o
   campo "title"/"keywords" de um indice (s3 v4) que so guarda "contents"
   concatenado, ou do indice s2 que nao guarda raw nenhum. A fonte real eh o
   corpus original data/corpus/entities.jsonl. load_corpus_fields() le os
   campos verdadeiros de la, com UM passe streaming, coletando so os docids
   necessarios (memoria limitada).

Features novas (entity search):
- f_fw_norm, f_rm3_norm       : score normalizado pelo max da query
- f_title_jaccard             : Jaccard(query_tokens, title_tokens)
- f_title_bigram_overlap      : % de bigramas da query presentes no titulo
- f_title_match_pos           : posicao normalizada do 1o match no titulo
- f_query_coverage            : % de tokens da query presentes no doc inteiro

Nota: features binarias grosseiras (f_title_exact_match, f_title_full_match)
foram REMOVIDAS por terem gain=0 no LightGBM: sao redundantes com as versoes
continuas (f_title_overlap, f_title_jaccard). Ex.: f_title_full_match==1 em
23/28833 rows, todas tambem com jaccard==1 (100% redundante).
"""

import json
import sys

# Ordem CANONICA das features. Treino e predicao DEVEM usar exatamente esta
# ordem/conjunto. 02_train_ltr.py le qualquer coluna que comece com "f_".
FEATURE_ORDER = [
    "f_bm25",
    "f_field_weights",
    "f_rm3",
    "f_bm25_norm",
    "f_fw_norm",
    "f_rm3_norm",
    "f_bm25_rank",
    "f_query_length",
    "f_mean_df",
    "f_doc_length",
    "f_title_overlap",
    "f_keyword_match",
    "f_title_jaccard",
    "f_title_bigram_overlap",
    "f_title_match_pos",
    "f_query_coverage",
]

_ID_PREFIX = '{"id": "'  # formato observado em entities.jsonl
_ID_LEN = 7              # ids zero-padded de 7 digitos


def _tokens(s: str) -> list[str]:
    """Tokenizacao simples e consistente: lowercase + whitespace split."""
    return s.lower().split() if s else []


def load_corpus_fields(corpus_path: str, docids) -> dict[str, dict]:
    """Le title/text/keywords reais do corpus para o conjunto de docids dado.

    Faz UM passe streaming sobre o jsonl. Para evitar json.loads em 4.6M linhas,
    extrai o id por slice barato e so parseia as linhas necessarias.

    Retorna {docid: {"title": str, "keywords": str, "doc_len": int,
                      "token_set": frozenset[str]}}.
    """
    need = set(docids)
    out: dict[str, dict] = {}
    total_need = len(need)
    parsed = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not need:
                break
            # Extracao barata do id: {"id": "XXXXXXX", ...
            if line.startswith(_ID_PREFIX):
                did = line[len(_ID_PREFIX):len(_ID_PREFIX) + _ID_LEN]
                if did not in need:
                    continue
            else:
                # Fallback: formato inesperado -> parseia para pegar o id
                try:
                    did = str(json.loads(line).get("id", ""))
                except Exception:
                    continue
                if did not in need:
                    continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            title = obj.get("title", "") or ""
            text = obj.get("text", "") or ""
            kw = obj.get("keywords", [])
            if isinstance(kw, list):
                kw_str = " ".join(str(k) for k in kw)
            else:
                kw_str = str(kw)

            combined = f"{title} {text} {kw_str}"
            out[did] = {
                "title": title,
                "keywords": kw_str,
                "doc_len": len(text.split()),
                "token_set": frozenset(_tokens(combined)),
            }
            need.discard(did)
            parsed += 1
            if parsed % 5000 == 0:
                print(f"[corpus] {parsed}/{total_need} docs lidos...",
                      file=sys.stderr)

    if need:
        print(f"[corpus] AVISO: {len(need)} docids nao encontrados no corpus "
              f"(de {total_need})", file=sys.stderr)
    print(f"[corpus] {len(out)}/{total_need} docids resolvidos", file=sys.stderr)
    return out


_EMPTY_DOC = {"title": "", "keywords": "", "doc_len": 0, "token_set": frozenset()}


def compute_features(
    qtext: str,
    *,
    bm25_s: float,
    fw_s: float,
    rm3_s: float,
    bm25_rank: int,
    max_bm25: float,
    max_fw: float,
    max_rm3: float,
    query_length: int,
    q_mean_df: float,
    doc: dict | None,
) -> dict[str, float]:
    """Computa todas as features de um par (query, doc). Usado por treino e teste."""
    d = doc if doc is not None else _EMPTY_DOC
    title = d["title"]
    kw_str = d["keywords"]
    doc_tokens = d["token_set"]

    q_tokens = _tokens(qtext)
    q_set = set(q_tokens)
    title_tokens = _tokens(title)
    title_set = set(title_tokens)

    # --- content features ---
    title_overlap = (
        sum(1 for t in q_tokens if t in title_set) / len(q_tokens)
        if q_tokens else 0.0
    )
    kw_lower = kw_str.lower()
    kw_match = sum(1 for t in q_tokens if t in kw_lower) if kw_str else 0

    union = q_set | title_set
    jaccard = len(q_set & title_set) / len(union) if union else 0.0

    # bigramas
    q_bigrams = set(zip(q_tokens, q_tokens[1:]))
    t_bigrams = set(zip(title_tokens, title_tokens[1:]))
    bigram_overlap = (
        len(q_bigrams & t_bigrams) / len(q_bigrams) if q_bigrams else 0.0
    )

    # posicao normalizada do 1o match no titulo (menor = melhor; 1.0 = sem match)
    match_pos = 1.0
    if title_tokens:
        for i, tok in enumerate(title_tokens):
            if tok in q_set:
                match_pos = i / len(title_tokens)
                break

    # cobertura: % dos tokens da query presentes em qualquer lugar do doc
    coverage = (
        sum(1 for t in q_set if t in doc_tokens) / len(q_set)
        if q_set else 0.0
    )

    return {
        "f_bm25": bm25_s,
        "f_field_weights": fw_s,
        "f_rm3": rm3_s,
        "f_bm25_norm": bm25_s / max_bm25 if max_bm25 else 0.0,
        "f_fw_norm": fw_s / max_fw if max_fw else 0.0,
        "f_rm3_norm": rm3_s / max_rm3 if max_rm3 else 0.0,
        "f_bm25_rank": float(bm25_rank),
        "f_query_length": float(query_length),
        "f_mean_df": q_mean_df,
        "f_doc_length": float(d["doc_len"]),
        "f_title_overlap": title_overlap,
        "f_keyword_match": float(kw_match),
        "f_title_jaccard": jaccard,
        "f_title_bigram_overlap": bigram_overlap,
        "f_title_match_pos": match_pos,
        "f_query_coverage": coverage,
    }


def feature_vector(feat: dict[str, float]) -> list[float]:
    """Converte o dict de features em vetor na ordem canonica."""
    return [feat[name] for name in FEATURE_ORDER]
