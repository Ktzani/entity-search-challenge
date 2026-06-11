"""
Query: representa uma query apos pre-processamento.

Mantem raw_text (para o JSON de output) e terms (lista deduplicada
preservando ordem, para o DAAT).

IMPORTANTE: usa o MESMO Tokenizer + Normalizer do indexer. Caso contrario
queries nao casariam com documentos.

Removemos duplicatas porque a intersecao conjuntiva nao se beneficia
delas e o TF que importa é o do documento, nao o da query.
"""

from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.normalizer import Normalizer


class Query:
    """Uma query de busca, com texto cru e termos preprocessados."""

    def __init__(
        self,
        raw_text: str,
        tokenizer: Tokenizer,
        normalizer: Normalizer,
    ):
        self._raw_text = raw_text.strip()
        tokens = tokenizer.tokenize(self._raw_text)
        normalized = normalizer.normalize(tokens)
        # Dedup preservando ordem para outputs deterministicos.
        seen = set()
        self._terms: list[str] = []
        for t in normalized:
            if t not in seen:
                seen.add(t)
                self._terms.append(t)

    @property
    def raw_text(self) -> str:
        return self._raw_text

    @property
    def terms(self) -> list[str]:
        return self._terms

    def is_empty(self) -> bool:
        """True se a query nao gerou termos uteis (vazia, so stopwords,
        ou termos descartados por tamanho)."""
        return len(self._terms) == 0

    def __repr__(self) -> str:
        return f"Query(raw={self._raw_text!r}, terms={self._terms})"