"""
Normalizer: converte tokens brutos em termos prontos para indexacao.

Pipeline:
    tokens [ja em lowercase, sem pontuacao pura]
      -> remove stopwords
      -> filtra por comprimento [MIN, MAX]
      -> opcionalmente descarta puros numericos
      -> aplica stemming Snowball
      -> termos finais

A ordem importa: filtramos ANTES do stemmer porque as checagens sao
mais baratas.

IMPORTANTE: indexer e processor DEVEM usar a mesma config deste
Normalizer, senao queries nao casam com documentos.
"""

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from src.config.preprocessing import (
    DROP_PURE_NUMERIC,
    MAX_TOKEN_LENGTH,
    MIN_TOKEN_LENGTH,
    STEMMER_LANGUAGE,
)


class Normalizer:
    """Aplica stopword removal + stemming + filtros."""

    def __init__(
        self,
        language: str = STEMMER_LANGUAGE,
        min_length: int = MIN_TOKEN_LENGTH,
        max_length: int = MAX_TOKEN_LENGTH,
        drop_numeric: bool = DROP_PURE_NUMERIC,
    ):
        self._stemmer = SnowballStemmer(language)
        self._stopwords = frozenset(stopwords.words(language))
        self._min_length = min_length
        self._max_length = max_length
        self._drop_numeric = drop_numeric

    def normalize(self, tokens: list[str]) -> list[str]:
        """
        Recebe tokens (esperados em lowercase) e retorna termos
        normalizados prontos para indexacao.
        """
        result = []
        for tok in tokens:
            if tok in self._stopwords:
                continue
            if len(tok) < self._min_length or len(tok) > self._max_length:
                continue
            if self._drop_numeric and tok.isdigit():
                continue
            stem = self._stemmer.stem(tok)

            if len(stem) < self._min_length:
                continue
            result.append(stem)
        return result
