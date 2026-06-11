"""
Tokenizer: texto bruto -> lista de tokens lowercase.

Usa NLTK word_tokenize (Treebank), depois descarta tokens de
pontuacao pura e residuos de contracoes. Stopwords, stemming e
filtro por tamanho NAO sao feitos aqui - sao responsabilidade do
Normalizer.
"""

import re

from nltk.tokenize import word_tokenize


# Token "lixo" se tiver SO pontuacao/simbolos. Mantemos tudo que tem
# pelo menos uma letra ou digito.
_HAS_ALNUM = re.compile(r"[a-z0-9]", re.IGNORECASE)

# Contracoes residuais do tokenizer Treebank (e.g., "don't" -> "do" + "n't").
# Esses pedacos nao tem valor semantico isolado e geram ruido no indice.
_CONTRACTION_RESIDUALS = frozenset([
    "n't", "'s", "'re", "'ve", "'ll", "'d", "'m", "'t",
])


class Tokenizer:
    """Tokenizador baseado em NLTK + filtros minimos."""

    def __init__(self, language: str = "english"):
        self._language = language

    def tokenize(self, text: str) -> list[str]:
        """
        Retorna a lista de tokens em lowercase, descartando pontuacao
        pura e residuos de contracoes ('n't, 's, etc.).
        """
        if not text:
            return []

        raw_tokens = word_tokenize(text, language=self._language)

        result = []
        for tok in raw_tokens:
            tok = tok.lower()
            if not _HAS_ALNUM.search(tok):
                continue
            if tok in _CONTRACTION_RESIDUALS:
                continue
            if tok.startswith("'"):
                continue
            result.append(tok)
        return result
