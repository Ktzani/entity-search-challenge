"""
Leitura do lexicon: term -> (offset, df).

Carregado inteiro em memoria. CRUCIAL no DAAT: se um termo da query
nao existe no lexicon, a intersecao conjuntiva é vazia imediatamente
(curto-circuito sem ler o disco).
"""

import os
import pickle

from src.config.indexer import TERM_LEXICON_FILENAME


class TermLexicon:
    """Leitura do lexicon."""

    def __init__(self, index_dir: str):
        """Carrega lexicon.pkl em RAM."""
        path = os.path.join(index_dir, TERM_LEXICON_FILENAME)
        with open(path, "rb") as f:
            self._lex: dict[str, tuple[int, int]] = pickle.load(f)

    def has_term(self, term: str) -> bool:
        """True se o termo existe no indice."""
        return term in self._lex

    def get_entry(self, term: str) -> tuple[int, int] | None:
        """
        Retorna (offset, df) do termo no inverted index, ou None se
        o termo nao existe no indice.
        """
        return self._lex.get(term)

    def get_df(self, term: str) -> int:
        """Document frequency do termo (0 se nao existe)."""
        entry = self._lex.get(term)
        return entry[1] if entry else 0

    def num_terms(self) -> int:
        """Numero de termos unicos no lexicon."""
        return len(self._lex)

    def __contains__(self, term: str) -> bool:
        return term in self._lex
