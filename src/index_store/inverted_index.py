"""
Leitura random-access do arquivo inverted.idx.

Abre o arquivo UMA vez e mantem o handle durante todo o query
processing. Para cada query, faz seek(offset) + read das postings.

Formato (por termo, em ordem alfabetica):
    [num_postings: uint32][posting_1][posting_2]...
O offset vem do lexicon; o termo NAO esta no arquivo.
"""

import os

from src.config.indexer import INVERTED_INDEX_FILENAME
from src.index_store.posting import Posting
from src.utils.io_utils import read_posting, read_uint32


class InvertedIndex:
    """Random-access reader do inverted index."""

    def __init__(self, index_dir: str):
        path = os.path.join(index_dir, INVERTED_INDEX_FILENAME)
        self._f = open(path, "rb")

    def read_postings(self, offset: int, expected_df: int) -> list[Posting]:
        """
        Le a inverted list completa de um termo.

        Parametros:
            offset: offset no arquivo (do lexicon)
            expected_df: numero esperado de postings (do lexicon)

        Retorna:
            lista de Posting(doc_id, tf), em ordem crescente de doc_id
            (invariante garantido pelo merger).
        """
        self._f.seek(offset)
        num_postings = read_uint32(self._f)
        if num_postings != expected_df:
            raise IOError(
                f"Inconsistencia: lexicon diz df={expected_df}, "
                f"arquivo diz num_postings={num_postings} no offset {offset}"
            )
        postings = []
        for _ in range(num_postings):
            p = read_posting(self._f)
            if p is None:
                raise IOError(f"EOF inesperado lendo postings no offset {offset}")
            postings.append(Posting(doc_id=p[0], tf=p[1]))
        return postings

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
