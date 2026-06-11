"""
Leitura do mapping internal_doc_id -> (original_id, length).

Persistido como tupla (original_ids: list[str], lengths: array('I')),
indexada por internal_id. Esta representacao gasta ~10x menos memoria
que dict[int, dict] em corpus grandes (4.6M docs: ~70 MB vs ~1.2 GB).
"""

import array
import os
import pickle

from src.config.indexer import DOCUMENT_INDEX_FILENAME


class DocumentIndex:
    """Leitura do document index."""

    def __init__(self, index_dir: str):
        """Carrega doc_index.pkl em RAM e pre-calcula avg_doc_length."""
        path = os.path.join(index_dir, DOCUMENT_INDEX_FILENAME)
        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, tuple) and len(data) == 2:
            self._original_ids, self._lengths = data
        elif isinstance(data, dict):
            n = len(data)
            self._original_ids = [""] * n
            self._lengths = array.array("I", [0] * n)
            for internal_id, info in data.items():
                self._original_ids[internal_id] = info["original_id"]
                self._lengths[internal_id] = info["length"]
        else:
            raise ValueError(f"Formato desconhecido em {path}")

        n = len(self._lengths)
        if n > 0:
            total = sum(self._lengths)
            self._avg_length = total / n
        else:
            self._avg_length = 0.0

    def get_length(self, doc_id: int) -> int:
        """Numero de tokens (apos preprocessing) do documento."""
        return self._lengths[doc_id]

    def get_original_id(self, doc_id: int) -> str:
        """Id original do documento (string do JSONL)."""
        return self._original_ids[doc_id]

    def avg_doc_length(self) -> float:
        """Comprimento medio dos documentos. Usado pelo BM25."""
        return self._avg_length

    def num_docs(self) -> int:
        """Numero total de documentos (= N para TF-IDF/BM25)."""
        return len(self._lengths)

    def __contains__(self, doc_id: int) -> bool:
        return 0 <= doc_id < len(self._lengths)