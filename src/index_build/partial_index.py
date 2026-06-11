"""
Indice invertido parcial em memoria.

Acumula postings ate o orcamento de memoria estourar; entao é
despejado em disco como um block ordenado por termo. O ordenamento
permite k-way merge linear na fase final.

Formato do block em disco (para cada termo, em ordem alfabetica):
    [term_length: uint16][term: utf-8 bytes][num_postings: uint32]
    [posting_1 = (doc_id, tf) = 8 bytes] ...

As postings dentro de cada termo ja saem em ordem de doc_id porque
docs sao processados sequencialmente (reader atribui doc_ids
monotonicos). Logo, as postings de cada termo ja saem em ordem 
crescente de doc_id.
"""

from collections import Counter, defaultdict

from src.utils.io_utils import write_posting, write_uint16, write_uint32


class PartialIndex:
    """Indice invertido parcial em memoria. NAO é thread-safe; usar
    com lock externo no SPIMI."""

    def __init__(self):
        self._index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._num_postings = 0
        self._num_docs = 0

    def add_document(self, doc_id: int, terms: list[str]):
        """
        Adiciona um documento ao indice. Recebe a lista de termos ja
        normalizados (com repeticoes). Calcula TF e adiciona uma
        posting por termo unico.
        """
        if not terms:
            return
        tfs = Counter(terms)
        for term, tf in tfs.items():
            self._index[term].append((doc_id, tf))
            self._num_postings += 1
        self._num_docs += 1

    def num_terms(self) -> int:
        """Termos unicos atualmente no indice."""
        return len(self._index)

    def num_postings(self) -> int:
        """Total de postings (sum over terms of len(postings))."""
        return self._num_postings

    def num_docs(self) -> int:
        """Documentos contribuidos a este indice parcial."""
        return self._num_docs

    def is_empty(self) -> bool:
        return self._num_postings == 0

    def clear(self):
        """
        Zera o indice apos flush. Recria o defaultdict para garantir
        liberacao do overhead interno do dict anterior (buckets, etc.).
        """
        self._index.clear()
        self._index = defaultdict(list)
        self._num_postings = 0
        self._num_docs = 0

    def dump_to_disk(self, path: str) -> dict:
        """
        Escreve o block em disco ordenado por termo. Retorna metadados
        {"path", "num_terms", "num_postings"}.
        """
        if self.is_empty():
            raise ValueError("PartialIndex vazio; nao ha o que despejar")

        sorted_terms = sorted(self._index.keys())

        with open(path, "wb") as f:
            for term in sorted_terms:
                postings = self._index[term]
                term_bytes = term.encode("utf-8")
                write_uint16(f, len(term_bytes))
                f.write(term_bytes)
                write_uint32(f, len(postings))
                for doc_id, tf in postings:
                    write_posting(f, doc_id, tf)

        return {
            "path": path,
            "num_terms": len(sorted_terms),
            "num_postings": self._num_postings,
        }
