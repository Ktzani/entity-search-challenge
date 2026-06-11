"""
K-way merge externo dos blocks parciais produzidos pelo SPIMI.

Cada block é um iterator de (term, postings) em ordem alfabetica.
Heap minima de (term, block_idx) seleciona o proximo termo a emitir;
postings dos blocks com o mesmo termo no topo sao combinadas e
escritas no arquivo final, atualizando o lexicon.

Complexidade: O(N log K) em tempo, O(K) em espaco.

Formato da inverted list no arquivo final:
    [num_postings: uint32][posting_1][posting_2]...
    (sem term: o lexicon resolve term -> offset)
"""

import heapq
import os
import sys

from src.utils.io_utils import (
    read_posting,
    read_uint16,
    read_uint32,
    write_posting,
    write_uint32,
)


class _BlockReader:
    """Iterator sobre um block binario, em ordem alfabetica de termo."""

    def __init__(self, path: str):
        self._path = path
        self._f = open(path, "rb")
        self._eof = False

    def next_entry(self) -> tuple[str, list[tuple[int, int]]] | None:
        """Le proxima entrada (term, postings) ou None ao EOF."""
        if self._eof:
            return None
        term_len = read_uint16(self._f)
        if term_len == -1:
            self._eof = True
            return None
        term_bytes = self._f.read(term_len)
        if len(term_bytes) < term_len:
            self._eof = True
            return None
        term = term_bytes.decode("utf-8")
        num_postings = read_uint32(self._f)
        if num_postings == -1:
            self._eof = True
            return None
        postings = []
        for _ in range(num_postings):
            p = read_posting(self._f)
            if p is None:
                self._eof = True
                return None
            postings.append(p)
        return term, postings

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


def merge_blocks(
    block_paths: list[str],
    inverted_index_path: str,
) -> dict[str, tuple[int, int]]:
    """
    Faz o k-way merge dos blocks parciais.

    Parametros:
        block_paths: lista de paths de blocks binarios (gerados pelo SPIMI)
        inverted_index_path: arquivo de saida (sera sobrescrito)

    Retorna:
        lexicon: dict term -> (offset_no_arquivo, document_frequency)
    """
    if not block_paths:
        open(inverted_index_path, "wb").close()
        return {}

    readers = [_BlockReader(p) for p in block_paths]

    # heap: (term, block_idx); block_idx é tie-breaker.
    heap: list[tuple[str, int]] = []
    current: list[tuple[str, list[tuple[int, int]]] | None] = []

    for i, reader in enumerate(readers):
        entry = reader.next_entry()
        if entry is not None:
            term, postings = entry
            current.append(entry)
            heapq.heappush(heap, (term, i))
        else:
            current.append(None)

    lexicon: dict[str, tuple[int, int]] = {}

    try:
        with open(inverted_index_path, "wb") as out:
            while heap:
                top_term, _ = heap[0]

                # Failsafe: somamos TFs caso um doc_id apareca em
                # multiplos blocks (nao deveria, mas defesa barata).
                combined: dict[int, int] = {}

                while heap and heap[0][0] == top_term:
                    _, block_idx = heapq.heappop(heap)
                    entry = current[block_idx]
                    if entry is None:
                        continue
                    _, postings = entry
                    for doc_id, tf in postings:
                        if doc_id in combined:
                            combined[doc_id] += tf
                        else:
                            combined[doc_id] = tf

                    next_entry = readers[block_idx].next_entry()
                    current[block_idx] = next_entry
                    if next_entry is not None:
                        next_term, _ = next_entry
                        heapq.heappush(heap, (next_term, block_idx))

                final_postings = sorted(combined.items())
                df = len(final_postings)

                offset = out.tell()
                lexicon[top_term] = (offset, df)

                write_uint32(out, df)
                for doc_id, tf in final_postings:
                    write_posting(out, doc_id, tf)
    finally:
        for r in readers:
            r.close()

    return lexicon


def cleanup_blocks(block_paths: list[str]):
    """Remove os arquivos de blocks parciais apos o merge."""
    for p in block_paths:
        try:
            os.remove(p)
        except OSError as e:
            print(f"[merger] aviso: nao removi {p}: {e}", file=sys.stderr)
