"""
Persiste lexicon e document index em disco (pickle) e calcula as
estatisticas finais do indice. O arquivo inverted.idx é produzido
pelo merger;
"""

import os
import pickle

from src.config.indexer import (
    DOCUMENT_INDEX_FILENAME,
    TERM_LEXICON_FILENAME,
    INVERTED_INDEX_FILENAME,
)


_PICKLE_PROTOCOL = 5


def write_lexicon(
    lexicon: dict[str, tuple[int, int]],
    index_dir: str,
) -> str:
    """Persiste lexicon (term -> (offset, df)) como pickle."""
    path = os.path.join(index_dir, TERM_LEXICON_FILENAME)
    with open(path, "wb") as f:
        pickle.dump(lexicon, f, protocol=_PICKLE_PROTOCOL)
    return path


def write_doc_index(
    doc_index_data,
    index_dir: str,
) -> str:
    """
    Persiste o document index como tupla (original_ids, lengths).
    Aceita tambem o formato legado dict[int, dict] e converte.
    """
    if isinstance(doc_index_data, dict):
        import array
        n = len(doc_index_data)
        original_ids: list[str] = [""] * n
        lengths = array.array("I", [0] * n)
        for internal_id, info in doc_index_data.items():
            original_ids[internal_id] = info["original_id"]
            lengths[internal_id] = info["length"]
        payload = (original_ids, lengths)
    else:
        payload = doc_index_data

    path = os.path.join(index_dir, DOCUMENT_INDEX_FILENAME)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=_PICKLE_PROTOCOL)
    return path


def compute_statistics(
    index_dir: str,
    lexicon: dict[str, tuple[int, int]],
) -> dict:
    """
    Estatisticas finais do indice: Index Size (MB), Number of Lists,
    Average List Size. Elapsed Time é preenchido pelo entry point.
    """
    inverted_path = os.path.join(index_dir, INVERTED_INDEX_FILENAME)
    lexicon_path = os.path.join(index_dir, TERM_LEXICON_FILENAME)
    doc_index_path = os.path.join(index_dir, DOCUMENT_INDEX_FILENAME)

    total_bytes = 0
    for p in (inverted_path, lexicon_path, doc_index_path):
        if os.path.exists(p):
            total_bytes += os.path.getsize(p)

    num_lists = len(lexicon)
    if num_lists > 0:
        total_postings = sum(df for _, df in lexicon.values())
        avg_list_size = total_postings / num_lists
    else:
        avg_list_size = 0.0

    return {
        "Index Size": round(total_bytes / (1024.0 * 1024.0), 3),
        "Number of Lists": num_lists,
        "Average List Size": round(avg_list_size, 2),
    }