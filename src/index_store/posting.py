"""
Posting (doc_id, tf) numa inverted list.

NamedTuple: imutavel, desempacotavel (doc_id, tf = posting) e
lightweight (mesma performance de tupla pura).
"""

from typing import NamedTuple


class Posting(NamedTuple):
    doc_id: int
    tf: int
