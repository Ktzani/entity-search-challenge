"""
Utilitarios de I/O binario do indice. LITTLE-ENDIAN explicito para
portabilidade entre plataformas (Windows -> Linux nos testes).

UINT16 (2B) = term_length, UINT32 (4B) = doc_id/tf/num_postings.
"""

import struct

FMT_UINT16 = "<H"
FMT_UINT32 = "<I"
FMT_POSTING = "<II"  # (doc_id, tf), 8 bytes

SIZEOF_UINT16 = 2
SIZEOF_UINT32 = 4
SIZEOF_POSTING = 8


def write_uint16(f, value: int):
    if value < 0 or value > 0xFFFF:
        raise ValueError(f"uint16 fora do range: {value}")
    f.write(struct.pack(FMT_UINT16, value))


def write_uint32(f, value: int):
    if value < 0 or value > 0xFFFFFFFF:
        raise ValueError(f"uint32 fora do range: {value}")
    f.write(struct.pack(FMT_UINT32, value))


def read_uint16(f) -> int:
    """Le um uint16. Retorna -1 se EOF."""
    data = f.read(SIZEOF_UINT16)
    if len(data) < SIZEOF_UINT16:
        return -1
    return struct.unpack(FMT_UINT16, data)[0]


def read_uint32(f) -> int:
    """Le um uint32. Retorna -1 se EOF."""
    data = f.read(SIZEOF_UINT32)
    if len(data) < SIZEOF_UINT32:
        return -1
    return struct.unpack(FMT_UINT32, data)[0]


def write_posting(f, doc_id: int, tf: int):
    f.write(struct.pack(FMT_POSTING, doc_id, tf))


def read_posting(f) -> tuple[int, int] | None:
    """Le uma posting de 8 bytes. Retorna None se EOF."""
    data = f.read(SIZEOF_POSTING)
    if len(data) < SIZEOF_POSTING:
        return None
    return struct.unpack(FMT_POSTING, data)


def file_size_mb(path: str) -> float:
    import os
    return os.path.getsize(path) / (1024.0 * 1024.0)
