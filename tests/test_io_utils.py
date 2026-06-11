"""Testes das funcoes de I/O binario."""

import io
import unittest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io_utils import (
    read_posting,
    read_uint16,
    read_uint32,
    write_posting,
    write_uint16,
    write_uint32,
)


class TestIOUtils(unittest.TestCase):

    def test_uint16_roundtrip(self):
        buf = io.BytesIO()
        write_uint16(buf, 42)
        write_uint16(buf, 65535)
        buf.seek(0)
        self.assertEqual(read_uint16(buf), 42)
        self.assertEqual(read_uint16(buf), 65535)

    def test_uint32_roundtrip(self):
        buf = io.BytesIO()
        write_uint32(buf, 0)
        write_uint32(buf, 1_000_000)
        write_uint32(buf, 4_294_967_295)  # uint32 max
        buf.seek(0)
        self.assertEqual(read_uint32(buf), 0)
        self.assertEqual(read_uint32(buf), 1_000_000)
        self.assertEqual(read_uint32(buf), 4_294_967_295)

    def test_posting_roundtrip(self):
        buf = io.BytesIO()
        write_posting(buf, doc_id=42, tf=7)
        write_posting(buf, doc_id=100000, tf=1)
        buf.seek(0)
        self.assertEqual(read_posting(buf), (42, 7))
        self.assertEqual(read_posting(buf), (100000, 1))

    def test_uint16_out_of_range(self):
        buf = io.BytesIO()
        with self.assertRaises(ValueError):
            write_uint16(buf, -1)
        with self.assertRaises(ValueError):
            write_uint16(buf, 70000)

    def test_uint32_out_of_range(self):
        buf = io.BytesIO()
        with self.assertRaises(ValueError):
            write_uint32(buf, -1)
        with self.assertRaises(ValueError):
            write_uint32(buf, 2**32)

    def test_eof_returns_sentinel(self):
        """Leitura de buffer vazio deve retornar -1 (uint) ou None (posting)."""
        buf = io.BytesIO(b"")
        self.assertEqual(read_uint16(buf), -1)
        self.assertEqual(read_uint32(buf), -1)
        self.assertIsNone(read_posting(buf))

    def test_little_endian(self):
        """Confirma que os bytes sao little-endian (portabilidade)."""
        buf = io.BytesIO()
        write_uint16(buf, 0x0102)  # 258 em decimal
        # Little-endian: byte menos significativo primeiro
        self.assertEqual(buf.getvalue(), b"\x02\x01")

        buf = io.BytesIO()
        write_uint32(buf, 0x01020304)
        self.assertEqual(buf.getvalue(), b"\x04\x03\x02\x01")


if __name__ == "__main__":
    unittest.main()
