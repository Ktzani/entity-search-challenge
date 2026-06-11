"""Testes do PartialIndex."""

import os
import unittest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import TempDirTestCase

from src.index_build.partial_index import PartialIndex
from src.utils.io_utils import read_posting, read_uint16, read_uint32


class TestPartialIndex(TempDirTestCase):

    def test_empty_index(self):
        pi = PartialIndex()
        self.assertTrue(pi.is_empty())
        self.assertEqual(pi.num_terms(), 0)
        self.assertEqual(pi.num_postings(), 0)
        self.assertEqual(pi.num_docs(), 0)

    def test_add_single_doc(self):
        pi = PartialIndex()
        pi.add_document(doc_id=0, terms=["albert", "einstein", "physics"])
        self.assertFalse(pi.is_empty())
        self.assertEqual(pi.num_terms(), 3)
        self.assertEqual(pi.num_postings(), 3)
        self.assertEqual(pi.num_docs(), 1)

    def test_tf_counted_correctly(self):
        """Termos repetidos no mesmo doc viram TF, nao postings duplicadas."""
        pi = PartialIndex()
        # 'albert' aparece 3x, deve gerar 1 posting com tf=3
        pi.add_document(doc_id=0, terms=["albert", "albert", "einstein", "albert"])
        self.assertEqual(pi.num_terms(), 2)
        self.assertEqual(pi.num_postings(), 2)

    def test_postings_ordered_by_doc_id(self):
        """Docs processados em ordem -> postings em ordem de doc_id."""
        pi = PartialIndex()
        pi.add_document(0, ["x"])
        pi.add_document(1, ["x"])
        pi.add_document(2, ["x"])

        path = os.path.join(self.tmpdir, "block.bin")
        pi.dump_to_disk(path)

        with open(path, "rb") as f:
            term_len = read_uint16(f)
            term = f.read(term_len).decode("utf-8")
            n = read_uint32(f)
            postings = [read_posting(f) for _ in range(n)]

        self.assertEqual(term, "x")
        doc_ids = [p[0] for p in postings]
        self.assertEqual(doc_ids, sorted(doc_ids))

    def test_dump_terms_sorted_alphabetically(self):
        pi = PartialIndex()
        pi.add_document(0, ["zebra", "alpha", "mango", "beta"])
        path = os.path.join(self.tmpdir, "block.bin")
        pi.dump_to_disk(path)

        terms_read = []
        with open(path, "rb") as f:
            while True:
                term_len = read_uint16(f)
                if term_len == -1:
                    break
                term = f.read(term_len).decode("utf-8")
                n = read_uint32(f)
                for _ in range(n):
                    read_posting(f)
                terms_read.append(term)

        self.assertEqual(terms_read, ["alpha", "beta", "mango", "zebra"])

    def test_clear(self):
        pi = PartialIndex()
        pi.add_document(0, ["a", "b", "c"])
        self.assertFalse(pi.is_empty())
        pi.clear()
        self.assertTrue(pi.is_empty())
        self.assertEqual(pi.num_terms(), 0)

    def test_add_after_clear(self):
        """Apos clear, deve aceitar novos documentos normalmente."""
        pi = PartialIndex()
        pi.add_document(0, ["a", "b"])
        pi.clear()
        pi.add_document(0, ["c", "d"])
        self.assertEqual(pi.num_terms(), 2)
        self.assertEqual(pi.num_postings(), 2)

    def test_dump_empty_raises(self):
        pi = PartialIndex()
        with self.assertRaises(ValueError):
            pi.dump_to_disk(os.path.join(self.tmpdir, "block.bin"))

    def test_empty_terms_list_does_not_count(self):
        """Adicionar doc com terms vazio nao incrementa num_docs."""
        pi = PartialIndex()
        pi.add_document(0, [])
        self.assertTrue(pi.is_empty())
        self.assertEqual(pi.num_docs(), 0)


if __name__ == "__main__":
    unittest.main()
