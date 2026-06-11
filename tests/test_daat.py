"""Testes do conjunctive DAAT."""

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.index_store.posting import Posting
from src.retrieval.daat import _Cursor


class TestCursor(unittest.TestCase):
    """Testes do galloping search."""

    def test_advance_to_target_in_list(self):
        postings = [Posting(i, 1) for i in [0, 5, 10, 15, 20]]
        c = _Cursor("t", postings)
        c.advance_to(10)
        self.assertEqual(c.current_doc_id, 10)

    def test_advance_to_target_not_in_list(self):
        """advance_to(7) deve parar no primeiro doc_id >= 7."""
        postings = [Posting(i, 1) for i in [0, 5, 10, 15, 20]]
        c = _Cursor("t", postings)
        c.advance_to(7)
        self.assertEqual(c.current_doc_id, 10)

    def test_advance_to_already_at_target(self):
        postings = [Posting(i, 1) for i in [0, 5, 10]]
        c = _Cursor("t", postings)
        c.advance_to(0)
        self.assertEqual(c.current_doc_id, 0)

    def test_advance_to_beyond_end(self):
        postings = [Posting(i, 1) for i in [0, 5, 10]]
        c = _Cursor("t", postings)
        c.advance_to(100)
        self.assertTrue(c.is_exhausted())

    def test_galloping_is_efficient(self):
        """Galloping deve visitar << posicoes que linear scan."""
        # 1000 docs sequenciais, target longe
        postings = [Posting(i, 1) for i in range(1000)]
        c = _Cursor("t", postings)
        c.advance_to(500)
        # Linear scan visitaria ~500 posicoes; galloping deve visitar <50
        self.assertLess(c.scanned, 50)

    def test_advance_one(self):
        postings = [Posting(0, 1), Posting(1, 1), Posting(2, 1)]
        c = _Cursor("t", postings)
        self.assertEqual(c.current_doc_id, 0)
        c.advance_one()
        self.assertEqual(c.current_doc_id, 1)
        c.advance_one()
        self.assertEqual(c.current_doc_id, 2)
        c.advance_one()
        self.assertTrue(c.is_exhausted())


class TestConjunctiveDAAT(unittest.TestCase):
    """
    Testes do DAAT requerem TermLexicon + InvertedIndex em disco.
    Esses sao testados via integracao no test_end_to_end.py.
    """

    def test_empty_terms(self):
        # Caso trivial: query sem termos -> resultado vazio
        from src.retrieval.daat import ConjunctiveDAAT, DAATResult
        # Nao precisamos de lexicon real para esse caso
        daat = ConjunctiveDAAT(lexicon=None, inverted_index=None)
        result = daat.intersect([])
        self.assertIsInstance(result, DAATResult)
        self.assertEqual(result.matched_doc_ids, [])
        self.assertEqual(result.postings_scanned, 0)


if __name__ == "__main__":
    unittest.main()
