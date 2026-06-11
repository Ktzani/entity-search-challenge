"""Testes do Query."""

import unittest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import ensure_nltk_data  # noqa

from src.preprocessing.normalizer import Normalizer
from src.preprocessing.tokenizer import Tokenizer
from src.retrieval.query import Query


class TestQuery(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()
        self.normalizer = Normalizer()

    def _q(self, text: str) -> Query:
        return Query(text, self.tokenizer, self.normalizer)

    def test_simple_query(self):
        q = self._q("albert einstein")
        self.assertEqual(q.terms, ["albert", "einstein"])
        self.assertFalse(q.is_empty())

    def test_raw_text_preserved(self):
        q = self._q("  Albert   Einstein  ")
        # strip aplicado
        self.assertEqual(q.raw_text, "Albert   Einstein")
        self.assertEqual(q.terms, ["albert", "einstein"])

    def test_duplicates_removed(self):
        q = self._q("Albert Einstein albert einstein")
        # 4 tokens -> 2 termos unicos (preservando ordem)
        self.assertEqual(q.terms, ["albert", "einstein"])

    def test_stopword_only_is_empty(self):
        q = self._q("the and a")
        self.assertTrue(q.is_empty())
        self.assertEqual(q.terms, [])

    def test_empty_string(self):
        q = self._q("")
        self.assertTrue(q.is_empty())
        self.assertEqual(q.raw_text, "")

    def test_stemming_applied(self):
        q = self._q("running runs")
        # Ambos viram 'run' apos stemming -> deduplicado
        self.assertEqual(q.terms, ["run"])

    def test_order_preserved_after_dedup(self):
        q = self._q("einstein albert einstein physics")
        self.assertEqual(q.terms, ["einstein", "albert", "physic"])


if __name__ == "__main__":
    unittest.main()
