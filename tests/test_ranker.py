"""Testes do Ranker."""

import unittest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.ranker import Ranker


class TestRanker(unittest.TestCase):

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            Ranker(k=0)
        with self.assertRaises(ValueError):
            Ranker(k=-1)

    def test_empty_candidates(self):
        r = Ranker(k=10)
        self.assertEqual(r.top_k([]), [])

    def test_fewer_than_k(self):
        r = Ranker(k=10)
        candidates = [(0, 1.5), (1, 3.2), (2, 0.8)]
        result = r.top_k(candidates)
        # Todos retornados, em ordem decrescente de score
        self.assertEqual(result, [(1, 3.2), (0, 1.5), (2, 0.8)])

    def test_top_k(self):
        r = Ranker(k=3)
        candidates = [(0, 1.0), (1, 3.0), (2, 0.5), (3, 5.0), (4, 2.0), (5, 4.0)]
        result = r.top_k(candidates)
        self.assertEqual(result, [(3, 5.0), (5, 4.0), (1, 3.0)])

    def test_tie_breaking_by_doc_id_ascending(self):
        """Em empate de score, doc_id MENOR ganha."""
        r = Ranker(k=3)
        candidates = [(0, 2.0), (1, 2.0), (2, 1.0), (3, 2.0), (4, 3.0)]
        result = r.top_k(candidates)
        self.assertEqual(result, [(4, 3.0), (0, 2.0), (1, 2.0)])

    def test_all_same_score(self):
        r = Ranker(k=2)
        candidates = [(5, 1.0), (1, 1.0), (3, 1.0), (2, 1.0)]
        result = r.top_k(candidates)
        # Empate total -> doc_ids menores ganham
        result_ids = [d for d, s in result]
        self.assertEqual(result_ids, [1, 2])

    def test_negative_scores_handled(self):
        """Scores negativos sao validos (improvavel mas defensivel)."""
        r = Ranker(k=2)
        candidates = [(0, -1.0), (1, -0.5), (2, -2.0)]
        result = r.top_k(candidates)
        # Maior é menos negativo
        self.assertEqual(result[0], (1, -0.5))

    def test_large_input_performance(self):
        """Heap deve ser O(N log K), nao O(N log N)."""
        import random, time
        random.seed(42)
        candidates = [(i, random.random()) for i in range(100_000)]

        r = Ranker(k=10)
        start = time.perf_counter()
        result = r.top_k(candidates)
        elapsed = time.perf_counter() - start

        # 100k candidatos devem ser processados em < 1 segundo
        self.assertLess(elapsed, 1.0)
        self.assertEqual(len(result), 10)


if __name__ == "__main__":
    unittest.main()
