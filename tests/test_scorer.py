"""
Testes do Scorer (TFIDF e BM25).

Esses testes exigem TermLexicon e DocumentIndex carregados. Para
isolar a logica de scoring sem precisar de um indice real,
usamos mocks minimos.
"""

import math
import unittest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _MockLexicon:
    """Mock minimo do TermLexicon."""

    def __init__(self, dfs: dict[str, int]):
        self._dfs = dfs

    def get_df(self, term: str) -> int:
        return self._dfs.get(term, 0)


class _MockDocIndex:
    """Mock minimo do DocumentIndex."""

    def __init__(self, lengths: dict[int, int]):
        self._lengths = lengths
        self._avg = (
            sum(lengths.values()) / len(lengths) if lengths else 0.0
        )

    def get_length(self, doc_id: int) -> int:
        return self._lengths[doc_id]

    def avg_doc_length(self) -> float:
        return self._avg

    def num_docs(self) -> int:
        return len(self._lengths)


class _MockPosting:
    """Posting com .tf - so o que o scorer precisa."""

    def __init__(self, tf: int):
        self.tf = tf


class TestTFIDFScorer(unittest.TestCase):

    def _make_scorer(self, num_docs: int, dfs: dict, lengths: dict):
        """Helper: cria scorer com mocks consistentes."""
        from src.retrieval.scorer import TFIDFScorer
        # Garante que num_docs >= max(df) para evitar IDF negativo
        assert all(df <= num_docs for df in dfs.values()), \
            "df nao pode exceder num_docs"
        # Adiciona docs ate atingir num_docs
        full_lengths = dict(lengths)
        while len(full_lengths) < num_docs:
            full_lengths[len(full_lengths)] = 100
        return TFIDFScorer(_MockDocIndex(full_lengths), _MockLexicon(dfs))

    def test_basic_score(self):
        # 100 docs no corpus, 'einstein' aparece em 10
        scorer = self._make_scorer(
            num_docs=100,
            dfs={"einstein": 10},
            lengths={0: 50, 1: 100},
        )

        # Doc 0 tem tf=2 para 'einstein'
        score = scorer.score(0, {"einstein": _MockPosting(2)})
        # esperado: (1 + log(2)) * log(100/10) = (1+0.693) * 2.302 ~= 3.90
        expected = (1.0 + math.log(2)) * math.log(100 / 10)
        self.assertAlmostEqual(score, expected, places=3)

    def test_higher_tf_higher_score(self):
        scorer = self._make_scorer(
            num_docs=100,
            dfs={"x": 50},
            lengths={0: 100, 1: 100},
        )

        score_low_tf = scorer.score(0, {"x": _MockPosting(1)})
        score_high_tf = scorer.score(0, {"x": _MockPosting(10)})
        self.assertGreater(score_high_tf, score_low_tf)

    def test_rarer_term_higher_score(self):
        scorer = self._make_scorer(
            num_docs=100,
            dfs={"rare": 1, "common": 50},
            lengths={0: 100, 1: 100},
        )

        score_rare = scorer.score(0, {"rare": _MockPosting(1)})
        score_common = scorer.score(0, {"common": _MockPosting(1)})
        self.assertGreater(score_rare, score_common)


class TestBM25Scorer(unittest.TestCase):

    def _make_scorer(self, num_docs: int, dfs: dict, lengths: dict):
        from src.retrieval.scorer import BM25Scorer
        assert all(df <= num_docs for df in dfs.values())
        full_lengths = dict(lengths)
        while len(full_lengths) < num_docs:
            full_lengths[len(full_lengths)] = 100
        return BM25Scorer(_MockDocIndex(full_lengths), _MockLexicon(dfs))

    def test_basic_score(self):
        scorer = self._make_scorer(
            num_docs=100,
            dfs={"einstein": 10},
            lengths={0: 100},
        )
        score = scorer.score(0, {"einstein": _MockPosting(2)})
        self.assertGreater(score, 0)

    def test_short_doc_gets_boost(self):
        """BM25 favorece docs onde o termo tem peso relativo maior."""
        scorer = self._make_scorer(
            num_docs=100,
            dfs={"x": 10},
            lengths={0: 10, 1: 1000},  # doc 0 curto, doc 1 longo
        )
        score_short = scorer.score(0, {"x": _MockPosting(1)})
        score_long = scorer.score(1, {"x": _MockPosting(1)})
        self.assertGreater(score_short, score_long)

    def test_tf_saturation(self):
        """TF saturation: dobrar TF nao dobra o score (BM25 satura)."""
        scorer = self._make_scorer(
            num_docs=100,
            dfs={"x": 10},
            lengths={0: 100},
        )
        score_tf1 = scorer.score(0, {"x": _MockPosting(1)})
        score_tf10 = scorer.score(0, {"x": _MockPosting(10)})
        ratio = score_tf10 / score_tf1
        self.assertLess(ratio, 5)


if __name__ == "__main__":
    unittest.main()
