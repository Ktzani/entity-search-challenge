"""Testes do Tokenizer e Normalizer."""

import unittest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import ensure_nltk_data  # noqa: garantir setup

from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.normalizer import Normalizer


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_simple_text(self):
        tokens = self.tokenizer.tokenize("Hello World")
        self.assertEqual(tokens, ["hello", "world"])

    def test_empty_text(self):
        self.assertEqual(self.tokenizer.tokenize(""), [])
        self.assertEqual(self.tokenizer.tokenize("   "), [])

    def test_only_punctuation_dropped(self):
        tokens = self.tokenizer.tokenize("!!! ??? ...")
        self.assertEqual(tokens, [])

    def test_contractions_residuals_removed(self):
        tokens = self.tokenizer.tokenize("Don't worry, it's a test.")
        # "n't" e "'s" devem ser descartados
        self.assertNotIn("n't", tokens)
        self.assertNotIn("'s", tokens)
        self.assertIn("do", tokens)
        self.assertIn("worry", tokens)
        self.assertIn("test", tokens)

    def test_tokens_starting_with_apostrophe_dropped(self):
        tokens = self.tokenizer.tokenize("'em going home")
        self.assertNotIn("'em", tokens)
        self.assertIn("going", tokens)

    def test_numbers_kept_in_tokenizer(self):
        # Numeros sao mantidos no tokenizer; descartados no normalizer.
        tokens = self.tokenizer.tokenize("Year 2024 was great")
        self.assertIn("2024", tokens)

    def test_hyphenated_kept(self):
        tokens = self.tokenizer.tokenize("dance-punk band")
        self.assertIn("dance-punk", tokens)


class TestNormalizer(unittest.TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer()
        self.normalizer = Normalizer()

    def test_stopwords_removed(self):
        tokens = ["the", "physics", "is", "a", "science"]
        terms = self.normalizer.normalize(tokens)
        self.assertNotIn("the", terms)
        self.assertNotIn("is", terms)
        self.assertNotIn("a", terms)

    def test_stemming_applied(self):
        terms = self.normalizer.normalize(["running", "runs", "ran"])
        # Snowball stemmer aplica diferente; verifiquemos que pelo menos
        # 'running' e 'runs' tem mesmo stem
        self.assertEqual(terms[0], "run")
        self.assertEqual(terms[1], "run")

    def test_min_length_filter(self):
        # Tokens curtos demais (< 2) sao descartados
        terms = self.normalizer.normalize(["a", "ab", "abc"])
        self.assertNotIn("a", terms)
        self.assertIn("ab", terms)
        self.assertIn("abc", terms)

    def test_max_length_filter(self):
        long_token = "a" * 50  # acima do max=40
        terms = self.normalizer.normalize([long_token])
        self.assertEqual(terms, [])

    def test_pure_numeric_dropped(self):
        terms = self.normalizer.normalize(["1996", "physics", "2024"])
        self.assertNotIn("1996", terms)
        self.assertNotIn("2024", terms)
        self.assertIn("physic", terms)

    def test_alphanumeric_kept(self):
        # "19th" tem letras, nao é puro numero
        terms = self.normalizer.normalize(["19th", "century"])
        self.assertIn("19th", terms)

    def test_full_pipeline(self):
        text = "!!! is a dance-punk band that formed in Sacramento, California, in 1996."
        tokens = self.tokenizer.tokenize(text)
        terms = self.normalizer.normalize(tokens)
        # Verifiquemos que stopwords foram removidas, stemming aplicado,
        # e 1996 descartado.
        self.assertNotIn("is", terms)
        self.assertNotIn("a", terms)
        self.assertNotIn("1996", terms)
        self.assertIn("form", terms)  # 'formed' -> 'form'
        self.assertIn("band", terms)
        self.assertIn("sacramento", terms)


if __name__ == "__main__":
    unittest.main()
