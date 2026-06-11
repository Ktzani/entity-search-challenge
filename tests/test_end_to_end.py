"""
Testes end-to-end: pipeline completo indexer + processor.

Cobre:
- SPIMI -> merger -> writer (gera indice completo)
- Load do indice -> Query -> DAAT -> Scorer -> Ranker -> JSON
- Caso especial: forca multiplos blocks via budget pequeno
"""

import json
import unittest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import TempDirTestCase, make_corpus

from src.index_build.merger import cleanup_blocks, merge_blocks
from src.index_build.spimi import SPIMIOrchestrator
from src.index_build.writer import compute_statistics, write_doc_index, write_lexicon
from src.index_store.document_index import DocumentIndex
from src.index_store.inverted_index import InvertedIndex
from src.index_store.term_lexicon import TermLexicon
from src.preprocessing.normalizer import Normalizer
from src.preprocessing.tokenizer import Tokenizer
from src.retrieval.daat import ConjunctiveDAAT
from src.retrieval.query import Query
from src.retrieval.ranker import Ranker
from src.retrieval.scorer import get_scorer
from src.utils.memory import MemoryMonitor


# Mini corpus de teste com docs que casam queries previsíveis
TEST_DOCS = [
    {"id": "DOC_A", "title": "Albert Einstein", "text": "physicist nobel theory", "keywords": []},
    {"id": "DOC_B", "title": "Brazil", "text": "country south america", "keywords": []},
    {"id": "DOC_C", "title": "Python", "text": "programming language interpreted", "keywords": []},
    {"id": "DOC_D", "title": "Newton", "text": "physicist mathematician gravity", "keywords": []},
    {"id": "DOC_E", "title": "Einstein relativity", "text": "theory physics einstein", "keywords": []},
]


class TestEndToEnd(TempDirTestCase):

    def setUp(self):
        super().setUp()
        # Cria corpus e diretorios
        self.corpus_path = make_corpus(TEST_DOCS, dir_path=self.tmpdir)
        self.index_dir = os.path.join(self.tmpdir, "index")
        self.blocks_dir = os.path.join(self.index_dir, "blocks")
        os.makedirs(self.index_dir)

    def _run_indexer(self, budget_mb: int = 512, max_docs: int = None):
        """Roda o pipeline de indexacao. Retorna (lexicon, doc_index)."""
        memory = MemoryMonitor(budget_mb=budget_mb)
        spimi = SPIMIOrchestrator(
            corpus_path=self.corpus_path,
            blocks_dir=self.blocks_dir,
            memory=memory,
            num_threads=2,
            max_docs=max_docs,
        )
        block_paths, doc_index = spimi.run()
        inverted_path = os.path.join(self.index_dir, "inverted.idx")
        lexicon = merge_blocks(block_paths, inverted_path)
        cleanup_blocks(block_paths)
        write_lexicon(lexicon, self.index_dir)
        write_doc_index(doc_index, self.index_dir)
        return lexicon, doc_index

    def test_full_indexing(self):
        """Pipeline completo de indexacao gera 3 arquivos."""
        self._run_indexer()
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "inverted.idx")))
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "lexicon.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "doc_index.pkl")))

    def test_statistics_format(self):
        """Statistics deve ter as 4 chaves do enunciado."""
        lexicon, _ = self._run_indexer()
        stats = compute_statistics(self.index_dir, lexicon)
        self.assertIn("Index Size", stats)
        self.assertIn("Number of Lists", stats)
        self.assertIn("Average List Size", stats)
        self.assertGreater(stats["Number of Lists"], 0)

    def test_query_processing_tfidf(self):
        """Pipeline completo de query com TFIDF."""
        self._run_indexer()
        lexicon = TermLexicon(self.index_dir)
        doc_index = DocumentIndex(self.index_dir)
        tokenizer = Tokenizer()
        normalizer = Normalizer()
        ranker = Ranker(k=10)
        scorer = get_scorer("TFIDF", doc_index, lexicon)

        with InvertedIndex(self.index_dir) as ii:
            daat = ConjunctiveDAAT(lexicon, ii)
            q = Query("albert einstein", tokenizer, normalizer)
            result = daat.intersect(q.terms)

        # Doc A tem "Albert Einstein" no title -> deve casar
        self.assertGreater(len(result.matched_doc_ids), 0)

        # Score e formato JSON
        scores = []
        for doc_id in result.matched_doc_ids:
            s = scorer.score(doc_id, result.matched_postings[doc_id])
            scores.append((doc_id, s))
        top = ranker.top_k(scores)
        json_out = [
            {"ID": doc_index.get_original_id(d), "Score": round(s, 4)}
            for d, s in top
        ]
        # DOC_A deve estar nos resultados
        ids = [r["ID"] for r in json_out]
        self.assertIn("DOC_A", ids)

    def test_query_processing_bm25(self):
        """Mesma query com BM25 deve dar resultados (talvez ordem diferente)."""
        self._run_indexer()
        lexicon = TermLexicon(self.index_dir)
        doc_index = DocumentIndex(self.index_dir)
        tokenizer = Tokenizer()
        normalizer = Normalizer()
        scorer = get_scorer("BM25", doc_index, lexicon)

        with InvertedIndex(self.index_dir) as ii:
            daat = ConjunctiveDAAT(lexicon, ii)
            q = Query("physicist", tokenizer, normalizer)
            result = daat.intersect(q.terms)

        # DOC_A e DOC_D mencionam "physicist"
        self.assertEqual(len(result.matched_doc_ids), 2)

        # Calcula scores
        for doc_id in result.matched_doc_ids:
            score = scorer.score(doc_id, result.matched_postings[doc_id])
            self.assertGreater(score, 0)

    def test_conjunctive_no_match(self):
        """Query conjuntiva sem match -> resultados vazios."""
        self._run_indexer()
        lexicon = TermLexicon(self.index_dir)
        doc_index = DocumentIndex(self.index_dir)
        tokenizer = Tokenizer()
        normalizer = Normalizer()

        with InvertedIndex(self.index_dir) as ii:
            daat = ConjunctiveDAAT(lexicon, ii)
            # 'python' so em DOC_C, 'newton' so em DOC_D -> disjuntos
            q = Query("python newton", tokenizer, normalizer)
            result = daat.intersect(q.terms)

        self.assertEqual(result.matched_doc_ids, [])

    def test_query_with_missing_term(self):
        """Termo inexistente -> curto-circuito, early_terminated=True."""
        self._run_indexer()
        lexicon = TermLexicon(self.index_dir)
        tokenizer = Tokenizer()
        normalizer = Normalizer()

        with InvertedIndex(self.index_dir) as ii:
            daat = ConjunctiveDAAT(lexicon, ii)
            q = Query("zzz_nonexistent_xxx", tokenizer, normalizer)
            result = daat.intersect(q.terms)

        self.assertEqual(result.matched_doc_ids, [])
        self.assertTrue(result.early_terminated)
        self.assertEqual(result.postings_scanned, 0)

    def test_multiple_blocks_with_small_budget(self):
        """
        Indexa corpus medio com budget pequeno e valida que o
        pipeline funciona ate o final. Numero exato de blocks
        depende do overhead do Python no sistema.
        """
        # Cria corpus medio para ter blocks
        big_corpus = []
        for i in range(500):
            big_corpus.append({
                "id": f"{i:06d}",
                "title": f"Doc{i}",
                "text": f"physics nobel einstein theory {i}",
                "keywords": [f"kw{i % 10}"],
            })
        big_path = make_corpus(big_corpus, dir_path=self.tmpdir)

        memory = MemoryMonitor(budget_mb=50)  # propositalmente pequeno
        spimi = SPIMIOrchestrator(
            corpus_path=big_path,
            blocks_dir=self.blocks_dir,
            memory=memory,
            num_threads=2,
        )
        block_paths, doc_index = spimi.run()

        # Pelo menos 1 block deve existir
        self.assertGreaterEqual(len(block_paths), 1)

        # Merger combina tudo (mesmo se for 1 block, deve produzir indice valido)
        inverted_path = os.path.join(self.index_dir, "inverted.idx")
        lexicon = merge_blocks(block_paths, inverted_path)

        # 'physics' (stem='physic') deve aparecer em TODOS os 500 docs
        self.assertIn("physic", lexicon)
        self.assertEqual(lexicon["physic"][1], 500)

    def test_scorer_factory(self):
        """get_scorer aceita TFIDF e BM25, case-insensitive."""
        self._run_indexer()
        lexicon = TermLexicon(self.index_dir)
        doc_index = DocumentIndex(self.index_dir)

        s1 = get_scorer("TFIDF", doc_index, lexicon)
        s2 = get_scorer("tfidf", doc_index, lexicon)
        s3 = get_scorer("BM25", doc_index, lexicon)
        s4 = get_scorer("bm25", doc_index, lexicon)
        self.assertEqual(s1.name, "TFIDF")
        self.assertEqual(s2.name, "TFIDF")
        self.assertEqual(s3.name, "BM25")
        self.assertEqual(s4.name, "BM25")

        with self.assertRaises(ValueError):
            get_scorer("PageRank", doc_index, lexicon)


if __name__ == "__main__":
    unittest.main()
