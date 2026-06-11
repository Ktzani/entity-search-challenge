"""Testes do CorpusReader."""

import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from tests.conftest import TempDirTestCase, make_corpus

from src.index_build.corpus_reader import CorpusReader


class TestCorpusReader(TempDirTestCase):

    def test_simple_corpus(self):
        path = make_corpus([
            {"id": "001", "title": "T1", "text": "text1", "keywords": ["k1"]},
            {"id": "002", "title": "T2", "text": "text2", "keywords": ["k2"]},
        ], dir_path=self.tmpdir)

        reader = CorpusReader(path)
        docs = list(reader)
        self.assertEqual(len(docs), 2)
        self.assertEqual(reader.docs_read, 2)
        self.assertEqual(reader.docs_skipped, 0)
        self.assertEqual(docs[0].id, "001")
        self.assertEqual(docs[0].title, "T1")
        self.assertEqual(docs[1].keywords, ["k2"])

    def test_max_docs_limit(self):
        path = make_corpus([
            {"id": f"{i:03d}", "title": "", "text": "txt", "keywords": []}
            for i in range(10)
        ], dir_path=self.tmpdir)

        reader = CorpusReader(path, max_docs=3)
        docs = list(reader)
        self.assertEqual(len(docs), 3)
        self.assertEqual(reader.docs_read, 3)

    def test_tolerates_invalid_json(self):
        """Linhas com JSON invalido sao puladas, nao quebram a leitura."""
        # Cria arquivo misto manualmente (make_corpus serializa tudo OK)
        path = os.path.join(self.tmpdir, "mixed.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"id":"001","title":"OK","text":"","keywords":[]}\n')
            f.write('isso nao é JSON\n')
            f.write('{"id":"002","title":"OK2","text":"","keywords":[]}\n')

        reader = CorpusReader(path)
        docs = list(reader)
        self.assertEqual(len(docs), 2)
        self.assertEqual(reader.docs_read, 2)
        self.assertEqual(reader.docs_skipped, 1)

    def test_tolerates_missing_id(self):
        path = os.path.join(self.tmpdir, "no_id.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"id":"001","title":"OK","text":"","keywords":[]}\n')
            f.write('{"title":"sem id","text":"","keywords":[]}\n')

        reader = CorpusReader(path)
        docs = list(reader)
        self.assertEqual(len(docs), 1)
        self.assertEqual(reader.docs_skipped, 1)

    def test_optional_fields(self):
        """Docs sem title/text/keywords ainda sao validos se tiverem id."""
        path = os.path.join(self.tmpdir, "minimal.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"id":"001"}\n')

        reader = CorpusReader(path)
        docs = list(reader)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].id, "001")
        self.assertEqual(docs[0].title, "")
        self.assertEqual(docs[0].text, "")
        self.assertEqual(docs[0].keywords, [])

    def test_id_coerced_to_string(self):
        """Se id vier como int, deve ser convertido para string."""
        path = os.path.join(self.tmpdir, "int_id.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"id":42,"title":"OK","text":"","keywords":[]}\n')

        reader = CorpusReader(path)
        docs = list(reader)
        self.assertEqual(docs[0].id, "42")
        self.assertIsInstance(docs[0].id, str)

    def test_full_content(self):
        """Document.full_content() concatena title + text + keywords."""
        path = make_corpus([
            {"id": "001", "title": "Title", "text": "body", "keywords": ["k1", "k2"]},
        ], dir_path=self.tmpdir)

        reader = CorpusReader(path)
        docs = list(reader)
        content = docs[0].full_content()
        self.assertIn("Title", content)
        self.assertIn("body", content)
        self.assertIn("k1", content)
        self.assertIn("k2", content)


if __name__ == "__main__":
    unittest.main()
