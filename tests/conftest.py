"""
Helpers compartilhados pelos testes. Centraliza criacao de
corpus sinteticos, garante setup do NLTK uma vez, etc.
"""

import json
import tempfile
import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# Garante que o NLTK esta pronto antes de qualquer teste rodar.
# Idempotente: faz download na primeira execucao apenas.
from src.preprocessing.nltk_setup import ensure_nltk_data
ensure_nltk_data()


def make_corpus(docs: list[dict], dir_path: str = None) -> str:
    """
    Cria um arquivo JSONL temporario com os documentos fornecidos.
    Retorna o caminho. O caller é responsavel pela remocao.
    """
    if dir_path:
        fd, path = tempfile.mkstemp(suffix=".jsonl", dir=dir_path)
    else:
        fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return path


class TempDirTestCase(unittest.TestCase):
    """
    TestCase base que cria um diretorio temporario por teste e o
    limpa ao final. Subclasses tem self.tmpdir disponivel.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="pa2_test_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
