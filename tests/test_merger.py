"""Testes do k-way merger externo."""

import os
import unittest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.conftest import TempDirTestCase

from src.index_build.merger import merge_blocks, cleanup_blocks
from src.index_build.partial_index import PartialIndex
from src.utils.io_utils import read_posting, read_uint32


def _read_block(path: str) -> list[tuple[str, list[tuple[int, int]]]]:
    """Helper: le um arquivo de inverted_index final e retorna [(term, postings)]."""
    # Note: arquivo final NAO tem term_length/term, so num_postings + postings
    # Esse helper é para validar atraves do lexicon.
    raise NotImplementedError(
        "Use o lexicon retornado por merge_blocks() para validar"
    )


class TestMerger(TempDirTestCase):

    def _build_block(self, block_idx: int, contents: dict) -> str:
        """
        Constroi um block parcial a partir de um dict
        {term: [(doc_id, tf), ...]}. Retorna o caminho.
        """
        pi = PartialIndex()
        # Construimos diretamente o dict interno para controle total
        # sobre o conteudo do block.
        for term, postings in contents.items():
            for doc_id, tf in postings:
                # Re-cria via add_document para passar pelas validacoes
                # Hack: cada chamada cria 1 posting unica.
                pi._index[term].append((doc_id, tf))
                pi._num_postings += 1
        # marca docs (nao usado pelo merger, mas para is_empty)
        pi._num_docs = 1

        path = os.path.join(self.tmpdir, f"block_{block_idx:05d}.bin")
        pi.dump_to_disk(path)
        return path

    def test_empty_block_list(self):
        out = os.path.join(self.tmpdir, "out.idx")
        lex = merge_blocks([], out)
        self.assertEqual(lex, {})
        self.assertTrue(os.path.exists(out))

    def test_single_block_passthrough(self):
        """Merge de 1 block deve produzir as mesmas listas."""
        b0 = self._build_block(0, {
            "albert": [(0, 2), (5, 1)],
            "einstein": [(0, 1), (3, 4)],
        })

        out = os.path.join(self.tmpdir, "out.idx")
        lex = merge_blocks([b0], out)

        self.assertEqual(len(lex), 2)
        self.assertIn("albert", lex)
        self.assertIn("einstein", lex)

        # Le postings finais usando o lexicon
        with open(out, "rb") as f:
            offset, df = lex["albert"]
            f.seek(offset)
            n = read_uint32(f)
            self.assertEqual(n, df)
            self.assertEqual(n, 2)
            postings = [read_posting(f) for _ in range(n)]
            self.assertEqual(postings, [(0, 2), (5, 1)])

    def test_merge_two_blocks_disjoint_terms(self):
        b0 = self._build_block(0, {"albert": [(0, 1)]})
        b1 = self._build_block(1, {"einstein": [(1, 1)]})

        out = os.path.join(self.tmpdir, "out.idx")
        lex = merge_blocks([b0, b1], out)

        self.assertEqual(len(lex), 2)
        self.assertIn("albert", lex)
        self.assertIn("einstein", lex)

    def test_merge_combines_postings_same_term(self):
        """Mesmo termo em multiplos blocks -> postings combinadas."""
        b0 = self._build_block(0, {"physics": [(0, 1), (2, 2)]})
        b1 = self._build_block(1, {"physics": [(5, 3), (8, 1)]})

        out = os.path.join(self.tmpdir, "out.idx")
        lex = merge_blocks([b0, b1], out)

        offset, df = lex["physics"]
        self.assertEqual(df, 4)

        with open(out, "rb") as f:
            f.seek(offset)
            n = read_uint32(f)
            postings = [read_posting(f) for _ in range(n)]

        # Postings devem estar ordenadas por doc_id
        doc_ids = [p[0] for p in postings]
        self.assertEqual(doc_ids, [0, 2, 5, 8])

    def test_merge_many_blocks(self):
        """Stress: 10 blocks pequenos com termos sobrepostos."""
        blocks = []
        for i in range(10):
            # Cada bloco contribui 2 postings para 'shared' e 1 unica
            blocks.append(self._build_block(i, {
                "shared": [(i, 1)],
                f"unique_{i}": [(i * 100, 1)],
            }))

        out = os.path.join(self.tmpdir, "out.idx")
        lex = merge_blocks(blocks, out)

        # 1 termo compartilhado + 10 unicos = 11 entradas
        self.assertEqual(len(lex), 11)

        # 'shared' deve ter df=10
        self.assertEqual(lex["shared"][1], 10)

    def test_alphabetical_offsets(self):
        """Termos no arquivo final ficam em ordem alfabetica de offset."""
        b0 = self._build_block(0, {
            "zebra": [(0, 1)],
            "alpha": [(0, 1)],
            "mango": [(0, 1)],
        })

        out = os.path.join(self.tmpdir, "out.idx")
        lex = merge_blocks([b0], out)

        # Offsets devem ser crescentes em ordem alfabetica
        offset_alpha = lex["alpha"][0]
        offset_mango = lex["mango"][0]
        offset_zebra = lex["zebra"][0]
        self.assertLess(offset_alpha, offset_mango)
        self.assertLess(offset_mango, offset_zebra)

    def test_cleanup_blocks(self):
        b0 = self._build_block(0, {"a": [(0, 1)]})
        b1 = self._build_block(1, {"b": [(0, 1)]})
        self.assertTrue(os.path.exists(b0))
        self.assertTrue(os.path.exists(b1))

        cleanup_blocks([b0, b1])
        self.assertFalse(os.path.exists(b0))
        self.assertFalse(os.path.exists(b1))


if __name__ == "__main__":
    unittest.main()
