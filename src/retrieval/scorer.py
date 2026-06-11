"""
Scorers: TF-IDF e BM25.

TF-IDF (variante ltn x lnc sem cosine — ranking preservado, so muda
scale):
    score(doc, q) = sum_{t em q} (1 + log(tf_doc)) * log(N / df_t)

Como Query.terms é deduplicado, o tf_query é sempre 1.

BM25 (Robertson):
    score(doc, q) = sum_{t em q}
                    idf(t) * (tf * (k1+1)) /
                    (tf + k1 * (1 - b + b * |doc|/avgdl))
    idf(t) = log((N - df + 0.5) / (df + 0.5) + 1)
    (forma "BM25+" do Lucene, evita IDF negativo)

Hiperparametros (config/processor.py): k1=1.2, b=0.75.
"""

import math
from abc import ABC, abstractmethod

from src.config.processor import BM25_B, BM25_K1
from src.index_store.document_index import DocumentIndex
from src.index_store.posting import Posting
from src.index_store.term_lexicon import TermLexicon


class Scorer(ABC):
    """Interface comum para scorers."""

    def __init__(
        self,
        doc_index: DocumentIndex,
        lexicon: TermLexicon,
    ):
        self._doc_index = doc_index
        self._lexicon = lexicon
        self._num_docs = doc_index.num_docs()

    @abstractmethod
    def score(
        self,
        doc_id: int,
        query_postings: dict[str, Posting],
    ) -> float:
        """
        Calcula o score do doc_id dado as postings dos termos da query
        que existem no doc.

        Parametros:
            doc_id: id interno do documento
            query_postings: dict term -> Posting (do DAATResult)

        Retorna:
            score escalar (maior = mais relevante)
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome do scorer (para debug/log)."""
        ...


class TFIDFScorer(Scorer):
    """TF-IDF variante lnc.ltn (sem cosine no doc)."""

    @property
    def name(self) -> str:
        return "TFIDF"

    def score(
        self,
        doc_id: int,
        query_postings: dict[str, Posting],
    ) -> float:
        score = 0.0
        for term, posting in query_postings.items():
            tf = posting.tf
            df = self._lexicon.get_df(term)
            if df == 0:
                continue
            log_tf = 1.0 + math.log(tf)
            idf = math.log(self._num_docs / df)
            score += log_tf * idf
        return score


class BM25Scorer(Scorer):
    """BM25 (Robertson) com IDF "BM25+" (sem negativos)."""

    def __init__(
        self,
        doc_index: DocumentIndex,
        lexicon: TermLexicon,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ):
        super().__init__(doc_index, lexicon)
        self._k1 = k1
        self._b = b
        self._avgdl = doc_index.avg_doc_length()

    @property
    def name(self) -> str:
        return "BM25"

    def score(
        self,
        doc_id: int,
        query_postings: dict[str, Posting],
    ) -> float:
        doc_length = self._doc_index.get_length(doc_id)

        normalization = 1.0 - self._b + self._b * (doc_length / self._avgdl)

        score = 0.0
        for term, posting in query_postings.items():
            tf = posting.tf
            df = self._lexicon.get_df(term)
            if df == 0:
                continue
            idf = math.log((self._num_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf_component = (tf * (self._k1 + 1.0)) / (tf + self._k1 * normalization)
            score += idf * tf_component
        return score


def get_scorer(
    name: str,
    doc_index: DocumentIndex,
    lexicon: TermLexicon,
) -> Scorer:
    """Factory: 'TFIDF' ou 'BM25' (case-insensitive)."""
    name_normalized = name.strip().upper()
    if name_normalized == "TFIDF":
        return TFIDFScorer(doc_index, lexicon)
    if name_normalized == "BM25":
        return BM25Scorer(doc_index, lexicon)
    raise ValueError(f"Scorer desconhecido: {name!r}. Use 'TFIDF' ou 'BM25'.")
