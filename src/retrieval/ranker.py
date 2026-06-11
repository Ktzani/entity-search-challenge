"""
Ranker: top-K documentos por score usando min-heap.

Min-heap de tamanho fixo K com (score, -doc_id). heap[0] é o
elemento mais facil de descartar; novos candidatos so entram se
score > heap[0] (ou empate com doc_id menor). Min-heap é ideal
porque o gate para o top-K é o MENOR score da heap.

Complexidade: O(N log K) para inserir, O(K log K) para extrair.
"""

import heapq

from src.config.processor import TOP_K


class Ranker:
    """Seleciona o top-K de candidatos por score usando min-heap."""

    def __init__(self, k: int = TOP_K):
        if k <= 0:
            raise ValueError(f"k deve ser positivo, recebido {k}")
        self._k = k

    def top_k(
        self,
        candidates: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """
        Seleciona os k candidatos com maior score.

        Parametros:
            candidates: lista de (doc_id, score)

        Retorna:
            lista de (doc_id, score) ordenada por score DECRESCENTE.
            Em caso de empate, ordena por doc_id crescente.
        """
        if not candidates:
            return []

        # Negamos doc_id para que empate de score considere doc_id
        # MAIOR como "menor" na heap (mais facil de descartar) - assim
        # mantemos doc_ids MENORES no top-K, deterministico.
        heap: list[tuple[float, int]] = []

        for doc_id, score in candidates:
            if len(heap) < self._k:
                heapq.heappush(heap, (score, -doc_id))
            elif score > heap[0][0] or (
                score == heap[0][0] and -doc_id > heap[0][1]
            ):
                heapq.heappushpop(heap, (score, -doc_id))

        result = [(-neg_doc_id, score) for score, neg_doc_id in heap]
        result.sort(key=lambda x: (-x[1], x[0]))
        return result
