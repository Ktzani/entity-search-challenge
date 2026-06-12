"""
Conjunctive Document-at-a-Time (DAAT) matching com galloping search.

Algoritmo:
    1. Para cada termo da query, carrega a inverted list em um Cursor.
    2. Se algum termo nao existe no indice -> retorna [] (curto-circuito).
    3. Ordena cursors por df crescente (menor lista primeiro).
    4. Loop: usa o doc_id maximo entre cursors como pivot, avanca todos
       os outros ate >= pivot. Se todos coincidem no pivot, é match.

Galloping (exponencial + binary): listas de tamanhos muito diferentes
forcam saltos grandes na lista longa. Linear seria O(gap); galloping
é O(log gap). Ganho substancial em queries conjuntivas.

Galloping: tecnica hibrida que combina busca exponencial seguida de
binary search. Na Fase 1, salta posicoes dobrando o step (1, 2, 4, 8...)
ate ultrapassar o alvo. Na Fase 2, executa binary search no intervalo
delimitado. Mais eficiente que linear para listas desiguais.

NAO é thread-safe; cada thread deve criar sua propria instancia.
"""

from dataclasses import dataclass

from src.index_store.inverted_index import InvertedIndex
from src.index_store.posting import Posting
from src.index_store.term_lexicon import TermLexicon

import heapq



@dataclass
class DAATResult:
    """Resultado de uma operacao de matching conjuntivo."""

    matched_doc_ids: list[int]
    # doc_id -> {term -> Posting} - usado pelo scorer (precisa do TF).
    matched_postings: dict[int, dict[str, Posting]]

    postings_scanned: int        # postings efetivamente examinadas
    postings_in_lists: int       # tamanho total das listas (sum of df)
    early_terminated: bool       # true se abortou cedo (curto-circuito)

    @property
    def skip_ratio(self) -> float:
        """Fracao de postings economizadas vs linear scan completo."""
        if self.postings_in_lists == 0:
            return 0.0
        return 1.0 - (self.postings_scanned / self.postings_in_lists)


class _Cursor:
    """
    Cursor sobre uma inverted list (postings ja carregadas em memoria).
    Suporta galloping search para avanco rapido.

    'scanned' conta posicoes DISTINTAS visitadas (set, nao multiset),
    para que binary search nao infle a metrica ao revisitar posicoes
    ja contadas no galloping.
    """

    def __init__(self, term: str, postings: list[Posting]):
        self.term = term
        self.postings = postings
        self.pos = 0
        self._visited: set[int] = set()
        if postings:
            self._visited.add(0)

    @property
    def current_doc_id(self) -> int:
        return self.postings[self.pos].doc_id

    @property
    def current_posting(self) -> Posting:
        return self.postings[self.pos]

    @property
    def scanned(self) -> int:
        return len(self._visited)

    def is_exhausted(self) -> bool:
        return self.pos >= len(self.postings)

    def advance_to(self, target: int):
        """
        Avanca a posicao ate o primeiro doc_id >= target via galloping
        search. Se nao houver, o cursor fica exhausted.
        """
        if self.is_exhausted():
            return

        if self.postings[self.pos].doc_id >= target:
            return

        # Fase 1: galloping exponencial.
        lo = self.pos
        jump = 1
        hi = lo + jump
        n = len(self.postings)

        while hi < n and self.postings[hi].doc_id < target:
            self._visited.add(hi)
            lo = hi
            jump *= 2
            hi = lo + jump

        hi = min(hi, n - 1)
        self._visited.add(hi)

        if self.postings[hi].doc_id < target:
            self.pos = n
            return

        # Fase 2: binary search em [lo+1, hi]
        # (lo tem doc_id < target; hi tem doc_id >= target).
        left = lo + 1
        right = hi
        while left < right:
            mid = (left + right) // 2
            self._visited.add(mid)
            if self.postings[mid].doc_id < target:
                left = mid + 1
            else:
                right = mid
        self.pos = left
        self._visited.add(self.pos)

    def advance_one(self):
        """Avanca uma posicao (apos match) para evitar reprocess."""
        self.pos += 1
        if not self.is_exhausted():
            self._visited.add(self.pos)


class ConjunctiveDAAT:
    """
    Matching conjuntivo (AND) entre as listas de varios termos.

    Pre-carrega todas as listas em memoria. Para o corpus deste TP
    (4.6M docs), mesmo termos populares cabem em poucos MB; o
    trade-off favorece simplicidade.
    """

    def __init__(self, lexicon: TermLexicon, inverted_index: InvertedIndex):
        self._lexicon = lexicon
        self._ii = inverted_index

    def intersect(self, terms: list[str]) -> DAATResult:
        """
        Encontra todos os documentos que contem TODOS os termos.

        Retorna um DAATResult com:
            matched_doc_ids: docs que casam, em ordem crescente
            matched_postings: para cada doc_match, dict term -> Posting
                              (necessario para o scorer calcular TF*IDF)
            postings_scanned: numero de postings examinadas
            postings_in_lists: numero total de postings nas listas
                               (= soma dos df dos termos)
            early_terminated: true se abortou cedo
        """
        if not terms:
            return DAATResult(
                matched_doc_ids=[],
                matched_postings={},
                postings_scanned=0,
                postings_in_lists=0,
                early_terminated=False,
            )

        # Curto-circuito: se algum termo nao existe no lexicon, a
        # intersecao conjuntiva é vazia sem ler disco.
        term_entries: list[tuple[str, int, int]] = []  # (term, offset, df)
        for term in terms:
            entry = self._lexicon.get_entry(term)
            if entry is None:
                return DAATResult(
                    matched_doc_ids=[],
                    matched_postings={},
                    postings_scanned=0,
                    postings_in_lists=0,
                    early_terminated=True,
                )
            offset, df = entry
            term_entries.append((term, offset, df))

        # Menor lista controla o numero de iteracoes do loop.
        term_entries.sort(key=lambda x: x[2])

        cursors: list[_Cursor] = []
        total_postings_in_lists = 0
        for term, offset, df in term_entries:
            postings = self._ii.read_postings(offset, df)
            cursors.append(_Cursor(term, postings))
            total_postings_in_lists += df

        matched_doc_ids: list[int] = []
        matched_postings: dict[int, dict[str, Posting]] = {}

        while True:
            if any(c.is_exhausted() for c in cursors):
                break

            pivot = max(c.current_doc_id for c in cursors)

            for c in cursors:
                if c.current_doc_id < pivot:
                    c.advance_to(pivot)

            if any(c.is_exhausted() for c in cursors):
                break

            if all(c.current_doc_id == pivot for c in cursors):
                matched_doc_ids.append(pivot)
                postings_for_doc = {}
                for c in cursors:
                    postings_for_doc[c.term] = c.current_posting
                matched_postings[pivot] = postings_for_doc
                for c in cursors:
                    c.advance_one()

        total_scanned = sum(c.scanned for c in cursors)

        return DAATResult(
            matched_doc_ids=matched_doc_ids,
            matched_postings=matched_postings,
            postings_scanned=total_scanned,
            postings_in_lists=total_postings_in_lists,
            early_terminated=False,
        )


###############################################################################
# DisjunctiveDAAT - matching disjuntivo (OR) para o Research Challenge
#
# Em contraste com ConjunctiveDAAT (que requer que todos os termos casem),
# DisjunctiveDAAT coleta como candidato qualquer documento que case com
# PELO MENOS UM dos termos da consulta. Termos ausentes do lexicon sao
# silenciosamente pulados.
#
# Necessario para recuperacao top-K disjuntiva (Kaggle pede top-100, que
# raramente seria preenchido com matching conjuntivo AND strict).
#
# Implementacao com min-heap sobre cursors:
#   - Mantem heap (doc_id, cursor_idx) com a posicao corrente de cada cursor
#   - A cada iteracao, extrai o menor doc_id; se varios cursors casarem o
#     mesmo doc_id, todos contribuem com sua posting; cursor avanca e
#     eh reinserido na heap se ainda tiver postings
#   - Itera ate todos os cursors se esgotarem
#
# Complexidade: O(sum(df_t) * log(n_terms)) - mais caro que conjunctive
# mas necessario.
#
# Interface compativel com ConjunctiveDAAT (reusa DAATResult e _Cursor).
###############################################################################
class DisjunctiveDAAT:
    """
    Matching disjuntivo (OR) com DAAT.
 
    Coleta qualquer documento que case com pelo menos um termo da consulta.
    Termos nao presentes no lexicon sao silenciosamente pulados (em vez de
    fazer curto-circuito como no ConjunctiveDAAT).
    """
 
    def __init__(self, lexicon: TermLexicon, inverted_index: InvertedIndex):
        self._lexicon = lexicon
        self._ii = inverted_index
 
    def intersect(self, terms: list[str]) -> DAATResult:
        """
        Coleta candidatos disjuntivamente.
 
        Apesar do nome 'intersect' (mantido por compatibilidade com a
        interface do ConjunctiveDAAT), aqui a semantica eh UNION:
        retorna todos os doc_ids que casam ao menos um termo.
        """
        # 1. Resolve termos no lexicon, ignorando os ausentes
        cursors: list[_Cursor] = []
        postings_in_lists = 0
        for term in terms:
            entry = self._lexicon.get_entry(term)
            if entry is None:
                continue  # termo desconhecido: pula silenciosamente
            offset, df = entry
            postings = self._ii.read_postings(offset, df)
            if not postings:
                continue
            cursors.append(_Cursor(term=term, postings=postings))
            postings_in_lists += df
 
        # Nenhum termo tem postings -> resultado vazio
        if not cursors:
            return DAATResult(
                matched_doc_ids=[],
                matched_postings={},
                postings_scanned=0,
                postings_in_lists=0,
                early_terminated=False,
            )
 
        # 2. Inicializa min-heap com (doc_id_atual, cursor_idx)
        # cursor_idx eh tie-breaker para evitar comparacao de _Cursor
        heap: list[tuple[int, int]] = []
        for idx, cursor in enumerate(cursors):
            heap.append((cursor.current_doc_id, idx))
        heapq.heapify(heap)
 
        # 3. Loop principal: itera doc_ids em ordem crescente, acumulando
        # contribuicao de cada cursor que aponta para o mesmo doc_id
        matched_doc_ids: list[int] = []
        matched_postings: dict[int, dict[str, Posting]] = {}
        postings_scanned = 0
 
        while heap:
            current_doc_id = heap[0][0]
            current_doc_postings: dict[str, Posting] = {}
 
            # Extrai todos os cursors que apontam para o MESMO doc_id atual
            while heap and heap[0][0] == current_doc_id:
                _, cursor_idx = heapq.heappop(heap)
                cursor = cursors[cursor_idx]
 
                # Adiciona contribuicao desse termo ao doc atual
                current_doc_postings[cursor.term] = cursor.current_posting
                postings_scanned += 1
 
                # Avanca cursor; se ainda tem postings, reinsere na heap
                cursor.advance_one()
                if not cursor.is_exhausted():
                    heapq.heappush(heap, (cursor.current_doc_id, cursor_idx))
 
            # Registra o doc atual com todas as contribuicoes coletadas
            matched_doc_ids.append(current_doc_id)
            matched_postings[current_doc_id] = current_doc_postings
 
        return DAATResult(
            matched_doc_ids=matched_doc_ids,
            matched_postings=matched_postings,
            postings_scanned=postings_scanned,
            postings_in_lists=postings_in_lists,
            early_terminated=False,
        )