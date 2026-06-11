"""
SPIMI Orchestrator.

Pipeline:
- 1 thread Reader le o JSONL em batches e atribui doc_ids sequenciais
- N threads Processor consomem batches: tokenize -> normalize -> add ao PartialIndex
- A thread principal monitora memoria e dispara flushes do PartialIndex

Saida: lista de paths de blocks parciais ordenados por termo, e um par
(original_ids, lengths) representando o document index.

Document index como arrays paralelos:
    Em vez de dict[int, dict] (que gastaria ~250-300 bytes/entry por
    causa do overhead do CPython), usamos dois arrays paralelos:
        original_ids: list[str]   (indexado por internal_id)
        lengths: array('I')       (uint32, 4 bytes/entry)

    - Para 4.6M docs, isso é ~70 MB em vez de ~1.2 GB.

"""

import array
import os
import queue
import sys
import threading
import time
import gc

from src.config.indexer import (
    NUM_PROCESSOR_THREADS,
    QUEUE_MAX_BATCHES,
    READER_BATCH_SIZE,
)
from src.index_build.corpus_reader import CorpusReader
from src.index_build.partial_index import PartialIndex
from src.preprocessing.tokenizer import Tokenizer
from src.preprocessing.normalizer import Normalizer
from src.utils.memory import MemoryMonitor

_END_OF_INPUT = object()

class SPIMIOrchestrator:
    """Coordena o pipeline de indexacao parcial."""

    def __init__(
        self,
        corpus_path: str,
        blocks_dir: str,
        memory: MemoryMonitor,
        num_threads: int = NUM_PROCESSOR_THREADS,
        max_docs: int | None = None,
    ):
        self._corpus_path = corpus_path
        self._blocks_dir = blocks_dir
        self._memory = memory
        self._num_threads = num_threads
        self._max_docs = max_docs

        os.makedirs(self._blocks_dir, exist_ok=True)

        self._pi = PartialIndex()
        self._pi_lock = threading.Lock()
        self._original_ids: list[str] = []
        self._lengths: array.array = array.array("I")
        self._doc_index_lock = threading.Lock()

        self._batch_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX_BATCHES)
        self._reader_done = threading.Event()
        self._flush_event = threading.Event()
        self._flush_done = threading.Event()
        self._stop_event = threading.Event()

        self._block_paths: list[str] = []
        self._next_block_idx = 0

        self._docs_processed = 0
        self._docs_processed_lock = threading.Lock()

    def run(self) -> tuple[list[str], tuple[list[str], array.array]]:
        """
        Executa o pipeline. Retorna (block_paths, (original_ids, lengths)).
        """
        reader_thread = threading.Thread(
            target=self._reader_loop, name="reader", daemon=True
        )
        reader_thread.start()

        processors = []
        for i in range(self._num_threads):
            t = threading.Thread(
                target=self._processor_loop, name=f"processor-{i}", daemon=True
            )
            t.start()
            processors.append(t)

        self._main_loop(reader_thread, processors)

        reader_thread.join(timeout=5.0)
        for t in processors:
            t.join(timeout=5.0)

        if not self._pi.is_empty():
            self._do_flush(final=True)

        return self._block_paths, (self._original_ids, self._lengths)

    def _reader_loop(self):
        """Le o JSONL, atribui doc_ids monotonicos e enfileira batches."""
        reader = CorpusReader(self._corpus_path, max_docs=self._max_docs)
        batch: list[tuple[int, object]] = []
        next_id = 0

        try:
            for doc in reader:
                if self._stop_event.is_set():
                    break
                batch.append((next_id, doc))
                next_id += 1
                if len(batch) >= READER_BATCH_SIZE:
                    self._enqueue_batch_blocking(batch)
                    batch = []
            if batch and not self._stop_event.is_set():
                self._enqueue_batch_blocking(batch)
        except Exception as e:
            print(f"[reader] erro fatal: {e}", file=sys.stderr)
            self._stop_event.set()
        finally:
            self._reader_done.set()
            for _ in range(self._num_threads):
                try:
                    self._batch_queue.put(_END_OF_INPUT, timeout=1.0)
                except queue.Full:
                    pass

    def _enqueue_batch_blocking(self, batch: list[tuple[int, object]]):
        while not self._stop_event.is_set():
            try:
                self._batch_queue.put(batch, timeout=0.5)
                return
            except queue.Full:
                continue

    def _processor_loop(self):
        """Consome batches: tokeniza, normaliza, adiciona ao PI."""
        tokenizer = Tokenizer()
        normalizer = Normalizer()

        while not self._stop_event.is_set():
            if self._flush_event.is_set():
                self._flush_done.wait(timeout=0.5)
                continue

            try:
                item = self._batch_queue.get(timeout=0.5)
            except queue.Empty:
                if self._reader_done.is_set() and self._batch_queue.empty():
                    return
                continue

            if item is _END_OF_INPUT:
                return

            batch = item
            for internal_id, doc in batch:
                if self._stop_event.is_set():
                    return

                # Tokeniza/normaliza FORA do lock (parte cara).
                content = doc.full_content()
                tokens = tokenizer.tokenize(content)
                terms = normalizer.normalize(tokens)
                length = len(terms)

                # Processors podem terminar fora de ordem; extende-se
                # os arrays sob demanda para cobrir o internal_id.
                with self._doc_index_lock:
                    needed = internal_id + 1
                    if len(self._original_ids) < needed:
                        gap = needed - len(self._original_ids)
                        self._original_ids.extend([""] * gap)
                        self._lengths.extend([0] * gap)
                    self._original_ids[internal_id] = doc.id
                    self._lengths[internal_id] = length

                with self._pi_lock:
                    self._pi.add_document(internal_id, terms)

                with self._docs_processed_lock:
                    self._docs_processed += 1

    def _main_loop(
        self,
        reader_thread: threading.Thread,
        processors: list[threading.Thread],
    ):
        """Monitora memoria e dispara flushes ate todos terminarem."""
        last_log_time = time.time()
        last_logged_docs = 0

        while True:
            now = time.time()
            if now - last_log_time >= 10.0:
                with self._docs_processed_lock:
                    cur = self._docs_processed
                rate = (cur - last_logged_docs) / (now - last_log_time)
                used = self._memory.used_mb()
                print(
                    f"[spimi] {cur} docs, {rate:.0f} docs/s, "
                    f"mem={used:.0f}MB ({self._memory.used_ratio()*100:.0f}%), "
                    f"blocks={len(self._block_paths)}",
                    file=sys.stderr,
                )
                last_log_time = now
                last_logged_docs = cur

            if self._memory.should_flush():
                self._do_flush()

            all_done = (
                not reader_thread.is_alive()
                and all(not t.is_alive() for t in processors)
            )
            if all_done:
                break

            time.sleep(0.5)

    def _do_flush(self, final: bool = False):
        """
        Despeja o PartialIndex em disco como um novo block.

        Sinaliza flush_event para pausar processors, dumpa o PI,
        limpa e libera os processors.
        """
        self._flush_event.set()
        self._flush_done.clear()

        # Pequena pausa para processors liberarem o lock; nao precisa
        # de barreira rigorosa pois o PI lock serializa writes.
        time.sleep(0.1)

        with self._pi_lock:
            if self._pi.is_empty():
                self._flush_done.set()
                self._flush_event.clear()
                return

            block_path = os.path.join(
                self._blocks_dir, f"block_{self._next_block_idx:05d}.bin"
            )
            metadata = self._pi.dump_to_disk(block_path)
            self._next_block_idx += 1
            self._block_paths.append(block_path)

            tag = "final" if final else "flush"
            used = self._memory.used_mb()
            print(
                f"[spimi] {tag} -> {os.path.basename(block_path)}: "
                f"{metadata['num_terms']} termos, "
                f"{metadata['num_postings']} postings, "
                f"mem={used:.0f}MB",
                file=sys.stderr,
            )

            self._pi.clear()
            
            # GC explicito: sem isso o RSS - Resident Set Size (a quantidade de RAM) 
            # pode demorar a baixar e disparar novo flush prematuramente. Forçar o GC 
            # garante que a leitura de RSS reflete o estado real de uso depois do flush.
            gc.collect()
            self._memory.reset_peak()

        self._flush_done.set()
        self._flush_event.clear()
