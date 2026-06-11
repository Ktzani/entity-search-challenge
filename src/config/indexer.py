"""
Constantes do processo de indexacao.
"""

# Threads do pool de processadores (parsing + normalizacao).
NUM_PROCESSOR_THREADS = 4

# Batches maiores reduzem overhead da queue, mas aumentam pico de memoria.
READER_BATCH_SIZE = 500

# Limita pico de memoria caso processors fiquem mais lentos que reader.
QUEUE_MAX_BATCHES = 8

# Fracao do orcamento consumida antes de disparar flush. Margem de
# seguranca cobre overheads (I/O buffers, GC, etc.).
MEMORY_FLUSH_THRESHOLD = 0.75

# Periodicidade da checagem de memoria (psutil tem custo).
MEMORY_CHECK_EVERY_DOCS = 500

BLOCKS_DIR_NAME = "blocks"

INVERTED_INDEX_FILENAME = "inverted.idx"
TERM_LEXICON_FILENAME = "lexicon.pkl"
DOCUMENT_INDEX_FILENAME = "doc_index.pkl"

TEXT_ENCODING = "utf-8"
