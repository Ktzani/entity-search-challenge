"""
Constantes de pre-processamento usadas tanto pelo indexer quanto pelo
processor. CRITICO: indexer e processor DEVEM usar os mesmos valores
aqui, senao queries nao casam com documentos.
"""

# Wikipedia inglesa: usado por SnowballStemmer e pela lista de stopwords.
STEMMER_LANGUAGE = "english"

# Filtra ruido (letras soltas, restos de tokenizacao).
MIN_TOKEN_LENGTH = 2

# Descarta tokens absurdamente longos (hashes, lixo de HTML).
MAX_TOKEN_LENGTH = 40

# Indexar numeros polui o lexicon sem beneficio para queries em entidades.
DROP_PURE_NUMERIC = True
