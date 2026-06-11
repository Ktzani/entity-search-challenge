"""
Constantes do processador de queries.
"""

TOP_K = 10

# BM25 classico (Robertson et al.): k1 controla saturacao do TF
# (1.2-2.0 tipicos); b controla normalizacao por tamanho do doc.
BM25_K1 = 1.2
BM25_B = 0.75

TEXT_ENCODING = "utf-8"
