"""
Garante que os dados necessarios do NLTK (punkt, stopwords) estao
disponiveis, baixando-os se nao estiverem. Idempotente.
"""

import sys
import nltk


def _ensure_resource(resource_path: str, resource_name: str):
    """
    Localiza um recurso NLTK; se nao existir, baixa silenciosamente.

    Parametros:
        resource_path: caminho NLTK do recurso (e.g., 'tokenizers/punkt')
        resource_name: nome do pacote a baixar (e.g., 'punkt')
    """
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"[nltk_setup] baixando {resource_name}...", file=sys.stderr)
        nltk.download(resource_name, quiet=True)


def ensure_nltk_data():
    """Garante punkt (tokenizer) e stopwords. Idempotente."""
    _ensure_resource("tokenizers/punkt", "punkt")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("[nltk_setup] baixando punkt_tab...", file=sys.stderr)
        nltk.download("punkt_tab", quiet=True)
    _ensure_resource("corpora/stopwords", "stopwords")
