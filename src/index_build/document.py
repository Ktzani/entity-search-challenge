"""
Representa um documento do corpus, ja parseado e pronto para indexacao.

Combina-se title + text + keywords em um unico campo textual durante
a indexacao (sem boost de campo). Os 4 campos originais sao mantidos
no dataclass para facilitar debug e permitir mudar a estrategia depois
sem mexer no CorpusReader.
"""

from dataclasses import dataclass, field


@dataclass
class Document:
    """Documento do corpus apos parsing do JSON."""

    id: str
    title: str
    text: str
    keywords: list[str] = field(default_factory=list)

    def full_content(self) -> str:
        """
        Concatena title + text + keywords numa unica string para
        tokenizacao. Espaco entre campos evita que tokens de campos
        diferentes sejam grudados pelo tokenizer.
        """
        parts = []
        if self.title:
            parts.append(self.title)
        if self.text:
            parts.append(self.text)
        if self.keywords:
            parts.extend(self.keywords)
        return " ".join(parts)
