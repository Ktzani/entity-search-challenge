"""
convert_corpus.py - Converte entities.jsonl (formato PA2) para o formato
JsonCollection do Pyserini.

Formato de entrada:
    {"id": "0000001", "title": "...", "text": "...", "keywords": [...]}

Formato de saida (JsonCollection do Pyserini):
    {"id": "0000001", "contents": "title texto concatenado keywords ..."}

Decisao: concatena title + text + keywords (junta com espaco) em "contents".
Mesma estrategia do indexer do PA2, permitindo comparacao limpa entre
implementacoes (Lucene vs Python puro).

Uso:
    python submissions/s2_pyserini_bm25/convert_corpus.py \\
        -i data/corpus/entities.jsonl \\
        -o data/corpus/entities_pyserini.jsonl
"""

import argparse
import json
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte entities.jsonl para o formato JsonCollection do Pyserini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path do entities.jsonl original (PA2)",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path do entities_pyserini.jsonl de saida",
    )
    return parser.parse_args()


def convert(input_path: str, output_path: str) -> None:
    t_start = time.perf_counter()
    n_docs = 0
    n_skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"[convert] linha {line_num} malformada, pulando: {e}",
                    file=sys.stderr,
                )
                n_skipped += 1
                continue

            # Extrai campos com defaults seguros
            doc_id = doc.get("id", "").strip()
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            keywords_list = doc.get("keywords", [])

            if not doc_id:
                n_skipped += 1
                continue

            # Concatena keywords (lista) com espaco
            if isinstance(keywords_list, list):
                keywords = " ".join(str(k).strip() for k in keywords_list if k)
            else:
                keywords = str(keywords_list).strip()

            # Concatena todos os campos em "contents"
            contents = " ".join(part for part in (title, text, keywords) if part)

            # Escreve formato Pyserini JsonCollection
            out_doc = {"id": doc_id, "contents": contents}
            fout.write(json.dumps(out_doc, ensure_ascii=False) + "\n")
            n_docs += 1

            # Log de progresso a cada 500k docs
            if n_docs % 500_000 == 0:
                elapsed = time.perf_counter() - t_start
                rate = n_docs / elapsed
                print(
                    f"[convert] {n_docs} docs convertidos em {elapsed:.1f}s "
                    f"({rate:.0f} docs/s)",
                    file=sys.stderr,
                )

    elapsed = time.perf_counter() - t_start
    print(
        f"[convert] CONCLUIDO: {n_docs} docs escritos em {output_path} "
        f"({elapsed:.1f}s, {n_skipped} pulados)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)
