"""
convert_corpus_weighted.py - Converte entities.jsonl para JsonCollection
do Pyserini com FIELD WEIGHTING via repeticao.

Estrategia: repete o conteudo de cada campo N vezes no campo "contents",
o que multiplica o term frequency (tf) e portanto o peso BM25 dos termos
daquele campo. Equivale a boost multifield sem precisar reindexar campos
separados.

Pesos comuns para entity search:
    title=3, text=1, keywords=2    (entidade-aware moderado)
    title=5, text=1, keywords=3    (entidade-aware agressivo)
    title=4, text=1, keywords=4    (titulo e keywords iguais)

Uso:
    python submissions/s3_field_weights/convert_corpus_weighted.py \\
        -i data/corpus/entities.jsonl \\
        -o data/corpus/v1/entities_weighted.jsonl \\
        --w-title 3 --w-text 1 --w-keywords 2
"""

import argparse
import json
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Converte entities.jsonl com pesos por campo (via repeticao).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True,
                        help="entities.jsonl original")
    parser.add_argument("-o", "--output", required=True,
                        help="entities_weighted.jsonl de saida")
    parser.add_argument("--w-title", type=int, default=3,
                        help="Peso do title (numero de repeticoes)")
    parser.add_argument("--w-text", type=int, default=1,
                        help="Peso do text (numero de repeticoes)")
    parser.add_argument("--w-keywords", type=int, default=2,
                        help="Peso das keywords (numero de repeticoes)")
    return parser.parse_args()


def repeat_field(s: str, n: int) -> str:
    """Repete a string n vezes separadas por espaco. n=0 retorna vazio."""
    if n <= 0 or not s:
        return ""
    return (s + " ") * n


def convert(input_path: str, output_path: str,
            w_title: int, w_text: int, w_keywords: int) -> None:
    t_start = time.perf_counter()
    n_docs = 0
    n_skipped = 0

    print(
        f"[convert] pesos: title={w_title}, text={w_text}, keywords={w_keywords}",
        file=sys.stderr,
    )

    # Cria diretorio de saida se nao existir
    import os
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[convert] linha {line_num} malformada: {e}", file=sys.stderr)
                n_skipped += 1
                continue

            doc_id = doc.get("id", "").strip()
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            keywords_list = doc.get("keywords", [])

            if not doc_id:
                n_skipped += 1
                continue

            if isinstance(keywords_list, list):
                keywords = " ".join(str(k).strip() for k in keywords_list if k)
            else:
                keywords = str(keywords_list).strip()

            # Aplica pesos via repeticao
            title_weighted = repeat_field(title, w_title)
            text_weighted = repeat_field(text, w_text)
            keywords_weighted = repeat_field(keywords, w_keywords)

            contents = (title_weighted + text_weighted + keywords_weighted).strip()

            out_doc = {"id": doc_id, "contents": contents}
            fout.write(json.dumps(out_doc, ensure_ascii=False) + "\n")
            n_docs += 1

            if n_docs % 500_000 == 0:
                elapsed = time.perf_counter() - t_start
                rate = n_docs / elapsed
                print(
                    f"[convert] {n_docs} docs em {elapsed:.1f}s ({rate:.0f} docs/s)",
                    file=sys.stderr,
                )

    elapsed = time.perf_counter() - t_start
    print(
        f"[convert] CONCLUIDO: {n_docs} docs em {output_path} "
        f"({elapsed:.1f}s, {n_skipped} pulados)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output,
            args.w_title, args.w_text, args.w_keywords)
