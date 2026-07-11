"""Convert a Stanford GloVe text file to word2vec text format.

Example:
    python tools/convert_glove.py \
        glove/glove.840B.300d.txt \
        glove/glove.840B.300d.gensim.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gensim.scripts.glove2word2vec import glove2word2vec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a GloVe text file to word2vec/Gensim text format."
    )
    parser.add_argument("input", type=Path, help="Path to the original GloVe .txt file")
    parser.add_argument("output", type=Path, help="Path for the converted .txt file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"GloVe input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count, dimensions = glove2word2vec(str(args.input), str(args.output))
    print(f"Converted {count:,} vectors with {dimensions} dimensions")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
