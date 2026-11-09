"""Command-line interface for DocuChat.

Usage:
    python -m docuchat ask "How do I get a refund?" [--docs docs]
    python -m docuchat serve [--docs docs] [--port 8000]
    python -m docuchat index [--docs docs]      # show index stats
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .answer import answer_question, build_index
from .chunking import load_documents


def _load_index(docs_dir: Path):
    documents = load_documents(docs_dir)
    if not documents:
        print(f"No .txt or .md documents found in {docs_dir}. "
              "Add some and try again.", file=sys.stderr)
        raise SystemExit(2)
    index, chunks = build_index(documents)
    return documents, index, chunks


def cmd_ask(args: argparse.Namespace) -> int:
    _documents, index, _chunks = _load_index(args.docs)
    answer = answer_question(index, args.question)
    print(answer.text)
    if answer.sources:
        print("\nSources:")
        for result in answer.sources:
            print(f"  - {result.chunk.source} "
                  f"(chunk {result.chunk.index + 1}, "
                  f"score {result.score:.2f})")
    return 0 if answer.found else 1


def cmd_index(args: argparse.Namespace) -> int:
    documents, _index, chunks = _load_index(args.docs)
    print(f"{len(documents)} document(s), {len(chunks)} chunk(s) indexed "
          f"from {args.docs}:")
    for name in sorted(documents):
        count = sum(1 for c in chunks if c.source == name)
        print(f"  - {name}: {count} chunk(s)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .web import serve

    serve(args.docs, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="docuchat",
        description="Ask questions over your own documents (RAG, offline).")
    parser.add_argument("--docs", type=Path, default=Path("docs"),
                        help="folder of .txt/.md documents (default: docs)")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer one question and print it")
    ask.add_argument("question", help="the question to answer")
    ask.set_defaults(func=cmd_ask)

    idx = sub.add_parser("index", help="show how many chunks were indexed")
    idx.set_defaults(func=cmd_index)

    serve_p = sub.add_parser("serve", help="start the web UI")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
