"""Web UI for DocuChat, built on the standard library only.

Serves a single page with a question box, an answer area and source
citations. ``GET /`` renders the page, ``POST /ask`` answers a question
from the HTML form, and ``POST /api/ask`` answers it as JSON.
"""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from .answer import answer_question
from .chunking import load_documents
from .retrieval import BM25Index
from .answer import build_index

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DocuChat — ask your documents</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto;
       padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
h1 {{ font-size: 1.6rem; }}
form {{ display: flex; gap: 0.5rem; margin: 1rem 0; }}
input[type=text] {{ flex: 1; padding: 0.6rem; font-size: 1rem; }}
button {{ padding: 0.6rem 1.2rem; font-size: 1rem; cursor: pointer; }}
.answer {{ background: #f4f6f8; border-left: 4px solid #2563eb;
           padding: 1rem; white-space: pre-wrap; }}
.not-found {{ border-left-color: #d97706; }}
.sources {{ margin-top: 1rem; font-size: 0.9rem; color: #444; }}
.sources li {{ margin-bottom: 0.25rem; }}
small {{ color: #666; }}
</style>
</head>
<body>
<h1>DocuChat</h1>
<p><small>Ask a question — answers come only from your documents, with
citations. {doc_count} document(s), {chunk_count} chunks indexed.</small></p>
<form method="post" action="/ask">
<input type="text" name="question" placeholder="Ask about your documents..."
       value="{question}" autofocus>
<button type="submit">Ask</button>
</form>
{answer_block}
</body>
</html>
"""


def render_page(question: str = "", answer_text: str = "",
                found: bool = True, sources: list | None = None,
                doc_count: int = 0, chunk_count: int = 0) -> str:
    """Render the main page, optionally with an answer block."""
    answer_block = ""
    if answer_text:
        cls = "answer" if found else "answer not-found"
        answer_block = (f'<div class="{cls}">'
                        f"{html.escape(answer_text)}</div>")
        if sources:
            items = "".join(
                f"<li>{html.escape(s.chunk.source)} — chunk "
                f"{s.chunk.index + 1} (score {s.score:.2f})</li>"
                for s in sources
            )
            answer_block += f'<div class="sources"><strong>Sources</strong><ul>{items}</ul></div>'
    return PAGE_TEMPLATE.format(
        question=html.escape(question, quote=True),
        answer_block=answer_block,
        doc_count=doc_count,
        chunk_count=chunk_count,
    )


class DocuChatHandler(BaseHTTPRequestHandler):
    """HTTP handler bound to a pre-built index via ``server.index``."""

    index: BM25Index
    doc_count: int = 0

    def log_message(self, *args):  # keep test/server output quiet
        pass

    def _send_html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if self.path in ("/", "/index.html"):
            self._send_html(render_page(doc_count=self.doc_count,
                                        chunk_count=len(self.index)))
        else:
            self._send_html("<h1>404 — not found</h1>", status=404)

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        if self.path == "/api/ask":
            try:
                question = json.loads(raw).get("question", "")
            except json.JSONDecodeError:
                question = ""
            answer = answer_question(self.index, question)
            self._send_json({
                "question": question,
                "answer": answer.text,
                "found": answer.found,
                "sources": [
                    {"source": s.chunk.source, "chunk": s.chunk.index + 1,
                     "score": round(s.score, 3), "text": s.chunk.text}
                    for s in answer.sources
                ],
            })
            return
        if self.path == "/ask":
            question = parse_qs(raw).get("question", [""])[0]
            answer = answer_question(self.index, question)
            self._send_html(render_page(
                question=question, answer_text=answer.text,
                found=answer.found, sources=answer.sources,
                doc_count=self.doc_count, chunk_count=len(self.index)))
            return
        self._send_html("<h1>404 — not found</h1>", status=404)


def make_server(docs_dir: str | Path, host: str = "127.0.0.1",
                port: int = 8000) -> ThreadingHTTPServer:
    """Build the index from *docs_dir* and return a ready HTTP server."""
    documents = load_documents(docs_dir)
    index, _chunks = build_index(documents)
    handler = type("BoundHandler", (DocuChatHandler,), {})
    handler.index = index
    handler.doc_count = len(documents)
    return ThreadingHTTPServer((host, port), handler)


def serve(docs_dir: str | Path, host: str = "127.0.0.1",
          port: int = 8000) -> None:
    """Index *docs_dir* and serve the web UI until interrupted."""
    server = make_server(docs_dir, host=host, port=port)
    print(f"DocuChat serving {server.server_address[0]}:{port} "
          f"({len(server.RequestHandlerClass.index)} chunks indexed).")
    print("Open http://127.0.0.1:8000 in your browser. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping DocuChat.")
