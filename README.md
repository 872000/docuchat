# DocuChat

A retrieval-augmented chatbot that answers questions over **your own
documents** — no API keys, no paid services, no network calls. Everything
runs locally with the Python standard library.

How it works:

1. **Ingest** — plain-text and Markdown files from a folder are split into
   overlapping chunks.
2. **Retrieve** — a from-scratch BM25 index ranks the chunks most relevant
   to your question.
3. **Answer** — the answer is built from sentences that actually appear in
   the retrieved chunks, each cited with its source document and chunk
   number. If the answer is not in your documents, DocuChat says so
   plainly instead of inventing one.

## Quick start

Requires Python 3.10 or newer. No dependencies to install.

```bash
git clone https://github.com/872000/docuchat.git
cd docuchat

# Ask a question from the command line (uses the sample docs)
python -m docuchat ask "How do I get a refund?"

# See what got indexed
python -m docuchat index

# Start the web UI at http://127.0.0.1:8000
python -m docuchat serve
```

Example output:

```
You can return any unused product within 30 days of delivery for a full
refund. [support-handbook.md chunk 1]

Sources:
  - support-handbook.md (chunk 1, score 4.12)
```

And for a question the documents cannot answer:

```
$ python -m docuchat ask "What is the capital of France?"
I couldn't find an answer to that in your documents. Try rephrasing the
question or adding a document that covers it.
```

## Adding your own documents

1. Drop `.txt` or `.md` files into a folder (the default is `docs/`).
2. Point DocuChat at it with `--docs`:

```bash
python -m docuchat --docs ~/my-notes ask "What is the onboarding checklist?"
python -m docuchat --docs ~/my-notes serve --port 8080
```

Tips for good answers:

- One topic per file keeps citations clean, but long files are fine —
  they are chunked automatically.
- Use plain language in your documents; retrieval matches on words, so
  the same terms you would ask about should appear in the text.

## Web UI

`python -m docuchat serve` serves a single page at
`http://127.0.0.1:8000` with a question box, the answer, and a source
list. There is also a JSON API:

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How often should I water tomatoes?"}'
```

```json
{
  "question": "How often should I water tomatoes?",
  "answer": "Water deeply two to three times per week ... [tomato-guide.md chunk 1]",
  "found": true,
  "sources": [
    {"source": "tomato-guide.md", "chunk": 1, "score": 5.31, "text": "..."}
  ]
}
```

## How retrieval works

- **Chunking** (`docuchat/chunking.py`) — paragraphs are packed into
  chunks of up to 120 words with a 30-word overlap, so facts are never
  stranded on a chunk boundary.
- **Ranking** (`docuchat/retrieval.py`) — BM25 (k1=1.5, b=0.75) with
  stopword removal. Scores are pure term statistics over your documents.
- **Answering** (`docuchat/answer.py`) — the best sentences from the top
  chunks are quoted with citations. Zero matching terms means a
  "not in the documents" answer, never a guess.

## Project layout

```
docuchat/
  __init__.py      package version
  __main__.py      `python -m docuchat` entry point
  chunking.py      document loading + overlapping chunker
  retrieval.py     from-scratch BM25 index
  answer.py        grounded, cited answer generation
  cli.py           `ask` / `index` / `serve` commands
  web.py           stdlib http.server web UI + JSON API
docs/              sample documents (support handbook, tomato guide, Python notes)
tests/             unittest suite (chunking, retrieval, answers, CLI, web)
```

## Running the tests

```bash
python -m unittest discover -s tests -v
```

The suite covers chunking (bounds, overlap, indexing), retrieval ranking
(a query must retrieve the right document), grounded answers (including
the "not in the documents" case and a check that answer sentences come
verbatim from the documents), and the CLI and web paths end to end. It
runs fully offline.

## License

MIT — see [LICENSE](LICENSE).
