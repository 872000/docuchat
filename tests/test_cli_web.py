"""Tests for the CLI and web UI paths (in-process, offline)."""

import json
import subprocess
import sys
import threading
import unittest
import urllib.request
from pathlib import Path

from docuchat.web import make_server

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"


def run_cli(*argv):
    return subprocess.run(
        [sys.executable, "-m", "docuchat", *argv],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )


class TestCLI(unittest.TestCase):
    def test_ask_returns_grounded_answer(self):
        result = run_cli("--docs", str(DOCS_DIR), "ask",
                         "How do I get a refund?")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("30 days", result.stdout)
        self.assertIn("support-handbook.md", result.stdout)

    def test_ask_off_topic_exits_nonzero_with_not_found(self):
        result = run_cli("--docs", str(DOCS_DIR), "ask",
                         "What is the capital of France?")
        self.assertEqual(result.returncode, 1)
        self.assertIn("couldn't find", result.stdout)

    def test_index_lists_documents_and_chunks(self):
        result = run_cli("--docs", str(DOCS_DIR), "index")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("3 document(s)", result.stdout)
        self.assertIn("support-handbook.md", result.stdout)
        self.assertIn("tomato-guide.md", result.stdout)
        self.assertIn("python-basics.md", result.stdout)

    def test_ask_missing_docs_folder_fails(self):
        result = run_cli("--docs", str(ROOT / "no-such-folder"), "ask", "hi")
        self.assertEqual(result.returncode, 2)


class TestWebUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(DOCS_DIR, port=0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def test_homepage_has_question_box(self):
        with urllib.request.urlopen(self._url("/"), timeout=10) as resp:
            body = resp.read().decode()
        self.assertEqual(resp.status, 200)
        self.assertIn("<form", body)
        self.assertIn('name="question"', body)
        self.assertIn("DocuChat", body)

    def test_form_ask_returns_grounded_answer(self):
        data = urllib.parse.urlencode(
            {"question": "How often should I water tomatoes?"}).encode()
        req = urllib.request.Request(self._url("/ask"), data=data)
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
        self.assertEqual(resp.status, 200)
        self.assertIn("tomato-guide.md", body)

    def test_json_api_returns_answer_and_sources(self):
        payload = json.dumps(
            {"question": "How do I define a function in Python?"}).encode()
        req = urllib.request.Request(
            self._url("/api/ask"), data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        self.assertTrue(data["found"])
        self.assertIn("python-basics.md", data["answer"])
        self.assertTrue(data["sources"])
        self.assertEqual(data["sources"][0]["source"], "python-basics.md")

    def test_json_api_not_found_case(self):
        payload = json.dumps(
            {"question": "What is the capital of France?"}).encode()
        req = urllib.request.Request(
            self._url("/api/ask"), data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        self.assertFalse(data["found"])
        self.assertIn("couldn't find", data["answer"])
        self.assertEqual(data["sources"], [])

    def test_unknown_path_is_404(self):
        try:
            urllib.request.urlopen(self._url("/nope"), timeout=10)
        except urllib.error.HTTPError as err:
            self.assertEqual(err.code, 404)
        else:
            self.fail("expected a 404")


if __name__ == "__main__":
    unittest.main()
