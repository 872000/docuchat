"""Tests for document loading and chunking."""

import unittest

from docuchat.chunking import chunk_documents, chunk_text, load_documents

LONG_TEXT = " ".join(f"word{i}" for i in range(500))


class TestChunkText(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        chunks = chunk_text("Hello world. Short doc.", source="a.md")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source, "a.md")
        self.assertEqual(chunks[0].index, 0)
        self.assertIn("Hello world", chunks[0].text)

    def test_empty_text_has_no_chunks(self):
        self.assertEqual(chunk_text("", source="a.md"), [])
        self.assertEqual(chunk_text("   \n\n  ", source="a.md"), [])

    def test_long_text_splits_into_bounded_chunks(self):
        chunks = chunk_text(LONG_TEXT, source="big.md", max_words=120,
                            overlap_words=30)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text.split()), 120)
        # Chunks are sequential and indexed in order.
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))

    def test_overlap_shares_words_between_chunks(self):
        chunks = chunk_text(LONG_TEXT, source="big.md", max_words=100,
                            overlap_words=20)
        first_words = set(chunks[0].text.split())
        second_words = set(chunks[1].text.split())
        self.assertTrue(first_words & second_words)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            chunk_text("x", source="a.md", max_words=0)
        with self.assertRaises(ValueError):
            chunk_text("x", source="a.md", max_words=10, overlap_words=10)

    def test_every_word_appears_in_some_chunk(self):
        chunks = chunk_text(LONG_TEXT, source="big.md", max_words=50,
                            overlap_words=10)
        seen = set()
        for chunk in chunks:
            seen.update(chunk.text.split())
        self.assertEqual(seen, set(LONG_TEXT.split()))


class TestLoadAndChunkDocuments(unittest.TestCase):
    def test_chunk_documents_keeps_sources(self):
        docs = {"a.md": "Alpha beta gamma.", "b.md": "Delta epsilon zeta."}
        chunks = chunk_documents(docs)
        self.assertEqual({c.source for c in chunks}, {"a.md", "b.md"})

    def test_load_documents_reads_txt_and_md(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "one.md").write_text("# Title\n\nSome text here.")
            (folder / "two.txt").write_text("Plain text doc.")
            (folder / "skip.pdf").write_text("not a supported file")
            docs = load_documents(folder)
            self.assertEqual(set(docs), {"one.md", "two.txt"})
            self.assertIn("Some text here", docs["one.md"])

    def test_load_documents_missing_folder_is_empty(self):
        self.assertEqual(load_documents("/definitely/not/a/folder"), {})

    def test_load_documents_skips_unreadable_file(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "good.md").write_text("Readable doc.")
            (folder / "bad.md").write_text("Unreadable doc.")
            real_read = Path.read_text

            def flaky_read(self, *args, **kwargs):
                if self.name == "bad.md":
                    raise OSError("permission denied")
                return real_read(self, *args, **kwargs)

            with mock.patch.object(Path, "read_text", flaky_read):
                docs = load_documents(folder)
            self.assertEqual(docs, {"good.md": "Readable doc."})


if __name__ == "__main__":
    unittest.main()
