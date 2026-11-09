"""Tests for grounded-answer behaviour, including the not-found case."""

import unittest
from pathlib import Path

from docuchat.answer import NOT_FOUND_MESSAGE, answer_question, build_index
from docuchat.chunking import load_documents

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def make_index():
    index, _chunks = build_index(load_documents(DOCS_DIR))
    return index


class TestGroundedAnswers(unittest.TestCase):
    def test_refund_answer_contains_fact_and_citation(self):
        answer = answer_question(make_index(),
                                 "How long do I have to return a product?")
        self.assertTrue(answer.found)
        self.assertIn("30 days", answer.text)
        self.assertIn("[support-handbook.md chunk", answer.text)
        self.assertTrue(answer.sources)

    def test_watering_answer_is_grounded_in_tomato_guide(self):
        answer = answer_question(make_index(),
                                 "How often should I water tomato plants?")
        self.assertTrue(answer.found)
        self.assertIn("tomato-guide.md", answer.text)
        self.assertIn("water", answer.text.lower())

    def test_python_answer_cites_python_notes(self):
        answer = answer_question(make_index(),
                                 "How do I define a function in Python?")
        self.assertTrue(answer.found)
        self.assertIn("python-basics.md", answer.text)

    def test_off_topic_question_says_not_in_documents(self):
        answer = answer_question(make_index(),
                                 "What is the capital of France?")
        self.assertFalse(answer.found)
        self.assertEqual(answer.text, NOT_FOUND_MESSAGE)
        self.assertEqual(answer.sources, [])

    def test_gibberish_query_says_not_in_documents(self):
        answer = answer_question(make_index(), "zqxj wobble florp")
        self.assertFalse(answer.found)
        self.assertIn("couldn't find", answer.text)

    def test_empty_question_is_rejected(self):
        answer = answer_question(make_index(), "   ")
        self.assertFalse(answer.found)

    def test_answer_never_invents_unrelated_facts(self):
        # A question sharing no content terms with any document must not
        # produce a confident-sounding answer.
        answer = answer_question(make_index(),
                                 "How do I bake sourdough bread from scratch?")
        self.assertFalse(answer.found)

    def test_answer_uses_sentences_from_documents(self):
        import re

        index = make_index()
        answer = answer_question(index, "How long does a refund take?")
        self.assertTrue(answer.found)
        # Strip citations, then every answer sentence must appear (verbatim)
        # in some indexed chunk.
        clean = re.sub(r"\[[^\]]*\]", "", answer.text)
        all_chunk_text = " ".join(c.text for c in index.chunks)
        for sentence in re.split(r"(?<=[.!?])\s+", clean):
            sentence = sentence.strip()
            if sentence:
                first_words = " ".join(sentence.split()[:6])
                self.assertIn(first_words, all_chunk_text)


if __name__ == "__main__":
    unittest.main()
