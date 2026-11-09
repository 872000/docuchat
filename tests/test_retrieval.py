"""Tests for BM25 retrieval ranking."""

import unittest

from docuchat.answer import build_index
from docuchat.retrieval import BM25Index, tokenize

DOCS = {
    "support-handbook.md": (
        "You can return any unused product within 30 days of delivery for a "
        "full refund. Refunds are issued to the original payment method "
        "within 5 business days. Express shipping takes 1 to 2 business "
        "days and costs $12.95."
    ),
    "tomato-guide.md": (
        "Water tomato plants deeply two to three times per week, giving each "
        "plant about 2 to 3 litres per watering. Tomatoes need at least 6 to "
        "8 hours of direct sunlight per day. Blossom end rot is caused by "
        "uneven watering or low calcium."
    ),
    "python-basics.md": (
        "Python is dynamically typed, so you do not declare a variable's "
        "type. Define a function with the def keyword. A dictionary maps "
        "keys to values and is written with curly braces."
    ),
}


def make_index():
    index, _chunks = build_index(DOCS)
    return index


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_drops_stopwords(self):
        tokens = tokenize("What is the Refund Policy?")
        self.assertEqual(tokens, ["refund", "policy"])

    def test_numbers_kept(self):
        self.assertIn("30", tokenize("within 30 days"))


class TestBM25Ranking(unittest.TestCase):
    def test_refund_query_retrieves_handbook(self):
        results = make_index().search("How do I get a refund for a return?")
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.source, "support-handbook.md")

    def test_watering_query_retrieves_tomato_guide(self):
        results = make_index().search("How often should I water tomatoes?")
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.source, "tomato-guide.md")

    def test_dictionary_query_retrieves_python_notes(self):
        results = make_index().search("How do dictionaries map keys to values?")
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.source, "python-basics.md")

    def test_unrelated_query_returns_nothing(self):
        results = make_index().search("quantum entanglement warp drive")
        self.assertEqual(results, [])

    def test_empty_query_returns_nothing(self):
        self.assertEqual(make_index().search(""), [])
        self.assertEqual(make_index().search("   "), [])

    def test_scores_descend(self):
        results = make_index().search("tomato watering sunlight")
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(s > 0 for s in scores))

    def test_top_k_limits_results(self):
        results = make_index().search("tomato", top_k=1)
        self.assertEqual(len(results), 1)

    def test_empty_index_searches_nothing(self):
        self.assertEqual(BM25Index([]).search("anything"), [])


if __name__ == "__main__":
    unittest.main()
