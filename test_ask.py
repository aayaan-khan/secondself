import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ask


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(ask.cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(ask.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector(self):
        self.assertEqual(ask.cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)

    def test_mismatched_length(self):
        self.assertEqual(ask.cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]), 0.0)


class TestParseNoteFile(unittest.TestCase):
    def test_with_frontmatter(self):
        text = (
            "---\n"
            "id: ec8a7dbe-10e0-4ade-bcc5-361976b5c399\n"
            "category: Projects\n"
            "tags: [hiking, weekend, outdoor]\n"
            "summary: \"Try new hiking trail\"\n"
            "---\n"
            "\n"
            "Idea for weekend: try that new hiking trail near the reservoir, check weather first.\n"
        )
        with patch.object(Path, "read_text", return_value=text):
            result = ask.parse_note_file(Path("wiki/Projects/ec8a7dbe.md"))
        self.assertEqual(result["id"], "ec8a7dbe-10e0-4ade-bcc5-361976b5c399")
        self.assertEqual(result["category"], "Projects")
        self.assertIn("hiking trail", result["body"])
        self.assertIn("hiking", result["tags"])

    def test_without_frontmatter_returns_none(self):
        text = "Just a plain note without frontmatter."
        with patch.object(Path, "read_text", return_value=text):
            result = ask.parse_note_file(Path("wiki/plain.md"))
        self.assertIsNone(result)

    def test_read_error_returns_none(self):
        with patch.object(Path, "read_text", side_effect=OSError("boom")):
            result = ask.parse_note_file(Path("wiki/missing.md"))
        self.assertIsNone(result)


class TestLoadEmbeddingsCache(unittest.TestCase):
    def test_load_existing(self):
        data = {"note-1": {"hash": "abc", "embedding": [0.1, 0.2, 0.3]}}
        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(data))):
                result = ask.load_embeddings_cache(Path("embeddings.json"))
        self.assertEqual(result, data)

    def test_load_missing_returns_empty_dict(self):
        with patch.object(Path, "exists", return_value=False):
            result = ask.load_embeddings_cache(Path("embeddings.json"))
        self.assertEqual(result, {})

    def test_load_corrupt_json_returns_empty_dict(self):
        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", unittest.mock.mock_open(read_data="not valid json")):
                result = ask.load_embeddings_cache(Path("embeddings.json"))
        self.assertEqual(result, {})


class TestRetrieve(unittest.TestCase):
    @patch("ask.find_note_by_id")
    @patch("ask.embed_text")
    @patch("ask.load_embeddings_cache")
    def test_top_k_ordered_by_score(self, mock_load, mock_embed, mock_find):
        mock_load.return_value = {
            "note-a": {"embedding": [1.0, 0.0, 0.0]},
            "note-b": {"embedding": [0.0, 1.0, 0.0]},
            "note-c": {"embedding": [0.95, 0.05, 0.0]},
        }
        mock_embed.return_value = [1.0, 0.0, 0.0]
        mock_find.side_effect = lambda nid, wiki_dir=None: {
            "body": f"Content of {nid}", "summary": "", "category": "Resources"
        }

        results = ask.retrieve("hiking", k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "note-a")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=5)
        self.assertEqual(results[1]["id"], "note-c")

    @patch("ask.load_embeddings_cache")
    @patch("ask.embed_text")
    def test_empty_question_returns_empty_without_embedding(self, mock_embed, mock_load):
        self.assertEqual(ask.retrieve(""), [])
        self.assertEqual(ask.retrieve("   "), [])
        mock_embed.assert_not_called()

    @patch("ask.load_embeddings_cache")
    @patch("ask.embed_text")
    def test_no_embeddings_cache_returns_empty(self, mock_embed, mock_load):
        mock_load.return_value = {}
        mock_embed.return_value = [1.0, 0.0]
        self.assertEqual(ask.retrieve("anything"), [])

    @patch("ask.load_embeddings_cache")
    @patch("ask.embed_text")
    def test_below_similarity_cutoff_excluded(self, mock_embed, mock_load):
        mock_load.return_value = {"note-low": {"embedding": [0.0, 1.0, 0.0]}}
        mock_embed.return_value = [1.0, 0.0, 0.0]
        # cosine similarity is 0.0, below default 0.30 cutoff
        results = ask.retrieve("hiking", similarity_cutoff=0.30)
        self.assertEqual(results, [])

    @patch("ask.find_note_by_id")
    @patch("ask.embed_text")
    @patch("ask.load_embeddings_cache")
    def test_missing_note_file_skipped(self, mock_load, mock_embed, mock_find):
        mock_load.return_value = {"note-x": {"embedding": [1.0, 0.0]}}
        mock_embed.return_value = [1.0, 0.0]
        mock_find.return_value = None  # note in cache but file no longer exists
        results = ask.retrieve("test")
        self.assertEqual(results, [])

    @patch("ask.load_embeddings_cache")
    @patch("ask.embed_text")
    def test_malformed_cache_entry_skipped(self, mock_embed, mock_load):
        # entry with no "embedding" key and not a list — must not crash
        mock_load.return_value = {"note-bad": {"hash": "abc"}, "note-str": "not-a-vector"}
        mock_embed.return_value = [1.0, 0.0]
        results = ask.retrieve("anything")
        self.assertEqual(results, [])


class TestTruncateContext(unittest.TestCase):
    def test_under_limit_includes_all(self):
        notes = [{"id": "n1", "score": 0.9, "content": "Short content."}]
        ctx = ask.truncate_context(notes, max_chars=1000)
        self.assertIn("Short content.", ctx)
        self.assertIn("n1", ctx)

    def test_over_limit_truncates(self):
        notes = [
            {"id": "n1", "score": 0.9, "content": "A" * 5000},
            {"id": "n2", "score": 0.8, "content": "B" * 5000},
        ]
        ctx = ask.truncate_context(notes, max_chars=200)
        self.assertLess(len(ctx), 300)
        self.assertIn("n1", ctx)
        self.assertNotIn("n2", ctx)


class TestGenerate(unittest.TestCase):
    @patch("ask.Groq" if hasattr(ask, "Groq") else "groq.Groq")
    def test_successful_generation(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "  Synthesized answer  "
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        notes = [{"id": "n1", "score": 0.92, "content": "Hiking trail near reservoir."}]
        with patch("groq.Groq", mock_groq_cls):
            result = ask.generate("Where should I hike?", notes)
        self.assertEqual(result, "Synthesized answer")

    def test_no_notes_returns_fallback_without_calling_llm(self):
        """Critical: no context notes must short-circuit before any API call."""
        result = ask.generate("Anything?", [])
        self.assertEqual(result, "I don't have notes on this")

    @patch("groq.Groq")
    def test_api_error_returns_error_string(self, mock_groq_cls):
        mock_groq_cls.side_effect = Exception("Rate limit exceeded")
        notes = [{"id": "n1", "score": 0.9, "content": "Note"}]
        result = ask.generate("What?", notes)
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("Rate limit exceeded", result)


class TestAsk(unittest.TestCase):
    @patch("ask.generate")
    @patch("ask.retrieve")
    def test_success(self, mock_retrieve, mock_generate):
        mock_retrieve.return_value = [{"id": "n1", "score": 0.85, "content": "Content"}]
        mock_generate.return_value = "The answer is 42."

        result = ask.ask("What is the answer?")
        self.assertEqual(result["answer"], "The answer is 42.")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["id"], "n1")
        self.assertAlmostEqual(result["sources"][0]["score"], 0.85)

    @patch("ask.retrieve")
    def test_no_relevant_notes_returns_fallback_not_hallucination(self, mock_retrieve):
        """Critical test: below-cutoff questions must NOT reach the LLM at all."""
        mock_retrieve.return_value = []
        result = ask.ask("What is quantum chromodynamics?")
        self.assertEqual(result["answer"], "I don't have notes on this")
        self.assertEqual(result["sources"], [])

    def test_empty_question_rejected_before_retrieval(self):
        with patch("ask.retrieve") as mock_retrieve:
            result = ask.ask("")
            mock_retrieve.assert_not_called()
        self.assertIn("non-empty", result["answer"].lower())
        self.assertEqual(result["sources"], [])

        with patch("ask.retrieve") as mock_retrieve:
            result = ask.ask("   \n\t  ")
            mock_retrieve.assert_not_called()
        self.assertIn("non-empty", result["answer"].lower())

    @patch("ask.generate")
    @patch("ask.retrieve")
    def test_api_failure_surfaces_as_error_string(self, mock_retrieve, mock_generate):
        mock_retrieve.return_value = [{"id": "n1", "score": 0.9, "content": "Note"}]
        mock_generate.return_value = "Error: Unable to generate answer (Exception: boom)"
        result = ask.ask("Question?")
        self.assertTrue(result["answer"].startswith("Error:"))


if __name__ == "__main__":
    unittest.main(verbosity=2)