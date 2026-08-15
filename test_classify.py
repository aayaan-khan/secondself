import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import classify


class TestClassifyPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.test_dir) / "raw"
        self.wiki_dir = Path(self.test_dir) / "wiki"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.wiki_dir.mkdir(parents=True, exist_ok=True)

        os.environ["SECONDSELF_RAW_DIR"] = str(self.raw_dir)
        os.environ["SECONDSELF_WIKI_DIR"] = str(self.wiki_dir)
        self.orig_raw_dir = classify.RAW_DIR
        self.orig_wiki_dir = classify.WIKI_DIR
        classify.RAW_DIR = self.raw_dir
        classify.WIKI_DIR = self.wiki_dir

    def tearDown(self):
        classify.RAW_DIR = self.orig_raw_dir
        classify.WIKI_DIR = self.orig_wiki_dir
        if "SECONDSELF_RAW_DIR" in os.environ:
            del os.environ["SECONDSELF_RAW_DIR"]
        if "SECONDSELF_WIKI_DIR" in os.environ:
            del os.environ["SECONDSELF_WIKI_DIR"]
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_normalize_category(self):
        self.assertEqual(classify.normalize_category("Projects"), "Projects")
        self.assertEqual(classify.normalize_category("projects"), "Projects")
        self.assertEqual(classify.normalize_category("AREAS"), "Areas")
        self.assertEqual(classify.normalize_category("resources"), "Resources")
        self.assertEqual(classify.normalize_category("Archives"), "Archives")
        # Invalid categories fallback to Resources
        self.assertEqual(classify.normalize_category("Random"), "Resources")
        self.assertEqual(classify.normalize_category(""), "Resources")
        self.assertEqual(classify.normalize_category(None), "Resources")

    def test_clean_tags(self):
        # Clean hashtags and whitespace
        tags = classify.clean_tags(["#Python", "Machine Learning", "#deep_learning!", "AI"])
        self.assertEqual(tags, ["python", "machine-learning", "deep-learning", "ai"])

        # Comma-separated string
        tags_str = classify.clean_tags("tools, productivity, notes")
        self.assertEqual(tags_str, ["tools", "productivity", "notes"])

        # Empty fallback
        self.assertEqual(classify.clean_tags([]), ["unclassified"])
        self.assertEqual(classify.clean_tags(None), ["unclassified"])

        # Cap at 5 tags
        long_tags = classify.clean_tags(["t1", "t2", "t3", "t4", "t5", "t6", "t7"])
        self.assertEqual(len(long_tags), 5)

    def test_clean_summary(self):
        self.assertEqual(
            classify.clean_summary('"A simple summary with quotes."'),
            "A simple summary with quotes."
        )
        self.assertEqual(
            classify.clean_summary("First line\nSecond line"),
            "First line Second line"
        )
        self.assertEqual(
            classify.clean_summary("", default_text="First line of note\nSecond line"),
            "Summary: First line of note"
        )

    def test_parse_llm_response_valid_json(self):
        valid_json = json.dumps({
            "category": "Projects",
            "tags": ["secondself", "coding", "agent"],
            "summary": "Building SecondSelf personal AI second brain."
        })
        res = classify.parse_llm_response(valid_json)
        self.assertEqual(res["category"], "Projects")
        self.assertEqual(res["tags"], ["secondself", "coding", "agent"])
        self.assertEqual(res["summary"], "Building SecondSelf personal AI second brain.")

    def test_parse_llm_response_markdown_fenced(self):
        fenced = """```json
{
  "category": "Resources",
  "tags": ["python", "tutorial"],
  "summary": "Comprehensive Python programming guide."
}
```"""
        res = classify.parse_llm_response(fenced)
        self.assertEqual(res["category"], "Resources")
        self.assertEqual(res["tags"], ["python", "tutorial"])
        self.assertEqual(res["summary"], "Comprehensive Python programming guide.")

    def test_parse_llm_response_prose_wrapped(self):
        prose = """Here is the PARA classification for your note:
{
  "category": "Areas",
  "tags": ["health", "fitness", "running"],
  "summary": "Weekly marathon training log and routine."
}
Hope this helps organize your wiki!"""
        res = classify.parse_llm_response(prose)
        self.assertEqual(res["category"], "Areas")
        self.assertEqual(res["tags"], ["health", "fitness", "running"])
        self.assertEqual(res["summary"], "Weekly marathon training log and routine.")

    def test_parse_llm_response_broken_json_fallback(self):
        broken = "category: Archives\ntags: ['old', 'deprecated']\nsummary: Old archived project files"
        res = classify.parse_llm_response(broken, fallback_content="Old archived project files")
        self.assertEqual(res["category"], "Archives")
        self.assertIn("old", res["tags"])
        self.assertIn("Old archived project files", res["summary"])

    def test_format_wiki_markdown(self):
        md = classify.format_wiki_markdown(
            capture_id="test-uuid-1234",
            category="Resources",
            tags=["ai", "notes"],
            summary='Notes on "AI and LLMs"',
            source_raw_filename="20260815T090000Z_test-uuid-1234.json",
            created_timestamp="2026-08-15T09:00:00+00:00",
            content="Detailed discussion on local sentence embeddings."
        )

        self.assertIn("---", md)
        self.assertIn("id: test-uuid-1234", md)
        self.assertIn("category: Resources", md)
        self.assertIn("tags: [ai, notes]", md)
        self.assertIn('summary: "Notes on \\"AI and LLMs\\""', md)
        self.assertIn("source_raw: raw/20260815T090000Z_test-uuid-1234.json", md)
        self.assertIn("links: []", md)
        self.assertIn("Detailed discussion on local sentence embeddings.", md)

    def test_classify_raw_file_with_mock_client(self):
        # Create a raw JSON capture file
        raw_capture = {
            "id": "11111111-2222-3333-4444-555555555555",
            "timestamp": "2026-08-15T09:00:00+00:00",
            "type": "note",
            "content": "Sprint plan: Finish capture, classify, and link modules by Friday.",
            "source_path": None
        }
        raw_file = self.raw_dir / "20260815T090000Z_11111111-2222-3333-4444-555555555555.json"
        raw_file.write_text(json.dumps(raw_capture), encoding="utf-8")

        # Mock Groq client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "category": "Projects",
                "tags": ["sprint", "development", "deadline"],
                "summary": "Sprint plan for completing SecondSelf core modules by Friday."
            })))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Classify
        res = classify.classify_raw_file(raw_file, client=mock_client, wiki_dir=self.wiki_dir)
        self.assertEqual(res["status"], "classified")
        self.assertEqual(res["category"], "Projects")

        dest_md = self.wiki_dir / "Projects" / "11111111-2222-3333-4444-555555555555.md"
        self.assertTrue(dest_md.exists())

        content = dest_md.read_text(encoding="utf-8")
        self.assertIn("category: Projects", content)
        self.assertIn("tags: [sprint, development, deadline]", content)
        self.assertIn("Sprint plan: Finish capture", content)

    def test_idempotency_skips_already_classified(self):
        # Setup existing classified file
        capture_id = "already-classified-uuid"
        existing_md = self.wiki_dir / "Resources" / f"{capture_id}.md"
        existing_md.parent.mkdir(parents=True, exist_ok=True)
        existing_md.write_text("Existing note", encoding="utf-8")

        raw_capture = {
            "id": capture_id,
            "timestamp": "2026-08-15T09:00:00+00:00",
            "type": "note",
            "content": "Some resource note",
            "source_path": None
        }
        raw_file = self.raw_dir / f"{capture_id}.json"
        raw_file.write_text(json.dumps(raw_capture), encoding="utf-8")

        mock_client = MagicMock()

        # Call classify without force
        res = classify.classify_raw_file(raw_file, client=mock_client, force=False, wiki_dir=self.wiki_dir)
        self.assertEqual(res["status"], "skipped")
        self.assertEqual(res["reason"], "already_classified")
        # Verify LLM was NOT called
        mock_client.chat.completions.create.assert_not_called()

        # Call with force=True
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "category": "Resources",
                "tags": ["updated"],
                "summary": "Updated summary"
            })))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        res_force = classify.classify_raw_file(raw_file, client=mock_client, force=True, wiki_dir=self.wiki_dir)
        self.assertEqual(res_force["status"], "classified")
        mock_client.chat.completions.create.assert_called_once()

    def test_null_content_binary_file_handling(self):
        raw_capture = {
            "id": "binary-file-uuid",
            "timestamp": "2026-08-15T09:00:00+00:00",
            "type": "file",
            "content": None,
            "source_path": "/path/to/archive_2026.zip"
        }
        raw_file = self.raw_dir / "binary_test.json"
        raw_file.write_text(json.dumps(raw_capture), encoding="utf-8")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "category": "Archives",
                "tags": ["zip", "backup", "files"],
                "summary": "Backup zip archive of 2026 project assets."
            })))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        res = classify.classify_raw_file(raw_file, client=mock_client, wiki_dir=self.wiki_dir)
        self.assertEqual(res["status"], "classified")
        self.assertEqual(res["category"], "Archives")

        dest_md = self.wiki_dir / "Archives" / "binary-file-uuid.md"
        self.assertTrue(dest_md.exists())
        text = dest_md.read_text(encoding="utf-8")
        self.assertIn("category: Archives", text)
        self.assertIn("*(No textual content extracted)*", text)

    def test_cli_help(self):
        python_bin = sys.executable
        cli_script = PROJECT_ROOT / "classify.py"

        result = subprocess.run(
            [python_bin, str(cli_script), "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(PROJECT_ROOT)
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("SecondSelf Auto-Classify", result.stdout)


if __name__ == "__main__":
    unittest.main()
