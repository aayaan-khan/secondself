import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import link


class TestLinkPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.wiki_dir = Path(self.test_dir) / "wiki"
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = Path(self.test_dir) / "embeddings.json"

        os.environ["SECONDSELF_WIKI_DIR"] = str(self.wiki_dir)
        os.environ["SECONDSELF_EMBEDDINGS_FILE"] = str(self.cache_file)
        self.orig_wiki_dir = link.WIKI_DIR
        self.orig_cache_file = link.CACHE_FILE
        link.WIKI_DIR = self.wiki_dir
        link.CACHE_FILE = self.cache_file

    def tearDown(self):
        link.WIKI_DIR = self.orig_wiki_dir
        link.CACHE_FILE = self.orig_cache_file
        if "SECONDSELF_WIKI_DIR" in os.environ:
            del os.environ["SECONDSELF_WIKI_DIR"]
        if "SECONDSELF_EMBEDDINGS_FILE" in os.environ:
            del os.environ["SECONDSELF_EMBEDDINGS_FILE"]
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_sample_note(self, note_id: str, category: str, summary: str, content: str, links=None) -> Path:
        cat_dir = self.wiki_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        note_file = cat_dir / f"{note_id}.md"
        links_list = links or []
        links_str = ", ".join(links_list)

        doc = f"""---
id: {note_id}
category: {category}
tags: [test, sample]
summary: "{summary}"
source_raw: raw/{note_id}.json
created: 2026-08-15T10:00:00+00:00
links: [{links_str}]
---

{content}
"""
        note_file.write_text(doc, encoding="utf-8")
        return note_file

    def test_cosine_similarity(self):
        # Identical
        self.assertAlmostEqual(link.cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 1.0)
        # Orthogonal
        self.assertAlmostEqual(link.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        # Opposing
        self.assertAlmostEqual(link.cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)
        # Zero vector handling
        self.assertEqual(link.cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)
        self.assertEqual(link.cosine_similarity([], []), 0.0)
        self.assertEqual(link.cosine_similarity([1.0], [1.0, 2.0]), 0.0)

    def test_content_hash(self):
        h1 = link.compute_content_hash("Hello World")
        h2 = link.compute_content_hash("Hello World")
        h3 = link.compute_content_hash("Hello World modified")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

    def test_parse_and_update_note_links(self):
        note_path = self.create_sample_note(
            note_id="test-1",
            category="Resources",
            summary="Test Note",
            content="Some body text."
        )

        # Parse
        parsed = link.parse_note_file(note_path)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["id"], "test-1")
        self.assertEqual(parsed["summary"], "Test Note")
        self.assertEqual(parsed["links"], [])
        self.assertEqual(parsed["body"], "Some body text.")

        # Update links
        success = link.update_note_links(note_path, ["linked-id-2", "linked-id-3"])
        self.assertTrue(success)

        # Re-parse to verify update
        reparsed = link.parse_note_file(note_path)
        self.assertEqual(reparsed["links"], ["linked-id-2", "linked-id-3"])
        self.assertEqual(reparsed["summary"], "Test Note")
        self.assertEqual(reparsed["body"], "Some body text.")

    def test_empty_wiki_and_single_note_edge_case(self):
        # 0 notes
        res0 = link.auto_link_wiki(wiki_dir=self.wiki_dir, cache_path=self.cache_file)
        self.assertEqual(res0["total_notes"], 0)
        self.assertEqual(res0["total_edges"], 0)

        # 1 note
        self.create_sample_note("single-1", "Projects", "Single Note", "Sole content in wiki.")
        res1 = link.auto_link_wiki(wiki_dir=self.wiki_dir, cache_path=self.cache_file)
        self.assertEqual(res1["total_notes"], 1)
        self.assertEqual(res1["total_edges"], 0)
        self.assertEqual(len(res1["orphans"]), 1)

    def test_compute_links_bidirectional_and_no_self_link(self):
        notes = [
            {"id": "A", "category": "Resources", "summary": "AI Note", "body": "embeddings"},
            {"id": "B", "category": "Resources", "summary": "ML Note", "body": "embeddings vectors"},
            {"id": "C", "category": "Areas", "summary": "Gardening", "body": "planting tomatoes"}
        ]
        # Synthetic mock embeddings: A and B are very similar, C is orthogonal
        embeddings_map = {
            "A": [0.9, 0.1, 0.0],
            "B": [0.85, 0.15, 0.0],
            "C": [0.0, 0.0, 1.0]
        }

        res = link.compute_links(notes, embeddings_map, threshold=0.70)
        links_by_id = res["links_by_id"]

        # A links to B and B links to A
        self.assertIn("B", links_by_id["A"])
        self.assertIn("A", links_by_id["B"])
        # No self links
        self.assertNotIn("A", links_by_id["A"])
        self.assertNotIn("B", links_by_id["B"])
        self.assertNotIn("C", links_by_id["C"])
        # C is orphan
        self.assertEqual(links_by_id["C"], [])
        self.assertEqual(res["total_edges"], 1)

    def test_threshold_filtering(self):
        notes = [
            {"id": "A", "summary": "A", "body": "A"},
            {"id": "B", "summary": "B", "body": "B"}
        ]
        embeddings_map = {
            "A": [1.0, 0.0],
            "B": [0.6, 0.8]  # dot product = 0.60
        }

        # Threshold 0.50 -> Linked
        res_low = link.compute_links(notes, embeddings_map, threshold=0.50)
        self.assertEqual(res_low["total_edges"], 1)

        # Threshold 0.70 -> Not linked
        res_high = link.compute_links(notes, embeddings_map, threshold=0.70)
        self.assertEqual(res_high["total_edges"], 0)

    def test_embeddings_cache_invalidation_on_content_change(self):
        note_path = self.create_sample_note("note-cache-1", "Resources", "Initial Title", "Initial content")
        notes = link.load_all_wiki_notes(self.wiki_dir)

        # Sync embeddings (computes and caches)
        emb_map1, cache1 = link.sync_embeddings(notes, cache_path=self.cache_file)
        self.assertIn("note-cache-1", cache1)
        initial_hash = cache1["note-cache-1"]["hash"]

        # Second sync without edit should reuse cache
        emb_map2, cache2 = link.sync_embeddings(notes, cache_path=self.cache_file)
        self.assertEqual(cache2["note-cache-1"]["hash"], initial_hash)

        # Modify note content
        note_path.write_text("""---
id: note-cache-1
category: Resources
tags: [test]
summary: "Modified Title"
source_raw: raw/note-cache-1.json
created: 2026-08-15T10:00:00+00:00
links: []
---

Completely modified text about vector databases and retrieval.
""", encoding="utf-8")

        notes_edited = link.load_all_wiki_notes(self.wiki_dir)
        emb_map3, cache3 = link.sync_embeddings(notes_edited, cache_path=self.cache_file)
        new_hash = cache3["note-cache-1"]["hash"]
        self.assertNotEqual(initial_hash, new_hash)

    def test_cli_report(self):
        self.create_sample_note("cli-1", "Resources", "Python Guide", "Python syntax and libraries.")
        self.create_sample_note("cli-2", "Resources", "Python Tips", "Python data science techniques.")

        python_bin = sys.executable
        cli_script = PROJECT_ROOT / "link.py"

        result = subprocess.run(
            [python_bin, str(cli_script), "--report", "--threshold", "0.50"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(PROJECT_ROOT)
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("SecondSelf Auto-Link Report", result.stdout)
        self.assertIn("Similarity Threshold:  0.50", result.stdout)


if __name__ == "__main__":
    unittest.main()
