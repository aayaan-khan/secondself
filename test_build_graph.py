"""
test_build_graph.py — Unit tests for build_graph.py (Phase 4)

Covers all edge cases from edge-case.md §4:
  1. Orphan nodes (zero links) included in output
  2. Broken link references skipped with warning
  3. Malformed frontmatter — file skipped, build continues
  4. Correct deduplication of bidirectional edges (A↔B → 1 edge, not 2)
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import build_graph
from build_graph import parse_frontmatter, load_notes, build_graph as _build_graph


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def create_note(wiki_dir: Path, category: str, note_id: str, summary: str,
                 tags: list[str] | None = None, links: list[str] | None = None,
                 raw_content: str | None = None) -> Path:
    """Write a well-formed wiki note and return its Path."""
    cat_dir = wiki_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    path = cat_dir / f"{note_id}.md"
    tags_str = ", ".join(tags or ["test"])
    links_str = ", ".join(links or [])
    if raw_content is not None:
        path.write_text(raw_content, encoding="utf-8")
    else:
        path.write_text(
            f"---\n"
            f"id: {note_id}\n"
            f"category: {category}\n"
            f"tags: [{tags_str}]\n"
            f"summary: \"{summary}\"\n"
            f"source_raw: raw/{note_id}.json\n"
            f"created: 2026-01-01T00:00:00+00:00\n"
            f"links: [{links_str}]\n"
            f"---\n\n"
            f"Body text for {summary}.\n",
            encoding="utf-8",
        )
    return path


# ---------------------------------------------------------------------------
# parse_frontmatter unit tests
# ---------------------------------------------------------------------------

class TestParseFrontmatter(unittest.TestCase):

    def test_parses_valid_frontmatter(self):
        md = "---\nid: abc-123\ncategory: Resources\ntags: [ai, ml]\nsummary: \"A test note\"\nlinks: []\n---\n\nBody."
        fm = parse_frontmatter(md)
        self.assertIsNotNone(fm)
        self.assertEqual(fm["id"], "abc-123")
        self.assertEqual(fm["category"], "Resources")
        self.assertEqual(fm["tags"], ["ai", "ml"])
        self.assertEqual(fm["summary"], "A test note")
        self.assertEqual(fm["links"], [])

    def test_returns_none_for_missing_frontmatter(self):
        md = "No frontmatter here at all."
        self.assertIsNone(parse_frontmatter(md))

    def test_returns_none_for_empty_frontmatter_block(self):
        md = "---\n---\nBody."
        result = parse_frontmatter(md)
        # Either None or empty dict is acceptable — neither should crash
        self.assertTrue(result is None or result == {})

    def test_parses_links_list(self):
        md = "---\nid: x\ncategory: Projects\nlinks: [id-1, id-2, id-3]\nsummary: \"x\"\n---\n"
        fm = parse_frontmatter(md)
        self.assertEqual(fm["links"], ["id-1", "id-2", "id-3"])

    def test_parses_empty_links(self):
        md = "---\nid: x\ncategory: Projects\nlinks: []\nsummary: \"x\"\n---\n"
        fm = parse_frontmatter(md)
        self.assertEqual(fm["links"], [])


# ---------------------------------------------------------------------------
# load_notes tests
# ---------------------------------------------------------------------------

class TestLoadNotes(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loads_well_formed_notes(self):
        create_note(self.tmp, "Resources", "note-1", "Resource Note")
        create_note(self.tmp, "Projects", "note-2", "Project Note")
        notes, warnings = load_notes(self.tmp)
        ids = {n["id"] for n in notes}
        self.assertIn("note-1", ids)
        self.assertIn("note-2", ids)
        self.assertEqual(len(warnings), 0)

    def test_skips_malformed_frontmatter_with_warning(self):
        # Well-formed note
        create_note(self.tmp, "Resources", "good-1", "Good Note")
        # Malformed: no frontmatter delimiters
        bad_path = (self.tmp / "Resources" / "bad-1.md")
        bad_path.write_text("Just plain text, no frontmatter at all.", encoding="utf-8")
        notes, warnings = load_notes(self.tmp)
        ids = {n["id"] for n in notes}
        self.assertIn("good-1", ids)
        self.assertNotIn("bad-1", ids)
        self.assertTrue(any("Malformed" in w for w in warnings))

    def test_empty_wiki_dir_returns_empty(self):
        notes, warnings = load_notes(self.tmp)
        self.assertEqual(notes, [])
        self.assertEqual(warnings, [])

    def test_nonexistent_wiki_dir_returns_empty(self):
        notes, warnings = load_notes(self.tmp / "nonexistent")
        self.assertEqual(notes, [])
        self.assertEqual(warnings, [])

    def test_note_without_id_gets_warning(self):
        path = (self.tmp / "Resources")
        path.mkdir(parents=True, exist_ok=True)
        (path / "noid.md").write_text(
            "---\ncategory: Resources\nsummary: \"No ID\"\nlinks: []\n---\n\nBody.",
            encoding="utf-8"
        )
        notes, warnings = load_notes(self.tmp)
        self.assertEqual(notes, [])
        self.assertTrue(any("No 'id'" in w for w in warnings))


# ---------------------------------------------------------------------------
# build_graph edge case tests
# ---------------------------------------------------------------------------

class TestBuildGraph(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.wiki = self.tmp / "wiki"
        self.wiki.mkdir()
        self.out = self.tmp / "graph.json"
        # Patch module-level defaults
        self._orig_wiki = build_graph.WIKI_DIR
        self._orig_graph = build_graph.GRAPH_FILE
        build_graph.WIKI_DIR = self.wiki
        build_graph.GRAPH_FILE = self.out

    def tearDown(self):
        build_graph.WIKI_DIR = self._orig_wiki
        build_graph.GRAPH_FILE = self._orig_graph
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    # Edge case 1: Orphan nodes included
    # ------------------------------------------------------------------
    def test_orphan_nodes_included_in_output(self):
        create_note(self.wiki, "Resources", "orphan-1", "Orphan Note A")
        create_note(self.wiki, "Areas",     "orphan-2", "Orphan Note B")
        result = _build_graph(wiki_dir=self.wiki, out_path=self.out)
        node_ids = {n["id"] for n in result["nodes"]}
        self.assertIn("orphan-1", node_ids)
        self.assertIn("orphan-2", node_ids)
        self.assertEqual(len(result["edges"]), 0)

    # ------------------------------------------------------------------
    # Edge case 2: Broken link reference — skipped with warning
    # ------------------------------------------------------------------
    def test_broken_link_skipped_with_warning(self):
        create_note(self.wiki, "Resources", "real-note", "Real Note",
                    links=["ghost-id-that-does-not-exist"])
        result = _build_graph(wiki_dir=self.wiki, out_path=self.out)
        # Edge must NOT be added
        self.assertEqual(len(result["edges"]), 0)
        # Warning must be emitted
        self.assertTrue(any("ghost-id-that-does-not-exist" in w for w in result["warnings"]))
        # Node still present
        self.assertEqual(len(result["nodes"]), 1)

    # ------------------------------------------------------------------
    # Edge case 3: Malformed frontmatter — file skipped, build continues
    # ------------------------------------------------------------------
    def test_malformed_frontmatter_skipped_build_continues(self):
        create_note(self.wiki, "Resources", "valid-1", "Valid Note")
        # Malformed note
        (self.wiki / "Resources" / "malformed.md").write_text(
            "Not valid YAML frontmatter at all, no --- delimiters", encoding="utf-8"
        )
        result = _build_graph(wiki_dir=self.wiki, out_path=self.out)
        node_ids = {n["id"] for n in result["nodes"]}
        self.assertIn("valid-1", node_ids)
        self.assertEqual(len(result["nodes"]), 1)
        self.assertTrue(any("Malformed" in w for w in result["warnings"]))

    # ------------------------------------------------------------------
    # Edge case 4: Bidirectional links → deduplicated to one edge
    # ------------------------------------------------------------------
    def test_bidirectional_links_produce_single_edge(self):
        create_note(self.wiki, "Resources", "note-a", "Note A", links=["note-b"])
        create_note(self.wiki, "Resources", "note-b", "Note B", links=["note-a"])
        result = _build_graph(wiki_dir=self.wiki, out_path=self.out)
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["edges"]), 1)
        edge = result["edges"][0]
        self.assertIn(edge["source"], {"note-a", "note-b"})
        self.assertIn(edge["target"], {"note-a", "note-b"})
        self.assertNotEqual(edge["source"], edge["target"])

    # ------------------------------------------------------------------
    # Additional: Multiple valid edges are all preserved
    # ------------------------------------------------------------------
    def test_multiple_edges_preserved(self):
        create_note(self.wiki, "Resources", "a", "A", links=["b", "c"])
        create_note(self.wiki, "Resources", "b", "B", links=["a"])
        create_note(self.wiki, "Resources", "c", "C", links=["a"])
        result = _build_graph(wiki_dir=self.wiki, out_path=self.out)
        self.assertEqual(len(result["nodes"]), 3)
        self.assertEqual(len(result["edges"]), 2)  # a-b and a-c

    # ------------------------------------------------------------------
    # Output schema validation
    # ------------------------------------------------------------------
    def test_graph_json_schema(self):
        create_note(self.wiki, "Resources", "n1", "Note 1", tags=["ai", "ml"], links=["n2"])
        create_note(self.wiki, "Projects",  "n2", "Note 2", links=["n1"])
        result = _build_graph(wiki_dir=self.wiki, out_path=self.out)

        # Check written file matches in-memory result
        with open(self.out, encoding="utf-8") as f:
            on_disk = json.load(f)

        self.assertEqual(on_disk["nodes"], result["nodes"])
        self.assertEqual(on_disk["edges"], result["edges"])

        # Node schema
        for node in on_disk["nodes"]:
            self.assertIn("id", node)
            self.assertIn("label", node)
            self.assertIn("category", node)
            self.assertIn("tags", node)
            self.assertIsInstance(node["tags"], list)

        # Edge schema
        for edge in on_disk["edges"]:
            self.assertIn("source", edge)
            self.assertIn("target", edge)

    # ------------------------------------------------------------------
    # Self-loop safety
    # ------------------------------------------------------------------
    def test_self_loop_not_included(self):
        create_note(self.wiki, "Resources", "self-note", "Self-Referencing",
                    links=["self-note"])
        result = _build_graph(wiki_dir=self.wiki, out_path=self.out)
        self.assertEqual(len(result["edges"]), 0)

    # ------------------------------------------------------------------
    # Empty wiki
    # ------------------------------------------------------------------
    def test_empty_wiki_produces_empty_graph(self):
        result = _build_graph(wiki_dir=self.wiki, out_path=self.out)
        self.assertEqual(result["nodes"], [])
        self.assertEqual(result["edges"], [])
        self.assertTrue(self.out.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
