"""
build_graph.py — Phase 4 (Graph Data Model)

Walks wiki/, reads YAML frontmatter from every note, and exports graph.json
with the schema from architecture.md §2.4:

    {
        "nodes": [{ "id": "...", "label": "...", "category": "...", "tags": [...] }],
        "edges": [{ "source": "...", "target": "..." }]
    }

Edge cases handled (edge-case.md §4):
  - Orphan nodes (zero links) — included in output, never dropped.
  - Broken link references — skipped with a warning, build continues.
  - Malformed YAML frontmatter — file skipped with a warning, build continues.
  - Large graph — no performance mitigation needed at current scale (noted in docs).

CLI:
    python build_graph.py               # reads wiki/, writes graph.json
    python build_graph.py --wiki wiki   # override wiki dir
    python build_graph.py --out out.json # override output path
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Reconfigure stdout/stderr to UTF-8 so emoji/arrows don't crash on Windows cp1252
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Defaults (override via env or CLI flags)
WIKI_DIR = Path(os.environ.get("SECONDSELF_WIKI_DIR", Path(__file__).resolve().parent / "wiki"))
GRAPH_FILE = Path(os.environ.get("SECONDSELF_GRAPH_FILE", Path(__file__).resolve().parent / "graph.json"))

# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def _parse_yaml_value(value: str) -> Any:
    """
    Minimal YAML scalar / list parser (no external dependency required).
    Handles:
      - bare scalars       → str
      - quoted scalars     → str (strips quotes)
      - inline lists       → List[str]  e.g. [a, b, c]
    """
    v = value.strip()

    # Inline list: [a, b, c]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1]
        if not inner.strip():
            return []
        items = [i.strip().strip('"').strip("'") for i in inner.split(",")]
        return [i for i in items if i]

    # Quoted scalar
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]

    return v


def parse_frontmatter(md_content: str) -> Optional[dict]:
    """
    Extract and parse the YAML frontmatter block from a markdown string.
    Returns a dict or None if the block is missing / malformed.
    """
    match = _FRONTMATTER_RE.match(md_content.strip())
    if not match:
        return None

    raw_yaml = match.group(1)
    result: dict = {}

    for line in raw_yaml.splitlines():
        # Skip blank lines and YAML comments
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key:
            result[key] = _parse_yaml_value(value)

    return result if result else None


# ---------------------------------------------------------------------------
# Wiki walker
# ---------------------------------------------------------------------------

def load_notes(wiki_dir: Optional[Path] = None) -> tuple[list[dict], list[str]]:
    """
    Walk wiki/ and return (notes, warnings).

    Each note is:
        {
            "id": str,
            "label": str,   # summary field
            "category": str,
            "tags": List[str],
            "links": List[str],
            "file": Path,
        }
    warnings is a list of human-readable problem strings.
    """
    target = wiki_dir or WIKI_DIR
    notes: list[dict] = []
    warnings: list[str] = []

    if not target.exists():
        return notes, warnings

    for md_file in sorted(target.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception as exc:
            warnings.append(f"Could not read {md_file}: {exc}")
            continue

        fm = parse_frontmatter(content)
        if fm is None:
            warnings.append(f"Malformed / missing frontmatter in {md_file} — skipping.")
            continue

        note_id = fm.get("id", "").strip()
        if not note_id:
            warnings.append(f"No 'id' in frontmatter of {md_file} — skipping.")
            continue

        summary = fm.get("summary", "").strip()
        category = fm.get("category", "Resources").strip()
        tags = fm.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        links_raw = fm.get("links", [])
        if isinstance(links_raw, str):
            links_raw = [links_raw] if links_raw else []

        notes.append(
            {
                "id": note_id,
                "label": summary or note_id,
                "category": category,
                "tags": tags,
                "links": [lnk.strip() for lnk in links_raw if lnk.strip()],
                "file": md_file,
            }
        )

    return notes, warnings


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    wiki_dir: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> dict:
    """
    Build the graph dict from wiki notes and write graph.json.

    Returns:
        {
            "nodes": [...],
            "edges": [...],
            "warnings": [...],   # diagnostic info, not in schema but useful for callers
        }
    """
    target_wiki = wiki_dir or WIKI_DIR
    target_out = out_path or GRAPH_FILE

    notes, warnings = load_notes(target_wiki)

    # Build a lookup of valid ids for broken-link detection
    valid_ids: set[str] = {n["id"] for n in notes}

    # Build nodes (all notes, including orphans)
    nodes = [
        {
            "id": n["id"],
            "label": n["label"],
            "category": n["category"],
            "tags": n["tags"],
        }
        for n in notes
    ]

    # Build deduplicated edges.
    # Since links are bidirectional (A.links contains B AND B.links contains A),
    # we canonicalise each edge as (min(a,b), max(a,b)) and add to a set.
    seen_edges: set[tuple[str, str]] = set()
    edges: list[dict] = []

    for note in notes:
        for linked_id in note["links"]:
            if linked_id not in valid_ids:
                warnings.append(
                    f"Broken link in note '{note['id']}': references missing id '{linked_id}' — skipping edge."
                )
                continue

            if linked_id == note["id"]:
                # Self-loop — skip silently (shouldn't happen, but be safe)
                continue

            canonical = (min(note["id"], linked_id), max(note["id"], linked_id))
            if canonical not in seen_edges:
                seen_edges.add(canonical)
                edges.append({"source": canonical[0], "target": canonical[1]})

    graph: dict = {"nodes": nodes, "edges": edges}

    # Write output
    target_out.parent.mkdir(parents=True, exist_ok=True)
    with open(target_out, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    return {**graph, "warnings": warnings}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build graph.json from wiki/ frontmatter.")
    parser.add_argument("--wiki", type=Path, default=None, help="Path to wiki dir (default: wiki/)")
    parser.add_argument("--out", type=Path, default=None, help="Path for graph.json output (default: graph.json)")
    args = parser.parse_args()

    wiki_dir = args.wiki or WIKI_DIR
    out_path = args.out or GRAPH_FILE

    print(f"Building graph from: {wiki_dir}")
    result = build_graph(wiki_dir=wiki_dir, out_path=out_path)

    for w in result.get("warnings", []):
        print(f"  WARNING: {w}", file=sys.stderr)

    n_nodes = len(result["nodes"])
    n_edges = len(result["edges"])
    orphans = [nd for nd in result["nodes"] if not any(
        e["source"] == nd["id"] or e["target"] == nd["id"] for e in result["edges"]
    )]

    print(f"\n================ graph.json Build Summary ================")
    print(f"  Nodes  : {n_nodes}")
    print(f"  Edges  : {n_edges}")
    print(f"  Orphans: {len(orphans)}")
    print(f"  Output : {out_path}")
    print(f"==========================================================")

    if orphans:
        print("\n  Orphan nodes (no edges):")
        for nd in orphans:
            print(f"    [{nd['category']}] {nd['label']}")

    print(f"\nDone. graph.json written to {out_path}")


if __name__ == "__main__":
    main()
