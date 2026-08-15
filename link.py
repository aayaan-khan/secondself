#!/usr/bin/env python3
"""SecondSelf - Auto-Link Pipeline (Week 2.2 / The Librarian, part 2)

Finds related notes using sentence embeddings (all-MiniLM-L6-v2) and cosine similarity,
caches embeddings to embeddings.json with SHA-256 content invalidation, and inserts
bidirectional links into note YAML frontmatter above a similarity threshold.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Reconfigure stdout/stderr on Windows for UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configuration
DEFAULT_THRESHOLD = 0.30
DEFAULT_MODEL_NAME = os.environ.get("SECONDSELF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
WIKI_DIR = Path(os.environ.get("SECONDSELF_WIKI_DIR", Path(__file__).resolve().parent / "wiki"))
CACHE_FILE = Path(os.environ.get("SECONDSELF_EMBEDDINGS_FILE", Path(__file__).resolve().parent / "embeddings.json"))

_MODEL_INSTANCE = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """Lazy-load sentence-transformers model."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL_INSTANCE = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}")
    return _MODEL_INSTANCE


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of text for cache invalidation."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_embeddings_cache(cache_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load cached embeddings from JSON file."""
    target_path = cache_path or CACHE_FILE
    if not target_path.exists():
        return {}
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Warning: Failed to load embeddings cache ({e}). Starting fresh.", file=sys.stderr)
        return {}


def save_embeddings_cache(cache: Dict[str, Any], cache_path: Optional[Path] = None) -> None:
    """Save embeddings cache to JSON file."""
    target_path = cache_path or CACHE_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def parse_note_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Parse YAML frontmatter and content body from a markdown note."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}", file=sys.stderr)
        return None

    # Match YAML frontmatter between opening and closing ---
    match = re.match(r"^---\s*\n([\s\S]*?)\n---\s*\n?([\s\S]*)$", content)
    if not match:
        return None

    frontmatter_raw = match.group(1)
    body = match.group(2).strip()

    # Extract frontmatter fields
    note_id_m = re.search(r"^id:\s*(.+)$", frontmatter_raw, re.MULTILINE)
    cat_m = re.search(r"^category:\s*(.+)$", frontmatter_raw, re.MULTILINE)
    tags_m = re.search(r"^tags:\s*\[(.*?)\]", frontmatter_raw, re.MULTILINE)
    sum_m = re.search(r"^summary:\s*[\"']?(.*?)[\"']?$", frontmatter_raw, re.MULTILINE)
    created_m = re.search(r"^created:\s*(.+)$", frontmatter_raw, re.MULTILINE)
    source_m = re.search(r"^source_raw:\s*(.+)$", frontmatter_raw, re.MULTILINE)
    links_m = re.search(r"^links:\s*\[(.*?)\]", frontmatter_raw, re.MULTILINE)

    if not note_id_m:
        return None

    note_id = note_id_m.group(1).strip()
    category = cat_m.group(1).strip() if cat_m else "Resources"
    summary = sum_m.group(1).strip() if sum_m else ""
    created = created_m.group(1).strip() if created_m else ""
    source_raw = source_m.group(1).strip() if source_m else ""

    tags = []
    if tags_m and tags_m.group(1).strip():
        tags = [t.strip().strip("\"'") for t in tags_m.group(1).split(",") if t.strip()]

    links = []
    if links_m and links_m.group(1).strip():
        links = [l.strip().strip("\"'") for l in links_m.group(1).split(",") if l.strip()]

    return {
        "id": note_id,
        "category": category,
        "tags": tags,
        "summary": summary,
        "source_raw": source_raw,
        "created": created,
        "links": links,
        "body": body,
        "raw_frontmatter": frontmatter_raw,
        "path": file_path
    }


def load_all_wiki_notes(wiki_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Scan wiki/ and load all markdown notes."""
    target_dir = wiki_dir or WIKI_DIR
    if not target_dir.exists():
        return []

    notes = []
    for md_path in target_dir.glob("**/*.md"):
        note = parse_note_file(md_path)
        if note:
            notes.append(note)
    return notes


def update_note_links(file_path: Path, new_links: List[str]) -> bool:
    """Update only the links: [...] field in note frontmatter in-place."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"⚠️  Error reading {file_path} for link update: {e}", file=sys.stderr)
        return False

    # Format sorted, unique links
    sorted_links = sorted(list(set(new_links)))
    links_formatted = ", ".join(sorted_links)
    new_links_line = f"links: [{links_formatted}]"

    # Replace existing links: [...] or append to frontmatter
    if re.search(r"^links:\s*\[.*?\]", content, re.MULTILINE):
        updated_content = re.sub(
            r"^links:\s*\[.*?\]",
            new_links_line,
            content,
            flags=re.MULTILINE
        )
    else:
        # If links line was missing in frontmatter, insert before closing ---
        updated_content = re.sub(
            r"\n---\s*\n",
            f"\n{new_links_line}\n---\n",
            content,
            count=1
        )

    try:
        file_path.write_text(updated_content, encoding="utf-8")
        return True
    except Exception as e:
        print(f"⚠️  Error writing updated links to {file_path}: {e}", file=sys.stderr)
        return False


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2) or not vec1:
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(vec1, vec2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def get_note_embed_text(note: Dict[str, Any]) -> str:
    """Construct text representation of a note for embedding."""
    summary = note.get("summary", "").strip()
    body = note.get("body", "").strip()
    if summary and body:
        return f"{summary}\n\n{body}"
    elif summary:
        return summary
    elif body:
        return body
    return note.get("id", "note")


def sync_embeddings(
    notes: List[Dict[str, Any]],
    cache_path: Optional[Path] = None,
    force_recompute: bool = False,
    model_name: str = DEFAULT_MODEL_NAME
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """Compute and cache embeddings for all notes, invalidating on content hash mismatch."""
    cache = {} if force_recompute else load_embeddings_cache(cache_path)
    updated_cache = {}
    embeddings_map: Dict[str, List[float]] = {}
    notes_to_embed = []
    embed_texts = []

    for note in notes:
        note_id = note["id"]
        embed_text = get_note_embed_text(note)
        content_hash = compute_content_hash(embed_text)

        # Check if cache is valid (same ID and matching content hash)
        if (
            not force_recompute
            and note_id in cache
            and cache[note_id].get("hash") == content_hash
            and "embedding" in cache[note_id]
        ):
            vector = cache[note_id]["embedding"]
            embeddings_map[note_id] = vector
            updated_cache[note_id] = cache[note_id]
        else:
            notes_to_embed.append((note, content_hash))
            embed_texts.append(embed_text)

    # Batch compute new embeddings if needed
    if notes_to_embed:
        model = get_embedding_model(model_name)
        new_vectors = model.encode(embed_texts, show_progress_bar=False, normalize_embeddings=True)

        for (note, content_hash), vector in zip(notes_to_embed, new_vectors):
            vec_list = [float(x) for x in vector]
            note_id = note["id"]
            embeddings_map[note_id] = vec_list
            updated_cache[note_id] = {
                "hash": content_hash,
                "embedding": vec_list,
                "summary": note.get("summary", ""),
                "category": note.get("category", ""),
                "path": str(note["path"].relative_to(WIKI_DIR) if note["path"].is_relative_to(WIKI_DIR) else note["path"].name)
            }

    # Save updated cache to disk
    save_embeddings_cache(updated_cache, cache_path)
    return embeddings_map, updated_cache


def compute_links(
    notes: List[Dict[str, Any]],
    embeddings_map: Dict[str, List[float]],
    threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, Any]:
    """Compute pairwise cosine similarities and determine bidirectional links."""
    links_by_id: Dict[str, Set[str]] = {n["id"]: set() for n in notes}
    pairwise_scores: List[Dict[str, Any]] = []

    n_notes = len(notes)
    for i in range(n_notes):
        for j in range(i + 1, n_notes):
            id_a = notes[i]["id"]
            id_b = notes[j]["id"]

            vec_a = embeddings_map.get(id_a)
            vec_b = embeddings_map.get(id_b)

            if not vec_a or not vec_b:
                continue

            sim = cosine_similarity(vec_a, vec_b)
            is_linked = sim >= threshold

            if is_linked:
                links_by_id[id_a].add(id_b)
                links_by_id[id_b].add(id_a)

            pairwise_scores.append({
                "note_a": notes[i],
                "note_b": notes[j],
                "similarity": sim,
                "is_linked": is_linked
            })

    # Sort pairs by similarity descending
    pairwise_scores.sort(key=lambda x: x["similarity"], reverse=True)

    # Check for near-duplicates (similarity > 0.95)
    near_duplicates = [p for p in pairwise_scores if p["similarity"] >= 0.95]

    # Identify orphans (0 links)
    orphans = [n for n in notes if len(links_by_id[n["id"]]) == 0]

    # Calculate total unique bidirectional edges
    total_edges = sum(len(links) for links in links_by_id.values()) // 2

    return {
        "links_by_id": {k: sorted(list(v)) for k, v in links_by_id.items()},
        "pairwise_scores": pairwise_scores,
        "near_duplicates": near_duplicates,
        "orphans": orphans,
        "total_edges": total_edges,
        "threshold": threshold
    }


def auto_link_wiki(
    wiki_dir: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    threshold: float = DEFAULT_THRESHOLD,
    dry_run: bool = False,
    force_recompute: bool = False
) -> Dict[str, Any]:
    """Execute complete auto-link pipeline on wiki/."""
    target_wiki_dir = wiki_dir or WIKI_DIR
    target_cache_path = cache_path or CACHE_FILE

    notes = load_all_wiki_notes(target_wiki_dir)
    if not notes:
        print("ℹ️  No notes found in wiki/.")
        return {
            "total_notes": 0,
            "total_edges": 0,
            "threshold": threshold,
            "orphans": [],
            "pairwise_scores": []
        }

    # Sync and compute embeddings
    embeddings_map, _ = sync_embeddings(
        notes,
        cache_path=target_cache_path,
        force_recompute=force_recompute
    )

    # Compute similarity and links
    results = compute_links(notes, embeddings_map, threshold=threshold)
    results["total_notes"] = len(notes)

    # Update frontmatter in markdown files if not dry_run
    if not dry_run:
        updated_count = 0
        for note in notes:
            new_links = results["links_by_id"].get(note["id"], [])
            if new_links != note.get("links", []):
                if update_note_links(note["path"], new_links):
                    updated_count += 1
        results["files_updated"] = updated_count

    return results


def print_report(results: Dict[str, Any]) -> None:
    """Print readable report of link results and similarity statistics."""
    total_notes = results.get("total_notes", 0)
    total_edges = results.get("total_edges", 0)
    threshold = results.get("threshold", DEFAULT_THRESHOLD)
    pairwise = results.get("pairwise_scores", [])
    orphans = results.get("orphans", [])
    near_dups = results.get("near_duplicates", [])

    print(f"\n================ SecondSelf Auto-Link Report ================")
    print(f"Notes in Wiki:         {total_notes}")
    print(f"Similarity Threshold:  {threshold:.2f}")
    print(f"Total Linked Pairs:    {total_edges}")
    print(f"Orphan Notes (0 links): {len(orphans)}")
    print(f"Near Duplicates (≥0.95): {len(near_dups)}")
    print("=============================================================\n")

    if near_dups:
        print("⚠️  Near-Duplicate Notes Detected (similarity ≥ 0.95):")
        for dup in near_dups:
            a = dup["note_a"]
            b = dup["note_b"]
            print(f"   • [{dup['similarity']:.4f}] '{a['summary']}' <-> '{b['summary']}'")
        print()

    print("📊 Top Pairwise Semantic Similarities:")
    for idx, pair in enumerate(pairwise[:15], 1):
        a = pair["note_a"]
        b = pair["note_b"]
        sim = pair["similarity"]
        marker = "🔗 LINKED" if pair["is_linked"] else "   unlinked"
        print(f" {idx:2d}. [{sim:.4f}] {marker} | ({a['category']}) {a['summary'][:32]} <-> ({b['category']}) {b['summary'][:32]}")

    if orphans:
        print(f"\n🌱 Orphan Notes (no links above {threshold:.2f}):")
        for orphan in orphans:
            print(f"   • ({orphan['category']}) {orphan['summary']} [ID: {orphan['id'][:8]}...]")
    else:
        print(f"\n✨ Fully Connected: All {total_notes} notes have at least one link!")


def main():
    parser = argparse.ArgumentParser(
        description="SecondSelf Auto-Link — compute note embeddings and insert semantic links"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold for linking (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Print detailed similarity report and link statistics"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Compute links and display report without updating markdown files"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of all embeddings, ignoring cache"
    )

    args = parser.parse_args()

    # Determine if dry_run or reporting
    is_dry_run = args.dry_run or args.report

    results = auto_link_wiki(
        threshold=args.threshold,
        dry_run=is_dry_run,
        force_recompute=args.recompute
    )

    print_report(results)

    if not is_dry_run:
        print(f"\n✅ Auto-linking complete: {results.get('files_updated', 0)} note file(s) updated in wiki/.")
    else:
        print("\nℹ️  Dry-run mode: no note files were modified.")


if __name__ == "__main__":
    main()
