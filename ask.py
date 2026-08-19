#!/usr/bin/env python3
"""SecondSelf - Ask Your Brain (Week 4.1 / The Oracle, part 1)

Retrieval-augmented Q&A over the wiki. Embeds the question, retrieves the
most similar cached notes (same embeddings.json cache as link.py), and
synthesizes an answer via Groq/Llama3 using ONLY the retrieved content.
"""

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reconfigure stdout/stderr on Windows for UTF-8 compatibility — same pattern as link.py
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

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — matches link.py's env var names and defaults exactly
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = os.environ.get("SECONDSELF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
WIKI_DIR = Path(os.environ.get("SECONDSELF_WIKI_DIR", Path(__file__).resolve().parent / "wiki"))
CACHE_FILE = Path(os.environ.get("SECONDSELF_EMBEDDINGS_FILE", Path(__file__).resolve().parent / "embeddings.json"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
DEFAULT_SIMILARITY_CUTOFF = float(os.getenv("SECONDSELF_SIMILARITY_CUTOFF", "0.30"))
MAX_CONTEXT_CHARS = int(os.getenv("SECONDSELF_MAX_CONTEXT", "6000"))
DEFAULT_K = 5

_MODEL_INSTANCE = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """Lazy-load sentence-transformers model. Same pattern as link.py."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL_INSTANCE = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}")
    return _MODEL_INSTANCE


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors. Identical to link.py's implementation."""
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


def load_embeddings_cache(cache_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load cached embeddings from JSON file. Same format/behavior as link.py."""
    target_path = cache_path or CACHE_FILE
    if not target_path.exists():
        return {}
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Warning: Failed to load embeddings cache ({e}).", file=sys.stderr)
        return {}


def parse_note_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Parse YAML frontmatter and content body from a markdown note.

    Regex-based, no PyYAML dependency — mirrors link.py's parse_note_file
    exactly so both scripts read the same frontmatter schema identically.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}", file=sys.stderr)
        return None

    match = re.match(r"^---\s*\n([\s\S]*?)\n---\s*\n?([\s\S]*)$", content)
    if not match:
        return None

    frontmatter_raw = match.group(1)
    body = match.group(2).strip()

    note_id_m = re.search(r"^id:\s*(.+)$", frontmatter_raw, re.MULTILINE)
    cat_m = re.search(r"^category:\s*(.+)$", frontmatter_raw, re.MULTILINE)
    tags_m = re.search(r"^tags:\s*\[(.*?)\]", frontmatter_raw, re.MULTILINE)
    sum_m = re.search(r"^summary:\s*[\"']?(.*?)[\"']?$", frontmatter_raw, re.MULTILINE)

    if not note_id_m:
        return None

    note_id = note_id_m.group(1).strip()
    category = cat_m.group(1).strip() if cat_m else "Resources"
    summary = sum_m.group(1).strip() if sum_m else ""

    tags = []
    if tags_m and tags_m.group(1).strip():
        tags = [t.strip().strip("\"'") for t in tags_m.group(1).split(",") if t.strip()]

    return {
        "id": note_id,
        "category": category,
        "tags": tags,
        "summary": summary,
        "body": body,
        "path": file_path,
    }


def find_note_by_id(note_id: str, wiki_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Walk WIKI_DIR and return the note whose frontmatter id matches."""
    target_dir = wiki_dir or WIKI_DIR
    if not target_dir.exists():
        return None
    for md_path in target_dir.glob("**/*.md"):
        note = parse_note_file(md_path)
        if note and note["id"] == note_id:
            return note
    return None


def embed_text(text: str) -> List[float]:
    """Embed a single string using the shared sentence-transformers model."""
    model = get_embedding_model()
    vector = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
    return [float(x) for x in vector]


def _get_embedding_vector(cache_entry: Any) -> Optional[List[float]]:
    """embeddings.json stores entries as {"hash":..., "embedding": [...], ...} per link.py's
    sync_embeddings(). Support both that shape and a bare list, defensively."""
    if isinstance(cache_entry, dict):
        return cache_entry.get("embedding")
    if isinstance(cache_entry, list):
        return cache_entry
    return None


def retrieve(
    question: str,
    k: int = DEFAULT_K,
    similarity_cutoff: float = DEFAULT_SIMILARITY_CUTOFF,
) -> List[Dict[str, Any]]:
    """
    Embed the question, score every cached note via cosine similarity,
    and return the top-k notes whose score is >= similarity_cutoff.
    Notes with no resolvable file on disk are skipped defensively.
    """
    if not question or not question.strip():
        return []

    q_embedding = embed_text(question)
    cache = load_embeddings_cache()

    if not cache:
        return []

    scored = []
    for note_id, entry in cache.items():
        note_embedding = _get_embedding_vector(entry)
        if not note_embedding:
            continue
        score = cosine_similarity(q_embedding, note_embedding)
        if score >= similarity_cutoff:
            scored.append((note_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_k = scored[:k]

    results = []
    for note_id, score in top_k:
        note = find_note_by_id(note_id)
        if note is not None:
            results.append({
                "id": note_id,
                "score": score,
                "content": note["body"],
                "summary": note.get("summary", ""),
                "category": note.get("category", ""),
            })
    return results


def truncate_context(notes: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Build a single context string from retrieved notes, capping total length
    to avoid exceeding the LLM context window.
    """
    parts = []
    current_len = 0
    for note in notes:
        header = f"--- Note {note['id']} (score: {note['score']:.3f}) ---\n"
        body = note["content"]
        part = f"{header}{body}\n\n"

        if current_len + len(part) > max_chars:
            remaining = max_chars - current_len - len(header)
            if remaining > 50:
                parts.append(f"{header}{body[:remaining]}\n")
            break

        parts.append(part)
        current_len += len(part)

    return "".join(parts)


def generate(question: str, context_notes: List[Dict[str, Any]]) -> str:
    """
    Build a RAG prompt from the retrieved notes and call Groq (Llama 3).
    Returns the synthesized answer, or a clear error string on failure.
    """
    if not context_notes:
        return "I don't have notes on this"

    context = truncate_context(context_notes)

    system_msg = (
        "You are SecondSelf, a personal knowledge assistant. "
        "Answer the user's question using ONLY the provided notes. "
        "Do not use outside knowledge. "
        "If the notes don't contain the answer, say you don't have notes on that."
    )

    user_msg = (
        f"The user has the following notes:\n\n"
        f"{context}\n"
        f"Based ONLY on these notes, answer the following question:\n"
        f"{question}\n\n"
        f"Answer:"
    )

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        models_to_try = [GROQ_MODEL, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        # Deduplicate while preserving order
        models_to_try = list(dict.fromkeys(models_to_try))

        last_error = None
        for model in models_to_try:
            try:
                chat_completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.2,
                    max_tokens=2048,
                    top_p=0.9,
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if "model_not_found" in err_str or "does not exist" in err_str:
                    continue
                raise e

        if last_error:
            raise last_error
        return "I don't have notes on this"
    except Exception as e:
        return f"Error: Unable to generate answer ({type(e).__name__}: {e})"


def ask(
    question: str,
    k: int = DEFAULT_K,
    similarity_cutoff: float = DEFAULT_SIMILARITY_CUTOFF,
) -> Dict[str, Any]:
    """
    End-to-end RAG wrapper: retrieve, generate, and return the answer
    together with the source note IDs and scores.
    """
    if not question or not question.strip():
        return {
            "answer": "Please provide a non-empty question.",
            "sources": [],
        }

    sources = retrieve(question, k=k, similarity_cutoff=similarity_cutoff)

    if not sources:
        return {
            "answer": "I don't have notes on this",
            "sources": [],
        }

    answer = generate(question, sources)
    return {
        "answer": answer,
        "sources": [{"id": s["id"], "score": s["score"]} for s in sources],
    }


def main():
    """CLI entrypoint for quick manual testing: python ask.py "your question"""
    if len(sys.argv) < 2:
        print("Usage: python ask.py \"your question here\"")
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    result = ask(question)
    print(f"\nQ: {question}\n")
    print(f"A: {result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  • {s['id']}  (score: {s['score']:.3f})")


if __name__ == "__main__":
    main()