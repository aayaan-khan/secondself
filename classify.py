#!/usr/bin/env python3
"""SecondSelf - Auto-Classify Pipeline (Week 2.1 / The Librarian, part 1)

Classifies raw captures from raw/*.json into PARA categories (Projects, Areas, Resources, Archives),
generates tags and summary via Groq (Llama 3), and writes structured Markdown files with YAML
frontmatter to wiki/{category}/{id}.md.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

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

# Load environment variables (.env) with UTF-8 BOM support
load_dotenv(encoding="utf-8-sig")

# Configuration
RAW_DIR = Path(os.environ.get("SECONDSELF_RAW_DIR", Path(__file__).resolve().parent / "raw"))
WIKI_DIR = Path(os.environ.get("SECONDSELF_WIKI_DIR", Path(__file__).resolve().parent / "wiki"))
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
MAX_PROMPT_CONTENT_CHARS = 12000

PARA_CATEGORIES = {"Projects", "Areas", "Resources", "Archives"}


def get_groq_client():
    """Initialize and return the Groq client if API key is set."""
    api_key = (
        os.environ.get("GROQ_API_KEY") or
        os.environ.get("\ufeffGROQ_API_KEY") or
        ""
    ).strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set in environment or .env file.\n"
            "Please create a .env file with your key (e.g. GROQ_API_KEY=gsk_...)."
        )
    import groq
    return groq.Groq(api_key=api_key)


def find_existing_wiki_note(capture_id: str, wiki_dir: Optional[Path] = None) -> Optional[Path]:
    """Check if wiki/**/{capture_id}.md already exists."""
    target_dir = wiki_dir or WIKI_DIR
    if not target_dir.exists():
        return None
    for note_path in target_dir.glob(f"**/{capture_id}.md"):
        if note_path.is_file():
            return note_path
    return None


def normalize_category(category_raw: Optional[str]) -> str:
    """Validate and normalize category to one of the 4 PARA options."""
    if not category_raw:
        return "Resources"
    cleaned = category_raw.strip().capitalize()
    for valid_cat in PARA_CATEGORIES:
        if cleaned.lower() == valid_cat.lower():
            return valid_cat
    print(f"⚠️  Invalid category '{category_raw}' received. Defaulting to 'Resources'.", file=sys.stderr)
    return "Resources"


def clean_tags(tags_raw: Any) -> List[str]:
    """Clean and normalize tags to a list of lowercase alphanumeric strings."""
    if not tags_raw:
        return ["unclassified"]
    if isinstance(tags_raw, str):
        # Split by comma or whitespace
        tags_raw = [t.strip() for t in re.split(r"[,;]+", tags_raw) if t.strip()]
    if not isinstance(tags_raw, list):
        return ["unclassified"]

    cleaned_tags = []
    for tag in tags_raw:
        if not isinstance(tag, str):
            continue
        # Remove hashtags and leading/trailing punctuation
        t = re.sub(r"^#+", "", tag.strip().lower())
        # Replace spaces or underscores with hyphens
        t = re.sub(r"[\s_]+", "-", t)
        # Remove any character not alphanumeric or hyphen
        t = re.sub(r"[^a-z0-9\-]", "", t)
        if t and t not in cleaned_tags:
            cleaned_tags.append(t)

    if not cleaned_tags:
        return ["unclassified"]
    return cleaned_tags[:5]  # Cap at 5 tags


def clean_summary(summary_raw: Optional[str], default_text: str = "") -> str:
    """Normalize and sanitize single-line summary."""
    if not summary_raw or not summary_raw.strip():
        if default_text:
            first_line = default_text.strip().split("\n")[0][:100].strip()
            return f"Summary: {first_line}" if first_line else "Captured knowledge item"
        return "Captured knowledge item"
    # Ensure single line, remove outer quotes
    cleaned = summary_raw.strip().replace("\n", " ").replace("\r", "")
    cleaned = re.sub(r"^[\"']|[\"']$", "", cleaned).strip()
    return cleaned[:150]


def parse_llm_response(raw_text: str, fallback_content: str = "") -> Dict[str, Any]:
    """Defensively parse LLM response to extract category, tags, and summary."""
    cleaned_text = raw_text.strip()

    # Step 1: Remove markdown code block fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned_text, re.IGNORECASE)
    if fence_match:
        cleaned_text = fence_match.group(1).strip()

    # Step 2: Attempt standard JSON parse
    try:
        data = json.loads(cleaned_text)
        if isinstance(data, dict):
            return {
                "category": normalize_category(data.get("category")),
                "tags": clean_tags(data.get("tags")),
                "summary": clean_summary(data.get("summary"), fallback_content)
            }
    except Exception:
        pass

    # Step 3: Attempt to extract JSON object via regex
    json_obj_match = re.search(r"\{[\s\S]*\}", cleaned_text)
    if json_obj_match:
        try:
            data = json.loads(json_obj_match.group(0))
            if isinstance(data, dict):
                return {
                    "category": normalize_category(data.get("category")),
                    "tags": clean_tags(data.get("tags")),
                    "summary": clean_summary(data.get("summary"), fallback_content)
                }
        except Exception:
            pass

    # Step 4: Line-by-line regex fallback
    category = "Resources"
    cat_match = re.search(r'["\']?category["\']?\s*:\s*["\']?([A-Za-z]+)["\']?', cleaned_text, re.IGNORECASE)
    if cat_match:
        category = normalize_category(cat_match.group(1))

    tags = ["unclassified"]
    tags_match = re.search(r'["\']?tags["\']?\s*:\s*\[(.*?)\]', cleaned_text, re.IGNORECASE)
    if tags_match:
        extracted_tags = re.findall(r'["\']([^"\']+)["\']', tags_match.group(1))
        if extracted_tags:
            tags = clean_tags(extracted_tags)

    summary = clean_summary(None, fallback_content)
    sum_match = re.search(r'["\']?summary["\']?\s*:\s*["\']([^"\'\n]+)["\']', cleaned_text, re.IGNORECASE)
    if sum_match:
        summary = clean_summary(sum_match.group(1), fallback_content)

    print("⚠️  LLM returned malformed JSON; parsed with fallback regex.", file=sys.stderr)
    return {
        "category": category,
        "tags": tags,
        "summary": summary
    }


def build_classification_prompt(capture_type: str, content: Optional[str], source_path: Optional[str]) -> str:
    """Build the prompt for the PARA classifier."""
    display_content = content or ""
    if len(display_content) > MAX_PROMPT_CONTENT_CHARS:
        display_content = display_content[:MAX_PROMPT_CONTENT_CHARS] + "\n\n[...truncated for prompt...]"

    if not display_content.strip():
        display_content = f"(No textual content extracted. File path: {source_path or 'unknown'})"

    prompt = f"""You are an expert knowledge organizer applying Tiago Forte's PARA method to classify personal captures.

The PARA framework categories:
1. Projects: Active, goal-oriented efforts with a specific outcome and deadline (e.g. active coding tasks, deliverables, projects in progress).
2. Areas: Long-term responsibilities and spheres of activity to maintain over time (e.g. health, finances, professional skills, ongoing maintenance).
3. Resources: Topics of ongoing interest, reference material, tutorials, guides, bookmarks, documentation, research notes.
4. Archives: Completed, inactive, or historical items from the other categories.

Classify this capture and respond with valid JSON ONLY:
{{
  "category": "Projects | Areas | Resources | Archives",
  "tags": ["tag1", "tag2", "tag3"],
  "summary": "One concise sentence summarizing the core idea (max 120 characters)"
}}

Rules:
- Category MUST be exactly one of: "Projects", "Areas", "Resources", "Archives".
- Tags MUST be 2 to 5 specific, lowercase keywords (no hashtags).
- Summary MUST be a single line.

---
Capture Type: {capture_type}
Source: {source_path or 'Direct capture'}
Content:
{display_content}
---"""
    return prompt


def call_groq_with_retry(
    client: Any,
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 5
) -> str:
    """Call Groq API with exponential backoff and dynamic model fallback cascade."""
    models_to_try = list(dict.fromkeys([model] + FALLBACK_MODELS))
    last_exception = None

    for candidate_model in models_to_try:
        delay = 2.0
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=candidate_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise knowledge classification assistant. Always respond with strict JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                is_model_not_found = "model_not_found" in err_str or "does not exist" in err_str
                is_rate_limit = "rate_limit" in err_str or "429" in err_str or "resource_exhausted" in err_str
                is_transient = "500" in err_str or "503" in err_str or "connection" in err_str or "timeout" in err_str

                if is_model_not_found:
                    # Immediately try next candidate model in cascade
                    break

                if (is_rate_limit or is_transient) and attempt < max_retries:
                    jitter = random.uniform(0.1, 0.5)
                    wait_time = delay + jitter
                    print(
                        f"⚠️  Groq API error on model '{candidate_model}' ({e}). Retrying in {wait_time:.1f}s (attempt {attempt}/{max_retries})...",
                        file=sys.stderr
                    )
                    time.sleep(wait_time)
                    delay *= 2.0
                else:
                    break

    raise last_exception or RuntimeError("Failed to get response from Groq LLM across all models")


def format_wiki_markdown(
    capture_id: str,
    category: str,
    tags: List[str],
    summary: str,
    source_raw_filename: str,
    created_timestamp: str,
    content: Optional[str]
) -> str:
    """Generate Markdown with YAML frontmatter per architecture.md §2.2."""
    # Escape quotes in summary for YAML safety
    safe_summary = summary.replace('"', '\\"')
    tags_formatted = ", ".join(tags)

    markdown_body = content if (content is not None and content.strip()) else "*(No textual content extracted)*"

    doc = f"""---
id: {capture_id}
category: {category}
tags: [{tags_formatted}]
summary: "{safe_summary}"
source_raw: raw/{source_raw_filename}
created: {created_timestamp}
links: []
---

{markdown_body}
"""
    return doc


def classify_raw_file(
    raw_file_path: Path,
    client: Optional[Any] = None,
    force: bool = False,
    wiki_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Classify a single raw/*.json capture file and write wiki/{category}/{id}.md."""
    target_wiki_dir = wiki_dir or WIKI_DIR

    if not raw_file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_file_path}")

    with open(raw_file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    capture_id = raw_data.get("id")
    if not capture_id:
        raise ValueError(f"Missing 'id' in raw capture file: {raw_file_path}")

    # Check cache / idempotency
    existing_note = find_existing_wiki_note(capture_id, target_wiki_dir)
    if existing_note and not force:
        return {
            "id": capture_id,
            "status": "skipped",
            "reason": "already_classified",
            "note_path": str(existing_note)
        }

    capture_type = raw_data.get("type", "note")
    content = raw_data.get("content")
    source_path = raw_data.get("source_path")
    created_timestamp = raw_data.get("timestamp", datetime.now(timezone.utc).isoformat())

    # Build prompt and call LLM
    prompt = build_classification_prompt(capture_type, content, source_path)

    if client is None:
        client = get_groq_client()

    raw_llm_response = call_groq_with_retry(client, prompt)
    parsed = parse_llm_response(raw_llm_response, fallback_content=content or "")

    category = parsed["category"]
    tags = parsed["tags"]
    summary = parsed["summary"]

    # Ensure destination category folder exists
    cat_dir = target_wiki_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    dest_file = cat_dir / f"{capture_id}.md"

    # If the note was previously filed in a different category under force re-classify, remove old file
    if existing_note and existing_note.resolve() != dest_file.resolve():
        try:
            existing_note.unlink()
        except Exception:
            pass

    markdown_text = format_wiki_markdown(
        capture_id=capture_id,
        category=category,
        tags=tags,
        summary=summary,
        source_raw_filename=raw_file_path.name,
        created_timestamp=created_timestamp,
        content=content
    )

    with open(dest_file, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    return {
        "id": capture_id,
        "status": "classified",
        "category": category,
        "tags": tags,
        "summary": summary,
        "note_path": str(dest_file),
        "source_raw": raw_file_path.name
    }


def batch_classify(force: bool = False, wiki_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Batch classify all captures in raw/."""
    target_raw_dir = RAW_DIR
    target_wiki_dir = wiki_dir or WIKI_DIR

    if not target_raw_dir.exists():
        print("ℹ️  No raw/ directory found.")
        return []

    raw_files = sorted(list(target_raw_dir.glob("*.json")))
    if not raw_files:
        print("ℹ️  No captures found in raw/.")
        return []

    client = get_groq_client()
    results = []
    print(f"🚀 Processing {len(raw_files)} raw capture(s)...")

    for idx, raw_file in enumerate(raw_files, 1):
        try:
            res = classify_raw_file(raw_file, client=client, force=force, wiki_dir=target_wiki_dir)
            results.append(res)
            if res["status"] == "classified":
                print(f"[{idx}/{len(raw_files)}] ✅ {raw_file.name} -> wiki/{res['category']}/{res['id']}.md")
                print(f"       Category: {res['category']} | Tags: {res['tags']}")
                print(f"       Summary:  {res['summary']}")
            else:
                print(f"[{idx}/{len(raw_files)}] ⏭️  {raw_file.name} (already classified, skipped)")
        except Exception as e:
            print(f"[{idx}/{len(raw_files)}] ❌ Failed to classify {raw_file.name}: {e}", file=sys.stderr)
            results.append({
                "file": raw_file.name,
                "status": "error",
                "error": str(e)
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="SecondSelf Auto-Classify — categorize raw captures into PARA wiki notes via LLM"
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Path to a single raw JSON file to classify"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Batch classify all unclassified captures in raw/"
    )
    parser.add_argument(
        "--id",
        dest="capture_id",
        help="Classify capture with specific ID from raw/"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-classification even if wiki note already exists"
    )

    args = parser.parse_args()

    # Determine execution mode
    if args.target:
        target_path = Path(args.target).resolve()
        try:
            res = classify_raw_file(target_path, force=args.force)
            if res["status"] == "classified":
                print(f"✅ Classified {target_path.name} -> wiki/{res['category']}/{res['id']}.md")
                print(f"   Category: {res['category']}")
                print(f"   Tags:     {', '.join(res['tags'])}")
                print(f"   Summary:  {res['summary']}")
            else:
                print(f"ℹ️  Note already classified at: {res['note_path']} (use --force to re-classify)")
        except Exception as e:
            print(f"❌ Classification failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.capture_id:
        matching = list(RAW_DIR.glob(f"*_{args.capture_id}.json")) or list(RAW_DIR.glob(f"{args.capture_id}.json"))
        if not matching:
            print(f"❌ No raw capture found for ID '{args.capture_id}' in {RAW_DIR}", file=sys.stderr)
            sys.exit(1)
        try:
            res = classify_raw_file(matching[0], force=args.force)
            if res["status"] == "classified":
                print(f"✅ Classified {matching[0].name} -> wiki/{res['category']}/{res['id']}.md")
                print(f"   Category: {res['category']}")
                print(f"   Tags:     {', '.join(res['tags'])}")
                print(f"   Summary:  {res['summary']}")
            else:
                print(f"ℹ️  Note already classified at: {res['note_path']}")
        except Exception as e:
            print(f"❌ Classification failed: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.all or not args.target:
        try:
            batch_classify(force=args.force)
        except Exception as e:
            print(f"❌ Batch classification failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
