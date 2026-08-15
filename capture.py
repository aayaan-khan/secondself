#!/usr/bin/env python3
"""SecondSelf - Capture Pipeline (Week 1 / The Archivist)

One command captures a note, a link, or a file into raw/.
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

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
RAW_DIR = Path(os.environ.get("SECONDSELF_RAW_DIR", Path(__file__).resolve().parent / "raw"))
MAX_CONTENT_LENGTH = 50000

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".csv", ".tsv", ".html", ".htm", ".xml", ".yaml", ".yml",
    ".rst", ".log", ".sh", ".bat", ".ps1", ".css", ".scss", ".sql",
    ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".java", ".env", ".toml",
    ".ini", ".cfg", ".conf"
}


def get_existing_captures():
    """Load existing captures to detect duplicates."""
    captures = []
    if not RAW_DIR.exists():
        return captures
    for file_path in RAW_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                captures.append((file_path.name, data))
        except Exception:
            continue
    return captures


def check_duplicate(content: Optional[str], capture_type: str, source_path: Optional[str]) -> Optional[str]:
    """Check if an identical capture already exists in raw/."""
    if content is None and source_path is None:
        return None
    for filename, capture in get_existing_captures():
        if capture.get("type") == capture_type:
            if source_path and capture.get("source_path") == source_path:
                return capture.get("id")
            if content and capture.get("content") == content:
                return capture.get("id")
    return None


def extract_url_content(url: str) -> str:
    """Extract readable text from a URL using trafilatura, with fallback to raw URL."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_links=True, include_comments=False)
            if extracted and extracted.strip():
                return extracted.strip()
    except Exception as e:
        print(f"⚠️  URL extraction notice: {e}. Falling back to raw URL.", file=sys.stderr)
    return url


def extract_pdf_content(file_path: Path) -> Optional[str]:
    """Extract text from a PDF file using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(file_path))
        text_parts = []
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())
        full_text = "\n\n".join(text_parts).strip()
        return full_text if full_text else None
    except Exception as e:
        print(f"⚠️  PDF extraction failed ({e}). Storing null content.", file=sys.stderr)
        return None


def extract_text_file(file_path: Path) -> Optional[str]:
    """Read plain text files with encoding fallbacks."""
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠️  File read error ({e}).", file=sys.stderr)
            return None
    # Final fallback with replace
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️  File read fallback failed ({e}). Storing null content.", file=sys.stderr)
        return None


def extract_file_content(file_path_str: str) -> Tuple[Optional[str], str]:
    """Extract content from local file based on extension/type."""
    file_path = Path(file_path_str).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path_str}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a regular file: {file_path_str}")

    suffix = file_path.suffix.lower()
    content: Optional[str] = None

    if suffix == ".pdf":
        content = extract_pdf_content(file_path)
    elif suffix in TEXT_EXTENSIONS or suffix == "":
        content = extract_text_file(file_path)
    else:
        # Binary or unsupported file type
        print(f"ℹ️  Binary/unsupported file type '{suffix}'. Capturing file reference with content: null.")
        content = None

    if content is not None and len(content) > MAX_CONTENT_LENGTH:
        print(f"ℹ️  Content exceeded {MAX_CONTENT_LENGTH} chars. Truncating with marker.")
        content = content[:MAX_CONTENT_LENGTH] + "\n\n[...truncated...]"

    return content, str(file_path)


def capture_item(
    capture_type: str,
    raw_input: str,
    force: bool = False
) -> dict:
    """Process and save a captured item into raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    content: Optional[str] = None
    source_path: Optional[str] = None

    if capture_type == "note":
        cleaned = raw_input.strip()
        if not cleaned:
            raise ValueError("Note content cannot be empty or whitespace-only.")
        content = cleaned
        source_path = None
    elif capture_type == "link":
        cleaned_url = raw_input.strip()
        if not cleaned_url:
            raise ValueError("URL cannot be empty.")
        if not (cleaned_url.startswith("http://") or cleaned_url.startswith("https://")):
            cleaned_url = "https://" + cleaned_url
        content = extract_url_content(cleaned_url)
        source_path = None
    elif capture_type == "file":
        cleaned_path = raw_input.strip()
        if not cleaned_path:
            raise ValueError("File path cannot be empty.")
        content, source_path = extract_file_content(cleaned_path)
    else:
        raise ValueError(f"Unknown capture type: {capture_type}")

    # Duplicate check
    if not force:
        existing_id = check_duplicate(content, capture_type, source_path)
        if existing_id:
            print(f"⚠️  Duplicate detected: identical item already captured with ID {existing_id}.")
            print("   Use --force to capture anyway.")
            return {
                "id": existing_id,
                "status": "duplicate",
                "message": "Duplicate skipped"
            }

    capture_id = str(uuid.uuid4())
    now_utc = datetime.now(timezone.utc)
    iso_timestamp = now_utc.isoformat()
    filename_timestamp = now_utc.strftime("%Y%m%dT%H%M%SZ")

    record = {
        "id": capture_id,
        "timestamp": iso_timestamp,
        "type": capture_type,
        "content": content,
        "source_path": source_path
    }

    filename = f"{filename_timestamp}_{capture_id}.json"
    dest_file = RAW_DIR / filename

    with open(dest_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return {
        "id": capture_id,
        "filename": filename,
        "path": str(dest_file),
        "type": capture_type,
        "status": "saved",
        "record": record
    }


def main():
    parser = argparse.ArgumentParser(
        description="SecondSelf Capture Pipeline — capture notes, URLs, and files into raw/"
    )
    parser.add_argument(
        "positional_note",
        nargs="?",
        help="Text note to capture (e.g. python capture.py 'My quick note')"
    )
    parser.add_argument(
        "--note", "-n",
        dest="flag_note",
        help="Explicit text note to capture"
    )
    parser.add_argument(
        "--url", "-u",
        dest="url",
        help="Web URL to capture and extract (e.g. python capture.py --url https://...)"
    )
    parser.add_argument(
        "--file", "-f",
        dest="file",
        help="File path to capture and extract (e.g. python capture.py --file document.pdf)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force capture even if duplicate content exists in raw/"
    )

    args = parser.parse_args()

    # Determine inputs
    inputs = []
    if args.positional_note:
        inputs.append(("note", args.positional_note))
    if args.flag_note:
        inputs.append(("note", args.flag_note))
    if args.url:
        inputs.append(("link", args.url))
    if args.file:
        inputs.append(("file", args.file))

    if len(inputs) == 0:
        parser.print_help()
        sys.exit(1)

    if len(inputs) > 1:
        print("❌ Error: Please specify only one capture target at a time (note, --url, or --file).", file=sys.stderr)
        sys.exit(1)

    capture_type, target = inputs[0]

    try:
        result = capture_item(capture_type, target, force=args.force)
        if result.get("status") == "saved":
            print(f"✅ Captured [{capture_type.upper()}] -> raw/{result['filename']}")
            print(f"   ID: {result['id']}")
            preview = (result['record']['content'] or '')[:120].replace('\n', ' ')
            if preview:
                print(f"   Preview: {preview}...")
            if result['record']['source_path']:
                print(f"   Source: {result['record']['source_path']}")
        else:
            sys.exit(0)
    except Exception as err:
        print(f"❌ Capture failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
