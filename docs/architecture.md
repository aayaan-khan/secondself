# SecondSelf — Architecture

## 1. System Overview

SecondSelf is a pipeline, not a single app. Data flows one direction through five stages, and each stage's output is a plain file (JSON/Markdown) that the next stage reads. This keeps every week independently testable — you can inspect `raw/`, `wiki/`, or `graph.json` directly at any point without running the whole system.

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌────────────┐   ┌─────────────┐
│ capture  │──▶│ classify  │──▶│  link    │──▶│ build_graph│──▶│  ask + app  │
│ (Wk 1)   │   │ (Wk 2.1)  │   │ (Wk 2.2) │   │  (Wk 3)    │   │  (Wk 4)     │
└──────────┘   └───────────┘   └──────────┘   └────────────┘   └─────────────┘
     │               │               │               │                │
   raw/            wiki/           wiki/         graph.json        Streamlit
  (*.json)      (*.md + meta)  (*.md + links)                     (public URL)
```

Design principle: **files are the interface between components.** No component calls another directly in-process except through app.py at the very end. This makes each week debuggable in isolation — a strict requirement given each week is graded on real data, not mocks.

---

## 2. Component Breakdown

### 2.1 `capture.py` (Week 1)

**Responsibility:** Get anything into `raw/` with zero friction.

- CLI entrypoint: `python capture.py "some note text"` or `python capture.py --file path/to/file.pdf` or `python capture.py --url https://...`
- Each capture is written as one JSON file to `raw/`:
  ```json
  {
    "id": "uuid4",
    "timestamp": "ISO-8601",
    "type": "note | link | file",
    "content": "raw text, or extracted text for files, or the URL",
    "source_path": "original file path if type=file, else null"
  }
  ```
- File naming: `raw/{timestamp}_{id}.json` — sortable by filename, uniquely identifiable.
- Link and file captures get lightweight content extraction (e.g. `trafilatura` for URLs, `pypdf` for PDFs) so downstream classification has actual text to work with, not just a path.

**Why JSON here and Markdown later:** `raw/` is machine-only scratch space, never hand-edited. `wiki/` is the human-readable, AI-organized layer — Markdown makes sense there.

### 2.2 `classify.py` (Week 2.1)

**Responsibility:** Turn one raw capture into one categorized wiki note.

- Reads a raw JSON file, sends its content to Groq (Llama 3) with a structured prompt requesting:
  - `category`: one of `Projects | Areas | Resources | Archives`
  - `tags`: list of 2–5 keywords
  - `summary`: one line
- Writes output as a Markdown file with YAML frontmatter to `wiki/{category}/{id}.md`:
  ```markdown
  ---
  id: uuid4
  category: Resources
  tags: [ai, notes, productivity]
  summary: One-line summary of the note
  source_raw: raw/{original_filename}.json
  created: ISO-8601
  links: []
  ---

  {full original content}
  ```
- **Caching layer:** before calling the LLM, check if `wiki/**/{id}.md` already exists — never re-classify the same raw item twice. This matters because free-tier LLM rate limits will get hit fast during dev iteration.

### 2.3 `link.py` (Week 2.2)

**Responsibility:** Find related notes and insert links, with no manual tagging.

- Uses `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) to embed each note's `summary + content`.
- Embeddings are cached to `embeddings.json` (or a local vector store like `chromadb`) keyed by note id — recomputing embeddings for the whole wiki on every run doesn't scale past a few dozen notes.
- For each new/updated note, compute cosine similarity against all existing embeddings.
- Above a threshold (start at `0.55`–`0.65`, tune against real notes — see note below), append the related note's id to the `links:` frontmatter field of both notes (bidirectional).
- **Threshold tuning is a manual step, not a one-shot guess.** Plan to run this against your real 15+ notes, eyeball the resulting links, and adjust — don't hardcode a number and move on.

### 2.4 `build_graph.py` (Week 3.1)

**Responsibility:** Turn the wiki's frontmatter into a graph JSON.

- Walks `wiki/`, reads every note's frontmatter.
- Nodes: `{id, label: summary, category, tags}`
- Edges: one per entry in each note's `links:` list (deduplicated, since links are bidirectional).
- Output: `graph.json`:
  ```json
  {
    "nodes": [{ "id": "...", "label": "...", "category": "...", "tags": [...] }],
    "edges": [{ "source": "...", "target": "..." }]
  }
  ```
- This script has no LLM/embedding dependency — it's pure file I/O, so it's fast to re-run anytime the wiki changes.

### 2.5 Graph rendering (Week 3.2)

**Responsibility:** Render `graph.json` as an interactive, hoverable, force-directed graph.

- Library: `vis-network` (simpler API, good defaults) or `Cytoscape.js` (more control if you want custom layouts later).
- Runs client-side in the browser — either as a standalone HTML file during dev, or embedded directly into the Streamlit app via `streamlit.components.v1.html()` in the final version.
- Node color = category, node size = number of connections (optional polish, not core).
- Hover → popup showing summary + tags. Click → optionally expand to full content.

### 2.6 `ask.py` (Week 4.1)

**Responsibility:** Retrieval-augmented Q&A over the wiki.

- `ask(question: str) -> str`:
  1. Embed the question using the same model as `link.py`.
  2. Retrieve top-k (e.g. 5) most similar notes by cosine similarity against cached embeddings.
  3. Build a prompt: system instruction + retrieved notes' content + the question.
  4. Send to Groq/Llama 3, return the synthesized answer.
- Keep retrieval and generation as separate, testable functions (`retrieve(question, k)` and `generate(question, context_notes)`) — you'll want to debug retrieval quality independently of answer quality.

### 2.7 `app.py` (Week 4.2)

**Responsibility:** One Streamlit app, two views.

- Sidebar or tabs: **Graph** view / **Ask** view.
- Graph view: embeds the Week 3 graph render.
- Ask view: text input → calls `ask()` → displays answer + which notes were used as sources (important for trust/debuggability).
- On startup, app should be able to run the full pipeline if `wiki/` or `graph.json` don't exist yet (or at minimum, fail with a clear message rather than crashing silently) — this matters once deployed, since the deployed environment starts from a fresh checkout of the repo.

---

## 3. Data Flow Summary

| Stage | Input | Output | Trigger |
|---|---|---|---|
| Capture | user input (text/url/file) | `raw/*.json` | manual CLI call |
| Classify | `raw/*.json` | `wiki/{category}/*.md` | manual/batch run over `raw/` |
| Link | `wiki/**/*.md` | updated frontmatter + `embeddings.json` | run after classify |
| Build graph | `wiki/**/*.md` | `graph.json` | run after link, or on-demand in app |
| Ask | question + `embeddings.json` + `wiki/` | answer string | called live from `app.py` |

---

## 4. Tech Stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | matches skill/tooling ecosystem |
| LLM | Groq API (Llama 3) | free tier, fast inference |
| Embeddings | `sentence-transformers` (local) | free, no API dependency, works offline |
| Graph viz | `vis-network` or `Cytoscape.js` | mature, force-directed, hover/drag built in |
| App framework | Streamlit | fastest path to a deployed public URL |
| Storage | Flat files (JSON/Markdown) | no DB setup needed, git-friendly, human-inspectable |
| Deployment | Streamlit Community Cloud or HF Spaces | free, direct GitHub integration |

No database. At this scale (tens of notes), flat files plus a cached embeddings file are simpler to reason about and version-control than standing up Postgres/Chroma-as-a-service.

---

## 5. Key Constraints & Decisions

- **Every component is independently runnable from the CLI.** No component should require the full app to be running to test it. This is what makes each week's acceptance criteria checkable in isolation.
- **Idempotency matters more than speed.** Re-running `classify.py` or `link.py` on already-processed notes should be a no-op (checked via cache), not a re-spend of LLM calls or a duplicate-link bug.
- **The frontmatter in `wiki/*.md` is the single source of truth** for category, tags, links. `graph.json` is a derived artifact, not authoritative — it can always be rebuilt from the wiki.
- **Deployment reads `wiki/` and `graph.json` from the repo itself** (committed, not gitignored) so the deployed app has real data to show without needing your local machine or API keys to regenerate everything first. API keys still stay in Streamlit secrets, never in the repo.

---

## 6. Open Questions to Resolve in `implementation-plan.md`

- Exact similarity threshold for auto-linking (tune empirically in Phase 3/4).
- Whether `embeddings.json` gets committed to git (binary-ish, will grow) or regenerated on deploy.
- Rate-limit handling/backoff strategy for the Groq API during batch classification runs.
