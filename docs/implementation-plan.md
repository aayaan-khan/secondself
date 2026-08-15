# SecondSelf — Implementation Plan

Phase-wise plan derived from `ProblemStatement.md` and `architecture.md`. Each phase has a scope, concrete tasks, exit criteria, and a suggested commit. Phases map onto the 4-week structure but are broken down finer so each is a single, reviewable session of work.

---

## Phase 0 — Setup

**Goal:** A scaffolded repo that runs, with no functionality yet.

**Tasks**
- Create repo structure:
  ```
  secondself/
  ├── raw/
  ├── wiki/
  ├── docs/            (already has ProblemStatement.md, architecture.md, edge-case.md)
  ├── capture.py
  ├── classify.py
  ├── link.py
  ├── build_graph.py
  ├── ask.py
  ├── app.py
  ├── requirements.txt
  └── README.md        (placeholder, filled in at the end)
  ```
- `requirements.txt` — pin initial dependencies: `groq`, `sentence-transformers`, `streamlit`, `trafilatura`, `pypdf`, `python-dotenv`, plus `vis-network` handled client-side (no pip package needed).
- Create `.env.example` (not `.env`) with `GROQ_API_KEY=` as a placeholder — real key goes in your local `.env`, which is gitignored.
- Confirm `.gitignore` covers: `venv/`, `__pycache__/`, `*.pyc`, `.env`, `.streamlit/secrets.toml`.
- Set up and activate a virtual environment, `pip install -r requirements.txt`, confirm it installs clean.

**Exit criteria**
- [ ] Folder structure matches architecture.md
- [ ] `requirements.txt` installs without errors in a clean venv
- [ ] `.env` is gitignored, `.env.example` is committed
- [ ] Empty `raw/` and `wiki/` exist (git doesn't track empty folders — add a `.gitkeep` to each)

**Commit:** `chore: scaffold repo structure, deps, and env template`

---

## Phase 1 — Capture Pipeline (Week 1 / The Archivist)

**Goal:** One command captures a note, a link, or a file into `raw/`.

**Tasks**
- Implement `capture.py` CLI:
  - `python capture.py "some note"` → note capture
  - `python capture.py --url https://...` → link capture, extract text via `trafilatura`
  - `python capture.py --file path.pdf` → file capture, extract text via `pypdf` (or store `content: null` if extraction isn't supported — see edge-case.md §1)
- Write each capture as `raw/{timestamp}_{id}.json` per the schema in architecture.md §2.1.
- Handle edge cases from `edge-case.md` §1 as you go: empty input rejected, dead URLs fall back to storing just the URL, oversized files truncated with a marker.
- Run it on 10+ real pieces of your own scattered information (not test strings) — per acceptance criteria, this must be real data.

**Exit criteria (from ProblemStatement.md, Week 1)**
- [ ] `raw/` and `wiki/` folder structure exists
- [ ] One command captures a note, a link, AND a file
- [ ] Every capture has a timestamp + unique ID
- [ ] 10+ real items captured

**Commit sequence (small, incremental — don't wait for all three types to work):**
1. `feat: capture text notes to raw/`
2. `feat: capture URLs with content extraction`
3. `feat: capture files with content extraction`
4. `test: capture 10+ real items` (or fold into the above if done together)

---

## Phase 2 — Auto-Classify (Week 2.1 / The Librarian, part 1)

**Goal:** Every raw capture gets a PARA category, tags, and summary via LLM.

**Tasks**
- Implement `classify.py`:
  - Read one `raw/*.json` file, build a structured prompt, call Groq (Llama 3).
  - Parse the response defensively — see edge-case.md §2 for malformed-JSON and invalid-category handling.
  - Write `wiki/{category}/{id}.md` with YAML frontmatter per architecture.md §2.2.
- Add caching: skip re-classification if `wiki/**/{id}.md` already exists for that raw item.
- Add retry-with-backoff for rate limits.
- Batch-run across all of last week's real captures.

**Exit criteria (from ProblemStatement.md, Week 2, classify half)**
- [ ] Any raw capture → category + tags + summary automatically
- [ ] PARA categorization working
- [ ] Idempotent — re-running doesn't re-call the LLM on already-classified items

**Commit sequence:**
1. `feat: classify single raw item via Groq/Llama3`
2. `feat: cache classification results, skip already-processed items`
3. `feat: batch classify all captured items`

---

## Phase 3 — Auto-Link (Week 2.2 / The Librarian, part 2)

**Goal:** Notes automatically link to related notes via embedding similarity, no manual tagging.

**Tasks**
- Implement `link.py`:
  - Compute embeddings per note (`sentence-transformers`, e.g. `all-MiniLM-L6-v2`) over `summary + content`.
  - Cache embeddings to `embeddings.json` keyed by note id.
  - For each new/updated note, compute cosine similarity against existing embeddings; above threshold, append bidirectional links to frontmatter.
- **Tune the threshold empirically** against your real wiki — start at 0.55–0.65, run it, eyeball the resulting links, adjust. This is a manual step, not a one-shot guess (see architecture.md §2.3).
- Handle edge cases from edge-case.md §3: zero-notes-to-compare, near-duplicates, bidirectional consistency, self-similarity exclusion.
- Run on 15+ real items → confirm an organized, linked `wiki/`.

**Exit criteria (from ProblemStatement.md, Week 2, link half)**
- [ ] Embeddings computed per note
- [ ] Related notes auto-linked (no manual tagging)
- [ ] Runs on 15+ real items → organized `wiki/`
- [ ] Links are bidirectional and don't self-reference

**Commit sequence:**
1. `feat: compute and cache note embeddings`
2. `feat: auto-link related notes above similarity threshold`
3. `tune: adjust similarity threshold based on real wiki results` (commit each time you change it, with the value and reasoning in the message)

---

## Phase 4 — Graph Data Model (Week 3.1 / The Cartographer, part 1)

**Goal:** Wiki frontmatter → clean `graph.json` (nodes + edges).

**Tasks**
- Implement `build_graph.py`:
  - Walk `wiki/`, read frontmatter from every note.
  - Build nodes (`id`, `label`, `category`, `tags`) and edges (deduplicated from `links:`).
  - Handle edge cases from edge-case.md §4: orphan nodes still included, broken link references skipped with a warning, malformed frontmatter skipped rather than crashing the build.
  - Export to `graph.json` per the schema in architecture.md §2.4.

**Exit criteria**
- [ ] Script builds nodes + edges from notes and exports clean JSON
- [ ] Orphan and malformed-frontmatter cases don't crash the build
- [ ] Rebuildable from wiki alone (no dependency on prior graph.json state)

**Commit:** `feat: build graph.json from wiki frontmatter`

---

## Phase 5 — Interactive Graph Rendering (Week 3.2 / The Cartographer, part 2)

**Goal:** `graph.json` rendered as a hoverable, draggable, zoomable force-directed graph.

**Tasks**
- Build the render using `vis-network` (or Cytoscape.js) as a standalone HTML page first (fastest to iterate on before embedding in Streamlit).
- Implement: node coloring by category, hover popup with summary/tags (truncated per edge-case.md §5), drag-to-explore, zoom.
- Handle empty-graph state (fresh wiki) — render an empty-state message, not a blank broken canvas.
- Sanitize/escape note content injected into tooltips.

**Exit criteria**
- [ ] Interactive force-directed graph renders from `graph.json`
- [ ] Hover reveals note content
- [ ] Drag + zoom work
- [ ] Built from real notes, not dummy data
- [ ] Empty-state handled gracefully

**Commit:** `feat: interactive graph render (vis-network)`

---

## Phase 6 — Ask Your Brain (Week 4.1 / The Oracle, part 1)

**Goal:** Retrieval-augmented Q&A over your own notes.

**Tasks**
- Implement `ask.py` with two separable functions (per architecture.md §2.6):
  - `retrieve(question, k)` — embed the question, cosine-similarity search against cached embeddings, return top-k notes.
  - `generate(question, context_notes)` — build a prompt with retrieved content, call Groq, return the answer.
  - `ask(question)` — wraps both.
- Handle edge cases from edge-case.md §6: no relevant notes found → say so, don't hallucinate; minimum similarity cutoff for retrieval, not just top-k regardless of score; context-length capping if retrieved notes are too long combined.
- Test against real questions about your own captured notes — not synthetic Q&A pairs.

**Exit criteria**
- [ ] `ask()` returns answers synthesized from your own notes (retrieval + LLM)
- [ ] Explicitly handles "no relevant notes" instead of hallucinating
- [ ] Retrieval and generation are independently testable

**Commit sequence:**
1. `feat: retrieval function over cached embeddings`
2. `feat: generate answer from retrieved notes via LLM`
3. `feat: ask() end-to-end, handle no-match case`

---

## Phase 7 — Local Integration Testing

**Goal:** Confirm the full pipeline works end-to-end locally before touching deployment.

**Tasks**
- Run the complete flow on a clean-ish state: capture → classify → link → build_graph → ask, using real data throughout.
- Walk through the cross-cutting edge cases from edge-case.md §8: kill a batch classify run mid-way and confirm it resumes cleanly rather than corrupting state; hand-edit a wiki note's tags and confirm classify.py doesn't silently overwrite it on next run.
- Fix anything that breaks under this integration pass before moving to the UI phase — this is cheaper to catch now than after wiring Streamlit on top.

**Exit criteria**
- [ ] Full pipeline runs end-to-end locally without manual intervention
- [ ] Resumability confirmed for classify/link batch runs
- [ ] Manual wiki edits survive a re-run

**Commit:** `test: verify end-to-end pipeline locally, fix integration issues` (or several small fix commits as issues surface)

---

## Phase 8 — UI & Deployment (Week 4.2 / The Oracle, part 2)

**Goal:** One Streamlit app (graph + ask), deployed with a public URL.

**Tasks**
- Build `app.py`:
  - Tabs or sidebar: Graph view / Ask view.
  - Graph view embeds the Week 3 render via `streamlit.components.v1.html()`.
  - Ask view: text input → `ask()` → display answer plus which notes were used as sources.
  - Handle fresh-checkout startup (edge-case.md §7): if `wiki/`/`graph.json` are missing, either they're committed (see the open question below) or the app fails with a clear message rather than crashing silently.
- **Resolve the open question from architecture.md §6 now, explicitly:** commit `wiki/`, `graph.json`, and `embeddings.json` to the repo so the deployed app has real data without needing local regeneration. Document this decision in the README once written.
- Move the API key out of any hardcoded location into `st.secrets` / environment variables; confirm it works from Streamlit's secrets manager, not just a local `.env`.
- Test with a clean venv locally against the exact `requirements.txt` before deploying, to catch missing-dependency issues before they surface as a deploy failure.
- Deploy to Streamlit Community Cloud (or HF Spaces).
- Confirm cold-start behavior — model download time for `sentence-transformers`, whether secrets are picked up correctly.

**Exit criteria**
- [ ] One Streamlit app contains both the graph and the search bar
- [ ] Deployed live with a public URL
- [ ] Full pipeline works end to end in the deployed app (not just locally)
- [ ] API keys confirmed working via Streamlit secrets, not committed to the repo

**Commit sequence:**
1. `feat: streamlit app combining graph and ask views`
2. `chore: move API key handling to st.secrets`
3. `chore: commit wiki/graph.json/embeddings for deployed app`
4. `deploy: configure Streamlit Cloud deployment`

---

## Phase 9 — Final Testing, Polish, README

**Goal:** Ship a project someone else could clone, understand, and run.

**Tasks**
- Final round of testing directly against the deployed public URL (not local), covering the main edge cases from edge-case.md that are user-facing: empty question, no-match question, concurrent-ish usage.
- Write `README.md`: what it is, the pipeline diagram (reuse architecture.md §1), setup instructions, how to run each phase locally, the deployed URL, and the "wiki/graph.json is committed" decision explained so a reader isn't confused by data already being in the repo.
- Clean up any dead code, stray debug prints, or leftover test files from `raw/`/`wiki/` that aren't part of your real captured data.
- Final commit and push.

**Exit criteria (Final Deliverables from ProblemStatement.md)**
- [ ] Public GitHub repo with a clean README + setup instructions
- [ ] Live deployed URL — interactive graph + ask-your-brain search, both working
- [ ] End-to-end flow verified: capture → classify → link → graph → ask
- [ ] All 4 weekly milestones complete (Capture Pipeline, Self-Organizing Wiki, Living Brain, SecondSelf deployment)

**Commit:** `docs: add final README` → `chore: final cleanup and polish`

---

## Phase-to-Badge Mapping

| Phase(s) | Week | Badge |
|---|---|---|
| 0, 1 | Week 1 | 🏅 The Archivist |
| 2, 3 | Week 2 | 🏅 The Librarian |
| 4, 5 | Week 3 | 🏅 The Cartographer |
| 6, 7, 8, 9 | Week 4 | 🏅 The Oracle |
