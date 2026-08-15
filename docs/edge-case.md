# SecondSelf — Edge Cases & Corner Scenarios

Organized by component. Each item is something the implementation plan should explicitly handle — not just something to "keep in mind."

---

## 1. Capture (`capture.py`)

- **Duplicate capture** — same URL or same note text captured twice. Decide: allow duplicates (simplest, dedupe later) or hash content and skip if identical hash already exists in `raw/`.
- **Empty/whitespace-only input** — `python capture.py ""` should reject with a clear error, not create a blank raw file.
- **File type not extractable** — a `.zip`, `.exe`, or image with no OCR path. Capture the file reference and metadata anyway, but flag `content: null` so classify.py can skip/handle gracefully instead of sending `None` to the LLM.
- **URL capture fails** — dead link, paywall, JS-rendered page `trafilatura` can't parse, or the site blocks scrapers. Fall back to storing just the raw URL as content rather than crashing the whole capture.
- **Very large file** (e.g. a 300-page PDF) — decide a max content length to extract/store, or you'll blow past LLM context limits in classify.py later. Truncate with a clear marker (`[...truncated...]`) rather than silently cutting off.
- **Non-UTF8 / weird encoding text files** — handle decode errors instead of crashing the CLI.
- **Timestamp collisions** — two captures in the same second. UUID in the filename already prevents overwrite, but worth confirming sort order still makes sense when timestamps tie.

## 2. Classify (`classify.py`)

- **LLM returns malformed JSON** — Groq/Llama occasionally wraps output in prose or breaks schema. Always parse defensively (try/except + regex fallback to extract JSON), and log the raw failed response somewhere for debugging rather than silently dropping the item.
- **LLM returns a category outside the 4 PARA options** — validate against the allowed enum; if invalid, default to `Resources` and flag for manual review rather than letting garbage categories propagate into the wiki folder structure.
- **Rate limit / API downtime** — batch classification over 15+ items will hit Groq free-tier limits. Implement retry-with-backoff, and make the batch runner resumable (skip already-classified items) rather than restarting from zero.
- **Already-classified item re-run** — must be a no-op (check cache/existing `wiki/**/{id}.md` before calling the LLM), otherwise you burn API calls and risk overwriting manually-adjusted tags.
- **Content too short to classify meaningfully** — e.g. a two-word note. Decide a minimum content length below which you either skip LLM classification and default to `Resources`/`uncategorized`, or pass it through anyway and accept a possibly weak result.
- **Content too long for the LLM's context window** — truncate or summarize-then-classify for oversized captures (relevant for large PDFs from Week 1).

## 3. Link (`link.py`)

- **Zero existing notes to compare against** — first note ever classified has nothing to link to. Must not crash; just produce empty `links: []`.
- **Threshold produces zero links across the whole wiki** — likely means the threshold is too high; this should be visible/loggable during a run, not silently invisible.
- **Threshold produces a fully-connected graph** — every note links to every note (threshold too low). Log link counts per note so you can spot this during tuning instead of only noticing it visually in Week 3.
- **Near-duplicate notes** — two captures that are almost the same content (e.g. same article saved twice from different sources). Similarity will be ~1.0; decide whether that's a "link" or should instead trigger a duplicate-merge flag.
- **Embeddings cache goes stale** — a note's content is edited after its embedding was cached. Either recompute on every classify run (simplest, costs a bit of time) or track a content hash to know when to invalidate the cached embedding.
- **Bidirectional link consistency** — if note A links to B, B must also link to A. A bug here (one-directional link) won't show up until Week 3's graph looks asymmetric — worth a small consistency check/test.
- **Self-similarity** — make sure a note is never compared against itself and doesn't end up in its own `links:` list.

## 4. Build Graph (`build_graph.py`)

- **Orphan nodes** — a note with zero links still needs to render as an isolated node, not be dropped from `graph.json`.
- **Broken link reference** — a note's frontmatter references a link id that no longer exists (e.g. the linked note was deleted/moved). Skip and log rather than crashing the graph build.
- **Malformed frontmatter** — a hand-edited or partially-written `.md` file with invalid YAML. Skip that file with a warning rather than failing the whole graph build.
- **Large graph performance** — once you're past ~100+ notes, force-directed layout can get sluggish in the browser. Not a Week 3 blocker at your current scale, but worth noting for later if you keep using this for real.

## 5. Graph Rendering (Week 3.2)

- **Empty wiki** — `graph.json` has zero nodes (fresh clone, nothing captured yet). The graph view should render an empty-state message, not a blank broken canvas.
- **Very long note content in hover popup** — truncate the popup preview; don't dump 2000 words into a hover tooltip.
- **Special characters in note content breaking HTML rendering** — sanitize/escape content injected into `vis-network`/`Cytoscape` tooltips to avoid broken layout or (if this ever goes further) injection issues.

## 6. Ask (`ask.py`)

- **Question has no relevant notes** — e.g. asking about a topic you never captured. `ask()` should say "I don't have notes on this" rather than the LLM hallucinating an answer from general knowledge — this is the single most important edge case for a RAG system, since silently hallucinating breaks the entire premise of the product.
- **Retrieved notes are only weakly related** — decide a minimum similarity cutoff for retrieval, not just "top-k regardless of score." Otherwise a weak match gets passed to the LLM as if it were solid context.
- **LLM context limit exceeded** — top-k notes combined are too long for one prompt. Cap total retrieved content length, or summarize each note before inserting into the prompt.
- **Ambiguous question spanning multiple categories** — e.g. "what have I saved about productivity" pulls from Projects, Areas, and Resources at once. Retrieval should not filter by category unless the question implies one.
- **Empty question string** — reject before hitting retrieval/LLM.
- **API failure mid-answer** — Groq call fails or times out. Return a clear error to the UI, don't let the Streamlit app hang or crash.

## 7. App / Deployment (Week 4.2)

- **Fresh deploy with no local state** — the deployed environment is a clean checkout of the repo. If `wiki/`, `graph.json`, and `embeddings.json` aren't committed, the deployed app has nothing to show and no way to regenerate it (no API keys / local files available at runtime). Decide explicitly what gets committed vs. regenerated at deploy time — this is called out as an open question in `architecture.md` and needs to be resolved before Phase 8.
- **API keys in the repo** — a hardcoded key in `classify.py` or `ask.py` will break deployment (missing on Streamlit Cloud) or, worse, leak if pushed. Must use `st.secrets` / environment variables, confirmed working in the deployed environment, not just locally.
- **Streamlit app crashes on missing dependency** — free deployment platforms build from `requirements.txt` fresh; a package that's installed locally but missing from `requirements.txt` will only surface as a deploy-time failure. Test with a clean venv locally before deploying, not just your dev environment.
- **Large `embeddings.json` or model download on cold start** — `sentence-transformers` downloading its model weights on every cold start of a free-tier deployment can be slow or hit storage/memory limits. Worth checking cold-start behavior once deployed, not assuming local speed carries over.
- **Concurrent users on a public URL** — two people asking questions at once on a shared free-tier deployment. Not a hard requirement to solve, but worth at least confirming the app doesn't hard-crash under two simultaneous requests.

## 8. Cross-Cutting

- **Partial pipeline runs** — a crash mid-batch (e.g. classify.py dies on item 8 of 15) shouldn't corrupt already-processed items or require restarting from zero. Every stage should be safely re-runnable/resumable.
- **Folder/category renames after the fact** — if you ever decide to rename a PARA category or restructure `wiki/`, links and `graph.json` must be rebuildable from frontmatter alone, not dependent on folder paths staying fixed.
- **Manual edits to wiki notes** — if you hand-edit a note's tags or category after auto-classification, the system shouldn't silently overwrite that edit on the next pipeline run. Decide whether classify.py ever re-classifies an already-classified note, and default to "no" unless explicitly forced.
