#!/usr/bin/env python3
"""SecondSelf — Streamlit App (Week 4.2 / The Oracle, part 2)

Two-tab UI:
  • Graph  — interactive vis-network force-directed graph of the wiki
  • Ask    — RAG question-answering over your personal notes

Secrets / env-var priority:
  1. st.secrets["GROQ_API_KEY"]  (Streamlit Cloud)
  2. os.getenv("GROQ_API_KEY")   (local .env via python-dotenv)
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

WIKI_DIR = Path(os.environ.get("SECONDSELF_WIKI_DIR", ROOT / "wiki"))
GRAPH_JSON = Path(os.environ.get("SECONDSELF_GRAPH_FILE", ROOT / "graph.json"))
EMBEDDINGS_JSON = Path(os.environ.get("SECONDSELF_EMBEDDINGS_FILE", ROOT / "embeddings.json"))
GRAPH_HTML_PATH = ROOT / "graph.html"

# ── Secrets bootstrap ────────────────────────────────────────────────────────
# Load .env for local runs first (no-op if already set)
load_dotenv()

# Prefer st.secrets (Streamlit Cloud) → fall back to env
def _resolve_groq_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")

# Inject resolved key into environment so ask.py picks it up normally
_key = _resolve_groq_key()
if _key:
    os.environ["GROQ_API_KEY"] = _key

# ── Streamlit page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="SecondSelf — Your Second Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    padding: 0 20px;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    font-size: 14px;
}

/* Source card */
.source-card {
    background: rgba(124,106,247,0.08);
    border: 1px solid rgba(124,106,247,0.25);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
}
.source-card .score-badge {
    display: inline-block;
    background: rgba(124,106,247,0.2);
    color: #a78bfa;
    border-radius: 999px;
    padding: 1px 8px;
    font-size: 11px;
    font-weight: 600;
    margin-left: 8px;
}
.source-card .note-id {
    font-family: 'Courier New', monospace;
    font-size: 10px;
    color: rgba(255,255,255,0.35);
    margin-top: 3px;
}

/* Answer box */
.answer-box {
    background: rgba(34,211,238,0.05);
    border-left: 3px solid #22d3ee;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    line-height: 1.7;
}

/* No-notes warning */
.no-notes-banner {
    background: rgba(249,115,22,0.1);
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 10px;
    padding: 16px;
    color: #fb923c;
    font-size: 14px;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 48px 24px;
    color: rgba(255,255,255,0.4);
}
</style>
""", unsafe_allow_html=True)


# ── Health check helpers ─────────────────────────────────────────────────────
def _check_data_readiness() -> dict:
    """Return a dict of {resource: bool} indicating which data files exist."""
    notes = list(WIKI_DIR.glob("**/*.md")) if WIKI_DIR.exists() else []
    return {
        "wiki_dir": WIKI_DIR.exists(),
        "wiki_notes": len(notes),
        "graph_json": GRAPH_JSON.exists(),
        "embeddings_json": EMBEDDINGS_JSON.exists(),
    }


def _show_missing_data_banner(status: dict):
    missing = []
    if not status["wiki_dir"] or status["wiki_notes"] == 0:
        missing.append("`wiki/` — no notes captured yet")
    if not status["graph_json"]:
        missing.append("`graph.json` — run `python build_graph.py` to generate")
    if not status["embeddings_json"]:
        missing.append("`embeddings.json` — run `python link.py` to generate")
    if missing:
        st.error(
            "**SecondSelf data not found.** Missing:\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n\nRun the pipeline locally (`capture.py → classify.py → link.py → build_graph.py`) "
            "and commit the output files, or run them in the deployed environment.",
            icon="⚠️"
        )
    return missing


# ── Graph HTML builder ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _build_graph_html(graph_json_mtime: float) -> str:
    """
    Read graph.html template and inline graph.json data directly —
    replacing the fetch('./graph.json') call with an inline JS object.
    This is required because Streamlit's html() component has no file server
    that would respond to relative fetch() requests.
    """
    if not GRAPH_HTML_PATH.exists():
        return "<p style='color:#fb923c;padding:24px'>graph.html not found.</p>"
    if not GRAPH_JSON.exists():
        return "<p style='color:#fb923c;padding:24px'>graph.json not found — run build_graph.py first.</p>"

    graph_data = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))
    inline_json = json.dumps(graph_data, ensure_ascii=False)

    html = GRAPH_HTML_PATH.read_text(encoding="utf-8")

    # Replace the async fetch block with a synchronous inline assignment
    old_block = (
        "  let graphData;\n"
        "  try {\n"
        "    const resp = await fetch('./graph.json');\n"
        "    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);\n"
        "    graphData = await resp.json();\n"
        "  } catch (err) {\n"
        "    console.error('Failed to load graph.json:', err);\n"
        "    graphData = { nodes: [], edges: [] };\n"
        "  }"
    )
    new_block = f"  const graphData = {inline_json};"

    if old_block in html:
        html = html.replace(old_block, new_block)
    else:
        # Fallback: inject a <script> with the data before </body>
        inject = f"<script>window.__GRAPH_DATA__={inline_json};</script>"
        html = html.replace("</body>", inject + "\n</body>")

    return html


# ── Ask module lazy loader ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def _load_ask_module():
    import ask as ask_module
    return ask_module


# ── Main UI ──────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:26px;font-weight:700;margin-bottom:4px'>🧠 SecondSelf</h1>"
    "<p style='color:rgba(255,255,255,0.45);font-size:14px;margin-bottom:20px'>"
    "Your personal knowledge graph &amp; AI question-answering brain</p>",
    unsafe_allow_html=True
)

status = _check_data_readiness()
missing = _show_missing_data_banner(status)

tab_graph, tab_ask = st.tabs(["🗺️  Graph", "💬  Ask"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: GRAPH VIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_graph:
    if not status["graph_json"]:
        st.markdown(
            "<div class='empty-state'>"
            "<div style='font-size:48px;margin-bottom:16px'>🗺️</div>"
            "<div style='font-size:16px;font-weight:600;margin-bottom:8px'>No graph yet</div>"
            "<div style='font-size:13px'>Run <code>python build_graph.py</code> to generate graph.json</div>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        graph_mtime = GRAPH_JSON.stat().st_mtime
        graph_html = _build_graph_html(graph_mtime)

        col_stats1, col_stats2, col_stats3 = st.columns(3)
        try:
            gd = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))
            with col_stats1:
                st.metric("Notes", len(gd.get("nodes", [])))
            with col_stats2:
                st.metric("Links", len(gd.get("edges", [])))
            with col_stats3:
                orphans = len([
                    n for n in gd.get("nodes", [])
                    if not any(
                        e["source"] == n["id"] or e["target"] == n["id"]
                        for e in gd.get("edges", [])
                    )
                ])
                st.metric("Orphan Notes", orphans)
        except Exception:
            pass

        st.markdown("---")
        from streamlit.components.v1 import html as st_html
        st_html(graph_html, height=680, scrolling=False)

        st.caption(
            "💡 Drag nodes to explore · hover for summary · click to see details · "
            "refresh after running `build_graph.py` to pick up new notes"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: ASK VIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    can_ask = status["embeddings_json"] and status["wiki_notes"] > 0

    if not can_ask:
        st.markdown(
            "<div class='no-notes-banner'>"
            "⚠️ <strong>Ask is unavailable</strong> — embeddings.json or wiki notes are missing. "
            "Run <code>python link.py</code> first to build the embeddings cache."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        if not _resolve_groq_key():
            st.warning(
                "**GROQ_API_KEY not set.** Add it to `.env` locally or to Streamlit secrets for deployment.",
                icon="🔑"
            )

        st.markdown(
            "<p style='font-size:14px;color:rgba(255,255,255,0.55);margin-bottom:16px'>"
            "Ask anything — answers are synthesized exclusively from your captured notes.</p>",
            unsafe_allow_html=True
        )

        with st.form("ask_form", clear_on_submit=False):
            question = st.text_input(
                "Your question",
                placeholder="What did I capture about productivity systems?",
                label_visibility="collapsed",
            )
            col_submit, col_k, col_cutoff = st.columns([3, 1, 1])
            with col_submit:
                submitted = st.form_submit_button("Ask 🔍", use_container_width=True, type="primary")
            with col_k:
                k = st.number_input("Top-k", min_value=1, max_value=10, value=5, step=1)
            with col_cutoff:
                cutoff = st.number_input("Min similarity", min_value=0.0, max_value=1.0, value=0.30, step=0.05, format="%.2f")

        if submitted:
            question = question.strip()

            if not question:
                st.warning("Please enter a question.", icon="💬")
            else:
                ask_module = _load_ask_module()

                with st.spinner("Searching your notes…"):
                    try:
                        result = ask_module.ask(question, k=int(k), similarity_cutoff=float(cutoff))
                    except Exception as e:
                        st.error(f"Unexpected error: {e}", icon="🚨")
                        result = None

                if result:
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])

                    # ── Answer ──────────────────────────────────────────────
                    if answer.startswith("I don't have notes on this"):
                        st.markdown(
                            "<div class='no-notes-banner'>"
                            "🔍 <strong>No relevant notes found</strong> — "
                            "I don't have notes on this topic above the similarity threshold "
                            f"({cutoff:.2f}). Try a lower threshold or capture more notes on this topic."
                            "</div>",
                            unsafe_allow_html=True
                        )
                    elif answer.startswith("Error:"):
                        st.error(answer, icon="🚨")
                    else:
                        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

                    # ── Sources ─────────────────────────────────────────────
                    if sources:
                        st.markdown(
                            f"<p style='font-size:12px;font-weight:600;letter-spacing:.08em;"
                            f"text-transform:uppercase;color:rgba(255,255,255,0.4);margin-top:16px'>"
                            f"Sources used ({len(sources)})</p>",
                            unsafe_allow_html=True
                        )
                        # Enrich sources with summary from embeddings cache
                        try:
                            emb_cache = json.loads(EMBEDDINGS_JSON.read_text(encoding="utf-8"))
                        except Exception:
                            emb_cache = {}

                        for src in sources:
                            note_id = src["id"]
                            score = src["score"]
                            cache_entry = emb_cache.get(note_id, {})
                            summary = cache_entry.get("summary", "—") if isinstance(cache_entry, dict) else "—"
                            category = cache_entry.get("category", "—") if isinstance(cache_entry, dict) else "—"

                            cat_colors = {
                                "Projects": "#f97316",
                                "Areas": "#22d3ee",
                                "Resources": "#a78bfa",
                                "Archives": "#6b7280",
                            }
                            cat_color = cat_colors.get(category, "#94a3b8")

                            st.markdown(
                                f"<div class='source-card'>"
                                f"<span style='color:{cat_color};font-weight:600;font-size:11px'>{category}</span>"
                                f"<span class='score-badge'>{score:.3f}</span><br>"
                                f"<span style='font-size:13px'>{summary}</span>"
                                f"<div class='note-id'>{note_id}</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

        # ── Example questions ────────────────────────────────────────────────
        with st.expander("💡 Example questions from your wiki", expanded=False):
            examples = [
                "How do I make butter chicken?",
                "What is the PARA method?",
                "What did I capture about habits and consistency?",
                "Where should I go hiking?",
                "What Python libraries do I need?",
            ]
            for ex in examples:
                st.markdown(f"- *{ex}*")
