# SecondSelf — Your Personal AI Second Brain

> **Not a notes app. Not a chatbot. A brain that organizes itself and answers for you.**

Every notes app fails the same way: you capture hundreds of notes, bookmarks, PDFs, and ideas — and then you never find them again. Information goes in, but nothing comes back out. Notes sit in folders nobody re-reads. Bookmarks pile up unread. Knowledge doesn't compound.

**SecondSelf** is an end-to-end knowledge pipeline that captures scattered information, automatically categorizes and links it via local embeddings and LLMs, visualizes your knowledge in an interactive force-directed graph, and lets you ask plain-English questions grounded exclusively in your notes.

---

## 🚀 Live Demo

- **Deployed URL:** *[Coming soon / Insert your deployed Streamlit Community Cloud URL here]*

---

## 🏗️ Architecture & Pipeline Flow

SecondSelf is a modular pipeline where plain files (JSON and Markdown) serve as the interface between components. Each stage can be inspected, tested, and debugged independently:

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌────────────┐   ┌─────────────┐
│ capture  │──▶│ classify  │──▶│  link    │──▶│ build_graph│──▶│  ask + app  │
│ (Wk 1)   │   │ (Wk 2.1)  │   │ (Wk 2.2) │   │  (Wk 3)    │   │  (Wk 4)     │
└──────────┘   └───────────┘   └──────────┘   └────────────┘   └─────────────┘
     │               │               │               │                │
   raw/            wiki/           wiki/         graph.json        Streamlit
  (*.json)      (*.md + meta)  (*.md + links)                     (public URL)
```

1. **Capture (`capture.py`):** Low-friction CLI that ingests raw notes, URLs (via `trafilatura`), or PDFs/files (via `pypdf`) into `raw/*.json` with unique UUIDs and timestamps.
2. **Auto-Classify (`classify.py`):** Passes raw captures through Groq (Llama 3 / open LLMs) to assign PARA categories (`Projects`, `Areas`, `Resources`, `Archives`), tags, and concise summaries formatted as YAML frontmatter in `wiki/`.
3. **Auto-Link (`link.py`):** Computes sentence embeddings (`all-MiniLM-L6-v2`) cached to `embeddings.json` and automatically inserts bidirectional semantic links above a tuned cosine similarity threshold.
4. **Graph Data Model (`build_graph.py`):** Converts note frontmatter into a clean, deduplicated `graph.json` node-edge representation.
5. **Ask Your Brain (`ask.py`):** Retrieval-augmented Q&A that searches cached note embeddings and synthesizes answers using Groq LLMs without hallucination.
6. **Web App (`app.py`):** Unified Streamlit interface featuring an interactive force-directed `vis-network` graph and live RAG search.

---

## 📦 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/aayaan-khan/secondself.git
cd secondself
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the root directory (based on `.env.example`):
```bash
GROQ_API_KEY=your_groq_api_key_here
```
*(Optional custom overrides: `GROQ_MODEL`, `SECONDSELF_SIMILARITY_CUTOFF`, `SECONDSELF_MAX_CONTEXT`)*

---

## 🛠️ Running the Pipeline Locally

Run each pipeline stage independently from the command line:

### Step 1: Capture knowledge
```bash
# Capture a text note
python capture.py "Review quarterly cloud infrastructure budget"

# Capture a web article with automatic content extraction
python capture.py --url https://en.wikipedia.org/wiki/Vector_database

# Capture a PDF document
python capture.py --file path/to/document.pdf
```

### Step 2: Auto-classify captures into the wiki
```bash
# Batch classify all unprocessed captures in raw/
python classify.py --all
```

### Step 3: Compute embeddings and auto-link related notes
```bash
# Compute embeddings and insert bidirectional links
python link.py

# View detailed similarity report and link statistics
python link.py --report
```

### Step 4: Build the knowledge graph
```bash
python build_graph.py
```

### Step 5: Ask questions via CLI or launch the Web App
```bash
# CLI query
python ask.py "What are the core principles of the PARA method?"

# Launch the Streamlit Web Application
streamlit run app.py
```

---

## 🧪 Running Automated Tests

Run the full test suite (73 unit & integration tests covering idempotency, crash resumption, edge cases, and RAG):

```bash
python -m unittest discover -p "test_*.py"
```

---

## 💡 Cold-Start Deployment & Data Architecture

To support instantaneous cold starts on serverless hosting platforms (such as Streamlit Community Cloud or Hugging Face Spaces) without requiring local regeneration or API keys on build:
- **`wiki/`**, **`graph.json`**, and **`embeddings.json`** are intentionally committed to git.
- When deployed, the app renders the complete graph and performs instant embedding retrieval immediately upon container launch.
- Secrets are resolved with fallback priority: `st.secrets["GROQ_API_KEY"]` → `os.getenv("GROQ_API_KEY")` → `.env`.

---

## ⚠️ Known Limitations & Design Decisions

1. **Similarity Cutoff & Empirical Tuning:** The cosine similarity cutoff (`0.30`) is tuned against the current corpus. As the wiki expands to hundreds of notes, the threshold or top-$k$ parameters can be tuned in `.env` (`SECONDSELF_SIMILARITY_CUTOFF`) to balance precision and recall.
2. **Upstream LLM Model Identifiers:** Cloud LLM providers occasionally update or deprecate specific model identifiers (e.g. `llama-3.1-8b-instant`). Both `classify.py` and `ask.py` implement an automated fallback cascade (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) to guarantee continuous uptime without manual script modifications.
3. **No Database Requirement:** At personal scale (tens to thousands of notes), flat files and cached JSON embeddings are chosen for inspection simplicity, offline portability, and git-native version control.

---

## 🏅 Badges & Milestones

- 🏅 **Week 1 — The Archivist:** Capture Pipeline
- 🏅 **Week 2 — The Librarian:** Auto-Classification & Auto-Linking
- 🏅 **Week 3 — The Cartographer:** Graph Data Model & Interactive Render
- 🏅 **Week 4 — The Oracle:** Retrieval-Augmented Q&A & Streamlit Deployment
