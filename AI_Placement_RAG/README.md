# 🎓 AI Placement Preparation Assistant using RAG

Ask questions about **technical interviews, aptitude, HR questions and resume writing**.
The assistant answers **only from your 4 PDFs** and shows the **source file and page**.
If the answer is not in the PDFs, it says: *"I could not find the answer in the uploaded documents."*

**Stack:** LangChain · Google Gemini · FAISS · ChromaDB · BM25 · Streamlit · LangSmith

| Version | What it adds | Where |
|---|---|---|
| V1 | Simple RAG, in-memory vector store (+ `v1_basics.py`: Prompt → LLM → Parser → Runnable) | `notebooks/RAG_V1…`, `src/v1_basics.py` |
| V2 | FAISS index saved to disk + distance threshold (refuses weak matches) | `notebooks/RAG_V2…` |
| V3 | ChromaDB + metadata (source, page, category) + category filter | `notebooks/RAG_V3…` |
| V4 | **Advanced RAG:** query rewriting → FAISS + BM25 → RRF → Gemini reranking → relevance gate | `notebooks/RAG_V4…` |
| Final | Streamlit web app + LangSmith monitoring + 30-question evaluation | `app.py`, `src/` |

---

## 1. Run it in VS Code (5 minutes)

1. **Unzip** and open the folder in VS Code (`File → Open Folder → AI_Placement_RAG`).
2. Install the **Python** and **Jupyter** extensions (VS Code will offer them).
3. Open a terminal (`Terminal → New Terminal`) and create an environment:

   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate

   pip install -r requirements.txt
   ```
   (Windows: if activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)
   Python **3.10 – 3.13** works.
4. **Add your keys:** copy `.env.example` to `.env` and paste your Gemini key
   (free key: https://aistudio.google.com/apikey). Never share or commit `.env`.
5. **Start the app:**

   ```bash
   streamlit run app.py
   ```
   The first start embeds the PDFs once (about 53 chunks) and saves the indexes in `vectorstores/`;
   later starts load them instantly. Indexes rebuild automatically if a PDF or setting changes.

### Notebooks
Open any file in `notebooks/` and pick the `.venv` kernel (top right). They run in **VS Code and Google Colab**:
they look for the PDFs in `data/` (VS Code) or `/content` (Colab) and read the key from `.env`
or the Colab secret `Gemini_API`.

---

## 2. Project structure

```
AI_Placement_RAG/
├── data/                 the 4 placement PDFs
├── notebooks/            V1–V4 teaching notebooks (fixed, see section 6)
├── src/
│   ├── config.py         all settings (paths, models, thresholds, LangSmith)
│   ├── ingest.py         load PDFs + category metadata + chunking
│   ├── llm.py            Gemini LLM/embeddings + safe error handling (429 / 503)
│   ├── stores.py         in-memory / FAISS / ChromaDB (saved + auto-rebuilt)
│   ├── retrieval.py      BM25, RRF, query rewriting, reranking
│   ├── pipelines.py      V1–V4 behind one class: PlacementRAG.ask(...)
│   ├── v1_basics.py      LangChain basics chain (Prompt → LLM → Parser → Runnable)
│   ├── evaluate.py       30-question comparison of V2 / V3 / V4
│   └── calibrate.py      picks SCORE_THRESHOLD from your own data
├── evaluation/questions.json   30 questions (25 answerable + 5 not in the PDFs)
├── tests/test_offline.py       24 tests, no API key or internet needed
├── app.py                Streamlit web app
├── requirements.txt  .env.example  .gitignore
└── .vscode/              VS Code settings + run configurations
```

---

## 3. Evaluation (project document, section 6)

```bash
python -m src.evaluate --no-llm            # retrieval only: fast, almost no quota
python -m src.evaluate --delay 5           # full run with Gemini for V2, V3, V4
python -m src.evaluate --delay 5 --resume  # continue after a quota stop
```
Results go to `evaluation/results/` (`results.csv` + `summary.md`). It measures:

* **Answer accuracy** – answerable: Gemini answered and ≥ 50 % of the expected keywords appear;
  not-in-PDF questions: the system correctly refused.
* **Retrieval quality** – hit rate: an expected (file, page) is among the returned sources.
* **Response time** – average seconds per question.

⚠️ The Gemini free tier is small (a few requests/day per model). V4 uses ~3 calls per question,
so a full run may need several days, a second model in `LLM_MODELS`, or billing enabled.
`--resume` saves after every question, so nothing is lost.

Paste `summary.md` into your presentation and explain *why* V4 is (or is not) better – that is what the mentor asks.

---

## 4. LangSmith monitoring (section 8)

1. Create a free account at https://smith.langchain.com → *Settings → API Keys*.
2. Put `LANGSMITH_API_KEY=...` (and optionally `LANGSMITH_PROJECT`) in `.env`.
3. Run the app and ask a few questions. The sidebar shows **🟢 LangSmith tracing ON**.

Each question becomes one trace: `placement_rag` → `vector_search` / `keyword_search_bm25`
(retrieved documents) → `rewrite` + `rerank` + `generate_answer` (prompt, LLM call, token usage, latency, errors).
Take screenshots of a V4 trace for the presentation.

---

## 5. Deploy (project document, section 7) – Streamlit Community Cloud

1. Push this folder to a **private** GitHub repository (`.env` is already git-ignored; commit `data/`).
2. https://share.streamlit.io → *New app* → pick the repo, main file `app.py`.
3. *Advanced settings → Secrets*: paste the contents of `.streamlit/secrets.toml.example` with your real keys.
4. Deploy. Anyone with the link can use it, so keep the key's quota in mind (or let users type their own key in the sidebar).

---

## 6. What was fixed compared with your earlier notebooks

**V4 (bugs that stopped it running)**
* `re` was used in `rewrite_query` before it was imported → `NameError`.
* `advanced_ask` was defined three times; the last copy had no error handling and overrode the safe one, and then ran the test automatically.
* The "Step 11 – Gemini LLM" and "Step 15 – Testing" headings were missing.
* A failed rewrite was detected by comparing text to the fallback; now it returns `None` explicitly.
* BM25 searched words glued to punctuation (`method?`); it now uses a proper tokenizer.
* Candidates after RRF raised from 8 to 12 (the project document says top 10–20 → top 3–5).

**V3**
* `SCORE_THRESHOLD = 1.2` could never refuse anything (out-of-scope questions score ≈ 0.85–0.94). Now 0.73, the value you measured in V2.
* Step numbers repeated (two "Step 10"s); duplicate sources were printed; 503 errors are now retried.

**V2**
* The calibrated threshold printed as `0.7300000190734863` (numpy float) – now a clean `0.73`.
* 503 "model busy" errors are retried instead of failing; stale error outputs removed.

**V1** – 503 retry; model list updated.

**All notebooks** – same model list (`gemini-3.5-flash-lite`, backup `gemini-3.6-flash`), work in VS Code and Colab, indexes are saved next to the project (not in random folders), `%pip` instead of `!pip -U`.

**Project document vs. your V1:** the document defines V1 as *LangChain basics* (Prompt → LLM → Parser → Runnable, no retrieval). Your V1 notebook is a simple RAG. I kept your notebook and added `src/v1_basics.py` so both are covered.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `429 RESOURCE_EXHAUSTED` | Gemini quota used up. Wait for reset, add a second model in `LLM_MODELS`, or use a key from a new project / enable billing. The app still shows the retrieved passages. |
| `503 UNAVAILABLE` | Model busy; retried automatically. Try again in a few seconds. |
| "PDFs were not found" | Keep the 4 PDFs, with these exact names, inside `data/`. |
| Everything is refused / nothing is refused | Run `python -m src.calibrate` and set `SCORE_THRESHOLD` in `.env`. |
| Changed a PDF | Nothing to do – indexes rebuild by themselves (or press **Rebuild indexes** in the app). |
| Model name not found | Check your models in Google AI Studio and edit `LLM_MODELS` in `.env`. |

Run the offline tests any time with `pytest -q`.
