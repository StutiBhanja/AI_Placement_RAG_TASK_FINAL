"""AI Placement Preparation Assistant - Streamlit web app.

Run:   streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from src import config
from src.llm import LLMUnavailable
from src.pipelines import VERSIONS, PlacementRAG, RAGResult

st.set_page_config(page_title="AI Placement Preparation Assistant", page_icon="🎓", layout="wide")

# --------------------------------------------------------------------------- #
# Keys: .env / environment first, then Streamlit secrets (for deployment)
# --------------------------------------------------------------------------- #
def _from_secrets(name: str) -> None:
    try:
        if name in st.secrets and not os.getenv(name):
            os.environ[name] = str(st.secrets[name])
    except Exception:  # no secrets.toml: perfectly fine locally
        pass


for _name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
    _from_secrets(_name)

with st.sidebar:
    st.header("⚙️ Settings")
    if not config.get_api_key():
        typed = st.text_input("Gemini API key", type="password",
                              help="Get a free key at https://aistudio.google.com/apikey")
        if typed:
            os.environ["GOOGLE_API_KEY"] = typed
    api_key = config.get_api_key()
    tracing = config.setup_langsmith()

    version = st.selectbox("RAG version", list(VERSIONS), index=3, format_func=VERSIONS.get)
    category = st.selectbox("Search only in", ["All"] + config.CATEGORIES,
                            help="Filters by document category (technical / resume / hr / aptitude).")
    show_details = st.checkbox("Show retrieval details", value=False)
    st.caption("🟢 LangSmith tracing ON" if tracing else "⚪ LangSmith tracing off (add LANGSMITH_API_KEY)")
    st.caption(f"LLM: {', '.join(config.LLM_MODELS)}")
    if version == "v4":
        st.caption("V4 makes ~3 Gemini calls per question (rewrite, rerank, answer).")

st.title("🎓 AI Placement Preparation Assistant")
st.write("Ask about **technical interviews, aptitude, HR questions or resume writing**. "
         "Answers come **only from the uploaded PDFs**, with the source file and page. "
         "If the answer is not in them, the assistant says so.")

if not api_key:
    st.info("Enter your Gemini API key in the sidebar (or put it in the `.env` file) to start.")
    st.stop()


@st.cache_resource(show_spinner="Loading PDFs and building / loading the vector indexes...")
def get_engine(_key: str) -> PlacementRAG:
    rag = PlacementRAG()
    _ = rag.chunks                # fail early and clearly if PDFs are missing
    return rag


try:
    engine = get_engine(api_key)
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()
except Exception as err:  # noqa: BLE001
    st.error(f"Could not start the assistant: {type(err).__name__}: {err}")
    st.stop()

with st.sidebar:
    st.caption(f"📚 {len(engine.chunks)} chunks from {len(config.CATEGORY_BY_FILE)} PDFs")
    if st.button("🔄 Rebuild indexes"):
        with st.spinner("Re-embedding all chunks..."):
            try:
                engine.rebuild_indexes()
                st.success("Indexes rebuilt.")
            except LLMUnavailable as err:
                st.error(err.message)
            except Exception as err:  # noqa: BLE001
                st.error(f"Rebuild failed: {err}")
    if st.button("🧹 Clear chat"):
        st.session_state.history = []

st.session_state.setdefault("history", [])


def render(res: RAGResult, details: bool) -> None:
    if res.error:
        st.warning(res.error)
    st.markdown(res.answer)
    if res.sources:
        with st.expander(f"📎 Sources ({len(res.sources)})", expanded=True):
            for s in res.sources:
                score = ""
                if s.score is not None:
                    score = f" · rerank score {s.score:.0f}/10" if res.version == "v4" else f" · distance {s.score:.3f}"
                st.markdown(f"**{s.label}** · _{s.category}_{score}")
                st.caption(s.snippet + "…")
    cap = f"{res.version.upper()} · {res.latency_s:.2f}s" + (" · from cache" if res.cached else "")
    st.caption(cap)
    if details:
        with st.expander("🔍 Retrieval details"):
            if res.queries:
                st.write("**Queries used:**")
                for q in res.queries:
                    st.write(f"- {q}")
            if res.best_distance is not None:
                st.write(f"Best vector distance: `{res.best_distance:.3f}` "
                         f"(threshold {config.SCORE_THRESHOLD})")
            for n in res.notes:
                st.write(f"ℹ️ {n}")


# Example questions (one per category)
examples = ["What is the STAR method?", "What is a confusion matrix?",
            "What should a fresher resume include?", "How do I find average speed for equal distances?"]
cols = st.columns(len(examples))
clicked = None
for col, ex in zip(cols, examples):
    if col.button(ex, use_container_width=True):
        clicked = ex

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render(turn["result"], turn["details"])

question = st.chat_input("Ask a placement-preparation question...") or clicked
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            try:
                result = engine.ask(question, version=version,
                                    category=None if category == "All" else category)
            except LLMUnavailable as err:
                result = RAGResult(question=question, version=version, answer=err.message, error=err.kind)
            except Exception as err:  # noqa: BLE001 - never show a raw traceback to the student
                result = RAGResult(question=question, version=version,
                                   answer=f"Something went wrong: {type(err).__name__}: {err}",
                                   error="unexpected")
        render(result, show_details)
    st.session_state.history.append({"question": question, "result": result, "details": show_details})
