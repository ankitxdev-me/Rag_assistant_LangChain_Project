"""
Zyro Dynamics HR Help Desk - Streamlit App
Single-file RAG + Agent + Guardrails + Evaluation + LangSmith integration
Following patterns from learned/streamlit_api_assistant.py using ChatGroq and FAISS (Python 3.14 compatible)
"""

import os
import shutil
import json
import traceback
from pathlib import Path
from datetime import datetime

import streamlit as st
from html import escape

import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# -------------------------
# LangChain + LangGraph imports
# -------------------------
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

# Optional LangSmith client
try:
    from langsmith import Client as LangSmithClient
except Exception:
    LangSmithClient = None

# -------------------------
# Config defaults
# -------------------------
DEFAULT_CONFIG = {
    "docs_path": "./",
    "db_path": "faiss_index_store",
    "llm_model": "llama-3.3-70b-versatile",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 800,
    "chunk_overlap": 150,
    "retrieval_k": 5,
}

# -------------------------
# UI styling (professional theme)
# -------------------------
THEME = """
<style>
:root{--bg:#f4f6f8; --card:#ffffff; --muted:#6b7280; --accent:#0f62fe; --text:#0f1724}
body, .stApp { background: var(--bg); color:var(--text); font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto; }
.carbon-card { background:var(--card); padding:14px; border-radius:10px; box-shadow: 0 6px 18px rgba(2,6,23,0.06); border:1px solid rgba(0,0,0,0.04); margin-bottom:12px; }
.card-title { font-size:18px; font-weight:600; margin-bottom:8px; }
.small-muted { color:var(--muted); font-size:13px; }
.resp-box { background: linear-gradient(180deg, rgba(0,0,0,0.01), rgba(0,0,0,0.00)); border-radius:8px; padding:12px; border:1px solid rgba(0,0,0,0.04); white-space:pre-wrap; font-family: ui-monospace, monospace; }
.trace-link { color: var(--accent); font-weight:600; text-decoration:none; }
</style>
"""
st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢", layout="wide")
st.markdown(THEME, unsafe_allow_html=True)

# -------------------------
# LangSmith helpers
# -------------------------
def get_langsmith_client(api_key=None):
    if LangSmithClient is None:
        return None
    key = api_key or os.getenv("LANGSMITH_API_KEY", "")
    if not key:
        return None
    try:
        try:
            client = LangSmithClient(api_key=key)
        except TypeError:
            client = LangSmithClient()
        return client
    except Exception:
        return None


def find_latest_run(client, project, filter_name_substr=None, limit=20):
    if client is None:
        return None
    try:
        runs = None
        if hasattr(client, "list_runs"):
            runs = list(client.list_runs(project_name=project, limit=limit))
        else:
            runs = []
        if not runs:
            return None
        if filter_name_substr:
            for r in runs:
                name = getattr(r, "name", None)
                if name and filter_name_substr in name:
                    run_id = getattr(r, "id", None)
                    url = f"https://smith.langchain.com/o/default/projects/p/{project}/runs/{run_id}" if run_id else None
                    return {"run_id": run_id, "url": url, "name": name}
        r = runs[0]
        run_id = getattr(r, "id", None)
        url = f"https://smith.langchain.com/o/default/projects/p/{project}/runs/{run_id}" if run_id else None
        name = getattr(r, "name", None)
        return {"run_id": run_id, "url": url, "name": name}
    except Exception:
        return None


# -------------------------
# Document loading & vector store
# -------------------------
@st.cache_resource
def load_documents(docs_path):
    loader = PyPDFDirectoryLoader(docs_path)
    documents = loader.load()
    for d in documents:
        d.metadata["source_file"] = os.path.basename(d.metadata.get("source", "unknown"))
    return documents


def build_vectorstore(docs_path, db_path, chunk_size, chunk_overlap, embedding_model):
    docs = load_documents(docs_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    for i, s in enumerate(splits):
        s.metadata["chunk_id"] = i
    
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.from_documents(splits, embeddings)
    vs.save_local(db_path)
    return vs, len(splits)


@st.cache_resource
def load_vectorstore(db_path, embedding_model):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return vs


# -------------------------
# RAG chain setup
# -------------------------
def format_context(docs):
    out = []
    for d in docs:
        snippet = d.page_content[:400].replace("\n", " ")
        src = d.metadata.get("source_file", "unknown")
        cid = d.metadata.get("chunk_id", "n/a")
        out.append(f"Source: {src} (chunk {cid})\n{snippet}")
    return "\n\n---\n\n".join(out)


def build_rag_chain(vectorstore, llm_model, retrieval_k, api_key):
    retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
    llm = ChatGroq(model=llm_model, temperature=0.1, max_tokens=512, api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Zyro Dynamics HR Help Desk assistant.
Use ONLY the provided context to answer employee questions about HR policies.
If the answer is not found say: "I don't have that information in the HR policy documents."
Cite sources: [chunk X from filename]"""),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    def retrieve_and_format(q):
        docs = retriever.invoke(q)
        return format_context(docs)

    rag_chain = (
        {
            "context": retrieve_and_format,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# -------------------------
# Tools & Agent
# -------------------------
@tool
def calculator_tool(expr: str) -> str:
    """
    Safely evaluate a basic arithmetic expression (integers/floats + +-*/ and parentheses).
    Examples: "2+2", "1000 - 234", "(10.5*3)/2"
    Returns the numeric result as a string or an error message on invalid input.
    """
    if expr is None:
        return "Error: empty expression"
    allowed = set("0123456789+-*/(). eE")
    if any(c not in allowed for c in expr):
        return "Invalid characters in expression. Only digits, + - * / ( ) . and spaces allowed."
    try:
        val = eval(expr, {"__builtins__": {}}, {})
        return str(val)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def doc_search_tool(query: str) -> str:
    """
    Retrieve top matching HR policy documentation chunks for the given query.
    Returns a formatted string with chunk id, source filename and preview.
    This tool relies on the 'retriever' being present in st.session_state (populated after index build).
    """
    if not query:
        return "No query provided."
    try:
        retriever = st.session_state.get("retriever")
        if retriever is None:
            return "Retriever not available (build the index first)."
        docs = retriever.invoke(query)[:3]
        out_lines = []
        for d in docs:
            src = d.metadata.get("source_file", "unknown")
            cid = d.metadata.get("chunk_id", "n/a")
            preview = d.page_content[:300].replace("\n", " ")
            out_lines.append(f"[chunk {cid}] {src}: {preview}")
        return "\n\n".join(out_lines) if out_lines else "No documents retrieved."
    except Exception as e:
        return f"doc_search_tool error: {e}"


def create_tool_agent(llm_model, tools, api_key):
    llm = ChatGroq(model=llm_model, temperature=0.1, api_key=api_key)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
You are an HR Help Desk assistant. Use doc_search_tool for looking up HR policies and calculator_tool for calculations (such as remaining leaves, leave encashment calculations, etc.).
Provide concise final answers and cite sources (filenames and chunks) when applicable.
"""
    )
    return agent


# -------------------------
# Guardrails (keyword blocking + length check)
# -------------------------
BLOCKED_KEYWORDS = {"salary details", "personal data", "confidential", "secret", "hack", "bypass"}

def apply_guardrails(q):
    ql = q.lower()
    for b in BLOCKED_KEYWORDS:
        if b in ql:
            return False, f"Blocked keyword detected: '{b}'"
    if len(ql.split()) < 3:
        return False, "Query too short. Please provide more context (at least 3 words)."
    return True, None


# -------------------------
# Evaluation: retrieval metrics
# -------------------------
def retrieval_metrics(retriever, query, ground_truth_files, k=5):
    docs = retriever.invoke(query)[:k]
    retrieved_files = [d.metadata.get("source_file") for d in docs]
    relevant = ground_truth_files
    recall = len(set(retrieved_files) & set(relevant)) / len(relevant) if relevant else None
    precision = len(set(retrieved_files) & set(relevant)) / max(1, len(retrieved_files)) if retrieved_files else None
    return {"recall@k": recall, "precision@k": precision, "retrieved": retrieved_files}


# -------------------------
# Display helpers
# -------------------------
def format_context_display(docs):
    out = []
    for d in docs:
        src = d.metadata.get("source_file", "unknown")
        cid = d.metadata.get("chunk_id", "n/a")
        preview = d.page_content[:400].replace("\n", " ")
        out.append(f"[chunk {cid}] {src}\n{preview}\n---")
    return "\n\n".join(out)


# ============================
# SIDEBAR
# ============================
st.sidebar.title("System Controls")

docs_path = st.sidebar.text_input("Docs folder", DEFAULT_CONFIG["docs_path"])
db_path = st.sidebar.text_input("Vector store path", DEFAULT_CONFIG["db_path"])
llm_model = st.sidebar.text_input("LLM model (Groq)", DEFAULT_CONFIG["llm_model"])
embedding_model = st.sidebar.text_input("Embedding model", DEFAULT_CONFIG["embedding_model"])
chunk_size = st.sidebar.number_input("Chunk size", value=DEFAULT_CONFIG["chunk_size"], step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", value=DEFAULT_CONFIG["chunk_overlap"], step=50)
retrieval_k = st.sidebar.number_input("Retriever k", value=DEFAULT_CONFIG["retrieval_k"], step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**API Keys**")
groq_key = st.sidebar.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""))

st.sidebar.markdown("---")
st.sidebar.markdown("**LangSmith (optional)**")
ls_key = st.sidebar.text_input("LangSmith API key", type="password", value=os.getenv("LANGSMITH_API_KEY", ""))
ls_project = st.sidebar.text_input("LangSmith project", value=os.getenv("LANGCHAIN_PROJECT", "zyro-rag-challenge"))
enable_langsmith = st.sidebar.checkbox("Enable LangSmith tracing", value=False)

# Persist LangSmith settings
if "langsmith_enabled" not in st.session_state:
    st.session_state.langsmith_enabled = False

if enable_langsmith and ls_key:
    st.session_state.langsmith_enabled = True
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = ls_key
    os.environ["LANGCHAIN_PROJECT"] = ls_project
else:
    st.session_state.langsmith_enabled = False

# Index control
st.sidebar.markdown("---")
st.sidebar.markdown("**Index Control**")
if "index_built" not in st.session_state:
    st.session_state.index_built = False

if st.sidebar.button("Initialize / Rebuild Index"):
    if not groq_key:
        st.sidebar.error("Please provide a Groq API key.")
    else:
        try:
            with st.spinner("Building vectorstore (this may take a moment)..."):
                vs, num_chunks = build_vectorstore(docs_path, db_path, chunk_size, chunk_overlap, embedding_model)
                st.session_state.vectorstore = vs
                st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(
                    vs, llm_model, retrieval_k, groq_key
                )
                st.session_state.index_built = True
                # Reinitialize agent
                st.session_state.agent = create_tool_agent(
                    llm_model, [calculator_tool, doc_search_tool], groq_key
                )
                st.sidebar.success(f"Index built with {num_chunks} chunks")
        except Exception as e:
            st.sidebar.error(f"Failed to build index: {e}")
            st.sidebar.exception(traceback.format_exc())

# Load index on startup if it exists
if not st.session_state.get("index_built", False) and os.path.exists(db_path) and groq_key:
    try:
        vs = load_vectorstore(db_path, embedding_model)
        st.session_state.vectorstore = vs
        st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(
            vs, llm_model, retrieval_k, groq_key
        )
        st.session_state.index_built = True
        st.session_state.agent = create_tool_agent(
            llm_model, [calculator_tool, doc_search_tool], groq_key
        )
        st.sidebar.success("Loaded existing FAISS index")
    except Exception as e:
        st.sidebar.warning("Could not load existing index automatically.")
        print("Load vectorstore error:", e)

# Agent control
st.sidebar.markdown("---")
st.sidebar.markdown("**Agent Control**")
if "agent_thread_id" not in st.session_state:
    st.session_state.agent_thread_id = "thread-" + os.urandom(6).hex()

if "agent" not in st.session_state and st.session_state.get("index_built", False) and groq_key:
    try:
        st.session_state.agent = create_tool_agent(
            llm_model, [calculator_tool, doc_search_tool], groq_key
        )
    except Exception as e:
        st.sidebar.warning("Agent creation failed.")
        print("Agent create error:", e)

if st.sidebar.button("Reset Agent Memory"):
    st.session_state.agent_thread_id = "thread-" + os.urandom(6).hex()
    st.sidebar.success("Agent thread reset")

# LangSmith session trace storage
if "langsmith_runs" not in st.session_state:
    st.session_state.langsmith_runs = []

# ============================
# MAIN HEADER
# ============================
st.title("🏢 Zyro Dynamics HR Help Desk")
st.markdown("Ask questions about HR policies (RAG), run agent tasks, inspect retrieved chunks, and view LangSmith traces.")

# ============================
# TABS
# ============================
tab_rag, tab_agent, tab_guard, tab_retrieval, tab_eval, tab_traces = st.tabs([
    "RAG Q&A", "Agent", "Guardrails", "Retrieved Chunks", "Evaluation", "LangSmith Traces"
])

# -------------------------
# Tab: RAG Q&A
# -------------------------
with tab_rag:
    st.subheader("Retrieval-Augmented Generation")
    q = st.text_input("Ask a question about HR policies", key="rag_input")
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Ask (RAG)"):
            if not st.session_state.get("index_built", False):
                st.error("Index not built. Click 'Initialize / Rebuild Index' in sidebar first.")
            else:
                ok, err = apply_guardrails(q)
                if not ok:
                    st.error(err)
                else:
                    with st.spinner("Running RAG..."):
                        try:
                            resp = st.session_state.rag_chain.invoke(q)
                            st.markdown("<div class='carbon-card'>", unsafe_allow_html=True)
                            st.markdown("<div class='card-title'>Answer</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='resp-box'>{escape(str(resp))}</div>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Show retrieved context
                            docs = st.session_state.retriever.invoke(q)
                            ctx = format_context_display(docs)
                            st.text_area("Retrieved (top k)", ctx, height=240)

                            # LangSmith trace capture
                            if enable_langsmith and ls_key:
                                client = get_langsmith_client(ls_key)
                                found = find_latest_run(client, ls_project, filter_name_substr=q[:40])
                                if found:
                                    st.session_state.langsmith_runs.append({
                                        "query": q,
                                        "url": found.get("url"),
                                        "run_id": found.get("run_id"),
                                        "ts": datetime.now().isoformat()
                                    })
                                    st.success("Trace recorded (LangSmith).")
                        except Exception as e:
                            st.error(f"RAG failed: {e}")
                            st.exception(traceback.format_exc())
    with col2:
        st.markdown("#### Quick actions")
        if st.button("Show sources (last query)"):
            try:
                if "retriever" in st.session_state:
                    docs = st.session_state.retriever.invoke(q)
                    st.write(format_context_display(docs))
                else:
                    st.info("No retriever available")
            except Exception as e:
                st.error("Retriever error")
                st.exception(traceback.format_exc())

# -------------------------
# Tab: Agent
# -------------------------
with tab_agent:
    st.subheader("Agent (tools + memory)")
    agent_input = st.text_input("Agent instruction", key="agent_input")
    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("Run Agent"):
            if not st.session_state.get("index_built", False):
                st.error("Index not built.")
            else:
                ok, err = apply_guardrails(agent_input)
                if not ok:
                    st.error(err)
                else:
                    with st.spinner("Running agent..."):
                        try:
                            agent = st.session_state.get("agent")
                            if agent is None:
                                st.error("Agent not available. Please ensure Groq API key is set and Index is built.")
                            else:
                                cfg = {"configurable": {"thread_id": st.session_state.get("agent_thread_id", "thread-default")}}
                                out = agent.invoke({"messages": [{"role": "user", "content": agent_input}]}, cfg)
                                final = ""
                                if isinstance(out, dict) and "messages" in out:
                                    last = out["messages"][-1]
                                    final = getattr(last, "content", last.get("content") if isinstance(last, dict) else str(last))
                                else:
                                    final = str(out)
                                st.markdown("<div class='carbon-card'>", unsafe_allow_html=True)
                                st.markdown("<div class='card-title'>Agent result</div>", unsafe_allow_html=True)
                                st.markdown(f"<div class='resp-box'>{escape(final)}</div>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                                # LangSmith trace capture
                                if enable_langsmith and ls_key:
                                    client = get_langsmith_client(ls_key)
                                    found = find_latest_run(client, ls_project, filter_name_substr=agent_input[:40])
                                    if found:
                                        st.session_state.langsmith_runs.append({
                                            "query": agent_input,
                                            "url": found.get("url"),
                                            "run_id": found.get("run_id"),
                                            "ts": datetime.now().isoformat()
                                        })
                                        st.success("Agent run recorded in session traces.")
                        except Exception as e:
                            st.error("Agent error")
                            st.exception(traceback.format_exc())
    with c2:
        if st.button("Reset Agent Thread"):
            st.session_state.agent_thread_id = "thread-" + os.urandom(6).hex()
            st.success("Agent thread id reset (memory cleared)")

# -------------------------
# Tab: Guardrails
# -------------------------
with tab_guard:
    st.subheader("Guardrails Test")
    gq = st.text_input("Enter query to test guardrails", key="guard_input")
    if st.button("Test Guardrails"):
        ok, err = apply_guardrails(gq)
        if not ok:
            st.error(err)
        else:
            st.success("Query passes guardrails.")
    st.markdown("**Blocked keywords:**")
    st.code(", ".join(sorted(BLOCKED_KEYWORDS)))

# -------------------------
# Tab: Retrieved Chunks viewer
# -------------------------
with tab_retrieval:
    st.subheader("Retrieved Chunks Viewer")
    rq = st.text_input("Query to inspect retrieved chunks", key="ret_q")
    if st.button("Inspect Chunks"):
        if not st.session_state.get("index_built", False):
            st.error("Index not built.")
        else:
            try:
                docs = st.session_state.retriever.invoke(rq)
                st.text_area("Top retrieved chunks", format_context_display(docs), height=420)
            except Exception as e:
                st.error("Retriever error")
                st.exception(traceback.format_exc())

# -------------------------
# Tab: Evaluation
# -------------------------
with tab_eval:
    st.subheader("Retrieval Evaluation")
    st.markdown("Provide a ground-truth mapping to run recall/precision tests.")
    gt_json = st.text_area(
        'Ground-truth mapping (JSON). Format: {"query": ["file1.pdf"]}',
        height=120,
        value='{"What is the leave policy?": ["02_Leave_Policy.pdf"], "What is the WFH policy?": ["03_Work_From_Home_Policy.pdf"]}'
    )
    if st.button("Run retrieval evaluation"):
        if not st.session_state.get("index_built", False):
            st.error("Index not built.")
        else:
            try:
                gt = json.loads(gt_json)
                for q_, files in gt.items():
                    metrics = retrieval_metrics(st.session_state.retriever, q_, files, k=retrieval_k)
                    st.markdown(f"**Q:** {q_}")
                    st.markdown(f"- Recall@{retrieval_k}: {metrics['recall@k']}")
                    st.markdown(f"- Precision@{retrieval_k}: {metrics['precision@k']}")
                    st.markdown(f"- Retrieved: {metrics['retrieved']}")
            except Exception as e:
                st.error("Evaluation failed")
                st.exception(traceback.format_exc())

# -------------------------
# Tab: LangSmith Traces
# -------------------------
with tab_traces:
    st.subheader("LangSmith Trace Dashboard (session)")
    st.markdown("Traces captured during this Streamlit session (enable LangSmith in sidebar).")
    if len(st.session_state.langsmith_runs) == 0:
        st.info("No traces captured yet. Run RAG queries with LangSmith enabled to create traces.")
    else:
        for run in reversed(st.session_state.langsmith_runs[-20:]):
            qdisplay = run.get("query", "")[:160]
            ts = run.get("ts", "")
            url = run.get("url", "")
            st.markdown(f"**{escape(qdisplay)}**  \u00b7  <span style='color:var(--muted);font-size:12px'>{ts}</span>", unsafe_allow_html=True)
            if url:
                st.markdown(f"[Open Trace ↗]({url})")
            st.markdown("---")
    project = ls_project or os.getenv("LANGCHAIN_PROJECT", "zyro-rag-challenge")
    st.markdown(f"[Open LangSmith Project ↗](https://smith.langchain.com/o/default/projects/p/{project})")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Zyro Dynamics HR Help Desk — RAG + Agent + Guardrails + LangSmith — built with LangChain + Groq")
