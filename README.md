# Zyro Dynamics HR Help Desk — RAG Challenge

This repository contains the complete implementation for the **Zyro Dynamics HR Help Desk — RAG Challenge** (NxtWave Masterclass). The objective of this project is to build a Retrieval-Augmented Generation (RAG) pipeline that answers employee HR questions using internal policy documents, wraps it in a Streamlit interface, and integrates LangSmith tracing.

---

## 📂 Project Structure

- `Starter_Notebook.ipynb` — The development notebook containing step-by-step implementation, testing, evaluation, and code to generate the Kaggle submission file.
- `Completed_Notebook.ipynb` — The reference notebook containing the completed RAG pipeline and non-blocking submission generator code.
- `app.py` — A production-ready Streamlit dashboard containing the single-file RAG pipeline, interactive chat, configuration controls, and LangSmith trace generation.
- `requirements.txt` — Python package dependencies required for this project.
- `*.pdf` — Internal policy documents (Company Profile, Employee Handbook, Leave Policy, WFH Policy, etc.) acting as the knowledge base (corpus).

---

## 🛠️ Technical Stack

- **Core Framework**: Python, LangChain, LCEL (LangChain Expression Language)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace
- **LLM Provider**: ChatGroq (`llama-3.3-70b-versatile` by default) / Google Gemini / OpenAI
- **User Interface**: Streamlit with custom CSS (glassmorphism/vibrant dark elements)
- **Observability**: LangSmith Tracing

---

## 🚀 Getting Started

### 1. Installation

Clone this repository or navigate to the workspace directory, then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys:

```ini
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=zyro-rag-challenge
```

### 3. Launching the Streamlit Chatbot

To start the interactive Streamlit chatbot application, run:

```bash
streamlit run app.py
```

This will launch a web dashboard on `http://localhost:8501` allowing you to:
- Build and save the FAISS vector index from the PDF policies.
- Ask HR-related questions.
- View retrieved text chunks and source file metadata.
- Automatically track and extract **LangSmith Trace URLs** directly from the UI.

---

## 📊 Generating the Submission File (`submission.csv`)

Cell 16 in the notebooks processes the encrypted evaluation questions and outputs `submission.csv` for Kaggle submission.

### Background Commit / "Save Version" Setup
When submitting on Kaggle or committing the notebook via **"Save Version"**, the environment runs headlessly (non-interactively). The interactive `input()` statements for **Streamlit App URL** and **LangSmith Trace URL** are designed to fall back gracefully to pre-defined variables without crashing. 

To configure these, follow the step-by-step guide below to get your links, and then assign them to the variables in Cell 16 of the notebook.

### 🌐 Step 1: Deploy to Streamlit Community Cloud and Get App URL
1. Push your project code (including `app.py`, `requirements.txt`, and the policy PDFs) to a repository on **GitHub**.
2. Visit **[Streamlit Community Cloud](https://share.streamlit.io/)** and sign in using your GitHub account.
3. Click on the **"New app"** button.
4. Select your repository, branch, and specify the file path as `app.py`.
5. Under **Advanced settings**, add your environment variables/secrets (like `GROQ_API_KEY` or `GOOGLE_API_KEY`).
6. Click **"Deploy"**.
7. Once your app is running live, copy the URL from the browser's address bar. It will look like this:
   `https://<your-app-name>.streamlit.app/`
8. Paste this URL into the `streamlit_link` variable in Cell 16.

### 🔍 Step 2: Get Your Public LangSmith Trace URL
1. Log in to your **[LangSmith Dashboard](https://smith.langchain.com/)**.
2. Go to **Projects** on the left menu and select your project (`zyro-rag-challenge`).
3. Click on any successful chain trace from your run history.
4. In the top-right corner of the trace panel, click the **"Share"** button.
5. Toggle the switch to **"Enable Public Link"**.
6. Copy the generated URL to your clipboard. It will look like:
   `https://smith.langchain.com/public/<unique-id>/r`
7. Paste this URL into the `langsmith_link` variable in Cell 16.

---

## 🛡️ Guardrail Features
The RAG agent is built with basic safety guardrails:
1. **Keyword Guardrail**: Rejects queries containing blocked phrases (e.g. asking for confidential salary details, personal data, etc.) and returns a safety warning.
2. **Length Guardrail**: Refuses questions under 3 words to encourage high-quality search queries.
3. **Out-of-Scope Fallback**: If the query is outside the scope of the loaded PDF documents, the chatbot outputs: `"I don't have that information in the HR policy documents."`
