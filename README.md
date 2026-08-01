<div align="center">

# 🧠 Adaptive RAG — Agentic AI Chatbot

**Intelligent Retrieval-Augmented Generation with adaptive query routing, vector search, and web-grounded answers**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-orange.svg)](https://www.langchain.com/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-339933.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B.svg)](https://streamlit.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Latest-47A248.svg)](https://www.mongodb.com/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)](.github/workflows/ci.yml)

</div>

---

## ✨ What is this?

**Adaptive RAG** is an end-to-end, production-grade **agentic RAG (Retrieval-Augmented Generation)** system that doesn't just retrieve — it *thinks* about **how** to answer.

Every user query is routed through a **LangGraph orchestration pipeline** that classifies the question and chooses the smartest path:

| Route | When? | What happens |
|-------|-------|--------------|
| 📚 **Index** | Answer exists in your uploaded documents | Vector search → relevance grading → answer generation → **faithfulness verification** |
| 🧠 **General** | Everyday knowledge / casual chat | Direct LLM response |
| 🌐 **Search** | Real-time / niche info | Live web search via Tavily → answer generation |

The result? **Fast, accurate, grounded answers** — with less hallucination, thanks to automatic grading, query rewriting, and answer verification built right into the graph.

---

## 🚀 Key Features

### 🤖 Agentic Query Routing
- **Adaptive classification** into 3 paths — no more "everything goes to the vector store"
- **ReAct agent** handles document retrieval with reasoning steps
- **Query rewriting** when initial retrieval isn't relevant

### 🛡️ Anti-Hallucination Pipeline
- **Relevance grading** — retrieved docs are scored before use
- **Answer verification** — a fact-checker compares the answer against context; ungrounded answers are **regenerated up to 2×**
- **Source citations** returned with every answer

### 📂 Document Intelligence
- Upload **PDF / TXT** — auto chunked (1000 chars, 150 overlap)
- **FAISS** vector store with disk persistence across restarts
- Semantic embeddings via **OpenAI**

### 💾 State & Memory
- **MongoDB** chat history (Motor async driver) — full conversation context
- JWT-based **authentication** (register / login)
- Session-scoped conversations

### ⚡ Production-Ready API
- **FastAPI** + async endpoints
- **Rate limiting** on all query/upload endpoints
- **SSE streaming** endpoint for smooth chat UX
- `/health`, `/rag/stats` monitoring endpoints

### 🎨 Streamlit Frontend
- Login / register + chat interface
- Sidebar document upload with descriptions
- Route badges, response times, and source expanders

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           Streamlit Frontend            │
                    │        (auth · chat · uploads)          │
                    └──────────────────┬──────────────────────┘
                                       │ REST / SSE
                    ┌──────────────────▼──────────────────────┐
                    │            FastAPI Backend              │
                    │     JWT auth · rate limits · stats      │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │         LangGraph State Graph           │
                    │                                         │
                    │   ┌────────────┐                        │
                    │   │ query_analy│ssis │                    │
                    │   └─────┬──────┘                        │
                    │   ┌─────┴─────┬───────┬──────────┐      │
                    │   ▼           ▼       ▼          ▼      │
                    │ retriever  general_llm web_search ... │
                    │   │                     │             │
                    │   ▼                     │             │
                    │  grade ◄── yes ── generate ──► verify │
                    │   │  no                    ▲  faithful│
                    │   ▼                        │          │
                    │ rewrite ────► retriever    └─ regenerate
                    └───────────────────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────┐
                    │   FAISS (embeddings) · MongoDB (history)│
                    └─────────────────────────────────────────┘
```

### Graph Flow
1. **query_classifier** — classifies into `index` / `general` / `search`
2. **retriever** — ReAct agent queries the FAISS vector store
3. **grade** — scores relevance (`yes` / `no`)
4. **rewrite** — rewrites query & re-retrieves if graded `no`
5. **generate** — produces a readable answer from context
6. **verify_answer** — fact-checks; regenerates if ungrounded (max 2×)

---

## 📁 Project Structure

```
Adaptive-Rag/
├── src/                        # Backend source
│   ├── main.py                 # FastAPI app entry point
│   ├── api/
│   │   ├── routes.py           # RAG / query / upload / stats endpoints
│   │   └── auth.py             # JWT auth (register, login, verify)
│   ├── config/
│   │   ├── settings.py         # YAML config loader
│   │   └── prompts.yaml        # LLM system prompts
│   ├── core/
│   │   ├── config.py           # Environment settings
│   │   └── logger.py           # Logging setup
│   ├── db/
│   │   └── mongo_client.py     # Motor async MongoDB client
│   ├── llms/
│   │   └── openai.py           # OpenAI GPT-4o LLM
│   ├── memory/
│   │   ├── chat_history_mongo.py    # MongoDB chat history
│   │   └── chathistory_in_memory.py # In-memory fallback
│   ├── models/                 # Pydantic schemas & graph State
│   ├── rag/
│   │   ├── graph_builder.py    # LangGraph workflow construction
│   │   ├── nodes.py            # Graph node implementations
│   │   ├── retriever_setup.py  # FAISS vector store + persistence
│   │   ├── document_upload.py  # PDF/TXT processing & chunking
│   │   └── reAct_agent.py      # ReAct agent factory
│   └── tools/
│       ├── common_tools.py     # Shared utilities
│       └── graph_tools.py      # Routing / grading / verification logic
├── streamlit_app/              # Streamlit frontend
│   ├── home.py                 # Login / register
│   ├── pages/chat.py           # Chat + document upload
│   └── utils/api_client.py     # Backend API client
├── tests/                      # Pytest suite (29 tests)
├── .github/workflows/ci.yml    # CI pipeline (lint + test)
├── .env.example                # Environment template
├── docker-compose.yml          # MongoDB + API + Streamlit
└── Dockerfile
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| 🧠 **LLM Framework** | LangChain 1.x, LangGraph 1.x |
| 🚀 **Backend** | FastAPI + Uvicorn (async) |
| 🗄️ **Vector Store** | FAISS (local, disk-persisted) — Qdrant optional |
| 💬 **Chat DB** | MongoDB (Motor async driver) |
| 🔐 **Auth** | JWT (HMAC-SHA256), password hashing |
| 🖥️ **Frontend** | Streamlit |
| 🔎 **Web Search** | Tavily |
| 🧾 **Models** | OpenAI GPT-4o |
| ⚙️ **Quality** | Pytest · Ruff · MyPy · GitHub Actions CI |
| 🐳 **Deployment** | Docker + Docker Compose |

---

## 📦 Installation

### Prerequisites
- Python **3.11+**
- MongoDB running locally (`mongodb://localhost:27017`) — or via Docker
- **OpenAI API key** & **Tavily API key**

### 1. Clone & install

```bash
git clone https://github.com/your-username/Adaptive-Rag.git
cd Adaptive-Rag

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Then fill in `.env`:

```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key
TAVILY_API_KEY=tvly-your-tavily-api-key

# MongoDB
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=adaptive_rag

# Optional
# JWT_SECRET=your-secret-at-least-32-chars
# FAISS_INDEX_DIR=faiss_index
```

---

## ▶️ Running the App

### Option A — Local

```bash
# Terminal 1 — Backend
uvicorn src.main:app --reload --port 8000

# Terminal 2 — Frontend
streamlit run streamlit_app/home.py
```

Then open:
- 🖥️ **Chat app** → http://localhost:8501
- 📚 **API docs (Swagger)** → http://localhost:8000/docs

### Option B — Docker Compose

```bash
docker-compose up --build
```

Starts MongoDB, the FastAPI backend, and Streamlit together.

---

## 🔌 API Reference

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `POST` | `/auth/register` | Create account | — |
| `POST` | `/auth/login` | Login → JWT token | — |
| `POST` | `/rag/query` | Ask the RAG system | 15/min |
| `POST` | `/rag/query/stream` | Streamed answer (SSE) | 10/min |
| `POST` | `/rag/documents/upload` | Upload PDF/TXT (header `X-Description`) | 5/min |
| `GET` | `/rag/documents/count` | Chunk count in vector store | — |
| `DELETE` | `/rag/documents` | Clear all documents | — |
| `GET` | `/rag/stats` | System statistics | — |
| `GET` | `/health` | Health check | — |

### Example query

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What topics does the uploaded document cover?", "session_id": "user_123"}'
```

**Response:**

```json
{
  "result": {
    "type": "ai",
    "content": "The document covers LangGraph workflows, state graphs, and tool calling..."
  },
  "route": "index",
  "time_seconds": 2.34,
  "source_documents": [{ "source": "guide.pdf", "page": 3 }]
}
```

---

## 🧪 Testing

```bash
# Run the full test suite
pytest tests/ -v

# Lint & type-check
ruff check src/ streamlit_app/ tests/
mypy src/ --ignore-missing-imports
```

> **29 tests** cover models, auth (JWT hashing/verification), graph routing logic, and API endpoints. Tests that need live OpenAI/MongoDB are marked to skip in CI without credentials.

CI runs automatically on every push via **GitHub Actions** (`.github/workflows/ci.yml`).

---

## 🗺️ Roadmap

- [x] Core agentic RAG pipeline (routing, grading, verification)
- [x] FAISS persistence + document upload
- [x] JWT auth & rate limiting
- [x] CI/CD (GitHub Actions)
- [ ] Per-user document isolation
- [ ] Real token-by-token streaming
- [ ] React/Next.js frontend
- [ ] Multi-LLM provider support

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes
4. Push & open a Pull Request

Please keep code styled with **Ruff** and add tests for new features.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

**Made with ❤️ by [Akshat Gupta](mailto:akshat.gupta13@outlook.com)**

⭐ Star this repo if you find it useful!

</div>
