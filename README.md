---
title: SHL Assessment Recommender
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
license: mit
short_description: Conversational AI agent for SHL assessment recommendations
---

# SHL Conversational Assessment Recommender

A conversational AI agent that helps hiring managers find the right SHL assessments through dialogue. The agent clarifies vague requests, recommends assessments grounded in the catalog, supports refinement and comparison, and refuses off-topic queries.

**Live API:**
- **Base:** [https://hey-nehaa-shl-assessment-recommender.hf.space](https://hey-nehaa-shl-assessment-recommender.hf.space)
- **Health:** [https://hey-nehaa-shl-assessment-recommender.hf.space/health](https://hey-nehaa-shl-assessment-recommender.hf.space/health)
- **Chat:** POST [https://hey-nehaa-shl-assessment-recommender.hf.space/chat](https://hey-nehaa-shl-assessment-recommender.hf.space/chat)

## Architecture

```
POST /chat → Parse History → Build Search Query → Hybrid Retrieval (FAISS + Keywords)
    → Route Model (Light/Powerful) → Groq LLM → Parse JSON → Validate Against Catalog → Response
```

| Component | Technology | Purpose |
|-----------|-----------|---------|
| API | FastAPI | Stateless REST endpoints |
| LLM | Groq (Llama 3.3 70B / Llama 3.1 8B) | Dual-model routing |
| Embeddings | all-MiniLM-L6-v2 | Semantic search |
| Vector Store | FAISS (IndexFlatIP) | Cosine similarity search |
| Retrieval | Hybrid | Semantic + keyword boosting |

### Model Routing

| Scenario | Model | Reason |
|----------|-------|--------|
| Early vague query | `llama-3.1-8b-instant` | Fast clarification |
| Recommendations / comparisons / refinements | `llama-3.3-70b-versatile` | Quality reasoning |
| Near turn budget (≥5 turns) | `llama-3.3-70b-versatile` | Must deliver results |
| Long input (JD, 150+ chars) | `llama-3.3-70b-versatile` | Complex context |

## Setup

### Prerequisites
- Python 3.11+
- [Groq API key](https://console.groq.com/keys) (free)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### Run Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
# Unit tests (no API key needed)
pytest tests/test_core.py -v

# Evaluation harness (requires running server)
python tests/eval_harness.py
```

## API Endpoints

### `GET /health`
Returns `{"status": "ok"}` with HTTP 200.

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need assessments for a senior Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, 4 years experience. Add personality tests."}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are assessments for a mid-level Java developer with personality evaluation.",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/", "test_type": "K"},
    {"name": "Spring (New)", "url": "https://www.shl.com/products/product-catalog/view/spring-new/", "test_type": "K"},
    {"name": "OPQ Universal Competency Report 2.0", "url": "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` when clarifying or refusing
- `recommendations` contains 1-10 items when recommending
- `end_of_conversation` is `true` only when the user confirms the shortlist

## Deployment

### Hugging Face Spaces (Primary)

Live at: [hey-nehaa/shl-assessment-recommender](https://huggingface.co/spaces/hey-nehaa/shl-assessment-recommender)

1. Create a Docker Space on HF
2. Add secret: `GROQ_API_KEY`
3. Push code — auto-builds and deploys

> **Note:** Free-tier HF Spaces spin down after inactivity. The first `/health` call allows up to 2 minutes for the service to wake up.

### Docker

```bash
docker build -t shl-recommender .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key shl-recommender
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key |
| `MODEL_POWERFUL` | No | `llama-3.3-70b-versatile` | Model for recommendations |
| `MODEL_LIGHT` | No | `llama-3.1-8b-instant` | Model for clarification |
| `PORT` | No | `7860` | Server port |

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── agent.py          # Orchestration: retrieval → LLM → validation
│   ├── catalog.py         # Catalog loading, parsing, indexing
│   ├── config.py          # Environment-based configuration
│   ├── conversation.py    # Conversation history utilities
│   ├── main.py            # FastAPI application and endpoints
│   ├── models.py          # Pydantic request/response schemas
│   ├── prompts.py         # System prompt and retrieval templates
│   └── retrieval.py       # FAISS + keyword hybrid search engine
├── data/raw/
│   └── shl_product_catalog.json
├── tests/
│   ├── test_core.py       # Unit tests (18 tests)
│   └── eval_harness.py    # Sample conversation replay + Recall@10
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Design Decisions

1. **Retrieval-grounded responses** — The LLM only sees retrieved assessments (top 30), not the full catalog. This prevents hallucination and keeps token usage low.
2. **Hybrid retrieval** — FAISS semantic search finds conceptually relevant assessments. Keyword matching catches exact technology names (Java, SQL, OPQ). Combined scoring outperforms either alone.
3. **Post-generation validation** — Every recommendation URL and name is validated against the catalog index. Hallucinated entries are fuzzy-matched or silently dropped.
4. **Turn budget awareness** — The agent knows remaining turns and escalates to recommendations when budget is low.
5. **Dual-model routing** — Cheap model for simple clarification, powerful model for complex reasoning. Automatic fallback if a model is unavailable.
