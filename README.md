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

# 🎯 SHL Conversational Assessment Recommender

An intelligent, production-ready conversational AI agent that assists hiring managers in discovering and shortlisting the right SHL assessments from the product catalog. Built on top of Google's **Gemini API** and a custom hybrid semantic search engine, the recommender dynamically refines selections, clarifies vague requirements, compares assessments, and structures outputs to align directly with official catalog specs.

---

## 🌐 Live API & Links

*   **API Base URL:** [https://hey-neha-shl-assessment-recommender.hf.space](https://hey-neha-shl-assessment-recommender.hf.space)
*   **Health Status:** [/health](https://hey-neha-shl-assessment-recommender.hf.space/health)
*   **Chat Endpoint:** POST [/chat](https://hey-neha-shl-assessment-recommender.hf.space/chat)
*   **GitHub Repository:** [hey-nehaa/shl-assessment-recommender](https://github.com/hey-nehaa/shl-assessment-recommender)

---

## 🏗️ System Architecture

```
                                  [ USER CONVERSATION ]
                                            │
                                            ▼
                                     [ POST /chat ]
                                            │
                                            ▼
                                    Parse Chat History
                                            │
                                            ▼
                                  Build Vector Query
                                            │
                                            ▼
                           Hybrid Retrieval (FAISS + Keywords)
                                            │
                                            ▼
                              Model Routing (Flash vs Pro)
                                            │
                                            ▼
                                     Gemini API Call
                                            │
                                            ▼
                                     Parse JSON Schema
                                            │
                                            ▼
                                Validate & Match Catalog
                                            │
                                            ▼
                                   [ API RESPONSE ]
```

### Tech Stack Overview

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **API Framework** | FastAPI | High-performance, stateless REST endpoints |
| **LLM Engine** | Gemini (`gemini-3.1-flash-live-preview`) | High-performance reasoning on a generous free-tier |
| **Embeddings** | `all-MiniLM-L6-v2` | Dense vector generation for semantic query matching |
| **Vector Store** | FAISS (`IndexFlatIP`) | High-speed, local cosine similarity search |
| **Retrieval** | Hybrid (Semantic + TF-IDF) | Combining contextual intent with direct keyword boosting |

### Intelligent Model Routing

To maximize responsiveness while keeping API costs low, requests are routed based on turn characteristics (both default to `gemini-3.1-flash-live-preview` for rate-limit stability, but can be customized to any model including `gemini-3.5-flash` or `gemini-3.1-pro`):

| Scenario | Model Routed | Rationale |
| :--- | :--- | :--- |
| **Early conversation / vague query** | `gemini-3.1-flash-live-preview` | Ultra-fast responses to clarify customer needs |
| **Shortlisting / complex refinements** | `gemini-3.1-flash-live-preview` | Deep logical reasoning to select precise metrics |
| **Near turn budget limit (Turn ≥ 5)** | `gemini-3.1-flash-live-preview` | High precision to finalize recommendations in time |
| **Detailed job description / long input** | `gemini-3.1-flash-live-preview` | Large context processing to match job competencies |

---

## 🛠️ Getting Started

### Prerequisites

*   Python 3.11+
*   **Gemini API Key** (obtainable from [Google AI Studio](https://aistudio.google.com/))

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hey-nehaa/shl-assessment-recommender.git
    cd shl-assessment-recommender
    ```

2.  **Set up virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment:**
    ```bash
    cp .env.example .env
    # Edit the .env file and set your GEMINI_API_KEY
    ```

### Run Locally

Start the FastAPI application development server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run Tests

```bash
# Run unit test suite (offline, no API key required)
pytest tests/test_core.py -v

# Run the conversation evaluation harness (requires local server running)
python tests/eval_harness.py
```

---

## 🔌 API Endpoints Reference

### 1. Readiness Check (`GET /health`)
Verifies that the service is running and vector indexes are loaded.

*   **Response:**
    ```json
    {
      "status": "ok"
    }
    ```

### 2. Conversation Interface (`POST /chat`)
Process a message within a stateless conversation turn.

*   **Request Body:**
    ```json
    {
      "messages": [
        {"role": "user", "content": "I need assessments for a senior Java developer"},
        {"role": "assistant", "content": "What seniority level and team requirements do you have?"},
        {"role": "user", "content": "Mid-level, 4 years experience. Also include personality tests."}
      ]
    }
    ```

*   **Response Body:**
    ```json
    {
      "reply": "Here are my recommended assessments for a mid-level Java developer, focusing on core programming competencies and workplace behavior.",
      "recommendations": [
        {
          "name": "Core Java (Advanced Level) (New)",
          "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
          "test_type": "K"
        },
        {
          "name": "Spring (New)",
          "url": "https://www.shl.com/products/product-catalog/view/spring-new/",
          "test_type": "K"
        },
        {
          "name": "OPQ Universal Competency Report 2.0",
          "url": "https://www.shl.com/products/product-catalog/view/opq-universal-competency-report-2-0/",
          "test_type": "P"
        }
      ],
      "end_of_conversation": false
    }
    ```

> [NOTE]
> - `recommendations` is empty (`[]`) when the agent is clarifying parameters or handling off-topic requests.
> - `end_of_conversation` becomes `true` when the user has confirmed they are satisfied with the final shortlisted recommendations.

---

## ⚙️ Environment Variables

Customize application behavior via the following environment variables in `.env`:

| Key | Required | Default | Description |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | **Yes** | — | Google Gemini developer API key |
| `MODEL_POWERFUL` | No | `gemini-3.1-flash-live-preview` | Model for recommendations, reasoning, and finalizations |
| `MODEL_LIGHT` | No | `gemini-3.1-flash-live-preview` | Model for intent classification and early clarifications |
| `PORT` | No | `7860` | Server binding port |
| `HOST` | No | `0.0.0.0` | Server binding address |
| `TOP_K_RETRIEVAL` | No | `30` | Number of context documents retrieved for the model |
| `TOP_K_RECOMMEND` | No | `10` | Maximum recommendations allowed in the API payload |

---

## 📁 Directory Structure

```
├── app/
│   ├── __init__.py
│   ├── agent.py          # Core coordinator (context builder, routing, API client, fallback)
│   ├── catalog.py         # Catalog loading, indexing, canonicalization, and validation
│   ├── config.py          # Environment settings and defaults
│   ├── conversation.py    # Chat length, turn counts, and history managers
│   ├── main.py            # FastAPI service entry point, lifespans, CORS and routes
│   ├── models.py          # Pydantic schema validation structures
│   ├── prompts.py         # System prompt instructions and context formats
│   └── retrieval.py       # FAISS indexing + keyword boosting search engine
├── data/
│   └── raw/
│       └── shl_product_catalog.json  # Reference product catalog
├── tests/
│   ├── test_core.py       # pytest unit test cases (18 items)
│   └── eval_harness.py    # Conversation replay simulation & Recall@10 evaluator
├── Dockerfile             # Multi-stage Docker config
├── render.yaml            # Render Blueprint deployment specification
├── requirements.txt       # Project python dependencies
├── .env.example           # Config template
└── README.md              # Documentation
```

---

## 💡 Key Design Decisions

1.  **Strict Grounding via Retrieval:** The LLM does not generate recommendation items from memory. Instead, it is fed the top 30 matched catalog entities. This ensures that recommended assessments exist in the real SHL portfolio, preventing hallucination.
2.  **Post-Generation Validation:** Recommendation names and links returned by the model are parsed and validated against the catalog indexes. Any hallucinated items are resolved using substring and keyword matching or silently discarded.
3.  **Hybrid Retrieval Engine:** Combines FAISS semantic embeddings (using `all-MiniLM-L6-v2`) with custom keyword boosting (checking exact tags like "Java", "SQL", "OPQ"). This delivers highly relevant grounding context to the LLM.
4.  **Turn-Aware Budgeting:** The system prompt injects turn-budget context (`[Turns used: X/8]`). As turns approach 8, the prompt automatically shifts intent to force shortlisting rather than continuing to clarify.
5.  **Graceful Key Failures:** If initialized without `GEMINI_API_KEY`, the agent enters a warning state rather than crashing the FastAPI lifespan, allowing `/health` checks and vector indexing to load successfully in deployment builders.
