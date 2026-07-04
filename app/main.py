"""FastAPI application for the SHL Assessment Recommender.

Exposes two endpoints:
- GET /health — readiness check
- POST /chat — stateless conversation endpoint
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent import SHLAgent
from app.catalog import load_catalog
from app.config import HOST, PORT
from app.models import ChatRequest, ChatResponse, HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global agent instance
agent: SHLAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the agent and retrieval engine at startup."""
    global agent
    logger.info("Starting SHL Assessment Recommender...")
    start = time.time()

    # Load catalog
    assessments = load_catalog()
    logger.info(f"Loaded {len(assessments)} assessments from catalog")

    # Initialize agent (includes embedding generation and FAISS indexing)
    agent = SHLAgent(assessments)
    elapsed = time.time() - start
    logger.info(f"Agent ready in {elapsed:.1f}s")

    yield

    logger.info("Shutting down SHL Assessment Recommender")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI agent for recommending SHL assessments",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "SHL Assessment Recommender",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Readiness check endpoint."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return agent response.

    The API is stateless — every request carries the full conversation history.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")

    # Convert Pydantic models to dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Process through the agent
    response = await agent.process_chat(messages)

    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
