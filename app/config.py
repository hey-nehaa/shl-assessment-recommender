"""Configuration for the SHL Assessment Recommender."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Powerful model for recommendations, comparisons, complex refinements
MODEL_POWERFUL = os.getenv("MODEL_POWERFUL", "llama-3.3-70b-versatile")
# Lightweight model for clarification, intent extraction, simple routing
MODEL_LIGHT = os.getenv("MODEL_LIGHT", "llama-3.1-8b-instant")

# --- Retrieval ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "30"))
TOP_K_RECOMMEND = int(os.getenv("TOP_K_RECOMMEND", "10"))

# --- Catalog ---
CATALOG_PATH = os.getenv(
    "CATALOG_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "shl_product_catalog.json"),
)

# --- Server ---
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860"))
