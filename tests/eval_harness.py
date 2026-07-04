"""Evaluation harness: replay sample conversations and measure Recall@10.

Parses the sample conversation markdown files to extract:
- User messages (the persona's turns)
- Expected recommendations (assessment names from the final shortlist)

Then replays against the live API and computes Recall@10.
"""

from __future__ import annotations

import json
import re
import sys
import os
import asyncio
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


API_BASE = os.getenv("API_BASE", "http://localhost:8000")
SAMPLE_CONVERSATIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "GenAI_SampleConversations",
)


def parse_sample_conversation(filepath: str) -> dict:
    """Parse a sample conversation markdown file.

    Returns:
        Dict with 'user_messages', 'expected_names', 'expected_urls'.
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Extract user messages
    user_messages = []
    user_pattern = re.compile(r"\*\*User\*\*\s*\n\s*\n\s*>\s*(.+?)(?:\n\n|\Z)", re.DOTALL)
    for match in user_pattern.finditer(content):
        msg = match.group(1).strip()
        # Handle multi-line quotes
        msg = re.sub(r"\n>\s*", "\n", msg)
        user_messages.append(msg)

    # Extract expected assessment names and URLs from the LAST recommendation table
    # Find all tables in the content
    table_pattern = re.compile(
        r"\|\s*#\s*\|.*?\n\|[-\s|]+\n((?:\|.*\n?)+)",
        re.MULTILINE,
    )
    tables = list(table_pattern.finditer(content))

    expected_names = []
    expected_urls = []

    if tables:
        # Use the last table as the expected final shortlist
        last_table = tables[-1].group(0)
        # Parse rows
        row_pattern = re.compile(r"\|\s*\d+\s*\|\s*(.+?)\s*\|.*?\|\s*<?(https://[^>|\s]+)>?\s*\|")
        for row_match in row_pattern.finditer(last_table):
            name = row_match.group(1).strip()
            url = row_match.group(2).strip()
            expected_names.append(name)
            expected_urls.append(url)

    return {
        "user_messages": user_messages,
        "expected_names": expected_names,
        "expected_urls": expected_urls,
    }


def compute_recall_at_k(
    recommended_urls: list[str],
    expected_urls: list[str],
    k: int = 10,
) -> float:
    """Compute Recall@K.

    Recall@K = |relevant ∩ recommended[:K]| / |relevant|
    """
    if not expected_urls:
        return 1.0  # No expected items = trivially satisfied

    recommended_set = set(recommended_urls[:k])
    expected_set = set(expected_urls)

    hits = len(recommended_set & expected_set)
    return hits / len(expected_set)


async def replay_conversation(
    conversation: dict,
    client: httpx.AsyncClient,
    max_turns: int = 8,
) -> dict:
    """Replay a sample conversation against the live API.

    Uses only the first user message to start, then feeds each subsequent
    user message after receiving the agent's response.

    Returns:
        Dict with 'recommended_names', 'recommended_urls', 'turns', 'responses'.
    """
    messages = []
    responses = []
    final_recommendations = []

    for i, user_msg in enumerate(conversation["user_messages"]):
        if len(messages) >= max_turns:
            break

        # Add user message
        messages.append({"role": "user", "content": user_msg})

        # Call the API
        try:
            resp = await client.post(
                f"{API_BASE}/chat",
                json={"messages": messages},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  API error on turn {i+1}: {e}")
            break

        responses.append(data)

        # Add assistant response to history
        messages.append({"role": "assistant", "content": data.get("reply", "")})

        # Track recommendations
        recs = data.get("recommendations", [])
        if recs:
            final_recommendations = recs

        # Check if conversation ended
        if data.get("end_of_conversation", False):
            break

    return {
        "recommended_names": [r["name"] for r in final_recommendations],
        "recommended_urls": [r["url"] for r in final_recommendations],
        "turns": len(messages),
        "responses": responses,
    }


async def run_evaluation():
    """Run the full evaluation suite."""
    print("=" * 60)
    print("SHL Assessment Recommender — Evaluation Harness")
    print("=" * 60)

    # Check health
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{API_BASE}/health", timeout=120.0)
            resp.raise_for_status()
            print(f"\n✓ Health check passed: {resp.json()}")
        except Exception as e:
            print(f"\n✗ Health check failed: {e}")
            print("  Make sure the server is running: uvicorn app.main:app")
            return

    # Load sample conversations
    conv_dir = Path(SAMPLE_CONVERSATIONS_DIR)
    if not conv_dir.exists():
        print(f"\n✗ Sample conversations not found at {conv_dir}")
        return

    conv_files = sorted(conv_dir.glob("C*.md"))
    print(f"\nFound {len(conv_files)} sample conversations")

    # Replay each conversation
    recalls = []
    async with httpx.AsyncClient() as client:
        for filepath in conv_files:
            conv_name = filepath.stem
            print(f"\n--- {conv_name} ---")

            conversation = parse_sample_conversation(str(filepath))
            print(f"  User messages: {len(conversation['user_messages'])}")
            print(f"  Expected: {conversation['expected_names']}")

            result = await replay_conversation(conversation, client)
            print(f"  Turns used: {result['turns']}")
            print(f"  Recommended: {result['recommended_names']}")

            recall = compute_recall_at_k(
                result["recommended_urls"],
                conversation["expected_urls"],
                k=10,
            )
            recalls.append(recall)
            print(f"  Recall@10: {recall:.3f}")

    # Summary
    if recalls:
        mean_recall = sum(recalls) / len(recalls)
        print(f"\n{'=' * 60}")
        print(f"MEAN RECALL@10: {mean_recall:.3f}")
        print(f"Min: {min(recalls):.3f} | Max: {max(recalls):.3f}")
        print(f"Conversations evaluated: {len(recalls)}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
