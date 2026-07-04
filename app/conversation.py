"""Conversation utilities for the SHL Assessment Recommender.

Provides helpers for analyzing conversation history: counting turns,
extracting user messages, and detecting prior recommendations.
"""

from __future__ import annotations


def count_turns(messages: list[dict]) -> int:
    """Count the total number of messages in the conversation."""
    return len(messages)


def get_last_user_message(messages: list[dict]) -> str:
    """Get the last user message from the conversation history."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""
