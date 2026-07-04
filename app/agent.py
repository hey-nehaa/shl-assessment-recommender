"""Core agent logic: orchestrates retrieval, LLM calls, and response validation.

This is the heart of the system. It:
1. Extracts intent from conversation history
2. Retrieves relevant assessments
3. Routes to the appropriate Groq model (light vs powerful)
4. Calls the LLM with grounding context
5. Validates and fixes the response schema
6. Ensures all recommendations are from the catalog
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from openai import OpenAI

from app.catalog import Assessment, build_name_index, build_url_index
from app.config import (
    GROQ_API_KEY,
    GROQ_BASE_URL,
    MODEL_LIGHT,
    MODEL_POWERFUL,
    TOP_K_RECOMMEND,
    TOP_K_RETRIEVAL,
)
from app.conversation import count_turns
from app.models import ChatResponse, Recommendation
from app.prompts import (
    RETRIEVAL_CONTEXT_TEMPLATE,
    SYSTEM_PROMPT,
    format_retrieved_assessments,
)
from app.retrieval import RetrievalEngine

logger = logging.getLogger(__name__)

# Keywords that signal a powerful model is needed
_POWERFUL_SIGNALS = {
    "compare", "comparison", "difference", "versus", "vs",
    "recommend", "shortlist", "battery", "confirmed", "finalize",
    "perfect", "that works", "locking", "keep", "drop",
    "add personality", "add cognitive", "remove", "replace", "update", "actually",
    "job description", "jd", "here is the", "here's the",
}


def _needs_powerful_model(messages: list[dict], turn_count: int) -> bool:
    """Decide whether to route to the powerful model.

    Uses powerful model for:
    - Recommendation generation (turn >= 2 or detailed first message)
    - Refinement requests (user changing constraints)
    - Comparison requests
    - Near-budget conversations (turn >= 5)
    - Detailed first-turn queries (job descriptions, multiple requirements)

    Uses light model for:
    - Simple clarification questions (early turns, vague input)
    """
    # Always use powerful model when budget is tight
    if turn_count >= 5:
        return True

    last_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_msg = msg.get("content", "").lower()
            break

    # Check for powerful signals in the last user message
    for signal in _POWERFUL_SIGNALS:
        if signal in last_msg:
            return True

    # Long messages (likely JD or detailed requirements) → powerful
    if len(last_msg) > 150:
        return True

    # If we've had at least one exchange, use powerful for the response
    if turn_count >= 3:
        return True

    # Default: light model for early clarification
    return False


class SHLAgent:
    """The SHL Assessment Recommender agent."""

    def __init__(self, assessments: list[Assessment]):
        self.assessments = assessments
        self.url_index = build_url_index(assessments)
        self.name_index = build_name_index(assessments)
        self.retrieval = RetrievalEngine(assessments)

        # Initialize Groq client (OpenAI-compatible)
        self.client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
        )

        logger.info(
            f"Agent initialized with {len(assessments)} assessments | "
            f"Powerful: {MODEL_POWERFUL} | Light: {MODEL_LIGHT}"
        )

    async def process_chat(self, messages: list[dict]) -> ChatResponse:
        """Process a chat request and return a response.

        Args:
            messages: Full conversation history as list of {role, content} dicts.

        Returns:
            ChatResponse with reply, recommendations, and end_of_conversation flag.
        """
        try:
            turn_count = count_turns(messages)

            # 2. Build a comprehensive search query from conversation context
            search_query = self._build_search_query(messages)

            # 3. Retrieve relevant assessments
            retrieved = self.retrieval.search(search_query, top_k=TOP_K_RETRIEVAL)

            # 4. Format retrieved assessments for the prompt
            retrieved_context = format_retrieved_assessments(retrieved)
            retrieval_prompt = RETRIEVAL_CONTEXT_TEMPLATE.format(
                retrieved_assessments=retrieved_context
            )

            # 5. Route to appropriate model
            use_powerful = _needs_powerful_model(messages, turn_count)
            model = MODEL_POWERFUL if use_powerful else MODEL_LIGHT

            # 6. Build the conversation for the LLM
            llm_messages = self._build_llm_messages(messages, retrieval_prompt, turn_count)

            # 7. Call the LLM
            raw_response = await self._call_llm(llm_messages, model)

            # 8. Parse and validate the response
            response = self._parse_and_validate(raw_response)

            logger.info(
                f"Turn {turn_count} | Model: {model.split('/')[-1]} | "
                f"Recs: {len(response.recommendations)} | EOC: {response.end_of_conversation}"
            )

            return response

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            return ChatResponse(
                reply="I apologize, but I encountered an issue processing your request. Could you please rephrase your question about SHL assessments?",
                recommendations=[],
                end_of_conversation=False,
            )

    def _build_search_query(self, messages: list[dict]) -> str:
        """Build a comprehensive search query from the full conversation."""
        user_parts = []
        for msg in messages:
            if msg.get("role") == "user":
                user_parts.append(msg.get("content", ""))
        return " ".join(user_parts)

    def _build_llm_messages(
        self, messages: list[dict], retrieval_context: str, turn_count: int
    ) -> list[dict]:
        """Build the OpenAI-format message array for the LLM call."""
        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            llm_messages.append({"role": role, "content": content})

        # Append retrieval context and turn budget to the last user message
        turn_budget = 8 - turn_count
        budget_note = f"\n\n[Turns used: {turn_count}/{8}. "
        if turn_budget <= 2:
            budget_note += "MUST recommend NOW.]"
        elif turn_budget <= 4:
            budget_note += "Recommend soon if enough context.]"
        else:
            budget_note += "Clarify if needed.]"

        if llm_messages and llm_messages[-1]["role"] == "user":
            llm_messages[-1]["content"] += f"\n\n{retrieval_context}{budget_note}"

        return llm_messages

    async def _call_llm(self, messages: list[dict], model: str) -> str:
        """Call Groq via OpenAI-compatible API with retry logic."""
        max_retries = 3
        current_model = model
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=current_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2048,
                    top_p=0.9,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                # If model is blocked, fallback to the other model
                if "403" in error_str and current_model == MODEL_LIGHT:
                    logger.warning(f"Model {current_model} blocked, falling back to {MODEL_POWERFUL}")
                    current_model = MODEL_POWERFUL
                    continue
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Groq API error (attempt {attempt+1}), retrying in {wait}s: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise

    def _parse_and_validate(self, raw_response: str) -> ChatResponse:
        """Parse the LLM response and validate against the schema.

        Handles common LLM output issues:
        - Markdown code fences
        - Extra fields
        - Invalid URLs
        - Hallucinated assessment names
        """
        # Strip markdown code fences if present
        text = raw_response.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)

        # Try to parse JSON
        try:
            data = json.loads(text, strict=False)
        except json.JSONDecodeError:
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                try:
                    data = json.loads(json_match.group(), strict=False)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM response as JSON: {text[:200]}")
                    return ChatResponse(
                        reply=text[:500] if text else "I can help you find the right SHL assessments. What role are you hiring for?",
                        recommendations=[],
                        end_of_conversation=False,
                    )
            else:
                return ChatResponse(
                    reply=text[:500] if text else "I can help you find the right SHL assessments. What role are you hiring for?",
                    recommendations=[],
                    end_of_conversation=False,
                )

        # Extract fields
        reply = data.get("reply", "")
        recommendations_raw = data.get("recommendations", [])
        end_of_conversation = bool(data.get("end_of_conversation", False))

        # Validate and fix recommendations
        valid_recommendations = []
        if recommendations_raw and isinstance(recommendations_raw, list):
            for rec in recommendations_raw:
                if not isinstance(rec, dict):
                    continue

                name = rec.get("name", "")
                url = rec.get("url", "")
                test_type = rec.get("test_type", "K")

                # Validate URL exists in catalog
                if url in self.url_index:
                    canonical = self.url_index[url]
                    valid_recommendations.append(
                        Recommendation(
                            name=canonical.name,
                            url=canonical.url,
                            test_type=canonical.test_type_codes,
                        )
                    )
                elif name.lower() in self.name_index:
                    canonical = self.name_index[name.lower()]
                    valid_recommendations.append(
                        Recommendation(
                            name=canonical.name,
                            url=canonical.url,
                            test_type=canonical.test_type_codes,
                        )
                    )
                else:
                    matched = self._fuzzy_match_assessment(name)
                    if matched:
                        valid_recommendations.append(
                            Recommendation(
                                name=matched.name,
                                url=matched.url,
                                test_type=matched.test_type_codes,
                            )
                        )
                    else:
                        logger.warning(f"Dropping hallucinated recommendation: {name} ({url})")

        # Deduplicate by URL
        seen_urls = set()
        deduped = []
        for rec in valid_recommendations:
            if rec.url not in seen_urls:
                seen_urls.add(rec.url)
                deduped.append(rec)
        valid_recommendations = deduped[:TOP_K_RECOMMEND]

        return ChatResponse(
            reply=reply,
            recommendations=valid_recommendations,
            end_of_conversation=end_of_conversation,
        )

    def _fuzzy_match_assessment(self, name: str) -> Assessment | None:
        """Try to fuzzy-match an assessment name to the catalog."""
        name_lower = name.lower().strip()

        # Try substring matching
        for a_name, assessment in self.name_index.items():
            if name_lower in a_name or a_name in name_lower:
                return assessment

        # Try keyword overlap
        name_words = set(name_lower.split())
        best_match = None
        best_overlap = 0
        for a_name, assessment in self.name_index.items():
            a_words = set(a_name.split())
            overlap = len(name_words & a_words)
            if overlap > best_overlap and overlap >= 2:
                best_overlap = overlap
                best_match = assessment

        return best_match
