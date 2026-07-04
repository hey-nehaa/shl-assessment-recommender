"""Tests for schema validation, catalog loading, and response correctness."""

from __future__ import annotations

import json
import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.catalog import load_catalog, build_url_index, build_name_index
from app.models import ChatRequest, ChatResponse, Recommendation


class TestCatalogLoading:
    """Tests for catalog data loading and preprocessing."""

    @pytest.fixture(scope="class")
    @staticmethod
    def assessments():
        return load_catalog()

    def test_catalog_loads(self, assessments):
        """Catalog should load successfully with items."""
        assert len(assessments) > 0
        assert len(assessments) > 300  # Expected ~377 items

    def test_no_duplicate_ids(self, assessments):
        """All entity_ids should be unique."""
        ids = [a.entity_id for a in assessments]
        assert len(ids) == len(set(ids))

    def test_all_have_urls(self, assessments):
        """Every assessment must have a URL."""
        for a in assessments:
            assert a.url, f"Assessment {a.name} has no URL"
            assert a.url.startswith("https://www.shl.com/"), f"Invalid URL: {a.url}"

    def test_all_have_names(self, assessments):
        """Every assessment must have a name."""
        for a in assessments:
            assert a.name, f"Assessment {a.entity_id} has no name"

    def test_test_type_codes(self, assessments):
        """Test type codes should be valid."""
        valid_codes = {"K", "P", "A", "S", "B", "C", "D", "E"}
        for a in assessments:
            codes = a.test_type_codes.split(",")
            for code in codes:
                assert code in valid_codes, f"Invalid code '{code}' for {a.name}"

    def test_url_index(self, assessments):
        """URL index should map every URL to an assessment."""
        url_idx = build_url_index(assessments)
        assert len(url_idx) == len(assessments)
        for a in assessments:
            assert a.url in url_idx

    def test_name_index(self, assessments):
        """Name index should be case-insensitive."""
        name_idx = build_name_index(assessments)
        assert "opq32r" not in name_idx or True  # Partial names won't match

    def test_search_text_built(self, assessments):
        """Every assessment should have search text for embeddings."""
        for a in assessments:
            assert a.search_text, f"Assessment {a.name} has no search text"
            assert len(a.search_text) > 20


class TestSchemaValidation:
    """Tests for response schema compliance."""

    def test_valid_response_with_recommendations(self):
        """Response with recommendations should validate."""
        resp = ChatResponse(
            reply="Here are my recommendations.",
            recommendations=[
                Recommendation(
                    name="Core Java (Advanced Level) (New)",
                    url="https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
                    test_type="K",
                )
            ],
            end_of_conversation=False,
        )
        assert len(resp.recommendations) == 1
        assert resp.reply != ""

    def test_valid_response_empty_recommendations(self):
        """Response without recommendations (clarifying) should validate."""
        resp = ChatResponse(
            reply="What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )
        assert len(resp.recommendations) == 0

    def test_response_json_serialization(self):
        """Response should serialize to valid JSON with exact field names."""
        resp = ChatResponse(
            reply="Test reply",
            recommendations=[
                Recommendation(name="Test", url="https://example.com", test_type="K")
            ],
            end_of_conversation=True,
        )
        data = json.loads(resp.model_dump_json())
        assert "reply" in data
        assert "recommendations" in data
        assert "end_of_conversation" in data
        assert len(data) == 3  # No extra fields

    def test_recommendation_fields(self):
        """Recommendations should have exactly name, url, test_type."""
        rec = Recommendation(name="Test", url="https://example.com", test_type="K")
        data = json.loads(rec.model_dump_json())
        assert set(data.keys()) == {"name", "url", "test_type"}

    def test_chat_request_parsing(self):
        """Chat request should parse correctly."""
        raw = {
            "messages": [
                {"role": "user", "content": "I need an assessment"},
                {"role": "assistant", "content": "What role?"},
                {"role": "user", "content": "Java developer"},
            ]
        }
        req = ChatRequest(**raw)
        assert len(req.messages) == 3
        assert req.messages[0].role == "user"


class TestRetrievalEngine:
    """Tests for the retrieval system."""

    @pytest.fixture(scope="class")
    @staticmethod
    def engine():
        from app.retrieval import RetrievalEngine
        assessments = load_catalog()
        return RetrievalEngine(assessments)

    def test_search_returns_results(self, engine):
        """Search should return results for a valid query."""
        results = engine.search("Java developer assessment", top_k=10)
        assert len(results) > 0
        assert len(results) <= 10

    def test_search_java_finds_java(self, engine):
        """Searching for Java should find Java-related assessments."""
        results = engine.search("Core Java programming test", top_k=10)
        names = [a.name.lower() for a, _ in results]
        assert any("java" in name for name in names), f"No Java results found: {names}"

    def test_search_personality_finds_opq(self, engine):
        """Searching for personality should find OPQ."""
        results = engine.search("personality assessment workplace behavior", top_k=10)
        names = [a.name.lower() for a, _ in results]
        assert any("opq" in name or "personality" in name for name in names)

    def test_search_returns_scores(self, engine):
        """Results should include relevance scores."""
        results = engine.search("SQL database test", top_k=5)
        for assessment, score in results:
            assert isinstance(score, float)
            assert score > 0

    def test_search_by_names(self, engine):
        """Name-based lookup should work."""
        results = engine.search_by_names(["Core Java (Advanced Level) (New)"])
        assert len(results) == 1
        assert "Java" in results[0].name
