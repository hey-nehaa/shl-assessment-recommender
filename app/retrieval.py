"""Retrieval system: FAISS-based semantic search with keyword boosting.

Builds embeddings at startup and provides hybrid search combining
semantic similarity with keyword matching for robust retrieval.
"""

from __future__ import annotations

import logging
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.catalog import Assessment
from app.config import EMBEDDING_MODEL, TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Hybrid retrieval engine combining FAISS semantic search with keyword boosting."""

    def __init__(self, assessments: list[Assessment]):
        self.assessments = assessments
        self._name_lower_map = {a.name.lower(): i for i, a in enumerate(assessments)}
        self._keyword_index: dict[str, set[int]] = {}

        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Build search texts and embeddings
        search_texts = [a.search_text for a in assessments]
        logger.info(f"Encoding {len(search_texts)} assessments...")
        embeddings = self.model.encode(search_texts, show_progress_bar=False, normalize_embeddings=True)
        self.embeddings = np.array(embeddings, dtype=np.float32)

        # Build FAISS index (inner product since embeddings are normalized = cosine similarity)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        # Build keyword index for exact-match boosting
        self._build_keyword_index()

        logger.info(f"Retrieval engine ready with {len(assessments)} assessments, dim={dim}")

    def _build_keyword_index(self):
        """Build inverted index of keywords -> assessment indices."""
        for i, a in enumerate(self.assessments):
            # Extract keywords from name and description
            text = f"{a.name} {a.description}".lower()
            # Include key categories
            for key in a.keys:
                text += f" {key.lower()}"
            # Include job levels
            for level in a.job_levels:
                text += f" {level.lower()}"

            words = set(re.findall(r"[a-z0-9#+.]+", text))
            for word in words:
                self._keyword_index.setdefault(word, set()).add(i)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        job_level_filter: str | None = None,
        type_filter: list[str] | None = None,
    ) -> list[tuple[Assessment, float]]:
        """Search for relevant assessments.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            job_level_filter: Optional job level to filter by.
            type_filter: Optional list of key categories to filter by.

        Returns:
            List of (Assessment, score) tuples sorted by relevance.
        """
        k = top_k or TOP_K_RETRIEVAL

        # Encode query
        query_embedding = self.model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )
        query_vec = np.array(query_embedding, dtype=np.float32)

        # Semantic search — retrieve more than needed for re-ranking
        n_search = min(len(self.assessments), k * 3)
        scores, indices = self.index.search(query_vec, n_search)

        # Build result set with scores
        results: dict[int, float] = {}
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results[idx] = float(score)

        # Keyword boosting
        query_words = set(re.findall(r"[a-z0-9#+.]+", query.lower()))
        for word in query_words:
            matching_indices = self._keyword_index.get(word, set())
            for idx in matching_indices:
                if idx in results:
                    results[idx] += 0.15  # Boost existing semantic matches
                else:
                    results[idx] = 0.3  # Add keyword-only matches with lower base score

        # Apply filters
        filtered_results: list[tuple[int, float]] = []
        for idx, score in results.items():
            a = self.assessments[idx]

            # Job level filter
            if job_level_filter:
                level_lower = job_level_filter.lower()
                a_levels = [l.lower() for l in a.job_levels]
                if not any(level_lower in al or al in level_lower for al in a_levels):
                    # Don't hard-filter; just reduce score
                    score *= 0.7

            # Type filter
            if type_filter:
                type_match = any(
                    tf.lower() in [k.lower() for k in a.keys] for tf in type_filter
                )
                if not type_match:
                    score *= 0.6

            filtered_results.append((idx, score))

        # Sort by score descending
        filtered_results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        return [
            (self.assessments[idx], score)
            for idx, score in filtered_results[:k]
        ]

    def search_by_names(self, names: list[str]) -> list[Assessment]:
        """Look up assessments by name (fuzzy match)."""
        results = []
        for name in names:
            name_lower = name.lower().strip()
            # Exact match
            if name_lower in self._name_lower_map:
                results.append(self.assessments[self._name_lower_map[name_lower]])
                continue
            # Partial match
            for a_name, idx in self._name_lower_map.items():
                if name_lower in a_name or a_name in name_lower:
                    results.append(self.assessments[idx])
                    break
        return results

    def get_all_assessments(self) -> list[Assessment]:
        """Return all assessments (for catalog-level queries)."""
        return self.assessments
