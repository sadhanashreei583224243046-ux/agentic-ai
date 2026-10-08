"""
Hybrid Retriever module combining Gemini Dense Semantic Vector Search
with BM25/Keyword Sparse Search for robust, high-accuracy document retrieval.
"""

import math
import re
from typing import List, Dict, Optional, Any
import numpy as np
from google import genai
from .indexer import DocumentChunk


class HybridRetriever:
    """
    Retriever that blends dense vector embeddings with sparse keyword scoring.
    """

    def __init__(
        self,
        chunks: List[DocumentChunk],
        pages_text: Optional[List[str]] = None,
        sections_found: Optional[List[str]] = None,
        client: Optional[genai.Client] = None,
        embedding_model: str = "gemini-embedding-001",
    ):
        self.chunks = chunks
        self.pages_text = pages_text or []
        self.sections_found = sections_found or []
        self.client = client
        self.embedding_model = embedding_model

        # Precompute chunk embeddings matrix if available
        self._build_embedding_matrix()

    def _build_embedding_matrix(self) -> None:
        self.embedded_indices = []
        vectors = []
        for idx, chunk in enumerate(self.chunks):
            if chunk.embedding is not None and len(chunk.embedding) > 0:
                self.embedded_indices.append(idx)
                vectors.append(chunk.embedding)

        if vectors:
            self.embeddings_matrix = np.array(vectors, dtype=np.float32)
            # Normalize vectors for fast cosine similarity via dot product
            norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            self.normalized_embeddings = self.embeddings_matrix / norms
        else:
            self.embeddings_matrix = None
            self.normalized_embeddings = None

    def search_semantic(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Dense semantic search using Gemini embeddings."""
        if self.normalized_embeddings is None or not self.client:
            return []

        try:
            res = self.client.models.embed_content(
                model=self.embedding_model,
                contents=query,
            )
            if not res.embeddings or len(res.embeddings) == 0:
                return []

            q_vec = np.array(res.embeddings[0].values, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return []
            q_vec /= q_norm

            scores = np.dot(self.normalized_embeddings, q_vec)
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for rank_idx in top_indices:
                chunk_idx = self.embedded_indices[rank_idx]
                chunk = self.chunks[chunk_idx]
                score = float(scores[rank_idx])
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "text": chunk.text,
                    "score": score,
                    "retrieval_type": "semantic",
                })
            return results
        except Exception as e:
            # Fallback gracefully
            return []

    def search_keyword(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Sparse keyword & token matching search."""
        query_clean = query.lower().strip()
        tokens = [t for t in re.findall(r"\w+", query_clean) if len(t) > 1]
        if not tokens:
            return []

        results = []
        for chunk in self.chunks:
            # Avoid full page duplicate in keyword search if section chunks exist
            if chunk.section.startswith("PAGE_") and chunk.section.endswith("_FULL"):
                continue

            text_lower = chunk.text.lower()
            score = 0.0

            # Exact full phrase match boost
            if query_clean in text_lower:
                score += 3.0

            # Individual token matches
            for token in tokens:
                matches = text_lower.count(token)
                if matches > 0:
                    score += 1.0 + math.log(1 + matches)

            if score > 0:
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "text": chunk.text,
                    "score": score,
                    "retrieval_type": "keyword",
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_hybrid(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Combines semantic and keyword scores with rank fusion and deduplication.
        """
        semantic_results = self.search_semantic(query, top_k=top_k * 2)
        keyword_results = self.search_keyword(query, top_k=top_k * 2)

        # Reciprocal Rank Fusion (RRF)
        combined_scores: Dict[str, float] = {}
        chunks_map: Dict[str, Dict[str, Any]] = {}
        rrf_k = 40

        for rank, item in enumerate(semantic_results):
            cid = item["chunk_id"]
            combined_scores[cid] = combined_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            chunks_map[cid] = item

        for rank, item in enumerate(keyword_results):
            cid = item["chunk_id"]
            combined_scores[cid] = combined_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            if cid not in chunks_map:
                chunks_map[cid] = item

        # If semantic failed (e.g. offline/no key yet), keyword scores serve directly
        if not semantic_results and keyword_results:
            return keyword_results[:top_k]

        # Sort by combined score
        sorted_cids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        final_results = []
        for cid, score in sorted_cids[:top_k]:
            item = dict(chunks_map[cid])
            item["score"] = round(score, 4)
            item["retrieval_type"] = "hybrid"
            final_results.append(item)

        return final_results

    def get_page_content(self, page_number: int) -> str:
        """Retrieve full text of a given page number (1-indexed)."""
        if 1 <= page_number <= len(self.pages_text):
            return f"--- PAGE {page_number} CONTENT ---\n" + self.pages_text[page_number - 1]
        return f"Error: Page number {page_number} out of range (document has {len(self.pages_text)} pages)."

    def list_sections(self) -> List[str]:
        """List detected sections in the document."""
        return self.sections_found
