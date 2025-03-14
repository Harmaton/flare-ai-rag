"""
Advanced retrieval system implementing hybrid search and query expansion.
"""
from typing import Any, List
from dataclasses import dataclass
import logging
import asyncio

import numpy as np
from qdrant_client import QdrantClient
from flare_ai_rag.embeddings.gemini_embeddings import GeminiEmbeddings
import google.generativeai as genai

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Search result with content and metadata."""
    content: str
    metadata: dict[str, Any]
    score: float

class AdvancedRetriever:
    """Advanced retrieval system with hybrid search capabilities."""

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str = "flare_knowledge_base",
        embedding_model: Any | None = None,
        gemini_api_key: str | None = None,
    ):
        """Initialize the retriever."""
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model or GeminiEmbeddings()
        
        # Initialize Gemini
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be provided")
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-pro')

    async def expand_query(self, query: str) -> list[str]:
        """Expand query using Gemini to improve search coverage."""
        prompt = f"""Given the search query: "{query}"
        Generate 2-3 alternative ways to express this query to improve search results.
        Focus on semantic variations that might capture different relevant aspects.
        Return only the variations, one per line."""
        
        response = await asyncio.to_thread(
            self.llm.generate_content, prompt
        )
        variations = [query] + response.text.strip().split('\n')
        return variations[:3]  # Limit to 3 variations

    async def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.7
    ) -> list[SearchResult]:
        """Perform hybrid search combining semantic and keyword-based approaches."""
        expanded_queries = await self.expand_query(query)
        embeddings = await asyncio.to_thread(
            self.embedding_model.embed_documents, expanded_queries
        )
        
        # Combine results from all query variations
        all_results = []
        for query_embedding in embeddings:
            vector_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=threshold
            )
            
            for result in vector_results:
                search_result = SearchResult(
                    content=result.payload.get("content", ""),
                    metadata=result.payload.get("metadata", {}),
                    score=float(result.score)
                )
                all_results.append(search_result)
        
        # Remove duplicates and sort by score
        unique_results = self._deduplicate_results(all_results)
        return sorted(unique_results, key=lambda x: x.score, reverse=True)[:limit]

    def _deduplicate_results(
        self,
        results: list[SearchResult]
    ) -> list[SearchResult]:
        """Remove duplicate results based on content similarity."""
        seen_contents = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(result.content)
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
        
        return unique_results
