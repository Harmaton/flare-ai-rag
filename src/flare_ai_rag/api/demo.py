"""
FastAPI demo interface for the Flare AI RAG system.
"""
from typing import Any, List
from pathlib import Path
import os
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from flare_ai_rag.ingestion.pipeline import DataIngestionPipeline, DataSource
from flare_ai_rag.retrieval.advanced_retriever import AdvancedRetriever
from flare_ai_rag.embeddings.gemini_embeddings import GeminiEmbeddings

app = FastAPI(title="Flare AI RAG Demo")
logger = logging.getLogger(__name__)

# Initialize components
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable must be set")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedding_model = GeminiEmbeddings(api_key=GEMINI_API_KEY)
pipeline = DataIngestionPipeline(client, embedding_model=embedding_model)
retriever = AdvancedRetriever(
    client,
    embedding_model=embedding_model,
    gemini_api_key=GEMINI_API_KEY
)

class IngestRequest(BaseModel):
    """Request model for data ingestion."""
    source_type: str
    path: str
    metadata: dict[str, Any]
    verification_score: float = 1.0

class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""
    query: str
    limit: int = 5
    threshold: float = 0.7

class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str
    conversation_id: str | None = None
    context_window: int = 2000

class SearchResponse(BaseModel):
    """Response model for search results."""
    content: str
    metadata: dict[str, Any]
    score: float

class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str
    conversation_id: str
    sources: list[dict[str, Any]]
    confidence_score: float

@app.post("/ingest")
async def ingest_data(request: IngestRequest) -> dict[str, Any]:
    """Ingest data from a source."""
    try:
        source = DataSource(
            source_type=request.source_type,
            path=request.path,
            metadata=request.metadata,
            last_updated=datetime.now(),
            verification_score=request.verification_score
        )
        num_chunks = pipeline.ingest_source(source)
        return {"status": "success", "chunks_processed": num_chunks}
    except Exception as e:
        logger.error(f"Error ingesting data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_knowledge_base(request: QueryRequest) -> list[SearchResponse]:
    """Query the knowledge base."""
    try:
        results = await retriever.hybrid_search(
            request.query,
            limit=request.limit,
            threshold=request.threshold
        )
        return [
            SearchResponse(
                content=result.content,
                metadata=result.metadata,
                score=result.score
            )
            for result in results
        ]
    except Exception as e:
        logger.error(f"Error querying knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat with the RAG system."""
    try:
        # First, retrieve relevant context
        results = await retriever.hybrid_search(
            request.message,
            limit=5,
            threshold=0.7
        )
        
        # Format context for the model
        context = "\n\n".join(f"Source: {r.metadata.get('source', 'Unknown')}\n{r.content}" 
                            for r in results)
        
        # Generate response using Gemini
        prompt = f"""Based on the following context, provide a helpful response to the user's message. 
        If the answer cannot be derived from the context, say so.
        
        Context:
        {context}
        
        User message: {request.message}"""
        
        response = retriever.llm.generate_content(prompt)
        
        # Calculate confidence based on retrieval scores
        confidence = sum(r.score for r in results) / len(results) if results else 0.0
        
        return ChatResponse(
            response=response.text,
            conversation_id=request.conversation_id or datetime.now().isoformat(),
            sources=[{
                "content": r.content,
                "metadata": r.metadata,
                "score": r.score
            } for r in results],
            confidence_score=confidence
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources")
async def list_sources() -> list[dict[str, Any]]:
    """List available data sources."""
    try:
        # In a real implementation, this would query the database
        return [
            {
                "source_type": "git",
                "path": "https://github.com/flare-foundation/go-flare",
                "metadata": {"description": "Main Flare Network repository"},
            },
            {
                "source_type": "markdown",
                "path": "https://dev.flare.network/ftso/overview",
                "metadata": {"description": "Official Flare Documentation"},
            }
        ]
    except Exception as e:
        logger.error(f"Error listing sources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> dict[str, str]:
    """Check system health."""
    return {"status": "healthy"}
