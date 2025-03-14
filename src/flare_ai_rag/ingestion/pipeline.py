"""
Enhanced data ingestion pipeline for Flare AI RAG system.
Supports multiple data sources and implements sophisticated preprocessing.
"""
from typing import Any
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    GitLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
from flare_ai_rag.embeddings.gemini_embeddings import GeminiEmbeddings

logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for a data source"""
    source_type: str  # git, csv, markdown, text
    path: str
    metadata: dict[str, Any]
    last_updated: datetime
    verification_score: float = 1.0  # Source reliability score (0-1)

class DataIngestionPipeline:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str = "flare_knowledge_base",
        embedding_model: Any | None = None,
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model or GeminiEmbeddings()
        self._ensure_collection()
        
        # Configure text splitter for optimal chunk sizes
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

    def _ensure_collection(self):
        """Ensure Qdrant collection exists with proper configuration"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimensions
                    distance=Distance.COSINE,
                ),
            )

    def load_source(self, source: DataSource) -> list[dict[str, Any]]:
        """Load and preprocess documents from a data source"""
        documents = []
        
        try:
            if source.source_type == "git":
                loader = GitLoader(
                    clone_url=source.path,
                    branch="main",
                    file_filter=lambda file_path: any(
                        file_path.endswith(ext) 
                        for ext in [".md", ".py", ".js", ".ts", ".txt"]
                    ),
                )
                documents = loader.load()
                
            elif source.source_type == "csv":
                loader = CSVLoader(file_path=source.path)
                documents = loader.load()
                
            elif source.source_type == "markdown":
                loader = UnstructuredMarkdownLoader(file_path=source.path)
                documents = loader.load()
                
            elif source.source_type == "text":
                loader = TextLoader(file_path=source.path)
                documents = loader.load()
                
            else:
                raise ValueError(f"Unsupported source type: {source.source_type}")

            # Split documents into chunks
            chunks = []
            for doc in documents:
                doc_chunks = self.text_splitter.split_text(doc.page_content)
                for chunk in doc_chunks:
                    chunks.append({
                        "content": chunk,
                        "metadata": {
                            **doc.metadata,
                            "source_type": source.source_type,
                            "source_path": source.path,
                            "verification_score": source.verification_score,
                            "last_updated": source.last_updated.isoformat(),
                            "chunk_size": len(chunk),
                        }
                    })
            
            return chunks

        except Exception as e:
            logger.error(f"Error loading source {source.path}: {str(e)}")
            return []

    def process_and_index(self, chunks: list[dict[str, Any]]):
        """Process chunks and index them in Qdrant"""
        try:
            # Generate embeddings
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                points.append({
                    "id": i,
                    "vector": embedding,
                    "payload": {
                        "content": chunk["content"],
                        **chunk["metadata"]
                    }
                })
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully indexed {len(points)} chunks")
            
        except Exception as e:
            logger.error(f"Error indexing chunks: {str(e)}")

    def ingest_source(self, source: DataSource):
        """Main method to ingest a data source"""
        chunks = self.load_source(source)
        if chunks:
            self.process_and_index(chunks)
            return len(chunks)
        return 0

    def get_source_stats(self) -> dict[str, Any]:
        """Get statistics about indexed sources"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "total_points": collection_info.points_count,
                "vectors_size": collection_info.vectors_size,
                "status": collection_info.status,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
