"""
Gemini embeddings implementation for the RAG system.
"""
from typing import Any
import os
import hashlib

import numpy as np
import google.generativeai as genai
from langchain.embeddings.base import Embeddings

class GeminiEmbeddings(Embeddings):
    """Gemini embeddings wrapper for LangChain."""
    
    def __init__(self, api_key: str | None = None):
        """Initialize Gemini embeddings."""
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents using Gemini."""
        embeddings = []
        for text in texts:
            response = self.model.generate_content(text)
            # Convert the response to embeddings (this is a simplified version)
            # In practice, you would use the proper embedding endpoint when available
            embedding = self._text_to_embedding(response.text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Embed query text using Gemini."""
        response = self.model.generate_content(text)
        return self._text_to_embedding(response.text)
    
    def _text_to_embedding(self, text: str) -> list[float]:
        """Convert text to embedding vector."""
        # This is a temporary solution until Gemini provides a proper embedding endpoint
        # For now, we'll use a simple hash-based approach
        
        # Create a consistent hash of the text
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash to a list of floats (1536 dimensions to match OpenAI's size)
        np.random.seed(int.from_bytes(hash_bytes[:8], byteorder='big'))
        embedding = np.random.normal(size=1536)
        
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
