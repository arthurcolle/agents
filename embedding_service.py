#!/usr/bin/env python3
"""
Embeddings Service Client

This module provides a client for the Embeddings Service API, allowing:
- Generation of embeddings for texts
- Computation of cosine similarity between texts
- Pairwise cosine similarity for a list of texts
- Model management
"""

import requests
import json
import uuid
from typing import List, Dict, Union, Optional, Any, Tuple
import numpy as np

class EmbeddingServiceClient:
    """Client for the Embeddings Service API"""
    
    def __init__(self, base_url: str = "https://arthurcolle--embeddings.modal.run"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def generate_embeddings(self, 
                           texts: List[str], 
                           batch_size: int = 32, 
                           max_length: int = 8192) -> Dict[str, Any]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            max_length: Maximum token length to process
            
        Returns:
            Dictionary containing embeddings and pairwise similarities
        """
        url = f"{self.base_url}/generate_embeddings"
        
        payload = {
            "texts": texts,
            "batch_size": batch_size,
            "max_length": max_length
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def compute_cosine_similarity(self, 
                                 texts1: Union[str, List[str]], 
                                 texts2: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Compute cosine similarity between two texts or lists of texts
        
        Args:
            texts1: First text or list of texts
            texts2: Second text or list of texts
            
        Returns:
            Dictionary containing similarity scores
        """
        url = f"{self.base_url}/cosine_similarity"
        
        payload = {
            "texts1": texts1,
            "texts2": texts2
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def compute_pairwise_cosine_similarity(self, texts: List[str]) -> Dict[str, Any]:
        """
        Compute pairwise cosine similarity for a list of texts
        
        Args:
            texts: List of texts for pairwise comparison
            
        Returns:
            Dictionary containing similarity matrix
        """
        url = f"{self.base_url}/pairwise_cosine_similarity"
        
        payload = {
            "texts": texts
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def load_model(self, model_name: str) -> Dict[str, str]:
        """
        Load a specific model by name
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Dictionary containing status and message
        """
        url = f"{self.base_url}/load_model/{model_name}"
        
        response = self.session.post(url)
        response.raise_for_status()
        
        return response.json()
    
    def list_available_models(self) -> Dict[str, Any]:
        """
        List all available models and currently loaded model
        
        Returns:
            Dictionary containing list of models and current model
        """
        url = f"{self.base_url}/available_models"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as a list of floats
        """
        result = self.generate_embeddings([text])
        
        if 'embeddings' in result and len(result['embeddings']) > 0:
            return result['embeddings'][0]['embedding']
        
        raise ValueError("Failed to generate embedding")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of vector embeddings
        """
        result = self.generate_embeddings(texts)
        
        if 'embeddings' in result and len(result['embeddings']) > 0:
            return [item['embedding'] for item in result['embeddings']]
        
        raise ValueError("Failed to generate embeddings")
    
    def semantic_search(self, 
                       query: str, 
                       documents: List[str], 
                       top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Perform semantic search to find documents most similar to the query
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document_index, similarity_score)
        """
        # Get query embedding
        query_embedding = self.embed_text(query)
        
        # Get document embeddings
        doc_embeddings = self.embed_batch(documents)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity as a float between -1 and 1
        """
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        
        dot_product = np.dot(vec1_array, vec2_array)
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


def compute_embedding(text, db_conn=None):
    """
    Compute embedding for a text using the Embeddings Service
    
    This function serves as a bridge to the holographic_memory.py compute_embedding function,
    but uses the Embeddings Service instead.
    
    Args:
        text: Text to embed
        db_conn: Optional database connection (not used in this implementation)
        
    Returns:
        Vector embedding as a list of floats
    """
    try:
        client = EmbeddingServiceClient()
        embedding = client.embed_text(text)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return a dummy embedding if the service fails
        return [0.0] * 768


if __name__ == "__main__":
    """Test the Embeddings Service client"""
    client = EmbeddingServiceClient()
    
    # Test available models
    print("Available models:")
    models = client.list_available_models()
    print(json.dumps(models, indent=2))
    
    # Test embeddings
    test_texts = [
        "This is a test of the embedding service.",
        "Embeddings are useful for semantic search and similarity matching."
    ]
    
    print("\nGenerating embeddings:")
    result = client.generate_embeddings(test_texts)
    # Print just the first few dimensions of each embedding
    for item in result['embeddings']:
        print(f"ID: {item['id']}")
        print(f"Content: {item['content']}")
        print(f"Embedding (first 5 dims): {item['embedding'][:5]}...")
    
    # Test cosine similarity
    print("\nComputing cosine similarity:")
    similarity = client.compute_cosine_similarity(test_texts[0], test_texts[1])
    print(f"Similarity: {similarity['similarity']}")
    
    # Test semantic search
    print("\nPerforming semantic search:")
    documents = [
        "The weather is lovely today with clear skies and sunshine.",
        "Python is a popular programming language for data science.",
        "Machine learning models require quality training data.",
        "Embedding vectors represent semantic meaning in high-dimensional space.",
        "The concept of similarity is central to information retrieval."
    ]
    
    query = "How do vector embeddings work in NLP?"
    search_results = client.semantic_search(query, documents)
    
    print(f"Query: {query}")
    print("Top results:")
    for idx, score in search_results:
        print(f"  {score:.4f}: {documents[idx]}")