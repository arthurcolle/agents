#!/usr/bin/env python3
"""
optimized_vector_memory.py
--------------------------
An optimized, high-performance vector memory system for the Llama4 agent architecture.
This implementation focuses on efficient local operation, improved caching, and reduced
dependency on external services when running locally.
"""

import os
import json
import time
import uuid
import hashlib
import numpy as np
import sqlite3
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import threading
from contextlib import contextmanager

# Try to import optional dependencies with appropriate fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import hnswlib
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Attempt to use faster numpy operations where available
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# Only define if numba is available
if NUMBA_AVAILABLE:
    @numba.njit(parallel=True, fastmath=True)
    def fast_cosine_similarity(a, b):
        """Optimized cosine similarity with numba"""
        dot_product = np.dot(a, b)
        norm_a = np.sqrt(np.sum(a**2))
        norm_b = np.sqrt(np.sum(b**2))
        return dot_product / (norm_a * norm_b)
else:
    def fast_cosine_similarity(a, b):
        """Standard cosine similarity fallback"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Thread-local SQLite connections for thread safety
local = threading.local()

@contextmanager
def get_db_connection(db_path):
    """Thread-safe database connection manager"""
    if not hasattr(local, 'connections'):
        local.connections = {}
    
    if db_path not in local.connections:
        local.connections[db_path] = sqlite3.connect(db_path)
        # Enable WAL mode for better concurrent performance
        local.connections[db_path].execute('PRAGMA journal_mode=WAL')
        local.connections[db_path].execute('PRAGMA synchronous=NORMAL')
    
    try:
        yield local.connections[db_path]
    except Exception as e:
        local.connections[db_path].rollback()
        raise e

class OptimizedEmbeddingProvider:
    """Optimized local embedding provider with caching"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser('~'), '.cache', 'llama4_embeddings')
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.model = None
        self.cache = {}  # In-memory cache
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize SQLite cache
        self.db_path = os.path.join(self.cache_dir, 'embedding_cache.db')
        self._init_db()
        
        # Initialize model if available
        self._init_model()
    
    def _init_db(self):
        """Initialize the SQLite cache database"""
        with get_db_connection(self.db_path) as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                hash TEXT PRIMARY KEY,
                content TEXT,
                embedding BLOB,
                created_at REAL
            )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_hash ON embeddings(hash)')
            conn.commit()
    
    def _init_model(self):
        """Initialize the embedding model if available"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                return True
            except Exception as e:
                print(f"Warning: Failed to initialize SentenceTransformer: {e}")
        return False
    
    def _compute_hash(self, content: str) -> str:
        """Compute a hash for the content to use as a cache key"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _get_from_cache(self, content_hash: str) -> Optional[np.ndarray]:
        """Get embedding from cache (memory or disk)"""
        # Check in-memory cache first
        if content_hash in self.cache:
            self.cache_hits += 1
            return self.cache[content_hash]
        
        # Check disk cache
        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT embedding FROM embeddings WHERE hash = ?', 
                    (content_hash,)
                )
                row = cursor.fetchone()
                if row:
                    embedding = np.frombuffer(row[0], dtype=np.float32)
                    # Add to in-memory cache
                    self.cache[content_hash] = embedding
                    self.cache_hits += 1
                    return embedding
        except Exception as e:
            print(f"Warning: Error retrieving from cache: {e}")
        
        self.cache_misses += 1
        return None
    
    def _save_to_cache(self, content_hash: str, content: str, embedding: np.ndarray):
        """Save embedding to cache (memory and disk)"""
        # Save to in-memory cache
        self.cache[content_hash] = embedding
        
        # Save to disk cache
        try:
            with get_db_connection(self.db_path) as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO embeddings (hash, content, embedding, created_at) VALUES (?, ?, ?, ?)',
                    (content_hash, content, embedding.tobytes(), time.time())
                )
                conn.commit()
        except Exception as e:
            print(f"Warning: Error saving to cache: {e}")
    
    def get_embedding(self, content: str) -> np.ndarray:
        """Get embedding for a text, using cache if available"""
        if not content:
            # Return zero vector for empty content
            return np.zeros(self.dimension, dtype=np.float32)
        
        # Compute hash for cache lookup
        content_hash = self._compute_hash(content)
        
        # Try to get from cache
        cached_embedding = self._get_from_cache(content_hash)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate new embedding if model is available
        if self.model:
            embedding = self.model.encode(content, convert_to_numpy=True).astype(np.float32)
            self._save_to_cache(content_hash, content, embedding)
            return embedding
        
        # Fallback: return random embedding (useful for testing without model)
        embedding = np.random.randn(self.dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        # Count entries in SQLite cache
        try:
            with get_db_connection(self.db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM embeddings')
                disk_cache_size = cursor.fetchone()[0]
        except Exception:
            disk_cache_size = 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.cache),
            "disk_cache_size": disk_cache_size,
        }

class OptimizedVectorIndex:
    """High-performance vector index with multiple backend options"""
    
    def __init__(self, dimension: int = 384, index_path: str = None, 
                 max_elements: int = 100000, space: str = "cosine"):
        self.dimension = dimension
        self.index_path = index_path
        self.max_elements = max_elements
        self.space = space
        self.index = None
        self.id_to_metadata = {}
        self.next_id = 0
        
        # Select best available backend
        if FAISS_AVAILABLE:
            self.backend = "faiss"
        elif HNSW_AVAILABLE:
            self.backend = "hnswlib"
        else:
            self.backend = "numpy"
        
        self._init_index()
    
    def _init_index(self):
        """Initialize the vector index based on available backends"""
        if self.backend == "faiss":
            # Use FAISS for high-performance nearest neighbor search
            if self.space == "cosine":
                # L2 normalized vectors + L2 distance = cosine similarity
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                # Inner product is another option
                self.index = faiss.IndexFlatIP(self.dimension)
        
        elif self.backend == "hnswlib":
            # Use HNSW for fast approximate nearest neighbor search
            self.index = hnswlib.Index(space=self.space, dim=self.dimension)
            self.index.init_index(max_elements=self.max_elements, ef_construction=200, M=16)
            self.index.set_ef(50)  # Controls accuracy vs speed tradeoff for search
        
        else:
            # Fallback to simple numpy-based index
            self.vectors = []
    
    def add_item(self, vector: np.ndarray, metadata: Dict[str, Any]) -> int:
        """Add an item to the index"""
        item_id = self.next_id
        self.next_id += 1
        
        # Store metadata mapped to the ID
        self.id_to_metadata[item_id] = metadata
        
        # Normalize vector if using cosine similarity
        if self.space == "cosine":
            vector = vector / np.linalg.norm(vector)
        
        # Add to the appropriate index type
        if self.backend == "faiss":
            self.index.add(np.array([vector], dtype=np.float32))
        
        elif self.backend == "hnswlib":
            self.index.add_items(vector, item_id)
        
        else:
            # Simple numpy storage
            self.vectors.append((item_id, vector))
        
        return item_id
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Search for similar vectors and return (id, similarity) pairs"""
        if not self.next_id:
            return []  # Empty index
        
        # Normalize query vector if using cosine similarity
        if self.space == "cosine":
            query_vector = query_vector / np.linalg.norm(query_vector)
        
        if self.backend == "faiss":
            D, I = self.index.search(np.array([query_vector], dtype=np.float32), k)
            # Convert distances to similarities
            if self.space == "cosine":
                similarities = 1.0 - np.sqrt(D[0]) / 2.0
            else:
                similarities = D[0]  # For inner product, the distance is already a similarity
            return [(int(idx), float(sim)) for idx, sim in zip(I[0], similarities) if idx != -1]
        
        elif self.backend == "hnswlib":
            labels, distances = self.index.knn_query(query_vector, k=k)
            # Convert distances to similarities
            if self.space == "cosine":
                similarities = 1.0 - distances[0] / 2.0
            else:
                similarities = distances[0]  # For inner product, distance is similarity
            return [(int(label), float(sim)) for label, sim in zip(labels[0], similarities)]
        
        else:
            # Simple numpy search
            similarities = []
            for item_id, vector in self.vectors:
                if self.space == "cosine":
                    similarity = fast_cosine_similarity(query_vector, vector)
                else:
                    similarity = np.dot(query_vector, vector)
                similarities.append((item_id, similarity))
            
            # Sort by similarity (highest first) and return top k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
    
    def get_metadata(self, item_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific item ID"""
        return self.id_to_metadata.get(item_id)
    
    def save(self, path: Optional[str] = None):
        """Save the index to disk"""
        save_path = path or self.index_path
        if not save_path:
            return False
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save metadata
            with open(f"{save_path}_metadata.json", 'w') as f:
                json.dump({
                    'id_to_metadata': {str(k): v for k, v in self.id_to_metadata.items()},
                    'next_id': self.next_id,
                    'dimension': self.dimension,
                    'backend': self.backend,
                    'space': self.space
                }, f)
            
            # Save the vectors based on backend
            if self.backend == "faiss":
                faiss.write_index(self.index, f"{save_path}.faiss")
            
            elif self.backend == "hnswlib":
                self.index.save_index(f"{save_path}.hnswlib")
            
            else:
                # Save numpy vectors
                np.save(f"{save_path}_vectors.npy", 
                       np.array([(id, vec) for id, vec in self.vectors], 
                               dtype=[('id', np.int32), ('vector', np.float32, (self.dimension,))]))
            
            return True
        except Exception as e:
            print(f"Error saving index: {e}")
            return False
    
    def load(self, path: Optional[str] = None):
        """Load the index from disk"""
        load_path = path or self.index_path
        if not load_path or not os.path.exists(f"{load_path}_metadata.json"):
            return False
        
        try:
            # Load metadata
            with open(f"{load_path}_metadata.json", 'r') as f:
                metadata = json.load(f)
                self.id_to_metadata = {int(k): v for k, v in metadata['id_to_metadata'].items()}
                self.next_id = metadata['next_id']
                self.dimension = metadata['dimension']
                self.backend = metadata.get('backend', self.backend)
                self.space = metadata.get('space', self.space)
            
            # Load vectors based on backend
            if self.backend == "faiss" and os.path.exists(f"{load_path}.faiss"):
                self.index = faiss.read_index(f"{load_path}.faiss")
            
            elif self.backend == "hnswlib" and os.path.exists(f"{load_path}.hnswlib"):
                self._init_index()  # Re-initialize the index
                self.index.load_index(f"{load_path}.hnswlib", max_elements=self.max_elements)
            
            elif os.path.exists(f"{load_path}_vectors.npy"):
                # Load numpy vectors
                data = np.load(f"{load_path}_vectors.npy")
                self.vectors = [(int(item['id']), item['vector']) for item in data]
            
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

class OptimizedVectorMemory:
    """High-performance vector memory system using optimized local components"""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', 
                 storage_path: str = None, dimension: int = 384):
        self.storage_path = storage_path or os.path.join(
            os.path.expanduser('~'), '.cache', 'llama4_vector_memory')
        self.dimension = dimension
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize embedding provider
        self.embedding_provider = OptimizedEmbeddingProvider(
            model_name=embedding_model,
            cache_dir=os.path.join(self.storage_path, 'embedding_cache')
        )
        
        # Update dimension based on the actual embedding size
        if self.embedding_provider.dimension:
            self.dimension = self.embedding_provider.dimension
        
        # Initialize vector index
        self.vector_index = OptimizedVectorIndex(
            dimension=self.dimension,
            index_path=os.path.join(self.storage_path, 'vector_index')
        )
        
        # Try to load existing index
        self.vector_index.load()
        
        # Metrics for performance monitoring
        self.metrics = {
            "items_added": 0,
            "searches_performed": 0,
            "avg_search_time": 0.0,
            "total_search_time": 0.0,
        }
    
    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a memory item to the vector store"""
        if not content:
            return None
        
        # Generate a unique ID for this memory
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Prepare metadata
        memory_metadata = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": timestamp,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "access_count": 0
        }
        
        # Get embedding for the content
        embedding = self.embedding_provider.get_embedding(content)
        
        # Add to vector index
        self.vector_index.add_item(embedding, memory_metadata)
        
        # Update metrics
        self.metrics["items_added"] += 1
        
        # Save index periodically
        if self.metrics["items_added"] % 100 == 0:
            self.vector_index.save()
        
        return memory_id
    
    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for memories similar to the query"""
        if not query:
            return []
        
        start_time = time.time()
        
        # Get embedding for the query
        query_embedding = self.embedding_provider.get_embedding(query)
        
        # Search the vector index
        search_results = self.vector_index.search(query_embedding, k=limit)
        
        # Get full results with metadata
        results = []
        for item_id, similarity in search_results:
            item_metadata = self.vector_index.get_metadata(item_id)
            if item_metadata:
                # Add similarity score and increment access count
                item_metadata["relevance_score"] = float(similarity)
                item_metadata["access_count"] += 1
                results.append(item_metadata)
        
        # Update metrics
        search_time = time.time() - start_time
        self.metrics["searches_performed"] += 1
        self.metrics["total_search_time"] += search_time
        self.metrics["avg_search_time"] = (
            self.metrics["total_search_time"] / self.metrics["searches_performed"]
        )
        
        return results
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID (less efficient than vector search)"""
        # Simple linear search through metadata
        for item_id, metadata in self.vector_index.id_to_metadata.items():
            if metadata.get("id") == memory_id:
                metadata["access_count"] += 1
                return metadata
        return None
    
    def forget_memory(self, memory_id: str) -> bool:
        """
        Mark a memory as forgotten
        
        Note: This doesn't actually remove from the index (would require rebuilding),
        but marks it as deleted in the metadata
        """
        for item_id, metadata in self.vector_index.id_to_metadata.items():
            if metadata.get("id") == memory_id:
                metadata["deleted"] = True
                metadata["deletion_timestamp"] = time.time()
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        stats = {
            "memory_count": len(self.vector_index.id_to_metadata),
            "active_count": sum(1 for m in self.vector_index.id_to_metadata.values() 
                              if not m.get("deleted", False)),
            "embedding_dimension": self.dimension,
            "backend": self.vector_index.backend,
            "performance": self.metrics,
            "embedding_cache": self.embedding_provider.get_cache_stats(),
        }
        return stats
    
    def save(self):
        """Save the memory system to disk"""
        return self.vector_index.save()

# Simple usage example
if __name__ == "__main__":
    memory = OptimizedVectorMemory()
    
    # Add some memories
    memory.add_memory("Python is a high-level programming language", 
                     {"category": "programming", "tags": ["python", "language"]})
    
    memory.add_memory("The Llama4 project is an advanced agent system with AI capabilities",
                     {"category": "project", "tags": ["llama4", "agent", "AI"]})
    
    memory.add_memory("Vector databases use embeddings to store and retrieve information",
                     {"category": "technology", "tags": ["vectors", "database", "embeddings"]})
    
    # Search for similar memories
    results = memory.search_memory("How do vector embeddings work?")
    
    print("Search Results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['content']} (Score: {result['relevance_score']:.4f})")
    
    # Print stats
    print("\nMemory Stats:")
    stats = memory.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Save to disk
    memory.save()