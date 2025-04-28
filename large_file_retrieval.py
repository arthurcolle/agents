#!/usr/bin/env python3
"""
Llama4 Memory Chunking System - Implements efficient retrieval against large files
for LLMs by dividing them into semantically meaningful chunks with hierarchical embeddings.
"""

import os
import sys
import hashlib
import time
import json
import sqlite3
import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass, field
import threading
import concurrent.futures
import re
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("llama4-memory")

@dataclass
class ChunkMetadata:
    """Metadata and context for a code or text chunk"""
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # 'code', 'text', 'mixed'
    parent_chunk_id: Optional[str] = None
    importance: float = 1.0
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    context_tokens: int = 0
    tags: Set[str] = field(default_factory=set)
    agent_id: Optional[str] = None
    last_modified: float = field(default_factory=time.time)
    related_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "parent_chunk_id": self.parent_chunk_id,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_access": self.last_access,
            "context_tokens": self.context_tokens,
            "tags": list(self.tags),
            "agent_id": self.agent_id,
            "last_modified": self.last_modified,
            "related_files": self.related_files
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkMetadata':
        """Create from dictionary"""
        metadata = cls(
            chunk_id=data["chunk_id"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            chunk_type=data["chunk_type"],
            parent_chunk_id=data["parent_chunk_id"],
            importance=data["importance"],
            access_count=data["access_count"],
            last_access=data["last_access"],
            context_tokens=data["context_tokens"],
            agent_id=data.get("agent_id"),
            last_modified=data.get("last_modified", time.time()),
            related_files=data.get("related_files", [])
        )
        metadata.tags = set(data["tags"])
        return metadata

@dataclass
class Chunk:
    """A chunk of content with metadata and embedding"""
    metadata: ChunkMetadata
    content: str
    embedding: Optional[np.ndarray] = None
    children: List[str] = field(default_factory=list)  # List of child chunk IDs
    
    def access(self) -> 'Chunk':
        """Record chunk access"""
        self.metadata.access_count += 1
        self.metadata.last_access = time.time()
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "metadata": self.metadata.to_dict(),
            "content": self.content,
            "embedding_base64": self._encode_embedding() if self.embedding is not None else None,
            "children": self.children,
        }
    
    def _encode_embedding(self) -> str:
        """Encode embedding as base64 string"""
        import base64
        return base64.b64encode(self.embedding.tobytes()).decode("utf-8")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """Create from dictionary"""
        metadata = ChunkMetadata.from_dict(data["metadata"])
        
        # Decode embedding if present
        embedding = None
        if data.get("embedding_base64"):
            import base64
            embedding_bytes = base64.b64decode(data["embedding_base64"])
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        return cls(
            metadata=metadata,
            content=data["content"],
            embedding=embedding,
            children=data.get("children", [])
        )

class Llama4Memory:
    """
    Efficient memory chunking system for large file retrieval with LLMs.
    
    This system breaks down large files into semantically meaningful chunks,
    creates hierarchical embeddings for efficient retrieval, and implements
    advanced memory management to minimize token usage. The chunks are stored
    in a way that enables agent modification and understanding of file relationships.
    """
    
    def __init__(self, 
                db_path: str = "llama4_knowledge.db",
                embedding_dimensions: int = 384,
                max_chunk_size: int = 1000,
                min_chunk_size: int = 50,
                embedding_batch_size: int = 10,
                use_mock_embeddings: bool = False,
                cache_embeddings: bool = True,
                auto_save: bool = True):
        """
        Initialize the Llama4Memory system.
        
        Args:
            db_path: Path to SQLite database for persistent storage
            embedding_dimensions: Dimension of embedding vectors
            max_chunk_size: Maximum lines per chunk
            min_chunk_size: Minimum lines per chunk
            embedding_batch_size: Number of chunks to embed in one batch
            use_mock_embeddings: Whether to use deterministic mock embeddings
            cache_embeddings: Whether to cache embeddings in memory
            auto_save: Whether to automatically save changes to disk
        """
        self.db_path = db_path
        self.embedding_dimensions = embedding_dimensions
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.embedding_batch_size = embedding_batch_size
        self.use_mock_embeddings = use_mock_embeddings
        self.cache_embeddings = cache_embeddings
        self.auto_save = auto_save
        
        # Memory storage
        self.chunks: Dict[str, Chunk] = {}
        self.file_index: Dict[str, List[str]] = {}  # file_path -> [chunk_ids]
        
        # Performance metrics
        self.metrics = {
            "total_files_processed": 0,
            "total_chunks_created": 0,
            "total_embeddings_generated": 0,
            "total_queries": 0,
            "cache_hits": 0,
            "avg_query_time": 0.0,
            "last_save_time": time.time()
        }
        
        # Thread lock for concurrency safety
        self.lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Auto-saving thread
        if self.auto_save:
            self._start_auto_save()
            
        logger.info(f"Llama4Memory initialized with database: {db_path}")
        logger.info(f"Chunk sizes: min={min_chunk_size}, max={max_chunk_size} lines")

    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        create_db = not os.path.exists(self.db_path)
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")  # Use Write-Ahead Logging for better concurrency
        
        if create_db:
            logger.info(f"Creating new database at {self.db_path}")
            
            # Create tables
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT,
                    content TEXT,
                    metadata TEXT,
                    embedding BLOB
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path)
            """)
            
            self.conn.commit()
        else:
            logger.info(f"Using existing database at {self.db_path}")
            self._load_from_database()

    def _load_from_database(self):
        """Load chunks and metrics from database"""
        with self.lock:
            logger.info("Loading chunks from database...")
            
            # Load chunks
            cursor = self.conn.execute("SELECT chunk_id, file_path, content, metadata, embedding FROM chunks")
            rows = cursor.fetchall()
            
            for chunk_id, file_path, content, metadata_json, embedding_blob in rows:
                try:
                    metadata_dict = json.loads(metadata_json)
                    metadata = ChunkMetadata.from_dict(metadata_dict)
                    
                    # Deserialize embedding if present
                    embedding = None
                    if embedding_blob is not None:
                        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    
                    # Create chunk and add to memory
                    chunk = Chunk(metadata=metadata, content=content, embedding=embedding)
                    self.chunks[chunk_id] = chunk
                    
                    # Update file index
                    if file_path not in self.file_index:
                        self.file_index[file_path] = []
                    self.file_index[file_path].append(chunk_id)
                except Exception as e:
                    logger.error(f"Error loading chunk {chunk_id}: {e}")
            
            # Update child references
            for chunk in self.chunks.values():
                if chunk.metadata.parent_chunk_id and chunk.metadata.parent_chunk_id in self.chunks:
                    parent = self.chunks[chunk.metadata.parent_chunk_id]
                    if chunk.metadata.chunk_id not in parent.children:
                        parent.children.append(chunk.metadata.chunk_id)
            
            # Load metrics
            cursor = self.conn.execute("SELECT key, value FROM metrics")
            for key, value in cursor.fetchall():
                try:
                    self.metrics[key] = json.loads(value)
                except:
                    self.metrics[key] = value
            
            logger.info(f"Loaded {len(self.chunks)} chunks from database")

    def process_file(self, file_path: str, force_reprocess: bool = False, agent_id: str = None) -> Tuple[bool, List[str]]:
        """
        Process a file into chunks and generate embeddings.
        
        Args:
            file_path: Path to the file to process
            force_reprocess: Whether to reprocess even if already in index
            agent_id: Optional identifier for the agent processing this file
            
        Returns:
            (success, list of chunk_ids)
        """
        with self.lock:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False, []
            
            # Check if file is already processed and we don't need to reprocess
            if file_path in self.file_index and not force_reprocess:
                logger.info(f"File already processed: {file_path}")
                return True, self.file_index[file_path]
            
            try:
                # If reprocessing, remove old chunks first
                if file_path in self.file_index:
                    self._remove_file_chunks(file_path)
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # Create chunks from file
                chunks = self._create_chunks(file_path, content, agent_id=agent_id)
                
                # Generate embeddings for chunks
                self._generate_embeddings(chunks)
                
                # Store chunks
                chunk_ids = []
                for chunk in chunks:
                    self.chunks[chunk.metadata.chunk_id] = chunk
                    chunk_ids.append(chunk.metadata.chunk_id)
                
                # Update file index
                self.file_index[file_path] = chunk_ids
                
                # Update metrics
                self.metrics["total_files_processed"] += 1
                self.metrics["total_chunks_created"] += len(chunks)
                
                logger.info(f"Processed {file_path} into {len(chunks)} chunks")
                
                # Save changes if auto-save is disabled (otherwise the auto-save thread will handle it)
                if not self.auto_save:
                    self.save_to_database()
                
                return True, chunk_ids
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                return False, []

    def _create_chunks(self, file_path: str, content: str, agent_id: str = None) -> List[Chunk]:
        """
        Create semantically meaningful chunks from file content.
        
        Args:
            file_path: Source file path
            content: File content
            agent_id: Optional identifier for the agent processing this file
            
        Returns:
            List of chunks
        """
        # Split content into lines
        lines = content.split('\n')
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Determine chunk type based on file extension
        is_code_file = file_ext in ['.py', '.js', '.ts', '.java', '.c', '.cpp', '.go', '.rs', '.php', '.rb', '.cs']
        
        # Choose chunking strategy based on file type
        if is_code_file:
            return self._create_code_chunks(file_path, lines, agent_id)
        else:
            return self._create_text_chunks(file_path, lines, agent_id)

    def _create_code_chunks(self, file_path: str, lines: List[str], agent_id: str = None) -> List[Chunk]:
        """
        Create chunks from code file with structure awareness.
        
        Args:
            file_path: Source file path
            lines: List of lines from the file
            agent_id: Optional identifier for the agent processing this file
            
        Returns:
            List of chunks
        """
        chunks = []
        current_chunk_lines = []
        current_chunk_start = 0
        boundaries = []
        
        # Find code structure boundaries (classes, functions, docstrings, etc.)
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check for class/function definitions, docstrings, etc.
            if re.match(r'^(class|def)\s+\w+', stripped) or re.match(r'^"""', stripped) or re.match(r"^'''", stripped):
                if current_chunk_lines and i > current_chunk_start + self.min_chunk_size:
                    boundaries.append((current_chunk_start, i - 1))
                    current_chunk_start = i
            
            current_chunk_lines.append(line)
            
            # Create a chunk if we hit max size
            if len(current_chunk_lines) >= self.max_chunk_size:
                boundaries.append((current_chunk_start, i))
                current_chunk_start = i + 1
                current_chunk_lines = []
        
        # Add final chunk if exists
        if current_chunk_lines:
            boundaries.append((current_chunk_start, current_chunk_start + len(current_chunk_lines) - 1))
        
        # Create chunks from boundaries
        for start, end in boundaries:
            chunk_content = '\n'.join(lines[start:end+1])
            chunk_id = self._generate_chunk_id(file_path, start, end)
            
            # Create metadata
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                file_path=file_path,
                start_line=start,
                end_line=end,
                chunk_type='code',
                context_tokens=self._estimate_tokens(chunk_content),
                agent_id=agent_id
            )
            
            # Create chunk
            chunk = Chunk(metadata=metadata, content=chunk_content)
            chunks.append(chunk)
        
        # Add parent-child relationships for nested definitions
        self._establish_parent_child_relationships(chunks)
        
        return chunks

    def _create_text_chunks(self, file_path: str, lines: List[str], agent_id: str = None) -> List[Chunk]:
        """
        Create chunks from text file with paragraph awareness.
        
        Args:
            file_path: Source file path
            lines: List of lines from the file
            agent_id: Optional identifier for the agent processing this file
            
        Returns:
            List of chunks
        """
        chunks = []
        current_chunk_lines = []
        current_chunk_start = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            current_chunk_lines.append(line)
            
            # Create a chunk at a paragraph boundary or if we hit max size
            if (len(line.strip()) == 0 and len(current_chunk_lines) > self.min_chunk_size) or \
               len(current_chunk_lines) >= self.max_chunk_size:
                
                # Skip if it's just a single blank line
                if len(current_chunk_lines) > 1 or len(current_chunk_lines[0].strip()) > 0:
                    chunk_content = '\n'.join(current_chunk_lines)
                    chunk_id = self._generate_chunk_id(file_path, current_chunk_start, current_chunk_start + len(current_chunk_lines) - 1)
                    
                    # Create metadata
                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        file_path=file_path,
                        start_line=current_chunk_start,
                        end_line=current_chunk_start + len(current_chunk_lines) - 1,
                        chunk_type='text',
                        context_tokens=self._estimate_tokens(chunk_content),
                        agent_id=agent_id
                    )
                    
                    # Create chunk
                    chunk = Chunk(metadata=metadata, content=chunk_content)
                    chunks.append(chunk)
                
                current_chunk_lines = []
                current_chunk_start = i + 1
            
            i += 1
        
        # Add final chunk if exists
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunk_id = self._generate_chunk_id(file_path, current_chunk_start, current_chunk_start + len(current_chunk_lines) - 1)
            
            # Create metadata
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                file_path=file_path,
                start_line=current_chunk_start,
                end_line=current_chunk_start + len(current_chunk_lines) - 1,
                chunk_type='text',
                context_tokens=self._estimate_tokens(chunk_content),
                agent_id=agent_id
            )
            
            # Create chunk
            chunk = Chunk(metadata=metadata, content=chunk_content)
            chunks.append(chunk)
        
        return chunks

    def _establish_parent_child_relationships(self, chunks: List[Chunk]):
        """
        Establish parent-child relationships between chunks based on line ranges.
        
        Args:
            chunks: List of chunks to process
        """
        # Sort chunks by size (larger chunks first)
        sorted_chunks = sorted(chunks, key=lambda c: (c.metadata.end_line - c.metadata.start_line), reverse=True)
        
        # For each chunk, find if it contains other chunks
        for parent_chunk in sorted_chunks:
            parent_start = parent_chunk.metadata.start_line
            parent_end = parent_chunk.metadata.end_line
            
            for child_chunk in sorted_chunks:
                # Skip self
                if parent_chunk.metadata.chunk_id == child_chunk.metadata.chunk_id:
                    continue
                
                child_start = child_chunk.metadata.start_line
                child_end = child_chunk.metadata.end_line
                
                # Check if child is contained within parent
                if child_start >= parent_start and child_end <= parent_end and \
                   not (child_start == parent_start and child_end == parent_end):
                    
                    # Add child to parent's children list
                    if child_chunk.metadata.chunk_id not in parent_chunk.children:
                        parent_chunk.children.append(child_chunk.metadata.chunk_id)
                    
                    # Set parent reference in child
                    child_chunk.metadata.parent_chunk_id = parent_chunk.metadata.chunk_id

    def _generate_chunk_id(self, file_path: str, start_line: int, end_line: int) -> str:
        """
        Generate a unique ID for a chunk based on its file path and line range.
        
        Args:
            file_path: Source file path
            start_line: Start line number
            end_line: End line number
            
        Returns:
            Unique chunk ID
        """
        # Create a unique identifier
        unique_str = f"{file_path}:{start_line}-{end_line}"
        hash_obj = hashlib.sha256(unique_str.encode())
        return hash_obj.hexdigest()[:16]  # Use first 16 chars of hex digest

    def _generate_embeddings(self, chunks: List[Chunk]):
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunks to embed
        """
        if not chunks:
            return
        
        # Process in batches
        for i in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[i:i+self.embedding_batch_size]
            
            if self.use_mock_embeddings:
                # Generate mock embeddings (deterministic based on content)
                for chunk in batch:
                    chunk.embedding = self._generate_mock_embedding(chunk.content)
            else:
                # Try to use actual embedding API here (placeholder for actual implementation)
                try:
                    # This would be replaced with actual embedding API call
                    for chunk in batch:
                        chunk.embedding = self._generate_mock_embedding(chunk.content)
                except Exception as e:
                    logger.warning(f"Failed to generate embeddings via API: {e}. Using mock embeddings.")
                    for chunk in batch:
                        chunk.embedding = self._generate_mock_embedding(chunk.content)
            
            # Update metrics
            self.metrics["total_embeddings_generated"] += len(batch)

    def _generate_mock_embedding(self, text: str) -> np.ndarray:
        """
        Generate a deterministic mock embedding for testing.
        
        Args:
            text: Text to embed
            
        Returns:
            Mock embedding vector
        """
        # Create a deterministic seed based on text hash
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert first N bytes to floats
        embedding = np.array([float(b) / 255.0 for b in hash_bytes[:self.embedding_dimensions]], dtype=np.float32)
        
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Simple approximation: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)

    def search(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Tuple[Chunk, float]]:
        """
        Search for chunks most relevant to a query.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of (chunk, similarity) tuples
        """
        start_time = time.time()
        
        with self.lock:
            if not self.chunks:
                logger.warning("No chunks available for search")
                return []
            
            # Generate embedding for query
            if self.use_mock_embeddings:
                query_embedding = self._generate_mock_embedding(query)
            else:
                # Placeholder for actual embedding API call
                try:
                    # This would be replaced with actual embedding API call
                    query_embedding = self._generate_mock_embedding(query)
                except Exception as e:
                    logger.warning(f"Failed to generate query embedding via API: {e}. Using mock embedding.")
                    query_embedding = self._generate_mock_embedding(query)
            
            # Calculate similarities
            similarities = []
            for chunk_id, chunk in self.chunks.items():
                if chunk.embedding is None:
                    continue
                
                similarity = self._calculate_similarity(query_embedding, chunk.embedding)
                if similarity >= threshold:
                    similarities.append((chunk, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Update metrics
            self.metrics["total_queries"] += 1
            query_time = time.time() - start_time
            self.metrics["avg_query_time"] = (self.metrics["avg_query_time"] * (self.metrics["total_queries"] - 1) + query_time) / self.metrics["total_queries"]
            
            # Return top-k results
            results = similarities[:top_k]
            
            # Update access counts for retrieved chunks
            for chunk, _ in results:
                chunk.access()
            
            logger.info(f"Search for '{query[:30]}...' returned {len(results)} results in {query_time:.3f}s")
            return results

    def search_by_file(self, query: str, file_path: str, top_k: int = 5, threshold: float = 0.5) -> List[Tuple[Chunk, float]]:
        """
        Search for chunks in a specific file most relevant to a query.
        
        Args:
            query: Search query
            file_path: File path to search in
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of (chunk, similarity) tuples
        """
        with self.lock:
            # Check if file exists in index
            if file_path not in self.file_index:
                logger.warning(f"File not found in index: {file_path}")
                return []
            
            # Get chunk IDs for file
            chunk_ids = self.file_index[file_path]
            
            # Generate embedding for query
            if self.use_mock_embeddings:
                query_embedding = self._generate_mock_embedding(query)
            else:
                # Placeholder for actual embedding API call
                try:
                    # This would be replaced with actual embedding API call
                    query_embedding = self._generate_mock_embedding(query)
                except Exception as e:
                    logger.warning(f"Failed to generate query embedding via API: {e}. Using mock embedding.")
                    query_embedding = self._generate_mock_embedding(query)
            
            # Calculate similarities for file chunks
            similarities = []
            for chunk_id in chunk_ids:
                chunk = self.chunks.get(chunk_id)
                if chunk is None or chunk.embedding is None:
                    continue
                
                similarity = self._calculate_similarity(query_embedding, chunk.embedding)
                if similarity >= threshold:
                    similarities.append((chunk, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Update access counts for retrieved chunks
            for chunk, _ in similarities[:top_k]:
                chunk.access()
            
            # Return top-k results
            return similarities[:top_k]

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity (0-1)
        """
        # Ensure embeddings are normalized
        if np.linalg.norm(embedding1) > 0:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
        if np.linalg.norm(embedding2) > 0:
            embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)

    def find_related_documents(self, chunk_id: str = None, file_path: str = None, threshold: float = 0.7, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Find documents that are related to a specific chunk or file.
        
        Args:
            chunk_id: ID of the chunk to find related documents for
            file_path: Path of the file to find related documents for
            threshold: Minimum similarity threshold (0-1)
            top_k: Maximum number of results to return
            
        Returns:
            List of (chunk, similarity) tuples
        """
        with self.lock:
            # Get the source chunk or file embeddings
            source_embedding = None
            
            if chunk_id and chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                if chunk.embedding is not None:
                    source_embedding = chunk.embedding
            elif file_path:
                # If file path is provided, use the combined embedding of all chunks in the file
                if file_path in self.file_index:
                    chunk_ids = self.file_index[file_path]
                    embeddings = []
                    
                    for cid in chunk_ids:
                        if cid in self.chunks and self.chunks[cid].embedding is not None:
                            embeddings.append(self.chunks[cid].embedding)
                    
                    if embeddings:
                        # Average the embeddings
                        source_embedding = np.mean(embeddings, axis=0)
                        
                        # Normalize
                        norm = np.linalg.norm(source_embedding)
                        if norm > 0:
                            source_embedding = source_embedding / norm
            
            if source_embedding is None:
                return []
            
            # Calculate similarities with all other chunks
            similarities = []
            for other_id, other_chunk in self.chunks.items():
                # Skip the source chunk and chunks from the same file
                if (chunk_id and other_id == chunk_id) or \
                   (file_path and other_chunk.metadata.file_path == file_path):
                    continue
                
                if other_chunk.embedding is None:
                    continue
                
                similarity = self._calculate_similarity(source_embedding, other_chunk.embedding)
                if similarity >= threshold:
                    similarities.append((other_chunk, similarity))
            
            # Also check the explicitly defined related files
            if chunk_id and chunk_id in self.chunks:
                chunk = self.chunks[chunk_id]
                for related_path in chunk.metadata.related_files:
                    if related_path not in self.file_index:
                        continue
                    
                    for related_chunk_id in self.file_index[related_path]:
                        related_chunk = self.chunks.get(related_chunk_id)
                        if related_chunk is not None and related_chunk.embedding is not None:
                            # Boost similarity for explicitly related files
                            similarity = self._calculate_similarity(source_embedding, related_chunk.embedding) * 1.2
                            if similarity >= threshold and (related_chunk, similarity) not in similarities:
                                similarities.append((related_chunk, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k results
            return similarities[:top_k]
    
    def get_file_chunks(self, file_path: str) -> List[Chunk]:
        """
        Get all chunks for a specific file.
        
        Args:
            file_path: File path
            
        Returns:
            List of chunks
        """
        with self.lock:
            # Check if file exists in index
            if file_path not in self.file_index:
                logger.warning(f"File not found in index: {file_path}")
                return []
            
            # Get chunk IDs for file
            chunk_ids = self.file_index[file_path]
            
            # Get chunks
            chunks = []
            for chunk_id in chunk_ids:
                chunk = self.chunks.get(chunk_id)
                if chunk is not None:
                    chunks.append(chunk)
            
            return chunks

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """
        Get a chunk by its ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk or None if not found
        """
        with self.lock:
            return self.chunks.get(chunk_id)

    def _remove_file_chunks(self, file_path: str):
        """
        Remove all chunks for a specific file.
        
        Args:
            file_path: File path to remove
        """
        with self.lock:
            # Check if file exists in index
            if file_path not in self.file_index:
                return
            
            # Get chunk IDs for file
            chunk_ids = self.file_index[file_path]
            
            # Remove chunks from memory
            for chunk_id in chunk_ids:
                if chunk_id in self.chunks:
                    del self.chunks[chunk_id]
            
            # Remove file from index
            del self.file_index[file_path]
            
            # Remove from database
            self.conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            self.conn.commit()
            
            logger.info(f"Removed {len(chunk_ids)} chunks for file: {file_path}")

    def save_to_database(self):
        """Save all chunks and metrics to database"""
        with self.lock:
            try:
                # Start a transaction
                self.conn.execute("BEGIN TRANSACTION")
                
                # Save chunks
                for chunk_id, chunk in self.chunks.items():
                    # Serialize metadata
                    metadata_json = json.dumps(chunk.metadata.to_dict())
                    
                    # Serialize embedding if present
                    embedding_blob = None
                    if chunk.embedding is not None:
                        embedding_blob = chunk.embedding.tobytes()
                    
                    # Upsert the chunk
                    self.conn.execute("""
                        INSERT OR REPLACE INTO chunks (chunk_id, file_path, content, metadata, embedding)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        chunk_id,
                        chunk.metadata.file_path,
                        chunk.content,
                        metadata_json,
                        embedding_blob
                    ))
                
                # Save metrics
                for key, value in self.metrics.items():
                    value_json = json.dumps(value)
                    self.conn.execute("""
                        INSERT OR REPLACE INTO metrics (key, value)
                        VALUES (?, ?)
                    """, (key, value_json))
                
                # Commit transaction
                self.conn.commit()
                
                # Update last save time
                self.metrics["last_save_time"] = time.time()
                
                logger.info(f"Saved {len(self.chunks)} chunks to database")
                return True
            except Exception as e:
                # Rollback on error
                self.conn.rollback()
                logger.error(f"Error saving to database: {e}")
                return False

    def _start_auto_save(self):
        """Start a background thread for automatic saving"""
        def auto_save_worker():
            while self.auto_save:
                try:
                    # Sleep for 30 seconds
                    time.sleep(30)
                    
                    # Save if changes have been made
                    if time.time() - self.metrics["last_save_time"] > 30:
                        self.save_to_database()
                except Exception as e:
                    logger.error(f"Error in auto-save worker: {e}")
        
        # Start thread
        thread = threading.Thread(target=auto_save_worker, daemon=True)
        thread.start()
        logger.info("Auto-save thread started")

    def process_directory(self, directory_path: str, extensions: List[str] = None, recurse: bool = True) -> Tuple[int, int]:
        """
        Process all files in a directory.
        
        Args:
            directory_path: Directory path to process
            extensions: List of file extensions to process (e.g. ['.py', '.txt'])
            recurse: Whether to recursively process subdirectories
            
        Returns:
            (files_processed, files_failed)
        """
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return 0, 0
        
        files_processed = 0
        files_failed = 0
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Function to process a single file
            def process_file_task(file_path):
                success, _ = self.process_file(file_path)
                return file_path, success
            
            # Get all files to process
            file_paths = []
            for root, dirs, files in os.walk(directory_path):
                if not recurse and root != directory_path:
                    continue
                
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check extension if specified
                    if extensions:
                        file_ext = os.path.splitext(file_path)[1].lower()
                        if file_ext not in extensions:
                            continue
                    
                    file_paths.append(file_path)
            
            # Submit all file processing tasks
            future_to_file = {executor.submit(process_file_task, fp): fp for fp in file_paths}
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    _, success = future.result()
                    if success:
                        files_processed += 1
                    else:
                        files_failed += 1
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    files_failed += 1
        
        logger.info(f"Processed {files_processed} files, {files_failed} failed from directory: {directory_path}")
        return files_processed, files_failed

    def process_file_content(self, file_path: str, content: str, agent_id: str = None, related_files: List[str] = None) -> Tuple[bool, List[str]]:
        """
        Process file content without reading from disk. Can be used by agents to add content
        to the memory system directly.
        
        Args:
            file_path: Virtual file path (identifier)
            content: File content
            agent_id: Optional identifier for the agent that generated this content
            related_files: List of related file paths that this content references
            
        Returns:
            (success, list of chunk_ids)
        """
        with self.lock:
            try:
                # If file already exists, remove old chunks first
                if file_path in self.file_index:
                    self._remove_file_chunks(file_path)
                
                # Create chunks from content
                chunks = self._create_chunks(file_path, content)
                
                # Add agent and relationship metadata
                for chunk in chunks:
                    if agent_id:
                        chunk.metadata.agent_id = agent_id
                    
                    if related_files:
                        chunk.metadata.related_files = related_files
                
                # Generate embeddings for chunks
                self._generate_embeddings(chunks)
                
                # Store chunks
                chunk_ids = []
                for chunk in chunks:
                    self.chunks[chunk.metadata.chunk_id] = chunk
                    chunk_ids.append(chunk.metadata.chunk_id)
                
                # Update file index
                self.file_index[file_path] = chunk_ids
                
                # Update metrics
                self.metrics["total_files_processed"] += 1
                self.metrics["total_chunks_created"] += len(chunks)
                
                logger.info(f"Processed content for {file_path} into {len(chunks)} chunks")
                
                # Save changes if auto-save is disabled
                if not self.auto_save:
                    self.save_to_database()
                
                return True, chunk_ids
            except Exception as e:
                logger.error(f"Error processing content for {file_path}: {e}")
                return False, []

    def update_chunk(self, chunk_id: str, new_content: str, agent_id: str = None) -> bool:
        """
        Update a chunk's content and regenerate its embedding. This allows agents
        to modify content while preserving relationships and metadata.
        
        Args:
            chunk_id: ID of the chunk to update
            new_content: New content for the chunk
            agent_id: Optional identifier for the agent making this update
            
        Returns:
            Success status
        """
        with self.lock:
            if chunk_id not in self.chunks:
                logger.warning(f"Chunk not found: {chunk_id}")
                return False
            
            try:
                chunk = self.chunks[chunk_id]
                
                # Update content
                chunk.content = new_content
                
                # Update metadata
                chunk.metadata.last_modified = time.time()
                if agent_id:
                    chunk.metadata.agent_id = agent_id
                
                # Regenerate embedding
                if self.use_mock_embeddings:
                    chunk.embedding = self._generate_mock_embedding(new_content)
                else:
                    try:
                        # This would be replaced with actual embedding API call
                        chunk.embedding = self._generate_mock_embedding(new_content)
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding via API: {e}. Using mock embedding.")
                        chunk.embedding = self._generate_mock_embedding(new_content)
                
                logger.info(f"Updated chunk {chunk_id} with new content")
                
                # Auto-save if enabled
                if not self.auto_save:
                    self.save_to_database()
                
                return True
            except Exception as e:
                logger.error(f"Error updating chunk {chunk_id}: {e}")
                return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Statistics dictionary
        """
        with self.lock:
            file_count = len(self.file_index)
            chunk_count = len(self.chunks)
            
            # Calculate total storage
            total_content_size = sum(len(chunk.content.encode('utf-8')) for chunk in self.chunks.values())
            total_embedding_size = sum(chunk.embedding.nbytes if chunk.embedding is not None else 0 for chunk in self.chunks.values())
            
            # Calculate average chunk size
            avg_chunk_size = total_content_size / max(1, chunk_count)
            
            # Create stats dictionary
            stats = {
                "total_files": file_count,
                "total_chunks": chunk_count,
                "total_content_size_bytes": total_content_size,
                "total_embedding_size_bytes": total_embedding_size,
                "average_chunk_size_bytes": avg_chunk_size,
                "db_path": self.db_path,
                "metrics": self.metrics
            }
            
            return stats

    def close(self):
        """Close database connection and save changes"""
        with self.lock:
            # Save any unsaved changes
            self.save_to_database()
            
            # Close database connection
            if self.conn:
                self.conn.close()
                self.conn = None
            
            logger.info("Memory system closed")

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

def main():
    """Example usage of Llama4Memory"""
    import argparse
    
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Llama4 Memory Chunking System")
    parser.add_argument("--db", type=str, default="llama4_knowledge.db", help="Path to database file")
    parser.add_argument("--dir", type=str, help="Directory to process")
    parser.add_argument("--file", type=str, help="File to process")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock embeddings")
    parser.add_argument("--agent", type=str, help="Agent ID for file processing")
    parser.add_argument("--find-related", type=str, help="Find documents related to the specified file")
    
    args = parser.parse_args()
    
    # Initialize memory system
    memory = Llama4Memory(
        db_path=args.db,
        use_mock_embeddings=args.mock,
        auto_save=True
    )
    
    try:
        # Process directory if specified
        if args.dir:
            memory.process_directory(args.dir, extensions=['.py', '.txt', '.md', '.json', '.js', '.ts', '.html', '.css'])
        
        # Process file if specified
        if args.file:
            success, chunk_ids = memory.process_file(args.file, agent_id=args.agent)
            if success:
                print(f"Processed {args.file} into {len(chunk_ids)} chunks")
            else:
                print(f"Failed to process {args.file}")
        
        # Find related documents if specified
        if args.find_related:
            results = memory.find_related_documents(file_path=args.find_related, top_k=args.top_k)
            
            print(f"\nDocuments related to: '{args.find_related}'")
            print("-" * 50)
            
            for i, (chunk, similarity) in enumerate(results):
                print(f"{i+1}. [Score: {similarity:.4f}] {chunk.metadata.file_path}:{chunk.metadata.start_line}-{chunk.metadata.end_line}")
                # Print agent if available
                if chunk.metadata.agent_id:
                    print(f"   Agent: {chunk.metadata.agent_id}")
                # Print a snippet of the content
                content_lines = chunk.content.split('\n')
                snippet = '\n'.join(content_lines[:min(3, len(content_lines))])
                print(f"   {snippet}...")
                print()
        
        # Search if query specified
        if args.query:
            results = memory.search(args.query, top_k=args.top_k)
            
            print(f"\nSearch results for: '{args.query}'")
            print("-" * 50)
            
            for i, (chunk, similarity) in enumerate(results):
                print(f"{i+1}. [Score: {similarity:.4f}] {chunk.metadata.file_path}:{chunk.metadata.start_line}-{chunk.metadata.end_line}")
                # Print agent if available
                if chunk.metadata.agent_id:
                    print(f"   Agent: {chunk.metadata.agent_id}")
                # Print a snippet of the content
                content_lines = chunk.content.split('\n')
                snippet = '\n'.join(content_lines[:min(3, len(content_lines))])
                print(f"   {snippet}...")
                print()
        
        # Print stats
        stats = memory.get_stats()
        print("\nMemory System Statistics:")
        print(f"Total Files: {stats['total_files']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Total Content Size: {stats['total_content_size_bytes'] / 1024:.2f} KB")
        print(f"Total Embedding Size: {stats['total_embedding_size_bytes'] / 1024:.2f} KB")
        print(f"Average Chunk Size: {stats['average_chunk_size_bytes']:.2f} bytes")
        print(f"Average Query Time: {stats['metrics']['avg_query_time']:.4f} seconds")
    
    finally:
        # Close memory system
        memory.close()

if __name__ == "__main__":
    main()