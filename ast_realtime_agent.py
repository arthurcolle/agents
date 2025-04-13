#!/usr/bin/env python3
"""
AST Realtime Agent: An intelligent code analysis and exploration tool using AST parsing and LLM insights

Features:
- AST-based code structure analysis with semantic search capabilities
- LLM-powered code understanding and exploration
- Hierarchical context management with memory prioritization
- Interactive exploration mode with natural language queries
- Tool integration for enhanced code navigation
- SQLite persistence for long-term knowledge storage
"""

import ast
import os
import sys
import json
import time
import logging
import numpy as np
import sqlite3
import asyncio
import argparse
import uuid
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Third-party imports with fallbacks
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    logger.warning("Pydantic not found. Some features will be limited.")
    PYDANTIC_AVAILABLE = False
    # Simple BaseModel fallback
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    Field = lambda **kwargs: None

# Try to import OpenAI for enhanced code understanding
try:
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI package found. Enhanced code understanding enabled.")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.info("OpenAI package not found. Using basic analysis mode.")

# Import local Jina client implementation if available, otherwise use jina package
try:
    from modules.jina_client import JinaClient
    CUSTOM_JINA_CLIENT = True
    logger.info("Using custom Jina client from modules.jina_client")
except ImportError:
    CUSTOM_JINA_CLIENT = False
    try:
        from jina import Document, DocumentArray
        from jina.clients import Client
        logger.info("Using jina package client")
    except ImportError:
        logger.error("Jina not found. Install with: pip install jina")
        sys.exit(1)

# Import Document class conditionally to avoid errors
if not CUSTOM_JINA_CLIENT:
    from jina import Document
else:
    # Define a minimal Document class for compatibility when using custom client
    class Document:
        def __init__(self, text=None):
            self.text = text
            self.embedding = None

# Constants
MAX_CONTEXT_SIZE = 8192  # Maximum tokens in context
RELEVANCE_THRESHOLD = 0.7  # Similarity threshold for relevance
CACHE_EXPIRY = 300  # Cache expiry time in seconds (5 minutes)
EMBEDDING_DIMENSION = 768  # Default dimension for embeddings

class ASTNode:
    """Represents a node in the AST with embedding and metadata."""
    
    def __init__(self, node: ast.AST, source_code: str, path: str = "", parent: Optional['ASTNode'] = None):
        self.ast_node = node
        self.node_type = type(node).__name__
        self.source_code = source_code
        self.embedding = None
        self.children = []
        self.parent = parent
        self.path = path
        self.line_start, self.line_end = self._get_line_range(node)
        self.importance_score = 0.0
        self.last_accessed = time.time()
        self.access_count = 0
        
    def _get_line_range(self, node: ast.AST) -> Tuple[int, int]:
        """Get the line range of an AST node."""
        if hasattr(node, 'lineno'):
            start = node.lineno
            if hasattr(node, 'end_lineno'):
                end = node.end_lineno
            else:
                # Approximation for Python < 3.8
                end = start + self.source_code[start-1:].count('\n', 0, 50)
            return start, end
        return 0, 0
    
    def get_source_segment(self) -> str:
        """Get the source code segment for this node."""
        if self.line_start == 0:
            return ""
        lines = self.source_code.splitlines()
        return "\n".join(lines[self.line_start-1:self.line_end])
    
    def get_representation(self) -> str:
        """Get a textual representation of this node for embedding."""
        source = self.get_source_segment()
        if not source:
            return f"{self.node_type} at unknown location"
        
        # Truncate long source segments
        if len(source) > 1000:
            source = source[:500] + "..." + source[-500:]
            
        return f"{self.node_type}: {source}"
    
    def update_access_time(self):
        """Update the last access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1
        
    def calculate_importance(self, query_embedding=None):
        """Calculate importance score based on recency, frequency and relevance."""
        # Recency factor (time decay)
        time_factor = np.exp(-(time.time() - self.last_accessed) / CACHE_EXPIRY)
        
        # Frequency factor
        freq_factor = np.log1p(self.access_count)
        
        # Relevance to current query if available
        relevance = 0.0
        if query_embedding is not None and self.embedding is not None:
            relevance = np.dot(query_embedding, self.embedding)
        
        # Combine factors
        self.importance_score = (0.3 * time_factor + 0.3 * freq_factor + 0.4 * relevance)
        return self.importance_score

class ASTEmbedder:
    """Embeds AST nodes using advanced embedding services with parallel processing capabilities."""
    
    def __init__(self, model_name: str = "jina-v2-clip", api_key: Optional[str] = None):
        self.model_name = model_name
        self.embedding_cache = {}  # Cache for embeddings
        self.api_key = api_key or os.getenv("JINA_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.embedding_lock = asyncio.Lock()  # Lock for thread-safe concurrent embedding operations
        
        # Set up appropriate client based on availability
        if OPENAI_AVAILABLE and 'openai' in self.model_name.lower():
            try:
                # Handle different versions of the OpenAI client
                try:
                    # First try without proxies parameter
                    self.openai_client = openai.AsyncClient(api_key=self.api_key)
                except TypeError:
                    # If that fails, try with simpler initialization
                    self.openai_client = openai.AsyncClient(api_key=self.api_key)
                logger.info(f"Using OpenAI for embeddings with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None
                
        # Setup Jina client as fallback or primary if specified
        if CUSTOM_JINA_CLIENT:
            try:
                # Don't pass proxies parameter since some versions of the client don't support it
                self.jina_client = JinaClient(token=self.api_key)
                logger.info(f"Using custom JinaClient for embeddings with model {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize custom JinaClient: {e}")
                self.jina_client = None
        else:
            try:
                self.jina_client = Client(host=f"grpc://api.jina.ai:8443")
                logger.info(f"Using Jina package Client for embeddings with model {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Jina Client: {e}")
                self.jina_client = None
        
        # Initialize semaphore for rate limiting
        self.embedding_semaphore = asyncio.Semaphore(10)  # Limit concurrent embedding requests
    
    async def _embed_with_openai(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using OpenAI's embedding API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        try:
            # Default to text-embedding-3-large if not specified
            model = self.model_name if 'text-embedding' in self.model_name else "text-embedding-3-large"
            
            # Process in batches of 1000 to avoid API limits
            batch_size = 1000
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                response = await self.openai_client.embeddings.create(
                    input=batch,
                    model=model
                )
                
                # Extract and add embeddings
                batch_embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            return all_embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings from OpenAI: {e}")
            raise
    
    async def _embed_with_jina(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using Jina's embedding API."""
        if CUSTOM_JINA_CLIENT and self.jina_client:
            try:
                # Use Jina's embedding endpoint directly
                response = await self.jina_client.embed(texts)
                
                if isinstance(response, dict) and 'embeddings' in response:
                    return [np.array(emb) for emb in response['embeddings']]
                elif isinstance(response, list) and len(response) > 0:
                    # Handle array response format
                    return [np.array(emb) for emb in response]
                else:
                    logger.error(f"Unexpected response format from Jina: {type(response)}")
                    raise ValueError("Invalid response format from Jina embedding API")
            except Exception as e:
                logger.error(f"Error with Jina embedding: {e}")
                raise
        elif self.jina_client:
            try:
                # Use standard Jina client
                docs = DocumentArray([Document(text=text) for text in texts])
                response = await self.jina_client.post("/embed", 
                                                     inputs=docs, 
                                                     parameters={"model": self.model_name})
                return [doc.embedding for doc in response]
            except Exception as e:
                logger.error(f"Error with Jina package client: {e}")
                raise
        else:
            logger.error("No Jina client available")
            raise ValueError("No Jina client initialized")
    
    async def _get_syntax_aware_embedding(self, node: ASTNode) -> np.ndarray:
        """
        Generate a syntax-aware embedding that considers code semantics
        using a combination of AST structure and text content.
        """
        # Get the basic representation
        text = node.get_representation()
        
        # Add syntax context based on node type and properties
        syntax_context = f"Type: {node.node_type}\n"
        
        # Add more specific context based on node type
        if isinstance(node.ast_node, ast.FunctionDef):
            syntax_context += f"Function name: {node.ast_node.name}\n"
            syntax_context += f"Arguments: {[a.arg for a in node.ast_node.args.args]}\n"
            
        elif isinstance(node.ast_node, ast.ClassDef):
            syntax_context += f"Class name: {node.ast_node.name}\n"
            bases = []
            for base in node.ast_node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
            syntax_context += f"Bases: {bases}\n"
            
        elif isinstance(node.ast_node, ast.ImportFrom):
            syntax_context += f"Import from: {node.ast_node.module}\n"
            syntax_context += f"Names: {[n.name for n in node.ast_node.names]}\n"
            
        elif isinstance(node.ast_node, ast.Import):
            syntax_context += f"Import: {[n.name for n in node.ast_node.names]}\n"
            
        # Combine with the actual code for a richer representation
        enriched_text = f"{syntax_context}\nCode:\n{text}"
        
        # Generate embedding for the enriched text
        if self.openai_client:
            embeddings = await self._embed_with_openai([enriched_text])
            return embeddings[0]
        elif self.jina_client:
            embeddings = await self._embed_with_jina([enriched_text])
            return embeddings[0]
        else:
            # Fallback to a deterministic but more sophisticated embedding
            import hashlib
            
            # Create a hash of the text for deterministic results
            hasher = hashlib.sha256()
            hasher.update(enriched_text.encode('utf-8'))
            hash_bytes = hasher.digest()
            
            # Create a deterministic but structured embedding from the hash
            embedding = np.zeros(EMBEDDING_DIMENSION)
            for i, byte in enumerate(hash_bytes):
                if i < EMBEDDING_DIMENSION:
                    embedding[i] = (byte / 255.0) * 2 - 1  # Scale to [-1, 1]
                
            # Ensure the embedding is normalized
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
    async def embed_node(self, node: ASTNode) -> np.ndarray:
        """Embed a single AST node with semantic understanding."""
        text = node.get_representation()
        
        # Check cache first
        if text in self.embedding_cache:
            node.embedding = self.embedding_cache[text]
            return self.embedding_cache[text]
        
        try:
            # Get syntax-aware embedding
            embedding = await self._get_syntax_aware_embedding(node)
            
            # Cache the embedding
            self.embedding_cache[text] = embedding
            node.embedding = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding node: {e}")
            # Provide a fallback embedding
            embedding = np.zeros(EMBEDDING_DIMENSION)
            # Set a few values based on node type hash for minimal differentiation
            node_type_hash = hash(node.node_type) % EMBEDDING_DIMENSION
            embedding[node_type_hash % EMBEDDING_DIMENSION] = 1.0
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            # Cache and return
            self.embedding_cache[text] = embedding
            node.embedding = embedding
            return embedding
    
    async def embed_nodes(self, nodes: List[ASTNode]) -> List[np.ndarray]:
        """Embed multiple AST nodes in batch efficiently with parallel processing."""
        # Map nodes to texts and track which nodes need embedding
        texts = [node.get_representation() for node in nodes]
        to_embed_indices = []
        to_embed_nodes = []
        
        # Check cache and collect nodes needing embedding with thread safety
        async with self.embedding_lock:
            for i, text in enumerate(texts):
                cache_key = f"{text}:{self.model_name}"
                if cache_key in self.embedding_cache:
                    nodes[i].embedding = self.embedding_cache[cache_key]
                    self.cache_hits += 1
                else:
                    to_embed_indices.append(i)
                    to_embed_nodes.append(nodes[i])
                    self.cache_misses += 1
        
        # If nothing to embed, we're done
        if not to_embed_nodes:
            return [node.embedding for node in nodes]
            
        try:
            # Process in appropriate batch sizes with improved parallelism
            batch_size = 50  # Increased batch size for better throughput
            all_results = []
            
            # Prepare a list of tasks for concurrent processing
            async def process_node(node):
                embedding = await self._get_syntax_aware_embedding(node)
                text = node.get_representation()
                cache_key = f"{text}:{self.model_name}"
                # Cache the result with thread safety
                async with self.embedding_lock:
                    self.embedding_cache[cache_key] = embedding
                node.embedding = embedding
                return embedding
            
            # Process batches with controlled concurrency
            for i in range(0, len(to_embed_nodes), batch_size):
                batch = to_embed_nodes[i:i+batch_size]
                
                # Process each batch with controlled parallelism
                batch_tasks = []
                for node in batch:
                    # Use semaphore to limit concurrent API calls
                    async with self.embedding_semaphore:
                        batch_tasks.append(process_node(node))
                
                # Wait for all tasks in this batch to complete
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle results, skipping any that raised exceptions
                for result in batch_results:
                    if not isinstance(result, Exception):
                        all_results.append(result)
                    else:
                        logger.warning(f"Error embedding node: {result}")
                        all_results.append(None)
                    
            # Map results back to original node order
            for i, orig_idx in enumerate(to_embed_indices):
                if i < len(all_results) and all_results[i] is not None:
                    nodes[orig_idx].embedding = all_results[i]
                else:
                    # Handle missing or error embeddings
                    self._create_fallback_embedding(nodes[orig_idx])
            
            return [node.embedding for node in nodes]
            
        except Exception as e:
            logger.error(f"Error batch embedding nodes: {e}")
            
            # Generate fallback embeddings for remaining nodes
            for i in to_embed_indices:
                if nodes[i].embedding is None:
                    self._create_fallback_embedding(nodes[i])
            
            return [node.embedding for node in nodes]
    
    def _create_fallback_embedding(self, node: ASTNode):
        """Create a fallback embedding when normal embedding fails."""
        text = node.get_representation()
        cache_key = f"{text}:{self.model_name}"
        
        # Create a fallback embedding based on node type
        embedding = np.zeros(EMBEDDING_DIMENSION)
        node_type_hash = hash(node.node_type) % EMBEDDING_DIMENSION
        embedding[node_type_hash % EMBEDDING_DIMENSION] = 1.0
        embedding = embedding / np.linalg.norm(embedding)
        
        # Cache the fallback result
        self.embedding_cache[cache_key] = embedding
        node.embedding = embedding
        return embedding

class ASTParser:
    """Parses source code into hierarchical AST nodes."""
    
    def __init__(self):
        self.root_nodes = {}  # Map of file paths to their root nodes
        
    def parse_file(self, file_path: str) -> Optional[ASTNode]:
        """Parse a file into AST nodes with hierarchical structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            root = ASTNode(tree, source_code, path=file_path)
            self._build_hierarchy(root, tree, source_code, file_path)
            self.root_nodes[file_path] = root
            return root
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return None
            
    def __del__(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def _build_hierarchy(self, parent_node: ASTNode, ast_node: ast.AST, source_code: str, file_path: str):
        """Recursively build the AST node hierarchy."""
        for field, value in ast.iter_fields(ast_node):
            if isinstance(value, ast.AST):
                child = ASTNode(value, source_code, path=file_path, parent=parent_node)
                parent_node.children.append(child)
                self._build_hierarchy(child, value, source_code, file_path)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        child = ASTNode(item, source_code, path=file_path, parent=parent_node)
                        parent_node.children.append(child)
                        self._build_hierarchy(child, item, source_code, file_path)

class CodeMemory(BaseModel):
    """Structure to represent code snippets in memory with metadata and embeddings."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str
    code_segment: str
    node_type: str
    line_start: int
    line_end: int
    embedding: Optional[List[float]] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    importance_score: float = 0.0
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        result = {
            "id": self.id,
            "file_path": self.file_path,
            "code_segment": self.code_segment[:200] + "..." if len(self.code_segment) > 200 else self.code_segment,
            "node_type": self.node_type,
            "line_range": f"{self.line_start}-{self.line_end}",
            "importance": self.importance_score,
            "access_count": self.access_count,
            "last_accessed": datetime.fromtimestamp(self.last_accessed).isoformat()
        }
        
        # Add summary if available in metadata
        if "summary" in self.metadata:
            result["summary"] = self.metadata["summary"]
            
        return result

class CodeAnalyzer:
    """Analyzes code semantics using LLM if available."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        self.analysis_cache = {}  # Cache for code analysis results
        
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.openai_client = openai.AsyncClient(api_key=self.api_key)
                logger.info("Initialized OpenAI client for code analysis")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
    
    async def summarize_code(self, node: ASTNode) -> Dict[str, Any]:
        """Generate a concise summary of the code using OpenAI."""
        if not self.openai_client:
            return {"summary": f"{node.node_type} code segment"}
            
        # Check cache first
        cache_key = f"{node.path}:{node.line_start}-{node.line_end}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
            
        try:
            code = node.get_source_segment()
            context = ""
            
            # Add contextual info based on node type
            if isinstance(node.ast_node, ast.FunctionDef):
                context = f"Function definition: {node.ast_node.name}"
            elif isinstance(node.ast_node, ast.ClassDef):
                context = f"Class definition: {node.ast_node.name}"
            elif isinstance(node.ast_node, ast.Import) or isinstance(node.ast_node, ast.ImportFrom):
                context = "Import statement"
            
            # Prepare the system message with syntax highlighting info
            system_message = """You are a code analysis assistant that specializes in Python.
            Analyze the provided code and provide a short, concise summary of what it does.
            Be specific but brief (1-2 lines maximum).
            Include key information like function purpose, class behavior, or important logic.
            Output in JSON format with 'summary', 'complexity', 'purpose', and 'dependencies' fields."""
            
            # Get analysis from OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Context: {context}\n\nCode:\n\n```python\n{code}\n```"}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            # Extract and process the response
            try:
                analysis = json.loads(response.choices[0].message.content)
                
                # Ensure we have the required fields
                if "summary" not in analysis:
                    analysis["summary"] = f"{node.node_type} at {node.path}:{node.line_start}"
                    
                # Cache the result
                self.analysis_cache[cache_key] = analysis
                return analysis
                
            except json.JSONDecodeError:
                # Fallback summary if JSON parsing fails
                fallback = {"summary": f"{node.node_type} code segment at lines {node.line_start}-{node.line_end}"}
                self.analysis_cache[cache_key] = fallback
                return fallback
                
        except Exception as e:
            logger.error(f"Error analyzing code with OpenAI: {e}")
            fallback = {"summary": f"{node.node_type} at {node.path}:{node.line_start}-{node.line_end}"}
            self.analysis_cache[cache_key] = fallback
            return fallback
    
    def get_node_complexity(self, node: ASTNode) -> float:
        """Estimate code complexity based on AST structure."""
        # Base complexity by node type
        base_complexity = {
            "Module": 0.1,
            "Import": 0.2,
            "ImportFrom": 0.2,
            "ClassDef": 0.5,
            "FunctionDef": 0.4,
            "AsyncFunctionDef": 0.45,
            "For": 0.6,
            "While": 0.6,
            "If": 0.5,
            "Try": 0.7,
            "With": 0.5,
            "Match": 0.6,
            "BinOp": 0.3,
            "Call": 0.4,
            "ListComp": 0.5,
            "DictComp": 0.6
        }.get(node.node_type, 0.3)
        
        # Add complexity based on code size
        code_size = node.line_end - node.line_start + 1
        size_factor = min(1.0, code_size / 20)  # Cap at 1.0 for files over 20 lines
        
        # Add complexity based on child count
        child_factor = min(1.0, len(node.children) / 10)  # Cap at 1.0 for nodes with 10+ children
        
        # Combine factors
        complexity = (0.4 * base_complexity) + (0.3 * size_factor) + (0.3 * child_factor)
        return min(1.0, complexity)  # Ensure final complexity is between 0 and 1
    
    def extract_dependencies(self, node: ASTNode) -> List[str]:
        """Extract dependencies and imports from code."""
        dependencies = []
        
        # Handle direct imports
        if isinstance(node.ast_node, ast.Import):
            for name in node.ast_node.names:
                dependencies.append(name.name)
        
        # Handle from imports
        elif isinstance(node.ast_node, ast.ImportFrom):
            if node.ast_node.module:
                for name in node.ast_node.names:
                    dependencies.append(f"{node.ast_node.module}.{name.name}")
        
        # For other node types, look for imported modules in children
        elif node.children:
            for child in node.children:
                if isinstance(child.ast_node, (ast.Import, ast.ImportFrom)):
                    dependencies.extend(self.extract_dependencies(child))
        
        return dependencies

class ContextCache:
    """Manages a context cache with realtime loading and eviction.
    Supports SQLite persistence and LLM-enhanced code understanding."""
    
    def __init__(self, max_size: int = MAX_CONTEXT_SIZE, api_key: Optional[str] = None, 
                 model_name: str = "text-embedding-3-large", use_sqlite: bool = True, 
                 db_path: str = "ast_nodes.db"):
        self.max_size = max_size
        self.cache = {}  # path -> {node_id -> ASTNode}
        self.memory_map = {}  # memory_id -> CodeMemory
        self.embedder = ASTEmbedder(model_name=model_name, api_key=api_key)
        self.code_analyzer = CodeAnalyzer(api_key=api_key)
        self.parser = ASTParser()
        self.current_size = 0
        self.access_history = deque(maxlen=150)  # For LRU tracking
        
        # Configuration for hierarchical embeddings and relationships
        self.use_hierarchical = True
        self.hierarchy_weight = 0.15  # Weight for parent-child relationship influence
        
        # SQLite integration
        self.use_sqlite = use_sqlite
        self.db_path = db_path
        if use_sqlite:
            # Use the specific path, converting to absolute if needed
            db_file = Path(db_path)
            if not db_file.is_absolute():
                db_file = Path.cwd() / db_file
                
            self.conn = sqlite3.connect(str(db_file))
            self.conn.row_factory = sqlite3.Row
            self._initialize_db()
            logger.info(f"Using SQLite database for persistence at {db_file}")
        else:
            self.conn = None
            logger.info("Using in-memory storage only (no SQLite persistence)")
    
    def _initialize_db(self):
        """Initialize the database schema for code memory and relationships."""
        cursor = self.conn.cursor()
        
        # Create code memory table with UUID primary key
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS code_memories (
            id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            code_segment TEXT NOT NULL,
            node_type TEXT NOT NULL,
            line_start INTEGER NOT NULL,
            line_end INTEGER NOT NULL,
            parent_id TEXT,
            embedding BLOB,
            importance_score REAL DEFAULT 0.0,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 0,
            metadata TEXT
        )
        ''')
        
        # Create relationship table to track parent-child connections
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS code_relationships (
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            PRIMARY KEY (parent_id, child_id),
            FOREIGN KEY (parent_id) REFERENCES code_memories(id),
            FOREIGN KEY (child_id) REFERENCES code_memories(id)
        )
        ''')
        
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON code_memories(file_path)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_node_type ON code_memories(node_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON code_memories(last_accessed)')
        
        self.conn.commit()
    
    async def load_file(self, file_path: str) -> bool:
        """Load a file into the cache with intelligent code understanding."""
        # Check if file already loaded
        if file_path in self.cache:
            return True
        
        # Parse the file
        root = self.parser.parse_file(file_path)
        if not root:
            logger.error(f"Failed to parse file: {file_path}")
            return False
        
        # Create cache entry for the file
        self.cache[file_path] = {}
        
        # Traverse the AST and collect all nodes
        nodes = self._traverse_ast(root)
        logger.info(f"Parsed {len(nodes)} nodes from {file_path}")
        
        # Sort nodes by hierarchy for processing (parents before children)
        nodes_by_depth = self._sort_nodes_by_hierarchy(nodes)
        
        # Keep track of created memories and their relationships
        created_memories = []
        
        # Process nodes by depth (breadth-first)
        for depth, depth_nodes in nodes_by_depth.items():
            # Embed the nodes at this depth
            await self.embedder.embed_nodes(depth_nodes)
            
            # Apply hierarchical influence if enabled
            if self.use_hierarchical and depth > 0:
                self._apply_hierarchical_influence(depth_nodes)
            
            # Process each node to create memories
            for node in depth_nodes:
                # Create a unique ID for this node
                memory_id = str(uuid.uuid4())
                
                # Find parent memory ID if exists
                parent_id = None
                if node.parent:
                    parent_node_id = id(node.parent)
                    for file_nodes in self.cache.values():
                        for cache_node_id, cache_node in file_nodes.items():
                            if cache_node_id == parent_node_id:
                                # Find the memory ID for this node
                                for mem in created_memories:
                                    if mem.file_path == cache_node.path and mem.line_start == cache_node.line_start:
                                        parent_id = mem.id
                                        break
                                break
                
                # Get code complexity
                complexity = self.code_analyzer.get_node_complexity(node)
                
                # Initialize metadata with basic information
                metadata = {
                    "complexity": complexity,
                    "dependencies": self.code_analyzer.extract_dependencies(node)
                }
                
                # Create the code memory
                memory = CodeMemory(
                    id=memory_id,
                    file_path=file_path,
                    code_segment=node.get_source_segment(),
                    node_type=node.node_type,
                    line_start=node.line_start,
                    line_end=node.line_end,
                    embedding=node.embedding.tolist() if node.embedding is not None else None,
                    parent_id=parent_id,
                    importance_score=complexity,  # Initial score based on complexity
                    metadata=metadata
                )
                
                # Add to created memories list
                created_memories.append(memory)
                
                # Add to cache for quick lookup
                node_id = id(node)
                self.cache[file_path][node_id] = node
                self.memory_map[memory_id] = memory
                self.current_size += 1
        
        # Update the relationships between memories
        for memory in created_memories:
            if memory.parent_id:
                parent = self.memory_map.get(memory.parent_id)
                if parent:
                    if memory.id not in parent.children_ids:
                        parent.children_ids.append(memory.id)
        
        # Save to SQLite if enabled
        if self.use_sqlite:
            self._save_memories_to_db(created_memories)
        
        # Run LLM analysis on important nodes in the background
        asyncio.create_task(self._analyze_important_nodes(file_path, created_memories))
        
        # Evict if needed to stay within cache limits
        self._evict_if_needed()
        
        return True
    
    async def _analyze_important_nodes(self, file_path: str, memories: List[CodeMemory]):
        """Analyze important nodes with LLM to enhance understanding."""
        # Filter to more important/complex nodes to reduce API costs
        important_nodes = []
        for memory in memories:
            # Only analyze substantial code segments
            if memory.line_end - memory.line_start >= 3 and memory.importance_score >= 0.4:
                # Find the corresponding AST node
                for node_id, node in self.cache[file_path].items():
                    if node.line_start == memory.line_start and node.line_end == memory.line_end:
                        important_nodes.append((memory, node))
                        break
        
        # Sort by importance descending, limit to 10 nodes per file
        important_nodes.sort(key=lambda x: x[0].importance_score, reverse=True)
        important_nodes = important_nodes[:10]
        
        logger.info(f"Analyzing {len(important_nodes)} important nodes from {file_path}")
        
        # Process each node with the LLM
        for memory, node in important_nodes:
            try:
                # Get LLM summary
                analysis = await self.code_analyzer.summarize_code(node)
                
                # Update the memory with enhanced information
                memory.metadata.update(analysis)
                
                # Save to database if enabled
                if self.use_sqlite:
                    self._update_memory_metadata(memory)
                    
            except Exception as e:
                logger.error(f"Error analyzing node {memory.id}: {e}")
    
    def _save_memories_to_db(self, memories: List[CodeMemory]):
        """Save multiple memories to the database efficiently."""
        if not self.use_sqlite or not self.conn:
            return
            
        cursor = self.conn.cursor()
        
        try:
            # Begin transaction for better performance
            self.conn.execute("BEGIN TRANSACTION")
            
            # Insert memories
            for memory in memories:
                # Convert embedding to binary blob if available
                embedding_blob = None
                if memory.embedding:
                    embedding_blob = np.array(memory.embedding, dtype=np.float32).tobytes()
                
                # Insert the memory
                cursor.execute(
                    """INSERT OR REPLACE INTO code_memories 
                    (id, file_path, code_segment, node_type, line_start, line_end, 
                    parent_id, embedding, importance_score, last_accessed, access_count, metadata) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        memory.id,
                        memory.file_path,
                        memory.code_segment,
                        memory.node_type,
                        memory.line_start,
                        memory.line_end,
                        memory.parent_id,
                        embedding_blob,
                        memory.importance_score,
                        memory.last_accessed,
                        memory.access_count,
                        json.dumps(memory.metadata)
                    )
                )
                
                # Insert relationships
                for child_id in memory.children_ids:
                    cursor.execute(
                        """INSERT OR REPLACE INTO code_relationships 
                        (parent_id, child_id, relationship_type) VALUES (?, ?, ?)""",
                        (memory.id, child_id, "parent-child")
                    )
            
            # Commit the transaction
            self.conn.commit()
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error saving memories to database: {e}")
    
    def _update_memory_metadata(self, memory: CodeMemory):
        """Update a memory's metadata in the database."""
        if not self.use_sqlite or not self.conn:
            return
            
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(
                "UPDATE code_memories SET metadata = ? WHERE id = ?",
                (json.dumps(memory.metadata), memory.id)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating memory metadata: {e}")
    
    def _load_memories_from_db(self, file_path: Optional[str] = None) -> List[CodeMemory]:
        """Load memories from the database, optionally filtered by file path."""
        if not self.use_sqlite or not self.conn:
            return []
            
        cursor = self.conn.cursor()
        
        try:
            if file_path:
                cursor.execute("SELECT * FROM code_memories WHERE file_path = ?", (file_path,))
            else:
                cursor.execute("SELECT * FROM code_memories ORDER BY last_accessed DESC LIMIT 1000")
                
            rows = cursor.fetchall()
            
            memories = []
            for row in rows:
                # Extract embedding if available
                embedding = None
                if row['embedding']:
                    embedding = np.frombuffer(row['embedding'], dtype=np.float32).tolist()
                
                # Create memory object
                memory = CodeMemory(
                    id=row['id'],
                    file_path=row['file_path'],
                    code_segment=row['code_segment'],
                    node_type=row['node_type'],
                    line_start=row['line_start'],
                    line_end=row['line_end'],
                    embedding=embedding,
                    parent_id=row['parent_id'],
                    importance_score=row['importance_score'],
                    last_accessed=row['last_accessed'],
                    access_count=row['access_count'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                
                # Load children IDs
                cursor.execute(
                    "SELECT child_id FROM code_relationships WHERE parent_id = ?",
                    (memory.id,)
                )
                children_rows = cursor.fetchall()
                memory.children_ids = [child_row[0] for child_row in children_rows]
                
                memories.append(memory)
                
            return memories
            
        except Exception as e:
            logger.error(f"Error loading memories from database: {e}")
            return []
    
    def _sort_nodes_by_hierarchy(self, nodes: List[ASTNode]) -> Dict[int, List[ASTNode]]:
        """Sort nodes by hierarchy level for breadth-first processing."""
        nodes_by_depth = defaultdict(list)
        
        # First pass: calculate depth for each node
        for node in nodes:
            depth = 0
            parent = node.parent
            while parent:
                depth += 1
                parent = parent.parent
            nodes_by_depth[depth].append(node)
            
        return nodes_by_depth
    
    def _apply_hierarchical_influence(self, nodes: List[ASTNode]):
        """Apply hierarchical influence from parents to children with semantic meaning preservation."""
        for node in nodes:
            if node.parent and node.parent.embedding is not None and node.embedding is not None:
                # Store original embedding for reference
                original = np.array(node.embedding)
                
                # Get parent embedding
                parent_embedding = np.array(node.parent.embedding)
                
                # Blend child embedding with parent embedding
                parent_influence = parent_embedding * self.hierarchy_weight
                child_component = original * (1 - self.hierarchy_weight)
                blended = parent_influence + child_component
                
                # Normalize the blended embedding
                norm = np.linalg.norm(blended)
                if norm > 0:
                    node.embedding = blended / norm
                    
                    # Store hierarchical relationship strength in metadata
                    similarity = np.dot(original, parent_embedding)
                    node.metadata = getattr(node, 'metadata', {})
                    node.metadata['parent_similarity'] = float(similarity)
    
    def _traverse_ast(self, root: ASTNode) -> List[ASTNode]:
        """Traverse the AST and return all nodes."""
        nodes = [root]
        stack = [root]
        
        while stack:
            node = stack.pop()
            for child in node.children:
                nodes.append(child)
                stack.append(child)
                
        return nodes
    
    def _evict_if_needed(self, query_embedding: Optional[np.ndarray] = None):
        """Evict nodes if cache exceeds max size using sophisticated importance scoring."""
        if self.current_size <= self.max_size:
            return
            
        # Calculate importance for all nodes with multiple factors
        all_memories = list(self.memory_map.values())
        for memory in all_memories:
            # Calculate importance with recency, frequency, complexity and query relevance
            time_factor = np.exp(-(time.time() - memory.last_accessed) / CACHE_EXPIRY)
            freq_factor = np.log1p(memory.access_count)
            complexity = memory.metadata.get('complexity', 0.5)
            
            # Hierarchy bonus: important if it has children or is a top-level node
            hierarchy_bonus = 0.2 if memory.children_ids else 0.0
            if not memory.parent_id:  # Top-level node
                hierarchy_bonus += 0.2
                
            # Relevance to current query if available
            relevance = 0.0
            if query_embedding is not None and memory.embedding is not None:
                relevance = np.dot(query_embedding, memory.embedding)
                
            # Special bonus for certain node types
            type_bonus = {
                "ClassDef": 0.3,
                "FunctionDef": 0.25,
                "AsyncFunctionDef": 0.25,
                "Module": 0.2
            }.get(memory.node_type, 0.0)
            
            # Combine all factors
            memory.importance_score = (
                0.25 * time_factor + 
                0.15 * freq_factor + 
                0.20 * complexity + 
                0.15 * hierarchy_bonus + 
                0.15 * relevance +
                0.10 * type_bonus
            )
        
        # Sort by importance (ascending) to evict least important first
        all_memories.sort(key=lambda x: x.importance_score)
        
        # Evict least important nodes
        to_evict = self.current_size - int(self.max_size * 0.8)  # Remove extra 20% to avoid frequent evictions
        logger.info(f"Evicting {to_evict} nodes from cache")
        
        for i in range(min(to_evict, len(all_memories))):
            memory = all_memories[i]
            
            # Find the AST node for this memory
            node_to_remove = None
            file_path = memory.file_path
            
            if file_path in self.cache:
                for node_id, node in self.cache[file_path].items():
                    if node.line_start == memory.line_start and node.line_end == memory.line_end:
                        node_to_remove = node_id
                        break
            
            # Remove from cache
            if node_to_remove and file_path in self.cache:
                if node_to_remove in self.cache[file_path]:
                    del self.cache[file_path][node_to_remove]
                    
                    # If file has no nodes left, remove it
                    if not self.cache[file_path]:
                        del self.cache[file_path]
            
            # Remove from memory map
            if memory.id in self.memory_map:
                del self.memory_map[memory.id]
                self.current_size -= 1
    
    async def query(self, query: str, file_paths: List[str], top_k: int = 10) -> List[Tuple[CodeMemory, float]]:
        """Query code memories based on semantic similarity with enhanced understanding."""
        # Create a query embedding
        query_node = ASTNode(ast.AST(), query, "query")
        query_node.get_representation = lambda: query  # Override to use query directly
        query_embedding = await self.embedder.embed_node(query_node)
        
        # Load files if not in cache
        for file_path in file_paths:
            if file_path not in self.cache:
                try:
                    await self.load_file(file_path)
                except Exception as e:
                    logger.error(f"Error loading file {file_path}: {e}")
        
        # Collect all relevant memories, either from cache or database
        memories_to_search = []
        
        # First get memories from cache
        for file_path in file_paths:
            memories = [mem for mem in self.memory_map.values() if mem.file_path == file_path]
            memories_to_search.extend(memories)
        
        # If using SQLite, also search database for any memories not in cache
        if self.use_sqlite and len(memories_to_search) < 100:  # Limit database search
            for file_path in file_paths:
                db_memories = self._load_memories_from_db(file_path)
                
                # Add only memories not already in search list
                existing_ids = {mem.id for mem in memories_to_search}
                for memory in db_memories:
                    if memory.id not in existing_ids:
                        # Add to memory map and search list if it has an embedding
                        if memory.embedding:
                            self.memory_map[memory.id] = memory
                            memories_to_search.append(memory)
        
        # Calculate similarity for all memories with embeddings
        results = []
        for memory in memories_to_search:
            if memory.embedding:
                # Convert embedding to numpy array if needed
                if isinstance(memory.embedding, list):
                    embedding_array = np.array(memory.embedding)
                else:
                    embedding_array = memory.embedding
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, embedding_array)
                
                if similarity >= RELEVANCE_THRESHOLD:
                    # Update access time and count
                    memory.last_accessed = time.time()
                    memory.access_count += 1
                    self.access_history.append(memory.id)
                    
                    # Add to results
                    results.append((memory, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Update database if using SQLite
        if self.use_sqlite:
            for memory, _ in results[:top_k]:
                self._update_memory_metadata(memory)
        
        # Evict if needed based on query embedding
        self._evict_if_needed(query_embedding)
        
        return results[:top_k]
    
    def get_node_by_location(self, file_path: str, line: int) -> Optional[CodeMemory]:
        """Get the most specific memory at a given location with efficient index structure."""
        # Check if file is loaded in cache
        if file_path not in self.cache:
            # Try to load from database if using SQLite
            if self.use_sqlite:
                db_memories = self._load_memories_from_db(file_path)
                
                # Find memories that contain the line
                matching_memories = []
                for memory in db_memories:
                    if memory.line_start <= line <= memory.line_end:
                        matching_memories.append(memory)
                
                if matching_memories:
                    # Sort by specificity (smallest line range)
                    matching_memories.sort(key=lambda x: x.line_end - x.line_start)
                    memory = matching_memories[0]
                    
                    # Update access info
                    memory.last_accessed = time.time()
                    memory.access_count += 1
                    
                    # Add to memory map
                    self.memory_map[memory.id] = memory
                    
                    # Update in database
                    if self.use_sqlite:
                        self._update_memory_metadata(memory)
                        
                    return memory
            
            return None
        
        # Search in-memory cache
        matching_memories = []
        for memory in self.memory_map.values():
            if memory.file_path == file_path and memory.line_start <= line <= memory.line_end:
                matching_memories.append(memory)
        
        if not matching_memories:
            return None
        
        # Return the most specific (smallest) memory that contains the line
        matching_memories.sort(key=lambda x: x.line_end - x.line_start)
        memory = matching_memories[0]
        
        # Update access time and count
        memory.last_accessed = time.time()
        memory.access_count += 1
        self.access_history.append(memory.id)
        
        # Update in database if using SQLite
        if self.use_sqlite:
            self._update_memory_metadata(memory)
        
        return memory
    
    def get_memory_tree(self, memory_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get a hierarchical view of a memory and its relationships."""
        if memory_id not in self.memory_map:
            return {"error": "Memory not found"}
        
        memory = self.memory_map[memory_id]
        result = {
            "id": memory.id,
            "type": memory.node_type,
            "file": memory.file_path,
            "lines": f"{memory.line_start}-{memory.line_end}",
            "summary": memory.metadata.get("summary", "No summary available"),
            "complexity": memory.metadata.get("complexity", 0.0)
        }
        
        # Add children if depth remaining
        if depth > 0 and memory.children_ids:
            result["children"] = []
            for child_id in memory.children_ids:
                if child_id in self.memory_map:
                    child_tree = self.get_memory_tree(child_id, depth - 1)
                    result["children"].append(child_tree)
        
        # Add parent if available
        if memory.parent_id and depth > 0:
            parent = self.memory_map.get(memory.parent_id)
            if parent:
                result["parent"] = {
                    "id": parent.id,
                    "type": parent.node_type,
                    "file": parent.file_path,
                    "lines": f"{parent.line_start}-{parent.line_end}",
                    "summary": parent.metadata.get("summary", "No summary available")
                }
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        file_count = len(self.cache)
        memory_count = len(self.memory_map)
        
        # Count nodes by type
        node_types = {}
        for memory in self.memory_map.values():
            node_type = memory.node_type
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += 1
        
        # Get most accessed memories
        top_accessed = sorted(
            self.memory_map.values(), 
            key=lambda x: x.access_count, 
            reverse=True
        )[:10]
        
        return {
            "total_memories": memory_count,
            "files_loaded": file_count,
            "max_cache_size": self.max_size,
            "current_size": self.current_size,
            "node_type_distribution": sorted(node_types.items(), key=lambda x: x[1], reverse=True),
            "top_accessed": [
                {
                    "file": mem.file_path,
                    "lines": f"{mem.line_start}-{mem.line_end}",
                    "type": mem.node_type,
                    "access_count": mem.access_count,
                    "summary": mem.metadata.get("summary", "No summary available")
                }
                for mem in top_accessed
            ]
        }
    
    def __del__(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

class RealtimeASTAgent:
    """Intelligent agent that provides realtime AST-based code understanding and assistance using LLMs 
    with enhanced parallel processing and incremental updates."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "text-embedding-3-large",
                use_sqlite: bool = True, db_path: str = "ast_nodes.db",
                llm_enabled: bool = True, watch_mode: bool = False):
        """Initialize the agent with the specified configuration.
        
        Args:
            api_key: API key for OpenAI or Jina (will check env vars if not provided)
            model_name: Name of the embedding model to use
            use_sqlite: Whether to use SQLite for persistent storage
            db_path: Path to the SQLite database file
            llm_enabled: Whether to use LLM for enhanced code understanding
            watch_mode: Whether to watch files for changes and update in real-time
        """
        # Setup API keys and clients
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("JINA_API_KEY")
        self.llm_enabled = llm_enabled and (OPENAI_AVAILABLE or CUSTOM_JINA_CLIENT)
        self.watch_mode = watch_mode
        self.file_watchers = {}  # Map file paths to their watchers
        
        # Performance tracking and optimization
        self.query_times = []
        self.embedding_times = []
        self.file_load_times = {}
        self.last_refresh_time = time.time()
        
        # Semaphores for concurrent operations
        self.file_semaphore = asyncio.Semaphore(5)  # Limit concurrent file operations
        self.query_semaphore = asyncio.Semaphore(3)  # Limit concurrent queries
        
        # Initialize context cache with advanced memory and embedding capabilities
        self.context_cache = ContextCache(
            api_key=self.api_key,
            model_name=model_name,
            use_sqlite=use_sqlite,
            db_path=db_path
        )
        
        # Tracking files and query history with timestamps
        self.current_files = set()
        self.query_history = []
        self.file_modified_times = {}  # Track file modification times
        
        # Setup OpenAI client for enhanced code understanding if available
        self.openai_client = None
        if OPENAI_AVAILABLE and self.api_key and self.llm_enabled:
            try:
                # Handle different versions of the OpenAI client
                try:
                    # First try without proxies parameter
                    self.openai_client = openai.AsyncClient(api_key=self.api_key)
                except TypeError:
                    # If that fails, try with simpler initialization
                    self.openai_client = openai.AsyncClient(api_key=self.api_key)
                logger.info("Initialized OpenAI client for enhanced code understanding")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")
        
        # Setup Jina client for additional capabilities
        self.jina_client = None
        if CUSTOM_JINA_CLIENT and self.api_key:
            try:
                from modules.jina_client import JinaClient
                self.jina_client = JinaClient(token=self.api_key, openai_key=self.api_key)
                logger.info("Initialized Jina client for enhanced querying capabilities")
            except Exception as e:
                logger.warning(f"Could not initialize Jina client: {e}")
    
    async def initialize(self, file_paths: List[str]):
        """Initialize the agent with a set of files."""
        loaded_files = []
        failed_files = []
        
        # Load files concurrently for better performance
        load_tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(self._load_file_with_logging(file_path))
            load_tasks.append(task)
        
        # Wait for all loading tasks to complete
        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        for file_path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                logger.error(f"Error loading {file_path}: {result}")
                failed_files.append(file_path)
            elif result:
                loaded_files.append(file_path)
                self.current_files.add(file_path)
            else:
                failed_files.append(file_path)
        
        # Log summary of initialization
        logger.info(f"Initialization complete: {len(loaded_files)} files loaded, {len(failed_files)} files failed")
        
        # Return initialization status
        return {
            "success": len(failed_files) == 0,
            "loaded_files": loaded_files,
            "failed_files": failed_files,
            "total_nodes": self.context_cache.current_size
        }
    
    async def _load_file_with_logging(self, file_path: str) -> bool:
        """Load a file with appropriate logging."""
        start_time = time.time()
        logger.info(f"Loading file: {file_path}")
        
        try:
            success = await self.context_cache.load_file(file_path)
            duration = time.time() - start_time
            
            if success:
                logger.info(f"Successfully loaded {file_path} in {duration:.2f}s")
            else:
                logger.warning(f"Failed to load {file_path} after {duration:.2f}s")
                
            return success
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return False
    
    async def add_file(self, file_path: str) -> Dict[str, Any]:
        """Add a new file to the agent's context."""
        # Check if file exists
        if not os.path.isfile(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
        
        # Load the file
        start_time = time.time()
        success = await self.context_cache.load_file(file_path)
        duration = time.time() - start_time
        
        if success:
            self.current_files.add(file_path)
            return {
                "success": True,
                "file": file_path,
                "time_taken": f"{duration:.2f}s"
            }
        else:
            return {
                "success": False,
                "file": file_path,
                "error": "Failed to parse file"
            }
    
    async def query_code(self, query: str, file_paths: Optional[List[str]] = None, 
                        max_results: int = 10, enhanced: bool = True) -> Dict[str, Any]:
        """Query code based on semantic similarity with LLM-enhanced understanding.
        
        Args:
            query: The natural language query to search for in the code
            file_paths: List of file paths to search (defaults to all loaded files)
            max_results: Maximum number of results to return
            enhanced: Whether to enhance results with LLM analysis
            
        Returns:
            Dictionary with query results and metadata
        """
        # Use all loaded files if none specified
        if file_paths is None or not file_paths:
            file_paths = list(self.current_files)
            
        # No files loaded yet
        if not file_paths:
            return {
                "success": False,
                "error": "No files loaded. Use add_file() first."
            }
        
        # Add to query history
        self.query_history.append(query)
        start_time = time.time()
        
        # Get semantic search results
        try:
            results = await self.context_cache.query(query, file_paths, top_k=max_results)
        except Exception as e:
            logger.error(f"Error querying code: {e}")
            return {
                "success": False,
                "error": f"Query failed: {str(e)}"
            }
        
        # Convert to structured results
        structured_results = []
        for memory, similarity in results:
            result = {
                "file": memory.file_path,
                "line_start": memory.line_start,
                "line_end": memory.line_end,
                "code": memory.code_segment,
                "node_type": memory.node_type,
                "similarity": float(similarity)  # Ensure JSON serializable
            }
            
            # Add metadata if available
            if memory.metadata:
                if "summary" in memory.metadata:
                    result["summary"] = memory.metadata["summary"]
                if "purpose" in memory.metadata:
                    result["purpose"] = memory.metadata["purpose"]
                if "complexity" in memory.metadata:
                    result["complexity"] = memory.metadata["complexity"]
            
            structured_results.append(result)
        
        # Enhance results with LLM if enabled and available
        if enhanced and self.llm_enabled and structured_results and self.openai_client:
            try:
                enhanced_results = await self._enhance_results_with_llm(query, structured_results)
                structured_results = enhanced_results
            except Exception as e:
                logger.warning(f"Error enhancing results with LLM: {e}")
        
        # Total search time
        duration = time.time() - start_time
        
        # Add additional context from Jina if available
        context = []
        if enhanced and self.jina_client is not None:
            try:
                # Use Jina search to enhance understanding of the query
                search_result = await self.jina_client.search(query)
                
                # Extract useful information from search results
                if isinstance(search_result, dict) and "extraction" in search_result:
                    extraction = search_result.get("extraction", {})
                    facts = extraction.get("important_facts", [])
                    context = facts[:5]  # Limit to top 5 facts
            except Exception as e:
                logger.warning(f"Error getting context from Jina: {e}")
        
        return {
            "success": True,
            "query": query,
            "results_count": len(structured_results),
            "time_taken": f"{duration:.2f}s",
            "results": structured_results,
            "context": context
        }
    
    async def _enhance_results_with_llm(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance search results with LLM analysis for deeper code understanding."""
        if not self.openai_client or not results:
            return results
            
        try:
            # Prepare the prompt with query and top result snippets
            prompt = f"""I'm searching in code for: "{query}"
            
            I found these code snippets (showing top 3):
            
            """
            
            # Include top 3 results in the prompt
            for i, result in enumerate(results[:3]):
                prompt += f"Result {i+1} ({result['node_type']} in {result['file']}:{result['line_start']}-{result['line_end']}):\n```python\n{result['code']}\n```\n\n"
            
            prompt += "Please provide a brief analysis of these results to help me understand how they relate to my query."
            
            # Get analysis from OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a code analysis assistant helping understand search results in a codebase."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            # Extract the analysis
            analysis = response.choices[0].message.content
            
            # Add the analysis to the search results
            for result in results:
                result["llm_analysis"] = analysis
                
            return results
            
        except Exception as e:
            logger.error(f"Error enhancing results with LLM: {e}")
            return results
    
    async def get_code_at_location(self, file_path: str, line: int) -> Dict[str, Any]:
        """Get detailed code information at a specific location with semantic understanding."""
        # Make sure the file is loaded
        if file_path not in self.current_files:
            try:
                success = await self.context_cache.load_file(file_path)
                if success:
                    self.current_files.add(file_path)
                else:
                    return {
                        "success": False,
                        "error": f"Could not load file: {file_path}"
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Error loading file: {str(e)}"
                }
        
        # Get the CodeMemory at the location
        memory = self.context_cache.get_node_by_location(file_path, line)
        if not memory:
            return {
                "success": False,
                "error": f"No code found at {file_path}:{line}"
            }
        
        # Get tree view if available
        tree_view = self.context_cache.get_memory_tree(memory.id, depth=1)
        
        # Build the result with rich information
        result = {
            "success": True,
            "file": memory.file_path,
            "line_start": memory.line_start,
            "line_end": memory.line_end,
            "code": memory.code_segment,
            "node_type": memory.node_type,
            "importance": memory.importance_score,
            "metadata": memory.metadata,
            "tree_view": tree_view
        }
        
        # Add LLM analysis if enabled and not already provided
        if self.llm_enabled and self.openai_client and "summary" not in memory.metadata:
            try:
                # Get code analysis
                analysis = await self._analyze_code_with_llm(memory.code_segment, memory.node_type)
                
                # Add analysis to result
                if analysis:
                    result["llm_analysis"] = analysis
                    
                    # Update memory metadata
                    memory.metadata.update(analysis)
                    self.context_cache._update_memory_metadata(memory)
                    
            except Exception as e:
                logger.warning(f"Error analyzing code with LLM: {e}")
        
        return result
    
    async def _analyze_code_with_llm(self, code: str, node_type: str) -> Dict[str, Any]:
        """Analyze code with LLM to provide semantic understanding."""
        if not self.openai_client:
            return {}
            
        try:
            # Prepare the system message
            system_message = """You are a code analysis assistant specialized in Python.
            Analyze the provided code and provide a concise summary of what it does.
            Include key functionality, purpose, and any notable patterns.
            Output in JSON format with 'summary', 'complexity', and 'key_points' fields."""
            
            # Get analysis from OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Node type: {node_type}\n\nCode:\n```python\n{code}\n```"}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            # Extract and process the response
            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                return {"summary": "Code analysis unavailable"}
                
        except Exception as e:
            logger.error(f"Error analyzing code with LLM: {e}")
            return {}
    
    async def refresh_file(self, file_path: str) -> Dict[str, Any]:
        """Refresh a file in the cache with incremental updates for faster real-time processing."""
        # Check if file exists
        if not os.path.isfile(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
        
        start_time = time.time()
        
        # Check if file has been modified since last refresh
        try:
            current_mtime = os.path.getmtime(file_path)
            previous_mtime = self.file_modified_times.get(file_path, 0)
            
            # Only refresh if file has been modified or never loaded
            if current_mtime <= previous_mtime and file_path in self.current_files:
                return {
                    "success": True,
                    "file": file_path,
                    "status": "not_modified",
                    "time_taken": "0.00s"
                }
            
            # Update modification time
            self.file_modified_times[file_path] = current_mtime
        except Exception as e:
            logger.warning(f"Error checking file modification time: {e}")
        
        # Use semaphore to control concurrent file operations
        async with self.file_semaphore:
            # Remove from cache if present
            removed = False
            if file_path in self.current_files:
                self.current_files.remove(file_path)
                removed = True
                
            if file_path in self.context_cache.cache:
                # Intelligent incremental update logic
                try:
                    # Read the new file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        new_content = f.read()
                    
                    # Parse the new AST
                    parser = ASTParser()
                    new_root = parser.parse_file(file_path)
                    
                    if new_root:
                        # Store existing memories for this file before clearing cache
                        existing_memories = {
                            memory_id: memory 
                            for memory_id, memory in self.context_cache.memory_map.items()
                            if memory.file_path == file_path
                        }
                        
                        # Clear file from cache
                        del self.context_cache.cache[file_path]
                        
                        # Clear related memories
                        to_remove = []
                        for memory_id, memory in self.context_cache.memory_map.items():
                            if memory.file_path == file_path:
                                to_remove.append(memory_id)
                                
                        for memory_id in to_remove:
                            if memory_id in self.context_cache.memory_map:
                                del self.context_cache.memory_map[memory_id]
                                self.context_cache.current_size -= 1
                        
                        # Load the file with optimized processing
                        success = await self.context_cache.load_file(file_path)
                        
                        # Preserve access history and important metadata from existing memories
                        if success:
                            for new_memory_id, new_memory in self.context_cache.memory_map.items():
                                if new_memory.file_path == file_path:
                                    # Look for equivalent memory in the old set
                                    for old_memory in existing_memories.values():
                                        if (new_memory.node_type == old_memory.node_type and 
                                            new_memory.line_start == old_memory.line_start and
                                            new_memory.line_end == old_memory.line_end):
                                            # Preserve access statistics and metadata
                                            new_memory.access_count = old_memory.access_count
                                            new_memory.last_accessed = old_memory.last_accessed
                                            if "summary" in old_memory.metadata:
                                                new_memory.metadata["summary"] = old_memory.metadata["summary"]
                    else:
                        # Fallback to normal loading if parsing fails
                        success = await self.context_cache.load_file(file_path)
                except Exception as e:
                    logger.error(f"Error during incremental update: {e}")
                    # Fallback to regular reload
                    success = await self.context_cache.load_file(file_path)
            else:
                # Regular loading for new files
                success = await self.context_cache.load_file(file_path)
        
        duration = time.time() - start_time
        
        # Track file load time for performance analysis
        self.file_load_times[file_path] = duration
        
        if success:
            self.current_files.add(file_path)
            
            # Start file watcher if watch mode is enabled
            if self.watch_mode and file_path not in self.file_watchers:
                self._setup_file_watcher(file_path)
            
            return {
                "success": True,
                "file": file_path,
                "was_previously_loaded": removed,
                "time_taken": f"{duration:.2f}s"
            }
        else:
            return {
                "success": False,
                "file": file_path,
                "error": "Failed to parse file"
            }
    
    def _setup_file_watcher(self, file_path: str):
        """Set up a file watcher for real-time updates."""
        try:
            import watchdog
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            class FileChangeHandler(FileSystemEventHandler):
                def __init__(self, agent, file_path):
                    self.agent = agent
                    self.file_path = file_path
                
                def on_modified(self, event):
                    if event.src_path == self.file_path:
                        logger.info(f"File changed: {self.file_path}")
                        asyncio.create_task(self.agent.refresh_file(self.file_path))
            
            # Create observer and handler
            observer = Observer()
            handler = FileChangeHandler(self, file_path)
            
            # Schedule watching this file
            observer.schedule(handler, os.path.dirname(file_path), recursive=False)
            observer.start()
            
            # Store observer for cleanup
            self.file_watchers[file_path] = observer
            logger.info(f"File watcher started for {file_path}")
            
        except ImportError:
            logger.warning("watchdog module not found. File watching disabled.")
            self.watch_mode = False
        except Exception as e:
            logger.error(f"Error setting up file watcher: {e}")
    
    def get_ast_structure(self, file_path: str) -> Dict[str, Any]:
        """Get a rich hierarchical representation of the code structure."""
        # Check if file is loaded
        if file_path not in self.current_files:
            return {
                "success": False,
                "error": f"File not loaded: {file_path}"
            }
        
        # Find top-level memories for this file
        top_level_memories = []
        for memory in self.context_cache.memory_map.values():
            if memory.file_path == file_path and not memory.parent_id:
                top_level_memories.append(memory)
        
        if not top_level_memories:
            return {
                "success": False,
                "error": f"No structure found for {file_path}"
            }
        
        # Sort by line number
        top_level_memories.sort(key=lambda x: x.line_start)
        
        # Build tree structure for each top-level node
        structure = {
            "success": True,
            "file": file_path,
            "elements": []
        }
        
        for memory in top_level_memories:
            # Get tree view with 2 levels of depth
            tree = self.context_cache.get_memory_tree(memory.id, depth=2)
            structure["elements"].append(tree)
        
        return structure
    
    def get_file_summary(self, file_path: str) -> Dict[str, Any]:
        """Get a summary of file contents and structure."""
        # Check if file is loaded
        if file_path not in self.current_files:
            return {
                "success": False,
                "error": f"File not loaded: {file_path}"
            }
        
        # Collect stats about the file
        memories = [m for m in self.context_cache.memory_map.values() if m.file_path == file_path]
        
        if not memories:
            return {
                "success": False,
                "error": f"No data found for {file_path}"
            }
        
        # Count by node type
        node_types = {}
        for memory in memories:
            if memory.node_type not in node_types:
                node_types[memory.node_type] = 0
            node_types[memory.node_type] += 1
        
        # Find classes and functions
        classes = []
        functions = []
        imports = []
        
        for memory in memories:
            if memory.node_type == "ClassDef":
                summary = memory.metadata.get("summary", f"Class at lines {memory.line_start}-{memory.line_end}")
                classes.append({
                    "lines": f"{memory.line_start}-{memory.line_end}",
                    "summary": summary
                })
            elif memory.node_type == "FunctionDef":
                summary = memory.metadata.get("summary", f"Function at lines {memory.line_start}-{memory.line_end}")
                functions.append({
                    "lines": f"{memory.line_start}-{memory.line_end}",
                    "summary": summary
                })
            elif memory.node_type in ("Import", "ImportFrom"):
                imports.append(memory.code_segment.strip())
        
        # Sort by line number
        classes.sort(key=lambda x: int(x["lines"].split("-")[0]))
        functions.sort(key=lambda x: int(x["lines"].split("-")[0]))
        
        return {
            "success": True,
            "file": file_path,
            "total_nodes": len(memories),
            "node_types": node_types,
            "classes": classes,
            "functions": functions,
            "imports": imports[:10]  # Limit to first 10 imports
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics and information about the agent's state."""
        # Get cache stats
        cache_stats = self.context_cache.get_stats()
        
        # Add agent-specific stats
        stats = {
            "success": True,
            "files_loaded": len(self.current_files),
            "total_memories": cache_stats["total_memories"],
            "max_cache_size": cache_stats["max_cache_size"],
            "llm_enabled": self.llm_enabled,
            "openai_available": self.openai_client is not None,
            "jina_available": self.jina_client is not None,
            "node_distribution": dict(cache_stats["node_type_distribution"][:10]),  # Top 10 node types
            "recent_queries": self.query_history[-5:] if self.query_history else [],
            "active_files": list(self.current_files)[:10],  # Show first 10 files
            "top_accessed": cache_stats["top_accessed"][:5]  # Top 5 accessed nodes
        }
        
        return stats
    
    async def explain_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Explain a code snippet with detailed analysis using LLM."""
        if not self.llm_enabled or not self.openai_client:
            return {
                "success": False,
                "error": "LLM functionality is not enabled or available"
            }
            
        try:
            # Prepare the system message
            system_message = f"""You are a {language} code analysis expert.
            Analyze the provided code snippet and provide a detailed explanation.
            Include:
            1. A high-level summary of what the code does
            2. Line-by-line explanation of key parts
            3. Any potential issues or improvements
            4. How this code might fit into a larger system
            
            Provide your response in an educational and clear format."""
            
            # Get analysis from OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"```{language}\n{code}\n```"}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract and return the explanation
            explanation = response.choices[0].message.content
            
            return {
                "success": True,
                "explanation": explanation,
                "language": language,
                "code_length": len(code)
            }
                
        except Exception as e:
            logger.error(f"Error explaining code with LLM: {e}")
            return {
                "success": False,
                "error": f"Error explaining code: {str(e)}"
            }

async def main():
    """Main function that runs the AST Realtime Agent."""
    import asyncio
    import argparse
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    
    try:
        from rich.pretty import Pretty
        HAS_RICH = True
    except ImportError:
        HAS_RICH = False
        print("Rich library not found. Install with: pip install rich")
    
    # Create command-line argument parser
    parser = argparse.ArgumentParser(
        description="AST Realtime Agent: Intelligent code exploration with LLM assistance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # File and operation options
    parser.add_argument("--file", "-f", type=str, nargs="+", help="Files to analyze")
    parser.add_argument("--query", "-q", type=str, help="Natural language query to search in code")
    parser.add_argument("--line", "-l", type=int, help="Line number to analyze")
    parser.add_argument("--explain", "-e", type=str, help="File to explain with LLM")
    
    # Display options
    parser.add_argument("--summarize", "-s", action="store_true", help="Summarize loaded files")
    parser.add_argument("--load-only", "-lo", action="store_true", help="Only load files without analysis")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    
    # Configuration options
    parser.add_argument("--api-key", "-k", type=str, help="OpenAI or Jina API key")
    parser.add_argument("--model", "-m", type=str, default="text-embedding-3-large", 
                        help="Embedding model to use")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM-enhanced understanding")
    parser.add_argument("--no-cache", action="store_true", help="Disable SQLite caching")
    parser.add_argument("--db-path", "-db", type=str, default="ast_nodes.db", 
                        help="Path to SQLite database file")
    
    # Mode options
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--concise", "-c", action="store_true", help="Show concise output")
    
    args = parser.parse_args()
    
    # Initialize rich console for pretty output
    console = Console(highlight=True)
    
    # Validate arguments
    if not args.file and not args.interactive and not args.stats:
        console.print("[bold red]Please provide at least one file to analyze with --file or use --interactive mode[/]")
        return
    
    # Print header
    console.print("[bold blue]AST Realtime Agent[/bold blue] - Intelligent Code Understanding", justify="center")
    console.print("=" * 80, justify="center")
    
    # Initialize agent with configured options
    agent = RealtimeASTAgent(
        api_key=args.api_key,
        model_name=args.model, 
        use_sqlite=not args.no_cache,
        db_path=args.db_path,
        llm_enabled=not args.no_llm
    )
    
    # Display configuration
    config_table = Table(title="Configuration", show_header=False, border_style="dim")
    config_table.add_column("Setting", style="green")
    config_table.add_column("Value", style="cyan")
    
    config_table.add_row("Embedding Model", args.model)
    config_table.add_row("LLM Enhanced", "Enabled" if not args.no_llm else "Disabled")
    config_table.add_row("Storage", f"SQLite ({args.db_path})" if not args.no_cache else "In-memory only")
    
    if not args.concise:
        console.print(config_table)
    
    # Load initial files if provided
    if args.file:
        with console.status(f"Loading {len(args.file)} files...", spinner="dots"):
            result = await agent.initialize(args.file)
        
        if result["success"]:
            console.print(f"✅ Loaded {len(result['loaded_files'])} files with {result['total_nodes']} code nodes")
        else:
            console.print(f"⚠️  Loaded {len(result['loaded_files'])} files successfully. {len(result['failed_files'])} files failed.", style="yellow")
            if result["failed_files"]:
                console.print("Failed files:", style="red")
                for file in result["failed_files"]:
                    console.print(f"  - {file}", style="red")
    
    if args.load_only:
        console.print("[green]Files loaded successfully. Exiting.[/green]")
        return
    
    # Process requests in non-interactive mode
    if args.query:
        with console.status(f"Searching for: {args.query}", spinner="dots"):
            result = await agent.query_code(args.query, file_paths=args.file)
        
        if result["success"]:
            console.print(f"\n[bold green]Query:[/] {result['query']}")
            console.print(f"[bold green]Found:[/] {result['results_count']} results in {result['time_taken']}")
            
            if "context" in result and result["context"]:
                console.print("\n[bold]Context:[/]")
                for fact in result["context"]:
                    console.print(f"• {fact}")
            
            if "llm_analysis" in result["results"][0]:
                # Show LLM analysis of results
                console.print("\n[bold]Analysis:[/]")
                console.print(Panel(result["results"][0]["llm_analysis"], 
                                   border_style="dim", 
                                   title="LLM Analysis"))
            
            # Show results
            for i, res in enumerate(result["results"]):
                console.print(f"\n[bold]Result {i+1}[/] ({res['similarity']:.4f}) - {res['node_type']} in {res['file']}:{res['line_start']}-{res['line_end']}")
                
                if "summary" in res:
                    console.print(f"[italic]{res['summary']}[/italic]")
                
                # Use syntax highlighting for code
                syntax = Syntax(
                    res["code"], 
                    "python", 
                    line_numbers=True, 
                    start_line=res["line_start"],
                    highlight_lines=range(res["line_start"], res["line_end"]+1)
                )
                console.print(syntax)
        else:
            console.print(f"[bold red]Error:[/] {result['error']}")
    
    # Show code at specific line
    if args.line is not None and args.file and len(args.file) == 1:
        with console.status(f"Analyzing code at {args.file[0]}:{args.line}", spinner="dots"):
            result = await agent.get_code_at_location(args.file[0], args.line)
        
        if result["success"]:
            console.print(f"\n[bold]Code at {result['file']}:{args.line}[/]")
            console.print(f"[green]Node type:[/] {result['node_type']}")
            
            if "metadata" in result and "summary" in result["metadata"]:
                console.print(f"[green]Summary:[/] {result['metadata']['summary']}")
            
            # Show code with syntax highlighting
            syntax = Syntax(
                result["code"], 
                "python", 
                line_numbers=True, 
                start_line=result["line_start"]
            )
            console.print(syntax)
            
            # Show tree view if available and not in concise mode
            if "tree_view" in result and not args.concise:
                console.print("\n[bold]Code Context:[/]")
                if HAS_RICH:
                    console.print(Pretty(result["tree_view"]))
                else:
                    console.print(json.dumps(result["tree_view"], indent=2))
            
            # Show LLM analysis if available
            if "llm_analysis" in result:
                console.print("\n[bold]Analysis:[/]")
                console.print(Panel(result["llm_analysis"]["summary"], 
                                   border_style="dim", 
                                   title="LLM Analysis"))
        else:
            console.print(f"[bold red]Error:[/] {result['error']}")
    
    # Explain a file
    if args.explain and os.path.isfile(args.explain):
        with open(args.explain, 'r') as f:
            code = f.read()
        
        with console.status(f"Analyzing {args.explain} with LLM...", spinner="dots"):
            result = await agent.explain_code(code)
        
        if result["success"]:
            console.print(f"\n[bold]Explanation of {args.explain}:[/]")
            console.print(Panel(result["explanation"], 
                               border_style="blue", 
                               title="Code Explanation"))
        else:
            console.print(f"[bold red]Error:[/] {result['error']}")
    
    # Summarize files
    if args.summarize and args.file:
        for file_path in args.file:
            if file_path in agent.current_files:
                summary = agent.get_file_summary(file_path)
                
                if summary["success"]:
                    console.print(f"\n[bold]Summary of {file_path}:[/]")
                    console.print(f"Total nodes: {summary['total_nodes']}")
                    
                    # Show classes
                    if summary["classes"]:
                        console.print("\n[bold green]Classes:[/]")
                        for cls in summary["classes"]:
                            console.print(f"• {cls['summary']} (lines {cls['lines']})")
                    
                    # Show functions
                    if summary["functions"]:
                        console.print("\n[bold green]Functions:[/]")
                        for func in summary["functions"][:10]:  # Show top 10
                            console.print(f"• {func['summary']} (lines {func['lines']})")
                        
                        if len(summary["functions"]) > 10:
                            console.print(f"...and {len(summary['functions']) - 10} more functions")
                    
                    # Show imports
                    if summary["imports"]:
                        console.print("\n[bold green]Imports:[/]")
                        for imp in summary["imports"]:
                            console.print(f"• {imp}")
                else:
                    console.print(f"[bold red]Error summarizing {file_path}:[/] {summary['error']}")
    
    # Show statistics if requested
    if args.stats:
        with console.status("Getting statistics...", spinner="dots"):
            stats = agent.get_stats()
        
        stats_table = Table(title="Agent Statistics")
        stats_table.add_column("Statistic", style="green")
        stats_table.add_column("Value", style="cyan")
        
        stats_table.add_row("Files Loaded", str(stats["files_loaded"]))
        stats_table.add_row("Total Memories", str(stats["total_memories"]))
        stats_table.add_row("Memory Usage", f"{stats['total_memories']} / {stats['max_cache_size']}")
        stats_table.add_row("LLM Enabled", str(stats["llm_enabled"]))
        
        console.print(stats_table)
        
        # Show node distribution in a separate table
        if "node_distribution" in stats:
            dist_table = Table(title="Node Type Distribution")
            dist_table.add_column("Node Type", style="green")
            dist_table.add_column("Count", style="cyan")
            
            for node_type, count in stats["node_distribution"].items():
                dist_table.add_row(node_type, str(count))
                
            console.print(dist_table)
        
        # Show recent queries
        if stats["recent_queries"]:
            console.print("\n[bold]Recent Queries:[/]")
            for query in stats["recent_queries"]:
                console.print(f"• {query}")
    
    # Run in interactive mode if requested
    if args.interactive:
        console.print("\n[bold blue]AST Realtime Agent Interactive Mode[/]")
        console.print("[dim]Type 'help' for instructions, 'exit' to quit[/]")
        
        while True:
            try:
                cmd = input("\n> ").strip()
                
                if cmd.lower() in ('exit', 'quit'):
                    console.print("[green]Goodbye![/]")
                    break
                    
                elif cmd.lower() in ('help', '?'):
                    help_table = Table(title="Available Commands")
                    help_table.add_column("Command", style="cyan")
                    help_table.add_column("Description", style="green")
                    
                    help_table.add_row("load <file_path>", "Load a file into the agent")
                    help_table.add_row("query <query>", "Search code with natural language")
                    help_table.add_row("info <file> <line>", "Get code at specific location")
                    help_table.add_row("explain <file>", "Explain a file with LLM")
                    help_table.add_row("summary <file>", "Get a summary of a file")
                    help_table.add_row("refresh <file>", "Reload a file if it changed")
                    help_table.add_row("stats", "Show cache statistics")
                    help_table.add_row("exit/quit", "Exit interactive mode")
                    
                    console.print(help_table)
                    
                elif cmd.lower().startswith('load '):
                    file_path = cmd[5:].strip()
                    
                    with console.status(f"Loading {file_path}...", spinner="dots"):
                        result = await agent.add_file(file_path)
                    
                    if result["success"]:
                        console.print(f"✅ Successfully loaded {file_path} in {result['time_taken']}")
                    else:
                        console.print(f"❌ Failed to load {file_path}: {result.get('error', 'Unknown error')}", style="red")
                        
                elif cmd.lower().startswith('query '):
                    query = cmd[6:].strip()
                    
                    with console.status(f"Searching: {query}", spinner="dots"):
                        result = await agent.query_code(query)
                    
                    if result["success"]:
                        console.print(f"[bold green]Found {result['results_count']} results in {result['time_taken']}[/]")
                        
                        # Show LLM analysis if available
                        if result['results'] and "llm_analysis" in result['results'][0]:
                            console.print(Panel(result['results'][0]["llm_analysis"], 
                                               border_style="dim", 
                                               title="Analysis"))
                        
                        # Show individual results
                        for i, res in enumerate(result["results"][:5]):  # Show top 5 results
                            console.print(f"\n[bold]Result {i+1}[/] ({res['similarity']:.4f}) - {res['node_type']} in {res['file']}:{res['line_start']}-{res['line_end']}")
                            
                            if "summary" in res:
                                console.print(f"[italic]{res['summary']}[/italic]")
                            
                            # Show code with syntax highlighting
                            syntax = Syntax(
                                res["code"], 
                                "python", 
                                line_numbers=True, 
                                start_line=res["line_start"]
                            )
                            console.print(syntax)
                        
                        if result['results_count'] > 5:
                            console.print(f"\n[dim]...and {result['results_count'] - 5} more results[/]")
                    else:
                        console.print(f"❌ Query failed: {result.get('error', 'Unknown error')}", style="red")
                        
                elif cmd.lower().startswith('info '):
                    parts = cmd[5:].strip().split()
                    if len(parts) >= 2:
                        file_path = parts[0]
                        try:
                            line = int(parts[1])
                            
                            with console.status(f"Getting info for {file_path}:{line}...", spinner="dots"):
                                result = await agent.get_code_at_location(file_path, line)
                            
                            if result["success"]:
                                console.print(f"\n[bold]Code at {file_path}:{line}[/]")
                                console.print(f"[green]Node type:[/] {result['node_type']}")
                                
                                if "metadata" in result and "summary" in result["metadata"]:
                                    console.print(f"[green]Summary:[/] {result['metadata']['summary']}")
                                
                                # Show code with syntax highlighting
                                syntax = Syntax(
                                    result["code"], 
                                    "python", 
                                    line_numbers=True, 
                                    start_line=result["line_start"]
                                )
                                console.print(syntax)
                                
                                # Show LLM analysis if available
                                if "llm_analysis" in result:
                                    console.print(Panel(
                                        result["llm_analysis"]["summary"], 
                                        border_style="dim", 
                                        title="Analysis"
                                    ))
                            else:
                                console.print(f"❌ {result['error']}", style="red")
                        except ValueError:
                            console.print("❌ Line number must be an integer", style="red")
                    else:
                        console.print("❌ Usage: info <file_path> <line_number>", style="red")
                        
                elif cmd.lower().startswith('explain '):
                    file_path = cmd[8:].strip()
                    
                    if not os.path.isfile(file_path):
                        console.print(f"❌ File not found: {file_path}", style="red")
                        continue
                    
                    with open(file_path, 'r') as f:
                        code = f.read()
                    
                    with console.status(f"Analyzing {file_path} with LLM...", spinner="dots"):
                        result = await agent.explain_code(code)
                    
                    if result["success"]:
                        console.print(f"\n[bold]Explanation of {file_path}:[/]")
                        console.print(Panel(
                            result["explanation"], 
                            border_style="blue", 
                            title="Code Explanation"
                        ))
                    else:
                        console.print(f"❌ Error: {result['error']}", style="red")
                        
                elif cmd.lower().startswith('summary '):
                    file_path = cmd[8:].strip()
                    
                    if file_path not in agent.current_files:
                        console.print(f"❌ File not loaded: {file_path}", style="red")
                        console.print("Use 'load <file_path>' to load the file first.")
                        continue
                    
                    summary = agent.get_file_summary(file_path)
                    
                    if summary["success"]:
                        console.print(f"\n[bold]Summary of {file_path}:[/]")
                        console.print(f"Total nodes: {summary['total_nodes']}")
                        
                        # Show classes
                        if summary["classes"]:
                            console.print("\n[bold green]Classes:[/]")
                            for cls in summary["classes"]:
                                console.print(f"• {cls['summary']} (lines {cls['lines']})")
                        
                        # Show functions
                        if summary["functions"]:
                            console.print("\n[bold green]Functions:[/]")
                            for func in summary["functions"][:10]:  # Show top 10
                                console.print(f"• {func['summary']} (lines {func['lines']})")
                            
                            if len(summary["functions"]) > 10:
                                console.print(f"...and {len(summary['functions']) - 10} more functions")
                        
                        # Show imports
                        if summary["imports"]:
                            console.print("\n[bold green]Imports:[/]")
                            for imp in summary["imports"]:
                                console.print(f"• {imp}")
                    else:
                        console.print(f"❌ Error: {summary['error']}", style="red")
                        
                elif cmd.lower().startswith('refresh '):
                    file_path = cmd[8:].strip()
                    
                    with console.status(f"Refreshing {file_path}...", spinner="dots"):
                        result = await agent.refresh_file(file_path)
                    
                    if result["success"]:
                        console.print(f"✅ Successfully refreshed {file_path} in {result['time_taken']}")
                    else:
                        console.print(f"❌ Failed to refresh {file_path}: {result.get('error', 'Unknown error')}", style="red")
                        
                elif cmd.lower() == 'stats':
                    with console.status("Getting statistics...", spinner="dots"):
                        stats = agent.get_stats()
                    
                    stats_table = Table(title="Agent Statistics")
                    stats_table.add_column("Statistic", style="green")
                    stats_table.add_column("Value", style="cyan")
                    
                    stats_table.add_row("Files Loaded", str(stats["files_loaded"]))
                    stats_table.add_row("Total Memories", str(stats["total_memories"]))
                    stats_table.add_row("Memory Usage", f"{stats['total_memories']} / {stats['max_cache_size']}")
                    stats_table.add_row("LLM Enabled", str(stats["llm_enabled"]))
                    
                    console.print(stats_table)
                    
                    # Show node distribution in a separate table
                    if "node_distribution" in stats:
                        dist_table = Table(title="Node Type Distribution")
                        dist_table.add_column("Node Type", style="green")
                        dist_table.add_column("Count", style="cyan")
                        
                        for node_type, count in stats["node_distribution"].items():
                            dist_table.add_row(node_type, str(count))
                            
                        console.print(dist_table)
                    
                    # Show top accessed nodes
                    if stats["top_accessed"]:
                        console.print("\n[bold]Most Accessed Nodes:[/]")
                        for node in stats["top_accessed"]:
                            console.print(f"• {node['summary']} ({node['type']} in {node['file']}:{node['lines']})")
                else:
                    console.print("❌ Unknown command. Type 'help' for instructions.", style="red")
                    
            except KeyboardInterrupt:
                console.print("\nExiting...")
                break
            except Exception as e:
                console.print(f"❌ Error: {str(e)}", style="red")
        return

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())