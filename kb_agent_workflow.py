#!/usr/bin/env python3
"""
KB Agent Workflow Demo

This script demonstrates a multi-agent workflow that integrates:
- Polymorphic prompts
- Knowledge bases
- Facts database
- Multiple specialized agents

The workflow performs a research and analysis task across domains.
"""

import os
import sys
import json
import time
import argparse
import sqlite3
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
import yaml
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from uuid import uuid4
import traceback

# Import our custom modules
from embedding_service import EmbeddingServiceClient, compute_embedding
from advanced_polymorphic_prompts import AdvancedPromptManager, Prompt, PromptChain

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("kb-agent-workflow")

# Check for the holographic_memory module for vector database operations
try:
    from holographic_memory import HolographicMemoryDB
    has_holographic = True
except ImportError:
    has_holographic = False
    logger.warning("Warning: HolographicMemoryDB not available. Using simplified vector storage.")


class EmbeddingDBStorage:
    """SQLite-based embedding storage with vector search capability"""
    
    def __init__(self, db_path: str):
        """Initialize the embedding database storage"""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialize_db()
        self.embedding_client = EmbeddingServiceClient()
        
    def _initialize_db(self):
        """Initialize the SQLite database with necessary tables"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create knowledge_items table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_items (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT NOT NULL,
                source TEXT,
                domain TEXT,
                created_at REAL,
                access_count INTEGER DEFAULT 0
            )
            ''')
            
            # Create embeddings table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                knowledge_id TEXT NOT NULL,
                vector BLOB NOT NULL,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_items(id)
            )
            ''')
            
            # Create tags table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
            ''')
            
            # Create knowledge_tags table (many-to-many)
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_tags (
                knowledge_id TEXT NOT NULL,
                tag_id INTEGER NOT NULL,
                PRIMARY KEY (knowledge_id, tag_id),
                FOREIGN KEY (knowledge_id) REFERENCES knowledge_items(id),
                FOREIGN KEY (tag_id) REFERENCES tags(id)
            )
            ''')
            
            # Add index for faster vector search
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge_items(domain)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_knowledge ON embeddings(knowledge_id)')
            
            self.conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
            raise
    
    def add_knowledge(self, title: str, content: str, metadata: Dict = None, tags: List[str] = None):
        """Add knowledge to the database with embedding"""
        metadata = metadata or {}
        tags = tags or []
        
        try:
            # Generate a unique ID
            knowledge_id = str(uuid4())
            
            # Extract metadata fields
            source = metadata.get('source', 'unknown')
            domain = metadata.get('domain', 'general')
            
            # Insert knowledge item
            self.cursor.execute(
                'INSERT INTO knowledge_items (id, title, content, source, domain, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (knowledge_id, title, content, source, domain, time.time())
            )
            
            # Generate and store embedding
            try:
                embedding = self.embedding_client.embed_text(content)
                embedding_id = str(uuid4())
                
                # Convert embedding to bytes for storage
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                
                # Store embedding
                self.cursor.execute(
                    'INSERT INTO embeddings (id, knowledge_id, vector) VALUES (?, ?, ?)',
                    (embedding_id, knowledge_id, embedding_bytes)
                )
            except Exception as e:
                logger.warning(f"Error generating embedding: {e}")
            
            # Process tags
            for tag in tags:
                # Insert tag if not exists
                self.cursor.execute(
                    'INSERT OR IGNORE INTO tags (name) VALUES (?)',
                    (tag,)
                )
                
                # Get tag ID
                self.cursor.execute('SELECT id FROM tags WHERE name = ?', (tag,))
                tag_id = self.cursor.fetchone()[0]
                
                # Link knowledge to tag
                self.cursor.execute(
                    'INSERT INTO knowledge_tags (knowledge_id, tag_id) VALUES (?, ?)',
                    (knowledge_id, tag_id)
                )
            
            self.conn.commit()
            return knowledge_id
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding knowledge: {e}")
            return None
    
    def batch_add_knowledge(self, items: List[Dict]):
        """Add multiple knowledge items in batch with optimized embedding generation"""
        if not items:
            return []
        
        try:
            # Extract titles and contents for batch embedding
            titles = [item.get('title', f"Item_{i}") for i, item in enumerate(items)]
            contents = [item['content'] for item in items if 'content' in item]
            
            # Generate embeddings in batch
            try:
                results = self.embedding_client.embed_batch(contents)
                embeddings = results
            except Exception as e:
                logger.warning(f"Error in batch embedding: {e}")
                embeddings = [None] * len(contents)
            
            # Begin transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            knowledge_ids = []
            
            # Insert each knowledge item
            for i, item in enumerate(items):
                if 'content' not in item:
                    logger.warning(f"Skipping item without content: {item}")
                    continue
                
                # Generate a unique ID
                knowledge_id = str(uuid4())
                knowledge_ids.append(knowledge_id)
                
                # Extract fields
                content = item['content']
                title = item.get('title', f"Item_{i}")
                metadata = item.get('metadata', {})
                tags = item.get('tags', [])
                
                source = metadata.get('source', 'unknown')
                domain = metadata.get('domain', 'general')
                
                # Insert knowledge item
                self.cursor.execute(
                    'INSERT INTO knowledge_items (id, title, content, source, domain, created_at) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (knowledge_id, title, content, source, domain, time.time())
                )
                
                # Store embedding if available
                if i < len(embeddings) and embeddings[i] is not None:
                    embedding = embeddings[i]
                    embedding_id = str(uuid4())
                    
                    # Convert embedding to bytes for storage
                    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                    
                    # Store embedding
                    self.cursor.execute(
                        'INSERT INTO embeddings (id, knowledge_id, vector) VALUES (?, ?, ?)',
                        (embedding_id, knowledge_id, embedding_bytes)
                    )
                
                # Process tags
                for tag in tags:
                    # Insert tag if not exists
                    self.cursor.execute(
                        'INSERT OR IGNORE INTO tags (name) VALUES (?)',
                        (tag,)
                    )
                    
                    # Get tag ID
                    self.cursor.execute('SELECT id FROM tags WHERE name = ?', (tag,))
                    tag_id = self.cursor.fetchone()[0]
                    
                    # Link knowledge to tag
                    self.cursor.execute(
                        'INSERT INTO knowledge_tags (knowledge_id, tag_id) VALUES (?, ?)',
                        (knowledge_id, tag_id)
                    )
            
            self.conn.commit()
            return knowledge_ids
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error in batch adding knowledge: {e}")
            return []
    
    def search(self, query: str, top_k: int = 5, domain: str = None, tags: List[str] = None):
        """Search knowledge using vector similarity"""
        try:
            # Generate embedding for query
            query_embedding = np.array(self.embedding_client.embed_text(query), dtype=np.float32)
            
            # Build a dynamic SQL query based on filters
            sql_filters = []
            sql_params = []
            join_clause = ""
            
            # Filter by domain if specified
            if domain:
                sql_filters.append("k.domain = ?")
                sql_params.append(domain)
            
            # Filter by tags if specified
            if tags and len(tags) > 0:
                placeholders = ", ".join(["?"] * len(tags))
                join_clause = """
                JOIN knowledge_tags kt ON k.id = kt.knowledge_id
                JOIN tags t ON kt.tag_id = t.id
                """
                sql_filters.append(f"t.name IN ({placeholders})")
                sql_params.extend(tags)
            
            # Build where clause
            where_clause = ""
            if sql_filters:
                where_clause = "WHERE " + " AND ".join(sql_filters)
            
            # Get all embeddings with their knowledge items
            query = f"""
            SELECT 
                e.vector, 
                k.id,
                k.title,
                k.content,
                k.source,
                k.domain,
                k.created_at,
                k.access_count
            FROM embeddings e
            JOIN knowledge_items k ON e.knowledge_id = k.id
            {join_clause}
            {where_clause}
            GROUP BY k.id
            """
            
            self.cursor.execute(query, sql_params)
            rows = self.cursor.fetchall()
            
            # Calculate similarities
            results = []
            for row in rows:
                vector_bytes = row[0]
                knowledge_id = row[1]
                title = row[2]
                content = row[3]
                source = row[4]
                domain = row[5]
                created_at = row[6]
                access_count = row[7]
                
                # Convert bytes to vector
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, vector)
                
                # Get tags for this knowledge item
                self.cursor.execute("""
                SELECT t.name FROM tags t
                JOIN knowledge_tags kt ON t.id = kt.tag_id
                WHERE kt.knowledge_id = ?
                """, (knowledge_id,))
                
                item_tags = [tag[0] for tag in self.cursor.fetchall()]
                
                # Add to results
                results.append({
                    "id": knowledge_id,
                    "title": title,
                    "content": content,
                    "similarity": float(similarity),
                    "source": source,
                    "domain": domain,
                    "created_at": created_at,
                    "access_count": access_count,
                    "tags": item_tags
                })
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Update access counts for returned items
            for result in results[:top_k]:
                self.cursor.execute(
                    'UPDATE knowledge_items SET access_count = access_count + 1 WHERE id = ?',
                    (result["id"],)
                )
                result["access_count"] += 1
            
            self.conn.commit()
            
            # Return top results
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            # Fallback to basic keyword search if vector search fails
            return self._keyword_search(query, top_k, domain, tags)
    
    def _keyword_search(self, query: str, top_k: int = 5, domain: str = None, tags: List[str] = None):
        """Fallback keyword-based search"""
        try:
            # Split query into terms
            query_terms = query.lower().split()
            
            # Build SQL query with filters
            sql_filters = []
            sql_params = []
            join_clause = ""
            
            # Always add LIKE conditions for each term
            for term in query_terms:
                sql_filters.append("(LOWER(k.title) LIKE ? OR LOWER(k.content) LIKE ?)")
                sql_params.extend([f"%{term}%", f"%{term}%"])
            
            # Filter by domain if specified
            if domain:
                sql_filters.append("k.domain = ?")
                sql_params.append(domain)
            
            # Filter by tags if specified
            if tags and len(tags) > 0:
                placeholders = ", ".join(["?"] * len(tags))
                join_clause = """
                JOIN knowledge_tags kt ON k.id = kt.knowledge_id
                JOIN tags t ON kt.tag_id = t.id
                """
                sql_filters.append(f"t.name IN ({placeholders})")
                sql_params.extend(tags)
            
            # Build where clause
            where_clause = "WHERE " + " AND ".join(sql_filters)
            
            # Get knowledge items matching the query terms
            query = f"""
            SELECT 
                k.id,
                k.title,
                k.content,
                k.source,
                k.domain,
                k.created_at,
                k.access_count
            FROM knowledge_items k
            {join_clause}
            {where_clause}
            GROUP BY k.id
            """
            
            self.cursor.execute(query, sql_params)
            rows = self.cursor.fetchall()
            
            # Calculate relevance scores and build results
            results = []
            for row in rows:
                knowledge_id = row[0]
                title = row[1]
                content = row[2]
                source = row[3]
                domain = row[4]
                created_at = row[5]
                access_count = row[6]
                
                # Calculate relevance score based on term frequency
                title_lower = title.lower()
                content_lower = content.lower()
                score = sum(title_lower.count(term) * 3 + content_lower.count(term) for term in query_terms)
                
                # Get tags for this knowledge item
                self.cursor.execute("""
                SELECT t.name FROM tags t
                JOIN knowledge_tags kt ON t.id = kt.tag_id
                WHERE kt.knowledge_id = ?
                """, (knowledge_id,))
                
                item_tags = [tag[0] for tag in self.cursor.fetchall()]
                
                # Add to results
                results.append({
                    "id": knowledge_id,
                    "title": title,
                    "content": content,
                    "similarity": score / (10 * len(query_terms)),  # Normalize to approximate similarity score
                    "source": source,
                    "domain": domain,
                    "created_at": created_at,
                    "access_count": access_count,
                    "tags": item_tags
                })
            
            # Sort by score (highest first)
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Update access counts for returned items
            for result in results[:top_k]:
                self.cursor.execute(
                    'UPDATE knowledge_items SET access_count = access_count + 1 WHERE id = ?',
                    (result["id"],)
                )
                result["access_count"] += 1
            
            self.conn.commit()
            
            # Return top results
            return results[:top_k]
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def load_from_directory(self, directory: str, chunk_size: int = 1000, batch_size: int = 10):
        """Load knowledge from a directory of JSON files with batch processing"""
        if not os.path.exists(directory):
            logger.error(f"Directory {directory} does not exist")
            return 0
        
        count = 0
        batch = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = json.load(f)
                        
                        title = content.get('title', os.path.splitext(file)[0])
                        text = json.dumps(content, ensure_ascii=False)
                        
                        # Determine domain from path or content
                        domain_parts = file_path.split(os.sep)[-2:]  # Use last two directory components
                        domain = domain_parts[0] if len(domain_parts) > 1 else "general"
                        
                        # Add to batch
                        if len(text) > chunk_size:
                            # Split into chunks if needed
                            chunks = self._chunk_text(text, chunk_size)
                            for i, chunk in enumerate(chunks):
                                batch.append({
                                    'title': f"{title}_chunk_{i+1}",
                                    'content': chunk,
                                    'metadata': {'source': file_path, 'domain': domain},
                                    'tags': [domain, 'chunked']
                                })
                        else:
                            batch.append({
                                'title': title,
                                'content': text,
                                'metadata': {'source': file_path, 'domain': domain},
                                'tags': [domain]
                            })
                        
                        # Process batch if it reaches the batch size
                        if len(batch) >= batch_size:
                            self.batch_add_knowledge(batch)
                            count += len(batch)
                            logger.info(f"Processed {count} items...")
                            batch = []
                    
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
        
        # Process any remaining items
        if batch:
            self.batch_add_knowledge(batch)
            count += len(batch)
        
        logger.info(f"Finished loading {count} knowledge items from {directory}")
        return count
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of approximately chunk_size characters"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for the space
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def get_stats(self):
        """Get statistics about the knowledge base"""
        try:
            # Get count of knowledge items
            self.cursor.execute("SELECT COUNT(*) FROM knowledge_items")
            knowledge_count = self.cursor.fetchone()[0]
            
            # Get count of embeddings
            self.cursor.execute("SELECT COUNT(*) FROM embeddings")
            embedding_count = self.cursor.fetchone()[0]
            
            # Get count of tags
            self.cursor.execute("SELECT COUNT(*) FROM tags")
            tag_count = self.cursor.fetchone()[0]
            
            # Get domains and their counts
            self.cursor.execute("""
            SELECT domain, COUNT(*) as count 
            FROM knowledge_items 
            GROUP BY domain 
            ORDER BY count DESC
            """)
            domains = dict(self.cursor.fetchall())
            
            # Get popular tags
            self.cursor.execute("""
            SELECT t.name, COUNT(kt.knowledge_id) as count 
            FROM tags t
            JOIN knowledge_tags kt ON t.id = kt.tag_id
            GROUP BY t.name
            ORDER BY count DESC
            LIMIT 10
            """)
            popular_tags = dict(self.cursor.fetchall())
            
            return {
                "knowledge_count": knowledge_count,
                "embedding_count": embedding_count,
                "tag_count": tag_count,
                "domains": domains,
                "popular_tags": popular_tags
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "knowledge_count": 0,
                "embedding_count": 0,
                "tag_count": 0,
                "domains": {},
                "popular_tags": {}
            }
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()


class KnowledgeConnector:
    """Connector for knowledge bases using vector search"""
    
    def __init__(self, kb_type: str, connection_params: Dict[str, Any]):
        self.kb_type = kb_type
        self.connection_params = connection_params
        self.embedding_client = EmbeddingServiceClient()
        
        if kb_type == "vector":
            db_path = connection_params.get("path", "./kb_vector.db")
            self.db = EmbeddingDBStorage(db_path)
        else:
            raise ValueError(f"Unsupported KB type: {kb_type}")
            
    def initialize_from_directory(self, directory: str, chunk_size: int = 1000):
        """Initialize the knowledge base from a directory of JSON files"""
        logger.info(f"Initializing knowledge base from {directory}")
        
        if not os.path.exists(directory):
            logger.error(f"Directory {directory} does not exist")
            return
        
        # Use batch loading
        count = self.db.load_from_directory(directory, chunk_size=chunk_size, batch_size=20)
        logger.info(f"Finished initializing knowledge base with {count} items")
        
    def add_knowledge(self, key: str, text: str, metadata: Dict = None):
        """Add knowledge to the vector database with embeddings"""
        try:
            # Extract domain from metadata
            domain = metadata.get('domain', 'general') if metadata else 'general'
            source = metadata.get('source', 'user') if metadata else 'user'
            
            # Extract tags if any
            tags = metadata.get('tags', []) if metadata else []
            if domain and domain not in tags:
                tags.append(domain)
            
            # Add to database
            self.db.add_knowledge(
                title=key,
                content=text,
                metadata={'source': source, 'domain': domain},
                tags=tags
            )
            return True
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return False
            
    def query(self, query: str, top_k: int = 5, domain: str = None, tags: List[str] = None):
        """Query the knowledge base for relevant information"""
        try:
            # Search the database
            results = self.db.search(
                query=query,
                top_k=top_k,
                domain=domain,
                tags=tags
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result["id"],
                    "title": result["title"],
                    "similarity": result["similarity"],
                    "text": result["content"],
                    "metadata": {
                        "source": result["source"],
                        "domain": result["domain"],
                        "tags": result["tags"],
                        "created_at": result["created_at"],
                        "access_count": result["access_count"]
                    }
                })
                
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return []
    
    def get_stats(self):
        """Get statistics about the knowledge base"""
        return self.db.get_stats()
    
    def close(self):
        """Close the knowledge base connection"""
        if hasattr(self, 'db') and hasattr(self.db, 'close'):
            self.db.close()


class SimplifiedVectorDB:
    """Simple vector database when holographic_memory is not available"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.vectors = {}
        self.load()
        
    def load(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    self.vectors = json.load(f)
        except Exception as e:
            logger.error(f"Error loading vector DB: {e}")
            self.vectors = {}
            
    def save(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'w') as f:
                json.dump(self.vectors, f)
        except Exception as e:
            logger.error(f"Error saving vector DB: {e}")
            
    def add_vector(self, key: str, vector: List[float], metadata: Dict = None):
        self.vectors[key] = {
            "vector": vector,
            "metadata": metadata or {}
        }
        self.save()
        
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        if not self.vectors:
            return []
            
        results = []
        for key, data in self.vectors.items():
            similarity = self._cosine_similarity(query_vector, data["vector"])
            results.append((key, similarity, data["metadata"]))
            
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        
        dot_product = np.dot(vec1_array, vec2_array)
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


@dataclass
class Fact:
    """A fact with metadata and confidence"""
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    source: str = ""
    confidence: float = 0.5
    domain: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    supporting_facts: List[str] = field(default_factory=list)
    contradicting_facts: List[str] = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)


class FactsDatabase:
    """Database for storing and retrieving facts"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.facts = {}
        self.embedding_client = EmbeddingServiceClient()
        self.vector_db = SimplifiedVectorDB(f"{os.path.splitext(db_path)[0]}_vectors.json")
        self.load()
        
    def load(self):
        """Load facts from database file"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    facts_data = json.load(f)
                    
                self.facts = {}
                for fact_id, fact_data in facts_data.items():
                    # Convert dict to Fact
                    self.facts[fact_id] = Fact(**fact_data)
            else:
                self.facts = {}
        except Exception as e:
            logger.error(f"Error loading facts DB: {e}")
            self.facts = {}
            
    def save(self):
        """Save facts to database file"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            facts_data = {}
            for fact_id, fact in self.facts.items():
                facts_data[fact_id] = fact.to_dict()
                
            with open(self.db_path, 'w') as f:
                json.dump(facts_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving facts DB: {e}")
            
    def add_fact(self, content: str, source: str, confidence: float = 0.5, domain: str = None) -> Fact:
        """Add a new fact to the database"""
        # Check for duplicate or contradicting facts
        existing_fact = self.find_similar_fact(content)
        
        if existing_fact:
            # If existing fact has higher confidence, keep it
            if existing_fact.confidence >= confidence:
                return existing_fact
                
            # Update existing fact with higher confidence
            existing_fact.confidence = confidence
            existing_fact.source = source
            existing_fact.timestamp = datetime.now().isoformat()
            if domain:
                existing_fact.domain = domain
                
            # Update vector DB
            self._update_fact_embedding(existing_fact)
            self.save()
            return existing_fact
            
        # Create new fact
        fact = Fact(
            content=content,
            source=source,
            confidence=confidence,
            domain=domain or ""
        )
        
        # Add to database
        self.facts[fact.id] = fact
        
        # Add to vector DB
        self._update_fact_embedding(fact)
        
        self.save()
        return fact
        
    def find_similar_fact(self, content: str, threshold: float = 0.9) -> Optional[Fact]:
        """Find a fact that's similar to the given content"""
        try:
            # Generate embedding for content
            content_embedding = self.embedding_client.embed_text(content)
            
            # Search for similar facts
            results = self.vector_db.search(content_embedding, top_k=1)
            
            if results and results[0][1] >= threshold:
                fact_id = results[0][0]
                return self.facts.get(fact_id)
                
            return None
        except Exception as e:
            logger.error(f"Error finding similar fact: {e}")
            return None
            
    def query_facts(self, query: str, confidence_threshold: float = 0.5, top_k: int = 5) -> List[Fact]:
        """Query facts database for facts relevant to the query"""
        try:
            # Generate embedding for query
            query_embedding = self.embedding_client.embed_text(query)
            
            # Search for relevant facts
            results = self.vector_db.search(query_embedding, top_k=top_k*2)  # Get extra to filter by confidence
            
            # Filter by confidence and get fact objects
            facts = []
            for fact_id, similarity, _ in results:
                fact = self.facts.get(fact_id)
                if fact and fact.confidence >= confidence_threshold:
                    facts.append(fact)
                    
            return facts[:top_k]
        except Exception as e:
            logger.error(f"Error querying facts: {e}")
            return []
            
    def get_facts_by_domain(self, domain: str) -> List[Fact]:
        """Get all facts for a specific domain"""
        return [fact for fact in self.facts.values() if fact.domain == domain]
        
    def _update_fact_embedding(self, fact: Fact):
        """Update or add the embedding for a fact"""
        try:
            embedding = self.embedding_client.embed_text(fact.content)
            self.vector_db.add_vector(fact.id, embedding, {"fact_id": fact.id})
        except Exception as e:
            logger.error(f"Error updating fact embedding: {e}")


class KBAgent:
    """Agent that uses knowledge bases and prompts to perform tasks"""
    
    def __init__(self, agent_id: str, kb_connector: KnowledgeConnector, prompt_manager: AdvancedPromptManager, facts_db: FactsDatabase = None):
        self.agent_id = agent_id
        self.kb = kb_connector
        self.prompt_manager = prompt_manager
        self.facts_db = facts_db
        self.state = {}
        
    def process(self, task: str, input_data: Dict = None) -> Dict:
        """Process a task using knowledge bases and prompts"""
        logger.info(f"Agent {self.agent_id} processing task: {task}")
        
        input_data = input_data or {}
        self.state.update(input_data)
        
        # Different tasks require different handling
        if task == "gather_facts":
            return self._gather_facts(input_data.get("query", ""))
        elif task == "analyze":
            return self._analyze_facts(input_data.get("facts", []))
        elif task == "synthesize":
            return self._synthesize_data(input_data)
        else:
            return {"error": f"Unknown task: {task}"}
            
    def _gather_facts(self, query: str) -> Dict:
        """Gather facts from knowledge base"""
        # Find an appropriate prompt
        prompt = self._find_best_prompt("research")
        
        if not prompt:
            return {"error": "No suitable prompt found for research task"}
            
        # Query knowledge base
        kb_results = self.kb.query(query, top_k=5)
        
        # Extract facts
        facts = []
        for result in kb_results:
            # Create a fact from each knowledge item
            if self.facts_db:
                fact = self.facts_db.add_fact(
                    content=result["text"][:500],  # Limit length
                    source=result.get("metadata", {}).get("source", "knowledge_base"),
                    confidence=result["similarity"],
                    domain=self.agent_id
                )
                facts.append(fact.to_dict())
            else:
                # If no facts DB, just store the raw data
                facts.append({
                    "id": result["id"],
                    "content": result["text"][:500],
                    "source": result.get("metadata", {}).get("source", "knowledge_base"),
                    "confidence": result["similarity"],
                    "domain": self.agent_id
                })
                
        return {
            "facts": facts,
            "query": query,
            "prompt_used": prompt.name
        }
        
    def _analyze_facts(self, facts: List[Dict]) -> Dict:
        """Analyze a set of facts"""
        # Find an appropriate prompt
        prompt = self._find_best_prompt("analysis")
        
        if not prompt:
            return {"error": "No suitable prompt found for analysis task"}
            
        # Prepare data for analysis
        fact_contents = [f["content"] for f in facts]
        facts_text = "\n\n".join(fact_contents)
        
        # In a real implementation, this would use an LLM to process the facts with the prompt
        # For now, we'll simulate the analysis
        analysis = {
            "summary": f"Analysis of {len(facts)} facts",
            "key_points": [
                f"The data suggests multiple perspectives on {facts[0]['content'][:50]}..." if facts else "No facts provided",
                "Point 2...",
                "Point 3..."
            ],
            "confidence": sum(f.get("confidence", 0.5) for f in facts) / max(len(facts), 1)
        }
        
        return {
            "analysis": analysis,
            "facts_used": [f["id"] for f in facts],
            "prompt_used": prompt.name
        }
        
    def _synthesize_data(self, input_data: Dict) -> Dict:
        """Synthesize multiple analyses into a comprehensive report"""
        # Find an appropriate prompt
        prompt = self._find_best_prompt("synthesis")
        
        if not prompt:
            return {"error": "No suitable prompt found for synthesis task"}
            
        # Extract data from inputs
        facts = input_data.get("facts", [])
        analyses = input_data.get("analyses", [input_data.get("analysis")] if input_data.get("analysis") else [])
        
        # In a real implementation, this would use an LLM to synthesize with the prompt
        # For now, we'll simulate the synthesis
        synthesis = {
            "title": "Synthesized Report",
            "summary": f"A comprehensive analysis of {len(facts)} facts across {len(analyses)} domains",
            "conclusions": [
                "Conclusion 1...",
                "Conclusion 2...",
                "Conclusion 3..."
            ],
            "recommendations": [
                "Recommendation 1...",
                "Recommendation 2..."
            ]
        }
        
        return {
            "synthesis": synthesis,
            "facts_used": [f["id"] for f in facts],
            "analyses_used": len(analyses),
            "prompt_used": prompt.name
        }
        
    def _find_best_prompt(self, task_type: str) -> Optional[Prompt]:
        """Find the best prompt for a given task type"""
        # In a real implementation, this would use semantic search
        # For now, we'll use a simple mapping
        prompt_mapping = {
            "research": "holographic_example",
            "analysis": "multi_agent_example", 
            "synthesis": "temporal_example"
        }
        
        prompt_name = prompt_mapping.get(task_type)
        if prompt_name:
            return self.prompt_manager.load_prompt(prompt_name)
            
        return None


class WorkflowEngine:
    """Engine for executing workflows with multiple agents"""
    
    def __init__(self):
        self.agents = {}
        self.workflows = {}
        
    def register_agent(self, agent_id: str, agent: KBAgent):
        """Register an agent with the workflow engine"""
        self.agents[agent_id] = agent
        
    def define_workflow(self, workflow_id: str, steps: List[Dict]):
        """Define a workflow with multiple steps"""
        self.workflows[workflow_id] = steps
        
    def execute_workflow(self, workflow_id: str, initial_context: Dict = None) -> Dict:
        """Execute a workflow with the registered agents"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {"error": f"Workflow '{workflow_id}' not found"}
            
        context = initial_context or {}
        results = []
        
        logger.info(f"Executing workflow '{workflow_id}' with {len(workflow)} steps")
        
        for i, step in enumerate(workflow):
            logger.info(f"\nStep {i+1}: {step['agent']} - {step['task']}")
            
            agent = self.agents.get(step["agent"])
            if not agent:
                logger.error(f"Agent '{step['agent']}' not found")
                results.append({"error": f"Agent '{step['agent']}' not found"})
                continue
                
            # Prepare inputs for this step
            step_inputs = {}
            for input_name in step.get("inputs", []):
                if input_name in context:
                    step_inputs[input_name] = context[input_name]
                else:
                    logger.warning(f"Warning: Input '{input_name}' not found in context")
                    
            # Execute the step
            try:
                start_time = time.time()
                result = agent.process(step["task"], step_inputs)
                duration = time.time() - start_time
                
                logger.info(f"Step completed in {duration:.2f} seconds")
                
                # Store outputs in context
                for output_name in step.get("outputs", []):
                    if output_name in result:
                        context[output_name] = result[output_name]
                    else:
                        logger.warning(f"Warning: Output '{output_name}' not found in result")
                        
                # Add to results
                results.append({
                    "step": i+1,
                    "agent": step["agent"],
                    "task": step["task"],
                    "duration": duration,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Error executing step: {e}")
                traceback.print_exc()
                results.append({
                    "step": i+1,
                    "agent": step["agent"],
                    "task": step["task"],
                    "error": str(e)
                })
                
        # Return final context and results
        return {
            "workflow_id": workflow_id,
            "context": context,
            "results": results
        }


def setup_demo_environment():
    """Set up a demo environment with KB agents and workflows"""
    # Create directories
    os.makedirs("./data", exist_ok=True)
    
    # Initialize prompt manager
    prompt_manager = AdvancedPromptManager("./prompts")
    
    # Initialize knowledge connectors
    academic_kb = KnowledgeConnector("vector", {"path": "./data/academic_kb.json", "use_holographic": False})
    business_kb = KnowledgeConnector("vector", {"path": "./data/business_kb.json", "use_holographic": False})
    
    # Initialize facts database
    facts_db = FactsDatabase("./data/facts.json")
    
    # Add sample knowledge items directly instead of loading from files
    logger.info("Adding sample knowledge items...")
    
    # Academic knowledge items
    academic_kb.add_knowledge(
        "machine_learning_basics",
        "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed. "
        "It focuses on algorithms that can learn from and make predictions on data. Common types include supervised learning, "
        "unsupervised learning, and reinforcement learning. Supervised learning uses labeled training data to learn a function "
        "that maps inputs to outputs. Unsupervised learning finds patterns in unlabeled data. Reinforcement learning enables "
        "an agent to learn through interaction with its environment.",
        {"source": "academic", "domain": "computer_science"}
    )
    
    academic_kb.add_knowledge(
        "deep_learning",
        "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers. "
        "These deep neural networks can learn hierarchical features from data. Convolutional Neural Networks (CNNs) "
        "are specialized for processing grid-like data such as images. Recurrent Neural Networks (RNNs) are designed "
        "for sequential data like text or time series. Transformers have become dominant in natural language processing "
        "thanks to their attention mechanisms.",
        {"source": "academic", "domain": "computer_science"}
    )
    
    academic_kb.add_knowledge(
        "ai_ethics",
        "AI ethics examines the moral implications of artificial intelligence systems. Key concerns include bias in "
        "training data leading to unfair outcomes, privacy implications of data collection, transparency of AI decision-making, "
        "and potential job displacement. Researchers advocate for fairness, accountability, and transparency in AI development. "
        "Regulations like GDPR in Europe aim to protect individual rights in an AI-powered world.",
        {"source": "academic", "domain": "ethics"}
    )
    
    # Business knowledge items
    business_kb.add_knowledge(
        "ml_business_applications",
        "Machine learning has transformed business operations across industries. In finance, it powers fraud detection, "
        "algorithmic trading, and credit scoring. Healthcare uses ML for disease diagnosis, personalized treatment plans, "
        "and medical image analysis. Retail companies leverage ML for customer segmentation, recommendation systems, and "
        "demand forecasting. Manufacturing employs ML for predictive maintenance, quality control, and supply chain optimization.",
        {"source": "business", "domain": "applications"}
    )
    
    business_kb.add_knowledge(
        "ml_market_trends",
        "The global machine learning market is experiencing rapid growth, expected to reach $190 billion by 2025. "
        "Key trends include increased adoption of cloud-based ML services, democratization of ML tools for non-experts, "
        "and rising demand for explainable AI. Industries making the largest investments include finance, healthcare, "
        "automotive, and retail. The talent shortage remains a significant challenge, with demand for ML engineers "
        "far exceeding supply.",
        {"source": "business", "domain": "market_analysis"}
    )
    
    business_kb.add_knowledge(
        "ml_implementation_challenges",
        "Organizations implementing machine learning face several challenges: data quality issues including incomplete or "
        "biased datasets; integration difficulties with legacy systems; shortage of skilled personnel; computational resource "
        "limitations; explainability requirements especially in regulated industries; and scalability concerns. Successful "
        "implementation requires cross-functional teams, clear business objectives, and realistic expectations about timeline "
        "and outcomes.",
        {"source": "business", "domain": "implementation"}
    )
    
    # Create agents
    research_agent = KBAgent("research", academic_kb, prompt_manager, facts_db)
    analysis_agent = KBAgent("analysis", academic_kb, prompt_manager, facts_db)
    business_agent = KBAgent("business", business_kb, prompt_manager, facts_db)
    synthesis_agent = KBAgent("synthesis", None, prompt_manager, facts_db)
    
    # Create workflow engine
    workflow_engine = WorkflowEngine()
    workflow_engine.register_agent("research", research_agent)
    workflow_engine.register_agent("analysis", analysis_agent)
    workflow_engine.register_agent("business", business_agent)
    workflow_engine.register_agent("synthesis", synthesis_agent)
    
    # Define a research workflow
    workflow_engine.define_workflow("research_report", [
        {"agent": "research", "task": "gather_facts", "outputs": ["facts"]},
        {"agent": "analysis", "task": "analyze", "inputs": ["facts"], "outputs": ["analysis"]},
        {"agent": "synthesis", "task": "synthesize", "inputs": ["facts", "analysis"], "outputs": ["report"]}
    ])
    
    # Define a business analysis workflow
    workflow_engine.define_workflow("business_analysis", [
        {"agent": "business", "task": "gather_facts", "outputs": ["business_facts"]},
        {"agent": "research", "task": "gather_facts", "outputs": ["research_facts"]},
        {"agent": "analysis", "task": "analyze", "inputs": ["business_facts", "research_facts"], "outputs": ["analysis"]},
        {"agent": "business", "task": "analyze", "inputs": ["business_facts", "analysis"], "outputs": ["business_analysis"]},
        {"agent": "synthesis", "task": "synthesize", "inputs": ["business_facts", "research_facts", "analysis", "business_analysis"], "outputs": ["report"]}
    ])
    
    return workflow_engine


def autonomous_problem_solver(query, kb_files=None, domains=None, max_steps=5, kb_dir="./knowledge_bases"):
    """
    Autonomous agent that solves a problem through step-by-step reasoning
    
    Args:
        query: The problem or question to solve
        kb_files: List of specific knowledge base files to use (optional)
        domains: List of domain tags to filter knowledge by (optional)
        max_steps: Maximum number of reasoning steps
        kb_dir: Directory containing knowledge base files
    
    Returns:
        Dict with reasoning steps and final answer
    """
    # Create a dedicated knowledge base for the solver
    logger.info(f"Creating autonomous problem solver for query: {query}")
    
    db_path = "./data/solver_kb.db"
    kb_storage = EmbeddingDBStorage(db_path)
    kb_connector = KnowledgeConnector("vector", {"path": db_path})
    
    # If no specific KB files provided, automatically find relevant ones
    if not kb_files:
        logger.info("No knowledge bases specified, automatically discovering relevant ones...")
        kb_files = discover_relevant_knowledge_bases(query, kb_dir)
        if kb_files:
            logger.info(f"Discovered {len(kb_files)} potentially relevant knowledge bases:")
            for kb_file in kb_files:
                logger.info(f"  - {os.path.basename(kb_file)}")
        else:
            logger.warning("No relevant knowledge bases found. Using all available knowledge.")
            kb_files = _get_all_kb_files(kb_dir)
    
    # Count how many KB files successfully loaded
    loaded_count = 0
    
    # Load knowledge bases
    for kb_file in kb_files:
        logger.info(f"Loading knowledge from {kb_file}...")
        if os.path.exists(kb_file):
            try:
                with open(kb_file, 'r') as f:
                    content = json.load(f)
                
                # Determine domain from filename
                domain = os.path.basename(kb_file).split('.')[0]
                
                # Process the content
                if isinstance(content, dict):
                    kb_connector.add_knowledge(
                        key=content.get('title', domain),
                        text=json.dumps(content, ensure_ascii=False),
                        metadata={'source': kb_file, 'domain': domain}
                    )
                    loaded_count += 1
                elif isinstance(content, list):
                    # Process each item in the list
                    for i, item in enumerate(content):
                        if isinstance(item, dict):
                            kb_connector.add_knowledge(
                                key=item.get('title', f"{domain}_{i}"),
                                text=json.dumps(item, ensure_ascii=False),
                                metadata={'source': kb_file, 'domain': domain}
                            )
                    loaded_count += 1
            except Exception as e:
                logger.error(f"Error processing knowledge file {kb_file}: {e}")
    
    logger.info(f"Successfully loaded {loaded_count} knowledge sources")
    
    # Initialize the reasoning process
    logger.info("Beginning step-by-step reasoning process")
    
    # Initial state
    context = {
        "query": query,
        "current_step": 0,
        "steps": [],
        "known_facts": [],
        "hypotheses": [],
        "conclusions": [],
        "domains": domains or []
    }
    
    # Step-by-step reasoning loop
    while context["current_step"] < max_steps:
        step_num = context["current_step"] + 1
        logger.info(f"\n--- Step {step_num} ---")
        
        # 1. Formulate sub-question based on current context
        sub_question = _formulate_sub_question(context)
        logger.info(f"Sub-question: {sub_question}")
        
        # 2. Query knowledge base for relevant information
        kb_results = kb_connector.query(
            query=sub_question, 
            top_k=3,
            tags=context["domains"] if context["domains"] else None
        )
        
        if kb_results:
            logger.info(f"Found {len(kb_results)} relevant knowledge items")
            # Extract new facts from knowledge base results
            for result in kb_results:
                if result["similarity"] > 0.3:  # Only use reasonably similar results
                    new_fact = {
                        "content": result["text"][:300] + "..." if len(result["text"]) > 300 else result["text"],
                        "similarity": result["similarity"],
                        "source": result["metadata"].get("source", "knowledge_base"),
                        "domain": result["metadata"].get("domain", "general")
                    }
                    if new_fact not in context["known_facts"]:
                        context["known_facts"].append(new_fact)
                        logger.info(f"Added new fact from {new_fact['domain']} (relevance: {new_fact['similarity']:.2f})")
        else:
            logger.info("No relevant knowledge found for this sub-question")
        
        # 3. Apply reasoning to derive new insights
        reasoning_result = _apply_reasoning(context, sub_question)
        
        # 4. Update context with new step
        context["steps"].append({
            "step_number": step_num,
            "sub_question": sub_question,
            "relevant_facts": kb_results,
            "reasoning": reasoning_result["reasoning"],
            "new_insights": reasoning_result["new_insights"]
        })
        
        # Add any new hypotheses
        for hypothesis in reasoning_result.get("hypotheses", []):
            if hypothesis not in context["hypotheses"]:
                context["hypotheses"].append(hypothesis)
                logger.info(f"New hypothesis: {hypothesis}")
        
        # Add any new conclusions
        for conclusion in reasoning_result.get("conclusions", []):
            if conclusion not in context["conclusions"]:
                context["conclusions"].append(conclusion)
                logger.info(f"New conclusion: {conclusion}")
        
        # Check if we've reached a final answer
        if reasoning_result.get("final_answer"):
            logger.info(f"Final answer reached: {reasoning_result['final_answer']}")
            context["final_answer"] = reasoning_result["final_answer"]
            break
        
        # Move to next step
        context["current_step"] += 1
        
        # Simulated thinking pause
        time.sleep(0.5)
    
    # If we've exhausted steps without a final answer, provide best conclusion
    if "final_answer" not in context and context["conclusions"]:
        context["final_answer"] = context["conclusions"][-1]
        logger.info(f"Maximum steps reached. Best conclusion: {context['final_answer']}")
    elif "final_answer" not in context:
        context["final_answer"] = "Insufficient information to reach a conclusion."
        logger.info("Maximum steps reached without conclusion.")
    
    # Save the reasoning process
    output_path = "./data/autonomous_reasoning.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(context, f, indent=2)
        logger.info(f"Reasoning process saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving reasoning: {e}")
    
    return context


def discover_relevant_knowledge_bases(query, kb_dir, max_results=5):
    """
    Automatically discover knowledge bases relevant to a query
    
    Args:
        query: The user query
        kb_dir: Directory containing knowledge base files
        max_results: Maximum number of knowledge bases to return
        
    Returns:
        List of paths to relevant knowledge base files
    """
    # Extract key terms from the query
    key_terms = extract_key_terms(query)
    logger.info(f"Extracted key terms: {', '.join(key_terms)}")
    
    # Get all knowledge base files
    all_kb_files = _get_all_kb_files(kb_dir)
    if not all_kb_files:
        logger.warning(f"No knowledge base files found in {kb_dir}")
        return []
    
    # Calculate relevance score for each knowledge base
    kb_scores = []
    for kb_file in all_kb_files:
        score = calculate_kb_relevance(kb_file, key_terms)
        kb_scores.append((kb_file, score))
    
    # Sort by relevance score (descending)
    kb_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take the top N most relevant
    most_relevant = [kb for kb, score in kb_scores[:max_results] if score > 0]
    
    return most_relevant


def _get_all_kb_files(kb_dir):
    """Get all JSON files in the knowledge base directory"""
    if not os.path.exists(kb_dir):
        logger.warning(f"Knowledge base directory {kb_dir} does not exist")
        return []
    
    kb_files = []
    for root, _, files in os.walk(kb_dir):
        for file in files:
            if file.endswith('.json'):
                kb_files.append(os.path.join(root, file))
    
    return kb_files


def extract_key_terms(query):
    """Extract key terms from a query"""
    # Remove common words and punctuation
    common_words = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'of', 'to', 'for', 'and', 
                   'or', 'on', 'with', 'by', 'as', 'at', 'from', 'how', 'why', 'when', 'where',
                   'which', 'who', 'whom', 'whose', 'that', 'this', 'these', 'those', 'between',
                   'relationship', 'applications', 'most', 'promising'}
    
    # Split query into words and convert to lowercase
    words = query.lower().replace('?', '').replace('.', '').replace(',', '').split()
    
    # Filter out common words and keep only substantive terms
    key_terms = [word for word in words if word not in common_words and len(word) > 2]
    
    # Handle compound terms like "machine learning" or "quantum computing"
    if "machine" in key_terms and "learning" in words:
        key_terms.append("machine_learning")
    if "quantum" in key_terms and "computing" in words:
        key_terms.append("quantum_computing")
    if "artificial" in key_terms and "intelligence" in words:
        key_terms.append("artificial_intelligence")
    
    return key_terms


def calculate_kb_relevance(kb_file, key_terms):
    """Calculate relevance score of a knowledge base to the key terms"""
    try:
        # Extract domain from filename as a primary relevance signal
        filename = os.path.basename(kb_file).lower()
        domain = os.path.splitext(filename)[0]
        
        # Initial score based on filename match
        score = 0
        for term in key_terms:
            if term.lower() in domain:
                score += 3  # Strong match if term appears in filename
        
        # Look into the file content for additional signal
        try:
            with open(kb_file, 'r') as f:
                try:
                    content = json.load(f)
                    
                    # Check the title if it's a dict
                    if isinstance(content, dict) and 'title' in content:
                        title = content['title'].lower()
                        for term in key_terms:
                            if term.lower() in title:
                                score += 2  # Good match if term appears in title
                    
                    # Check content text
                    content_str = json.dumps(content).lower()
                    for term in key_terms:
                        if term.lower() in content_str:
                            score += 1  # Basic match if term appears in content
                            
                except json.JSONDecodeError:
                    # Not a valid JSON, just check if terms appear in raw text
                    raw_content = f.read().lower()
                    for term in key_terms:
                        if term.lower() in raw_content:
                            score += 0.5
        except Exception as e:
            # If we can't read the file, just use the filename score
            logger.warning(f"Couldn't analyze content of {kb_file}: {e}")
        
        return score
        
    except Exception as e:
        logger.error(f"Error calculating relevance for {kb_file}: {e}")
        return 0


def _formulate_sub_question(context):
    """Generate the next sub-question based on current context"""
    query = context["query"]
    current_step = context["current_step"]
    
    # For demonstration purposes, we'll use a simple approach
    if current_step == 0:
        # Initial exploration
        return f"What are the key concepts related to: {query}?"
    elif current_step == 1:
        # Deeper understanding
        return f"What are the main challenges or problems associated with: {query}?"
    elif current_step == 2:
        # Solutions or approaches
        return f"What approaches or methods exist to address: {query}?"
    elif current_step == 3:
        # Evaluation or comparison
        return f"What are the advantages and disadvantages of different approaches to: {query}?"
    else:
        # Synthesis and conclusion
        return f"What are the most important considerations for addressing: {query}?"


def _apply_reasoning(context, sub_question):
    """Apply reasoning to current knowledge to derive new insights"""
    # This is a simplified implementation for demonstration
    # In a real system, this would use an LLM or reasoning engine
    
    known_facts = context["known_facts"]
    current_step = context["current_step"]
    
    # Simple reasoning simulation
    if not known_facts:
        reasoning = "No facts available for reasoning at this stage."
        new_insights = ["Insufficient information to draw meaningful conclusions."]
        return {"reasoning": reasoning, "new_insights": new_insights}
    
    # Extract domains present in our facts
    domains = list(set(fact["domain"] for fact in known_facts))
    
    # Create a simple knowledge graph from extracted facts
    knowledge_graph = {}
    
    # Process all facts to build a knowledge graph of concepts and relationships
    for fact in known_facts:
        content = fact["content"].lower()
        
        # Extract key concepts by frequency analysis
        words = content.replace(".", " ").replace(",", " ").replace("(", " ").replace(")", " ").split()
        word_freq = {}
        
        for word in words:
            if len(word) > 3 and word not in {"with", "that", "this", "from", "what", "when", "they", "their", "would", "could"}:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Extract key phrases (simple n-gram analysis)
        phrases = []
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            if any(tech_term in phrase for tech_term in ["quantum", "algorithm", "machine", "learning", "neural", "feature", "vector", "model"]):
                phrases.append(phrase)
                
        # Store in knowledge graph
        domain = fact["domain"]
        if domain not in knowledge_graph:
            knowledge_graph[domain] = {
                "frequent_terms": word_freq,
                "key_phrases": phrases,
                "raw_facts": [content]
            }
        else:
            # Update existing domain knowledge
            for term, freq in word_freq.items():
                knowledge_graph[domain]["frequent_terms"][term] = knowledge_graph[domain]["frequent_terms"].get(term, 0) + freq
            knowledge_graph[domain]["key_phrases"].extend(phrases)
            knowledge_graph[domain]["raw_facts"].append(content)
    
    # Analyze the knowledge graph based on sub-question to derive insights
    applications = []
    if "most promising applications" in sub_question.lower() or "key concepts" in sub_question.lower():
        # Extract potential applications by analyzing patterns in the knowledge graph
        for domain, domain_data in knowledge_graph.items():
            # Find frequent technical terms
            tech_terms = sorted(domain_data["frequent_terms"].items(), key=lambda x: x[1], reverse=True)
            tech_terms = [term for term, freq in tech_terms if freq > 1]
            
            # Process raw facts to extract application patterns
            for fact in domain_data["raw_facts"]:
                # Look for application-indicating patterns using dynamic text analysis
                sentences = fact.split(".")
                
                for sentence in sentences:
                    # Look for sentences that suggest applications
                    if any(app_indicator in sentence for app_indicator in ["application", "used for", "applied to", "potential", "promising", "advantage"]):
                        applications.append(sentence.strip())
                    
                    # Look for specific algorithm mentions
                    if any(alg in sentence for alg in ["svm", "pca", "neural network", "boltzmann", "hhl", "qaoa", "vqe"]):
                        applications.append(sentence.strip())
    
    # Synthesize insights based on step
    reasoning = f"Analyzing information from {len(known_facts)} facts across domains: {', '.join(domains)}"
    
    # Extract concepts from knowledge graph for all steps
    key_concepts = []
    all_terms = {}
    
    # Aggregate terms across all domains
    for domain, data in knowledge_graph.items():
        for term, freq in data["frequent_terms"].items():
            all_terms[term] = all_terms.get(term, 0) + freq
    
    # Get top terms overall
    top_terms = sorted(all_terms.items(), key=lambda x: x[1], reverse=True)[:10]
    key_concepts = [term for term, _ in top_terms]
    
    # Generate insights based on the current step
    if current_step == 0:
        # Initial concepts
        new_insights = [f"The query relates to domains: {', '.join(domains)}"]
        
        # Extract key phrases from knowledge graph
        key_phrase_insights = []
        for domain, data in knowledge_graph.items():
            # Identify unique phrases
            unique_phrases = list(set(data["key_phrases"]))
            if unique_phrases:
                top_phrases = unique_phrases[:3] if len(unique_phrases) > 3 else unique_phrases
                key_phrase_insights.extend(top_phrases)
        
        if key_phrase_insights:
            new_insights.append(f"Key concepts include: {', '.join(key_phrase_insights[:5])}")
        
        # Extract applications if present
        if applications:
            # Clean and deduplicate applications
            clean_applications = []
            for app in applications:
                # Clean up and format application text
                app = app.strip()
                if len(app) > 15:  # Avoid very short fragments
                    clean_applications.append(app)
            
            unique_applications = list(set(clean_applications))
            if unique_applications:
                top_applications = unique_applications[:3] if len(unique_applications) > 3 else unique_applications
                new_insights.append(f"Potential applications include: {'; '.join(top_applications)}")
        
        hypotheses = [f"Understanding {context['query']} requires integrated knowledge across {', '.join(domains)}."]
        
    elif current_step == 1:
        # Challenges
        challenges = []
        
        # Search for challenge-related sentences across domains
        for domain, data in knowledge_graph.items():
            for content in data["raw_facts"]:
                sentences = content.split(".")
                for sentence in sentences:
                    if any(challenge_term in sentence.lower() for challenge_term in 
                           ["challenge", "limitation", "bottleneck", "difficult", "issue", "problem"]):
                        challenges.append(sentence.strip())
        
        new_insights = [
            f"There are multiple challenges associated with implementing quantum computing in machine learning",
            "These challenges span hardware limitations, algorithmic complexity, and theoretical questions about quantum advantage."
        ]
        
        if challenges:
            # Clean and deduplicate challenges
            unique_challenges = []
            for challenge in challenges:
                if len(challenge) > 20 and challenge not in unique_challenges:  # Avoid too short fragments
                    unique_challenges.append(challenge)
            
            if unique_challenges:
                top_challenges = unique_challenges[:2] if len(unique_challenges) > 2 else unique_challenges
                for challenge in top_challenges:
                    new_insights.append(f"Challenge: {challenge}")
        
        hypotheses = ["A comprehensive approach will need to address multiple technical challenges before widespread adoption."]
        
    elif current_step == 2:
        # Approaches and methods
        approaches = []
        
        # Extract approach-related terms from knowledge graph
        approach_terms = ["approach", "method", "technique", "algorithm", "framework"]
        
        for domain, data in knowledge_graph.items():
            for content in data["raw_facts"]:
                sentences = content.split(".")
                for sentence in sentences:
                    if any(approach_term in sentence.lower() for approach_term in approach_terms):
                        approaches.append(sentence.strip())
        
        new_insights = [
            "Multiple approaches exist with different strengths and limitations",
            "The choice of approach depends on specific requirements and available quantum resources."
        ]
        
        if approaches:
            # Clean and deduplicate approaches
            unique_approaches = []
            for approach in approaches:
                if len(approach) > 20 and approach not in unique_approaches:
                    unique_approaches.append(approach)
            
            if unique_approaches:
                top_approaches = unique_approaches[:2] if len(unique_approaches) > 2 else unique_approaches
                for approach in top_approaches:
                    new_insights.append(f"Approach: {approach}")
        
        hypotheses = ["Near-term quantum ML applications will likely focus on hybrid approaches that can work with NISQ devices."]
        
    elif current_step == 3:
        # Advantages and evaluation
        advantages = []
        
        # Extract advantage-related sentences
        advantage_terms = ["advantage", "speedup", "benefit", "outperform", "faster", "better", "improve"]
        
        for domain, data in knowledge_graph.items():
            for content in data["raw_facts"]:
                sentences = content.split(".")
                for sentence in sentences:
                    if any(advantage_term in sentence.lower() for advantage_term in advantage_terms):
                        advantages.append(sentence.strip())
        
        new_insights = [
            "Quantum ML approaches offer distinct advantages for specific computational bottlenecks",
            "The most promising applications address problems that are computationally intractable classically."
        ]
        
        if advantages:
            # Clean and deduplicate advantages
            unique_advantages = []
            for advantage in advantages:
                if len(advantage) > 20 and advantage not in unique_advantages:
                    unique_advantages.append(advantage)
            
            if unique_advantages:
                top_advantages = unique_advantages[:2] if len(unique_advantages) > 2 else unique_advantages
                for advantage in top_advantages:
                    new_insights.append(f"Advantage: {advantage}")
        
        conclusions = ["Different quantum approaches have distinct advantages for specific ML use cases."]
        
    else:
        # Synthesis and most promising applications
        # Now we'll use our aggregated applications list and the knowledge graph
        
        # Refine applications for final answer
        refined_applications = []
        
        # Group similar applications
        for app in applications:
            # Clean and process the application text
            app = app.strip()
            if len(app) > 15:  # Avoid very short fragments
                refined_applications.append(app)
        
        # Extract application categories by analyzing term frequencies
        app_categories = {}
        category_keywords = {
            "optimization": ["optimization", "optimizer", "minimization", "qaoa"],
            "classification": ["classification", "classifier", "svm", "support vector"],
            "dimensionality_reduction": ["dimension", "pca", "principal component"],
            "generative": ["generative", "sampling", "distribution", "boltzmann"],
            "drug_discovery": ["drug", "molecule", "chemical", "material"],
            "finance": ["finance", "financial", "portfolio", "trading"],
            "neural_networks": ["neural", "network", "qnn"],
            "kernel_methods": ["kernel", "feature space", "hilbert"]
        }
        
        # Analyze which categories the applications fall into
        for app in refined_applications:
            for category, keywords in category_keywords.items():
                if any(keyword in app.lower() for keyword in keywords):
                    if category not in app_categories:
                        app_categories[category] = []
                    app_categories[category].append(app)
        
        # Extract promising applications
        promising_applications = []
        for category, apps in app_categories.items():
            # Take the most detailed application from each category
            best_app = max(apps, key=len) if apps else None
            if best_app:
                if category == "optimization":
                    promising_applications.append("Quantum optimization for ML model training")
                elif category == "classification":
                    promising_applications.append("Quantum kernel methods for enhanced classification")
                elif category == "dimensionality_reduction":
                    promising_applications.append("Quantum dimensionality reduction for high-dimensional data")
                elif category == "generative":
                    promising_applications.append("Quantum generative models and sampling from complex distributions")
                elif category == "drug_discovery":
                    promising_applications.append("Drug discovery and materials science applications")
                elif category == "finance":
                    promising_applications.append("Financial modeling and optimization")
                elif category == "neural_networks":
                    promising_applications.append("Quantum neural networks for complex function approximation")
                elif category == "kernel_methods":
                    promising_applications.append("Quantum kernel methods for high-dimensional feature spaces")
        
        # Add general applications if we haven't found enough specific ones
        if len(promising_applications) < 3:
            general_apps = [
                "Quantum support vector machines for classification tasks",
                "Solving large systems of linear equations with HHL algorithm",
                "Quantum reinforcement learning",
                "Quantum computer vision and pattern recognition"
            ]
            # Add general applications that aren't already covered
            for app in general_apps:
                if not any(app.lower() in p.lower() for p in promising_applications):
                    promising_applications.append(app)
                    if len(promising_applications) >= 5:
                        break
        
        new_insights = [
            "The most promising quantum ML applications align with quantum computational advantages",
            "Near-term applications will likely focus on hybrid quantum-classical approaches"
        ]
        
        if promising_applications:
            new_insights.append(f"Most promising applications: {', '.join(promising_applications[:4])}")
        
        conclusions = [
            "The most promising applications of quantum computing in machine learning are those that leverage uniquely quantum properties to overcome classical computational bottlenecks."
        ]
        
        # Build a comprehensive final answer
        application_list = ""
        if promising_applications:
            application_list = "\n1. " + "\n2. ".join(promising_applications[:5])
        else:
            # Fallback if we couldn't extract structured applications
            application_list = "\n1. Quantum Support Vector Machines for classification tasks" + \
                              "\n2. Quantum Principal Component Analysis for dimensionality reduction" + \
                              "\n3. Quantum Neural Networks for complex function approximation" + \
                              "\n4. Quantum optimization for training machine learning models" + \
                              "\n5. Quantum algorithms for sampling from complex probability distributions"
            
        # Extract challenges from knowledge graph
        challenge_note = ""
        challenge_sentences = []
        
        for domain, data in knowledge_graph.items():
            for content in data["raw_facts"]:
                sentences = content.split(".")
                for sentence in sentences:
                    if any(challenge_term in sentence.lower() for challenge_term in 
                           ["challenge", "limitation", "bottleneck", "difficult", "issue", "problem"]):
                        challenge_sentences.append(sentence.strip())
        
        if challenge_sentences:
            challenge_note = "\n\nWhile these applications show promise, challenges remain in quantum hardware development, data loading efficiency, and identifying problems with genuine quantum advantage."
            
        final_answer = f"Based on analysis of knowledge from {', '.join(domains)}, the most promising applications of quantum computing in machine learning include:{application_list}{challenge_note}\n\nThe field is advancing rapidly with hybrid quantum-classical approaches showing the most near-term potential on NISQ devices."
        
        return {
            "reasoning": reasoning, 
            "new_insights": new_insights,
            "conclusions": conclusions,
            "final_answer": final_answer
        }
    
    # For middle steps, return without final answer
    result = {
        "reasoning": reasoning,
        "new_insights": new_insights
    }
    
    # Add hypotheses or conclusions if present
    if 'hypotheses' in locals():
        result["hypotheses"] = hypotheses
    if 'conclusions' in locals():
        result["conclusions"] = conclusions
    
    return result


def main():
    """Main function to run the KB agent workflow demo"""
    parser = argparse.ArgumentParser(description="KB Agent Workflow Demo")
    parser.add_argument("--workflow", type=str, default="research_report", 
                      help="Workflow to execute (research_report or business_analysis)")
    parser.add_argument("--query", type=str, default="artificial intelligence",
                      help="Query to use for the workflow")
    parser.add_argument("--output", type=str, default="./data/workflow_result.json",
                      help="Output file to save workflow results")
    parser.add_argument("--kb-dir", type=str, default="./knowledge_bases",
                      help="Directory containing knowledge base files")
    parser.add_argument("--embed-all", action="store_true",
                      help="Embed all knowledge bases at startup")
    parser.add_argument("--autonomous", action="store_true",
                      help="Run in autonomous problem-solving mode")
    parser.add_argument("--domains", type=str, nargs="+",
                      help="Specific domains to filter knowledge by")
    parser.add_argument("--kb-files", type=str, nargs="+",
                      help="Specific knowledge base files to load")
    
    args = parser.parse_args()
    
    # Run in autonomous problem-solving mode if requested
    if args.autonomous:
        autonomous_problem_solver(
            query=args.query,
            kb_files=args.kb_files,
            domains=args.domains
        )
        return
    
    # Regular workflow mode
    # Set up the environment
    workflow_engine = setup_demo_environment()
    
    # Load knowledge bases from directory if requested
    if args.embed_all:
        logger.info(f"Initializing knowledge base from {args.kb_dir}")
        for agent_id, agent in workflow_engine.agents.items():
            if agent.kb:
                logger.info(f"Initializing {agent_id} knowledge base...")
                agent.kb.initialize_from_directory(args.kb_dir)
    
    # Execute the workflow
    logger.info(f"\nExecuting workflow '{args.workflow}' with query '{args.query}'")
    result = workflow_engine.execute_workflow(args.workflow, {"query": args.query})
    
    # Save the result
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"\nWorkflow results saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Print summary
    logger.info("\nWorkflow Summary:")
    logger.info(f"Workflow: {args.workflow}")
    logger.info(f"Query: {args.query}")
    logger.info(f"Steps completed: {len(result['results'])}")
    
    if "report" in result["context"]:
        logger.info("\nReport Summary:")
        report = result["context"]["report"]
        for key, value in report.items():
            if isinstance(value, list):
                logger.info(f"- {key}:")
                for item in value[:3]:  # Show first 3 items
                    logger.info(f"  - {item}")
                if len(value) > 3:
                    logger.info(f"  - ... ({len(value) - 3} more)")
            else:
                logger.info(f"- {key}: {value}")
    
    logger.info("\nDone!")


if __name__ == "__main__":
    main()