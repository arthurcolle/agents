import duckdb
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hybrid-search")

class HybridSearch:
    """
    Hybrid search implementation combining vector similarity search and full-text search
    using DuckDB with FlockMTL, VSS, and FTS extensions.
    """
    
    def __init__(self, db_path: str = ':memory:', api_key: Optional[str] = None, 
                 use_mock_embeddings: bool = True):
        """
        Initialize the hybrid search system.
        
        Args:
            db_path: Path to the DuckDB database file or ':memory:' for in-memory database
            api_key: OpenAI API key for embedding generation (if not set in environment)
            use_mock_embeddings: Force using mock embeddings even if API key is available
        """
        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path)
        self.use_mock_embeddings = use_mock_embeddings
        
        # Set OpenAI API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Load required extensions
        self._load_extensions()
        
        logger.info("HybridSearch initialized with database: %s", db_path)
    
    def _load_extensions(self):
        """Load required DuckDB extensions"""
        try:
            # Install and load FlockMTL extension
            self.conn.execute("INSTALL flockmtl FROM community;")
            self.conn.execute("LOAD flockmtl;")
            
            # Install and load VSS extension for vector similarity search
            self.conn.execute("INSTALL vss;")
            self.conn.execute("LOAD vss;")
            
            # Install and load FTS extension for full-text search
            self.conn.execute("INSTALL fts;")
            self.conn.execute("LOAD fts;")
            
            # Load environment variables into DB
            self._load_env_vars_to_db()
            
            logger.info("All extensions loaded successfully")
        except Exception as e:
            logger.error(f"Error loading extensions: {e}")
            raise
    
    def _load_env_vars_to_db(self):
        """Load all environment variables into DuckDB as a local table"""
        try:
            # Create a temporary table to store environment variables
            self.conn.execute("CREATE TEMPORARY TABLE IF NOT EXISTS env_vars (name VARCHAR, value VARCHAR)")
            
            # Insert all environment variables
            for name, value in os.environ.items():
                # Skip variables with empty values
                if not value:
                    continue
                    
                # Escape single quotes in values
                escaped_value = value.replace("'", "''")
                
                # Insert into temporary table
                self.conn.execute(f"INSERT INTO env_vars VALUES ('{name}', '{escaped_value}')")
            
            logger.info(f"Loaded {self.conn.execute('SELECT COUNT(*) FROM env_vars').fetchone()[0]} environment variables into database")
        except Exception as e:
            logger.warning(f"Error loading environment variables into database: {e}")
            # Continue execution even if this fails
            pass
    
    def create_document_table(self, table_name: str = "documents"):
        """Create a table for storing documents with vector embeddings"""
        try:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY,
                    title VARCHAR,
                    content TEXT,
                    embedding FLOAT[]
                );
            """)
            logger.info(f"Created document table: {table_name}")
        except Exception as e:
            logger.error(f"Error creating document table: {e}")
            raise
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding for testing without API keys"""
        import hashlib
        import random
        
        # Create a deterministic seed based on text hash
        hash_obj = hashlib.sha256(text.encode())
        hash_hex = hash_obj.hexdigest()
        seed = int(hash_hex, 16) % (2**32)
        random.seed(seed)
        
        # Generate 1536 float values (same dimension as text-embedding-3-small)
        embedding = []
        for _ in range(1536):
            # Generate a small random value between -0.1 and 0.1
            val = (random.random() - 0.5) * 0.2
            embedding.append(val)
        
        # Normalize the embedding
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x/magnitude for x in embedding]
            
        return embedding
    
    def insert_documents(self, documents: List[Dict[str, Any]], table_name: str = "documents"):
        """
        Insert documents into the database and generate embeddings.
        
        Args:
            documents: List of document dictionaries with 'title' and 'content' fields
            table_name: Name of the table to insert into
        """
        try:
            # First create the table if it doesn't exist
            self.create_document_table(table_name)
            
            # Insert documents and generate embeddings
            for i, doc in enumerate(documents):
                try:
                    if self.use_mock_embeddings:
                        # Generate mock embedding
                        mock_embedding = self._generate_mock_embedding(doc['content'])
                        
                        # Convert to DuckDB array format with proper escaping
                        mock_embedding_values = ", ".join([str(val) for val in mock_embedding])
                        
                        # Insert with mock embedding
                        self.conn.execute(f"""
                            INSERT INTO {table_name} (id, title, content, embedding)
                            VALUES (
                                {i+1},
                                '{doc['title'].replace("'", "''")}',
                                '{doc['content'].replace("'", "''")}',
                                ARRAY[{mock_embedding_values}]::FLOAT[]
                            );
                        """)
                    else:
                        try:
                            # Try to generate embedding using FlockMTL's llm_embedding function
                            self.conn.execute(f"""
                                INSERT INTO {table_name} (id, title, content, embedding)
                                SELECT 
                                    {i+1} as id,
                                    '{doc['title'].replace("'", "''")}' as title,
                                    '{doc['content'].replace("'", "''")}' as content,
                                    llm_embedding({{'model_name':'text-embedding-3-small'}}, 
                                                 {{'content': '{doc['content'].replace("'", "''")}'}}
                                    )::FLOAT[] as embedding;
                            """)
                        except Exception as api_error:
                            # Fallback to mock embedding if API call fails
                            logger.warning(f"Failed to generate embedding via API: {api_error}. Using mock embedding.")
                            
                            # Generate mock embedding
                            mock_embedding = self._generate_mock_embedding(doc['content'])
                            
                            # Convert to DuckDB array format with proper escaping
                            mock_embedding_values = ", ".join([str(val) for val in mock_embedding])
                            
                            # Insert with mock embedding
                            self.conn.execute(f"""
                                INSERT INTO {table_name} (id, title, content, embedding)
                                VALUES (
                                    {i+1},
                                    '{doc['title'].replace("'", "''")}',
                                    '{doc['content'].replace("'", "''")}',
                                    ARRAY[{mock_embedding_values}]::FLOAT[]
                                );
                            """)
                except Exception as doc_error:
                    logger.error(f"Failed to insert document {i+1}: {doc_error}")
            
            logger.info(f"Inserted {len(documents)} documents into {table_name}")
        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            raise
    
    def create_indexes(self, table_name: str = "documents"):
        """Create FTS index on the document table (skip HNSW index for compatibility)"""
        try:
            # Create FTS index for full-text search
            self.conn.execute(f"""
                PRAGMA create_fts_index(
                    '{table_name}', 'id', 'title', 'content', 
                    stemmer = 'porter', 
                    stopwords = 'english', 
                    ignore = '(\\\\.|[^a-z])+',
                    strip_accents = 1, 
                    lower = 1, 
                    overwrite = 1
                );
            """)
            logger.info(f"Created FTS index on {table_name}")
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
            raise
    
            
    def hybrid_search(self, query: str, k: int = 5, table_name: str = "documents", 
                      fusion_method: str = "rrf", k_constant: float = 60.0) -> List[Dict[str, Any]]:
        """
        Perform search using BM25 text search instead of hybrid search.
        (Vector similarity disabled for compatibility)
        
        Args:
            query: Search query string
            k: Number of results to return
            table_name: Name of the table to search
            fusion_method: Not used in this version
            k_constant: Not used in this version
            
        Returns:
            List of document dictionaries with search scores
        """
        try:
            # Perform text search
            result = self.conn.execute(f"""
                SELECT 
                    d.id, 
                    d.title, 
                    d.content,
                    fts_main_{table_name}.match_bm25(id, '{query.replace("'", "''")}') AS score
                FROM {table_name} d
                WHERE fts_main_{table_name}.match_bm25(id, '{query.replace("'", "''")}') IS NOT NULL
                ORDER BY score DESC
                LIMIT {k};
            """).fetchall()
            
            # Convert result to list of dictionaries
            results = []
            for row in result:
                results.append({
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "score": row[3]
                })
            
            logger.info(f"Full-text search for '{query}' returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            raise
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Example usage of the HybridSearch class"""
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Hybrid search demo using DuckDB and FlockMTL")
    parser.add_argument("--query", "-q", type=str, default="vector similarity for search",
                        help="Search query to execute")
    parser.add_argument("--mock", "-m", action="store_true", default=True,
                        help="Use mock embeddings instead of API (default: True)")
    parser.add_argument("--results", "-r", type=int, default=3,
                        help="Number of results to return (default: 3)")
    
    args = parser.parse_args()
    
    # Initialize search system
    search = HybridSearch(use_mock_embeddings=args.mock)
    
    # Sample documents
    documents = [
        {
            "title": "Introduction to DuckDB",
            "content": "DuckDB is an in-process SQL OLAP database management system. It is designed to be embedded within applications and to execute analytical SQL queries on data."
        },
        {
            "title": "Vector Similarity Search",
            "content": "Vector similarity search allows finding similar items based on their vector representations. It's commonly used in recommendation systems and semantic search."
        },
        {
            "title": "Full-Text Search",
            "content": "Full-text search is a technique for searching text content in documents. It typically uses inverted indexes and relevance scoring like BM25."
        },
        {
            "title": "Hybrid Search Systems",
            "content": "Hybrid search combines multiple search techniques such as keyword-based search and vector similarity search to improve search quality and relevance."
        },
        {
            "title": "FlockMTL Extension",
            "content": "FlockMTL is a DuckDB extension that integrates LLM capabilities and retrieval-augmented generation directly into SQL queries for knowledge-intensive analytical applications."
        }
    ]
    
    # Insert documents and create indexes
    search.insert_documents(documents)
    search.create_indexes()
    
    # Perform search with the provided query
    results = search.hybrid_search(args.query, k=args.results)
    
    # Print results
    print(f"\nSearch results for: '{args.query}'")
    print("-" * 50)
    if results:
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
            print(f"   {result['content'][:100]}...")
            print()
    else:
        print("No results found.")
    
    # Close connection
    search.close()

if __name__ == "__main__":
    main()
