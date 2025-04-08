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
    
    def __init__(self, db_path: str = ':memory:', api_key: Optional[str] = None):
        """
        Initialize the hybrid search system.
        
        Args:
            db_path: Path to the DuckDB database file or ':memory:' for in-memory database
            api_key: OpenAI API key for embedding generation (if not set in environment)
        """
        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path)
        self.use_mock_embeddings = False
        
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
            
            # Set up OpenAI API key as a secret for FlockMTL
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key:
                self.conn.execute(f"CREATE OR REPLACE SECRET openai (api_key='{api_key}');")
            else:
                logger.warning("OPENAI_API_KEY environment variable not set. Using mock embeddings.")
                # Use mock embeddings if no API key is available
                self.use_mock_embeddings = True
            
            logger.info("All extensions loaded successfully")
        except Exception as e:
            logger.error(f"Error loading extensions: {e}")
            raise
    
    def create_document_table(self, table_name: str = "documents"):
        """Create a table for storing documents with vector embeddings"""
        try:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY,
                    title VARCHAR,
                    content TEXT,
                    embedding FLOAT[1536]
                );
            """)
            logger.info(f"Created document table: {table_name}")
        except Exception as e:
            logger.error(f"Error creating document table: {e}")
            raise
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate a deterministic mock embedding for testing without API keys"""
        import hashlib
        import struct
        
        # Create a deterministic but seemingly random embedding based on text hash
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Generate 1536 float values (same dimension as text-embedding-3-small)
        embedding = []
        for i in range(0, 1536):
            # Use different parts of the hash to seed different values
            byte_pos = i % 32
            val = struct.unpack('f', hash_bytes[byte_pos:byte_pos+1] * 4)[0]
            # Normalize to typical embedding range
            val = (val % 1.0) * 0.1
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
                if self.use_mock_embeddings:
                    # Generate mock embedding
                    mock_embedding = self._generate_mock_embedding(doc['content'])
                    mock_embedding_str = str(mock_embedding).replace('[', 'ARRAY[')
                    
                    # Insert with mock embedding
                    self.conn.execute(f"""
                        INSERT INTO {table_name} (id, title, content, embedding)
                        VALUES (
                            {i+1},
                            '{doc['title'].replace("'", "''")}',
                            '{doc['content'].replace("'", "''")}',
                            {mock_embedding_str}
                        );
                    """)
                else:
                    # Generate embedding using FlockMTL's llm_embedding function
                    self.conn.execute(f"""
                        INSERT INTO {table_name} (id, title, content, embedding)
                        SELECT 
                            {i+1} as id,
                            '{doc['title'].replace("'", "''")}' as title,
                            '{doc['content'].replace("'", "''")}' as content,
                            llm_embedding({{'model_name':'text-embedding-3-small'}}, 
                                         {{'content': '{doc['content'].replace("'", "''")}'}}
                            )::FLOAT[1536] as embedding;
                    """)
            
            logger.info(f"Inserted {len(documents)} documents into {table_name}")
        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            raise
    
    def create_indexes(self, table_name: str = "documents"):
        """Create HNSW and FTS indexes on the document table"""
        try:
            # Create HNSW index for vector similarity search
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS hnsw_idx ON {table_name} 
                USING HNSW (embedding) WITH (metric = 'cosine');
            """)
            logger.info(f"Created HNSW index on {table_name}")
            
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
        Perform hybrid search combining vector similarity and BM25 text search.
        
        Args:
            query: Search query string
            k: Number of results to return
            table_name: Name of the table to search
            fusion_method: Fusion method to use ('rrf' for Reciprocal Rank Fusion)
            k_constant: k constant for RRF fusion
            
        Returns:
            List of document dictionaries with search scores
        """
        try:
            # Generate query embedding
            if self.use_mock_embeddings:
                # Generate mock embedding for query
                mock_embedding = self._generate_mock_embedding(query)
                mock_embedding_str = str(mock_embedding).replace('[', 'ARRAY[')
                
                self.conn.execute(f"""
                    CREATE OR REPLACE TEMPORARY TABLE query_embedding AS
                    SELECT {mock_embedding_str}::FLOAT[1536] AS embedding;
                """)
            else:
                self.conn.execute(f"""
                    CREATE OR REPLACE TEMPORARY TABLE query_embedding AS
                    SELECT llm_embedding({{'model_name':'text-embedding-3-small'}}, 
                                       {{'query': '{query.replace("'", "''")}'}})::FLOAT[1536] AS embedding;
                """)
            
            # Perform hybrid search with fusion
            result = self.conn.execute(f"""
                WITH 
                -- Vector similarity search
                vector_results AS (
                    SELECT 
                        d.id, 
                        d.title, 
                        d.content,
                        array_cosine_similarity(q.embedding, d.embedding) AS score,
                        ROW_NUMBER() OVER (ORDER BY array_cosine_similarity(q.embedding, d.embedding) DESC) AS rank
                    FROM {table_name} d, query_embedding q
                    ORDER BY score DESC
                    LIMIT 100
                ),
                
                -- BM25 full-text search
                bm25_results AS (
                    SELECT 
                        d.id, 
                        d.title, 
                        d.content,
                        fts_main_{table_name}.match_bm25(id, '{query.replace("'", "''")}') AS score,
                        ROW_NUMBER() OVER (ORDER BY fts_main_{table_name}.match_bm25(id, '{query.replace("'", "''")}') DESC) AS rank
                    FROM {table_name} d
                    WHERE fts_main_{table_name}.match_bm25(id, '{query.replace("'", "''")}') IS NOT NULL
                    ORDER BY score DESC
                    LIMIT 100
                ),
                
                -- Combine results with fusion
                combined_results AS (
                    SELECT 
                        COALESCE(v.id, b.id) AS id,
                        COALESCE(v.title, b.title) AS title,
                        COALESCE(v.content, b.content) AS content,
                        -- RRF fusion score
                        (1.0 / ({k_constant} + COALESCE(v.rank, 1000))) + 
                        (1.0 / ({k_constant} + COALESCE(b.rank, 1000))) AS fusion_score
                    FROM vector_results v
                    FULL OUTER JOIN bm25_results b ON v.id = b.id
                    ORDER BY fusion_score DESC
                    LIMIT {k}
                )
                
                SELECT id, title, content, fusion_score
                FROM combined_results
                ORDER BY fusion_score DESC;
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
            
            logger.info(f"Hybrid search for '{query}' returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            raise
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Example usage of the HybridSearch class"""
    # Initialize with your OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    search = HybridSearch(api_key=api_key)
    
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
    
    # Perform hybrid search
    query = "vector similarity for search"
    results = search.hybrid_search(query, k=3)
    
    # Print results
    print(f"\nSearch results for: '{query}'")
    print("-" * 50)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   {result['content'][:100]}...")
        print()
    
    # Close connection
    search.close()

if __name__ == "__main__":
    main()
