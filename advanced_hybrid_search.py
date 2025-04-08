import duckdb
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("advanced-hybrid-search")

class AdvancedHybridSearch:
    """
    Advanced hybrid search implementation using FlockMTL's specialized functions
    for reranking and fusion, combined with VSS and FTS extensions.
    """
    
    def __init__(self, db_path: str = ':memory:', api_key: Optional[str] = None):
        """
        Initialize the advanced hybrid search system.
        
        Args:
            db_path: Path to the DuckDB database file or ':memory:' for in-memory database
            api_key: OpenAI API key for LLM operations (if not set in environment)
        """
        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path)
        
        # Set OpenAI API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Load required extensions
        self._load_extensions()
        
        # Set up model and prompt resources
        self._setup_resources()
        
        logger.info("AdvancedHybridSearch initialized with database: %s", db_path)
    
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
            
            logger.info("All extensions loaded successfully")
        except Exception as e:
            logger.error(f"Error loading extensions: {e}")
            raise
    
    def _setup_resources(self):
        """Set up model and prompt resources for FlockMTL"""
        try:
            # Create model resources
            self.conn.execute("""
                CREATE OR REPLACE MODEL('embedding-model', 'text-embedding-3-small', 'openai');
                CREATE OR REPLACE MODEL('reranking-model', 'gpt-4o-mini', 'openai');
            """)
            
            # Create prompt resources
            self.conn.execute("""
                CREATE OR REPLACE PROMPT('reranking-prompt', 
                    'Rank the following documents based on their relevance to the query. 
                     Consider semantic meaning, not just keyword matching. 
                     The most relevant document should be ranked first.');
            """)
            
            logger.info("Model and prompt resources set up successfully")
        except Exception as e:
            logger.error(f"Error setting up resources: {e}")
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
                # Generate embedding using FlockMTL's llm_embedding function with the defined model
                self.conn.execute(f"""
                    INSERT INTO {table_name} (id, title, content, embedding)
                    SELECT 
                        {i+1} as id,
                        '{doc['title'].replace("'", "''")}' as title,
                        '{doc['content'].replace("'", "''")}' as content,
                        llm_embedding({{'model_name': 'embedding-model'}}, 
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
    
    def advanced_hybrid_search(self, query: str, k: int = 5, table_name: str = "documents") -> List[Dict[str, Any]]:
        """
        Perform advanced hybrid search using FlockMTL's specialized functions.
        
        Args:
            query: Search query string
            k: Number of results to return
            table_name: Name of the table to search
            
        Returns:
            List of document dictionaries with search scores
        """
        try:
            # Generate query embedding
            self.conn.execute(f"""
                CREATE OR REPLACE TEMPORARY TABLE query_embedding AS
                SELECT llm_embedding({{'model_name': 'embedding-model'}}, 
                                   {{'query': '{query.replace("'", "''")}'}})::FLOAT[1536] AS embedding;
            """)
            
            # Perform hybrid search with fusion and reranking
            result = self.conn.execute(f"""
                WITH 
                -- Vector similarity search
                vector_results AS (
                    SELECT 
                        d.id, 
                        d.title, 
                        d.content,
                        array_cosine_similarity(q.embedding, d.embedding) AS score
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
                        fts_main_{table_name}.match_bm25(id, '{query.replace("'", "''")}') AS score
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
                        -- Use fusion function from FlockMTL
                        fusion(
                            COALESCE(v.score, 0)::DOUBLE / (SELECT MAX(score) FROM vector_results),
                            COALESCE(b.score, 0)::DOUBLE / (SELECT MAX(score) FROM bm25_results)
                        ) AS fusion_score
                    FROM vector_results v
                    FULL OUTER JOIN bm25_results b ON v.id = b.id
                    ORDER BY fusion_score DESC
                    LIMIT 20
                ),
                
                -- Prepare data for reranking
                rerank_input AS (
                    SELECT 
                        id,
                        title,
                        content,
                        fusion_score,
                        '{query.replace("'", "''")}' AS query
                    FROM combined_results
                    ORDER BY fusion_score DESC
                    LIMIT 20
                )
                
                -- Apply LLM reranking to the top candidates
                SELECT 
                    r.id,
                    r.title,
                    r.content,
                    r.fusion_score AS initial_score,
                    llm_rerank(
                        {{'model_name': 'reranking-model'}},
                        {{'prompt_name': 'reranking-prompt'}},
                        {{'query': r.query, 'document': r.content}}
                    ) AS reranked_score
                FROM rerank_input r
                ORDER BY reranked_score DESC
                LIMIT {k};
            """).fetchall()
            
            # Convert result to list of dictionaries
            results = []
            for row in result:
                results.append({
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "initial_score": row[3],
                    "reranked_score": row[4]
                })
            
            logger.info(f"Advanced hybrid search for '{query}' returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error performing advanced hybrid search: {e}")
            raise
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Example usage of the AdvancedHybridSearch class with command-line arguments"""
    parser = argparse.ArgumentParser(description="Advanced Hybrid Search using FlockMTL")
    parser.add_argument("--query", type=str, default="vector similarity for search",
                        help="Search query")
    parser.add_argument("--results", type=int, default=3,
                        help="Number of results to return")
    parser.add_argument("--db", type=str, default=":memory:",
                        help="Path to DuckDB database file (default: in-memory)")
    args = parser.parse_args()
    
    # Initialize with your OpenAI API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    search = AdvancedHybridSearch(db_path=args.db, api_key=api_key)
    
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
        },
        {
            "title": "Retrieval-Augmented Generation",
            "content": "Retrieval-Augmented Generation (RAG) is a technique that enhances language model outputs by retrieving relevant information from external sources before generating responses."
        },
        {
            "title": "SQL and LLMs",
            "content": "Combining SQL with Large Language Models allows for powerful data analysis workflows that can process both structured and unstructured data in a unified manner."
        },
        {
            "title": "BM25 Ranking Algorithm",
            "content": "BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query. It's based on the probabilistic retrieval framework."
        },
        {
            "title": "Embedding Models",
            "content": "Embedding models convert text into numerical vector representations that capture semantic meaning, allowing for similarity comparisons between different pieces of text."
        },
        {
            "title": "Reciprocal Rank Fusion",
            "content": "Reciprocal Rank Fusion (RRF) is a technique for combining multiple ranked lists into a single ranking, often used in hybrid search systems to merge results from different retrieval methods."
        }
    ]
    
    # Insert documents and create indexes
    search.insert_documents(documents)
    search.create_indexes()
    
    # Perform advanced hybrid search
    results = search.advanced_hybrid_search(args.query, k=args.results)
    
    # Print results
    print(f"\nAdvanced search results for: '{args.query}'")
    print("-" * 60)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']}")
        print(f"   Initial Score: {result['initial_score']:.4f}, Reranked Score: {result['reranked_score']:.4f}")
        print(f"   {result['content'][:100]}...")
        print()
    
    # Close connection
    search.close()

if __name__ == "__main__":
    main()
