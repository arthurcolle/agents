import duckdb
import os
import logging
import argparse
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-with-flockmtl")

class RAGWithFlockMTL:
    """
    Retrieval-Augmented Generation using FlockMTL with DuckDB.
    """
    
    def __init__(self, db_path: str = ':memory:', api_key: Optional[str] = None):
        """
        Initialize the RAG system with FlockMTL.
        
        Args:
            db_path: Path to the DuckDB database file or ':memory:' for in-memory database
            api_key: OpenAI API key for LLM operations (if not set in environment)
        """
        self.db_path = db_path
        self.conn = duckdb.connect(database=db_path)
        
        # Set OpenAI API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        # Load FlockMTL extension
        self._load_extension()
        
        # Set up model and prompt resources
        self._setup_resources()
        
        logger.info("RAG system initialized with database: %s", db_path)
    
    def _load_extension(self):
        """Load FlockMTL extension"""
        try:
            self.conn.execute("INSTALL flockmtl FROM community;")
            self.conn.execute("LOAD flockmtl;")
            logger.info("FlockMTL extension loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FlockMTL extension: {e}")
            raise
    
    def _setup_resources(self):
        """Set up model and prompt resources for FlockMTL"""
        try:
            # Create model resources - drop first if they exist
            self.conn.execute("DROP MODEL IF EXISTS 'embedding-model';")
            self.conn.execute("DROP MODEL IF EXISTS 'completion-model';")
            self.conn.execute("CREATE MODEL 'embedding-model' FROM openai (MODEL 'text-embedding-3-small');")
            self.conn.execute("CREATE MODEL 'completion-model' FROM openai (MODEL 'gpt-4o');")
            
            # Create prompt resources - drop first if they exist
            self.conn.execute("DROP PROMPT IF EXISTS 'retrieval-prompt';")
            self.conn.execute("DROP PROMPT IF EXISTS 'generation-prompt';")
            self.conn.execute("CREATE PROMPT 'retrieval-prompt' AS 'Search for documents that are relevant to answering this question.';")
            self.conn.execute("CREATE PROMPT 'generation-prompt' AS 'Based on the retrieved documents, answer the following question. If the documents do not contain relevant information, say so. Include citations to the document IDs you used in your answer.';")
            
            logger.info("Model and prompt resources set up successfully")
        except Exception as e:
            logger.error(f"Error setting up resources: {e}")
            raise
    
    def create_document_table(self):
        """Create a table for storing documents with vector embeddings"""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    title VARCHAR,
                    content TEXT,
                    embedding FLOAT[1536]
                );
            """)
            logger.info("Created document table")
        except Exception as e:
            logger.error(f"Error creating document table: {e}")
            raise
    
    def insert_documents(self, documents: List[Dict[str, Any]]):
        """
        Insert documents into the database and generate embeddings.
        
        Args:
            documents: List of document dictionaries with 'title' and 'content' fields
        """
        try:
            # First create the table if it doesn't exist
            self.create_document_table()
            
            # Insert documents and generate embeddings
            for i, doc in enumerate(documents):
                # Generate embedding using FlockMTL's llm_embedding function
                self.conn.execute(f"""
                    INSERT INTO documents (id, title, content, embedding)
                    SELECT 
                        {i+1} as id,
                        '{doc['title'].replace("'", "''")}' as title,
                        '{doc['content'].replace("'", "''")}' as content,
                        llm_embedding({{'model_name': 'embedding-model'}}, 
                                     {{'content': '{doc['content'].replace("'", "''")}'}}
                        )::FLOAT[1536] as embedding;
                """)
            
            logger.info(f"Inserted {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            raise
    
    def retrieve_and_generate(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Perform RAG: retrieve relevant documents and generate an answer.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with query, retrieved documents, and generated answer
        """
        try:
            # Generate query embedding
            self.conn.execute(f"""
                CREATE OR REPLACE TEMPORARY TABLE query_embedding AS
                SELECT llm_embedding({{'model_name': 'embedding-model'}}, 
                                   {{'query': '{query.replace("'", "''")}'}})::FLOAT[1536] AS embedding;
            """)
            
            # Retrieve relevant documents
            retrieved_docs = self.conn.execute(f"""
                SELECT 
                    d.id, 
                    d.title, 
                    d.content,
                    array_cosine_similarity(q.embedding, d.embedding) AS similarity
                FROM documents d, query_embedding q
                ORDER BY similarity DESC
                LIMIT {k};
            """).fetchall()
            
            # Format retrieved documents for context
            context = ""
            for i, doc in enumerate(retrieved_docs):
                context += f"Document {doc[0]}: {doc[1]}\n{doc[2]}\n\n"
            
            # Generate answer using llm_complete
            answer = self.conn.execute(f"""
                SELECT llm_complete(
                    {{'model_name': 'completion-model'}},
                    {{'prompt_name': 'generation-prompt'}},
                    {{'question': '{query.replace("'", "''")}', 'context': '{context.replace("'", "''")}'}}
                );
            """).fetchone()[0]
            
            # Prepare result
            result = {
                "query": query,
                "retrieved_documents": [
                    {"id": doc[0], "title": doc[1], "content": doc[2], "similarity": doc[3]}
                    for doc in retrieved_docs
                ],
                "answer": answer
            }
            
            logger.info(f"Generated answer for query: '{query}'")
            return result
        except Exception as e:
            logger.error(f"Error in retrieve_and_generate: {e}")
            raise
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Example usage of the RAGWithFlockMTL class"""
    # Initialize with your OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    rag = RAGWithFlockMTL(api_key=api_key)
    
    # Sample documents about FlockMTL and DuckDB
    documents = [
        {
            "title": "FlockMTL Overview",
            "content": "FlockMTL is an extension for DuckDB that deeply integrates LLM capabilities and retrieval-augmented generation. It includes model-driven scalar and aggregate functions, enabling chained predictions through tuple-level mappings and reductions."
        },
        {
            "title": "FlockMTL Features",
            "content": "FlockMTL supports a broad range of semantic operations, including classification, summarization, and reranking. It introduces new DDL resource types: MODELs and PROMPTs, treated as first-class schema objects akin to TABLEs."
        },
        {
            "title": "FlockMTL Optimizations",
            "content": "FlockMTL handles lower-level implementation details like LLM context management, batching on input tuples, caching and reusing of results, and predictions on unique input values seamlessly."
        },
        {
            "title": "DuckDB Introduction",
            "content": "DuckDB is an in-process SQL OLAP database management system. It is designed to be embedded within applications and to execute analytical SQL queries on data efficiently."
        },
        {
            "title": "Vector Similarity Search in DuckDB",
            "content": "The VSS extension adds indexing support to accelerate vector similarity search queries using DuckDB's fixed-size ARRAY type. It supports HNSW indexes with various distance metrics like L2, cosine, and inner product."
        }
    ]
    
    # Insert documents
    rag.insert_documents(documents)
    
    # Example queries
    queries = [
        "What is FlockMTL and how does it relate to DuckDB?",
        "How does FlockMTL optimize LLM operations?",
        "What types of vector similarity metrics are supported in DuckDB?"
    ]
    
    # Process each query
    for query in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)
        
        result = rag.retrieve_and_generate(query)
        
        print("\nRETRIEVED DOCUMENTS:")
        print("-" * 40)
        for i, doc in enumerate(result["retrieved_documents"]):
            print(f"{i+1}. {doc['title']} (Similarity: {doc['similarity']:.4f})")
            print(f"   {doc['content'][:100]}...")
        
        print("\nGENERATED ANSWER:")
        print("-" * 40)
        print(result["answer"])
    
    # Close connection
    rag.close()

if __name__ == "__main__":
    main()
