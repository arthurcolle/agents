#!/usr/bin/env python3
import ast
import os
import sys
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import time
import sqlite3
import uuid
import hashlib
import ast
from pathlib import Path
import importlib

try:
    import numpy as np
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

@dataclass
class CodeNode:
    """Base class for code structure nodes with embedding support."""
    node_id: str
    source_file: str
    line_start: int
    line_end: int
    code_text: str
    node_type: str
    name: Optional[str] = None
    parent_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StructuredOutput:
    """Base class for structured data extraction from model outputs."""
    schema_id: str
    timestamp: float = field(default_factory=time.time)
    
@dataclass
class FunctionCall(StructuredOutput):
    """Structured representation of a function call extracted from text."""
    name: str
    arguments: Dict[str, Any]
    raw_text: str
    
@dataclass
class ParallelFunctionCalls(StructuredOutput):
    """Container for multiple function calls to be executed in parallel."""
    calls: List[FunctionCall]
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class CodeStructureAnalyzer:
    """Analyzes and indexes code for efficient retrieval and manipulation."""
    
    def __init__(self, db_path: str = "code_artifacts.db"):
        """Initialize the code structure analyzer with storage for embeddings."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._initialize_db()
        self.model = None  # Lazy load embedding model
        
    def _initialize_db(self):
        """Initialize the database schema."""
        cursor = self.conn.cursor()
        
        # Create nodes table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS code_nodes (
            node_id TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            line_start INTEGER NOT NULL,
            line_end INTEGER NOT NULL,
            code_text TEXT NOT NULL,
            node_type TEXT NOT NULL,
            name TEXT,
            parent_id TEXT,
            embedding BLOB,
            metadata TEXT
        )
        ''')
        
        # Create index on source_file for faster lookups
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_code_nodes_source_file ON code_nodes(source_file)
        ''')
        
        # Create index on node_type for faster lookups
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_code_nodes_node_type ON code_nodes(node_type)
        ''')
        
        # Create function calls table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS function_calls (
            call_id TEXT PRIMARY KEY,
            schema_id TEXT NOT NULL,
            name TEXT NOT NULL,
            arguments TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            timestamp REAL NOT NULL,
            batch_id TEXT
        )
        ''')
        
        self.conn.commit()
    
    def _get_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model
    
    def compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for a text snippet."""
        model = self._get_model()
        embedding = model.encode(text)
        return embedding.tolist()
    
    def analyze_file(self, file_path: str):
        """Parse and index a Python file's structure."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content, filename=file_path)
            source_lines = content.splitlines()
            self._process_ast_node(tree, file_path, source_lines, content)
            return True
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return False
    
    def _process_ast_node(self, node, file_path: str, source_lines: List[str], full_content: str, parent_id: Optional[str] = None):
        """Process an AST node and its children to extract code structure."""
        # Skip nodes without line information
        if not hasattr(node, 'lineno'):
            return
        
        # Get node line information
        line_start = node.lineno
        line_end = node.end_lineno if hasattr(node, 'end_lineno') else line_start
        
        # Extract node code text
        if line_start <= len(source_lines) and line_end <= len(source_lines):
            if line_start == line_end:
                code_text = source_lines[line_start - 1]
            else:
                code_text = '\n'.join(source_lines[line_start - 1:line_end])
        else:
            code_text = ""
        
        # Determine node type and name
        node_type = node.__class__.__name__
        name = None
        
        # Extract names for common node types
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.ClassDef):
            name = node.name
        elif isinstance(node, ast.Assign) and node.targets:
            if isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
        
        # Create a unique node ID
        node_id = str(uuid.uuid4())
        
        # Create metadata based on node type
        metadata = {}
        
        # Add specific metadata for different node types
        if isinstance(node, ast.FunctionDef):
            metadata['args'] = [arg.arg for arg in node.args.args]
            metadata['returns'] = getattr(node, 'returns', None)
            # Extract docstring if available
            if (node.body and isinstance(node.body[0], ast.Expr) and 
                isinstance(node.body[0].value, ast.Str)):
                metadata['docstring'] = node.body[0].value.s
                
        elif isinstance(node, ast.ClassDef):
            metadata['bases'] = [base.id for base in node.bases if isinstance(base, ast.Name)]
            # Extract methods
            metadata['methods'] = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            
        # Compute embedding for the code text
        embedding = self.compute_embedding(code_text)
        
        # Create and store the code node
        code_node = CodeNode(
            node_id=node_id,
            source_file=file_path,
            line_start=line_start,
            line_end=line_end,
            code_text=code_text,
            node_type=node_type,
            name=name,
            parent_id=parent_id,
            embedding=embedding,
            metadata=metadata
        )
        
        self._store_node(code_node)
        
        # Recursively process children
        for child in ast.iter_child_nodes(node):
            self._process_ast_node(child, file_path, source_lines, full_content, node_id)
    
    def _store_node(self, node: CodeNode):
        """Store a code node in the database."""
        cursor = self.conn.cursor()
        
        # Convert embedding to binary blob
        embedding_blob = None
        if node.embedding:
            embedding_blob = np.array(node.embedding, dtype=np.float32).tobytes()
        
        # Convert metadata to JSON
        metadata_json = json.dumps(node.metadata)
        
        cursor.execute(
            "INSERT INTO code_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (node.node_id, node.source_file, node.line_start, node.line_end, 
             node.code_text, node.node_type, node.name, node.parent_id,
             embedding_blob, metadata_json)
        )
        
        self.conn.commit()
    
    def search(self, query: str, limit: int = 10) -> List[CodeNode]:
        """Search for code nodes matching a query using embeddings similarity."""
        query_embedding = self.compute_embedding(query)
        
        # Convert to numpy array for comparison
        query_embedding_np = np.array(query_embedding, dtype=np.float32)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM code_nodes")
        rows = cursor.fetchall()
        
        results = []
        for row in cursor.description:
            print(row[0])
        
        for row in rows:
            # Extract embedding from binary blob
            embedding_blob = row[8]
            if embedding_blob:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                
                # Compute cosine similarity
                similarity = np.dot(query_embedding_np, embedding) / (
                    np.linalg.norm(query_embedding_np) * np.linalg.norm(embedding)
                )
                
                node = CodeNode(
                    node_id=row[0],
                    source_file=row[1],
                    line_start=row[2],
                    line_end=row[3],
                    code_text=row[4],
                    node_type=row[5],
                    name=row[6],
                    parent_id=row[7],
                    embedding=None,  # Don't include embedding in results
                    metadata=json.loads(row[9])
                )
                
                results.append((node, similarity))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in results[:limit]]
    
    def search_by_regex(self, pattern: str) -> List[CodeNode]:
        """Search for code nodes by regex pattern."""
        compiled_pattern = re.compile(pattern)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM code_nodes")
        
        results = []
        for row in cursor.fetchall():
            if compiled_pattern.search(row[4]):  # code_text is at index 4
                node = CodeNode(
                    node_id=row[0],
                    source_file=row[1],
                    line_start=row[2],
                    line_end=row[3],
                    code_text=row[4],
                    node_type=row[5],
                    name=row[6],
                    parent_id=row[7],
                    embedding=None,
                    metadata=json.loads(row[9])
                )
                results.append(node)
        
        return results

# Function parsing utilities
def parse_function_calls_json(text: str, schema_id: str = "json_function_call") -> List[FunctionCall]:
    """
    Parse function calls from text using JSON schema format.
    
    Extracts function calls using the JSON schema defined format:
    {"name": "function_name", "arguments": {"arg1": "value1", "arg2": "value2"}}
    
    Args:
        text: Text containing potential JSON function calls
        schema_id: Identifier for the schema used
        
    Returns:
        List of FunctionCall objects
    """
    function_calls = []
    
    # Find JSON objects in the text
    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    potential_matches = re.findall(json_pattern, text)
    
    for potential_json in potential_matches:
        try:
            data = json.loads(potential_json)
            
            # Check if this looks like a function call
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                function_calls.append(FunctionCall(
                    schema_id=schema_id,
                    name=data["name"],
                    arguments=data["arguments"],
                    raw_text=potential_json
                ))
        except json.JSONDecodeError:
            continue
    
    return function_calls

def parse_parallel_function_calls(text: str) -> Optional[ParallelFunctionCalls]:
    """
    Parse parallel function calls from text.
    
    Looks for a JSON array of function calls in the format:
    [
        {"name": "func1", "arguments": {...}},
        {"name": "func2", "arguments": {...}}
    ]
    
    Args:
        text: Text containing potential parallel function calls
        
    Returns:
        ParallelFunctionCalls object if found, None otherwise
    """
    # Look for array of JSON objects
    array_pattern = r'\[\s*(\{.*?\}(?:\s*,\s*\{.*?\})*)\s*\]'
    array_matches = re.findall(array_pattern, text, re.DOTALL)
    
    for array_match in array_matches:
        try:
            # Add brackets back for proper JSON parsing
            json_array = f"[{array_match}]"
            data = json.loads(json_array)
            
            if isinstance(data, list) and all(
                isinstance(item, dict) and "name" in item and "arguments" in item
                for item in data
            ):
                calls = [
                    FunctionCall(
                        schema_id="json_function_call",
                        name=item["name"],
                        arguments=item["arguments"],
                        raw_text=json.dumps(item)
                    )
                    for item in data
                ]
                
                return ParallelFunctionCalls(
                    schema_id="parallel_function_calls",
                    calls=calls
                )
        except json.JSONDecodeError:
            continue
    
    return None

def format_json_schema_prompt(schema: Dict[str, Any]) -> str:
    """
    Format a JSON schema into a prompt instruction for an LLM.
    
    Args:
        schema: JSON schema dictionary
        
    Returns:
        Formatted instruction string for the LLM
    """
    return f"""
You must respond with a valid JSON object that conforms to this schema:
```json
{json.dumps(schema, indent=2)}
```

Your response must be a JSON object that follows this schema EXACTLY.
Do not include any explanations, only provide a valid JSON response.
"""

def together_json_mode_request(messages: List[Dict[str, str]], schema: Dict[str, Any], model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    """
    Make a request to Together API using JSON mode.
    
    Args:
        messages: List of message dictionaries (role, content)
        schema: JSON schema for structured output
        model: Model identifier to use
        
    Returns:
        Parsed JSON response
    """
    try:
        from together import Together
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "together"])
        from together import Together
    
    # Add JSON mode instruction to system message if not present
    has_json_instruction = False
    for msg in messages:
        if msg["role"] == "system" and "JSON" in msg["content"]:
            has_json_instruction = True
            break
    
    if not has_json_instruction:
        # If there's a system message, append to it
        system_found = False
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] += "\n\nYou must respond with valid JSON only."
                system_found = True
                break
        
        # If no system message, add one
        if not system_found:
            messages.insert(0, {
                "role": "system",
                "content": "You must respond with valid JSON only."
            })
    
    # Initialize Together client
    together = Together()
    
    # Make the request with JSON mode enabled
    response = together.chat.completions.create(
        messages=messages,
        model=model,
        response_format={
            "type": "json_object",
            "schema": schema
        }
    )
    
    # Parse and return the JSON response
    content = response.choices[0].message.content
    return json.loads(content)

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python code_structure_analyzer.py <python_file_to_analyze>")
        sys.exit(1)
        
    analyzer = CodeStructureAnalyzer()
    file_path = sys.argv[1]
    
    if analyzer.analyze_file(file_path):
        print(f"Successfully analyzed {file_path}")
        print("Top code structures:")
        results = analyzer.search("function definition", limit=5)
        for idx, node in enumerate(results):
            print(f"{idx+1}. {node.node_type}: {node.name} (lines {node.line_start}-{node.line_end})")
    else:
        print(f"Failed to analyze {file_path}")