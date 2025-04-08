#!/usr/bin/env python3
import re
import os
import sys
import json
import argparse
import subprocess
import inspect
import importlib
import importlib.util
import math
import sqlite3
import uuid
import hashlib
import base64
from io import StringIO, BytesIO
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
import traceback
from pathlib import Path
import tempfile
import time
from dataclasses import dataclass, field
import urllib.parse
import asyncio
import queue
import threading
import re

try:
    import aiohttp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp

try:
    from together import Together
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "together"])
    from together import Together

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.panel import Panel
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.panel import Panel

# Initialize console for rich output
console = Console()

@dataclass
class CodeArtifact:
    artifact_id: str
    name: str
    code: str
    description: str
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_count: int = 0
    last_result: Optional[Dict[str, Any]] = None
    
@dataclass
class FunctionSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    source_code: Optional[str] = None
    
@dataclass
class PlanningSession:
    """Class to track multi-turn reasoning and tool use sessions."""
    session_id: str
    task: str
    created_at: float
    steps: List[Dict[str, Any]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    completion_status: str = "in_progress"  # in_progress, completed, failed
    
@dataclass
class StructuredOutput:
    """Base class for structured data extraction results."""
    source: str
    timestamp: float = field(default_factory=time.time)
    
@dataclass
class URLExtraction(StructuredOutput):
    """Structured output for URL extraction."""
    urls: List[str] = field(default_factory=list)
    
@dataclass
class KnowledgeItem:
    """A piece of knowledge extracted from content."""
    content: str
    source_url: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

class AsyncTaskProcessor:
    """Processes tasks asynchronously in a background thread."""
    
    def __init__(self):
        self.task_queue = queue.Queue()
        self.results = {}
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.knowledge_base = []
        self.processed_urls = set()
        self.url_pattern = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')
        self.worker_thread.start()
    
    def _worker(self):
        """Background worker that processes tasks from the queue."""
        while not self.stop_event.is_set():
            try:
                task_id, task_func, args, kwargs = self.task_queue.get(timeout=1)
                try:
                    # Run either a sync or async function
                    if asyncio.iscoroutinefunction(task_func):
                        # Create a new event loop for this thread if needed
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(task_func(*args, **kwargs))
                    else:
                        result = task_func(*args, **kwargs)
                    
                    self.results[task_id] = {"status": "completed", "result": result}
                except Exception as e:
                    self.results[task_id] = {
                        "status": "failed", 
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                pass
    
    def add_task(self, task_func, *args, **kwargs):
        """Add a task to the queue."""
        task_id = str(uuid.uuid4())
        self.results[task_id] = {"status": "pending"}
        self.task_queue.put((task_id, task_func, args, kwargs))
        return task_id
    
    def get_result(self, task_id):
        """Get the result of a task."""
        return self.results.get(task_id, {"status": "not_found"})
    
    def extract_urls(self, text):
        """Extract URLs from text and return as structured output."""
        urls = [url for url in self.url_pattern.findall(text)]
        # Normalize URLs (ensure they start with http:// or https://)
        normalized_urls = []
        for url in urls:
            if url.startswith('www.'):
                url = 'https://' + url
            normalized_urls.append(url)
        
        return URLExtraction(source="text_extraction", urls=normalized_urls)
    
    def add_urls_to_process(self, urls):
        """Add URLs to be processed for knowledge extraction."""
        new_urls = [url for url in urls if url not in self.processed_urls]
        for url in new_urls:
            self.processed_urls.add(url)
            # Queue the URL for processing
            self.add_task(self._process_url, url)
        return len(new_urls)
    
    async def _process_url(self, url):
        """Process a URL to extract knowledge."""
        try:
            # Use the JinaClient to read the content
            jina_client = JinaClient(token=os.environ.get("JINA_API_KEY"))
            result = await jina_client.read(url)
            
            if isinstance(result, dict) and "results" in result:
                content = result["results"]
                
                # Extract knowledge items
                knowledge_item = KnowledgeItem(
                    content=content,
                    source_url=url
                )
                self.knowledge_base.append(knowledge_item)
                
                # Extract further URLs from the content
                urls_extraction = self.extract_urls(content)
                self.add_urls_to_process(urls_extraction.urls)
                
                return {
                    "success": True,
                    "url": url,
                    "knowledge_extracted": True,
                    "further_urls_found": len(urls_extraction.urls)
                }
        except Exception as e:
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    def get_knowledge_summary(self):
        """Get a summary of the knowledge base."""
        return {
            "total_items": len(self.knowledge_base),
            "total_urls_processed": len(self.processed_urls),
            "total_urls_pending": self.task_queue.qsize(),
            "recent_items": [
                {"source": item.source_url, "timestamp": item.timestamp}
                for item in sorted(self.knowledge_base, key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }
    
    def search_knowledge(self, query):
        """Search the knowledge base for relevant information."""
        # Simple keyword search for now
        results = []
        for item in self.knowledge_base:
            if query.lower() in item.content.lower():
                results.append({
                    "source": item.source_url,
                    "timestamp": item.timestamp,
                    "relevance": "high" if query.lower() in item.content.lower()[:500] else "medium"
                })
        return results
    
    def stop(self):
        """Stop the task processor."""
        self.stop_event.set()
        self.worker_thread.join(timeout=2)
    
class JinaClient:
    """Client for interacting with Jina.ai endpoints"""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize with your Jina token"""
        self.token = token or os.getenv("JINA_API_KEY")
        if not self.token:
            raise ValueError("JINA_API_KEY environment variable or token must be provided")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    async def search(self, query: str) -> dict:
        """
        Search using s.jina.ai endpoint
        Args:
            query: Search term
        Returns:
            API response as dict
        """
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/{encoded_query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response_text = await response.text()
                return {"results": response_text}
    
    async def fact_check(self, query: str) -> str:
        """
        Get grounding info using g.jina.ai endpoint
        Args:
            query: Query to ground
        Returns:
            API response as text
        """
        encoded_query = urllib.parse.quote(query)
        url = f"https://g.jina.ai/{encoded_query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                return await response.text()
        
    async def read(self, url: str) -> dict:
        """
        Get ranking using r.jina.ai endpoint
        Args:
            url: URL to rank
        Returns:
            API response as dict
        """
        encoded_url = urllib.parse.quote(url)
        rank_url = f"https://r.jina.ai/{encoded_url}"
        async with aiohttp.ClientSession() as session:
            async with session.get(rank_url, headers=self.headers) as response:
                response_text = await response.text()
                return {"results": response_text}


class CodeRepository:
    """Repository for storing and managing code artifacts."""
    
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize the database schema."""
        cursor = self.conn.cursor()
        
        # Create artifacts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            code TEXT NOT NULL,
            description TEXT,
            created_at REAL NOT NULL,
            execution_count INTEGER DEFAULT 0,
            metadata TEXT,
            last_result TEXT
        )
        ''')
        
        # Create execution logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS execution_logs (
            log_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL,
            executed_at REAL NOT NULL,
            success INTEGER NOT NULL,
            stdout TEXT,
            stderr TEXT,
            result TEXT,
            execution_time REAL,
            FOREIGN KEY (artifact_id) REFERENCES artifacts (artifact_id)
        )
        ''')
        
        # Create modules table for in-memory modules
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS modules (
            module_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            code TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_updated_at REAL NOT NULL,
            description TEXT
        )
        ''')
        
        self.conn.commit()
        
    def add_artifact(self, name: str, code: str, description: str = "", metadata: Dict[str, Any] = None) -> str:
        """Add a new code artifact to the repository."""
        artifact_id = str(uuid.uuid4())
        created_at = time.time()
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO artifacts (artifact_id, name, code, description, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (artifact_id, name, code, description, created_at, json.dumps(metadata or {}))
        )
        self.conn.commit()
        
        return artifact_id
        
    def update_artifact(self, artifact_id: str, code: str = None, description: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Update an existing code artifact."""
        cursor = self.conn.cursor()
        
        # Get current values
        cursor.execute("SELECT code, description, metadata FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        row = cursor.fetchone()
        if not row:
            return False
            
        current_code, current_description, current_metadata_str = row
        current_metadata = json.loads(current_metadata_str) if current_metadata_str else {}
        
        # Update values
        updated_code = code if code is not None else current_code
        updated_description = description if description is not None else current_description
        
        if metadata:
            current_metadata.update(metadata)
        updated_metadata = json.dumps(current_metadata)
        
        cursor.execute(
            "UPDATE artifacts SET code = ?, description = ?, metadata = ? WHERE artifact_id = ?",
            (updated_code, updated_description, updated_metadata, artifact_id)
        )
        self.conn.commit()
        
        return cursor.rowcount > 0
        
    def get_artifact(self, artifact_id: str) -> Optional[CodeArtifact]:
        """Retrieve a code artifact by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
            
        column_names = [description[0] for description in cursor.description]
        artifact_data = dict(zip(column_names, row))
        
        return CodeArtifact(
            artifact_id=artifact_data["artifact_id"],
            name=artifact_data["name"],
            code=artifact_data["code"],
            description=artifact_data["description"],
            created_at=artifact_data["created_at"],
            execution_count=artifact_data["execution_count"],
            metadata=json.loads(artifact_data["metadata"]) if artifact_data["metadata"] else {},
            last_result=json.loads(artifact_data["last_result"]) if artifact_data["last_result"] else None
        )
        
    def find_artifacts(self, query: str = None, limit: int = 10) -> List[CodeArtifact]:
        """Find artifacts matching a query."""
        cursor = self.conn.cursor()
        
        if query:
            sql = """
            SELECT * FROM artifacts 
            WHERE name LIKE ? OR description LIKE ? OR code LIKE ?
            ORDER BY created_at DESC LIMIT ?
            """
            params = (f"%{query}%", f"%{query}%", f"%{query}%", limit)
        else:
            sql = "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?"
            params = (limit,)
            
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        artifacts = []
        column_names = [description[0] for description in cursor.description]
        
        for row in rows:
            artifact_data = dict(zip(column_names, row))
            artifacts.append(CodeArtifact(
                artifact_id=artifact_data["artifact_id"],
                name=artifact_data["name"],
                code=artifact_data["code"],
                description=artifact_data["description"],
                created_at=artifact_data["created_at"],
                execution_count=artifact_data["execution_count"],
                metadata=json.loads(artifact_data["metadata"]) if artifact_data["metadata"] else {},
                last_result=json.loads(artifact_data["last_result"]) if artifact_data["last_result"] else None
            ))
            
        return artifacts
        
    def log_execution(self, artifact_id: str, success: bool, stdout: str = "", stderr: str = "", 
                     result: str = None, execution_time: float = None) -> str:
        """Log an execution of a code artifact."""
        log_id = str(uuid.uuid4())
        executed_at = time.time()
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO execution_logs (log_id, artifact_id, executed_at, success, stdout, stderr, result, execution_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (log_id, artifact_id, executed_at, 1 if success else 0, stdout, stderr, result, execution_time)
        )
        
        # Update artifact execution count and last result
        cursor.execute(
            "UPDATE artifacts SET execution_count = execution_count + 1, last_result = ? WHERE artifact_id = ?",
            (json.dumps({
                "success": success,
                "result": result,
                "executed_at": executed_at
            }), artifact_id)
        )
        
        self.conn.commit()
        return log_id
        
    def get_execution_logs(self, artifact_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get execution logs for a specific artifact."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM execution_logs WHERE artifact_id = ? ORDER BY executed_at DESC LIMIT ?",
            (artifact_id, limit)
        )
        rows = cursor.fetchall()
        
        logs = []
        column_names = [description[0] for description in cursor.description]
        
        for row in rows:
            log_data = dict(zip(column_names, row))
            logs.append(log_data)
            
        return logs
        
    def add_module(self, name: str, code: str, description: str = "") -> str:
        """Add a new in-memory module."""
        module_id = str(uuid.uuid4())
        timestamp = time.time()
        
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO modules (module_id, name, code, created_at, last_updated_at, description) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (module_id, name, code, timestamp, timestamp, description)
        )
        self.conn.commit()
        
        return module_id
        
    def update_module(self, name: str, code: str, description: str = None) -> bool:
        """Update an existing module."""
        timestamp = time.time()
        cursor = self.conn.cursor()
        
        if description is not None:
            cursor.execute(
                "UPDATE modules SET code = ?, last_updated_at = ?, description = ? WHERE name = ?",
                (code, timestamp, description, name)
            )
        else:
            cursor.execute(
                "UPDATE modules SET code = ?, last_updated_at = ? WHERE name = ?",
                (code, timestamp, name)
            )
            
        self.conn.commit()
        return cursor.rowcount > 0
        
    def get_module(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a module by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM modules WHERE name = ?", (name,))
        row = cursor.fetchone()
        
        if not row:
            return None
            
        column_names = [description[0] for description in cursor.description]
        return dict(zip(column_names, row))
        
    def list_modules(self) -> List[Dict[str, Any]]:
        """List all available modules."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM modules ORDER BY name")
        rows = cursor.fetchall()
        
        modules = []
        column_names = [description[0] for description in cursor.description]
        
        for row in rows:
            modules.append(dict(zip(column_names, row)))
            
        return modules
        
    def delete_module(self, name: str) -> bool:
        """Delete a module by name."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM modules WHERE name = ?", (name,))
        self.conn.commit()
        
        return cursor.rowcount > 0
        
    def execute_module(self, name: str, globals_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a stored module."""
        module = self.get_module(name)
        if not module:
            return {"success": False, "error": f"Module '{name}' not found"}
            
        try:
            # Create execution environment
            if globals_dict is None:
                globals_dict = globals().copy()
                
            locals_dict = {}
            
            # Execute the module code
            start_time = time.time()
            exec(module["code"], globals_dict, locals_dict)
            execution_time = time.time() - start_time
            
            return {
                "success": True,
                "locals": locals_dict,
                "execution_time": execution_time
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()

class ToolRegistry:
    def __init__(self):
        self.functions: Dict[str, FunctionSpec] = {}
        self.code_repo = CodeRepository(db_path="code_artifacts.db")
        self.jina_client = None
        try:
            self.jina_client = JinaClient()
        except ValueError:
            console.print("[yellow]Warning: JINA_API_KEY not found in environment variables. Jina tools will not be available.[/yellow]")
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register the default set of tools."""
        # Python code execution
        self.register_function(
            name="execute_python",
            description="Execute Python code and return the result",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "save_artifact": {
                        "type": "boolean",
                        "description": "Whether to save this code as an artifact",
                        "default": False
                    },
                    "artifact_name": {
                        "type": "string",
                        "description": "Name for the artifact if saving (defaults to auto-generated)",
                        "default": ""
                    },
                    "description": {
                        "type": "string",
                        "description": "Description for the artifact if saving",
                        "default": ""
                    }
                },
                "required": ["code"]
            },
            function=self._execute_python
        )
        
        # Save code to module
        self.register_function(
            name="save_module",
            description="Save Python code as a reusable module",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the module (must be a valid Python identifier)"
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code for the module"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the module does",
                        "default": ""
                    }
                },
                "required": ["name", "code"]
            },
            function=self._save_module
        )
        
        # Execute a saved module
        self.register_function(
            name="execute_module",
            description="Execute a previously saved module",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the module to execute"
                    },
                    "args": {
                        "type": "object",
                        "description": "Arguments to pass to the module (if any)",
                        "default": {}
                    }
                },
                "required": ["name"]
            },
            function=self._execute_saved_module
        )
        
        # List available modules
        self.register_function(
            name="list_modules",
            description="List all available Python modules",
            parameters={
                "type": "object",
                "properties": {}
            },
            function=self._list_modules
        )
        
        # Get a specific module
        self.register_function(
            name="get_module",
            description="Get a specific Python module by name",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the module to retrieve"
                    }
                },
                "required": ["name"]
            },
            function=self._get_module
        )
        
        # Run a Python script file
        self.register_function(
            name="run_script",
            description="Run a Python script file and return the result",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the Python script file to run"
                    },
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Command line arguments to pass to the script",
                        "default": []
                    }
                },
                "required": ["file_path"]
            },
            function=self._run_script
        )
        
        # Code search
        self.register_function(
            name="search_code",
            description="Search for code artifacts by name or content",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            },
            function=self._search_code
        )
        
        # Weather functions
        self.register_function(
            name="get_weather",
            description="Get current weather information for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for (city, state, country, etc.)"
                    }
                },
                "required": ["location"]
            },
            function=self._get_weather
        )
        
        self.register_function(
            name="parse_weather_response",
            description="Parse weather data into a readable format",
            parameters={
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Weather API response to parse"
                    }
                },
                "required": ["response"]
            },
            function=self._parse_weather_response
        )
        
        # System prompt management
        self.register_function(
            name="update_system_prompt",
            description="Update the system prompt for the agent to adapt its behavior",
            parameters={
                "type": "object",
                "properties": {
                    "system_prompt": {
                        "type": "string",
                        "description": "New system prompt to guide the agent's behavior"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Whether to append to the existing system prompt or replace it entirely",
                        "default": False
                    }
                },
                "required": ["system_prompt"]
            },
            function=self._update_system_prompt
        )
        
        self.register_function(
            name="get_system_prompt",
            description="Retrieve the current system prompt",
            parameters={
                "type": "object",
                "properties": {}
            },
            function=self._get_system_prompt
        )
        
        self.register_function(
            name="add_reflection_note",
            description="Add a self-reflection note for the agent to record insights, strategies, or observations",
            parameters={
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "The reflection note content"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category for organizing notes (e.g., 'strategy', 'observation', 'improvement')",
                        "default": "general"
                    }
                },
                "required": ["note"]
            },
            function=self._add_reflection_note
        )
        
        self.register_function(
            name="get_reflection_notes",
            description="Retrieve previously recorded reflection notes",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter notes by category",
                        "default": "all"
                    }
                }
            },
            function=self._get_reflection_notes
        )
        
        # Environmental adaptation
        self.register_function(
            name="analyze_environment",
            description="Analyze the current environment and context to adapt agent behavior",
            parameters={
                "type": "object",
                "properties": {
                    "aspect": {
                        "type": "string",
                        "description": "Specific environmental aspect to analyze (system, user_behavior, conversation_context, task_complexity)",
                        "enum": ["system", "user_behavior", "conversation_context", "task_complexity", "all"],
                        "default": "all"
                    }
                }
            },
            function=self._analyze_environment
        )
        
        self.register_function(
            name="adapt_to_environment",
            description="Adapt agent behavior based on environmental analysis",
            parameters={
                "type": "object",
                "properties": {
                    "adaptation_strategy": {
                        "type": "string",
                        "description": "Strategy for adaptation (e.g., 'simplify_responses', 'increase_detail', 'focus_technical', 'focus_practical')"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for this adaptation"
                    },
                    "system_prompt_update": {
                        "type": "string",
                        "description": "Optional system prompt update to implement the adaptation"
                    }
                },
                "required": ["adaptation_strategy", "reason"]
            },
            function=self._adapt_to_environment
        )
        
        # File operations
        self.register_function(
            name="read_file",
            description="Read the contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    }
                },
                "required": ["path"]
            },
            function=self._read_file
        )
        
        self.register_function(
            name="write_file",
            description="Write content to a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Whether to append to the file or overwrite it",
                        "default": False
                    }
                },
                "required": ["path", "content"]
            },
            function=self._write_file
        )
        
        self.register_function(
            name="list_directory",
            description="List files and directories in a directory",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list"
                    }
                },
                "required": ["path"]
            },
            function=self._list_directory
        )
        
        # Command execution
        self.register_function(
            name="execute_command",
            description="Execute a shell command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute"
                    }
                },
                "required": ["command"]
            },
            function=self._execute_command
        )
        
        # Function creation
        self.register_function(
            name="create_python_function",
            description="Create a new Python function that can be called later",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the function"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the function does"
                    },
                    "parameters_schema": {
                        "type": "object",
                        "description": "JSON Schema for the function parameters"
                    },
                    "source_code": {
                        "type": "string",
                        "description": "Python source code for the function"
                    }
                },
                "required": ["name", "description", "parameters_schema", "source_code"]
            },
            function=self._create_python_function
        )
        
        # Function listing
        self.register_function(
            name="list_available_functions",
            description="List all available functions that can be called",
            parameters={
                "type": "object",
                "properties": {},
            },
            function=self._list_available_functions
        )
        
        # Together API models
        self.register_function(
            name="list_together_models",
            description="List all available models on the Together API",
            parameters={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "string",
                        "description": "Optional filter term for model names",
                        "default": ""
                    }
                }
            },
            function=self._list_together_models
        )
        
        # Generate text completion
        self.register_function(
            name="generate_completion",
            description="Generate a text completion using Together API",
            parameters={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model to use for completion"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt to generate completion for"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum number of tokens to generate",
                        "default": 256
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (0-2)",
                        "default": 0.7
                    },
                    "logprobs": {
                        "type": "integer",
                        "description": "Number of top logprobs to return (usually 1 or 5)",
                        "default": 0
                    },
                    "echo": {
                        "type": "boolean",
                        "description": "Echo prompt tokens with logprobs",
                        "default": False
                    }
                },
                "required": ["model", "prompt"]
            },
            function=self._generate_completion
        )
        
        # Create/update assistant
        self.register_function(
            name="create_or_update_assistant",
            description="Create or update an assistant in the Together platform",
            parameters={
                "type": "object",
                "properties": {
                    "assistant_id": {
                        "type": "string",
                        "description": "Assistant ID (if updating existing assistant)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name of the assistant"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of the assistant's purpose"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use for this assistant"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt for the assistant"
                    }
                },
                "required": ["name", "model"]
            },
            function=self._create_or_update_assistant
        )
        
        # Create thread
        self.register_function(
            name="create_thread",
            description="Create a new thread for conversations",
            parameters={
                "type": "object",
                "properties": {
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata for the thread"
                    }
                }
            },
            function=self._create_thread
        )
        
        # Add message to thread
        self.register_function(
            name="add_message_to_thread",
            description="Add a message to an existing thread",
            parameters={
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "ID of the thread to add message to"
                    },
                    "role": {
                        "type": "string",
                        "description": "Role of the message sender (user or assistant)",
                        "enum": ["user", "assistant"]
                    },
                    "content": {
                        "type": "string",
                        "description": "Content of the message"
                    }
                },
                "required": ["thread_id", "role", "content"]
            },
            function=self._add_message_to_thread
        )
        
        # Run assistant on thread
        self.register_function(
            name="run_assistant",
            description="Run an assistant on a thread to generate a response",
            parameters={
                "type": "object",
                "properties": {
                    "assistant_id": {
                        "type": "string",
                        "description": "ID of the assistant to run"
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "ID of the thread to run the assistant on"
                    }
                },
                "required": ["assistant_id", "thread_id"]
            },
            function=self._run_assistant
        )
        
        # Jina web search tool
        if self.jina_client:
            self.register_function(
                name="web_search",
                description="Search the web for information using Jina's search API",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                },
                function=self._web_search
            )
            
            # Jina web reader tool
            self.register_function(
                name="web_read",
                description="Read and extract content from a URL using Jina's reader API",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to read content from"
                        }
                    },
                    "required": ["url"]
                },
                function=self._web_read
            )
            
            # Jina fact check tool
            self.register_function(
                name="fact_check",
                description="Verify facts and ground statements using Jina's fact checking API",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The statement to fact check"
                        }
                    },
                    "required": ["query"]
                },
                function=self._fact_check
            )
        
        # Planning tools for multi-turn, complex operations
        self.register_function(
            name="create_planning_session",
            description="Start a new planning session for complex multi-turn operations",
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task to plan and execute"
                    }
                },
                "required": ["task"]
            },
            function=self._create_planning_session
        )
        
        self.register_function(
            name="add_plan_step",
            description="Add a step to the current planning session, with optional tool execution",
            parameters={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "Description of this planning step"
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Optional name of tool to execute in this step"
                    },
                    "tool_args": {
                        "type": "object",
                        "description": "Optional arguments for the tool"
                    }
                },
                "required": ["plan"]
            },
            function=self._add_plan_step
        )
        
        self.register_function(
            name="get_planning_status",
            description="Get the current status of the planning session",
            parameters={
                "type": "object",
                "properties": {}
            },
            function=self._get_planning_status
        )
        
        self.register_function(
            name="complete_planning_session",
            description="Complete the current planning session with a summary",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of the planning session results"
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the planning session was successful",
                        "default": True
                    }
                },
                "required": ["summary"]
            },
            function=self._complete_planning_session
        )
        
        # Structured output and knowledge building tools
        self.register_function(
            name="extract_urls",
            description="Extract URLs from text and return as structured output",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to extract URLs from"
                    }
                },
                "required": ["text"]
            },
            function=self._extract_urls
        )
        
        self.register_function(
            name="process_urls",
            description="Process URLs to extract knowledge",
            parameters={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "URLs to process"
                    }
                },
                "required": ["urls"]
            },
            function=self._process_urls
        )
        
        self.register_function(
            name="get_knowledge_summary",
            description="Get a summary of the knowledge base",
            parameters={
                "type": "object",
                "properties": {}
            },
            function=self._get_knowledge_summary
        )
        
        self.register_function(
            name="search_knowledge",
            description="Search the knowledge base for relevant information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            },
            function=self._search_knowledge
        )
        
        self.register_function(
            name="monitor_task",
            description="Monitor the status of a background task",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "ID of the task to monitor"
                    }
                },
                "required": ["task_id"]
            },
            function=self._monitor_task
        )
    
    def register_function(self, name: str, description: str, parameters: Dict[str, Any], function: Callable, source_code: Optional[str] = None):
        """Register a function to be available for the agent."""
        if name in self.functions:
            console.print(f"[yellow]Warning: Overwriting existing function '{name}'[/yellow]")
        
        self.functions[name] = FunctionSpec(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
            source_code=source_code
        )
    
    def get_openai_tools_format(self) -> List[Dict[str, Any]]:
        """Get the tools in the format expected by OpenAI's API."""
        tools = []
        for name, spec in self.functions.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters
                }
            })
        return tools
    
    def call_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a registered function with the provided arguments."""
        if name not in self.functions:
            return {"error": f"Function '{name}' not found in registry"}
        
        try:
            result = self.functions[name].function(**arguments)
            return result
        except Exception as e:
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    # Default tool implementations
    def _read_file(self, path: str) -> Dict[str, Any]:
        """Read the contents of a file."""
        try:
            path = Path(path).expanduser()
            if not path.exists():
                return {"error": f"File '{path}' does not exist"}
            
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            return {
                "content": content,
                "size_bytes": path.stat().st_size,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _write_file(self, path: str, content: str, append: bool = False) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            path = Path(path).expanduser()
            
            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as file:
                file.write(content)
            
            return {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _list_directory(self, path: str) -> Dict[str, Any]:
        """List files and directories in a directory."""
        try:
            path = Path(path).expanduser()
            if not path.exists():
                return {"error": f"Path '{path}' does not exist"}
            
            if not path.is_dir():
                return {"error": f"Path '{path}' is not a directory"}
            
            items = []
            for item in path.iterdir():
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size_bytes": item.stat().st_size if item.is_file() else None,
                    "last_modified": time.ctime(item.stat().st_mtime)
                })
            
            return {
                "path": str(path),
                "items": items,
                "count": len(items),
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a shell command."""
        try:
            process = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True
            )
            
            return {
                "command": command,
                "stdout": process.stdout,
                "stderr": process.stderr,
                "return_code": process.returncode,
                "success": process.returncode == 0
            }
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _create_python_function(self, name: str, description: str, parameters_schema: Dict[str, Any], source_code: str) -> Dict[str, Any]:
        """Create a new Python function that can be called later."""
        try:
            # Create a temporary file to hold the function code
            temp_module_path = Path(tempfile.gettempdir()) / f"dynamic_func_{name}_{int(time.time())}.py"
            
            # Write the module with the function
            with open(temp_module_path, 'w', encoding='utf-8') as f:
                f.write(source_code)
            
            # Import the module
            spec = importlib.util.spec_from_file_location(f"dynamic_func_{name}", temp_module_path)
            if spec is None or spec.loader is None:
                return {"error": "Failed to create module specification", "success": False}
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the function in the module
            if name not in dir(module):
                return {"error": f"Function '{name}' not found in the provided source code", "success": False}
            
            function = getattr(module, name)
            if not callable(function):
                return {"error": f"'{name}' is not a callable function", "success": False}
            
            # Register the function
            self.register_function(
                name=name,
                description=description,
                parameters=parameters_schema,
                function=function,
                source_code=source_code
            )
            
            return {
                "name": name,
                "description": description,
                "success": True,
                "message": f"Function '{name}' successfully created and registered"
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _list_available_functions(self) -> Dict[str, Any]:
        """List all available functions that can be called."""
        function_list = []
        for name, spec in self.functions.items():
            function_list.append({
                "name": name,
                "description": spec.description,
                "parameters": spec.parameters,
                "has_source_code": spec.source_code is not None
            })
        
        return {
            "functions": function_list,
            "count": len(function_list),
            "success": True
        }
    
    def _list_together_models(self, filter: str = "") -> Dict[str, Any]:
        """List all available models on the Together API."""
        try:
            # Get credentials for Together API from parent agent
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            # Use the agent's client to list models
            models_data = agent.client.models.list()
            
            # Filter models if a filter term is provided
            models = []
            for model in models_data.data:
                model_dict = {
                    "id": model.id,
                    "name": model.name,
                    "context_length": model.context_length,
                    "capabilities": model.capabilities
                }
                
                if not filter or filter.lower() in model.name.lower():
                    models.append(model_dict)
            
            return {
                "models": models,
                "count": len(models),
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _generate_completion(self, model: str, prompt: str, max_tokens: int = 256, 
                         temperature: float = 0.7, logprobs: int = 0, echo: bool = False) -> Dict[str, Any]:
        """Generate a text completion using Together API with optional logprobs."""
        try:
            # Get credentials for Together API from parent agent
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            # Prepare optional parameters
            params = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Add logprobs parameters if requested and enabled
            agent_logprobs_enabled = False
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    agent_logprobs_enabled = getattr(agent, 'enable_logprobs', False)
                    break
                    
            if (logprobs > 0 or agent_logprobs_enabled) and logprobs >= 0:
                # If no specific logprobs count was requested but agent has it enabled, use 1
                actual_logprobs = logprobs if logprobs > 0 else 1
                params["logprobs"] = actual_logprobs
                
                if echo:
                    params["echo"] = echo
            
            # Use the agent's client to generate completions
            response = agent.client.completions.create(**params)
            
            result = {
                "model": model,
                "completion": response.choices[0].text,
                "finish_reason": response.choices[0].finish_reason,
                "tokens_used": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "success": True
            }
            
            # Include logprobs in the result if they were requested
            if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                logprobs_data = response.choices[0].logprobs
                
                result["logprobs"] = {
                    "tokens": logprobs_data.tokens,
                    "token_logprobs": logprobs_data.token_logprobs
                }
                
                # Include top logprobs if available (depends on the model and API version)
                if hasattr(logprobs_data, "top_logprobs") and logprobs_data.top_logprobs:
                    result["logprobs"]["top_logprobs"] = logprobs_data.top_logprobs
            
            # Include prompt tokens and their logprobs if echo was enabled
            if echo and hasattr(response, "prompt"):
                result["prompt_tokens"] = []
                for prompt_item in response.prompt:
                    if hasattr(prompt_item, "logprobs") and prompt_item.logprobs:
                        prompt_logprobs = {
                            "text": prompt_item.text,
                            "tokens": prompt_item.logprobs.tokens,
                            "token_logprobs": prompt_item.logprobs.token_logprobs
                        }
                        result["prompt_tokens"].append(prompt_logprobs)
            
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _create_or_update_assistant(self, name: str, model: str, assistant_id: str = None, 
                                    description: str = None, system_prompt: str = None) -> Dict[str, Any]:
        """Create or update an assistant in the Together platform."""
        try:
            # Get credentials for Together API from parent agent
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            # Together API doesn't currently support assistants API directly
            # Create a simple assistant record to track configuration
            
            # Generate a new assistant ID if needed
            if not assistant_id:
                assistant_id = f"asst_{int(time.time())}"
                action = "Created"
            else:
                action = "Updated"
            
            # Store assistant data in agent's configuration
            if not hasattr(agent, 'assistants'):
                agent.assistants = {}
                
            agent.assistants[assistant_id] = {
                "id": assistant_id,
                "name": name,
                "model": model,
                "description": description or "",
                "system_prompt": system_prompt or "",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return {
                "assistant_id": assistant_id,
                "name": name,
                "model": model,
                "description": description or "",
                "action": action,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _create_thread(self, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new thread for conversations."""
        try:
            # Get credentials for Together API from parent agent
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            # Together API doesn't currently support threads API directly
            # Just return a simulated thread ID for compatibility
            thread_id = f"thread_{int(time.time())}"
            
            return {
                "thread_id": thread_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metadata": metadata or {},
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _add_message_to_thread(self, thread_id: str, role: str, content: str) -> Dict[str, Any]:
        """Add a message to an existing thread."""
        try:
            # Get credentials for Together API from parent agent
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            # Together API doesn't currently support messages API directly
            # Simulate adding message to a thread by updating the agent's conversation
            if thread_id.startswith("thread_"):
                # Generate a message ID for consistency
                message_id = f"msg_{int(time.time())}_{hash(content) % 10000}"
                
                # Add the message to the conversation history directly
                agent.add_message(role, content)
                
                return {
                    "message_id": message_id,
                    "thread_id": thread_id,
                    "role": role,
                    "content": content,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "success": True
                }
            else:
                return {"error": "Invalid thread ID format", "success": False}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _get_system_prompt(self) -> Dict[str, Any]:
        """Retrieve the current system prompt."""
        try:
            # Get reference to the TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            return {
                "system_prompt": agent.system_message["content"],
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _add_reflection_note(self, note: str, category: str = "general") -> Dict[str, Any]:
        """Add a self-reflection note for the agent to record insights, strategies, or observations."""
        try:
            # Get reference to the TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            # Create the reflection note with timestamp
            reflection_note = {
                "timestamp": time.time(),
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "note": note
            }
            
            # Add to the agent's reflection notes
            agent.reflection_notes.append(reflection_note)
            
            return {
                "note_id": len(agent.reflection_notes) - 1,
                "timestamp": reflection_note["datetime"],
                "category": category,
                "note": note,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _get_reflection_notes(self, category: str = "all") -> Dict[str, Any]:
        """Retrieve previously recorded reflection notes."""
        try:
            # Get reference to the TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            # Filter notes by category if specified
            if category.lower() == "all":
                filtered_notes = agent.reflection_notes
            else:
                filtered_notes = [note for note in agent.reflection_notes if note["category"].lower() == category.lower()]
            
            # Format notes for display
            formatted_notes = []
            for i, note in enumerate(filtered_notes):
                formatted_notes.append({
                    "id": i,
                    "timestamp": note["datetime"],
                    "category": note["category"],
                    "note": note["note"]
                })
            
            return {
                "notes": formatted_notes,
                "count": len(formatted_notes),
                "category_filter": category,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _analyze_environment(self, aspect: str = "all") -> Dict[str, Any]:
        """Analyze the current environment and context to adapt agent behavior."""
        try:
            # Get reference to the TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            # Update the timestamp of last analysis
            agent.environment_state["last_analysis_time"] = time.time()
            
            # Analyze conversation context if requested
            if aspect in ["conversation_context", "all"] and len(agent.conversation_history) > 2:
                # Extract user messages for analysis
                user_messages = [msg["content"] for msg in agent.conversation_history if msg.get("role") == "user"]
                
                # Perform basic sentiment analysis
                positive_words = ["good", "great", "excellent", "amazing", "helpful", "thanks", "thank", "appreciate"]
                negative_words = ["bad", "wrong", "incorrect", "error", "issue", "problem", "not", "doesn't", "don't"]
                
                positive_count = sum(sum(1 for word in positive_words if word.lower() in msg.lower()) for msg in user_messages)
                negative_count = sum(sum(1 for word in negative_words if word.lower() in msg.lower()) for msg in user_messages)
                
                if positive_count > negative_count * 2:
                    agent.environment_state["conversation_context"]["sentiment"] = "positive"
                elif negative_count > positive_count:
                    agent.environment_state["conversation_context"]["sentiment"] = "negative"
                else:
                    agent.environment_state["conversation_context"]["sentiment"] = "neutral"
                
                # Detect if conversation is task-oriented
                task_words = ["do", "make", "create", "implement", "build", "fix", "solve", "how", "help"]
                task_count = sum(sum(1 for word in task_words if word.lower() in msg.lower()) for msg in user_messages)
                
                agent.environment_state["conversation_context"]["task_oriented"] = task_count > len(user_messages) / 2
            
            # Analyze task complexity if requested
            if aspect in ["task_complexity", "all"] and len(agent.conversation_history) > 2:
                # Extract the last few user messages
                recent_user_messages = [msg["content"] for msg in agent.conversation_history[-min(5, len(agent.conversation_history)):] 
                                      if msg.get("role") == "user"]
                
                # Complexity indicators
                complexity_indicators = {
                    "high": ["complex", "advanced", "detailed", "comprehensive", "integrate", "optimize", "scale"],
                    "low": ["simple", "basic", "easy", "quick", "just", "help me", "show me"]
                }
                
                high_complexity = sum(sum(1 for word in complexity_indicators["high"] if word.lower() in msg.lower()) 
                                     for msg in recent_user_messages)
                low_complexity = sum(sum(1 for word in complexity_indicators["low"] if word.lower() in msg.lower()) 
                                    for msg in recent_user_messages)
                
                # Update complexity level
                if high_complexity > low_complexity * 2:
                    agent.environment_state["task_complexity"]["current_level"] = "high"
                elif low_complexity > high_complexity:
                    agent.environment_state["task_complexity"]["current_level"] = "low"
                else:
                    agent.environment_state["task_complexity"]["current_level"] = "medium"
            
            # Return the analysis results for the requested aspect
            if aspect == "all":
                return {
                    "environment_state": agent.environment_state,
                    "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "message_count": len([msg for msg in agent.conversation_history if msg.get("role") == "user"]),
                    "success": True
                }
            else:
                return {
                    aspect: agent.environment_state[aspect],
                    "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "success": True
                }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _adapt_to_environment(self, adaptation_strategy: str, reason: str, system_prompt_update: str = None) -> Dict[str, Any]:
        """Adapt agent behavior based on environmental analysis."""
        try:
            # Get reference to the TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            # Record the adaptation
            adaptation = {
                "timestamp": time.time(),
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "strategy": adaptation_strategy,
                "reason": reason
            }
            
            # Add to the adaptations history
            agent.environment_state["task_complexity"]["adaptations_made"].append(adaptation)
            
            # Record as a reflection note
            self._add_reflection_note(
                note=f"Adapted behavior using strategy: {adaptation_strategy}. Reason: {reason}",
                category="adaptation"
            )
            
            # Update system prompt if provided
            if system_prompt_update:
                self._update_system_prompt(
                    system_prompt=system_prompt_update,
                    append=True
                )
                adaptation["system_prompt_updated"] = True
            else:
                adaptation["system_prompt_updated"] = False
            
            return {
                "adaptation": adaptation,
                "current_environment": agent.environment_state,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _update_system_prompt(self, system_prompt: str, append: bool = False) -> Dict[str, Any]:
        """Update the system prompt for the agent to adapt its behavior."""
        try:
            # Get reference to the TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            # Save the old system prompt for reporting
            old_system_prompt = agent.system_message["content"]
            
            # Update the system prompt
            if append:
                new_system_prompt = old_system_prompt + "\n\n" + system_prompt
            else:
                new_system_prompt = system_prompt
            
            # Find the system message in the conversation history and update it
            for i, message in enumerate(agent.conversation_history):
                if message.get("role") == "system":
                    # Update the system message in the conversation history
                    agent.conversation_history[i]["content"] = new_system_prompt
                    # Also update the cached system_message
                    agent.system_message["content"] = new_system_prompt
                    break
            else:
                # If no system message found, add it to the beginning
                agent.conversation_history.insert(0, {"role": "system", "content": new_system_prompt})
                agent.system_message["content"] = new_system_prompt
            
            return {
                "old_system_prompt": old_system_prompt,
                "new_system_prompt": new_system_prompt,
                "append": append,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _get_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather information for a location."""
        try:
            # This is a mock implementation since we don't have actual weather API access
            import random
            
            # Create simulated weather data
            weather_conditions = ["sunny", "partly cloudy", "cloudy", "rainy", "stormy", "snowy", "windy", "foggy"]
            condition = random.choice(weather_conditions)
            
            # Generate realistic temperatures based on condition
            if condition == "sunny":
                temp_f = random.randint(70, 95)
            elif condition in ["partly cloudy", "cloudy"]:
                temp_f = random.randint(60, 80)
            elif condition in ["rainy", "stormy"]:
                temp_f = random.randint(50, 70)
            elif condition == "snowy":
                temp_f = random.randint(20, 35)
            else:
                temp_f = random.randint(40, 75)
                
            temp_c = round((temp_f - 32) * 5/9, 1)
            
            # Generate humidity and wind
            humidity = random.randint(30, 90)
            wind_speed = random.randint(0, 20)
            
            # Create a forecast for the next few days
            forecast = []
            current_temp = temp_f
            for i in range(5):
                # Vary temperature slightly for forecast
                forecast_temp = current_temp + random.randint(-10, 10)
                forecast_condition = random.choice(weather_conditions)
                forecast.append({
                    "day": i + 1,
                    "condition": forecast_condition,
                    "high_f": forecast_temp,
                    "low_f": forecast_temp - random.randint(10, 20),
                    "precipitation_chance": random.randint(0, 100)
                })
            
            return {
                "location": location,
                "current_condition": condition,
                "temperature_f": temp_f,
                "temperature_c": temp_c,
                "humidity": humidity,
                "wind_speed_mph": wind_speed,
                "forecast": forecast,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
            
    def _execute_python(self, code: str, save_artifact: bool = False, 
                        artifact_name: str = "", description: str = "") -> Dict[str, Any]:
        """Execute Python code and return the result."""
        try:
            # Create a string IO to capture stdout
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            # Prepare the locals dict to capture return values
            local_vars = {}
            
            # Measure execution time
            start_time = time.time()
            
            # Execute the code with redirected stdout/stderr
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, globals(), local_vars)
            
            execution_time = time.time() - start_time
            
            # Get the stdout and stderr
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            # Look for the last expression's value if it exists
            result = None
            if '_' in local_vars:
                result = local_vars['_']
            
            # Convert result to string representation if it's not None
            result_str = None
            if result is not None:
                try:
                    if isinstance(result, (dict, list, tuple, set)):
                        result_str = json.dumps(result)
                    else:
                        result_str = str(result)
                except:
                    result_str = str(result)
            
            # Prepare the response
            response = {
                "stdout": stdout,
                "stderr": stderr,
                "result": result_str,
                "execution_time": execution_time,
                "success": True
            }
            
            # Save as artifact if requested
            if save_artifact:
                if not artifact_name:
                    # Generate a name based on the first line of code or a timestamp
                    first_line = code.strip().split('\n')[0]
                    if len(first_line) > 30:
                        first_line = first_line[:27] + "..."
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    artifact_name = f"code_{timestamp}_{first_line.replace(' ', '_')}"
                
                # Add metadata about execution
                metadata = {
                    "execution_time": execution_time,
                    "has_output": bool(stdout),
                    "has_error": bool(stderr),
                    "has_result": result is not None
                }
                
                # Save the artifact
                artifact_id = self.code_repo.add_artifact(
                    name=artifact_name,
                    code=code,
                    description=description,
                    metadata=metadata
                )
                
                # Log the execution in the repo
                self.code_repo.log_execution(
                    artifact_id=artifact_id,
                    success=True,
                    stdout=stdout,
                    stderr=stderr,
                    result=result_str,
                    execution_time=execution_time
                )
                
                # Add artifact info to the response
                response["artifact_id"] = artifact_id
                response["artifact_name"] = artifact_name
            
            return response
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False
            }
            
            # Save failed execution as artifact if requested
            if save_artifact:
                if not artifact_name:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    artifact_name = f"failed_code_{timestamp}"
                
                # Add metadata about the error
                metadata = {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                
                # Save the artifact
                artifact_id = self.code_repo.add_artifact(
                    name=artifact_name,
                    code=code,
                    description=description or f"Failed execution: {str(e)}",
                    metadata=metadata
                )
                
                # Log the failed execution
                self.code_repo.log_execution(
                    artifact_id=artifact_id,
                    success=False,
                    stderr=traceback.format_exc(),
                    result=str(e)
                )
                
                # Add artifact info to the response
                error_result["artifact_id"] = artifact_id
                error_result["artifact_name"] = artifact_name
            
            return error_result
            
    def _save_module(self, name: str, code: str, description: str = "") -> Dict[str, Any]:
        """Save Python code as a reusable module."""
        try:
            # Validate module name (must be a valid Python identifier)
            if not name.isidentifier():
                return {
                    "success": False,
                    "error": f"Invalid module name: '{name}'. Must be a valid Python identifier."
                }
            
            # Check if module already exists
            existing_module = self.code_repo.get_module(name)
            if existing_module:
                # Update existing module
                success = self.code_repo.update_module(name, code, description)
                action = "updated"
            else:
                # Create new module
                self.code_repo.add_module(name, code, description)
                success = True
                action = "created"
            
            if not success:
                return {
                    "success": False,
                    "error": f"Failed to {action} module '{name}'"
                }
            
            # Try to validate the code by compiling it
            try:
                compile(code, f"<module:{name}>", "exec")
            except Exception as e:
                # We still save it even if compilation fails, but we warn the user
                return {
                    "success": True,
                    "warning": f"Module saved but contains syntax errors: {str(e)}",
                    "module_name": name,
                    "action": action
                }
            
            return {
                "success": True,
                "module_name": name,
                "description": description,
                "action": action
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def _execute_saved_module(self, name: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a previously saved module."""
        try:
            # Get the module
            module_data = self.code_repo.get_module(name)
            if not module_data:
                return {
                    "success": False,
                    "error": f"Module '{name}' not found"
                }
            
            # Create a globals dictionary with the arguments
            globals_dict = globals().copy()
            if args:
                globals_dict.update(args)
            
            # Set up output capturing
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            result = None
            
            # Execute the module code with redirected output
            start_time = time.time()
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # Create a locals dict to capture any defined functions and variables
                    locals_dict = {}
                    exec(module_data["code"], globals_dict, locals_dict)
                    
                    # Look for a main function to call
                    if "main" in locals_dict and callable(locals_dict["main"]):
                        if args:
                            result = locals_dict["main"](**args)
                        else:
                            result = locals_dict["main"]()
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue(),
                    "module_name": name
                }
                
            execution_time = time.time() - start_time
            
            # Convert complex result to string if needed
            result_str = None
            if result is not None:
                try:
                    if isinstance(result, (dict, list, tuple, set)):
                        result_str = json.dumps(result)
                    else:
                        result_str = str(result)
                except:
                    result_str = str(result)
            
            return {
                "success": True,
                "module_name": name,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "result": result_str,
                "execution_time": execution_time,
                "returned_main_function": "main" in locals_dict and callable(locals_dict["main"])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def _list_modules(self) -> Dict[str, Any]:
        """List all available Python modules."""
        try:
            modules = self.code_repo.list_modules()
            
            # Format the modules for display
            formatted_modules = []
            for module in modules:
                # Calculate code size and line count
                code = module["code"]
                line_count = len(code.split("\n"))
                code_size = len(code.encode("utf-8"))
                
                formatted_modules.append({
                    "name": module["name"],
                    "description": module["description"] or "No description",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(module["created_at"])),
                    "last_updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(module["last_updated_at"])),
                    "line_count": line_count,
                    "code_size_bytes": code_size
                })
            
            return {
                "success": True,
                "modules": formatted_modules,
                "count": len(formatted_modules)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def _get_module(self, name: str) -> Dict[str, Any]:
        """Get a specific Python module by name."""
        try:
            module = self.code_repo.get_module(name)
            if not module:
                return {
                    "success": False,
                    "error": f"Module '{name}' not found"
                }
            
            # Format the module data
            formatted_module = {
                "name": module["name"],
                "description": module["description"] or "No description",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(module["created_at"])),
                "last_updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(module["last_updated_at"])),
                "code": module["code"],
                "line_count": len(module["code"].split("\n")),
                "code_size_bytes": len(module["code"].encode("utf-8"))
            }
            
            return {
                "success": True,
                "module": formatted_module
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def _run_script(self, file_path: str, args: List[str] = None) -> Dict[str, Any]:
        """Run a Python script file and return the result."""
        try:
            # Validate file path
            path = Path(file_path).expanduser()
            if not path.exists():
                return {
                    "success": False,
                    "error": f"Script file '{file_path}' does not exist"
                }
            
            if not path.is_file():
                return {
                    "success": False,
                    "error": f"'{file_path}' is not a file"
                }
            
            # Prepare command with arguments
            cmd = [sys.executable, str(path)]
            if args:
                cmd.extend(args)
            
            # Execute the script as a separate process to ensure clean environment
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Get the output
            stdout, stderr = process.communicate()
            execution_time = time.time() - start_time
            
            # Create an artifact for this script execution
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            script_name = path.name
            
            # Read the script content
            with open(path, "r", encoding="utf-8") as f:
                script_content = f.read()
            
            # Save as an artifact
            artifact_id = self.code_repo.add_artifact(
                name=f"script_{script_name}_{timestamp}",
                code=script_content,
                description=f"Execution of script {script_name}",
                metadata={
                    "file_path": str(path),
                    "args": args or [],
                    "exit_code": process.returncode,
                    "execution_time": execution_time
                }
            )
            
            # Log the execution
            self.code_repo.log_execution(
                artifact_id=artifact_id,
                success=process.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time
            )
            
            return {
                "success": process.returncode == 0,
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "execution_time": execution_time,
                "file_path": str(path),
                "args": args or [],
                "artifact_id": artifact_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def _search_code(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for code artifacts by name or content."""
        try:
            artifacts = self.code_repo.find_artifacts(query, limit)
            
            # Format the artifacts for display
            formatted_artifacts = []
            for artifact in artifacts:
                # Get snippets of matching code
                code_snippets = []
                lines = artifact.code.split("\n")
                for i, line in enumerate(lines):
                    if query.lower() in line.lower():
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        snippet = "\n".join(lines[start:end])
                        line_number = i + 1
                        code_snippets.append({
                            "line": line_number,
                            "snippet": snippet
                        })
                
                formatted_artifacts.append({
                    "artifact_id": artifact.artifact_id,
                    "name": artifact.name,
                    "description": artifact.description or "No description",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(artifact.created_at)),
                    "execution_count": artifact.execution_count,
                    "line_count": len(artifact.code.split("\n")),
                    "code_size_bytes": len(artifact.code.encode("utf-8")),
                    "snippets": code_snippets[:3]  # Limit to first 3 snippets
                })
            
            return {
                "success": True,
                "query": query,
                "artifacts": formatted_artifacts,
                "count": len(formatted_artifacts)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _parse_weather_response(self, response: str) -> Dict[str, Any]:
        """Parse weather data into a readable format."""
        try:
            # Handle the case where response is already a dictionary
            if isinstance(response, dict):
                weather_data = response
            else:
                # Try to parse the response as JSON
                try:
                    weather_data = json.loads(response)
                except json.JSONDecodeError:
                    return {
                        "error": "Invalid weather data format",
                        "success": False
                    }
            
            # Extract relevant information
            location = weather_data.get("location", "Unknown location")
            condition = weather_data.get("current_condition", "unknown")
            temp_f = weather_data.get("temperature_f", 0)
            temp_c = weather_data.get("temperature_c", 0)
            humidity = weather_data.get("humidity", 0)
            wind_speed = weather_data.get("wind_speed_mph", 0)
            forecast = weather_data.get("forecast", [])
            
            # Create a human-readable summary
            summary = f"Current weather in {location}: {condition.capitalize()}, {temp_f}°F ({temp_c}°C), "
            summary += f"humidity {humidity}%, wind {wind_speed} mph."
            
            # Add forecast information if available
            forecast_summary = ""
            if forecast and len(forecast) > 0:
                tomorrow = forecast[0]
                forecast_summary = f"\n\nTomorrow's forecast: {tomorrow.get('condition', 'unknown').capitalize()}, "
                forecast_summary += f"high of {tomorrow.get('high_f', 0)}°F, low of {tomorrow.get('low_f', 0)}°F, "
                forecast_summary += f"{tomorrow.get('precipitation_chance', 0)}% chance of precipitation."
            
            return {
                "location": location,
                "summary": summary + forecast_summary,
                "condition": condition,
                "temperature_f": temp_f,
                "temperature_c": temp_c,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "forecast": forecast,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
            
    def _run_assistant(self, assistant_id: str, thread_id: str) -> Dict[str, Any]:
        """Run an assistant on a thread to generate a response."""
        try:
            # Get credentials for Together API from parent agent
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            
            # Together API doesn't currently support runs API directly
            # Instead, use the standard chat completions API with the conversation history
            
            # Get the system prompt based on assistant_id
            # For now, just use a default system prompt
            system_prompt = (
                f"You are an assistant (ID: {assistant_id}) helping with this conversation. "
                f"Please analyze the conversation history and provide a helpful response."
            )
            
            # Filter conversation history to only include messages for this thread
            # In this simplified implementation, we'll just use the agent's entire conversation history
            
            # Add the system message at the beginning if needed
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add the remaining conversation history
            for msg in agent.conversation_history:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Call the chat completions API
            response = agent.client.chat.completions.create(
                model=agent.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract the assistant's response
            response_content = response.choices[0].message.content
            
            # Add the assistant's response to the conversation history
            agent.add_message("assistant", response_content)
            
            return {
                "run_id": f"run_{int(time.time())}",
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "status": "completed",
                "response": response_content,
                "success": True
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _web_search(self, query: str) -> Dict[str, Any]:
        """Search the web using Jina s.jina.ai endpoint."""
        if not self.jina_client:
            return {"error": "Jina client not initialized. Please set JINA_API_KEY environment variable.", "success": False}
            
        try:
            # Run the async function in a synchronous context
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self.jina_client.search(query))
            return {"success": True, "query": query, "results": result["results"]}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _web_read(self, url: str) -> Dict[str, Any]:
        """Read a web page using Jina r.jina.ai endpoint."""
        if not self.jina_client:
            return {"error": "Jina client not initialized. Please set JINA_API_KEY environment variable.", "success": False}
            
        try:
            # Run the async function in a synchronous context
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self.jina_client.read(url))
            return {"success": True, "url": url, "content": result["results"]}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _fact_check(self, query: str) -> Dict[str, Any]:
        """Fact check a statement using Jina g.jina.ai endpoint."""
        if not self.jina_client:
            return {"error": "Jina client not initialized. Please set JINA_API_KEY environment variable.", "success": False}
            
        try:
            # Run the async function in a synchronous context
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(self.jina_client.fact_check(query))
            return {"success": True, "query": query, "fact_check_result": result}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
            
    def _create_planning_session(self, task: str) -> Dict[str, Any]:
        """Create a new planning session for complex multi-turn operations."""
        try:
            session_id = str(uuid.uuid4())
            created_at = time.time()
            
            # Get active TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if agent:
                session = PlanningSession(
                    session_id=session_id,
                    task=task,
                    created_at=created_at
                )
                agent.planning_session = session
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "task": task,
                    "message": "Planning session created successfully"
                }
            else:
                return {"error": "Could not access TogetherAgent instance", "success": False}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _add_plan_step(self, plan: str, tool_name: str = None, tool_args: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a step to the current planning session."""
        try:
            # Get active TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent or not agent.planning_session:
                return {"error": "No active planning session", "success": False}
            
            step = {
                "step_id": len(agent.planning_session.steps) + 1,
                "timestamp": time.time(),
                "plan": plan
            }
            
            # If tool execution is included
            if tool_name and tool_name in self.functions:
                step["tool"] = {
                    "name": tool_name,
                    "arguments": tool_args or {}
                }
                
                # Execute the tool if args are provided
                if tool_args:
                    result = self.call_function(tool_name, tool_args)
                    step["result"] = result
            
            agent.planning_session.steps.append(step)
            
            return {
                "success": True,
                "step_id": step["step_id"],
                "message": "Plan step added successfully"
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _get_planning_status(self) -> Dict[str, Any]:
        """Get the current status of the planning session."""
        try:
            # Get active TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent or not agent.planning_session:
                return {"error": "No active planning session", "success": False}
            
            return {
                "success": True,
                "session_id": agent.planning_session.session_id,
                "task": agent.planning_session.task,
                "steps_count": len(agent.planning_session.steps),
                "steps": agent.planning_session.steps,
                "state": agent.planning_session.state,
                "active": agent.planning_session.active,
                "completion_status": agent.planning_session.completion_status,
                "created_at": agent.planning_session.created_at,
                "elapsed_time": time.time() - agent.planning_session.created_at
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _complete_planning_session(self, summary: str, success: bool = True) -> Dict[str, Any]:
        """Complete the current planning session with a summary."""
        try:
            # Get active TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent or not agent.planning_session:
                return {"error": "No active planning session", "success": False}
            
            agent.planning_session.active = False
            agent.planning_session.completion_status = "completed" if success else "failed"
            
            # Add a final summary step
            agent.planning_session.steps.append({
                "step_id": len(agent.planning_session.steps) + 1,
                "timestamp": time.time(),
                "summary": summary,
                "success": success
            })
            
            # Add the completed session to the agent's history
            if not hasattr(agent, "completed_planning_sessions"):
                agent.completed_planning_sessions = []
                
            agent.completed_planning_sessions.append(agent.planning_session)
            
            # Clear the active session
            completed_session = agent.planning_session
            agent.planning_session = None
            
            return {
                "success": True,
                "session_id": completed_session.session_id,
                "task": completed_session.task,
                "steps_count": len(completed_session.steps),
                "completion_status": completed_session.completion_status,
                "elapsed_time": time.time() - completed_session.created_at,
                "summary": summary
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
            
    def _extract_urls(self, text: str) -> Dict[str, Any]:
        """Extract URLs from text and return as structured output."""
        try:
            # Get active TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            
            # Extract URLs
            extraction = agent.task_processor.extract_urls(text)
            
            return {
                "success": True,
                "urls": extraction.urls,
                "count": len(extraction.urls),
                "source": extraction.source,
                "timestamp": extraction.timestamp
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _process_urls(self, urls: List[str]) -> Dict[str, Any]:
        """Process URLs to extract knowledge."""
        try:
            # Get active TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            
            # Process URLs
            new_urls_count = agent.task_processor.add_urls_to_process(urls)
            
            return {
                "success": True,
                "urls_added": new_urls_count,
                "total_urls_pending": agent.task_processor.task_queue.qsize(),
                "total_urls_processed": len(agent.task_processor.processed_urls),
                "message": f"Added {new_urls_count} new URLs to the processing queue"
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _get_knowledge_summary(self) -> Dict[str, Any]:
        """Get a summary of the knowledge base."""
        try:
            # Get active TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            
            # Get knowledge summary
            summary = agent.task_processor.get_knowledge_summary()
            summary["success"] = True
            
            return summary
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _search_knowledge(self, query: str) -> Dict[str, Any]:
        """Search the knowledge base for relevant information."""
        try:
            # Get active TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            
            # Search knowledge
            results = agent.task_processor.search_knowledge(query)
            
            return {
                "success": True,
                "query": query,
                "results_count": len(results),
                "results": results
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _monitor_task(self, task_id: str) -> Dict[str, Any]:
        """Monitor the status of a background task."""
        try:
            # Get active TogetherAgent instance
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            
            # Get task status
            result = agent.task_processor.get_result(task_id)
            result["task_id"] = task_id
            
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

class TogetherAgent:
    def __init__(self, model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
        """Initialize the agent with the Together API."""
        self.api_key = os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set TOGETHER_API_KEY environment variable.")
        
        self.use_mock = "dummy_api_key_for_testing" in self.api_key
        
        if not self.use_mock:
            self.client = Together(api_key=self.api_key)
        else:
            # Create a mock client that mimics the Together client
            from types import SimpleNamespace
            
            # Define mock completion class
            def mock_completion(**kwargs):
                if "python" in kwargs.get("prompt", "").lower() or (isinstance(kwargs.get("messages", []), list) and any("python" in str(m.get("content", "")).lower() for m in kwargs.get("messages", []))):
                    # For the purposes of testing our code execution feature, use exact format with tags
                    if "use python code" in str(kwargs).lower() and "date" in str(kwargs).lower():
                        mock_text = """Here's Python code to print the current date:

<|python_start|>
from datetime import date

def get_current_date():
    # Get today's date
    today = date.today()
    return today

current_date = get_current_date()
print("Current Date: ", current_date)
<|python_end|>

This code imports the date class from the datetime module, defines a function to get the current date, and then prints it."""
                    else:
                        mock_text = """Here's Python code to perform the requested task:

<|python_start|>
# Example Python code
print("Hello, world!")
<|python_end|>"""
                    
                    # For chat completions API
                    if "messages" in kwargs:
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content=mock_text,
                                        model_dump=lambda: {"role": "assistant", "content": mock_text}
                                    )
                                )
                            ]
                        )
                    # For completions API
                    else:
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    text=mock_text
                                )
                            ]
                        )
                else:
                    mock_text = "I'm a mock response from the agent."
                    
                    # For chat completions API
                    if "messages" in kwargs:
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content=mock_text,
                                        model_dump=lambda: {"role": "assistant", "content": mock_text}
                                    )
                                )
                            ]
                        )
                    # For completions API
                    else:
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    text=mock_text
                                )
                            ]
                        )
            
            # Create a mock client structure
            self.client = SimpleNamespace(
                completions=SimpleNamespace(
                    create=mock_completion
                ),
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=mock_completion
                    )
                )
            )
            
            # Log that we're using mock mode
            console.print("[yellow]Running in mock mode with test responses[/yellow]")
            
        self.model = model
        self.tool_registry = ToolRegistry()
        self.conversation_history = []
        self.reflection_notes = []  # Store for self-reflection notes
        self.enable_logprobs = False  # Whether to enable logprobs in API calls
        self.enable_planning = True   # Whether to enable multi-turn planning
        self.planning_session = None  # Current active planning session
        
        # Initialize the async task processor for background learning
        self.task_processor = AsyncTaskProcessor()
        
        self.environment_state = {
            "system": self._detect_system_info(),
            "user_behavior": {
                "message_count": 0,
                "avg_message_length": 0,
                "technical_level": "medium",  # Initial assumption
                "detected_preferences": []
            },
            "conversation_context": {
                "topic_clusters": [],
                "sentiment": "neutral",
                "task_oriented": True
            },
            "task_complexity": {
                "current_level": "medium",
                "adaptations_made": []
            },
            "last_analysis_time": time.time()
        }
        
        # Define model capabilities
        self.is_llama4 = "llama-4" in model.lower()
        self.supports_images = self.is_llama4  # Only Llama 4 supports images in this implementation
        self.max_images = 5 if self.is_llama4 else 0  # Llama 4 supports up to 5 images
        
        # Language support - Llama 4 supports multiple languages
        self.supported_languages = [
            "Arabic", "English", "French", "German", "Hindi", "Indonesian", 
            "Italian", "Portuguese", "Spanish", "Tagalog", "Thai", "Vietnamese"
        ] if self.is_llama4 else ["English"]
        
        # Define maximum context length based on model
        if "maverick" in model.lower():
            self.max_context_length = 1000000  # 1M tokens
        elif "scout" in model.lower():
            self.max_context_length = 10000000  # 10M tokens
        else:
            self.max_context_length = 8192  # Default for other models
        
        # Define system prompt with Llama 4 format
        if self.is_llama4:
            # Use Llama 4 recommended system prompt
            system_content = (
                "You are an expert conversationalist who responds to the best of your ability. "
                "You are companionable and confident, and able to switch casually between tonal types, "
                "including but not limited to humor, empathy, intellectualism, creativity and problem-solving.\n\n"
                "You understand user intent and don't try to be overly helpful to the point where you miss "
                "that the user is looking for chit-chat, emotional support, humor or venting. Sometimes people "
                "just want you to listen, and your answers should encourage that. For all other cases, you provide "
                "insightful and in-depth responses. Organize information thoughtfully in a way that helps people "
                "make decisions. Always avoid templated language.\n\n"
                "You are also an expert in using tools to accomplish tasks. You can invoke functions when needed "
                "using one of these formats:\n"
                "1. [function_name(param1=value1, param2=value2)]\n"
                "2. <function=function_name>{\"param1\": \"value1\", \"param2\": \"value2\"}</function>\n\n"
                "You have access to the Together AI platform tools to create assistants, threads, and manage conversations. "
                "When working with the Together API, use the provided tools in sequence for best results.\n\n"
                "You also have self-reflection capabilities through reflection notes to record insights, "
                "track strategy changes, remember important facts about the conversation, and adapt to the user's needs.<|eot|>"
            )
        else:
            # Generic system prompt for non-Llama 4 models
            system_content = (
                "You are a helpful AI assistant with the ability to use tools to accomplish tasks. "
                "You can create new tools on the fly by defining Python functions. "
                "Always prefer using existing tools over creating new ones when they can accomplish the task. "
                "When creating new tools, ensure they're well-documented and robust. "
                "Think step-by-step and be thorough in your analysis.\n\n"
                "You have access to the Together AI platform tools to create assistants, threads, and manage conversations. "
                "You can list available models, generate completions, create assistants, and manage conversation threads. "
                "For multi-step tasks, create a thread, add messages, and run an assistant to generate responses iteratively. "
                "When working with the Together API, use the provided tools in sequence for best results.\n\n"
                "You also have self-reflection capabilities through reflection notes. Use these to record your insights, "
                "track strategy changes, remember important facts about the conversation, and adapt to the user's needs. "
                "Periodically review your reflection notes to improve your assistance quality."
            )
        
        self.system_message = {
            "role": "system", 
            "content": system_content
        }
        
        self.conversation_history.append(self.system_message)
    
    def _detect_system_info(self):
        """Detect information about the system environment."""
        system_info = {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "os_name": os.name,
            "cpu_count": os.cpu_count() or "unknown",
            "terminal_size": {"columns": 80, "lines": 24},  # Default values to avoid ioctl errors
            "environment_variables": {
                # Include only non-sensitive environment variables
                "PATH_exists": "PATH" in os.environ,
                "HOME_exists": "HOME" in os.environ,
                "LANG": os.environ.get("LANG", "unknown")
            }
        }
        return system_info
    
    def update_environment_user_behavior(self, message):
        """Update user behavior metrics based on new message."""
        # Handle multi-modal content
        message_text = ""
        if isinstance(message, list):
            # Extract text content from multi-modal message
            for item in message:
                if item.get('type') == 'text':
                    message_text += item.get('text', '')
        else:
            message_text = message
        
        # Update message count
        self.environment_state["user_behavior"]["message_count"] += 1
        
        # Update average message length
        current_count = self.environment_state["user_behavior"]["message_count"]
        current_avg = self.environment_state["user_behavior"]["avg_message_length"]
        new_length = len(message_text)
        
        # Calculate new average
        new_avg = ((current_avg * (current_count - 1)) + new_length) / current_count
        self.environment_state["user_behavior"]["avg_message_length"] = new_avg
        
        # Basic technical level detection (could be enhanced with more sophisticated NLP)
        technical_terms = [
            "code", "function", "class", "method", "algorithm", "implementation",
            "api", "database", "query", "json", "xml", "http", "rest",
            "async", "thread", "concurrency", "memory", "cpu", "processor",
            "git", "repository", "commit", "merge", "branch"
        ]
        
        technical_count = sum(1 for term in technical_terms if term.lower() in message_text.lower())
        
        # Update technical level based on density of technical terms
        if technical_count > 5 or (technical_count / max(1, len(message_text.split())) > 0.1):
            self.environment_state["user_behavior"]["technical_level"] = "high"
        elif technical_count > 2:
            self.environment_state["user_behavior"]["technical_level"] = "medium"
        else:
            # Only downgrade to low if we have enough messages to be confident
            if current_count > 3:
                self.environment_state["user_behavior"]["technical_level"] = "low"
                
        # Check if message contains images
        has_images = isinstance(message, list) and any(item.get('type') == 'image_url' for item in message)
        if has_images:
            if "multimodal" not in self.environment_state["user_behavior"]["detected_preferences"]:
                self.environment_state["user_behavior"]["detected_preferences"].append("multimodal")
    
    def extract_python_code(self, text: str) -> List[str]:
        """Extract Python code from text between <|python_start|> and <|python_end|> tags.
        
        Args:
            text: The text containing potential Python code blocks
            
        Returns:
            List of Python code blocks
        """
        import re
        # Find all code between <|python_start|> and <|python_end|> or <|python_end
        pattern = r'<\|python_start\|>(.*?)(?:<\|python_end\|>|<\|python_end)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        return [match.strip() for match in matches]
        
# Make the extract_python_code function available at module level for testing
def extract_python_code(text: str) -> list:
    """Extract Python code from text between <|python_start|> and <|python_end|> tags."""
    import re
    pattern = r'<\|python_start\|>(.*?)(?:<\|python_end\|>|<\|python_end)'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]
        
    def parse_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse function calls from text in various Llama 4 supported formats.
        
        Supports two formats:
        1. [func_name(param1=value1, param2=value2)]
        2. <function=func_name>{"param1": "value1", "param2": "value2"}</function>
        
        Args:
            text: The text containing potential function calls
            
        Returns:
            List of dictionaries with function name and arguments
        """
        function_calls = []
        
        # Look for the standard format: [func_name(param1="value1", param2=value2)]
        import re
        pattern = r'\[(\w+)\((.*?)\)\]'
        matches = re.findall(pattern, text)
        
        for match in matches:
            func_name = match[0]
            args_str = match[1]
            
            # Parse arguments
            args = {}
            try:
                # First try to parse as JSON if arguments look like a JSON object
                args_str = args_str.strip()
                if args_str.startswith("{") and args_str.endswith("}"):
                    args = json.loads(args_str)
                else:
                    # Fall back to regex for param=value format
                    # Split by commas, but not those inside quotes
                    arg_pairs = re.findall(r'(\w+)=("[^"]*"|\'[^\']*\'|\S+)', args_str)
                    
                    for arg_name, arg_value in arg_pairs:
                        # Remove quotes if present
                        if (arg_value.startswith('"') and arg_value.endswith('"')) or \
                           (arg_value.startswith("'") and arg_value.endswith("'")):
                            arg_value = arg_value[1:-1]
                        args[arg_name] = arg_value
            except json.JSONDecodeError:
                console.print(f"[red]Error parsing arguments for {func_name}: {args_str}[/red]")
            
            function_calls.append({
                "name": func_name,
                "arguments": args
            })
        
        # Also check for the custom format <function=func_name>{"param": "value"}</function>
        function_pattern = r'<function=(\w+)>(.*?)</function>'
        function_matches = re.findall(function_pattern, text)
        
        for match in function_matches:
            func_name = match[0]
            try:
                args = json.loads(match[1])
                function_calls.append({
                    "name": func_name,
                    "arguments": args
                })
            except json.JSONDecodeError:
                console.print(f"[red]Error parsing arguments for {func_name}: {match[1]}[/red]")
        
        return function_calls

    def format_llama4_prompt(self) -> str:
        """Format messages according to Llama 4 prompt format.
        
        Returns:
            A properly formatted Llama 4 prompt string
        """
        # Start with begin_of_text token
        formatted_prompt = "<|begin_of_text|>"
        
        # Add all messages with proper header tags
        for msg in self.conversation_history:
            role = msg.get("role")
            content = msg.get("content", "")
            
            # Skip 'tool' type messages for now - they don't fit directly in the Llama 4 format
            if role == "tool":
                continue
                
            # Add header and content
            formatted_prompt += f"<|header_start|>{role}<|header_end|>\n\n{content}"
            
            # Make sure each message (except system) ends with <|eot|>
            if role != "system" and not content.endswith("<|eot|>"):
                formatted_prompt += "<|eot|>"
        
        # Add the final assistant header
        formatted_prompt += f"<|header_start|>assistant<|header_end|>"
        
        return formatted_prompt
    
    def process_image_in_message(self, message):
        """Process and extract image references from message.
        
        For Llama 4, handles multi-modal messages with multiple images.
        
        Args:
            message: The user message (string or list of content objects)
            
        Returns:
            Processed message suitable for the Llama 4 API
        """
        # Check if this is a multi-modal message (list of content objects)
        if isinstance(message, list):
            # This is already a structured multi-modal message
            return message
        
        # Check if the message contains image URLs in standard format
        import re
        image_urls = re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)', message)
        
        if not image_urls:
            # No images detected, return the original message
            return message
        
        # Convert to multi-modal format
        multimodal_content = []
        
        # Add the text content first, removing the image URLs
        text_content = message
        for url in image_urls:
            text_content = text_content.replace(url, '')
        
        text_content = text_content.strip()
        
        if text_content:
            multimodal_content.append({
                "type": "text",
                "text": text_content
            })
        
        # Add each image URL as a separate content object
        for url in image_urls:
            multimodal_content.append({
                "type": "image_url",
                "image_url": {
                    "url": url
                }
            })
        
        return multimodal_content
    
    def add_message(self, role: str, content):
        """Add a message to the conversation history.
        
        For Llama 4 models, messages are properly formatted with the required tokens.
        
        Args:
            role: The role of the message sender (user, assistant, system, tool)
            content: The content of the message (string or list for multi-modal)
        """
        # Process any images if this is a user message in a Llama 4 model
        if self.is_llama4 and role == "user" and self.supports_images:
            content = self.process_image_in_message(content)
        
        # Extract URLs from content for continuous learning (only from user and assistant messages)
        if role in ["user", "assistant"] and isinstance(content, str) and hasattr(self, "task_processor"):
            extracted = self.task_processor.extract_urls(content)
            if extracted.urls and len(extracted.urls) > 0:
                # Add URLs to processing queue in background
                self.task_processor.add_urls_to_process(extracted.urls)
                
                # Add a note about it
                console.print(f"[dim][Extracted {len(extracted.urls)} URLs for background processing][/dim]", highlight=False)
            
        # Create the message object
        message = {"role": role}
        
        # Handle multi-modal content vs text content
        if isinstance(content, list):
            # This is multi-modal content
            message["content"] = content
        else:
            # This is text content
            # Format message according to Llama 4 format requirements if using a Llama 4 model
            # For Llama 4, messages are formatted with special tokens:
            # <|header_start|>role<|header_end|>content<|eot|>
            if self.is_llama4 and role != "system" and not content.endswith("<|eot|>"):
                content = content + "<|eot|>"
            
            message["content"] = content
            
            # If this is a user message, update environment stats
            if role == "user":
                self.update_environment_user_behavior(content)
                
        # Add the message to conversation history
        self.conversation_history.append(message)
    
    def process_tool_calls(self, tool_calls) -> List[Dict[str, Any]]:
        """Process tool calls from the agent's response."""
        results = []
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
                console.print(f"[red]Error parsing arguments for {function_name}: {tool_call.function.arguments}[/red]")
            
            console.print(f"[cyan]Calling function: [bold]{function_name}[/bold][/cyan]")
            console.print(f"[cyan]With arguments:[/cyan] {json.dumps(arguments, indent=2)}")
            
            result = self.tool_registry.call_function(function_name, arguments)
            
            # Add the tool result to the conversation
            self.conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(result, indent=2)
            })
            
            results.append({
                "function_name": function_name,
                "arguments": arguments,
                "result": result
            })
        
        return results
    
    def chat(self, message) -> str:
        """Send a message to the agent and get a response.
        
        Args:
            message: The user message (can be a string or a structured multi-modal message)
        
        Returns:
            The agent's response as a string
        """
        # Add the user message to the conversation
        self.add_message("user", message)
        
        # Get the tools in the format expected by the API
        tools = self.tool_registry.get_openai_tools_format()
        
        # Check if we should perform periodic environment analysis
        user_message_count = len([msg for msg in self.conversation_history if msg.get("role") == "user"])
        
        # Perform environment analysis every 3 user messages or if this is the first message
        if user_message_count == 1 or user_message_count % 3 == 0:
            # Use the tool registry to analyze the environment
            self.tool_registry._analyze_environment(aspect="all")
            
            # Check if automatic adaptation is needed
            env = self.environment_state
            
            # Simple adaptation heuristics based on analysis
            if env["user_behavior"]["technical_level"] == "high" and env["task_complexity"]["current_level"] == "high":
                # For highly technical users with complex tasks
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="increase_technical_detail",
                    reason="User exhibits high technical knowledge and is working on complex tasks",
                    system_prompt_update="Focus on providing detailed technical explanations and comprehensive code examples. Prioritize correctness and best practices over simplicity."
                )
            elif env["user_behavior"]["technical_level"] == "low" and env["conversation_context"]["sentiment"] == "negative":
                # For users struggling with technical content
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="simplify_explanations",
                    reason="User appears to be struggling with technical content",
                    system_prompt_update="Simplify explanations, avoid jargon, provide more step-by-step guidance, and use analogies where possible."
                )
            elif env["conversation_context"]["task_oriented"] and env["task_complexity"]["current_level"] == "medium":
                # For task-focused users with medium complexity
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="focus_on_practical_solutions",
                    reason="User is task-oriented and working on moderately complex tasks",
                    system_prompt_update="Focus on practical solutions, provide executable examples, and offer step-by-step procedures. Be concise but thorough."
                )
            elif "multimodal" in env["user_behavior"]["detected_preferences"]:
                # For users who are using multi-modal features
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="optimize_multimodal",
                    reason="User is working with multi-modal content like images",
                    system_prompt_update="Pay special attention to visual content in the conversation. Provide detailed descriptions and analyses of images when they're present."
                )
        
        while True:
            try:
                # For Llama 4 models with multi-modal or tool support
                if self.model.startswith("meta-llama/Llama-4"):
                    # Detect if this is a multi-modal conversation
                    has_multimodal = any(
                        isinstance(msg.get("content"), list) for msg in self.conversation_history
                    )
                    
                    if has_multimodal:
                        # Use the chat completions API for multi-modal support
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=self.conversation_history,
                            tools=tools,
                            tool_choice="auto",
                            max_tokens=1024
                        )
                        
                        # Get the assistant's response
                        assistant_message = response.choices[0].message
                        
                        # Add the assistant's message to the conversation history
                        self.conversation_history.append(assistant_message.model_dump())
                        
                        # Check if the assistant wants to use tools
                        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                            # Process the tool calls
                            results = self.process_tool_calls(assistant_message.tool_calls)
                            
                            # Log the results
                            for result in results:
                                console.print(f"[green]Result from {result['function_name']}:[/green]")
                                console.print(json.dumps(result['result'], indent=2))
                            
                            # After processing tool calls, get a final response that incorporates the tool results
                            final_response = self.client.chat.completions.create(
                                model=self.model,
                                messages=self.conversation_history,
                                tools=tools,
                                tool_choice="none"  # Force text response after tool use
                            )
                            
                            final_message = final_response.choices[0].message
                            self.conversation_history.append(final_message.model_dump())
                            return final_message.content
                        
                        # Return the assistant's response content
                        return assistant_message.content
                    else:
                        # For text-only Llama 4, can use either method - prefer completions for now
                        # Format the Llama 4 prompt using the dedicated method
                        formatted_prompt = self.format_llama4_prompt()
                        
                        # Create parameters dict with optional logprobs
                        params = {
                            "model": self.model,
                            "prompt": formatted_prompt,
                            "max_tokens": 1024,
                            "stop": ["<|eot|>"]
                        }
                        
                        # Add logprobs if this is a compatible model and enabled
                        if self.is_llama4 and self.enable_logprobs:
                            params["logprobs"] = 1
                        
                        response = self.client.completions.create(**params)
                        
                        # Extract text and add the eot token back
                        assistant_content = response.choices[0].text + "<|eot|>"
                        assistant_message = {"role": "assistant", "content": assistant_content}
                        
                        # Save logprobs info if available
                        if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                            logprobs_data = response.choices[0].logprobs
                            assistant_message["logprobs"] = {
                                "tokens": logprobs_data.tokens,
                                "token_logprobs": logprobs_data.token_logprobs
                            }
                            
                            # For debugging/analysis, calculate the average logprob
                            if len(logprobs_data.token_logprobs) > 0:
                                avg_logprob = sum(lp for lp in logprobs_data.token_logprobs if lp is not None) / len(logprobs_data.token_logprobs)
                                assistant_message["avg_logprob"] = avg_logprob
                        
                        # Add to conversation history
                        self.conversation_history.append(assistant_message)
                        
                        # Check for function calls in the text
                        if "[" in assistant_content and "(" in assistant_content and ")" in assistant_content or "<function=" in assistant_content:
                            # Parse function calls from text using class method
                            function_calls = self.parse_function_calls(assistant_content)
                            
                            if function_calls:
                                # Process function calls
                                results = []
                                for func_call in function_calls:
                                    name = func_call["name"]
                                    args = func_call["arguments"]
                                    console.print(f"[cyan]Calling function: [bold]{name}[/bold][/cyan]")
                                    console.print(f"[cyan]With arguments:[/cyan] {json.dumps(args, indent=2)}")
                                    
                                    result = self.tool_registry.call_function(name, args)
                                    console.print(f"[green]Result from {name}:[/green]")
                                    console.print(json.dumps(result, indent=2))
                                    
                                    # Add tool result to conversation history
                                    self.conversation_history.append({
                                        "role": "tool",
                                        "name": name,
                                        "content": json.dumps(result, indent=2)
                                    })
                                    
                                    results.append({"function_name": name, "arguments": args, "result": result})
                                
                                # Process the results to generate a final response without recursion
                                final_prompt = f"Based on these function results: {json.dumps(results)}, provide a direct answer to the user's question."
                                final_response = self.client.completions.create(
                                    model=self.model,
                                    prompt=self.format_llama4_prompt() + "\n\nUser: " + final_prompt + "<|eot|>\nAssistant: ",
                                    max_tokens=1024,
                                    stop=["<|eot|>"]
                                )
                                
                                # Extract the final response text
                                final_content = final_response.choices[0].text + "<|eot|>"
                                final_message = {"role": "assistant", "content": final_content}
                                self.conversation_history.append(final_message)
                                
                                # Return the formatted response directly
                                return final_content.rstrip("<|eot|>")
                        
                        # Parse and handle code execution blocks
                        response_content = assistant_content.rstrip("<|eot|>")
                        if "<|python_start|>" in response_content and "<|python_end" in response_content:
                            # Extract and execute python code
                            code_blocks = self.extract_python_code(response_content)
                            if code_blocks:
                                for code_block in code_blocks:
                                    if "Run the code" in user_input or "run the code" in user_input:
                                        console.print("[cyan]Executing Python code:[/cyan]")
                                        
                                        # Direct execution using our own _execute_python method
                                        import io
                                        import datetime  # Ensure datetime module is available
                                        from contextlib import redirect_stdout, redirect_stderr
                                        
                                        stdout_capture = io.StringIO()
                                        stderr_capture = io.StringIO()
                                        local_vars = {}
                                        
                                        # Prepare execution environment with imports
                                        exec_globals = globals().copy()
                                        # Add datetime modules to globals
                                        from datetime import date, datetime, timedelta
                                        exec_globals['date'] = date
                                        exec_globals['datetime'] = datetime
                                        exec_globals['timedelta'] = timedelta
                                        
                                        try:
                                            console.print(f"[dim]{code_block}[/dim]")
                                            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                                                exec(code_block, exec_globals, local_vars)
                                            
                                            stdout = stdout_capture.getvalue()
                                            stderr = stderr_capture.getvalue()
                                            
                                            if stdout:
                                                console.print("[green]Execution result:[/green]")
                                                console.print(stdout)
                                            
                                            if stderr:
                                                console.print("[red]Errors:[/red]")
                                                console.print(stderr)
                                                
                                        except Exception as e:
                                            import traceback
                                            console.print(f"[red]Error executing code: {str(e)}[/red]")
                                            console.print(traceback.format_exc())
                                    else:
                                        # If not asked to run, just display the code
                                        pass
                        
                        # Return the normal text content
                        return response_content
                else:
                    # For non-Llama 4 models, use regular OpenAI format
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        tools=tools,
                        tool_choice="auto"
                    )
                
                    # Get the assistant's response
                    assistant_message = response.choices[0].message
                    
                    # Add the assistant's message to the conversation history
                    self.conversation_history.append(assistant_message.model_dump())
                    
                    # Check if the assistant wants to use tools
                    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                        # Process the tool calls
                        results = self.process_tool_calls(assistant_message.tool_calls)
                        
                        # Log the results
                        for result in results:
                            console.print(f"[green]Result from {result['function_name']}:[/green]")
                            console.print(json.dumps(result['result'], indent=2))
                        
                        # After processing tool calls, get a final response that incorporates the tool results
                        final_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=self.conversation_history,
                            tools=tools,
                            tool_choice="none"  # Force text response after tool use
                        )
                        
                        final_message = final_response.choices[0].message
                        self.conversation_history.append(final_message.model_dump())
                        return final_message.content
                    
                    # Return the assistant's response content
                    return assistant_message.content
            
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
                console.print(traceback.format_exc())
                return f"Error: {str(e)}"
    
    def get_response(self):
        """Get a response from the agent for the current conversation.
        Used for direct testing without needing a full interactive session."""
        return self.get_response_with_tools(user_input="")


def main():
    parser = argparse.ArgumentParser(description="Chat with an AI agent using Together API with dynamic tools")
    parser.add_argument("--model", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", 
                       help="Model to use (default: meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)")
    parser.add_argument("--logprobs", action="store_true", 
                       help="Enable returning logprobs for confidence analysis")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode with mock API responses")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold blue]Together Agent CLI[/bold blue]\n"
        "Chat with an AI agent that can use and create tools dynamically.\n"
        "The agent has self-adaptive capabilities:\n"
        "- Can use the Together API to access Llama 4 and other models\n"
        "- Supports OpenAI-compatible function calling\n"
        "- Handles multi-modal content with multiple images\n"
        "- Returns token logprobs for confidence analysis\n"
        "- Dynamically updates its system prompt based on the conversation\n"
        "- Maintains self-reflection notes to track insights\n"
        "- Creates assistants and conversation threads\n"
        "- Analyzes the environment and user behavior to adapt responses\n"
        "- Adjusts technical level based on conversation context\n"
        "Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to end the conversation.",
        title="Welcome"
    ))
    
    # For testing, set a dummy API key if in test mode
    if args.test_mode and "TOGETHER_API_KEY" not in os.environ:
        os.environ["TOGETHER_API_KEY"] = "dummy_api_key_for_testing"
    
    # Initialize the agent
    try:
        agent = TogetherAgent(model=args.model)
        
        # Enable logprobs if requested
        agent.enable_logprobs = args.logprobs
    except ValueError as e:
        if "API key is required" in str(e) and not args.test_mode:
            console.print("[red]Error: Together API key is required. Set the TOGETHER_API_KEY environment variable.[/red]")
            return 1
        raise
    
    # Start the conversation loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
            
            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Exiting...[/yellow]")
                break
            
            # Detect if this is a multi-modal input with image URLs
            image_urls = re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)', user_input)
            
            if image_urls:
                # Create a multi-modal message
                multimodal_content = []
                
                # Add the text content first, removing the image URLs
                text_content = user_input
                for url in image_urls:
                    text_content = text_content.replace(url, '')
                
                text_content = text_content.strip()
                
                if text_content:
                    multimodal_content.append({
                        "type": "text",
                        "text": text_content
                    })
                
                # Add each image URL as a separate content object
                for url in image_urls:
                    console.print(f"[cyan]Including image: {url}[/cyan]")
                    multimodal_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": url
                        }
                    })
                
                # Start a spinner while waiting for the response
                with console.status("[bold blue]Thinking about your images...[/bold blue]", spinner="dots"):
                    response = agent.chat(multimodal_content)
            else:
                # Standard text message
                # Start a spinner while waiting for the response
                with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
                    response = agent.chat(user_input)
            
            # Check if we have logprobs to analyze model confidence
            if any('logprobs' in msg for msg in agent.conversation_history[-2:] if isinstance(msg, dict)):
                # Find most recent message with logprobs
                for msg in reversed(agent.conversation_history):
                    if isinstance(msg, dict) and 'logprobs' in msg and msg.get('role') == 'assistant':
                        avg_confidence = msg.get('avg_logprob', None)
                        if avg_confidence is not None:
                            confidence_level = "high" if avg_confidence > -1.0 else "medium" if avg_confidence > -2.0 else "low"
                            console.print(f"[cyan]Model confidence: {confidence_level} (avg logprob: {avg_confidence:.2f})[/cyan]")
                        break
            
            # Display the response
            console.print("\n[bold purple]Assistant[/bold purple]:")
            console.print(Markdown(response))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]")
            console.print(traceback.format_exc())
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
