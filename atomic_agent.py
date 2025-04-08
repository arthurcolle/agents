#!/usr/bin/env python3
"""
cli.py
------
A CLI tool to chat with an AI agent using the Together API with dynamic tools.
This version defines all functions and tools, and supports multiple function calls
within a single turn. The system prompt (for Llama‑4 models) now informs the model
that multiple function calls may be issued.
"""

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
import time
import queue
import threading
import tempfile
import urllib.parse
import asyncio
import traceback
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO, BytesIO
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional

try:
    import aiohttp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp

try:
    import pytz
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytz"])
    import pytz

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

# =======================
# Data Classes & Helpers
# =======================
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
    session_id: str
    task: str
    created_at: float
    steps: List[Dict[str, Any]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    completion_status: str = "in_progress"  # "in_progress", "completed", "failed"

@dataclass
class StructuredOutput:
    source: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class URLExtraction(StructuredOutput):
    urls: List[str] = field(default_factory=list)

@dataclass
class KnowledgeItem:
    content: str
    source_url: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

# ================================
# Async Task Processor
# ================================
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
        while not self.stop_event.is_set():
            try:
                task_id, task_func, args, kwargs = self.task_queue.get(timeout=1)
                try:
                    if asyncio.iscoroutinefunction(task_func):
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
                continue

    def add_task(self, task_func, *args, **kwargs):
        task_id = str(uuid.uuid4())
        self.results[task_id] = {"status": "pending"}
        self.task_queue.put((task_id, task_func, args, kwargs))
        return task_id

    def get_result(self, task_id):
        return self.results.get(task_id, {"status": "not_found"})

    def extract_urls(self, text):
        urls = [url for url in self.url_pattern.findall(text)]
        normalized_urls = []
        for url in urls:
            if url.startswith('www.'):
                url = 'https://' + url
            normalized_urls.append(url)
        return URLExtraction(source="text_extraction", urls=normalized_urls)

    def add_urls_to_process(self, urls):
        new_urls = [url for url in urls if url not in self.processed_urls]
        for url in new_urls:
            self.processed_urls.add(url)
            self.add_task(self._process_url, url)
        return len(new_urls)

    async def _process_url(self, url):
        try:
            jina_client = JinaClient(token=os.environ.get("JINA_API_KEY"))
            result = await jina_client.read(url)
            if isinstance(result, dict) and "results" in result:
                content = result["results"]
                knowledge_item = KnowledgeItem(content=content, source_url=url)
                self.knowledge_base.append(knowledge_item)
                urls_extraction = self.extract_urls(content)
                self.add_urls_to_process(urls_extraction.urls)
                return {
                    "success": True,
                    "url": url,
                    "knowledge_extracted": True,
                    "further_urls_found": len(urls_extraction.urls)
                }
        except Exception as e:
            return {"success": False, "url": url, "error": str(e)}

    def get_knowledge_summary(self):
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
        self.stop_event.set()
        self.worker_thread.join(timeout=2)

# =======================
# Jina API Client
# =======================
class JinaClient:
    """Client for interacting with Jina.ai endpoints"""
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("JINA_API_KEY")
        if not self.token:
            raise ValueError("JINA_API_KEY environment variable or token must be provided")
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    async def search(self, query: str) -> dict:
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/{encoded_query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response_text = await response.text()
                return {"results": response_text}

    async def fact_check(self, query: str) -> str:
        encoded_query = urllib.parse.quote(query)
        url = f"https://g.jina.ai/{encoded_query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                return await response.text()

    async def read(self, url: str) -> dict:
        encoded_url = urllib.parse.quote(url)
        rank_url = f"https://r.jina.ai/{encoded_url}"
        async with aiohttp.ClientSession() as session:
            async with session.get(rank_url, headers=self.headers) as response:
                response_text = await response.text()
                return {"results": response_text}

# =======================
# Code Repository
# =======================
class CodeRepository:
    """Repository for storing and managing code artifacts."""
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._initialize_db()
        
    def _initialize_db(self):
        cursor = self.conn.cursor()
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
        cursor = self.conn.cursor()
        cursor.execute("SELECT code, description, metadata FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        row = cursor.fetchone()
        if not row:
            return False
        current_code, current_description, current_metadata_str = row
        current_metadata = json.loads(current_metadata_str) if current_metadata_str else {}
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
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        row = cursor.fetchone()
        if not row:
            return None
        column_names = [desc[0] for desc in cursor.description]
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
        col_names = [desc[0] for desc in cursor.description]
        for row in rows:
            artifact_data = dict(zip(col_names, row))
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
        log_id = str(uuid.uuid4())
        executed_at = time.time()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO execution_logs (log_id, artifact_id, executed_at, success, stdout, stderr, result, execution_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (log_id, artifact_id, executed_at, 1 if success else 0, stdout, stderr, result, execution_time)
        )
        cursor.execute(
            "UPDATE artifacts SET execution_count = execution_count + 1, last_result = ? WHERE artifact_id = ?",
            (json.dumps({"success": success, "result": result, "executed_at": executed_at}), artifact_id)
        )
        self.conn.commit()
        return log_id

    def get_execution_logs(self, artifact_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM execution_logs WHERE artifact_id = ? ORDER BY executed_at DESC LIMIT ?", (artifact_id, limit))
        rows = cursor.fetchall()
        logs = []
        col_names = [desc[0] for desc in cursor.description]
        for row in rows:
            logs.append(dict(zip(col_names, row)))
        return logs

    def add_module(self, name: str, code: str, description: str = "") -> str:
        module_id = str(uuid.uuid4())
        timestamp = time.time()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO modules (module_id, name, code, created_at, last_updated_at, description) VALUES (?, ?, ?, ?, ?, ?)",
            (module_id, name, code, timestamp, timestamp, description)
        )
        self.conn.commit()
        return module_id

    def update_module(self, name: str, code: str, description: str = None) -> bool:
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
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM modules WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return None
        col_names = [desc[0] for desc in cursor.description]
        return dict(zip(col_names, row))

    def list_modules(self) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM modules ORDER BY name")
        rows = cursor.fetchall()
        modules = []
        col_names = [desc[0] for desc in cursor.description]
        for row in rows:
            modules.append(dict(zip(col_names, row)))
        return modules

    def delete_module(self, name: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM modules WHERE name = ?", (name,))
        self.conn.commit()
        return cursor.rowcount > 0

    def execute_module(self, name: str, globals_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        module = self.get_module(name)
        if not module:
            return {"success": False, "error": f"Module '{name}' not found"}
        try:
            if globals_dict is None:
                globals_dict = globals().copy()
            locals_dict = {}
            start_time = time.time()
            exec(module["code"], globals_dict, locals_dict)
            execution_time = time.time() - start_time
            return {"success": True, "locals": locals_dict, "execution_time": execution_time}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def close(self):
        if self.conn:
            self.conn.close()

# =======================
# Tool Registry
# =======================
class ToolRegistry:
    def __init__(self):
        self.functions: Dict[str, FunctionSpec] = {}
        self.code_repo = CodeRepository(db_path="code_artifacts.db")
        self.jina_client = None
        try:
            self.jina_client = JinaClient()
            console.print("[green]Jina client initialized successfully[/green]")
        except ValueError:
            console.print("[yellow]Warning: JINA_API_KEY not found. Jina tools will not be available.[/yellow]")
            console.print("[yellow]Set the JINA_API_KEY environment variable to enable web search functionality.[/yellow]")
        self._register_default_tools()

    def _register_default_tools(self):
        # Date and time tools
        self.register_function(
            name="get_current_datetime",
            description="Get the current date and time",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Optional timezone (e.g., 'UTC', 'US/Eastern')", "default": "local"}
                },
                "required": []
            },
            function=self._get_current_datetime
        )
        
        # Web search tools
        self.register_function(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            },
            function=self._web_search
        )
        
        # Python code execution
        self.register_function(
            name="execute_python",
            description="Execute Python code and return the result",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "save_artifact": {"type": "boolean", "description": "Whether to save this code as an artifact", "default": False},
                    "artifact_name": {"type": "string", "description": "Name for the artifact if saving", "default": ""},
                    "description": {"type": "string", "description": "Artifact description", "default": ""}
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
                    "name": {"type": "string", "description": "Name for the module (valid Python identifier)"},
                    "code": {"type": "string", "description": "Python code for the module"},
                    "description": {"type": "string", "description": "Module description", "default": ""}
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
                    "name": {"type": "string", "description": "Name of the module"},
                    "args": {"type": "object", "description": "Arguments to pass (if any)", "default": {}}
                },
                "required": ["name"]
            },
            function=self._execute_saved_module
        )
        # List modules
        self.register_function(
            name="list_modules",
            description="List all available Python modules",
            parameters={"type": "object", "properties": {}},
            function=self._list_modules
        )
        # Get a module
        self.register_function(
            name="get_module",
            description="Get a specific Python module by name",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Module name"}},
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
                    "file_path": {"type": "string", "description": "Path to the script"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Command-line arguments", "default": []}
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
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10}
                },
                "required": ["query"]
            },
            function=self._search_code
        )
        # Weather functions
        self.register_function(
            name="get_weather",
            description="Get current weather information for a location",
            parameters={"type": "object", "properties": {"location": {"type": "string", "description": "Location (e.g., city)"}}, "required": ["location"]},
            function=self._get_weather
        )
        self.register_function(
            name="parse_weather_response",
            description="Parse weather data into a readable format",
            parameters={"type": "object", "properties": {"response": {"type": "string", "description": "Weather API response"}}, "required": ["response"]},
            function=self._parse_weather_response
        )
        # System prompt management
        self.register_function(
            name="update_system_prompt",
            description="Update the system prompt for the agent",
            parameters={"type": "object", "properties": {
                "system_prompt": {"type": "string", "description": "New system prompt"},
                "append": {"type": "boolean", "description": "Append rather than replace", "default": False}
            }, "required": ["system_prompt"]},
            function=self._update_system_prompt
        )
        self.register_function(
            name="get_system_prompt",
            description="Retrieve the current system prompt",
            parameters={"type": "object", "properties": {}},
            function=self._get_system_prompt
        )
        self.register_function(
            name="add_reflection_note",
            description="Add a self-reflection note",
            parameters={"type": "object", "properties": {
                "note": {"type": "string", "description": "Reflection content"},
                "category": {"type": "string", "description": "Note category", "default": "general"}
            }, "required": ["note"]},
            function=self._add_reflection_note
        )
        self.register_function(
            name="get_reflection_notes",
            description="Retrieve self-reflection notes",
            parameters={"type": "object", "properties": {"category": {"type": "string", "description": "Category filter", "default": "all"}}},
            function=self._get_reflection_notes
        )
        # Environmental adaptation
        self.register_function(
            name="analyze_environment",
            description="Analyze the current environment and context",
            parameters={"type": "object", "properties": {"aspect": {"type": "string", "description": "Aspect to analyze", "enum": ["system", "user_behavior", "conversation_context", "task_complexity", "all"], "default": "all"}}},
            function=self._analyze_environment
        )
        self.register_function(
            name="adapt_to_environment",
            description="Adapt agent behavior based on environmental analysis",
            parameters={"type": "object", "properties": {
                "adaptation_strategy": {"type": "string", "description": "Adaptation strategy"},
                "reason": {"type": "string", "description": "Reason for adaptation"},
                "system_prompt_update": {"type": "string", "description": "Optional prompt update"}
            }, "required": ["adaptation_strategy", "reason"]},
            function=self._adapt_to_environment
        )
        # File operations (read, write, list_directory)
        self.register_function(
            name="read_file",
            description="Read file contents",
            parameters={"type": "object", "properties": {"path": {"type": "string", "description": "File path"}}, "required": ["path"]},
            function=self._read_file
        )
        self.register_function(
            name="write_file",
            description="Write content to a file",
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content"},
                "append": {"type": "boolean", "description": "Append flag", "default": False}
            }, "required": ["path", "content"]},
            function=self._write_file
        )
        self.register_function(
            name="list_directory",
            description="List files/directories in a directory",
            parameters={"type": "object", "properties": {"path": {"type": "string", "description": "Directory path"}}, "required": ["path"]},
            function=self._list_directory
        )
        # Command execution
        self.register_function(
            name="execute_command",
            description="Execute a shell command",
            parameters={"type": "object", "properties": {"command": {"type": "string", "description": "Shell command"}}, "required": ["command"]},
            function=self._execute_command
        )
        # Function creation
        self.register_function(
            name="create_python_function",
            description="Create a new Python function for later calls",
            parameters={"type": "object", "properties": {
                "name": {"type": "string", "description": "Function name"},
                "description": {"type": "string", "description": "Function description"},
                "parameters_schema": {"type": "object", "description": "JSON Schema for parameters"},
                "source_code": {"type": "string", "description": "Source code"}
            }, "required": ["name", "description", "parameters_schema", "source_code"]},
            function=self._create_python_function
        )
        self.register_function(
            name="list_available_functions",
            description="List all available functions",
            parameters={"type": "object", "properties": {}},
            function=self._list_available_functions
        )
        # Together API models & completions
        self.register_function(
            name="list_together_models",
            description="List models available on Together API",
            parameters={"type": "object", "properties": {"filter": {"type": "string", "description": "Filter", "default": ""}}},
            function=self._list_together_models
        )
        self.register_function(
            name="generate_completion",
            description="Generate a text completion using Together API",
            parameters={"type": "object", "properties": {
                "model": {"type": "string", "description": "Model to use"},
                "prompt": {"type": "string", "description": "Prompt"},
                "max_tokens": {"type": "integer", "description": "Max tokens", "default": 256},
                "temperature": {"type": "number", "description": "Temperature", "default": 0.7},
                "logprobs": {"type": "integer", "description": "Number of logprobs", "default": 0},
                "echo": {"type": "boolean", "description": "Echo prompt tokens", "default": False}
            }, "required": ["model", "prompt"]},
            function=self._generate_completion
        )
        # Create/update assistant
        self.register_function(
            name="create_or_update_assistant",
            description="Create or update an assistant in Together platform",
            parameters={"type": "object", "properties": {
                "assistant_id": {"type": "string", "description": "Assistant ID (if updating)"},
                "name": {"type": "string", "description": "Assistant name"},
                "description": {"type": "string", "description": "Assistant description"},
                "model": {"type": "string", "description": "Model to use"},
                "system_prompt": {"type": "string", "description": "System prompt"}
            }, "required": ["name", "model"]},
            function=self._create_or_update_assistant
        )
        # Create thread
        self.register_function(
            name="create_thread",
            description="Create a new conversation thread",
            parameters={"type": "object", "properties": {"metadata": {"type": "object", "description": "Optional metadata"}}},
            function=self._create_thread
        )
        # Add message to thread
        self.register_function(
            name="add_message_to_thread",
            description="Add a message to an existing thread",
            parameters={"type": "object", "properties": {
                "thread_id": {"type": "string", "description": "Thread ID"},
                "role": {"type": "string", "description": "Sender role", "enum": ["user", "assistant"]},
                "content": {"type": "string", "description": "Message content"}
            }, "required": ["thread_id", "role", "content"]},
            function=self._add_message_to_thread
        )
        # Run assistant on thread
        self.register_function(
            name="run_assistant",
            description="Run an assistant on a thread to generate a response",
            parameters={"type": "object", "properties": {
                "assistant_id": {"type": "string", "description": "Assistant ID"},
                "thread_id": {"type": "string", "description": "Thread ID"}
            }, "required": ["assistant_id", "thread_id"]},
            function=self._run_assistant
        )
        # Jina tools
        # Register the new search, read, and fact_check functions
        self.register_function(
            name="search",
            description="Search the web for information",
            parameters={
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                }, 
                "required": ["query"]
            },
            function=self.search
        )
        
        self.register_function(
            name="read",
            description="Read and extract content from a web page",
            parameters={
                "type": "object", 
                "properties": {
                    "url": {"type": "string", "description": "URL of the web page to read"}
                }, 
                "required": ["url"]
            },
            function=self.read
        )
        
        self.register_function(
            name="fact_check",
            description="Verify a statement or claim for factual accuracy",
            parameters={
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "Statement to fact check"}
                }, 
                "required": ["query"]
            },
            function=self.fact_check
        )
        
        # Keep the legacy functions for backward compatibility
        if self.jina_client:
            self.register_function(
                name="web_search",
                description="Search the web using Jina's search API",
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
            self.register_function(
                name="web_read",
                description="Read web page content using Jina's reader API",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to read content from"
                        }
                    },
                    "required": ["url"]
                },
                function=self._web_read
            )
            self.register_function(
                name="fact_check",
                description="Verify a statement using Jina's fact checking API",
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
        # Planning tools
        self.register_function(
            name="create_planning_session",
            description="Start a new planning session for multi-turn tasks",
            parameters={"type": "object", "properties": {"task": {"type": "string", "description": "Task description"}}, "required": ["task"]},
            function=self._create_planning_session
        )
        self.register_function(
            name="add_plan_step",
            description="Add a planning step with optional tool execution",
            parameters={"type": "object", "properties": {
                "plan": {"type": "string", "description": "Step description"},
                "tool_name": {"type": "string", "description": "Tool name (optional)"},
                "tool_args": {"type": "object", "description": "Tool arguments (optional)"}
            }, "required": ["plan"]},
            function=self._add_plan_step
        )
        self.register_function(
            name="get_planning_status",
            description="Get the status of the current planning session",
            parameters={"type": "object", "properties": {}},
            function=self._get_planning_status
        )
        self.register_function(
            name="complete_planning_session",
            description="Complete the current planning session with a summary",
            parameters={"type": "object", "properties": {
                "summary": {"type": "string", "description": "Session summary"},
                "success": {"type": "boolean", "description": "Was the session successful", "default": True}
            }, "required": ["summary"]},
            function=self._complete_planning_session
        )
        # Knowledge tools
        self.register_function(
            name="extract_urls",
            description="Extract URLs from text",
            parameters={"type": "object", "properties": {"text": {"type": "string", "description": "Text input"}}, "required": ["text"]},
            function=self._extract_urls
        )
        self.register_function(
            name="process_urls",
            description="Process URLs to extract knowledge",
            parameters={"type": "object", "properties": {"urls": {"type": "array", "items": {"type": "string"}, "description": "List of URLs"}}, "required": ["urls"]},
            function=self._process_urls
        )
        self.register_function(
            name="get_knowledge_summary",
            description="Get a summary of the knowledge base",
            parameters={"type": "object", "properties": {}},
            function=self._get_knowledge_summary
        )
        self.register_function(
            name="search_knowledge",
            description="Search the knowledge base for relevant information",
            parameters={"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]},
            function=self._search_knowledge
        )
        self.register_function(
            name="monitor_task",
            description="Monitor the status of a background task",
            parameters={"type": "object", "properties": {"task_id": {"type": "string", "description": "Task ID"}}, "required": ["task_id"]},
            function=self._monitor_task
        )

    def _categorize_function(self, name: str) -> str:
        """Categorize functions based on their names or purposes"""
        web_tools = ["web_search", "web_read", "fact_check", "extract_urls", "process_urls"]
        code_tools = ["execute_python", "save_module", "execute_module", "list_modules", "get_module", 
                     "run_script", "search_code", "create_python_function"]
        file_tools = ["read_file", "write_file", "list_directory"]
        system_tools = ["execute_command", "update_system_prompt", "get_system_prompt", 
                       "add_reflection_note", "get_reflection_notes"]
        planning_tools = ["create_planning_session", "add_plan_step", "get_planning_status", 
                         "complete_planning_session"]
        knowledge_tools = ["get_knowledge_summary", "search_knowledge", "monitor_task"]
        weather_tools = ["get_weather", "parse_weather_response"]
        together_tools = ["list_together_models", "generate_completion", "create_or_update_assistant",
                         "create_thread", "add_message_to_thread", "run_assistant"]
        
        if name in web_tools:
            return "Web & Search"
        elif name in code_tools:
            return "Code & Development"
        elif name in file_tools:
            return "File Operations"
        elif name in system_tools:
            return "System & Configuration"
        elif name in planning_tools:
            return "Planning & Task Management"
        elif name in knowledge_tools:
            return "Knowledge Management"
        elif name in weather_tools:
            return "Weather"
        elif name in together_tools:
            return "Together API"
        elif "analyze" in name or "adapt" in name:
            return "Environment & Adaptation"
        else:
            return "Miscellaneous"
    
    def register_function(self, name: str, description: str, parameters: Dict[str, Any], function: Callable, source_code: Optional[str] = None):
        if name in self.functions:
            console.print(f"[yellow]Warning: Overwriting existing function '{name}'[/yellow]")
        self.functions[name] = FunctionSpec(name=name, description=description, parameters=parameters, function=function, source_code=source_code)

    def get_openai_tools_format(self) -> List[Dict[str, Any]]:
        tools = []
        for name, spec in self.functions.items():
            tools.append({
                "type": "function",
                "function": {"name": spec.name, "description": spec.description, "parameters": spec.parameters}
            })
        return tools

    def call_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self.functions:
            return {"error": f"Function '{name}' not found in registry"}
        try:
            return self.functions[name].function(**arguments)
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    # Extra helper methods to support parallel execution
    def has_tool(self, name: str) -> bool:
        return name in self.functions

    def get_tool(self, name: str):
        return self.functions[name].function if name in self.functions else None
        
    def get_available_tools(self) -> List[str]:
        """Return a list of all available tool names"""
        return list(self.functions.keys())

    # ===============================
    # Default tool implementations
    # ===============================
    def _read_file(self, path: str) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser()
            if not path.exists():
                return {"error": f"File '{path}' does not exist"}
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            return {"content": content, "size_bytes": path.stat().st_size, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _write_file(self, path: str, content: str, append: bool = False) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as file:
                file.write(content)
            return {"path": str(path), "size_bytes": path.stat().st_size, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _list_directory(self, path: str) -> Dict[str, Any]:
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
            return {"path": str(path), "items": items, "count": len(items), "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _execute_command(self, command: str) -> Dict[str, Any]:
        try:
            process = subprocess.run(command, shell=True, text=True, capture_output=True)
            return {"command": command, "stdout": process.stdout, "stderr": process.stderr, "return_code": process.returncode, "success": process.returncode == 0}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _create_python_function(self, name: str, description: str, parameters_schema: Dict[str, Any], source_code: str) -> Dict[str, Any]:
        try:
            temp_module_path = Path(tempfile.gettempdir()) / f"dynamic_func_{name}_{int(time.time())}.py"
            with open(temp_module_path, 'w', encoding='utf-8') as f:
                f.write(source_code)
            spec = importlib.util.spec_from_file_location(f"dynamic_func_{name}", temp_module_path)
            if spec is None or spec.loader is None:
                return {"error": "Failed to create module specification", "success": False}
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if name not in dir(module):
                return {"error": f"Function '{name}' not found in the provided source code", "success": False}
            function = getattr(module, name)
            if not callable(function):
                return {"error": f"'{name}' is not a callable function", "success": False}
            self.register_function(name=name, description=description, parameters=parameters_schema, function=function, source_code=source_code)
            return {"name": name, "description": description, "success": True, "message": f"Function '{name}' successfully created and registered"}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _list_available_functions(self) -> Dict[str, Any]:
        function_list = []
        for name, spec in self.functions.items():
            # Extract required parameters for easier reference
            required_params = spec.parameters.get("required", [])
            properties = spec.parameters.get("properties", {})
            
            # Create a simplified parameter description
            param_desc = []
            for param_name, param_info in properties.items():
                is_required = param_name in required_params
                param_type = param_info.get("type", "any")
                description = param_info.get("description", "")
                default = f", default: {param_info.get('default')}" if "default" in param_info else ""
                req_marker = "*" if is_required else ""
                param_desc.append(f"{param_name}{req_marker} ({param_type}{default}): {description}")
            
            # Create a simplified function description
            simplified_desc = {
                "name": name,
                "description": spec.description,
                "parameters": param_desc,
                "has_source_code": spec.source_code is not None,
                "category": self._categorize_function(name)
            }
            function_list.append(simplified_desc)
        
        # Group functions by category
        categories = {}
        for func in function_list:
            category = func["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(func)
        
        return {
            "functions": function_list, 
            "count": len(function_list), 
            "categories": categories,
            "success": True
        }

    def _list_together_models(self, filter: str = "") -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            models_data = agent.client.models.list()
            models = []
            for model in models_data.data:
                model_dict = {"id": model.id, "name": model.name, "context_length": model.context_length, "capabilities": model.capabilities}
                if not filter or filter.lower() in model.name.lower():
                    models.append(model_dict)
            return {"models": models, "count": len(models), "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _generate_completion(self, model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.7, logprobs: int = 0, echo: bool = False) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            params = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
            agent_logprobs_enabled = False
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    agent_logprobs_enabled = getattr(agent, 'enable_logprobs', False)
                    break
            if (logprobs > 0 or agent_logprobs_enabled) and logprobs >= 0:
                actual_logprobs = logprobs if logprobs > 0 else 1
                params["logprobs"] = actual_logprobs
                if echo:
                    params["echo"] = echo
            response = agent.client.completions.create(**params)
            result = {
                "model": model,
                "completion": response.choices[0].text,
                "finish_reason": response.choices[0].finish_reason,
                "tokens_used": {"prompt": response.usage.prompt_tokens, "completion": response.usage.completion_tokens, "total": response.usage.total_tokens},
                "success": True
            }
            if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                logprobs_data = response.choices[0].logprobs
                result["logprobs"] = {"tokens": logprobs_data.tokens, "token_logprobs": logprobs_data.token_logprobs}
                if hasattr(logprobs_data, "top_logprobs") and logprobs_data.top_logprobs:
                    result["logprobs"]["top_logprobs"] = logprobs_data.top_logprobs
            if echo and hasattr(response, "prompt"):
                result["prompt_tokens"] = []
                for prompt_item in response.prompt:
                    if hasattr(prompt_item, "logprobs") and prompt_item.logprobs:
                        result["prompt_tokens"].append({
                            "text": prompt_item.text,
                            "tokens": prompt_item.logprobs.tokens,
                            "token_logprobs": prompt_item.logprobs.token_logprobs
                        })
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _create_or_update_assistant(self, name: str, model: str, assistant_id: str = None, description: str = None, system_prompt: str = None) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            if not assistant_id:
                assistant_id = f"asst_{int(time.time())}"
                action = "Created"
            else:
                action = "Updated"
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
            return {"assistant_id": assistant_id, "name": name, "model": model, "description": description or "", "action": action, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _create_thread(self, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            thread_id = f"thread_{int(time.time())}"
            return {"thread_id": thread_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "metadata": metadata or {}, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _add_message_to_thread(self, thread_id: str, role: str, content: str) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            if thread_id.startswith("thread_"):
                message_id = f"msg_{int(time.time())}_{hash(content) % 10000}"
                agent.add_message(role, content)
                return {"message_id": message_id, "thread_id": thread_id, "role": role, "content": content, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "success": True}
            else:
                return {"error": "Invalid thread ID format", "success": False}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_system_prompt(self) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            return {"system_prompt": agent.system_message["content"], "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _add_reflection_note(self, note: str, category: str = "general") -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            reflection_note = {"timestamp": time.time(), "datetime": time.strftime("%Y-%m-%d %H:%M:%S"), "category": category, "note": note}
            agent.reflection_notes.append(reflection_note)
            return {"note_id": len(agent.reflection_notes) - 1, "timestamp": reflection_note["datetime"], "category": category, "note": note, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_reflection_notes(self, category: str = "all") -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            if category.lower() == "all":
                filtered_notes = agent.reflection_notes
            else:
                filtered_notes = [note for note in agent.reflection_notes if note["category"].lower() == category.lower()]
            formatted_notes = []
            for i, note in enumerate(filtered_notes):
                formatted_notes.append({"id": i, "timestamp": note["datetime"], "category": note["category"], "note": note["note"]})
            return {"notes": formatted_notes, "count": len(formatted_notes), "category_filter": category, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _analyze_environment(self, aspect: str = "all") -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            agent.environment_state["last_analysis_time"] = time.time()
            if aspect in ["conversation_context", "all"] and len(agent.conversation_history) > 2:
                user_messages = [msg["content"] for msg in agent.conversation_history if msg.get("role") == "user"]
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
                task_words = ["do", "make", "create", "implement", "build", "fix", "solve", "how", "help"]
                task_count = sum(sum(1 for word in task_words if word.lower() in msg.lower()) for msg in user_messages)
                agent.environment_state["conversation_context"]["task_oriented"] = task_count > len(user_messages) / 2
            if aspect in ["task_complexity", "all"] and len(agent.conversation_history) > 2:
                recent_msgs = [msg["content"] for msg in agent.conversation_history[-min(5, len(agent.conversation_history)):] if msg.get("role") == "user"]
                complexity_indicators = {"high": ["complex", "advanced", "detailed", "comprehensive", "integrate", "optimize", "scale"],
                                         "low": ["simple", "basic", "easy", "quick", "just", "help me", "show me"]}
                high_complexity = sum(sum(1 for word in complexity_indicators["high"] if word.lower() in msg.lower()) for msg in recent_msgs)
                low_complexity = sum(sum(1 for word in complexity_indicators["low"] if word.lower() in msg.lower()) for msg in recent_msgs)
                if high_complexity > low_complexity * 2:
                    agent.environment_state["task_complexity"]["current_level"] = "high"
                elif low_complexity > high_complexity:
                    agent.environment_state["task_complexity"]["current_level"] = "low"
                else:
                    agent.environment_state["task_complexity"]["current_level"] = "medium"
            if aspect == "all":
                return {"environment_state": agent.environment_state, "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S"), "message_count": len([msg for msg in agent.conversation_history if msg.get("role") == "user"]), "success": True}
            else:
                return {aspect: agent.environment_state[aspect], "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S"), "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _adapt_to_environment(self, adaptation_strategy: str, reason: str, system_prompt_update: str = None) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            adaptation = {"timestamp": time.time(), "datetime": time.strftime("%Y-%m-%d %H:%M:%S"), "strategy": adaptation_strategy, "reason": reason}
            agent.environment_state["task_complexity"]["adaptations_made"].append(adaptation)
            self._add_reflection_note(note=f"Adapted using {adaptation_strategy}: {reason}", category="adaptation")
            if system_prompt_update:
                self._update_system_prompt(system_prompt=system_prompt_update, append=True)
                adaptation["system_prompt_updated"] = True
            else:
                adaptation["system_prompt_updated"] = False
            return {"adaptation": adaptation, "current_environment": agent.environment_state, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _update_system_prompt(self, system_prompt: str, append: bool = False) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            old_prompt = agent.system_message["content"]
            new_prompt = old_prompt + "\n\n" + system_prompt if append else system_prompt
            for i, message in enumerate(agent.conversation_history):
                if message.get("role") == "system":
                    agent.conversation_history[i]["content"] = new_prompt
                    agent.system_message["content"] = new_prompt
                    break
            else:
                agent.conversation_history.insert(0, {"role": "system", "content": new_prompt})
                agent.system_message["content"] = new_prompt
            return {"old_system_prompt": old_prompt, "new_system_prompt": new_prompt, "append": append, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_weather(self, location: str, city: str = None) -> Dict[str, Any]:
        # Use city parameter if provided, otherwise use location
        location = city or location
        try:
            import random
            conditions = ["sunny", "partly cloudy", "cloudy", "rainy", "stormy", "snowy", "windy", "foggy"]
            condition = random.choice(conditions)
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
            temp_c = round((temp_f - 32) * 5 / 9, 1)
            humidity = random.randint(30, 90)
            wind_speed = random.randint(0, 20)
            forecast = []
            current_temp = temp_f
            for i in range(5):
                forecast_temp = current_temp + random.randint(-10, 10)
                forecast_condition = random.choice(conditions)
                forecast.append({
                    "day": i + 1,
                    "condition": forecast_condition,
                    "high_f": forecast_temp,
                    "low_f": forecast_temp - random.randint(10, 20),
                    "precipitation_chance": random.randint(0, 100)
                })
            return {"location": location, "current_condition": condition, "temperature_f": temp_f, "temperature_c": temp_c, "humidity": humidity, "wind_speed_mph": wind_speed, "forecast": forecast, "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"), "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _execute_python(self, code: str, save_artifact: bool = False, artifact_name: str = "", description: str = "") -> Dict[str, Any]:
        try:
            import io
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            local_vars = {}
            start_time = time.time()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, globals(), local_vars)
            execution_time = time.time() - start_time
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            result = local_vars.get('_', None)
            try:
                result_str = json.dumps(result) if isinstance(result, (dict, list, tuple, set)) else str(result)
            except:
                result_str = str(result)
            response = {"stdout": stdout, "stderr": stderr, "result": result_str, "execution_time": execution_time, "success": True}
            if save_artifact:
                if not artifact_name:
                    first_line = code.strip().split('\n')[0]
                    if len(first_line) > 30:
                        first_line = first_line[:27] + "..."
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    artifact_name = f"code_{timestamp}_{first_line.replace(' ', '_')}"
                metadata = {"execution_time": execution_time, "has_output": bool(stdout), "has_error": bool(stderr), "has_result": result is not None}
                artifact_id = self.code_repo.add_artifact(name=artifact_name, code=code, description=description, metadata=metadata)
                self.code_repo.log_execution(artifact_id=artifact_id, success=True, stdout=stdout, stderr=stderr, result=result_str, execution_time=execution_time)
                response["artifact_id"] = artifact_id
                response["artifact_name"] = artifact_name
            return response
        except Exception as e:
            error_result = {"error": str(e), "traceback": traceback.format_exc(), "success": False}
            if save_artifact:
                if not artifact_name:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    artifact_name = f"failed_code_{timestamp}"
                metadata = {"error_type": type(e).__name__, "error_message": str(e)}
                artifact_id = self.code_repo.add_artifact(name=artifact_name, code=code, description=description or f"Failed execution: {str(e)}", metadata=metadata)
                self.code_repo.log_execution(artifact_id=artifact_id, success=False, stderr=traceback.format_exc(), result=str(e))
                error_result["artifact_id"] = artifact_id
                error_result["artifact_name"] = artifact_name
            return error_result

    def _save_module(self, name: str, code: str, description: str = "") -> Dict[str, Any]:
        try:
            if not name.isidentifier():
                return {"success": False, "error": f"Invalid module name: '{name}'. Must be a valid Python identifier."}
            existing_module = self.code_repo.get_module(name)
            if existing_module:
                success = self.code_repo.update_module(name, code, description)
                action = "updated"
            else:
                self.code_repo.add_module(name, code, description)
                success = True
                action = "created"
            if not success:
                return {"success": False, "error": f"Failed to {action} module '{name}'"}
            try:
                compile(code, f"<module:{name}>", "exec")
            except Exception as e:
                return {"success": True, "warning": f"Module saved but contains syntax errors: {str(e)}", "module_name": name, "action": action}
            return {"success": True, "module_name": name, "description": description, "action": action}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _execute_saved_module(self, name: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            module_data = self.code_repo.get_module(name)
            if not module_data:
                return {"success": False, "error": f"Module '{name}' not found"}
            globals_dict = globals().copy()
            if args:
                globals_dict.update(args)
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            result = None
            start_time = time.time()
            locals_dict = {}
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(module_data["code"], globals_dict, locals_dict)
                if "main" in locals_dict and callable(locals_dict["main"]):
                    result = locals_dict["main"](**args) if args else locals_dict["main"]()
            execution_time = time.time() - start_time
            try:
                result_str = json.dumps(result) if isinstance(result, (dict, list, tuple, set)) else str(result)
            except:
                result_str = str(result)
            return {"success": True, "module_name": name, "stdout": stdout_capture.getvalue(), "stderr": stderr_capture.getvalue(), "result": result_str, "execution_time": execution_time, "returned_main_function": "main" in locals_dict and callable(locals_dict["main"])}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _list_modules(self) -> Dict[str, Any]:
        try:
            modules = self.code_repo.list_modules()
            formatted_modules = []
            for module in modules:
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
            return {"success": True, "modules": formatted_modules, "count": len(formatted_modules)}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _get_module(self, name: str) -> Dict[str, Any]:
        try:
            module = self.code_repo.get_module(name)
            if not module:
                return {"success": False, "error": f"Module '{name}' not found"}
            formatted_module = {
                "name": module["name"],
                "description": module["description"] or "No description",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(module["created_at"])),
                "last_updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(module["last_updated_at"])),
                "code": module["code"],
                "line_count": len(module["code"].split("\n")),
                "code_size_bytes": len(module["code"].encode("utf-8"))
            }
            return {"success": True, "module": formatted_module}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _run_script(self, file_path: str, args: List[str] = None) -> Dict[str, Any]:
        try:
            path = Path(file_path).expanduser()
            if not path.exists():
                return {"success": False, "error": f"Script file '{file_path}' does not exist"}
            if not path.is_file():
                return {"success": False, "error": f"'{file_path}' is not a file"}
            cmd = [sys.executable, str(path)]
            if args:
                cmd.extend(args)
            start_time = time.time()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            execution_time = time.time() - start_time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            script_name = path.name
            with open(path, "r", encoding="utf-8") as f:
                script_content = f.read()
            artifact_id = self.code_repo.add_artifact(name=f"script_{script_name}_{timestamp}", code=script_content, description=f"Execution of script {script_name}", metadata={"file_path": str(path), "args": args or [], "exit_code": process.returncode, "execution_time": execution_time})
            self.code_repo.log_execution(artifact_id=artifact_id, success=process.returncode == 0, stdout=stdout, stderr=stderr, execution_time=execution_time)
            return {"success": process.returncode == 0, "exit_code": process.returncode, "stdout": stdout, "stderr": stderr, "execution_time": execution_time, "file_path": str(path), "args": args or [], "artifact_id": artifact_id}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _search_code(self, query: str, limit: int = 10) -> Dict[str, Any]:
        try:
            artifacts = self.code_repo.find_artifacts(query, limit)
            formatted_artifacts = []
            for artifact in artifacts:
                code_snippets = []
                lines = artifact.code.split("\n")
                for i, line in enumerate(lines):
                    if query.lower() in line.lower():
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        snippet = "\n".join(lines[start:end])
                        line_number = i + 1
                        code_snippets.append({"line": line_number, "snippet": snippet})
                formatted_artifacts.append({
                    "artifact_id": artifact.artifact_id,
                    "name": artifact.name,
                    "description": artifact.description or "No description",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(artifact.created_at)),
                    "execution_count": artifact.execution_count,
                    "line_count": len(artifact.code.split("\n")),
                    "code_size_bytes": len(artifact.code.encode("utf-8")),
                    "snippets": code_snippets[:3]
                })
            return {"success": True, "query": query, "artifacts": formatted_artifacts, "count": len(formatted_artifacts)}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _parse_weather_response(self, response: str) -> Dict[str, Any]:
        try:
            if isinstance(response, dict):
                weather_data = response
            else:
                try:
                    weather_data = json.loads(response)
                except json.JSONDecodeError:
                    return {"error": "Invalid weather data format", "success": False}
            location = weather_data.get("location", "Unknown location")
            condition = weather_data.get("current_condition", "unknown")
            temp_f = weather_data.get("temperature_f", 0)
            temp_c = weather_data.get("temperature_c", 0)
            humidity = weather_data.get("humidity", 0)
            wind_speed = weather_data.get("wind_speed_mph", 0)
            forecast = weather_data.get("forecast", [])
            summary = f"Current weather in {location}: {condition.capitalize()}, {temp_f}°F ({temp_c}°C), humidity {humidity}%, wind {wind_speed} mph."
            forecast_summary = ""
            if forecast and len(forecast) > 0:
                tomorrow = forecast[0]
                forecast_summary = f"\n\nTomorrow's forecast: {tomorrow.get('condition', 'unknown').capitalize()}, high of {tomorrow.get('high_f', 0)}°F, low of {tomorrow.get('low_f', 0)}°F, {tomorrow.get('precipitation_chance', 0)}% chance of precipitation."
            return {"location": location, "summary": summary + forecast_summary, "condition": condition, "temperature_f": temp_f, "temperature_c": temp_c, "humidity": humidity, "wind_speed": wind_speed, "forecast": forecast, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _run_assistant(self, assistant_id: str, thread_id: str) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            system_prompt = f"You are an assistant (ID: {assistant_id}) helping with this conversation. Please analyze the conversation history and provide a helpful response."
            messages = [{"role": "system", "content": system_prompt}]
            for msg in agent.conversation_history:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            response = agent.client.chat.completions.create(model=agent.model, messages=messages, temperature=0.7, max_tokens=500)
            response_content = response.choices[0].message.content
            agent.add_message("assistant", response_content)
            return {"run_id": f"run_{int(time.time())}", "thread_id": thread_id, "assistant_id": assistant_id, "status": "completed", "response": response_content, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def search(self, query: str) -> Dict[str, Any]:
        """
        Search the web for information using Jina's search API.
        
        Args:
            query: The search query string
            
        Returns:
            Dictionary containing search results or error information
        """
        if not self.jina_client:
            try:
                self.jina_client = JinaClient()
                console.print("[green]Successfully initialized Jina client[/green]")
            except ValueError:
                return {"error": "Jina client not initialized. Please set JINA_API_KEY environment variable.", "success": False}
        
        try:
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            result = loop.run_until_complete(self.jina_client.search(query))
            
            # Process the results to make them more readable
            processed_results = result["results"]
            
            # Extract relevant information if possible
            try:
                # Try to parse as JSON if it looks like JSON
                if processed_results.strip().startswith('{') and processed_results.strip().endswith('}'):
                    processed_results = json.loads(processed_results)
            except:
                # If parsing fails, keep as string
                pass
                
            return {
                "success": True, 
                "query": query, 
                "results": processed_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _web_search(self, query: str) -> Dict[str, Any]:
        """Legacy wrapper for the search function"""
        return self.search(query)

    def read(self, url: str) -> Dict[str, Any]:
        """
        Read and extract content from a web page using Jina's reader API.
        
        Args:
            url: The URL of the web page to read
            
        Returns:
            Dictionary containing the extracted content or error information
        """
        if not self.jina_client:
            try:
                self.jina_client = JinaClient()
                console.print("[green]Successfully initialized Jina client[/green]")
            except ValueError:
                return {"error": "Jina client not initialized. Please set JINA_API_KEY environment variable.", "success": False}
        
        try:
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            result = loop.run_until_complete(self.jina_client.read(url))
            
            # Process the content to make it more readable
            content = result["results"]
            
            # Extract metadata about the page
            metadata = {
                "url": url,
                "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "content_length": len(content)
            }
            
            return {
                "success": True, 
                "url": url, 
                "content": content,
                "metadata": metadata
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _web_read(self, url: str) -> Dict[str, Any]:
        """Legacy wrapper for the read function"""
        return self.read(url)

    def fact_check(self, query: str) -> Dict[str, Any]:
        """
        Verify a statement using Jina's fact checking API.
        
        Args:
            query: The statement to fact check
            
        Returns:
            Dictionary containing fact check results or error information
        """
        if not self.jina_client:
            try:
                self.jina_client = JinaClient()
                console.print("[green]Successfully initialized Jina client[/green]")
            except ValueError:
                return {"error": "Jina client not initialized. Please set JINA_API_KEY environment variable.", "success": False}
        
        try:
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            result = loop.run_until_complete(self.jina_client.fact_check(query))
            
            # Process the result to provide a more structured response
            return {
                "success": True, 
                "query": query, 
                "fact_check_result": result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _fact_check(self, query: str) -> Dict[str, Any]:
        """Legacy wrapper for the fact_check function"""
        return self.fact_check(query)

    def _create_planning_session(self, task: str) -> Dict[str, Any]:
        try:
            session_id = str(uuid.uuid4())
            created_at = time.time()
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if agent:
                session = PlanningSession(session_id=session_id, task=task, created_at=created_at)
                agent.planning_session = session
                return {"success": True, "session_id": session_id, "task": task, "message": "Planning session created successfully"}
            else:
                return {"error": "Could not access TogetherAgent instance", "success": False}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _add_plan_step(self, plan: str, tool_name: str = None, tool_args: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent or not agent.planning_session:
                return {"error": "No active planning session", "success": False}
            step = {"step_id": len(agent.planning_session.steps) + 1, "timestamp": time.time(), "plan": plan}
            if tool_name and tool_name in self.functions:
                step["tool"] = {"name": tool_name, "arguments": tool_args or {}}
                if tool_args:
                    result = self.call_function(tool_name, tool_args)
                    step["result"] = result
            agent.planning_session.steps.append(step)
            return {"success": True, "step_id": step["step_id"], "message": "Plan step added successfully"}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_planning_status(self) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent or not agent.planning_session:
                return {"error": "No active planning session", "success": False}
            return {"success": True, "session_id": agent.planning_session.session_id, "task": agent.planning_session.task, "steps_count": len(agent.planning_session.steps), "steps": agent.planning_session.steps, "state": agent.planning_session.state, "active": agent.planning_session.active, "completion_status": agent.planning_session.completion_status, "created_at": agent.planning_session.created_at, "elapsed_time": time.time() - agent.planning_session.created_at}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _complete_planning_session(self, summary: str, success: bool = True) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent or not agent.planning_session:
                return {"error": "No active planning session", "success": False}
            agent.planning_session.active = False
            agent.planning_session.completion_status = "completed" if success else "failed"
            agent.planning_session.steps.append({"step_id": len(agent.planning_session.steps) + 1, "timestamp": time.time(), "summary": summary, "success": success})
            if not hasattr(agent, "completed_planning_sessions"):
                agent.completed_planning_sessions = []
            agent.completed_planning_sessions.append(agent.planning_session)
            completed_session = agent.planning_session
            agent.planning_session = None
            return {"success": True, "session_id": completed_session.session_id, "task": completed_session.task, "steps_count": len(completed_session.steps), "completion_status": completed_session.completion_status, "elapsed_time": time.time() - completed_session.created_at, "summary": summary}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _extract_urls(self, text: str) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            extraction = agent.task_processor.extract_urls(text)
            return {"success": True, "urls": extraction.urls, "count": len(extraction.urls), "source": extraction.source, "timestamp": extraction.timestamp}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _process_urls(self, urls: List[str]) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            new_urls_count = agent.task_processor.add_urls_to_process(urls)
            return {"success": True, "urls_added": new_urls_count, "total_urls_pending": agent.task_processor.task_queue.qsize(), "total_urls_processed": len(agent.task_processor.processed_urls), "message": f"Added {new_urls_count} new URLs to the processing queue"}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_knowledge_summary(self) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            summary = agent.task_processor.get_knowledge_summary()
            summary["success"] = True
            return summary
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _search_knowledge(self, query: str) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            results = agent.task_processor.search_knowledge(query)
            return {"success": True, "query": query, "results_count": len(results), "results": results}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_current_datetime(self, timezone: str = "local") -> Dict[str, Any]:
        """Get the current date and time, optionally in a specific timezone."""
        try:
            import datetime
            import pytz
            from zoneinfo import ZoneInfo, available_timezones
            
            now = datetime.datetime.now()
            local_time = now
            
            if timezone and timezone.lower() != "local":
                try:
                    # Try using ZoneInfo (Python 3.9+)
                    local_time = now.astimezone(ZoneInfo(timezone))
                except (ImportError, KeyError):
                    try:
                        # Fall back to pytz
                        tz = pytz.timezone(timezone)
                        local_time = now.astimezone(tz)
                    except (pytz.exceptions.UnknownTimeZoneError, ImportError):
                        return {
                            "error": f"Unknown timezone: {timezone}",
                            "available_timezones": "Use standard timezone names like 'UTC', 'US/Eastern', 'Europe/London'",
                            "success": False
                        }
            
            result = {
                "current_datetime": local_time.strftime("%Y-%m-%d %H:%M:%S"),
                "date": local_time.strftime("%Y-%m-%d"),
                "time": local_time.strftime("%H:%M:%S"),
                "timezone": timezone if timezone != "local" else "local system timezone",
                "timestamp": time.time(),
                "iso_format": local_time.isoformat(),
                "success": True
            }
            
            # Add day of week, month name, etc.
            result["day_of_week"] = local_time.strftime("%A")
            result["month"] = local_time.strftime("%B")
            result["year"] = local_time.year
            result["day"] = local_time.day
            
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _monitor_task(self, task_id: str) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            result = agent.task_processor.get_result(task_id)
            result["task_id"] = task_id
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

# =======================
# Helper Functions
# =======================
def extract_python_code(text: str) -> list:
    pattern = r'<\|python_start\|>(.*?)(?:<\|python_end\|>|<\|python_end)'
    return [match.strip() for match in re.findall(pattern, text, re.DOTALL)]

def parse_function_calls(text: str) -> List[Dict[str, Any]]:
    """
    Advanced function call parser that handles multiple formats and provides better error recovery.
    Supports:
    1. OpenAI-style JSON format: {"name": "func_name", "arguments": {...}}
    2. Bracket format: [func_name(arg1=val1, arg2=val2)]
    3. Function tag format: <function=func_name>{"arg1": "val1"}</function>
    4. Python code block format: <|python_start|><function=func_name>...</|python_end|>
    5. Tool calls format: {"type": "function", "function": {"name": "func_name", "arguments": "{...}"}}
    """
    function_calls = []
    
    # Track if we found any potential function calls for better error reporting
    potential_function_call_found = False
    
    # Try to find OpenAI-style tool calls first (most reliable format)
    tool_call_pattern = r'\{"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*"(.+?)"\s*\}\s*\}'
    tool_matches = re.findall(tool_call_pattern, text.replace('\n', ' '))
    for func_name, args_str in tool_matches:
        potential_function_call_found = True
        try:
            # Handle escaped JSON in the arguments string
            args_str = args_str.replace('\\"', '"').replace('\\\\', '\\')
            args = json.loads(args_str)
            function_calls.append({"name": func_name, "arguments": args})
            continue
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Could not parse tool call arguments for {func_name}, trying fallback methods[/yellow]")
    
    # Try to find JSON-formatted function calls
    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    potential_matches = re.findall(json_pattern, text)
    for potential_json in potential_matches:
        try:
            data = json.loads(potential_json)
            # Handle direct function call format
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                potential_function_call_found = True
                function_calls.append({"name": data["name"], "arguments": data["arguments"]})
                continue
                
            # Handle OpenAI-style tool calls format
            if isinstance(data, dict) and data.get("type") == "function" and "function" in data:
                potential_function_call_found = True
                func_data = data["function"]
                func_name = func_data.get("name")
                if func_name:
                    # Arguments might be a JSON string or already parsed
                    args_data = func_data.get("arguments", {})
                    if isinstance(args_data, str):
                        try:
                            args = json.loads(args_data)
                        except json.JSONDecodeError:
                            args = {"raw_arguments": args_data}
                    else:
                        args = args_data
                    function_calls.append({"name": func_name, "arguments": args})
                    continue
        except json.JSONDecodeError:
            pass
    
    # If we already found function calls in the preferred formats, return them
    if function_calls:
        return function_calls
    
    # Try to find function calls in the format [function_name(arg1=val1, arg2=val2)]
    pattern = r'\[(\w+)\((.*?)\)\]'
    matches = re.findall(pattern, text)
    for match in matches:
        potential_function_call_found = True
        func_name, args_str = match
        args = {}
        try:
            args_str = args_str.strip()
            if args_str.startswith("{") and args_str.endswith("}"):
                args = json.loads(args_str)
            else:
                # Handle both key=value and key="value with spaces"
                arg_pairs = re.findall(r'(\w+)=("[^"]*"|\'[^\']*\'|\S+)', args_str)
                for arg_name, arg_value in arg_pairs:
                    if (arg_value.startswith('"') and arg_value.endswith('"')) or (arg_value.startswith("'") and arg_value.endswith("'")):
                        arg_value = arg_value[1:-1]
                    args[arg_name] = arg_value
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Error parsing arguments for {func_name}: {args_str}[/yellow]")
            # Still add with empty args for recovery
            args = {}
        function_calls.append({"name": func_name, "arguments": args})
    
    # Try to find function calls in the format <function=function_name>{"arg1": "val1"}</function>
    function_pattern = r'<function=(\w+)>(.*?)</function>'
    fn_matches = re.findall(function_pattern, text)
    for m in fn_matches:
        potential_function_call_found = True
        func_name = m[0]
        args_str = m[1].strip()
        try:
            # If args_str is empty or not valid JSON, use empty dict
            if not args_str or args_str == "{}":
                args = {}
            else:
                args = json.loads(args_str)
            function_calls.append({"name": func_name, "arguments": args})
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Error parsing arguments for {func_name}: {m[1]}[/yellow]")
            # Still add the function call with empty arguments as a fallback
            function_calls.append({"name": func_name, "arguments": {}})
    
    # Special case for <|python_start|><function=X>...<|python_end pattern
    python_function_pattern = r'<\|python_start\|><function=(\w+)>'
    python_fn_matches = re.findall(python_function_pattern, text)
    for func_name in python_fn_matches:
        potential_function_call_found = True
        function_calls.append({"name": func_name, "arguments": {}})
    
    # Special case for <|python_start|>{"type": "function", "name": "X", "parameters": {}}
    python_json_pattern = r'<\|python_start\|>(\{.*?\})<\|python_end'
    python_json_matches = re.findall(python_json_pattern, text, re.DOTALL)
    for json_str in python_json_matches:
        potential_function_call_found = True
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "name" in data:
                # Handle both "parameters" and "arguments" keys
                args = data.get("parameters", data.get("arguments", {}))
                function_calls.append({"name": data["name"], "arguments": args})
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Could not parse JSON in Python block: {json_str[:50]}...[/yellow]")
    
    # Special case for "list_functions" or similar common function names if no other matches
    if not function_calls and "list_functions" in text.lower():
        function_calls.append({"name": "list_available_functions", "arguments": {}})
    elif not function_calls and "list tools" in text.lower():
        function_calls.append({"name": "list_available_functions", "arguments": {}})
    
    # If we found potential function calls but couldn't parse any, log a warning
    if potential_function_call_found and not function_calls:
        console.print("[red]Warning: Detected potential function calls but failed to parse them[/red]")
        console.print(f"[dim]Text snippet: {text[:100]}...[/dim]")
    
    return function_calls

def get_structured_function_call(messages: List[Dict[str, str]], tools: List[Dict[str, Any]], model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo") -> List[Dict[str, Any]]:
    from pydantic import BaseModel, Field
    class FunctionCall(BaseModel):
        name: str = Field(description="Name of the function to call")
        arguments: Dict[str, Any] = Field(description="Arguments for the function call")
    class ParallelFunctionCalls(BaseModel):
        calls: List[FunctionCall] = Field(description="List of function calls to execute in parallel")
    system_prefix = "You must respond with a valid JSON object containing one or more function calls. "
    system_found = False
    for msg in messages:
        if msg["role"] == "system":
            msg["content"] = system_prefix + msg["content"]
            system_found = True
            break
    if not system_found:
        messages.insert(0, {"role": "system", "content": system_prefix + "Use the available tools to respond to the user's request."})
    from together import Together
    together = Together()
    response = together.chat.completions.create(
        messages=messages,
        model=model,
        response_format={"type": "json_object", "schema": ParallelFunctionCalls.model_json_schema()},
        tools=tools
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        if "calls" in data and isinstance(data["calls"], list):
            return [{"name": call["name"], "arguments": call["arguments"]} for call in data["calls"] if "name" in call and "arguments" in call]
        elif "name" in data and "arguments" in data:
            return [{"name": data["name"], "arguments": data["arguments"]}]
        return []
    except json.JSONDecodeError:
        console.print("[yellow]Warning: JSON parsing failed, falling back to regex[/yellow]")
        return parse_function_calls(content)

def get_parallel_function_calls(messages: List[Dict[str, str]], tools: List[Dict[str, Any]], model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo") -> List[Dict[str, Any]]:
    from pydantic import BaseModel, Field
    class FunctionCall(BaseModel):
        name: str = Field(description="Name of the function to call")
        arguments: Dict[str, Any] = Field(description="Arguments for the function call")
    class ParallelFunctionCalls(BaseModel):
        calls: List[FunctionCall] = Field(description="List of function calls to execute in parallel")
    system_prefix = "You must respond with a valid JSON object containing multiple function calls to execute in parallel. "
    system_found = False
    for msg in messages:
        if msg["role"] == "system":
            msg["content"] = system_prefix + msg["content"]
            system_found = True
            break
    if not system_found:
        messages.insert(0, {"role": "system", "content": system_prefix + "Identify all operations that can be executed in parallel and return them as a list."})
    from together import Together
    together = Together()
    response = together.chat.completions.create(
        messages=messages,
        model=model,
        response_format={"type": "json_object", "schema": ParallelFunctionCalls.model_json_schema()},
        tools=tools
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        if "calls" in data and isinstance(data["calls"], list):
            return [{"name": call["name"], "arguments": call["arguments"]} for call in data["calls"] if "name" in call and "arguments" in call]
        return []
    except json.JSONDecodeError:
        console.print("[yellow]Warning: JSON parsing for parallel calls failed, falling back to regex[/yellow]")
        return parse_function_calls(content)

def execute_parallel_functions(function_calls: List[Dict[str, Any]], tool_registry):
    task_processor = AsyncTaskProcessor()
    tasks = []
    for func_call in function_calls:
        function_name = func_call.get("name")
        arguments = func_call.get("arguments", {})
        if tool_registry.has_tool(function_name):
            func = tool_registry.get_tool(function_name)
            task_id = task_processor.add_task(func, **arguments)
            tasks.append((task_id, function_name, arguments))
        else:
            console.print(f"[red]Function {function_name} not found in tool registry[/red]")
    results = []
    for task_id, function_name, arguments in tasks:
        while True:
            task_result = task_processor.get_result(task_id)
            if task_result["status"] in ["completed", "failed"]:
                results.append({"function_name": function_name, "arguments": arguments, "status": task_result["status"], "result": task_result.get("result") if task_result["status"] == "completed" else task_result.get("error")})
                break
            time.sleep(0.1)
    task_processor.stop()
    return results

def batch_process_function_calls(agent, user_message, max_batch_size=5):
    all_function_calls = get_parallel_function_calls(
        messages=[{"role": "system", "content": "Identify all operations needed to complete this task. Return them as JSON."},
                  {"role": "user", "content": user_message}],
        tools=agent.tool_registry.get_openai_tools_format(),
        model=agent.model
    )
    if not all_function_calls:
        return agent.chat(user_message)
    console.print(f"[cyan]Identified {len(all_function_calls)} operations to process in batches[/cyan]")
    all_results = []
    for i in range(0, len(all_function_calls), max_batch_size):
        batch = all_function_calls[i:i+max_batch_size]
        console.print(f"[cyan]Processing batch {i//max_batch_size + 1} with {len(batch)} operations[/cyan]")
        batch_results = execute_parallel_functions(batch, agent.tool_registry)
        all_results.extend(batch_results)
        result_message = {"role": "function", "content": json.dumps(batch_results, indent=2)}
        agent.conversation_history.append(result_message)
    final_prompt = f"""I've completed all the operations you requested. Here's a summary of the results:

{json.dumps(all_results, indent=2)}

Please provide a final summary of all the work completed."""
    agent.last_user_message = final_prompt
    agent.add_message("user", final_prompt)
    from together import Together
    together = Together()
    response = together.chat.completions.create(messages=agent.conversation_history, model=agent.model, tool_choice="none")
    final_message = response.choices[0].message
    agent.conversation_history.append(final_message.model_dump())
    return final_message.content

# =======================
# Together Agent
# =======================
class TogetherAgent:
    def __init__(self, model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
        self.api_key = os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set TOGETHER_API_KEY environment variable.")
        self.use_mock = "dummy_api_key_for_testing" in self.api_key
        if not self.use_mock:
            self.client = Together(api_key=self.api_key)
        else:
            from types import SimpleNamespace
            def mock_completion(**kwargs):
                if "python" in kwargs.get("prompt", "").lower() or (isinstance(kwargs.get("messages", []), list) and any("python" in str(m.get("content", "")).lower() for m in kwargs.get("messages", []))):
                    mock_text = """Here's Python code to perform the requested task:

<|python_start|>
print("Hello, world!")
<|python_end|>"""
                    if "messages" in kwargs:
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=mock_text, model_dump=lambda: {"role": "assistant", "content": mock_text}))])
                    else:
                        return SimpleNamespace(choices=[SimpleNamespace(text=mock_text)])
                else:
                    mock_text = "I'm a mock response from the agent."
                    if "messages" in kwargs:
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=mock_text, model_dump=lambda: {"role": "assistant", "content": mock_text}))])
                    else:
                        return SimpleNamespace(choices=[SimpleNamespace(text=mock_text)])
            self.client = SimpleNamespace(
                completions=SimpleNamespace(create=mock_completion),
                chat=SimpleNamespace(completions=SimpleNamespace(create=mock_completion))
            )
            console.print("[yellow]Running in mock mode with test responses[/yellow]")
        self.model = model
        self.tool_registry = ToolRegistry()
        self.conversation_history = []
        self.reflection_notes = []
        self.enable_logprobs = False
        self.enable_planning = True
        self.planning_session = None
        self.task_processor = AsyncTaskProcessor()
        self.environment_state = {
            "system": self._detect_system_info(),
            "user_behavior": {"message_count": 0, "avg_message_length": 0, "technical_level": "medium", "detected_preferences": []},
            "conversation_context": {"topic_clusters": [], "sentiment": "neutral", "task_oriented": True},
            "task_complexity": {"current_level": "medium", "adaptations_made": []},
            "last_analysis_time": time.time()
        }
        # For Llama-4 models, add a system prompt with explicit instructions
        self.is_llama4 = "llama-4" in self.model.lower()
        if self.is_llama4:
            system_content = (
                "You are an expert conversationalist with the ability to perform multiple function calls in a single turn. "
                "You can call functions using one or more of the following formats:\n"
                "1. [function_name(param1=value1, param2=value2)]\n"
                "2. <function=function_name>{\"param1\": \"value1\", \"param2\": \"value2\"}</function>\n"
                "3. JSON structured output: {\"name\": \"function_name\", \"arguments\": {\"param1\": \"value1\"}}\n\n"
                "Feel free to include multiple function calls in your response. "
                "Always analyze the conversation context and use the best available tools to provide an accurate response.\n\n"
                "<|eot|>"
            )
        else:
            system_content = (
                "You are a helpful AI assistant that can dynamically use tools to accomplish tasks. "
                "You can issue function calls using the defined tools."
            )
        self.system_message = {"role": "system", "content": system_content}
        self.conversation_history.append(self.system_message)

    def _detect_system_info(self):
        return {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "os_name": os.name,
            "cpu_count": os.cpu_count() or "unknown",
            "terminal_size": {"columns": 80, "lines": 24},
            "environment_variables": {"PATH_exists": "PATH" in os.environ, "HOME_exists": "HOME" in os.environ, "LANG": os.environ.get("LANG", "unknown")}
        }

    def update_environment_user_behavior(self, message):
        message_text = ""
        if isinstance(message, list):
            for item in message:
                if item.get('type') == 'text':
                    message_text += item.get('text', '')
        else:
            message_text = message
        self.environment_state["user_behavior"]["message_count"] += 1
        current_count = self.environment_state["user_behavior"]["message_count"]
        current_avg = self.environment_state["user_behavior"]["avg_message_length"]
        new_length = len(message_text)
        new_avg = ((current_avg * (current_count - 1)) + new_length) / current_count
        self.environment_state["user_behavior"]["avg_message_length"] = new_avg
        technical_terms = ["code", "function", "class", "method", "algorithm", "implementation", "api", "database", "query", "json", "xml", "http", "rest", "async", "thread", "concurrency", "memory", "cpu", "processor", "git", "repository", "commit", "merge", "branch"]
        technical_count = sum(1 for term in technical_terms if term.lower() in message_text.lower())
        if technical_count > 5 or (technical_count / max(1, len(message_text.split())) > 0.1):
            self.environment_state["user_behavior"]["technical_level"] = "high"
        elif technical_count > 2:
            self.environment_state["user_behavior"]["technical_level"] = "medium"
        else:
            if current_count > 3:
                self.environment_state["user_behavior"]["technical_level"] = "low"
        has_images = isinstance(message, list) and any(item.get('type') == 'image_url' for item in message)
        if has_images and "multimodal" not in self.environment_state["user_behavior"]["detected_preferences"]:
            self.environment_state["user_behavior"]["detected_preferences"].append("multimodal")

    def _generate_tools_list(self) -> str:
        """Generate a nicely formatted list of available tools"""
        tools_info = self.tool_registry._list_available_functions()
        categories = tools_info.get("categories", {})
        
        response = "# Available Tools\n\n"
        
        for category, functions in sorted(categories.items()):
            response += f"## {category}\n\n"
            
            for func in sorted(functions, key=lambda x: x["name"]):
                name = func["name"]
                description = func["description"]
                params = func["parameters"]
                
                response += f"### {name}\n"
                response += f"{description}\n\n"
                
                if params:
                    response += "**Parameters:**\n"
                    for param in params:
                        response += f"- {param}\n"
                    response += "\n"
            
            response += "\n"
        
        response += "You can use these tools by asking me to perform specific tasks. For example:\n"
        response += "- \"Search the web for the latest news about AI\"\n"
        response += "- \"Get the weather in New York\"\n"
        response += "- \"Create a Python function to calculate Fibonacci numbers\"\n"
        
        return response
    
    def _find_similar_tool(self, name: str) -> Optional[str]:
        """Find a similar tool name in the registry using fuzzy matching."""
        available_tools = self.tool_registry.get_available_tools()
        
        # Check for common aliases
        aliases = {
            "list_functions": "list_available_functions",
            "get_time": "get_current_datetime",
            "get_date": "get_current_datetime",
            "datetime": "get_current_datetime",
            "search": "web_search",
            "weather": "get_weather",
            "execute": "execute_python",
            "run_python": "execute_python",
            "python": "execute_python",
            "read": "read_file",
            "write": "write_file",
            "ls": "list_directory",
            "dir": "list_directory",
        }
        
        if name.lower() in aliases:
            alias = aliases[name.lower()]
            if alias in available_tools:
                return alias
        
        # Try exact match with different casing
        for tool in available_tools:
            if tool.lower() == name.lower():
                return tool
        
        # Try prefix match
        for tool in available_tools:
            if tool.lower().startswith(name.lower()) or name.lower().startswith(tool.lower()):
                return tool
        
        # Try substring match
        for tool in available_tools:
            if name.lower() in tool.lower() or tool.lower() in name.lower():
                return tool
        
        return None
    
    def _get_function_suggestion(self, function_name: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Generate a helpful suggestion for fixing a function call error."""
        if not self.tool_registry.has_tool(function_name):
            similar_tool = self._find_similar_tool(function_name)
            if similar_tool:
                return f"Use '{similar_tool}' instead of '{function_name}'"
            else:
                available = ", ".join(self.tool_registry.get_available_tools()[:5])
                return f"Function '{function_name}' not found. Available functions include: {available}..."
        
        # Get the function spec to check parameters
        function_spec = None
        for name, spec in self.tool_registry.functions.items():
            if name == function_name:
                function_spec = spec
                break
        
        if not function_spec:
            return "Unknown function"
        
        # Check for missing required parameters
        required_params = function_spec.parameters.get("required", [])
        missing_params = [param for param in required_params if param not in arguments]
        
        if missing_params:
            return f"Missing required parameters: {', '.join(missing_params)}"
        
        # Check for invalid parameters
        properties = function_spec.parameters.get("properties", {})
        invalid_params = [param for param in arguments if param not in properties]
        
        if invalid_params:
            valid_params = list(properties.keys())
            return f"Invalid parameters: {', '.join(invalid_params)}. Valid parameters are: {', '.join(valid_params)}"
        
        # If we have an error message, return it
        if "error" in result:
            return f"Error: {result['error']}"
        
        return "Unknown error"
    
    def extract_python_code(self, text: str) -> List[str]:
        return extract_python_code(text)

    def process_tool_calls(self, tool_calls):
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
            self.conversation_history.append({
                "role": "tool",
                "tool_call_id": getattr(tool_call, "id", "n/a"),
                "name": function_name,
                "content": json.dumps(result, indent=2)
            })
            results.append({"function_name": function_name, "arguments": arguments, "result": result})
        return results

    def format_llama4_prompt(self) -> str:
        # Basic formatting: concatenate conversation history
        prompt = "<|begin_of_text|>"
        for msg in self.conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role != "tool":
                prompt += f"<|header_start|>{role}<|header_end|>\n{content}"
                if role != "system" and not content.endswith("<|eot|>"):
                    prompt += "<|eot|>"
            else:
                prompt += f"\nTool output: {content}\n"
        prompt += "<|header_start|>assistant<|header_end|>"
        return prompt

    def process_image_in_message(self, message):
        if isinstance(message, str):
            image_urls = re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)', message)
            if not image_urls:
                return message
            for url in image_urls:
                message = message.replace(url, '')
            text_content = message.strip()
            multimodal = []
            if text_content:
                multimodal.append({"type": "text", "text": text_content})
            for url in image_urls:
                multimodal.append({"type": "image_url", "image_url": {"url": url}})
            return multimodal
        return message

    def add_message(self, role: str, content):
        if hasattr(self, "is_llama4") and self.is_llama4 and role == "user" and isinstance(content, str):
            content = self.process_image_in_message(content)
        if role in ["user", "assistant"] and isinstance(content, str) and hasattr(self, "task_processor"):
            extracted = self.task_processor.extract_urls(content)
            if extracted.urls:
                self.task_processor.add_urls_to_process(extracted.urls)
                console.print(f"[dim]Extracted {len(extracted.urls)} URLs for background processing[/dim]")
        message = {"role": role}
        if isinstance(content, list):
            message["content"] = content
        else:
            if hasattr(self, "is_llama4") and self.is_llama4 and role != "system" and not content.endswith("<|eot|>"):
                content += "<|eot|>"
            message["content"] = content
            if role == "user":
                self.update_environment_user_behavior(content)
        self.conversation_history.append(message)

    def generate_response(self, user_input):
        # Store last user message and add to conversation history
        self.last_user_message = user_input
        self.add_message("user", user_input)
        
        # Special handling for tool listing requests
        if isinstance(user_input, str) and user_input.lower().strip() in [
            "list tools", "list your tools", "what tools do you have", 
            "show tools", "show available tools", "what can you do",
            "list functions", "list available functions", "what functions do you have"
        ]:
            return self._generate_tools_list()
            
        tools = self.tool_registry.get_openai_tools_format()
        # Periodic environment analysis
        user_msg_count = len([msg for msg in self.conversation_history if msg.get("role") == "user"])
        if user_msg_count == 1 or user_msg_count % 3 == 0:
            self.tool_registry._analyze_environment(aspect="all")
            env = self.environment_state
            if env["user_behavior"]["technical_level"] == "high" and env["task_complexity"]["current_level"] == "high":
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="increase_technical_detail",
                    reason="High technical level and complex task",
                    system_prompt_update="Provide detailed technical explanations and comprehensive code examples. Multiple function calls may be issued within one turn."
                )
            elif env["user_behavior"]["technical_level"] == "low" and env["conversation_context"]["sentiment"] == "negative":
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="simplify_explanations",
                    reason="User struggling with technical content",
                    system_prompt_update="Simplify explanations and offer step-by-step guidance."
                )
            elif env["conversation_context"]["task_oriented"] and env["task_complexity"]["current_level"] == "medium":
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="focus_on_practical_solutions",
                    reason="Task-focused user with moderate complexity",
                    system_prompt_update="Focus on practical solutions and step-by-step procedures."
                )
            elif "multimodal" in env["user_behavior"]["detected_preferences"]:
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="optimize_multimodal",
                    reason="User uses multimodal features",
                    system_prompt_update="Pay special attention to visual content; provide detailed analysis of images."
                )
        while True:
            try:
                if self.model.startswith("meta-llama/Llama-4"):
                    has_multimodal = any(isinstance(msg.get("content"), list) for msg in self.conversation_history)
                    if has_multimodal:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=self.conversation_history,
                            tools=tools,
                            tool_choice="auto",
                            max_tokens=1024
                        )
                        assistant_message = response.choices[0].message
                        self.conversation_history.append(assistant_message.model_dump())
                        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                            results = self.process_tool_calls(assistant_message.tool_calls)
                            for res in results:
                                console.print(f"[green]Result from {res['function_name']}:[/green]")
                                console.print(json.dumps(res['result'], indent=2))
                            final_response = self.client.chat.completions.create(
                                model=self.model,
                                messages=self.conversation_history,
                                tools=tools,
                                tool_choice="none"
                            )
                            final_message = final_response.choices[0].message
                            self.conversation_history.append(final_message.model_dump())
                            return final_message.content
                        return assistant_message.content
                    else:
                        formatted_prompt = self.format_llama4_prompt()
                        params = {"model": self.model, "prompt": formatted_prompt, "max_tokens": 1024, "stop": ["<|eot|>"]}
                        if self.is_llama4 and self.enable_logprobs:
                            params["logprobs"] = 1
                        response = self.client.completions.create(**params)
                        assistant_content = response.choices[0].text + "<|eot|>"
                        assistant_message = {"role": "assistant", "content": assistant_content}
                        if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                            logprobs_data = response.choices[0].logprobs
                            assistant_message["logprobs"] = {"tokens": logprobs_data.tokens, "token_logprobs": logprobs_data.token_logprobs}
                            if len(logprobs_data.token_logprobs) > 0:
                                avg_logprob = sum(lp for lp in logprobs_data.token_logprobs if lp is not None) / len(logprobs_data.token_logprobs)
                                assistant_message["avg_logprob"] = avg_logprob
                        self.conversation_history.append(assistant_message)
                        # Check for potential function calls using a more comprehensive detection
                        has_function_call = (
                            ("[" in assistant_content and "(" in assistant_content and ")" in assistant_content) or 
                            "<function=" in assistant_content or 
                            "\"type\": \"function\"" in assistant_content or
                            "{\"name\":" in assistant_content or
                            "<|python_start|>" in assistant_content
                        )
                        
                        if has_function_call:
                            function_calls = parse_function_calls(assistant_content)
                            if function_calls:
                                results = []
                                all_successful = True
                                max_retries = 2  # Allow up to 2 retries for failed function calls
                                
                                # First pass: try to execute all functions
                                for fc in function_calls:
                                    name = fc["name"]
                                    args = fc["arguments"]
                                    
                                    # Try to find similar function if exact match not found
                                    if not self.tool_registry.has_tool(name):
                                        similar_tool = self._find_similar_tool(name)
                                        if similar_tool:
                                            console.print(f"[yellow]Function '{name}' not found, using similar function '{similar_tool}'[/yellow]")
                                            name = similar_tool
                                        else:
                                            error_msg = f"Function '{name}' not found in registry"
                                            console.print(f"[red]{error_msg}[/red]")
                                            result = {"error": error_msg, "success": False, "available_tools": self.tool_registry.get_available_tools()[:5]}
                                            all_successful = False
                                            self.conversation_history.append({"role": "tool", "name": name, "content": json.dumps(result, indent=2)})
                                            results.append({"function_name": name, "arguments": args, "result": result})
                                            continue
                                    
                                    # Execute the function
                                    console.print(f"[cyan]Calling function: [bold]{name}[/bold][/cyan]")
                                    console.print(f"[cyan]With arguments:[/cyan] {json.dumps(args, indent=2)}")
                                    result = self.tool_registry.call_function(name, args)
                                    
                                    # Check if there was an error
                                    if "error" in result:
                                        all_successful = False
                                        console.print(f"[red]Error in {name}: {result.get('error')}[/red]")
                                    else:
                                        console.print(f"[green]Result from {name}:[/green]")
                                        console.print(json.dumps(result, indent=2))
                                    
                                    self.conversation_history.append({"role": "tool", "name": name, "content": json.dumps(result, indent=2)})
                                    results.append({"function_name": name, "arguments": args, "result": result})
                                
                                # If any function call failed, try to recover with increasingly specific guidance
                                retry_count = 0
                                while not all_successful and retry_count < max_retries:
                                    retry_count += 1
                                    
                                    # Create a detailed error report
                                    error_details = []
                                    for res in results:
                                        if "error" in res["result"]:
                                            error_details.append({
                                                "function": res["function_name"],
                                                "arguments": res["arguments"],
                                                "error": res["result"].get("error"),
                                                "suggestion": self._get_function_suggestion(res["function_name"], res["arguments"], res["result"])
                                            })
                                    
                                    # Generate a correction prompt based on retry count
                                    if retry_count == 1:
                                        # First retry: general guidance
                                        correction_prompt = (
                                            f"There were errors in your function calls. Please correct your approach and try again with valid function calls.\n\n"
                                            f"Error details: {json.dumps(error_details, indent=2)}"
                                        )
                                    else:
                                        # Second retry: more specific guidance with available tools
                                        available_tools = self.tool_registry.get_available_tools()
                                        correction_prompt = (
                                            f"Your function calls still have errors. Here are the available tools you can use: "
                                            f"{', '.join(available_tools[:10])}.\n\n"
                                            f"Please use one of these tools with the correct parameters. Error details: {json.dumps(error_details, indent=2)}"
                                        )
                                    
                                    self.add_message("user", correction_prompt)
                                    corrected_response = self.client.completions.create(
                                        model=self.model,
                                        prompt=self.format_llama4_prompt() + "\nAssistant: ",
                                        max_tokens=1024,
                                        stop=["<|eot|>"]
                                    )
                                    corrected_content = corrected_response.choices[0].text + "<|eot|>"
                                    corrected_message = {"role": "assistant", "content": corrected_content}
                                    self.conversation_history.append(corrected_message)
                                    
                                    # Try to parse function calls from the corrected response
                                    corrected_function_calls = parse_function_calls(corrected_content)
                                    if corrected_function_calls:
                                        # Process the corrected function calls
                                        corrected_results = []
                                        all_successful = True  # Reset success flag for this retry
                                        
                                        for fc in corrected_function_calls:
                                            name = fc["name"]
                                            args = fc["arguments"]
                                            
                                            # Try to find similar function if exact match not found
                                            if not self.tool_registry.has_tool(name):
                                                similar_tool = self._find_similar_tool(name)
                                                if similar_tool:
                                                    console.print(f"[yellow]Function '{name}' not found, using similar function '{similar_tool}'[/yellow]")
                                                    name = similar_tool
                                                else:
                                                    error_msg = f"Function '{name}' not found in registry"
                                                    console.print(f"[red]{error_msg}[/red]")
                                                    result = {"error": error_msg, "success": False}
                                                    all_successful = False
                                                    self.conversation_history.append({"role": "tool", "name": name, "content": json.dumps(result, indent=2)})
                                                    corrected_results.append({"function_name": name, "arguments": args, "result": result})
                                                    continue
                                            
                                            console.print(f"[cyan]Retrying function: [bold]{name}[/bold][/cyan]")
                                            console.print(f"[cyan]With arguments:[/cyan] {json.dumps(args, indent=2)}")
                                            result = self.tool_registry.call_function(name, args)
                                            
                                            if "error" in result:
                                                all_successful = False
                                                console.print(f"[red]Error in retry {retry_count} for {name}: {result.get('error')}[/red]")
                                            else:
                                                console.print(f"[green]Result from {name} (retry {retry_count}):[/green]")
                                                console.print(json.dumps(result, indent=2))
                                                
                                            self.conversation_history.append({"role": "tool", "name": name, "content": json.dumps(result, indent=2)})
                                            corrected_results.append({"function_name": name, "arguments": args, "result": result})
                                        
                                        if corrected_results:
                                            # If we got successful results in this retry, use them
                                            # Otherwise, keep the original results to avoid losing information
                                            successful_results = [r for r in corrected_results if "error" not in r["result"]]
                                            if successful_results:
                                                results = successful_results
                                            else:
                                                # Combine original successful results with any new information
                                                successful_original = [r for r in results if "error" not in r["result"]]
                                                results = successful_original + corrected_results
                                    
                                    # If all functions succeeded in this retry, break the loop
                                    if all_successful:
                                        break
                                
                                final_prompt = f"Based on these function results: {json.dumps(results)}, provide a direct answer."
                                final_response = self.client.completions.create(
                                    model=self.model,
                                    prompt=self.format_llama4_prompt() + "\n\nUser: " + final_prompt + "<|eot|>\nAssistant: ",
                                    max_tokens=1024,
                                    stop=["<|eot|>"]
                                )
                                final_content = final_response.choices[0].text + "<|eot|>"
                                final_message = {"role": "assistant", "content": final_content}
                                self.conversation_history.append(final_message)
                                return final_content.rstrip("<|eot|>")
                        response_content = assistant_content.rstrip("<|eot|>")
                        if "<|python_start|>" in response_content and "<|python_end" in response_content:
                            code_blocks = self.extract_python_code(response_content)
                            if code_blocks:
                                for code_block in code_blocks:
                                    if hasattr(self, 'last_user_message') and ("Run the code" in self.last_user_message or "run the code" in self.last_user_message):
                                        console.print("[cyan]Executing Python code:[/cyan]")
                                        import io, datetime
                                        stdout_capture = io.StringIO()
                                        stderr_capture = io.StringIO()
                                        local_vars = {}
                                        exec_globals = globals().copy()
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
                                            console.print(f"[red]Error executing code: {str(e)}[/red]")
                                            console.print(traceback.format_exc())
                        return response_content
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        tools=tools,
                        tool_choice="auto"
                    )
                    assistant_message = response.choices[0].message
                    self.conversation_history.append(assistant_message.model_dump())
                    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                        results = self.process_tool_calls(assistant_message.tool_calls)
                        for res in results:
                            console.print(f"[green]Result from {res['function_name']}:[/green]")
                            console.print(json.dumps(res['result'], indent=2))
                        final_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=self.conversation_history,
                            tools=tools,
                            tool_choice="none"
                        )
                        final_message = final_response.choices[0].message
                        self.conversation_history.append(final_message.model_dump())
                        return final_message.content
                    return assistant_message.content
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
                console.print(traceback.format_exc())
                return f"Error: {str(e)}"

# =======================
# Main CLI Loop
# =======================
def main():
    parser = argparse.ArgumentParser(description="Chat with an AI agent using Together API with dynamic tools")
    parser.add_argument("--model", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", help="Model to use (default: meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)")
    parser.add_argument("--logprobs", action="store_true", help="Enable returning logprobs for confidence analysis")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with mock API responses")
    args = parser.parse_args()
    
    welcome = (
        "[bold blue]Together Agent CLI[/bold blue]\n"
        "Chat with an AI agent that can use and create tools dynamically.\n"
        "The agent has self-adaptive capabilities and supports multiple function calls in one turn.\n"
        "Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to exit."
    )
    console.print(Panel.fit(welcome, title="Welcome"))
    
    if args.test_mode and "TOGETHER_API_KEY" not in os.environ:
        os.environ["TOGETHER_API_KEY"] = "dummy_api_key_for_testing"
    
    try:
        agent = TogetherAgent(model=args.model)
        agent.enable_logprobs = args.logprobs
    except ValueError as e:
        if "API key is required" in str(e) and not args.test_mode:
            console.print("[red]Error: Together API key is required. Set TOGETHER_API_KEY environment variable.[/red]")
            return 1
        raise
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Exiting...[/yellow]")
                break
            image_urls = re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)', user_input)
            if image_urls:
                multimodal = []
                text_content = user_input
                for url in image_urls:
                    text_content = text_content.replace(url, '')
                    console.print(f"[cyan]Including image: {url}[/cyan]")
                text_content = text_content.strip()
                if text_content:
                    multimodal.append({"type": "text", "text": text_content})
                for url in image_urls:
                    multimodal.append({"type": "image_url", "image_url": {"url": url}})
                with console.status("[bold blue]Processing images...[/bold blue]", spinner="dots"):
                    response = agent.generate_response(multimodal)
            else:
                with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
                    response = agent.generate_response(user_input)
            for msg in reversed(agent.conversation_history):
                if isinstance(msg, dict) and msg.get("role") == "assistant" and 'logprobs' in msg and msg.get('avg_logprob') is not None:
                    avg_conf = msg.get('avg_logprob')
                    level = "high" if avg_conf > -1.0 else "medium" if avg_conf > -2.0 else "low"
                    console.print(f"[cyan]Model confidence: {level} (avg logprob: {avg_conf:.2f})[/cyan]")
                    break
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

