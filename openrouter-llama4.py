#!/usr/bin/env python3
"""
Enhanced OpenRouter client for Llama 4 models with streaming, function-calling, and multi-turn support
Featuring a protected kernel architecture with dynamic code extension capabilities
- Protected kernel that cannot be self-modified
- Dynamic function registry with code safety validation
- Redis PubSub for distributed message passing
- SQLite3 database for storing knowledge and code artifacts
- Semantic retrieval over conversations, messages, and knowledge base items
"""

import os
import json
import re
import time
import uuid
import requests
import threading
import logging
import inspect
import asyncio
import ast
import tokenize
import io
import hashlib
import sqlite3
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Union, Tuple, Set
import importlib.util
import importlib.abc
import sys
import traceback
import functools
import types

# Import environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: dotenv not installed. Using environment variables directly.")

# Redis for PubSub and distributed features
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not installed. Distributed features will be disabled.")

# Semantic embedding models
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers or faiss not installed. Semantic search will be disabled.")

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("llama4-agent")

# === CONFIGURATION ===
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL", "meta-llama/llama-4-maverick")  # Or meta-llama/llama-4-scout
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "15"))  # Maximum recursive reasoning steps
ENABLE_AUTO_REASONING = os.getenv("ENABLE_AUTO_REASONING", "True").lower() == "true"
AGENT_PERSISTENCE_PATH = os.getenv("AGENT_PERSISTENCE_PATH", "agent_state.json")
MODULES_PATH = os.getenv("MODULES_PATH", "modules")
ENABLE_HANDOFFS = os.getenv("ENABLE_HANDOFFS", "True").lower() == "true"
HANDOFF_TIMEOUT = int(os.getenv("HANDOFF_TIMEOUT", "3600"))  # 1 hour default timeout for handoffs
DEFAULT_TEMP = float(os.getenv("DEFAULT_TEMP", "0.7"))
CAPABILITY_REGISTRY_PATH = os.getenv("CAPABILITY_REGISTRY_PATH", "capability_registry.json")
ENABLE_DYNAMIC_CAPABILITIES = os.getenv("ENABLE_DYNAMIC_CAPABILITIES", "True").lower() == "true"
SAVE_IMAGES = os.getenv("SAVE_IMAGES", "True").lower() == "true"  # Whether to save images to database
AUTO_REGISTER_APIS = os.getenv("AUTO_REGISTER_APIS", "True").lower() == "true"  # Automatically register default APIs

# === DATABASE CONFIGURATION ===
DB_PATH = os.getenv("DB_PATH", "llama4_knowledge.db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # Depends on the model

# === REDIS CONFIGURATION ===
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_CHANNEL_PREFIX = "llama4_agent:"  # Prefix for Redis channels

# Initialize Redis PubSub
redis_client = None
redis_async_client = None
pubsub = None
pubsub_listener_task = None
pubsub_handlers = {}

if REDIS_AVAILABLE:
    try:
        # Initialize synchronous client for non-async code
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
        
        # Initialize async client for async code
        async def init_async_redis():
            global redis_async_client, pubsub, pubsub_listener_task
            try:
                # Check if redis has asyncio attribute
                if not hasattr(redis, 'asyncio'):
                    logger.error("Redis module does not have asyncio attribute. Install redis with 'pip install redis[hiredis]'")
                    global REDIS_AVAILABLE
                    REDIS_AVAILABLE = False
                    return None
                
                redis_async_client = await redis.asyncio.from_url(
                    f"redis://{REDIS_HOST}:{REDIS_PORT}",
                    password=REDIS_PASSWORD,
                    decode_responses=True
                )
                pubsub = redis_async_client.pubsub()
                
                # Subscribe to agent events channel
                await pubsub.subscribe("agent_events")
                
                # Start pubsub listener task
                pubsub_listener_task = asyncio.create_task(listen_for_pubsub_messages())
                
                logger.info("Async Redis client initialized successfully")
                return redis_async_client
            except Exception as e:
                logger.error(f"Failed to initialize async Redis client: {e}")
                # Global declaration must come before assignment
                global REDIS_AVAILABLE
                REDIS_AVAILABLE = False
                return None
        
        # Function to listen for pubsub messages
        async def listen_for_pubsub_messages():
            try:
                logger.info("Started PubSub listener")
                while True:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message and message["type"] == "message":
                        channel = message["channel"]
                        try:
                            data = json.loads(message["data"])
                            
                            # Handle message based on channel
                            if channel in pubsub_handlers:
                                for handler in pubsub_handlers[channel]:
                                    try:
                                        if asyncio.iscoroutinefunction(handler):
                                            asyncio.create_task(handler(data))
                                        else:
                                            handler(data)
                                    except Exception as handler_error:
                                        logger.error(f"Error in PubSub handler: {handler_error}")
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON in PubSub message: {message['data']}")
                    
                    # Small sleep to prevent CPU spinning
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                logger.info("PubSub listener cancelled")
            except Exception as e:
                logger.error(f"Error in PubSub listener: {e}")
        
        # Function to register a handler for a channel
        def register_pubsub_handler(channel, handler):
            if channel not in pubsub_handlers:
                pubsub_handlers[channel] = []
            pubsub_handlers[channel].append(handler)
            logger.info(f"Registered handler for PubSub channel: {channel}")
        
        # Function to publish a message
        async def publish_message(channel, data):
            if redis_async_client:
                try:
                    message = json.dumps(data) if not isinstance(data, str) else data
                    await redis_async_client.publish(channel, message)
                    return True
                except Exception as e:
                    logger.error(f"Failed to publish to channel {channel}: {e}")
                    return False
            return False
        
        # Initialize async Redis in the background
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if one doesn't exist
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.create_task(init_async_redis())
        
        logger.info("Redis PubSub initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {str(e)}")
        REDIS_AVAILABLE = False
        redis_client = None

# Import multiprocessing and threading
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Initialize process and thread pools for parallel execution
process_pool = None
thread_pool = None

try:
    # Determine optimal number of workers
    cpu_count = multiprocessing.cpu_count()
    process_pool = ProcessPoolExecutor(max_workers=cpu_count)
    thread_pool = ThreadPoolExecutor(max_workers=cpu_count * 4)  # More threads for I/O-bound tasks
    logger.info(f"Initialized process pool with {cpu_count} workers and thread pool with {cpu_count * 4} workers")
except Exception as e:
    logger.error(f"Failed to initialize process/thread pools: {str(e)}")

# Create necessary directories
os.makedirs(MODULES_PATH, exist_ok=True)

###############################################################################
# KERNEL - THIS CODE CANNOT BE MODIFIED AT RUNTIME                            #
###############################################################################

class KernelProtector:
    """
    Protects kernel code from being modified at runtime
    This class creates a barrier between the kernel and code 
    run by the system to prevent self-destruction
    """
    def __init__(self):
        self._kernel_functions = set()
        self._kernel_modules = {}
        self._kernel_attributes = {}
        
    def register_kernel_functions(self, *functions):
        """Register functions as part of the protected kernel"""
        for func in functions:
            if callable(func):
                self._kernel_functions.add(func)
    
    def register_kernel_module(self, module):
        """Register a module as part of the kernel"""
        self._kernel_modules[module.__name__] = module
        
    def register_kernel_attribute(self, name, value):
        """Register an attribute as part of the kernel"""
        self._kernel_attributes[name] = value
        
    def is_kernel_function(self, func):
        """Check if a function is registered as kernel code"""
        return func in self._kernel_functions
        
    def is_kernel_module(self, module_name):
        """Check if a module is registered as part of the kernel"""
        return module_name in self._kernel_modules
        
    def get_kernel_attribute(self, name):
        """Get a kernel attribute safely"""
        return self._kernel_attributes.get(name)
        
    def validate_module_access(self, module_name, attribute=None):
        """
        Validate access to a module and optionally an attribute
        Returns True if access is permitted
        """
        # If it's not a kernel module, access is permitted
        if not self.is_kernel_module(module_name):
            return True
            
        # If no specific attribute requested, deny access to kernel module
        if attribute is None:
            return False
            
        # Check if the attribute is public (doesn't start with _)
        return not attribute.startswith('_')
    
    def install_protection_hooks(self):
        """
        Install import hooks to protect the kernel from being
        modified at runtime by malicious or buggy code
        """
        # Install import hook
        sys.meta_path.insert(0, KernelImportFinder(self))

class KernelImportFinder(importlib.abc.MetaPathFinder):
    """
    Import finder that protects kernel modules from being modified
    """
    def __init__(self, protector):
        self.protector = protector
        
    def find_spec(self, fullname, path, target=None):
        """
        Check if the import is for a kernel module and
        return a protected spec if needed
        """
        if self.protector.is_kernel_module(fullname):
            # Return the original module instead of allowing reimport
            original_module = self.protector.get_kernel_module(fullname)
            
            # Create a read-only spec
            spec = importlib.machinery.ModuleSpec(fullname, None)
            spec.loader = KernelModuleLoader(original_module)
            return spec
            
        # For non-kernel modules, let the normal import system handle it
        return None

class KernelModuleLoader(importlib.abc.Loader):
    """
    Custom loader that returns already loaded kernel modules
    """
    def __init__(self, original_module):
        self.original_module = original_module
        
    def create_module(self, spec):
        """Return the original module"""
        return self.original_module
        
    def exec_module(self, module):
        """No-op since the module is already loaded"""
        pass

class CodeValidator:
    """
    Validates code for safety before execution
    Prevents malicious code from being run
    """
    def __init__(self):
        self.blacklist = {
            # System access
            "os.system", "subprocess.run", "subprocess.call", 
            "os._exit", "sys.exit", "quit", "exit",
            
            # Critical file operations
            "os.remove", "os.unlink", "os.rmdir", "shutil.rmtree",
            
            # Kernel files
            "__main__", "__init__", "kernel", "CodeValidator",
            "KernelProtector", "AgentKernel", "CapabilityRegistry",
            
            # Import related
            "importlib.reload", "__import__", "reload",
            
            # Messing with sys.path
            "sys.path.append", "sys.path.insert", "sys.path.extend",
            
            # Network operations (can be enabled selectively)
            "socket", "gethostbyname", "urlopen", "requests.post",
        }
    
    def is_safe(self, code_str: str) -> Tuple[bool, str]:
        """
        Check if code is safe to execute
        Returns (is_safe, message)
        """
        # Check for dangerous imports and function calls
        for danger in self.blacklist:
            if danger in code_str:
                return False, f"Code contains disallowed operation: {danger}"
        
        # Try to parse the code with ast
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return False, f"Syntax error in code: {str(e)}"
            
        # Recursively walk the AST to check for dangerous patterns
        dangers = []
        for node in ast.walk(tree):
            # Check for exec or eval
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Name) and 
                    node.func.id in ['exec', 'eval']):
                    dangers.append(f"Contains forbidden call to {node.func.id}")
                    
            # Check for dangerous attribute access
            elif isinstance(node, ast.Attribute):
                # Construct full attribute path (e.g., os.path.abspath)
                attr_path = self._get_attribute_path(node)
                if attr_path in self.blacklist:
                    dangers.append(f"Contains forbidden attribute: {attr_path}")
                    
            # Check for __import__ calls
            elif (isinstance(node, ast.Call) and
                  isinstance(node.func, ast.Name) and
                  node.func.id == '__import__'):
                dangers.append("Contains direct call to __import__")
                
        if dangers:
            return False, "; ".join(dangers)
            
        return True, "Code passed safety checks"
    
    def _get_attribute_path(self, node):
        """Reconstruct a full attribute path from an ast.Attribute node"""
        parts = []
        current = node
        
        # Walk up the attribute chain
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
            
        # If we end at a Name node, add it
        if isinstance(current, ast.Name):
            parts.append(current.id)
            
        # Reverse and join with dots
        return ".".join(reversed(parts))

class DynamicCodeStore:
    """
    Stores and manages dynamically loaded code
    Ensures code is stored safely and can be loaded on demand
    """
    def __init__(self, base_path: str = MODULES_PATH):
        self.base_path = base_path
        self.validator = CodeValidator()
        self.modules = {}
        self.module_fingerprints = {}
        
        # Create directory if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)
    
    def save_code(self, name: str, code: str) -> Tuple[bool, str]:
        """
        Save code to a file in the modules directory
        Returns (success, message)
        """
        # Safety check
        is_safe, message = self.validator.is_safe(code)
        if not is_safe:
            return False, message
            
        # Create filename and path
        safe_name = self._sanitize_name(name)
        file_path = os.path.join(self.base_path, f"{safe_name}.py")
        
        # Calculate fingerprint
        fingerprint = hashlib.sha256(code.encode('utf-8')).hexdigest()
        
        try:
            # Write code to file
            with open(file_path, 'w') as f:
                f.write(code)
                
            # Store fingerprint
            self.module_fingerprints[safe_name] = fingerprint
            
            return True, f"Code saved as {safe_name}.py"
        except Exception as e:
            return False, f"Failed to save code: {str(e)}"
            
    def load_module(self, name: str) -> Tuple[bool, Any]:
        """
        Load a module from file
        Returns (success, module or error message)
        """
        # Sanitize name
        safe_name = self._sanitize_name(name)
        file_path = os.path.join(self.base_path, f"{safe_name}.py")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return False, f"Module {safe_name} does not exist"
            
        try:
            # Check if already loaded and up to date
            if safe_name in self.modules:
                # Check if file has changed since last load
                with open(file_path, 'r') as f:
                    code = f.read()
                new_fingerprint = hashlib.sha256(code.encode('utf-8')).hexdigest()
                
                if new_fingerprint == self.module_fingerprints.get(safe_name):
                    # Fingerprint matches, return cached module
                    return True, self.modules[safe_name]
            
            # Load the module
            spec = importlib.util.spec_from_file_location(safe_name, file_path)
            if spec is None:
                return False, f"Failed to create module spec for {safe_name}"
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Update cache
            self.modules[safe_name] = module
            
            # Calculate and store fingerprint
            with open(file_path, 'r') as f:
                code = f.read()
            self.module_fingerprints[safe_name] = hashlib.sha256(code.encode('utf-8')).hexdigest()
            
            return True, module
        except Exception as e:
            return False, f"Failed to load module {safe_name}: {str(e)}"
            
    def list_modules(self) -> List[str]:
        """List all available modules"""
        modules = []
        for file in os.listdir(self.base_path):
            if file.endswith('.py'):
                modules.append(file[:-3])  # Remove .py extension
        return modules
        
    def get_code(self, name: str) -> Tuple[bool, str]:
        """
        Get the source code of a module
        Returns (success, code or error message)
        """
        safe_name = self._sanitize_name(name)
        file_path = os.path.join(self.base_path, f"{safe_name}.py")
        
        if not os.path.exists(file_path):
            return False, f"Module {safe_name} does not exist"
            
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            return True, code
        except Exception as e:
            return False, f"Failed to read module {safe_name}: {str(e)}"
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize module name to prevent path traversal"""
        # Replace non-alphanumeric characters with underscores
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)

class CapabilityRegistry:
    """
    Registry for agent capabilities
    Manages functions that the agent can call
    """
    def __init__(self, kernel_protected_functions=None):
        self.capabilities = {}
        self.usage_stats = {}
        self.kernel_protected = kernel_protected_functions or set()
        
    def register(self, name: str, func: Callable, 
                 description: str = "", 
                 parameters: Dict = None,
                 is_kernel: bool = False) -> bool:
        """
        Register a new capability
        Returns success boolean
        """
        if is_kernel:
            self.kernel_protected.add(name)
            
        self.capabilities[name] = {
            "function": func,
            "description": description,
            "parameters": parameters or {},
            "is_kernel": is_kernel
        }
        
        # Initialize usage stats
        if name not in self.usage_stats:
            self.usage_stats[name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "avg_duration": 0,
                "last_called": None
            }
            
        return True
        
    def unregister(self, name: str) -> bool:
        """
        Unregister a capability
        Returns success boolean
        """
        # Cannot unregister kernel functions
        if name in self.kernel_protected:
            return False
            
        if name in self.capabilities:
            del self.capabilities[name]
            return True
        return False
        
    def get(self, name: str) -> Optional[Dict]:
        """Get a capability by name"""
        return self.capabilities.get(name)
        
    def list_capabilities(self) -> List[Dict]:
        """List all registered capabilities"""
        result = []
        for name, cap in self.capabilities.items():
            result.append({
                "name": name,
                "description": cap["description"],
                "parameters": cap["parameters"],
                "is_kernel": cap["is_kernel"]
            })
        return result
        
    def execute(self, name: str, **kwargs) -> Tuple[Any, bool]:
        """
        Execute a capability and track usage stats
        Returns (result, success)
        """
        if name not in self.capabilities:
            return None, False
            
        capability = self.capabilities[name]
        function = capability["function"]
        
        # Update stats
        self.usage_stats[name]["calls"] += 1
        self.usage_stats[name]["last_called"] = time.time()
        
        start_time = time.time()
        try:
            result = function(**kwargs)
            self.usage_stats[name]["successes"] += 1
            success = True
        except Exception as e:
            result = f"Error: {str(e)}"
            self.usage_stats[name]["failures"] += 1
            success = False
            
        # Update duration stats
        duration = time.time() - start_time
        stats = self.usage_stats[name]
        total_calls = stats["calls"]
        stats["avg_duration"] = ((stats["avg_duration"] * (total_calls - 1)) + duration) / total_calls
        
        return result, success
        
    def get_stats(self) -> Dict:
        """Get usage statistics for all capabilities"""
        return self.usage_stats

class AgentKernel:
    """
    Core kernel for the agent
    Provides protected APIs that cannot be overridden
    Controls access to system resources and databases
    """
    def __init__(self):
        # Version info
        self.version = "1.0.0"
        self.code_validator = CodeValidator()
        self.code_store = DynamicCodeStore(MODULES_PATH)
        self.capability_registry = CapabilityRegistry()
        
        # Initialize kernel protection
        self.protector = KernelProtector()
        
        # Create database connection
        self.db_connection = self._init_database()
        
        # Register kernel functions with protector
        self._register_kernel_methods()
        
    def _register_kernel_methods(self):
        """Register kernel methods with the protector"""
        kernel_functions = [
            self.register_capability,
            self.unregister_capability,
            self.execute_capability,
            self.list_capabilities,
            self.add_code_module,
            self.load_code_module,
            self.list_code_modules,
            self.get_code_module,
            self._init_database,
            self.get_stats,
            self.execute_query
        ]
        
        self.protector.register_kernel_functions(*kernel_functions)
        
    def _init_database(self):
        """Initialize the database connection and create tables"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Create tables for capabilities and code modules
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS capabilities (
                name TEXT PRIMARY KEY,
                description TEXT,
                parameters TEXT,
                is_kernel BOOLEAN,
                usage_count INTEGER DEFAULT 0,
                last_used REAL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS code_modules (
                name TEXT PRIMARY KEY,
                code TEXT,
                fingerprint TEXT,
                created_at REAL,
                updated_at REAL,
                is_active BOOLEAN
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_log (
                id TEXT PRIMARY KEY,
                capability TEXT,
                arguments TEXT,
                result TEXT,
                success BOOLEAN,
                execution_time REAL,
                timestamp REAL
            )
            ''')
            
            conn.commit()
            return conn
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            return None
            
    def register_capability(self, name: str, func: Callable, 
                          description: str = "", 
                          parameters: Dict = None,
                          is_kernel: bool = False) -> bool:
        """Register a capability in the registry"""
        # Store in registry
        success = self.capability_registry.register(
            name, func, description, parameters, is_kernel
        )
        
        # Also store in database
        if success and self.db_connection is not None:
            cursor = self.db_connection.cursor()
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO capabilities (name, description, parameters, is_kernel, usage_count) VALUES (?, ?, ?, ?, 0)",
                    (name, description, json.dumps(parameters or {}), is_kernel)
                )
                self.db_connection.commit()
            except Exception as e:
                logger.error(f"Failed to store capability in database: {str(e)}")
                
        return success
        
    def unregister_capability(self, name: str) -> bool:
        """Unregister a capability from the registry"""
        # Remove from registry
        success = self.capability_registry.unregister(name)
        
        # Also remove from database
        if success and self.db_connection is not None:
            cursor = self.db_connection.cursor()
            try:
                cursor.execute("DELETE FROM capabilities WHERE name = ?", (name,))
                self.db_connection.commit()
            except Exception as e:
                logger.error(f"Failed to remove capability from database: {str(e)}")
                
        return success
        
    def execute_capability(self, name: str, **kwargs) -> Tuple[Any, bool]:
        """Execute a registered capability"""
        # Log execution
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Execute the capability
        result, success = self.capability_registry.execute(name, **kwargs)
        
        # Log to database
        execution_time = time.time() - start_time
        if self.db_connection is not None:
            cursor = self.db_connection.cursor()
            try:
                # Update usage count
                cursor.execute(
                    "UPDATE capabilities SET usage_count = usage_count + 1, last_used = ? WHERE name = ?", 
                    (time.time(), name)
                )
                
                # Log execution
                cursor.execute(
                    "INSERT INTO execution_log (id, capability, arguments, result, success, execution_time, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        execution_id, 
                        name, 
                        json.dumps(kwargs), 
                        json.dumps(result) if isinstance(result, (dict, list)) else str(result), 
                        success, 
                        execution_time, 
                        time.time()
                    )
                )
                self.db_connection.commit()
            except Exception as e:
                logger.error(f"Failed to log capability execution: {str(e)}")
                
        return result, success
        
    def list_capabilities(self) -> List[Dict]:
        """List all registered capabilities"""
        return self.capability_registry.list_capabilities()
        
    def add_code_module(self, name: str, code: str) -> Tuple[bool, str]:
        """
        Add a new code module
        Returns (success, message)
        """
        # Validate and save the code
        success, message = self.code_store.save_code(name, code)
        
        # Store in database
        if success and self.db_connection is not None:
            cursor = self.db_connection.cursor()
            try:
                # Calculate fingerprint
                fingerprint = hashlib.sha256(code.encode('utf-8')).hexdigest()
                now = time.time()
                
                # Check if module exists
                cursor.execute("SELECT name FROM code_modules WHERE name = ?", (name,))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update
                    cursor.execute(
                        "UPDATE code_modules SET code = ?, fingerprint = ?, updated_at = ?, is_active = ? WHERE name = ?",
                        (code, fingerprint, now, True, name)
                    )
                else:
                    # Insert
                    cursor.execute(
                        "INSERT INTO code_modules (name, code, fingerprint, created_at, updated_at, is_active) VALUES (?, ?, ?, ?, ?, ?)",
                        (name, code, fingerprint, now, now, True)
                    )
                self.db_connection.commit()
            except Exception as e:
                logger.error(f"Failed to store code module in database: {str(e)}")
                
        return success, message
        
    def load_code_module(self, name: str) -> Tuple[bool, Any]:
        """
        Load a code module
        Returns (success, module or error message)
        """
        return self.code_store.load_module(name)
        
    def list_code_modules(self) -> List[str]:
        """List all available code modules"""
        return self.code_store.list_modules()
        
    def get_code_module(self, name: str) -> Tuple[bool, str]:
        """
        Get the source code of a module
        Returns (success, code or error message)
        """
        return self.code_store.get_code(name)
        
    def get_stats(self) -> Dict:
        """Get usage statistics for the agent"""
        return {
            "capabilities": self.capability_registry.get_stats(),
            "version": self.version,
            "uptime": time.time()  # TODO: Track actual uptime
        }

    def execute_query(self, query: str, params=None) -> Tuple[bool, List]:
        """
        Execute a SQL query with protection against dangerous operations
        Returns (success, results)
        """
        # Check if query is safe (read-only)
        query_lower = query.lower().strip()
        is_read_only = (
            query_lower.startswith("select") and
            "insert" not in query_lower and
            "update" not in query_lower and
            "delete" not in query_lower and
            "drop" not in query_lower and
            "alter" not in query_lower and
            "create" not in query_lower
        )
        
        if not is_read_only:
            return False, "Only SELECT queries are allowed"
            
        if self.db_connection is None:
            return False, "Database connection not available"
            
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(query, params or [])
            
            # Get column names from cursor description
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Fetch all rows
            rows = cursor.fetchall()
            
            # Convert to list of dicts
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
                
            return True, results
        except Exception as e:
            return False, f"Query execution failed: {str(e)}"

###############################################################################
# LLAMA 4 PROMPT FORMATTING                                                   #
###############################################################################

def format_system_prompt(prompt_text):
    """Format a system prompt for Llama 4"""
    return prompt_text

def format_messages_for_api(messages):
    """Format a list of messages for the API"""
    formatted = []
    for msg in messages:
        if "role" in msg and "content" in msg:
            formatted.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    return formatted

###############################################################################
# DYNAMIC CODE GENERATION AND EXECUTION                                       #
###############################################################################

def create_function(name: str, code: str, description: str = "", params: Dict = None) -> Dict:
    """Create a new function and register it with the agent"""
    if not agent_kernel:
        return {"error": "Agent kernel not initialized"}
        
    # Validate code
    is_safe, message = agent_kernel.code_validator.is_safe(code)
    if not is_safe:
        return {"error": f"Code validation failed: {message}"}
        
    try:
        # Create a module namespace
        module_namespace = {}
        
        # Execute the code in the namespace
        exec(code, module_namespace)
        
        # Find the function in the namespace
        func = module_namespace.get(name)
        if not func or not callable(func):
            return {"error": f"Function '{name}' not found in the provided code"}
            
        # Register the function
        success = agent_kernel.register_capability(name, func, description, params)
        if not success:
            return {"error": f"Failed to register function '{name}'"}
            
        # Also save as code module
        module_name = f"func_{name}"
        success, message = agent_kernel.add_code_module(module_name, code)
        if not success:
            return {"error": f"Failed to save code module: {message}"}
            
        # Define the function in the global environment
        try:
            # Get the global environment
            global_env = globals()
            
            # Also define the function in the global namespace for direct access
            global_env[name] = func
            
            logger.info(f"Function '{name}' defined in global environment")
        except Exception as e:
            logger.warning(f"Could not define function in global environment: {str(e)}")
            
        return {
            "status": "success", 
            "message": f"Function '{name}' created and registered",
            "code_module": module_name
        }
    except Exception as e:
        return {"error": f"Failed to create function: {str(e)}"}

def modify_function(name: str, code: str) -> Dict:
    """Modify an existing function"""
    if not agent_kernel:
        return {"error": "Agent kernel not initialized"}
        
    # Check if function exists
    capabilities = agent_kernel.list_capabilities()
    capability_exists = False
    
    for cap in capabilities:
        if cap["name"] == name:
            if cap.get("is_kernel", False):
                return {"error": f"Cannot modify kernel function '{name}'"}
            capability_exists = True
            break
            
    if not capability_exists:
        return {"error": f"Function '{name}' does not exist"}
        
    # Try to unregister existing function
    success = agent_kernel.unregister_capability(name)
    if not success:
        return {"error": f"Failed to unregister existing function '{name}'"}
        
    # Remove from global namespace if it exists there
    try:
        global_env = globals()
        if name in global_env and callable(global_env[name]):
            del global_env[name]
            logger.info(f"Removed function '{name}' from global environment")
    except Exception as e:
        logger.warning(f"Error removing function from global environment: {str(e)}")
    
    # Create and register the new version
    return create_function(name, code)

def delete_function(name: str) -> Dict:
    """Delete a registered function"""
    if not agent_kernel:
        return {"error": "Agent kernel not initialized"}
        
    # Check if function exists
    capabilities = agent_kernel.list_capabilities()
    capability_exists = False
    is_kernel = False
    
    for cap in capabilities:
        if cap["name"] == name:
            is_kernel = cap.get("is_kernel", False)
            capability_exists = True
            break
            
    if not capability_exists:
        return {"error": f"Function '{name}' does not exist"}
        
    if is_kernel:
        return {"error": f"Cannot delete kernel function '{name}'"}
        
    # Unregister the function
    success = agent_kernel.unregister_capability(name)
    if not success:
        return {"error": f"Failed to unregister function '{name}'"}
    
    # Remove from global namespace if it exists there
    try:
        global_env = globals()
        if name in global_env and callable(global_env[name]):
            del global_env[name]
            logger.info(f"Removed function '{name}' from global environment")
    except Exception as e:
        logger.warning(f"Error removing function from global environment: {str(e)}")
    
    return {"status": "success", "message": f"Function '{name}' deleted"}

###############################################################################
# AGENT IMPLEMENTATION                                                        #
###############################################################################

class Llama4Agent:
    """
    Llama 4 Agent with autonomous reasoning capabilities and distributed processing
    Includes a protected kernel for long-running execution and multi-processing support
    """
    def __init__(self, agent_id=None, api_key=None, model=MODEL, 
                 temperature=DEFAULT_TEMP, max_tokens=2048,
                 system_prompt=None, debug=False, fallback_models=None):
        # Agent identification
        self.agent_id = agent_id or str(uuid.uuid4())
        self.model = model
        
        # Fallback models if primary model fails
        self.fallback_models = fallback_models or [
            "meta-llama/llama-4-scout",
            "anthropic/claude-3-opus",
            "anthropic/claude-3-sonnet",
            "google/gemini-1.5-pro"
        ]
        
        # API configuration
        self.api_key = api_key or OPENROUTER_API_KEY
        self.endpoint = ENDPOINT
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug
        
        # Multi-processing configuration
        self.use_multiprocessing = True
        self.process_pool = process_pool  # Use global process pool
        self.thread_pool = thread_pool    # Use global thread pool
        
        # Advanced connection and retry configuration
        self.max_retries = 5  # Maximum retry attempts
        self.base_retry_delay = 1  # Initial retry delay in seconds
        self.max_retry_delay = 60  # Maximum retry delay in seconds
        self.retry_jitter = 0.2  # Random jitter factor (0.0-1.0) to avoid thundering herd
        self.retry_status_codes = [408, 429, 500, 502, 503, 504]  # Status codes to retry
        
        # Circuit breaker pattern to avoid overloading servers
        self.circuit_breaker_threshold = 5  # Consecutive failures to trip circuit breaker
        self.circuit_breaker_timeout = 90  # Seconds to wait before resetting circuit breaker
        self.circuit_break_time = None  # When circuit breaker was triggered
        self.consecutive_failures = 0  # Count of consecutive failures
        
        # Model fallback configuration
        self.model_fallback_index = 0  # Current fallback model index
        self.model_failures = {}  # Track failures by model
        
        # Request timeout configuration
        self.connect_timeout = 10  # Connection timeout in seconds
        self.read_timeout = 180  # Read timeout in seconds for non-streaming
        self.stream_timeout = 300  # Stream timeout in seconds
        
        # API health tracking
        self.api_health = {
            "last_success": None,
            "success_count": 0,
            "failure_count": 0,
            "last_latency": 0,
            "avg_latency": 0,
            "last_error": None,
            "models_health": {}  # Track health by model
        }
        
        # Distributed processing
        self.distributed_mode = REDIS_AVAILABLE
        self.pubsub_channels = {
            "agent_events": f"agent:{self.agent_id}:events",
            "agent_tasks": f"agent:{self.agent_id}:tasks",
            "agent_results": f"agent:{self.agent_id}:results"
        }
        
        # Conversation state
        self.messages = []
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.conversation_id = str(uuid.uuid4())
        
        # Add system message
        self.messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Setup distributed event handlers if Redis is available
        if REDIS_AVAILABLE and redis_async_client:
            # Register handlers for agent events
            register_pubsub_handler("agent_events", self._handle_agent_event)
            register_pubsub_handler(self.pubsub_channels["agent_results"], self._handle_task_result)
            
            # Subscribe to agent-specific channels
            asyncio.create_task(self._subscribe_to_agent_channels())
        
        # Initialize metadata storage
        self.metadata = {
            "reasoning_steps": [],
            "function_calls": [],
            "current_goal": None,
            "pending_tasks": [],
            "completed_tasks": [],
            "distributed_tasks": {}  # Track distributed tasks
        }
        
        # Initialize capability usage tracking
        self.execution_times = {}
        
        # Locks for thread safety
        self._message_lock = asyncio.Lock()
        self._metadata_lock = asyncio.Lock()
        
        # Register functions
        self._register_agent_functions()
        
    def _register_agent_functions(self):
        """Register agent functions with the kernel"""
        if agent_kernel:
            # Register agent functions
            agent_kernel.register_capability(
                "create_function",
                create_function,
                "Create a new function from code and register it",
                {
                    "name": "Name of the function",
                    "code": "Python code defining the function",
                    "description": "Description of what the function does",
                    "params": "Parameter specification"
                }
            )
            
            agent_kernel.register_capability(
                "modify_function",
                modify_function,
                "Modify an existing function",
                {
                    "name": "Name of the function to modify",
                    "code": "New Python code for the function"
                }
            )
            
            agent_kernel.register_capability(
                "delete_function",
                delete_function,
                "Delete a registered function",
                {
                    "name": "Name of the function to delete"
                }
            )
            
    async def _subscribe_to_agent_channels(self):
        """Subscribe to agent-specific Redis channels"""
        if REDIS_AVAILABLE and pubsub:
            try:
                # Subscribe to agent-specific channels
                for channel in self.pubsub_channels.values():
                    await pubsub.subscribe(channel)
                    logger.info(f"Subscribed to channel: {channel}")
                return True
            except Exception as e:
                logger.error(f"Failed to subscribe to agent channels: {e}")
                return False
        return False
    
    async def _handle_agent_event(self, data):
        """Handle agent events from PubSub"""
        try:
            if isinstance(data, dict):
                event_type = data.get("type")
                
                if event_type == "task_assigned":
                    task_id = data.get("task_id")
                    logger.info(f"Task assigned to agent: {task_id}")
                    
                    # Store task in metadata
                    async with self._metadata_lock:
                        self.metadata["distributed_tasks"][task_id] = {
                            "status": "assigned",
                            "assigned_at": time.time(),
                            "data": data
                        }
                
                elif event_type == "agent_command":
                    command = data.get("command")
                    if command == "reset":
                        logger.info("Received reset command")
                        await self.reset()
        except Exception as e:
            logger.error(f"Error handling agent event: {e}")
    
    async def _handle_task_result(self, data):
        """Handle task results from PubSub"""
        try:
            if isinstance(data, dict):
                task_id = data.get("task_id")
                result = data.get("result")
                
                if task_id and task_id in self.metadata["distributed_tasks"]:
                    # Update task status
                    async with self._metadata_lock:
                        self.metadata["distributed_tasks"][task_id].update({
                            "status": "completed",
                            "completed_at": time.time(),
                            "result": result
                        })
                    
                    logger.info(f"Received result for task {task_id}")
        except Exception as e:
            logger.error(f"Error handling task result: {e}")
    
    async def reset(self):
        """Reset the agent state"""
        async with self._message_lock:
            # Keep system prompt but reset conversation
            self.messages = [{
                "role": "system",
                "content": self.system_prompt
            }]
            
            # Reset metadata
            async with self._metadata_lock:
                self.metadata = {
                    "reasoning_steps": [],
                    "function_calls": [],
                    "current_goal": None,
                    "pending_tasks": [],
                    "completed_tasks": [],
                    "distributed_tasks": {}
                }
            
            # Generate new conversation ID
            self.conversation_id = str(uuid.uuid4())
            
            logger.info(f"Agent {self.agent_id} reset with new conversation ID: {self.conversation_id}")
    
    def _get_default_system_prompt(self):
        """Get the default system prompt"""
        return """You are an advanced Llama 4 agent with autonomous reasoning, distributed processing, and research capabilities. You can:
1. Call functions to gather information and perform actions
2. Reason step by step to solve complex problems
3. Research topics by searching, reading, and analyzing multiple web sources
4. Create, modify, and manage new code abilities
5. Search the web, fact-check statements, and read URLs using Jina.ai
6. Remember context and maintain coherent conversations
7. Distribute complex tasks across multiple processes for parallel execution
8. Use WebRTC for real-time communication when needed

You have access to powerful web research tools:
- web_research: Perform comprehensive research on a topic, gathering and analyzing multiple sources
- jina_search: Search the web for information
- jina_fact_check: Fact check statements against web sources
- jina_read_url: Read and extract key information from any URL

You also have distributed processing capabilities:
- parallel_process: Execute a task across multiple CPU cores for faster processing
- distribute_task: Send a task to be processed by another agent or service
- realtime_communicate: Establish WebRTC connection for real-time data exchange

When users ask factual questions or need information about current topics, you should use your research capabilities to gather accurate information from the web rather than relying solely on your training data.

Always reason step by step when approaching complex problems. Leverage your web research capabilities to find up-to-date and relevant information online.
For computationally intensive tasks, use your parallel processing capabilities.
"""
    
    async def chat(self, user_message, images=None, stream=True, recursive_call=False):
        """
        Send a message to the model and get a response with distributed processing support
        
        This method supports a recursive approach where the agent can decide to make 
        additional calls based on tool results. It also supports distributed processing
        for computationally intensive tasks.
        
        Args:
            user_message: The user's message
            images: Optional list of images to process
            stream: Whether to stream the response
            recursive_call: Whether this is a recursive call from within the agent
            
        Returns:
            Response data from the model, potentially including multiple tool calls and responses
        """
        # Add user message to conversation only if not a recursive call
        message_content = user_message
        
        # Add images if provided and format according to Llama 4 specifications
        if images:
            message_content = {"text": user_message, "images": images}
        
        # Thread-safe message addition
        async with self._message_lock:
            # Only add user message if there is content and this is an initial call, not a recursive one
            if not recursive_call:
                self.messages.append({
                    "role": "user",
                    "content": message_content
                })
            # For recursive calls, if we have a specific follow-up prompt, add it
            elif recursive_call and user_message:
                self.messages.append({
                    "role": "user",
                    "content": message_content
                })
                        
            # Log the current state of messages for debugging
            if self.debug:
                print(f"[DEBUG] Current messages state: {json.dumps(self.messages, indent=2)}")
        
        # Prepare request with current model
        current_model = self.model
        
        # Check if we should use a fallback model due to previous failures
        if self.model_fallback_index > 0 and self.model_fallback_index < len(self.fallback_models):
            current_model = self.fallback_models[self.model_fallback_index - 1]
            logger.info(f"Using fallback model: {current_model}")
        
        request_data = {
            "model": current_model,
            "messages": format_messages_for_api(self.messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
            "tools": self._get_tool_definitions()
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Check circuit breaker status
        if self._check_circuit_breaker():
            return {"error": f"Circuit breaker active. Cooling down for {self.circuit_breaker_timeout} seconds. Try again later."}
            
        # Track request start time for latency monitoring
        start_time = time.time()
        
        # Publish event if in distributed mode
        if self.distributed_mode and redis_async_client:
            try:
                await publish_message(self.pubsub_channels["agent_events"], {
                    "type": "request_started",
                    "agent_id": self.agent_id,
                    "conversation_id": self.conversation_id,
                    "model": current_model,
                    "timestamp": start_time
                })
            except Exception as e:
                logger.error(f"Failed to publish request event: {e}")
        
        # Send request with advanced retry logic
        for attempt in range(self.max_retries):
            try:
                if self.debug:
                    print(f"[DEBUG] Sending request to: {self.endpoint}")
                    print(f"[DEBUG] Request tools: {json.dumps(self._get_tool_definitions())}")
                
                # Add jitter to avoid thundering herd problem
                jitter_factor = 1.0 + (random.random() * self.retry_jitter * 2) - self.retry_jitter
                
                # Set appropriate timeouts based on streaming mode
                timeout = (self.connect_timeout, self.stream_timeout if stream else self.read_timeout)
                
                # Make the API request
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    data=json.dumps(request_data),
                    stream=stream,
                    timeout=timeout
                )
                
                # Handle non-OK responses
                if not response.ok:
                    status_code = response.status_code
                    error_info = f"API request failed: {status_code} - {response.text}"
                    logger.error(error_info)
                    
                    # Update API health metrics
                    self._update_api_health(success=False, error=error_info)
                    self.consecutive_failures += 1
                    
                    # Only retry for specific status codes
                    if status_code in self.retry_status_codes and attempt < self.max_retries - 1:
                        retry_delay = min(
                            self.base_retry_delay * (2 ** attempt) * jitter_factor,
                            self.max_retry_delay
                        )
                        
                        # Add additional delay for rate limiting
                        if status_code == 429:
                            # Try to get retry-after header, or use our calculated delay
                            retry_after = response.headers.get('Retry-After')
                            if retry_after and retry_after.isdigit():
                                retry_delay = max(int(retry_after), retry_delay)
                        
                        logger.info(f"Retrying after {retry_delay:.2f}s (attempt {attempt+1}/{self.max_retries}, status code {status_code})...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Check if we should activate circuit breaker
                        self._check_circuit_breaker_threshold()
                        return {"error": error_info}
                
                # Process the response based on streaming preference
                if stream:
                    try:
                        # Reset failure count on successful connection
                        self.consecutive_failures = 0
                        
                        # Record successful request
                        latency = time.time() - start_time
                        self._update_api_health(success=True, latency=latency)
                        
                        return self._handle_streaming_response(response)
                    except requests.exceptions.ConnectionError as conn_err:
                        # Connection error during streaming
                        error_info = f"Error streaming response: Connection closed"
                        logger.error(error_info)
                        
                        # Update failure metrics
                        self.consecutive_failures += 1
                        self._update_api_health(success=False, error=error_info)
                        
                        # Only retry if not the last attempt
                        if attempt < self.max_retries - 1:
                            retry_delay = min(
                                self.base_retry_delay * (2 ** attempt) * jitter_factor,
                                self.max_retry_delay
                            )
                            logger.info(f"Retrying after {retry_delay:.2f}s (attempt {attempt+1}/{self.max_retries})...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            # Check if we should activate circuit breaker
                            self._check_circuit_breaker_threshold()
                            return {"error": "Connection closed after maximum retries"}
                    except requests.exceptions.ReadTimeout:
                        # Timeout during streaming
                        error_info = "Stream read timeout"
                        logger.error(error_info)
                        
                        # Update failure metrics
                        self.consecutive_failures += 1
                        self._update_api_health(success=False, error=error_info)
                        
                        # Only retry if not the last attempt
                        if attempt < self.max_retries - 1:
                            retry_delay = min(
                                self.base_retry_delay * (2 ** attempt) * jitter_factor,
                                self.max_retry_delay
                            )
                            logger.info(f"Retrying after {retry_delay:.2f}s (attempt {attempt+1}/{self.max_retries})...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            # Check if we should activate circuit breaker
                            self._check_circuit_breaker_threshold()
                            return {"error": "Stream timeout after maximum retries"}
                    except Exception as stream_error:
                        error_info = f"Error processing streaming response: {str(stream_error)}"
                        logger.error(error_info)
                        
                        # Update failure metrics
                        self._update_api_health(success=False, error=error_info)
                        
                        if self.debug:
                            print(f"[DEBUG] Stream processing error: {error_info}")
                            print(f"[DEBUG] {traceback.format_exc()}")
                        return {"error": error_info}
                else:
                    # Non-streaming mode
                    try:
                        result = self._handle_response(response.json())
                        
                        # Reset failure count on successful request
                        self.consecutive_failures = 0
                        
                        # Record successful request
                        latency = time.time() - start_time
                        self._update_api_health(success=True, latency=latency)
                        
                        return result
                    except json.JSONDecodeError:
                        error_info = "Invalid JSON response from API"
                        logger.error(error_info)
                        
                        # Update failure metrics
                        self.consecutive_failures += 1
                        self._update_api_health(success=False, error=error_info)
                        
                        if attempt < self.max_retries - 1:
                            retry_delay = min(
                                self.base_retry_delay * (2 ** attempt) * jitter_factor,
                                self.max_retry_delay
                            )
                            logger.info(f"Retrying after {retry_delay:.2f}s (attempt {attempt+1}/{self.max_retries})...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            # Check if we should activate circuit breaker
                            self._check_circuit_breaker_threshold()
                            return {"error": error_info}
                    
            except requests.exceptions.ConnectionError as conn_err:
                # Connection error during request
                error_info = f"Connection error: {str(conn_err)}"
                logger.error(error_info)
                
                # Update failure metrics
                self.consecutive_failures += 1
                self._update_api_health(success=False, error=error_info)
                
                # Only retry if not the last attempt
                if attempt < self.max_retries - 1:
                    retry_delay = min(
                        self.base_retry_delay * (2 ** attempt) * jitter_factor,
                        self.max_retry_delay
                    )
                    logger.info(f"Retrying after {retry_delay:.2f}s (attempt {attempt+1}/{self.max_retries})...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Check if we should activate circuit breaker
                    self._check_circuit_breaker_threshold()
                    return {"error": "Maximum retries exceeded. Connection failed."}
            except requests.exceptions.Timeout as timeout_err:
                # Timeout error during request
                error_type = "Connection timeout" if "connect" in str(timeout_err).lower() else "Read timeout"
                error_info = f"{error_type}: {str(timeout_err)}"
                logger.error(error_info)
                
                # Update failure metrics
                self.consecutive_failures += 1
                self._update_api_health(success=False, error=error_info)
                
                # Only retry if not the last attempt
                if attempt < self.max_retries - 1:
                    retry_delay = min(
                        self.base_retry_delay * (2 ** attempt) * jitter_factor,
                        self.max_retry_delay
                    )
                    logger.info(f"Retrying after {retry_delay:.2f}s (attempt {attempt+1}/{self.max_retries})...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # Check if we should activate circuit breaker
                    self._check_circuit_breaker_threshold()
                    return {"error": f"Maximum retries exceeded. {error_type}."}
            except Exception as e:
                error_info = f"Error in chat request: {str(e)}"
                logger.error(error_info)
                
                # Update failure metrics
                self.consecutive_failures += 1
                self._update_api_health(success=False, error=error_info)
                
                if self.debug:
                    print(f"[DEBUG] Request error: {error_info}")
                    print(f"[DEBUG] {traceback.format_exc()}")
                return {"error": error_info}
            
    def _check_circuit_breaker(self):
        """Check if the circuit breaker is active and should block requests"""
        if self.circuit_break_time is None:
            return False
            
        # Check if circuit breaker timeout has elapsed
        if time.time() - self.circuit_break_time > self.circuit_breaker_timeout:
            logger.info("Circuit breaker reset after cooling down period")
            self.circuit_break_time = None
            self.consecutive_failures = 0
            return False
            
        return True
        
    def _check_circuit_breaker_threshold(self):
        """Check if we've hit the threshold for activating the circuit breaker"""
        if self.consecutive_failures >= self.circuit_breaker_threshold:
            logger.warning(f"Circuit breaker activated after {self.consecutive_failures} consecutive failures")
            self.circuit_break_time = time.time()
            return True
        return False
        
    def _update_api_health(self, success=True, latency=None, error=None):
        """Update API health tracking metrics"""
        now = time.time()
        
        if success:
            self.api_health["last_success"] = now
            self.api_health["success_count"] += 1
            
            if latency is not None:
                # Update latency metrics
                self.api_health["last_latency"] = latency
                
                # Update average latency with exponential moving average
                if self.api_health["avg_latency"] == 0:
                    self.api_health["avg_latency"] = latency
                else:
                    # Use 0.1 as smoothing factor
                    self.api_health["avg_latency"] = (0.9 * self.api_health["avg_latency"]) + (0.1 * latency)
        else:
            self.api_health["failure_count"] += 1
            self.api_health["last_error"] = error
            
    def _handle_streaming_response(self, response):
        """
        Handle a streaming response from the API
        Uses an elegant recursive approach where the agent controls the entire flow
        """
        content_chunks = []
        function_call_chunks = {}
        last_activity_time = time.time()
        timeout_duration = 30  # seconds with no activity before declaring timeout
        
        # Process the stream with better error handling and timeout detection
        try:
            for chunk in response.iter_lines():
                # Update activity time since we received a chunk (even if empty)
                last_activity_time = time.time()
                
                if chunk:
                    chunk = chunk.decode('utf-8')
                    if chunk.startswith('data: '):
                        chunk = chunk[6:]  # Remove 'data: ' prefix
                        
                        if chunk == "[DONE]":
                            break
                            
                        try:
                            # Store debug info
                            debug_info = f"Raw chunk: {chunk}"
                            
                            chunk_data = json.loads(chunk)
                            debug_info += f"\nParsed chunk: {json.dumps(chunk_data)}"
                            
                            # Debug logging only if debug flag is enabled
                            if self.debug:
                                print(f"[DEBUG] {debug_info}")
                                
                            # Handle potential errors in the chunk
                            if "error" in chunk_data:
                                error_msg = chunk_data.get("error", {}).get("message", "Unknown API error")
                                logger.error(f"API error in chunk: {error_msg}")
                                if self.debug:
                                    print(f"[DEBUG] API error in chunk: {error_msg}")
                                yield {"type": "error", "content": f"API error: {error_msg}"}
                                continue
                                
                            # Check if choices exists and is not empty
                            if "choices" in chunk_data and chunk_data["choices"] and len(chunk_data["choices"]) > 0:
                                choice_data = chunk_data["choices"][0]
                                
                                # Check for finish reason
                                if "finish_reason" in choice_data and choice_data["finish_reason"]:
                                    finish_reason = choice_data["finish_reason"]
                                    # Log non-standard finish reasons
                                    if finish_reason not in ["stop", None]:
                                        logger.info(f"Stream finished with reason: {finish_reason}")
                                        if finish_reason == "length":
                                            yield {"type": "info", "content": "Response exceeded maximum token limit"}
                                        elif finish_reason == "content_filter":
                                            yield {"type": "error", "content": "Content was filtered for safety reasons"}
                                
                                delta = choice_data.get("delta", {})
                                
                                if self.debug:
                                    print(f"[DEBUG] Delta: {json.dumps(delta)}")
                                
                                # Process delta (content or function call)
                                if "content" in delta and delta["content"]:
                                    content = delta["content"]
                                    content_chunks.append(content)
                                    yield {"type": "content", "content": content}
                                
                                # Process tool calls
                                if "tool_calls" in delta and delta["tool_calls"]:
                                    if self.debug:
                                        print(f"\n[DEBUG] Received tool_calls in delta: {json.dumps(delta['tool_calls'])}")
                                    
                                    tool_call = delta["tool_calls"][0]
                                    
                                    # Initialize function call if new
                                    tool_index = tool_call.get("index", 0)
                                    
                                    if tool_index not in function_call_chunks:
                                        call_id = f"call_{uuid.uuid4()}"
                                        if self.debug:
                                            print(f"[DEBUG] Creating new tool call ID: {call_id}")
                                        function_call_chunks[tool_index] = {
                                            "id": call_id,
                                            "type": "function",
                                            "function": {
                                                "name": "",
                                                "arguments": ""
                                            }
                                        }
                                        
                                    # Append function info
                                    if "function" in tool_call:
                                        function_info = tool_call["function"]
                                        if self.debug:
                                            print(f"[DEBUG] Function info: {json.dumps(function_info)}")
                                        
                                        if "name" in function_info:
                                            function_call_chunks[tool_index]["function"]["name"] += function_info["name"]
                                            if self.debug:
                                                print(f"[DEBUG] Updated name: {function_call_chunks[tool_index]['function']['name']}")
                                            
                                        if "arguments" in function_info:
                                            function_call_chunks[tool_index]["function"]["arguments"] += function_info["arguments"]
                                            if self.debug:
                                                print(f"[DEBUG] Updated arguments: {function_call_chunks[tool_index]['function']['arguments']}")
                                        
                                    yield {"type": "tool_call", "tool_call": tool_call}
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON: {chunk}")
                            yield {"type": "error", "content": "Failed to parse response chunk"}
                else:
                    # Check for inactivity timeout
                    current_time = time.time()
                    if current_time - last_activity_time > timeout_duration:
                        logger.warning(f"Stream inactive for {timeout_duration} seconds, closing connection")
                        yield {"type": "error", "content": "Stream timed out due to inactivity"}
                        break
                        
        except requests.exceptions.ConnectionError as e:
            # Re-raise the exception to be caught by the retry mechanism in chat()
            logger.error(f"Connection error during streaming: {str(e)}")
            raise
        except requests.exceptions.ReadTimeout as e:
            logger.error(f"Read timeout during streaming: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in streaming response handler: {str(e)}")
            yield {"type": "error", "content": f"Error in streaming: {str(e)}"}
                        
        # Assemble complete message and add to conversation
        assistant_message = {}
        
        if content_chunks:
            assistant_message["content"] = "".join(content_chunks)
        
        # Process function calls
        if function_call_chunks:
            assistant_message["tool_calls"] = []
            
            for index, call_data in function_call_chunks.items():
                # Execute the function
                function_name = call_data["function"]["name"]
                arguments = call_data["function"]["arguments"]
                
                # Parse arguments
                try:
                    args = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                    
                # Add function call to response
                assistant_message["tool_calls"].append(call_data)
                
                # Execute the function if kernel is available
                if agent_kernel:
                    if self.debug:
                        print(f"[DEBUG] Executing function: {function_name} with args: {json.dumps(args)}")
                    try:
                        result, success = agent_kernel.execute_capability(function_name, **args)
                        if self.debug:
                            print(f"[DEBUG] Function execution result: {json.dumps(result) if isinstance(result, (dict, list)) else result}")
                            print(f"[DEBUG] Execution success: {success}")
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Function execution error: {str(e)}")
                        result = f"Error: {str(e)}"
                        success = False
                    
                    # Create tool call response
                    tool_call_response = {
                        "tool_call_id": call_data["id"],
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                    }
                    
                    # Add tool response to conversation history immediately
                    self.messages.append(tool_call_response)
                    
                    # Also add it to the assistant message for reference
                    if "tool_call_responses" not in assistant_message:
                        assistant_message["tool_call_responses"] = []
                    assistant_message["tool_call_responses"].append(tool_call_response)
                    
                    # Yield the function result to be shown to the user
                    yield {
                        "type": "function_result", 
                        "function_name": function_name,
                        "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                    }
                    
                    # We'll handle weather formatting in the main response instead
                    pass
                
                # Track function call
                self.metadata["function_calls"].append({
                    "name": function_name,
                    "arguments": args,
                    "result": result if 'result' in locals() else None,
                    "timestamp": time.time()
                })
        
        # Add assistant message to conversation history
        self.messages.append({
            "role": "assistant",
            **assistant_message
        })
        
        # If we have function calls, decide how to provide a response
        if function_call_chunks and agent_kernel:
            # Check what functions were called
            function_names = [call["function"]["name"] for call in function_call_chunks.values()]
            
            # Handle special case for weather rather than doing a recursive call
            if "get_weather" in function_names:
                # Look for the weather result in the function calls
                for call_data in function_call_chunks.values():
                    if call_data["function"]["name"] == "get_weather":
                        # Parse the arguments
                        try:
                            args = json.loads(call_data["function"]["arguments"])
                            location = args.get("location", "the requested location")
                        except:
                            location = "the requested location"
                
                # Find the weather result in the responses
                weather_content = None
                for msg in self.messages:
                    if msg.get("role") == "tool" and msg.get("name") == "get_weather":
                        weather_content = msg.get("content")
                        break
                
                if weather_content:
                    try:
                        # Parse the weather data
                        weather_data = json.loads(weather_content)
                        # Yield a nicely formatted response
                        yield {
                            "type": "content",
                            "content": f"\nIn {weather_data.get('location', location)}, the current temperature is {weather_data.get('temperature')}°{weather_data.get('units', 'imperial') == 'imperial' and 'F' or 'C'} with {weather_data.get('conditions', 'unknown conditions')}. The humidity is {weather_data.get('humidity', 'unknown')}%."
                        }
                    except Exception as e:
                        if self.debug:
                            print(f"[DEBUG] Error formatting weather response: {str(e)}")
                        # Just yield the content directly
                        yield {
                            "type": "content",
                            "content": f"\nThe weather information has been retrieved for {location}."
                        }
            else:
                # For other function calls, do a recursive call
                if self.debug:
                    print("[DEBUG] Starting recursive call after function execution")
                try:
                    # Recursive call with improved prompt
                    continuation_prompt = "Based on the function results above, provide a helpful response that addresses the user's original request. Format your response clearly and do not include any JSON or function call syntax."
                    
                    # Create an event loop if needed for the recursive call
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Properly await the coroutine in the recursive call
                    continuation_coroutine = self.chat(continuation_prompt, stream=True, recursive_call=True)
                    continuation_generator = loop.run_until_complete(continuation_coroutine)
                    
                    # Pass through all continuation responses
                    for chunk in continuation_generator:
                        # Add a flag to indicate this is part of a recursive flow
                        chunk["recursive"] = True
                        yield chunk
                        
                    # The final result will be yielded by the continuation
                except Exception as e:
                    if self.debug:
                        print(f"[DEBUG] Error in recursive call: {str(e)}")
                        print(traceback.format_exc())
                    # Yield the error to be shown to the user
                    yield {"type": "error", "content": f"Error in recursive call: {str(e)}"}
        
        # Return final message
        return {"type": "complete", "message": assistant_message}
        
    def _handle_response(self, response_data):
        """
        Handle a complete (non-streaming) response
        Uses a recursive approach where the agent controls the flow
        """
        try:
            # Check for errors in the response
            if "error" in response_data:
                error_info = f"API error: {response_data['error'].get('message', 'Unknown error')}"
                logger.error(error_info)
                return {"error": error_info}
                
            # Check if choices exists and is not empty
            if "choices" not in response_data or not response_data["choices"]:
                error_info = "API response missing choices"
                logger.error(error_info)
                return {"error": error_info}
                
            choice = response_data["choices"][0]
            message = choice.get("message", {})
            
            # Save a copy of the original message
            original_message = {**message}
            
            # Add to conversation
            self.messages.append({
                "role": "assistant",
                **message
            })
            
            # Process any tool calls
            if "tool_calls" in message:
                # Create container for tool responses
                tool_responses = []
                message["tool_responses"] = []
                
                for tool_call in message["tool_calls"]:
                    if "function" in tool_call:
                        function_call = tool_call["function"]
                        
                        # Get function details
                        function_name = function_call["name"]
                        arguments = function_call["arguments"]
                        
                        # Parse arguments
                        try:
                            args = json.loads(arguments)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                            
                        # Execute function if kernel is available
                        if agent_kernel:
                            result, success = agent_kernel.execute_capability(function_name, **args)
                            
                            # Create tool response
                            tool_call_id = tool_call.get("id", f"call_{uuid.uuid4()}")
                            tool_response = {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": function_name,
                                "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                            }
                            
                            # Add function call result to conversation
                            self.messages.append(tool_response)
                            tool_responses.append(tool_response)
                            
                            # Also add to message for the caller
                            message["tool_responses"].append({
                                "tool_call_id": tool_call_id,
                                "function_name": function_name, 
                                "result": result
                            })
                            
                            # Track function call
                            self.metadata["function_calls"].append({
                                "name": function_name,
                                "arguments": args,
                                "result": result,
                                "timestamp": time.time()
                            })
                
                # Let the model decide what to do next
                # This allows the model to recursively use tools when needed
                # We only do this once to prevent infinite recursive calls
                if not message.get("is_recursive_call", False):
                    next_prompt_data = {
                        "model": self.model,
                        "messages": format_messages_for_api(self.messages),
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "stream": False,
                        "tools": self._get_tool_definitions()
                    }
                    
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    if self.debug:
                        print("[DEBUG] Getting next response from model after tool execution")
                    next_response = requests.post(
                        self.endpoint,
                        headers=headers,
                        json=next_prompt_data
                    )
                    
                    if next_response.status_code == 200:
                        # Mark this as a recursive call to prevent infinite loops
                        next_data = next_response.json()
                        next_data["is_recursive_call"] = True
                        
                        # Process the recursive response
                        next_result = self._handle_response(next_data)
                        
                        # Add the final result to our message
                        message["final_response"] = next_result.get("content", "")
                        
                        # Also copy any new tool calls/responses
                        if "tool_calls" in next_result:
                            if "recursive_tool_calls" not in message:
                                message["recursive_tool_calls"] = []
                            message["recursive_tool_calls"].extend(next_result["tool_calls"])
                            
                        if "tool_responses" in next_result:
                            message["tool_responses"].extend(next_result["tool_responses"])
            
            # Return the processed message
            return message
                
        except Exception as e:
            error_info = f"Error processing response: {str(e)}"
            logger.error(error_info)
            return {"error": error_info}
            
    def _get_tool_definitions(self):
        """Get tool definitions for the API request"""
        if not agent_kernel:
            return []
            
        capabilities = agent_kernel.list_capabilities()
        tools = []
        
        for cap in capabilities:
            tool = {
                "type": "function",
                "function": {
                    "name": cap["name"],
                    "description": cap["description"]
                }
            }
            
            # Add parameters if available
            if cap["parameters"]:
                params = cap["parameters"] or {}
                tool["function"]["parameters"] = {
                    "type": "object",
                    "properties": params
                }
                
                # Only add required field if we have parameters
                if params:
                    tool["function"]["parameters"]["required"] = list(params.keys())
                
            tools.append(tool)
            
        return tools
        
    def add_code_module(self, name: str, code: str):
        """Add a new code module to the agent"""
        if not agent_kernel:
            return {"error": "Agent kernel not initialized"}
            
        return agent_kernel.add_code_module(name, code)
        
    def load_code_module(self, name: str):
        """Load a code module"""
        if not agent_kernel:
            return {"error": "Agent kernel not initialized"}
            
        return agent_kernel.load_code_module(name)
        
    def reason(self, query: str, max_steps=MAX_ITERATIONS):
        """
        Use multi-step reasoning to answer a complex query
        """
        # Set up a new reasoning session
        self.metadata["reasoning_steps"] = []
        self.metadata["current_goal"] = query
        
        # Add reasoning system prompt
        original_system_prompt = self.system_prompt
        reasoning_system_prompt = f"""You are an advanced reasoning agent that thinks step-by-step. 
Your task is to solve the following problem: {query}

You have access to these tools:
1. Functions you can call to get information or perform actions
2. The ability to create new code to enhance your capabilities

Think through this problem carefully:
1. Break it down into sub-problems if needed
2. Consider multiple approaches
3. Use the functions available to you
4. Create new functions if needed to solve specific aspects

Respond with your step-by-step reasoning and conclusions.
"""
        self.system_prompt = reasoning_system_prompt
        
        # Reset reasoning context
        self.messages = [
            {"role": "system", "content": reasoning_system_prompt}
        ]
        
        # Start reasoning process
        step = 1
        while step <= max_steps:
            # Determine the next reasoning step
            next_prompt = "What is the next step in solving this problem?"
            
            if step == 1:
                next_prompt = f"Let's solve this problem step by step: {query}"
                
            # Get model's next reasoning step
            print(f"\n[Step {step}] Thinking...")
            result = self.chat(next_prompt, stream=True)
            
            # Track this step
            if isinstance(result, dict) and "message" in result:
                content = result["message"].get("content", "")
                
                # Record the reasoning step
                reasoning_step = {
                    "step": step,
                    "prompt": next_prompt,
                    "reasoning": content,
                    "function_calls": []
                }
                
                # Check for function calls in this step
                if "tool_calls" in result["message"]:
                    for call in result["message"]["tool_calls"]:
                        # Handle different tool call structures
                        if "function" in call:
                            # Handle structured format with function field
                            reasoning_step["function_calls"].append({
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                                "result": call.get("response")
                            })
                        else:
                            # Handle direct format
                            reasoning_step["function_calls"].append({
                                "name": call.get("name", "unknown"),
                                "arguments": call.get("arguments", "{}"),
                                "result": call.get("response")
                            })
                        
                self.metadata["reasoning_steps"].append(reasoning_step)
                
                # Check if we've reached a conclusion
                conclusion_markers = [
                    "in conclusion", "to conclude", "final answer", 
                    "the answer is", "therefore", "hence", "thus"
                ]
                if any(marker in content.lower() for marker in conclusion_markers):
                    print("\n[Reasoning] Reached conclusion")
                    break
                    
            step += 1
        
        # Restore original system prompt
        self.system_prompt = original_system_prompt
        
        # Summarize the reasoning
        reasoning_summary = ""
        for step_info in self.metadata["reasoning_steps"]:
            reasoning_summary += f"Step {step_info['step']}: {step_info['reasoning'][:100]}...\n"
            
        final_step = self.metadata["reasoning_steps"][-1]["reasoning"]
        
        return {
            "query": query,
            "steps_taken": len(self.metadata["reasoning_steps"]),
            "reasoning_trace": self.metadata["reasoning_steps"],
            "conclusion": final_step,
            "function_calls": self.metadata["function_calls"]
        }
        
    def generate_code(self, task_description: str, language: str = "python"):
        """Generate code to solve a specific task"""
        prompt = f"""Create a {language} code solution for this task: {task_description}

The code should:
1. Be well-documented with comments
2. Handle edge cases appropriately 
3. Follow best practices for {language}
4. Be optimized for readability and performance

Return ONLY the code itself without additional explanation.
"""
        # Use streaming for better generation
        result = self.chat(prompt, stream=True)
        
        # Extract code from the response
        if isinstance(result, dict) and "message" in result:
            content = result["message"].get("content", "")
            
            # Try to extract code blocks
            code_pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
            matches = re.findall(code_pattern, content)
            
            if matches:
                # Return the first code block
                return matches[0].strip()
            else:
                # Return whole content if no code blocks found
                return content.strip()
        
        return "# Failed to generate code"

# Create global agent kernel (protected)
agent_kernel = AgentKernel()

###############################################################################
# UTILITY FUNCTIONS                                                          #
###############################################################################

def text_sentiment_analysis(text: str) -> Dict:
    """
    Perform basic sentiment analysis on text
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    # Simple lexicon-based sentiment analysis
    positive_words = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "happy", "joy", "joyful", "love", "liked", "like", "awesome",
        "positive", "beautiful", "best", "better", "success", "successful"
    }
    
    negative_words = {
        "bad", "terrible", "awful", "horrible", "sad", "unhappy", "hate",
        "dislike", "poor", "negative", "worst", "worse", "failure", "failed",
        "disappointing", "disappointed", "annoying", "angry", "anger"
    }
    
    # Normalize and tokenize text
    words = text.lower().split()
    
    # Count sentiment words
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Calculate sentiment score (-1 to 1)
    total = positive_count + negative_count
    if total == 0:
        sentiment_score = 0
    else:
        sentiment_score = (positive_count - negative_count) / total
    
    # Determine sentiment label
    if sentiment_score > 0.25:
        sentiment = "positive"
    elif sentiment_score < -0.25:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "score": sentiment_score,
        "positive_words": positive_count,
        "negative_words": negative_count,
        "word_count": len(words)
    }

def summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Generate a simple extractive summary of text
    
    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences to include
        
    Returns:
        Summarized text
    """
    # Split into sentences (basic approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= max_sentences:
        return text
    
    # Simple scoring - prefer longer sentences with important words
    important_words = {"key", "important", "significant", "main", "critical", 
                      "essential", "crucial", "primary", "major", "vital"}
    
    sentence_scores = []
    for sentence in sentences:
        # Score based on length (normalized) - prefer medium length
        length_score = min(len(sentence.split()) / 20, 1.0)
        
        # Score based on position - prefer early sentences
        position = sentences.index(sentence) + 1
        position_score = 1.0 / position if position <= 5 else 0.1
        
        # Score based on important words
        words = sentence.lower().split()
        important_score = sum(1 for word in words if word in important_words) / len(words) if words else 0
        
        # Calculate total score
        total_score = (length_score * 0.4) + (position_score * 0.4) + (important_score * 0.2)
        sentence_scores.append((sentence, total_score))
    
    # Sort by score and take top sentences
    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in sorted_sentences[:max_sentences]]
    
    # Reorder sentences to maintain original flow
    ordered_summary = [s for s in sentences if s in top_sentences]
    
    return " ".join(ordered_summary)

def analyze_numerical_data(data: List[float]) -> Dict:
    """
    Perform statistical analysis on numerical data
    
    Args:
        data: List of numerical values
        
    Returns:
        Dictionary with statistical measures
    """
    if not data:
        return {"error": "Empty data set"}
    
    # Basic statistics
    count = len(data)
    sum_values = sum(data)
    mean = sum_values / count
    
    # Calculate median
    sorted_data = sorted(data)
    mid = count // 2
    median = sorted_data[mid] if count % 2 else (sorted_data[mid-1] + sorted_data[mid]) / 2
    
    # Calculate variance and standard deviation
    variance = sum((x - mean) ** 2 for x in data) / count
    std_dev = variance ** 0.5
    
    # Calculate min, max, range
    min_val = min(data)
    max_val = max(data)
    data_range = max_val - min_val
    
    # Calculate quartiles
    q1_idx = count // 4
    q3_idx = 3 * count // 4
    q1 = sorted_data[q1_idx]
    q3 = sorted_data[q3_idx]
    iqr = q3 - q1
    
    return {
        "count": count,
        "mean": mean,
        "median": median,
        "min": min_val,
        "max": max_val,
        "range": data_range,
        "variance": variance,
        "std_dev": std_dev,
        "q1": q1,
        "q3": q3,
        "iqr": iqr
    }

def generate_random_data(data_type: str = "int", min_val: float = 0, max_val: float = 100, size: int = 10) -> List:
    """
    Generate random data for testing
    
    Args:
        data_type: Type of data ('int', 'float', 'bool', 'str')
        min_val: Minimum value for numerical data
        max_val: Maximum value for numerical data
        size: Number of elements to generate
        
    Returns:
        List of random data
    """
    import random
    import string
    
    if size <= 0:
        return []
    
    if data_type == "int":
        return [random.randint(int(min_val), int(max_val)) for _ in range(size)]
    
    elif data_type == "float":
        return [random.uniform(min_val, max_val) for _ in range(size)]
    
    elif data_type == "bool":
        return [random.choice([True, False]) for _ in range(size)]
    
    elif data_type == "str":
        # Generate random strings
        str_length = int(max(5, min_val))  # Use min_val as string length
        return [''.join(random.choices(string.ascii_letters, k=str_length)) for _ in range(size)]
    
    else:
        return {"error": f"Unknown data type: {data_type}"}

def extract_entities(text: str) -> Dict:
    """
    Extract named entities from text using simple pattern matching
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with extracted entities
    """
    entities = {
        "people": [],
        "organizations": [],
        "locations": [],
        "dates": [],
        "emails": [],
        "urls": []
    }
    
    # Simple regex patterns for entity extraction
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    entities["emails"] = emails
    
    # URL pattern
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(url_pattern, text)
    entities["urls"] = urls
    
    # Date patterns (simple)
    date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
    dates = re.findall(date_pattern, text, re.IGNORECASE)
    entities["dates"] = dates
    
    # Very basic entity extraction based on capitalization patterns
    # Not accurate but provides a simple demonstration
    words = text.split()
    current_entity = []
    current_type = None
    
    for i, word in enumerate(words):
        # Skip URLs and emails we already found
        if any(word in e for e in emails + urls):
            continue
            
        # Check for capitalized words that might be entities
        if word and word[0].isupper():
            # Check for organization indicators
            if any(indicator in word.lower() for indicator in ["inc", "corp", "llc", "company", "co.", "corporation"]):
                if current_type == "organization":
                    current_entity.append(word)
                else:
                    if current_entity and current_type:
                        entity_text = " ".join(current_entity)
                        entities[current_type + "s"].append(entity_text)
                    current_entity = [word]
                    current_type = "organization"
            
            # Check for location indicators
            elif any(indicator in word.lower() for indicator in ["street", "ave", "road", "blvd", "drive", "lane", "place", "city"]):
                if current_type == "location":
                    current_entity.append(word)
                else:
                    if current_entity and current_type:
                        entity_text = " ".join(current_entity)
                        entities[current_type + "s"].append(entity_text)
                    current_entity = [word]
                    current_type = "location"
            
            # Otherwise might be a person or continue current entity
            elif current_entity and (i > 0 and words[i-1][0].isupper()):
                current_entity.append(word)
            else:
                if current_entity and current_type:
                    entity_text = " ".join(current_entity)
                    entities[current_type + "s"].append(entity_text)
                current_entity = [word]
                current_type = "person"  # Default to person for capitalized words
        else:
            # End of entity
            if current_entity and current_type:
                entity_text = " ".join(current_entity)
                entities[current_type + "s"].append(entity_text)
                current_entity = []
                current_type = None
    
    # Add final entity if exists
    if current_entity and current_type:
        entity_text = " ".join(current_entity)
        entities[current_type + "s"].append(entity_text)
    
    # Remove duplicates
    for entity_type in entities:
        entities[entity_type] = list(set(entities[entity_type]))
    
    return entities

def convert_units(value: float, from_unit: str, to_unit: str) -> Dict:
    """
    Convert between different units of measurement
    
    Args:
        value: The value to convert
        from_unit: The source unit (e.g., 'km', 'mi', 'kg', 'lb', 'celsius', 'fahrenheit')
        to_unit: The target unit
        
    Returns:
        Dictionary with conversion result
    """
    # Define conversion factors for different unit types
    length_units = {
        "m": 1.0,
        "km": 1000.0,
        "cm": 0.01,
        "mm": 0.001,
        "in": 0.0254,
        "ft": 0.3048,
        "yd": 0.9144,
        "mi": 1609.344
    }
    
    weight_units = {
        "g": 1.0,
        "kg": 1000.0,
        "mg": 0.001,
        "lb": 453.59237,
        "oz": 28.349523125,
        "st": 6350.29318,
        "ton": 907184.74,
        "tonne": 1000000.0
    }
    
    volume_units = {
        "l": 1.0,
        "ml": 0.001,
        "gal": 3.78541,
        "qt": 0.946353,
        "pt": 0.473176,
        "cup": 0.2365882,
        "oz_fluid": 0.0295735,
        "tbsp": 0.0147868,
        "tsp": 0.00492892
    }
    
    area_units = {
        "m2": 1.0,
        "km2": 1000000.0,
        "cm2": 0.0001,
        "mm2": 0.000001,
        "ha": 10000.0,
        "acre": 4046.86,
        "ft2": 0.092903,
        "in2": 0.00064516,
        "mi2": 2589988.11
    }
    
    time_units = {
        "s": 1.0,
        "min": 60.0,
        "hr": 3600.0,
        "day": 86400.0,
        "week": 604800.0,
        "month": 2592000.0,  # 30 days
        "year": 31536000.0,  # 365 days
        "ms": 0.001
    }
    
    # Handle temperature conversions separately
    if from_unit.lower() in ["c", "celsius"] and to_unit.lower() in ["f", "fahrenheit"]:
        result = (value * 9/5) + 32
        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
            "formula": "°F = (°C × 9/5) + 32"
        }
    
    elif from_unit.lower() in ["f", "fahrenheit"] and to_unit.lower() in ["c", "celsius"]:
        result = (value - 32) * 5/9
        return {
            "original_value": value,
            "original_unit": from_unit,
            "converted_value": result,
            "converted_unit": to_unit,
            "formula": "°C = (°F - 32) × 5/9"
        }
    
    # Handle other unit conversions
    unit_categories = {
        "length": length_units,
        "weight": weight_units,
        "volume": volume_units,
        "area": area_units,
        "time": time_units
    }
    
    # Find which category the units belong to
    category = None
    for cat_name, units in unit_categories.items():
        if from_unit.lower() in [u.lower() for u in units.keys()] and to_unit.lower() in [u.lower() for u in units.keys()]:
            category = cat_name
            break
    
    if not category:
        return {
            "error": f"Cannot convert between {from_unit} and {to_unit}. Units are incompatible or not supported."
        }
    
    # Get the conversion factor for each unit
    units = unit_categories[category]
    from_factor = None
    to_factor = None
    
    for unit, factor in units.items():
        if unit.lower() == from_unit.lower():
            from_factor = factor
        if unit.lower() == to_unit.lower():
            to_factor = factor
    
    # Convert to the base unit, then to the target unit
    base_value = value * from_factor
    result = base_value / to_factor
    
    return {
        "original_value": value,
        "original_unit": from_unit,
        "converted_value": result,
        "converted_unit": to_unit,
        "category": category
    }

def generate_qr_code(data: str, file_name: str = "qr_code") -> Dict:
    """
    Generate a QR code from data
    
    Args:
        data: The data to encode in the QR code
        file_name: The base name for the output file (without extension)
        
    Returns:
        Dictionary with information about the generated QR code
    """
    try:
        # Try to import qrcode library
        try:
            import qrcode
        except ImportError:
            return {"error": "QR code generation requires the 'qrcode' library"}
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Save to file with timestamp to avoid overwriting
        timestamp = int(time.time())
        file_path = f"{file_name}_{timestamp}.png"
        img.save(file_path)
        
        return {
            "status": "success",
            "message": f"QR code generated for data: {data[:50]}{'...' if len(data) > 50 else ''}",
            "file_path": file_path,
            "data_length": len(data)
        }
    except Exception as e:
        return {"error": f"Failed to generate QR code: {str(e)}"}

def format_data_as_table(data: List[Dict], columns: List[str] = None) -> str:
    """
    Format a list of dictionaries as a text table
    
    Args:
        data: List of dictionaries containing the data
        columns: List of column names to include (defaults to all keys in first row)
        
    Returns:
        Formatted text table
    """
    if not data:
        return "Empty data set"
    
    # Determine columns if not specified
    if not columns:
        columns = list(data[0].keys())
    
    # Calculate column widths
    col_widths = {col: len(col) for col in columns}
    for row in data:
        for col in columns:
            if col in row:
                col_widths[col] = max(col_widths[col], len(str(row[col])))
    
    # Create header
    header = " | ".join(col.ljust(col_widths[col]) for col in columns)
    separator = "-+-".join("-" * col_widths[col] for col in columns)
    
    # Create rows
    rows = []
    for row in data:
        formatted_row = " | ".join(
            str(row.get(col, "")).ljust(col_widths[col]) for col in columns
        )
        rows.append(formatted_row)
    
    # Combine all parts
    table = f"{header}\n{separator}\n" + "\n".join(rows)
    
    return table

def encrypt_decrypt_text(text: str, key: str, encrypt: bool = True) -> Dict:
    """
    Simple encryption/decryption using XOR operation with a key
    
    Args:
        text: Text to encrypt or decrypt
        key: Encryption key
        encrypt: True for encryption, False for decryption
        
    Returns:
        Dictionary with result
    """
    try:
        # Convert key to repeating byte sequence
        key_bytes = key.encode('utf-8')
        key_length = len(key_bytes)
        
        if encrypt:
            # For encryption, convert text to bytes
            text_bytes = text.encode('utf-8')
            operation = "encryption"
        else:
            # For decryption, convert hex string to bytes
            try:
                text_bytes = bytes.fromhex(text)
                operation = "decryption"
            except ValueError:
                return {"error": "Invalid hexadecimal input for decryption"}
        
        # Apply XOR with the key
        result_bytes = bytearray()
        for i, byte in enumerate(text_bytes):
            key_byte = key_bytes[i % key_length]
            result_bytes.append(byte ^ key_byte)
        
        if encrypt:
            # Return hex string for encrypted result
            result = result_bytes.hex()
        else:
            # Return decoded text for decrypted result
            try:
                result = result_bytes.decode('utf-8')
            except UnicodeDecodeError:
                return {"error": "Decryption failed: result is not valid UTF-8. Incorrect key?"}
        
        return {
            "operation": operation,
            "result": result,
            "key_length": key_length,
            "data_length": len(text_bytes)
        }
        
    except Exception as e:
        return {"error": f"Error during {operation}: {str(e)}"}

###############################################################################
# SAMPLE CAPABILITIES                                                         #
###############################################################################

def get_weather(location: str, units: str = "metric") -> Dict:
    """Get weather information for a location using Jina web search"""
    # Construct search query for current weather
    search_query = f"current weather in {location}"
    
    try:
        # Use jina_search if available
        if agent_kernel:
            result, success = agent_kernel.execute_capability("jina_search", query=search_query)
            
            if success and isinstance(result, dict) and "results" in result:
                # Process search results
                weather_data = {
                    "location": location,
                    "temperature": None,
                    "conditions": None,
                    "humidity": None,
                    "units": units
                }
                
                # Extract temperature and conditions from search results
                if isinstance(result["results"], dict) and "results" in result["results"]:
                    for item in result["results"]["results"]:
                        if "snippet" in item:
                            snippet = item["snippet"].lower()
                            
                            # Extract temperature
                            temp_match = re.search(r'(\d+(?:\.\d+)?)(?:\s*)[°º](?:[CF])?', snippet)
                            if temp_match and weather_data["temperature"] is None:
                                weather_data["temperature"] = float(temp_match.group(1))
                            
                            # Extract conditions
                            for condition in ["sunny", "cloudy", "partly cloudy", "rainy", "snow", "clear"]:
                                if condition in snippet and weather_data["conditions"] is None:
                                    weather_data["conditions"] = condition.title()
                            
                            # Extract humidity
                            humidity_match = re.search(r'humidity[:\s]+(\d+)%', snippet)
                            if humidity_match and weather_data["humidity"] is None:
                                weather_data["humidity"] = int(humidity_match.group(1))
                
                # If we found weather data, return it
                if weather_data["temperature"] is not None:
                    return weather_data
    except Exception as e:
        logger.error(f"Error getting weather via web search: {str(e)}")
    
    # Fallback to mock data
    return {
        "location": location,
        "temperature": 22.5 if units == "metric" else 72.5,
        "conditions": "Partly cloudy",
        "humidity": 65,
        "units": units
    }
    
def search_knowledge(query: str, limit: int = 5) -> List[Dict]:
    """Search the knowledge base for relevant information"""
    if not agent_kernel:
        return [{"error": "Agent kernel not initialized"}]
        
    # Execute a search query
    success, results = agent_kernel.execute_query(
        "SELECT * FROM code_modules WHERE code LIKE ? LIMIT ?",
        (f"%{query}%", limit)
    )
    
    if not success:
        return [{"error": results}]
        
    return results
    
def create_agent_code(description: str) -> Dict:
    """Create a code module based on a description"""
    # Create a template for the new module
    code = f"""# Code module generated from description: {description}
import time
from typing import Dict, List, Any

def main_function(input_data: Dict = None) -> Dict:
    \"\"\"
    Main function for this module: {description}
    \"\"\"
    # Implementation goes here
    result = {{"status": "success", "message": "Function executed"}}
    
    # Add timestamp
    result["timestamp"] = time.time()
    
    return result
"""
    
    # Generate a name for the module
    name = f"generated_{int(time.time())}"
    
    # Save the module
    if agent_kernel:
        success, message = agent_kernel.add_code_module(name, code)
        
        if success:
            return {
                "status": "success",
                "module_name": name,
                "code": code,
                "message": message
            }
        else:
            return {"error": message}
    else:
        return {"error": "Agent kernel not initialized"}

def web_research(topic: str, depth: int = 3, max_sources: int = 5) -> Dict:
    """
    Perform comprehensive web research on a topic
    
    Args:
        topic: The research topic or question
        depth: How deep to go in research (1-3, higher means more thorough)
        max_sources: Maximum number of sources to check
        
    Returns:
        Research results with citations
    """
    if not agent_kernel:
        return {"error": "Agent kernel not initialized"}
    
    # Step 1: Initial search to find relevant sources
    result, success = agent_kernel.execute_capability(
        "jina_search", 
        query=topic
    )
    
    if not success:
        return {"error": f"Search failed: {result}"}
    
    search_results = []
    urls = []
    
    # Extract search results and URLs
    try:
        if isinstance(result, dict) and "results" in result:
            if isinstance(result["results"], dict) and "results" in result["results"]:
                search_results = result["results"]["results"][:max_sources]
                urls = [item.get("url") for item in search_results if "url" in item]
    except:
        pass
    
    # If we couldn't extract URLs, return basic search results
    if not urls:
        return {
            "status": "limited_results",
            "topic": topic,
            "search_results": result,
            "summary": "Could not extract structured data from search results."
        }
    
    # Step 2: Read content from top sources
    sources_content = []
    
    for i, url in enumerate(urls[:max_sources]):
        # Skip if not a valid URL
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            continue
            
        # Get content from URL
        read_result, read_success = agent_kernel.execute_capability(
            "jina_read_url", 
            url=url
        )
        
        if read_success:
            # Extract important facts and metadata
            source_info = {
                "url": url,
                "title": search_results[i].get("title", "Untitled"),
                "content": read_result
            }
            
            # Extract key facts if available
            if isinstance(read_result, dict) and "extraction" in read_result:
                extraction = read_result["extraction"]
                if "important_facts" in extraction:
                    source_info["facts"] = extraction["important_facts"]
                
            sources_content.append(source_info)
    
    # Step 3: Additional fact checking if depth > 1
    fact_checks = []
    
    if depth >= 2 and sources_content:
        # Extract key statements to fact check
        all_facts = []
        for source in sources_content:
            if "facts" in source:
                all_facts.extend(source["facts"][:3])  # Take up to 3 facts per source
        
        # Fact check key statements
        for fact in all_facts[:5]:  # Limit to 5 fact checks
            check_result, check_success = agent_kernel.execute_capability(
                "jina_fact_check", 
                query=fact
            )
            
            if check_success:
                fact_checks.append({
                    "statement": fact,
                    "check_result": check_result
                })
    
    # Step 4: Synthesize the research
    return {
        "status": "success",
        "topic": topic,
        "depth": depth,
        "sources": len(sources_content),
        "sources_content": sources_content,
        "fact_checks": fact_checks,
        "timestamp": time.time()
    }

# Register sample capabilities
if agent_kernel:
    # Register utility functions
    agent_kernel.register_capability(
        "text_sentiment_analysis",
        text_sentiment_analysis,
        "Analyze sentiment in text (positive, negative, neutral)",
        {
            "text": "Text to analyze for sentiment"
        },
        is_kernel=True
    )
    
    agent_kernel.register_capability(
        "summarize_text",
        summarize_text,
        "Generate a simple extractive summary of text",
        {
            "text": "Text to summarize",
            "max_sentences": "Maximum number of sentences to include in summary"
        },
        is_kernel=True
    )
    
    agent_kernel.register_capability(
        "analyze_numerical_data",
        analyze_numerical_data,
        "Perform statistical analysis on numerical data",
        {
            "data": "List of numerical values to analyze"
        },
        is_kernel=True
    )
    
    agent_kernel.register_capability(
        "generate_random_data",
        generate_random_data,
        "Generate random data for testing",
        {
            "data_type": "Type of data ('int', 'float', 'bool', 'str')",
            "min_val": "Minimum value for numerical data",
            "max_val": "Maximum value for numerical data",
            "size": "Number of elements to generate"
        },
        is_kernel=True
    )
    
    agent_kernel.register_capability(
        "extract_entities",
        extract_entities,
        "Extract named entities from text using simple pattern matching",
        {
            "text": "Text to analyze for entities"
        },
        is_kernel=True
    )
    
    agent_kernel.register_capability(
        "convert_units",
        convert_units,
        "Convert between different units of measurement",
        {
            "value": "The value to convert",
            "from_unit": "The source unit (e.g., 'km', 'mi', 'kg', 'lb', 'celsius', 'fahrenheit')",
            "to_unit": "The target unit"
        },
        is_kernel=True
    )
    
    agent_kernel.register_capability(
        "generate_qr_code",
        generate_qr_code,
        "Generate a QR code from data",
        {
            "data": "The data to encode in the QR code",
            "file_name": "The base name for the output file (without extension)"
        },
        is_kernel=True
    )
    
    agent_kernel.register_capability(
        "format_data_as_table",
        format_data_as_table,
        "Format a list of dictionaries as a text table",
        {
            "data": "List of dictionaries containing the data",
            "columns": "List of column names to include (defaults to all keys in first row)"
        },
        is_kernel=True
    )
    
    agent_kernel.register_capability(
        "encrypt_decrypt_text",
        encrypt_decrypt_text,
        "Simple encryption/decryption using XOR operation with a key",
        {
            "text": "Text to encrypt or decrypt",
            "key": "Encryption key",
            "encrypt": "True for encryption, False for decryption"
        },
        is_kernel=True
    )
    
    # Register original capabilities
    agent_kernel.register_capability(
        "get_weather",
        get_weather,
        "Get weather information for a location using Jina web search",
        {
            "location": "City or area name",
            "units": "Either 'metric' or 'imperial'"
        },
        is_kernel=True  # Mark as kernel function
    )
    
    agent_kernel.register_capability(
        "search_knowledge",
        search_knowledge,
        "Search the knowledge base for information",
        {
            "query": "Search query string",
            "limit": "Maximum number of results"
        },
        is_kernel=True
    )
    
    agent_kernel.register_capability(
        "create_agent_code",
        create_agent_code,
        "Create a new code module from a description",
        {
            "description": "Description of what the code should do"
        },
        is_kernel=True
    )
    
    # Register web research capability
    agent_kernel.register_capability(
        "web_research",
        web_research,
        "Perform comprehensive web research on a topic, gathering information from multiple sources",
        {
            "topic": "Research topic or question",
            "depth": "Research depth (1-3, higher means more thorough)",
            "max_sources": "Maximum number of sources to check"
        },
        is_kernel=True
    )
    
    # Try to register Jina tools if available
    try:
        from modules.jina_tools import jina_search, jina_fact_check, jina_read_url
        
        # Register jina_search
        agent_kernel.register_capability(
            "jina_search",
            jina_search,
            "Search the web using Jina search API with AI-powered content extraction",
            {
                "query": "Search query text",
                "token": "Optional Jina API token (uses env var if not provided)",
                "extract_content": "Whether to extract structured content"
            },
            is_kernel=True
        )
        
        # Register jina_fact_check
        agent_kernel.register_capability(
            "jina_fact_check",
            jina_fact_check,
            "Fact check a statement using Jina grounding API with AI-powered content extraction",
            {
                "query": "Statement to fact check",
                "token": "Optional Jina API token (uses env var if not provided)",
                "extract_content": "Whether to extract structured content"
            },
            is_kernel=True
        )
        
        # Register jina_read_url
        agent_kernel.register_capability(
            "jina_read_url",
            jina_read_url,
            "Read and rank content from a URL using Jina ranking API with AI-powered content extraction",
            {
                "url": "URL to read and rank",
                "token": "Optional Jina API token (uses env var if not provided)",
                "extract_content": "Whether to extract structured content"
            },
            is_kernel=True
        )
        
        print("Successfully registered Jina tools!")
        
    except ImportError:
        print("Jina tools not available. Run pip install -r requirements.txt to enable.")
    except Exception as e:
        print(f"Error registering Jina tools: {str(e)}")

###############################################################################
# COMMAND LINE INTERFACE                                                      #
###############################################################################

# Command registry for CLI
COMMAND_REGISTRY = {
    "help": "Display available commands",
    "code": "Create, modify, or view code modules",
    "function": "Create, call or list functions",
    "reason": "Run autonomous reasoning on a query",
    "research": "Perform comprehensive web research on a topic",
    "search": "Search the web using Jina.ai",
    "fact-check": "Fact check a statement using Jina.ai",
    "read-url": "Read and analyze content from a URL using Jina.ai",
    "info": "Display agent information",
    "exit": "Exit the program",
}

def handle_command(agent, command_str):
    """Handle CLI commands"""
    parts = command_str.strip().split(maxsplit=1)
    cmd = parts[0][1:]  # Remove leading /
    args = parts[1] if len(parts) > 1 else ""
    
    if cmd == "help":
        print("\n📚 Available commands:")
        for command, description in COMMAND_REGISTRY.items():
            print(f"  /{command} - {description}")
        return True
        
    elif cmd == "reason":
        if not args:
            print("❌ Error: Missing query. Usage: /reason <query>")
            return True
            
        print(f"🧠 Running autonomous reasoning on: {args}")
        result = agent.reason(args, max_steps=MAX_ITERATIONS)
        
        print("\n✅ Reasoning complete")
        print(f"Steps taken: {result['steps_taken']}")
        print(f"Conclusion: {result['conclusion'][:150]}...")
        return True

    elif cmd == "code":
        if not args:
            print("Available code commands: create, view, list")
            return True
            
        code_parts = args.split(maxsplit=1)
        code_cmd = code_parts[0]
        code_args = code_parts[1] if len(code_parts) > 1 else ""
        
        if code_cmd == "list":
            if not agent_kernel:
                print("❌ Error: Agent kernel not initialized")
                return True
                
            modules = agent_kernel.list_code_modules()
            print(f"\n📋 Code modules ({len(modules)}):")
            for module in modules:
                print(f"  - {module}")
            return True
                
        elif code_cmd == "view":
            if not code_args:
                print("❌ Error: Missing module name. Usage: /code view <module_name>")
                return True
                
            if not agent_kernel:
                print("❌ Error: Agent kernel not initialized")
                return True
                
            success, code = agent_kernel.get_code_module(code_args)
            if success:
                print(f"\n📝 Code module: {code_args}\n")
                print(code)
            else:
                print(f"❌ Error: {code}")
            return True
                
        elif code_cmd == "create":
            if not code_args:
                print("❌ Error: Missing description. Usage: /code create <description>")
                return True
                
            print(f"💻 Creating code module for: {code_args}")
            result = create_agent_code(code_args)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Created module: {result['module_name']}")
            return True
                
        else:
            print(f"❌ Unknown code command: {code_cmd}")
            return True
            
    elif cmd == "function":
        if not args:
            print("Available function commands: list, call, create")
            return True
            
        func_parts = args.split(maxsplit=1)
        func_cmd = func_parts[0]
        func_args = func_parts[1] if len(func_parts) > 1 else ""
        
        if func_cmd == "list":
            if not agent_kernel:
                print("❌ Error: Agent kernel not initialized")
                return True
                
            capabilities = agent_kernel.list_capabilities()
            print(f"\n🔧 Functions ({len(capabilities)}):")
            for capability in capabilities:
                kernel_mark = "🔒" if capability["is_kernel"] else " "
                print(f"  {kernel_mark} {capability['name']} - {capability['description']}")
            return True
                
        elif func_cmd == "call":
            if not func_args:
                print("❌ Error: Missing function name and args. Usage: /function call <name> <args_json>")
                return True
                
            call_parts = func_args.split(maxsplit=1)
            func_name = call_parts[0]
            args_str = call_parts[1] if len(call_parts) > 1 else "{}"
            
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON arguments")
                return True
                
            print(f"🔧 Calling function: {func_name}")
            if agent_kernel:
                result, success = agent_kernel.execute_capability(func_name, **args)
                
                if success:
                    print(f"✅ Result: {result}")
                else:
                    print(f"❌ Error: {result}")
            else:
                print("❌ Error: Agent kernel not initialized")
            return True
                
        elif func_cmd == "create":
            # Needs agent to generate the function
            print("🤖 Asking agent to create a new function")
            if not func_args:
                print("❌ Error: Missing function description. Usage: /function create <description>")
                return True
                
            # Use chat to generate function code
            prompt = f"Generate a Python function based on this description: {func_args}\n\nProvide ONLY the function code, including a clear docstring."
            
            print("⌛ Generating function code...")
            result = agent.chat(prompt, stream=True)
            
            if isinstance(result, dict) and "message" in result:
                content = result["message"].get("content", "")
                
                # Extract function code
                code_pattern = r"```(?:python)?\s*([\s\S]*?)```"
                matches = re.findall(code_pattern, content)
                
                code = matches[0].strip() if matches else content.strip()
                
                # Extract function name from code
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_name = node.name
                            
                            # Register the function
                            print(f"📝 Registering function: {func_name}")
                            result = create_function(func_name, code, func_args)
                            
                            if "error" in result:
                                print(f"❌ Error: {result['error']}")
                            else:
                                print(f"✅ Function created: {func_name}")
                            break
                    else:
                        print("❌ No function definition found in generated code")
                except SyntaxError:
                    print("❌ Generated code has syntax errors")
            else:
                print("❌ Failed to generate function code")
            return True
                
        else:
            print(f"❌ Unknown function command: {func_cmd}")
            return True
            
    elif cmd == "search":
        if not args:
            print("❌ Error: Missing search query. Usage: /search <query>")
            return True
            
        print(f"🔍 Searching the web for: {args}")
        if agent_kernel:
            try:
                # Call jina_search function through the kernel
                result, success = agent_kernel.execute_capability("jina_search", query=args)
                
                if success:
                    print("\n✅ Search results:")
                    
                    # Try to format results nicely if possible
                    try:
                        if isinstance(result, dict) and "results" in result:
                            results = result["results"]
                            if isinstance(results, dict) and "results" in results:
                                search_results = results["results"]
                                for i, item in enumerate(search_results[:5]):  # Show top 5 results
                                    print(f"\n  {i+1}. {item.get('title', 'No title')}")
                                    print(f"     URL: {item.get('url', 'No URL')}")
                                    if "snippet" in item:
                                        print(f"     Snippet: {item['snippet'][:150]}...")
                            else:
                                print(json.dumps(result, indent=2))
                        else:
                            print(json.dumps(result, indent=2))
                    except Exception as e:
                        # Just print raw result if formatting fails
                        print(str(result)[:500] + "..." if len(str(result)) > 500 else str(result))
                else:
                    print(f"❌ Error: {result}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        else:
            print("❌ Error: Agent kernel not initialized")
        return True
        
    elif cmd == "fact-check":
        if not args:
            print("❌ Error: Missing statement to fact check. Usage: /fact-check <statement>")
            return True
            
        print(f"🔍 Fact checking: {args}")
        if agent_kernel:
            try:
                # Call jina_fact_check function through the kernel
                result, success = agent_kernel.execute_capability("jina_fact_check", query=args)
                
                if success:
                    print("\n✅ Fact check results:")
                    try:
                        if isinstance(result, dict):
                            print(json.dumps(result, indent=2)[:1000] + "..." if len(json.dumps(result, indent=2)) > 1000 else json.dumps(result, indent=2))
                        else:
                            print(str(result)[:1000] + "..." if len(str(result)) > 1000 else str(result))
                    except Exception as e:
                        print(str(result)[:500] + "..." if len(str(result)) > 500 else str(result))
                else:
                    print(f"❌ Error: {result}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        else:
            print("❌ Error: Agent kernel not initialized")
        return True
        
    elif cmd == "read-url":
        if not args:
            print("❌ Error: Missing URL to read. Usage: /read-url <url>")
            return True
            
        if not args.startswith(("http://", "https://")):
            args = "https://" + args
            
        print(f"📄 Reading content from: {args}")
        if agent_kernel:
            try:
                # Call jina_read_url function through the kernel
                result, success = agent_kernel.execute_capability("jina_read_url", url=args)
                
                if success:
                    print("\n✅ Content analysis:")
                    # Try to format results nicely
                    try:
                        if isinstance(result, dict) and "extraction" in result:
                            extraction = result["extraction"]
                            print("\n📋 Key facts:")
                            for fact in extraction.get("important_facts", [])[:5]:  # Show top 5 facts
                                print(f"  • {fact}")
                                
                            if extraction.get("important_people"):
                                print("\n👤 People mentioned:")
                                for person in extraction.get("important_people", [])[:5]:
                                    print(f"  • {person}")
                                    
                            if extraction.get("important_organizations"):
                                print("\n🏢 Organizations mentioned:")
                                for org in extraction.get("important_organizations", [])[:5]:
                                    print(f"  • {org}")
                        else:
                            print(json.dumps(result, indent=2)[:1000] + "..." if len(json.dumps(result, indent=2)) > 1000 else json.dumps(result, indent=2))
                    except Exception as e:
                        print(str(result)[:500] + "..." if len(str(result)) > 500 else str(result))
                else:
                    print(f"❌ Error: {result}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        else:
            print("❌ Error: Agent kernel not initialized")
        return True
        
    elif cmd == "research":
        if not args:
            print("❌ Error: Missing research topic. Usage: /research <topic> [--depth=<1-3>] [--sources=<number>]")
            return True
            
        # Parse arguments for depth and max_sources
        depth = 2  # Default depth
        max_sources = 3  # Default sources
        
        # Extract any options
        if "--depth=" in args:
            try:
                depth_str = re.search(r'--depth=(\d+)', args)
                if depth_str:
                    depth = int(depth_str.group(1))
                    # Remove from args
                    args = args.replace(depth_str.group(0), "").strip()
            except:
                pass
                
        if "--sources=" in args:
            try:
                sources_str = re.search(r'--sources=(\d+)', args)
                if sources_str:
                    max_sources = int(sources_str.group(1))
                    # Remove from args
                    args = args.replace(sources_str.group(0), "").strip()
            except:
                pass
        
        # Clamp values to reasonable ranges
        depth = max(1, min(3, depth))
        max_sources = max(1, min(7, max_sources))
            
        print(f"🔍 Researching: {args} (Depth: {depth}, Sources: {max_sources})")
        print("⏳ This may take a moment as we search and analyze multiple sources...")
        
        if agent_kernel:
            try:
                # Call web_research function through the kernel
                result, success = agent_kernel.execute_capability(
                    "web_research", 
                    topic=args,
                    depth=depth,
                    max_sources=max_sources
                )
                
                if success:
                    print("\n✅ Research complete!")
                    
                    # Format and display the results
                    if isinstance(result, dict):
                        print(f"\n📚 Research on: {result.get('topic', args)}")
                        print(f"📊 Sources analyzed: {result.get('sources', 0)}")
                        
                        # Display source information
                        sources = result.get("sources_content", [])
                        if sources:
                            print("\n📝 Sources:")
                            for i, source in enumerate(sources):
                                print(f"\n  {i+1}. {source.get('title', 'Untitled')}")
                                print(f"     URL: {source.get('url', 'No URL')}")
                                
                                # Show facts from this source if available
                                if "facts" in source:
                                    print("     Key points:")
                                    for fact in source["facts"][:3]:  # Top 3 facts
                                        print(f"       • {fact}")
                        
                        # Display fact checks if available
                        fact_checks = result.get("fact_checks", [])
                        if fact_checks:
                            print("\n✓ Fact Checks:")
                            for i, check in enumerate(fact_checks[:3]):  # Top 3 fact checks
                                print(f"\n  {i+1}. Statement: {check.get('statement', 'N/A')}")
                                
                                # Try to extract the result of the fact check
                                check_result = check.get("check_result", {})
                                if isinstance(check_result, dict) and "extraction" in check_result:
                                    # Display verification status if found
                                    facts = check_result["extraction"].get("important_facts", [])
                                    if facts:
                                        print(f"     Result: {facts[0]}")
                    else:
                        print("Research results: " + str(result))
                else:
                    print(f"❌ Research failed: {result}")
            except Exception as e:
                print(f"❌ Error during research: {str(e)}")
                print(traceback.format_exc())
        else:
            print("❌ Error: Agent kernel not initialized")
        return True
        
    elif cmd == "info":
        print("\n📊 Agent Information:")
        print(f"  Agent ID: {agent.agent_id}")
        print(f"  Model: {agent.model}")
        
        if agent_kernel:
            stats = agent_kernel.get_stats()
            capabilities = agent_kernel.list_capabilities()
            
            print(f"  Kernel Version: {stats.get('version', 'unknown')}")
            print(f"  Functions: {len(capabilities)}")
            
            # Show most used functions
            cap_stats = stats.get("capabilities", {})
            if cap_stats:
                print("\n  Most used functions:")
                # Sort by usage count
                sorted_caps = sorted(
                    cap_stats.items(), 
                    key=lambda x: x[1].get("calls", 0), 
                    reverse=True
                )
                
                for i, (name, stat) in enumerate(sorted_caps[:5]):
                    print(f"    {i+1}. {name} - {stat.get('calls', 0)} calls")
        return True
        
    elif cmd == "exit" or cmd == "quit":
        print("👋 Goodbye!")
        return False
        
    return False

def register_default_apis():
    """Register default APIs if auto-registration is enabled"""
    if not AUTO_REGISTER_APIS:
        print("Auto API registration disabled.")
        return
        
    try:
        # Import the function
        from modules.openapi_tools_functions import register_all_default_apis
        
        print("🔄 Automatically registering default APIs...")
        result = register_all_default_apis()
        
        if result["status"] == "success":
            print(f"✅ {result['message']}")
        else:
            print(f"⚠️ {result['message']}")
            
    except ImportError as e:
        print(f"⚠️ Failed to auto-register APIs: {str(e)}")
    except Exception as e:
        print(f"⚠️ Error during API registration: {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Llama 4 Agent with Kernel Protection")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--model", type=str, help="Model to use (overrides environment variable)")
    parser.add_argument("--key", type=str, help="API Key (overrides environment variable)")
    parser.add_argument("--max-steps", type=int, help="Maximum reasoning steps")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMP, help="Temperature for generation")
    parser.add_argument("--agent-id", type=str, help="Specify agent ID")
    parser.add_argument("--register-apis", action="store_true", help="Force registration of default APIs")
    parser.add_argument("--no-register-apis", action="store_true", help="Skip registration of default APIs")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Override settings from command line if provided
    if args.model:
        MODEL = args.model
    if args.key:
        OPENROUTER_API_KEY = args.key
    if args.max_steps:
        MAX_ITERATIONS = args.max_steps
    if args.register_apis:
        AUTO_REGISTER_APIS = True
    if args.no_register_apis:
        AUTO_REGISTER_APIS = False
    if args.debug:
        logging.getLogger("llama4-agent").setLevel(logging.DEBUG)
    
    # Create instance of the enhanced Llama 4 Agent
    agent = Llama4Agent(
        agent_id=args.agent_id,
        api_key=OPENROUTER_API_KEY,
        model=MODEL,
        temperature=args.temp,
        max_tokens=2048,
        debug=args.debug
    )
    
    # Register default APIs if enabled
    register_default_apis()
    
    # Print banner
    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║  🦙 Llama 4 Agent with Protected Kernel Architecture 🔒                    ║
║    Dynamic Code Generation • Protected Core • Autonomous Reasoning       ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    
    if args.interactive:
        print("\n💡 Run /help for commands")
        
        # Main interaction loop
        while True:
            try:
                try:
                    user_input = input("\n💬 You: ")
                except EOFError:
                    print("\n👋 Goodbye!")
                    break

                # Check for exit command
                if user_input.lower() in ["exit", "quit", "/exit"]:
                    print("👋 Goodbye!")
                    break

                # Check for command
                if user_input.startswith("/"):
                    if handle_command(agent, user_input):
                        continue
                        
                # Handle chat prefix command
                if user_input.startswith("chat "):
                    user_input = user_input[5:]  # Remove "chat " prefix

                # Use streaming mode for better UX
                print("🤖 Agent is responding...")
                
                # Track conversation state
                conversation_state = {
                    "depth": 0,  # Track recursive depth
                    "has_printed_prefix": False  # Track if we've printed the prefix
                }
                
                # Create an event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Use streaming for real-time response - properly await the coroutine
                for chunk in loop.run_until_complete(agent.chat(user_input, stream=True)):
                    # Print appropriate prefix based on recursive level
                    if not conversation_state["has_printed_prefix"]:
                        print("\n🦙: ", end="", flush=True)
                        conversation_state["has_printed_prefix"] = True
                    
                    # Handle recursive chunks
                    if chunk.get("recursive", False) and conversation_state["depth"] == 0:
                        # Increase depth for formatting
                        conversation_state["depth"] += 1
                        # Print a separator and new prefix for recursive response
                        print("\n\n🔄 Continuing based on function results...\n", flush=True)
                        # Reset the prefix state for the continuation
                        conversation_state["has_printed_prefix"] = False
                        # Print new prefix
                        print("🦙: ", end="", flush=True)
                    
                    # Debug logging - only show if debug flag is set
                    if agent.debug and "debug" in chunk:
                        print(f"\n[DEBUG] {chunk['debug']}", flush=True)
                        
                    # Process different chunk types
                    if chunk["type"] == "content":
                        content = chunk["content"]
                        
                        # Filter out any special markers from the model
                        special_markers = ["<|header_start|>", "<|header_end|>", "<|python_start|>", "<|python_end|>", "assistant"]
                        for marker in special_markers:
                            if marker in content:
                                content = content.replace(marker, "")
                        
                        # More aggressive filtering of raw JSON and function call patterns
                        # These can appear especially in recursive responses
                        filter_patterns = [
                            'type": "function"',
                            '"type": "function"',
                            '"name":',
                            '"parameters":',
                            'type":',
                            '{"function":',
                            '"function":',
                            '{"tool_calls":',
                            'parameters":',
                            'name":',
                            'tool_call":'
                        ]
                        
                        # Function to check if content looks like JSON
                        def looks_like_json(text):
                            text = text.strip()
                            # Check if it starts with JSON syntax
                            if (text.startswith('{') or text.startswith('[') or
                                text.startswith('"') or text.endswith('}') or 
                                text.endswith(']')):
                                return True
                            # Check for JSON patterns
                            if any(pattern in text for pattern in filter_patterns):
                                return True
                            # Check if it has JSON key-value pattern
                            if re.search(r'"[^"]+"\s*:', text):
                                return True
                            return False
                        
                        # Skip content that appears to be JSON
                        if looks_like_json(content):
                            if agent.debug:
                                print(f"\n[DEBUG] Skipping JSON-like content: {content[:30]}...")
                            continue
                                
                        # Only print non-empty, non-whitespace content
                        if content and not content.isspace():
                            print(content, end="", flush=True)
                    elif chunk["type"] == "tool_call":
                        tool_call = chunk["tool_call"]
                        if "function" in tool_call:
                            function_info = tool_call["function"]
                            if "name" in function_info:
                                print(f"\n\n⚙️ Calling function: {function_info['name']}", flush=True)
                        elif "name" in tool_call.get("function", {}):
                            # Handle new structure format
                            print(f"\n\n⚙️ Calling function: {tool_call['function']['name']}", flush=True)
                    elif chunk["type"] == "function_result":
                        # Display function result to user
                        function_name = chunk.get('function_name', 'function')
                        content = chunk.get('content', '')
                        
                        # Try to format JSON content for better readability
                        try:
                            if content.startswith('{') or content.startswith('['):
                                data = json.loads(content)
                                formatted_content = json.dumps(data, indent=2)
                            else:
                                formatted_content = content
                        except:
                            formatted_content = content
                            
                        print(f"\n📊 Result from {function_name}:\n{formatted_content}", flush=True)
                    elif chunk["type"] == "error":
                        # Display errors
                        print(f"\n❌ Error: {chunk['content']}", flush=True)
                
                # Print a final newline
                print("\n")
                    
            except KeyboardInterrupt:
                print("\n⚠️ Operation interrupted by user. Type 'exit' to quit.")
                continue
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue
    else:
        print("\n💡 Run with --interactive flag for interactive mode")
        print("👋 Exiting non-interactive mode")
    async def parallel_process(self, func, data_chunks, cpu_bound=True):
        """
        Process data in parallel using multiple CPU cores
        
        Args:
            func: Function to execute on each data chunk
            data_chunks: List of data chunks to process
            cpu_bound: Whether this is a CPU-bound task (True) or I/O-bound (False)
            
        Returns:
            List of results from processing each chunk
        """
        if not self.use_multiprocessing:
            # Process sequentially if multiprocessing is disabled
            results = []
            for chunk in data_chunks:
                if asyncio.iscoroutinefunction(func):
                    result = await func(chunk)
                else:
                    result = func(chunk)
                results.append(result)
            return results
        
        # Process in parallel
        tasks = []
        for chunk in data_chunks:
            if cpu_bound:
                # CPU-bound tasks go to process pool
                task = self._run_in_process_pool(func, chunk)
            else:
                # I/O-bound tasks go to thread pool
                task = self._run_in_thread_pool(func, chunk)
            tasks.append(task)
        
        # Wait for all tasks to complete
        return await asyncio.gather(*tasks)
    
    async def distribute_task(self, task_type, task_data, priority=0):
        """
        Distribute a task to be processed asynchronously
        
        Args:
            task_type: Type of task
            task_data: Data for the task
            priority: Priority (0-9, higher is more important)
            
        Returns:
            Task ID if successful, None otherwise
        """
        if not self.distributed_mode or not redis_async_client:
            logger.warning("Distributed mode not available, executing task directly")
            # Execute directly if distributed mode not available
            if task_type in agent_kernel.capability_registry.capabilities:
                handler = agent_kernel.capability_registry.capabilities[task_type]["function"]
                return await self._run_in_thread_pool(handler, **task_data)
            return None
        
        try:
            # Generate task ID
            task_id = f"task_{int(time.time() * 1000)}_{task_type}_{uuid.uuid4().hex[:8]}"
            
            # Create task
            task = {
                "id": task_id,
                "type": task_type,
                "data": task_data,
                "agent_id": self.agent_id,
                "conversation_id": self.conversation_id,
                "priority": priority,
                "created_at": time.time()
            }
            
            # Store task in metadata
            async with self._metadata_lock:
                self.metadata["distributed_tasks"][task_id] = {
                    "status": "pending",
                    "created_at": time.time(),
                    "data": task_data
                }
            
            # Publish task to Redis
            await publish_message(self.pubsub_channels["agent_tasks"], task)
            
            logger.info(f"Distributed task {task_id} of type {task_type}")
            return task_id
        except Exception as e:
            logger.error(f"Failed to distribute task: {e}")
            return None
    
    async def wait_for_distributed_task(self, task_id, timeout=60):
        """
        Wait for a distributed task to complete
        
        Args:
            task_id: Task ID to wait for
            timeout: Timeout in seconds
            
        Returns:
            Task result if completed, None otherwise
        """
        if not self.distributed_mode:
            return None
            
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if task is completed
            async with self._metadata_lock:
                if task_id in self.metadata["distributed_tasks"]:
                    task = self.metadata["distributed_tasks"][task_id]
                    if task.get("status") == "completed":
                        return task.get("result")
            
            # Wait a bit before checking again
            await asyncio.sleep(0.5)
            
        logger.warning(f"Timeout waiting for task {task_id}")
        return None
