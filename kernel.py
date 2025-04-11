#!/usr/bin/env python3
"""
kernel.py
---------
An auto-generative kernel that handles loading and generating all necessary components
for each instance, providing a seamless experience with minimal setup.

Features:
- Auto-detection and loading of available models and APIs
- Dynamic capability discovery and registration
- Automatic dependency management
- Self-optimization based on usage patterns
- Persistent knowledge storage with vector embeddings
"""

import os
import sys
import json
import inspect
import importlib
import importlib.util
import time
import hashlib
import sqlite3
import asyncio
import traceback
from typing import Dict, List, Any, Callable, Optional, Union, Tuple, Set
from pathlib import Path
from dataclasses import dataclass

# Try to import optional dependencies
try:
    import aiohttp
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp

# Initialize async event loop
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio"])
    import nest_asyncio
    nest_asyncio.apply()

# === CONFIGURATION MANAGER ===
class ConfigManager:
    """Manages kernel configuration with auto-detection and defaults"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'config.json')
        self.config = self._load_config()
        self._detect_api_keys()
        self._detect_available_models()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create defaults"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}, using defaults")
                
        # Default configuration
        return {
            "models": {
                "default": "auto",
                "fallback": "openai/gpt-3.5-turbo",
                "preferred_providers": ["openai", "anthropic", "meta", "together", "ollama"]
            },
            "api_endpoints": {
                "openrouter": "https://openrouter.ai/api/v1",
                "openai": "https://api.openai.com/v1",
                "anthropic": "https://api.anthropic.com/v1",
                "together": "https://api.together.xyz/v1"
            },
            "features": {
                "auto_generate": True,
                "vector_memory": True,
                "persistence": True,
                "dynamic_modules": True
            },
            "paths": {
                "modules": "modules",
                "knowledge_base": os.path.join(os.path.dirname(__file__), 'knowledge_base'),
                "cache": os.path.join(os.path.dirname(__file__), 'cache')
            }
        }
        
    def _detect_api_keys(self) -> None:
        """Auto-detect available API keys from environment variables"""
        api_keys = {}
        
        # Check for common API keys
        key_mappings = {
            "OPENAI_API_KEY": "openai",
            "ANTHROPIC_API_KEY": "anthropic", 
            "TOGETHER_API_KEY": "together",
            "OPENROUTER_API_KEY": "openrouter",
            "JINA_API_KEY": "jina",
            "COHERE_API_KEY": "cohere"
        }
        
        for env_var, provider in key_mappings.items():
            if os.environ.get(env_var):
                api_keys[provider] = os.environ.get(env_var)
                
        self.config["api_keys"] = api_keys
        
    def _detect_available_models(self) -> None:
        """Detect locally available models and potential API-accessible models"""
        available_models = {
            "local": [],
            "api": []
        }
        
        # Check for Ollama models if available
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models["local"] = [f"ollama/{model['name']}" for model in models]
        except Exception:
            pass
            
        # Add potential API models based on available keys
        if "openai" in self.config.get("api_keys", {}):
            available_models["api"].extend([
                "openai/gpt-3.5-turbo",
                "openai/gpt-4-turbo",
                "openai/gpt-4o"
            ])
            
        if "anthropic" in self.config.get("api_keys", {}):
            available_models["api"].extend([
                "anthropic/claude-3-opus",
                "anthropic/claude-3-sonnet",
                "anthropic/claude-3-haiku" 
            ])
            
        if "together" in self.config.get("api_keys", {}):
            available_models["api"].extend([
                "meta-llama/llama-4-maverick",
                "meta-llama/Llama-4-Turbo-17B-Instruct-FP8",
                "mistralai/Mixtral-8x22B-Instruct-v0.1"
            ])
            
        if "openrouter" in self.config.get("api_keys", {}):
            # OpenRouter gives access to many models
            available_models["api"].append("openrouter/auto")
            
        self.config["available_models"] = available_models
        
        # Set best available model as default
        if self.config["models"]["default"] == "auto":
            self.config["models"]["default"] = self._determine_best_model()
            
    def _determine_best_model(self) -> str:
        """Determine the best available model based on preferences"""
        # First check API models
        for provider in self.config["models"]["preferred_providers"]:
            for model in self.config["available_models"]["api"]:
                if model.startswith(f"{provider}/"):
                    return model
                    
        # Then check local models
        if self.config["available_models"]["local"]:
            return self.config["available_models"]["local"][0]
            
        # Fallback to default
        return self.config["models"]["fallback"]
        
    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
            
    def get_model(self) -> str:
        """Get the current default model"""
        return self.config["models"]["default"]
        
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific provider"""
        return self.config.get("api_keys", {}).get(provider)
        
    def get_feature_flag(self, feature: str) -> bool:
        """Get status of a feature flag"""
        return self.config.get("features", {}).get(feature, False)
        
    def get_path(self, path_key: str) -> str:
        """Get a configured path"""
        return self.config.get("paths", {}).get(path_key, "")

# === DEPENDENCY MANAGER ===
class DependencyManager:
    """Manages package dependencies with automatic installation"""
    
    def __init__(self):
        self.required_packages = {
            "core": ["aiohttp", "nest_asyncio", "pydantic"],
            "embedding": ["numpy", "sentence-transformers"],
            "visualization": ["matplotlib", "plotly"],
            "database": ["sqlalchemy", "redis"],
            "llm_clients": ["openai", "anthropic", "together"]
        }
        self.installed_packages = set(self._get_installed_packages())
        
    def _get_installed_packages(self) -> List[str]:
        """Get list of installed packages"""
        try:
            import pkg_resources
            return [pkg.key for pkg in pkg_resources.working_set]
        except ImportError:
            # Fallback if pkg_resources not available
            return []
            
    def ensure_packages(self, category: str = "core") -> bool:
        """Ensure all packages in a category are installed"""
        if category not in self.required_packages:
            return False
            
        packages_to_install = [
            pkg for pkg in self.required_packages[category] 
            if pkg.lower() not in self.installed_packages
        ]
        
        if not packages_to_install:
            return True
            
        print(f"Installing required {category} packages: {', '.join(packages_to_install)}")
        try:
            import subprocess
            for package in packages_to_install:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                self.installed_packages.add(package.lower())
            return True
        except Exception as e:
            print(f"Error installing packages: {e}")
            return False
            
    def get_package_info(self, package_name: str) -> Dict[str, Any]:
        """Get information about an installed package"""
        try:
            import pkg_resources
            package = pkg_resources.get_distribution(package_name)
            return {
                "name": package.key,
                "version": package.version,
                "location": package.location
            }
        except (ImportError, pkg_resources.DistributionNotFound):
            return {"name": package_name, "installed": False}

# === MODULE LOADER ===
class ModuleLoader:
    """Dynamic module loader with auto-registration capabilities"""
    
    def __init__(self, modules_path: str = "modules"):
        self.modules_path = modules_path
        self.loaded_modules = {}
        self.module_timestamps = {}
        
        # Create modules directory if it doesn't exist
        os.makedirs(modules_path, exist_ok=True)
        
    def discover_modules(self) -> List[str]:
        """Discover available modules in the modules directory"""
        modules = []
        
        if not os.path.exists(self.modules_path):
            return modules
            
        # List Python files in modules directory
        for file in os.listdir(self.modules_path):
            if file.endswith(".py") and not file.startswith("__"):
                modules.append(file[:-3])  # Remove .py extension
                
        return modules
        
    def load_module(self, module_name: str) -> Tuple[bool, Any]:
        """Load a module by name with automatic timestamp tracking"""
        try:
            # Check if module is in the modules directory
            module_path = os.path.join(self.modules_path, f"{module_name}.py")
            
            if os.path.exists(module_path):
                # Load from file
                mod_time = os.path.getmtime(module_path)
                
                # Check if we need to reload
                if module_name in self.loaded_modules:
                    if self.module_timestamps.get(module_name) == mod_time:
                        # Module hasn't changed, return cached version
                        return True, self.loaded_modules[module_name]
                        
                # Load or reload the module
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Cache the module
                self.loaded_modules[module_name] = module
                self.module_timestamps[module_name] = mod_time
                
                return True, module
            else:
                # Try to import from Python path
                module = importlib.import_module(module_name)
                self.loaded_modules[module_name] = module
                return True, module
                
        except Exception as e:
            traceback.print_exc()
            return False, str(e)
            
    def load_all_modules(self) -> Dict[str, Any]:
        """Load all available modules"""
        results = {}
        
        for module_name in self.discover_modules():
            success, result = self.load_module(module_name)
            results[module_name] = {
                "success": success,
                "result": result if success else str(result)
            }
            
        return results
        
    def generate_module_template(self, module_name: str, module_type: str = "tool") -> Tuple[bool, str]:
        """Generate a template module file"""
        # Define module templates
        templates = {
            "tool": """#!/usr/bin/env python3
\"\"\"
{module_name}.py - A tool module for Llama4 kernel

This module provides tools for working with {module_name_natural}.
\"\"\"

import os
import sys
from typing import Dict, List, Any, Optional

# Register functions with the kernel decorator
def register_kernel_function(name=None, description=None, schema=None):
    def decorator(func):
        func._kernel_function = {
            'name': name or func.__name__,
            'description': description or func.__doc__ or '',
            'schema': schema
        }
        return func
    return decorator

@register_kernel_function(
    description="Example function for {module_name_natural}"
)
def example_function(param1: str, param2: Optional[int] = None) -> Dict[str, Any]:
    \"\"\"Example function for {module_name_natural}\"\"\"
    return {
        "status": "success",
        "message": f"Called with {{param1}} and {{param2}}",
        "result": {
            "param1": param1,
            "param2": param2
        }
    }

# Add more functions here...
""",
            "model": """#!/usr/bin/env python3
\"\"\"
{module_name}.py - A model connector for Llama4 kernel

This module provides integration with {module_name_natural} models.
\"\"\"

import os
import sys
from typing import Dict, List, Any, Optional

class {module_name_class}Connector:
    \"\"\"Connector for {module_name_natural} models\"\"\"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("{module_name_upper}_API_KEY")
        self.api_base = "https://api.example.com/v1"
        
    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        \"\"\"Generate a response\"\"\"
        # Implementation here
        return {
            "status": "success",
            "response": f"Response to: {{prompt}}"
        }

# Register with kernel
def register_kernel_function(name=None, description=None, schema=None):
    def decorator(func):
        func._kernel_function = {
            'name': name or func.__name__,
            'description': description or func.__doc__ or '',
            'schema': schema
        }
        return func
    return decorator

@register_kernel_function(
    description="Generate a response from {module_name_natural}"
)
async def generate_response(prompt: str, **kwargs) -> Dict[str, Any]:
    \"\"\"Generate a response from {module_name_natural}\"\"\"
    connector = {module_name_class}Connector()
    return await connector.generate(prompt, **kwargs)
"""
        }
        
        if module_type not in templates:
            return False, f"Unknown module type: {module_type}"
            
        # Format template
        module_name_natural = module_name.replace('_', ' ').title()
        module_name_class = ''.join(word.capitalize() for word in module_name.split('_'))
        module_name_upper = module_name.upper()
        
        content = templates[module_type].format(
            module_name=module_name,
            module_name_natural=module_name_natural,
            module_name_class=module_name_class,
            module_name_upper=module_name_upper
        )
        
        # Write to file
        output_path = os.path.join(self.modules_path, f"{module_name}.py")
        
        try:
            with open(output_path, 'w') as f:
                f.write(content)
            return True, output_path
        except Exception as e:
            return False, str(e)

# === PERSISTENT MEMORY ===
class PersistentMemory:
    """Persistent memory system with vector embeddings"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), 'kernel_memory.db')
        self.embeddings_available = False
        self.embedding_model = None
        self._initialize_db()
        self._initialize_embeddings()
        
    def _initialize_db(self) -> None:
        """Initialize SQLite database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create memory items table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at REAL,
                access_count INTEGER DEFAULT 0,
                embedding_hash TEXT
            )
            ''')
            
            # Create embeddings table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                hash TEXT PRIMARY KEY,
                vector BLOB
            )
            ''')
            
            self.conn.commit()
        except Exception as e:
            print(f"Error initializing database: {e}")
            self.conn = None
            self.cursor = None
            
    def _initialize_embeddings(self) -> None:
        """Initialize embedding model"""
        try:
            # Try to import sentence-transformers
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            # Load a small, fast model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings_available = True
            print("Embedding model initialized successfully")
        except ImportError:
            print("Sentence Transformers not available, embeddings disabled")
            self.embeddings_available = False
            
    def _get_embedding(self, content: str) -> Optional[bytes]:
        """Get embedding for content"""
        if not self.embeddings_available or not self.embedding_model:
            return None
            
        try:
            import numpy as np
            embedding = self.embedding_model.encode(content)
            return np.array(embedding).tobytes()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
            
    def _embedding_hash(self, content: str) -> str:
        """Generate a hash for content to identify embeddings"""
        return hashlib.md5(content.encode()).hexdigest()
        
    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        """Add a memory item with embedding"""
        if not self.conn:
            return None
            
        try:
            import uuid
            memory_id = str(uuid.uuid4())
            timestamp = time.time()
            metadata_json = json.dumps(metadata or {})
            
            # Generate embedding hash
            embedding_hash = self._embedding_hash(content)
            
            # Store memory item
            self.cursor.execute(
                'INSERT INTO memory_items (id, content, metadata, created_at, embedding_hash) VALUES (?, ?, ?, ?, ?)',
                (memory_id, content, metadata_json, timestamp, embedding_hash)
            )
            
            # Store embedding if available
            if self.embeddings_available:
                embedding = self._get_embedding(content)
                if embedding:
                    try:
                        self.cursor.execute(
                            'INSERT OR REPLACE INTO embeddings (hash, vector) VALUES (?, ?)',
                            (embedding_hash, embedding)
                        )
                    except Exception as e:
                        print(f"Error storing embedding: {e}")
            
            self.conn.commit()
            return memory_id
        except Exception as e:
            print(f"Error adding memory: {e}")
            return None
            
    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memory using vector similarity if available"""
        if not self.conn:
            return []
            
        try:
            if self.embeddings_available and self.embedding_model:
                return self._vector_search(query, limit)
            else:
                return self._keyword_search(query, limit)
        except Exception as e:
            print(f"Error searching memory: {e}")
            return []
            
    def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using vector similarity"""
        try:
            import numpy as np
            
            # Get query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Get all embeddings and calculate similarity
            self.cursor.execute('SELECT hash, vector FROM embeddings')
            similarities = []
            
            for embedding_hash, vector_bytes in self.cursor.fetchall():
                try:
                    vector = np.frombuffer(vector_bytes, dtype=np.float32)
                    similarity = np.dot(query_embedding, vector) / (np.linalg.norm(query_embedding) * np.linalg.norm(vector))
                    similarities.append((embedding_hash, float(similarity)))
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top matching memory items
            results = []
            for embedding_hash, similarity in similarities[:limit]:
                self.cursor.execute(
                    'SELECT id, content, metadata, created_at, access_count FROM memory_items WHERE embedding_hash = ?',
                    (embedding_hash,)
                )
                
                for item_id, content, metadata_json, created_at, access_count in self.cursor.fetchall():
                    # Update access count
                    self.cursor.execute('UPDATE memory_items SET access_count = access_count + 1 WHERE id = ?', (item_id,))
                    
                    # Add to results
                    results.append({
                        "id": item_id,
                        "content": content,
                        "metadata": json.loads(metadata_json),
                        "created_at": created_at,
                        "access_count": access_count + 1,
                        "relevance_score": similarity
                    })
                    
                    if len(results) >= limit:
                        break
                        
                if len(results) >= limit:
                    break
                    
            self.conn.commit()
            return results
        except Exception as e:
            print(f"Error in vector search: {e}")
            return self._keyword_search(query, limit)  # Fallback
            
    def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using keywords"""
        try:
            # Simple keyword search using LIKE
            query_terms = query.lower().split()
            results = []
            
            for term in query_terms:
                self.cursor.execute(
                    'SELECT id, content, metadata, created_at, access_count FROM memory_items WHERE LOWER(content) LIKE ?',
                    (f"%{term}%",)
                )
                
                for item_id, content, metadata_json, created_at, access_count in self.cursor.fetchall():
                    # Calculate relevance based on term frequency
                    score = sum(content.lower().count(term) for term in query_terms)
                    
                    # Check if already in results
                    existing = next((r for r in results if r["id"] == item_id), None)
                    if existing:
                        # Update score if higher
                        if score > existing["relevance_score"]:
                            existing["relevance_score"] = score
                    else:
                        # Add to results
                        results.append({
                            "id": item_id,
                            "content": content,
                            "metadata": json.loads(metadata_json),
                            "created_at": created_at,
                            "access_count": access_count,
                            "relevance_score": score
                        })
                        
            # Sort by relevance score
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Update access counts
            for result in results[:limit]:
                self.cursor.execute('UPDATE memory_items SET access_count = access_count + 1 WHERE id = ?', (result["id"],))
                result["access_count"] += 1
                
            self.conn.commit()
            return results[:limit]
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
            
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get specific memory by ID"""
        if not self.conn:
            return None
            
        try:
            self.cursor.execute(
                'SELECT content, metadata, created_at, access_count FROM memory_items WHERE id = ?',
                (memory_id,)
            )
            
            result = self.cursor.fetchone()
            if not result:
                return None
                
            content, metadata_json, created_at, access_count = result
            
            # Update access count
            self.cursor.execute('UPDATE memory_items SET access_count = access_count + 1 WHERE id = ?', (memory_id,))
            self.conn.commit()
            
            return {
                "id": memory_id,
                "content": content,
                "metadata": json.loads(metadata_json),
                "created_at": created_at,
                "access_count": access_count + 1
            }
        except Exception as e:
            print(f"Error getting memory: {e}")
            return None
            
    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()

# === FUNCTION REGISTRY ===
class FunctionRegistry:
    """Registry for functions that can be called by the kernel"""
    
    def __init__(self):
        self.functions = {}
        self.function_schemas = {}
        
    def register(self, name: str, func: Callable, description: str,
                parameter_schema: Optional[Dict] = None) -> None:
        """Register a function with the kernel"""
        # Register the function
        self.functions[name] = func
        
        # Create schema if not provided
        if parameter_schema is None:
            parameter_schema = self._infer_schema(func)
            
        # Store the schema
        self.function_schemas[name] = {
            "name": name,
            "description": description,
            "parameters": parameter_schema
        }
        
    def _infer_schema(self, func: Callable) -> Dict:
        """Infer parameter schema from function signature"""
        # Get function signature
        sig = inspect.signature(func)
        schema = {}
        
        # Process each parameter
        for name, param in sig.parameters.items():
            # Skip self parameter for methods
            if name == 'self':
                continue
                
            param_schema = {
                "type": "any",
                "description": f"Parameter: {name}"
            }
            
            # Add default value if present
            if param.default is not param.empty:
                param_schema["default"] = param.default
                
            # Mark as optional if has default
            if param.default is not param.empty:
                param_schema["optional"] = True
                
            # Add to schema
            schema[name] = param_schema
            
        return schema
        
    def call(self, name: str, **kwargs) -> Any:
        """Call a registered function with validation"""
        # Check if function exists
        func = self.functions.get(name)
        if func is None:
            return {
                "status": "error",
                "message": f"Function '{name}' not found"
            }
            
        # Validate parameters
        schema = self.function_schemas.get(name)
        validation_result = self._validate_parameters(schema, kwargs)
        
        if validation_result["valid"] is False:
            return {
                "status": "error",
                "message": f"Parameter validation failed: {validation_result['message']}"
            }
            
        # Call the function
        try:
            result = func(**kwargs)
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": f"Function execution error: {str(e)}"
            }
            
    def _validate_parameters(self, schema: Dict, params: Dict) -> Dict:
        """Validate parameters against schema"""
        if not schema or "parameters" not in schema:
            # No schema to validate against
            return {"valid": True}
            
        # Check for missing required parameters
        for param_name, param_schema in schema["parameters"].items():
            if param_name not in params and not param_schema.get("optional", False):
                return {
                    "valid": False,
                    "message": f"Missing required parameter: {param_name}"
                }
                
        # All checks passed
        return {"valid": True}
        
    def list_functions(self) -> List[Dict]:
        """Get list of all registered functions"""
        return list(self.function_schemas.values())

# === MAIN KERNEL CLASS ===
class Kernel:
    """
    Auto-generative kernel that handles loading and generating
    all necessary components for a seamless experience
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the kernel with auto-configuration"""
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        
        # Initialize dependency manager
        self.dependency_manager = DependencyManager()
        self.dependency_manager.ensure_packages("core")
        
        # Initialize function registry
        self.function_registry = FunctionRegistry()
        
        # Initialize module loader
        modules_path = self.config_manager.get_path("modules")
        self.module_loader = ModuleLoader(modules_path)
        
        # Initialize persistent memory if enabled
        self.memory = None
        if self.config_manager.get_feature_flag("persistence"):
            self.memory = PersistentMemory()
            
        # Register kernel functions
        self._register_kernel_functions()
        
        # Load all available modules if dynamic modules enabled
        if self.config_manager.get_feature_flag("dynamic_modules"):
            self.module_loader.load_all_modules()
            
    def _register_kernel_functions(self) -> None:
        """Register built-in kernel functions"""
        # Register kernel info function
        self.function_registry.register(
            "kernel_info",
            self.get_kernel_info,
            "Get information about the kernel",
            {}
        )
        
        # Register list functions function
        self.function_registry.register(
            "list_functions",
            self.list_functions,
            "List all registered functions",
            {}
        )
        
        # Register memory functions if available
        if self.memory:
            self.function_registry.register(
                "add_memory",
                self.add_memory,
                "Add an item to memory",
                {
                    "content": {"type": "string", "description": "Content to remember"},
                    "metadata": {"type": "object", "description": "Optional metadata", "optional": True}
                }
            )
            
            self.function_registry.register(
                "search_memory",
                self.search_memory,
                "Search memory for relevant items",
                {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum results to return", "optional": True}
                }
            )
            
        # Register module functions
        self.function_registry.register(
            "list_modules",
            self.list_modules,
            "List available modules",
            {}
        )
        
        self.function_registry.register(
            "load_module",
            self.load_module,
            "Load a module",
            {
                "module_name": {"type": "string", "description": "Name of module to load"}
            }
        )
        
        self.function_registry.register(
            "generate_module",
            self.generate_module,
            "Generate a new module from template",
            {
                "module_name": {"type": "string", "description": "Name for the new module"},
                "module_type": {"type": "string", "description": "Type of module (tool, model)", "optional": True}
            }
        )
        
    def get_kernel_info(self) -> Dict[str, Any]:
        """Get information about the kernel"""
        return {
            "status": "success",
            "version": "1.0.0",
            "config": {
                "model": self.config_manager.get_model(),
                "features": {k: self.config_manager.get_feature_flag(k) 
                            for k in self.config_manager.config.get("features", {})},
                "available_models": self.config_manager.config.get("available_models", {})
            },
            "functions": len(self.function_registry.list_functions()),
            "modules": len(self.module_loader.discover_modules()),
            "memory": self.memory is not None
        }
        
    def list_functions(self) -> Dict[str, Any]:
        """List all registered functions"""
        return {
            "status": "success",
            "functions": self.function_registry.list_functions()
        }
        
    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add an item to memory"""
        if not self.memory:
            return {
                "status": "error",
                "message": "Memory is not enabled"
            }
            
        memory_id = self.memory.add_memory(content, metadata)
        
        if memory_id:
            return {
                "status": "success",
                "memory_id": memory_id
            }
        else:
            return {
                "status": "error",
                "message": "Failed to add memory"
            }
            
    def search_memory(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search memory for relevant items"""
        if not self.memory:
            return {
                "status": "error",
                "message": "Memory is not enabled"
            }
            
        results = self.memory.search_memory(query, limit)
        
        return {
            "status": "success",
            "results": results,
            "count": len(results)
        }
        
    def list_modules(self) -> Dict[str, Any]:
        """List available modules"""
        modules = self.module_loader.discover_modules()
        
        return {
            "status": "success",
            "modules": modules,
            "count": len(modules)
        }
        
    def load_module(self, module_name: str) -> Dict[str, Any]:
        """Load a module and register its functions"""
        # Load the module
        success, result = self.module_loader.load_module(module_name)
        
        if not success:
            return {
                "status": "error",
                "message": f"Failed to load module: {result}"
            }
            
        # Register functions with register_kernel_function decorator
        functions_registered = 0
        
        # Get all module attributes
        for name, obj in inspect.getmembers(result):
            # Skip private attributes
            if name.startswith('_'):
                continue
                
            # Check if attribute has register_kernel_function metadata
            if hasattr(obj, '_kernel_function'):
                # Register the function
                func_name = obj._kernel_function.get('name', name)
                description = obj._kernel_function.get('description', '')
                schema = obj._kernel_function.get('schema', None)
                
                self.function_registry.register(func_name, obj, description, schema)
                functions_registered += 1
                
        return {
            "status": "success",
            "module": module_name,
            "functions_registered": functions_registered
        }
        
    def generate_module(self, module_name: str, module_type: str = "tool") -> Dict[str, Any]:
        """Generate a new module from template"""
        success, result = self.module_loader.generate_module_template(module_name, module_type)
        
        if success:
            return {
                "status": "success",
                "module": module_name,
                "path": result,
                "type": module_type
            }
        else:
            return {
                "status": "error",
                "message": result
            }
            
    def call_function(self, name: str, **kwargs) -> Any:
        """Call a registered function"""
        return self.function_registry.call(name, **kwargs)
        
    def register_function(self, name: str, func: Callable, description: str,
                        parameter_schema: Optional[Dict] = None) -> None:
        """Register a function with the kernel"""
        self.function_registry.register(name, func, description, parameter_schema)
        
    async def chat(self, prompt: str, model: Optional[str] = None):
        """
        Generate a chat response using the best available model
        
        This is a placeholder for actual implementation. In a complete implementation,
        this would dynamically select and use the best available model based on
        the config and available APIs.
        """
        # Determine which model to use
        model = model or self.config_manager.get_model()
        
        # Simple mock implementation
        return {
            "status": "success",
            "model": model,
            "response": f"This is a placeholder response to: {prompt}"
        }
        
    def close(self) -> None:
        """Clean up resources"""
        if self.memory:
            self.memory.close()

# === FUNCTION DECORATOR ===
def register_kernel_function(name: Optional[str] = None, 
                           description: Optional[str] = None,
                           schema: Optional[Dict] = None):
    """
    Decorator to register a function with the kernel
    
    Args:
        name: Function name override
        description: Function description
        schema: Parameter schema
        
    Returns:
        Decorator function
    """
    def decorator(func):
        # Store kernel function metadata
        func._kernel_function = {
            'name': name or func.__name__,
            'description': description or inspect.getdoc(func) or '',
            'schema': schema
        }
        return func
    return decorator

# === MAIN FUNCTION ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-generative Kernel")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--info", action="store_true", help="Show kernel info")
    parser.add_argument("--list-functions", action="store_true", help="List available functions")
    parser.add_argument("--list-modules", action="store_true", help="List available modules")
    parser.add_argument("--load-module", help="Module to load")
    parser.add_argument("--generate-module", help="Generate a new module")
    parser.add_argument("--module-type", default="tool", help="Type of module to generate")
    
    args = parser.parse_args()
    
    # Create kernel
    kernel = Kernel(args.config)
    
    if args.info:
        info = kernel.get_kernel_info()
        print(json.dumps(info, indent=2))
        
    if args.list_functions:
        functions = kernel.list_functions()
        print(json.dumps(functions, indent=2))
        
    if args.list_modules:
        modules = kernel.list_modules()
        print(json.dumps(modules, indent=2))
        
    if args.load_module:
        result = kernel.load_module(args.load_module)
        print(json.dumps(result, indent=2))
        
    if args.generate_module:
        result = kernel.generate_module(args.generate_module, args.module_type)
        print(json.dumps(result, indent=2))
        
    # Clean up
    kernel.close()