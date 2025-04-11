#!/usr/bin/env python3
"""
OpenRouter Kernel - Core agent kernel with robust function registry

An extensible kernel for the Llama4 agent system that provides:
- Protected kernel with safe dynamic extension
- Function registry with parameter validation
- Hot module reloading
- Connection to OpenRouter API for Llama 4 access
- Dynamic system prompt enhancement
- Automatic capability inference
"""

import os
import sys
import json
import inspect
import importlib
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

import requests
import aiohttp
import nest_asyncio
nest_asyncio.apply()

# Optional capability for dynamic prompt enhancement
try:
    from modules.prompt_enhancer import enhance_system_prompt, PromptManager
    PROMPT_ENHANCEMENT_AVAILABLE = True
except ImportError:
    PROMPT_ENHANCEMENT_AVAILABLE = False

# === FUNCTION REGISTRY ===
class FunctionRegistry:
    """Manages functions that can be called by the agent"""
    def __init__(self):
        self.functions = {}
        self.function_schemas = {}
        
    def register(self, name: str, func: Callable, description: str,
                parameter_schema: Optional[Dict] = None) -> None:
        """
        Register a function with the kernel
        
        Args:
            name: Function name for the registry
            func: Function to register
            description: Function description
            parameter_schema: Optional schema for parameters
        """
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
        
    def unregister(self, name: str) -> bool:
        """
        Remove a function from the registry
        
        Args:
            name: Function name to remove
            
        Returns:
            Whether function was found and removed
        """
        if name in self.functions:
            del self.functions[name]
            del self.function_schemas[name]
            return True
        return False
        
    def get(self, name: str) -> Optional[Callable]:
        """
        Get a function by name
        
        Args:
            name: Function name
            
        Returns:
            Function or None if not found
        """
        return self.functions.get(name)
        
    def get_schema(self, name: str) -> Optional[Dict]:
        """
        Get function schema by name
        
        Args:
            name: Function name
            
        Returns:
            Schema or None if not found
        """
        return self.function_schemas.get(name)
        
    def list_functions(self) -> List[Dict]:
        """
        Get list of all registered functions
        
        Returns:
            List of function schemas
        """
        return list(self.function_schemas.values())
        
    def _infer_schema(self, func: Callable) -> Dict:
        """
        Infer parameter schema from function signature
        
        Args:
            func: Function to analyze
            
        Returns:
            Parameter schema
        """
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
        """
        Call a registered function with validation
        
        Args:
            name: Function name
            **kwargs: Parameters for the function
            
        Returns:
            Function result
        """
        # Check if function exists
        func = self.get(name)
        if func is None:
            return {
                "status": "error",
                "message": f"Function '{name}' not found"
            }
            
        # Validate parameters
        schema = self.get_schema(name)
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
        """
        Validate parameters against schema
        
        Args:
            schema: Function schema
            params: Parameters to validate
            
        Returns:
            Validation result
        """
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
                
        # Check for unexpected parameters
        for param_name in params:
            if param_name not in schema["parameters"]:
                return {
                    "valid": False,
                    "message": f"Unexpected parameter: {param_name}"
                }
                
        # All checks passed
        return {"valid": True}

# === CODE VALIDATION ===
class CodeValidator:
    """Validates code for potential security issues"""
    def __init__(self):
        # Define patterns that should be blocked
        self.blocked_patterns = [
            "os.system", "subprocess.call", "subprocess.Popen",
            "exec(", "eval(", "__import__(",
            "open(", "file.write", "file.close"
        ]
        
        # Define modules that should be blocked
        self.blocked_modules = [
            "subprocess", "pty", "socket", "ctypes"
        ]
        
    def validate(self, code: str) -> Dict:
        """
        Validate code for security issues
        
        Args:
            code: Python code to validate
            
        Returns:
            Dictionary with validation result
        """
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern in code:
                return {
                    "valid": False,
                    "message": f"Code contains blocked pattern: {pattern}"
                }
                
        # Check for blocked module imports
        for module in self.blocked_modules:
            if f"import {module}" in code or f"from {module}" in code:
                return {
                    "valid": False,
                    "message": f"Code imports blocked module: {module}"
                }
                
        # Basic syntax check
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return {
                "valid": False,
                "message": f"Syntax error: {str(e)}"
            }
            
        # All checks passed
        return {"valid": True}

# === MODULE MANAGER ===
class ModuleManager:
    """Manages dynamic module loading and hot reloading"""
    def __init__(self, modules_dir: str = "modules"):
        self.modules_dir = modules_dir
        self.loaded_modules = {}
        self.module_timestamps = {}
        
        # Create modules directory if it doesn't exist
        if not os.path.exists(modules_dir):
            os.makedirs(modules_dir)
            
    def load_module(self, module_name: str) -> Tuple[bool, Any]:
        """
        Load a module by name
        
        Args:
            module_name: Name of module to load
            
        Returns:
            Tuple of (success, module/error)
        """
        try:
            # Check if module is in the modules directory
            module_path = os.path.join(self.modules_dir, f"{module_name}.py")
            
            if os.path.exists(module_path):
                # Load from file
                return self._load_from_file(module_name, module_path)
            else:
                # Try to import from Python path
                return self._load_from_import(module_name)
        except Exception as e:
            return False, str(e)
            
    def _load_from_file(self, module_name: str, module_path: str) -> Tuple[bool, Any]:
        """
        Load module from file
        
        Args:
            module_name: Name of module
            module_path: Path to module file
            
        Returns:
            Tuple of (success, module/error)
        """
        try:
            # Get file modification time
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
        except Exception as e:
            return False, str(e)
            
    def _load_from_import(self, module_name: str) -> Tuple[bool, Any]:
        """
        Load module from Python path
        
        Args:
            module_name: Name of module
            
        Returns:
            Tuple of (success, module/error)
        """
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            
            # Cache the module
            self.loaded_modules[module_name] = module
            self.module_timestamps[module_name] = time.time()
            
            return True, module
        except ImportError as e:
            return False, str(e)
            
    def reload_module(self, module_name: str) -> Tuple[bool, Any]:
        """
        Reload a module by name
        
        Args:
            module_name: Name of module to reload
            
        Returns:
            Tuple of (success, module/error)
        """
        # Remove from cache
        if module_name in self.loaded_modules:
            del self.loaded_modules[module_name]
            
        if module_name in self.module_timestamps:
            del self.module_timestamps[module_name]
            
        # Load the module again
        return self.load_module(module_name)
        
    def list_modules(self) -> List[str]:
        """
        Get list of available modules
        
        Returns:
            List of module names
        """
        if not os.path.exists(self.modules_dir):
            return []
            
        modules = []
        
        # List Python files in modules directory
        for file in os.listdir(self.modules_dir):
            if file.endswith(".py") and not file.startswith("__"):
                modules.append(file[:-3])  # Remove .py extension
                
        return modules

# === OPENROUTER KERNEL ===
class OpenRouterKernel:
    """
    Main kernel for OpenRouter-based agent system
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/llama-4-maverick"):
        """
        Initialize the kernel
        
        Args:
            api_key: OpenRouter API key
            model: Model to use
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model
        self.api_base = "https://openrouter.ai/api/v1"
        
        # Create function registry
        self.function_registry = FunctionRegistry()
        
        # Create code validator
        self.code_validator = CodeValidator()
        
        # Create module manager
        self.module_manager = ModuleManager()
        
        # Create prompt manager if available
        self.prompt_manager = None
        if PROMPT_ENHANCEMENT_AVAILABLE:
            self.prompt_manager = PromptManager(self.default_system_prompt())
            
        # Register built-in functions
        self._register_built_ins()
        
    def default_system_prompt(self) -> str:
        """
        Get the default system prompt
        
        Returns:
            Default system prompt
        """
        return """You are an advanced AI assistant powered by Llama 4 Maverick.
You have access to a variety of tools and capabilities that help you assist the user.
You can execute code, manage files, search for information, and much more.
Always be helpful, accurate, and responsive to the user's needs."""
        
    def _register_built_ins(self) -> None:
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
        
        # Register load module function
        self.function_registry.register(
            "load_module",
            self.load_module,
            "Load a module",
            {
                "module_name": {"type": "string", "description": "Name of module to load"}
            }
        )
        
        # Register list modules function
        self.function_registry.register(
            "list_modules",
            self.list_modules,
            "List available modules",
            {}
        )
        
        # Register enhance prompt function if available
        if PROMPT_ENHANCEMENT_AVAILABLE:
            self.function_registry.register(
                "enhance_system_prompt",
                self.enhance_system_prompt,
                "Enhance the system prompt with capabilities",
                {
                    "context": {"type": "string", "description": "Optional context for prompt", "optional": True}
                }
            )
            
    def register_function(self, name: str, func: Callable, description: str,
                         parameter_schema: Optional[Dict] = None) -> None:
        """
        Register a function with the kernel
        
        Args:
            name: Function name for the registry
            func: Function to register
            description: Function description
            parameter_schema: Optional schema for parameters
        """
        self.function_registry.register(name, func, description, parameter_schema)
        
    def call_function(self, name: str, **kwargs) -> Any:
        """
        Call a registered function
        
        Args:
            name: Function name
            **kwargs: Parameters for the function
            
        Returns:
            Function result
        """
        return self.function_registry.call(name, **kwargs)
        
    def get_kernel_info(self) -> Dict:
        """
        Get information about the kernel
        
        Returns:
            Dictionary with kernel info
        """
        return {
            "status": "success",
            "model": self.model,
            "functions": len(self.function_registry.list_functions()),
            "modules": len(self.module_manager.list_modules()),
            "prompt_enhancement": PROMPT_ENHANCEMENT_AVAILABLE
        }
        
    def list_functions(self) -> Dict:
        """
        List all registered functions
        
        Returns:
            Dictionary with function list
        """
        return {
            "status": "success",
            "functions": self.function_registry.list_functions()
        }
        
    def load_module(self, module_name: str) -> Dict:
        """
        Load a module and register its functions
        
        Args:
            module_name: Name of module to load
            
        Returns:
            Dictionary with load result
        """
        # Load the module
        success, result = self.module_manager.load_module(module_name)
        
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
                
                self.register_function(func_name, obj, description, schema)
                functions_registered += 1
                
        # Update prompt manager with new capabilities
        if self.prompt_manager and PROMPT_ENHANCEMENT_AVAILABLE:
            self.prompt_manager.scan_for_capabilities()
                
        return {
            "status": "success",
            "module": module_name,
            "functions_registered": functions_registered
        }
        
    def list_modules(self) -> Dict:
        """
        List available modules
        
        Returns:
            Dictionary with module list
        """
        modules = self.module_manager.list_modules()
        
        return {
            "status": "success",
            "modules": modules
        }
        
    def enhance_system_prompt(self, context: Optional[str] = None) -> Dict:
        """
        Enhance the system prompt with capabilities
        
        Args:
            context: Optional conversation context
            
        Returns:
            Dictionary with enhanced prompt
        """
        if not PROMPT_ENHANCEMENT_AVAILABLE:
            return {
                "status": "error",
                "message": "Prompt enhancement not available"
            }
            
        if self.prompt_manager:
            # Scan for capabilities
            self.prompt_manager.scan_for_capabilities()
            
            # Generate enhanced prompt
            enhanced_prompt = self.prompt_manager.generate_prompt(context)
            
            return {
                "status": "success",
                "prompt": enhanced_prompt
            }
        else:
            # Use the standalone function
            try:
                enhanced_prompt = enhance_system_prompt(
                    self.default_system_prompt(), 
                    context
                )
                
                return {
                    "status": "success",
                    "prompt": enhanced_prompt
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Error enhancing prompt: {str(e)}"
                }
                
    def chat(self, prompt: str, system_prompt: Optional[str] = None, 
             functions: Optional[List[Dict]] = None) -> Dict:
        """
        Send a chat request to the OpenRouter API
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            functions: Optional function definitions
            
        Returns:
            OpenRouter API response
        """
        if not self.api_key:
            return {
                "status": "error",
                "message": "OpenRouter API key not set"
            }
            
        # Use enhanced system prompt if available
        if system_prompt is None and self.prompt_manager and PROMPT_ENHANCEMENT_AVAILABLE:
            system_prompt = self.prompt_manager.generate_prompt()
        elif system_prompt is None:
            system_prompt = self.default_system_prompt()
            
        # Prepare request
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
        }
        
        # Add functions if provided
        if functions:
            data["functions"] = functions
            
        # Send request
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {
                "status": "error",
                "message": f"API request failed: {str(e)}"
            }
            
    async def async_chat(self, prompt: str, system_prompt: Optional[str] = None,
                        functions: Optional[List[Dict]] = None) -> Dict:
        """
        Send an async chat request to the OpenRouter API
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            functions: Optional function definitions
            
        Returns:
            OpenRouter API response
        """
        if not self.api_key:
            return {
                "status": "error",
                "message": "OpenRouter API key not set"
            }
            
        # Use enhanced system prompt if available
        if system_prompt is None and self.prompt_manager and PROMPT_ENHANCEMENT_AVAILABLE:
            system_prompt = self.prompt_manager.generate_prompt()
        elif system_prompt is None:
            system_prompt = self.default_system_prompt()
            
        # Prepare request
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
        }
        
        # Add functions if provided
        if functions:
            data["functions"] = functions
            
        # Send request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        return {
                            "status": "error",
                            "message": f"API request failed with status {response.status}"
                        }
                    return await response.json()
        except Exception as e:
            return {
                "status": "error",
                "message": f"API request failed: {str(e)}"
            }
            
    def stream_chat(self, prompt: str, system_prompt: Optional[str] = None,
                  functions: Optional[List[Dict]] = None):
        """
        Stream a chat response from the OpenRouter API
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            functions: Optional function definitions
            
        Yields:
            Response chunks
        """
        if not self.api_key:
            yield {
                "type": "error",
                "content": "OpenRouter API key not set"
            }
            return
            
        # Use enhanced system prompt if available
        if system_prompt is None and self.prompt_manager and PROMPT_ENHANCEMENT_AVAILABLE:
            system_prompt = self.prompt_manager.generate_prompt()
        elif system_prompt is None:
            system_prompt = self.default_system_prompt()
            
        # Prepare request
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True
        }
        
        # Add functions if provided
        if functions:
            data["functions"] = functions
            
        # Send request
        try:
            response = requests.post(url, headers=headers, json=data, stream=True)
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    # Skip "data: " prefix
                    if line.startswith(b"data: "):
                        line = line[6:]
                        
                    # Skip empty lines and "[DONE]"
                    if line and line != b"[DONE]":
                        try:
                            chunk = json.loads(line)
                            
                            if "choices" in chunk and chunk["choices"]:
                                choice = chunk["choices"][0]
                                
                                if "delta" in choice:
                                    delta = choice["delta"]
                                    
                                    # Handle content
                                    if "content" in delta and delta["content"]:
                                        yield {
                                            "type": "content",
                                            "content": delta["content"]
                                        }
                                        
                                    # Handle function call
                                    if "function_call" in delta:
                                        yield {
                                            "type": "function_call",
                                            "function_call": delta["function_call"]
                                        }
                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            pass
        except Exception as e:
            yield {
                "type": "error",
                "content": f"API request failed: {str(e)}"
            }

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
    
    parser = argparse.ArgumentParser(description="OpenRouter Kernel")
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--model", default="meta-llama/llama-4-maverick", help="Model to use")
    parser.add_argument("--prompt", help="Test prompt")
    parser.add_argument("--system-prompt", help="System prompt file")
    parser.add_argument("--load-module", help="Module to load")
    parser.add_argument("--list-modules", action="store_true", help="List available modules")
    parser.add_argument("--list-functions", action="store_true", help="List registered functions")
    parser.add_argument("--enhance-prompt", action="store_true", help="Enhance system prompt")
    
    args = parser.parse_args()
    
    # Create kernel
    kernel = OpenRouterKernel(api_key=args.api_key, model=args.model)
    
    # Load module if specified
    if args.load_module:
        result = kernel.load_module(args.load_module)
        print(json.dumps(result, indent=2))
        
    # List modules if requested
    if args.list_modules:
        result = kernel.list_modules()
        print(json.dumps(result, indent=2))
        
    # List functions if requested
    if args.list_functions:
        result = kernel.list_functions()
        print(json.dumps(result, indent=2))
        
    # Enhance prompt if requested
    if args.enhance_prompt:
        result = kernel.enhance_system_prompt()
        if result["status"] == "success":
            print(result["prompt"])
        else:
            print(json.dumps(result, indent=2))
        
    # Send test prompt if specified
    if args.prompt:
        system_prompt = None
        
        # Load system prompt from file if specified
        if args.system_prompt:
            try:
                with open(args.system_prompt, 'r') as f:
                    system_prompt = f.read()
            except Exception as e:
                print(f"Error reading system prompt: {e}")
                sys.exit(1)
                
        # Send the prompt
        for chunk in kernel.stream_chat(args.prompt, system_prompt):
            if chunk["type"] == "content":
                print(chunk["content"], end="")
            elif chunk["type"] == "error":
                print(f"\nError: {chunk['content']}")
                
        print()  # Add newline at end