#!/usr/bin/env python3
"""
Register Tools - Register available tools with the agent kernel

This script registers all available tools with the OpenRouter kernel:
- Jupyter tools for notebook integration
- Code execution tools for safe code running
- Filesystem tools for file operations
- Jina tools for web search and fact checking
- Math utilities
"""

import os
import sys
import importlib
import json
from typing import Dict, List, Any, Optional, Tuple, Union

# Import the OpenRouter kernel
from openrouter_kernel import OpenRouterKernel

# === REGISTER JUPYTER TOOLS ===
def register_jupyter_tools(kernel):
    """Register Jupyter tools with the kernel"""
    try:
        # Import the module
        import modules.jupyter_tools as jupyter_tools
        
        # Register functions
        kernel.register_function(
            "jupyter_execute_code",
            jupyter_tools.jupyter_execute_code,
            "Execute Python code in a Jupyter kernel",
            {
                "code": {"type": "string", "description": "Python code to execute"},
                "kernel_name": {"type": "string", "description": "Name of kernel to use", "default": "default"}
            }
        )
        
        kernel.register_function(
            "jupyter_create_notebook",
            jupyter_tools.jupyter_create_notebook,
            "Create a new Jupyter notebook",
            {
                "cells": {"type": "array", "description": "List of cell dictionaries with 'type' and 'content'", "optional": True}
            }
        )
        
        kernel.register_function(
            "jupyter_open_notebook",
            jupyter_tools.jupyter_open_notebook,
            "Open a Jupyter notebook from file",
            {
                "file_path": {"type": "string", "description": "Path to notebook file"}
            }
        )
        
        kernel.register_function(
            "jupyter_save_notebook",
            jupyter_tools.jupyter_save_notebook,
            "Save a Jupyter notebook to file",
            {
                "notebook": {"type": "object", "description": "Notebook object"},
                "file_path": {"type": "string", "description": "Path to save notebook"}
            }
        )
        
        kernel.register_function(
            "jupyter_run_notebook",
            jupyter_tools.jupyter_run_notebook,
            "Run all cells in a Jupyter notebook",
            {
                "file_path": {"type": "string", "description": "Path to notebook file"},
                "kernel_name": {"type": "string", "description": "Name of kernel to use", "default": "default"}
            }
        )
        
        print("✅ Jupyter tools registered successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to register Jupyter tools: {str(e)}")
        print("   Install required packages with: pip install jupyter notebook ipykernel")
        return False

# === REGISTER CODE EXECUTION TOOLS ===
def register_code_execution_tools(kernel):
    """Register code execution tools with the kernel"""
    try:
        # Import the module
        import modules.code_execution as code_execution
        
        # Register functions
        kernel.register_function(
            "execute_code",
            code_execution.execute_code,
            "Execute code safely with resource limits",
            {
                "code": {"type": "string", "description": "Code to execute"},
                "language": {"type": "string", "description": "Programming language", "default": "python"},
                "time_limit": {"type": "integer", "description": "Maximum execution time in seconds", "default": 10},
                "memory_limit": {"type": "integer", "description": "Maximum memory usage in MB", "default": 1024}
            }
        )
        
        kernel.register_function(
            "run_script",
            code_execution.run_script,
            "Run a script file safely",
            {
                "file_path": {"type": "string", "description": "Path to the script file"},
                "time_limit": {"type": "integer", "description": "Maximum execution time in seconds", "default": 10},
                "memory_limit": {"type": "integer", "description": "Maximum memory usage in MB", "default": 1024}
            }
        )
        
        kernel.register_function(
            "test_function",
            code_execution.test_function,
            "Test a function with multiple inputs",
            {
                "function": {"type": "callable", "description": "Function to test"},
                "test_inputs": {"type": "array", "description": "List of dictionaries with args and kwargs"}
            }
        )
        
        kernel.register_function(
            "evaluate_performance",
            code_execution.evaluate_performance,
            "Benchmark code performance",
            {
                "code": {"type": "string", "description": "Code to benchmark"},
                "iterations": {"type": "integer", "description": "Number of iterations", "default": 5}
            }
        )
        
        kernel.register_function(
            "is_code_safe",
            code_execution.is_code_safe,
            "Check if code is safe to execute",
            {
                "code": {"type": "string", "description": "Code to check"}
            }
        )
        
        print("✅ Code execution tools registered successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to register code execution tools: {str(e)}")
        return False

# === REGISTER FILESYSTEM TOOLS ===
def register_filesystem_tools(kernel):
    """Register filesystem tools with the kernel"""
    try:
        # Import the module
        import modules.filesystem as filesystem
        
        # Register functions
        kernel.register_function(
            "read_file",
            filesystem.read_file,
            "Read a file safely",
            {
                "path": {"type": "string", "description": "Path to file"},
                "binary": {"type": "boolean", "description": "Whether to read in binary mode", "default": False},
                "offset": {"type": "integer", "description": "Byte/character offset to start from", "default": 0},
                "limit": {"type": "integer", "description": "Maximum bytes/characters to read", "optional": True},
                "root_dir": {"type": "string", "description": "Root directory to restrict operations to", "optional": True}
            }
        )
        
        kernel.register_function(
            "write_file",
            filesystem.write_file,
            "Write to a file safely",
            {
                "path": {"type": "string", "description": "Path to file"},
                "content": {"type": "string", "description": "Content to write"},
                "append": {"type": "boolean", "description": "Whether to append to the file", "default": False},
                "backup": {"type": "boolean", "description": "Whether to create a backup", "default": True},
                "root_dir": {"type": "string", "description": "Root directory to restrict operations to", "optional": True}
            }
        )
        
        kernel.register_function(
            "list_directory",
            filesystem.list_directory,
            "List contents of a directory",
            {
                "path": {"type": "string", "description": "Directory path"},
                "recursive": {"type": "boolean", "description": "Whether to list subdirectories", "default": False},
                "include_hidden": {"type": "boolean", "description": "Whether to include hidden files/dirs", "default": False},
                "include_details": {"type": "boolean", "description": "Whether to include detailed stats", "default": False},
                "root_dir": {"type": "string", "description": "Root directory to restrict operations to", "optional": True}
            }
        )
        
        kernel.register_function(
            "search_files",
            filesystem.search_files,
            "Search for files matching a pattern",
            {
                "path": {"type": "string", "description": "Directory path to search in"},
                "pattern": {"type": "string", "description": "Glob pattern to match"},
                "recursive": {"type": "boolean", "description": "Whether to search subdirectories", "default": True},
                "file_type": {"type": "string", "description": "Filter by file type (extension)", "optional": True},
                "root_dir": {"type": "string", "description": "Root directory to restrict operations to", "optional": True}
            }
        )
        
        kernel.register_function(
            "get_file_info",
            filesystem.get_file_info,
            "Get detailed information about a file",
            {
                "path": {"type": "string", "description": "Path to file"},
                "root_dir": {"type": "string", "description": "Root directory to restrict operations to", "optional": True}
            }
        )
        
        kernel.register_function(
            "is_path_allowed",
            filesystem.is_path_allowed,
            "Check if a path is allowed by safety controls",
            {
                "path": {"type": "string", "description": "Path to check"},
                "write_access": {"type": "boolean", "description": "Whether write access is requested", "default": False},
                "root_dir": {"type": "string", "description": "Root directory to restrict operations to", "optional": True}
            }
        )
        
        print("✅ Filesystem tools registered successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to register filesystem tools: {str(e)}")
        return False

# === REGISTER JINA TOOLS ===
def register_jina_tools(kernel):
    """Register Jina tools with the kernel"""
    try:
        # Import the module
        import modules.jina_tools as jina_tools
        
        # Register functions
        kernel.register_function(
            "jina_search",
            jina_tools.jina_search,
            "Search the web using Jina",
            {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Maximum number of results", "default": 5},
                "detailed": {"type": "boolean", "description": "Whether to include detailed info", "default": False}
            }
        )
        
        kernel.register_function(
            "jina_fact_check",
            jina_tools.jina_fact_check,
            "Fact check a claim using Jina",
            {
                "claim": {"type": "string", "description": "Claim to fact check"},
                "detailed": {"type": "boolean", "description": "Whether to include detailed info", "default": False}
            }
        )
        
        kernel.register_function(
            "jina_read_url",
            jina_tools.jina_read_url,
            "Read and extract content from a URL using Jina",
            {
                "url": {"type": "string", "description": "URL to read"},
                "query": {"type": "string", "description": "Query to find relevant content", "optional": True},
                "summarize": {"type": "boolean", "description": "Whether to summarize content", "default": False}
            }
        )
        
        kernel.register_function(
            "jina_weather",
            jina_tools.jina_weather,
            "Get weather information for a location using Jina web search",
            {
                "location": {"type": "string", "description": "City name or location to get weather for"},
                "token": {"type": "string", "description": "Optional Jina API token (uses env var if not provided)", "optional": True},
                "openai_key": {"type": "string", "description": "Optional OpenAI API key (uses env var if not provided)", "optional": True}
            }
        )
        
        print("✅ Jina tools registered successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to register Jina tools: {str(e)}")
        print("   Make sure to set JINA_API_KEY environment variable")
        return False

# === REGISTER MATH TOOLS ===
def register_math_tools(kernel):
    """Register math tools with the kernel"""
    try:
        # Import the module
        import modules.math_utils as math_utils
        
        # Register functions
        kernel.register_function(
            "calculate",
            math_utils.calculate,
            "Evaluate a mathematical expression",
            {
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
            }
        )
        
        print("✅ Math tools registered successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to register math tools: {str(e)}")
        return False

# === REGISTER OPENAPI TOOLS ===
def register_openapi_tools(kernel):
    """Register OpenAPI tools with the kernel"""
    try:
        # Import the module
        from modules.openapi_tools_functions import (
            register_openapi_service,
            list_openapi_services,
            get_openapi_service_functions,
            get_openapi_function_info,
            call_openapi_function,
            refresh_openapi_service,
            generate_openapi_client_code,
            register_tool_management_api,
            register_tool_management_api_dev,
            register_embeddings_api,
            register_dynamic_schema_api,
            register_realtime_relay_api,
            register_all_default_apis
        )
        
        # Register functions
        kernel.register_function(
            "register_openapi_service",
            register_openapi_service,
            "Register an OpenAPI service from a URL",
            {
                "url": {"type": "string", "description": "URL to the OpenAPI specification or service"},
                "name": {"type": "string", "description": "Optional name for the API", "optional": True}
            }
        )
        
        kernel.register_function(
            "list_openapi_services",
            list_openapi_services,
            "List all registered OpenAPI services",
            {}
        )
        
        kernel.register_function(
            "get_openapi_service_functions",
            get_openapi_service_functions,
            "Get functions for a specific OpenAPI service",
            {
                "api_name": {"type": "string", "description": "Name of the API"}
            }
        )
        
        kernel.register_function(
            "get_openapi_function_info",
            get_openapi_function_info,
            "Get information about a specific OpenAPI function",
            {
                "function_name": {"type": "string", "description": "Name of the function"}
            }
        )
        
        kernel.register_function(
            "call_openapi_function",
            call_openapi_function,
            "Call an OpenAPI function",
            {
                "function_name": {"type": "string", "description": "Name of the function to call"}
                # Additional parameters will be passed through
            }
        )
        
        kernel.register_function(
            "refresh_openapi_service",
            refresh_openapi_service,
            "Refresh an OpenAPI service by re-fetching its specification",
            {
                "api_name": {"type": "string", "description": "Name of the API to refresh"}
            }
        )
        
        kernel.register_function(
            "generate_openapi_client_code",
            generate_openapi_client_code,
            "Generate client code for an OpenAPI service",
            {
                "api_name": {"type": "string", "description": "Name of the API"},
                "language": {"type": "string", "description": "Programming language for the client code", "default": "python"}
            }
        )
        
        kernel.register_function(
            "register_tool_management_api",
            register_tool_management_api,
            "Register the Tool Management API (shortcut function)",
            {}
        )
        
        kernel.register_function(
            "register_tool_management_api_dev",
            register_tool_management_api_dev,
            "Register the Tool Management API Dev version (shortcut function)",
            {}
        )
        
        kernel.register_function(
            "register_embeddings_api",
            register_embeddings_api,
            "Register the Embeddings API (shortcut function)",
            {}
        )
        
        kernel.register_function(
            "register_dynamic_schema_api",
            register_dynamic_schema_api,
            "Register the Dynamic Schema API (shortcut function)",
            {}
        )
        
        kernel.register_function(
            "register_realtime_relay_api",
            register_realtime_relay_api,
            "Register the Realtime Relay API (shortcut function)",
            {}
        )
        
        kernel.register_function(
            "register_all_default_apis",
            register_all_default_apis,
            "Register all default APIs: Tool Management, Embeddings, Dynamic Schema, and Realtime Relay",
            {}
        )
        
        print("✅ OpenAPI tools registered successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to register OpenAPI tools: {str(e)}")
        return False

# === REGISTER DYNAMIC TOOLS ===
def register_dynamic_tools(kernel):
    """Register dynamic tools management functions with the kernel"""
    try:
        # Import the modules
        from modules.dynamic_tools_functions import (
            create_dynamic_tool,
            approve_dynamic_tool,
            reject_dynamic_tool,
            list_dynamic_tools,
            list_pending_dynamic_tools,
            get_dynamic_tool_code,
            execute_dynamic_tool,
            get_tool_signature
        )
        
        # Register functions
        kernel.register_function(
            "create_dynamic_tool",
            create_dynamic_tool,
            "Create a new dynamic tool from code with pre-verification",
            {
                "name": {
                    "type": "string",
                    "description": "Name for the new tool"
                },
                "code": {
                    "type": "string",
                    "description": "Python code defining the tool functions"
                },
                "description": {
                    "type": "string",
                    "description": "Description of what the tool does"
                },
                "auto_approve": {
                    "type": "boolean",
                    "description": "Whether to automatically approve the tool",
                    "default": False
                }
            }
        )
        
        kernel.register_function(
            "approve_dynamic_tool",
            approve_dynamic_tool,
            "Approve a pending dynamic tool",
            {
                "name": {
                    "type": "string",
                    "description": "Tool name to approve"
                }
            }
        )
        
        kernel.register_function(
            "reject_dynamic_tool",
            reject_dynamic_tool,
            "Reject a pending dynamic tool",
            {
                "name": {
                    "type": "string",
                    "description": "Tool name to reject"
                }
            }
        )
        
        kernel.register_function(
            "list_dynamic_tools",
            list_dynamic_tools,
            "List all registered dynamic tools",
            {}
        )
        
        kernel.register_function(
            "list_pending_dynamic_tools",
            list_pending_dynamic_tools,
            "List all pending dynamic tools",
            {}
        )
        
        kernel.register_function(
            "get_dynamic_tool_code",
            get_dynamic_tool_code,
            "Get the code for a dynamic tool",
            {
                "name": {
                    "type": "string",
                    "description": "Tool name"
                }
            }
        )
        
        kernel.register_function(
            "execute_dynamic_tool",
            execute_dynamic_tool,
            "Execute a dynamic tool",
            {
                "name": {
                    "type": "string",
                    "description": "Tool name to execute"
                }
                # Additional parameters will be passed through
            }
        )
        
        kernel.register_function(
            "get_tool_signature",
            get_tool_signature,
            "Get the signature for a dynamic tool",
            {
                "name": {
                    "type": "string",
                    "description": "Tool name"
                }
            }
        )
        
        print("✅ Dynamic tools registered successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to register dynamic tools: {str(e)}")
        return False

# === MAIN FUNCTION ===
def main():
    """Main function to register all tools"""
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY=your_api_key_here")
        return 1
        
    # Create kernel
    kernel = OpenRouterKernel(api_key)
    
    # Register tools
    results = []
    results.append(("Jupyter Tools", register_jupyter_tools(kernel)))
    results.append(("Code Execution Tools", register_code_execution_tools(kernel)))
    results.append(("Filesystem Tools", register_filesystem_tools(kernel)))
    results.append(("Jina Tools", register_jina_tools(kernel)))
    results.append(("Math Tools", register_math_tools(kernel)))
    results.append(("Dynamic Tools", register_dynamic_tools(kernel)))
    results.append(("OpenAPI Tools", register_openapi_tools(kernel)))
    
    # Print summary
    print("\n=== Tool Registration Summary ===")
    for name, success in results:
        status = "✅ Registered" if success else "❌ Failed"
        print(f"{status}: {name}")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())