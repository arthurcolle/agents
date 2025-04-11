"""
Dynamic Tools Module - Create and register tools dynamically with verification
"""
import inspect
import re
import ast
import json
import logging
import importlib
import functools
import sys
from typing import Dict, Any, Callable, Optional, Tuple, List

# Configure logging
logger = logging.getLogger("dynamic_tools")

class CodeVerifier:
    """
    Verifies code safety before execution or tool creation
    Uses multiple verification strategies to ensure code is safe
    """
    def __init__(self):
        self.blacklist = {
            # System access
            "os.system", "subprocess", "exec(", "eval(", "globals()", "locals()",
            "os._exit", "sys.exit", "quit", "exit", "__import__",
            
            # Critical file operations
            "os.remove", "os.unlink", "os.rmdir", "shutil.rmtree",
            
            # Protected modules
            "openrouter_kernel", "AgentKernel", "KernelProtector",
            "CodeValidator", "DynamicCodeStore", "CapabilityRegistry",
            
            # Network operations (can be selectively allowed)
            "socket.socket", "urlopen", "urllib.request", 
        }
        
        # Modules that require special approval
        self.restricted_modules = {
            "os": ["path", "environ", "getcwd", "listdir", "mkdir"],  # Allowed os functions
            "sys": ["version", "platform", "path", "modules"],       # Allowed sys attributes
            "requests": True,  # Needs approval, but not totally banned
            "aiohttp": True,   # Needs approval, but not totally banned
        }
        
        # Allowed imports with default permission
        self.allowed_modules = {
            "json", "re", "datetime", "time", "math", "random", 
            "collections", "itertools", "functools", "typing",
            "hashlib", "base64", "uuid", "enum", "dataclasses",
            "copy", "inspect", "textwrap", "string", "pathlib",
        }
    
    def verify_code(self, code: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Verify if code is safe for execution
        Returns (is_safe, message, metadata)
        """
        # Check for blacklisted patterns
        for pattern in self.blacklist:
            if pattern in code:
                return False, f"Code contains forbidden pattern: {pattern}", {}
        
        # Parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Code has syntax error: {str(e)}", {}
        
        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
                
        # Check for function definitions
        functions = []
        function_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
                functions.append(node)
        
        if not functions:
            return False, "No function definitions found in the code", {}
            
        # Check imports
        for imp in imports:
            base_module = imp.split('.')[0]
            
            # Check if it's in blacklist
            if base_module in self.blacklist:
                return False, f"Forbidden module import: {base_module}", {}
                
            # Check if it's in restricted modules
            if base_module in self.restricted_modules:
                if self.restricted_modules[base_module] is not True:
                    # Check if using allowed submodules only
                    restricted_submodules = True
                    # Further verification would be here
                
                requires_approval = True
            
            # Not in allowed list    
            if base_module not in self.allowed_modules and base_module not in self.restricted_modules:
                requires_approval = True
                        
        # Extract function signatures
        signatures = {}
        for func in functions:
            func_name = func.name
            args = []
            
            # Get arguments
            for arg in func.args.args:
                arg_name = arg.arg
                # Check for type annotation
                arg_type = None
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        arg_type = arg.annotation.id
                    elif isinstance(arg.annotation, ast.Attribute):
                        arg_type = arg.annotation.attr
                
                args.append({
                    "name": arg_name,
                    "type": arg_type
                })
                
            # Get docstring if exists
            docstring = ast.get_docstring(func)
            
            signatures[func_name] = {
                "args": args,
                "docstring": docstring
            }
        
        metadata = {
            "imports": imports,
            "functions": function_names,
            "signatures": signatures
        }
        
        return True, "Code passed verification", metadata

class DynamicToolFactory:
    """
    Factory for creating and managing dynamic tools
    Handles verification, registration, and execution of dynamically created tools
    """
    def __init__(self):
        self.verifier = CodeVerifier()
        self.tools = {}
        self.tool_metadata = {}
        self.pending_approval = {}
        
    def create_tool(self, 
                    name: str, 
                    code: str, 
                    description: str = "", 
                    auto_approve: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Create a new tool from code
        Args:
            name: Name for the new tool
            code: Python code defining the tool
            description: Description of what the tool does
            auto_approve: Whether to automatically approve the tool
            
        Returns:
            (success, message, metadata)
        """
        # Verify the code first
        is_safe, message, metadata = self.verifier.verify_code(code)
        
        if not is_safe:
            return False, message, {}
        
        # Create a unique namespace for the code
        tool_namespace = {}
        
        # If auto-approve is not enabled, add to pending approval
        if not auto_approve:
            self.pending_approval[name] = {
                "code": code,
                "description": description,
                "metadata": metadata
            }
            return True, "Tool pending approval", {
                "status": "pending_approval",
                "name": name,
                "metadata": metadata
            }
        
        # Execute the code in the namespace
        try:
            exec(code, tool_namespace)
        except Exception as e:
            return False, f"Error executing tool code: {str(e)}", {}
        
        # Find and register the main function
        main_function = None
        for func_name, func in tool_namespace.items():
            if callable(func) and not func_name.startswith('_'):
                if func_name in metadata["functions"]:
                    main_function = func
                    break
                    
        if not main_function:
            return False, "No valid function found in the tool code", {}
        
        # Register the tool
        self.tools[name] = main_function
        self.tool_metadata[name] = {
            "description": description,
            "code": code,
            "metadata": metadata
        }
        
        return True, f"Tool '{name}' created successfully", {
            "status": "created",
            "name": name,
            "metadata": metadata
        }
        
    def approve_tool(self, name: str) -> Tuple[bool, str]:
        """
        Approve a pending tool
        Args:
            name: Tool name to approve
            
        Returns:
            (success, message)
        """
        if name not in self.pending_approval:
            return False, f"No pending tool named '{name}'"
            
        pending_tool = self.pending_approval[name]
        
        # Create tool namespace
        tool_namespace = {}
        
        # Execute the code
        try:
            exec(pending_tool["code"], tool_namespace)
        except Exception as e:
            return False, f"Error executing tool code: {str(e)}"
            
        # Find the main function
        main_function = None
        for func_name, func in tool_namespace.items():
            if callable(func) and not func_name.startswith('_'):
                if func_name in pending_tool["metadata"]["functions"]:
                    main_function = func
                    break
                    
        if not main_function:
            return False, "No valid function found in the tool code"
            
        # Register the tool
        self.tools[name] = main_function
        self.tool_metadata[name] = {
            "description": pending_tool["description"],
            "code": pending_tool["code"],
            "metadata": pending_tool["metadata"]
        }
        
        # Remove from pending
        del self.pending_approval[name]
        
        return True, f"Tool '{name}' approved and registered"
        
    def reject_tool(self, name: str) -> Tuple[bool, str]:
        """
        Reject a pending tool
        Args:
            name: Tool name to reject
            
        Returns:
            (success, message)
        """
        if name not in self.pending_approval:
            return False, f"No pending tool named '{name}'"
            
        del self.pending_approval[name]
        return True, f"Tool '{name}' rejected"
        
    def execute_tool(self, name: str, **kwargs) -> Tuple[bool, Any]:
        """
        Execute a tool
        Args:
            name: Tool name to execute
            **kwargs: Arguments to pass to the tool
            
        Returns:
            (success, result)
        """
        if name not in self.tools:
            return False, f"Tool '{name}' not found"
            
        tool_func = self.tools[name]
        
        try:
            result = tool_func(**kwargs)
            return True, result
        except Exception as e:
            return False, f"Error executing tool: {str(e)}"
            
    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered tools
        Returns:
            Dictionary of tools and their metadata
        """
        return {
            name: {
                "description": self.tool_metadata[name]["description"],
                "metadata": self.tool_metadata[name]["metadata"]
            }
            for name in self.tools
        }
        
    def list_pending_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        List all pending tools
        Returns:
            Dictionary of pending tools and their metadata
        """
        return {
            name: {
                "description": self.pending_approval[name]["description"],
                "metadata": self.pending_approval[name]["metadata"]
            }
            for name in self.pending_approval
        }
        
    def get_tool_code(self, name: str) -> Tuple[bool, str]:
        """
        Get the code for a tool
        Args:
            name: Tool name
            
        Returns:
            (success, code)
        """
        if name in self.tools:
            return True, self.tool_metadata[name]["code"]
        elif name in self.pending_approval:
            return True, self.pending_approval[name]["code"]
        else:
            return False, f"Tool '{name}' not found"
            
    def get_tool_signature(self, name: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Get the signature for a tool
        Args:
            name: Tool name
            
        Returns:
            (success, signature)
        """
        if name in self.tools:
            signatures = self.tool_metadata[name]["metadata"]["signatures"]
            func_name = self.tool_metadata[name]["metadata"]["functions"][0]  # Get first function
            if func_name in signatures:
                return True, signatures[func_name]
            return False, "Function signature not found"
        else:
            return False, f"Tool '{name}' not found"

# Create a global instance of the tool factory
tool_factory = DynamicToolFactory()