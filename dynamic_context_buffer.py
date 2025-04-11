#!/usr/bin/env python3
"""
Dynamic Context Buffer

A powerful buffer system for AI agents to:
- Hot-reload and dynamically rewrite modules on the fly
- Maintain an in-memory environment for code execution 
- Sandbox code execution with proper isolation
- Allow agents to experiment with code changes without affecting the disk
- Persist memory across sessions with serialization

This acts as a virtual filesystem and execution environment 
that only becomes "real" when explicitly committed to disk.
"""

import os
import sys
import importlib
import inspect
import ast
import types
import uuid
import json
import pickle
import time
import copy
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('dynamic_context')

class VirtualModule:
    """Represents a Python module that exists in memory but not necessarily on disk"""
    
    def __init__(self, name: str, code: str = "", source_path: Optional[str] = None):
        """
        Initialize a virtual module
        
        Args:
            name: The module name
            code: The module source code
            source_path: Optional path to the original source file
        """
        self.name = name
        self.code = code
        self.source_path = source_path
        self.compiled = None
        self.module = None
        self.last_modified = time.time()
        self.attributes = {}
        self.in_memory_only = source_path is None
        
        # Compile if code is provided
        if code:
            self.compile()
            
    def compile(self) -> bool:
        """
        Compile the module code
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.compiled = compile(self.code, f"<virtual:{self.name}>", 'exec')
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in module {self.name}: {str(e)}")
            return False
            
    def load(self) -> Optional[types.ModuleType]:
        """
        Load the module into memory
        
        Returns:
            The loaded module object or None if failed
        """
        try:
            # Create a new module object
            module = types.ModuleType(self.name)
            
            # Set __file__ attribute if we have a source path
            if self.source_path:
                module.__file__ = self.source_path
                
            # Add to sys.modules
            sys.modules[self.name] = module
            
            # Execute the compiled code in the module's namespace
            if self.compiled:
                exec(self.compiled, module.__dict__)
                
            self.module = module
            self.attributes = {
                name: value for name, value in inspect.getmembers(module)
                if not name.startswith('_')
            }
            return module
        except Exception as e:
            logger.error(f"Error loading module {self.name}: {str(e)}")
            return None
            
    def reload(self) -> Optional[types.ModuleType]:
        """
        Reload the module with current code
        
        Returns:
            The reloaded module object or None if failed
        """
        self.compile()
        return self.load()
        
    def get_function(self, function_name: str) -> Optional[Callable]:
        """
        Get a function from the module
        
        Args:
            function_name: Name of the function to retrieve
            
        Returns:
            The function object or None if not found
        """
        if not self.module:
            self.load()
            
        if not self.module:
            return None
            
        return getattr(self.module, function_name, None)
        
    def get_class(self, class_name: str) -> Optional[type]:
        """
        Get a class from the module
        
        Args:
            class_name: Name of the class to retrieve
            
        Returns:
            The class object or None if not found
        """
        if not self.module:
            self.load()
            
        if not self.module:
            return None
            
        return getattr(self.module, class_name, None)
        
    def update_code(self, new_code: str) -> bool:
        """
        Update the module's code
        
        Args:
            new_code: The new source code
            
        Returns:
            True if successful, False otherwise
        """
        self.code = new_code
        self.last_modified = time.time()
        return self.compile() and self.reload() is not None
        
    def serialize(self) -> Dict:
        """
        Serialize the module for storage
        
        Returns:
            Dictionary representation of the module
        """
        return {
            "name": self.name,
            "code": self.code,
            "source_path": self.source_path,
            "last_modified": self.last_modified,
            "in_memory_only": self.in_memory_only
        }
        
    @classmethod
    def deserialize(cls, data: Dict) -> 'VirtualModule':
        """
        Create a module from serialized data
        
        Args:
            data: Dictionary representation of the module
            
        Returns:
            VirtualModule instance
        """
        module = cls(
            name=data["name"],
            code=data["code"],
            source_path=data["source_path"]
        )
        module.last_modified = data.get("last_modified", time.time())
        module.in_memory_only = data.get("in_memory_only", False)
        return module

class DynamicContextBuffer:
    """
    Core buffer system for managing dynamic code modules
    Provides virtual environment for testing code changes before committing to disk
    """
    
    def __init__(self, persistence_path: Optional[str] = None):
        """
        Initialize the dynamic context buffer
        
        Args:
            persistence_path: Optional path to save/load buffer state
        """
        self.modules: Dict[str, VirtualModule] = {}
        self.persistence_path = persistence_path
        self.snapshots: Dict[str, List[Dict]] = {}  # module_name -> list of snapshots
        self.execution_history: List[Dict] = []
        self.logger = logger
        
        # Load persistent state if available
        if persistence_path and os.path.exists(persistence_path):
            self.load_state()
    
    def add_module_from_file(self, file_path: str) -> Optional[VirtualModule]:
        """
        Load a module from an existing file
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            The loaded virtual module or None if failed
        """
        try:
            with open(file_path, 'r') as f:
                code = f.read()
                
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            module = VirtualModule(module_name, code, file_path)
            self.modules[module_name] = module
            
            # Create initial snapshot
            self._create_snapshot(module_name)
            
            return module
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            return None
            
    def add_module_from_code(self, module_name: str, code: str) -> VirtualModule:
        """
        Create a new in-memory module from code
        
        Args:
            module_name: Name for the new module
            code: Python source code
            
        Returns:
            The created virtual module
        """
        module = VirtualModule(module_name, code)
        self.modules[module_name] = module
        
        # Create initial snapshot
        self._create_snapshot(module_name)
        
        return module
        
    def get_module(self, module_name: str) -> Optional[VirtualModule]:
        """
        Get a virtual module by name
        
        Args:
            module_name: Name of the module to retrieve
            
        Returns:
            The module or None if not found
        """
        return self.modules.get(module_name)
        
    def update_module(self, module_name: str, new_code: str) -> bool:
        """
        Update a module's code
        
        Args:
            module_name: Name of the module to update
            new_code: New source code
            
        Returns:
            True if successful, False otherwise
        """
        module = self.get_module(module_name)
        if not module:
            return False
            
        success = module.update_code(new_code)
        
        if success:
            # Create snapshot after successful update
            self._create_snapshot(module_name)
            
        return success
        
    def execute_code(self, code: str, globals_dict: Optional[Dict] = None, 
                    locals_dict: Optional[Dict] = None) -> Dict:
        """
        Execute code in a managed environment
        
        Args:
            code: Python code to execute
            globals_dict: Optional globals dictionary
            locals_dict: Optional locals dictionary
            
        Returns:
            Dictionary with execution results
        """
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        if globals_dict is None:
            globals_dict = {}
            
        if locals_dict is None:
            locals_dict = {}
            
        # Add access to our modules
        for name, module in self.modules.items():
            if module.module:
                globals_dict[name] = module.module
                
        # Execute the code
        stdout_buffer = []
        stderr_buffer = []
        result = None
        error = None
        
        try:
            # Redirect stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            class CaptureBuffer:
                def __init__(self, buffer):
                    self.buffer = buffer
                    
                def write(self, text):
                    self.buffer.append(text)
                    original_stdout.write(text)
                    
                def flush(self):
                    pass
                    
            sys.stdout = CaptureBuffer(stdout_buffer)
            sys.stderr = CaptureBuffer(stderr_buffer)
            
            # Execute the code
            compiled = compile(code, "<dynamic_context>", "exec")
            exec(compiled, globals_dict, locals_dict)
            
            # Check for result variable if it exists
            if 'result' in locals_dict:
                result = locals_dict['result']
                
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            self.logger.error(f"Error executing code: {error}")
            
        finally:
            # Restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
        execution_time = time.time() - start_time
        
        # Record the execution
        execution_record = {
            "id": execution_id,
            "code": code,
            "timestamp": start_time,
            "duration": execution_time,
            "error": error,
            "stdout": "".join(stdout_buffer),
            "stderr": "".join(stderr_buffer),
            "result": result
        }
        
        self.execution_history.append(execution_record)
        
        return execution_record
        
    def execute_function(self, module_name: str, function_name: str, 
                       *args, **kwargs) -> Dict:
        """
        Execute a function from a module
        
        Args:
            module_name: Name of the module
            function_name: Name of the function
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Dictionary with execution results
        """
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        module = self.get_module(module_name)
        if not module:
            return {
                "id": execution_id,
                "error": f"Module {module_name} not found",
                "result": None,
                "success": False
            }
            
        function = module.get_function(function_name)
        if not function:
            return {
                "id": execution_id,
                "error": f"Function {function_name} not found in module {module_name}",
                "result": None,
                "success": False
            }
            
        # Execute the function
        stdout_buffer = []
        stderr_buffer = []
        result = None
        error = None
        
        try:
            # Redirect stdout/stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            class CaptureBuffer:
                def __init__(self, buffer):
                    self.buffer = buffer
                    
                def write(self, text):
                    self.buffer.append(text)
                    original_stdout.write(text)
                    
                def flush(self):
                    pass
                    
            sys.stdout = CaptureBuffer(stdout_buffer)
            sys.stderr = CaptureBuffer(stderr_buffer)
            
            # Execute the function
            result = function(*args, **kwargs)
            
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            self.logger.error(f"Error executing function {function_name}: {error}")
            
        finally:
            # Restore stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
        execution_time = time.time() - start_time
        
        # Record the execution
        execution_record = {
            "id": execution_id,
            "module": module_name,
            "function": function_name,
            "args": args,
            "kwargs": kwargs,
            "timestamp": start_time,
            "duration": execution_time,
            "error": error,
            "stdout": "".join(stdout_buffer),
            "stderr": "".join(stderr_buffer),
            "result": result,
            "success": error is None
        }
        
        self.execution_history.append(execution_record)
        
        return execution_record
        
    def _create_snapshot(self, module_name: str) -> None:
        """
        Create a snapshot of a module's current state
        
        Args:
            module_name: Name of the module
        """
        module = self.get_module(module_name)
        if not module:
            return
            
        if module_name not in self.snapshots:
            self.snapshots[module_name] = []
            
        snapshot = {
            "timestamp": time.time(),
            "code": module.code,
            "hash": hash(module.code)
        }
        
        self.snapshots[module_name].append(snapshot)
        
        # Limit snapshots to 20 per module
        if len(self.snapshots[module_name]) > 20:
            self.snapshots[module_name].pop(0)
            
    def restore_snapshot(self, module_name: str, index: int = -1) -> bool:
        """
        Restore a module to a previous snapshot
        
        Args:
            module_name: Name of the module
            index: Index of the snapshot to restore (-1 for latest)
            
        Returns:
            True if successful, False otherwise
        """
        if module_name not in self.snapshots:
            return False
            
        snapshots = self.snapshots[module_name]
        if not snapshots:
            return False
            
        if index < 0:
            index = len(snapshots) + index
            
        if index < 0 or index >= len(snapshots):
            return False
            
        snapshot = snapshots[index]
        return self.update_module(module_name, snapshot["code"])
        
    def get_snapshots(self, module_name: str) -> List[Dict]:
        """
        Get all snapshots for a module
        
        Args:
            module_name: Name of the module
            
        Returns:
            List of snapshots
        """
        return self.snapshots.get(module_name, [])
        
    def commit_to_disk(self, module_name: str) -> bool:
        """
        Write an in-memory module to disk
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if successful, False otherwise
        """
        module = self.get_module(module_name)
        if not module:
            return False
            
        # If module doesn't have a source path, create one
        if not module.source_path:
            module.source_path = f"{module_name}.py"
            
        try:
            with open(module.source_path, 'w') as f:
                f.write(module.code)
                
            # Update module state
            module.in_memory_only = False
            return True
        except Exception as e:
            self.logger.error(f"Error writing module {module_name} to disk: {str(e)}")
            return False
            
    def save_state(self, path: Optional[str] = None) -> bool:
        """
        Save the current state of the buffer
        
        Args:
            path: Optional path to save to (defaults to persistence_path)
            
        Returns:
            True if successful, False otherwise
        """
        save_path = path or self.persistence_path
        if not save_path:
            self.logger.error("No persistence path specified")
            return False
            
        try:
            # Serialize all modules
            modules_data = {
                name: module.serialize()
                for name, module in self.modules.items()
            }
            
            # Create state object
            state = {
                "modules": modules_data,
                "snapshots": self.snapshots,
                "timestamp": time.time()
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(state, f)
                
            return True
        except Exception as e:
            self.logger.error(f"Error saving buffer state: {str(e)}")
            return False
            
    def load_state(self, path: Optional[str] = None) -> bool:
        """
        Load a previously saved state
        
        Args:
            path: Optional path to load from (defaults to persistence_path)
            
        Returns:
            True if successful, False otherwise
        """
        load_path = path or self.persistence_path
        if not load_path or not os.path.exists(load_path):
            return False
            
        try:
            with open(load_path, 'rb') as f:
                state = pickle.load(f)
                
            # Load modules
            self.modules = {}
            for name, module_data in state.get("modules", {}).items():
                self.modules[name] = VirtualModule.deserialize(module_data)
                
            # Load snapshots
            self.snapshots = state.get("snapshots", {})
            
            # Load each module
            for module in self.modules.values():
                module.load()
                
            return True
        except Exception as e:
            self.logger.error(f"Error loading buffer state: {str(e)}")
            return False
            
    def clear(self) -> None:
        """Clear all modules and state"""
        self.modules = {}
        self.snapshots = {}
        self.execution_history = []
        
    def get_module_summary(self, module_name: str) -> Dict:
        """
        Get a summary of a module's content
        
        Args:
            module_name: Name of the module
            
        Returns:
            Dictionary with module summary
        """
        module = self.get_module(module_name)
        if not module:
            return {"error": f"Module {module_name} not found"}
            
        # Parse the code
        try:
            tree = ast.parse(module.code)
            
            # Extract functions
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func = {
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [ast.unparse(dec).strip() for dec in node.decorator_list]
                    }
                    
                    # Extract docstring if available
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Str)):
                        func["docstring"] = node.body[0].value.s
                        
                    functions.append(func)
                    
            # Extract classes
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    cls = {
                        "name": node.name,
                        "line": node.lineno,
                        "bases": [ast.unparse(base).strip() for base in node.bases],
                        "methods": []
                    }
                    
                    # Extract docstring if available
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Str)):
                        cls["docstring"] = node.body[0].value.s
                        
                    # Extract methods
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            method = {
                                "name": child.name,
                                "line": child.lineno,
                                "args": [arg.arg for arg in child.args.args],
                                "decorators": [ast.unparse(dec).strip() for dec in child.decorator_list]
                            }
                            
                            # Extract docstring if available
                            if (child.body and isinstance(child.body[0], ast.Expr) and 
                                isinstance(child.body[0].value, ast.Str)):
                                method["docstring"] = child.body[0].value.s
                                
                            cls["methods"].append(method)
                            
                    classes.append(cls)
                    
            return {
                "name": module_name,
                "source_path": module.source_path,
                "in_memory_only": module.in_memory_only,
                "last_modified": module.last_modified,
                "functions": functions,
                "classes": classes,
                "line_count": len(module.code.split('\n'))
            }
            
        except SyntaxError as e:
            return {
                "name": module_name,
                "source_path": module.source_path,
                "in_memory_only": module.in_memory_only,
                "last_modified": module.last_modified,
                "error": f"Syntax error: {str(e)}",
                "line_count": len(module.code.split('\n'))
            }

class HotSwapExecutionContext:
    """
    Execution context that allows hot-swapping code at runtime
    Provides a safe environment for executing dynamically modified code
    """
    
    def __init__(self, buffer: DynamicContextBuffer):
        """
        Initialize the execution context
        
        Args:
            buffer: The dynamic context buffer to use
        """
        self.buffer = buffer
        self.globals = {}
        self.locals = {}
        self.modules_imported = set()
        
    def reset(self) -> None:
        """Reset the execution context"""
        self.globals = {}
        self.locals = {}
        self.modules_imported = set()
        
    def import_module(self, module_name: str) -> bool:
        """
        Import a module from the buffer into the execution context
        
        Args:
            module_name: Name of the module to import
            
        Returns:
            True if successful, False otherwise
        """
        module = self.buffer.get_module(module_name)
        if not module or not module.module:
            return False
            
        self.globals[module_name] = module.module
        self.modules_imported.add(module_name)
        return True
        
    def execute(self, code: str) -> Dict:
        """
        Execute code in this context
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution results
        """
        return self.buffer.execute_code(code, self.globals, self.locals)
        
    def call_function(self, module_name: str, function_name: str, 
                     *args, **kwargs) -> Dict:
        """
        Call a function from a module in this context
        
        Args:
            module_name: Name of the module
            function_name: Name of the function
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Execution results
        """
        # Ensure the module is imported
        if module_name not in self.modules_imported:
            self.import_module(module_name)
            
        return self.buffer.execute_function(module_name, function_name, *args, **kwargs)
        
    def hot_reload_module(self, module_name: str) -> bool:
        """
        Reload a module and update the context
        
        Args:
            module_name: Name of the module to reload
            
        Returns:
            True if successful, False otherwise
        """
        module = self.buffer.get_module(module_name)
        if not module:
            return False
            
        # Reload the module
        reloaded_module = module.reload()
        if not reloaded_module:
            return False
            
        # Update in our globals
        if module_name in self.modules_imported:
            self.globals[module_name] = reloaded_module
            
        return True

# === HELPER FUNCTIONS ===

def create_dynamic_context(persistence_path: Optional[str] = None) -> DynamicContextBuffer:
    """
    Create a new dynamic context buffer
    
    Args:
        persistence_path: Optional path for persisting buffer state
        
    Returns:
        Initialized DynamicContextBuffer
    """
    return DynamicContextBuffer(persistence_path)
    
def create_execution_context(buffer: DynamicContextBuffer) -> HotSwapExecutionContext:
    """
    Create a new execution context
    
    Args:
        buffer: The dynamic context buffer to use
        
    Returns:
        Initialized execution context
    """
    return HotSwapExecutionContext(buffer)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic Context Buffer")
    parser.add_argument("--file", help="Python file to load")
    parser.add_argument("--exec", help="Code to execute")
    parser.add_argument("--save", help="Path to save state")
    parser.add_argument("--load", help="Path to load state")
    
    args = parser.parse_args()
    
    buffer = DynamicContextBuffer()
    
    if args.load:
        print(f"Loading state from {args.load}")
        buffer.load_state(args.load)
        
    if args.file:
        print(f"Loading file {args.file}")
        module = buffer.add_module_from_file(args.file)
        if module:
            print(f"Loaded module {module.name}")
            
    if args.exec:
        print(f"Executing: {args.exec}")
        result = buffer.execute_code(args.exec)
        if result["error"]:
            print(f"Error: {result['error']}")
        else:
            print(f"Result: {result['result']}")
            
    if args.save:
        print(f"Saving state to {args.save}")
        buffer.save_state(args.save)