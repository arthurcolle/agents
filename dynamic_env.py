#!/usr/bin/env python3
"""
Dynamic Environment for Hot Reloading and Code Modification

This module provides a powerful environment for LLM agents to:
- Create, modify and manage code modules in memory
- Hot-reload modules without restarting the application
- Transform code safely with AST-based operations
- Track dependencies and manage module references

The environment maintains proper isolation while allowing
dynamic code modifications to propagate to all references.
"""

import sys
import os
import ast
import importlib
import importlib.util
import types
import inspect
import logging
import time
import hashlib
import traceback
from typing import Dict, List, Any, Optional, Set, Union, Callable, TypeVar, cast

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dynamic_env")

# Type definitions
T = TypeVar('T')
CodeModule = types.ModuleType
ModuleDict = Dict[str, CodeModule]
CodeTransformer = Callable[[str], str]


class CodeBuffer:
    """
    In-memory buffer for code storage and version tracking
    
    Provides facilities for:
    - Storing multiple versions of module code
    - Tracking code changes and history
    - Basic validation and formatting
    """
    def __init__(self):
        self.code_store: Dict[str, List[Dict[str, Any]]] = {}
        
    def create_module(self, name: str, code: str) -> bool:
        """Create a new module in the buffer"""
        if name in self.code_store:
            logger.warning(f"Module '{name}' already exists in buffer")
            return False
            
        self.code_store[name] = [{
            "timestamp": time.time(),
            "code": code,
            "hash": hashlib.sha256(code.encode()).hexdigest(),
            "version": 1
        }]
        logger.info(f"Created new module '{name}' in buffer (v1)")
        return True
        
    def update_module(self, name: str, code: str) -> bool:
        """Update an existing module with new code"""
        if name not in self.code_store:
            logger.warning(f"Module '{name}' not found in buffer")
            return False
            
        versions = self.code_store[name]
        current_version = len(versions)
        new_version = current_version + 1
        
        # Check if code actually changed
        current_hash = versions[-1]["hash"]
        new_hash = hashlib.sha256(code.encode()).hexdigest()
        
        if current_hash == new_hash:
            logger.info(f"Module '{name}' not updated - code unchanged")
            return False
            
        self.code_store[name].append({
            "timestamp": time.time(),
            "code": code,
            "hash": new_hash,
            "version": new_version
        })
        logger.info(f"Updated module '{name}' to version {new_version}")
        return True
        
    def get_module_code(self, name: str, version: Optional[int] = None) -> Optional[str]:
        """Get module code at a specific version (latest if not specified)"""
        if name not in self.code_store:
            logger.warning(f"Module '{name}' not found in buffer")
            return None
            
        versions = self.code_store[name]
        
        if version is None:
            # Get latest version
            return versions[-1]["code"]
        elif 1 <= version <= len(versions):
            # Get specific version
            return versions[version-1]["code"]
        else:
            logger.warning(f"Invalid version {version} for module '{name}'")
            return None
            
    def get_version_count(self, name: str) -> int:
        """Get the number of versions for a module"""
        if name not in self.code_store:
            return 0
        return len(self.code_store[name])
        
    def list_modules(self) -> List[Dict[str, Any]]:
        """List all modules in the buffer with their latest version info"""
        return [
            {
                "name": name,
                "version": versions[-1]["version"],
                "timestamp": versions[-1]["timestamp"],
                "hash": versions[-1]["hash"]
            }
            for name, versions in self.code_store.items()
        ]
        
    def get_module_history(self, name: str) -> List[Dict[str, Any]]:
        """Get the full version history for a module"""
        if name not in self.code_store:
            return []
        
        # Return copies without the full code to reduce size
        return [
            {
                "version": v["version"], 
                "timestamp": v["timestamp"],
                "hash": v["hash"]
            }
            for v in self.code_store[name]
        ]
        
    def delete_module(self, name: str) -> bool:
        """Delete a module from the buffer"""
        if name not in self.code_store:
            return False
            
        del self.code_store[name]
        logger.info(f"Deleted module '{name}' from buffer")
        return True


class CodeTransformer:
    """
    Transforms code using AST operations
    
    Provides:
    - AST-based code analysis
    - Code safety validation
    - Source transformations
    """
    def __init__(self, unsafe_modules: Optional[List[str]] = None):
        self.unsafe_modules = unsafe_modules or [
            "os", "subprocess", "socket", "shutil", "pathlib", 
            "pickle", "sys", "importlib"
        ]
        
    def validate_code(self, code: str) -> List[str]:
        """
        Validate code for security and correctness
        Returns a list of potential issues
        """
        issues = []
        
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Check for dangerous imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name in self.unsafe_modules:
                            issues.append(f"Unsafe import: {name.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.unsafe_modules:
                        issues.append(f"Unsafe import from: {node.module}")
                
                # Check for exec/eval calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ["exec", "eval"]:
                        issues.append(f"Dangerous call to {node.func.id}")
                        
                # Check for __import__ calls
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    issues.append("Dangerous call to __import__")
        
        except SyntaxError as e:
            issues.append(f"Syntax error: {str(e)}")
        
        return issues
    
    def apply_transformations(self, code: str) -> str:
        """Apply transformations to code (placeholder for more advanced transforms)"""
        return code


class ModuleRegistry:
    """
    Manages dynamic modules and their imports
    
    Handles:
    - Module creation, loading and unloading
    - Import hooks to capture module references
    - Dependency tracking between modules
    """
    def __init__(self):
        self.modules: ModuleDict = {}
        self.virtual_namespace = "__dynamic_modules__"
        self.references: Dict[str, List[Any]] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self.transformer = CodeTransformer()
        
        # Hook into sys.meta_path to capture imports
        self._register_import_hook()
    
    def _register_import_hook(self) -> None:
        """Register an import hook to intercept dynamic modules"""
        class DynamicModuleFinder:
            def __init__(self, registry):
                self.registry = registry
                
            def find_spec(self, fullname, path, target=None):
                # Check if this is a dynamic module
                if fullname.startswith(self.registry.virtual_namespace + "."):
                    module_name = fullname.split(".")[-1]
                    if module_name in self.registry.modules:
                        # Return a spec for this module
                        return importlib.machinery.ModuleSpec(
                            fullname, 
                            None, 
                            is_package=False
                        )
                return None
        
        # Add our finder to the meta path
        sys.meta_path.insert(0, DynamicModuleFinder(self))
    
    def _track_dependency(self, module_name: str, dependent_module: Optional[str] = None) -> None:
        """
        Track a dependency between modules
        If dependent_module is None, use the calling module
        """
        if dependent_module is None:
            # Try to get the calling module
            frame = inspect.currentframe()
            if frame:
                try:
                    frame = frame.f_back  # Get the caller's frame
                    if frame and frame.f_globals and "__name__" in frame.f_globals:
                        dependent_module = frame.f_globals["__name__"]
                finally:
                    del frame  # Avoid reference cycles
        
        if dependent_module and dependent_module != module_name:
            if dependent_module not in self.dependencies:
                self.dependencies[dependent_module] = set()
            self.dependencies[dependent_module].add(module_name)
            logger.debug(f"Tracked dependency: {dependent_module} -> {module_name}")
    
    def _track_reference(self, module_name: str, obj: Any) -> None:
        """Track a reference to a module object"""
        if module_name not in self.references:
            self.references[module_name] = []
        self.references[module_name].append(obj)
    
    def create_module(self, name: str, code: str) -> Optional[CodeModule]:
        """
        Create a new dynamic module
        Returns the module object or None if creation failed
        """
        # Validate the code
        issues = self.transformer.validate_code(code)
        if issues:
            for issue in issues:
                logger.warning(f"Code validation issue: {issue}")
            return None
        
        # Create a new module
        full_name = f"{self.virtual_namespace}.{name}"
        module = types.ModuleType(full_name)
        module.__file__ = f"<dynamic-module:{name}>"
        
        # Execute the code in the module's context
        try:
            # Apply transformations
            transformed_code = self.transformer.apply_transformations(code)
            
            # Compile and execute
            compiled_code = compile(transformed_code, module.__file__, 'exec')
            exec(compiled_code, module.__dict__)
            
            # Store in registry
            self.modules[name] = module
            
            # Add to sys.modules for standard imports to work
            sys.modules[full_name] = module
            
            logger.info(f"Created dynamic module: {name}")
            return module
        
        except Exception as e:
            logger.error(f"Error creating module {name}: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    def import_module(self, name: str) -> Optional[CodeModule]:
        """
        Import a dynamic module
        Returns the module object or None if import fails
        """
        # Check if already loaded
        if name in self.modules:
            module = self.modules[name]
            # Track dependency
            self._track_dependency(name)
            return module
        
        # Not found
        logger.warning(f"Dynamic module not found: {name}")
        return None
    
    def update_module(self, name: str, code: str) -> bool:
        """
        Update a module with new code
        Returns True if the update succeeded
        """
        if name not in self.modules:
            logger.warning(f"Module {name} not found for update")
            return False
        
        # Validate the code
        issues = self.transformer.validate_code(code)
        if issues:
            for issue in issues:
                logger.warning(f"Code validation issue: {issue}")
            return False
        
        try:
            # Get the existing module
            module = self.modules[name]
            
            # Apply transformations
            transformed_code = self.transformer.apply_transformations(code)
            
            # Keep a copy of the old state
            old_state = module.__dict__.copy()
            
            # Clear the module namespace, but keep certain special attributes
            keep_attrs = {"__name__", "__file__", "__path__", "__loader__", "__package__"}
            for key in list(module.__dict__.keys()):
                if key not in keep_attrs:
                    delattr(module, key)
            
            # Execute the new code
            compiled_code = compile(transformed_code, module.__file__, 'exec')
            exec(compiled_code, module.__dict__)
            
            # Update references to this module
            if name in self.references:
                self._update_references(name, old_state)
            
            logger.info(f"Updated module: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating module {name}: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def _update_references(self, module_name: str, old_state: Dict[str, Any]) -> None:
        """
        Update all references to a module after it's been changed
        This is a critical part of hot module replacement
        """
        module = self.modules[module_name]
        new_state = module.__dict__
        
        # Find all modules that depend on this one
        dependent_modules = set()
        for dep_name, deps in self.dependencies.items():
            if module_name in deps:
                dependent_modules.add(dep_name)
        
        # Update class implementations
        for name, obj in old_state.items():
            # Skip special attributes and non-class items
            if name.startswith('__') or not isinstance(obj, type):
                continue
            
            # Check if this class still exists in new state
            if name in new_state and isinstance(new_state[name], type):
                new_class = new_state[name]
                
                # Update class methods and attributes
                for attr_name, attr_value in new_class.__dict__.items():
                    if not attr_name.startswith('__'):
                        setattr(obj, attr_name, attr_value)
        
        logger.debug(f"Updated references for module {module_name}")
    
    def reload_module(self, name: str) -> Optional[CodeModule]:
        """
        Reload a module from its current code
        This is a convenience method mainly used for testing
        """
        if name not in self.modules:
            logger.warning(f"Module {name} not found for reload")
            return None
        
        # Re-execute the module
        module = self.modules[name]
        try:
            # Get code via dis or other means
            module_code = inspect.getsource(module)
            if self.update_module(name, module_code):
                return module
            return None
        except Exception as e:
            logger.error(f"Error reloading module {name}: {str(e)}")
            return None
    
    def unload_module(self, name: str) -> bool:
        """
        Unload a module from the registry
        Returns True if the module was unloaded
        """
        if name not in self.modules:
            return False
        
        full_name = f"{self.virtual_namespace}.{name}"
        
        # Remove from sys.modules
        if full_name in sys.modules:
            del sys.modules[full_name]
        
        # Remove from registry
        del self.modules[name]
        
        # Clear references and dependencies
        if name in self.references:
            del self.references[name]
        
        # Clean up dependencies
        for dep_name, deps in list(self.dependencies.items()):
            if name in deps:
                deps.remove(name)
            if name == dep_name:
                del self.dependencies[dep_name]
        
        logger.info(f"Unloaded module: {name}")
        return True
    
    def list_modules(self) -> List[Dict[str, Any]]:
        """List all modules in the registry with metadata"""
        return [
            {
                "name": name,
                "attributes": list(module.__dict__.keys()),
                "dependencies": list(self.dependencies.get(name, set())),
                "references": len(self.references.get(name, []))
            }
            for name, module in self.modules.items()
        ]


class ModuleBuffer:
    """
    High-level interface combining CodeBuffer and ModuleRegistry
    
    Provides a unified API for:
    - Creating and managing code in the buffer
    - Loading modules from the buffer into the registry
    - Updating modules and propagating changes
    """
    def __init__(self, registry: Optional[ModuleRegistry] = None):
        self.buffer = CodeBuffer()
        self.registry = registry or ModuleRegistry()
        
    def create_module(self, name: str, code: str) -> Optional[CodeModule]:
        """
        Create a new module in the buffer and load it
        Returns the module object if successful
        """
        # Add to buffer
        if not self.buffer.create_module(name, code):
            return None
        
        # Load into registry
        return self.registry.create_module(name, code)
    
    def update_module(self, name: str, code: str) -> bool:
        """
        Update a module in both buffer and registry
        Returns True if the update succeeded
        """
        # Update in buffer
        if not self.buffer.update_module(name, code):
            return False
        
        # Update in registry
        return self.registry.update_module(name, code)
    
    def get_module(self, name: str) -> Optional[CodeModule]:
        """Get a module from the registry"""
        return self.registry.import_module(name)
    
    def list_modules(self) -> List[Dict[str, Any]]:
        """List all modules with metadata from both buffer and registry"""
        buffer_modules = {m["name"]: m for m in self.buffer.list_modules()}
        registry_modules = {m["name"]: m for m in self.registry.list_modules()}
        
        # Combine information
        result = []
        all_names = set(buffer_modules.keys()) | set(registry_modules.keys())
        
        for name in all_names:
            module_info = {
                "name": name,
                "in_buffer": name in buffer_modules,
                "in_registry": name in registry_modules,
            }
            
            if name in buffer_modules:
                module_info.update({
                    "version": buffer_modules[name]["version"],
                    "timestamp": buffer_modules[name]["timestamp"]
                })
                
            if name in registry_modules:
                module_info.update({
                    "attributes": registry_modules[name].get("attributes", []),
                    "dependencies": registry_modules[name].get("dependencies", []),
                    "references": registry_modules[name].get("references", 0)
                })
                
            result.append(module_info)
            
        return result
    
    def get_module_history(self, name: str) -> List[Dict[str, Any]]:
        """Get version history for a module"""
        return self.buffer.get_module_history(name)
    
    def rollback_module(self, name: str, version: int) -> bool:
        """
        Rollback a module to a specific version
        Returns True if the rollback succeeded
        """
        # Get the code at the specified version
        code = self.buffer.get_module_code(name, version)
        if not code:
            return False
        
        # Update to this version
        if not self.buffer.update_module(name, code):
            return False
        
        # Update in registry
        return self.registry.update_module(name, code)


class CodeRegistry:
    """
    Wrapper for ModuleRegistry providing a simpler interface 
    focused on importing and using dynamic modules
    """
    def __init__(self, registry: Optional[ModuleRegistry] = None):
        self.registry = registry or ModuleRegistry()
    
    def import_module(self, name: str) -> Any:
        """Import a module by name"""
        return self.registry.import_module(name)
    
    def reload_module(self, name: str) -> bool:
        """Reload a module by name"""
        return self.registry.reload_module(name) is not None
    
    def list_modules(self) -> List[str]:
        """Get a list of all available module names"""
        return [m["name"] for m in self.registry.list_modules()]


# Demo code when run directly
if __name__ == "__main__":
    # Set up the environment
    buffer = ModuleBuffer()
    registry = CodeRegistry(buffer.registry)
    
    # Create a sample module
    code = """
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b
    """
    
    module = buffer.create_module("demo", code)
    if module:
        print("Created module 'demo'")
        print(f"greet('World') = {module.greet('World')}")
        print(f"add(5, 3) = {module.add(5, 3)}")
        
        # Update the module
        new_code = """
def greet(name):
    return f"Greetings, {name}!"

def add(a, b):
    return a + b
    
def subtract(a, b):
    return a - b
        """
        
        if buffer.update_module("demo", new_code):
            print("\nUpdated module 'demo'")
            print(f"greet('World') = {module.greet('World')}")
            print(f"subtract(10, 4) = {module.subtract(10, 4)}")
            
    # List all modules
    print("\nAvailable modules:")
    for module_info in buffer.list_modules():
        print(f"- {module_info['name']} (v{module_info.get('version', '?')})")