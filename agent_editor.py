#!/usr/bin/env python3
"""
Agent Editor - Module for code analysis and modification

Provides tools for an AI agent to safely modify its own code or other code:
- AST-based code chunking to break files into manageable pieces
- Memory snapshots to track versions and enable rollbacks
- Hot module reloading to apply changes without restarting
- Validation of code modifications before applying them
- Methods to search, modify, and analyze code
"""

import os
import sys
import ast
import copy
import re
import importlib
import importlib.util
import inspect
import hashlib
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable

# === CODE CHUNKING AND ANALYSIS ===
class CodeChunk:
    """Represents a meaningful chunk of code (class, function, block)"""
    def __init__(self, node, source_code, parent=None, file_path=None):
        self.node = node
        self.source_code = source_code
        self.parent = parent
        self.file_path = file_path
        self.start_line = getattr(node, 'lineno', 0)
        self.end_line = self._calculate_end_line()
        self.children = []
        
    def _calculate_end_line(self):
        """Calculate the ending line number of this chunk"""
        if hasattr(self.node, 'end_lineno'):
            return self.node.end_lineno
            
        # For older Python versions or when end_lineno is not available
        if hasattr(self.node, 'body') and isinstance(self.node.body, list) and self.node.body:
            max_line = 0
            for child in self.node.body:
                if hasattr(child, 'end_lineno'):
                    max_line = max(max_line, child.end_lineno)
                elif hasattr(child, 'lineno'):
                    max_line = max(max_line, child.lineno)
            return max_line if max_line > 0 else self.start_line
            
        return self.start_line
        
    def extract_code(self):
        """Extract the source code for this chunk"""
        lines = self.source_code.split('\n')
        return '\n'.join(lines[self.start_line - 1:self.end_line])
        
    def get_summary(self):
        """Get a summary of this chunk"""
        chunk_type = type(self.node).__name__
        
        if isinstance(self.node, ast.ClassDef):
            name = self.node.name
            methods = [method.name for method in self.node.body if isinstance(method, ast.FunctionDef)]
            return {
                "type": "class",
                "name": name,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "methods": methods
            }
            
        elif isinstance(self.node, ast.FunctionDef):
            name = self.node.name
            args = [arg.arg for arg in self.node.args.args if arg.arg != 'self']
            return {
                "type": "function",
                "name": name,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "args": args,
                "decorators": [
                    ast.unparse(dec).strip() for dec in self.node.decorator_list
                ]
            }
            
        elif isinstance(self.node, ast.Module):
            return {
                "type": "module",
                "start_line": self.start_line,
                "end_line": self.end_line,
                "path": self.file_path
            }
            
        else:
            return {
                "type": chunk_type.lower(),
                "start_line": self.start_line,
                "end_line": self.end_line
            }

class CodeAnalyzer:
    """Analyzes code to extract meaningful chunks and insights"""
    def __init__(self):
        self.chunks = []
        
    def analyze_file(self, file_path: str) -> List[CodeChunk]:
        """Analyze a file and extract code chunks"""
        with open(file_path, 'r') as f:
            source_code = f.read()
            
        return self.analyze_code(source_code, file_path)
        
    def analyze_code(self, source_code: str, file_path: Optional[str] = None) -> List[CodeChunk]:
        """
        Analyze code string and extract code chunks
        Returns a list of CodeChunk objects
        """
        self.chunks = []
        
        try:
            # Parse the code into an AST
            tree = ast.parse(source_code)
            
            # Create the root chunk (module)
            root_chunk = CodeChunk(tree, source_code, file_path=file_path)
            self.chunks.append(root_chunk)
            
            # Process all top-level chunks
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    chunk = CodeChunk(node, source_code, parent=root_chunk, file_path=file_path)
                    root_chunk.children.append(chunk)
                    self.chunks.append(chunk)
                    
                    # Process class methods
                    if isinstance(node, ast.ClassDef):
                        for method_node in node.body:
                            if isinstance(method_node, ast.FunctionDef):
                                method_chunk = CodeChunk(method_node, source_code, parent=chunk, file_path=file_path)
                                chunk.children.append(method_chunk)
                                self.chunks.append(method_chunk)
                                
        except SyntaxError as e:
            print(f"Syntax error in code: {str(e)}")
            return []
            
        return self.chunks
        
    def find_chunk_containing_line(self, line_number: int) -> Optional[CodeChunk]:
        """Find the smallest chunk containing the given line number"""
        matching_chunks = [
            chunk for chunk in self.chunks 
            if chunk.start_line <= line_number <= chunk.end_line
        ]
        
        # Sort by size (smallest first)
        matching_chunks.sort(key=lambda c: c.end_line - c.start_line)
        
        return matching_chunks[0] if matching_chunks else None
        
    def get_chunk_by_name(self, name: str, chunk_type: Optional[str] = None) -> Optional[CodeChunk]:
        """Find a chunk by name and optional type"""
        for chunk in self.chunks:
            if (
                (isinstance(chunk.node, ast.ClassDef) and chunk.node.name == name and 
                 (chunk_type is None or chunk_type == 'class')) or
                (isinstance(chunk.node, ast.FunctionDef) and chunk.node.name == name and
                 (chunk_type is None or chunk_type == 'function'))
            ):
                return chunk
                
        return None
        
    def get_all_functions(self) -> List[CodeChunk]:
        """Get all function chunks"""
        return [c for c in self.chunks if isinstance(c.node, ast.FunctionDef)]
        
    def get_all_classes(self) -> List[CodeChunk]:
        """Get all class chunks"""
        return [c for c in self.chunks if isinstance(c.node, ast.ClassDef)]

# === CODE MODIFICATION ===
class CodeEditor:
    """Edits code safely, with validation and undo capability"""
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.snapshots = {}  # file_path -> list of snapshots
        
    def load_file(self, file_path: str) -> str:
        """Load a file and create an initial snapshot"""
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Create first snapshot
        self._create_snapshot(file_path, content)
        
        # Analyze the code
        self.analyzer.analyze_code(content, file_path)
        
        return content
        
    def _create_snapshot(self, file_path: str, content: str) -> None:
        """Create a snapshot of the file content"""
        if file_path not in self.snapshots:
            self.snapshots[file_path] = []
            
        self.snapshots[file_path].append({
            "timestamp": time.time(),
            "content": content,
            "hash": hashlib.sha256(content.encode()).hexdigest()
        })
        
    def save_file(self, file_path: str, content: str) -> bool:
        """
        Save content to a file, creating a new snapshot
        Returns success boolean
        """
        try:
            # First validate the code
            if not self._validate_code(content):
                return False
                
            # Create backup if this is a new modification
            if os.path.exists(file_path):
                backup_path = f"{file_path}.bak"
                with open(backup_path, 'w') as f:
                    with open(file_path, 'r') as src:
                        f.write(src.read())
                        
            # Write the new content
            with open(file_path, 'w') as f:
                f.write(content)
                
            # Create snapshot
            self._create_snapshot(file_path, content)
            
            return True
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return False
            
    def _validate_code(self, content: str) -> bool:
        """
        Validate that code compiles and doesn't contain dangerous patterns
        Returns True if valid, False otherwise
        """
        # Check that it parses
        try:
            ast.parse(content)
        except SyntaxError as e:
            print(f"Syntax error in modified code: {str(e)}")
            return False
            
        # Check for dangerous patterns (very basic)
        dangerous_patterns = [
            'os.system', 'subprocess.call', 'subprocess.run',
            'exec(', 'eval(', '__import__('
        ]
        
        for pattern in dangerous_patterns:
            if pattern in content:
                print(f"Dangerous pattern detected in code: {pattern}")
                return False
                
        return True
        
    def get_snapshots(self, file_path: str) -> List[Dict]:
        """Get all snapshots for a file"""
        return self.snapshots.get(file_path, [])
        
    def restore_snapshot(self, file_path: str, index: int) -> bool:
        """
        Restore a previous snapshot
        Returns success boolean
        """
        if file_path not in self.snapshots:
            return False
            
        snapshots = self.snapshots[file_path]
        if index < 0 or index >= len(snapshots):
            return False
            
        # Get the snapshot
        snapshot = snapshots[index]
        
        # Restore the content
        return self.save_file(file_path, snapshot["content"])
        
    def replace_function(self, file_path: str, function_name: str, new_code: str) -> bool:
        """
        Replace a function with new code
        Returns success boolean
        """
        # Load the current file
        content = self.load_file(file_path)
        
        # Find the function in the code
        chunks = self.analyzer.analyze_code(content, file_path)
        function_chunk = self.analyzer.get_chunk_by_name(function_name, 'function')
        
        if not function_chunk:
            print(f"Function '{function_name}' not found in {file_path}")
            return False
            
        # Validate the new function code
        try:
            tree = ast.parse(new_code)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                print("New code must contain exactly one function definition")
                return False
                
            func_node = tree.body[0]
            if func_node.name != function_name:
                print(f"New function name ({func_node.name}) doesn't match the function to replace ({function_name})")
                return False
        except SyntaxError as e:
            print(f"Syntax error in new function code: {str(e)}")
            return False
            
        # Replace the function in the content
        lines = content.split('\n')
        start_line = function_chunk.start_line - 1  # 0-indexed
        end_line = function_chunk.end_line
        
        # Create the new content
        new_lines = lines[:start_line] + new_code.split('\n') + lines[end_line:]
        new_content = '\n'.join(new_lines)
        
        # Save the modified content
        return self.save_file(file_path, new_content)
        
    def replace_method(self, file_path: str, class_name: str, method_name: str, new_code: str) -> bool:
        """
        Replace a class method with new code
        Returns success boolean
        """
        # Load the current file
        content = self.load_file(file_path)
        
        # Find the class and method in the code
        chunks = self.analyzer.analyze_code(content, file_path)
        class_chunk = self.analyzer.get_chunk_by_name(class_name, 'class')
        
        if not class_chunk:
            print(f"Class '{class_name}' not found in {file_path}")
            return False
            
        method_chunk = None
        for child in class_chunk.children:
            if isinstance(child.node, ast.FunctionDef) and child.node.name == method_name:
                method_chunk = child
                break
                
        if not method_chunk:
            print(f"Method '{method_name}' not found in class '{class_name}'")
            return False
            
        # Validate the new method code
        try:
            tree = ast.parse(new_code)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                print("New code must contain exactly one method definition")
                return False
                
            func_node = tree.body[0]
            if func_node.name != method_name:
                print(f"New method name ({func_node.name}) doesn't match the method to replace ({method_name})")
                return False
        except SyntaxError as e:
            print(f"Syntax error in new method code: {str(e)}")
            return False
            
        # Replace the method in the content
        lines = content.split('\n')
        start_line = method_chunk.start_line - 1  # 0-indexed
        end_line = method_chunk.end_line
        
        # Get the indentation level
        indent = re.match(r'^\s*', lines[start_line]).group(0)
        
        # Indent the new code lines
        new_code_lines = new_code.split('\n')
        indented_new_code_lines = [indent + line for line in new_code_lines]
        
        # Create the new content
        new_lines = lines[:start_line] + indented_new_code_lines + lines[end_line:]
        new_content = '\n'.join(new_lines)
        
        # Save the modified content
        return self.save_file(file_path, new_content)
        
    def add_function(self, file_path: str, new_function_code: str, 
                   insert_after: Optional[str] = None) -> bool:
        """
        Add a new function to the file
        If insert_after is provided, add after that function/class, otherwise add at the end
        Returns success boolean
        """
        # Load the current file
        content = self.load_file(file_path)
        
        # Validate the new function code
        try:
            tree = ast.parse(new_function_code)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                print("New code must contain exactly one function definition")
                return False
                
            func_node = tree.body[0]
            function_name = func_node.name
            
            # Check if function already exists
            existing_func = self.analyzer.get_chunk_by_name(function_name, 'function')
            if existing_func:
                print(f"Function '{function_name}' already exists in {file_path}")
                return False
        except SyntaxError as e:
            print(f"Syntax error in new function code: {str(e)}")
            return False
            
        # Determine where to insert the new function
        lines = content.split('\n')
        
        if insert_after:
            # Find the chunk to insert after
            chunk = self.analyzer.get_chunk_by_name(insert_after)
            if not chunk:
                print(f"Function/class '{insert_after}' not found in {file_path}")
                return False
                
            insert_point = chunk.end_line
            # Add the new function after the specified function/class
            new_lines = lines[:insert_point] + ['', new_function_code] + lines[insert_point:]
        else:
            # Add the new function at the end of the file
            new_lines = lines + ['', new_function_code]
            
        new_content = '\n'.join(new_lines)
        
        # Save the modified content
        return self.save_file(file_path, new_content)
        
    def add_method(self, file_path: str, class_name: str, new_method_code: str) -> bool:
        """
        Add a new method to a class
        Returns success boolean
        """
        # Load the current file
        content = self.load_file(file_path)
        
        # Find the class in the code
        chunks = self.analyzer.analyze_code(content, file_path)
        class_chunk = self.analyzer.get_chunk_by_name(class_name, 'class')
        
        if not class_chunk:
            print(f"Class '{class_name}' not found in {file_path}")
            return False
            
        # Validate the new method code
        try:
            tree = ast.parse(new_method_code)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                print("New code must contain exactly one method definition")
                return False
                
            func_node = tree.body[0]
            method_name = func_node.name
            
            # Check if method already exists
            for child in class_chunk.children:
                if isinstance(child.node, ast.FunctionDef) and child.node.name == method_name:
                    print(f"Method '{method_name}' already exists in class '{class_name}'")
                    return False
        except SyntaxError as e:
            print(f"Syntax error in new method code: {str(e)}")
            return False
            
        # Determine where to insert the new method
        lines = content.split('\n')
        
        # Find the last method in the class
        if class_chunk.children:
            last_child = max(class_chunk.children, key=lambda c: c.end_line)
            insert_point = last_child.end_line
        else:
            # If no methods, insert after class definition
            insert_point = class_chunk.start_line
            
        # Get the indentation level
        # Try to determine from last line of class or from class definition
        if class_chunk.children:
            indent = re.match(r'^\s*', lines[last_child.start_line - 1]).group(0)
        else:
            # Default indentation (4 spaces)
            class_line = lines[class_chunk.start_line - 1]
            indent = re.match(r'^\s*', class_line).group(0) + '    '
            
        # Indent the new code lines
        new_code_lines = new_method_code.split('\n')
        indented_new_code_lines = [indent + line for line in new_code_lines]
        
        # Create the new content
        new_lines = lines[:insert_point] + [''] + indented_new_code_lines + lines[insert_point:]
        new_content = '\n'.join(new_lines)
        
        # Save the modified content
        return self.save_file(file_path, new_content)

# === HOT MODULE RELOADING ===
class ModuleReloader:
    """Handles hot reloading of modified Python modules"""
    def __init__(self):
        self.module_cache = {}
        
    def reload_module(self, module_name: str) -> Tuple[bool, Any]:
        """
        Reload a module dynamically
        Returns (success, module)
        """
        try:
            if module_name in sys.modules:
                module = importlib.reload(sys.modules[module_name])
                return True, module
            else:
                module = importlib.import_module(module_name)
                return True, module
        except Exception as e:
            print(f"Error reloading module {module_name}: {str(e)}")
            return False, str(e)
            
    def reload_file(self, file_path: str) -> Tuple[bool, Any]:
        """
        Reload a module from a file path
        Returns (success, module)
        """
        try:
            # Get module name from file path
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Create a spec from the file path
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                return False, f"Could not create spec for {file_path}"
                
            # Create a module from the spec
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules
            sys.modules[module_name] = module
            
            # Execute the module
            spec.loader.exec_module(module)
            
            # Cache the module
            self.module_cache[file_path] = module
            
            return True, module
        except Exception as e:
            print(f"Error reloading file {file_path}: {str(e)}")
            return False, str(e)
            
    def get_module_attributes(self, module) -> Dict:
        """Get all public attributes of a module"""
        return {
            name: value for name, value in inspect.getmembers(module)
            if not name.startswith('_')
        }
        
    def get_function_signature(self, func: Callable) -> Dict:
        """Get function signature information"""
        sig = inspect.signature(func)
        return {
            "name": func.__name__,
            "parameters": [param for param in sig.parameters],
            "return_annotation": str(sig.return_annotation),
            "doc": func.__doc__
        }

# === MAIN EDITOR CLASS ===
class AgentEditor:
    """
    Main editor class for agent self-modification
    Provides tools for analyzing, modifying, and reloading code
    """
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.editor = CodeEditor()
        self.reloader = ModuleReloader()
        
    def analyze_file(self, file_path: str) -> Dict:
        """
        Analyze a file and return its structure
        Returns dict with file info and chunks
        """
        chunks = self.analyzer.analyze_file(file_path)
        
        # Get functions and classes
        functions = [c for c in chunks if isinstance(c.node, ast.FunctionDef)]
        classes = [c for c in chunks if isinstance(c.node, ast.ClassDef)]
        
        return {
            "file_path": file_path,
            "line_count": len(open(file_path).readlines()),
            "functions": [f.get_summary() for f in functions],
            "classes": [c.get_summary() for c in classes]
        }
        
    def extract_function(self, file_path: str, function_name: str) -> Optional[str]:
        """Extract a function's code from a file"""
        chunks = self.analyzer.analyze_file(file_path)
        func_chunk = self.analyzer.get_chunk_by_name(function_name, 'function')
        
        if not func_chunk:
            return None
            
        return func_chunk.extract_code()
        
    def extract_method(self, file_path: str, class_name: str, method_name: str) -> Optional[str]:
        """Extract a method's code from a class"""
        chunks = self.analyzer.analyze_file(file_path)
        class_chunk = self.analyzer.get_chunk_by_name(class_name, 'class')
        
        if not class_chunk:
            return None
            
        for child in class_chunk.children:
            if isinstance(child.node, ast.FunctionDef) and child.node.name == method_name:
                return child.extract_code()
                
        return None
        
    def replace_function(self, file_path: str, function_name: str, new_code: str) -> bool:
        """Replace a function with new code"""
        return self.editor.replace_function(file_path, function_name, new_code)
        
    def replace_method(self, file_path: str, class_name: str, method_name: str, new_code: str) -> bool:
        """Replace a class method with new code"""
        return self.editor.replace_method(file_path, class_name, method_name, new_code)
        
    def add_function(self, file_path: str, new_function_code: str, 
                   insert_after: Optional[str] = None) -> bool:
        """Add a new function to a file"""
        return self.editor.add_function(file_path, new_function_code, insert_after)
        
    def add_method(self, file_path: str, class_name: str, new_method_code: str) -> bool:
        """Add a new method to a class"""
        return self.editor.add_method(file_path, class_name, new_method_code)
        
    def reload_module(self, file_path: str) -> Tuple[bool, Any]:
        """Reload a module after changes"""
        return self.reloader.reload_file(file_path)
        
    def get_snapshots(self, file_path: str) -> List[Dict]:
        """Get modification history for a file"""
        return self.editor.get_snapshots(file_path)
        
    def restore_snapshot(self, file_path: str, index: int) -> bool:
        """Restore a previous version of a file"""
        return self.editor.restore_snapshot(file_path, index)

if __name__ == "__main__":
    # Simple CLI for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Editor for Code Analysis and Modification")
    parser.add_argument("file", help="Path to the file to analyze")
    parser.add_argument("--function", help="Function to extract")
    parser.add_argument("--class", dest="class_name", help="Class to analyze")
    parser.add_argument("--method", help="Method to extract")
    
    args = parser.parse_args()
    
    editor = AgentEditor()
    
    if args.function:
        code = editor.extract_function(args.file, args.function)
        if code:
            print(f"Function '{args.function}':")
            print(code)
        else:
            print(f"Function '{args.function}' not found")
    elif args.class_name and args.method:
        code = editor.extract_method(args.file, args.class_name, args.method)
        if code:
            print(f"Method '{args.class_name}.{args.method}':")
            print(code)
        else:
            print(f"Method '{args.class_name}.{args.method}' not found")
    else:
        analysis = editor.analyze_file(args.file)
        print(f"File: {analysis['file_path']}")
        print(f"Lines: {analysis['line_count']}")
        print(f"Functions: {len(analysis['functions'])}")
        for func in analysis['functions']:
            print(f"  - {func['name']} (lines {func['start_line']}-{func['end_line']})")
        print(f"Classes: {len(analysis['classes'])}")
        for cls in analysis['classes']:
            print(f"  - {cls['name']} (lines {cls['start_line']}-{cls['end_line']})")
            print(f"    Methods: {', '.join(cls['methods'])}")