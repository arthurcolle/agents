#!/usr/bin/env python3
"""
Advanced Code Editor and Network Operations Tool

This module provides advanced capabilities for:
1. Code editing and manipulation
2. Network operations and API interactions
3. Enhanced file system operations
"""

import os
import sys
import re
import json
import shutil
import tempfile
import subprocess
import difflib
import urllib.request
import urllib.parse
import urllib.error
import http.client
import socket
import ssl
import threading
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("editor")

#######################################
# Advanced Code Editing Capabilities
#######################################

class CodeTransformation(Enum):
    """Types of code transformations that can be applied"""
    REFACTOR = "refactor"
    OPTIMIZE = "optimize"
    FORMAT = "format"
    ANALYZE = "analyze"
    GENERATE = "generate"
    COMPLETE = "complete"

@dataclass
class CodeEdit:
    """Represents a single edit operation on code"""
    start_line: int
    end_line: int
    replacement: str
    description: str = ""

@dataclass
class CodeAnalysisResult:
    """Results of code analysis"""
    issues: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    suggestions: List[str]
    complexity: Dict[str, Any]

class CodeEditor:
    """
    Advanced code editor with capabilities for refactoring, optimization,
    and intelligent code manipulation
    """
    
    def __init__(self, workspace_dir: Optional[str] = None):
        """Initialize the code editor with an optional workspace directory"""
        self.workspace_dir = workspace_dir or os.getcwd()
        self.history = []  # Edit history for undo/redo
        self.language_handlers = {
            "python": self._handle_python,
            "javascript": self._handle_javascript,
            "typescript": self._handle_typescript,
            "java": self._handle_java,
            "c": self._handle_c,
            "cpp": self._handle_cpp,
            "go": self._handle_go,
            "rust": self._handle_rust,
            "html": self._handle_html,
            "css": self._handle_css,
            "markdown": self._handle_markdown,
        }
    
    def read_file(self, filepath: str) -> str:
        """Read a file and return its contents"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            raise
    
    def write_file(self, filepath: str, content: str) -> bool:
        """Write content to a file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing to file {filepath}: {e}")
            return False
    
    def apply_edits(self, filepath: str, edits: List[CodeEdit]) -> Tuple[bool, str]:
        """Apply a list of edits to a file"""
        try:
            # Read the file
            content = self.read_file(filepath)
            lines = content.splitlines()
            
            # Sort edits in reverse order to avoid line number changes
            sorted_edits = sorted(edits, key=lambda e: e.start_line, reverse=True)
            
            # Apply edits
            for edit in sorted_edits:
                if edit.start_line < 1 or edit.end_line > len(lines):
                    return False, f"Invalid line range: {edit.start_line}-{edit.end_line}"
                
                # Replace the lines
                replacement_lines = edit.replacement.splitlines()
                lines[edit.start_line-1:edit.end_line] = replacement_lines
            
            # Join lines and write back to file
            new_content = "\n".join(lines)
            if self.write_file(filepath, new_content):
                # Add to history
                self.history.append({
                    "filepath": filepath,
                    "edits": edits,
                    "timestamp": time.time()
                })
                return True, "Edits applied successfully"
            else:
                return False, "Failed to write file"
        except Exception as e:
            logger.error(f"Error applying edits to {filepath}: {e}")
            return False, str(e)
    
    def generate_diff(self, original: str, modified: str) -> str:
        """Generate a unified diff between original and modified text"""
        original_lines = original.splitlines()
        modified_lines = modified.splitlines()
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            lineterm='',
            n=3  # Context lines
        )
        
        return '\n'.join(diff)
    
    def refactor_code(self, filepath: str, transformation: CodeTransformation, 
                     options: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Refactor code using the appropriate language handler
        
        Args:
            filepath: Path to the file to refactor
            transformation: Type of transformation to apply
            options: Additional options for the transformation
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Determine file language from extension
            _, ext = os.path.splitext(filepath)
            ext = ext.lstrip('.').lower()
            
            # Map extension to language
            ext_to_lang = {
                'py': 'python',
                'js': 'javascript',
                'ts': 'typescript',
                'java': 'java',
                'c': 'c',
                'cpp': 'cpp',
                'cc': 'cpp',
                'h': 'c',
                'hpp': 'cpp',
                'go': 'go',
                'rs': 'rust',
                'html': 'html',
                'css': 'css',
                'md': 'markdown',
            }
            
            language = ext_to_lang.get(ext)
            if not language:
                return False, f"Unsupported file extension: {ext}"
            
            # Get the appropriate handler
            handler = self.language_handlers.get(language)
            if not handler:
                return False, f"No handler available for language: {language}"
            
            # Read the file
            content = self.read_file(filepath)
            
            # Apply the transformation
            options = options or {}
            success, result = handler(content, transformation, options)
            
            if success:
                # Write the result back to the file
                if self.write_file(filepath, result):
                    return True, f"Successfully applied {transformation.value} transformation"
                else:
                    return False, "Failed to write transformed code to file"
            else:
                return False, f"Failed to apply {transformation.value} transformation: {result}"
        except Exception as e:
            logger.error(f"Error refactoring {filepath}: {e}")
            return False, str(e)
    
    def analyze_code(self, filepath: str) -> Tuple[bool, Union[CodeAnalysisResult, str]]:
        """
        Analyze code for quality, complexity, and potential issues
        
        Args:
            filepath: Path to the file to analyze
            
        Returns:
            Tuple of (success, result)
            where result is either a CodeAnalysisResult or an error message
        """
        try:
            # Determine file language from extension
            _, ext = os.path.splitext(filepath)
            ext = ext.lstrip('.').lower()
            
            # Map extension to language
            ext_to_lang = {
                'py': 'python',
                'js': 'javascript',
                'ts': 'typescript',
                'java': 'java',
                'c': 'c',
                'cpp': 'cpp',
                'cc': 'cpp',
                'h': 'c',
                'hpp': 'cpp',
                'go': 'go',
                'rs': 'rust',
                'html': 'html',
                'css': 'css',
                'md': 'markdown',
            }
            
            language = ext_to_lang.get(ext)
            if not language:
                return False, f"Unsupported file extension: {ext}"
            
            # Get the appropriate handler
            handler = self.language_handlers.get(language)
            if not handler:
                return False, f"No handler available for language: {language}"
            
            # Read the file
            content = self.read_file(filepath)
            
            # Apply the analysis transformation
            success, result = handler(content, CodeTransformation.ANALYZE, {})
            
            if success and isinstance(result, dict):
                # Convert the result to a CodeAnalysisResult
                analysis_result = CodeAnalysisResult(
                    issues=result.get('issues', []),
                    metrics=result.get('metrics', {}),
                    suggestions=result.get('suggestions', []),
                    complexity=result.get('complexity', {})
                )
                return True, analysis_result
            else:
                return False, f"Failed to analyze code: {result}"
        except Exception as e:
            logger.error(f"Error analyzing {filepath}: {e}")
            return False, str(e)
    
    def _handle_python(self, content: str, transformation: CodeTransformation, 
                      options: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle Python code transformations"""
        if transformation == CodeTransformation.FORMAT:
            try:
                # Use black for formatting if available
                with tempfile.NamedTemporaryFile(suffix='.py', mode='w+', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp.flush()
                    
                    try:
                        result = subprocess.run(
                            ['black', tmp.name],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Read the formatted content
                        tmp.seek(0)
                        formatted_content = tmp.read()
                        return True, formatted_content
                    except subprocess.CalledProcessError as e:
                        return False, f"Black formatting failed: {e.stderr}"
                    except FileNotFoundError:
                        # Black not installed, fall back to basic formatting
                        pass
            except Exception as e:
                logger.warning(f"Error using black for formatting: {e}")
            
            # Basic formatting fallback
            import ast
            try:
                # Parse the code to ensure it's valid
                ast.parse(content)
                
                # Apply basic formatting (indentation, etc.)
                lines = content.splitlines()
                formatted_lines = []
                indent_level = 0
                
                for line in lines:
                    stripped = line.strip()
                    
                    # Adjust indent level based on content
                    if stripped.endswith(':'):
                        formatted_lines.append('    ' * indent_level + stripped)
                        indent_level += 1
                    elif stripped in ('break', 'continue', 'pass', 'return'):
                        formatted_lines.append('    ' * indent_level + stripped)
                    elif stripped.startswith(('elif', 'else', 'except', 'finally')):
                        indent_level = max(0, indent_level - 1)
                        formatted_lines.append('    ' * indent_level + stripped)
                        indent_level += 1
                    else:
                        formatted_lines.append('    ' * indent_level + stripped)
                
                return True, '\n'.join(formatted_lines)
            except SyntaxError as e:
                return False, f"Syntax error in Python code: {e}"
        
        elif transformation == CodeTransformation.ANALYZE:
            try:
                import ast
                
                # Parse the code
                tree = ast.parse(content)
                
                # Initialize analysis results
                issues = []
                metrics = {
                    "lines_of_code": len(content.splitlines()),
                    "character_count": len(content),
                }
                suggestions = []
                complexity = {}
                
                # Find imports
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            imports.append(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)
                
                metrics["imports"] = imports
                
                # Find functions and classes
                functions = []
                classes = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                
                metrics["functions"] = functions
                metrics["classes"] = classes
                
                # Check for potential issues
                for node in ast.walk(tree):
                    # Check for bare excepts
                    if isinstance(node, ast.ExceptHandler) and node.type is None:
                        issues.append({
                            "type": "bare_except",
                            "message": "Bare except clause should be avoided",
                            "line": node.lineno
                        })
                    
                    # Check for mutable default arguments
                    if isinstance(node, ast.FunctionDef):
                        for arg in node.args.defaults:
                            if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                                issues.append({
                                    "type": "mutable_default",
                                    "message": f"Mutable default argument in function {node.name}",
                                    "line": node.lineno
                                })
                
                # Add suggestions
                if len(functions) > 10 and len(classes) == 0:
                    suggestions.append("Consider organizing code into classes for better structure")
                
                if any("import *" in line for line in content.splitlines()):
                    suggestions.append("Avoid wildcard imports (import *) for better code clarity")
                
                # Calculate complexity metrics
                try:
                    # Try to use radon for cyclomatic complexity if available
                    import radon.complexity as cc
                    
                    with tempfile.NamedTemporaryFile(suffix='.py', mode='w+', encoding='utf-8') as tmp:
                        tmp.write(content)
                        tmp.flush()
                        
                        complexity_metrics = cc.cc_visit(tmp.name)
                        complexity = {
                            "functions": [
                                {
                                    "name": metric.name,
                                    "complexity": metric.complexity,
                                    "rank": cc.rank(metric.complexity)
                                }
                                for metric in complexity_metrics
                            ]
                        }
                except ImportError:
                    # Radon not available, use basic complexity estimation
                    complexity = {
                        "estimation": "Basic complexity estimation (radon not available)",
                        "branches": content.count("if ") + content.count("elif ") + content.count("else:") + 
                                   content.count("for ") + content.count("while ") + content.count("try:") +
                                   content.count("except") + content.count("with ")
                    }
                
                return True, {
                    "issues": issues,
                    "metrics": metrics,
                    "suggestions": suggestions,
                    "complexity": complexity
                }
            except SyntaxError as e:
                return False, f"Syntax error in Python code: {e}"
            except Exception as e:
                return False, f"Error analyzing Python code: {e}"
        
        elif transformation == CodeTransformation.REFACTOR:
            # Basic refactoring operations
            refactor_type = options.get("type", "extract_function")
            
            if refactor_type == "extract_function":
                start_line = options.get("start_line")
                end_line = options.get("end_line")
                function_name = options.get("function_name", "extracted_function")
                
                if not all([start_line, end_line, function_name]):
                    return False, "Missing required options for extract_function"
                
                lines = content.splitlines()
                if start_line < 1 or end_line > len(lines):
                    return False, f"Invalid line range: {start_line}-{end_line}"
                
                # Extract the code to be refactored
                code_to_extract = lines[start_line-1:end_line]
                
                # Determine indentation
                indent = ""
                for line in code_to_extract:
                    if line.strip():
                        indent = re.match(r'^(\s*)', line).group(1)
                        break
                
                # Remove common indentation
                if indent:
                    code_to_extract = [line[len(indent):] if line.startswith(indent) else line for line in code_to_extract]
                
                # Create the function
                function_def = [f"def {function_name}():"]
                function_body = ["    " + line for line in code_to_extract]
                
                # Replace the extracted code with a function call
                lines[start_line-1:end_line] = [f"{indent}{function_name}()"]
                
                # Insert the function before the extraction point
                insertion_point = start_line - 1
                while insertion_point > 0 and lines[insertion_point-1].strip() == "":
                    insertion_point -= 1
                
                lines[insertion_point:insertion_point] = [""] + function_def + function_body + [""]
                
                return True, "\n".join(lines)
            
            elif refactor_type == "rename_variable":
                old_name = options.get("old_name")
                new_name = options.get("new_name")
                
                if not all([old_name, new_name]):
                    return False, "Missing required options for rename_variable"
                
                # Simple string replacement (not semantically aware)
                # For a real implementation, use the ast module to find variable references
                pattern = r'\b' + re.escape(old_name) + r'\b'
                refactored_content = re.sub(pattern, new_name, content)
                
                return True, refactored_content
            
            else:
                return False, f"Unsupported refactor type: {refactor_type}"
        
        else:
            return False, f"Unsupported transformation for Python: {transformation.value}"
    
    def _handle_javascript(self, content: str, transformation: CodeTransformation, 
                          options: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle JavaScript code transformations"""
        if transformation == CodeTransformation.FORMAT:
            try:
                # Use prettier for formatting if available
                with tempfile.NamedTemporaryFile(suffix='.js', mode='w+', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp.flush()
                    
                    try:
                        result = subprocess.run(
                            ['prettier', '--write', tmp.name],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Read the formatted content
                        tmp.seek(0)
                        formatted_content = tmp.read()
                        return True, formatted_content
                    except subprocess.CalledProcessError as e:
                        return False, f"Prettier formatting failed: {e.stderr}"
                    except FileNotFoundError:
                        # Prettier not installed, fall back to basic formatting
                        pass
            except Exception as e:
                logger.warning(f"Error using prettier for formatting: {e}")
            
            # Basic formatting fallback
            # This is a very simplistic approach and won't handle all cases correctly
            try:
                lines = content.splitlines()
                formatted_lines = []
                indent_level = 0
                
                for line in lines:
                    stripped = line.strip()
                    
                    # Adjust indent level based on content
                    if stripped.endswith('{'):
                        formatted_lines.append('  ' * indent_level + stripped)
                        indent_level += 1
                    elif stripped.startswith('}'):
                        indent_level = max(0, indent_level - 1)
                        formatted_lines.append('  ' * indent_level + stripped)
                    else:
                        formatted_lines.append('  ' * indent_level + stripped)
                
                return True, '\n'.join(formatted_lines)
            except Exception as e:
                return False, f"Error formatting JavaScript code: {e}"
        
        elif transformation == CodeTransformation.ANALYZE:
            # Basic JavaScript analysis
            lines = content.splitlines()
            
            issues = []
            metrics = {
                "lines_of_code": len(lines),
                "character_count": len(content),
            }
            suggestions = []
            complexity = {}
            
            # Check for common issues
            for i, line in enumerate(lines):
                # Check for console.log statements
                if "console.log" in line:
                    issues.append({
                        "type": "console_log",
                        "message": "Console.log statement should be removed in production code",
                        "line": i + 1
                    })
                
                # Check for alert statements
                if "alert(" in line:
                    issues.append({
                        "type": "alert",
                        "message": "Alert statements should be avoided in modern web applications",
                        "line": i + 1
                    })
                
                # Check for eval
                if "eval(" in line:
                    issues.append({
                        "type": "eval",
                        "message": "Eval is dangerous and should be avoided",
                        "line": i + 1
                    })
            
            # Count functions and classes
            function_count = sum(1 for line in lines if re.search(r'function\s+\w+\s*\(', line))
            class_count = sum(1 for line in lines if re.search(r'class\s+\w+', line))
            
            metrics["function_count"] = function_count
            metrics["class_count"] = class_count
            
            # Add suggestions
            if "var " in content:
                suggestions.append("Consider using 'let' and 'const' instead of 'var' for better scoping")
            
            if not "use strict" in content:
                suggestions.append("Add 'use strict' directive for safer JavaScript execution")
            
            # Basic complexity estimation
            complexity = {
                "estimation": "Basic complexity estimation",
                "branches": content.count("if ") + content.count("else ") + 
                           content.count("for ") + content.count("while ") + 
                           content.count("switch ") + content.count("case ") +
                           content.count("try ") + content.count("catch ")
            }
            
            return True, {
                "issues": issues,
                "metrics": metrics,
                "suggestions": suggestions,
                "complexity": complexity
            }
        
        else:
            return False, f"Unsupported transformation for JavaScript: {transformation.value}"
    
    def _handle_typescript(self, content: str, transformation: CodeTransformation, 
                          options: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle TypeScript code transformations"""
        # For now, delegate to JavaScript handler with some TypeScript-specific additions
        if transformation == CodeTransformation.ANALYZE:
            success, result = self._handle_javascript(content, transformation, options)
            
            if success and isinstance(result, dict):
                # Add TypeScript-specific analysis
                issues = result.get("issues", [])
                suggestions = result.get("suggestions", [])
                
                # Check for any type
                if "any" in content:
                    issues.append({
                        "type": "any_type",
                        "message": "Avoid using 'any' type as it defeats TypeScript's type checking",
                        "line": None  # Would need more complex parsing to determine line numbers
                    })
                
                # Check for non-null assertions
                non_null_count = content.count("!")
                if non_null_count > 0:
                    issues.append({
                        "type": "non_null_assertion",
                        "message": f"Found {non_null_count} non-null assertions (!), which can lead to runtime errors",
                        "line": None
                    })
                
                # Add TypeScript-specific suggestions
                if "interface " in content and "type " not in content:
                    suggestions.append("Consider using 'type' aliases for more complex types")
                
                result["issues"] = issues
                result["suggestions"] = suggestions
            
            return success, result
        else:
            return self._handle_javascript(content, transformation, options)
    
    def _handle_java(self, content: str, transformation: CodeTransformation, 
                    options: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle Java code transformations"""
        if transformation == CodeTransformation.FORMAT:
            try:
                # Use google-java-format if available
                with tempfile.NamedTemporaryFile(suffix='.java', mode='w+', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp.flush()
                    
                    try:
                        result = subprocess.run(
                            ['google-java-format', tmp.name],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Read the formatted content from stdout
                        formatted_content = result.stdout
                        return True, formatted_content
                    except subprocess.CalledProcessError as e:
                        return False, f"Java formatting failed: {e.stderr}"
                    except FileNotFoundError:
                        # google-java-format not installed, fall back to basic formatting
                        pass
            except Exception as e:
                logger.warning(f"Error using google-java-format: {e}")
            
            # Basic formatting fallback
            try:
                lines = content.splitlines()
                formatted_lines = []
                indent_level = 0
                
                for line in lines:
                    stripped = line.strip()
                    
                    # Adjust indent level based on content
                    if stripped.endswith('{'):
                        formatted_lines.append('    ' * indent_level + stripped)
                        indent_level += 1
                    elif stripped.startswith('}'):
                        indent_level = max(0, indent_level - 1)
                        formatted_lines.append('    ' * indent_level + stripped)
                    else:
                        formatted_lines.append('    ' * indent_level + stripped)
                
                return True, '\n'.join(formatted_lines)
            except Exception as e:
                return False, f"Error formatting Java code: {e}"
        
        elif transformation == CodeTransformation.ANALYZE:
            # Basic Java analysis
            lines = content.splitlines()
            
            issues = []
            metrics = {
                "lines_of_code": len(lines),
                "character_count": len(content),
            }
            suggestions = []
            complexity = {}
            
            # Count classes, methods, and fields
            class_pattern = r'(public|private|protected)?\s+class\s+(\w+)'
            method_pattern = r'(public|private|protected)?\s+\w+\s+(\w+)\s*\([^)]*\)\s*(\{|throws)'
            field_pattern = r'(public|private|protected)?\s+\w+\s+(\w+)\s*;'
            
            classes = re.findall(class_pattern, content)
            methods = re.findall(method_pattern, content)
            fields = re.findall(field_pattern, content)
            
            metrics["class_count"] = len(classes)
            metrics["method_count"] = len(methods)
            metrics["field_count"] = len(fields)
            
            # Check for common issues
            for i, line in enumerate(lines):
                # Check for System.out.println
                if "System.out.println" in line:
                    issues.append({
                        "type": "println",
                        "message": "System.out.println should be replaced with proper logging in production code",
                        "line": i + 1
                    })
                
                # Check for empty catch blocks
                if re.search(r'catch\s*\([^)]+\)\s*\{\s*\}', line):
                    issues.append({
                        "type": "empty_catch",
                        "message": "Empty catch block should include at least a comment or logging",
                        "line": i + 1
                    })
            
            # Add suggestions
            if "extends Thread" in content:
                suggestions.append("Consider implementing Runnable instead of extending Thread for better flexibility")
            
            if "synchronized" in content:
                suggestions.append("Consider using java.util.concurrent utilities instead of synchronized blocks")
            
            # Basic complexity estimation
            complexity = {
                "estimation": "Basic complexity estimation",
                "branches": content.count("if ") + content.count("else ") + 
                           content.count("for ") + content.count("while ") + 
                           content.count("switch ") + content.count("case ") +
                           content.count("try ") + content.count("catch ")
            }
            
            return True, {
                "issues": issues,
                "metrics": metrics,
                "suggestions": suggestions,
                "complexity": complexity
            }
        
        else:
            return False, f"Unsupported transformation for Java: {transformation.value}"
    
    def _handle_c(self, content: str, transformation: CodeTransformation, 
                 options: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle C code transformations"""
        # Basic implementation for C
        if transformation == CodeTransformation.FORMAT:
            try:
                # Use clang-format if available
                with tempfile.NamedTemporaryFile(suffix='.c', mode='w+', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp.flush()
                    
                    try:
                        result = subprocess.run(
                            ['clang-format', tmp.name],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Read the formatted content from stdout
                        formatted_content = result.stdout
                        return True, formatted_content
                    except subprocess.CalledProcessError as e:
                        return False, f"C formatting failed: {e.stderr}"
                    except FileNotFoundError:
                        # clang-format not installed, fall back to basic formatting
                        pass
            except Exception as e:
                logger.warning(f"Error using clang-format: {e}")
            
            # Basic formatting fallback
            try:
                lines = content.splitlines()
                formatted_lines = []
                indent_level = 0
                
                for line in lines:
                    stripped = line.strip()
                    
                    # Adjust indent level based on content
                    if stripped.endswith('{'):
                        formatted_lines.append('    ' * indent_level + stripped)
                        indent_level += 1
                    elif stripped.startswith('}'):
                        indent_level = max(0, indent_level - 1)
                        formatted_lines.append('    ' * indent_level + stripped)
                    else:
                        formatted_lines.append('    ' * indent_level + stripped)
                
                return True, '\n'.join(formatted_lines)
            except Exception as e:
                return False, f"Error formatting C code: {e}"
        
        elif transformation == CodeTransformation.ANALYZE:
            # Basic C analysis
            lines = content.splitlines()
            
            issues = []
            metrics = {
                "lines_of_code": len(lines),
                "character_count": len(content),
            }
            suggestions = []
            complexity = {}
            
            # Count functions
            function_pattern = r'\w+\s+(\w+)\s*\([^)]*\)\s*\{'
            functions = re.findall(function_pattern, content)
            
            metrics["function_count"] = len(functions)
            
            # Check for common issues
            for i, line in enumerate(lines):
                # Check for gets (unsafe)
                if "gets(" in line:
                    issues.append({
                        "type": "gets",
                        "message": "gets() is unsafe and has been deprecated, use fgets() instead",
                        "line": i + 1
                    })
                
                # Check for strcpy (potentially unsafe)
                if "strcpy(" in line:
                    issues.append({
                        "type": "strcpy",
                        "message": "strcpy() can lead to buffer overflows, consider strncpy() or strlcpy()",
                        "line": i + 1
                    })
                
                # Check for malloc without free
                if "malloc(" in line:
                    issues.append({
                        "type": "malloc",
                        "message": "Ensure malloc() is paired with free() to avoid memory leaks",
                        "line": i + 1
                    })
            
            # Add suggestions
            if "goto" in content:
                suggestions.append("Avoid using goto statements as they make code harder to understand")
            
            if "#define" in content:
                suggestions.append("Consider using const variables instead of #define for better type safety")
            
            # Basic complexity estimation
            complexity = {
                "estimation": "Basic complexity estimation",
                "branches": content.count("if ") + content.count("else ") + 
                           content.count("for ") + content.count("while ") + 
                           content.count("switch ") + content.count("case ")
            }
            
            return True, {
                "issues": issues,
                "metrics": metrics,
                "suggestions": suggestions,
                "complexity": complexity
            }
        
        else:
            return False, f"Unsupported transformation for C: {transformation.value}"
    
    def _handle_cpp(self, content: str, transformation: CodeTransformation, 
                   options: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle C++ code transformations"""
        # For now, extend the C handler with C++-specific additions
        if transformation == CodeTransformation.ANALYZE:
            success, result = self._handle_c(content, transformation, options)
            
            if success and isinstance(result, dict):
                # Add C++-specific analysis
                issues = result.get("issues", [])
                suggestions = result.get("suggestions", [])
                
                # Check for using namespace std
                if "using namespace std" in content:
                    issues.append({
                        "type": "using_namespace",
                        "message": "Avoid 'using namespace std' in header files as it pollutes the global namespace",
                        "line": None
                    })
                
                # Check for raw pointers
                raw_pointer_count = content.count("new ") + content.count("delete ")
                if raw_pointer_count > 0:
                    suggestions.append("Consider using smart pointers (std::unique_ptr, std::shared_ptr) instead of raw pointers")
                
                # Check for C-style casts
                c_cast_pattern = r'\([a-zA-Z_][a-zA-Z0-9_]*\s*\*?\)'
                c_casts = re.findall(c_cast_pattern, content)
                if c_casts:
                    suggestions.append("Use C++ style casts (static_cast, dynamic_cast) instead of C-style casts")
                
                result["issues"] = issues
                result["suggestions"] = suggestions
            
            return success, result
        else:
            return self._handle_c(content, transformation, options)
    
    def _handle_go(self, content: str, transformation: CodeTransformation, 
                  options: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle Go code transformations"""
        if transformation == CodeTransformation.FORMAT:
            try:
                # Use gofmt if available
                with tempfile.NamedTemporaryFile(suffix='.go', mode='w+', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp.flush()
                    
                    try:
                        result = subprocess.run(
                            ['gofmt', tmp.name],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Read the formatted content from stdout
                        formatted_content = result.stdout
                        return True, formatted_content
                    except subprocess.CalledProcessError as e:
                        return False, f"Go formatting failed: {e.stderr}"
                    except FileNotFoundError:
                        # gofmt not installed, fall back to basic formatting
                        pass
            except Exception as e:
                logger.warning(f"Error using gofmt: {e}")
            
            # For Go, we don't provide a fallback formatter as gofmt is the standard
            return False, "gofmt not available and no fallback formatter implemented for Go"
        
        elif transformation == CodeTransformation.ANALYZE:
            # Basic Go analysis
            lines = content.splitlines()
            
            issues = []
            metrics = {
                "lines_of_code": len(lines),
                "character_count": len(content),
            }
            suggestions = []
            complexity = {}
            
            # Count functions and structs
            function_pattern = r'func\s+(\w+)'
            struct_pattern = r'type\s+(\w+)\s+struct'
            
            functions = re.findall(function_pattern, content)
            structs = re.findall(struct_pattern, content)
            
            metrics["function_count"] = len(functions)
            metrics["struct_count"] = len(structs)
            
            # Check for common issues
            for i, line in enumerate(lines):
                # Check for naked returns
                if re.search(r'func\s+\w+[^{]*\)\s*\([^)]+\)\s*{', line):
                    if "return" in content and not re.search(r'return\s+\w+', content):
                        issues.append({
                            "type": "naked_return",
                            "message": "Naked returns can reduce code clarity, consider explicit returns",
                            "line": i + 1
                        })
                
                # Check for fmt.Println in production code
                if "fmt.Println" in line:
                    issues.append({
                        "type": "fmt_println",
                        "message": "fmt.Println should be replaced with proper logging in production code",
                        "line": i + 1
                    })
            
            # Add suggestions
            if "var " in content and ":=" not in content:
                suggestions.append("Consider using short variable declarations (:=) where appropriate")
            
            if "panic(" in content:
                suggestions.append("Avoid using panic() in production code, handle errors explicitly")
            
            # Basic complexity estimation
            complexity = {
                "estimation": "Basic complexity estimation",
                "branches": content.count("if ") + content.count("else ") + 
                           content.count("for ") + content.count("switch ") + 
                           content.count("case ") + content.count("select ")
            }
            
            return True, {
                "issues": issues,
                "metrics": metrics,
                "suggestions": suggestions,
                "complexity": complexity
            }
        
        else:
            return False, f"Unsupported transformation for Go: {transformation.value}"
    
    def _handle_rust(self, content: str, transformation: CodeTransformation, 
                    options: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle Rust code transformations"""
        if transformation == CodeTransformation.FORMAT:
            try:
                # Use rustfmt if available
                with tempfile.NamedTemporaryFile(suffix='.rs', mode='w+', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp.flush()
                    
                    try:
                        result = subprocess.run(
                            ['rustfmt', tmp.name],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Read the formatted content
                        with open(tmp.name, 'r', encoding='utf-8') as f:
                            formatted_content = f.read()
                        return True, formatted_content
                    except subprocess.CalledProcessError as e:
                        return False, f"Rust formatting failed: {e.stderr}"
                    except FileNotFoundError:
                        # rustfmt not installed, fall back to basic formatting
                        pass
            except Exception as e:
                logger.warning(f"Error using rustfmt: {e}")
            
            # For Rust, we don't provide a fallback formatter as rustfmt is the standard
            return False, "rustfmt not available and no fallback formatter implemented for Rust"
        
        elif transformation == CodeTransformation.ANALYZE:
            # Basic Rust analysis
            lines = content.splitlines()
            
            issues = []
            metrics = {
                "lines_of_code": len(lines),
                "character_count": len(content),
            }
            suggestions = []
            complexity = {}
            
            # Count functions, structs, and enums
            function_pattern = r'fn\s+(\w+)'
            struct_pattern = r'struct\s+(\w+)'
            enum_pattern = r'enum\s+(\w+)'
            
            functions = re.findall(function_pattern, content)
            structs = re.findall(struct_pattern, content)
            enums = re.findall(enum_pattern, content)
            
            metrics["function_count"] = len(functions)
            metrics["struct_count"] = len(structs)
            metrics["enum_count"] = len(enums)
            
            # Check for common issues
            for i, line in enumerate(lines):
                # Check for unwrap() without error handling
                if ".unwrap()" in line:
                    issues.append({
                        "type": "unwrap",
                        "message": "unwrap() can cause panics, consider proper error handling",
                        "line": i + 1
                    })
                
                # Check for expect() without error handling
                if ".expect(" in line:
                    issues.append({
                        "type": "expect",
                        "message": "expect() can cause panics, consider proper error handling",
                        "line": i + 1
                    })
                
                # Check for println! in production code
                if "println!" in line:
                    issues.append({
                        "type": "println",
                        "message": "println! should be replaced with proper logging in production code",
                        "line": i + 1
                    })
            
            # Add suggestions
            if "unsafe" in content:
                suggestions.append("Minimize the use of unsafe blocks and document why they are necessary")
            
            if "mut " in content:
                suggestions.append("Consider reducing the number of mutable variables for better safety")
            
            # Basic complexity estimation
            complexity = {
                "estimation": "Basic complexity estimation",
                "branches": content.count("if ") + content.count("else ") + 
                           content.count("for ") + content.count("while ") + 
                           content.count("match ") + content.count("loop ")
            }
            
            return True, {
                "issues": issues,
                "metrics": metrics,
                "suggestions": suggestions,
                "complexity": complexity
            }
        
        else:
            return False, f"Unsupported transformation for Rust: {transformation.value}"
    
    def _handle_html(self, content: str, transformation: CodeTransformation, 
                    options: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle HTML code transformations"""
        if transformation == CodeTransformation.FORMAT:
            try:
                # Use html-tidy if available
                with tempfile.NamedTemporaryFile(suffix='.html', mode='w+', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp.flush()
                    
                    try:
                        result = subprocess.run(
                            ['tidy', '-i', '-w', '0', '-q', tmp.name],
                            capture_output=True,
                            text=True
                        )
                        
                        # Read the formatted content from stdout
                        formatted_content = result.stdout
                        return True, formatted_content
                    except subprocess.CalledProcessError as e:
                        # tidy returns non-zero exit code for warnings too
                        if e.stdout:
                            return True, e.stdout
                        return False, f"HTML formatting failed: {e.stderr}"
                    except FileNotFoundError:
                        # html-tidy not installed, fall back to basic formatting
                        pass
            except Exception as e:
                logger.warning(f"Error using html-tidy: {e}")
            
            # Basic HTML formatting
            try:
                import html.parser
                
                class HTMLFormatter(html.parser.HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.result = []
                        self.indent = 0
                        self.in_pre = False
                    
                    def handle_starttag(self, tag, attrs):
                        if tag == 'pre':
                            self.in_pre = True
                        
                        if not self.in_pre:
                            self.result.append('  ' * self.indent + self.get_starttag_text())
                            if tag not in ['br', 'img', 'input', 'hr', 'meta', 'link']:
                                self.indent += 1
                        else:
                            self.result.append(self.get_starttag_text())
                    
                    def handle_endtag(self, tag):
                        if not self.in_pre:
                            if tag not in ['br', 'img', 'input', 'hr', 'meta', 'link']:
                                self.indent -= 1
                            self.result.append('  ' * self.indent + f'</{tag}>')
                        else:
                            self.result.append(f'</{tag}>')
                        
                        if tag == 'pre':
                            self.in_pre = False
                    
                    def handle_data(self, data):
                        if not self.in_pre:
                            data = data.strip()
                            if data:
                                self.result.append('  ' * self.indent + data)
                        else:
                            self.result.append(data)
                    
                    def handle_comment(self, data):
                        if not self.in_pre:
                            self.result.append('  ' * self.indent + f'<!--{data}-->')
                        else:
                            self.result.append(f'<!--{data}-->')
                
                formatter = HTMLFormatter()
                formatter.feed(content)
                
                return True, '\n'.join(formatter.result)
            except Exception as e:
                return False, f"Error formatting HTML code: {e}"
        
        elif transformation == CodeTransformation.ANALYZE:
            # Basic HTML analysis
            lines = content.splitlines()
            
            issues = []
            metrics = {
                "lines_of_code": len(lines),
                "character_count": len(content),
            }
            suggestions = []
            complexity = {}
            
            # Count tags
            tag_pattern = r'<(\w+)[^>]*>'
            tags = re.findall(tag_pattern, content)
            
            tag_counts = {}
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            metrics["tag_counts"] = tag_counts
            metrics["total_tags"] = len(tags)
            
            # Check for common issues
            for i, line in enumerate(lines):
                # Check for deprecated tags
                deprecated_tags = ['font', 'center', 'strike', 'marquee', 'frame', 'frameset']
                for tag in deprecated_tags:
                    if f'<{tag}' in line.lower():
                        issues.append({
                            "type": "deprecated_tag",
                            "message": f"The <{tag}> tag is deprecated in HTML5",
                            "line": i + 1
                        })
                
                # Check for inline styles
                if 'style="' in line:
                    issues.append({
                        "type": "inline_style",
                        "message": "Inline styles should be avoided in favor of external CSS",
                        "line": i + 1
                    })
                
                # Check for missing alt attributes in images
                if '<img' in line.lower() and 'alt="' not in line.lower():
                    issues.append({
                        "type": "missing_alt",
                        "message": "Images should have alt attributes for accessibility",
                        "line": i + 1
                    })
            
            # Add suggestions
            if '<!DOCTYPE html>' not in content:
                suggestions.append("Add <!DOCTYPE html> declaration at the beginning of the document")
            
            if '<table' in content.lower() and '<th' not in content.lower():
                suggestions.append("Use <th> elements for table headers for better accessibility")
            
            # Basic complexity estimation
            complexity = {
                "estimation": "Basic complexity estimation",
                "nesting_level": max([line.count('<') - line.count('</') for line in lines]),
                "unique_tags": len(set(tags))
            }
            
            return True, {
                "issues": issues,
                "metrics": metrics,
                "suggestions": suggestions,
                "complexity": complexity
            }
        
        else:
            return False, f"Unsupported transformation for HTML: {transformation.value}"
    
    def _handle_css(self, content: str, transformation: CodeTransformation, 
                   options: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle CSS code transformations"""
        if transformation == CodeTransformation.FORMAT:
            try:
                # Use prettier for formatting if available
                with tempfile.NamedTemporaryFile(suffix='.css', mode='w+', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp.flush()
                    
                    try:
                        result = subprocess.run(
                            ['prettier', '--parser', 'css', tmp.name],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Read the formatted content from stdout
                        formatted_content = result.stdout
                        return True, formatted_content
                    except subprocess.CalledProcessError as e:
                        return False, f"CSS formatting failed: {e.stderr}"
                    except FileNotFoundError:
                        # prettier not installed, fall back to basic formatting
                        pass
            except Exception as e:
                logger.warning(f"Error using prettier for CSS: {e}")
            
            # Basic CSS formatting
            try:
                # Very simplistic CSS formatter
                result = []
                in_rule = False
                indent = 0
                
                for line in content.splitlines():
                    line = line.strip()
                    
                    if not line:
                        result.append('')
                        continue
                    
                    if '{' in line:
                        in_rule = True
                        if '}' in line:
                            # Single-line rule
                            result.append(line)
                            in_rule = False
                        else:
                            # Start of multi-line rule
                            result.append(line)
                            indent = 2
                    elif '}' in line:
                        # End of rule
                        in_rule = False
                        indent = 0
                        result.append(line)
                    elif in_rule:
                        # Inside a rule
                        result.append(' ' * indent + line)
                    else:
                        # Outside a rule
                        result.append(line)
                
                return True, '\n'.join(result)
            except Exception as e:
                return False, f"Error formatting CSS code: {e}"
        
        elif transformation == CodeTransformation.ANALYZE:
            # Basic CSS analysis
            lines = content.splitlines()
            
            issues = []
            metrics = {
                "lines_of_code": len(lines),
                "character_count": len(content),
            }
            suggestions = []
            complexity = {}
            
            # Count selectors and properties
            selector_pattern = r'([^{]+)\s*\{'
            property_pattern = r'([a-zA-Z-]+)\s*:'
            
            selectors = re.findall(selector_pattern, content)
            properties = re.findall(property_pattern, content)
            
            # Clean up selectors
            selectors = [s.strip() for s in selectors]
            
            metrics["selector_count"] = len(selectors)
            metrics["property_count"] = len(properties)
            
            # Count unique properties
            unique_properties = set(properties)
            metrics["unique_property_count"] = len(unique_properties)
            
            # Check for common issues
            for i, line in enumerate(lines):
                # Check for !important
                if '!important' in line:
                    issues.append({
                        "type": "important",
                        "message": "Avoid using !important as it breaks the natural cascading of CSS",
                        "line": i + 1
                    })
                
                # Check for vendor prefixes
                if re.search(r'-(webkit|moz|ms|o)-', line):
                    issues.append({
                        "type": "vendor_prefix",
                        "message": "Consider using a CSS preprocessor or autoprefixer for vendor prefixes",
                        "line": i + 1
                    })
                
                # Check for complex selectors
                if '>' in line or '+' in line or '~' in line:
                    if line.count('>') + line.count('+') + line.count('~') > 2:
                        issues.append({
                            "type": "complex_selector",
                            "message": "Overly complex selectors can impact performance",
                            "line": i + 1
                        })
            
            # Add suggestions
            if '@media' not in content:
                suggestions.append("Consider using media queries for responsive design")
            
            if 'px' in content and 'rem' not in content and 'em' not in content:
                suggestions.append("Consider using relative units (rem, em) instead of pixels for better accessibility")
            
            # Check for potential overspecificity
            id_selectors = [s for s in selectors if '#' in s]
            if len(id_selectors) > len(selectors) / 3:
                suggestions.append("Reduce the use of ID selectors to avoid specificity issues")
            
            # Basic complexity estimation
            complexity = {
                "estimation": "Basic complexity estimation",
                "selector_specificity": {
                    "id_selectors": sum(s.count('#') for s in selectors),
                    "class_selectors": sum(s.count('.') for s in selectors),
                    "element_selectors": sum(len(re.findall(r'\b[a-zA-Z]+\b', s)) for s in selectors)
                }
            }
            
            return True, {
                "issues": issues,
                "metrics": metrics,
                "suggestions": suggestions,
                "complexity": complexity
            }
        
        else:
            return False, f"Unsupported transformation for CSS: {transformation.value}"
    
    def _handle_markdown(self, content: str, transformation: CodeTransformation, 
                        options: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle Markdown code transformations"""
        if transformation == CodeTransformation.FORMAT:
            try:
                # Use prettier for formatting if available
                with tempfile.NamedTemporaryFile(suffix='.md', mode='w+', encoding='utf-8') as tmp:
                    tmp.write(content)
                    tmp.flush()
                    
                    try:
                        result = subprocess.run(
                            ['prettier', '--parser', 'markdown', tmp.name],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Read the formatted content from stdout
                        formatted_content = result.stdout
                        return True, formatted_content
                    except subprocess.CalledProcessError as e:
                        return False, f"Markdown formatting failed: {e.stderr}"
                    except FileNotFoundError:
                        # prettier not installed, fall back to basic formatting
                        pass
            except Exception as e:
                logger.warning(f"Error using prettier for Markdown: {e}")
            
            # Basic Markdown formatting
            try:
                # Very simplistic Markdown formatter
                result = []
                in_code_block = False
                prev_line_empty = False
                
                for line in content.splitlines():
                    # Preserve code blocks as-is
                    if line.startswith('```'):
                        in_code_block = not in_code_block
                        result.append(line)
                        continue
                    
                    if in_code_block:
                        result.append(line)
                        continue
                    
                    # Handle regular markdown
                    stripped = line.strip()
                    
                    # Add empty lines between sections
                    if stripped.startswith('#') and result and not prev_line_empty:
                        result.append('')
                    
                    # Format lists with proper spacing
                    if stripped.startswith(('- ', '* ', '+ ', '1. ')):
                        if result and not result[-1].strip().startswith(('- ', '* ', '+ ', '1. ')) and not prev_line_empty:
                            result.append('')
                    
                    # Add the line
                    result.append(line)
                    
                    # Track empty lines
                    prev_line_empty = not stripped
                
                return True, '\n'.join(result)
            except Exception as e:
                return False, f"Error formatting Markdown code: {e}"
        
        elif transformation == CodeTransformation.ANALYZE:
            # Basic Markdown analysis
            lines = content.splitlines()
            
            issues = []
            metrics = {
                "lines_of_code": len(lines),
                "character_count": len(content),
            }
            suggestions = []
            complexity = {}
            
            # Count headings, lists, and code blocks
            heading_pattern = r'^(#{1,6})\s+'
            list_item_pattern = r'^(\s*)([-*+]|\d+\.)\s+'
            code_block_pattern = r'^```'
            
            headings = [line for line in lines if re.match(heading_pattern, line)]
            list_items = [line for line in lines if re.match(list_item_pattern, line)]
            code_blocks = [i for i, line in enumerate(lines) if re.match(code_block_pattern, line)]
            
            metrics["heading_count"] = len(headings)
            metrics["list_item_count"] = len(list_items)
            metrics["code_block_count"] = len(code_blocks) // 2  # Each block has start and end markers
            
            # Count links and images
            link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
            image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
            
            links = re.findall(link_pattern, content)
            images = re.findall(image_pattern, content)
            
            metrics["link_count"] = len(links)
            metrics["image_count"] = len(images)
            
            # Check for common issues
            for i, line in enumerate(lines):
                # Check for very long lines
                if len(line) > 100:
                    issues.append({
                        "type": "long_line",
                        "message": "Line exceeds 100 characters, consider breaking it for better readability",
                        "line": i + 1
                    })
                
                # Check for heading hierarchy
                if re.match(r'^#{1,6}\s+', line):
                    level = line.count('#')
                    if i > 0 and re.match(r'^#{1,6}\s+', lines[i-1]):
                        prev_level = lines[i-1].count('#')
                        if level > prev_level + 1:
                            issues.append({
                                "type": "heading_hierarchy",
                                "message": f"Heading level jumps from {prev_level} to {level}, consider using sequential levels",
                                "line": i + 1
                            })
            
            # Check for broken links
            for link_text, link_url in links:
                if link_url.startswith('http'):
                    # Skip external links for now
                    continue
                
                if link_url.startswith('#'):
                    # Check if the anchor exists
                    anchor = link_url[1:]
                    anchor_pattern = re.compile(r'^#{1,6}\s+.*\{#' + re.escape(anchor) + r'\}|^#{1,6}\s+' + re.escape(anchor) + r'\b', re.MULTILINE)
                    if not anchor_pattern.search(content):
                        issues.append({
                            "type": "broken_anchor",
                            "message": f"Anchor '{anchor}' referenced but not defined",
                            "line": None
                        })
            
            # Add suggestions
            if not any(line.startswith('# ') for line in lines):
                suggestions.append("Add a top-level heading (# Title) at the beginning of the document")
            
            if len(headings) > 0 and len(content.splitlines()) / len(headings) > 20:
                suggestions.append("Consider adding more headings to break up long sections of text")
            
            if len(links) == 0 and len(content) > 500:
                suggestions.append("Consider adding links to relevant resources or sections")
            
            # Basic complexity estimation
            complexity = {
                "estimation": "Basic complexity estimation",
                "heading_depth": max([line.count('#') for line in headings]) if headings else 0,
                "list_nesting": max([len(re.match(r'^\s*', line).group(0)) // 2 for line in list_items]) if list_items else 0
            }
            
            return True, {
                "issues": issues,
                "metrics": metrics,
                "suggestions": suggestions,
                "complexity": complexity
            }
        
        else:
            return False, f"Unsupported transformation for Markdown: {transformation.value}"

#######################################
# Network Operations
#######################################

class HttpMethod(Enum):
    """HTTP methods for network operations"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"

@dataclass
class HttpResponse:
    """HTTP response data structure"""
    status_code: int
    headers: Dict[str, str]
    content: bytes
    text: str
    json: Optional[Any] = None
    elapsed: float = 0.0
    url: str = ""
    
    @property
    def is_success(self) -> bool:
        """Check if the response was successful (status code 200-299)"""
        return 200 <= self.status_code < 300
    
    @property
    def is_redirect(self) -> bool:
        """Check if the response is a redirect (status code 300-399)"""
        return 300 <= self.status_code < 400
    
    @property
    def is_client_error(self) -> bool:
        """Check if the response is a client error (status code 400-499)"""
        return 400 <= self.status_code < 500
    
    @property
    def is_server_error(self) -> bool:
        """Check if the response is a server error (status code 500-599)"""
        return 500 <= self.status_code < 600

class NetworkOperations:
    """
    Advanced network operations for HTTP requests, API interactions,
    and network diagnostics
    """
    
    def __init__(self, timeout: int = 30, verify_ssl: bool = True):
        """Initialize network operations with default settings"""
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = None
        self.history = []  # Request history
        self.user_agent = "Python NetworkOperations/1.0"
    
    def create_session(self) -> None:
        """Create a persistent HTTP session"""
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent}
            )
    
    async def close_session(self) -> None:
        """Close the HTTP session"""
        if self.session is not None:
            await self.session.close()
            self.session = None
    
    async def request(self, method: HttpMethod, url: str, 
                     headers: Dict[str, str] = None, 
                     params: Dict[str, str] = None,
                     data: Any = None,
                     json_data: Any = None,
                     timeout: int = None,
                     verify_ssl: bool = None) -> HttpResponse:
        """
        Make an HTTP request
        
        Args:
            method: HTTP method to use
            url: URL to request
            headers: Optional headers to include
            params: Optional query parameters
            data: Optional form data or raw data
            json_data: Optional JSON data (will be serialized)
            timeout: Optional timeout override
            verify_ssl: Optional SSL verification override
            
        Returns:
            HttpResponse object with the response data
        """
        # Create session if needed
        self.create_session()
        
        # Use defaults if not specified
        timeout = timeout if timeout is not None else self.timeout
        verify_ssl = verify_ssl if verify_ssl is not None else self.verify_ssl
        
        # Prepare request parameters
        import aiohttp
        request_kwargs = {
            "headers": headers,
            "params": params,
            "ssl": verify_ssl,
            "timeout": aiohttp.ClientTimeout(total=timeout)
        }
        
        if data is not None:
            request_kwargs["data"] = data
        
        if json_data is not None:
            request_kwargs["json"] = json_data
        
        # Record start time
        start_time = time.time()
        
        try:
            # Make the request
            async with self.session.request(method.value, url, **request_kwargs) as response:
                # Read response data
                content = await response.read()
                text = content.decode('utf-8', errors='replace')
                
                # Try to parse JSON if the content type is JSON
                json_data = None
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type or 'application/ld+json' in content_type:
                    try:
                        json_data = json.loads(text)
                    except json.JSONDecodeError:
                        pass
                
                # Calculate elapsed time
                elapsed = time.time() - start_time
                
                # Create response object
                http_response = HttpResponse(
                    status_code=response.status,
                    headers=dict(response.headers),
                    content=content,
                    text=text,
                    json=json_data,
                    elapsed=elapsed,
                    url=str(response.url)
                )
                
                # Add to history
                self.history.append({
                    "method": method.value,
                    "url": url,
                    "status_code": response.status,
                    "elapsed": elapsed,
                    "timestamp": time.time()
                })
                
                return http_response
        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            raise
    
    async def get(self, url: str, **kwargs) -> HttpResponse:
        """Make a GET request"""
        return await self.request(HttpMethod.GET, url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> HttpResponse:
        """Make a POST request"""
        return await self.request(HttpMethod.POST, url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> HttpResponse:
        """Make a PUT request"""
        return await self.request(HttpMethod.PUT, url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> HttpResponse:
        """Make a DELETE request"""
        return await self.request(HttpMethod.DELETE, url, **kwargs)
    
    async def patch(self, url: str, **kwargs) -> HttpResponse:
        """Make a PATCH request"""
        return await self.request(HttpMethod.PATCH, url, **kwargs)
    
    async def head(self, url: str, **kwargs) -> HttpResponse:
        """Make a HEAD request"""
        return await self.request(HttpMethod.HEAD, url, **kwargs)
    
    async def options(self, url: str, **kwargs) -> HttpResponse:
        """Make an OPTIONS request"""
        return await self.request(HttpMethod.OPTIONS, url, **kwargs)
    
    async def download_file(self, url: str, filepath: str, 
                           chunk_size: int = 8192,
                           show_progress: bool = True) -> Dict[str, Any]:
        """
        Download a file from a URL
        
        Args:
            url: URL to download from
            filepath: Path to save the file to
            chunk_size: Size of chunks to download
            show_progress: Whether to show a progress bar
            
        Returns:
            Dictionary with download statistics
        """
        # Create session if needed
        self.create_session()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Make the request
            async with self.session.get(url, timeout=self.timeout) as response:
                if not response.ok:
                    return {
                        "success": False,
                        "message": f"Failed to download file: HTTP {response.status}",
                        "status_code": response.status,
                        "elapsed": time.time() - start_time
                    }
                
                # Get total size if available
                total_size = int(response.headers.get('Content-Length', 0))
                
                # Open file for writing
                with open(filepath, 'wb') as f:
                    downloaded = 0
                    
                    # Download in chunks
                    async for chunk in response.content.iter_chunked(chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress
                        if show_progress and total_size > 0:
                            percent = downloaded / total_size * 100
                            logger.info(f"Downloaded {downloaded}/{total_size} bytes ({percent:.1f}%)")
                
                # Calculate elapsed time
                elapsed = time.time() - start_time
                
                # Add to history
                self.history.append({
                    "method": "GET",
                    "url": url,
                    "status_code": response.status,
                    "elapsed": elapsed,
                    "timestamp": time.time(),
                    "downloaded_bytes": downloaded,
                    "filepath": filepath
                })
                
                return {
                    "success": True,
                    "message": f"Successfully downloaded {downloaded} bytes to {filepath}",
                    "filepath": filepath,
                    "size": downloaded,
                    "elapsed": elapsed,
                    "speed": downloaded / elapsed if elapsed > 0 else 0
                }
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return {
                "success": False,
                "message": f"Error downloading file: {str(e)}",
                "elapsed": time.time() - start_time
            }
    
    async def ping(self, host: str, count: int = 4, timeout: int = 5) -> Dict[str, Any]:
        """
        Ping a host to check connectivity
        
        Args:
            host: Hostname or IP address to ping
            count: Number of ping packets to send
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with ping statistics
        """
        try:
            # Use the ping command
            if sys.platform == "win32":
                # Windows
                cmd = ["ping", "-n", str(count), "-w", str(timeout * 1000), host]
            else:
                # Unix/Linux/MacOS
                cmd = ["ping", "-c", str(count), "-W", str(timeout), host]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse the output
            output = result.stdout
            
            # Extract statistics
            if sys.platform == "win32":
                # Windows output parsing
                packets_sent = count
                packets_received = 0
                min_time = max_time = avg_time = 0
                
                # Parse received packets
                if "Received = " in output:
                    received_line = re.search(r"Received = (\d+)", output)
                    if received_line:
                        packets_received = int(received_line.group(1))
                
                # Parse times
                if "Minimum = " in output:
                    times_line = re.search(r"Minimum = (\d+)ms, Maximum = (\d+)ms, Average = (\d+)ms", output)
                    if times_line:
                        min_time = int(times_line.group(1))
                        max_time = int(times_line.group(2))
                        avg_time = int(times_line.group(3))
            else:
                # Unix/Linux/MacOS output parsing
                packets_sent = count
                packets_received = 0
                min_time = max_time = avg_time = 0
                
                # Parse received packets
                if " packets transmitted, " in output:
                    stats_line = re.search(r"(\d+) packets transmitted, (\d+) (packets received|received)", output)
                    if stats_line:
                        packets_sent = int(stats_line.group(1))
                        packets_received = int(stats_line.group(2))
                
                # Parse times
                if "min/avg/max" in output:
                    times_line = re.search(r"min/avg/max[^=]+=\s*([\d.]+)/([\d.]+)/([\d.]+)", output)
                    if times_line:
                        min_time = float(times_line.group(1))
                        avg_time = float(times_line.group(2))
                        max_time = float(times_line.group(3))
            
            # Calculate packet loss
            packet_loss = 100 - (packets_received / packets_sent * 100) if packets_sent > 0 else 100
            
            return {
                "success": packets_received > 0,
                "host": host,
                "packets_sent": packets_sent,
                "packets_received": packets_received,
                "packet_loss": packet_loss,
                "min_time": min_time,
                "max_time": max_time,
                "avg_time": avg_time,
                "output": output
            }
        except Exception as e:
            logger.error(f"Error pinging host {host}: {e}")
            return {
                "success": False,
                "host": host,
                "message": f"Error pinging host: {str(e)}"
            }
    
    async def traceroute(self, host: str, max_hops: int = 30, timeout: int = 5) -> Dict[str, Any]:
        """
        Perform a traceroute to a host
        
        Args:
            host: Hostname or IP address to trace
            max_hops: Maximum number of hops
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with traceroute results
        """
        try:
            # Use the traceroute command
            if sys.platform == "win32":
                # Windows
                cmd = ["tracert", "-d", "-h", str(max_hops), "-w", str(timeout * 1000), host]
            else:
                # Unix/Linux/MacOS
                cmd = ["traceroute", "-n", "-m", str(max_hops), "-w", str(timeout), host]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse the output
            output = result.stdout
            
            # Extract hops
            hops = []
            
            if sys.platform == "win32":
                # Windows output parsing
                hop_pattern = r"^\s*(\d+)\s+(\d+)\s+ms\s+(\d+)\s+ms\s+(\d+)\s+ms\s+(.+)$"
                for line in output.splitlines():
                    match = re.search(hop_pattern, line)
                    if match:
                        hop_num = int(match.group(1))
                        times = [int(match.group(2)), int(match.group(3)), int(match.group(4))]
                        ip = match.group(5).strip()
                        
                        hops.append({
                            "hop": hop_num,
                            "ip": ip,
                            "times": times,
                            "avg_time": sum(times) / len(times)
                        })
            else:
                # Unix/Linux/MacOS output parsing
                hop_pattern = r"^\s*(\d+)\s+([^\s]+)(?:\s+([0-9.]+)\s+ms)?(?:\s+([0-9.]+)\s+ms)?(?:\s+([0-9.]+)\s+ms)?.*$"
                for line in output.splitlines():
                    match = re.search(hop_pattern, line)
                    if match:
                        hop_num = int(match.group(1))
                        ip = match.group(2)
                        
                        # Extract times
                        times = []
                        for i in range(3, 6):
                            if match.group(i):
                                times.append(float(match.group(i)))
                        
                        if times:
                            hops.append({
                                "hop": hop_num,
                                "ip": ip,
                                "times": times,
                                "avg_time": sum(times) / len(times)
                            })
            
            return {
                "success": len(hops) > 0,
                "host": host,
                "hops": hops,
                "hop_count": len(hops),
                "output": output
            }
        except Exception as e:
            logger.error(f"Error performing traceroute to {host}: {e}")
            return {
                "success": False,
                "host": host,
                "message": f"Error performing traceroute: {str(e)}"
            }
    
    async def check_port(self, host: str, port: int, timeout: int = 5) -> Dict[str, Any]:
        """
        Check if a port is open on a host
        
        Args:
            host: Hostname or IP address to check
            port: Port number to check
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with port check results
        """
        try:
            # Create a socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            
            # Record start time
            start_time = time.time()
            
            # Try to connect
            result = sock.connect_ex((host, port))
            
            # Calculate elapsed time
            elapsed = time.time() - start_time
            
            # Close the socket
            sock.close()
            
            # Check the result
            is_open = result == 0
            
            return {
                "success": True,
                "host": host,
                "port": port,
                "is_open": is_open,
                "elapsed": elapsed,
                "message": f"Port {port} is {'open' if is_open else 'closed'} on {host}"
            }
        except Exception as e:
            logger.error(f"Error checking port {port} on {host}: {e}")
            return {
                "success": False,
                "host": host,
                "port": port,
                "is_open": False,
                "message": f"Error checking port: {str(e)}"
            }
    
    async def dns_lookup(self, hostname: str) -> Dict[str, Any]:
        """
        Perform a DNS lookup for a hostname
        
        Args:
            hostname: Hostname to look up
            
        Returns:
            Dictionary with DNS lookup results
        """
        try:
            # Get IP addresses
            addresses = socket.getaddrinfo(hostname, None)
            
            # Extract unique IP addresses
            ips = set()
            for addr in addresses:
                family, type, proto, canonname, sockaddr = addr
                if family == socket.AF_INET:  # IPv4
                    ips.add(sockaddr[0])
                elif family == socket.AF_INET6:  # IPv6
                    ips.add(sockaddr[0])
            
            return {
                "success": True,
                "hostname": hostname,
                "ip_addresses": list(ips),
                "address_count": len(ips)
            }
        except Exception as e:
            logger.error(f"Error performing DNS lookup for {hostname}: {e}")
            return {
                "success": False,
                "hostname": hostname,
                "message": f"Error performing DNS lookup: {str(e)}"
            }
    
    def get_request_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the request history
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of request history items
        """
        # Sort by timestamp (newest first) and limit
        sorted_history = sorted(self.history, key=lambda x: x["timestamp"], reverse=True)[:limit]
        return sorted_history

#######################################
# Enhanced File System Operations
#######################################

class FileSystemOperations:
    """
    Enhanced file system operations with advanced capabilities
    for file manipulation, searching, and monitoring
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize file system operations with an optional base directory"""
        self.base_dir = base_dir or os.getcwd()
        self.history = []  # Operation history
        self.watchers = {}  # File watchers
    
    def resolve_path(self, path: str) -> str:
        """Resolve a path relative to the base directory"""
        if os.path.isabs(path):
            return os.path.normpath(path)
        else:
            return os.path.normpath(os.path.join(self.base_dir, path))
    
    def list_files(self, directory: str = ".", pattern: str = "*", 
                  recursive: bool = False, include_hidden: bool = False,
                  include_dirs: bool = True, include_files: bool = True,
                  sort_by: str = "name") -> Dict[str, Any]:
        """
        List files in a directory with advanced filtering and sorting
        
        Args:
            directory: Directory to list files from
            pattern: Glob pattern to filter files
            recursive: Whether to recursively list files in subdirectories
            include_hidden: Whether to include hidden files
            include_dirs: Whether to include directories in the results
            include_files: Whether to include files in the results
            sort_by: How to sort the results (name, size, modified, type)
            
        Returns:
            Dictionary with file listing results
        """
        try:
            # Resolve the directory path
            dir_path = self.resolve_path(directory)
            
            # Check if the directory exists
            if not os.path.isdir(dir_path):
                return {
                    "success": False,
                    "message": f"Directory not found: {dir_path}",
                    "data": None
                }
            
            # Get files matching the pattern
            if recursive:
                matches = []
                for root, dirnames, filenames in os.walk(dir_path):
                    # Filter directories if needed
                    if include_dirs:
                        for dirname in dirnames:
                            if include_hidden or not dirname.startswith('.'):
                                if fnmatch.fnmatch(dirname, pattern):
                                    matches.append(os.path.join(root, dirname))
                    
                    # Filter files if needed
                    if include_files:
                        for filename in filenames:
                            if include_hidden or not filename.startswith('.'):
                                if fnmatch.fnmatch(filename, pattern):
                                    matches.append(os.path.join(root, filename))
                
                files = matches
            else:
                matches = glob.glob(os.path.join(dir_path, pattern))
                
                # Filter hidden files if needed
                if not include_hidden:
                    matches = [f for f in matches if not os.path.basename(f).startswith('.')]
                
                # Filter directories and files if needed
                files = []
                for match in matches:
                    if os.path.isdir(match) and include_dirs:
                        files.append(match)
                    elif os.path.isfile(match) and include_files:
                        files.append(match)
            
            # Get file info
            file_info = []
            for file_path in files:
                try:
                    stat = os.stat(file_path)
                    is_dir = os.path.isdir(file_path)
                    
                    # Get file type
                    file_type = "directory" if is_dir else "file"
                    
                    # Get file extension
                    ext = os.path.splitext(file_path)[1].lower() if not is_dir else ""
                    
                    # Calculate human-readable size
                    size_bytes = stat.st_size
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if size_bytes < 1024 or unit == 'TB':
                            human_size = f"{size_bytes:.2f} {unit}"
                            break
                        size_bytes /= 1024
                    
                    # Get relative path
                    rel_path = os.path.relpath(file_path, dir_path)
                    
                    file_info.append({
                        "name": os.path.basename(file_path),
                        "path": file_path,
                        "relative_path": rel_path,
                        "size": stat.st_size,
                        "human_size": human_size,
                        "modified": time.ctime(stat.st_mtime),
                        "modified_timestamp": stat.st_mtime,
                        "created": time.ctime(stat.st_ctime),
                        "created_timestamp": stat.st_ctime,
                        "accessed": time.ctime(stat.st_atime),
                        "is_dir": is_dir,
                        "type": file_type,
                        "extension": ext,
                        "permissions": oct(stat.st_mode)[-3:],
                        "owner": stat.st_uid,
                        "group": stat.st_gid
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for {file_path}: {e}")
            
            # Sort the results
            if sort_by == "name":
                file_info.sort(key=lambda x: x["name"].lower())
            elif sort_by == "size":
                file_info.sort(key=lambda x: x["size"], reverse=True)
            elif sort_by == "modified":
                file_info.sort(key=lambda x: x["modified_timestamp"], reverse=True)
            elif sort_by == "type":
                file_info.sort(key=lambda x: (x["is_dir"], x["extension"], x["name"].lower()), reverse=True)
            
            # Add summary statistics
            total_size = sum(item["size"] for item in file_info)
            dir_count = sum(1 for item in file_info if item["is_dir"])
            file_count = len(file_info) - dir_count
            
            # Add to history
            self.history.append({
                "operation": "list_files",
                "directory": dir_path,
                "pattern": pattern,
                "recursive": recursive,
                "timestamp": time.time(),
                "file_count": file_count,
                "dir_count": dir_count
            })
            
            return {
                "success": True,
                "message": f"Found {len(file_info)} items matching pattern '{pattern}' in '{dir_path}'",
                "data": {
                    "items": file_info,
                    "summary": {
                        "total_items": len(file_info),
                        "directories": dir_count,
                        "files": file_count,
                        "total_size": total_size,
                        "path": dir_path,
                        "pattern": pattern
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return {
                "success": False,
                "message": f"Error listing files: {str(e)}",
                "data": None
            }
    
    def search_files(self, directory: str = ".", pattern: str = "*", 
                    content_pattern: Optional[str] = None, 
                    max_size: int = 10 * 1024 * 1024,  # 10 MB
                    max_results: int = 100,
                    case_sensitive: bool = False,
                    recursive: bool = True) -> Dict[str, Any]:
        """
        Search for files by name and/or content
        
        Args:
            directory: Directory to search in
            pattern: Glob pattern to filter files by name
            content_pattern: Optional regex pattern to search in file contents
            max_size: Maximum file size to search in
            max_results: Maximum number of results to return
            case_sensitive: Whether the content search is case-sensitive
            recursive: Whether to recursively search in subdirectories
            
        Returns:
            Dictionary with search results
        """
        try:
            # Resolve the directory path
            dir_path = self.resolve_path(directory)
            
            # Check if the directory exists
            if not os.path.isdir(dir_path):
                return {
                    "success": False,
                    "message": f"Directory not found: {dir_path}",
                    "data": None
                }
            
            # Compile the content pattern if provided
            content_regex = None
            if content_pattern:
                flags = 0 if case_sensitive else re.IGNORECASE
                try:
                    content_regex = re.compile(content_pattern, flags)
                except re.error as e:
                    return {
                        "success": False,
                        "message": f"Invalid regex pattern: {e}",
                        "data": None
                    }
            
            # Get files matching the name pattern
            matches = []
            if recursive:
                for root, _, filenames in os.walk(dir_path):
                    for filename in filenames:
                        if fnmatch.fnmatch(filename, pattern):
                            matches.append(os.path.join(root, filename))
            else:
                matches = glob.glob(os.path.join(dir_path, pattern))
                matches = [m for m in matches if os.path.isfile(m)]
            
            # Search for content if needed
            results = []
            for file_path in matches:
                try:
                    # Check file size
                    size = os.path.getsize(file_path)
                    if size > max_size:
                        continue
                    
                    # If no content pattern, just add the file
                    if not content_regex:
                        stat = os.stat(file_path)
                        results.append({
                            "path": file_path,
                            "relative_path": os.path.relpath(file_path, dir_path),
                            "size": size,
                            "modified": time.ctime(stat.st_mtime),
                            "matches": []
                        })
                        
                        # Check if we've reached the maximum results
                        if len(results) >= max_results:
                            break
                    else:
                        # Search for the content pattern
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                content = f.read()
                                
                                # Find all matches
                                content_matches = []
                                for match in content_regex.finditer(content):
                                    # Get the line number
                                    line_num = content.count('\n', 0, match.start()) + 1
                                    
                                    # Get the line content
                                    line_start = content.rfind('\n', 0, match.start()) + 1
                                    line_end = content.find('\n', match.start())
                                    if line_end == -1:
                                        line_end = len(content)
                                    
                                    line_content = content[line_start:line_end]
                                    
                                    content_matches.append({
                                        "line": line_num,
                                        "start": match.start(),
                                        "end": match.end(),
                                        "text": match.group(0),
                                        "line_content": line_content
                                    })
                                
                                # If there are matches, add the file to the results
                                if content_matches:
                                    stat = os.stat(file_path)
                                    results.append({
                                        "path": file_path,
                                        "relative_path": os.path.relpath(file_path, dir_path),
                                        "size": size,
                                        "modified": time.ctime(stat.st_mtime),
                                        "matches": content_matches
                                    })
                                    
                                    # Check if we've reached the maximum results
                                    if len(results) >= max_results:
                                        break
                        except UnicodeDecodeError:
                            # Skip binary files
                            pass
                except Exception as e:
                    logger.warning(f"Error searching in {file_path}: {e}")
            
            # Add to history
            self.history.append({
                "operation": "search_files",
                "directory": dir_path,
                "pattern": pattern,
                "content_pattern": content_pattern,
                "timestamp": time.time(),
                "result_count": len(results)
            })
            
            return {
                "success": True,
                "message": f"Found {len(results)} files matching the search criteria",
                "data": {
                    "results": results,
                    "count": len(results),
                    "has_more": len(matches) > max_results,
                    "total_matches": sum(len(r["matches"]) for r in results)
                }
            }
        except Exception as e:
            logger.error(f"Error searching files: {e}")
            return {
                "success": False,
                "message": f"Error searching files: {str(e)}",
                "data": None
            }
    
    def read_file(self, filepath: str, encoding: str = 'utf-8', 
                 max_size: int = 10 * 1024 * 1024,  # 10 MB
                 chunk_size: Optional[int] = None,
                 line_numbers: bool = False) -> Dict[str, Any]:
        """
        Read a file with advanced options
        
        Args:
            filepath: Path to the file to read
            encoding: File encoding to use
            max_size: Maximum file size to read
            chunk_size: Optional size of chunk to read
            line_numbers: Whether to include line numbers
            
        Returns:
            Dictionary with file contents and metadata
        """
        try:
            # Resolve the file path
            file_path = self.resolve_path(filepath)
            
            # Check if the file exists
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "message": f"File not found: {file_path}",
                    "data": None
                }
            
            # Check if it's a directory
            if os.path.isdir(file_path):
                return {
                    "success": False,
                    "message": f"Cannot read directory as file: {file_path}",
                    "data": None
                }
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > max_size:
                return {
                    "success": False,
                    "message": f"File too large ({file_size} bytes). Max size is {max_size} bytes.",
                    "data": None
                }
            
            # Detect if the file is binary
            try:
                is_binary = False
                with open(file_path, 'rb') as f:
                    chunk = f.read(1024)
                    if b'\0' in chunk:  # Simple binary detection
                        is_binary = True
                
                if is_binary:
                    # For binary files, return hex dump
                    with open(file_path, 'rb') as f:
                        binary_data = f.read(chunk_size or max_size)
                    
                    hex_dump = ' '.join(f'{b:02x}' for b in binary_data[:100])  # First 100 bytes
                    
                    # Add to history
                    self.history.append({
                        "operation": "read_file",
                        "filepath": file_path,
                        "timestamp": time.time(),
                        "size": len(binary_data),
                        "is_binary": True
                    })
                    
                    return {
                        "success": True,
                        "message": f"Successfully read binary file: {file_path} ({file_size} bytes)",
                        "data": {
                            "content": f"Binary file: first 100 bytes: {hex_dump}...",
                            "is_binary": True,
                            "size": file_size,
                            "path": file_path,
                            "binary_preview": hex_dump
                        }
                    }
            except Exception as e:
                logger.warning(f"Error detecting binary file: {e}")
            
            # Read the file content
            content = ""
            if chunk_size:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read(chunk_size)
                    truncated = file_size > chunk_size
            else:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                    truncated = False
            
            # Process content based on options
            if line_numbers:
                lines = content.splitlines()
                content_with_lines = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
                content = content_with_lines
            
            # Get file metadata
            stat = os.stat(file_path)
            
            # Detect file type based on extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lstrip('.').lower()
            
            # Map extension to file type
            ext_to_type = {
                'py': 'Python',
                'js': 'JavaScript',
                'html': 'HTML',
                'css': 'CSS',
                'json': 'JSON',
                'xml': 'XML',
                'md': 'Markdown',
                'txt': 'Text',
                'csv': 'CSV',
                'pdf': 'PDF',
                'jpg': 'JPEG Image',
                'jpeg': 'JPEG Image',
                'png': 'PNG Image',
                'gif': 'GIF Image',
                'svg': 'SVG Image',
                'mp3': 'MP3 Audio',
                'mp4': 'MP4 Video',
                'zip': 'ZIP Archive',
                'tar': 'TAR Archive',
                'gz': 'GZIP Archive',
                'doc': 'Word Document',
                'docx': 'Word Document',
                'xls': 'Excel Spreadsheet',
                'xlsx': 'Excel Spreadsheet',
                'ppt': 'PowerPoint Presentation',
                'pptx': 'PowerPoint Presentation',
            }
            
            file_type = ext_to_type.get(ext, 'Unknown')
            
            # Add to history
            self.history.append({
                "operation": "read_file",
                "filepath": file_path,
                "timestamp": time.time(),
                "size": len(content),
                "is_binary": False
            })
            
            return {
                "success": True,
                "message": f"Successfully read file: {file_path} ({len(content)} bytes)",
                "data": {
                    "content": content,
                    "size": file_size,
                    "path": file_path,
                    "encoding": encoding,
                    "truncated": truncated,
                    "line_count": content.count('\n') + 1,
                    "modified": time.ctime(stat.st_mtime),
                    "created": time.ctime(stat.st_ctime),
                    "accessed": time.ctime(stat.st_atime),
                    "file_type": file_type,
                    "extension": ext,
                    "is_binary": False
                }
            }
        except UnicodeDecodeError:
            # If we hit a decode error, try to read as binary
            try:
                with open(file_path, 'rb') as f:
                    binary_data = f.read(100)  # Just read a small preview
                
                hex_dump = ' '.join(f'{b:02x}' for b in binary_data)
                
                # Add to history
                self.history.append({
                    "operation": "read_file",
                    "filepath": file_path,
                    "timestamp": time.time(),
                    "size": file_size,
                    "is_binary": True
                })
                
                return {
                    "success": True,
                    "message": f"File appears to be binary: {file_path}",
                    "data": {
                        "content": f"Binary file: first 100 bytes: {hex_dump}...",
                        "is_binary": True,
                        "size": file_size,
                        "path": file_path,
                        "binary_preview": hex_dump
                    }
                }
            except Exception as e:
                logger.error(f"Error reading binary file: {e}")
                return {
                    "success": False,
                    "message": f"Error reading file: {str(e)}",
                    "data": None
                }
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return {
                "success": False,
                "message": f"Error reading file: {str(e)}",
                "data": None
            }
    
    def write_file(self, filepath: str, content: Union[str, bytes], 
                  mode: str = 'w', encoding: str = 'utf-8',
                  create_backup: bool = False,
                  make_dirs: bool = True) -> Dict[str, Any]:
        """
        Write content to a file with advanced options
        
        Args:
            filepath: Path to the file to write
            content: Content to write to the file
            mode: File mode ('w' for write, 'a' for append, 'wb' for binary write)
            encoding: File encoding to use (for text modes)
            create_backup: Whether to create a backup of the existing file
            make_dirs: Whether to create parent directories if they don't exist
            
        Returns:
            Dictionary with write operation results
        """
        try:
            # Resolve the file path
            file_path = self.resolve_path(filepath)
            
            # Check if the file exists
            file_exists = os.path.exists(file_path)
            
            # Create a backup if requested
            if file_exists and create_backup:
                backup_path = f"{file_path}.bak"
                shutil.copy2(file_path, backup_path)
                logger.info(f"Created backup of {file_path} at {backup_path}")
            
            # Create parent directories if needed
            if make_dirs:
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Determine if we're writing binary data
            is_binary = isinstance(content, bytes) or 'b' in mode
            
            # Write the file
            if is_binary:
                # Ensure we have binary content
                if isinstance(content, str):
                    content = content.encode(encoding)
                
                # Use binary mode
                with open(file_path, mode.replace('w', 'wb').replace('a', 'ab'), encoding=None) as f:
                    f.write(content)
            else:
                # Use text mode
                with open(file_path, mode, encoding=encoding) as f:
                    f.write(content)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            # Add to history
            self.history.append({
                "operation": "write_file",
                "filepath": file_path,
                "timestamp": time.time(),
                "size": file_size,
                "mode": mode,
                "backup": backup_path if file_exists and create_backup else None
            })
            
            return {
                "success": True,
                "message": f"Successfully wrote {file_size} bytes to {file_path}",
                "data": {
                    "path": file_path,
                    "size": file_size,
                    "mode": mode,
                    "is_binary": is_binary,
                    "backup": backup_path if file_exists and create_backup else None
                }
            }
        except Exception as e:
            logger.error(f"Error writing file: {e}")
            return {
                "success": False,
                "message": f"Error writing file: {str(e)}",
                "data": None
            }
    
    def copy(self, source: str, destination: str, 
            overwrite: bool = False,
            create_backup: bool = False,
            preserve_metadata: bool = True) -> Dict[str, Any]:
        """
        Copy a file or directory
        
        Args:
            source: Source path
            destination: Destination path
            overwrite: Whether to overwrite existing files
            create_backup: Whether to create a backup of existing files
            preserve_metadata: Whether to preserve file metadata
            
        Returns:
            Dictionary with copy operation results
        """
        try:
            # Resolve the paths
            source_path = self.resolve_path(source)
            dest_path = self.resolve_path(destination)
            
            # Check if the source exists
            if not os.path.exists(source_path):
                return {
                    "success": False,
                    "message": f"Source not found: {source_path}",
                    "data": None
                }
            
            # Check if the destination exists and overwrite is False
            if os.path.exists(dest_path) and not overwrite:
                return {
                    "success": False,
                    "message": f"Destination already exists: {dest_path}. Set overwrite=true to overwrite.",
                    "data": None
                }
            
            # Create a backup if requested
            backup_path = None
            if os.path.exists(dest_path) and create_backup:
                backup_path = f"{dest_path}.bak"
                if os.path.isdir(dest_path):
                    shutil.copytree(dest_path, backup_path)
                else:
                    shutil.copy2(dest_path, backup_path)
                logger.info(f"Created backup of {dest_path} at {backup_path}")
            
            # Copy the file or directory
            if os.path.isdir(source_path):
                # Copy directory
                if os.path.exists(dest_path) and overwrite:
                    shutil.rmtree(dest_path)
                
                if preserve_metadata:
                    shutil.copytree(source_path, dest_path, symlinks=True)
                else:
                    shutil.copytree(source_path, dest_path, symlinks=True, copy_function=shutil.copy)
                
                # Count files and directories
                file_count = 0
                dir_count = 0
                total_size = 0
                
                for root, dirs, files in os.walk(dest_path):
                    dir_count += len(dirs)
                    file_count += len(files)
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                
                # Add to history
                self.history.append({
                    "operation": "copy_directory",
                    "source": source_path,
                    "destination": dest_path,
                    "timestamp": time.time(),
                    "file_count": file_count,
                    "dir_count": dir_count,
                    "total_size": total_size,
                    "backup": backup_path
                })
                
                return {
                    "success": True,
                    "message": f"Successfully copied directory from {source_path} to {dest_path}",
                    "data": {
                        "source": source_path,
                        "destination": dest_path,
                        "is_directory": True,
                        "file_count": file_count,
                        "dir_count": dir_count,
                        "total_size": total_size,
                        "backup": backup_path
                    }
                }
            else:
                # Copy file
                if preserve_metadata:
                    shutil.copy2(source_path, dest_path)
                else:
                    shutil.copy(source_path, dest_path)
                
                # Get file size
                file_size = os.path.getsize(dest_path)
                
                # Add to history
                self.history.append({
                    "operation": "copy_file",
                    "source": source_path,
                    "destination": dest_path,
                    "timestamp": time.time(),
                    "size": file_size,
                    "backup": backup_path
                })
                
                return {
                    "success": True,
                    "message": f"Successfully copied file from {source_path} to {dest_path}",
                    "data": {
                        "source": source_path,
                        "destination": dest_path,
                        "is_directory": False,
                        "size": file_size,
                        "backup": backup_path
                    }
                }
        except Exception as e:
            logger.error(f"Error copying: {e}")
            return {
                "success": False,
                "message": f"Error copying: {str(e)}",
                "data": None
            }
    
    def move(self, source: str, destination: str, 
            overwrite: bool = False,
            create_backup: bool = False) -> Dict[str, Any]:
        """
        Move a file or directory
        
        Args:
            source: Source path
            destination: Destination path
            overwrite: Whether to overwrite existing files
            create_backup: Whether to create a backup of existing files
            
        Returns:
            Dictionary with move operation results
        """
        try:
            # Resolve the paths
            source_path = self.resolve_path(source)
            dest_path = self.resolve_path(destination)
            
            # Check if the source exists
            if not os.path.exists(source_path):
                return {
                    "success": False,
                    "message": f"Source not found: {source_path}",
                    "data": None
                }
            
            # Check if the destination exists and overwrite is False
            if os.path.exists(dest_path) and not overwrite:
                return {
                    "success": False,
                    "message": f"Destination already exists: {dest_path}. Set overwrite=true to overwrite.",
                    "data": None
                }
            
            # Create a backup if requested
            backup_path = None
            if os.path.exists(dest_path) and create_backup:
                backup_path = f"{dest_path}.bak"
                if os.path.isdir(dest_path):
                    shutil.copytree(dest_path, backup_path)
                else:
                    shutil.copy2(dest_path, backup_path)
                logger.info(f"Created backup of {dest_path} at {backup_path}")
            
            # Get source info before moving
            is_directory = os.path.isdir(source_path)
            
            if is_directory:
                # Count files and directories
                file_count = 0
                dir_count = 0
                total_size = 0
                
                for root, dirs, files in os.walk(source_path):
                    dir_count += len(dirs)
                    file_count += len(files)
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
            else:
                # Get file size
                file_size = os.path.getsize(source_path)
            
            # Move the file or directory
            if os.path.exists(dest_path) and overwrite:
                if os.path.isdir(dest_path):
                    shutil.rmtree(dest_path)
                else:
                    os.remove(dest_path)
            
            # Create parent directories if needed
            os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
            
            # Move the file or directory
            shutil.move(source_path, dest_path)
            
            # Add to history
            if is_directory:
                self.history.append({
                    "operation": "move_directory",
                    "source": source_path,
                    "destination": dest_path,
                    "timestamp": time.time(),
                    "file_count": file_count,
                    "dir_count": dir_count,
                    "total_size": total_size,
                    "backup": backup_path
                })
                
                return {
                    "success": True,
                    "message": f"Successfully moved directory from {source_path} to {dest_path}",
                    "data": {
                        "source": source_path,
                        "destination": dest_path,
                        "is_directory": True,
                        "file_count": file_count,
                        "dir_count": dir_count,
                        "total_size": total_size,
                        "backup": backup_path
                    }
                }
            else:
                self.history.append({
                    "operation": "move_file",
                    "source": source_path,
                    "destination": dest_path,
                    "timestamp": time.time(),
                    "size": file_size,
                    "backup": backup_path
                })
                
                return {
                    "success": True,
                    "message": f"Successfully moved file from {source_path} to {dest_path}",
                    "data": {
                        "source": source_path,
                        "destination": dest_path,
                        "is_directory": False,
                        "size": file_size,
                        "backup": backup_path
                    }
                }
        except Exception as e:
            logger.error(f"Error moving: {e}")
            return {
                "success": False,
                "message": f"Error moving: {str(e)}",
                "data": None
            }
    
    def delete(self, path: str, recursive: bool = False) -> Dict[str, Any]:
        """
        Delete a file or directory
        
        Args:
            path: Path to delete
            recursive: Whether to recursively delete directories
            
        Returns:
            Dictionary with delete operation results
        """
        try:
            # Resolve the path
            resolved_path = self.resolve_path(path)
            
            # Check if the path exists
            if not os.path.exists(resolved_path):
                return {
                    "success": False,
                    "message": f"Path not found: {resolved_path}",
                    "data": None
                }
            
            # Get path info before deleting
            is_directory = os.path.isdir(resolved_path)
            
            if is_directory:
                # Check if recursive is required
                if not recursive and len(os.listdir(resolved_path)) > 0:
                    return {
                        "success": False,
                        "message": f"Directory not empty: {resolved_path}. Set recursive=true to delete recursively.",
                        "data": None
                    }
                
                # Count files and directories
                file_count = 0
                dir_count = 0
                total_size = 0
                
                for root, dirs, files in os.walk(resolved_path):
                    dir_count += len(dirs)
                    file_count += len(files)
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                
                # Delete the directory
                if recursive:
                    shutil.rmtree(resolved_path)
                else:
                    os.rmdir(resolved_path)
                
                # Add to history
                self.history.append({
                    "operation": "delete_directory",
                    "path": resolved_path,
                    "timestamp": time.time(),
                    "recursive": recursive,
                    "file_count": file_count,
                    "dir_count": dir_count,
                    "total_size": total_size
                })
                
                return {
                    "success": True,
                    "message": f"Successfully deleted directory: {resolved_path}",
                    "data": {
                        "path": resolved_path,
                        "is_directory": True,
                        "file_count": file_count,
                        "dir_count": dir_count,
                        "total_size": total_size
                    }
                }
            else:
                # Get file size
                file_size = os.path.getsize(resolved_path)
                
                # Delete the file
                os.remove(resolved_path)
                
                # Add to history
                self.history.append({
                    "operation": "delete_file",
                    "path": resolved_path,
                    "timestamp": time.time(),
                    "size": file_size
                })
                
                return {
                    "success": True,
                    "message": f"Successfully deleted file: {resolved_path}",
                    "data": {
                        "path": resolved_path,
                        "is_directory": False,
                        "size": file_size
                    }
                }
        except Exception as e:
            logger.error(f"Error deleting: {e}")
            return {
                "success": False,
                "message": f"Error deleting: {str(e)}",
                "data": None
            }
    
    def create_directory(self, path: str, mode: int = 0o755, 
                        exist_ok: bool = True) -> Dict[str, Any]:
        """
        Create a directory
        
        Args:
            path: Path to create
            mode: Directory permissions mode
            exist_ok: Whether it's okay if the directory already exists
            
        Returns:
            Dictionary with create directory operation results
        """
        try:
            # Resolve the path
            resolved_path = self.resolve_path(path)
            
            # Check if the path already exists
            if os.path.exists(resolved_path):
                if not exist_ok:
                    return {
                        "success": False,
                        "message": f"Path already exists: {resolved_path}",
                        "data": None
                    }
                elif not os.path.isdir(resolved_path):
                    return {
                        "success": False,
                        "message": f"Path exists but is not a directory: {resolved_path}",
                        "data": None
                    }
                else:
                    # Directory already exists and exist_ok is True
                    return {
                        "success": True,
                        "message": f"Directory already exists: {resolved_path}",
                        "data": {
                            "path": resolved_path,
                            "already_existed": True
                        }
                    }
            
            # Create the directory
            os.makedirs(resolved_path, mode=mode, exist_ok=exist_ok)
            
            # Add to history
            self.history.append({
                "operation": "create_directory",
                "path": resolved_path,
                "timestamp": time.time(),
                "mode": oct(mode)
            })
            
            return {
                "success": True,
                "message": f"Successfully created directory: {resolved_path}",
                "data": {
                    "path": resolved_path,
                    "mode": oct(mode),
                    "already_existed": False
                }
            }
        except Exception as e:
            logger.error(f"Error creating directory: {e}")
            return {
                "success": False,
                "message": f"Error creating directory: {str(e)}",
                "data": None
            }
    
    def get_file_info(self, path: str) -> Dict[str, Any]:
        """
        Get detailed information about a file or directory
        
        Args:
            path: Path to get information about
            
        Returns:
            Dictionary with file information
        """
        try:
            # Resolve the path
            resolved_path = self.resolve_path(path)
            
            # Check if the path exists
            if not os.path.exists(resolved_path):
                return {
                    "success": False,
                    "message": f"Path not found: {resolved_path}",
                    "data": None
                }
            
            # Get basic file info
            stat = os.stat(resolved_path)
            is_dir = os.path.isdir(resolved_path)
            
            # Get file type
            file_type = "directory" if is_dir else "file"
            
            # Get file extension
            ext = os.path.splitext(resolved_path)[1].lower() if not is_dir else ""
            
            # Calculate human-readable size
            if is_dir:
                # Calculate directory size
                total_size = 0
                for root, dirs, files in os.walk(resolved_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        total_size += os.path.getsize(file_path)
                
                size_bytes = total_size
            else:
                size_bytes = stat.st_size
            
            # Format human-readable size
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size_bytes < 1024 or unit == 'TB':
                    human_size = f"{size_bytes:.2f} {unit}"
                    break
                size_bytes /= 1024
            
            # Get file owner and group names
            try:
                import pwd
                import grp
                owner_name = pwd.getpwuid(stat.st_uid).pw_name
                group_name = grp.getgrgid(stat.st_gid).gr_name
            except (ImportError, KeyError):
                owner_name = str(stat.st_uid)
                group_name = str(stat.st_gid)
            
            # Get additional file info
            info = {
                "name": os.path.basename(resolved_path),
                "path": resolved_path,
                "size": stat.st_size if not is_dir else total_size,
                "human_size": human_size,
                "modified": time.ctime(stat.st_mtime),
                "modified_timestamp": stat.st_mtime,
                "created": time.ctime(stat.st_ctime),
                "created_timestamp": stat.st_ctime,
                "accessed": time.ctime(stat.st_atime),
                "accessed_timestamp": stat.st_atime,
                "is_dir": is_dir,
                "type": file_type,
                "extension": ext,
                "permissions": oct(stat.st_mode)[-3:],
                "owner": stat.st_uid,
                "owner_name": owner_name,
                "group": stat.st_gid,
                "group_name": group_name,
                "is_symlink": os.path.islink(resolved_path)
            }
            
            # Add symlink target if applicable
            if info["is_symlink"]:
                info["symlink_target"] = os.readlink(resolved_path)
            
            # Add directory-specific info
            if is_dir:
                # Count files and subdirectories
                try:
                    entries = os.listdir(resolved_path)
                    files = [e for e in entries if os.path.isfile(os.path.join(resolved_path, e))]
                    dirs = [e for e in entries if os.path.isdir(os.path.join(resolved_path, e))]
                    
                    info["file_count"] = len(files)
                    info["dir_count"] = len(dirs)
                    info["total_entries"] = len(entries)
                except Exception as e:
                    logger.warning(f"Error counting directory entries: {e}")
            
            # Add file-specific info
            else:
                # Try to detect MIME type
                try:
                    import mimetypes
                    mime_type, encoding = mimetypes.guess_type(resolved_path)
                    info["mime_type"] = mime_type or "application/octet-stream"
                    if encoding:
                        info["encoding"] = encoding
                except Exception as e:
                    logger.warning(f"Error detecting MIME type: {e}")
            
            # Add to history
            self.history.append({
                "operation": "get_file_info",
                "path": resolved_path,
                "timestamp": time.time()
            })
            
            return {
                "success": True,
                "message": f"Successfully retrieved information for: {resolved_path}",
                "data": info
            }
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {
                "success": False,
                "message": f"Error getting file info: {str(e)}",
                "data": None
            }
    
    def start_file_watcher(self, path: str, 
                          callback: Optional[Callable[[str, str], None]] = None) -> Dict[str, Any]:
        """
        Start watching a file or directory for changes
        
        Args:
            path: Path to watch
            callback: Optional callback function to call when changes occur
            
        Returns:
            Dictionary with file watcher results
        """
        try:
            # Resolve the path
            resolved_path = self.resolve_path(path)
            
            # Check if the path exists
            if not os.path.exists(resolved_path):
                return {
                    "success": False,
                    "message": f"Path not found: {resolved_path}",
                    "data": None
                }
            
            # Check if we're already watching this path
            if resolved_path in self.watchers:
                return {
                    "success": False,
                    "message": f"Already watching path: {resolved_path}",
                    "data": None
                }
            
            # Create a watcher ID
            watcher_id = str(uuid.uuid4())
            
            # Create a thread to watch the file
            def watch_thread():
                logger.info(f"Started watching {resolved_path}")
                
                # Get initial file info
                if os.path.isdir(resolved_path):
                    # Watch a directory
                    last_state = {}
                    for root, dirs, files in os.walk(resolved_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                last_state[file_path] = os.path.getmtime(file_path)
                            except Exception:
                                pass
                else:
                    # Watch a file
                    try:
                        last_mtime = os.path.getmtime(resolved_path)
                    except Exception:
                        last_mtime = 0
                
                # Watch for changes
                while watcher_id in self.watchers:
                    try:
                        if os.path.isdir(resolved_path):
                            # Check for changes in the directory
                            current_state = {}
                            for root, dirs, files in os.walk(resolved_path):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    try:
                                        current_state[file_path] = os.path.getmtime(file_path)
                                    except Exception:
                                        pass
                            
                            # Find added, modified, and deleted files
                            for file_path, mtime in current_state.items():
                                if file_path not in last_state:
                                    logger.info(f"File added: {file_path}")
                                    if callback:
                                        callback(file_path, "added")
                                elif mtime > last_state[file_path]:
                                    logger.info(f"File modified: {file_path}")
                                    if callback:
                                        callback(file_path, "modified")
                            
                            for file_path in last_state:
                                if file_path not in current_state:
                                    logger.info(f"File deleted: {file_path}")
                                    if callback:
                                        callback(file_path, "deleted")
                            
                            last_state = current_state
                        else:
                            # Check for changes in the file
                            try:
                                current_mtime = os.path.getmtime(resolved_path)
                                if current_mtime > last_mtime:
                                    logger.info(f"File modified: {resolved_path}")
                                    if callback:
                                        callback(resolved_path, "modified")
                                    last_mtime = current_mtime
                            except FileNotFoundError:
                                logger.info(f"File deleted: {resolved_path}")
                                if callback:
                                    callback(resolved_path, "deleted")
                                break
                    except Exception as e:
                        logger.error(f"Error watching file: {e}")
                    
                    # Sleep for a bit
                    time.sleep(1)
                
                logger.info(f"Stopped watching {resolved_path}")
            
            # Start the thread
            thread = threading.Thread(target=watch_thread, daemon=True)
            thread.start()
            
            # Add to watchers
            self.watchers[resolved_path] = {
                "id": watcher_id,
                "thread": thread,
                "started": time.time()
            }
            
            # Add to history
            self.history.append({
                "operation": "start_file_watcher",
                "path": resolved_path,
                "timestamp": time.time(),
                "watcher_id": watcher_id
            })
            
            return {
                "success": True,
                "message": f"Successfully started watching: {resolved_path}",
                "data": {
                    "path": resolved_path,
                    "watcher_id": watcher_id
                }
            }
        except Exception as e:
            logger.error(f"Error starting file watcher: {e}")
            return {
                "success": False,
                "message": f"Error starting file watcher: {str(e)}",
                "data": None
            }
    
    def stop_file_watcher(self, path: str) -> Dict[str, Any]:
        """
        Stop watching a file or directory for changes
        
        Args:
            path: Path to stop watching
            
        Returns:
            Dictionary with file watcher results
        """
        try:
            # Resolve the path
            resolved_path = self.resolve_path(path)
            
            # Check if we're watching this path
            if resolved_path not in self.watchers:
                return {
                    "success": False,
                    "message": f"Not watching path: {resolved_path}",
                    "data": None
                }
            
            # Get the watcher info
            watcher = self.watchers[resolved_path]
            
            # Remove from watchers
            del self.watchers[resolved_path]
            
            # Add to history
            self.history.append({
                "operation": "stop_file_watcher",
                "path": resolved_path,
                "timestamp": time.time(),
                "watcher_id": watcher["id"]
            })
            
            return {
                "success": True,
                "message": f"Successfully stopped watching: {resolved_path}",
                "data": {
                    "path": resolved_path,
                    "watcher_id": watcher["id"],
                    "duration": time.time() - watcher["started"]
                }
            }
        except Exception as e:
            logger.error(f"Error stopping file watcher: {e}")
            return {
                "success": False,
                "message": f"Error stopping file watcher: {str(e)}",
                "data": None
            }
    
    def get_file_watchers(self) -> Dict[str, Any]:
        """
        Get a list of active file watchers
        
        Returns:
            Dictionary with file watcher results
        """
        try:
            # Get the watchers
            watchers = []
            for path, watcher in self.watchers.items():
                watchers.append({
                    "path": path,
                    "watcher_id": watcher["id"],
                    "started": watcher["started"],
                    "duration": time.time() - watcher["started"]
                })
            
            return {
                "success": True,
                "message": f"Found {len(watchers)} active file watchers",
                "data": {
                    "watchers": watchers,
                    "count": len(watchers)
                }
            }
        except Exception as e:
            logger.error(f"Error getting file watchers: {e}")
            return {
                "success": False,
                "message": f"Error getting file watchers: {str(e)}",
                "data": None
            }
    
    def get_operation_history(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get the operation history
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            Dictionary with operation history results
        """
        try:
            # Sort by timestamp (newest first) and limit
            sorted_history = sorted(self.history, key=lambda x: x["timestamp"], reverse=True)[:limit]
            
            return {
                "success": True,
                "message": f"Retrieved {len(sorted_history)} operation history records",
                "data": {
                    "history": sorted_history,
                    "count": len(sorted_history),
                    "total_operations": len(self.history)
                }
            }
        except Exception as e:
            logger.error(f"Error getting operation history: {e}")
            return {
                "success": False,
                "message": f"Error getting operation history: {str(e)}",
                "data": None
            }

#######################################
# Main Editor Class
#######################################

class Editor:
    """
    Main editor class that integrates all capabilities
    """
    
    def __init__(self, workspace_dir: Optional[str] = None):
        """Initialize the editor with an optional workspace directory"""
        self.workspace_dir = workspace_dir or os.getcwd()
        self.code_editor = CodeEditor(workspace_dir)
        self.network = NetworkOperations()
        self.fs = FileSystemOperations(workspace_dir)
    
    async def initialize(self):
        """Initialize the editor components"""
        # Create a network session
        self.network.create_session()
    
    async def shutdown(self):
        """Shutdown the editor components"""
        # Close the network session
        await self.network.close_session()
        
        # Stop all file watchers
        watchers = self.fs.get_file_watchers()
        if watchers["success"]:
            for watcher in watchers["data"]["watchers"]:
                self.fs.stop_file_watcher(watcher["path"])
    
    def get_version(self) -> str:
        """Get the editor version"""
        return "1.0.0"
    
    def get_status(self) -> Dict[str, Any]:
        """Get the editor status"""
        return {
            "version": self.get_version(),
            "workspace_dir": self.workspace_dir,
            "active_watchers": len(self.fs.watchers),
            "network_requests": len(self.network.history),
            "file_operations": len(self.fs.history)
        }

#######################################
# Command Line Interface
#######################################

async def main():
    """Main function for the editor"""
    parser = argparse.ArgumentParser(description="Advanced Code Editor and Network Operations Tool")
    parser.add_argument("--workspace", "-w", help="Workspace directory")
    parser.add_argument("--command", "-c", help="Command to execute")
    parser.add_argument("--file", "-f", help="File to operate on")
    parser.add_argument("--url", "-u", help="URL to operate on")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create the editor
    editor = Editor(args.workspace)
    await editor.initialize()
    
    try:
        # Process commands
        if args.command:
            if args.command == "status":
                # Show editor status
                status = editor.get_status()
                print(json.dumps(status, indent=2))
            
            elif args.command == "list":
                # List files
                if not args.file:
                    print("Error: No directory specified")
                    return
                
                result = editor.fs.list_files(args.file, recursive=True)
                if result["success"]:
                    print(f"Found {result['data']['summary']['total_items']} items")
                    for item in result["data"]["items"]:
                        print(f"{item['type']:<10} {item['human_size']:<10} {item['path']}")
                else:
                    print(f"Error: {result['message']}")
            
            elif args.command == "read":
                # Read a file
                if not args.file:
                    print("Error: No file specified")
                    return
                
                result = editor.fs.read_file(args.file)
                if result["success"]:
                    if result["data"]["is_binary"]:
                        print(f"Binary file: {result['data']['path']}")
                        print(f"Size: {result['data']['size']} bytes")
                        print(f"Preview: {result['data']['binary_preview']}")
                    else:
                        print(result["data"]["content"])
                else:
                    print(f"Error: {result['message']}")
            
            elif args.command == "analyze":
                # Analyze a file
                if not args.file:
                    print("Error: No file specified")
                    return
                
                result = editor.code_editor.analyze_code(args.file)
                if result[0]:
                    analysis = result[1]
                    print(f"Analysis results for {args.file}:")
                    print(f"Issues: {len(analysis.issues)}")
                    for issue in analysis.issues:
                        print(f"  - {issue['type']}: {issue['message']}")
                    
                    print(f"Metrics:")
                    for key, value in analysis.metrics.items():
                        print(f"  - {key}: {value}")
                    
                    print(f"Suggestions:")
                    for suggestion in analysis.suggestions:
                        print(f"  - {suggestion}")
                else:
                    print(f"Error: {result[1]}")
            
            elif args.command == "format":
                # Format a file
                if not args.file:
                    print("Error: No file specified")
                    return
                
                result = editor.code_editor.refactor_code(args.file, CodeTransformation.FORMAT)
                if result[0]:
                    print(f"Successfully formatted {args.file}")
                else:
                    print(f"Error: {result[1]}")
            
            elif args.command == "download":
                # Download a file
                if not args.url:
                    print("Error: No URL specified")
                    return
                
                if not args.output:
                    print("Error: No output file specified")
                    return
                
                result = await editor.network.download_file(args.url, args.output)
                if result["success"]:
                    print(f"Successfully downloaded {result['size']} bytes to {result['filepath']}")
                    print(f"Speed: {result['speed']:.2f} bytes/second")
                else:
                    print(f"Error: {result['message']}")
            
            elif args.command == "ping":
                # Ping a host
                if not args.url:
                    print("Error: No host specified")
                    return
                
                result = await editor.network.ping(args.url)
                if result["success"]:
                    print(f"Ping results for {result['host']}:")
                    print(f"Packets: {result['packets_sent']} sent, {result['packets_received']} received")
                    print(f"Packet loss: {result['packet_loss']:.1f}%")
                    print(f"Round-trip time: min={result['min_time']}ms, avg={result['avg_time']}ms, max={result['max_time']}ms")
                else:
                    print(f"Error: {result['message']}")
            
            else:
                print(f"Unknown command: {args.command}")
        else:
            # Show help
            print("Advanced Code Editor and Network Operations Tool")
            print("Commands:")
            print("  status              Show editor status")
            print("  list -f <dir>       List files in a directory")
            print("  read -f <file>      Read a file")
            print("  analyze -f <file>   Analyze a file")
            print("  format -f <file>    Format a file")
            print("  download -u <url> -o <file>  Download a file")
            print("  ping -u <host>      Ping a host")
    finally:
        # Shutdown the editor
        await editor.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
