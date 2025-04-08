#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import json
import requests
import asyncio
import time
import random
import uuid
import shutil
import glob
import fnmatch
import subprocess
import re
import inspect
import importlib
import pickle
from collections import deque
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Deque
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Import agent hooks and advanced visualizer
from agent_hooks import CLIAgentHooks
from data_visualizer import AdvancedDataVisualizer
from dynamic_agents import registry, AgentContext, execute_agent_command

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cli-agent")

# Rich console for better formatting
console = Console()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DataAnalysisTools:
    """
    Tools for data analysis that can be used by the agent
    """
    def __init__(self):
        logger.info("Initializing data analysis tools")
        self.visualizer = AdvancedDataVisualizer()
    
    def load_csv(self, filepath: str) -> Dict:
        """Load a CSV file and return basic statistics"""
        try:
            import pandas as pd
            
            # Load the data
            df = pd.read_csv(filepath)
            
            # Generate basic statistics
            stats = {
                "columns": list(df.columns),
                "shape": df.shape,
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "head": df.head(5).to_dict(orient="records"),
                "describe": df.describe().to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            return {
                "success": True,
                "message": f"Successfully loaded CSV file with {df.shape[0]} rows and {df.shape[1]} columns",
                "data": stats
            }
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return {
                "success": False,
                "message": f"Error loading CSV file: {str(e)}",
                "data": None
            }
    
    def plot_data(self, data: Dict, plot_type: str = "histogram", 
                 x_column: str = None, y_column: str = None,
                 title: str = "Data Visualization") -> Dict:
        """Generate a plot from data and save it to a file"""
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create a dataframe from the data
            if isinstance(data, dict) and "head" in data:
                # If data is from load_csv
                df = pd.DataFrame(data["head"])
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                # If data is a list of dictionaries
                df = pd.DataFrame(data)
            else:
                return {
                    "success": False,
                    "message": "Invalid data format for plotting",
                    "data": None
                }
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            if plot_type == "histogram" and x_column:
                sns.histplot(data=df, x=x_column)
            elif plot_type == "scatter" and x_column and y_column:
                sns.scatterplot(data=df, x=x_column, y=y_column)
            elif plot_type == "bar" and x_column and y_column:
                sns.barplot(data=df, x=x_column, y=y_column)
            elif plot_type == "line" and x_column and y_column:
                sns.lineplot(data=df, x=x_column, y=y_column)
            elif plot_type == "heatmap":
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            else:
                return {
                    "success": False,
                    "message": f"Invalid plot type or missing required columns for {plot_type}",
                    "data": None
                }
            
            plt.title(title)
            plt.tight_layout()
            
            # Save the plot
            output_file = f"{plot_type}_{x_column}_{y_column if y_column else ''}.png"
            plt.savefig(output_file)
            plt.close()
            
            return {
                "success": True,
                "message": f"Successfully created {plot_type} plot and saved to {output_file}",
                "data": {
                    "file": output_file,
                    "plot_type": plot_type,
                    "x_column": x_column,
                    "y_column": y_column
                }
            }
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            return {
                "success": False,
                "message": f"Error creating plot: {str(e)}",
                "data": None
            }
    
    def analyze_text(self, text: str) -> Dict:
        """Perform basic text analysis"""
        try:
            # Basic text statistics
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            
            # Word frequency
            import re
            from collections import Counter
            
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = Counter(words).most_common(10)
            
            return {
                "success": True,
                "message": "Successfully analyzed text",
                "data": {
                    "word_count": word_count,
                    "character_count": char_count,
                    "sentence_count": sentence_count,
                    "top_words": word_freq
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                "success": False,
                "message": f"Error analyzing text: {str(e)}",
                "data": None
            }

class ModalIntegration:
    """
    Integration with Modal for running functions in the cloud
    """
    def __init__(self, endpoint="https://arthurcolle--registry.modal.run"):
        self.endpoint = endpoint
        self.available = self._check_availability()
        
        if self.available:
            logger.info(f"Modal integration available at {endpoint}")
        else:
            logger.warning(f"Modal integration not available at {endpoint}")
    
    def _check_availability(self) -> bool:
        """Check if Modal endpoint is available"""
        try:
            response = requests.get(self.endpoint, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_functions(self) -> Dict:
        """List available functions in Modal"""
        if not self.available:
            return {
                "success": False,
                "message": "Modal integration not available",
                "data": None
            }
        
        try:
            response = requests.get(f"{self.endpoint}/functions")
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Successfully retrieved Modal functions",
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "message": f"Error retrieving Modal functions: {response.status_code}",
                    "data": None
                }
        except Exception as e:
            logger.error(f"Error listing Modal functions: {e}")
            return {
                "success": False,
                "message": f"Error listing Modal functions: {str(e)}",
                "data": None
            }
    
    def call_function(self, function_name: str, params: Dict) -> Dict:
        """Call a function in Modal"""
        if not self.available:
            return {
                "success": False,
                "message": "Modal integration not available",
                "data": None
            }
        
        try:
            response = requests.post(
                f"{self.endpoint}/functions/{function_name}",
                json=params
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": f"Successfully called Modal function {function_name}",
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "message": f"Error calling Modal function: {response.status_code}",
                    "data": None
                }
        except Exception as e:
            logger.error(f"Error calling Modal function: {e}")
            return {
                "success": False,
                "message": f"Error calling Modal function: {str(e)}",
                "data": None
            }

class FileSystemTools:
    """
    Advanced tools for interacting with the file system
    Provides comprehensive file operations with safety checks and detailed metadata
    """
    def __init__(self):
        logger.info("Initializing advanced file system tools")
        self.history = []  # Track file operations for potential undo
        self.safe_mode = True  # Safety mode to prevent destructive operations
        
    def list_files(self, path: str = ".", pattern: str = "*", recursive: bool = False, 
                  include_hidden: bool = False, sort_by: str = "name") -> Dict:
        """
        List files in a directory with advanced filtering and sorting options
        
        Args:
            path: Directory path to list files from
            pattern: Glob pattern to filter files
            recursive: Whether to recursively list files in subdirectories
            include_hidden: Whether to include hidden files (starting with .)
            sort_by: How to sort results (name, size, modified, type)
        """
        try:
            # Normalize path
            norm_path = os.path.normpath(os.path.expanduser(path))
            
            # Get files matching pattern
            if recursive:
                matches = []
                for root, dirnames, filenames in os.walk(norm_path):
                    for filename in filenames:
                        if fnmatch.fnmatch(filename, pattern):
                            if include_hidden or not filename.startswith('.'):
                                matches.append(os.path.join(root, filename))
                    # Add directories if requested
                    for dirname in dirnames:
                        if fnmatch.fnmatch(dirname, pattern):
                            if include_hidden or not dirname.startswith('.'):
                                matches.append(os.path.join(root, dirname))
                files = matches
            else:
                files = glob.glob(os.path.join(norm_path, pattern))
                if not include_hidden:
                    files = [f for f in files if not os.path.basename(f).startswith('.')]
            
            # Get detailed file info
            file_info = []
            for file_path in files:
                try:
                    stat = os.stat(file_path)
                    is_dir = os.path.isdir(file_path)
                    
                    # Get file type and mime type
                    file_type = "directory" if is_dir else "file"
                    mime_type = None
                    if not is_dir:
                        try:
                            import magic
                            mime_type = magic.from_file(file_path, mime=True)
                        except ImportError:
                            # Fallback to simple extension-based detection
                            ext = os.path.splitext(file_path)[1].lower()
                            mime_map = {
                                '.txt': 'text/plain', '.py': 'text/x-python',
                                '.jpg': 'image/jpeg', '.png': 'image/png',
                                '.pdf': 'application/pdf', '.json': 'application/json'
                            }
                            mime_type = mime_map.get(ext, 'application/octet-stream')
                    
                    # Calculate human-readable size
                    size_bytes = stat.st_size
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if size_bytes < 1024 or unit == 'TB':
                            human_size = f"{size_bytes:.2f} {unit}"
                            break
                        size_bytes /= 1024
                    
                    file_info.append({
                        "name": os.path.basename(file_path),
                        "path": file_path,
                        "size": stat.st_size,
                        "human_size": human_size,
                        "modified": time.ctime(stat.st_mtime),
                        "modified_timestamp": stat.st_mtime,
                        "created": time.ctime(stat.st_ctime),
                        "created_timestamp": stat.st_ctime,
                        "accessed": time.ctime(stat.st_atime),
                        "is_dir": is_dir,
                        "type": file_type,
                        "mime_type": mime_type,
                        "permissions": oct(stat.st_mode)[-3:],
                        "owner": stat.st_uid,
                        "group": stat.st_gid
                    })
                except Exception as e:
                    logger.warning(f"Error getting info for {file_path}: {e}")
            
            # Sort results
            if sort_by == "name":
                file_info.sort(key=lambda x: x["name"])
            elif sort_by == "size":
                file_info.sort(key=lambda x: x["size"], reverse=True)
            elif sort_by == "modified":
                file_info.sort(key=lambda x: x["modified_timestamp"], reverse=True)
            elif sort_by == "type":
                file_info.sort(key=lambda x: (x["is_dir"], x["name"]), reverse=True)
            
            # Add summary statistics
            total_size = sum(item["size"] for item in file_info)
            dir_count = sum(1 for item in file_info if item["is_dir"])
            file_count = len(file_info) - dir_count
            
            return {
                "success": True,
                "message": f"Found {len(file_info)} items matching pattern '{pattern}' in '{norm_path}'",
                "data": {
                    "items": file_info,
                    "summary": {
                        "total_items": len(file_info),
                        "directories": dir_count,
                        "files": file_count,
                        "total_size": total_size,
                        "path": norm_path,
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
    
    def read_file(self, filepath: str, max_size: int = 1024 * 1024, 
                 encoding: str = 'utf-8', chunk_size: int = None,
                 line_numbers: bool = False, syntax_highlight: bool = False) -> Dict:
        """
        Read the contents of a file with advanced options
        
        Args:
            filepath: Path to the file to read
            max_size: Maximum file size in bytes
            encoding: File encoding to use
            chunk_size: If set, read only this many bytes
            line_numbers: Whether to include line numbers
            syntax_highlight: Whether to detect and include syntax highlighting info
        """
        try:
            # Normalize path
            norm_path = os.path.normpath(os.path.expanduser(filepath))
            
            # Check if file exists
            if not os.path.exists(norm_path):
                return {
                    "success": False,
                    "message": f"File not found: {norm_path}",
                    "data": None
                }
            
            # Check if it's a directory
            if os.path.isdir(norm_path):
                return {
                    "success": False,
                    "message": f"Cannot read directory as file: {norm_path}",
                    "data": None
                }
            
            # Check file size
            file_size = os.path.getsize(norm_path)
            if file_size > max_size:
                return {
                    "success": False,
                    "message": f"File too large ({file_size} bytes). Max size is {max_size} bytes.",
                    "data": None
                }
            
            # Detect binary file
            try:
                is_binary = False
                with open(norm_path, 'rb') as f:
                    chunk = f.read(1024)
                    if b'\0' in chunk:  # Simple binary detection
                        is_binary = True
                
                if is_binary:
                    # For binary files, return hex dump instead of text content
                    with open(norm_path, 'rb') as f:
                        binary_data = f.read(chunk_size or max_size)
                    
                    hex_dump = ' '.join(f'{b:02x}' for b in binary_data[:100])  # First 100 bytes
                    
                    return {
                        "success": True,
                        "message": f"Successfully read binary file: {norm_path} ({file_size} bytes)",
                        "data": {
                            "content": f"Binary file: first 100 bytes: {hex_dump}...",
                            "is_binary": True,
                            "size": file_size,
                            "path": norm_path,
                            "binary_preview": hex_dump
                        }
                    }
            except Exception as e:
                logger.warning(f"Error detecting binary file: {e}")
            
            # Read file content
            content = ""
            if chunk_size:
                with open(norm_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read(chunk_size)
                    truncated = file_size > chunk_size
            else:
                with open(norm_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                    truncated = False
            
            # Process content based on options
            if line_numbers:
                lines = content.splitlines()
                content_with_lines = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
                content = content_with_lines
            
            # Detect file type for syntax highlighting
            file_type = None
            if syntax_highlight:
                ext = os.path.splitext(norm_path)[1].lower()
                file_type_map = {
                    '.py': 'python', '.js': 'javascript', '.html': 'html',
                    '.css': 'css', '.json': 'json', '.md': 'markdown',
                    '.xml': 'xml', '.yaml': 'yaml', '.yml': 'yaml',
                    '.sh': 'bash', '.bash': 'bash', '.sql': 'sql',
                    '.c': 'c', '.cpp': 'cpp', '.h': 'c', '.java': 'java'
                }
                file_type = file_type_map.get(ext)
            
            # Get file metadata
            stat = os.stat(norm_path)
            
            return {
                "success": True,
                "message": f"Successfully read file: {norm_path} ({len(content)} bytes)",
                "data": {
                    "content": content,
                    "size": file_size,
                    "path": norm_path,
                    "encoding": encoding,
                    "truncated": truncated,
                    "line_count": content.count('\n') + 1,
                    "modified": time.ctime(stat.st_mtime),
                    "file_type": file_type,
                    "is_binary": False
                }
            }
        except UnicodeDecodeError:
            # If we hit a decode error, try to read as binary
            try:
                with open(norm_path, 'rb') as f:
                    binary_data = f.read(100)  # Just read a small preview
                
                hex_dump = ' '.join(f'{b:02x}' for b in binary_data)
                
                return {
                    "success": True,
                    "message": f"File appears to be binary: {norm_path}",
                    "data": {
                        "content": f"Binary file: first 100 bytes: {hex_dump}...",
                        "is_binary": True,
                        "size": file_size,
                        "path": norm_path,
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
    
    def write_file(self, filepath: str, content: str, overwrite: bool = False, 
                  append: bool = False, encoding: str = 'utf-8', 
                  create_backup: bool = False, mode: str = None) -> Dict:
        """
        Write content to a file with advanced options
        
        Args:
            filepath: Path to the file to write
            content: Content to write to the file
            overwrite: Whether to overwrite existing files
            append: Whether to append to existing files
            encoding: File encoding to use
            create_backup: Whether to create a backup of existing file
            mode: File permissions mode (e.g., '644')
        """
        try:
            # Normalize path
            norm_path = os.path.normpath(os.path.expanduser(filepath))
            
            # Safety check for system directories
            system_dirs = ['/bin', '/sbin', '/usr/bin', '/usr/sbin', '/etc/passwd', '/etc/shadow']
            if any(norm_path.startswith(d) for d in system_dirs) and self.safe_mode:
                return {
                    "success": False,
                    "message": f"Safety check: Cannot write to system directory: {norm_path}",
                    "data": None
                }
            
            # Check if file exists
            file_exists = os.path.exists(norm_path)
            
            # Handle existing file
            if file_exists:
                if not (overwrite or append):
                    return {
                        "success": False,
                        "message": f"File already exists: {norm_path}. Set overwrite=true to overwrite or append=true to append.",
                        "data": None
                    }
                
                # Create backup if requested
                if create_backup:
                    backup_path = f"{norm_path}.bak"
                    shutil.copy2(norm_path, backup_path)
                    logger.info(f"Created backup of {norm_path} at {backup_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(norm_path)), exist_ok=True)
            
            # Determine write mode
            write_mode = 'a' if append else 'w'
            
            # Write file
            with open(norm_path, write_mode, encoding=encoding) as f:
                f.write(content)
            
            # Set file mode if specified
            if mode:
                try:
                    mode_int = int(mode, 8)
                    os.chmod(norm_path, mode_int)
                except ValueError:
                    logger.warning(f"Invalid mode format: {mode}. Expected octal (e.g., '644')")
            
            # Add to history for potential undo
            operation = "append" if append else "write"
            self.history.append({
                "operation": operation,
                "path": norm_path,
                "timestamp": time.time(),
                "size": len(content),
                "backup": f"{norm_path}.bak" if create_backup else None
            })
            
            return {
                "success": True,
                "message": f"Successfully {operation}ed {len(content)} bytes to {norm_path}",
                "data": {
                    "path": norm_path,
                    "size": len(content),
                    "operation": operation,
                    "backup": f"{norm_path}.bak" if create_backup else None
                }
            }
        except Exception as e:
            logger.error(f"Error writing file: {e}")
            return {
                "success": False,
                "message": f"Error writing file: {str(e)}",
                "data": None
            }
    
    def copy_file(self, source: str, destination: str, overwrite: bool = False) -> Dict:
        """Copy a file from source to destination"""
        try:
            # Normalize paths
            norm_source = os.path.normpath(os.path.expanduser(source))
            norm_dest = os.path.normpath(os.path.expanduser(destination))
            
            # Check if source exists
            if not os.path.exists(norm_source):
                return {
                    "success": False,
                    "message": f"Source file not found: {norm_source}",
                    "data": None
                }
            
            # Check if destination exists and overwrite is False
            if os.path.exists(norm_dest) and not overwrite:
                return {
                    "success": False,
                    "message": f"Destination file already exists: {norm_dest}. Set overwrite=true to overwrite.",
                    "data": None
                }
            
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(norm_dest)), exist_ok=True)
            
            # Copy file
            shutil.copy2(norm_source, norm_dest)
            
            return {
                "success": True,
                "message": f"Successfully copied {norm_source} to {norm_dest}",
                "data": {
                    "source": norm_source,
                    "destination": norm_dest
                }
            }
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            return {
                "success": False,
                "message": f"Error copying file: {str(e)}",
                "data": None
            }
    
    def delete_file(self, filepath: str, recursive: bool = False) -> Dict:
        """Delete a file or directory"""
        try:
            # Normalize path
            norm_path = os.path.normpath(os.path.expanduser(filepath))
            
            # Check if file exists
            if not os.path.exists(norm_path):
                return {
                    "success": False,
                    "message": f"File not found: {norm_path}",
                    "data": None
                }
            
            # Delete file or directory
            if os.path.isdir(norm_path):
                if recursive:
                    shutil.rmtree(norm_path)
                else:
                    os.rmdir(norm_path)
            else:
                os.remove(norm_path)
            
            return {
                "success": True,
                "message": f"Successfully deleted: {norm_path}",
                "data": {
                    "path": norm_path,
                    "was_directory": os.path.isdir(norm_path)
                }
            }
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return {
                "success": False,
                "message": f"Error deleting file: {str(e)}",
                "data": None
            }

class CodingTools:
    """
    Advanced tools for code execution and management with sandboxing and analysis
    """
    def __init__(self):
        logger.info("Initializing advanced coding tools")
        self.temp_dir = os.path.join(os.getcwd(), "temp_code")
        self.sandbox_dir = os.path.join(os.getcwd(), "sandbox")
        self.history_dir = os.path.join(os.getcwd(), "code_history")
        
        # Create necessary directories
        for directory in [self.temp_dir, self.sandbox_dir, self.history_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Track execution history
        self.execution_history = []
        
        # Restricted commands that won't be allowed in shell execution
        self.restricted_commands = [
            "rm -rf", "mkfs", "dd if=/dev/zero", ":(){ :|:& };:",  # Fork bomb
            "> /dev/sda", "chmod -R 777 /", "mv /* /dev/null"
        ]
        
        # Default allowed imports for Python code
        self.allowed_imports = {
            "safe": ["math", "random", "datetime", "collections", "itertools", 
                    "functools", "re", "json", "csv", "os.path", "time"],
            "data_science": ["numpy", "pandas", "matplotlib", "seaborn", "sklearn"],
            "standard_library": ["os", "sys", "subprocess", "pathlib", "shutil"],
            "web": ["requests", "bs4", "urllib"],
            "all": []  # Empty means no restrictions
        }
        
        # Current security level
        self.security_level = "standard_library"  # Default level
    
    def set_security_level(self, level: str) -> Dict:
        """Set the security level for code execution"""
        valid_levels = ["safe", "data_science", "standard_library", "web", "all"]
        
        if level not in valid_levels:
            return {
                "success": False,
                "message": f"Invalid security level: {level}. Valid levels are: {', '.join(valid_levels)}",
                "data": None
            }
        
        self.security_level = level
        return {
            "success": True,
            "message": f"Security level set to: {level}",
            "data": {
                "level": level,
                "allowed_imports": self.allowed_imports[level] if level != "all" else "All imports allowed"
            }
        }
    
    def _check_code_safety(self, code: str) -> Tuple[bool, str]:
        """Check if Python code is safe to execute"""
        import ast
        
        # Don't restrict if security level is 'all'
        if self.security_level == "all":
            return True, "No restrictions applied"
        
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Check for imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
            
            # Check if all imports are allowed
            allowed = self.allowed_imports[self.security_level]
            disallowed = [imp for imp in imports if imp not in allowed and imp != ""]
            
            if disallowed:
                return False, f"Disallowed imports: {', '.join(disallowed)}"
            
            # Check for potentially dangerous operations
            for node in ast.walk(tree):
                # Check for exec or eval
                if isinstance(node, ast.Call) and hasattr(node, 'func') and hasattr(node.func, 'id'):
                    if node.func.id in ['exec', 'eval']:
                        return False, "Use of exec() or eval() is not allowed"
                
                # Check for __import__
                if isinstance(node, ast.Call) and hasattr(node, 'func') and hasattr(node.func, 'id'):
                    if node.func.id == '__import__':
                        return False, "Use of __import__() is not allowed"
            
            return True, "Code passed safety checks"
        except SyntaxError as e:
            return False, f"Syntax error in code: {str(e)}"
        except Exception as e:
            return False, f"Error checking code safety: {str(e)}"
    
    def _check_shell_safety(self, command: str) -> Tuple[bool, str]:
        """Check if shell command is safe to execute"""
        # Check for restricted commands
        for restricted in self.restricted_commands:
            if restricted in command:
                return False, f"Command contains restricted pattern: {restricted}"
        
        # Check for pipe to shell or command substitution if in safe mode
        if self.security_level in ["safe", "data_science"]:
            if "|" in command or ">" in command or "$(" in command or "`" in command:
                return False, "Command contains pipes, redirections, or command substitution which are not allowed in current security level"
        
        return True, "Command passed safety checks"
    
    def execute_python(self, code: str, timeout: int = 10, save_history: bool = True,
                      sandbox: bool = True, allow_imports: List[str] = None,
                      provide_inputs: List[str] = None, environment: Dict[str, str] = None) -> Dict:
        """
        Execute Python code in a controlled environment with advanced options
        
        Args:
            code: Python code to execute
            timeout: Timeout in seconds
            save_history: Whether to save execution history
            sandbox: Whether to run in a sandbox directory
            allow_imports: Additional imports to allow for this execution
            provide_inputs: List of inputs to provide to the program
            environment: Environment variables to set
        """
        start_time = time.time()
        
        try:
            # Check code safety
            is_safe, safety_message = self._check_code_safety(code)
            if not is_safe and not allow_imports:
                return {
                    "success": False,
                    "message": f"Code safety check failed: {safety_message}",
                    "data": {
                        "code": code,
                        "safety_check": safety_message
                    }
                }
            
            # Create a unique ID for this execution
            exec_id = uuid.uuid4().hex
            
            # Determine execution directory
            exec_dir = self.sandbox_dir if sandbox else self.temp_dir
            temp_file = os.path.join(exec_dir, f"code_{exec_id}.py")
            
            # Save code to history if requested
            if save_history:
                history_file = os.path.join(self.history_dir, f"python_{exec_id}.py")
                with open(history_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Execution ID: {exec_id}\n")
                    f.write(f"# Timestamp: {time.ctime()}\n")
                    f.write(f"# Security level: {self.security_level}\n")
                    f.write(f"# Safety check: {safety_message}\n\n")
                    f.write(code)
            
            # Write code to temporary file
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Prepare environment
            env = os.environ.copy()
            if environment:
                env.update(environment)
            
            # Execute code in a subprocess
            if provide_inputs:
                # If inputs are provided, we need to communicate with the process
                input_text = "\n".join(provide_inputs)
                process = subprocess.Popen(
                    [sys.executable, temp_file],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=exec_dir,
                    env=env
                )
                
                try:
                    stdout, stderr = process.communicate(input=input_text, timeout=timeout)
                    return_code = process.returncode
                except subprocess.TimeoutExpired:
                    process.kill()
                    return {
                        "success": False,
                        "message": f"Code execution timed out after {timeout} seconds",
                        "data": {
                            "stdout": "",
                            "stderr": "Execution timed out",
                            "return_code": -1,
                            "execution_time": time.time() - start_time,
                            "exec_id": exec_id
                        }
                    }
            else:
                # No inputs needed
                process = subprocess.Popen(
                    [sys.executable, temp_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=exec_dir,
                    env=env
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    return_code = process.returncode
                except subprocess.TimeoutExpired:
                    process.kill()
                    return {
                        "success": False,
                        "message": f"Code execution timed out after {timeout} seconds",
                        "data": {
                            "stdout": "",
                            "stderr": "Execution timed out",
                            "return_code": -1,
                            "execution_time": time.time() - start_time,
                            "exec_id": exec_id
                        }
                    }
            
            # Clean up
            try:
                os.remove(temp_file)
            except:
                pass
            
            # Record execution in history
            execution_record = {
                "id": exec_id,
                "type": "python",
                "timestamp": time.time(),
                "code": code,
                "return_code": return_code,
                "execution_time": time.time() - start_time,
                "security_level": self.security_level,
                "sandbox": sandbox
            }
            self.execution_history.append(execution_record)
            
            # Check for syntax errors in stderr
            syntax_error = None
            if "SyntaxError" in stderr:
                for line in stderr.splitlines():
                    if "SyntaxError" in line:
                        syntax_error = line
                        break
            
            return {
                "success": return_code == 0,
                "message": "Code executed successfully" if return_code == 0 else f"Code execution failed with return code {return_code}",
                "data": {
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": return_code,
                    "execution_time": time.time() - start_time,
                    "exec_id": exec_id,
                    "syntax_error": syntax_error,
                    "history_file": history_file if save_history else None
                }
            }
        except Exception as e:
            logger.error(f"Error executing Python code: {e}")
            return {
                "success": False,
                "message": f"Error executing Python code: {str(e)}",
                "data": {
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }
            }
    
    def execute_shell(self, command: str, timeout: int = 10, save_history: bool = True,
                     sandbox: bool = True, environment: Dict[str, str] = None,
                     working_dir: str = None) -> Dict:
        """
        Execute a shell command with advanced options
        
        Args:
            command: Shell command to execute
            timeout: Timeout in seconds
            save_history: Whether to save execution history
            sandbox: Whether to run in a sandbox directory
            environment: Environment variables to set
            working_dir: Working directory for command execution
        """
        start_time = time.time()
        
        try:
            # Check command safety
            is_safe, safety_message = self._check_shell_safety(command)
            if not is_safe:
                return {
                    "success": False,
                    "message": f"Command safety check failed: {safety_message}",
                    "data": {
                        "command": command,
                        "safety_check": safety_message
                    }
                }
            
            # Create a unique ID for this execution
            exec_id = uuid.uuid4().hex
            
            # Determine execution directory
            if working_dir:
                exec_dir = os.path.expanduser(working_dir)
            else:
                exec_dir = self.sandbox_dir if sandbox else os.getcwd()
            
            # Save command to history if requested
            if save_history:
                history_file = os.path.join(self.history_dir, f"shell_{exec_id}.sh")
                with open(history_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Execution ID: {exec_id}\n")
                    f.write(f"# Timestamp: {time.ctime()}\n")
                    f.write(f"# Security level: {self.security_level}\n")
                    f.write(f"# Safety check: {safety_message}\n")
                    f.write(f"# Working directory: {exec_dir}\n\n")
                    f.write(command)
            
            # Prepare environment
            env = os.environ.copy()
            if environment:
                env.update(environment)
            
            # Execute command in a subprocess
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=exec_dir,
                env=env
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    "success": False,
                    "message": f"Command execution timed out after {timeout} seconds",
                    "data": {
                        "stdout": "",
                        "stderr": "Execution timed out",
                        "return_code": -1,
                        "execution_time": time.time() - start_time,
                        "exec_id": exec_id
                    }
                }
            
            # Record execution in history
            execution_record = {
                "id": exec_id,
                "type": "shell",
                "timestamp": time.time(),
                "command": command,
                "return_code": return_code,
                "execution_time": time.time() - start_time,
                "security_level": self.security_level,
                "working_dir": exec_dir
            }
            self.execution_history.append(execution_record)
            
            return {
                "success": return_code == 0,
                "message": "Command executed successfully" if return_code == 0 else f"Command execution failed with return code {return_code}",
                "data": {
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_code": return_code,
                    "command": command,
                    "execution_time": time.time() - start_time,
                    "exec_id": exec_id,
                    "working_dir": exec_dir,
                    "history_file": history_file if save_history else None
                }
            }
        except Exception as e:
            logger.error(f"Error executing shell command: {e}")
            return {
                "success": False,
                "message": f"Error executing shell command: {str(e)}",
                "data": {
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }
            }
    
    def analyze_code(self, code: str) -> Dict:
        """
        Analyze Python code for quality, complexity, and potential issues
        
        Args:
            code: Python code to analyze
        """
        try:
            # Create a temporary file
            temp_file = os.path.join(self.temp_dir, f"analysis_{uuid.uuid4().hex}.py")
            
            # Write code to file
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            results = {
                "syntax_valid": True,
                "imports": [],
                "functions": [],
                "classes": [],
                "complexity": {},
                "issues": []
            }
            
            # Check syntax
            try:
                import ast
                ast.parse(code)
            except SyntaxError as e:
                results["syntax_valid"] = False
                results["issues"].append({
                    "type": "syntax_error",
                    "message": str(e),
                    "line": e.lineno,
                    "offset": e.offset
                })
                return {
                    "success": True,
                    "message": "Code analysis completed with syntax errors",
                    "data": results
                }
            
            # Extract imports, functions, and classes
            try:
                tree = ast.parse(code)
                
                # Find imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            results["imports"].append({
                                "name": name.name,
                                "alias": name.asname
                            })
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for name in node.names:
                            results["imports"].append({
                                "name": f"{module}.{name.name}" if module else name.name,
                                "alias": name.asname,
                                "from_import": True,
                                "module": module
                            })
                
                # Find functions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        args = []
                        for arg in node.args.args:
                            args.append(arg.arg)
                        
                        results["functions"].append({
                            "name": node.name,
                            "args": args,
                            "line": node.lineno,
                            "docstring": ast.get_docstring(node)
                        })
                
                # Find classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        bases = []
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                bases.append(base.id)
                        
                        methods = []
                        for child in node.body:
                            if isinstance(child, ast.FunctionDef):
                                methods.append(child.name)
                        
                        results["classes"].append({
                            "name": node.name,
                            "bases": bases,
                            "methods": methods,
                            "line": node.lineno,
                            "docstring": ast.get_docstring(node)
                        })
            except Exception as e:
                results["issues"].append({
                    "type": "ast_error",
                    "message": f"Error parsing code structure: {str(e)}"
                })
            
            # Check for potential issues
            try:
                # Check for unused imports
                import_names = set()
                for imp in results["imports"]:
                    if imp.get("alias"):
                        import_names.add(imp["alias"])
                    else:
                        name = imp["name"].split(".")[-1] if "." in imp["name"] else imp["name"]
                        import_names.add(name)
                
                # Simple check for unused imports by looking for the name in the code
                for name in import_names:
                    # Count occurrences after import statement
                    if code.count(name) <= 1:  # Only appears in import statement
                        results["issues"].append({
                            "type": "unused_import",
                            "message": f"Potential unused import: {name}"
                        })
                
                # Check for TODO comments
                todo_pattern = r"#\s*TODO:?\s*(.*)"
                for i, line in enumerate(code.splitlines()):
                    match = re.search(todo_pattern, line, re.IGNORECASE)
                    if match:
                        results["issues"].append({
                            "type": "todo",
                            "message": match.group(1).strip(),
                            "line": i + 1
                        })
            except Exception as e:
                results["issues"].append({
                    "type": "analysis_error",
                    "message": f"Error during code analysis: {str(e)}"
                })
            
            # Calculate code complexity
            try:
                # Count lines of code
                lines = code.splitlines()
                non_empty_lines = [line for line in lines if line.strip()]
                comment_lines = [line for line in lines if line.strip().startswith("#")]
                
                results["complexity"] = {
                    "total_lines": len(lines),
                    "code_lines": len(non_empty_lines) - len(comment_lines),
                    "comment_lines": len(comment_lines),
                    "blank_lines": len(lines) - len(non_empty_lines),
                    "comment_ratio": len(comment_lines) / len(non_empty_lines) if non_empty_lines else 0
                }
                
                # Try to use radon for cyclomatic complexity if available
                try:
                    import radon.complexity as cc
                    import radon.metrics as metrics
                    
                    # Calculate cyclomatic complexity
                    complexity = cc.cc_visit(code)
                    if complexity:
                        results["complexity"]["cyclomatic"] = [
                            {
                                "name": c.name,
                                "complexity": c.complexity,
                                "rank": cc.rank(c.complexity)
                            } for c in complexity
                        ]
                    
                    # Calculate maintainability index
                    mi = metrics.mi_visit(code, True)
                    if mi:
                        results["complexity"]["maintainability_index"] = mi
                except ImportError:
                    # Radon not available
                    pass
            except Exception as e:
                results["issues"].append({
                    "type": "complexity_error",
                    "message": f"Error calculating code complexity: {str(e)}"
                })
            
            # Clean up
            try:
                os.remove(temp_file)
            except:
                pass
            
            return {
                "success": True,
                "message": "Code analysis completed successfully",
                "data": results
            }
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {
                "success": False,
                "message": f"Error analyzing code: {str(e)}",
                "data": None
            }
    
    def get_execution_history(self, limit: int = 10, execution_type: str = None) -> Dict:
        """Get the execution history"""
        try:
            # Filter by type if specified
            if execution_type:
                history = [h for h in self.execution_history if h["type"] == execution_type]
            else:
                history = self.execution_history
            
            # Sort by timestamp (newest first) and limit
            sorted_history = sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
            
            return {
                "success": True,
                "message": f"Retrieved {len(sorted_history)} execution history records",
                "data": sorted_history
            }
        except Exception as e:
            logger.error(f"Error getting execution history: {e}")
            return {
                "success": False,
                "message": f"Error getting execution history: {str(e)}",
                "data": None
            }

class DynamicToolRegistry:
    """
    Registry for dynamically adding and managing tools that the agent can use
    """
    def __init__(self, console=None):
        self.console = console or Console()
        self.tools = {}
        self.tool_descriptions = {}
        self.tool_categories = {}
        self.tool_modules = {}
        
    def register_tool(self, name: str, func: Callable, description: str, 
                     parameters: Dict[str, Any], category: str = "custom") -> Dict:
        """
        Register a new tool that the agent can use
        
        Args:
            name: Name of the tool
            func: Function to call when the tool is invoked
            description: Description of what the tool does
            parameters: OpenAI-compatible parameters schema
            category: Category for organizing tools
        """
        if name in self.tools:
            return {
                "success": False,
                "message": f"Tool with name '{name}' already exists",
                "data": None
            }
        
        self.tools[name] = func
        self.tool_descriptions[name] = description
        
        # Add to category
        if category not in self.tool_categories:
            self.tool_categories[category] = []
        self.tool_categories[category].append(name)
        
        # Store the module where this function is defined
        module = inspect.getmodule(func)
        if module:
            self.tool_modules[name] = module.__name__
        
        return {
            "success": True,
            "message": f"Successfully registered tool '{name}' in category '{category}'",
            "data": {
                "name": name,
                "description": description,
                "category": category,
                "parameters": parameters
            }
        }
    
    def unregister_tool(self, name: str) -> Dict:
        """Remove a tool from the registry"""
        if name not in self.tools:
            return {
                "success": False,
                "message": f"Tool with name '{name}' does not exist",
                "data": None
            }
        
        # Remove from tools and descriptions
        func = self.tools.pop(name)
        description = self.tool_descriptions.pop(name)
        
        # Remove from categories
        for category, tools in self.tool_categories.items():
            if name in tools:
                tools.remove(name)
                break
        
        # Remove from modules
        if name in self.tool_modules:
            module_name = self.tool_modules.pop(name)
        else:
            module_name = None
        
        return {
            "success": True,
            "message": f"Successfully unregistered tool '{name}'",
            "data": {
                "name": name,
                "description": description,
                "module": module_name
            }
        }
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_tool_description(self, name: str) -> Optional[str]:
        """Get a tool's description by name"""
        return self.tool_descriptions.get(name)
    
    def list_tools(self, category: str = None) -> List[str]:
        """List all tools or tools in a specific category"""
        if category:
            return self.tool_categories.get(category, [])
        else:
            return list(self.tools.keys())
    
    def list_categories(self) -> List[str]:
        """List all tool categories"""
        return list(self.tool_categories.keys())
    
    def get_openai_tools_format(self, category: str = None) -> List[Dict]:
        """
        Get tools in OpenAI's tool format for the API
        
        Args:
            category: Optional category to filter tools
        """
        tools = []
        
        tool_names = self.list_tools(category)
        for name in tool_names:
            # Skip if we don't have a description
            if name not in self.tool_descriptions:
                continue
                
            # Get the function
            func = self.tools[name]
            
            # Create the tool definition
            tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": self.tool_descriptions[name],
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
            
            # Try to get the function signature
            try:
                # Skip if func is not callable
                if not callable(func):
                    continue
                    
                sig = inspect.signature(func)
                
                # Add required parameters
                required = []
                for param_name, param in sig.parameters.items():
                    # Skip self parameter for methods
                    if param_name == "self":
                        continue
                        
                    # Add to required if no default value
                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)
                
                if required:
                    tool["function"]["parameters"]["required"] = required
            except (ValueError, TypeError) as e:
                # If we can't get the signature, just continue without required parameters
                logging.warning(f"Could not get signature for tool {name}: {e}")
                
            tools.append(tool)
            
        return tools
    
    def execute_tool(self, name: str, arguments: Dict) -> Dict:
        """Execute a tool by name with the given arguments"""
        if name not in self.tools:
            return {
                "success": False,
                "message": f"Tool with name '{name}' does not exist",
                "data": None
            }
        
        try:
            func = self.tools[name]
            result = func(**arguments)
            
            # If the result is already a dict with success/message/data, return it
            if isinstance(result, dict) and "success" in result and "message" in result:
                return result
                
            # Otherwise, wrap the result
            return {
                "success": True,
                "message": f"Successfully executed tool '{name}'",
                "data": result
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error executing tool '{name}': {str(e)}",
                "data": None
            }
    
    def load_tool_from_code(self, name: str, code: str, description: str, 
                           parameters: Dict[str, Any], category: str = "custom") -> Dict:
        """
        Load a tool from Python code
        
        Args:
            name: Name of the tool
            code: Python code defining the function
            description: Description of what the tool does
            parameters: OpenAI-compatible parameters schema
            category: Category for organizing tools
        """
        try:
            # Create a temporary module
            module_name = f"dynamic_tool_{name}"
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            
            # Execute the code in the module's namespace
            exec(code, module.__dict__)
            
            # Find the function in the module
            func = None
            for item_name, item in module.__dict__.items():
                if callable(item) and not item_name.startswith("__"):
                    func = item
                    break
            
            if not func:
                return {
                    "success": False,
                    "message": "No function found in the provided code",
                    "data": None
                }
            
            # Register the tool
            return self.register_tool(name, func, description, parameters, category)
        except Exception as e:
            return {
                "success": False,
                "message": f"Error loading tool from code: {str(e)}",
                "data": None
            }

class PerceptualMemory:
    """
    Advanced memory system for storing and retrieving interaction frames
    Enables experience replay and semantic search across past interactions
    """
    def __init__(self, max_frames=1000, embedding_dim=1536, vector_db_path="perceptual_memory.pkl"):
        self.frames = deque(maxlen=max_frames)  # Store recent frames in memory
        self.frame_embeddings = deque(maxlen=max_frames)  # Store embeddings for semantic search
        self.vector_db_path = vector_db_path
        self.embedding_dim = embedding_dim
        
        # Load existing memory if available
        self._load_memory()
        
        # Track statistics
        self.stats = {
            "total_frames_added": 0,
            "total_retrievals": 0,
            "total_replays": 0
        }
    
    def _load_memory(self):
        """Load memory from disk if available"""
        try:
            if os.path.exists(self.vector_db_path):
                with open(self.vector_db_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.frames = saved_data.get('frames', deque(maxlen=self.frames.maxlen))
                    self.frame_embeddings = saved_data.get('embeddings', deque(maxlen=self.frames.maxlen))
                    self.stats = saved_data.get('stats', self.stats)
                    logging.info(f"Loaded {len(self.frames)} perceptual frames from disk")
        except Exception as e:
            logging.error(f"Error loading perceptual memory: {e}")
    
    def _save_memory(self):
        """Save memory to disk"""
        try:
            with open(self.vector_db_path, 'wb') as f:
                pickle.dump({
                    'frames': self.frames,
                    'embeddings': self.frame_embeddings,
                    'stats': self.stats
                }, f)
            logging.info(f"Saved {len(self.frames)} perceptual frames to disk")
        except Exception as e:
            logging.error(f"Error saving perceptual memory: {e}")
    
    def add_frame(self, frame: Dict, embedding: Optional[np.ndarray] = None) -> str:
        """
        Add a perceptual frame to memory
        
        Args:
            frame: Dictionary containing interaction data
            embedding: Optional pre-computed embedding vector
            
        Returns:
            frame_id: Unique identifier for the stored frame
        """
        # Ensure frame has required fields
        if 'timestamp' not in frame:
            frame['timestamp'] = time.time()
        
        if 'id' not in frame:
            frame['id'] = str(uuid.uuid4())
        
        # Add frame to memory
        self.frames.append(frame)
        
        # Add embedding if provided, otherwise use zeros
        if embedding is not None:
            self.frame_embeddings.append(embedding)
        else:
            # Placeholder embedding (will be updated when get_embedding is called)
            self.frame_embeddings.append(np.zeros(self.embedding_dim))
        
        # Update stats
        self.stats["total_frames_added"] += 1
        
        # Periodically save memory
        if self.stats["total_frames_added"] % 10 == 0:
            self._save_memory()
        
        return frame['id']
    
    def get_embedding(self, text: str, client) -> np.ndarray:
        """Get embedding vector for text using OpenAI API"""
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logging.error(f"Error getting embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def search_memory(self, query: str, limit: int = 5, client=None) -> List[Dict]:
        """
        Search memory for relevant frames using semantic similarity
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            client: OpenAI client for computing embeddings
            
        Returns:
            List of relevant frames
        """
        if not self.frames:
            return []
        
        if client is None:
            # Return most recent frames if no client is provided
            return list(self.frames)[-limit:]
        
        # Get query embedding
        query_embedding = self.get_embedding(query, client)
        
        # Calculate similarity scores
        similarities = []
        for i, embedding in enumerate(self.frame_embeddings):
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding) + 1e-10
            )
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top results
        results = []
        for i, _ in similarities[:limit]:
            if i < len(self.frames):
                results.append(self.frames[i])
        
        # Update stats
        self.stats["total_retrievals"] += 1
        
        return results
    
    def sample_for_replay(self, batch_size: int = 3, strategy: str = "random") -> List[Dict]:
        """
        Sample frames for experience replay
        
        Args:
            batch_size: Number of frames to sample
            strategy: Sampling strategy (random, recent, prioritized)
            
        Returns:
            List of sampled frames
        """
        if not self.frames:
            return []
        
        frames_list = list(self.frames)
        
        if strategy == "recent":
            # Sample most recent frames
            samples = frames_list[-batch_size:]
        elif strategy == "prioritized":
            # Prioritize frames with higher importance (if available)
            frames_with_priority = [(i, f.get('importance', 0.5)) for i, f in enumerate(frames_list)]
            frames_with_priority.sort(key=lambda x: x[1], reverse=True)
            sample_indices = [x[0] for x in frames_with_priority[:batch_size]]
            samples = [frames_list[i] for i in sample_indices]
        else:
            # Random sampling
            if len(frames_list) <= batch_size:
                samples = frames_list
            else:
                sample_indices = random.sample(range(len(frames_list)), batch_size)
                samples = [frames_list[i] for i in sample_indices]
        
        # Update stats
        self.stats["total_replays"] += 1
        
        return samples
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            "current_size": len(self.frames),
            "max_size": self.frames.maxlen,
            "total_frames_added": self.stats["total_frames_added"],
            "total_retrievals": self.stats["total_retrievals"],
            "total_replays": self.stats["total_replays"]
        }

class MetaReflectionLayer:
    """
    Meta-cognitive layer for agent self-reflection and oversight
    Provides higher-order reasoning about the agent's own thought processes
    """
    def __init__(self, model="gpt-4o", console=None):
        self.model = model
        self.console = console or Console()
        self.reflection_history = []
        self.meta_prompts = {
            "evaluate_response": (
                "You are a meta-cognitive oversight system. Evaluate the following response "
                "from the primary reasoning system. Consider:\n"
                "1. Is the reasoning sound and logical?\n"
                "2. Are there any cognitive biases present?\n"
                "3. Is the response complete and addresses all aspects of the query?\n"
                "4. Are there alternative perspectives that should be considered?\n"
                "5. Is the confidence level appropriate given the available information?\n\n"
                "Primary system query: {query}\n\n"
                "Primary system response: {response}\n\n"
                "Provide a brief meta-evaluation and any suggestions for improvement."
            ),
            "improve_response": (
                "You are a meta-cognitive improvement system. The primary reasoning system "
                "generated a response, and the meta-evaluation identified potential issues. "
                "Your task is to improve the response based on this feedback.\n\n"
                "Original query: {query}\n\n"
                "Primary response: {response}\n\n"
                "Meta-evaluation: {evaluation}\n\n"
                "Generate an improved response that addresses the issues identified."
            )
        }
    
    async def reflect(self, query: str, response: str, client) -> Dict:
        """Perform meta-reflection on a response"""
        try:
            # Format the evaluation prompt
            eval_prompt = self.meta_prompts["evaluate_response"].format(
                query=query,
                response=response
            )
            
            # Get meta-evaluation
            meta_response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a meta-cognitive oversight system that evaluates the quality of responses."},
                    {"role": "user", "content": eval_prompt}
                ]
            )
            
            evaluation = meta_response.choices[0].message.content
            
            # Record the reflection
            reflection = {
                "timestamp": time.time(),
                "query": query,
                "primary_response": response,
                "meta_evaluation": evaluation
            }
            self.reflection_history.append(reflection)
            
            return {
                "evaluation": evaluation,
                "reflection": reflection
            }
        except Exception as e:
            logger.error(f"Error in meta-reflection: {e}")
            return {
                "evaluation": f"Error in meta-reflection: {str(e)}",
                "reflection": None
            }
    
    async def improve(self, query: str, response: str, evaluation: str, client) -> str:
        """Generate an improved response based on meta-reflection"""
        try:
            # Format the improvement prompt
            improve_prompt = self.meta_prompts["improve_response"].format(
                query=query,
                response=response,
                evaluation=evaluation
            )
            
            # Get improved response
            improved_response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a meta-cognitive improvement system that enhances responses."},
                    {"role": "user", "content": improve_prompt}
                ]
            )
            
            return improved_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error improving response: {e}")
            return response  # Return original response if improvement fails

class CLIAgent:
    """
    CLI agent for conversational data analysis with meta-recursive self-awareness
    Uses a two-layer architecture:
    1. Inner model: Primary reasoning and task execution
    2. Outer model: Meta-cognitive reflection and oversight
    
    Features:
    - Perceptual memory for storing interaction frames
    - Experience replay for learning from past interactions
    - Semantic search across interaction history
    """
    def __init__(self, model="gpt-4o", meta_model=None, console=None, enable_meta=True, 
                max_memory_frames=1000, enable_experience_replay=True):
        # Primary reasoning layer
        self.model = model
        self.meta_model = meta_model or model  # Use same model for meta layer if not specified
        self.data_tools = DataAnalysisTools()
        self.modal = ModalIntegration()
        self.file_tools = FileSystemTools()
        self.code_tools = CodingTools()
        self.conversation_history = []
        self.console = console or Console()
        self.hooks = CLIAgentHooks(self.console, display_name="Data Analysis Agent")
        
        # Meta-cognitive layer
        self.enable_meta = enable_meta
        self.meta_layer = MetaReflectionLayer(model=self.meta_model, console=self.console)
        
        # Dynamic tool registry
        self.tool_registry = DynamicToolRegistry(console=self.console)
        
        # Perceptual memory system
        self.perceptual_memory = PerceptualMemory(max_frames=max_memory_frames)
        self.enable_experience_replay = enable_experience_replay
        self.replay_batch_size = 3
        self.replay_frequency = 5
        self.interaction_count = 0
        
        # Dynamic agent contexts
        self.agent_contexts = {}
        
        # Initialize dynamic agent tools
        self._init_dynamic_agent_tools()
        
    def _init_dynamic_agent_tools(self):
        """Initialize tools for dynamic agent management"""
        # Create default agents
        try:
            # Create a file agent
            registry.create_agent("file_agent", "file")
            # Create a data analysis agent
            registry.create_agent("data_analysis_agent", "data_analysis")
            
            # Create contexts for each agent
            self.agent_contexts["file_agent"] = AgentContext(agent_id="file_agent")
            self.agent_contexts["data_analysis_agent"] = AgentContext(agent_id="data_analysis_agent")
            
            self.console.print("[dim]Initialized dynamic agents: file_agent, data_analysis_agent[/dim]")
        except Exception as e:
            self.console.print(f"[bold red]Error initializing dynamic agents: {e}[/bold red]")
    
        # Register built-in tools
        self._register_builtin_tools()
        
        # Register tool templates
        self._register_tool_templates()
        
        # Define available tools for OpenAI API
        self.tools = self.tool_registry.get_openai_tools_format() + [
            # File system tools
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory with advanced filtering and sorting options",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to list files from (default: current directory)"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Glob pattern to filter files (default: *)"
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Whether to recursively list files in subdirectories (default: false)"
                            },
                            "include_hidden": {
                                "type": "boolean",
                                "description": "Whether to include hidden files (starting with .) (default: false)"
                            },
                            "sort_by": {
                                "type": "string",
                                "description": "How to sort results (name, size, modified, type) (default: name)",
                                "enum": ["name", "size", "modified", "type"]
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file with advanced options",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the file to read"
                            },
                            "max_size": {
                                "type": "integer",
                                "description": "Maximum file size in bytes (default: 1MB)"
                            },
                            "encoding": {
                                "type": "string",
                                "description": "File encoding to use (default: utf-8)"
                            },
                            "chunk_size": {
                                "type": "integer",
                                "description": "If set, read only this many bytes"
                            },
                            "line_numbers": {
                                "type": "boolean",
                                "description": "Whether to include line numbers (default: false)"
                            },
                            "syntax_highlight": {
                                "type": "boolean",
                                "description": "Whether to detect and include syntax highlighting info (default: false)"
                            }
                        },
                        "required": ["filepath"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file with advanced options",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            },
                            "overwrite": {
                                "type": "boolean",
                                "description": "Whether to overwrite the file if it exists (default: false)"
                            },
                            "append": {
                                "type": "boolean",
                                "description": "Whether to append to existing files (default: false)"
                            },
                            "encoding": {
                                "type": "string",
                                "description": "File encoding to use (default: utf-8)"
                            },
                            "create_backup": {
                                "type": "boolean",
                                "description": "Whether to create a backup of existing file (default: false)"
                            },
                            "mode": {
                                "type": "string",
                                "description": "File permissions mode (e.g., '644')"
                            }
                        },
                        "required": ["filepath", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "copy_file",
                    "description": "Copy a file from source to destination",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "description": "Path to the source file"
                            },
                            "destination": {
                                "type": "string",
                                "description": "Path to the destination file"
                            },
                            "overwrite": {
                                "type": "boolean",
                                "description": "Whether to overwrite the destination file if it exists (default: false)"
                            }
                        },
                        "required": ["source", "destination"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file or directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the file or directory to delete"
                            },
                            "recursive": {
                                "type": "boolean",
                                "description": "Whether to recursively delete directories (default: false)"
                            }
                        },
                        "required": ["filepath"]
                    }
                }
            },
            # Coding tools
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute Python code in a controlled environment with advanced options",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default: 10)"
                            },
                            "save_history": {
                                "type": "boolean",
                                "description": "Whether to save execution history (default: true)"
                            },
                            "sandbox": {
                                "type": "boolean",
                                "description": "Whether to run in a sandbox directory (default: true)"
                            },
                            "allow_imports": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Additional imports to allow for this execution"
                            },
                            "provide_inputs": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of inputs to provide to the program"
                            },
                            "environment": {
                                "type": "object",
                                "description": "Environment variables to set"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_shell",
                    "description": "Execute a shell command with advanced options",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute"
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds (default: 10)"
                            },
                            "save_history": {
                                "type": "boolean",
                                "description": "Whether to save execution history (default: true)"
                            },
                            "sandbox": {
                                "type": "boolean",
                                "description": "Whether to run in a sandbox directory (default: true)"
                            },
                            "environment": {
                                "type": "object",
                                "description": "Environment variables to set"
                            },
                            "working_dir": {
                                "type": "string",
                                "description": "Working directory for command execution"
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_code",
                    "description": "Analyze Python code for quality, complexity, and potential issues",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to analyze"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_execution_history",
                    "description": "Get the execution history of code and commands",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of history items to return (default: 10)"
                            },
                            "execution_type": {
                                "type": "string",
                                "description": "Filter by execution type (python or shell)",
                                "enum": ["python", "shell"]
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_security_level",
                    "description": "Set the security level for code execution",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "level": {
                                "type": "string",
                                "description": "Security level to set",
                                "enum": ["safe", "data_science", "standard_library", "web", "all"]
                            }
                        },
                        "required": ["level"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_interaction_history",
                    "description": "Search through past interactions using semantic search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 5)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_memory_stats",
                    "description": "Get statistics about the perceptual memory system",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_agent",
                    "description": "Create a new specialized agent for a specific task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Unique identifier for the agent"
                            },
                            "agent_type": {
                                "type": "string",
                                "description": "Type of agent to create",
                                "enum": ["file", "data_analysis"]
                            }
                        },
                        "required": ["agent_id", "agent_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_agents",
                    "description": "List all available agents",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_agent_command",
                    "description": "Execute a command on a specific agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "ID of the agent to execute the command on"
                            },
                            "command": {
                                "type": "string",
                                "description": "Command to execute on the agent"
                            }
                        },
                        "required": ["agent_id", "command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_advanced_visualization",
                    "description": "Create advanced data visualizations including correlation matrices, pairplots, and more",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "description": "Data to visualize"
                            },
                            "viz_type": {
                                "type": "string",
                                "description": "Type of visualization to create",
                                "enum": ["correlation_matrix", "pairplot", "distribution", "boxplot", "timeseries", "3d_scatter"]
                            },
                            "title": {
                                "type": "string",
                                "description": "Title for the visualization"
                            },
                            "columns": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Specific columns to include in the visualization"
                            },
                            "date_column": {
                                "type": "string",
                                "description": "Column to use as date for time series plots"
                            },
                            "value_column": {
                                "type": "string",
                                "description": "Column to use as value for time series plots"
                            },
                            "x_column": {
                                "type": "string",
                                "description": "Column to use for x-axis in 3D scatter plots"
                            },
                            "y_column": {
                                "type": "string",
                                "description": "Column to use for y-axis in 3D scatter plots"
                            },
                            "z_column": {
                                "type": "string",
                                "description": "Column to use for z-axis in 3D scatter plots"
                            },
                            "color_column": {
                                "type": "string",
                                "description": "Column to use for color in 3D scatter plots"
                            }
                        },
                        "required": ["data", "viz_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "load_csv",
                    "description": "Load a CSV file and return basic statistics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            }
                        },
                        "required": ["filepath"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plot_data",
                    "description": "Generate a plot from data and save it to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "description": "Data to plot"
                            },
                            "plot_type": {
                                "type": "string",
                                "description": "Type of plot (histogram, scatter, bar, line, heatmap)",
                                "enum": ["histogram", "scatter", "bar", "line", "heatmap"]
                            },
                            "x_column": {
                                "type": "string",
                                "description": "Column to use for x-axis"
                            },
                            "y_column": {
                                "type": "string",
                                "description": "Column to use for y-axis (for scatter, bar, line plots)"
                            },
                            "title": {
                                "type": "string",
                                "description": "Title for the plot"
                            }
                        },
                        "required": ["data", "plot_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_text",
                    "description": "Perform basic text analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to analyze"
                            }
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_modal_functions",
                    "description": "List available functions in Modal",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "call_modal_function",
                    "description": "Call a function in Modal",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "function_name": {
                                "type": "string",
                                "description": "Name of the function to call"
                            },
                            "params": {
                                "type": "object",
                                "description": "Parameters for the function"
                            }
                        },
                        "required": ["function_name", "params"]
                    }
                }
            }
        ]
        
        logger.info(f"CLI agent initialized with model {self.model}")
    
    def _register_builtin_tools(self):
        """Register built-in tools with the tool registry"""
        # Register data analysis tools
        self.tool_registry.register_tool(
            "load_csv", 
            self.data_tools.load_csv,
            "Load a CSV file and return basic statistics",
            {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the CSV file"
                    }
                },
                "required": ["filepath"]
            },
            "data_analysis"
        )
        
        self.tool_registry.register_tool(
            "plot_data", 
            self.data_tools.plot_data,
            "Generate a plot from data and save it to a file",
            {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Data to plot"
                    },
                    "plot_type": {
                        "type": "string",
                        "description": "Type of plot (histogram, scatter, bar, line, heatmap)",
                        "enum": ["histogram", "scatter", "bar", "line", "heatmap"]
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Column to use for x-axis"
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Column to use for y-axis (for scatter, bar, line plots)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the plot"
                    }
                },
                "required": ["data", "plot_type"]
            },
            "data_analysis"
        )
        
        self.tool_registry.register_tool(
            "analyze_text", 
            self.data_tools.analyze_text,
            "Perform basic text analysis",
            {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to analyze"
                    }
                },
                "required": ["text"]
            },
            "data_analysis"
        )
        
        # Register Modal tools
        self.tool_registry.register_tool(
            "list_modal_functions", 
            self.modal.list_functions,
            "List available functions in Modal",
            {
                "type": "object",
                "properties": {}
            },
            "modal"
        )
        
        self.tool_registry.register_tool(
            "call_modal_function", 
            self.modal.call_function,
            "Call a function in Modal",
            {
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function to call"
                    },
                    "params": {
                        "type": "object",
                        "description": "Parameters for the function"
                    }
                },
                "required": ["function_name", "params"]
            },
            "modal"
        )
        
        # Register visualization tools
        self.tool_registry.register_tool(
            "create_advanced_visualization", 
            self.data_tools.visualizer.create_visualization,
            "Create advanced data visualizations including correlation matrices, pairplots, and more",
            {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Data to visualize"
                    },
                    "viz_type": {
                        "type": "string",
                        "description": "Type of visualization to create",
                        "enum": ["correlation_matrix", "pairplot", "distribution", "boxplot", "timeseries", "3d_scatter"]
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the visualization"
                    },
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Specific columns to include in the visualization"
                    }
                },
                "required": ["data", "viz_type"]
            },
            "visualization"
        )
        
        # Register file system tools
        self.tool_registry.register_tool(
            "list_files", 
            self.file_tools.list_files,
            "List files in a directory with advanced filtering and sorting options",
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list files from (default: current directory)"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to filter files (default: *)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to recursively list files in subdirectories (default: false)"
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Whether to include hidden files (starting with .) (default: false)"
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "How to sort results (name, size, modified, type) (default: name)",
                        "enum": ["name", "size", "modified", "type"]
                    }
                }
            },
            "file_system"
        )
        
        self.tool_registry.register_tool(
            "read_file", 
            self.file_tools.read_file,
            "Read the contents of a file with advanced options",
            {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "max_size": {
                        "type": "integer",
                        "description": "Maximum file size in bytes (default: 1MB)"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding to use (default: utf-8)"
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "If set, read only this many bytes"
                    },
                    "line_numbers": {
                        "type": "boolean",
                        "description": "Whether to include line numbers (default: false)"
                    },
                    "syntax_highlight": {
                        "type": "boolean",
                        "description": "Whether to detect and include syntax highlighting info (default: false)"
                    }
                },
                "required": ["filepath"]
            },
            "file_system"
        )
        
        self.tool_registry.register_tool(
            "write_file", 
            self.file_tools.write_file,
            "Write content to a file with advanced options",
            {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Whether to overwrite the file if it exists (default: false)"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Whether to append to existing files (default: false)"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "File encoding to use (default: utf-8)"
                    },
                    "create_backup": {
                        "type": "boolean",
                        "description": "Whether to create a backup of existing file (default: false)"
                    },
                    "mode": {
                        "type": "string",
                        "description": "File permissions mode (e.g., '644')"
                    }
                },
                "required": ["filepath", "content"]
            },
            "file_system"
        )
        
        self.tool_registry.register_tool(
            "copy_file", 
            self.file_tools.copy_file,
            "Copy a file from source to destination",
            {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Path to the source file"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Path to the destination file"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Whether to overwrite the destination file if it exists (default: false)"
                    }
                },
                "required": ["source", "destination"]
            },
            "file_system"
        )
        
        self.tool_registry.register_tool(
            "delete_file", 
            self.file_tools.delete_file,
            "Delete a file or directory",
            {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Path to the file or directory to delete"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to recursively delete directories (default: false)"
                    }
                },
                "required": ["filepath"]
            },
            "file_system"
        )
        
        # Register coding tools
        self.tool_registry.register_tool(
            "execute_python", 
            self.code_tools.execute_python,
            "Execute Python code in a controlled environment with advanced options",
            {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 10)"
                    },
                    "save_history": {
                        "type": "boolean",
                        "description": "Whether to save execution history (default: true)"
                    },
                    "sandbox": {
                        "type": "boolean",
                        "description": "Whether to run in a sandbox directory (default: true)"
                    },
                    "allow_imports": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Additional imports to allow for this execution"
                    },
                    "provide_inputs": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of inputs to provide to the program"
                    },
                    "environment": {
                        "type": "object",
                        "description": "Environment variables to set"
                    }
                },
                "required": ["code"]
            },
            "coding"
        )
        
        self.tool_registry.register_tool(
            "execute_shell", 
            self.code_tools.execute_shell,
            "Execute a shell command with advanced options",
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 10)"
                    },
                    "save_history": {
                        "type": "boolean",
                        "description": "Whether to save execution history (default: true)"
                    },
                    "sandbox": {
                        "type": "boolean",
                        "description": "Whether to run in a sandbox directory (default: true)"
                    },
                    "environment": {
                        "type": "object",
                        "description": "Environment variables to set"
                    },
                    "working_dir": {
                        "type": "string",
                        "description": "Working directory for command execution"
                    }
                },
                "required": ["command"]
            },
            "coding"
        )
        
        self.tool_registry.register_tool(
            "analyze_code", 
            self.code_tools.analyze_code,
            "Analyze Python code for quality, complexity, and potential issues",
            {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to analyze"
                    }
                },
                "required": ["code"]
            },
            "coding"
        )
        
        self.tool_registry.register_tool(
            "get_execution_history", 
            self.code_tools.get_execution_history,
            "Get the execution history of code and commands",
            {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of history items to return (default: 10)"
                    },
                    "execution_type": {
                        "type": "string",
                        "description": "Filter by execution type (python or shell)",
                        "enum": ["python", "shell"]
                    }
                }
            },
            "coding"
        )
        
        self.tool_registry.register_tool(
            "set_security_level", 
            self.code_tools.set_security_level,
            "Set the security level for code execution",
            {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "description": "Security level to set",
                        "enum": ["safe", "data_science", "standard_library", "web", "all"]
                    }
                },
                "required": ["level"]
            },
            "coding"
        )
        
        # Register dynamic agent tools
        self.tool_registry.register_tool(
            "create_agent", 
            self._create_agent,
            "Create a new specialized agent for a specific task",
            {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "Unique identifier for the agent"
                    },
                    "agent_type": {
                        "type": "string",
                        "description": "Type of agent to create",
                        "enum": ["file", "data_analysis"]
                    }
                },
                "required": ["agent_id", "agent_type"]
            },
            "agents"
        )
        
        self.tool_registry.register_tool(
            "list_agents", 
            self._list_agents,
            "List all available agents",
            {
                "type": "object",
                "properties": {}
            },
            "agents"
        )
        
        # Register tool management tools
        self.tool_registry.register_tool(
            "register_tool", 
            self._register_new_tool,
            "Register a new tool that the agent can use",
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the tool"
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code defining the function"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the tool does"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "OpenAI-compatible parameters schema"
                    },
                    "category": {
                        "type": "string",
                        "description": "Category for organizing tools (default: custom)"
                    }
                },
                "required": ["name", "code", "description", "parameters"]
            },
            "tool_management"
        )
        
        self.tool_registry.register_tool(
            "unregister_tool", 
            self._unregister_tool,
            "Remove a tool from the registry",
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the tool to remove"
                    }
                },
                "required": ["name"]
            },
            "tool_management"
        )
        
        self.tool_registry.register_tool(
            "list_tools", 
            self._list_tools,
            "List all available tools or tools in a specific category",
            {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category to filter tools (optional)"
                    }
                }
            },
            "tool_management"
        )
    
    def _register_tool_templates(self):
        """Register tool templates from the tool_templates module"""
        try:
            from tool_templates import get_all_tool_templates
            
            templates = get_all_tool_templates()
            registered_count = 0
            
            for template in templates:
                # Skip if tool already exists
                if self.tool_registry.get_tool(template["name"]):
                    continue
                    
                # Register the tool
                result = self.tool_registry.load_tool_from_code(
                    template["name"],
                    template["code"],
                    template["description"],
                    template["parameters"],
                    template["category"]
                )
                
                if result["success"]:
                    registered_count += 1
            
            if registered_count > 0:
                self.console.print(f"[dim]Registered {registered_count} tool templates[/dim]")
                
        except ImportError:
            self.console.print("[yellow]Tool templates module not found. Some advanced tools will not be available.[/yellow]")
        except Exception as e:
            self.console.print(f"[yellow]Error registering tool templates: {e}[/yellow]")

    def _register_new_tool(self, name: str, code: str, description: str, 
                          parameters: Dict[str, Any], category: str = "custom") -> Dict:
        """Register a new tool from code"""
        return self.tool_registry.load_tool_from_code(
            name, code, description, parameters, category
        )
    
    def _unregister_tool(self, name: str) -> Dict:
        """Remove a tool from the registry"""
        return self.tool_registry.unregister_tool(name)
    
    def _list_tools(self, category: str = None) -> Dict:
        """List all available tools or tools in a specific category"""
        tools = self.tool_registry.list_tools(category)
        categories = self.tool_registry.list_categories()
        
        return {
            "success": True,
            "message": f"Found {len(tools)} tools" + (f" in category '{category}'" if category else ""),
            "data": {
                "tools": tools,
                "categories": categories,
                "tool_count": len(tools)
            }
        }
    
    def _handle_tool_call(self, name: str, arguments: Dict) -> Dict:
        """Handle tool calls from the agent"""
        logger.info(f"Handling tool call: {name} with arguments {arguments}")
        
        # Notify hooks that a tool is starting
        self.hooks.on_tool_start("CLI Agent", name, arguments)
        
        result = None
        try:
            # Check if the tool is in the registry
            if self.tool_registry.get_tool(name):
                result = self.tool_registry.execute_tool(name, arguments)
            # Handle dynamic agent tools
            elif name == "execute_agent_command":
                result = asyncio.run(self._execute_agent_command(**arguments))
            else:
                result = {
                    "success": False,
                    "message": f"Unknown tool: {name}",
                    "data": None
                }
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            self.hooks.on_error("CLI Agent", e)
            result = {
                "success": False,
                "message": f"Error executing tool {name}: {str(e)}",
                "data": None
            }
        
        # Notify hooks that the tool has completed
        self.hooks.on_tool_end("CLI Agent", name, result)
        
        return result
    
    def _create_agent(self, agent_id: str, agent_type: str) -> Dict:
        """Create a new agent of the specified type"""
        try:
            # Generate a unique ID if not provided
            if not agent_id:
                agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
            
            # Create the agent
            agent = registry.create_agent(agent_id, agent_type)
            
            # Create a context for the agent
            self.agent_contexts[agent_id] = AgentContext(agent_id=agent_id)
            
            return {
                "success": True,
                "message": f"Created {agent_type} agent with ID: {agent_id}",
                "data": {
                    "agent_id": agent_id,
                    "agent_type": agent_type
                }
            }
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return {
                "success": False,
                "message": f"Error creating agent: {str(e)}",
                "data": None
            }
    
    def _list_agents(self) -> Dict:
        """List all available agents"""
        try:
            agents = registry.list_agents()
            agent_types = registry.list_agent_types()
            
            return {
                "success": True,
                "message": f"Found {len(agents)} agents",
                "data": {
                    "agents": agents,
                    "available_types": agent_types
                }
            }
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return {
                "success": False,
                "message": f"Error listing agents: {str(e)}",
                "data": None
            }
    
    async def _execute_agent_command(self, agent_id: str, command: str) -> Dict:
        """Execute a command on a specific agent"""
        try:
            # Get the agent context
            context = self.agent_contexts.get(agent_id)
            if not context:
                # Create a new context if it doesn't exist
                context = AgentContext(agent_id=agent_id)
                self.agent_contexts[agent_id] = context
            
            # Execute the command
            result = await execute_agent_command(agent_id, command, context)
            
            return {
                "success": result.get("success", False),
                "message": f"Executed command on agent {agent_id}",
                "data": result
            }
        except Exception as e:
            logger.error(f"Error executing agent command: {e}")
            return {
                "success": False,
                "message": f"Error executing agent command: {str(e)}",
                "data": None
            }
    
    def _create_interaction_frame(self, message: str, response: str, 
                                tool_calls: List = None, meta_reflection: Dict = None) -> Dict:
        """
        Create a perceptual frame from an interaction
        
        Args:
            message: User message
            response: Agent response
            tool_calls: List of tool calls made during the interaction
            meta_reflection: Meta-reflection data if available
            
        Returns:
            Frame dictionary
        """
        frame = {
            "id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "interaction": {
                "user_message": message,
                "agent_response": response
            },
            "context": {
                "conversation_length": len(self.conversation_history) // 2,  # Approximate turn count
                "model": self.model
            }
        }
        
        # Add tool calls if available
        if tool_calls:
            frame["tool_calls"] = tool_calls
        
        # Add meta-reflection if available
        if meta_reflection:
            frame["meta_reflection"] = meta_reflection
        
        return frame
    
    def _store_interaction(self, message: str, response: str, 
                         tool_calls: List = None, meta_reflection: Dict = None, client=None) -> str:
        """
        Store an interaction in perceptual memory
        
        Args:
            message: User message
            response: Agent response
            tool_calls: List of tool calls made
            meta_reflection: Meta-reflection data
            client: OpenAI client for computing embeddings
            
        Returns:
            Frame ID
        """
        # Create frame
        frame = self._create_interaction_frame(message, response, tool_calls, meta_reflection)
        
        # Compute embedding if client is available
        embedding = None
        if client:
            # Create a combined text representation for embedding
            text_for_embedding = f"User: {message}\nAgent: {response}"
            embedding = self.perceptual_memory.get_embedding(text_for_embedding, client)
        
        # Store frame
        frame_id = self.perceptual_memory.add_frame(frame, embedding)
        
        return frame_id
    
    async def _perform_experience_replay(self, client) -> Dict:
        """
        Perform experience replay to learn from past interactions
        
        Args:
            client: OpenAI client
            
        Returns:
            Results of the replay
        """
        # Sample frames for replay
        frames = self.perceptual_memory.sample_for_replay(batch_size=self.replay_batch_size)
        
        if not frames:
            return {"success": False, "message": "No frames available for replay"}
        
        # Format frames for reflection
        replay_prompt = "Review these past interactions and provide insights on how to improve:\n\n"
        
        for i, frame in enumerate(frames):
            interaction = frame.get("interaction", {})
            user_msg = interaction.get("user_message", "")
            agent_resp = interaction.get("agent_response", "")
            
            replay_prompt += f"Interaction {i+1}:\n"
            replay_prompt += f"User: {user_msg}\n"
            replay_prompt += f"Agent: {agent_resp}\n\n"
        
        replay_prompt += "Based on these interactions, what patterns do you notice? How could responses be improved? What strategies worked well?"
        
        try:
            # Get insights from model
            replay_response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are analyzing past interactions to improve future responses."},
                    {"role": "user", "content": replay_prompt}
                ]
            )
            
            insights = replay_response.choices[0].message.content
            
            # Store the insights in a special frame
            replay_frame = {
                "id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "type": "experience_replay",
                "sampled_frames": [frame["id"] for frame in frames],
                "insights": insights
            }
            
            self.perceptual_memory.add_frame(replay_frame)
            
            return {
                "success": True,
                "message": "Experience replay completed successfully",
                "insights": insights,
                "frames_analyzed": len(frames)
            }
        except Exception as e:
            logging.error(f"Error in experience replay: {e}")
            return {
                "success": False,
                "message": f"Error in experience replay: {str(e)}"
            }
    
    async def chat(self, message: str) -> str:
        """
        Chat with the agent using a two-layer architecture:
        1. Inner model generates the primary response
        2. Outer model performs meta-reflection and improves the response
        
        Also stores interaction frames in perceptual memory and performs
        experience replay for continuous learning.
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Notify hooks that the agent is starting
        self.hooks.on_start("CLI Agent")
        
        # Increment interaction counter
        self.interaction_count += 1
        
        # Run the primary reasoning layer (inner model)
        with self.console.status("[bold green]Thinking..."):
            try:
                # Check if the message contains any code execution requests
                if "```python" in message or "```py" in message or "execute this code" in message.lower():
                    self.console.print("[bold yellow]Warning: Code execution detected in request. Proceeding with caution.[/bold yellow]")
                
                # Get the latest tools (in case new ones were registered)
                current_tools = self.tool_registry.get_openai_tools_format() + self.tools
                
                # Primary reasoning with inner model
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    tools=current_tools,
                    tool_choice="auto"
                )
                
                # Get the response
                assistant_message = response.choices[0].message
                
                # Process tool calls if any
                if assistant_message.tool_calls:
                    # Add the assistant's message to the conversation
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            } for tool_call in assistant_message.tool_calls
                        ]
                    })
                    
                    # Process each tool call
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # Execute the function - don't use nested console.status
                        self.console.print(f"[bold blue]Running tool: {function_name}...[/bold blue]")
                        function_response = self._handle_tool_call(function_name, function_args)
                        
                        # Add the function response to the conversation
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(function_response)
                        })
                    
                    # Get the final response after tool calls
                    self.console.print("[bold green]Processing results...[/bold green]")
                    second_response = client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history
                    )
                    
                    primary_response = second_response.choices[0].message.content
                    
                    # Meta-cognitive reflection (outer model) if enabled
                    final_response = primary_response
                    meta_reflection = None
                    
                    if self.enable_meta:
                        self.console.print("[bold cyan]Performing meta-reflection...[/bold cyan]")
                        # Get meta-evaluation
                        reflection = await self.meta_layer.reflect(message, primary_response, client)
                        meta_reflection = reflection
                        
                        # Improve response based on meta-reflection
                        if reflection and "evaluation" in reflection:
                            improved_response = await self.meta_layer.improve(
                                message, 
                                primary_response, 
                                reflection["evaluation"],
                                client
                            )
                            final_response = improved_response
                            
                            # Add meta-reflection to conversation history as a system note
                            self.conversation_history.append({
                                "role": "system",
                                "content": f"Meta-reflection: {reflection['evaluation']}"
                            })
                        
                        # Add the final response to the conversation
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": final_response
                        })
                        
                        # Store interaction in perceptual memory
                        tool_calls_data = [
                            {
                                "name": tool_call.function.name,
                                "arguments": json.loads(tool_call.function.arguments)
                            } for tool_call in assistant_message.tool_calls
                        ]
                        
                        self._store_interaction(
                            message=message,
                            response=final_response,
                            tool_calls=tool_calls_data,
                            meta_reflection=meta_reflection,
                            client=client
                        )
                        
                        # Perform experience replay if enabled and it's time
                        if (self.enable_experience_replay and 
                            self.interaction_count % self.replay_frequency == 0 and
                            self.interaction_count > 1):
                            self.console.print("[bold magenta]Performing experience replay...[/bold magenta]")
                            replay_result = await self._perform_experience_replay(client)
                            if replay_result["success"] and self.console:
                                self.console.print(
                                    f"[dim][Experience replay: analyzed {replay_result.get('frames_analyzed', 0)} past interactions][/dim]",
                                    style="dim"
                                )
                        
                        return final_response
                else:
                    # No tool calls, just get the primary response
                    primary_response = assistant_message.content
                    
                    # Meta-cognitive reflection (outer model) if enabled
                    final_response = primary_response
                    meta_reflection = None
                    
                    if self.enable_meta:
                        self.console.print("[bold cyan]Performing meta-reflection...[/bold cyan]")
                        # Get meta-evaluation
                        reflection = await self.meta_layer.reflect(message, primary_response, client)
                        meta_reflection = reflection
                        
                        # Improve response based on meta-reflection
                        if reflection and "evaluation" in reflection:
                            improved_response = await self.meta_layer.improve(
                                message, 
                                primary_response, 
                                reflection["evaluation"],
                                client
                            )
                            final_response = improved_response
                            
                            # Add meta-reflection to conversation history as a system note
                            self.conversation_history.append({
                                "role": "system",
                                "content": f"Meta-reflection: {reflection['evaluation']}"
                            })
                    
                    # Add the response to the conversation history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": final_response
                    })
                    
                    # Store interaction in perceptual memory
                    self._store_interaction(
                        message=message,
                        response=final_response,
                        meta_reflection=meta_reflection,
                        client=client
                    )
                    
                    # Perform experience replay if enabled and it's time
                    if (self.enable_experience_replay and 
                        self.interaction_count % self.replay_frequency == 0 and
                        self.interaction_count > 1):
                        self.console.print("[bold magenta]Performing experience replay...[/bold magenta]")
                        replay_result = await self._perform_experience_replay(client)
                        if replay_result["success"] and self.console:
                            self.console.print(
                                f"[dim][Experience replay: analyzed {replay_result.get('frames_analyzed', 0)} past interactions][/dim]",
                                style="dim"
                            )
                    
                    # Notify hooks that the agent has completed
                    self.hooks.on_end("CLI Agent", final_response)
                    return final_response
                    
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                self.hooks.on_error("CLI Agent", e)
                return f"Error: {str(e)}"

    def search_interaction_history(self, query: str, limit: int = 5) -> Dict:
        """
        Search through past interactions using semantic search
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            Dictionary with search results
        """
        try:
            # Search memory
            results = self.perceptual_memory.search_memory(query, limit, client)
            
            # Format results for display
            formatted_results = []
            for frame in results:
                interaction = frame.get("interaction", {})
                formatted_results.append({
                    "id": frame.get("id", "unknown"),
                    "timestamp": frame.get("datetime", "unknown"),
                    "user_message": interaction.get("user_message", ""),
                    "agent_response": interaction.get("agent_response", "")
                })
            
            return {
                "success": True,
                "message": f"Found {len(formatted_results)} relevant interactions",
                "data": formatted_results
            }
        except Exception as e:
            logging.error(f"Error searching interaction history: {e}")
            return {
                "success": False,
                "message": f"Error searching interaction history: {str(e)}",
                "data": []
            }
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about the perceptual memory system"""
        return self.perceptual_memory.get_memory_stats()

def main():
    """Main function for the CLI agent"""
    parser = argparse.ArgumentParser(description="CLI Agent for Data Analysis")
    parser.add_argument("--model", default="gpt-4o", help="Model to use for the agent")
    parser.add_argument("--meta-model", default=None, help="Model to use for meta-reflection (defaults to same as primary model)")
    parser.add_argument("--disable-meta", action="store_true", help="Disable meta-cognitive reflection layer")
    parser.add_argument("--disable-replay", action="store_true", help="Disable experience replay")
    parser.add_argument("--memory-size", type=int, default=1000, help="Maximum number of interaction frames to store in memory")
    parser.add_argument("--trace", action="store_true", help="Enable tracing for debugging")
    parser.add_argument("--debug", action="store_true", help="Show debug information including agent hooks")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation (0.0-2.0)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens for model response")
    parser.add_argument("--list-agents", action="store_true", help="List available dynamic agents and exit")
    parser.add_argument("--list-tools", action="store_true", help="List available tools and exit")
    parser.add_argument("--list-categories", action="store_true", help="List available tool categories and exit")
    parser.add_argument("--list-templates", action="store_true", help="List available tool templates and exit")
    args = parser.parse_args()
    
    # Enable tracing if requested
    if args.trace:
        openai.debug.trace.enable()
    
    # Create console
    console = Console()
    
    # Create the agent with meta-cognitive layer and perceptual memory
    agent = CLIAgent(
        model=args.model, 
        meta_model=args.meta_model,
        console=console,
        enable_meta=not args.disable_meta,
        max_memory_frames=args.memory_size,
        enable_experience_replay=not args.disable_replay
    )
    
    
    # If --list-agents flag is provided, list agents and exit
    if args.list_agents:
        agents = registry.list_agents()
        agent_types = registry.list_agent_types()
        
        console.print(Panel.fit(
            f"[bold]Available Agent Types:[/bold] {', '.join(agent_types)}\n\n"
            f"[bold]Registered Agents:[/bold]",
            title="Dynamic Agents",
            border_style="green"
        ))
        
        if agents:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Agent ID")
            table.add_column("Type")
            
            for agent_info in agents:
                table.add_row(agent_info["id"], agent_info["type"])
            
            console.print(table)
        else:
            console.print("[yellow]No agents registered yet.[/yellow]")
        
        return
    
    # If --list-tools flag is provided, list tools and exit
    if args.list_tools:
        tools_result = agent._list_tools()
        
        if tools_result["success"]:
            tools = tools_result["data"]["tools"]
            categories = tools_result["data"]["categories"]
            
            console.print(Panel.fit(
                f"[bold]Available Tool Categories:[/bold] {', '.join(categories)}\n\n"
                f"[bold]Registered Tools:[/bold] {len(tools)} total",
                title="Available Tools",
                border_style="green"
            ))
            
            # Group tools by category
            tools_by_category = {}
            for category in categories:
                tools_by_category[category] = agent.tool_registry.list_tools(category)
            
            # Display tools by category
            for category, category_tools in tools_by_category.items():
                if category_tools:
                    console.print(f"\n[bold cyan]{category.upper()}[/bold cyan] ({len(category_tools)} tools)")
                    
                    table = Table(show_header=True, header_style="bold magenta", box=None)
                    table.add_column("Tool Name")
                    table.add_column("Description")
                    
                    for tool_name in category_tools:
                        description = agent.tool_registry.get_tool_description(tool_name) or ""
                        # Truncate description if too long
                        if len(description) > 60:
                            description = description[:57] + "..."
                        table.add_row(tool_name, description)
                    
                    console.print(table)
        else:
            console.print(f"[bold red]Error listing tools: {tools_result['message']}[/bold red]")
        
        return
    
    # If --list-categories flag is provided, list categories and exit
    if args.list_categories:
        categories = agent.tool_registry.list_categories()
        
        console.print(Panel.fit(
            f"[bold]Available Tool Categories:[/bold]\n",
            title="Tool Categories",
            border_style="green"
        ))
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Category")
        table.add_column("Tool Count")
        
        for category in categories:
            tool_count = len(agent.tool_registry.list_tools(category))
            table.add_row(category, str(tool_count))
        
        console.print(table)
        return
    
    # If --list-templates flag is provided, list tool templates and exit
    if args.list_templates:
        try:
            from tool_templates import get_all_tool_templates
            templates = get_all_tool_templates()
            
            console.print(Panel.fit(
                f"[bold]Available Tool Templates:[/bold] {len(templates)} templates\n",
                title="Tool Templates",
                border_style="green"
            ))
            
            # Group templates by category
            templates_by_category = {}
            for template in templates:
                category = template.get("category", "misc")
                if category not in templates_by_category:
                    templates_by_category[category] = []
                templates_by_category[category].append(template)
            
            # Display templates by category
            for category, category_templates in templates_by_category.items():
                console.print(f"\n[bold cyan]{category.upper()}[/bold cyan] ({len(category_templates)} templates)")
                
                table = Table(show_header=True, header_style="bold magenta", box=None)
                table.add_column("Name")
                table.add_column("Description")
                
                for template in category_templates:
                    description = template.get("description", "")
                    # Truncate description if too long
                    if len(description) > 60:
                        description = description[:57] + "..."
                    table.add_row(template["name"], description)
                
                console.print(table)
        except ImportError:
            console.print("[bold red]Tool templates module not found.[/bold red]")
        except Exception as e:
            console.print(f"[bold red]Error listing tool templates: {str(e)}[/bold red]")
        
        return
    
    # Configure OpenAI client with additional parameters
    client.temperature = args.temperature
    client.max_tokens = args.max_tokens
    
    # Welcome message
    meta_status = "enabled" if not args.disable_meta else "disabled"
    replay_status = "enabled" if not args.disable_replay else "disabled"
    memory_size = args.memory_size
    
    console.print(Panel.fit(
        "[bold blue]Welcome to the Data Analysis CLI Agent![/bold blue]\n"
        "You can chat with me about data analysis tasks, and I'll help you analyze data, "
        "create visualizations, and more.\n"
        f"[bold cyan]Meta-cognitive reflection:[/bold cyan] {meta_status}\n"
        f"[bold cyan]Experience replay:[/bold cyan] {replay_status} (memory: {memory_size} frames)\n"
        "[bold cyan]Dynamic Agents:[/bold cyan] You can create and use specialized agents for specific tasks.\n"
        "Type [bold green]'exit'[/bold green] to quit.",
        title="Data Analysis Assistant",
        border_style="blue"
    ))
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold green]You")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("[bold blue]Goodbye![/bold blue]")
                break
            
            # Process the input
            response = asyncio.run(agent.chat(user_input))
            
            # Display the response
            console.print("\n[bold blue]Assistant[/bold blue]")
            console.print(Markdown(response))
            
            # Show debug summary if requested
            if args.debug:
                agent.hooks.display_summary()
            
        except KeyboardInterrupt:
            console.print("\n[bold blue]Goodbye![/bold blue]")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
