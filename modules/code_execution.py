#!/usr/bin/env python3
"""
Code Execution Module - Safe code execution environment

Provides tools for an AI agent to safely execute code:
- Sandboxed code execution with resource limits
- Output and error capture
- Performance monitoring
- Support for multiple languages

This module implements safety measures including:
- Resource limits (time, memory, CPU)
- Module access restrictions
- Execution isolation
- Proper error handling
"""

import os
import sys
import ast
import time
import tempfile
import subprocess
import resource
import threading
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# === RESOURCE MANAGEMENT ===
class ResourceLimiter:
    """
    Manages resource limits for code execution
    """
    def __init__(self, 
                time_limit: int = 10,       # seconds
                memory_limit: int = 1024,   # MB
                cpu_limit: int = 1):        # cores
        """
        Initialize with resource limits
        
        Args:
            time_limit: Maximum execution time in seconds
            memory_limit: Maximum memory usage in MB
            cpu_limit: Maximum CPU cores to use
        """
        self.time_limit = time_limit
        self.memory_limit = memory_limit 
        self.cpu_limit = cpu_limit
        
    def set_process_limits(self):
        """
        Set resource limits for the current process
        Must be called in the target process/thread
        """
        try:
            # Set memory limit
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            # Convert MB to bytes
            memory_bytes = self.memory_limit * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, hard))
            
            # Set CPU time limit
            soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
            resource.setrlimit(resource.RLIMIT_CPU, (self.time_limit, hard))
            
            # Set max subprocesses
            soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
            resource.setrlimit(resource.RLIMIT_NPROC, (10, hard))
        except Exception as e:
            print(f"Warning: Couldn't set resource limits: {str(e)}")

# === SAFE CODE ANALYSIS ===
class CodeAnalyzer:
    """
    Analyzes code for security and safety issues
    """
    def __init__(self):
        """Initialize with default settings"""
        # Modules that should never be imported
        self.blocked_modules = {
            'os.system', 'subprocess', 'sys.modules', 
            'importlib', '__import__', 'eval', 'exec',
            'pty', 'socket', 'pickle', 'ctypes'
        }
        
        # Dangerous function calls
        self.dangerous_calls = {
            'open': {'mode': {'w', 'a', 'x', 'r+', 'w+', 'a+', 'x+'}},
            'write': {},
            'delete': {},
            'remove': {}
        }
        
    def analyze_code(self, code: str) -> Dict:
        """
        Analyze code for safety issues
        
        Args:
            code: Python code to analyze
            
        Returns:
            Dictionary with analysis results
        """
        try:
            issues = []
            
            # Parse the code
            tree = ast.parse(code)
            
            # Find imports of blocked modules
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name in self.blocked_modules:
                            issues.append(f"Blocked module import: {name.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.blocked_modules:
                        issues.append(f"Blocked module import: {node.module}")
                    for name in node.names:
                        full_name = f"{node.module}.{name.name}"
                        if full_name in self.blocked_modules:
                            issues.append(f"Blocked module import: {full_name}")
                            
                # Look for calls to eval or exec
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['exec', 'eval']:
                            issues.append(f"Dangerous call to {node.func.id}")
                    elif isinstance(node.func, ast.Attribute):
                        # Check for attribute calls like os.system
                        if hasattr(node.func, 'attr') and hasattr(node.func.value, 'id'):
                            full_name = f"{node.func.value.id}.{node.func.attr}"
                            if full_name in self.blocked_modules:
                                issues.append(f"Dangerous call to {full_name}")
                        
                        # Check for calls to open with write permissions
                        if hasattr(node.func, 'attr') and node.func.attr == 'open':
                            # Check args and kwargs for mode
                            if len(node.args) >= 2:
                                if isinstance(node.args[1], ast.Str):
                                    mode = node.args[1].s
                                    if any(w in mode for w in self.dangerous_calls['open']['mode']):
                                        issues.append(f"File opened with write permissions: mode={mode}")
                            
                            # Check for keyword args
                            for kw in node.keywords:
                                if kw.arg == 'mode' and isinstance(kw.value, ast.Str):
                                    mode = kw.value.s
                                    if any(w in mode for w in self.dangerous_calls['open']['mode']):
                                        issues.append(f"File opened with write permissions: mode={mode}")
                                        
            return {
                "status": "success",
                "issues": issues,
                "safe": len(issues) == 0
            }
            
        except SyntaxError as e:
            return {
                "status": "error",
                "message": f"Syntax error: {str(e)}",
                "safe": False
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error analyzing code: {str(e)}",
                "safe": False
            }

# === CODE EXECUTION ===
class CodeExecutor:
    """
    Manages safe code execution
    """
    def __init__(self, 
                time_limit: int = 10,
                memory_limit: int = 1024,
                cpu_limit: int = 1,
                safe_mode: bool = True):
        """
        Initialize the code executor
        
        Args:
            time_limit: Maximum execution time in seconds
            memory_limit: Maximum memory usage in MB
            cpu_limit: Maximum CPU cores to use
            safe_mode: Whether to enforce safety checks
        """
        self.resource_limiter = ResourceLimiter(time_limit, memory_limit, cpu_limit)
        self.code_analyzer = CodeAnalyzer()
        self.safe_mode = safe_mode
        
    def execute_code(self, code: str, language: str = "python") -> Dict:
        """
        Execute code safely
        
        Args:
            code: Code to execute
            language: Programming language (python, javascript, etc.)
            
        Returns:
            Dictionary with execution results
        """
        if language.lower() != "python":
            return self._execute_other_language(code, language)
            
        # For Python, we check safety and have multiple execution methods
        if self.safe_mode:
            # Analyze the code for safety
            analysis = self.code_analyzer.analyze_code(code)
            
            if not analysis["safe"]:
                return {
                    "status": "error",
                    "message": "Code contains unsafe operations",
                    "details": analysis["issues"]
                }
                
        # Execute using subprocess for best isolation
        return self._execute_subprocess(code)
        
    def _execute_subprocess(self, code: str) -> Dict:
        """
        Execute code in a subprocess
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
                temp_path = temp.name
                temp.write(code.encode('utf-8'))
                
            start_time = time.time()
            
            # Run the code in a subprocess
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.resource_limiter.time_limit
            )
            
            execution_time = time.time() - start_time
            
            # Clean up
            os.unlink(temp_path)
            
            # Process the results
            if result.returncode == 0:
                return {
                    "status": "success",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "execution_time": execution_time
                }
            else:
                return {
                    "status": "error",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "execution_time": execution_time,
                    "returncode": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return {
                "status": "timeout",
                "message": f"Execution timed out after {self.resource_limiter.time_limit} seconds"
            }
        except Exception as e:
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return {
                "status": "error",
                "message": f"Error executing code: {str(e)}"
            }
            
    def _execute_other_language(self, code: str, language: str) -> Dict:
        """
        Execute code in languages other than Python
        
        Args:
            code: Code to execute
            language: Programming language
            
        Returns:
            Dictionary with execution results
        """
        # Map of languages to their interpreters/compilers
        language_map = {
            "javascript": ["node", "-e"],
            "nodejs": ["node", "-e"],
            "js": ["node", "-e"],
            "bash": ["bash", "-c"],
            "sh": ["sh", "-c"],
            "ruby": ["ruby", "-e"],
            "perl": ["perl", "-e"],
            "php": ["php", "-r"]
        }
        
        if language.lower() not in language_map:
            return {
                "status": "error",
                "message": f"Unsupported language: {language}"
            }
            
        try:
            interpreter, exec_flag = language_map[language.lower()]
            
            # Check if the interpreter is available
            try:
                subprocess.run(["which", interpreter], 
                              check=True, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
            except subprocess.CalledProcessError:
                return {
                    "status": "error",
                    "message": f"{interpreter} not found. Please install it to execute {language} code."
                }
                
            start_time = time.time()
            
            # Run the code
            result = subprocess.run(
                [interpreter, exec_flag, code],
                capture_output=True,
                text=True,
                timeout=self.resource_limiter.time_limit
            )
            
            execution_time = time.time() - start_time
            
            # Process the results
            if result.returncode == 0:
                return {
                    "status": "success",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "execution_time": execution_time
                }
            else:
                return {
                    "status": "error",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "execution_time": execution_time,
                    "returncode": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "message": f"Execution timed out after {self.resource_limiter.time_limit} seconds"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing code: {str(e)}"
            }
            
    def run_script(self, file_path: str) -> Dict:
        """
        Run a script file
        
        Args:
            file_path: Path to the script file
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "status": "error",
                    "message": f"File not found: {file_path}"
                }
                
            # Determine language from file extension
            _, ext = os.path.splitext(file_path)
            language_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.rb': 'ruby',
                '.sh': 'bash',
                '.pl': 'perl',
                '.php': 'php'
            }
            
            language = language_map.get(ext.lower(), 'unknown')
            
            if language == 'unknown':
                return {
                    "status": "error",
                    "message": f"Unsupported file type: {ext}"
                }
                
            # Read the file
            with open(file_path, 'r') as f:
                code = f.read()
                
            # Execute the code
            return self.execute_code(code, language)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error running script: {str(e)}"
            }
            
    def test_function(self, function: Callable, test_inputs: List[Dict]) -> Dict:
        """
        Test a function with multiple inputs
        
        Args:
            function: Function to test
            test_inputs: List of dictionaries with args and kwargs
            
        Returns:
            Dictionary with test results
        """
        results = []
        
        for i, test_input in enumerate(test_inputs):
            try:
                args = test_input.get('args', [])
                kwargs = test_input.get('kwargs', {})
                
                start_time = time.time()
                output = function(*args, **kwargs)
                execution_time = time.time() - start_time
                
                results.append({
                    "test_index": i,
                    "status": "success",
                    "output": output,
                    "execution_time": execution_time
                })
            except Exception as e:
                results.append({
                    "test_index": i,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                
        return {
            "status": "success",
            "test_count": len(test_inputs),
            "passed": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "results": results
        }
        
    def evaluate_performance(self, code: str, iterations: int = 5) -> Dict:
        """
        Benchmark code performance
        
        Args:
            code: Code to benchmark
            iterations: Number of iterations
            
        Returns:
            Dictionary with benchmark results
        """
        if iterations < 1:
            return {
                "status": "error",
                "message": "Iterations must be at least 1"
            }
            
        execution_times = []
        
        for i in range(iterations):
            result = self.execute_code(code)
            
            if result["status"] != "success":
                return {
                    "status": "error",
                    "message": f"Execution failed on iteration {i+1}",
                    "details": result
                }
                
            execution_times.append(result["execution_time"])
            
        return {
            "status": "success",
            "iterations": iterations,
            "avg_time": sum(execution_times) / len(execution_times),
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "times": execution_times
        }

# === UTILITY FUNCTIONS ===
def execute_code(code: str, language: str = "python",
               time_limit: int = 10, memory_limit: int = 1024) -> Dict:
    """
    Execute code safely with resource limits
    
    Args:
        code: Code to execute
        language: Programming language (python, javascript, etc.)
        time_limit: Maximum execution time in seconds
        memory_limit: Maximum memory usage in MB
        
    Returns:
        Dictionary with execution results
    """
    executor = CodeExecutor(time_limit=time_limit, memory_limit=memory_limit)
    return executor.execute_code(code, language)

def run_script(file_path: str, time_limit: int = 10, memory_limit: int = 1024) -> Dict:
    """
    Run a script file safely
    
    Args:
        file_path: Path to the script file
        time_limit: Maximum execution time in seconds
        memory_limit: Maximum memory usage in MB
        
    Returns:
        Dictionary with execution results
    """
    executor = CodeExecutor(time_limit=time_limit, memory_limit=memory_limit)
    return executor.run_script(file_path)

def test_function(function: Callable, test_inputs: List[Dict]) -> Dict:
    """
    Test a function with multiple inputs
    
    Args:
        function: Function to test
        test_inputs: List of dictionaries with args and kwargs
        
    Returns:
        Dictionary with test results
    """
    executor = CodeExecutor()
    return executor.test_function(function, test_inputs)

def evaluate_performance(code: str, iterations: int = 5) -> Dict:
    """
    Benchmark code performance
    
    Args:
        code: Code to benchmark
        iterations: Number of iterations
        
    Returns:
        Dictionary with benchmark results
    """
    executor = CodeExecutor()
    return executor.evaluate_performance(code, iterations)

def is_code_safe(code: str) -> Dict:
    """
    Check if code is safe to execute
    
    Args:
        code: Code to check
        
    Returns:
        Dictionary with safety analysis results
    """
    analyzer = CodeAnalyzer()
    return analyzer.analyze_code(code)

# === COMMAND-LINE INTERFACE ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe Code Execution")
    parser.add_argument("--execute", help="Execute code from string")
    parser.add_argument("--file", help="Execute code from file")
    parser.add_argument("--language", default="python", help="Programming language")
    parser.add_argument("--time-limit", type=int, default=10, help="Time limit in seconds")
    parser.add_argument("--memory-limit", type=int, default=1024, help="Memory limit in MB")
    parser.add_argument("--analyze", action="store_true", help="Only analyze code for safety")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark performance")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark iterations")
    
    args = parser.parse_args()
    
    if args.execute:
        code = args.execute
        
        if args.analyze:
            result = is_code_safe(code)
        elif args.benchmark:
            result = evaluate_performance(code, args.iterations)
        else:
            result = execute_code(code, args.language, args.time_limit, args.memory_limit)
            
        print(json.dumps(result, indent=2))
    elif args.file:
        if not os.path.exists(args.file):
            print(json.dumps({
                "status": "error",
                "message": f"File not found: {args.file}"
            }, indent=2))
        else:
            with open(args.file, 'r') as f:
                code = f.read()
                
            if args.analyze:
                result = is_code_safe(code)
            elif args.benchmark:
                result = evaluate_performance(code, args.iterations)
            else:
                result = execute_code(code, args.language, args.time_limit, args.memory_limit)
                
            print(json.dumps(result, indent=2))
    else:
        # Read from stdin if no file or code provided
        if not sys.stdin.isatty():
            code = sys.stdin.read()
            
            if args.analyze:
                result = is_code_safe(code)
            elif args.benchmark:
                result = evaluate_performance(code, args.iterations)
            else:
                result = execute_code(code, args.language, args.time_limit, args.memory_limit)
                
            print(json.dumps(result, indent=2))
        else:
            parser.print_help()