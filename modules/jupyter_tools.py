#!/usr/bin/env python3
"""
Jupyter Tools - Module for interacting with Jupyter notebooks and kernels

Provides tools for an AI agent to work with Jupyter notebooks:
- Create and manage Jupyter kernels
- Execute code in notebooks
- Create, open, and save notebooks
- Convert between Python code and notebook format

This module requires the following packages:
- jupyter
- notebook
- ipykernel
"""

import os
import sys
import json
import time
import re
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union

try:
    import jupyter_client
    from jupyter_client.kernelspec import KernelSpecManager
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
except ImportError:
    print("Jupyter packages not installed. Please install jupyter, notebook, and ipykernel.")
    jupyter_client = None
    nbformat = None

# === KERNEL MANAGEMENT ===
class JupyterKernelManager:
    """
    Manages Jupyter kernels for code execution
    """
    def __init__(self):
        """Initialize the kernel manager"""
        if jupyter_client is None:
            raise ImportError("jupyter_client package is required")
            
        self.kernels = {}  # name -> (km, kc) kernel manager and client
        self.kernel_outputs = {}  # name -> list of outputs
        
    def start_kernel(self, name: str = "default") -> Tuple[bool, str]:
        """
        Start a new Jupyter kernel
        
        Args:
            name: Name to assign to this kernel
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if name in self.kernels:
                return False, f"Kernel '{name}' already exists"
                
            # Start the kernel
            km = jupyter_client.KernelManager(kernel_name="python3")
            km.start_kernel()
            
            # Start the client
            kc = km.client()
            kc.start_channels()
            
            # Wait for kernel to be ready
            kc.wait_for_ready(timeout=30)
            
            # Store the kernel
            self.kernels[name] = (km, kc)
            self.kernel_outputs[name] = []
            
            return True, f"Kernel '{name}' started successfully"
        except Exception as e:
            return False, f"Error starting kernel: {str(e)}"
            
    def stop_kernel(self, name: str = "default") -> Tuple[bool, str]:
        """
        Stop a running kernel
        
        Args:
            name: Name of the kernel to stop
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if name not in self.kernels:
                return False, f"Kernel '{name}' not found"
                
            # Get the kernel manager and client
            km, kc = self.kernels[name]
            
            # Stop the client and kernel
            kc.stop_channels()
            km.shutdown_kernel()
            
            # Remove from dictionaries
            del self.kernels[name]
            del self.kernel_outputs[name]
            
            return True, f"Kernel '{name}' stopped successfully"
        except Exception as e:
            return False, f"Error stopping kernel: {str(e)}"
            
    def list_kernels(self) -> List[str]:
        """
        List all running kernels
        
        Returns:
            List of kernel names
        """
        return list(self.kernels.keys())
        
    def execute_code(self, code: str, name: str = "default", 
                    timeout: int = 60, store_history: bool = True) -> Dict:
        """
        Execute code in a kernel
        
        Args:
            code: Python code to execute
            name: Name of the kernel to use
            timeout: Timeout in seconds
            store_history: Whether to store in kernel history
            
        Returns:
            Dictionary with execution results
        """
        try:
            if name not in self.kernels:
                # Try to start the kernel
                success, message = self.start_kernel(name)
                if not success:
                    return {"status": "error", "message": message}
                    
            # Get the kernel client
            _, kc = self.kernels[name]
            
            # Execute the code
            msg_id = kc.execute(code, store_history=store_history)
            
            # Collect the outputs
            outputs = []
            output_text = ""
            error = None
            
            # Process messages until idle
            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=timeout)
                    msg_type = msg['header']['msg_type']
                    content = msg['content']
                    
                    if msg_type == 'status' and content['execution_state'] == 'idle':
                        break
                        
                    if msg_type == 'execute_result':
                        output_text += str(content['data'].get('text/plain', ''))
                        outputs.append({
                            "type": "execute_result",
                            "data": content['data'],
                            "metadata": content.get('metadata', {})
                        })
                        
                    elif msg_type == 'display_data':
                        outputs.append({
                            "type": "display_data",
                            "data": content['data'],
                            "metadata": content.get('metadata', {})
                        })
                        
                    elif msg_type == 'stream':
                        stream_text = content['text']
                        output_text += stream_text
                        outputs.append({
                            "type": "stream",
                            "name": content['name'],
                            "text": stream_text
                        })
                        
                    elif msg_type == 'error':
                        error_text = '\n'.join(content['traceback'])
                        output_text += error_text
                        error = {
                            "ename": content['ename'],
                            "evalue": content['evalue'],
                            "traceback": content['traceback']
                        }
                        outputs.append({
                            "type": "error",
                            "ename": content['ename'],
                            "evalue": content['evalue'],
                            "traceback": content['traceback']
                        })
                        
                except Exception as e:
                    # Timeout or other error
                    error = {
                        "ename": type(e).__name__,
                        "evalue": str(e),
                        "traceback": []
                    }
                    break
                    
            # Store the outputs
            self.kernel_outputs[name].append({
                "code": code,
                "outputs": outputs,
                "output_text": output_text,
                "error": error,
                "timestamp": time.time()
            })
            
            # Return the results
            result = {
                "status": "error" if error else "success",
                "outputs": outputs,
                "output_text": output_text
            }
            
            if error:
                result["error"] = error
                
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing code: {str(e)}"
            }
            
    def get_kernel_history(self, name: str = "default") -> List[Dict]:
        """
        Get the execution history for a kernel
        
        Args:
            name: Name of the kernel
            
        Returns:
            List of execution results
        """
        if name not in self.kernel_outputs:
            return []
            
        return self.kernel_outputs[name]
        
    def clear_kernel_history(self, name: str = "default") -> Tuple[bool, str]:
        """
        Clear the execution history for a kernel
        
        Args:
            name: Name of the kernel
            
        Returns:
            Tuple of (success, message)
        """
        if name not in self.kernel_outputs:
            return False, f"Kernel '{name}' not found"
            
        self.kernel_outputs[name] = []
        return True, f"History for kernel '{name}' cleared"

# === NOTEBOOK MANAGEMENT ===
class JupyterNotebookManager:
    """
    Manages Jupyter notebooks
    """
    def __init__(self):
        """Initialize the notebook manager"""
        if nbformat is None:
            raise ImportError("nbformat package is required")
            
    def create_notebook(self, cells: List[Dict] = None) -> Dict:
        """
        Create a new notebook
        
        Args:
            cells: List of cell dictionaries with 'type' and 'content'
            
        Returns:
            Dictionary with notebook data
        """
        try:
            # Create a new notebook
            nb = new_notebook()
            
            # Add cells if provided
            if cells:
                for cell in cells:
                    if cell['type'] == 'code':
                        nb.cells.append(new_code_cell(cell['content']))
                    elif cell['type'] == 'markdown':
                        nb.cells.append(new_markdown_cell(cell['content']))
                        
            return {
                "status": "success",
                "notebook": nb
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error creating notebook: {str(e)}"
            }
            
    def open_notebook(self, file_path: str) -> Dict:
        """
        Open a notebook from a file
        
        Args:
            file_path: Path to the notebook file
            
        Returns:
            Dictionary with notebook data
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return {
                    "status": "error",
                    "message": f"File not found: {file_path}"
                }
                
            # Read the notebook
            with open(file_path, 'r') as f:
                nb = nbformat.read(f, as_version=4)
                
            # Extract cell data
            cells = []
            for cell in nb.cells:
                cell_data = {
                    "type": cell.cell_type,
                    "content": cell.source
                }
                
                if cell.cell_type == 'code' and hasattr(cell, 'outputs'):
                    cell_data["outputs"] = cell.outputs
                    
                cells.append(cell_data)
                
            return {
                "status": "success",
                "notebook": nb,
                "cells": cells,
                "file_path": file_path
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error opening notebook: {str(e)}"
            }
            
    def save_notebook(self, notebook, file_path: str) -> Dict:
        """
        Save a notebook to a file
        
        Args:
            notebook: Notebook object
            file_path: Path to save the notebook
            
        Returns:
            Dictionary with result status
        """
        try:
            # Create directories if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write the notebook
            with open(file_path, 'w') as f:
                nbformat.write(notebook, f)
                
            return {
                "status": "success",
                "message": f"Notebook saved to {file_path}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error saving notebook: {str(e)}"
            }
            
    def code_to_notebook(self, code: str) -> Dict:
        """
        Convert Python code to a notebook
        
        Args:
            code: Python code string
            
        Returns:
            Dictionary with notebook data
        """
        try:
            # Create a new notebook
            nb = new_notebook()
            
            # Split code into cells by looking for cell markers
            # Default marker is # %% or # %%% or # In[]:
            cell_markers = [
                r'# *%%.*?$',  # For VSCode/Spyder cells
                r'# *In\[\d*\]:.*?$',  # For Jupyter style cells
                r'# *\+\+\+.*?$'  # Another common style
            ]
            
            pattern = '|'.join(cell_markers)
            
            # Split code by cell markers
            chunks = re.split(pattern, code, flags=re.MULTILINE)
            markers = re.findall(pattern, code, flags=re.MULTILINE)
            
            if len(chunks) <= 1:
                # No cell markers found, treat as a single cell
                nb.cells.append(new_code_cell(code))
            else:
                # Skip the first chunk if it's empty
                if not chunks[0].strip():
                    chunks = chunks[1:]
                else:
                    # If first chunk is not empty, add it without marker
                    markers = ['# %%'] + markers
                    
                # Create cells from chunks and markers
                for marker, chunk in zip(markers, chunks):
                    if chunk.strip():
                        if '# md' in marker.lower() or 'markdown' in marker.lower():
                            # It's a markdown cell
                            nb.cells.append(new_markdown_cell(chunk.strip()))
                        else:
                            # It's a code cell
                            nb.cells.append(new_code_cell(chunk.strip()))
                        
            return {
                "status": "success",
                "notebook": nb
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error converting code to notebook: {str(e)}"
            }
            
    def notebook_to_code(self, notebook) -> Dict:
        """
        Convert a notebook to Python code
        
        Args:
            notebook: Notebook object
            
        Returns:
            Dictionary with code string
        """
        try:
            code_chunks = []
            
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type == 'code':
                    code_chunks.append(f"# %% Cell {i+1}")
                    code_chunks.append(cell.source)
                    code_chunks.append("")
                elif cell.cell_type == 'markdown':
                    code_chunks.append(f"# %% [markdown] Cell {i+1}")
                    # Comment out markdown lines
                    markdown_lines = [f"# {line}" for line in cell.source.split('\n')]
                    code_chunks.append('\n'.join(markdown_lines))
                    code_chunks.append("")
                    
            return {
                "status": "success",
                "code": '\n'.join(code_chunks)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error converting notebook to code: {str(e)}"
            }
            
    def run_notebook(self, notebook, kernel_name: str = "default") -> Dict:
        """
        Run all cells in a notebook
        
        Args:
            notebook: Notebook object
            kernel_name: Name of kernel to use
            
        Returns:
            Dictionary with execution results
        """
        try:
            # Create a kernel manager if needed
            kernel_manager = JupyterKernelManager()
            
            # Check if kernel exists, start if needed
            if kernel_name not in kernel_manager.list_kernels():
                success, message = kernel_manager.start_kernel(kernel_name)
                if not success:
                    return {
                        "status": "error",
                        "message": f"Failed to start kernel: {message}"
                    }
                    
            # Process each code cell
            results = []
            
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type == 'code':
                    result = kernel_manager.execute_code(
                        cell.source, name=kernel_name
                    )
                    
                    results.append({
                        "cell_index": i,
                        "status": result["status"],
                        "outputs": result.get("outputs", []),
                        "output_text": result.get("output_text", ""),
                        "error": result.get("error", None)
                    })
                    
                    # Update the cell outputs
                    if hasattr(cell, 'outputs'):
                        cell.outputs = result.get("outputs", [])
                        
            return {
                "status": "success",
                "results": results,
                "notebook": notebook
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error running notebook: {str(e)}"
            }

# === UTILITY FUNCTIONS ===
def jupyter_execute_code(code: str, kernel_name: str = "default") -> Dict:
    """
    Execute code in a Jupyter kernel
    
    Args:
        code: Python code to execute
        kernel_name: Name of kernel to use
        
    Returns:
        Dictionary with execution results
    """
    try:
        kernel_manager = JupyterKernelManager()
        return kernel_manager.execute_code(code, name=kernel_name)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error executing code: {str(e)}"
        }

def jupyter_create_notebook(cells: List[Dict] = None) -> Dict:
    """
    Create a new Jupyter notebook
    
    Args:
        cells: List of cell dictionaries with 'type' and 'content'
        
    Returns:
        Dictionary with notebook data
    """
    try:
        notebook_manager = JupyterNotebookManager()
        return notebook_manager.create_notebook(cells)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating notebook: {str(e)}"
        }

def jupyter_open_notebook(file_path: str) -> Dict:
    """
    Open a Jupyter notebook from file
    
    Args:
        file_path: Path to notebook file
        
    Returns:
        Dictionary with notebook data
    """
    try:
        notebook_manager = JupyterNotebookManager()
        return notebook_manager.open_notebook(file_path)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error opening notebook: {str(e)}"
        }

def jupyter_save_notebook(notebook, file_path: str) -> Dict:
    """
    Save a Jupyter notebook to file
    
    Args:
        notebook: Notebook object
        file_path: Path to save notebook
        
    Returns:
        Dictionary with result status
    """
    try:
        notebook_manager = JupyterNotebookManager()
        return notebook_manager.save_notebook(notebook, file_path)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error saving notebook: {str(e)}"
        }

def jupyter_run_notebook(file_path: str, kernel_name: str = "default") -> Dict:
    """
    Run all cells in a Jupyter notebook
    
    Args:
        file_path: Path to notebook file
        kernel_name: Name of kernel to use
        
    Returns:
        Dictionary with execution results
    """
    try:
        notebook_manager = JupyterNotebookManager()
        result = notebook_manager.open_notebook(file_path)
        
        if result["status"] != "success":
            return result
            
        notebook = result["notebook"]
        
        return notebook_manager.run_notebook(notebook, kernel_name)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error running notebook: {str(e)}"
        }

# === COMMAND-LINE INTERFACE ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Jupyter Tools")
    parser.add_argument("--execute", help="Execute code")
    parser.add_argument("--kernel", default="default", help="Kernel name")
    parser.add_argument("--create-notebook", action="store_true", help="Create a notebook from code")
    parser.add_argument("--open-notebook", help="Open a notebook file")
    parser.add_argument("--save-notebook", help="Save notebook to file")
    parser.add_argument("--run-notebook", help="Run a notebook file")
    
    args = parser.parse_args()
    
    if args.execute:
        result = jupyter_execute_code(args.execute, args.kernel)
        print(json.dumps(result, indent=2))
    elif args.create_notebook:
        code = sys.stdin.read()
        notebook_manager = JupyterNotebookManager()
        result = notebook_manager.code_to_notebook(code)
        print(json.dumps(result["notebook"], indent=2))
    elif args.open_notebook:
        result = jupyter_open_notebook(args.open_notebook)
        print(json.dumps(result, indent=2))
    elif args.run_notebook:
        result = jupyter_run_notebook(args.run_notebook, args.kernel)
        print(json.dumps(result, indent=2))
    elif args.save_notebook:
        # Not directly usable from command line
        print("Save notebook option requires notebook object")
    else:
        parser.print_help()