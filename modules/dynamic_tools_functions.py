"""
Dynamic Tools Functions - Functions for creating and managing dynamic tools.
These will be registered with the agent kernel to allow dynamic tool creation.
"""

from typing import Dict, Any, List, Optional
from modules.dynamic_tools import tool_factory

def create_dynamic_tool(name: str, code: str, description: str = "", auto_approve: bool = False) -> Dict[str, Any]:
    """
    Create a new dynamic tool from code with pre-verification
    
    Args:
        name: Name for the new tool
        code: Python code defining the tool functions
        description: Description of what the tool does
        auto_approve: Whether to automatically approve the tool
        
    Returns:
        Dictionary with result information
    """
    success, message, metadata = tool_factory.create_tool(
        name=name,
        code=code,
        description=description,
        auto_approve=auto_approve
    )
    
    if success:
        return {
            "status": "success",
            "message": message,
            "tool_name": name,
            "metadata": metadata
        }
    else:
        return {
            "status": "error",
            "message": message
        }

def approve_dynamic_tool(name: str) -> Dict[str, Any]:
    """
    Approve a pending dynamic tool
    
    Args:
        name: Tool name to approve
        
    Returns:
        Dictionary with result information
    """
    success, message = tool_factory.approve_tool(name)
    
    if success:
        return {
            "status": "success",
            "message": message,
            "tool_name": name
        }
    else:
        return {
            "status": "error",
            "message": message
        }
        
def reject_dynamic_tool(name: str) -> Dict[str, Any]:
    """
    Reject a pending dynamic tool
    
    Args:
        name: Tool name to reject
        
    Returns:
        Dictionary with result information
    """
    success, message = tool_factory.reject_tool(name)
    
    if success:
        return {
            "status": "success",
            "message": message,
            "tool_name": name
        }
    else:
        return {
            "status": "error",
            "message": message
        }
        
def list_dynamic_tools() -> Dict[str, Any]:
    """
    List all registered dynamic tools
    
    Returns:
        Dictionary with tools and their metadata
    """
    tools = tool_factory.list_tools()
    
    return {
        "status": "success",
        "count": len(tools),
        "tools": tools
    }
    
def list_pending_dynamic_tools() -> Dict[str, Any]:
    """
    List all pending dynamic tools
    
    Returns:
        Dictionary with pending tools and their metadata
    """
    tools = tool_factory.list_pending_tools()
    
    return {
        "status": "success",
        "count": len(tools),
        "tools": tools
    }
    
def get_dynamic_tool_code(name: str) -> Dict[str, Any]:
    """
    Get the code for a dynamic tool
    
    Args:
        name: Tool name
        
    Returns:
        Dictionary with tool code
    """
    success, code = tool_factory.get_tool_code(name)
    
    if success:
        return {
            "status": "success",
            "tool_name": name,
            "code": code
        }
    else:
        return {
            "status": "error",
            "message": code  # Error message
        }
        
def execute_dynamic_tool(name: str, **kwargs) -> Dict[str, Any]:
    """
    Execute a dynamic tool
    
    Args:
        name: Tool name to execute
        **kwargs: Arguments to pass to the tool
        
    Returns:
        Dictionary with execution result
    """
    success, result = tool_factory.execute_tool(name, **kwargs)
    
    if success:
        return {
            "status": "success",
            "tool_name": name,
            "result": result
        }
    else:
        return {
            "status": "error",
            "message": result  # Error message
        }
        
def get_tool_signature(name: str) -> Dict[str, Any]:
    """
    Get the signature for a dynamic tool
    
    Args:
        name: Tool name
        
    Returns:
        Dictionary with tool signature
    """
    success, signature = tool_factory.get_tool_signature(name)
    
    if success:
        return {
            "status": "success",
            "tool_name": name,
            "signature": signature
        }
    else:
        return {
            "status": "error",
            "message": signature  # Error message
        }