#!/usr/bin/env python3
"""
Register Jina tools with the agent kernel

This script registers the Jina client functions with the agent kernel
to make them available for function calling through the LLM API.
Includes OpenAI-powered content extraction capabilities that can
extract structured data from web content, including important facts,
dates, people, organizations, and more.
"""

import os
import sys
from typing import Dict, Any

# Import our kernel
from openrouter_kernel import OpenRouterKernel
from openrouter_kernel import register_kernel_function

# Import the Jina tools
from modules.jina_tools import jina_search, jina_fact_check, jina_read_url

# === MAIN FUNCTION ===
def main():
    """Register the Jina tools with the agent kernel"""
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return 1
        
    # Initialize kernel
    kernel = OpenRouterKernel(api_key)
    
    # Register the Jina tools
    print("Registering Jina tools with the agent kernel...")
    
    # Register jina_search
    kernel.register_function(
        "jina_search",
        jina_search,
        "Search the web using Jina search API with AI-powered content extraction",
        {
            "query": {
                "type": "string",
                "description": "Search query text"
            },
            "token": {
                "type": "string",
                "description": "Optional Jina API token (uses env var if not provided)"
            },
            "openai_key": {
                "type": "string",
                "description": "Optional OpenAI API key (uses env var if not provided)"
            },
            "extract_content": {
                "type": "boolean",
                "description": "Whether to extract structured content (requires OpenAI)",
                "default": True
            }
        }
    )
    
    # Register jina_fact_check
    kernel.register_function(
        "jina_fact_check",
        jina_fact_check,
        "Fact check a statement using Jina grounding API with AI-powered content extraction",
        {
            "query": {
                "type": "string",
                "description": "Statement to fact check"
            },
            "token": {
                "type": "string",
                "description": "Optional Jina API token (uses env var if not provided)"
            },
            "openai_key": {
                "type": "string",
                "description": "Optional OpenAI API key (uses env var if not provided)"
            },
            "extract_content": {
                "type": "boolean",
                "description": "Whether to extract structured content (requires OpenAI)",
                "default": True
            }
        }
    )
    
    # Register jina_read_url
    kernel.register_function(
        "jina_read_url",
        jina_read_url,
        "Read and rank content from a URL using Jina ranking API with AI-powered content extraction",
        {
            "url": {
                "type": "string",
                "description": "URL to read and rank"
            },
            "token": {
                "type": "string",
                "description": "Optional Jina API token (uses env var if not provided)"
            },
            "openai_key": {
                "type": "string",
                "description": "Optional OpenAI API key (uses env var if not provided)"
            },
            "extract_content": {
                "type": "boolean",
                "description": "Whether to extract structured content (requires OpenAI)",
                "default": True
            }
        }
    )
    
    print("Successfully registered Jina tools!")
    
    # List all registered functions
    functions = kernel.list_functions()
    print(f"Registered functions ({len(functions)}):")
    for func in functions:
        print(f"  - {func}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())