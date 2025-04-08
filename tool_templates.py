"""
Tool templates for the CLI agent

This module contains example templates for creating new tools that can be
registered with the CLI agent at runtime.
"""

def create_calculator_tool():
    """
    Create a calculator tool that can perform basic arithmetic operations
    """
    code = """
def calculator(operation, a, b):
    \"\"\"
    Perform basic arithmetic operations
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
        
    Returns:
        Result of the operation
    \"\"\"
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return {
                "success": False,
                "message": "Cannot divide by zero",
                "data": None
            }
        return a / b
    else:
        return {
            "success": False,
            "message": f"Unknown operation: {operation}",
            "data": None
        }
    """
    
    description = "Perform basic arithmetic operations"
    
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "The operation to perform",
                "enum": ["add", "subtract", "multiply", "divide"]
            },
            "a": {
                "type": "number",
                "description": "First number"
            },
            "b": {
                "type": "number",
                "description": "Second number"
            }
        },
        "required": ["operation", "a", "b"]
    }
    
    return {
        "name": "calculator",
        "code": code,
        "description": description,
        "parameters": parameters,
        "category": "math"
    }

def create_weather_tool():
    """
    Create a weather tool that can fetch weather data for a location
    """
    code = """
import requests
import json

def get_weather(location, units="metric"):
    \"\"\"
    Get weather data for a location
    
    Args:
        location: City name or zip code
        units: Units to use (metric, imperial)
        
    Returns:
        Weather data for the location
    \"\"\"
    # This is a mock implementation
    # In a real implementation, you would use a weather API
    
    # Simulate API call
    weather_data = {
        "location": location,
        "temperature": 22 if units == "metric" else 72,
        "units": units,
        "conditions": "Sunny",
        "humidity": 65,
        "wind_speed": 10,
        "forecast": [
            {"day": "Today", "high": 25, "low": 18, "conditions": "Sunny"},
            {"day": "Tomorrow", "high": 23, "low": 17, "conditions": "Partly Cloudy"},
            {"day": "Day 3", "high": 21, "low": 16, "conditions": "Cloudy"}
        ]
    }
    
    return {
        "success": True,
        "message": f"Weather data for {location}",
        "data": weather_data
    }
    """
    
    description = "Get weather data for a location"
    
    parameters = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or zip code"
            },
            "units": {
                "type": "string",
                "description": "Units to use (metric, imperial)",
                "enum": ["metric", "imperial"],
                "default": "metric"
            }
        },
        "required": ["location"]
    }
    
    return {
        "name": "get_weather",
        "code": code,
        "description": description,
        "parameters": parameters,
        "category": "weather"
    }

def create_text_summarizer_tool():
    """
    Create a text summarizer tool that can summarize text
    """
    code = """
def summarize_text(text, max_sentences=3):
    \"\"\"
    Summarize text by extracting the most important sentences
    
    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences to include in summary
        
    Returns:
        Summarized text
    \"\"\"
    # This is a simple extractive summarization implementation
    # In a real implementation, you might use a more sophisticated algorithm
    
    import re
    from collections import Counter
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= max_sentences:
        return {
            "success": True,
            "message": "Text already concise, no summarization needed",
            "data": {
                "original_text": text,
                "summary": text,
                "sentence_count": len(sentences),
                "original_length": len(text),
                "summary_length": len(text)
            }
        }
    
    # Tokenize and count word frequencies
    words = re.findall(r'\\w+', text.lower())
    word_freq = Counter(words)
    
    # Score sentences based on word frequency
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        score = sum(word_freq[word.lower()] for word in re.findall(r'\\w+', sentence))
        # Normalize by sentence length to avoid bias towards longer sentences
        score = score / max(1, len(re.findall(r'\\w+', sentence)))
        sentence_scores.append((i, score, sentence))
    
    # Get top sentences
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    
    # Sort by original position
    top_sentences = sorted(top_sentences, key=lambda x: x[0])
    
    # Join sentences
    summary = ' '.join(s[2] for s in top_sentences)
    
    return {
        "success": True,
        "message": f"Summarized text from {len(sentences)} to {max_sentences} sentences",
        "data": {
            "original_text": text,
            "summary": summary,
            "sentence_count": {
                "original": len(sentences),
                "summary": len(top_sentences)
            },
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": len(summary) / len(text) if len(text) > 0 else 1
        }
    }
    """
    
    description = "Summarize text by extracting the most important sentences"
    
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to summarize"
            },
            "max_sentences": {
                "type": "integer",
                "description": "Maximum number of sentences to include in summary",
                "default": 3
            }
        },
        "required": ["text"]
    }
    
    return {
        "name": "summarize_text",
        "code": code,
        "description": description,
        "parameters": parameters,
        "category": "text_processing"
    }

def create_file_search_tool():
    """
    Create a file search tool that can search for text in files
    """
    code = """
def search_files(directory=".", pattern="*", text=None, case_sensitive=False, 
               max_results=100, include_binary=False, recursive=True):
    \"\"\"
    Search for files containing specific text
    
    Args:
        directory: Directory to search in
        pattern: File pattern to match (glob syntax)
        text: Text to search for within files
        case_sensitive: Whether the search should be case-sensitive
        max_results: Maximum number of results to return
        include_binary: Whether to include binary files in the search
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of files matching the criteria
    \"\"\"
    import os
    import glob
    import fnmatch
    import re
    
    # Normalize directory path
    directory = os.path.expanduser(directory)
    
    # Find all files matching the pattern
    if recursive:
        matches = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(root, filename))
    else:
        matches = glob.glob(os.path.join(directory, pattern))
    
    # If no text search is required, return the matches
    if text is None:
        return {
            "success": True,
            "message": f"Found {len(matches)} files matching pattern '{pattern}'",
            "data": {
                "files": matches[:max_results],
                "total_matches": len(matches),
                "truncated": len(matches) > max_results
            }
        }
    
    # Prepare the search pattern
    if case_sensitive:
        search_pattern = re.compile(re.escape(text))
    else:
        search_pattern = re.compile(re.escape(text), re.IGNORECASE)
    
    # Search for text in files
    results = []
    for file_path in matches:
        try:
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            # Check if file is binary
            is_binary = False
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\\0' in chunk:  # Simple binary detection
                    is_binary = True
            
            # Skip binary files if not included
            if is_binary and not include_binary:
                continue
            
            # Search for text in file
            with open(file_path, 'r', errors='replace') as f:
                content = f.read()
                if search_pattern.search(content):
                    # Find line numbers where the text appears
                    lines = content.splitlines()
                    matching_lines = []
                    for i, line in enumerate(lines):
                        if search_pattern.search(line):
                            matching_lines.append({
                                "line_number": i + 1,
                                "line": line.strip()
                            })
                    
                    results.append({
                        "file": file_path,
                        "matching_lines": matching_lines[:5],  # Limit to 5 matching lines
                        "total_matches": len(matching_lines)
                    })
                    
                    # Stop if we've reached the maximum number of results
                    if len(results) >= max_results:
                        break
        except Exception as e:
            # Skip files that can't be read
            continue
    
    return {
        "success": True,
        "message": f"Found {len(results)} files containing '{text}'",
        "data": {
            "files": results,
            "total_matches": len(results),
            "truncated": len(matches) > max_results
        }
    }
    """
    
    description = "Search for files containing specific text"
    
    parameters = {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Directory to search in",
                "default": "."
            },
            "pattern": {
                "type": "string",
                "description": "File pattern to match (glob syntax)",
                "default": "*"
            },
            "text": {
                "type": "string",
                "description": "Text to search for within files"
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether the search should be case-sensitive",
                "default": False
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 100
            },
            "include_binary": {
                "type": "boolean",
                "description": "Whether to include binary files in the search",
                "default": False
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to search recursively in subdirectories",
                "default": True
            }
        },
        "required": ["text"]
    }
    
    return {
        "name": "search_files",
        "code": code,
        "description": description,
        "parameters": parameters,
        "category": "file_system"
    }

def get_all_tool_templates():
    """Get all available tool templates"""
    return [
        create_calculator_tool(),
        create_weather_tool(),
        create_text_summarizer_tool(),
        create_file_search_tool()
    ]
