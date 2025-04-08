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
    sentences = re.split(r'(?<=[.!?])\\s+', text)
    
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

def create_web_scraper_tool():
    """
    Create a web scraper tool that can extract data from websites
    """
    code = """
import requests
from bs4 import BeautifulSoup
import json
import re
import urllib.parse

def web_scraper(url, selector=None, extract_type="text", attributes=None, pagination_selector=None, max_pages=1):
    \"\"\"
    Extract data from a website using BeautifulSoup
    
    Args:
        url: URL of the website to scrape
        selector: CSS selector to target specific elements (e.g., "div.content", "h1", "table tr")
        extract_type: Type of data to extract (text, html, attributes)
        attributes: List of attributes to extract if extract_type is "attributes"
        pagination_selector: CSS selector for pagination links
        max_pages: Maximum number of pages to scrape
        
    Returns:
        Extracted data from the website
    \"\"\"
    try:
        # Add user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        all_results = []
        current_url = url
        pages_scraped = 0
        
        while pages_scraped < max_pages:
            # Fetch the page
            response = requests.get(current_url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract data based on selector
            if selector:
                elements = soup.select(selector)
            else:
                elements = [soup]  # Use the entire page if no selector
            
            # Process each element based on extraction type
            page_results = []
            for element in elements:
                if extract_type == "text":
                    page_results.append(element.get_text(strip=True))
                elif extract_type == "html":
                    page_results.append(str(element))
                elif extract_type == "attributes" and attributes:
                    attr_data = {}
                    for attr in attributes:
                        attr_data[attr] = element.get(attr, "")
                    page_results.append(attr_data)
                else:
                    # Default to dictionary with text and html
                    page_results.append({
                        "text": element.get_text(strip=True),
                        "html": str(element)
                    })
            
            all_results.extend(page_results)
            pages_scraped += 1
            
            # Handle pagination if requested
            if pagination_selector and pages_scraped < max_pages:
                next_link = soup.select_one(pagination_selector)
                if next_link and next_link.get('href'):
                    next_url = next_link.get('href')
                    # Handle relative URLs
                    if not next_url.startswith(('http://', 'https://')):
                        next_url = urllib.parse.urljoin(current_url, next_url)
                    current_url = next_url
                else:
                    break  # No more pages
            else:
                break  # No pagination requested or reached max pages
        
        return {
            "success": True,
            "message": f"Successfully scraped {len(all_results)} elements from {pages_scraped} pages",
            "data": {
                "url": url,
                "selector": selector,
                "pages_scraped": pages_scraped,
                "results": all_results
            }
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Error fetching URL: {str(e)}",
            "data": None
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error scraping website: {str(e)}",
            "data": None
        }
    """
    
    description = "Extract data from websites using web scraping"
    
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL of the website to scrape"
            },
            "selector": {
                "type": "string",
                "description": "CSS selector to target specific elements (e.g., 'div.content', 'h1', 'table tr')"
            },
            "extract_type": {
                "type": "string",
                "description": "Type of data to extract",
                "enum": ["text", "html", "attributes"],
                "default": "text"
            },
            "attributes": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of attributes to extract if extract_type is 'attributes'"
            },
            "pagination_selector": {
                "type": "string",
                "description": "CSS selector for pagination links (e.g., 'a.next', '.pagination .next')"
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum number of pages to scrape",
                "default": 1
            }
        },
        "required": ["url"]
    }
    
    return {
        "name": "web_scraper",
        "code": code,
        "description": description,
        "parameters": parameters,
        "category": "web"
    }

def create_image_analyzer_tool():
    """
    Create an image analyzer tool that can extract information from images
    """
    code = """
import os
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
from datetime import datetime

def image_analyzer(image_path=None, image_url=None, analysis_type="basic", 
                 draw_annotations=False, output_path=None):
    \"\"\"
    Analyze images and extract information
    
    Args:
        image_path: Path to local image file
        image_url: URL of image to analyze
        analysis_type: Type of analysis to perform (basic, colors, objects, text)
        draw_annotations: Whether to draw annotations on the image
        output_path: Path to save annotated image (if draw_annotations is True)
        
    Returns:
        Analysis results for the image
    \"\"\"
    try:
        # Load the image
        if image_path:
            if not os.path.exists(image_path):
                return {
                    "success": False,
                    "message": f"Image file not found: {image_path}",
                    "data": None
                }
            img = Image.open(image_path)
            source = "local_file"
        elif image_url:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            source = "url"
        else:
            return {
                "success": False,
                "message": "Either image_path or image_url must be provided",
                "data": None
            }
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Get basic image information
        width, height = img.size
        format_name = img.format or "Unknown"
        mode = img.mode
        
        # Create a copy for annotations if needed
        if draw_annotations:
            annotated_img = img.copy()
            draw = ImageDraw.Draw(annotated_img)
            # Try to load a font, use default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
        
        results = {
            "basic_info": {
                "width": width,
                "height": height,
                "format": format_name,
                "mode": mode,
                "aspect_ratio": round(width / height, 2) if height > 0 else 0
            }
        }
        
        # Perform analysis based on type
        if analysis_type == "basic" or analysis_type == "all":
            # Basic analysis is already done
            pass
            
        if analysis_type == "colors" or analysis_type == "all":
            # Convert to numpy array for color analysis
            img_array = np.array(img)
            
            # Get color statistics
            mean_color = img_array.mean(axis=(0, 1)).astype(int)
            median_color = np.median(img_array, axis=(0, 1)).astype(int)
            
            # Get dominant colors (simplified)
            pixels = img.resize((50, 50)).getdata()
            color_counts = {}
            for pixel in pixels:
                if isinstance(pixel, int):
                    # For grayscale images
                    pixel = (pixel, pixel, pixel)
                color_counts[pixel] = color_counts.get(pixel, 0) + 1
            
            # Get top 5 colors
            dominant_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            results["color_analysis"] = {
                "mean_color": {
                    "rgb": mean_color.tolist(),
                    "hex": f"#{mean_color[0]:02x}{mean_color[1]:02x}{mean_color[2]:02x}"
                },
                "median_color": {
                    "rgb": median_color.tolist(),
                    "hex": f"#{median_color[0]:02x}{median_color[1]:02x}{median_color[2]:02x}"
                },
                "dominant_colors": [
                    {
                        "rgb": color,
                        "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                        "frequency": count / len(pixels)
                    }
                    for color, count in dominant_colors
                ]
            }
            
            # Draw color palette if annotations requested
            if draw_annotations:
                palette_width = 50
                palette_height = 20
                x_offset = 10
                y_offset = height - 30 - (len(dominant_colors) * palette_height)
                
                for i, (color, _) in enumerate(dominant_colors):
                    y = y_offset + (i * palette_height)
                    draw.rectangle([x_offset, y, x_offset + palette_width, y + palette_height], fill=color)
                    hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    draw.text((x_offset + palette_width + 5, y), hex_color, fill=(255, 255, 255), font=font)
                    draw.text((x_offset + palette_width + 5, y), hex_color, fill=(0, 0, 0), font=font, stroke_width=1)
        
        # Save annotated image if requested
        if draw_annotations:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"annotated_image_{timestamp}.jpg"
            
            annotated_img.save(output_path)
            results["annotated_image"] = {
                "path": output_path
            }
        
        return {
            "success": True,
            "message": f"Successfully analyzed image ({width}x{height}, {format_name})",
            "data": results
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error analyzing image: {str(e)}",
            "data": None
        }
    """
    
    description = "Analyze images and extract information"
    
    parameters = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to local image file"
            },
            "image_url": {
                "type": "string",
                "description": "URL of image to analyze"
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis to perform",
                "enum": ["basic", "colors", "all"],
                "default": "basic"
            },
            "draw_annotations": {
                "type": "boolean",
                "description": "Whether to draw annotations on the image",
                "default": False
            },
            "output_path": {
                "type": "string",
                "description": "Path to save annotated image (if draw_annotations is True)"
            }
        }
    }
    
    return {
        "name": "image_analyzer",
        "code": code,
        "description": description,
        "parameters": parameters,
        "category": "image_processing"
    }

def create_nlp_tool():
    """
    Create a natural language processing tool
    """
    code = """
import re
import string
from collections import Counter
import math

def nlp_processor(text, analysis_type="all", language="english", max_keywords=10):
    \"\"\"
    Process text using natural language processing techniques
    
    Args:
        text: Text to analyze
        analysis_type: Type of analysis to perform (sentiment, keywords, entities, statistics, all)
        language: Language of the text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        NLP analysis results
    \"\"\"
    try:
        if not text or not isinstance(text, str):
            return {
                "success": False,
                "message": "Invalid text input",
                "data": None
            }
        
        results = {}
        
        # Basic text statistics
        if analysis_type in ["statistics", "all"]:
            # Count characters, words, sentences
            char_count = len(text)
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text)) - 1 if text else 0
            
            # Count paragraphs (text blocks separated by newlines)
            paragraph_count = len([p for p in text.split('\\n\\n') if p.strip()])
            
            # Calculate readability metrics (Flesch Reading Ease)
            if word_count > 0 and sentence_count > 0:
                avg_words_per_sentence = word_count / max(1, sentence_count)
                syllable_count = count_syllables(text)
                avg_syllables_per_word = syllable_count / max(1, word_count)
                flesch_reading_ease = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
                flesch_reading_ease = min(100, max(0, flesch_reading_ease))  # Clamp between 0-100
            else:
                avg_words_per_sentence = 0
                syllable_count = 0
                avg_syllables_per_word = 0
                flesch_reading_ease = 0
            
            results["statistics"] = {
                "character_count": char_count,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "avg_words_per_sentence": round(avg_words_per_sentence, 2),
                "avg_syllables_per_word": round(avg_syllables_per_word, 2),
                "readability": {
                    "flesch_reading_ease": round(flesch_reading_ease, 2),
                    "interpretation": interpret_flesch_score(flesch_reading_ease)
                }
            }
        
        # Keyword extraction
        if analysis_type in ["keywords", "all"]:
            keywords = extract_keywords(text, language, max_keywords)
            results["keywords"] = keywords
        
        # Simple sentiment analysis
        if analysis_type in ["sentiment", "all"]:
            sentiment = analyze_sentiment(text)
            results["sentiment"] = sentiment
        
        return {
            "success": True,
            "message": f"Successfully analyzed text ({len(text)} characters)",
            "data": results
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error processing text: {str(e)}",
            "data": None
        }

def count_syllables(text):
    \"\"\"Count syllables in text (English)\"\"\"
    # This is a simplified syllable counter for English
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    
    syllable_count = 0
    for word in words:
        word = word.strip()
        if not word:
            continue
            
        # Count vowel groups
        count = len(re.findall(r'[aeiouy]+', word))
        
        # Adjust for common patterns
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiouy':
            count += 1
        if count == 0:
            count = 1
            
        syllable_count += count
        
    return syllable_count

def interpret_flesch_score(score):
    \"\"\"Interpret Flesch Reading Ease score\"\"\"
    if score >= 90:
        return "Very Easy - 5th grade"
    elif score >= 80:
        return "Easy - 6th grade"
    elif score >= 70:
        return "Fairly Easy - 7th grade"
    elif score >= 60:
        return "Standard - 8th-9th grade"
    elif score >= 50:
        return "Fairly Difficult - 10th-12th grade"
    elif score >= 30:
        return "Difficult - College"
    else:
        return "Very Difficult - College Graduate"

def extract_keywords(text, language="english", max_keywords=10):
    \"\"\"Extract keywords using TF-IDF like approach\"\"\"
    # Simplified stopwords for English
    stopwords = set([
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
        "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
        "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", 
        "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", 
        "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", 
        "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", 
        "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", 
        "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", 
        "would", "you", "your", "yours", "yourself", "yourselves"
    ])
    
    # Clean and tokenize text
    text = text.lower()
    text = re.sub(r'[^\\w\\s]', '', text)
    words = text.split()
    
    # Remove stopwords and short words
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    
    # Calculate "importance" (simplified TF-IDF)
    word_importance = {}
    total_words = len(filtered_words)
    
    for word, count in word_freq.items():
        # Term frequency
        tf = count / max(1, total_words)
        
        # Inverse sentence frequency (simplified)
        sentences = re.split(r'[.!?]+', text)
        sf = sum(1 for s in sentences if word in s.lower())
        isf = math.log(len(sentences) / max(1, sf))
        
        # Importance score
        word_importance[word] = tf * isf
    
    # Get top keywords
    keywords = [
        {"word": word, "score": round(score, 4), "count": word_freq[word]}
        for word, score in sorted(word_importance.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
    ]
    
    return keywords

def analyze_sentiment(text):
    \"\"\"Simple rule-based sentiment analysis\"\"\"
    # Very basic positive and negative word lists
    positive_words = set([
        "good", "great", "excellent", "amazing", "wonderful", "fantastic", "terrific", "outstanding",
        "brilliant", "awesome", "superb", "perfect", "happy", "joy", "love", "like", "best", "better",
        "positive", "beautiful", "nice", "pleasant", "impressive", "remarkable", "exceptional"
    ])
    
    negative_words = set([
        "bad", "terrible", "awful", "horrible", "poor", "disappointing", "worst", "worse",
        "negative", "ugly", "unpleasant", "hate", "dislike", "failure", "fail", "problem",
        "difficult", "hard", "trouble", "wrong", "sad", "angry", "upset", "annoying", "terrible"
    ])
    
    # Clean text
    text = text.lower()
    words = re.findall(r'\\b[\\w\']+\\b', text)
    
    # Count positive and negative words
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Calculate sentiment score (-1 to 1)
    total = positive_count + negative_count
    if total == 0:
        sentiment_score = 0
    else:
        sentiment_score = (positive_count - negative_count) / total
    
    # Determine sentiment label
    if sentiment_score > 0.2:
        sentiment = "positive"
    elif sentiment_score < -0.2:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "score": round(sentiment_score, 2),
        "label": sentiment,
        "positive_words": positive_count,
        "negative_words": negative_count
    }
    """
    
    description = "Process text using natural language processing techniques"
    
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to analyze"
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis to perform",
                "enum": ["sentiment", "keywords", "statistics", "all"],
                "default": "all"
            },
            "language": {
                "type": "string",
                "description": "Language of the text",
                "default": "english"
            },
            "max_keywords": {
                "type": "integer",
                "description": "Maximum number of keywords to extract",
                "default": 10
            }
        },
        "required": ["text"]
    }
    
    return {
        "name": "nlp_processor",
        "code": code,
        "description": description,
        "parameters": parameters,
        "category": "nlp"
    }

def create_time_series_tool():
    """
    Create a time series analysis tool
    """
    code = """
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
import os

def time_series_analyzer(data=None, date_column=None, value_column=None, 
                        frequency=None, analysis_type="basic", forecast_periods=7,
                        csv_path=None, output_path=None):
    \"\"\"
    Analyze time series data and generate forecasts
    
    Args:
        data: JSON string or dictionary with time series data
        date_column: Name of the column containing dates
        value_column: Name of the column containing values to analyze
        frequency: Frequency for resampling (daily, weekly, monthly, quarterly, yearly)
        analysis_type: Type of analysis to perform (basic, decomposition, forecast, all)
        forecast_periods: Number of periods to forecast
        csv_path: Path to CSV file with time series data (alternative to data)
        output_path: Path to save plots and results
        
    Returns:
        Time series analysis results
    \"\"\"
    try:
        # Load data
        if csv_path:
            if not os.path.exists(csv_path):
                return {
                    "success": False,
                    "message": f"CSV file not found: {csv_path}",
                    "data": None
                }
            df = pd.read_csv(csv_path)
        elif data:
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except:
                    return {
                        "success": False,
                        "message": "Invalid JSON data",
                        "data": None
                    }
            
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
                
            df = pd.DataFrame(data)
        else:
            return {
                "success": False,
                "message": "Either data or csv_path must be provided",
                "data": None
            }
        
        # Validate required columns
        if not date_column or date_column not in df.columns:
            # Try to find a date column
            for col in df.columns:
                if "date" in col.lower() or "time" in col.lower():
                    date_column = col
                    break
            
            if not date_column:
                return {
                    "success": False,
                    "message": "Date column not specified or not found in data",
                    "data": None
                }
        
        if not value_column or value_column not in df.columns:
            # Try to find a numeric column
            for col in df.columns:
                if col != date_column and pd.api.types.is_numeric_dtype(df[col]):
                    value_column = col
                    break
            
            if not value_column:
                return {
                    "success": False,
                    "message": "Value column not specified or not found in data",
                    "data": None
                }
        
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date
        df = df.sort_values(by=date_column)
        
        # Set date as index
        df = df.set_index(date_column)
        
        # Resample if frequency is specified
        if frequency:
            freq_map = {
                "daily": "D",
                "weekly": "W",
                "monthly": "M",
                "quarterly": "Q",
                "yearly": "Y"
            }
            pandas_freq = freq_map.get(frequency.lower(), "D")
            df_resampled = df[value_column].resample(pandas_freq).mean()
        else:
            df_resampled = df[value_column]
        
        # Fill missing values with forward fill then backward fill
        df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
        
        results = {}
        
        # Basic statistics
        if analysis_type in ["basic", "all"]:
            stats = {
                "count": len(df_resampled),
                "mean": float(df_resampled.mean()),
                "std": float(df_resampled.std()),
                "min": float(df_resampled.min()),
                "25%": float(df_resampled.quantile(0.25)),
                "50%": float(df_resampled.median()),
                "75%": float(df_resampled.quantile(0.75)),
                "max": float(df_resampled.max()),
                "start_date": df_resampled.index.min().strftime("%Y-%m-%d"),
                "end_date": df_resampled.index.max().strftime("%Y-%m-%d"),
                "duration_days": (df_resampled.index.max() - df_resampled.index.min()).days
            }
            
            # Calculate growth rates
            if len(df_resampled) > 1:
                first_value = df_resampled.iloc[0]
                last_value = df_resampled.iloc[-1]
                if first_value != 0:
                    total_growth = (last_value - first_value) / first_value
                    stats["total_growth_rate"] = float(total_growth)
                    
                    # Annualized growth rate
                    years = (df_resampled.index.max() - df_resampled.index.min()).days / 365.25
                    if years > 0 and first_value > 0 and last_value > 0:
                        cagr = (last_value / first_value) ** (1 / years) - 1
                        stats["annualized_growth_rate"] = float(cagr)
            
            results["statistics"] = stats
            
            # Create basic plot
            plt.figure(figsize=(10, 6))
            plt.plot(df_resampled.index, df_resampled.values)
            plt.title(f"Time Series: {value_column}")
            plt.xlabel("Date")
            plt.ylabel(value_column)
            plt.grid(True)
            
            # Save or encode plot
            if output_path:
                plt.savefig(f"{output_path}_basic.png")
                results["plots"] = {"basic": f"{output_path}_basic.png"}
            else:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                results["plots"] = {
                    "basic": base64.b64encode(buf.read()).decode('utf-8')
                }
            plt.close()
        
        # Simple forecast (naive method)
        if analysis_type in ["forecast", "all"]:
            # Use a simple moving average for forecasting
            window = min(7, len(df_resampled) // 2)
            if window > 0:
                # Calculate moving average
                ma = df_resampled.rolling(window=window).mean()
                
                # Get the last value for forecasting
                last_ma = ma.iloc[-1]
                
                # Create forecast dates
                last_date = df_resampled.index[-1]
                forecast_dates = []
                
                if frequency:
                    # Use pandas date_range for regular frequencies
                    freq_map = {
                        "daily": "D",
                        "weekly": "W",
                        "monthly": "M",
                        "quarterly": "Q",
                        "yearly": "Y"
                    }
                    pandas_freq = freq_map.get(frequency.lower(), "D")
                    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                                 periods=forecast_periods, 
                                                 freq=pandas_freq)
                else:
                    # Estimate frequency from data
                    if len(df_resampled) > 1:
                        avg_days = (df_resampled.index[-1] - df_resampled.index[0]).days / (len(df_resampled) - 1)
                        for i in range(forecast_periods):
                            forecast_dates.append(last_date + timedelta(days=avg_days * (i + 1)))
                
                # Create forecast values (naive forecast using last value)
                forecast_values = [last_ma] * len(forecast_dates)
                
                # Store forecast results
                forecast_result = {
                    "dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
                    "values": [float(v) for v in forecast_values],
                    "method": "Simple Moving Average"
                }
                
                results["forecast"] = forecast_result
                
                # Plot with forecast
                plt.figure(figsize=(12, 6))
                plt.plot(df_resampled.index, df_resampled.values, label='Historical')
                plt.plot(forecast_dates, forecast_values, 'r--', label='Forecast')
                plt.title(f"Time Series Forecast: {value_column}")
                plt.xlabel("Date")
                plt.ylabel(value_column)
                plt.legend()
                plt.grid(True)
                
                # Save or encode plot
                if output_path:
                    plt.savefig(f"{output_path}_forecast.png")
                    if "plots" not in results:
                        results["plots"] = {}
                    results["plots"]["forecast"] = f"{output_path}_forecast.png"
                else:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    if "plots" not in results:
                        results["plots"] = {}
                    results["plots"]["forecast"] = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
        
        return {
            "success": True,
            "message": f"Successfully analyzed time series data ({len(df_resampled)} points)",
            "data": results
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error analyzing time series: {str(e)}",
            "data": None
        }
    """
    
    description = "Analyze time series data and generate forecasts"
    
    parameters = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "description": "JSON string or dictionary with time series data"
            },
            "date_column": {
                "type": "string",
                "description": "Name of the column containing dates"
            },
            "value_column": {
                "type": "string",
                "description": "Name of the column containing values to analyze"
            },
            "frequency": {
                "type": "string",
                "description": "Frequency for resampling",
                "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"]
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis to perform",
                "enum": ["basic", "forecast", "all"],
                "default": "basic"
            },
            "forecast_periods": {
                "type": "integer",
                "description": "Number of periods to forecast",
                "default": 7
            },
            "csv_path": {
                "type": "string",
                "description": "Path to CSV file with time series data (alternative to data)"
            },
            "output_path": {
                "type": "string",
                "description": "Path to save plots and results"
            }
        }
    }
    
    return {
        "name": "time_series_analyzer",
        "code": code,
        "description": description,
        "parameters": parameters,
        "category": "data_analysis"
    }

def create_database_tool():
    """
    Create a database operations tool
    """
    code = """
import sqlite3
import os
import json
import pandas as pd
import csv
import io

def database_tool(operation, database_path, query=None, table_name=None, 
                data=None, csv_path=None, output_format="json"):
    \"\"\"
    Perform database operations using SQLite
    
    Args:
        operation: Operation to perform (query, create_table, insert, export, list_tables, describe)
        database_path: Path to SQLite database file
        query: SQL query to execute
        table_name: Name of the table to operate on
        data: Data to insert (list of dictionaries or JSON string)
        csv_path: Path to CSV file for importing data
        output_format: Format for query results (json, csv, dataframe)
        
    Returns:
        Results of the database operation
    \"\"\"
    try:
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(database_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        # Connect to database
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        # Execute operation
        if operation == "query":
            if not query:
                return {
                    "success": False,
                    "message": "Query is required for 'query' operation",
                    "data": None
                }
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Format results
            if output_format == "json":
                results = [dict(row) for row in rows]
            elif output_format == "csv":
                if not rows:
                    results = ""
                else:
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(rows[0].keys())
                    writer.writerows([list(row) for row in rows])
                    results = output.getvalue()
            elif output_format == "dataframe":
                results = pd.DataFrame([dict(row) for row in rows])
            else:
                results = [dict(row) for row in rows]
            
            return {
                "success": True,
                "message": f"Query executed successfully. {len(rows)} rows returned.",
                "data": results
            }
            
        elif operation == "create_table":
            if not table_name:
                return {
                    "success": False,
                    "message": "Table name is required for 'create_table' operation",
                    "data": None
                }
            
            if query:
                # Use provided CREATE TABLE query
                create_query = query
            elif data:
                # Generate CREATE TABLE query from data structure
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        return {
                            "success": False,
                            "message": "Invalid JSON data",
                            "data": None
                        }
                
                if not data or not isinstance(data, list) or not data[0]:
                    return {
                        "success": False,
                        "message": "Data must be a non-empty list of dictionaries",
                        "data": None
                    }
                
                # Get column names and types from first row
                columns = []
                for key, value in data[0].items():
                    if isinstance(value, int):
                        col_type = "INTEGER"
                    elif isinstance(value, float):
                        col_type = "REAL"
                    elif isinstance(value, bool):
                        col_type = "BOOLEAN"
                    else:
                        col_type = "TEXT"
                    
                    columns.append(f"\"{key}\" {col_type}")
                
                create_query = f"CREATE TABLE IF NOT EXISTS \"{table_name}\" ({', '.join(columns)})"
            else:
                return {
                    "success": False,
                    "message": "Either query or data is required for 'create_table' operation",
                    "data": None
                }
            
            cursor.execute(create_query)
            conn.commit()
            
            return {
                "success": True,
                "message": f"Table '{table_name}' created successfully",
                "data": {
                    "table_name": table_name,
                    "query": create_query
                }
            }
            
        elif operation == "insert":
            if not table_name:
                return {
                    "success": False,
                    "message": "Table name is required for 'insert' operation",
                    "data": None
                }
            
            # Get data to insert
            if data:
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except:
                        return {
                            "success": False,
                            "message": "Invalid JSON data",
                            "data": None
                        }
            elif csv_path:
                if not os.path.exists(csv_path):
                    return {
                        "success": False,
                        "message": f"CSV file not found: {csv_path}",
                        "data": None
                    }
                
                # Read CSV file
                df = pd.read_csv(csv_path)
                data = df.to_dict(orient="records")
            else:
                return {
                    "success": False,
                    "message": "Either data or csv_path is required for 'insert' operation",
                    "data": None
                }
            
            # Ensure data is a list
            if not isinstance(data, list):
                data = [data]
            
            if not data:
                return {
                    "success": False,
                    "message": "No data to insert",
                    "data": None
                }
            
            # Get column names from first row
            columns = list(data[0].keys())
            
            # Prepare INSERT query
            placeholders = ", ".join(["?"] * len(columns))
            column_str = ", ".join([f'"{col}"' for col in columns])
            insert_query = f'INSERT INTO "{table_name}" ({column_str}) VALUES ({placeholders})'
            
            # Execute INSERT for each row
            rows_inserted = 0
            for row in data:
                values = [row.get(col) for col in columns]
                cursor.execute(insert_query, values)
                rows_inserted += 1
            
            conn.commit()
            
            return {
                "success": True,
                "message": f"Successfully inserted {rows_inserted} rows into '{table_name}'",
                "data": {
                    "table_name": table_name,
                    "rows_inserted": rows_inserted
                }
            }
            
        elif operation == "export":
            if not table_name and not query:
                return {
                    "success": False,
                    "message": "Either table_name or query is required for 'export' operation",
                    "data": None
                }
            
            # Execute query
            if query:
                cursor.execute(query)
            else:
                cursor.execute(f'SELECT * FROM "{table_name}"')
            
            rows = cursor.fetchall()
            
            # Format results
            if output_format == "json":
                results = [dict(row) for row in rows]
            elif output_format == "csv":
                if not rows:
                    results = ""
                else:
                    output = io.StringIO()
                    writer = csv.writer(output)
                    writer.writerow(rows[0].keys())
                    writer.writerows([list(row) for row in rows])
                    results = output.getvalue()
            elif output_format == "dataframe":
                results = pd.DataFrame([dict(row) for row in rows])
            else:
                results = [dict(row) for row in rows]
            
            return {
                "success": True,
                "message": f"Successfully exported {len(rows)} rows",
                "data": results
            }
            
        elif operation == "list_tables":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            return {
                "success": True,
                "message": f"Found {len(tables)} tables",
                "data": {
                    "tables": tables
                }
            }
            
        elif operation == "describe":
            if not table_name:
                return {
                    "success": False,
                    "message": "Table name is required for 'describe' operation",
                    "data": None
                }
            
            # Get table info
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns = [dict(row) for row in cursor.fetchall()]
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM '{table_name}'")
            row_count = cursor.fetchone()[0]
            
            # Get sample data
            cursor.execute(f"SELECT * FROM '{table_name}' LIMIT 5")
            sample_data = [dict(row) for row in cursor.fetchall()]
            
            return {
                "success": True,
                "message": f"Table '{table_name}' has {len(columns)} columns and {row_count} rows",
                "data": {
                    "table_name": table_name,
                    "columns": columns,
                    "row_count": row_count,
                    "sample_data": sample_data
                }
            }
            
        else:
            return {
                "success": False,
                "message": f"Unknown operation: {operation}",
                "data": None
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Database error: {str(e)}",
            "data": None
        }
    finally:
        if conn:
            conn.close()
    """
    
    description = "Perform database operations using SQLite"
    
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "Operation to perform",
                "enum": ["query", "create_table", "insert", "export", "list_tables", "describe"],
                "default": "query"
            },
            "database_path": {
                "type": "string",
                "description": "Path to SQLite database file"
            },
            "query": {
                "type": "string",
                "description": "SQL query to execute"
            },
            "table_name": {
                "type": "string",
                "description": "Name of the table to operate on"
            },
            "data": {
                "type": "object",
                "description": "Data to insert (list of dictionaries or JSON string)"
            },
            "csv_path": {
                "type": "string",
                "description": "Path to CSV file for importing data"
            },
            "output_format": {
                "type": "string",
                "description": "Format for query results",
                "enum": ["json", "csv", "dataframe"],
                "default": "json"
            }
        },
        "required": ["operation", "database_path"]
    }
    
    return {
        "name": "database_tool",
        "code": code,
        "description": description,
        "parameters": parameters,
        "category": "database"
    }

def get_all_tool_templates():
    """Get all available tool templates"""
    return [
        create_calculator_tool(),
        create_weather_tool(),
        create_text_summarizer_tool(),
        create_file_search_tool(),
        create_web_scraper_tool(),
        create_image_analyzer_tool(),
        create_nlp_tool(),
        create_time_series_tool(),
        create_database_tool()
    ]
