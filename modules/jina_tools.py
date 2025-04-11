"""
Jina Tools Module - Functions for the Jina client to be registered with the agent kernel.
"""
import asyncio
import json
import nest_asyncio
from typing import Dict, Any, Optional

# Import the JinaClient class from jina_client module
from modules.jina_client import JinaClient

# Apply nest_asyncio to allow nested event loops
try:
    nest_asyncio.apply()
except Exception as e:
    print(f"Warning: Failed to apply nest_asyncio: {e}")

# Global client instance (initialized on first use)
_jina_client = None

def _get_client(token: Optional[str] = None, openai_key: Optional[str] = None) -> JinaClient:
    """Get or initialize the Jina client instance with optional OpenAI support"""
    global _jina_client
    if _jina_client is None:
        try:
            _jina_client = JinaClient(token, openai_key)
        except Exception as e:
            return {"error": f"Failed to initialize Jina client: {str(e)}"}
    return _jina_client

def jina_search(query: str, token: Optional[str] = None, openai_key: Optional[str] = None, extract_content: bool = True) -> Dict:
    """
    Run a search query using Jina s.jina.ai with optional content extraction
    
    Args:
        query: Search query text
        token: Optional Jina API token (uses env var if not provided)
        openai_key: Optional OpenAI API key (uses env var if not provided)
        extract_content: Whether to extract structured content (requires OpenAI)
        
    Returns:
        Dictionary with search results and optional content extraction
    """
    client = _get_client(token, openai_key)
    
    # Check if client initialization failed
    if isinstance(client, dict) and "error" in client:
        return client
    
    try:
        # Run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(client.search(query))
        
        # Try to parse the JSON result if possible
        try:
            if isinstance(result.get("results"), str):
                json_result = json.loads(result["results"])
                result["results"] = json_result
        except (json.JSONDecodeError, AttributeError):
            pass
            
        # Include extraction if available
        if extract_content and "extraction" in result:
            return {
                "status": "success", 
                "results": result.get("results", {}),
                "extraction": result.get("extraction", {})
            }
            
        return {"status": "success", "results": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def jina_fact_check(query: str, token: Optional[str] = None, openai_key: Optional[str] = None, extract_content: bool = True) -> Dict:
    """
    Fact check a statement using Jina g.jina.ai with optional content extraction
    
    Args:
        query: Statement to fact check
        token: Optional Jina API token (uses env var if not provided)
        openai_key: Optional OpenAI API key (uses env var if not provided)
        extract_content: Whether to extract structured content (requires OpenAI)
        
    Returns:
        Dictionary with grounding information and optional content extraction
    """
    client = _get_client(token, openai_key)
    
    # Check if client initialization failed
    if isinstance(client, dict) and "error" in client:
        return client
    
    try:
        # Run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(client.fact_check(query))
        
        # Try to parse the JSON result if possible
        try:
            if isinstance(result.get("results"), str):
                json_result = json.loads(result["results"])
                result["results"] = json_result
        except (json.JSONDecodeError, AttributeError):
            pass
            
        # Include extraction if available
        if extract_content and "extraction" in result:
            return {
                "status": "success", 
                "grounding": result.get("results", {}),
                "extraction": result.get("extraction", {})
            }
            
        return {"status": "success", "grounding": result.get("results", result)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def jina_read_url(url: str, token: Optional[str] = None, openai_key: Optional[str] = None, extract_content: bool = True) -> Dict:
    """
    Read and rank content from a URL using Jina r.jina.ai with content extraction
    
    Args:
        url: URL to read and rank
        token: Optional Jina API token (uses env var if not provided)
        openai_key: Optional OpenAI API key (uses env var if not provided)
        extract_content: Whether to extract structured content (requires OpenAI)
        
    Returns:
        Dictionary with content ranking and optional structured extraction
    """
    client = _get_client(token, openai_key)
    
    # Check if client initialization failed
    if isinstance(client, dict) and "error" in client:
        return client
    
    try:
        # Run the async method in a synchronous context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(client.read(url))
        
        # Try to parse the JSON result if possible
        try:
            if isinstance(result.get("results"), str):
                json_result = json.loads(result["results"])
                result["results"] = json_result
        except (json.JSONDecodeError, AttributeError):
            pass
            
        # Include extraction if available
        if extract_content and "extraction" in result:
            return {
                "status": "success", 
                "content": result.get("results", {}),
                "extraction": result.get("extraction", {})
            }
            
        return {"status": "success", "content": result.get("results", result)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def jina_weather(location: str, token: Optional[str] = None, openai_key: Optional[str] = None) -> Dict:
    """
    Get weather information for a location using Jina search
    
    Args:
        location: City name or location to get weather for
        token: Optional Jina API token (uses env var if not provided)
        openai_key: Optional OpenAI API key (uses env var if not provided)
        
    Returns:
        Dictionary with weather information extracted from search results
    """
    query = f"current weather in {location}"
    
    # Get search results using Jina search
    search_result = jina_search(query, token, openai_key, extract_content=True)
    
    # Check if search was successful
    if search_result.get("status") != "success":
        return {
            "success": False,
            "message": f"Failed to get weather information for {location}",
            "error": search_result.get("error", "Unknown error")
        }
    
    # Extract the important facts about weather
    extraction = search_result.get("extraction", {})
    important_facts = extraction.get("important_facts", [])
    
    # Construct weather data from extracted information
    weather_data = {
        "location": location,
        "condition": "Unknown",
        "temperature": None,
        "feels_like": None,
        "humidity": None,
        "wind_speed": None,
        "wind_direction": None,
        "pressure": None,
        "visibility": None,
        "timestamp": None,
    }
    
    # Parse extracted facts to populate weather data
    for fact in important_facts:
        fact = fact.lower()
        if "temperature" in fact or "°c" in fact or "°f" in fact:
            weather_data["condition"] = "Based on web search"
            # Try to extract temperature
            if "°c" in fact:
                try:
                    temp_part = fact.split("°c")[0].strip()
                    temp_digits = ''.join(c for c in temp_part if c.isdigit() or c == '.' or c == '-')
                    weather_data["temperature"] = float(temp_digits)
                except (ValueError, IndexError):
                    pass
            elif "°f" in fact:
                try:
                    temp_part = fact.split("°f")[0].strip()
                    temp_digits = ''.join(c for c in temp_part if c.isdigit() or c == '.' or c == '-')
                    temp_f = float(temp_digits)
                    # Convert to Celsius
                    weather_data["temperature"] = round((temp_f - 32) * 5/9, 1)
                except (ValueError, IndexError):
                    pass
        
        # Extract other weather details if present
        if "humidity" in fact and "%" in fact:
            try:
                humidity_part = fact.split("%")[0].strip()
                humidity_digits = ''.join(c for c in humidity_part if c.isdigit())
                weather_data["humidity"] = int(humidity_digits)
            except (ValueError, IndexError):
                pass
        
        if "wind" in fact:
            # Try to extract wind speed
            for unit in ["mph", "km/h", "m/s"]:
                if unit in fact:
                    try:
                        wind_part = fact.split(unit)[0].strip().split()[-1]
                        wind_speed = float(wind_part)
                        # Convert to m/s if needed
                        if unit == "mph":
                            wind_speed = wind_speed * 0.44704
                        elif unit == "km/h":
                            wind_speed = wind_speed * 0.277778
                        weather_data["wind_speed"] = round(wind_speed, 1)
                    except (ValueError, IndexError):
                        pass
                    break
            
            # Try to extract wind direction
            for direction in ["north", "south", "east", "west", "nw", "ne", "sw", "se", "nnw", "nne", "ssw", "sse", "wnw", "wsw", "ene", "ese"]:
                if direction in fact.lower():
                    weather_data["wind_direction"] = direction.upper()
                    break
    
    # If we couldn't extract detailed weather, fall back to basic information
    if weather_data["temperature"] is None:
        # Use the full extraction to determine basic weather condition
        all_text = " ".join(important_facts)
        if "sun" in all_text.lower() or "clear" in all_text.lower():
            weather_data["condition"] = "Clear"
        elif "cloud" in all_text.lower():
            weather_data["condition"] = "Clouds"
        elif "rain" in all_text.lower():
            weather_data["condition"] = "Rain"
        elif "snow" in all_text.lower():
            weather_data["condition"] = "Snow"
        elif "fog" in all_text.lower() or "mist" in all_text.lower():
            weather_data["condition"] = "Fog"
    
    return {
        "success": True,
        "message": f"Weather information for {location}",
        "data": weather_data
    }