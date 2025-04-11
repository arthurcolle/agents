"""
Jina Client Module - Client for interacting with Jina.ai endpoints
Supports search, fact checking, URL reading, and content extraction functionality.
"""
from typing import Dict, Optional, List, Any
import os
import urllib.parse
import aiohttp
from pydantic import BaseModel
import json

# Initialize OpenAI client if available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI client not available. Web content extraction will be limited.")

# Define the extraction model
class WebContentExtractionModel(BaseModel):
    urls: List[str] = []
    important_facts: List[str] = []
    quantities: List[str] = []
    important_dates: List[str] = []
    important_people: List[str] = []
    important_places: List[str] = []
    important_organizations: List[str] = []
    important_events: List[str] = []
    important_documents: List[str] = []
    important_links: List[str] = []

class JinaClient:
    """Client for interacting with Jina.ai endpoints"""
    
    def __init__(self, token: Optional[str] = None, openai_key: Optional[str] = None):
        """Initialize with your Jina token and optional OpenAI key"""
        self.token = token or os.getenv("JINA_API_KEY")
        if not self.token:
            raise ValueError("JINA_API_KEY environment variable or token must be provided")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        # Initialize OpenAI client if available
        self.openai_client = None
        if OPENAI_AVAILABLE:
            openai_api_key = openai_key or os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
                print("OpenAI client initialized for content extraction")
    
    async def search(self, query: str) -> Dict:
        """
        Search using s.jina.ai endpoint
        Args:
            query: Search term
        Returns:
            API response as dict with optional content extraction
        """
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/{encoded_query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response_text = await response.text()
                result = {"results": response_text}
                
                # Add content extraction if available
                if self.openai_client:
                    try:
                        extraction = await self.extract_content(response_text)
                        if "status" in extraction and extraction["status"] == "success":
                            result["extraction"] = extraction["extraction"]
                    except Exception as e:
                        result["extraction_error"] = str(e)
                
                return result
    
    async def fact_check(self, query: str) -> Dict:
        """
        Get grounding info using g.jina.ai endpoint
        Args:
            query: Query to ground
        Returns:
            API response as dict with optional content extraction
        """
        encoded_query = urllib.parse.quote(query)
        url = f"https://g.jina.ai/{encoded_query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response_text = await response.text()
                result = {"results": response_text}
                
                # Add content extraction if available
                if self.openai_client:
                    try:
                        extraction = await self.extract_content(response_text)
                        if "status" in extraction and extraction["status"] == "success":
                            result["extraction"] = extraction["extraction"]
                    except Exception as e:
                        result["extraction_error"] = str(e)
                
                return result
        
    async def read(self, url: str) -> Dict:
        """
        Get ranking using r.jina.ai endpoint
        Args:
            url: URL to rank
        Returns:
            API response as dict with optional content extraction
        """
        encoded_url = urllib.parse.quote(url)
        rank_url = f"https://r.jina.ai/{encoded_url}"
        async with aiohttp.ClientSession() as session:
            async with session.get(rank_url, headers=self.headers) as response:
                response_text = await response.text()
                result = {"results": response_text}
                
                # Add content extraction if available
                if self.openai_client:
                    try:
                        extraction = await self.extract_content(response_text)
                        if "status" in extraction and extraction["status"] == "success":
                            result["extraction"] = extraction["extraction"]
                    except Exception as e:
                        result["extraction_error"] = str(e)
                
                return result
                
    async def extract_content(self, text: str) -> Dict:
        """
        Extract structured information from text using OpenAI
        Args:
            text: Text to analyze
        Returns:
            Dict with structured extraction or error
        """
        if not OPENAI_AVAILABLE or not self.openai_client:
            return {"error": "OpenAI client not available for content extraction"}
            
        try:
            response = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can extract information from a given text."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                response_model=WebContentExtractionModel
            )
            return {
                "status": "success",
                "extraction": response.model_dump()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}