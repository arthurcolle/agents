from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import urllib.parse
import aiohttp
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jina-client")

class JinaClient:
    """Client for interacting with Jina.ai endpoints"""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize with your Jina token"""
        self.token = token or os.getenv("JINA_API_KEY")
        if not self.token:
            logger.warning("JINA_API_KEY environment variable or token not provided. Some features may not work.")
        self.headers = {
            "Authorization": f"Bearer {self.token}" if self.token else "",
            "Content-Type": "application/json"
        }
    
    async def search(self, query: str) -> Dict[str, Any]:
        """
        Search using s.jina.ai endpoint
        Args:
            query: Search term
        Returns:
            API response as dict
        """
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://s.jina.ai/{encoded_query}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        response_text = await response.text()
                        logger.error(f"Jina search error: {response.status} - {response_text}")
                        return {
                            "error": True,
                            "status": response.status,
                            "message": f"Error from Jina search API: {response_text}"
                        }
        except Exception as e:
            logger.error(f"Error in Jina search: {str(e)}")
            return {"error": True, "message": f"Error connecting to Jina search API: {str(e)}"}
    
    async def fact_check(self, query: str) -> Dict[str, Any]:
        """
        Get grounding info using g.jina.ai endpoint
        Args:
            query: Query to ground
        Returns:
            API response as dict
        """
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://g.jina.ai/{encoded_query}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        response_text = await response.text()
                        logger.error(f"Jina fact-check error: {response.status} - {response_text}")
                        return {
                            "error": True,
                            "status": response.status,
                            "message": f"Error from Jina fact-check API: {response_text}"
                        }
        except Exception as e:
            logger.error(f"Error in Jina fact-check: {str(e)}")
            return {"error": True, "message": f"Error connecting to Jina fact-check API: {str(e)}"}
        
    async def read(self, url: str) -> Dict[str, Any]:
        """
        Get content using r.jina.ai endpoint
        Args:
            url: URL to read
        Returns:
            API response as dict
        """
        try:
            encoded_url = urllib.parse.quote(url)
            read_url = f"https://r.jina.ai/{encoded_url}"
            async with aiohttp.ClientSession() as session:
                async with session.get(read_url, headers=self.headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        response_text = await response.text()
                        logger.error(f"Jina read error: {response.status} - {response_text}")
                        return {
                            "error": True,
                            "status": response.status,
                            "message": f"Error from Jina read API: {response_text}"
                        }
        except Exception as e:
            logger.error(f"Error in Jina read: {str(e)}")
            return {"error": True, "message": f"Error connecting to Jina read API: {str(e)}"}

    def get_mock_search_results(self, query: str) -> Dict[str, Any]:
        """
        Get mock search results when API key is not available
        Args:
            query: Search term
        Returns:
            Mock search results
        """
        return {
            "results": [
                {
                    "title": f"Mock search result 1 for '{query}'",
                    "url": "https://example.com/result1",
                    "snippet": f"This is a mock search result for '{query}'. In a real implementation, this would be actual search results from Jina."
                },
                {
                    "title": f"Mock search result 2 for '{query}'",
                    "url": "https://example.com/result2",
                    "snippet": "Another mock search result. Please provide a valid Jina API key for real results."
                }
            ],
            "mock": True
        }
    
    def get_mock_fact_check(self, query: str) -> Dict[str, Any]:
        """
        Get mock fact check results when API key is not available
        Args:
            query: Query to fact check
        Returns:
            Mock fact check results
        """
        return {
            "factCheck": {
                "claim": query,
                "verdict": "UNKNOWN",
                "confidence": 0.5,
                "explanation": "This is a mock fact check result. Please provide a valid Jina API key for real fact checking."
            },
            "sources": [
                {
                    "title": "Mock Source 1",
                    "url": "https://example.com/source1",
                    "relevance": 0.8
                }
            ],
            "mock": True
        }
    
    def get_mock_read_results(self, url: str) -> Dict[str, Any]:
        """
        Get mock read results when API key is not available
        Args:
            url: URL to read
        Returns:
            Mock read results
        """
        return {
            "url": url,
            "title": "Mock Page Title",
            "content": f"This is mock content for {url}. In a real implementation, this would be the actual content extracted from the URL by Jina.",
            "summary": "This is a mock summary. Please provide a valid Jina API key for real content extraction.",
            "mock": True
        }

# Example usage
async def example_usage():
    client = JinaClient()
    
    # Search example
    search_results = await client.search("artificial intelligence")
    print("Search results:", search_results)
    
    # Fact check example
    fact_check_results = await client.fact_check("The Earth is flat")
    print("Fact check results:", fact_check_results)
    
    # Read URL example
    read_results = await client.read("https://jina.ai")
    print("Read results:", read_results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
