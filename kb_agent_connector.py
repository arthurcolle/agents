import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple

# Import the knowledge base dispatcher
from knowledge_base_dispatcher import dispatcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kb_agent_connector")

class KnowledgeBaseConnector:
    """
    Connector for integrating the knowledge base dispatcher with other agent systems.
    This acts as a bridge between the main agent and the knowledge base sub-agents.
    """
    
    def __init__(self):
        """Initialize the knowledge base connector"""
        self.cia = dispatcher.cia
    
    def set_kb_classification(self, kb_name: str, level: str) -> None:
        """
        Set the classification level for a knowledge base.

        Args:
            kb_name: Name of the knowledge base
            level: Classification level to set
        """
        self.cia.set_classification_level(kb_name, level)

    async def dispatch_to_kb_agent(self, kb_name: str, command: str) -> Dict[str, Any]:
        """
        Dispatch a command to a knowledge base agent.
        
        Args:
            kb_name: Name of the knowledge base
            command: Command to execute
            
        Returns:
            Command execution result
        """
        try:
            result = await self.cia.execute_command(kb_name, command)
            return result
        except Exception as e:
            logger.error(f"Error dispatching to KB agent: {e}")
            return {
                "success": False,
                "error": f"Error dispatching to KB agent: {str(e)}"
            }
    
    async def dispatch_query_to_all_kbs(self, query: str, max_results_per_kb: int = 3) -> Dict[str, Any]:
        # Advanced security check with logging
        if not self.security_check(query):
            logger.error("Security check failed for query")
            return {
                "success": False,
                "error": "Security check failed"
            }
        """
        Dispatch a search query to all knowledge bases and aggregate the results.
        
        Args:
            query: Search query
            max_results_per_kb: Maximum number of results per knowledge base
            
        Returns:
            Aggregated search results from all knowledge bases
        """
        kb_list = self.dispatcher.list_knowledge_bases()
        
        # Create async tasks for each knowledge base
        tasks = []
        for kb_info in kb_list:
            task = self.dispatcher.search_knowledge_base(kb_info["name"], query)
            tasks.append(task)
        
        # Real-time monitoring setup
        logger.info("Starting real-time monitoring of knowledge base queries")

        # Execute all search tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate and process results
        aggregated_results = []
        
        for i, result in enumerate(results):
            kb_name = kb_list[i]["name"] if i < len(kb_list) else "unknown"
            
            try:
                if isinstance(result, Exception):
                    logger.warning(f"Error searching knowledge base {kb_name}: {result}")
                    continue
                
                if result.get("success") and "data" in result:
                    kb_results = result["data"]
                    
                    # Limit results per knowledge base
                    if isinstance(kb_results, list):
                        kb_results = kb_results[:max_results_per_kb]
                    
                    # Add source information
                    for item in kb_results if isinstance(kb_results, list) else [kb_results]:
                        if isinstance(item, dict):
                            item["source_kb"] = kb_name
                            aggregated_results.append(item)
            except Exception as e:
                logger.error(f"Error processing results from {kb_name}: {e}")
        
        return {
            "success": True,
            "query": query,
            "total_results": len(aggregated_results),
            "results": aggregated_results
        }
        """
        Dispatch a search query to all knowledge bases and aggregate the results.
        
        Args:
            query: Search query
            max_results_per_kb: Maximum number of results per knowledge base
            
        Returns:
            Aggregated search results from all knowledge bases
        """
        kb_list = self.dispatcher.list_knowledge_bases()
        
        # Create async tasks for each knowledge base
        tasks = []
        for kb_info in kb_list:
            task = self.dispatcher.search_knowledge_base(kb_info["name"], query)
            tasks.append(task)
        
        # Execute all search tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate and process results
        aggregated_results = []
        
        for i, result in enumerate(results):
            kb_name = kb_list[i]["name"] if i < len(kb_list) else "unknown"
            
            try:
                if isinstance(result, Exception):
                    logger.warning(f"Error searching knowledge base {kb_name}: {result}")
                    continue
                
                if result.get("success") and "data" in result:
                    kb_results = result["data"]
                    
                    # Limit results per knowledge base
                    if isinstance(kb_results, list):
                        kb_results = kb_results[:max_results_per_kb]
                    
                    # Add source information
                    for item in kb_results if isinstance(kb_results, list) else [kb_results]:
                        if isinstance(item, dict):
                            item["source_kb"] = kb_name
                            aggregated_results.append(item)
            except Exception as e:
                logger.error(f"Error processing results from {kb_name}: {e}")
        
        return {
            "success": True,
            "query": query,
            "total_results": len(aggregated_results),
            "results": aggregated_results
        }
    
    def get_available_knowledge_bases(self) -> List[Dict[str, Any]]:
        """
        Get a list of all available knowledge bases.
        
        Returns:
            List of knowledge base information
        """
        return self.dispatcher.list_knowledge_bases()
    
    async def get_kb_data(self, kb_name: str, entry_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get knowledge base data - either the full knowledge base or a specific entry.
        
        Args:
            kb_name: Name of the knowledge base
            entry_id: Optional ID of a specific entry
            
        Returns:
            Knowledge base data
        """
        if entry_id is not None:
            return await self.dispatcher.execute_kb_command(kb_name, f"get_entry {entry_id}")
        else:
            # First get info about the KB
            info_result = await self.dispatcher.get_kb_info(kb_name)
            
            # Then get entries (limited to 10 by default)
            entries_result = await self.dispatcher.list_kb_entries(kb_name)
            
            return {
                "success": True,
                "kb_name": kb_name,
                "info": info_result.get("data", {}),
                "entries": entries_result.get("data", [])
            }

# Create a global connector instance for easy import
connector = KnowledgeBaseConnector()

async def main():
    """Test the knowledge base connector"""
    # List available knowledge bases
    kb_list = connector.get_available_knowledge_bases()
    print(f"Found {len(kb_list)} knowledge bases:")
    for kb in kb_list[:5]:  # Show first 5 for brevity
        print(f"- {kb['name']}")
    
    if kb_list:
        # Test with first knowledge base
        kb_name = kb_list[0]["name"]
        print(f"\nTesting with knowledge base: {kb_name}")
        
        # Get KB data
        kb_data = await connector.get_kb_data(kb_name)
        print(f"Knowledge base data: {kb_data}")
        
        # Test search across all KBs
        search_results = await connector.dispatch_query_to_all_kbs("science")
        print(f"Search results: {len(search_results.get('results', []))} matches found")

if __name__ == "__main__":
    asyncio.run(main())
