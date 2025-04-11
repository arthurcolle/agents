import os
import json
import glob
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from dynamic_agents import DynamicAgent, AgentContext, KnowledgeBaseAgent, registry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knowledge_base_dispatcher")

class KnowledgeBaseDispatcher:
    """
    Dispatcher for managing access to knowledge bases via specialized agents.
    Creates and manages sub-agents for each knowledge base in the knowledge_bases directory.
    """
    
    def __init__(self, knowledge_base_dir: str = "knowledge_bases"):
        """
        Initialize the knowledge base dispatcher.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base files
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.kb_agents = {}
        self._initialize_kb_agents()
    
    def _initialize_kb_agents(self):
        """Scan knowledge base directory and create agents for each knowledge base"""
        kb_path = Path(self.knowledge_base_dir)
        if not kb_path.exists() or not kb_path.is_dir():
            logger.warning(f"Knowledge base directory not found: {self.knowledge_base_dir}")
            return
        
        # Find all JSON files in the knowledge_bases directory
        kb_files = list(kb_path.glob("*.json"))
        logger.info(f"Found {len(kb_files)} knowledge base files")
        
        # Create an agent for each knowledge base file
        for kb_file in kb_files:
            try:
                # Read the knowledge base file
                with open(kb_file, 'r', encoding='utf-8') as f:
                    kb_data = json.load(f)
                
                # Get knowledge base name
                kb_name = kb_file.stem
                kb_id = f"kb_{kb_name}"
                
                # Create a context for this knowledge base
                context = AgentContext(agent_id=kb_id)
                context.set_variable("kb_name", kb_name)
                context.set_variable("kb_path", str(kb_file.parent))
                context.set_variable("kb_file", str(kb_file))
                
                # Add metadata from the KB file if available
                if isinstance(kb_data, dict):
                    if "name" in kb_data:
                        kb_name = kb_data["name"]
                        context.set_variable("kb_name", kb_name)
                    
                    # Store content or entries count
                    if "content" in kb_data:
                        context.set_variable("kb_content", kb_data["content"])
                        context.set_variable("kb_entries", 1)
                    elif isinstance(kb_data, list):
                        context.set_variable("kb_entries", len(kb_data))
                
                # Create a specialized agent for this knowledge base
                agent = registry.create_agent(kb_id, "knowledge_base")
                self.kb_agents[kb_name] = {
                    "agent": agent,
                    "context": context,
                    "file_path": str(kb_file)
                }
                
                logger.info(f"Created agent for knowledge base: {kb_name}")
                
            except Exception as e:
                logger.error(f"Error creating agent for {kb_file.name}: {e}")
    
    def get_kb_agent(self, kb_name: str) -> Optional[Tuple[DynamicAgent, AgentContext]]:
        """
        Get a knowledge base agent by name.
        
        Args:
            kb_name: Name of the knowledge base
            
        Returns:
            Tuple of (agent, context) if found, None otherwise
        """
        kb_info = self.kb_agents.get(kb_name)
        if not kb_info:
            return None
        
        return (kb_info["agent"], kb_info["context"])
    
    def list_knowledge_bases(self) -> List[Dict[str, Any]]:
        """
        List all available knowledge bases.
        
        Returns:
            List of knowledge base information
        """
        kb_list = []
        for kb_name, kb_info in self.kb_agents.items():
            kb_list.append({
                "name": kb_name,
                "agent_id": kb_info["agent"].agent_id,
                "file_path": kb_info["file_path"]
            })
        
        return kb_list
    
    async def execute_kb_command(self, kb_name: str, command: str) -> Dict[str, Any]:
        """
        Execute a command on a knowledge base agent.
        
        Args:
            kb_name: Name of the knowledge base
            command: Command to execute
            
        Returns:
            Command execution result
        """
        agent_info = self.get_kb_agent(kb_name)
        if not agent_info:
            return {
                "success": False,
                "error": f"Knowledge base not found: {kb_name}"
            }
        
        agent, context = agent_info
        result = await agent.execute(command, context)
        
        return result
    
    async def search_knowledge_base(self, kb_name: str, query: str) -> Dict[str, Any]:
        """
        Search a specific knowledge base.
        
        Args:
            kb_name: Name of the knowledge base
            query: Search query
            
        Returns:
            Search results
        """
        return await self.execute_kb_command(kb_name, f"search {query}")
    
    async def get_kb_info(self, kb_name: str) -> Dict[str, Any]:
        """
        Get information about a knowledge base.
        
        Args:
            kb_name: Name of the knowledge base
            
        Returns:
            Knowledge base information
        """
        return await self.execute_kb_command(kb_name, "info")
    
    async def list_kb_entries(self, kb_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        List entries in a knowledge base.
        
        Args:
            kb_name: Name of the knowledge base
            limit: Maximum number of entries to return
            
        Returns:
            List of knowledge base entries
        """
        return await self.execute_kb_command(kb_name, f"list_entries {limit}")

# Create a global instance for reuse
dispatcher = KnowledgeBaseDispatcher()

async def main():
    """Test the knowledge base dispatcher"""
    kb_list = dispatcher.list_knowledge_bases()
    print(f"Found {len(kb_list)} knowledge bases:")
    for kb in kb_list[:5]:  # Show first 5 for brevity
        print(f"- {kb['name']}")
    
    if kb_list:
        # Test with first knowledge base
        kb_name = kb_list[0]["name"]
        print(f"\nTesting with knowledge base: {kb_name}")
        
        # Get info
        info_result = await dispatcher.get_kb_info(kb_name)
        print(f"Info: {info_result}")
        
        # List entries
        entries_result = await dispatcher.list_kb_entries(kb_name, 3)
        print(f"Entries: {entries_result}")
        
        # Search
        search_result = await dispatcher.search_knowledge_base(kb_name, "definition")
        print(f"Search results: {search_result}")

if __name__ == "__main__":
    asyncio.run(main())