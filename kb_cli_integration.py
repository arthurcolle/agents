import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import importlib
import inspect

# Import the knowledge base connector
from kb_agent_connector import connector as kb_connector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kb_cli_integration")

class KnowledgeBaseIntegration:
    """
    Integration class for patching the CLIAgent with enhanced knowledge base capabilities.
    This class adds methods to CLIAgent to use the knowledge base dispatcher system.
    """
    
    @staticmethod
    def patch_cli_agent():
        """
        Patch the CLIAgent class with enhanced knowledge base capabilities.
        This adds new methods and overrides existing methods to use the knowledge base dispatcher.
        """
        try:
            # Dynamically import the CLIAgent class
            cli_agent_module = importlib.import_module("cli_agent")
            
            # Find the CLIAgent class
            cli_agent_class = None
            for name, obj in inspect.getmembers(cli_agent_module):
                if inspect.isclass(obj) and hasattr(obj, '_init_knowledge_bases'):
                    cli_agent_class = obj
                    break
            
            if not cli_agent_class:
                logger.error("CLIAgent class not found")
                return False
            
            # Patch the CLIAgent methods
            KnowledgeBaseIntegration._patch_init_knowledge_bases(cli_agent_class)
            KnowledgeBaseIntegration._patch_list_knowledge_bases(cli_agent_class)
            KnowledgeBaseIntegration._patch_search_knowledge_base(cli_agent_class)
            KnowledgeBaseIntegration._add_dispatch_to_kb_agent(cli_agent_class)
            
            logger.info("CLIAgent successfully patched with enhanced knowledge base capabilities")
            return True
            
        except Exception as e:
            logger.error(f"Error patching CLIAgent: {e}")
            return False
    
    @staticmethod
    def _patch_init_knowledge_bases(cli_agent_class):
        """
        Patch the _init_knowledge_bases method of CLIAgent.
        This keeps the original functionality but adds integration with the knowledge base dispatcher.
        """
        original_init_kb = getattr(cli_agent_class, '_init_knowledge_bases', None)
        
        async def enhanced_init_knowledge_bases(self, knowledge_base_dir=None):
            """Enhanced method to initialize knowledge bases with dispatcher integration"""
            # Call the original method if it exists
            if original_init_kb:
                await original_init_kb(self, knowledge_base_dir)
            
            # Add the knowledge base dispatcher integration
            self.kb_connector = kb_connector
            
            # Log available knowledge bases from the dispatcher
            kb_list = self.kb_connector.get_available_knowledge_bases()
            logger.info(f"Knowledge base dispatcher loaded {len(kb_list)} knowledge bases")
            
            # Add a reference to the knowledge base connector in the agent
            if hasattr(self, 'perceptual_memory') and isinstance(self.perceptual_memory, dict):
                self.perceptual_memory['knowledge_base_connector'] = self.kb_connector
            
            return True
        
        # Replace the original method with our enhanced version
        setattr(cli_agent_class, '_init_knowledge_bases', enhanced_init_knowledge_bases)
    
    @staticmethod
    def _patch_list_knowledge_bases(cli_agent_class):
        """
        Patch the _list_knowledge_bases method of CLIAgent.
        This keeps the original functionality but adds results from the knowledge base dispatcher.
        """
        original_list_kb = getattr(cli_agent_class, '_list_knowledge_bases', None)
        
        async def enhanced_list_knowledge_bases(self):
            """Enhanced method to list knowledge bases including from dispatcher"""
            # Call the original method if it exists
            original_results = {}
            if original_list_kb:
                original_results = await original_list_kb(self)
            
            # Get results from the knowledge base dispatcher
            dispatcher_kb_list = self.kb_connector.get_available_knowledge_bases()
            
            # Combine the results
            combined_results = {
                "success": True,
                "message": f"Found {len(dispatcher_kb_list)} knowledge bases",
                "data": dispatcher_kb_list
            }
            
            # If original returned results, merge them
            if original_results and original_results.get("success") and "data" in original_results:
                combined_results["message"] = f"Found {len(original_results['data']) + len(dispatcher_kb_list)} knowledge bases"
                combined_results["data"] = original_results["data"] + dispatcher_kb_list
            
            return combined_results
        
        # Replace the original method with our enhanced version
        setattr(cli_agent_class, '_list_knowledge_bases', enhanced_list_knowledge_bases)
    
    @staticmethod
    def _patch_search_knowledge_base(cli_agent_class):
        """
        Patch the _search_knowledge_base method of CLIAgent.
        This keeps the original functionality but adds results from the knowledge base dispatcher.
        """
        original_search_kb = getattr(cli_agent_class, '_search_knowledge_base', None)
        
        async def enhanced_search_knowledge_base(self, kb_name, query):
            """Enhanced method to search knowledge bases including from dispatcher"""
            # For "all" knowledge bases, use the dispatcher's method to search all KBs
            if kb_name.lower() == "all":
                search_results = await self.kb_connector.dispatch_query_to_all_kbs(query)
                return search_results
            
            # Check if the knowledge base is available in the dispatcher
            kb_list = self.kb_connector.get_available_knowledge_bases()
            kb_names = [kb["name"] for kb in kb_list]
            
            if kb_name in kb_names:
                # Use the dispatcher to search this KB
                search_result = await self.kb_connector.dispatch_to_kb_agent(kb_name, f"search {query}")
                return search_result
            
            # If not found in dispatcher, use the original method
            if original_search_kb:
                return await original_search_kb(self, kb_name, query)
            
            # If original method doesn't exist, return error
            return {
                "success": False,
                "error": f"Knowledge base not found: {kb_name}"
            }
        
        # Replace the original method with our enhanced version
        setattr(cli_agent_class, '_search_knowledge_base', enhanced_search_knowledge_base)
    
    @staticmethod
    def _add_dispatch_to_kb_agent(cli_agent_class):
        """
        Add a new method to CLIAgent for dispatching commands to knowledge base agents.
        """
        async def dispatch_to_kb_agent(self, kb_name, command):
            """Dispatch a command to a knowledge base agent"""
            return await self.kb_connector.dispatch_to_kb_agent(kb_name, command)
        
        # Add the new method to the class
        setattr(cli_agent_class, 'dispatch_to_kb_agent', dispatch_to_kb_agent)

# Function to apply the patches
def apply_kb_integration_patches():
    """Apply the knowledge base integration patches to the CLIAgent class"""
    return KnowledgeBaseIntegration.patch_cli_agent()

if __name__ == "__main__":
    # Test the patching
    success = apply_kb_integration_patches()
    print(f"Knowledge base integration patching {'succeeded' if success else 'failed'}")
    
    # Try to access the patched CLIAgent
    try:
        cli_agent_module = importlib.import_module("cli_agent")
        patched_methods = []
        
        for name, obj in inspect.getmembers(cli_agent_module):
            if inspect.isclass(obj):
                for method_name, method in inspect.getmembers(obj):
                    if method_name in ['_init_knowledge_bases', '_list_knowledge_bases', 
                                      '_search_knowledge_base', 'dispatch_to_kb_agent']:
                        patched_methods.append(method_name)
        
        print(f"Patched methods found: {patched_methods}")
    except Exception as e:
        print(f"Error testing patches: {e}")