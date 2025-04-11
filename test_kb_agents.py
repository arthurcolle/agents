#!/usr/bin/env python3
import os
import sys
import json
import asyncio
import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our knowledge base components
from knowledge_base_dispatcher import KnowledgeBaseDispatcher, AgentContext, KnowledgeBaseAgent
from kb_agent_connector import KnowledgeBaseConnector

class TestKnowledgeBaseAgents(unittest.TestCase):
    """Test cases for the knowledge base agents system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temp directory for test knowledge bases
        self.test_kb_dir = Path("test_knowledge_bases")
        self.test_kb_dir.mkdir(exist_ok=True)
        
        # Create a test knowledge base file
        self.test_kb_file = self.test_kb_dir / "Test_Knowledge.json"
        test_kb_data = {
            "name": "Test Knowledge",
            "content": "This is a test knowledge base for unit testing.",
            "url": "https://example.com/test",
            "embedding": None,
            "last_updated": 1743796484.877455,
            "query_count": 0
        }
        
        with open(self.test_kb_file, 'w', encoding='utf-8') as f:
            json.dump(test_kb_data, f, indent=2)
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove test knowledge base file
        if self.test_kb_file.exists():
            self.test_kb_file.unlink()
        
        # Remove test directory
        if self.test_kb_dir.exists():
            self.test_kb_dir.rmdir()
    
    def test_dispatcher_initialization(self):
        """Test the knowledge base dispatcher initialization"""
        dispatcher = KnowledgeBaseDispatcher(knowledge_base_dir=str(self.test_kb_dir))
        
        # Assert we have our test kb
        self.assertIn("Test_Knowledge", dispatcher.kb_agents)
        
        # Get agent info
        kb_info = dispatcher.kb_agents["Test_Knowledge"]
        
        # Check agent details
        self.assertEqual(kb_info["agent"].agent_type, "knowledge_base")
        self.assertEqual(kb_info["agent"].agent_id, "kb_Test_Knowledge")
        
        # Check context
        self.assertEqual(kb_info["context"].get_variable("kb_name"), "Test Knowledge")
        self.assertEqual(kb_info["context"].get_variable("kb_content"), 
                        "This is a test knowledge base for unit testing.")
    
    def test_list_knowledge_bases(self):
        """Test listing knowledge bases"""
        dispatcher = KnowledgeBaseDispatcher(knowledge_base_dir=str(self.test_kb_dir))
        kb_list = dispatcher.list_knowledge_bases()
        
        # Check that we have our test kb
        self.assertEqual(len(kb_list), 1)
        self.assertEqual(kb_list[0]["name"], "Test_Knowledge")
        self.assertTrue(kb_list[0]["file_path"].endswith("Test_Knowledge.json"))
    
    @patch('knowledge_base_dispatcher.KnowledgeBaseAgent.execute')
    async def test_execute_kb_command(self, mock_execute):
        """Test executing a command on a knowledge base agent"""
        # Set up the mock
        mock_execute.return_value = {
            "success": True,
            "message": "Command executed successfully",
            "data": {"test": "data"}
        }
        
        # Initialize the dispatcher
        dispatcher = KnowledgeBaseDispatcher(knowledge_base_dir=str(self.test_kb_dir))
        
        # Execute a command
        result = await dispatcher.execute_kb_command("Test_Knowledge", "test_command")
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Command executed successfully")
        self.assertEqual(result["data"], {"test": "data"})
        
        # Verify the mock was called
        mock_execute.assert_called_once()
    
    @patch('knowledge_base_dispatcher.KnowledgeBaseAgent.execute')
    async def test_connector_dispatch(self, mock_execute):
        """Test the connector dispatching to a knowledge base agent"""
        # Set up the mock
        mock_execute.return_value = {
            "success": True,
            "message": "Search executed successfully",
            "data": [{"result": "Test result"}]
        }
        
        # Create a connector with a mock dispatcher
        connector = KnowledgeBaseConnector()
        connector.dispatcher = KnowledgeBaseDispatcher(knowledge_base_dir=str(self.test_kb_dir))
        
        # Dispatch a search command
        result = await connector.dispatch_to_kb_agent("Test_Knowledge", "search test query")
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Search executed successfully")
        self.assertEqual(result["data"], [{"result": "Test result"}])
        
        # Verify the mock was called
        mock_execute.assert_called_once()
    
    @patch('knowledge_base_dispatcher.KnowledgeBaseAgent.execute')
    async def test_search_all_knowledge_bases(self, mock_execute):
        """Test searching all knowledge bases"""
        # Set up the mock
        mock_execute.return_value = {
            "success": True,
            "message": "Search executed successfully",
            "data": [{"result": "Test result"}]
        }
        
        # Create a connector with a mock dispatcher
        connector = KnowledgeBaseConnector()
        connector.dispatcher = KnowledgeBaseDispatcher(knowledge_base_dir=str(self.test_kb_dir))
        
        # Dispatch a search to all KBs
        result = await connector.dispatch_query_to_all_kbs("test query")
        
        # Check the result
        self.assertTrue(result["success"])
        self.assertEqual(result["query"], "test query")
        self.assertGreaterEqual(result["total_results"], 0)
        
        # Verify the mock was called at least once
        mock_execute.assert_called()

async def run_tests():
    """Run the unit tests asynchronously"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestKnowledgeBaseAgents)
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    # Run the async tests
    asyncio.run(run_tests())