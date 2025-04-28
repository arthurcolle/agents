#!/usr/bin/env python3
"""
Test script for the epistemic knowledge management system
"""

import os
import logging
import unittest
import tempfile
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test-epistemic")

# Import core components
from epistemic_core import (
    EpistemicUnit, 
    EpistemicStore, 
    KnowledgeGraph, 
    TemporalKnowledgeState,
    KnowledgeAPI,
    ReasoningWorkspace
)

# Import tools
from epistemic_tools import (
    initialize_knowledge_system,
    shutdown_knowledge_system,
    store_knowledge,
    query_knowledge,
    explore_concept,
    create_reasoning_workspace,
    workspace_add_step,
    workspace_derive_knowledge,
    workspace_commit_knowledge,
    workspace_get_chain,
    create_relationship,
    create_temporal_snapshot
)

# Import long-context components
from epistemic_long_context import (
    IncrementalReasoner,
    RecursiveDecomposer
)


class TestEpistemicCore(unittest.TestCase):
    """Tests for the core epistemic knowledge system components"""
    
    def setUp(self):
        """Set up a test database for each test"""
        self.test_dir = tempfile.mkdtemp(prefix="epistemic_test_")
        self.db_path = os.path.join(self.test_dir, "test_epistemic.db")
        initialize_knowledge_system(self.db_path)
        logger.info(f"Test database initialized at {self.db_path}")
    
    def tearDown(self):
        """Clean up test database"""
        shutdown_knowledge_system()
        shutil.rmtree(self.test_dir)
        logger.info(f"Test database removed from {self.test_dir}")
    
    def test_epistemic_unit(self):
        """Test EpistemicUnit functionality"""
        # Create a unit
        unit = EpistemicUnit(
            content="The sky is blue due to Rayleigh scattering of sunlight.",
            confidence=0.95,
            source="Physics textbook",
            evidence="Consistent with atmospheric physics models."
        )
        
        # Test properties
        self.assertEqual(unit.content, "The sky is blue due to Rayleigh scattering of sunlight.")
        self.assertEqual(unit.confidence, 0.95)
        self.assertEqual(unit.source, "Physics textbook")
        self.assertEqual(unit.evidence, "Consistent with atmospheric physics models.")
        
        # Test JSON serialization
        unit_dict = unit.to_dict()
        self.assertIn("content", unit_dict)
        self.assertIn("confidence", unit_dict)
        self.assertIn("source", unit_dict)
        self.assertIn("evidence", unit_dict)
    
    def test_knowledge_storage_retrieval(self):
        """Test storing and retrieving knowledge"""
        # Create and store units
        unit1 = EpistemicUnit(
            content="Machine learning algorithms learn patterns from data.",
            confidence=0.92,
            source="ML Textbook",
            evidence="Demonstrated through empirical results across domains."
        )
        
        unit2 = EpistemicUnit(
            content="Deep learning is a subset of machine learning.",
            confidence=0.98,
            source="AI Research Paper",
            evidence="Based on formal definitions in the literature."
        )
        
        # Store units
        result1 = store_knowledge(unit1)
        result2 = store_knowledge(unit2)
        
        # Check storage results
        self.assertIn("unit_id", result1)
        self.assertIn("status", result1)
        self.assertEqual(result1["status"], "success")
        
        # Query for ML-related knowledge
        query_result = query_knowledge("machine learning")
        
        # Check query results
        self.assertIn("direct_results", query_result)
        self.assertGreaterEqual(len(query_result["direct_results"]), 1)
        
        # Verify content is retrieved
        found_contents = [r.get("content", "") for r in query_result["direct_results"]]
        self.assertTrue(any("machine learning" in content.lower() for content in found_contents))
    
    def test_knowledge_graph(self):
        """Test knowledge graph functionality"""
        # Create graph
        graph = KnowledgeGraph(self.db_path)
        
        # Create units
        unit1 = EpistemicUnit(
            content="AI ethics focuses on responsible AI development.",
            confidence=0.9,
            source="Ethics Guideline",
            evidence="Based on consensus principles."
        )
        
        unit2 = EpistemicUnit(
            content="Responsible AI requires fairness considerations.",
            confidence=0.85,
            source="AI Policy Framework",
            evidence="Derived from legal and ethical standards."
        )
        
        # Store units
        result1 = store_knowledge(unit1)
        result2 = store_knowledge(unit2)
        
        # Create relationship
        rel_result = create_relationship(
            source_id=result1["unit_id"],
            relation_type="implies",
            target=result2["unit_id"],
            confidence=0.8
        )
        
        self.assertEqual(rel_result["status"], "success")
        
        # Explore a concept to verify graph connections
        exploration = explore_concept("AI ethics")
        
        # Check results
        self.assertIn("relationships", exploration)
        
        # At least one relationship should be found
        self.assertGreaterEqual(len(exploration["relationships"]), 1)
    
    def test_temporal_knowledge(self):
        """Test temporal knowledge tracking"""
        # Create temporal state manager
        temp_state = TemporalKnowledgeState(self.db_path)
        
        # Create initial knowledge
        unit1 = EpistemicUnit(
            content="The best algorithm for problem X is A.",
            confidence=0.75,
            source="Research Paper 2020",
            evidence="Based on empirical comparison of 3 algorithms."
        )
        
        result1 = store_knowledge(unit1)
        
        # Create first snapshot
        snapshot1 = create_temporal_snapshot("Initial knowledge")
        
        # Update with new knowledge
        unit2 = EpistemicUnit(
            content="The best algorithm for problem X is B, not A.",
            confidence=0.85,
            source="Research Paper 2023",
            evidence="Based on comprehensive comparison of 10 algorithms."
        )
        
        result2 = store_knowledge(unit2)
        
        # Create second snapshot
        snapshot2 = create_temporal_snapshot("Updated knowledge")
        
        # Check snapshots
        snapshots = temp_state.get_snapshots()
        self.assertGreaterEqual(len(snapshots), 2)
        
        # Get diffs
        diff = temp_state.get_snapshot_diff(snapshot1, snapshot2)
        self.assertIn("added", diff)
        self.assertGreaterEqual(len(diff["added"]), 1)
    
    def test_reasoning_workspace(self):
        """Test reasoning workspace functionality"""
        # Create a workspace
        workspace_result = create_reasoning_workspace("Evaluating algorithm performance")
        
        self.assertIn("workspace_id", workspace_result)
        workspace_id = workspace_result["workspace_id"]
        
        # Add steps
        step1 = workspace_add_step(
            workspace_id,
            "observation",
            "Algorithm A has O(n^2) time complexity."
        )
        
        step2 = workspace_add_step(
            workspace_id,
            "observation",
            "Algorithm B has O(n log n) time complexity."
        )
        
        step3 = workspace_add_step(
            workspace_id,
            "reasoning",
            "For large inputs, O(n log n) is more efficient than O(n^2)."
        )
        
        # Derive knowledge
        derived = workspace_derive_knowledge(
            workspace_id,
            "Algorithm B is more efficient than Algorithm A for large inputs.",
            confidence=0.9,
            evidence="Based on time complexity analysis."
        )
        
        # Get the reasoning chain
        chain = workspace_get_chain(workspace_id)
        
        # Verify chain
        self.assertIn("chain", chain)
        self.assertIn("steps", chain["chain"])
        self.assertIn("derived_knowledge", chain["chain"])
        
        # Check steps
        self.assertEqual(len(chain["chain"]["steps"]), 3)
        
        # Check derived knowledge
        self.assertGreaterEqual(len(chain["chain"]["derived_knowledge"]), 1)
        
        # Commit the derived knowledge
        commit_result = workspace_commit_knowledge(workspace_id)
        self.assertIn("committed_count", commit_result)
        self.assertGreater(commit_result["committed_count"], 0)


class TestLongContextReasoning(unittest.TestCase):
    """Tests for the long-context reasoning functionality"""
    
    def setUp(self):
        """Set up a test database for each test"""
        self.test_dir = tempfile.mkdtemp(prefix="epistemic_long_test_")
        self.db_path = os.path.join(self.test_dir, "test_long_context.db")
        os.makedirs(os.path.join(self.test_dir, "long_context_sessions"), exist_ok=True)
        logger.info(f"Test database initialized at {self.db_path}")
    
    def tearDown(self):
        """Clean up test database"""
        shutdown_knowledge_system()
        shutil.rmtree(self.test_dir)
        logger.info(f"Test database removed from {self.test_dir}")
    
    def test_incremental_reasoner(self):
        """Test incremental reasoning functionality"""
        # Create reasoner
        reasoner = IncrementalReasoner(self.db_path)
        
        # Set a problem
        problem = "What are the tradeoffs between model size and inference speed in deep learning?"
        result = reasoner.set_problem(problem)
        
        # Verify problem was set
        self.assertEqual(result["status"], "problem_set")
        self.assertIn("subproblems", result)
        self.assertGreater(len(result["subproblems"]), 0)
        
        # Process a few increments
        increment_results = []
        for i in range(3):  # Process 3 increments
            increment_result = reasoner.process_next_increment()
            increment_results.append(increment_result)
            
            self.assertIn("status", increment_result)
            self.assertIn("progress", increment_result)
        
        # Get reasoning trace
        trace = reasoner.get_reasoning_trace()
        
        # Verify trace
        self.assertIn("steps", trace)
        self.assertGreaterEqual(len(trace["steps"]), 3)  # At least 3 steps
        
        # Close session
        close_result = reasoner.close_session()
        self.assertEqual(close_result["status"], "session_closed")
    
    def test_recursive_decomposer(self):
        """Test recursive problem decomposition"""
        # Create decomposer
        decomposer = RecursiveDecomposer(self.db_path)
        
        # Set a complex problem
        problem = (
            "How can we design AI systems that balance performance with explainability "
            "across different domains such as healthcare, finance, and autonomous vehicles?"
        )
        
        result = decomposer.decompose_problem(problem)
        
        # Verify decomposition
        self.assertIn("subproblem_count", result)
        self.assertGreater(result["subproblem_count"], 0)
        
        # Process a few increments
        for i in range(3):  # Process 3 increments
            increment_result = decomposer.process_next_increment()
            
            self.assertIn("tree_progress", increment_result)
            
            if increment_result.get("status") == "complete":
                break
        
        # Get the problem tree
        tree = decomposer.get_problem_tree()
        
        # Verify tree
        self.assertIn("root", tree)
        self.assertIn("subproblems", tree)
        
        # Close
        close_result = decomposer.close()
        self.assertEqual(close_result["status"], "all_sessions_closed")


def run_tests():
    """Run all tests for the epistemic knowledge system"""
    print("\n===== TESTING EPISTEMIC KNOWLEDGE MANAGEMENT SYSTEM =====\n")
    
    # Load and run tests
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestEpistemicCore))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestLongContextReasoning))
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # Return True if successful, False otherwise
    return test_result.wasSuccessful()


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("./knowledge", exist_ok=True)
    
    # Run tests
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed! The epistemic knowledge system is functioning correctly.")
    else:
        print("\n❌ Some tests failed. Please check the logs for details.")
    
    # Exit with appropriate code
    exit(0 if success else 1)