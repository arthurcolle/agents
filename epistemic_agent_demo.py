#!/usr/bin/env python3
"""
Epistemic Agent Demo - Demonstrates how an AI agent can use epistemic tools

This script shows how an AI agent can use the epistemic knowledge system
to store, retrieve, and reason with knowledge without having to implement
these capabilities in its own code.
"""

import os
import json
import time
import argparse
import logging
from pprint import pprint

# Import epistemic tools
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
    create_temporal_snapshot,
    compare_snapshots
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agent-demo")


class EpistemicAgent:
    """
    Demo agent that leverages epistemic tools for knowledge management
    
    This simulates an AI agent that uses the epistemic tools to manage knowledge
    without implementing knowledge management itself.
    """
    
    def __init__(self, knowledge_db_path: str = "./knowledge/agent_demo.db"):
        """Initialize the agent with a knowledge database"""
        self.name = "EpistemoBot"
        
        # Initialize the knowledge system
        result = initialize_knowledge_system(knowledge_db_path)
        
        if result["status"] != "initialized":
            raise RuntimeError(f"Failed to initialize knowledge system: {result}")
        
        logger.info(f"Agent initialized with knowledge system at {knowledge_db_path}")
        logger.info(f"Initial knowledge count: {result['unit_count']}")
        
        # Create a baseline snapshot
        self.baseline_snapshot = create_temporal_snapshot("baseline")["snapshot_id"]
    
    def learn(self, content: str, source: str, confidence: float = 0.8, domain: str = None):
        """Learn new information by storing it in the knowledge system"""
        logger.info(f"Learning: {content[:50]}...")
        
        result = store_knowledge(
            content=content,
            source=source,
            confidence=confidence,
            domain=domain
        )
        
        if result["status"] == "success":
            logger.info(f"Learned new knowledge with ID: {result['id']}")
            
            # Check for contradictions
            if result["contradictions"]:
                logger.warning(f"Found {len(result['contradictions'])} contradictions with existing knowledge")
        else:
            logger.error(f"Failed to learn: {result.get('error', 'unknown error')}")
        
        return result
    
    def answer_question(self, question: str):
        """Answer a question by querying the knowledge system"""
        logger.info(f"Answering question: {question}")
        
        # Query the knowledge system
        results = query_knowledge(
            query=question,
            reasoning_depth=1,
            min_confidence=0.2
        )
        
        if results["status"] != "success" or not results.get("direct_results"):
            return "I don't have enough information to answer that question."
        
        # Formulate an answer based on the results
        answer = self._formulate_answer(question, results)
        
        return answer
    
    def solve_problem(self, problem: str):
        """Solve a problem using multi-step reasoning"""
        logger.info(f"Solving problem: {problem}")
        
        # Create a reasoning workspace
        workspace = create_reasoning_workspace(f"Solve: {problem}")
        
        if workspace["status"] != "success":
            return f"Failed to create reasoning workspace: {workspace.get('error', 'unknown error')}"
        
        workspace_id = workspace["workspace_id"]
        
        try:
            # Step 1: Gather relevant knowledge
            logger.info("Step 1: Gathering relevant knowledge")
            
            # Simulate AI retrieving relevant knowledge
            query_result = query_knowledge(problem)
            
            # Record this step
            workspace_add_step(
                workspace_id=workspace_id,
                step_type="information_gathering",
                content=f"Gathered knowledge related to: {problem}"
            )
            
            # Step 2: Analyze the problem
            logger.info("Step 2: Analyzing the problem")
            
            # Record this step
            workspace_add_step(
                workspace_id=workspace_id,
                step_type="analysis",
                content=f"The key aspects of this problem are: complexity, requirements, constraints"
            )
            
            # Step 3: Generate possible solutions
            logger.info("Step 3: Generating solutions")
            
            # Record this step
            workspace_add_step(
                workspace_id=workspace_id,
                step_type="solution_generation",
                content="Generated three potential solution approaches: A, B, and C"
            )
            
            # Step 4: Evaluate solutions
            logger.info("Step 4: Evaluating solutions")
            
            # Record this step
            workspace_add_step(
                workspace_id=workspace_id,
                step_type="evaluation",
                content="Evaluated solutions against criteria: Solution B is optimal"
            )
            
            # Step 5: Formulate final solution
            logger.info("Step 5: Formulating final solution")
            
            # Derive new knowledge
            solution = "The optimal solution is to implement approach B with modifications X and Y."
            
            workspace_derive_knowledge(
                workspace_id=workspace_id,
                content=solution,
                confidence=0.85
            )
            
            # Commit the derived knowledge
            commit_result = workspace_commit_knowledge(workspace_id)
            
            if commit_result["status"] == "success":
                logger.info(f"Committed {commit_result['committed_count']} knowledge units")
            
            # Get the full reasoning chain
            chain = workspace_get_chain(workspace_id)
            
            return {
                "solution": solution,
                "reasoning_steps": len(chain["chain"]["steps"]),
                "confidence": 0.85
            }
            
        except Exception as e:
            logger.error(f"Error during problem solving: {e}")
            return f"Failed to solve problem: {str(e)}"
    
    def review_knowledge_growth(self):
        """Review how the agent's knowledge has grown since baseline"""
        logger.info("Reviewing knowledge growth")
        
        # Create a current snapshot
        current_snapshot = create_temporal_snapshot("current")
        
        if current_snapshot["status"] != "success":
            return "Failed to create current knowledge snapshot"
        
        # Compare with baseline
        diff = compare_snapshots(self.baseline_snapshot, current_snapshot["snapshot_id"])
        
        if diff["status"] != "success":
            return "Failed to compare knowledge snapshots"
        
        # Extract the difference information
        added_count = diff["diff"]["added_count"]
        
        return f"Knowledge has grown by {added_count} units since initialization"
    
    def _formulate_answer(self, question: str, results: dict) -> str:
        """Formulate an answer based on query results"""
        # Simple answer formulation for demo
        if not results.get("direct_results"):
            return "I don't have specific information to answer that question."
        
        # Get the most relevant result
        top_result = results["direct_results"][0]
        
        # Extract the content
        content = top_result.get("content", "")
        
        # Simple answer formatting
        if len(content) > 200:
            content = content[:200] + "..."
        
        # Add confidence indicator
        confidence = results.get("confidence", 0)
        confidence_indicator = "low" if confidence < 0.4 else "medium" if confidence < 0.7 else "high"
        
        return f"Based on my knowledge ({confidence_indicator} confidence): {content}"
    
    def shutdown(self):
        """Shutdown the agent and knowledge system"""
        logger.info("Shutting down agent")
        shutdown_result = shutdown_knowledge_system()
        
        if shutdown_result["status"] == "shutdown_complete":
            logger.info("Knowledge system shut down successfully")
        else:
            logger.error(f"Error shutting down knowledge system: {shutdown_result}")


def main():
    """Main function to run the agent demo"""
    parser = argparse.ArgumentParser(description="Epistemic Agent Demo")
    parser.add_argument("--db-path", type=str, default="./knowledge/agent_demo.db",
                      help="Path to the knowledge database")
    args = parser.parse_args()
    
    try:
        # Initialize agent
        agent = EpistemicAgent(args.db_path)
        
        # Demo sequence
        print("\n1. Learning initial knowledge...")
        agent.learn(
            "Quantum computing uses quantum bits (qubits) that can exist in superposition states.",
            "quantum_textbook",
            0.95,
            "quantum_computing"
        )
        
        agent.learn(
            "Neural networks consist of layers of interconnected nodes modeling biological neurons.",
            "ai_textbook",
            0.9,
            "machine_learning"
        )
        
        agent.learn(
            "Quantum machine learning combines quantum computing with machine learning techniques.",
            "research_paper",
            0.85,
            "quantum_ml"
        )
        
        print("\n2. Answering a question...")
        answer = agent.answer_question("How does quantum computing work?")
        print(f"Answer: {answer}")
        
        print("\n3. Solving a problem...")
        solution = agent.solve_problem("How to implement a quantum neural network?")
        print("Solution:")
        pprint(solution)
        
        print("\n4. Reviewing knowledge growth...")
        growth = agent.review_knowledge_growth()
        print(growth)
        
        # Shutdown
        agent.shutdown()
        
    except Exception as e:
        logger.error(f"Error in demo: {e}")
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    main()