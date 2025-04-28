#!/usr/bin/env python3
"""
Multi-Knowledge Base Agent System - A framework for solving real-world problems
by orchestrating multiple specialized agents across different knowledge domains.

This system generalizes the pattern to allow multi-agent collaboration for any
given real-world problem by leveraging the extensive collection of knowledge bases.
"""

import os
import json
import asyncio
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import time
import uuid

# Import core components
from knowledge_base_dispatcher import KnowledgeBaseDispatcher
from dynamic_agents import DynamicAgent, AgentContext, KnowledgeBaseAgent, registry
from central_interaction_agent import CentralInteractionAgent
from pubsub_service import PubSubService
from agent_collective import AgentCollective, CollectiveRole, CollectiveAgent
from epistemic_core import EpistemicUnit, KnowledgeAPI, ReasoningWorkspace
from epistemic_tools import (
    initialize_knowledge_system,
    shutdown_knowledge_system,
    store_knowledge,
    query_knowledge,
    create_reasoning_workspace,
    workspace_add_step,
    workspace_derive_knowledge,
    workspace_commit_knowledge
)
from epistemic_long_context import IncrementalReasoner, RecursiveDecomposer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_kb_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multi-kb-agent")


class MultiKBAgentSystem:
    """
    A system that orchestrates multiple knowledge-base specialized agents to solve
    complex real-world problems requiring expertise across multiple domains.
    """
    
    def __init__(self, epistemic_db_path: str = "./knowledge/epistemic_multi_kb.db"):
        """
        Initialize the multi-knowledge base agent system.
        
        Args:
            epistemic_db_path: Path to the epistemic knowledge database
        """
        # Initialize the epistemic knowledge system
        self.epistemic_db_path = epistemic_db_path
        Path(epistemic_db_path).parent.mkdir(parents=True, exist_ok=True)
        initialize_knowledge_system(epistemic_db_path)
        self.knowledge_api = KnowledgeAPI(epistemic_db_path)
        
        # Initialize the knowledge base dispatcher
        self.kb_dispatcher = KnowledgeBaseDispatcher()
        
        # Initialize the pubsub service for agent communication
        self.pubsub = PubSubService()
        
        # Initialize the agent collective
        self.collective = AgentCollective(pubsub_service=self.pubsub)
        
        # Initialize the central interaction agent
        self.central_agent = CentralInteractionAgent(self.kb_dispatcher)
        
        # Workspace for overall problem solving
        self.main_workspace = create_reasoning_workspace("Multi-KB Problem Solving")
        self.main_workspace_id = self.main_workspace["workspace_id"]
        
        # Track active knowledge domains and reasoning processes
        self.active_domains = set()
        self.domain_relevance = {}
        self.reasoners = {}
        self.decomposers = {}
        
        # Problem tracking
        self.current_problem = None
        self.problem_decomposition = {}
        
        logger.info("MultiKBAgentSystem initialized")
    
    async def setup_domain_agents(self):
        """Set up specialized agents for different knowledge domains"""
        # Get the list of available knowledge bases
        kb_list = self.kb_dispatcher.list_knowledge_bases()
        logger.info(f"Found {len(kb_list)} knowledge bases")
        
        # Create collective agents for each knowledge domain
        for kb_info in kb_list:
            kb_name = kb_info["name"]
            agent_id = f"domain_{kb_name}"
            
            # Create a specialized agent for this domain
            role = CollectiveRole(
                name=f"{kb_name.replace('_', ' ').title()} Specialist",
                description=f"Expert in {kb_name.replace('_', ' ')} domain knowledge",
                capabilities=["domain_expertise", "knowledge_retrieval", "analysis"]
            )
            
            # Register with the collective
            await self.collective.add_agent(
                CollectiveAgent(
                    agent_id=agent_id,
                    role=role,
                    capabilities=[
                        "search_knowledge",
                        "analyze_domain",
                        "answer_domain_questions",
                        "identify_domain_relevance"
                    ]
                )
            )
            
            logger.info(f"Created collective agent for domain: {kb_name}")
        
        # Set up core system agents
        await self._setup_system_agents()
        
        # Connect to pubsub channels
        await self.pubsub.connect()
        await self.pubsub.subscribe("problem_solving", self._handle_problem_solving_message)
        await self.pubsub.subscribe("domain_insights", self._handle_domain_insights)
        await self.pubsub.subscribe("solution_proposals", self._handle_solution_proposals)
        
        logger.info("Domain agents setup completed")
    
    async def _setup_system_agents(self):
        """Set up the core system agents for orchestration and analysis"""
        # Coordinator agent
        coordinator_role = CollectiveRole(
            name="Problem Coordinator",
            description="Orchestrates the overall problem-solving process",
            capabilities=["coordination", "delegation", "synthesis"]
        )
        await self.collective.add_agent(
            CollectiveAgent(
                agent_id="coordinator",
                role=coordinator_role,
                capabilities=[
                    "decompose_problems", 
                    "assign_subproblems",
                    "synthesize_solutions",
                    "manage_workflow"
                ]
            )
        )
        
        # Research strategist
        research_role = CollectiveRole(
            name="Research Strategist",
            description="Plans and coordinates knowledge gathering across domains",
            capabilities=["research", "information_synthesis", "query_formulation"]
        )
        await self.collective.add_agent(
            CollectiveAgent(
                agent_id="researcher",
                role=research_role,
                capabilities=[
                    "formulate_queries",
                    "evaluate_information",
                    "identify_knowledge_gaps",
                    "synthesize_findings"
                ]
            )
        )
        
        # Solution architect
        architect_role = CollectiveRole(
            name="Solution Architect",
            description="Designs and refines solution approaches",
            capabilities=["solution_design", "integration", "evaluation"]
        )
        await self.collective.add_agent(
            CollectiveAgent(
                agent_id="architect",
                role=architect_role,
                capabilities=[
                    "design_solutions",
                    "evaluate_approaches",
                    "integrate_components",
                    "ensure_coherence"
                ]
            )
        )
        
        # Critical evaluator
        evaluator_role = CollectiveRole(
            name="Critical Evaluator",
            description="Critically evaluates proposed solutions",
            capabilities=["critical_thinking", "evaluation", "verification"]
        )
        await self.collective.add_agent(
            CollectiveAgent(
                agent_id="evaluator",
                role=evaluator_role,
                capabilities=[
                    "identify_weaknesses",
                    "validate_solutions",
                    "propose_improvements",
                    "risk_assessment"
                ]
            )
        )
        
        logger.info("System agents setup completed")
    
    async def solve_problem(self, problem_statement: str) -> Dict[str, Any]:
        """
        Solve a complex problem by coordinating multiple knowledge-base agents.
        
        Args:
            problem_statement: The problem to solve
            
        Returns:
            Solution details and reasoning process
        """
        self.current_problem = problem_statement
        logger.info(f"Starting to solve problem: {problem_statement}")
        
        # Record problem in workspace
        workspace_add_step(
            self.main_workspace_id,
            "problem_definition",
            f"PROBLEM STATEMENT: {problem_statement}"
        )
        
        # Create a decomposer for this problem
        decomposer_id = f"decomposer_{int(time.time())}"
        decomposer = RecursiveDecomposer(f"{self.epistemic_db_path}_decomp")
        self.decomposers[decomposer_id] = decomposer
        
        # Decompose the problem
        decomposition = decomposer.decompose_problem(problem_statement)
        self.problem_decomposition = decomposition
        
        workspace_add_step(
            self.main_workspace_id,
            "problem_decomposition",
            f"PROBLEM DECOMPOSED INTO {decomposition['subproblem_count']} SUBPROBLEMS:\n" + 
            "\n".join([f"- {sp['statement']}" for sp in decomposition['subproblems']])
        )
        
        # Identify relevant knowledge domains for each subproblem
        await self._identify_relevant_domains(decomposition['subproblems'])
        
        # Publish the problem to the problem solving channel
        await self.pubsub.publish("problem_solving", {
            "action": "new_problem",
            "problem_statement": problem_statement,
            "subproblems": decomposition['subproblems'],
            "domain_relevance": self.domain_relevance
        })
        
        # Process the problem incrementally
        solution_result = await self._process_problem_incrementally(decomposer_id)
        
        workspace_add_step(
            self.main_workspace_id,
            "solution_synthesis",
            f"SOLUTION SYNTHESIS:\n{solution_result['conclusion']}"
        )
        
        # Store the solution in the epistemic system
        solution_unit = EpistemicUnit(
            content=solution_result['conclusion'],
            confidence=0.85,
            source="Multi-KB Agent System",
            evidence=f"Collaborative solution from multiple domain specialists"
        )
        store_knowledge(solution_unit)
        
        # Commit the workspace knowledge
        workspace_commit_knowledge(self.main_workspace_id)
        
        # Close the decomposer
        decomposer.close()
        
        return {
            "problem": problem_statement,
            "solution": solution_result['conclusion'],
            "subproblems": decomposition['subproblems'],
            "domains_utilized": list(self.active_domains),
            "reasoning_trace": workspace_get_chain(self.main_workspace_id)["chain"]
        }
    
    async def _identify_relevant_domains(self, subproblems: List[Dict[str, Any]]):
        """
        Identify which knowledge domains are most relevant for each subproblem.
        
        Args:
            subproblems: List of subproblems
        """
        # Get all available knowledge bases
        kb_list = self.kb_dispatcher.list_knowledge_bases()
        
        # For each subproblem, analyze which domains are most relevant
        for i, subproblem in enumerate(subproblems):
            subproblem_id = f"subproblem_{i+1}"
            subproblem_statement = subproblem["statement"]
            
            logger.info(f"Identifying relevant domains for subproblem: {subproblem_statement[:50]}...")
            
            # Initialize relevance scores for this subproblem
            self.domain_relevance[subproblem_id] = {}
            
            # Query the epistemic knowledge system for initial insights
            query_result = query_knowledge(subproblem_statement)
            
            # Get domain insights from the central interaction agent
            domain_insights = await self.central_agent.analyze_problem_domains(subproblem_statement)
            
            # Combine the insights to determine domain relevance
            for kb_info in kb_list:
                kb_name = kb_info["name"]
                
                # Check relevance from domain insights
                relevance_score = 0.0
                
                if kb_name in domain_insights.get("relevant_domains", {}):
                    relevance_score = domain_insights["relevant_domains"][kb_name]
                
                # Add to domain relevance if score is above threshold
                if relevance_score >= 0.3:  # Minimal relevance threshold
                    self.domain_relevance[subproblem_id][kb_name] = relevance_score
                    self.active_domains.add(kb_name)
            
            # Ensure at least 3 domains per subproblem for diverse insights
            if len(self.domain_relevance[subproblem_id]) < 3:
                # Add more domains based on general relevance
                additional_domains = await self.central_agent.get_complementary_domains(
                    subproblem_statement, 
                    list(self.domain_relevance[subproblem_id].keys()), 
                    3 - len(self.domain_relevance[subproblem_id])
                )
                
                for domain, score in additional_domains.items():
                    self.domain_relevance[subproblem_id][domain] = score
                    self.active_domains.add(domain)
            
            # Record domain relevance in workspace
            domains_str = "\n".join([
                f"- {domain}: {score:.2f}" 
                for domain, score in self.domain_relevance[subproblem_id].items()
            ])
            
            workspace_add_step(
                self.main_workspace_id,
                "domain_relevance",
                f"RELEVANT DOMAINS FOR SUBPROBLEM {i+1}:\n{domains_str}"
            )
            
            logger.info(f"Identified {len(self.domain_relevance[subproblem_id])} relevant domains for subproblem {i+1}")
    
    async def _process_problem_incrementally(self, decomposer_id: str) -> Dict[str, Any]:
        """
        Process the problem incrementally using the recursive decomposer.
        
        Args:
            decomposer_id: ID of the decomposer
            
        Returns:
            Final problem solution
        """
        decomposer = self.decomposers[decomposer_id]
        
        # Process increments until completion
        max_increments = 30  # Limit for reasonable runtime
        increments_processed = 0
        
        for i in range(max_increments):
            result = decomposer.process_next_increment()
            increments_processed += 1
            
            # Log progress
            logger.info(f"Increment {i+1}: Progress {result.get('tree_progress', 0)*100:.1f}%")
            
            # Handle the current focus
            if "focus" in result:
                focus = result["focus"]
                
                if focus["type"] == "subproblem":
                    # Get insights from domain specialists
                    await self._get_domain_insights(focus["id"], focus["content"])
                
                # Record the step in our workspace
                workspace_add_step(
                    self.main_workspace_id,
                    "processing_increment",
                    f"PROCESSING: {focus['type']} - {focus['content'][:100]}..."
                )
            
            # If complete, we're done
            if result.get("status") == "complete":
                logger.info("Problem solving complete!")
                
                # Record conclusion in workspace
                workspace_add_step(
                    self.main_workspace_id,
                    "solution",
                    f"FINAL SOLUTION:\n{result['conclusion']}"
                )
                
                return result
        
        logger.warning(f"Reached maximum increments ({max_increments}). Solution may be incomplete.")
        
        # Generate the best solution we have so far
        final_result = decomposer._synthesize_final_answer()
        
        return final_result
    
    async def _get_domain_insights(self, subproblem_id: str, content: str):
        """
        Get insights from domain specialist agents for a specific subproblem.
        
        Args:
            subproblem_id: ID of the subproblem
            content: Content of the subproblem
        """
        if subproblem_id not in self.domain_relevance:
            logger.warning(f"No domain relevance information for {subproblem_id}")
            return
        
        relevant_domains = self.domain_relevance[subproblem_id]
        
        if not relevant_domains:
            logger.warning(f"No relevant domains for {subproblem_id}")
            return
        
        # Collect insights from each relevant domain
        for domain, relevance in relevant_domains.items():
            # Skip if relevance is too low
            if relevance < 0.3:
                continue
                
            logger.info(f"Getting insights from {domain} domain (relevance: {relevance:.2f})")
            
            # Query the domain knowledge base
            kb_result = await self.kb_dispatcher.search_knowledge_base(domain, content)
            
            if not kb_result.get("success", False):
                logger.warning(f"Failed to query {domain} knowledge base: {kb_result.get('error', 'Unknown error')}")
                continue
            
            # Process the domain insights
            insights = kb_result.get("results", [])
            
            if not insights:
                logger.info(f"No insights found in {domain} domain")
                continue
            
            # Record insights in the workspace
            insight_text = "\n".join([f"- {insight}" for insight in insights[:5]])
            workspace_add_step(
                self.main_workspace_id,
                "domain_insight",
                f"INSIGHTS FROM {domain.upper()} DOMAIN:\n{insight_text}"
            )
            
            # Publish the insights to the domain insights channel
            await self.pubsub.publish("domain_insights", {
                "subproblem_id": subproblem_id,
                "domain": domain,
                "relevance": relevance,
                "insights": insights
            })
            
            logger.info(f"Published {len(insights)} insights from {domain} domain")
    
    async def _handle_problem_solving_message(self, message: Dict[str, Any]):
        """
        Handle messages on the problem solving channel.
        
        Args:
            message: The message to handle
        """
        action = message.get("action")
        
        if action == "subproblem_assignment":
            subproblem_id = message.get("subproblem_id")
            agent_id = message.get("agent_id")
            logger.info(f"Subproblem {subproblem_id} assigned to agent {agent_id}")
            
        elif action == "progress_update":
            subproblem_id = message.get("subproblem_id")
            progress = message.get("progress", 0)
            logger.info(f"Subproblem {subproblem_id} progress: {progress:.1f}%")
    
    async def _handle_domain_insights(self, message: Dict[str, Any]):
        """
        Handle messages on the domain insights channel.
        
        Args:
            message: The message to handle
        """
        subproblem_id = message.get("subproblem_id")
        domain = message.get("domain")
        insights = message.get("insights", [])
        
        logger.info(f"Received {len(insights)} insights from {domain} for {subproblem_id}")
        
        # Here we would process the insights and integrate them into the solution
        # For now, we'll just acknowledge them
        
        # Publish acknowledgment
        await self.pubsub.publish("problem_solving", {
            "action": "insights_received",
            "subproblem_id": subproblem_id,
            "domain": domain,
            "insight_count": len(insights)
        })
    
    async def _handle_solution_proposals(self, message: Dict[str, Any]):
        """
        Handle messages on the solution proposals channel.
        
        Args:
            message: The message to handle
        """
        subproblem_id = message.get("subproblem_id")
        agent_id = message.get("agent_id")
        solution = message.get("solution")
        
        logger.info(f"Received solution proposal from {agent_id} for {subproblem_id}")
        
        # Record the proposal in the workspace
        workspace_add_step(
            self.main_workspace_id,
            "solution_proposal",
            f"SOLUTION PROPOSAL FROM {agent_id} FOR {subproblem_id}:\n{solution}"
        )
        
        # Here we would evaluate the proposal and potentially integrate it
        # For now, we'll just acknowledge it
        
        # Publish acknowledgment
        await self.pubsub.publish("problem_solving", {
            "action": "proposal_received",
            "subproblem_id": subproblem_id,
            "agent_id": agent_id
        })
    
    def close(self):
        """Clean up resources and shut down the system"""
        # Close all decomposers
        for decomposer_id, decomposer in self.decomposers.items():
            decomposer.close()
        
        # Close all reasoners
        for reasoner_id, reasoner in self.reasoners.items():
            reasoner.close_session()
        
        # Shut down the epistemic system
        shutdown_knowledge_system()
        
        logger.info("MultiKBAgentSystem shut down")


async def solve_example_problem():
    """Example usage of the multi-KB agent system"""
    system = MultiKBAgentSystem()
    
    try:
        # Setup the domain agents
        await system.setup_domain_agents()
        
        # Define a complex problem
        problem = (
            "Design a comprehensive urban sustainability plan that addresses transportation, "
            "energy efficiency, waste management, and social equity while accounting for "
            "climate change impacts and limited municipal budgets."
        )
        
        print("\n" + "="*80)
        print(f"SOLVING PROBLEM: {problem}")
        print("="*80 + "\n")
        
        # Solve the problem
        solution = await system.solve_problem(problem)
        
        print("\n" + "="*80)
        print("SOLUTION:")
        print(solution["solution"])
        print("\n" + "="*40)
        print(f"Domains utilized: {', '.join(solution['domains_utilized'])}")
        print(f"Subproblems addressed: {len(solution['subproblems'])}")
        print("="*80 + "\n")
        
    finally:
        # Clean up
        system.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Knowledge Base Agent System")
    parser.add_argument("--problem", type=str, help="Problem to solve")
    
    args = parser.parse_args()
    
    # Run the system
    if args.problem:
        # Custom problem from command line
        async def solve_custom_problem():
            system = MultiKBAgentSystem()
            try:
                await system.setup_domain_agents()
                solution = await system.solve_problem(args.problem)
                
                print("\n" + "="*80)
                print("SOLUTION:")
                print(solution["solution"])
                print("\n" + "="*40)
                print(f"Domains utilized: {', '.join(solution['domains_utilized'])}")
                print("="*80 + "\n")
            finally:
                system.close()
        
        asyncio.run(solve_custom_problem())
    else:
        # Example problem
        asyncio.run(solve_example_problem())