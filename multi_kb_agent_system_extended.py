#!/usr/bin/env python3
"""
Multi-Knowledge Base Agent System (Extended) - A framework for solving complex real-world problems
by orchestrating multiple specialized agents across different knowledge domains with enhanced
capabilities including autonomous learning, consensus mechanisms, and emergent insights.

This system generalizes the pattern to allow multi-agent collaboration for any
given real-world problem by leveraging the extensive collection of knowledge bases.
"""

import os
import json
import asyncio
import logging
import argparse
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
import time
import uuid
import random
from datetime import datetime

# Import core components
from knowledge_base_dispatcher import KnowledgeBaseDispatcher
from dynamic_agents import DynamicAgent, AgentContext, KnowledgeBaseAgent, registry
from central_interaction_agent import CentralInteractionAgent
from pubsub_service import PubSubService
from agent_collective import AgentCollective, CollectiveRole, CollectiveAgent
from holographic_memory import HolographicMemory
from epistemic_core import EpistemicUnit, KnowledgeAPI, ReasoningWorkspace, KnowledgeGraph, TemporalKnowledgeState
from epistemic_tools import (
    initialize_knowledge_system,
    shutdown_knowledge_system,
    store_knowledge,
    query_knowledge,
    create_reasoning_workspace,
    workspace_add_step,
    workspace_derive_knowledge,
    workspace_commit_knowledge,
    create_relationship,
    explore_concept,
    create_temporal_snapshot
)
from epistemic_long_context import IncrementalReasoner, RecursiveDecomposer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_kb_agent_extended.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multi-kb-agent-extended")


class EnhancedConsensus:
    """
    Enhanced consensus mechanism for coordinating agreement between multiple agents
    with weighted voting, reputation-based influence, and minority perspective preservation.
    """
    
    def __init__(self, pubsub_service: PubSubService):
        """Initialize the consensus mechanism"""
        self.pubsub = pubsub_service
        self.voting_sessions = {}
        self.agent_reputation = {}
        self.vote_history = {}
        
    async def initialize_voting(self, topic: str, options: List[str], 
                               weights: Optional[Dict[str, float]] = None,
                               timeout: int = 60) -> str:
        """
        Initialize a new voting session on a topic.
        
        Args:
            topic: The topic to vote on
            options: The available options
            weights: Optional weights for each agent
            timeout: Timeout in seconds
            
        Returns:
            Voting session ID
        """
        session_id = f"vote_{uuid.uuid4()}"
        
        self.voting_sessions[session_id] = {
            "topic": topic,
            "options": options,
            "weights": weights or {},
            "votes": {},
            "deadline": time.time() + timeout,
            "status": "open",
            "result": None,
            "minority_opinions": []
        }
        
        # Announce the voting session
        await self.pubsub.publish("consensus", {
            "action": "vote_started",
            "session_id": session_id,
            "topic": topic,
            "options": options,
            "deadline": self.voting_sessions[session_id]["deadline"]
        })
        
        return session_id
    
    async def cast_vote(self, session_id: str, agent_id: str, option: str, 
                       confidence: float = 1.0, rationale: Optional[str] = None) -> Dict[str, Any]:
        """
        Cast a vote in a voting session.
        
        Args:
            session_id: The voting session ID
            agent_id: The agent casting the vote
            option: The chosen option
            confidence: The confidence in the vote
            rationale: Optional rationale for the vote
            
        Returns:
            Vote acknowledgment
        """
        # Check if session exists and is open
        if session_id not in self.voting_sessions:
            return {"success": False, "error": "Voting session not found"}
        
        session = self.voting_sessions[session_id]
        
        if session["status"] != "open":
            return {"success": False, "error": "Voting session is closed"}
        
        if option not in session["options"]:
            return {"success": False, "error": f"Invalid option: {option}"}
        
        # Record the vote
        weight = session["weights"].get(agent_id, 1.0)
        
        # Apply reputation adjustment if available
        if agent_id in self.agent_reputation:
            weight *= self.agent_reputation[agent_id]
        
        # Apply confidence adjustment
        effective_weight = weight * confidence
        
        session["votes"][agent_id] = {
            "option": option,
            "confidence": confidence,
            "weight": weight,
            "effective_weight": effective_weight,
            "rationale": rationale,
            "timestamp": time.time()
        }
        
        # Track vote history
        if agent_id not in self.vote_history:
            self.vote_history[agent_id] = []
        
        self.vote_history[agent_id].append({
            "session_id": session_id,
            "topic": session["topic"],
            "option": option,
            "timestamp": time.time()
        })
        
        # Announce the vote
        await self.pubsub.publish("consensus", {
            "action": "vote_cast",
            "session_id": session_id,
            "agent_id": agent_id,
            "option": option
        })
        
        # Check if all expected votes are in or deadline reached
        if time.time() >= session["deadline"]:
            await self.finalize_voting(session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "vote_recorded": option
        }
    
    async def finalize_voting(self, session_id: str) -> Dict[str, Any]:
        """
        Finalize a voting session and determine the outcome.
        
        Args:
            session_id: The voting session ID
            
        Returns:
            Voting outcome
        """
        if session_id not in self.voting_sessions:
            return {"success": False, "error": "Voting session not found"}
        
        session = self.voting_sessions[session_id]
        
        if session["status"] != "open":
            return {"success": False, "error": "Voting session already finalized"}
        
        # Tally votes
        tallies = {}
        for option in session["options"]:
            tallies[option] = 0.0
        
        for agent_id, vote in session["votes"].items():
            tallies[vote["option"]] += vote["effective_weight"]
        
        # Find winner
        winner = max(tallies.items(), key=lambda x: x[1])
        
        # Record minority opinions (options with significant support but not winners)
        total_weight = sum(tallies.values())
        minority_threshold = 0.25  # Options with at least 25% support are significant minorities
        
        minority_opinions = []
        for option, tally in tallies.items():
            if option != winner[0] and tally / total_weight >= minority_threshold:
                # Find agents who voted for this option and their rationales
                supporters = [
                    {"agent_id": agent_id, "rationale": vote["rationale"]}
                    for agent_id, vote in session["votes"].items()
                    if vote["option"] == option
                ]
                
                minority_opinions.append({
                    "option": option,
                    "support": tally / total_weight,
                    "supporters": supporters
                })
        
        # Update session
        session["status"] = "closed"
        session["result"] = winner[0]
        session["tallies"] = tallies
        session["total_weight"] = total_weight
        session["minority_opinions"] = minority_opinions
        
        # Announce result
        await self.pubsub.publish("consensus", {
            "action": "vote_completed",
            "session_id": session_id,
            "topic": session["topic"],
            "result": winner[0],
            "tallies": tallies,
            "total_weight": total_weight,
            "minority_opinions": minority_opinions
        })
        
        # Update agent reputations based on consensus
        await self._update_reputations(session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "topic": session["topic"],
            "result": winner[0],
            "tallies": tallies,
            "support_ratio": winner[1] / total_weight if total_weight > 0 else 0,
            "minority_opinions": minority_opinions
        }
    
    async def _update_reputations(self, session_id: str):
        """
        Update agent reputations based on voting outcome.
        Agents who voted with the majority get a small reputation boost.
        
        Args:
            session_id: The voting session ID
        """
        session = self.voting_sessions[session_id]
        result = session["result"]
        
        # Small adjustments to prevent reputation from changing too quickly
        majority_boost = 0.02
        
        for agent_id, vote in session["votes"].items():
            if agent_id not in self.agent_reputation:
                self.agent_reputation[agent_id] = 1.0
            
            # Boost majority voters slightly
            if vote["option"] == result:
                self.agent_reputation[agent_id] = min(
                    1.5,  # Cap at 1.5x influence
                    self.agent_reputation[agent_id] + majority_boost
                )
        
        # Log reputation updates
        logger.info(f"Updated agent reputations after vote {session_id}")


class LearningSubsystem:
    """
    Advanced learning subsystem that enables agents to improve over time through
    experience, feedback, collective knowledge sharing, and automated insight generation.
    
    Features:
    - Continuous learning from agent interactions
    - Cross-domain knowledge synthesis
    - Neural embedding-based pattern recognition
    - Automatic skill acquisition and transfer
    - Meta-learning capabilities
    - Bayesian optimization of learning strategies
    """
    
    def __init__(self, epistemic_db_path: str, enable_meta_learning: bool = True, 
                skill_transfer: bool = True, continuous_optimization: bool = True):
        """Initialize the advanced learning subsystem"""
        self.knowledge_api = KnowledgeAPI(epistemic_db_path)
        self.temporal_state = TemporalKnowledgeState(epistemic_db_path)
        self.kb_graph = KnowledgeGraph(epistemic_db_path)
        
        # Advanced configuration
        self.enable_meta_learning = enable_meta_learning
        self.skill_transfer = skill_transfer
        self.continuous_optimization = continuous_optimization
        self.learning_rate_adaption = True
        self.exploration_coefficient = 0.2
        
        # Track learning activities with enhanced metrics
        self.learning_sessions = {}
        self.agent_skills = {}
        self.cross_domain_insights = []
        self.skill_transfer_history = []
        self.learning_curves = {}
        self.competency_matrix = {}
        self.concept_mastery_levels = {}
        
        # Enhanced memory systems
        self.holographic_memory = HolographicMemory(dimensions=512, semantic_weight=0.7)
        self.episodic_memory = {}
        self.procedural_memory = {}
        
        # Learning strategies and meta-learning
        self.learning_strategies = {
            "direct_instruction": {"efficiency": 0.8, "retention": 0.5, "transfer": 0.3},
            "guided_discovery": {"efficiency": 0.5, "retention": 0.8, "transfer": 0.7},
            "collaborative_learning": {"efficiency": 0.6, "retention": 0.7, "transfer": 0.8},
            "trial_and_error": {"efficiency": 0.3, "retention": 0.9, "transfer": 0.6}
        }
        
        # Agent-specific learning preferences
        self.agent_learning_profiles = {}
        
        # Optimization metrics
        self.learning_efficiency = {}
        self.retention_rates = {}
        self.transfer_effectiveness = {}
        
        # Initialize optimization models if enabled
        self.continuous_optimization = continuous_optimization
        if continuous_optimization:
            self._initialize_optimization()
            
    def _initialize_optimization(self):
        """Initialize optimization models for continuous learning improvement"""
        logger.info("Initializing learning optimization models")
        
        # Create optimization matrices for each learning dimension
        self.efficiency_matrix = np.zeros((len(self.learning_strategies), len(self.learning_strategies)))
        self.retention_matrix = np.zeros((len(self.learning_strategies), len(self.learning_strategies)))
        self.transfer_matrix = np.zeros((len(self.learning_strategies), len(self.learning_strategies)))
        
        # Initialize with baseline values
        for i, strategy1 in enumerate(self.learning_strategies.keys()):
            for j, strategy2 in enumerate(self.learning_strategies.keys()):
                # Fill diagonal with strategy's own effectiveness
                if i == j:
                    self.efficiency_matrix[i, j] = self.learning_strategies[strategy1]["efficiency"]
                    self.retention_matrix[i, j] = self.learning_strategies[strategy1]["retention"]
                    self.transfer_matrix[i, j] = self.learning_strategies[strategy1]["transfer"]
                else:
                    # Off-diagonal represents synergy between strategies
                    # Default to average of the two strategies with a small bonus for diversity
                    self.efficiency_matrix[i, j] = (self.learning_strategies[strategy1]["efficiency"] +
                                                  self.learning_strategies[strategy2]["efficiency"]) / 2 * 1.05
                    self.retention_matrix[i, j] = (self.learning_strategies[strategy1]["retention"] +
                                                 self.learning_strategies[strategy2]["retention"]) / 2 * 1.05
                    self.transfer_matrix[i, j] = (self.learning_strategies[strategy1]["transfer"] +
                                                self.learning_strategies[strategy2]["transfer"]) / 2 * 1.05
        
        logger.info("Learning optimization models initialized")
        
    async def register_agent_skill(self, agent_id: str, domain: str, skill: str, 
                                  proficiency: float = 0.5) -> Dict[str, Any]:
        """
        Register an agent's skill in a domain.
        
        Args:
            agent_id: The agent ID
            domain: The knowledge domain
            skill: The specific skill
            proficiency: Initial proficiency level (0.0-1.0)
            
        Returns:
            Registration result
        """
        if agent_id not in self.agent_skills:
            self.agent_skills[agent_id] = {}
        
        if domain not in self.agent_skills[agent_id]:
            self.agent_skills[agent_id][domain] = {}
        
        self.agent_skills[agent_id][domain][skill] = proficiency
        
        logger.info(f"Registered skill '{skill}' for agent {agent_id} in {domain} domain (proficiency: {proficiency:.2f})")
        
        return {
            "success": True,
            "agent_id": agent_id,
            "domain": domain,
            "skill": skill,
            "proficiency": proficiency
        }
    
    async def start_learning_session(self, agent_id: str, domain: str, 
                                    topic: str) -> Dict[str, Any]:
        """
        Start a learning session for an agent.
        
        Args:
            agent_id: The agent ID
            domain: The knowledge domain
            topic: The learning topic
            
        Returns:
            Learning session details
        """
        session_id = f"learn_{uuid.uuid4()}"
        
        self.learning_sessions[session_id] = {
            "agent_id": agent_id,
            "domain": domain,
            "topic": topic,
            "start_time": time.time(),
            "steps": [],
            "resources_used": [],
            "insights_gained": [],
            "status": "active"
        }
        
        logger.info(f"Started learning session {session_id} for agent {agent_id} on {topic} in {domain} domain")
        
        return {
            "success": True,
            "session_id": session_id,
            "agent_id": agent_id,
            "domain": domain,
            "topic": topic
        }
    
    async def record_learning_step(self, session_id: str, action: str, 
                                  content: str) -> Dict[str, Any]:
        """
        Record a step in a learning session.
        
        Args:
            session_id: The learning session ID
            action: The learning action performed
            content: The content of the learning step
            
        Returns:
            Step recording result
        """
        if session_id not in self.learning_sessions:
            return {"success": False, "error": "Learning session not found"}
        
        session = self.learning_sessions[session_id]
        
        if session["status"] != "active":
            return {"success": False, "error": "Learning session is not active"}
        
        # Record the step
        step = {
            "action": action,
            "content": content,
            "timestamp": time.time()
        }
        
        session["steps"].append(step)
        
        logger.info(f"Recorded learning step for session {session_id}: {action}")
        
        return {
            "success": True,
            "session_id": session_id,
            "step_index": len(session["steps"]) - 1
        }
    
    async def record_learning_insight(self, session_id: str, insight: str, 
                                     confidence: float = 0.7) -> Dict[str, Any]:
        """
        Record an insight gained during a learning session.
        
        Args:
            session_id: The learning session ID
            insight: The insight gained
            confidence: Confidence in the insight
            
        Returns:
            Insight recording result
        """
        if session_id not in self.learning_sessions:
            return {"success": False, "error": "Learning session not found"}
        
        session = self.learning_sessions[session_id]
        
        if session["status"] != "active":
            return {"success": False, "error": "Learning session is not active"}
        
        # Record the insight
        insight_record = {
            "content": insight,
            "confidence": confidence,
            "timestamp": time.time()
        }
        
        session["insights_gained"].append(insight_record)
        
        # Store in epistemic system
        insight_unit = EpistemicUnit(
            content=insight,
            confidence=confidence,
            source=f"Learning session {session_id}",
            evidence=f"Learned by agent {session['agent_id']} on topic {session['topic']}"
        )
        
        result = store_knowledge(insight_unit)
        
        # Add domain relationship
        create_relationship(
            source_id=result["unit_id"],
            relation_type="learned_in_domain",
            target=f"domain:{session['domain']}",
            confidence=confidence
        )
        
        logger.info(f"Recorded learning insight for session {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "insight_index": len(session["insights_gained"]) - 1,
            "epistemic_unit_id": result["unit_id"]
        }
    
    async def complete_learning_session(self, session_id: str, 
                                      proficiency_gain: float = 0.05) -> Dict[str, Any]:
        """
        Complete a learning session and update agent skills.
        
        Args:
            session_id: The learning session ID
            proficiency_gain: The amount of proficiency gained
            
        Returns:
            Session completion result
        """
        if session_id not in self.learning_sessions:
            return {"success": False, "error": "Learning session not found"}
        
        session = self.learning_sessions[session_id]
        
        if session["status"] != "active":
            return {"success": False, "error": "Learning session is not active"}
        
        # Update session status
        session["status"] = "completed"
        session["end_time"] = time.time()
        session["duration"] = session["end_time"] - session["start_time"]
        
        # Update agent skills
        agent_id = session["agent_id"]
        domain = session["domain"]
        
        # Extract skills from topic
        topic_parts = session["topic"].lower().split()
        related_skills = [
            part for part in topic_parts 
            if len(part) > 3 and part not in ["with", "using", "and", "the", "for"]
        ]
        
        # Update existing skills or add new ones
        if agent_id in self.agent_skills and domain in self.agent_skills[agent_id]:
            # Update existing skills
            for skill in self.agent_skills[agent_id][domain]:
                if any(skill_term in skill.lower() for skill_term in related_skills):
                    # Skill is related to the topic, increase proficiency
                    current = self.agent_skills[agent_id][domain][skill]
                    self.agent_skills[agent_id][domain][skill] = min(1.0, current + proficiency_gain)
                    
                    logger.info(f"Increased proficiency of agent {agent_id} in {skill} to {self.agent_skills[agent_id][domain][skill]:.2f}")
            
            # Add new skills if needed
            for skill_term in related_skills:
                if not any(skill_term in skill.lower() for skill in self.agent_skills[agent_id][domain]):
                    new_skill = skill_term.capitalize()
                    self.agent_skills[agent_id][domain][new_skill] = proficiency_gain
                    
                    logger.info(f"Added new skill '{new_skill}' for agent {agent_id} with proficiency {proficiency_gain:.2f}")
        
        # Create a temporal snapshot
        snapshot_id = create_temporal_snapshot(f"Learning session {session_id} completed")
        
        logger.info(f"Completed learning session {session_id}")
        
        return {
            "success": True,
            "session_id": session_id,
            "agent_id": agent_id,
            "domain": domain,
            "duration": session["duration"],
            "insights_gained": len(session["insights_gained"]),
            "skills_updated": list(self.agent_skills.get(agent_id, {}).get(domain, {}).keys()),
            "snapshot_id": snapshot_id
        }
    
    async def identify_cross_domain_patterns(self) -> Dict[str, Any]:
        """
        Identify patterns and connections across different knowledge domains.
        
        Returns:
            Cross-domain insights
        """
        # Get insights from each active domain
        insights = query_knowledge("cross-domain", reasoning_depth=2)
        
        # Use holographic memory to find patterns
        pattern_inputs = []
        for result in insights.get("direct_results", []):
            pattern_inputs.append(result.get("content", ""))
        
        if pattern_inputs:
            # Add to holographic memory
            for input_text in pattern_inputs:
                self.holographic_memory.store(input_text)
            
            # Retrieve related patterns
            patterns = self.holographic_memory.search("interdisciplinary connections", top_k=5)
            
            # Record cross-domain insights
            cross_domain_insight = {
                "timestamp": time.time(),
                "patterns": patterns,
                "source_insights": insights.get("direct_results", [])[:5]
            }
            
            self.cross_domain_insights.append(cross_domain_insight)
            
            logger.info(f"Identified {len(patterns)} cross-domain patterns")
            
            return {
                "success": True,
                "patterns_found": len(patterns),
                "patterns": patterns
            }
        
        return {
            "success": False,
            "error": "Insufficient insights for pattern identification"
        }


class MultiKBAgentSystem:
    """
    A system that orchestrates multiple knowledge-base specialized agents to solve
    complex real-world problems requiring expertise across multiple domains.
    
    Extended with advanced features:
    - Enhanced consensus mechanisms for better agent coordination
    - Learning subsystem for continuous agent improvement
    - Emergent insight detection for cross-domain discoveries
    - Dynamic agent specialization based on problem needs
    - Multiple modes of operation for different problem types
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
        
        # Initialize enhanced components
        self.consensus = EnhancedConsensus(self.pubsub)
        self.learning = LearningSubsystem(epistemic_db_path)
        
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
        
        # Operation mode (collaborative, competitive, emergent)
        self.operation_mode = "collaborative"
        
        # Performance metrics
        self.metrics = {
            "problems_solved": 0,
            "domains_utilized": {},
            "solution_quality": [],
            "average_resolution_time": []
        }
        
        logger.info("Extended MultiKBAgentSystem initialized")
    
    async def setup_domain_agents(self):
        """Set up specialized agents for different knowledge domains"""
        # Initialize KB sizes if needed
        if hasattr(self.kb_dispatcher, 'initialize_kb_sizes'):
            await self.kb_dispatcher.initialize_kb_sizes()
            
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
                    name=f"{kb_name.replace('_', ' ').title()} Specialist",
                    description=f"Expert in {kb_name.replace('_', ' ')} domain knowledge",
                    role=role,
                    capabilities=[
                        "search_knowledge",
                        "analyze_domain",
                        "answer_domain_questions",
                        "identify_domain_relevance"
                    ]
                )
            )
            
            # Register basic skills in the learning subsystem
            await self.learning.register_agent_skill(
                agent_id=agent_id,
                domain=kb_name,
                skill="Domain Knowledge",
                proficiency=0.8
            )
            
            await self.learning.register_agent_skill(
                agent_id=agent_id,
                domain=kb_name,
                skill="Information Retrieval",
                proficiency=0.7
            )
            
            logger.info(f"Created collective agent for domain: {kb_name}")
        
        # Set up core system agents
        await self._setup_system_agents()
        
        # Connect to pubsub channels
        await self.pubsub.connect()
        await self.pubsub.subscribe("problem_solving", self._handle_problem_solving_message)
        await self.pubsub.subscribe("domain_insights", self._handle_domain_insights)
        await self.pubsub.subscribe("solution_proposals", self._handle_solution_proposals)
        await self.pubsub.subscribe("consensus", self._handle_consensus_message)
        await self.pubsub.subscribe("learning", self._handle_learning_message)
        await self.pubsub.subscribe("emergent_insights", self._handle_emergent_insights)
        
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
                name="Problem Coordinator",
                description="Orchestrates the overall problem-solving process",
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
                name="Research Strategist",
                description="Plans and coordinates knowledge gathering across domains",
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
                name="Solution Architect",
                description="Designs and refines solution approaches",
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
                name="Critical Evaluator",
                description="Critically evaluates proposed solutions",
                role=evaluator_role,
                capabilities=[
                    "identify_weaknesses",
                    "validate_solutions",
                    "propose_improvements",
                    "risk_assessment"
                ]
            )
        )
        
        # New: Emergent Pattern Detector
        pattern_role = CollectiveRole(
            name="Pattern Detector",
            description="Identifies emergent patterns and connections across domains",
            capabilities=["pattern_recognition", "cross_domain_synthesis", "emergent_insight"]
        )
        await self.collective.add_agent(
            CollectiveAgent(
                agent_id="pattern_detector",
                name="Pattern Detector",
                description="Identifies emergent patterns and connections across domains",
                role=pattern_role,
                capabilities=[
                    "detect_patterns",
                    "connect_insights",
                    "identify_emergence",
                    "propose_novel_combinations"
                ]
            )
        )
        
        # New: Learning Facilitator
        learning_role = CollectiveRole(
            name="Learning Facilitator",
            description="Coordinates the learning and improvement of other agents",
            capabilities=["teaching", "knowledge_transfer", "skill_assessment"]
        )
        await self.collective.add_agent(
            CollectiveAgent(
                agent_id="learning_facilitator",
                name="Learning Facilitator",
                description="Coordinates the learning and improvement of other agents",
                role=learning_role,
                capabilities=[
                    "identify_learning_opportunities",
                    "create_learning_plans",
                    "assess_skill_improvement",
                    "transfer_knowledge"
                ]
            )
        )
        
        logger.info("System agents setup completed")
    
    async def solve_problem(self, problem_statement: str, 
                          mode: str = "collaborative") -> Dict[str, Any]:
        """
        Solve a complex problem by coordinating multiple knowledge-base agents.
        
        Args:
            problem_statement: The problem to solve
            mode: Operation mode (collaborative, competitive, or emergent)
            
        Returns:
            Solution details and reasoning process
        """
        start_time = time.time()
        self.current_problem = problem_statement
        self.operation_mode = mode
        
        logger.info(f"Starting to solve problem in {mode} mode: {problem_statement}")
        
        # Record problem in workspace
        workspace_add_step(
            self.main_workspace_id,
            "problem_definition",
            f"PROBLEM STATEMENT: {problem_statement}"
        )
        
        # Record operation mode
        workspace_add_step(
            self.main_workspace_id,
            "operation_mode",
            f"OPERATION MODE: {mode.upper()}"
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
            "domain_relevance": self.domain_relevance,
            "mode": mode
        })
        
        # Use different solution approaches based on the mode
        if mode == "collaborative":
            solution_result = await self._process_problem_incrementally(decomposer_id)
        elif mode == "competitive":
            solution_result = await self._process_problem_competitively(decomposer_id)
        elif mode == "emergent":
            solution_result = await self._process_problem_emergently(decomposer_id)
        else:
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
        store_result = store_knowledge(solution_unit)
        
        # Commit the workspace knowledge
        workspace_commit_knowledge(self.main_workspace_id)
        
        # Close the decomposer
        decomposer.close()
        
        # Create a temporal snapshot
        snapshot_id = create_temporal_snapshot(f"Problem solution: {problem_statement[:50]}...")
        
        # Update metrics
        end_time = time.time()
        self.metrics["problems_solved"] += 1
        self.metrics["average_resolution_time"].append(end_time - start_time)
        
        # Update domain utilization metrics
        for domain in self.active_domains:
            if domain not in self.metrics["domains_utilized"]:
                self.metrics["domains_utilized"][domain] = 0
            self.metrics["domains_utilized"][domain] += 1
        
        # Trigger learning from this problem-solving experience
        await self._learn_from_problem_solving(problem_statement, solution_result)
        
        return {
            "problem": problem_statement,
            "solution": solution_result['conclusion'],
            "subproblems": decomposition['subproblems'],
            "domains_utilized": list(self.active_domains),
            "reasoning_trace": workspace_get_chain(self.main_workspace_id)["chain"],
            "solution_unit_id": store_result["unit_id"],
            "snapshot_id": snapshot_id,
            "resolution_time": end_time - start_time,
            "mode": mode
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
        Collaborative approach with incremental progress.
        
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
    
    async def _process_problem_competitively(self, decomposer_id: str) -> Dict[str, Any]:
        """
        Process the problem using a competitive approach where multiple agents
        independently generate solutions and then vote on the best one.
        
        Args:
            decomposer_id: ID of the decomposer
            
        Returns:
            Final problem solution
        """
        decomposer = self.decomposers[decomposer_id]
        
        # Get main problem statement
        problem = self.current_problem
        
        # Identify key domains for overall problem
        primary_domains = []
        for subproblem_id, domain_scores in self.domain_relevance.items():
            for domain, score in domain_scores.items():
                if score >= 0.6:  # High relevance threshold
                    primary_domains.append(domain)
        
        primary_domains = list(set(primary_domains))[:5]  # Take up to 5 top domains
        
        logger.info(f"Selected {len(primary_domains)} primary domains for competitive approach")
        
        # Record the competitive approach
        workspace_add_step(
            self.main_workspace_id,
            "competitive_approach",
            f"USING COMPETITIVE APPROACH WITH {len(primary_domains)} DOMAINS:\n" +
            "\n".join([f"- {domain}" for domain in primary_domains])
        )
        
        # Each domain generates a solution independently
        solution_proposals = []
        
        for domain in primary_domains:
            # Create a workspace for this domain's solution
            domain_workspace = create_reasoning_workspace(f"{domain} Solution for {problem[:30]}...")
            
            # Get solution from this domain's perspective
            solution = await self.kb_dispatcher.execute_kb_command(
                domain,
                f"generate_solution {problem}"
            )
            
            if solution.get("success", False):
                proposal = {
                    "domain": domain,
                    "solution": solution.get("solution", "No solution provided"),
                    "confidence": solution.get("confidence", 0.5),
                    "workspace_id": domain_workspace["workspace_id"]
                }
                
                solution_proposals.append(proposal)
                
                # Record in domain workspace
                workspace_add_step(
                    domain_workspace["workspace_id"],
                    "domain_solution",
                    f"SOLUTION FROM {domain.upper()} PERSPECTIVE:\n{proposal['solution']}"
                )
                
                # Publish to solution proposals channel
                await self.pubsub.publish("solution_proposals", {
                    "action": "domain_solution",
                    "domain": domain,
                    "solution": proposal["solution"],
                    "confidence": proposal["confidence"]
                })
                
                logger.info(f"Generated solution from {domain} domain (confidence: {proposal['confidence']:.2f})")
        
        # Use consensus mechanism to evaluate and select best solution
        if solution_proposals:
            # Prepare voting options
            options = [f"solution_{i}" for i in range(len(solution_proposals))]
            weights = {f"evaluator": 1.5}  # Give evaluator agent more weight
            
            # Set up voting session
            vote_session = await self.consensus.initialize_voting(
                topic="solution_selection",
                options=options,
                weights=weights,
                timeout=30
            )
            
            # Record voting
            workspace_add_step(
                self.main_workspace_id,
                "solution_voting",
                f"VOTING ON {len(solution_proposals)} SOLUTION PROPOSALS"
            )
            
            # Simulate votes from agents
            for agent_id in ["coordinator", "researcher", "evaluator", "architect", "pattern_detector"]:
                # Evaluator tends to prefer solutions with higher confidence
                if agent_id == "evaluator":
                    confidence_values = [p["confidence"] for p in solution_proposals]
                    max_confidence_index = confidence_values.index(max(confidence_values))
                    vote_option = options[max_confidence_index]
                else:
                    # Other agents vote somewhat randomly but with preference for higher confidence
                    weighted_options = []
                    for i, proposal in enumerate(solution_proposals):
                        # Add options with weight proportional to confidence
                        weight = int(proposal["confidence"] * 10)
                        weighted_options.extend([options[i]] * weight)
                    
                    vote_option = random.choice(weighted_options)
                
                # Cast vote
                await self.consensus.cast_vote(
                    vote_session,
                    agent_id,
                    vote_option,
                    confidence=0.8,
                    rationale=f"Based on {agent_id}'s evaluation criteria"
                )
            
            # Finalize the vote
            vote_result = await self.consensus.finalize_voting(vote_session)
            
            if vote_result["success"]:
                winning_index = int(vote_result["result"].split("_")[1])
                winning_proposal = solution_proposals[winning_index]
                
                # Record the winning solution
                workspace_add_step(
                    self.main_workspace_id,
                    "winning_solution",
                    f"WINNING SOLUTION FROM {winning_proposal['domain'].upper()}:\n" +
                    f"Support: {vote_result['support_ratio']:.2f}\n\n" +
                    winning_proposal["solution"]
                )
                
                # Check for minority opinions
                if vote_result["minority_opinions"]:
                    minority_text = "\n\n".join([
                        f"ALTERNATIVE APPROACH FROM {solution_proposals[int(opinion['option'].split('_')[1])]['domain'].upper()}:\n" +
                        f"Support: {opinion['support']:.2f}\n\n" +
                        solution_proposals[int(opinion['option'].split('_')[1])]["solution"]
                        for opinion in vote_result["minority_opinions"]
                    ])
                    
                    workspace_add_step(
                        self.main_workspace_id,
                        "alternative_solutions",
                        f"ALTERNATIVE SOLUTIONS WITH SIGNIFICANT SUPPORT:\n{minority_text}"
                    )
                
                # Synthesize final answer incorporating minority views
                synthesis = decomposer._synthesize_final_answer()
                synthesis["conclusion"] = winning_proposal["solution"]
                
                # Add information about alternative approaches if there were minority opinions
                if vote_result["minority_opinions"]:
                    synthesis["conclusion"] += "\n\nAlternative approaches to consider:\n"
                    for opinion in vote_result["minority_opinions"]:
                        minority_index = int(opinion["option"].split("_")[1])
                        minority_domain = solution_proposals[minority_index]["domain"]
                        synthesis["conclusion"] += f"\n- From {minority_domain} perspective: " + \
                                                  solution_proposals[minority_index]["solution"][:200] + "..."
                
                return synthesis
        
        # Fallback to standard synthesis if voting fails
        return decomposer._synthesize_final_answer()
    
    async def _process_problem_emergently(self, decomposer_id: str) -> Dict[str, Any]:
        """
        Process the problem using an emergent approach that leverages cross-domain
        insights and pattern recognition for novel solutions.
        
        Args:
            decomposer_id: ID of the decomposer
            
        Returns:
            Final problem solution
        """
        # Create initial understanding of the problem with the decomposer
        decomposer = self.decomposers[decomposer_id]
        
        # Run several increments to establish basic understanding
        initial_increments = 5
        for i in range(initial_increments):
            result = decomposer.process_next_increment()
            logger.info(f"Initial increment {i+1}: Progress {result.get('tree_progress', 0)*100:.1f}%")
        
        # Record the emergent approach
        workspace_add_step(
            self.main_workspace_id,
            "emergent_approach",
            "USING EMERGENT APPROACH TO FIND NOVEL CROSS-DOMAIN SOLUTIONS"
        )
        
        # Gather insights from all relevant domains
        all_domain_insights = {}
        
        for subproblem_id, domain_scores in self.domain_relevance.items():
            for domain, relevance in domain_scores.items():
                # Skip if already processed
                if domain in all_domain_insights:
                    continue
                
                # Get insights from this domain
                kb_result = await self.kb_dispatcher.search_knowledge_base(
                    domain,
                    self.current_problem
                )
                
                if kb_result.get("success", False):
                    insights = kb_result.get("results", [])
                    
                    if insights:
                        all_domain_insights[domain] = {
                            "insights": insights[:5],  # Take top 5 insights
                            "relevance": relevance
                        }
                        
                        logger.info(f"Gathered {len(insights[:5])} insights from {domain} domain")
        
        # Record gathered insights
        insights_text = "\n\n".join([
            f"INSIGHTS FROM {domain.upper()} (relevance: {info['relevance']:.2f}):\n" +
            "\n".join([f"- {insight}" for insight in info["insights"]])
            for domain, info in all_domain_insights.items()
        ])
        
        workspace_add_step(
            self.main_workspace_id,
            "cross_domain_insights",
            f"INSIGHTS FROM {len(all_domain_insights)} DOMAINS:\n{insights_text}"
        )
        
        # Use pattern detector to identify cross-domain patterns
        patterns = await self.learning.identify_cross_domain_patterns()
        
        if patterns["success"]:
            # Record identified patterns
            patterns_text = "\n".join([
                f"- {pattern}" for pattern in patterns["patterns"]
            ])
            
            workspace_add_step(
                self.main_workspace_id,
                "emergent_patterns",
                f"EMERGENT PATTERNS ACROSS DOMAINS:\n{patterns_text}"
            )
            
            # Publish to emergent insights channel
            await self.pubsub.publish("emergent_insights", {
                "action": "cross_domain_patterns",
                "patterns": patterns["patterns"],
                "domains": list(all_domain_insights.keys())
            })
            
            # Create a novel solution based on identified patterns
            solution_components = []
            
            # Process remaining increments to generate complete solution
            max_remaining = 15
            for i in range(max_remaining):
                result = decomposer.process_next_increment()
                
                # If complete, we're done
                if result.get("status") == "complete":
                    break
            
            # Generate final synthesis
            synthesis = decomposer._synthesize_final_answer()
            
            # Enhance the conclusion with cross-domain patterns
            enhanced_conclusion = synthesis["conclusion"] + "\n\n"
            enhanced_conclusion += "CROSS-DOMAIN INSIGHTS:\n"
            enhanced_conclusion += patterns_text
            
            synthesis["conclusion"] = enhanced_conclusion
            
            workspace_add_step(
                self.main_workspace_id,
                "emergent_solution",
                f"EMERGENT SOLUTION INCORPORATING CROSS-DOMAIN INSIGHTS:\n{enhanced_conclusion}"
            )
            
            return synthesis
        
        # Fallback to standard synthesis if pattern identification fails
        return await self._process_problem_incrementally(decomposer_id)
    
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
    
    async def _learn_from_problem_solving(self, problem: str, solution: Dict[str, Any]):
        """
        Learn from the problem-solving experience to improve future performance.
        
        Args:
            problem: The problem that was solved
            solution: The solution that was generated
        """
        # Create a learning session for each active domain
        for domain in self.active_domains:
            agent_id = f"domain_{domain}"
            
            # Start a learning session
            session = await self.learning.start_learning_session(
                agent_id=agent_id,
                domain=domain,
                topic=f"Problem solving: {problem[:50]}..."
            )
            
            # Record key steps in the learning process
            await self.learning.record_learning_step(
                session["session_id"],
                "problem_analysis",
                f"Analyzed problem: {problem}"
            )
            
            # Record insights gained
            await self.learning.record_learning_insight(
                session["session_id"],
                f"Domain {domain} can contribute to solving problems related to: {problem[:100]}...",
                confidence=0.75
            )
            
            # Complete the learning session
            await self.learning.complete_learning_session(
                session["session_id"],
                proficiency_gain=0.03  # Small incremental improvement
            )
            
            logger.info(f"Completed learning session for {domain} domain")
        
        # Publish learning experience
        await self.pubsub.publish("learning", {
            "action": "problem_solving_experience",
            "problem": problem,
            "domains_utilized": list(self.active_domains),
            "solution_found": True
        })
    
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
        action = message.get("action")
        
        if action == "domain_solution":
            domain = message.get("domain")
            solution = message.get("solution")
            confidence = message.get("confidence", 0.5)
            
            logger.info(f"Received solution proposal from {domain} domain (confidence: {confidence:.2f})")
            
            # Record the proposal in the workspace
            workspace_add_step(
                self.main_workspace_id,
                "solution_proposal",
                f"SOLUTION PROPOSAL FROM {domain.upper()} DOMAIN (confidence: {confidence:.2f}):\n{solution}"
            )
    
    async def _handle_consensus_message(self, message: Dict[str, Any]):
        """
        Handle messages on the consensus channel.
        
        Args:
            message: The message to handle
        """
        action = message.get("action")
        
        if action == "vote_completed":
            session_id = message.get("session_id")
            topic = message.get("topic")
            result = message.get("result")
            
            logger.info(f"Voting completed for {topic} with result: {result}")
            
            # Record the consensus in the workspace
            workspace_add_step(
                self.main_workspace_id,
                "consensus_reached",
                f"CONSENSUS REACHED ON {topic.upper()}:\n{result}"
            )
    
    async def _handle_learning_message(self, message: Dict[str, Any]):
        """
        Handle messages on the learning channel.
        
        Args:
            message: The message to handle
        """
        action = message.get("action")
        
        if action == "problem_solving_experience":
            problem = message.get("problem")
            domains = message.get("domains_utilized", [])
            
            logger.info(f"Learning from problem solving experience with {len(domains)} domains")
    
    async def _handle_emergent_insights(self, message: Dict[str, Any]):
        """
        Handle messages on the emergent insights channel.
        
        Args:
            message: The message to handle
        """
        action = message.get("action")
        
        if action == "cross_domain_patterns":
            patterns = message.get("patterns", [])
            domains = message.get("domains", [])
            
            logger.info(f"Received {len(patterns)} cross-domain patterns from {len(domains)} domains")
            
            # Record the patterns in the workspace
            patterns_text = "\n".join([f"- {pattern}" for pattern in patterns])
            workspace_add_step(
                self.main_workspace_id,
                "emergent_patterns",
                f"EMERGENT PATTERNS ACROSS {len(domains)} DOMAINS:\n{patterns_text}"
            )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about system performance.
        
        Returns:
            System metrics
        """
        # Calculate aggregate metrics
        avg_resolution_time = sum(self.metrics["average_resolution_time"]) / len(self.metrics["average_resolution_time"]) if self.metrics["average_resolution_time"] else 0
        
        # Sort domains by utilization
        sorted_domains = sorted(
            self.metrics["domains_utilized"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_domains = sorted_domains[:10] if len(sorted_domains) > 10 else sorted_domains
        
        return {
            "problems_solved": self.metrics["problems_solved"],
            "avg_resolution_time": avg_resolution_time,
            "top_domains": dict(top_domains),
            "agent_count": len(self.collective.agents),
            "agent_reputation": self.consensus.agent_reputation
        }
    
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
        
        logger.info("Extended MultiKBAgentSystem shut down")


async def solve_example_problem():
    """Example usage of the extended multi-KB agent system"""
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
        
        # Solve the problem using all three modes
        modes = ["collaborative", "competitive", "emergent"]
        
        for mode in modes:
            print(f"\n{'-'*40}")
            print(f"USING {mode.upper()} MODE")
            print(f"{'-'*40}\n")
            
            # Solve the problem
            solution = await system.solve_problem(problem, mode=mode)
            
            print("\n" + "="*80)
            print(f"SOLUTION ({mode.upper()}):")
            print(solution["solution"])
            print("\n" + "="*40)
            print(f"Domains utilized: {', '.join(solution['domains_utilized'])}")
            print(f"Resolution time: {solution['resolution_time']:.2f} seconds")
            print("="*80 + "\n")
        
        # Print system metrics
        metrics = system.get_system_metrics()
        print("\n" + "="*80)
        print("SYSTEM METRICS:")
        print(f"Problems solved: {metrics['problems_solved']}")
        print(f"Average resolution time: {metrics['avg_resolution_time']:.2f} seconds")
        print("\nTop domains:")
        for domain, count in metrics['top_domains'].items():
            print(f"- {domain}: {count} uses")
        print("="*80 + "\n")
        
    finally:
        # Clean up
        system.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Knowledge Base Agent System (Extended)")
    parser.add_argument("--problem", type=str, help="Problem to solve")
    parser.add_argument("--mode", type=str, choices=["collaborative", "competitive", "emergent"], 
                       default="collaborative", help="Operation mode")
    parser.add_argument("--compare", action="store_true", help="Compare all operation modes")
    
    args = parser.parse_args()
    
    # Run the system
    if args.problem:
        # Custom problem from command line
        async def solve_custom_problem():
            system = MultiKBAgentSystem()
            try:
                await system.setup_domain_agents()
                
                if args.compare:
                    # Compare all modes
                    modes = ["collaborative", "competitive", "emergent"]
                    results = {}
                    
                    for mode in modes:
                        print(f"\nSolving with {mode} mode...")
                        solution = await system.solve_problem(args.problem, mode=mode)
                        results[mode] = solution
                    
                    # Display comparison
                    print("\n" + "="*80)
                    print("SOLUTION COMPARISON:")
                    
                    for mode, solution in results.items():
                        print(f"\n{'-'*40}")
                        print(f"{mode.upper()} SOLUTION (time: {solution['resolution_time']:.2f}s):")
                        print(f"{'-'*40}")
                        print(solution["solution"])
                    
                    print("\n" + "="*80)
                else:
                    # Single mode solution
                    solution = await system.solve_problem(args.problem, mode=args.mode)
                    
                    print("\n" + "="*80)
                    print(f"SOLUTION ({args.mode.upper()}):")
                    print(solution["solution"])
                    print("\n" + "="*40)
                    print(f"Domains utilized: {', '.join(solution['domains_utilized'])}")
                    print(f"Resolution time: {solution['resolution_time']:.2f} seconds")
                    print("="*80 + "\n")
            finally:
                system.close()
        
        asyncio.run(solve_custom_problem())
    else:
        # Example problem
        asyncio.run(solve_example_problem())