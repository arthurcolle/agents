#!/usr/bin/env python3
"""
Epistemic Tools - Tools for AI agents to interact with the epistemic knowledge system

This module provides tool functions that an AI agent can use to store and retrieve knowledge
without having to implement knowledge management in its own code.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from epistemic_core import (
    KnowledgeAPI, ReasoningWorkspace, EpistemicUnit, 
    KnowledgeGraph, TemporalKnowledgeState
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("epistemic-tools")

# Global knowledge API instance
_KNOWLEDGE_API = None
_ACTIVE_WORKSPACES = {}


def initialize_knowledge_system(db_path: str = "./knowledge/epistemic.db") -> Dict[str, Any]:
    """
    Tool: Initialize the knowledge system
    
    This must be called before using other epistemic tools. It sets up the knowledge
    system with the specified database path.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Status information about the knowledge system
    """
    global _KNOWLEDGE_API
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize the knowledge API
        _KNOWLEDGE_API = KnowledgeAPI(db_path)
        
        # Get basic stats about the knowledge system
        unit_count = 0
        domains = []
        
        try:
            _KNOWLEDGE_API.store.cursor.execute('SELECT COUNT(*) FROM epistemic_units')
            unit_count = _KNOWLEDGE_API.store.cursor.fetchone()[0]
            
            _KNOWLEDGE_API.store.cursor.execute('SELECT DISTINCT domain FROM epistemic_units')
            domains = [row[0] for row in _KNOWLEDGE_API.store.cursor.fetchall()]
        except Exception as e:
            logger.warning(f"Error getting knowledge stats: {e}")
        
        return {
            "status": "initialized",
            "db_path": db_path,
            "unit_count": unit_count,
            "domains": domains,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error initializing knowledge system: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def shutdown_knowledge_system() -> Dict[str, Any]:
    """
    Tool: Shut down the knowledge system
    
    This should be called when done using the knowledge system to ensure
    all data is properly saved and connections are closed.
    
    Returns:
        Status information
    """
    global _KNOWLEDGE_API, _ACTIVE_WORKSPACES
    
    try:
        # Close all active workspaces
        for workspace_id in list(_ACTIVE_WORKSPACES.keys()):
            try:
                _ACTIVE_WORKSPACES.pop(workspace_id)
            except Exception as e:
                logger.warning(f"Error closing workspace {workspace_id}: {e}")
        
        # Close the knowledge API
        if _KNOWLEDGE_API:
            _KNOWLEDGE_API.close()
            _KNOWLEDGE_API = None
        
        return {
            "status": "shutdown_complete",
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error shutting down knowledge system: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def store_knowledge(content: str, source: str, confidence: float = 0.7, 
                   domain: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Tool: Store a piece of knowledge in the knowledge system
    
    Use this to add new knowledge that the AI has learned or observed.
    
    Args:
        content: The actual knowledge content
        source: Where this knowledge came from
        confidence: How confident we are in this knowledge (0.0-1.0)
        domain: The knowledge domain (optional)
        metadata: Additional properties (optional)
        
    Returns:
        Information about the stored knowledge
    """
    _ensure_initialized()
    
    try:
        # Store the knowledge
        unit_id = _KNOWLEDGE_API.tell(
            content=content,
            source=source,
            confidence=confidence,
            domain=domain,
            metadata=metadata
        )
        
        # Get the stored unit to return details
        unit = _KNOWLEDGE_API.store.get_unit(unit_id)
        
        return {
            "status": "success",
            "id": unit_id,
            "content": content,
            "confidence": confidence,
            "domain": domain or "general",
            "timestamp": time.time(),
            "contradictions": unit.contradictions if unit else []
        }
    except Exception as e:
        logger.error(f"Error storing knowledge: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def query_knowledge(query: str, reasoning_depth: int = 2, 
                   min_confidence: float = 0.3, domain: str = None) -> Dict[str, Any]:
    """
    Tool: Query the knowledge system
    
    Use this to retrieve knowledge relevant to a query, with optional filters.
    
    Args:
        query: The question or topic to search for
        reasoning_depth: How deeply to explore supporting evidence (0-2)
        min_confidence: Minimum confidence threshold (0.0-1.0)
        domain: Limit results to a specific domain (optional)
        
    Returns:
        Relevant knowledge with justifications
    """
    _ensure_initialized()
    
    try:
        # Query the knowledge system
        result = _KNOWLEDGE_API.ask(
            query=query,
            reasoning_depth=reasoning_depth,
            min_confidence=min_confidence,
            domain=domain
        )
        
        # Add metadata about the query
        result["status"] = "success"
        result["timestamp"] = time.time()
        
        return result
    except Exception as e:
        logger.error(f"Error querying knowledge: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def explore_concept(concept: str, max_distance: int = 2) -> Dict[str, Any]:
    """
    Tool: Explore a concept in the knowledge graph
    
    Use this to understand relationships between concepts.
    
    Args:
        concept: The concept to explore
        max_distance: How many hops away to explore
        
    Returns:
        Concept subgraph and related knowledge
    """
    _ensure_initialized()
    
    try:
        # Explore the concept
        result = _KNOWLEDGE_API.explore(
            concept=concept,
            max_distance=max_distance
        )
        
        # Add metadata about the exploration
        result["status"] = "success"
        result["timestamp"] = time.time()
        
        return result
    except Exception as e:
        logger.error(f"Error exploring concept: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def create_reasoning_workspace(goal: str) -> Dict[str, Any]:
    """
    Tool: Create a new reasoning workspace
    
    Use this to start a multi-step reasoning process.
    
    Args:
        goal: The reasoning goal
        
    Returns:
        Workspace information
    """
    _ensure_initialized()
    
    try:
        # Create a new workspace
        workspace = ReasoningWorkspace(_KNOWLEDGE_API, goal)
        
        # Store in active workspaces
        _ACTIVE_WORKSPACES[workspace.id] = workspace
        
        return {
            "status": "success",
            "workspace_id": workspace.id,
            "goal": goal,
            "created_at": workspace.started_at
        }
    except Exception as e:
        logger.error(f"Error creating reasoning workspace: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def workspace_add_step(workspace_id: str, step_type: str, 
                      content: str, evidence_ids: List[str] = None) -> Dict[str, Any]:
    """
    Tool: Add a reasoning step to a workspace
    
    Use this to record steps in a multi-step reasoning process.
    
    Args:
        workspace_id: The ID of the workspace
        step_type: Type of reasoning step (e.g., "observation", "inference", "hypothesis")
        content: The content of this reasoning step
        evidence_ids: IDs of supporting evidence (optional)
        
    Returns:
        Step information
    """
    _ensure_initialized()
    
    try:
        # Get the workspace
        workspace = _get_workspace(workspace_id)
        
        # Add the reasoning step
        step = workspace.add_reasoning_step(
            step_type=step_type,
            content=content,
            evidence_ids=evidence_ids or []
        )
        
        return {
            "status": "success",
            "workspace_id": workspace_id,
            "step_number": step["step_number"],
            "step_type": step_type,
            "content": content
        }
    except Exception as e:
        logger.error(f"Error adding reasoning step: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def workspace_derive_knowledge(workspace_id: str, content: str, 
                              confidence: float, evidence_ids: List[str] = None) -> Dict[str, Any]:
    """
    Tool: Derive new knowledge in a reasoning workspace
    
    Use this to record a conclusion or derived knowledge from reasoning.
    
    Args:
        workspace_id: The ID of the workspace
        content: The derived knowledge
        confidence: Confidence in this knowledge (0.0-1.0)
        evidence_ids: IDs of supporting evidence (optional)
        
    Returns:
        Information about the derived knowledge
    """
    _ensure_initialized()
    
    try:
        # Get the workspace
        workspace = _get_workspace(workspace_id)
        
        # Derive the knowledge
        derived = workspace.derive_knowledge(
            content=content,
            confidence=confidence,
            evidence_ids=evidence_ids or []
        )
        
        return {
            "status": "success",
            "workspace_id": workspace_id,
            "content": content,
            "confidence": confidence,
            "derived_at": derived["derived_at"]
        }
    except Exception as e:
        logger.error(f"Error deriving knowledge: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def workspace_commit_knowledge(workspace_id: str) -> Dict[str, Any]:
    """
    Tool: Commit derived knowledge to the main knowledge store
    
    Use this to finalize reasoning and save conclusions.
    
    Args:
        workspace_id: The ID of the workspace
        
    Returns:
        Information about committed knowledge
    """
    _ensure_initialized()
    
    try:
        # Get the workspace
        workspace = _get_workspace(workspace_id)
        
        # Commit the knowledge
        committed_ids = workspace.commit_knowledge()
        
        return {
            "status": "success",
            "workspace_id": workspace_id,
            "committed_count": len(committed_ids),
            "committed_ids": committed_ids
        }
    except Exception as e:
        logger.error(f"Error committing knowledge: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def workspace_get_chain(workspace_id: str) -> Dict[str, Any]:
    """
    Tool: Get the complete reasoning chain from a workspace
    
    Use this to review the full reasoning process.
    
    Args:
        workspace_id: The ID of the workspace
        
    Returns:
        Complete reasoning chain
    """
    _ensure_initialized()
    
    try:
        # Get the workspace
        workspace = _get_workspace(workspace_id)
        
        # Get the reasoning chain
        chain = workspace.get_reasoning_chain()
        
        return {
            "status": "success",
            "workspace_id": workspace_id,
            "chain": chain
        }
    except Exception as e:
        logger.error(f"Error getting reasoning chain: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def create_relationship(source_id: str, relation_type: str, 
                       target_id: str, confidence: float = 0.8) -> Dict[str, Any]:
    """
    Tool: Create a relationship between knowledge units
    
    Use this to connect pieces of knowledge.
    
    Args:
        source_id: ID of the source knowledge unit
        relation_type: Type of relationship
        target_id: ID of the target knowledge unit
        confidence: Confidence in this relationship (0.0-1.0)
        
    Returns:
        Information about the created relationship
    """
    _ensure_initialized()
    
    try:
        # Create the relationship
        relation_id = _KNOWLEDGE_API.store._add_relation(
            from_id=source_id,
            relation_type=relation_type,
            to_id=target_id,
            confidence=confidence
        )
        
        return {
            "status": "success",
            "source_id": source_id,
            "relation_type": relation_type,
            "target_id": target_id,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def create_temporal_snapshot(name: str = None) -> Dict[str, Any]:
    """
    Tool: Create a snapshot of the current knowledge state
    
    Use this to mark a point in time for later comparison.
    
    Args:
        name: Optional name for the snapshot
        
    Returns:
        Information about the created snapshot
    """
    _ensure_initialized()
    
    try:
        # Create the snapshot
        snapshot_id = _KNOWLEDGE_API.temporal.create_snapshot(name)
        
        return {
            "status": "success",
            "snapshot_id": snapshot_id,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error creating snapshot: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def compare_snapshots(snapshot1_id: str, snapshot2_id: str) -> Dict[str, Any]:
    """
    Tool: Compare two knowledge snapshots
    
    Use this to see how knowledge has changed over time.
    
    Args:
        snapshot1_id: ID of the first snapshot
        snapshot2_id: ID of the second snapshot
        
    Returns:
        Differences between the snapshots
    """
    _ensure_initialized()
    
    try:
        # Compare the snapshots
        diff = _KNOWLEDGE_API.temporal.diff_snapshots(snapshot1_id, snapshot2_id)
        
        return {
            "status": "success",
            "diff": diff
        }
    except Exception as e:
        logger.error(f"Error comparing snapshots: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _ensure_initialized():
    """Ensure the knowledge system is initialized"""
    if _KNOWLEDGE_API is None:
        raise RuntimeError("Knowledge system not initialized. Call initialize_knowledge_system first.")


def _get_workspace(workspace_id: str) -> ReasoningWorkspace:
    """Get an active workspace by ID"""
    if workspace_id not in _ACTIVE_WORKSPACES:
        raise ValueError(f"Workspace {workspace_id} not found or no longer active")
    
    return _ACTIVE_WORKSPACES[workspace_id]


# Example usage
if __name__ == "__main__":
    # Initialize the knowledge system
    initialize_knowledge_system("./knowledge/example.db")
    
    # Store some knowledge
    quantum_id = store_knowledge(
        content="Quantum computing uses quantum bits that can exist in superpositions of states.",
        source="quantum_textbook", 
        confidence=0.9, 
        domain="quantum_computing"
    )
    
    # Query the knowledge
    result = query_knowledge("How does quantum computing work?")
    print(f"Found {len(result.get('direct_results', []))} results")
    
    # Create a reasoning workspace
    workspace = create_reasoning_workspace("Understand quantum computing")
    
    # Add reasoning steps
    workspace_add_step(
        workspace["workspace_id"],
        "observation",
        "Quantum computers leverage superposition for parallel computation"
    )
    
    # Derive and commit knowledge
    workspace_derive_knowledge(
        workspace["workspace_id"],
        "Quantum computers can theoretically perform certain calculations exponentially faster than classical computers.",
        0.8
    )
    
    workspace_commit_knowledge(workspace["workspace_id"])
    
    # Shut down the knowledge system
    shutdown_knowledge_system()