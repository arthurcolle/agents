#!/usr/bin/env python3
"""
Distributed Memory System - Advanced distributed vector-based memory with 
consensus mechanisms, dynamic knowledge routing, and cross-agent memory integration.
"""

import numpy as np
import logging
import time
import pickle
import os
import json
import asyncio
import hashlib
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
import scipy.spatial.distance as distance
from holographic_memory import HolographicMemory, MemoryTrace

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("distributed_memory")

@dataclass
class KnowledgeFragment:
    """A single fragment of knowledge with verification metadata"""
    vector: np.ndarray
    content: Any
    source_agent_id: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    verification_status: str = "unverified"  # unverified, verified, disputed
    verification_count: int = 0
    verification_sources: Set[str] = field(default_factory=set)
    dispute_count: int = 0
    dispute_sources: Set[str] = field(default_factory=set)
    fragment_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest())
    tags: Set[str] = field(default_factory=set)
    
    def verify(self, source_id: str, verification_confidence: float = 1.0) -> bool:
        """Mark the fragment as verified by a source"""
        if source_id == self.source_agent_id:
            # Self-verification is not allowed
            return False
            
        if source_id in self.verification_sources:
            # Already verified by this source
            return False
            
        self.verification_sources.add(source_id)
        self.verification_count += 1
        
        # Update verification status
        if self.verification_count > self.dispute_count:
            self.verification_status = "verified"
        
        return True
        
    def dispute(self, source_id: str, dispute_reason: str = "") -> bool:
        """Mark the fragment as disputed by a source"""
        if source_id == self.source_agent_id:
            # Self-dispute is not allowed
            return False
            
        if source_id in self.dispute_sources:
            # Already disputed by this source
            return False
            
        self.dispute_sources.add(source_id)
        self.dispute_count += 1
        
        # Update verification status
        if self.dispute_count >= self.verification_count:
            self.verification_status = "disputed"
            
        return True
    
    def to_memory_trace(self) -> MemoryTrace:
        """Convert to a memory trace for holographic memory storage"""
        return MemoryTrace(
            vector=self.vector,
            content=self.content,
            timestamp=self.timestamp,
            importance=self.confidence,
            tags=self.tags,
            memory_type="semantic"
        )
    
    @classmethod
    def from_memory_trace(cls, trace: MemoryTrace, source_agent_id: str, confidence: float = 0.7) -> 'KnowledgeFragment':
        """Create a knowledge fragment from a memory trace"""
        return cls(
            vector=trace.vector,
            content=trace.content,
            source_agent_id=source_agent_id,
            confidence=confidence,
            timestamp=trace.timestamp,
            tags=trace.tags
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "fragment_id": self.fragment_id,
            "content": self.content,
            "source_agent_id": self.source_agent_id,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "verification_status": self.verification_status,
            "verification_count": self.verification_count,
            "verification_sources": list(self.verification_sources),
            "dispute_count": self.dispute_count,
            "dispute_sources": list(self.dispute_sources),
            "tags": list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], vector: Optional[np.ndarray] = None) -> 'KnowledgeFragment':
        """Create a knowledge fragment from a dictionary"""
        fragment = cls(
            vector=vector if vector is not None else np.zeros(128),  # Placeholder
            content=data["content"],
            source_agent_id=data["source_agent_id"],
            confidence=data["confidence"],
            timestamp=data["timestamp"],
            verification_status=data["verification_status"],
            verification_count=data["verification_count"],
            dispute_count=data["dispute_count"],
            fragment_id=data["fragment_id"],
            tags=set(data["tags"])
        )
        fragment.verification_sources = set(data["verification_sources"])
        fragment.dispute_sources = set(data["dispute_sources"])
        return fragment

class DistributedMemorySystem:
    """
    Advanced distributed memory system that integrates knowledge across 
    multiple agents with consensus mechanisms and verification.
    """
    
    def __init__(self, 
                agent_id: str,
                dimensions: int = 256,
                memory_path: Optional[str] = None,
                consensus_threshold: float = 0.6,
                auto_verification: bool = True,
                verification_confidence_threshold: float = 0.8,
                knowledge_propagation_enabled: bool = True,
                trust_decay_factor: float = 0.05):
        """
        Initialize the distributed memory system.
        
        Args:
            agent_id: Unique identifier for this agent
            dimensions: Vector dimensionality for knowledge representation
            memory_path: Path for persisting memory
            consensus_threshold: Threshold for reaching consensus (0-1)
            auto_verification: Whether to automatically verify compatible knowledge
            verification_confidence_threshold: Confidence threshold for auto-verification
            knowledge_propagation_enabled: Whether to propagate knowledge to peers
            trust_decay_factor: Rate at which trust decays over time
        """
        self.agent_id = agent_id
        self.dimensions = dimensions
        self.memory_path = memory_path
        self.consensus_threshold = consensus_threshold
        self.auto_verification = auto_verification
        self.verification_confidence_threshold = verification_confidence_threshold
        self.knowledge_propagation_enabled = knowledge_propagation_enabled
        self.trust_decay_factor = trust_decay_factor
        
        # Core memory components
        self.local_memory = HolographicMemory(
            dimensions=dimensions,
            memory_path=memory_path,
            auto_save=True
        )
        
        # Knowledge fragments (including those from other agents)
        self.knowledge_fragments: Dict[str, KnowledgeFragment] = {}
        
        # Agent trust model
        self.agent_trust: Dict[str, float] = {}
        self.agent_interaction_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Knowledge propagation and consensus
        self.pending_verifications: Dict[str, KnowledgeFragment] = {}
        self.verification_callbacks: Dict[str, List[Callable]] = {}
        
        # Remote agent knowledge mapping
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.remote_knowledge_index: Dict[str, Dict[str, str]] = {}
        
        # Statistics and metadata
        self.last_consensus_update = time.time()
        self.total_consensus_rounds = 0
        self.knowledge_integration_stats = {
            "total_fragments_received": 0,
            "total_fragments_verified": 0,
            "total_fragments_disputed": 0,
            "total_fragments_integrated": 0,
            "total_fragments_propagated": 0
        }
        
        logger.info(f"Distributed Memory System initialized for agent {agent_id}")
    
    def add_knowledge_fragment(self, fragment: KnowledgeFragment) -> bool:
        """
        Add a knowledge fragment to the distributed memory system.
        
        Args:
            fragment: The knowledge fragment to add
            
        Returns:
            Success status
        """
        # Skip if we already have this fragment
        if fragment.fragment_id in self.knowledge_fragments:
            return False
        
        # Add to fragment store
        self.knowledge_fragments[fragment.fragment_id] = fragment
        
        # Add to local memory for retrieval
        self.local_memory.encode(
            data=fragment.content,
            tags=fragment.tags,
            importance=fragment.confidence,
            vector=fragment.vector
        )
        
        # Update stats
        self.knowledge_integration_stats["total_fragments_received"] += 1
        
        # Attempt auto-verification if enabled
        if self.auto_verification and fragment.source_agent_id != self.agent_id:
            self._attempt_auto_verification(fragment)
        
        # Propagate knowledge if enabled and from this agent
        if self.knowledge_propagation_enabled and fragment.source_agent_id == self.agent_id:
            self._mark_for_propagation(fragment)
            self.knowledge_integration_stats["total_fragments_propagated"] += 1
        
        logger.info(f"Added knowledge fragment {fragment.fragment_id[:8]} from agent {fragment.source_agent_id}")
        return True
    
    def create_knowledge_fragment(self, content: Any, tags: Set[str] = None, 
                                confidence: float = 0.8, vector: Optional[np.ndarray] = None) -> KnowledgeFragment:
        """
        Create a new knowledge fragment from this agent.
        
        Args:
            content: The knowledge content
            tags: Set of tags for categorization
            confidence: Confidence in this knowledge (0-1)
            vector: Optional pre-computed vector representation
            
        Returns:
            Created knowledge fragment
        """
        tags = tags or set()
        
        # Create vector representation if not provided
        if vector is None:
            # Use a deterministic method to generate consistent vectors
            hash_seed = abs(hash(str(content))) % (2**32 - 1)
            np.random.seed(hash_seed)
            vector = np.random.randn(self.dimensions)
            vector = vector / np.linalg.norm(vector)
            np.random.seed()
        
        # Create the fragment
        fragment = KnowledgeFragment(
            vector=vector,
            content=content,
            source_agent_id=self.agent_id,
            confidence=confidence,
            tags=tags
        )
        
        # Add to our system
        self.add_knowledge_fragment(fragment)
        
        return fragment
    
    def verify_knowledge_fragment(self, fragment_id: str, verification_confidence: float = 1.0) -> bool:
        """
        Verify a knowledge fragment as this agent.
        
        Args:
            fragment_id: ID of the fragment to verify
            verification_confidence: Confidence in the verification
            
        Returns:
            Success status
        """
        if fragment_id not in self.knowledge_fragments:
            logger.warning(f"Cannot verify unknown fragment {fragment_id}")
            return False
        
        fragment = self.knowledge_fragments[fragment_id]
        
        # Verify the fragment
        if fragment.verify(self.agent_id, verification_confidence):
            logger.info(f"Verified knowledge fragment {fragment_id[:8]}")
            self.knowledge_integration_stats["total_fragments_verified"] += 1
            
            # Execute verification callbacks
            if fragment_id in self.verification_callbacks:
                for callback in self.verification_callbacks[fragment_id]:
                    try:
                        callback(fragment)
                    except Exception as e:
                        logger.error(f"Error executing verification callback: {e}")
                
                del self.verification_callbacks[fragment_id]
            
            return True
        else:
            logger.warning(f"Failed to verify fragment {fragment_id[:8]} (already verified or self-verification)")
            return False
    
    def dispute_knowledge_fragment(self, fragment_id: str, dispute_reason: str = "") -> bool:
        """
        Dispute a knowledge fragment as this agent.
        
        Args:
            fragment_id: ID of the fragment to dispute
            dispute_reason: Reason for the dispute
            
        Returns:
            Success status
        """
        if fragment_id not in self.knowledge_fragments:
            logger.warning(f"Cannot dispute unknown fragment {fragment_id}")
            return False
        
        fragment = self.knowledge_fragments[fragment_id]
        
        # Dispute the fragment
        if fragment.dispute(self.agent_id, dispute_reason):
            logger.info(f"Disputed knowledge fragment {fragment_id[:8]}: {dispute_reason}")
            self.knowledge_integration_stats["total_fragments_disputed"] += 1
            
            # Update trust model for the source agent
            self._update_agent_trust(fragment.source_agent_id, -0.1)
            
            return True
        else:
            logger.warning(f"Failed to dispute fragment {fragment_id[:8]} (already disputed or self-dispute)")
            return False
    
    def query_knowledge(self, query: Any, top_k: int = 5, 
                     threshold: float = 0.6,
                     tags: Optional[Set[str]] = None,
                     require_verification: bool = False,
                     trusted_sources_only: bool = False) -> List[Tuple[Any, float, str]]:
        """
        Query the distributed knowledge system.
        
        Args:
            query: The query for knowledge retrieval
            top_k: Maximum results to return
            threshold: Similarity threshold
            tags: Filter by tags
            require_verification: Only return verified knowledge
            trusted_sources_only: Only return knowledge from trusted sources
            
        Returns:
            List of (content, similarity, source_agent_id) tuples
        """
        # Use local memory for fast retrieval
        memory_results = self.local_memory.retrieve(
            query=query,
            top_k=top_k * 2,  # Get more than needed for filtering
            threshold=threshold,
            tags=tags
        )
        
        # Filter and map results
        results = []
        for content, similarity in memory_results:
            # Find the fragment that contains this content
            matching_fragments = [f for f in self.knowledge_fragments.values() if f.content == content]
            
            if not matching_fragments:
                continue
                
            fragment = matching_fragments[0]
            
            # Apply filters
            if require_verification and fragment.verification_status != "verified":
                continue
                
            if trusted_sources_only and not self._is_trusted_source(fragment.source_agent_id):
                continue
            
            results.append((content, similarity, fragment.source_agent_id))
            
            if len(results) >= top_k:
                break
        
        return results

    def request_consensus(self, fragment_id: str, 
                        agent_ids: List[str], 
                        timeout: float = 30.0) -> asyncio.Future:
        """
        Request consensus verification from other agents for a knowledge fragment.
        
        Args:
            fragment_id: ID of the fragment to verify
            agent_ids: List of agent IDs to request verification from
            timeout: Time to wait for consensus in seconds
            
        Returns:
            Future that will be resolved with consensus result
        """
        if fragment_id not in self.knowledge_fragments:
            logger.warning(f"Cannot request consensus for unknown fragment {fragment_id}")
            future = asyncio.Future()
            future.set_result({"consensus": False, "reason": "Fragment not found"})
            return future
            
        fragment = self.knowledge_fragments[fragment_id]
        
        # Create a future to track consensus
        loop = asyncio.get_event_loop()
        consensus_future = loop.create_future()
        
        # Add to pending verifications
        self.pending_verifications[fragment_id] = fragment
        
        # This function will be called when verification arrives
        def check_consensus(updated_fragment):
            total_agents = len(agent_ids) + 1  # +1 for self
            verifications = updated_fragment.verification_count
            disputes = updated_fragment.dispute_count
            
            # Check if we have enough verifications for consensus
            if verifications / total_agents >= self.consensus_threshold:
                if not consensus_future.done():
                    consensus_future.set_result({
                        "consensus": True,
                        "fragment_id": fragment_id,
                        "verification_count": verifications,
                        "dispute_count": disputes,
                        "total_agents": total_agents
                    })
                    # Remove from pending
                    if fragment_id in self.pending_verifications:
                        del self.pending_verifications[fragment_id]
            
            # Check if we have too many disputes
            elif disputes / total_agents > (1 - self.consensus_threshold):
                if not consensus_future.done():
                    consensus_future.set_result({
                        "consensus": False,
                        "fragment_id": fragment_id,
                        "verification_count": verifications,
                        "dispute_count": disputes,
                        "total_agents": total_agents,
                        "reason": "Too many disputes"
                    })
                    # Remove from pending
                    if fragment_id in self.pending_verifications:
                        del self.pending_verifications[fragment_id]
        
        # Register callback for when verifications arrive
        if fragment_id not in self.verification_callbacks:
            self.verification_callbacks[fragment_id] = []
        self.verification_callbacks[fragment_id].append(check_consensus)
        
        # Set timeout
        def on_timeout():
            if not consensus_future.done():
                # Calculate current state
                total_agents = len(agent_ids) + 1
                verifications = fragment.verification_count
                disputes = fragment.dispute_count
                
                # Determine if we have consensus
                has_consensus = verifications / total_agents >= self.consensus_threshold
                
                consensus_future.set_result({
                    "consensus": has_consensus,
                    "fragment_id": fragment_id,
                    "verification_count": verifications,
                    "dispute_count": disputes,
                    "total_agents": total_agents,
                    "reason": "Timeout" if not has_consensus else None
                })
                
                # Clean up
                if fragment_id in self.pending_verifications:
                    del self.pending_verifications[fragment_id]
        
        # Schedule timeout
        loop.call_later(timeout, on_timeout)
        
        # Return future
        return consensus_future
    
    def receive_verification_request(self, fragment: KnowledgeFragment, 
                                  requesting_agent_id: str) -> Dict[str, Any]:
        """
        Process a verification request from another agent.
        
        Args:
            fragment: The fragment to verify
            requesting_agent_id: ID of the agent requesting verification
            
        Returns:
            Verification response
        """
        # Skip if this is our own fragment
        if fragment.source_agent_id == self.agent_id:
            return {
                "verified": False,
                "reason": "Self-verification not allowed",
                "fragment_id": fragment.fragment_id
            }
        
        # First check if we already have this fragment
        if fragment.fragment_id in self.knowledge_fragments:
            existing_fragment = self.knowledge_fragments[fragment.fragment_id]
            
            # Check if we've already verified or disputed
            if self.agent_id in existing_fragment.verification_sources:
                return {
                    "verified": True,
                    "fragment_id": fragment.fragment_id
                }
            elif self.agent_id in existing_fragment.dispute_sources:
                return {
                    "verified": False,
                    "reason": "Previously disputed",
                    "fragment_id": fragment.fragment_id
                }
        
        # Add the fragment to our system
        self.add_knowledge_fragment(fragment)
        
        # Perform auto-verification
        if self.auto_verification:
            verification_result = self._attempt_auto_verification(fragment)
            
            if verification_result["verified"]:
                return {
                    "verified": True,
                    "confidence": verification_result["confidence"],
                    "fragment_id": fragment.fragment_id
                }
        
        # Otherwise, this needs manual verification
        return {
            "verified": False,
            "reason": "Requires manual verification",
            "fragment_id": fragment.fragment_id,
            "pending": True
        }
    
    def handle_verification_response(self, response: Dict[str, Any], 
                                  source_agent_id: str) -> bool:
        """
        Handle a verification response from another agent.
        
        Args:
            response: The verification response
            source_agent_id: ID of the agent that sent the response
            
        Returns:
            Success status
        """
        fragment_id = response.get("fragment_id")
        
        if not fragment_id or fragment_id not in self.knowledge_fragments:
            logger.warning(f"Received verification for unknown fragment {fragment_id}")
            return False
        
        fragment = self.knowledge_fragments[fragment_id]
        
        if response.get("verified", False):
            # It was verified
            fragment.verify(source_agent_id, response.get("confidence", 1.0))
            logger.info(f"Fragment {fragment_id[:8]} verified by agent {source_agent_id}")
            
            # Update trust model
            if fragment.source_agent_id == self.agent_id:
                # Agent verified our knowledge, increase trust
                self._update_agent_trust(source_agent_id, 0.05)
            
            return True
        else:
            # It was disputed
            reason = response.get("reason", "No reason provided")
            fragment.dispute(source_agent_id, reason)
            logger.info(f"Fragment {fragment_id[:8]} disputed by agent {source_agent_id}: {reason}")
            
            # Update trust model
            if fragment.source_agent_id == self.agent_id:
                # Agent disputed our knowledge, decrease trust slightly
                self._update_agent_trust(source_agent_id, -0.02)
            
            return True
    
    def integrate_memory(self, other_memory: HolographicMemory, 
                      source_agent_id: str,
                      confidence: float = 0.7,
                      tags: Optional[Set[str]] = None,
                      trusted_only: bool = True) -> int:
        """
        Integrate memories from another agent's holographic memory.
        
        Args:
            other_memory: HolographicMemory instance from another agent
            source_agent_id: ID of the source agent
            confidence: Base confidence for imported memories
            tags: Additional tags to add to imported memories
            trusted_only: Only import if source agent is trusted
            
        Returns:
            Number of memories integrated
        """
        if trusted_only and not self._is_trusted_source(source_agent_id):
            logger.warning(f"Skipping memory integration from untrusted agent {source_agent_id}")
            return 0
        
        tags = tags or set()
        integrated_count = 0
        
        # Process each memory trace from the other memory
        for trace in other_memory.memory_traces:
            # Create knowledge fragment from memory trace
            fragment = KnowledgeFragment.from_memory_trace(
                trace=trace,
                source_agent_id=source_agent_id,
                confidence=confidence
            )
            
            # Add additional tags
            fragment.tags.update(tags)
            
            # Add to our system
            if self.add_knowledge_fragment(fragment):
                integrated_count += 1
        
        logger.info(f"Integrated {integrated_count} memories from agent {source_agent_id}")
        self.knowledge_integration_stats["total_fragments_integrated"] += integrated_count
        
        # Update trust model if successful integration
        if integrated_count > 0:
            self._update_agent_trust(source_agent_id, 0.02)
        
        return integrated_count
    
    def get_knowledge_for_agent(self, target_agent_id: str, 
                             max_fragments: int = 10,
                             min_confidence: float = 0.7,
                             tags: Optional[Set[str]] = None) -> List[KnowledgeFragment]:
        """
        Get knowledge fragments to share with another agent based on their capabilities.
        
        Args:
            target_agent_id: ID of the target agent
            max_fragments: Maximum fragments to return
            min_confidence: Minimum confidence threshold
            tags: Filter by tags
            
        Returns:
            List of knowledge fragments
        """
        tags = tags or set()
        
        # Get agent capabilities if available
        agent_tags = self.agent_capabilities.get(target_agent_id, set())
        if agent_tags:
            # Include agent-specific tags in search
            tags.update(agent_tags)
        
        # Find relevant fragments
        if tags:
            relevant_fragments = [
                f for f in self.knowledge_fragments.values()
                if f.confidence >= min_confidence and 
                (not tags or any(tag in f.tags for tag in tags)) and
                f.source_agent_id == self.agent_id  # Only share our own knowledge
            ]
        else:
            # No tags specified, just use confidence
            relevant_fragments = [
                f for f in self.knowledge_fragments.values()
                if f.confidence >= min_confidence and
                f.source_agent_id == self.agent_id  # Only share our own knowledge
            ]
        
        # Sort by confidence (descending)
        relevant_fragments.sort(key=lambda f: f.confidence, reverse=True)
        
        # Limit to max_fragments
        return relevant_fragments[:max_fragments]
    
    def _attempt_auto_verification(self, fragment: KnowledgeFragment) -> Dict[str, Any]:
        """
        Attempt to automatically verify a knowledge fragment.
        
        Args:
            fragment: Fragment to verify
            
        Returns:
            Verification result
        """
        # Skip if it's our own fragment
        if fragment.source_agent_id == self.agent_id:
            return {"verified": False, "reason": "Self-verification not allowed"}
        
        # Skip if source agent is not trusted
        if not self._is_trusted_source(fragment.source_agent_id):
            return {"verified": False, "reason": "Source agent not trusted"}
        
        # Check if we have similar knowledge
        results = self.local_memory.retrieve_by_similarity(
            query_vector=fragment.vector,
            top_k=5,
            threshold=self.verification_confidence_threshold
        )
        
        if not results:
            return {"verified": False, "reason": "No similar knowledge found"}
        
        # Check the best match
        best_match, similarity = results[0]
        
        # Get the fragment for this match
        matching_fragments = [f for f in self.knowledge_fragments.values() 
                             if f.content == best_match.content and 
                             f.source_agent_id == self.agent_id]
        
        if matching_fragments:
            # We have a similar fragment from ourselves, use it for verification
            confidence = similarity * matching_fragments[0].confidence
            
            if confidence >= self.verification_confidence_threshold:
                # Verify the fragment
                fragment.verify(self.agent_id, confidence)
                logger.info(f"Auto-verified fragment {fragment.fragment_id[:8]} with confidence {confidence:.3f}")
                
                return {
                    "verified": True,
                    "confidence": confidence,
                    "matching_fragment_id": matching_fragments[0].fragment_id
                }
        
        return {"verified": False, "reason": "No compatible knowledge found"}
    
    def _update_agent_trust(self, agent_id: str, change: float) -> float:
        """
        Update trust level for an agent.
        
        Args:
            agent_id: ID of the agent
            change: Amount to change trust (+/-)
            
        Returns:
            New trust level
        """
        # Initialize if not exists
        if agent_id not in self.agent_trust:
            self.agent_trust[agent_id] = 0.5  # Default neutral trust
            
        # Update trust
        current_trust = self.agent_trust[agent_id]
        new_trust = max(0.0, min(1.0, current_trust + change))  # Clamp to [0,1]
        self.agent_trust[agent_id] = new_trust
        
        # Record interaction
        if agent_id not in self.agent_interaction_history:
            self.agent_interaction_history[agent_id] = []
            
        self.agent_interaction_history[agent_id].append({
            "timestamp": time.time(),
            "change": change,
            "new_trust": new_trust
        })
        
        return new_trust
    
    def _is_trusted_source(self, agent_id: str) -> bool:
        """Check if an agent is trusted"""
        # Trust our own knowledge
        if agent_id == self.agent_id:
            return True
            
        # Check trust level
        trust_level = self.agent_trust.get(agent_id, 0.5)  # Default neutral trust
        return trust_level >= 0.6  # Trust threshold
    
    def _update_trust_decay(self) -> None:
        """Apply trust decay over time"""
        current_time = time.time()
        
        for agent_id, trust_level in list(self.agent_trust.items()):
            # Skip self
            if agent_id == self.agent_id:
                continue
                
            # Get last interaction time
            last_interaction = 0
            if agent_id in self.agent_interaction_history and self.agent_interaction_history[agent_id]:
                last_interaction = self.agent_interaction_history[agent_id][-1]["timestamp"]
                
            # Calculate days since last interaction
            days_since_interaction = (current_time - last_interaction) / (24 * 3600)
            
            if days_since_interaction > 1:  # Only decay after a day
                # Apply decay
                decay = self.trust_decay_factor * np.log1p(days_since_interaction)
                new_trust = max(0.3, trust_level - decay)  # Don't go below 0.3
                
                # Update trust without recording interaction
                self.agent_trust[agent_id] = new_trust
    
    def _mark_for_propagation(self, fragment: KnowledgeFragment) -> None:
        """Mark a fragment for propagation to other agents"""
        # This is a placeholder - in a real implementation, this would 
        # interface with the agent communication system to propagate
        # knowledge to other agents
        pass
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about the distributed memory system"""
        return {
            "agent_id": self.agent_id,
            "knowledge_fragments": len(self.knowledge_fragments),
            "verified_fragments": len([f for f in self.knowledge_fragments.values() 
                                    if f.verification_status == "verified"]),
            "disputed_fragments": len([f for f in self.knowledge_fragments.values() 
                                    if f.verification_status == "disputed"]),
            "local_memories": len(self.local_memory.memory_traces),
            "known_agents": len(self.agent_trust),
            "trusted_agents": len([a for a, t in self.agent_trust.items() if t >= 0.6]),
            "integration_stats": self.knowledge_integration_stats
        }
    
    def save_state(self, path: Optional[str] = None) -> bool:
        """
        Save the state of the distributed memory system.
        
        Args:
            path: Path to save the state (defaults to memory_path)
            
        Returns:
            Success status
        """
        save_path = path or self.memory_path
        if not save_path:
            logger.warning("No path specified for state save")
            return False
            
        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save local memory
            self.local_memory.save_memory()
            
            # Prepare state data
            state_data = {
                "agent_id": self.agent_id,
                "dimensions": self.dimensions,
                "knowledge_fragments": {
                    fid: {
                        "data": fragment.to_dict(),
                        "vector_b64": base64.b64encode(fragment.vector.tobytes()).decode("utf-8")
                    }
                    for fid, fragment in self.knowledge_fragments.items()
                },
                "agent_trust": self.agent_trust,
                "agent_interaction_history": self.agent_interaction_history,
                "agent_capabilities": {
                    aid: list(caps) for aid, caps in self.agent_capabilities.items()
                },
                "knowledge_integration_stats": self.knowledge_integration_stats,
                "timestamp": time.time(),
                "version": "1.0"
            }
            
            # Save to file
            state_path = f"{save_path}.state.json"
            with open(state_path, 'w') as f:
                json.dump(state_data, f)
                
            logger.info(f"Distributed memory state saved to {state_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving distributed memory state: {e}")
            return False
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """
        Load the state of the distributed memory system.
        
        Args:
            path: Path to load the state from (defaults to memory_path)
            
        Returns:
            Success status
        """
        load_path = path or self.memory_path
        if not load_path:
            logger.warning("No path specified for state load")
            return False
            
        state_path = f"{load_path}.state.json"
        if not os.path.exists(state_path):
            logger.warning(f"State file not found at {state_path}")
            return False
            
        try:
            # Load state data
            with open(state_path, 'r') as f:
                state_data = json.load(f)
                
            # Verify agent ID
            if state_data["agent_id"] != self.agent_id:
                logger.warning(f"State file agent ID mismatch: {state_data['agent_id']} != {self.agent_id}")
                return False
                
            # Set dimensions
            self.dimensions = state_data["dimensions"]
                
            # Load knowledge fragments
            import base64
            self.knowledge_fragments = {}
            for fid, fragment_data in state_data["knowledge_fragments"].items():
                vector_bytes = base64.b64decode(fragment_data["vector_b64"])
                vector = np.frombuffer(vector_bytes, dtype=np.float64).reshape(-1)
                
                # Create fragment from dict
                fragment = KnowledgeFragment.from_dict(fragment_data["data"], vector)
                self.knowledge_fragments[fid] = fragment
                
            # Load agent trust data
            self.agent_trust = state_data["agent_trust"]
            self.agent_interaction_history = state_data["agent_interaction_history"]
            
            # Load agent capabilities
            self.agent_capabilities = {
                aid: set(caps) for aid, caps in state_data["agent_capabilities"].items()
            }
            
            # Load stats
            self.knowledge_integration_stats = state_data["knowledge_integration_stats"]
            
            # Local memory should be loaded separately
            if self.memory_path:
                self.local_memory.load_memory()
                
            logger.info(f"Distributed memory state loaded from {state_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading distributed memory state: {e}")
            return False

# Run test if called directly
if __name__ == "__main__":
    # Create distributed memory system
    memory_system = DistributedMemorySystem(
        agent_id="test-agent-1",
        dimensions=128
    )
    
    # Add some knowledge
    fragment1 = memory_system.create_knowledge_fragment(
        content="The Earth orbits around the Sun",
        tags={"astronomy", "science"}
    )
    
    fragment2 = memory_system.create_knowledge_fragment(
        content="Python is a programming language",
        tags={"programming", "computers"}
    )
    
    # Create a second agent's memory
    agent2_memory = DistributedMemorySystem(
        agent_id="test-agent-2",
        dimensions=128
    )
    
    # Add knowledge to second agent
    fragment3 = agent2_memory.create_knowledge_fragment(
        content="The Earth is the third planet from the Sun",
        tags={"astronomy", "science"}
    )
    
    # Simulate knowledge exchange
    print("\nSimulating knowledge exchange...")
    # Get knowledge from agent 1 to share
    fragments_to_share = memory_system.get_knowledge_for_agent(
        target_agent_id="test-agent-2",
        tags={"science"}
    )
    
    # Agent 2 receives knowledge
    for fragment in fragments_to_share:
        response = agent2_memory.receive_verification_request(
            fragment=fragment,
            requesting_agent_id="test-agent-1"
        )
        print(f"Agent 2 received fragment: {fragment.content}")
        print(f"Verification response: {response}")
    
    # Print memory stats
    print("\nAgent 1 Stats:")
    for key, value in memory_system.get_agent_stats().items():
        print(f"  {key}: {value}")
    
    print("\nAgent 2 Stats:")
    for key, value in agent2_memory.get_agent_stats().items():
        print(f"  {key}: {value}")
    
    # Query knowledge
    print("\nQuerying knowledge in Agent 2:")
    results = agent2_memory.query_knowledge(
        query="What orbits the sun?",
        top_k=3
    )
    
    for content, similarity, source_id in results:
        print(f"  {content} (Similarity: {similarity:.3f}, Source: {source_id})")