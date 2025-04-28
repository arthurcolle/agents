#!/usr/bin/env python3
"""
Test script for the distributed memory system with consensus verification.
Demonstrates knowledge sharing, verification, dispute resolution, and trust-based knowledge integration
across multiple agents.
"""

import asyncio
import logging
import time
import os
import numpy as np
from typing import Dict, Set, List, Any, Tuple
from distributed_memory_system import DistributedMemorySystem, KnowledgeFragment

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("distributed-memory-test")

# Test data for different domains
ASTRONOMY_FACTS = [
    ("The Earth orbits around the Sun", {"astronomy", "science", "solar system"}),
    ("Mars is the fourth planet from the Sun", {"astronomy", "science", "planets", "solar system"}),
    ("Jupiter is the largest planet in our solar system", {"astronomy", "science", "planets", "solar system"}),
    ("A light year is the distance light travels in one year", {"astronomy", "science", "measurement"}),
    ("The Sun is a G-type main-sequence star", {"astronomy", "science", "stars", "solar system"}),
]

PHYSICS_FACTS = [
    ("The speed of light in vacuum is approximately 299,792,458 meters per second", {"physics", "science", "constants"}),
    ("E=mc² is Einstein's mass-energy equivalence formula", {"physics", "science", "relativity"}),
    ("Quantum mechanics describes nature at the atomic and subatomic scales", {"physics", "science", "quantum"}),
    ("Newton's third law states that for every action, there is an equal and opposite reaction", {"physics", "science", "mechanics"}),
    ("The standard model describes fundamental particles and their interactions", {"physics", "science", "particle physics"}),
]

COMPUTER_SCIENCE_FACTS = [
    ("Python is a high-level programming language", {"programming", "computers", "software"}),
    ("A database is an organized collection of data", {"computers", "data", "software"}),
    ("Machine learning is a subset of artificial intelligence", {"computers", "AI", "data science"}),
    ("HTTP is the foundation of data communication for the web", {"computers", "networking", "internet"}),
    ("Algorithms are step-by-step procedures for calculations", {"computers", "mathematics", "programming"}),
]

# Disputed and controversial facts (some correct, some incorrect)
DISPUTED_FACTS = [
    ("Pluto is classified as a dwarf planet, not a planet", {"astronomy", "science", "planets", "solar system"}),  # Correct but might be disputed
    ("The Earth is flat", {"astronomy", "science", "earth"}),  # Incorrect
    ("Humans only use 10% of their brains", {"biology", "science", "brain"}),  # Incorrect
    ("Cold fusion is a viable energy source", {"physics", "science", "energy"}),  # Disputed
    ("Quantum computers can break all encryption", {"computers", "quantum", "security"}),  # Oversimplified
]

class SimulatedAgentNetwork:
    """Simulate a network of distributed memory agents with different specialties"""
    
    def __init__(self):
        self.agents: Dict[str, DistributedMemorySystem] = {}
        self.agent_specialties: Dict[str, Set[str]] = {}
    
    def create_agent(self, agent_id: str, specialties: Set[str], dimensions: int = 128) -> DistributedMemorySystem:
        """Create a new agent with specified specialties"""
        # Create agent
        agent = DistributedMemorySystem(
            agent_id=agent_id,
            dimensions=dimensions
        )
        
        # Store agent and its specialties
        self.agents[agent_id] = agent
        self.agent_specialties[agent_id] = specialties
        
        logger.info(f"Created agent {agent_id} with specialties: {specialties}")
        return agent
    
    def populate_agent_knowledge(self, agent_id: str) -> None:
        """Populate an agent with knowledge based on its specialties"""
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return
        
        agent = self.agents[agent_id]
        specialties = self.agent_specialties[agent_id]
        
        # Add knowledge based on specialties
        knowledge_added = 0
        
        if "astronomy" in specialties:
            for fact, tags in ASTRONOMY_FACTS:
                agent.create_knowledge_fragment(content=fact, tags=tags, confidence=0.9)
                knowledge_added += 1
        
        if "physics" in specialties:
            for fact, tags in PHYSICS_FACTS:
                agent.create_knowledge_fragment(content=fact, tags=tags, confidence=0.9)
                knowledge_added += 1
        
        if "computers" in specialties:
            for fact, tags in COMPUTER_SCIENCE_FACTS:
                agent.create_knowledge_fragment(content=fact, tags=tags, confidence=0.9)
                knowledge_added += 1
        
        # Add some disputed knowledge (but only to certain agents)
        if "disputed" in specialties:
            for fact, tags in DISPUTED_FACTS:
                # Add with lower confidence to represent uncertainty
                agent.create_knowledge_fragment(content=fact, tags=tags, confidence=0.6)
                knowledge_added += 1
        
        logger.info(f"Added {knowledge_added} knowledge fragments to agent {agent_id}")
    
    def connect_agents(self) -> None:
        """Connect agents by registering their capabilities"""
        for agent_id, agent in self.agents.items():
            # Register other agents' capabilities
            for other_id, specialties in self.agent_specialties.items():
                if other_id != agent_id:
                    agent.agent_capabilities[other_id] = specialties
                    # Initialize with moderate trust
                    agent._update_agent_trust(other_id, 0.1)
    
    async def simulate_knowledge_exchange(self, rounds: int = 3) -> Dict[str, Any]:
        """Simulate knowledge exchange between agents for a number of rounds"""
        exchange_stats = {
            "rounds": rounds,
            "total_fragments_shared": 0,
            "total_verifications": 0,
            "total_disputes": 0,
            "consensus_reached": 0,
            "trust_changes": {}
        }
        
        # Record initial trust levels
        initial_trust = {}
        for agent_id, agent in self.agents.items():
            initial_trust[agent_id] = {other: level for other, level in agent.agent_trust.items()}
        
        # Run multiple rounds of exchange
        for round_num in range(1, rounds + 1):
            logger.info(f"\n=== Starting Knowledge Exchange Round {round_num} ===")
            
            # For each agent, share knowledge with others
            for agent_id, agent in self.agents.items():
                # Find knowledge to share based on other agents' specialties
                for other_id, other_agent in self.agents.items():
                    if other_id == agent_id:
                        continue
                    
                    # Get knowledge relevant to the other agent's specialties
                    other_specialties = self.agent_specialties[other_id]
                    fragments_to_share = agent.get_knowledge_for_agent(
                        target_agent_id=other_id,
                        tags=other_specialties,
                        min_confidence=0.7
                    )
                    
                    if not fragments_to_share:
                        continue
                    
                    # Share a subset of fragments (to limit the test volume)
                    fragments_to_share = fragments_to_share[:2]
                    exchange_stats["total_fragments_shared"] += len(fragments_to_share)
                    
                    logger.info(f"Agent {agent_id} sharing {len(fragments_to_share)} fragments with {other_id}")
                    
                    # Process each fragment
                    for fragment in fragments_to_share:
                        # Send to other agent for verification
                        response = other_agent.receive_verification_request(
                            fragment=fragment,
                            requesting_agent_id=agent_id
                        )
                        
                        if response.get("verified", False):
                            exchange_stats["total_verifications"] += 1
                            logger.info(f"Agent {other_id} verified '{fragment.content[:30]}...'")
                        elif response.get("pending", False):
                            # This would normally be handled by user intervention or another process
                            # For testing, we'll simulate a verification or dispute based on the content
                            
                            # Check if it's a disputed fact
                            is_disputed = any(fragment.content == fact for fact, _ in DISPUTED_FACTS)
                            
                            if is_disputed and "critical" in other_specialties:
                                # Critical agents will dispute controversial facts
                                other_agent.dispute_knowledge_fragment(
                                    fragment_id=fragment.fragment_id,
                                    dispute_reason="Content appears to be disputed by current knowledge"
                                )
                                exchange_stats["total_disputes"] += 1
                                logger.info(f"Agent {other_id} disputed '{fragment.content[:30]}...'")
                            else:
                                # Otherwise verify it
                                other_agent.verify_knowledge_fragment(
                                    fragment_id=fragment.fragment_id,
                                    verification_confidence=0.8
                                )
                                exchange_stats["total_verifications"] += 1
                                logger.info(f"Agent {other_id} manually verified '{fragment.content[:30]}...'")
                        else:
                            exchange_stats["total_disputes"] += 1
                            logger.info(f"Agent {other_id} disputed '{fragment.content[:30]}...' - {response.get('reason', 'No reason')}")
            
            # Allow time for processing
            await asyncio.sleep(0.5)
        
        # Record final trust levels and changes
        for agent_id, agent in self.agents.items():
            trust_changes = {}
            for other_id, final_trust in agent.agent_trust.items():
                if other_id in initial_trust.get(agent_id, {}):
                    initial = initial_trust[agent_id][other_id]
                    change = final_trust - initial
                    trust_changes[other_id] = {
                        "initial": initial,
                        "final": final_trust,
                        "change": change
                    }
            exchange_stats["trust_changes"][agent_id] = trust_changes
        
        return exchange_stats
    
    async def test_consensus_verification(self, fact: str, tags: Set[str]) -> Dict[str, Any]:
        """Test consensus verification across multiple agents"""
        # Select astronomy agents for this test
        astronomy_agents = [
            agent_id for agent_id, specialties in self.agent_specialties.items()
            if "astronomy" in specialties
        ]
        
        if len(astronomy_agents) < 3:
            logger.warning("Need at least 3 astronomy agents for consensus test")
            return {"success": False, "reason": "Not enough astronomy agents"}
        
        # Choose one agent to create knowledge
        source_agent_id = astronomy_agents[0]
        source_agent = self.agents[source_agent_id]
        
        # Create the fragment
        fragment = source_agent.create_knowledge_fragment(
            content=fact,
            tags=tags,
            confidence=0.85
        )
        
        logger.info(f"\n=== Testing Consensus Verification ===")
        logger.info(f"Agent {source_agent_id} created fragment: '{fact}'")
        
        # Select verification targets (other astronomy agents)
        verification_targets = astronomy_agents[1:]
        
        # Request consensus
        logger.info(f"Requesting consensus from {len(verification_targets)} agents...")
        consensus_future = source_agent.request_consensus(
            fragment_id=fragment.fragment_id,
            agent_ids=verification_targets,
            timeout=5.0
        )
        
        # Simulate verification by other agents
        for target_id in verification_targets:
            target_agent = self.agents[target_id]
            
            # Verify the fragment
            response = target_agent.receive_verification_request(
                fragment=fragment,
                requesting_agent_id=source_agent_id
            )
            
            # If not auto-verified, manually verify
            if not response.get("verified", False) and response.get("pending", False):
                # Check if it's a controversial fact
                is_disputed = any(fragment.content == fact for fact, _ in DISPUTED_FACTS)
                
                if is_disputed and "critical" in self.agent_specialties[target_id]:
                    # Dispute it
                    target_agent.dispute_knowledge_fragment(
                        fragment_id=fragment.fragment_id,
                        dispute_reason="Content appears to be disputed by current knowledge"
                    )
                    logger.info(f"Agent {target_id} disputed the fragment")
                else:
                    # Verify it
                    target_agent.verify_knowledge_fragment(
                        fragment_id=fragment.fragment_id,
                        verification_confidence=0.8
                    )
                    logger.info(f"Agent {target_id} verified the fragment")
        
        # Wait for consensus result
        consensus_result = await consensus_future
        
        logger.info(f"Consensus result: {consensus_result}")
        return consensus_result

    def test_query_across_all_agents(self, query: str, top_k: int = 3) -> None:
        """Test querying knowledge across all agents"""
        logger.info(f"\n=== Testing Knowledge Query: '{query}' ===")
        
        all_results = []
        
        # Query each agent
        for agent_id, agent in self.agents.items():
            results = agent.query_knowledge(
                query=query,
                top_k=top_k,
                threshold=0.1  # Lower threshold for testing
            )
            
            if results:
                logger.info(f"Results from agent {agent_id}:")
                for content, similarity, source_id in results:
                    logger.info(f"  - {content} (Similarity: {similarity:.3f}, Source: {source_id})")
                    all_results.append((content, similarity, source_id, agent_id))
            else:
                logger.info(f"No results from agent {agent_id}")
        
        # Analyze consolidated results
        if all_results:
            # Sort by similarity
            all_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("\nConsolidated top results across all agents:")
            for i, (content, similarity, source_id, agent_id) in enumerate(all_results[:5]):
                logger.info(f"{i+1}. {content} (Similarity: {similarity:.3f}, Source: {source_id}, Found by: {agent_id})")
            
            # Check for consensus
            contents = [r[0] for r in all_results]
            most_common = max(set(contents), key=contents.count)
            consensus_level = contents.count(most_common) / len(contents)
            
            logger.info(f"\nMost consistent result: '{most_common}'")
            logger.info(f"Consensus level: {consensus_level:.2f} ({contents.count(most_common)}/{len(contents)} agents)")
        else:
            logger.info("No results found by any agent")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        stats = {
            "total_agents": len(self.agents),
            "total_knowledge_fragments": sum(len(agent.knowledge_fragments) for agent in self.agents.values()),
            "total_verified_fragments": sum(
                len([f for f in agent.knowledge_fragments.values() if f.verification_status == "verified"])
                for agent in self.agents.values()
            ),
            "total_disputed_fragments": sum(
                len([f for f in agent.knowledge_fragments.values() if f.verification_status == "disputed"])
                for agent in self.agents.values()
            ),
            "average_trust_level": 0,
            "agent_stats": {}
        }
        
        # Calculate average trust
        trust_values = []
        for agent in self.agents.values():
            trust_values.extend(agent.agent_trust.values())
        
        if trust_values:
            stats["average_trust_level"] = sum(trust_values) / len(trust_values)
        
        # Add individual agent stats
        for agent_id, agent in self.agents.items():
            stats["agent_stats"][agent_id] = agent.get_agent_stats()
        
        return stats

async def run_tests():
    """Run the distributed memory system tests"""
    # Create agent network
    network = SimulatedAgentNetwork()
    
    # Create agents with different specialties
    network.create_agent("astronomy-1", {"astronomy", "science"})
    network.create_agent("astronomy-2", {"astronomy", "science", "physics"})
    network.create_agent("physics-1", {"physics", "science"})
    network.create_agent("computers-1", {"computers", "programming"})
    network.create_agent("critic-1", {"astronomy", "physics", "critical"})
    
    # Populate agents with knowledge
    for agent_id in network.agents:
        network.populate_agent_knowledge(agent_id)
    
    # Connect agents
    network.connect_agents()
    
    # Print initial stats
    logger.info("\n=== Initial System State ===")
    stats = network.get_system_stats()
    logger.info(f"Total Agents: {stats['total_agents']}")
    logger.info(f"Total Knowledge Fragments: {stats['total_knowledge_fragments']}")
    logger.info(f"Total Verified Fragments: {stats['total_verified_fragments']}")
    logger.info(f"Total Disputed Fragments: {stats['total_disputed_fragments']}")
    
    # Test knowledge exchange
    exchange_stats = await network.simulate_knowledge_exchange(rounds=2)
    
    # Test consensus verification (with correct fact)
    consensus_result = await network.test_consensus_verification(
        fact="Neptune is the eighth planet from the Sun",
        tags={"astronomy", "science", "planets", "solar system"}
    )
    
    # Test consensus verification (with disputed fact)
    disputed_consensus = await network.test_consensus_verification(
        fact="The Earth is flat",
        tags={"astronomy", "science", "earth"}
    )
    
    # Test querying across agents
    network.test_query_across_all_agents("What is the largest planet?")
    network.test_query_across_all_agents("Tell me about programming languages")
    network.test_query_across_all_agents("What is quantum mechanics?")
    
    # Print final stats
    logger.info("\n=== Final System State ===")
    final_stats = network.get_system_stats()
    logger.info(f"Total Agents: {final_stats['total_agents']}")
    logger.info(f"Total Knowledge Fragments: {final_stats['total_knowledge_fragments']}")
    logger.info(f"Total Verified Fragments: {final_stats['total_verified_fragments']}")
    logger.info(f"Total Disputed Fragments: {final_stats['total_disputed_fragments']}")
    logger.info(f"Average Trust Level: {final_stats['average_trust_level']:.3f}")
    
    # Print trust changes
    logger.info("\n=== Agent Trust Changes ===")
    for agent_id, trust_changes in exchange_stats["trust_changes"].items():
        logger.info(f"Agent {agent_id} trust changes:")
        for other_id, change_data in trust_changes.items():
            logger.info(f"  - {other_id}: {change_data['initial']:.3f} → {change_data['final']:.3f} (Δ{change_data['change']:.3f})")
    
    return final_stats

def main():
    """Main function"""
    asyncio.run(run_tests())

if __name__ == "__main__":
    main()