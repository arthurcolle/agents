#!/usr/bin/env python3
"""
Test script for the enhanced holographic memory system.
Demonstrates key features including encoding, retrieval, forgetting curves,
and memory merging capabilities.
"""

import numpy as np
import time
import os
import logging
from holographic_memory import HolographicMemory

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory-test")

def test_basic_memory_operations():
    """Test basic memory operations"""
    logger.info("=== Testing Basic Memory Operations ===")
    
    # Create memory
    memory = HolographicMemory(dimensions=128, auto_save=False)
    
    # Add semantic memories
    logger.info("Adding semantic memories...")
    facts = [
        ("Paris is the capital of France", {"geography", "europe", "cities"}),
        ("The speed of light is approximately 299,792,458 meters per second", {"physics", "science"}),
        ("Water boils at 100 degrees Celsius at sea level", {"chemistry", "science"}),
        ("The Earth orbits around the Sun", {"astronomy", "science"}),
        ("Python is a programming language", {"programming", "computers"}),
    ]
    
    for fact, tags in facts:
        memory.encode(fact, tags=tags, memory_type="semantic", importance=1.0)
        
    # Add episodic memories
    logger.info("Adding episodic memories...")
    experiences = [
        ("I enjoyed a beautiful sunset yesterday", {"personal", "nature"}),
        ("The conference presentation went really well", {"work", "events"}),
        ("Had a great conversation with my friend about philosophy", {"personal", "conversation"}),
    ]
    
    for experience, tags in experiences:
        memory.encode(experience, tags=tags, memory_type="episodic", importance=0.8)
    
    # Test basic retrieval
    logger.info("Testing basic retrieval...")
    queries = [
        "What is the capital of France?",
        "Tell me about light speed",
        "information about Python programming",
        "personal experiences about nature",
        "What temperature does water boil?",
    ]
    
    for query in queries:
        logger.info(f"\nQuery: {query}")
        results = memory.retrieve(query, top_k=2, threshold=0.1)
        for content, similarity in results:
            logger.info(f"  Result: {content} (Similarity: {similarity:.3f})")
    
    # Test tag-based retrieval
    logger.info("\nTesting tag-based retrieval...")
    science_results = memory.retrieve("science concepts", tags={"science"}, top_k=3)
    logger.info("Science-tagged memories:")
    for content, similarity in science_results:
        logger.info(f"  {content} (Similarity: {similarity:.3f})")
    
    # Test memory type filtering
    logger.info("\nTesting memory type filtering...")
    episodic_results = memory.retrieve("my experiences", memory_type="episodic", top_k=3)
    logger.info("Episodic memories:")
    for content, similarity in episodic_results:
        logger.info(f"  {content} (Similarity: {similarity:.3f})")
    
    # Test memory stats
    logger.info("\nMemory statistics:")
    stats = memory.get_stats()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return memory

def test_forgetting_curve(memory=None):
    """Test the forgetting curve mechanism"""
    logger.info("\n=== Testing Forgetting Curve ===")
    
    if memory is None:
        # Create new memory with fast forgetting
        memory = HolographicMemory(
            dimensions=128, 
            auto_save=False,
            forgetting_factor=0.9,  # High forgetting rate
            forgetting_interval=5    # Short interval for testing
        )
        
        # Add memories with different importance
        logger.info("Adding memories with different importance levels...")
        memory.encode("Critical security protocol: Never share your password", 
                     tags={"security"}, importance=1.0)
        memory.encode("The meeting is scheduled for Friday", 
                     tags={"work"}, importance=0.5)
        memory.encode("The cafeteria had sandwiches today", 
                     tags={"food"}, importance=0.1)
    
    # Get initial stats
    logger.info("Initial memory state:")
    logger.info(f"  Total memories: {len(memory.memory_traces)}")
    
    # Apply forgetting curve
    logger.info("Applying forgetting curve...")
    # Set time in the future to simulate passage of time
    future_time = time.time() + 86400 * 7  # 7 days in the future
    pruned = memory.forget(current_time=future_time)
    
    logger.info(f"Pruned {pruned} memories")
    logger.info(f"Remaining memories: {len(memory.memory_traces)}")
    
    # Show remaining memories
    logger.info("Remaining memories:")
    for i, trace in enumerate(memory.memory_traces):
        logger.info(f"  {i}: {trace.content[:50]}... (Importance: {trace.importance:.2f})")
    
    return memory

def test_memory_access_reinforcement():
    """Test how memory access reinforces memories"""
    logger.info("\n=== Testing Memory Access Reinforcement ===")
    
    # Create memory
    memory = HolographicMemory(
        dimensions=128, 
        auto_save=False,
        forgetting_factor=0.5
    )
    
    # Add test memories
    logger.info("Adding test memories...")
    memory.encode("Frequently accessed memory", tags={"test"}, importance=0.5)
    memory.encode("Rarely accessed memory", tags={"test"}, importance=0.5)
    
    # Access the first memory multiple times
    logger.info("Accessing 'Frequently accessed memory' multiple times...")
    for _ in range(5):
        results = memory.retrieve("Frequently accessed memory", top_k=1, threshold=0.5)
        time.sleep(0.1)  # Small delay between accesses
    
    # Show access stats
    logger.info("\nAccess statistics:")
    for i, trace in enumerate(memory.memory_traces):
        logger.info(f"  Memory {i}: '{trace.content}'")
        logger.info(f"    Access count: {trace.access_count}")
        logger.info(f"    Last access: {time.time() - trace.last_access:.2f} seconds ago")
    
    # Test forgetting with access reinforcement
    logger.info("\nSimulating time passage to test forgetting with access reinforcement...")
    future_time = time.time() + 86400 * 14  # 14 days in the future
    pruned = memory.forget(current_time=future_time)
    
    logger.info(f"Pruned {pruned} memories")
    logger.info(f"Remaining memories: {len(memory.memory_traces)}")
    
    return memory

def test_memory_persistence():
    """Test saving and loading memory"""
    logger.info("\n=== Testing Memory Persistence ===")
    
    # Create a temporary path for the test
    memory_path = "/tmp/test_holographic_memory.pkl"
    
    # Create and populate memory
    logger.info(f"Creating memory and saving to {memory_path}...")
    memory = HolographicMemory(
        dimensions=128,
        memory_path=memory_path,
        auto_save=False
    )
    
    # Add test data
    memory.encode("Memory persistence test data", tags={"test", "persistence"})
    memory.encode("Another test entry", tags={"test"})
    
    # Save manually
    save_success = memory.save_memory()
    logger.info(f"Memory save {'succeeded' if save_success else 'failed'}")
    
    # Create a new memory and load from file
    logger.info("Creating new memory instance and loading from file...")
    new_memory = HolographicMemory(
        dimensions=128,  # This will be overridden by loaded value
        memory_path=memory_path,
        auto_save=False
    )
    
    load_success = new_memory.load_memory()
    logger.info(f"Memory load {'succeeded' if load_success else 'failed'}")
    
    # Verify loaded data
    logger.info(f"Loaded memory contains {len(new_memory.memory_traces)} traces")
    
    for i, trace in enumerate(new_memory.memory_traces):
        logger.info(f"  Memory {i}: '{trace.content}'")
        logger.info(f"    Tags: {trace.tags}")
    
    # Clean up
    if os.path.exists(memory_path):
        os.remove(memory_path)
        
    return new_memory

def test_memory_merging():
    """Test merging memories from different sources"""
    logger.info("\n=== Testing Memory Merging ===")
    
    # Create first memory
    logger.info("Creating first memory with geography knowledge...")
    memory1 = HolographicMemory(dimensions=128, auto_save=False)
    memory1.encode("Paris is the capital of France", 
                  tags={"geography", "europe"}, memory_type="semantic")
    memory1.encode("Berlin is the capital of Germany", 
                  tags={"geography", "europe"}, memory_type="semantic")
    
    # Create second memory
    logger.info("Creating second memory with science knowledge...")
    memory2 = HolographicMemory(dimensions=128, auto_save=False)
    memory2.encode("The speed of light is approximately 299,792,458 meters per second", 
                  tags={"physics", "science"}, memory_type="semantic")
    memory2.encode("Paris is a city in Europe", 
                  tags={"geography", "europe"}, memory_type="semantic")  # Overlapping knowledge
    
    # Show initial stats
    logger.info(f"Memory 1 has {len(memory1.memory_traces)} traces")
    logger.info(f"Memory 2 has {len(memory2.memory_traces)} traces")
    
    # Merge memories
    logger.info("Merging memory 2 into memory 1 with 'newest' conflict strategy...")
    added_count = memory1.merge_memories(memory2, conflict_strategy="newest")
    
    logger.info(f"Added {added_count} new memories")
    logger.info(f"Memory 1 now has {len(memory1.memory_traces)} traces")
    
    # Verify merged memories
    logger.info("Memories in merged memory:")
    for i, trace in enumerate(memory1.memory_traces):
        logger.info(f"  Memory {i}: '{trace.content}'")
        logger.info(f"    Tags: {trace.tags}")
    
    return memory1

def main():
    """Run all tests"""
    # Run basic memory operations
    memory = test_basic_memory_operations()
    
    # Test forgetting curve
    test_forgetting_curve(memory)
    
    # Test memory access reinforcement
    test_memory_access_reinforcement()
    
    # Test memory persistence
    test_memory_persistence()
    
    # Test memory merging
    test_memory_merging()
    
    logger.info("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()