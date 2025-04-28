#!/usr/bin/env python3
"""
Epistemic Knowledge System - Comprehensive Examples

This script demonstrates various capabilities of the epistemic knowledge system:
1. Basic knowledge storage and retrieval
2. Knowledge graphs and concept relationships
3. Reasoning with workspaces
4. Contradiction detection and resolution
5. Temporal knowledge tracking
6. Complex, multi-step reasoning
"""

import os
import time
import json
import random
from pprint import pprint

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

# Clear terminal for better readability
os.system('clear' if os.name == 'posix' else 'cls')


def separator(title):
    """Print a section separator with title"""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")


def example_1_basic_knowledge():
    """Example 1: Basic knowledge storage and retrieval"""
    separator("EXAMPLE 1: BASIC KNOWLEDGE STORAGE AND RETRIEVAL")
    
    print("Storing knowledge about different domains...\n")
    
    # Store knowledge about programming
    programming_id = store_knowledge(
        content="Python is a high-level, interpreted programming language known for its readability and versatility.",
        source="programming_book",
        confidence=0.95,
        domain="programming"
    )
    print(f"Stored programming knowledge with ID: {programming_id['id']}")
    
    # Store knowledge about physics
    physics_id = store_knowledge(
        content="Quantum mechanics is a fundamental theory in physics that describes nature at the scale of atoms and subatomic particles.",
        source="physics_textbook",
        confidence=0.9,
        domain="physics"
    )
    print(f"Stored physics knowledge with ID: {physics_id['id']}")
    
    # Store knowledge about history
    history_id = store_knowledge(
        content="The Renaissance was a period of European cultural, artistic, political, and scientific rebirth that followed the Middle Ages.",
        source="history_encyclopedia",
        confidence=0.85,
        domain="history"
    )
    print(f"Stored history knowledge with ID: {history_id['id']}")
    
    # Query knowledge
    print("\nQuerying knowledge about programming...")
    programming_query = query_knowledge("What is Python?")
    print(f"Found {len(programming_query['direct_results'])} results")
    if programming_query['direct_results']:
        print(f"Top result content: {programming_query['direct_results'][0]['content'][:100]}...")
    
    print("\nQuerying knowledge about quantum physics...")
    physics_query = query_knowledge("quantum mechanics")
    print(f"Found {len(physics_query['direct_results'])} results")
    if physics_query['direct_results']:
        print(f"Top result content: {physics_query['direct_results'][0]['content'][:100]}...")
    
    # Cross-domain query
    print("\nCross-domain query that should return no results...")
    cross_query = query_knowledge("What is the relationship between Python and the Renaissance?")
    print(f"Found {len(cross_query['direct_results'])} results")


def example_2_knowledge_graph():
    """Example 2: Knowledge graph and concept relationships"""
    separator("EXAMPLE 2: KNOWLEDGE GRAPH AND CONCEPT RELATIONSHIPS")
    
    print("Building a knowledge graph about technology concepts...\n")
    
    # Create concept nodes
    ai_id = store_knowledge(
        content="CONCEPT: Artificial Intelligence\nPROPERTIES: {'field': 'computer science', 'type': 'technology'}",
        source="tech_taxonomy",
        confidence=0.95,
        domain="knowledge_graph",
        metadata={"node_type": "concept", "concept": "Artificial Intelligence"}
    )
    print(f"Created 'Artificial Intelligence' concept node: {ai_id['id']}")
    
    ml_id = store_knowledge(
        content="CONCEPT: Machine Learning\nPROPERTIES: {'field': 'AI', 'type': 'technique'}",
        source="tech_taxonomy",
        confidence=0.9,
        domain="knowledge_graph",
        metadata={"node_type": "concept", "concept": "Machine Learning"}
    )
    print(f"Created 'Machine Learning' concept node: {ml_id['id']}")
    
    dl_id = store_knowledge(
        content="CONCEPT: Deep Learning\nPROPERTIES: {'field': 'ML', 'type': 'technique'}",
        source="tech_taxonomy",
        confidence=0.85,
        domain="knowledge_graph",
        metadata={"node_type": "concept", "concept": "Deep Learning"}
    )
    print(f"Created 'Deep Learning' concept node: {dl_id['id']}")
    
    python_id = store_knowledge(
        content="CONCEPT: Python\nPROPERTIES: {'field': 'programming', 'type': 'language'}",
        source="tech_taxonomy",
        confidence=0.95,
        domain="knowledge_graph",
        metadata={"node_type": "concept", "concept": "Python"}
    )
    print(f"Created 'Python' concept node: {python_id['id']}")
    
    # Create relationships
    print("\nCreating relationships between concepts...")
    
    create_relationship(
        ai_id['id'],
        "has_subfield",
        ml_id['id'],
        confidence=0.95
    )
    print("Added: AI has_subfield Machine Learning")
    
    create_relationship(
        ml_id['id'],
        "has_technique",
        dl_id['id'],
        confidence=0.9
    )
    print("Added: Machine Learning has_technique Deep Learning")
    
    create_relationship(
        ml_id['id'],
        "implemented_in",
        python_id['id'],
        confidence=0.8
    )
    print("Added: Machine Learning implemented_in Python")
    
    # Store some facts about the relationships
    store_knowledge(
        content="Machine Learning is a subset of Artificial Intelligence focused on algorithms that can learn from data.",
        source="ai_textbook",
        confidence=0.95,
        domain="computer_science"
    )
    
    store_knowledge(
        content="Deep Learning is a type of Machine Learning that uses neural networks with many layers.",
        source="ai_textbook",
        confidence=0.9,
        domain="computer_science"
    )
    
    store_knowledge(
        content="Python is one of the most popular programming languages for implementing Machine Learning algorithms.",
        source="programming_book",
        confidence=0.85,
        domain="computer_science"
    )
    
    # Explore a concept
    print("\nExploring 'Machine Learning' concept and its relationships...")
    ml_exploration = explore_concept("Machine Learning")
    
    print(f"Found {len(ml_exploration.get('graph', {}).get('nodes', []))} related nodes")
    print(f"Found {len(ml_exploration.get('graph', {}).get('edges', []))} relationships")
    
    # Print connections
    if 'graph' in ml_exploration and 'edges' in ml_exploration['graph']:
        print("\nRelationships around Machine Learning:")
        for edge in ml_exploration['graph']['edges']:
            print(f"  {edge.get('source', '?')} --[{edge.get('type', '?')}]--> {edge.get('target', '?')}")


def example_3_contradictions():
    """Example 3: Contradiction detection and resolution"""
    separator("EXAMPLE 3: CONTRADICTION DETECTION AND RESOLUTION")
    
    print("Storing contradictory information about climate change...\n")
    
    # Store two contradictory facts
    fact1_id = store_knowledge(
        content="The Earth's climate is warming due to human activities.",
        source="climate_science_journal",
        confidence=0.95,
        domain="climate_science"
    )
    print(f"Stored fact 1 with ID: {fact1_id['id']}")
    
    # This should be flagged as contradictory
    fact2_id = store_knowledge(
        content="The Earth's climate is not warming due to human activities.",
        source="climate_skeptic_blog",
        confidence=0.4,
        domain="climate_science"
    )
    print(f"Stored contradictory fact 2 with ID: {fact2_id['id']}")
    
    # Check if contradiction was detected
    print(f"\nContradictions detected in fact 2: {len(fact2_id.get('contradictions', []))}")
    
    # Add more nuanced information
    fact3_id = store_knowledge(
        content="While natural factors affect climate, the primary driver of current warming is human greenhouse gas emissions.",
        source="ipcc_report",
        confidence=0.9,
        domain="climate_science"
    )
    print(f"Stored nuanced fact 3 with ID: {fact3_id['id']}")
    
    # Query about climate change and see what gets returned (should favor higher confidence)
    print("\nQuerying about climate change...")
    climate_query = query_knowledge("Is climate change caused by humans?", min_confidence=0.5)
    print(f"Found {len(climate_query['direct_results'])} results with confidence >= 0.5")
    
    if climate_query['direct_results']:
        print("\nTop results:")
        for i, result in enumerate(climate_query['direct_results']):
            print(f"Result {i+1}: (Confidence: {result['confidence']})")
            print(f"  {result['content'][:100]}...")


def example_4_temporal_knowledge():
    """Example 4: Temporal knowledge tracking"""
    separator("EXAMPLE 4: TEMPORAL KNOWLEDGE TRACKING")
    
    print("Creating knowledge snapshots and tracking changes...\n")
    
    # Create a snapshot of current state
    snapshot1 = create_temporal_snapshot("before_covid_knowledge")
    print(f"Created knowledge snapshot: {snapshot1['snapshot_id']}")
    
    # Add new knowledge about a recent event
    print("\nAdding knowledge about a recent event (COVID-19)...")
    
    covid1_id = store_knowledge(
        content="COVID-19 is a respiratory disease caused by the SARS-CoV-2 virus.",
        source="medical_journal",
        confidence=0.95,
        domain="medicine"
    )
    print(f"Stored COVID fact 1 with ID: {covid1_id['id']}")
    
    covid2_id = store_knowledge(
        content="Vaccines have been developed to help prevent severe COVID-19 infections.",
        source="health_organization",
        confidence=0.9,
        domain="medicine"
    )
    print(f"Stored COVID fact 2 with ID: {covid2_id['id']}")
    
    covid3_id = store_knowledge(
        content="COVID-19 was first identified in Wuhan, China in December 2019.",
        source="epidemiology_report",
        confidence=0.95,
        domain="medicine"
    )
    print(f"Stored COVID fact 3 with ID: {covid3_id['id']}")
    
    # Create another snapshot
    snapshot2 = create_temporal_snapshot("after_covid_knowledge")
    print(f"\nCreated second knowledge snapshot: {snapshot2['snapshot_id']}")
    
    # Compare snapshots
    print("\nComparing knowledge snapshots...")
    diff = compare_snapshots(snapshot1['snapshot_id'], snapshot2['snapshot_id'])
    
    if diff['status'] == 'success':
        print(f"Time between snapshots: {diff['diff']['time_delta']:.2f} seconds")
        print(f"Knowledge units added: {diff['diff']['added_count']}")
        print(f"Knowledge units removed: {diff['diff']['removed_count']}")
        
        print("\nNewly added knowledge:")
        for i, unit in enumerate(diff['diff']['added_units'][:3]):  # Show up to 3
            print(f"  {i+1}. {unit.get('content', '')[:100]}...")


def example_5_reasoning_workspace():
    """Example 5: Complex reasoning with workspace"""
    separator("EXAMPLE 5: COMPLEX REASONING WITH WORKSPACE")
    
    print("Solving a problem through multi-step reasoning...\n")
    
    # Define the problem
    problem = "What are the ethical implications of AI in healthcare?"
    print(f"Problem: {problem}\n")
    
    # Create reasoning workspace
    workspace = create_reasoning_workspace(f"Solving: {problem}")
    workspace_id = workspace['workspace_id']
    print(f"Created reasoning workspace with ID: {workspace_id}")
    
    # Step 1: Gather relevant knowledge
    print("\nStep 1: Gathering relevant knowledge...")
    
    # Query for relevant knowledge
    health_ai_query = query_knowledge("AI in healthcare ethics")
    result_count = len(health_ai_query.get('direct_results', []))
    
    # Record this step
    step1 = workspace_add_step(
        workspace_id,
        "information_gathering",
        f"Gathered {result_count} relevant knowledge units about AI in healthcare ethics."
    )
    print(f"Added reasoning step: {step1['step_type']}")
    
    # Add some facts to support reasoning
    store_knowledge(
        content="AI in healthcare can help diagnose diseases more accurately than human doctors in some cases.",
        source="medical_ai_journal",
        confidence=0.9,
        domain="healthcare_ai"
    )
    
    store_knowledge(
        content="There are privacy concerns about AI systems accessing and processing sensitive patient data.",
        source="ethics_paper",
        confidence=0.85,
        domain="healthcare_ai"
    )
    
    store_knowledge(
        content="AI systems may perpetuate biases present in their training data, leading to healthcare disparities.",
        source="ai_ethics_conference",
        confidence=0.8,
        domain="healthcare_ai"
    )
    
    # Step 2: Identify key ethical dimensions
    print("\nStep 2: Identifying key ethical dimensions...")
    step2 = workspace_add_step(
        workspace_id,
        "analysis",
        """Based on available knowledge, the key ethical dimensions of AI in healthcare are:
1. Privacy and data security
2. Algorithmic bias and fairness
3. Accountability and transparency
4. Impact on doctor-patient relationships
5. Accessibility and equity concerns"""
    )
    print(f"Added reasoning step: {step2['step_type']}")
    
    # Step 3: Analyze privacy concerns
    print("\nStep 3: Analyzing privacy concerns...")
    step3 = workspace_add_step(
        workspace_id,
        "deep_analysis",
        """Privacy concerns analysis:
- Healthcare data is among the most sensitive personal information
- AI systems require large amounts of data to be effective
- Challenges in obtaining proper informed consent for AI data usage
- Risks of data breaches and unauthorized access
- Tension between data sharing for improved outcomes and privacy protection"""
    )
    print(f"Added reasoning step: {step3['step_type']}")
    
    # Step 4: Analyze bias concerns
    print("\nStep 4: Analyzing bias concerns...")
    step4 = workspace_add_step(
        workspace_id,
        "deep_analysis",
        """Bias concerns analysis:
- Historic healthcare disparities may be encoded in training data
- Underrepresentation of minority populations in medical research
- Risk of reinforcing existing inequities in healthcare access
- Need for diverse and representative training datasets
- Importance of ongoing monitoring for algorithmic bias"""
    )
    print(f"Added reasoning step: {step4['step_type']}")
    
    # Step 5: Derive balanced conclusion
    print("\nStep 5: Deriving balanced conclusion...")
    conclusion = """The ethical implications of AI in healthcare are multifaceted, requiring careful consideration of:

1. Privacy and data protection: Healthcare organizations must implement robust safeguards for patient data used in AI systems, with clear consent mechanisms and transparency about data usage.

2. Addressing bias and ensuring fairness: AI systems must be trained on diverse, representative datasets and continuously monitored to prevent perpetuating or amplifying existing healthcare disparities.

3. Maintaining human oversight: While AI can enhance healthcare delivery, maintaining appropriate human clinical judgment and preserving the doctor-patient relationship remains essential.

4. Ensuring equitable access: Benefits of healthcare AI should be accessible across different populations and socioeconomic groups to avoid creating new healthcare divides.

5. Establishing clear accountability frameworks: Clear guidelines for responsibility when AI systems are involved in healthcare decisions are necessary.

The ethical implementation of AI in healthcare requires balancing innovation with careful consideration of these ethical dimensions."""

    derived = workspace_derive_knowledge(
        workspace_id,
        conclusion,
        0.85
    )
    print(f"Derived conclusion with confidence: {derived['confidence']}")
    
    # Commit knowledge
    print("\nCommitting derived knowledge to the main knowledge store...")
    commit = workspace_commit_knowledge(workspace_id)
    print(f"Committed {commit['committed_count']} knowledge units")
    
    # Get the reasoning chain
    print("\nRetrieving full reasoning chain...")
    chain = workspace_get_chain(workspace_id)
    print(f"Reasoning chain has {len(chain['chain']['steps'])} steps")
    
    # Query the committed knowledge
    print("\nQuerying for the committed knowledge...")
    results = query_knowledge("ethical implications of AI in healthcare")
    
    if results['direct_results']:
        print(f"\nFound our derived knowledge among {len(results['direct_results'])} results")
        print(f"Result confidence: {results['direct_results'][0]['confidence']}")
        # Just print the first 200 characters
        print(f"\nResult content: {results['direct_results'][0]['content'][:200]}...")


def run_all_examples():
    """Run all examples in sequence"""
    # Initialize the knowledge system
    db_path = "./knowledge/comprehensive_examples.db"
    
    # Remove old DB if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    print(f"Initializing knowledge system at: {db_path}")
    initialize_knowledge_system(db_path)
    
    try:
        # Run all examples
        example_1_basic_knowledge()
        example_2_knowledge_graph()
        example_3_contradictions()
        example_4_temporal_knowledge()
        example_5_reasoning_workspace()
        
    finally:
        # Always shut down the knowledge system
        shutdown_knowledge_system()
        print("\nKnowledge system shut down successfully.")


if __name__ == "__main__":
    run_all_examples()