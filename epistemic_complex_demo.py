#!/usr/bin/env python3
"""
Epistemic Complex Demo - Comprehensive demonstration of the epistemic knowledge system
handling a complex, multi-faceted problem with long-context reasoning.

This script demonstrates:
1. Advanced knowledge management with epistemological foundations
2. Incremental and recursive reasoning on complex problems
3. Knowledge synthesis across multiple domains
4. Temporal tracking of knowledge development
5. Handling of contradictions and uncertainties
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from epistemic_core import (
    EpistemicUnit,
    EpistemicStore,
    KnowledgeGraph,
    TemporalKnowledgeState,
    KnowledgeAPI,
    ReasoningWorkspace
)

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

from epistemic_long_context import (
    IncrementalReasoner,
    RecursiveDecomposer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("epistemic_complex_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("complex-demo")


class ComplexProblemSolver:
    """
    Demonstrates solving a complex, multi-faceted problem using the epistemic
    knowledge management system with advanced reasoning techniques.
    """
    
    def __init__(self, knowledge_path: str = "./knowledge/complex_demo"):
        """Initialize the complex problem solver"""
        self.knowledge_path = knowledge_path
        Path(knowledge_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize knowledge base
        self.db_path = f"{knowledge_path}/epistemic.db"
        initialize_knowledge_system(self.db_path)
        
        # Knowledge API for direct access
        self.knowledge_api = KnowledgeAPI(self.db_path)
        
        # For temporal tracking
        self.temporal_state = TemporalKnowledgeState(self.db_path)
        
        # For knowledge graph relationships
        self.knowledge_graph = KnowledgeGraph(self.db_path)
        
        # Track our reasoning components
        self.incrementals = {}
        self.decomposers = {}
        self.workspaces = {}
        
        logger.info(f"Initialized ComplexProblemSolver with knowledge path: {knowledge_path}")
    
    def seed_knowledge_base(self) -> Dict[str, Any]:
        """Seed the knowledge base with initial domain knowledge"""
        logger.info("Seeding knowledge base with domain expertise")
        
        # Seed with domain knowledge across multiple areas
        domains = {
            "ai_ethics": [
                {
                    "content": "AI systems should be designed to align with human values and respect human autonomy.",
                    "confidence": 0.92,
                    "source": "AI Ethics Framework, 2023",
                    "evidence": "Based on consensus across multiple AI ethics guidelines."
                },
                {
                    "content": "Transparency in AI decision-making is essential for accountability and trust.",
                    "confidence": 0.89,
                    "source": "IEEE Ethical AI Guidelines",
                    "evidence": "Multiple studies show transparent AI systems receive higher user trust."
                },
                {
                    "content": "AI systems should be fair and avoid discriminatory outcomes across different demographic groups.",
                    "confidence": 0.94,
                    "source": "ACM Code of Ethics",
                    "evidence": "Research shows biased AI can amplify existing societal inequalities."
                }
            ],
            "cognitive_science": [
                {
                    "content": "Working memory has limited capacity, typically 4-7 items at once.",
                    "confidence": 0.91,
                    "source": "Cognitive Psychology Journal, 2020",
                    "evidence": "Multiple experimental studies measuring working memory capacity."
                },
                {
                    "content": "System 1 thinking is fast, intuitive and emotional; System 2 is slower, deliberative and logical.",
                    "confidence": 0.88,
                    "source": "Kahneman, D. (2011). Thinking, Fast and Slow.",
                    "evidence": "Based on decades of cognitive bias research and experimental psychology."
                }
            ],
            "epistemology": [
                {
                    "content": "Knowledge can be defined as justified true belief with proper causal connections.",
                    "confidence": 0.82,
                    "source": "Contemporary Epistemology, 2018",
                    "evidence": "Extension of Gettier problem solutions in modern epistemology."
                },
                {
                    "content": "Epistemic uncertainty can be divided into aleatory (inherent randomness) and epistemic (lack of knowledge).",
                    "confidence": 0.85,
                    "source": "Philosophy of Science Journal, 2019",
                    "evidence": "Demonstrated applications in risk modeling and decision theory."
                },
                {
                    "content": "The Bayesian framework provides a mathematical basis for updating beliefs based on evidence.",
                    "confidence": 0.93,
                    "source": "Bayesian Epistemology, 2021",
                    "evidence": "Success of Bayesian methods in machine learning and cognitive science."
                }
            ],
            "data_science": [
                {
                    "content": "Feature engineering often has more impact on model performance than algorithm selection.",
                    "confidence": 0.87,
                    "source": "Applied Data Science Conference, 2022",
                    "evidence": "Comparative studies across multiple datasets and modeling tasks."
                },
                {
                    "content": "Cross-validation helps prevent overfitting by evaluating models on independent data splits.",
                    "confidence": 0.95,
                    "source": "Machine Learning Handbook, 2023",
                    "evidence": "Standard practice validated across thousands of ML applications."
                }
            ],
            "contradictory_evidence": [
                {
                    "content": "Larger language models always produce more accurate outputs.",
                    "confidence": 0.71,
                    "source": "Early LLM Research, 2020",
                    "evidence": "Based on limited benchmarks with early models."
                },
                {
                    "content": "Model size does not always correlate with accuracy; smaller fine-tuned models can outperform larger models.",
                    "confidence": 0.83, 
                    "source": "Advanced LLM Evaluation Study, 2023",
                    "evidence": "Recent studies comparing domain-specialized smaller models against general larger models."
                }
            ]
        }
        
        # Add all knowledge units to the store
        units_added = 0
        for domain, facts in domains.items():
            for fact in facts:
                unit = EpistemicUnit(
                    content=fact["content"],
                    confidence=fact["confidence"],
                    source=fact["source"],
                    evidence=fact["evidence"],
                    domain=domain
                )
                
                # Store the knowledge
                result = store_knowledge(unit)
                units_added += 1
                
                # Add domain categorization relationship
                create_relationship(
                    source_id=result["unit_id"],
                    relation_type="categorized_as",
                    target="domain:" + domain,
                    confidence=0.95
                )
        
        # Create concept relationships in the knowledge graph
        concept_relationships = [
            ("ai_ethics", "relates_to", "epistemology", 0.78),
            ("cognitive_science", "informs", "ai_ethics", 0.82),
            ("data_science", "applies", "epistemology", 0.85),
            ("cognitive_science", "overlaps_with", "epistemology", 0.79)
        ]
        
        for source, relation, target, confidence in concept_relationships:
            create_relationship(
                source_id="domain:" + source,
                relation_type=relation,
                target="domain:" + target,
                confidence=confidence
            )
        
        # Create a temporal snapshot
        snapshot_id = create_temporal_snapshot("Initial knowledge seed")
        
        return {
            "units_added": units_added,
            "domains": list(domains.keys()),
            "snapshot_id": snapshot_id
        }
    
    def solve_complex_problem(self, problem_statement: str) -> Dict[str, Any]:
        """
        Solve a complex problem using incremental reasoning and recursive decomposition
        """
        logger.info(f"Starting complex problem solution: {problem_statement[:50]}...")
        
        # Create a decomposer for this problem
        decomposer_id = f"decomposer_{int(time.time())}"
        decomposer = RecursiveDecomposer(f"{self.knowledge_path}/{decomposer_id}.db")
        self.decomposers[decomposer_id] = decomposer
        
        # Decompose the problem
        decomposition = decomposer.decompose_problem(problem_statement)
        logger.info(f"Problem decomposed into {decomposition['subproblem_count']} subproblems")
        
        # Process a significant number of increments to make progress
        increments_processed = 0
        max_increments = 30  # Limit for demo purposes
        
        results = []
        for i in range(max_increments):
            result = decomposer.process_next_increment()
            increments_processed += 1
            results.append(result)
            
            # Track progress
            logger.info(f"Increment {i+1}: Progress {result.get('tree_progress', 0)*100:.1f}%")
            
            # If complete, we're done
            if result.get("status") == "complete":
                logger.info("Problem solving complete!")
                break
        
        # Finalize and get the problem tree
        if increments_processed >= max_increments:
            logger.warning(f"Reached maximum increments ({max_increments}). Solution may be incomplete.")
        
        tree = decomposer.get_problem_tree()
        final_result = decomposer.close()
        
        # Add the entire reasoning process as structured knowledge
        self._store_reasoning_process(problem_statement, tree, results, final_result)
        
        return {
            "problem": problem_statement,
            "increments_processed": increments_processed,
            "tree": tree,
            "final_result": final_result,
            "decomposer_id": decomposer_id
        }
    
    def explore_complex_concept(self, concept: str, depth: int = 2) -> Dict[str, Any]:
        """
        Explore a complex concept using the knowledge graph and temporal dimensions
        """
        logger.info(f"Exploring complex concept: {concept}")
        
        # Direct exploration
        exploration = explore_concept(concept, depth=depth)
        
        # Get temporal evolution of this concept (if it exists)
        temporal_snapshots = self.temporal_state.get_concept_evolution(concept)
        
        # Track this exploration in a reasoning workspace
        workspace_id = f"concept_exploration_{int(time.time())}"
        workspace = create_reasoning_workspace(f"Exploration of concept: {concept}")
        self.workspaces[workspace_id] = workspace["workspace_id"]
        
        # Record exploration steps
        workspace_add_step(
            workspace["workspace_id"],
            "concept_query",
            f"EXPLORING CONCEPT: {concept}"
        )
        
        # Record findings
        direct_results = exploration.get("direct_results", [])
        if direct_results:
            result_summary = "\n".join([
                f"- {r.get('content', 'No content')} (confidence: {r.get('confidence', 0):.2f})"
                for r in direct_results[:5]
            ])
            
            workspace_add_step(
                workspace["workspace_id"],
                "concept_findings",
                f"DIRECT FINDINGS:\n{result_summary}"
            )
        
        # Record relationships
        relationships = exploration.get("relationships", [])
        if relationships:
            relationship_summary = "\n".join([
                f"- {r.get('source', '')} {r.get('relation_type', '')} {r.get('target', '')}"
                for r in relationships[:5]
            ])
            
            workspace_add_step(
                workspace["workspace_id"],
                "concept_relationships",
                f"CONCEPT RELATIONSHIPS:\n{relationship_summary}"
            )
        
        # Derive new knowledge from this exploration
        derived = self._derive_knowledge_from_exploration(concept, exploration)
        
        if derived:
            # Record derivation
            workspace_add_step(
                workspace["workspace_id"],
                "knowledge_derivation",
                f"DERIVED KNOWLEDGE:\n{derived}"
            )
            
            # Store the derived knowledge in the workspace
            workspace_derive_knowledge(
                workspace["workspace_id"],
                derived,
                confidence=0.75,
                evidence=f"Derived from exploration of concept: {concept}"
            )
        
        return {
            "concept": concept,
            "direct_results_count": len(direct_results),
            "relationships_count": len(relationships),
            "temporal_snapshots": len(temporal_snapshots),
            "workspace_id": workspace["workspace_id"],
            "derived_knowledge": derived is not None
        }
    
    def handle_contradictions(self) -> Dict[str, Any]:
        """
        Demonstrate handling of contradictory knowledge in the system
        """
        logger.info("Analyzing and resolving contradictions")
        
        # Create a workspace for contradiction analysis
        workspace = create_reasoning_workspace("Contradiction Resolution")
        
        # Find potentially contradictory knowledge
        query_result = query_knowledge("larger language models accuracy", reasoning_depth=2)
        
        # Record the potentially contradictory statements
        workspace_add_step(
            workspace["workspace_id"],
            "contradiction_identification",
            "ANALYZING POTENTIAL CONTRADICTIONS:\n" +
            "Searching for contradictory statements about language model size and accuracy"
        )
        
        # Find the contradictory units
        contradictory_units = []
        direct_results = query_result.get("direct_results", [])
        
        for unit in direct_results:
            content = unit.get("content", "").lower()
            if "larger" in content and "model" in content:
                if "always" in content and "more accurate" in content:
                    contradictory_units.append(unit)
                elif "not always" in content and "accuracy" in content:
                    contradictory_units.append(unit)
        
        if len(contradictory_units) >= 2:
            # Record the contradiction
            contradiction_text = "\n".join([
                f"CONTRADICTORY STATEMENT {i+1}:\n" +
                f"Content: {unit.get('content', 'No content')}\n" +
                f"Confidence: {unit.get('confidence', 0):.2f}\n" +
                f"Source: {unit.get('source', 'Unknown source')}\n" +
                f"Evidence: {unit.get('evidence', 'No evidence')}"
                for i, unit in enumerate(contradictory_units[:2])
            ])
            
            workspace_add_step(
                workspace["workspace_id"],
                "contradiction_detail",
                f"CONTRADICTORY STATEMENTS FOUND:\n{contradiction_text}"
            )
            
            # Analyze and resolve the contradiction
            resolution = self._resolve_contradiction(contradictory_units)
            
            workspace_add_step(
                workspace["workspace_id"],
                "contradiction_resolution",
                f"CONTRADICTION RESOLUTION:\n{resolution['explanation']}"
            )
            
            # Store the resolved knowledge
            resolved_unit = workspace_derive_knowledge(
                workspace["workspace_id"],
                resolution["resolved_statement"],
                confidence=resolution["confidence"],
                evidence=resolution["evidence"]
            )
            
            # Commit the resolved knowledge
            workspace_commit_knowledge(workspace["workspace_id"])
            
            return {
                "contradictions_found": True,
                "units_analyzed": len(contradictory_units),
                "resolution": resolution,
                "workspace_id": workspace["workspace_id"]
            }
        else:
            workspace_add_step(
                workspace["workspace_id"],
                "contradiction_result",
                "No clear contradictions found in the knowledge base."
            )
            
            return {
                "contradictions_found": False,
                "units_analyzed": len(direct_results),
                "workspace_id": workspace["workspace_id"]
            }
    
    def incremental_long_problem(self, problem_statement: str) -> Dict[str, Any]:
        """
        Solve a problem incrementally using the IncrementalReasoner
        """
        logger.info(f"Starting incremental reasoning: {problem_statement[:50]}...")
        
        # Create an incremental reasoner
        reasoner_id = f"incremental_{int(time.time())}"
        reasoner = IncrementalReasoner(f"{self.knowledge_path}/{reasoner_id}.db")
        self.incrementals[reasoner_id] = reasoner
        
        # Set the problem
        reasoner.set_problem(problem_statement)
        
        # Process increments
        max_increments = 20  # Limit for demo purposes
        results = []
        
        for i in range(max_increments):
            result = reasoner.process_next_increment()
            results.append(result)
            
            logger.info(f"Increment {i+1}: Progress {result.get('progress', 0)*100:.1f}%")
            
            if result.get("status") == "complete":
                logger.info("Incremental reasoning complete!")
                break
        
        # Get the reasoning trace
        trace = reasoner.get_reasoning_trace()
        
        # Commit derived knowledge
        commit_result = reasoner.commit_derived_knowledge()
        
        # Close the session
        close_result = reasoner.close_session()
        
        return {
            "problem": problem_statement,
            "reasoner_id": reasoner_id,
            "increments_processed": len(results),
            "trace_steps": len(trace.get("steps", [])),
            "insights": len(trace.get("insights", [])),
            "committed_units": commit_result.get("units_committed", 0),
            "status": close_result.get("status", "unknown")
        }
    
    def _store_reasoning_process(self, problem: str, tree: Dict[str, Any], 
                                results: List[Dict[str, Any]], final_result: Dict[str, Any]) -> None:
        """Store the entire reasoning process as structured knowledge"""
        # Create a unit for the overall problem
        problem_unit = EpistemicUnit(
            content=f"Problem: {problem}",
            confidence=0.99,
            source="Complex Problem Solver",
            evidence="Original problem statement"
        )
        problem_result = store_knowledge(problem_unit)
        
        # Store the final conclusion if available
        if "conclusion" in final_result:
            conclusion_unit = EpistemicUnit(
                content=final_result["conclusion"],
                confidence=0.85,
                source="Recursive Decomposer",
                evidence=f"Conclusion from solving: {problem}"
            )
            conclusion_result = store_knowledge(conclusion_unit)
            
            # Link problem to conclusion
            create_relationship(
                source_id=problem_result["unit_id"],
                relation_type="has_conclusion",
                target=conclusion_result["unit_id"],
                confidence=0.95
            )
        
        # Store key insights from subproblems
        for result in results:
            if "increment_result" in result and "result" in result["increment_result"]:
                if "insight" in result["increment_result"]["result"] and result["increment_result"]["result"]["insight"]:
                    insight = result["increment_result"]["result"]["insight"]
                    
                    insight_unit = EpistemicUnit(
                        content=insight,
                        confidence=0.8,
                        source=f"Subproblem Analysis: {result.get('statement', 'Unknown')}",
                        evidence="Derived during recursive problem solving"
                    )
                    insight_result = store_knowledge(insight_unit)
                    
                    # Link problem to insight
                    create_relationship(
                        source_id=problem_result["unit_id"],
                        relation_type="has_insight",
                        target=insight_result["unit_id"],
                        confidence=0.9
                    )
    
    def _derive_knowledge_from_exploration(self, concept: str, exploration: Dict[str, Any]) -> Optional[str]:
        """Derive new knowledge from concept exploration"""
        direct_results = exploration.get("direct_results", [])
        relationships = exploration.get("relationships", [])
        
        if not direct_results and not relationships:
            return None
        
        # Simple implementation - in a real system, this would use more sophisticated methods
        if len(direct_results) >= 2:
            # Find the two most confident results
            sorted_results = sorted(direct_results, key=lambda x: x.get("confidence", 0), reverse=True)
            top_results = sorted_results[:2]
            
            # Generate a synthesis of the top results
            synthesis = f"Synthesis regarding {concept}: "
            synthesis += f"According to {top_results[0].get('source', 'unknown')}, "
            synthesis += f"{top_results[0].get('content', '')}. "
            
            if len(top_results) > 1:
                synthesis += f"Additionally, {top_results[1].get('source', 'unknown')} indicates that "
                synthesis += f"{top_results[1].get('content', '')}."
            
            return synthesis
        
        return None
    
    def _resolve_contradiction(self, contradictory_units: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve a contradiction between knowledge units"""
        if len(contradictory_units) < 2:
            return {"error": "Need at least 2 units to resolve a contradiction"}
        
        # Sort by confidence and recency (assuming more recent sources are better)
        sorted_units = sorted(
            contradictory_units, 
            key=lambda x: (x.get("confidence", 0), "2023" in x.get("source", "")),
            reverse=True
        )
        
        highest_conf_unit = sorted_units[0]
        second_unit = sorted_units[1]
        
        # Check if the highest confidence unit is much more confident
        if highest_conf_unit.get("confidence", 0) > second_unit.get("confidence", 0) + 0.1:
            # Trust the higher confidence unit
            resolution = {
                "resolved_statement": highest_conf_unit.get("content", ""),
                "confidence": highest_conf_unit.get("confidence", 0) * 0.9,  # Slight reduction due to contradiction
                "evidence": f"Resolved contradiction favoring higher confidence source: {highest_conf_unit.get('source', 'unknown')}",
                "explanation": f"The contradiction was resolved by favoring the statement with higher confidence. " +
                               f"The statement from {highest_conf_unit.get('source', 'unknown')} " +
                               f"(confidence: {highest_conf_unit.get('confidence', 0):.2f}) " +
                               f"was preferred over {second_unit.get('source', 'unknown')} " +
                               f"(confidence: {second_unit.get('confidence', 0):.2f})."
            }
        else:
            # More nuanced reconciliation for similar confidence levels
            # For this demo, create a synthesis that acknowledges both perspectives
            
            # Special case for the LLM size contradiction
            if "larger language models" in highest_conf_unit.get("content", "").lower():
                resolution = {
                    "resolved_statement": "Model size correlation with accuracy depends on context; " +
                                         "smaller fine-tuned models can sometimes outperform larger general models, " +
                                         "especially in domain-specific tasks.",
                    "confidence": 0.85,
                    "evidence": f"Synthesized from contradictory sources: {highest_conf_unit.get('source', 'unknown')} " +
                               f"and {second_unit.get('source', 'unknown')}",
                    "explanation": "The contradiction about language model size and accuracy was resolved through nuanced synthesis. " +
                                  "While larger models often have more capability, specialized smaller models can be more accurate " +
                                  "in specific domains. Recent research (2023) shows the importance of fine-tuning and domain specialization " +
                                  "over raw model size."
                }
            else:
                # Generic synthesis for other contradictions
                resolution = {
                    "resolved_statement": f"There are conflicting views on this topic: " +
                                         f"{highest_conf_unit.get('content', '')} However, some sources suggest that " +
                                         f"{second_unit.get('content', '')}",
                    "confidence": max(highest_conf_unit.get("confidence", 0), second_unit.get("confidence", 0)) * 0.7,
                    "evidence": f"Combined from contradictory sources with similar confidence levels",
                    "explanation": "This contradiction could not be definitively resolved due to similar confidence levels. " +
                                  "Both perspectives have been preserved with appropriate caveats."
                }
        
        return resolution
    
    def close(self) -> None:
        """Close all components and shut down the knowledge system"""
        # Close all reasoners
        for reasoner_id, reasoner in self.incrementals.items():
            reasoner.close_session()
        
        # Close all decomposers
        for decomposer_id, decomposer in self.decomposers.items():
            decomposer.close()
        
        # Shut down the knowledge system
        shutdown_knowledge_system()
        
        logger.info("Complex problem solver closed")


def demonstrate_complex_problem_solving():
    """Run a comprehensive demonstration of the epistemic knowledge system"""
    print("\n========== EPISTEMIC KNOWLEDGE SYSTEM - COMPLEX PROBLEM DEMONSTRATION ==========\n")
    
    # Create the solver
    solver = ComplexProblemSolver()
    
    try:
        # Step 1: Seed the knowledge base
        print("\n===== STEP 1: SEEDING KNOWLEDGE BASE =====")
        seed_result = solver.seed_knowledge_base()
        print(f"Added {seed_result['units_added']} knowledge units across {len(seed_result['domains'])} domains")
        print(f"Created initial knowledge snapshot: {seed_result['snapshot_id']}")
        
        # Step 2: Explore a complex concept
        print("\n===== STEP 2: EXPLORING A COMPLEX CONCEPT =====")
        concept = "epistemic uncertainty in AI systems"
        concept_result = solver.explore_complex_concept(concept, depth=2)
        print(f"Explored concept: {concept}")
        print(f"Found {concept_result['direct_results_count']} direct results")
        print(f"Found {concept_result['relationships_count']} relationships")
        print(f"Workspace created: {concept_result['workspace_id']}")
        if concept_result['derived_knowledge']:
            print("Successfully derived new knowledge from exploration")
        
        # Step 3: Handle contradictions
        print("\n===== STEP 3: HANDLING CONTRADICTIONS =====")
        contradiction_result = solver.handle_contradictions()
        if contradiction_result['contradictions_found']:
            print("Found and resolved contradictions in the knowledge base")
            print(f"Resolution confidence: {contradiction_result['resolution']['confidence']:.2f}")
            print(f"Resolution explanation: {contradiction_result['resolution']['explanation'][:150]}...")
        else:
            print("No contradictions found in the current knowledge base")
        
        # Step 4: Solve an incremental long-context problem
        print("\n===== STEP 4: INCREMENTAL REASONING ON COMPLEX PROBLEM =====")
        incremental_problem = (
            "How can we create AI systems that properly manage epistemic uncertainty while balancing "
            "the need for decisive action in critical domains like healthcare, autonomous vehicles, "
            "and financial systems? Consider both technical and ethical dimensions."
        )
        incremental_result = solver.incremental_long_problem(incremental_problem)
        print(f"Processed {incremental_result['increments_processed']} increments")
        print(f"Generated {incremental_result['trace_steps']} reasoning steps")
        print(f"Produced {incremental_result['insights']} insights")
        print(f"Committed {incremental_result['committed_units']} knowledge units")
        
        # Step 5: Solve a complex multi-domain problem
        print("\n===== STEP 5: RECURSIVE DECOMPOSITION OF COMPLEX MULTI-DOMAIN PROBLEM =====")
        complex_problem = (
            "How can we design an AI-powered knowledge management system for scientific research "
            "that properly handles the epistemological dimensions of evidence, belief, and certainty "
            "across multiple scientific disciplines with varying standards of proof? The system should "
            "account for temporal evolution of knowledge, contradictory findings, and interdisciplinary "
            "synthesis while maintaining appropriate uncertainty quantification."
        )
        complex_result = solver.solve_complex_problem(complex_problem)
        print(f"Decomposed into {len(complex_result['tree']['subproblems'])} subproblems")
        print(f"Processed {complex_result['increments_processed']} increments")
        if "conclusion" in complex_result['final_result']:
            print("\nFinal conclusion:")
            conclusion = complex_result['final_result']['conclusion']
            print(f"{conclusion[:300]}...")
        
        print("\n===== DEMONSTRATION COMPLETE =====")
        print("The epistemic knowledge system has demonstrated:")
        print("1. Storage and retrieval of knowledge with epistemological properties")
        print("2. Exploration of concepts through knowledge graphs")
        print("3. Handling of contradictions and uncertainty")
        print("4. Incremental reasoning on long-context problems")
        print("5. Recursive decomposition of complex multi-domain problems")
        print("6. Integration of all these capabilities in a unified system")
    
    finally:
        # Close everything properly
        solver.close()


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("./knowledge", exist_ok=True)
    
    # Run the demonstration
    demonstrate_complex_problem_solving()