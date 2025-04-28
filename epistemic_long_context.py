#!/usr/bin/env python3
"""
Epistemic Long Context - Support for arbitrarily long reasoning problems

This module extends the epistemic knowledge system to handle arbitrarily long reasoning
processes through:
1. Incremental reasoning with state persistence
2. Evidence chunking and progressive summarization
3. Long-context session management
4. Recursive problem decomposition
"""

import os
import json
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("long-context")


class IncrementalReasoner:
    """
    Handles arbitrarily long reasoning processes by breaking them into incremental steps
    with state persistence and evidence chunking.
    """
    
    def __init__(self, db_path: str = "./knowledge/long_context.db", session_id: str = None):
        """Initialize the incremental reasoner with a knowledge database"""
        # Initialize the underlying knowledge system
        initialize_knowledge_system(db_path)
        
        # Create or use a session ID
        self.session_id = session_id or f"session_{str(uuid.uuid4())}"
        
        # Create a workspace for this session
        workspace = create_reasoning_workspace(f"Long context session: {self.session_id}")
        self.workspace_id = workspace["workspace_id"]
        
        # Initialize the context that carries through the session
        self.context = {
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "problem_statement": None,
            "subproblems": [],
            "current_focus": None,
            "open_questions": [],
            "insights": [],
            "evidence_chunks": [],
            "current_chunk_id": 0,
            "progress": 0.0  # 0.0 to 1.0
        }
        
        # Save initial state
        self._save_state()
        
        logger.info(f"Initialized incremental reasoner with session ID: {self.session_id}")
    
    def set_problem(self, problem_statement: str) -> Dict[str, Any]:
        """Set the main problem to be solved over multiple increments"""
        # Record the problem statement
        self.context["problem_statement"] = problem_statement
        self.context["open_questions"].append({
            "id": "main_problem",
            "question": problem_statement,
            "status": "open",
            "created_at": time.time()
        })
        
        # Add the problem statement as the first reasoning step
        workspace_add_step(
            self.workspace_id,
            "problem_definition",
            f"MAIN PROBLEM: {problem_statement}"
        )
        
        # Analyze the problem to break it into subproblems
        subproblems = self._decompose_problem(problem_statement)
        self.context["subproblems"] = subproblems
        
        # Add all subproblems to open questions
        for i, subproblem in enumerate(subproblems):
            self.context["open_questions"].append({
                "id": f"subproblem_{i+1}",
                "question": subproblem,
                "status": "open",
                "created_at": time.time()
            })
        
        # Set current focus to the first subproblem
        if subproblems:
            self.context["current_focus"] = {
                "type": "subproblem",
                "id": "subproblem_1",
                "content": subproblems[0]
            }
        
        # Update state
        self._save_state()
        
        return {
            "status": "problem_set",
            "problem": problem_statement,
            "subproblems": subproblems,
            "session_id": self.session_id
        }
    
    def process_next_increment(self) -> Dict[str, Any]:
        """Process the next increment of reasoning"""
        # Check if we have a problem set
        if not self.context["problem_statement"]:
            return {
                "status": "error",
                "message": "No problem statement set. Call set_problem() first."
            }
        
        # Check if we're done
        if not self._has_open_questions():
            return {
                "status": "complete",
                "message": "All questions have been answered.",
                "final_conclusion": self._generate_final_conclusion()
            }
        
        # Determine what to focus on next
        focus = self._determine_next_focus()
        self.context["current_focus"] = focus
        
        # Process the current focus
        if focus["type"] == "subproblem":
            result = self._process_subproblem(focus["id"], focus["content"])
        elif focus["type"] == "question":
            result = self._process_question(focus["id"], focus["content"])
        elif focus["type"] == "evidence":
            result = self._process_evidence(focus["id"], focus["content"])
        else:
            result = {
                "status": "error",
                "message": f"Unknown focus type: {focus['type']}"
            }
        
        # Update progress
        self._update_progress()
        
        # Save the updated state
        self._save_state()
        
        return {
            "status": "increment_processed",
            "focus": focus,
            "result": result,
            "progress": self.context["progress"],
            "remaining_questions": len([q for q in self.context["open_questions"] if q["status"] == "open"])
        }
    
    def get_reasoning_trace(self) -> Dict[str, Any]:
        """Get the full reasoning trace for the session"""
        # Get the workspace chain
        chain = workspace_get_chain(self.workspace_id)
        
        # Build a comprehensive reasoning trace
        trace = {
            "session_id": self.session_id,
            "problem_statement": self.context["problem_statement"],
            "subproblems": self.context["subproblems"],
            "insights": self.context["insights"],
            "progress": self.context["progress"],
            "steps": chain["chain"]["steps"],
            "derived_knowledge": chain["chain"]["derived_knowledge"]
        }
        
        return trace
    
    def commit_derived_knowledge(self) -> Dict[str, Any]:
        """Commit all derived knowledge to the main knowledge store"""
        result = workspace_commit_knowledge(self.workspace_id)
        
        return {
            "status": "knowledge_committed",
            "units_committed": result["committed_count"]
        }
    
    def close_session(self) -> Dict[str, Any]:
        """Close the reasoning session and commit all knowledge"""
        # Commit any remaining derived knowledge
        commit_result = self.commit_derived_knowledge()
        
        # Generate the final conclusion if not already done
        if not any(i for i in self.context["insights"] if i.get("type") == "final_conclusion"):
            final_conclusion = self._generate_final_conclusion()
            self.context["insights"].append({
                "type": "final_conclusion",
                "content": final_conclusion,
                "created_at": time.time()
            })
        
        # Save final state
        self._save_state()
        
        return {
            "status": "session_closed",
            "session_id": self.session_id,
            "committed_units": commit_result["units_committed"],
            "final_progress": self.context["progress"]
        }
    
    def _decompose_problem(self, problem: str) -> List[str]:
        """Break a complex problem into manageable subproblems"""
        # This is a simplified implementation
        # In a production system, this would use an LLM to decompose the problem
        
        # Parse complexity from the problem length
        complexity = min(5, max(2, len(problem) // 100))
        
        # Generate generic subproblems based on complexity
        subproblems = []
        
        if complexity >= 2:
            subproblems.append(f"What are the key concepts and terms in: '{problem}'?")
        
        if complexity >= 3:
            subproblems.append(f"What are the main challenges or difficulties in: '{problem}'?")
        
        if complexity >= 4:
            subproblems.append(f"What are the different approaches or methodologies for: '{problem}'?")
        
        subproblems.append(f"What evidence do we need to answer: '{problem}'?")
        subproblems.append(f"What conclusions can we draw about: '{problem}'?")
        
        return subproblems
    
    def _determine_next_focus(self) -> Dict[str, Any]:
        """Determine what to focus on next based on current state"""
        # Check for open subproblems
        for i, subproblem in enumerate(self.context["subproblems"]):
            subproblem_id = f"subproblem_{i+1}"
            if any(q["id"] == subproblem_id and q["status"] == "open" for q in self.context["open_questions"]):
                return {
                    "type": "subproblem",
                    "id": subproblem_id,
                    "content": subproblem
                }
        
        # Check for other open questions
        for question in self.context["open_questions"]:
            if question["status"] == "open" and question["id"] != "main_problem":
                return {
                    "type": "question",
                    "id": question["id"],
                    "content": question["question"]
                }
        
        # Check for unprocessed evidence
        for i, chunk in enumerate(self.context["evidence_chunks"]):
            if not chunk.get("processed", False):
                return {
                    "type": "evidence",
                    "id": f"evidence_{i+1}",
                    "content": chunk["content"]
                }
        
        # Default to the main problem if nothing else
        return {
            "type": "question",
            "id": "main_problem",
            "content": self.context["problem_statement"]
        }
    
    def _process_subproblem(self, subproblem_id: str, content: str) -> Dict[str, Any]:
        """Process a subproblem increment"""
        # Record this step in the workspace
        step = workspace_add_step(
            self.workspace_id,
            "subproblem_analysis",
            f"ANALYZING SUBPROBLEM: {content}"
        )
        
        # Query for knowledge related to this subproblem
        query_result = query_knowledge(content, reasoning_depth=1)
        relevant_count = len(query_result.get("direct_results", []))
        
        # Add an evidence chunk if we found relevant information
        if relevant_count > 0:
            chunk_id = len(self.context["evidence_chunks"]) + 1
            self.context["evidence_chunks"].append({
                "id": f"evidence_{chunk_id}",
                "type": "query_results",
                "content": f"Results for query: '{content}'",
                "data": query_result,
                "processed": False,
                "created_at": time.time()
            })
        
        # Add insights based on the subproblem
        insight = self._generate_insight_for_subproblem(content, query_result)
        if insight:
            self.context["insights"].append({
                "type": "subproblem_insight",
                "related_to": subproblem_id,
                "content": insight,
                "created_at": time.time()
            })
            
            # Record the insight in the workspace
            workspace_add_step(
                self.workspace_id,
                "insight",
                f"INSIGHT FOR SUBPROBLEM '{subproblem_id}': {insight}"
            )
        
        # Generate follow-up questions
        follow_ups = self._generate_follow_up_questions(content, query_result)
        for i, question in enumerate(follow_ups):
            question_id = f"{subproblem_id}_followup_{i+1}"
            self.context["open_questions"].append({
                "id": question_id,
                "question": question,
                "parent": subproblem_id,
                "status": "open",
                "created_at": time.time()
            })
        
        # Mark this subproblem as closed
        for q in self.context["open_questions"]:
            if q["id"] == subproblem_id:
                q["status"] = "closed"
                q["closed_at"] = time.time()
        
        return {
            "status": "subproblem_processed",
            "insight": insight,
            "follow_up_questions": follow_ups,
            "evidence_added": relevant_count > 0
        }
    
    def _process_question(self, question_id: str, content: str) -> Dict[str, Any]:
        """Process a question increment"""
        # Record this step in the workspace
        step = workspace_add_step(
            self.workspace_id,
            "question_analysis",
            f"ANALYZING QUESTION: {content}"
        )
        
        # Query for knowledge related to this question
        query_result = query_knowledge(content, reasoning_depth=1)
        
        # Generate answer based on available knowledge
        answer = self._generate_answer(content, query_result)
        
        # Record the answer
        workspace_add_step(
            self.workspace_id,
            "answer",
            f"ANSWER TO QUESTION '{question_id}': {answer}"
        )
        
        # Add insight
        self.context["insights"].append({
            "type": "question_insight",
            "related_to": question_id,
            "content": answer,
            "created_at": time.time()
        })
        
        # Mark question as answered
        for q in self.context["open_questions"]:
            if q["id"] == question_id:
                q["status"] = "answered"
                q["answer"] = answer
                q["answered_at"] = time.time()
        
        return {
            "status": "question_answered",
            "answer": answer
        }
    
    def _process_evidence(self, evidence_id: str, content: str) -> Dict[str, Any]:
        """Process an evidence chunk increment"""
        # Find the evidence chunk
        chunk = None
        chunk_index = None
        for i, c in enumerate(self.context["evidence_chunks"]):
            if c.get("id") == evidence_id:
                chunk = c
                chunk_index = i
                break
        
        if not chunk:
            return {
                "status": "error",
                "message": f"Evidence chunk {evidence_id} not found"
            }
        
        # Record this step in the workspace
        step = workspace_add_step(
            self.workspace_id,
            "evidence_analysis",
            f"ANALYZING EVIDENCE: {content}"
        )
        
        # Analyze the evidence
        summary = self._summarize_evidence(chunk)
        
        # Record the summary
        workspace_add_step(
            self.workspace_id,
            "evidence_summary",
            f"EVIDENCE SUMMARY FOR '{evidence_id}': {summary}"
        )
        
        # Add insight from evidence
        self.context["insights"].append({
            "type": "evidence_insight",
            "related_to": evidence_id,
            "content": summary,
            "created_at": time.time()
        })
        
        # Mark evidence as processed
        self.context["evidence_chunks"][chunk_index]["processed"] = True
        self.context["evidence_chunks"][chunk_index]["summary"] = summary
        
        return {
            "status": "evidence_processed",
            "summary": summary
        }
    
    def _generate_insight_for_subproblem(self, subproblem: str, query_result: Dict[str, Any]) -> str:
        """Generate an insight for a subproblem based on query results"""
        # This is a simplified implementation
        # In a real system, this would use an LLM to generate insights
        
        direct_results = query_result.get("direct_results", [])
        
        if not direct_results:
            return f"No specific information found about '{subproblem}' in the knowledge base."
        
        # Use the top result to generate an insight
        top_result = direct_results[0]
        content = top_result.get("content", "")
        confidence = top_result.get("confidence", 0)
        
        # Truncate content if it's too long
        if len(content) > 200:
            content = content[:200] + "..."
        
        if confidence > 0.8:
            return f"High confidence insight: {content}"
        elif confidence > 0.5:
            return f"Medium confidence insight: {content}"
        else:
            return f"Low confidence insight: {content}"
    
    def _generate_follow_up_questions(self, content: str, query_result: Dict[str, Any]) -> List[str]:
        """Generate follow-up questions based on a subproblem and query results"""
        # This is a simplified implementation
        # In a real system, this would use an LLM to generate follow-up questions
        
        # Generate 1-2 generic follow-up questions
        questions = []
        
        direct_results = query_result.get("direct_results", [])
        
        if len(direct_results) == 0:
            questions.append(f"What are the key terms and concepts related to '{content}'?")
        
        if "challenge" not in content.lower() and "problem" not in content.lower():
            questions.append(f"What are the main challenges or difficulties related to '{content}'?")
        
        if "approach" not in content.lower() and "method" not in content.lower():
            questions.append(f"What approaches or methodologies exist for addressing '{content}'?")
            
        # Limit to at most 2 follow-up questions
        return questions[:2]
    
    def _generate_answer(self, question: str, query_result: Dict[str, Any]) -> str:
        """Generate an answer to a question based on query results"""
        # This is a simplified implementation
        # In a real system, this would use an LLM to generate answers
        
        direct_results = query_result.get("direct_results", [])
        
        if not direct_results:
            return f"Insufficient information to answer the question: '{question}'"
        
        # Use the top result to generate an answer
        top_result = direct_results[0]
        content = top_result.get("content", "")
        confidence = top_result.get("confidence", 0)
        
        # Truncate content if it's too long
        if len(content) > 300:
            content = content[:300] + "..."
        
        confidence_str = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        
        return f"Based on available knowledge ({confidence_str} confidence): {content}"
    
    def _summarize_evidence(self, evidence: Dict[str, Any]) -> str:
        """Summarize an evidence chunk"""
        # This is a simplified implementation
        # In a real system, this would use an LLM to summarize evidence
        
        if evidence.get("type") == "query_results":
            direct_results = evidence.get("data", {}).get("direct_results", [])
            if not direct_results:
                return "No relevant information found in this evidence."
            
            # Summarize the top results
            summary_parts = []
            for i, result in enumerate(direct_results[:3]):
                content = result.get("content", "")
                confidence = result.get("confidence", 0)
                
                # Truncate content if it's too long
                if len(content) > 100:
                    content = content[:100] + "..."
                
                summary_parts.append(f"Result {i+1} (confidence: {confidence:.2f}): {content}")
            
            return "Evidence summary:\n" + "\n".join(summary_parts)
        
        return "Unknown evidence type. Cannot summarize."
    
    def _generate_final_conclusion(self) -> str:
        """Generate a final conclusion for the problem based on all insights"""
        # This is a simplified implementation
        # In a real system, this would use an LLM to generate a comprehensive conclusion
        
        # Get all insights
        insights = self.context["insights"]
        
        if not insights:
            return "Insufficient information to draw a conclusion for the main problem."
        
        # Start with a header
        conclusion = f"CONCLUSION FOR: {self.context['problem_statement']}\n\n"
        
        # Add a summary of key insights
        conclusion += "Based on the analysis, the key insights are:\n\n"
        
        insight_count = 0
        for insight in insights:
            if insight.get("type") in ["subproblem_insight", "evidence_insight"]:
                content = insight.get("content", "")
                if len(content) > 100:
                    content = content[:100] + "..."
                conclusion += f"- {content}\n"
                insight_count += 1
                if insight_count >= 5:  # Limit to 5 insights
                    break
        
        # Add answered questions
        answered_questions = [q for q in self.context["open_questions"] if q.get("status") == "answered"]
        if answered_questions:
            conclusion += "\nKey questions answered:\n\n"
            for i, q in enumerate(answered_questions[:3]):  # Limit to 3 questions
                question = q.get("question", "")
                answer = q.get("answer", "")
                
                if len(question) > 50:
                    question = question[:50] + "..."
                if len(answer) > 100:
                    answer = answer[:100] + "..."
                
                conclusion += f"{i+1}. Q: {question}\n   A: {answer}\n\n"
        
        # Add a final summary
        conclusion += "\nIn conclusion, "
        if insight_count > 0:
            conclusion += "the analysis provides valuable insights into the problem, "
            conclusion += "though further investigation may be needed for a complete understanding."
        else:
            conclusion += "there is insufficient information to provide a comprehensive answer to the problem."
        
        return conclusion
    
    def _has_open_questions(self) -> bool:
        """Check if there are any open questions remaining"""
        return any(q["status"] == "open" for q in self.context["open_questions"])
    
    def _update_progress(self) -> None:
        """Update the progress indicator based on current state"""
        # Calculate progress based on:
        # 1. Proportion of questions answered
        # 2. Proportion of evidence chunks processed
        # 3. Number of insights generated
        
        total_questions = len(self.context["open_questions"])
        answered_questions = sum(1 for q in self.context["open_questions"] if q["status"] != "open")
        
        total_evidence = len(self.context["evidence_chunks"])
        processed_evidence = sum(1 for c in self.context["evidence_chunks"] if c.get("processed", False))
        
        # Basic progress calculation
        if total_questions == 0:
            question_progress = 0
        else:
            question_progress = answered_questions / total_questions
        
        if total_evidence == 0:
            evidence_progress = 0
        else:
            evidence_progress = processed_evidence / total_evidence
        
        # Weight the components
        progress = 0.7 * question_progress + 0.3 * evidence_progress
        
        # Final adjustments
        if progress > 0.95 and self._has_open_questions():
            progress = 0.95  # Cap at 95% if there are still open questions
        
        self.context["progress"] = round(progress, 2)
    
    def _save_state(self) -> None:
        """Save the current state to disk for persistence"""
        state_dir = Path("./knowledge/long_context_sessions")
        state_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = state_dir / f"{self.session_id}.json"
        
        with open(state_file, "w") as f:
            json.dump(self.context, f, indent=2)
    
    @classmethod
    def load_session(cls, session_id: str) -> 'IncrementalReasoner':
        """Load a previously saved session"""
        state_file = Path(f"./knowledge/long_context_sessions/{session_id}.json")
        
        if not state_file.exists():
            raise ValueError(f"Session file for {session_id} not found")
        
        with open(state_file, "r") as f:
            context = json.load(f)
        
        # Create a new reasoner with the loaded session ID
        reasoner = cls(db_path=f"./knowledge/long_context.db", session_id=session_id)
        
        # Restore the context
        reasoner.context = context
        
        return reasoner


class RecursiveDecomposer:
    """
    Handles arbitrarily complex problems through recursive decomposition.
    
    This class decomposes a large problem into a tree of subproblems,
    solving each independently and then recombining the solutions.
    """
    
    def __init__(self, db_path: str = "./knowledge/recursive_decomp.db"):
        """Initialize the recursive decomposer"""
        # Initialize the underlying knowledge system
        initialize_knowledge_system(db_path)
        
        # Keep track of subproblem reasoners
        self.reasoners = {}
        self.main_session_id = None
        
        # Problem decomposition tree
        self.problem_tree = {
            "root": None,
            "subproblems": {},
            "dependencies": {}
        }
    
    def decompose_problem(self, problem_statement: str) -> Dict[str, Any]:
        """Decompose a complex problem into a tree of subproblems"""
        # Create the main problem session
        main_reasoner = IncrementalReasoner(session_id=f"main_{str(uuid.uuid4())}")
        main_reasoner.set_problem(problem_statement)
        
        self.main_session_id = main_reasoner.session_id
        self.reasoners[main_reasoner.session_id] = main_reasoner
        
        # Set as root problem
        self.problem_tree["root"] = {
            "id": main_reasoner.session_id,
            "statement": problem_statement
        }
        
        # Generate first-level decomposition
        subproblems = main_reasoner.context["subproblems"]
        
        # Create reasoners for each subproblem
        for i, subproblem in enumerate(subproblems):
            sub_reasoner = IncrementalReasoner(session_id=f"sub_{i+1}_{str(uuid.uuid4())}")
            sub_reasoner.set_problem(subproblem)
            
            self.reasoners[sub_reasoner.session_id] = sub_reasoner
            
            # Add to problem tree
            self.problem_tree["subproblems"][sub_reasoner.session_id] = {
                "id": sub_reasoner.session_id,
                "statement": subproblem,
                "parent": main_reasoner.session_id,
                "index": i+1
            }
            
            # Add dependency
            if main_reasoner.session_id not in self.problem_tree["dependencies"]:
                self.problem_tree["dependencies"][main_reasoner.session_id] = []
            
            self.problem_tree["dependencies"][main_reasoner.session_id].append(sub_reasoner.session_id)
        
        return {
            "main_session": main_reasoner.session_id,
            "subproblem_count": len(subproblems),
            "subproblems": [
                {"id": r_id, "statement": data["statement"]} 
                for r_id, data in self.problem_tree["subproblems"].items()
            ]
        }
    
    def process_next_increment(self) -> Dict[str, Any]:
        """Process the next increment in the problem tree"""
        # Identify the next session to work on
        session_id = self._get_next_session()
        
        if not session_id:
            # All sessions complete, synthesize final answer
            return self._synthesize_final_answer()
        
        # Process an increment for the selected session
        reasoner = self.reasoners[session_id]
        result = reasoner.process_next_increment()
        
        # Check if this session is complete
        if result["status"] == "complete":
            # Propagate insights to parent
            parent_id = self._get_parent_session(session_id)
            if parent_id:
                self._propagate_insights(session_id, parent_id)
        
        # Return information about the increment
        return {
            "status": "increment_processed",
            "session_id": session_id,
            "is_main": session_id == self.main_session_id,
            "statement": self._get_session_statement(session_id),
            "increment_result": result,
            "tree_progress": self._calculate_tree_progress()
        }
    
    def get_problem_tree(self) -> Dict[str, Any]:
        """Get the full problem decomposition tree"""
        tree = self.problem_tree.copy()
        
        # Add progress information
        for session_id, reasoner in self.reasoners.items():
            if session_id in tree["subproblems"]:
                tree["subproblems"][session_id]["progress"] = reasoner.context["progress"]
            elif tree["root"] and tree["root"]["id"] == session_id:
                tree["root"]["progress"] = reasoner.context["progress"]
        
        return tree
    
    def get_session_trace(self, session_id: str) -> Dict[str, Any]:
        """Get the reasoning trace for a specific session"""
        if session_id not in self.reasoners:
            return {"error": f"Session {session_id} not found"}
        
        return self.reasoners[session_id].get_reasoning_trace()
    
    def close(self) -> Dict[str, Any]:
        """Close all sessions and commit knowledge"""
        results = {}
        
        # Close all sessions
        for session_id, reasoner in self.reasoners.items():
            results[session_id] = reasoner.close_session()
        
        return {
            "status": "all_sessions_closed",
            "results": results
        }
    
    def _get_next_session(self) -> Optional[str]:
        """Determine which session to process next based on dependencies"""
        # Start with leaf nodes (sessions with no dependencies)
        for session_id, reasoner in self.reasoners.items():
            # Skip completed sessions
            if not reasoner._has_open_questions():
                continue
                
            # Check if this is a leaf node or if all dependencies are complete
            if session_id not in self.problem_tree["dependencies"] or self._are_dependencies_complete(session_id):
                return session_id
        
        # If no leaf nodes, check if main session has open questions
        if self.main_session_id and self.reasoners[self.main_session_id]._has_open_questions():
            return self.main_session_id
        
        # No sessions need processing
        return None
    
    def _are_dependencies_complete(self, session_id: str) -> bool:
        """Check if all dependencies of a session are complete"""
        if session_id not in self.problem_tree["dependencies"]:
            return True
        
        for dep_id in self.problem_tree["dependencies"][session_id]:
            if dep_id in self.reasoners and self.reasoners[dep_id]._has_open_questions():
                return False
        
        return True
    
    def _get_parent_session(self, session_id: str) -> Optional[str]:
        """Get the parent session of a subproblem"""
        if session_id == self.main_session_id:
            return None
        
        for sub_id, data in self.problem_tree["subproblems"].items():
            if sub_id == session_id and "parent" in data:
                return data["parent"]
        
        return None
    
    def _get_session_statement(self, session_id: str) -> str:
        """Get the problem statement for a session"""
        if session_id == self.main_session_id:
            return self.problem_tree["root"]["statement"]
        
        if session_id in self.problem_tree["subproblems"]:
            return self.problem_tree["subproblems"][session_id]["statement"]
        
        return "Unknown session"
    
    def _propagate_insights(self, from_session: str, to_session: str) -> None:
        """Propagate insights from a completed session to its parent"""
        if from_session not in self.reasoners or to_session not in self.reasoners:
            return
        
        # Get insights from the completed session
        insights = self.reasoners[from_session].context["insights"]
        
        if not insights:
            return
        
        # Find the most significant insight
        significant_insight = None
        for insight in insights:
            if insight.get("type") == "final_conclusion":
                significant_insight = insight
                break
        
        if not significant_insight:
            for insight in insights:
                if insight.get("type") == "subproblem_insight":
                    significant_insight = insight
                    break
        
        if not significant_insight and insights:
            significant_insight = insights[0]
        
        if not significant_insight:
            return
        
        # Add as evidence to the parent session
        statement = self._get_session_statement(from_session)
        content = significant_insight.get("content", "No content")
        
        evidence = {
            "id": f"subproblem_result_{from_session}",
            "type": "subproblem_result",
            "statement": statement,
            "content": f"RESULT FROM SUBPROBLEM: {statement}\n\n{content}",
            "processed": False,
            "created_at": time.time()
        }
        
        self.reasoners[to_session].context["evidence_chunks"].append(evidence)
    
    def _synthesize_final_answer(self) -> Dict[str, Any]:
        """Synthesize the final answer from all sessions"""
        if not self.main_session_id or self.main_session_id not in self.reasoners:
            return {
                "status": "error",
                "message": "Main session not found"
            }
        
        main_reasoner = self.reasoners[self.main_session_id]
        
        # Generate the final conclusion
        conclusion = main_reasoner._generate_final_conclusion()
        
        # Add to main session insights
        main_reasoner.context["insights"].append({
            "type": "final_conclusion",
            "content": conclusion,
            "created_at": time.time()
        })
        
        # Save the state
        main_reasoner._save_state()
        
        return {
            "status": "complete",
            "conclusion": conclusion,
            "main_session": self.main_session_id
        }
    
    def _calculate_tree_progress(self) -> float:
        """Calculate overall progress across the problem tree"""
        if not self.reasoners:
            return 0.0
        
        # Weight the main session more heavily
        main_weight = 0.5
        sub_weight = 0.5 / max(1, len(self.reasoners) - 1)
        
        progress = 0.0
        
        # Add main session progress
        if self.main_session_id in self.reasoners:
            progress += main_weight * self.reasoners[self.main_session_id].context["progress"]
        
        # Add subproblem progress
        for session_id, reasoner in self.reasoners.items():
            if session_id != self.main_session_id:
                progress += sub_weight * reasoner.context["progress"]
        
        return round(progress, 2)


def demo_incremental_reasoner():
    """Demonstrate the incremental reasoner on a complex problem"""
    print("\n===== INCREMENTAL REASONER DEMO =====\n")
    
    # Create the reasoner
    reasoner = IncrementalReasoner()
    
    # Set a complex problem
    print("Setting complex problem...")
    problem = (
        "What are the societal implications of widespread adoption of artificial intelligence "
        "in healthcare, education, and the workforce over the next decade, considering economic, "
        "ethical, and policy dimensions?"
    )
    reasoner.set_problem(problem)
    print(f"Problem: {problem}\n")
    print(f"Decomposed into {len(reasoner.context['subproblems'])} subproblems:")
    for i, subproblem in enumerate(reasoner.context["subproblems"]):
        print(f"{i+1}. {subproblem}")
    
    print("\nProcessing increments...\n")
    
    # Process several increments
    for i in range(10):  # Process 10 increments to demonstrate
        print(f"\n--- INCREMENT {i+1} ---")
        result = reasoner.process_next_increment()
        
        print(f"Focus: {result['focus']['type']} - {result['focus']['content'][:50]}...")
        
        if result["status"] == "complete":
            print("\nReasoning complete!")
            print(f"Final conclusion: {result['final_conclusion'][:200]}...")
            break
        
        print(f"Progress: {result['progress'] * 100:.0f}%")
        print(f"Remaining questions: {result['remaining_questions']}")
    
    # Get the reasoning trace
    trace = reasoner.get_reasoning_trace()
    print(f"\nReasoning trace has {len(trace['steps'])} steps and {len(trace['insights'])} insights")
    
    # Commit knowledge
    reasoner.commit_derived_knowledge()
    
    # Close the session
    reasoner.close_session()
    
    print("\nIncremental reasoning session closed")


def demo_recursive_decomposer():
    """Demonstrate the recursive decomposer on a complex problem"""
    print("\n===== RECURSIVE DECOMPOSER DEMO =====\n")
    
    # Create the decomposer
    decomposer = RecursiveDecomposer()
    
    # Set a complex problem
    print("Setting complex problem...")
    problem = (
        "How can we design a sustainable smart city transportation system that integrates "
        "autonomous vehicles, public transit, and micro-mobility options while addressing "
        "concerns of equity, privacy, environmental impact, and economic viability?"
    )
    
    result = decomposer.decompose_problem(problem)
    print(f"Problem: {problem}\n")
    print(f"Decomposed into {result['subproblem_count']} subproblems:")
    for i, subproblem in enumerate(result["subproblems"]):
        print(f"{i+1}. {subproblem['statement']}")
    
    print("\nProcessing increments across problem tree...\n")
    
    # Process several increments
    for i in range(15):  # Process 15 increments to demonstrate
        print(f"\n--- INCREMENT {i+1} ---")
        result = decomposer.process_next_increment()
        
        if result["status"] == "complete":
            print("\nReasoning complete!")
            print(f"Final conclusion: {result['conclusion'][:200]}...")
            break
        
        print(f"Working on: {result['statement'][:50]}...")
        print(f"Session: {result['session_id']} {'(MAIN)' if result['is_main'] else ''}")
        print(f"Tree progress: {result['tree_progress'] * 100:.0f}%")
    
    # Close all sessions
    decomposer.close()
    
    print("\nRecursive decomposition complete")


if __name__ == "__main__":
    # Create directories
    os.makedirs("./knowledge", exist_ok=True)
    os.makedirs("./knowledge/long_context_sessions", exist_ok=True)
    
    # Run demos
    try:
        demo_incremental_reasoner()
        demo_recursive_decomposer()
    finally:
        # Ensure knowledge system is shut down
        shutdown_knowledge_system()