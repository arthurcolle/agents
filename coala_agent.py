#!/usr/bin/env python3
"""
coala_agent.py
--------------
An advanced language agent based on the CoALA (Cognitive Architectures for Language Agents)
framework, designed for generalist computer usage tasks.
"""

import re
import os
import sys
import json
import argparse
import subprocess
import inspect
import importlib
import importlib.util
import math
import sqlite3
import uuid
import hashlib
import base64
import time
import queue
import threading
import tempfile
import urllib.parse
import asyncio
import traceback
from typing import Dict, List, Any, Callable, Optional, Union, Tuple, Set
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO, BytesIO
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

# --- Import from atomic_agent ---
try:
    from atomic_agent import AgentOrchestrator, ScoutAgent, ToolRegistry as AtomicToolRegistry, VectorMemory
    # Import additional components from atomic_agent
    from atomic_agent import AsyncTaskProcessor, JinaClient, CodeRepository, extract_python_code, parse_function_calls
    # Import embedding_model and vector_store from atomic_agent
    from atomic_agent import embedding_model, vector_store
    # Rename imported ToolRegistry to avoid conflict if needed later
except ImportError as e:
    from rich.console import Console
    console = Console()
    console.print(f"[red]Error importing from atomic_agent: {e}[/red]")
    console.print("[yellow]Please ensure atomic_agent.py is in the same directory or Python path.[/yellow]")
    sys.exit(1)

# --- Dependency Handling ---
try:
    import aiohttp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp

try:
    from together import Together
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "together"])
    from together import Together

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.panel import Panel
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.panel import Panel

# --- Initialization ---
# Initialize console early, before it might be used in except blocks
from rich.console import Console
console = Console()

# --- Import from atomic_agent ---
try:
    from atomic_agent import AgentOrchestrator, ScoutAgent, ToolRegistry as AtomicToolRegistry
    # Rename imported ToolRegistry to avoid conflict if needed later
except ImportError as e:
    console.print(f"[red]Error importing from atomic_agent: {e}[/red]")
    console.print("[yellow]Please ensure atomic_agent.py is in the same directory or Python path.[/yellow]")
# --- Dependency Handling ---
try:
    import aiohttp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp

try:
    from together import Together
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "together"])
    from together import Together

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.panel import Panel
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.panel import Panel

# --- Initialization ---
console = Console() # Initialize console early

# --- Import from atomic_agent ---
try:
    from atomic_agent import AgentOrchestrator, ScoutAgent, ToolRegistry as AtomicToolRegistry
    # Rename imported ToolRegistry to avoid conflict if needed later
except ImportError as e:
    console.print(f"[red]Error importing from atomic_agent: {e}[/red]")
    console.print("[yellow]Please ensure atomic_agent.py is in the same directory or Python path.[/yellow]")
    sys.exit(1)

# --- Dependency Handling ---
# (Keep other dependency imports here if needed)

# --- CoALA Core Components ---

@dataclass
class WorkingMemory:
    """Short-term memory holding current context, goals, and intermediate results."""
    current_goal: Optional[str] = None
    recent_observations: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_reasoning: List[str] = field(default_factory=list)
    proposed_actions: List[Dict[str, Any]] = field(default_factory=list)
    selected_action: Optional[Dict[str, Any]] = None
    variables: Dict[str, Any] = field(default_factory=dict) # General purpose storage

    def add_observation(self, obs: Any, source: str = "external"):
        self.recent_observations.append({"timestamp": time.time(), "source": source, "data": obs})
        # Limit history size if needed
        if len(self.recent_observations) > 10:
            self.recent_observations.pop(0)

    def clear_cycle_state(self):
        self.intermediate_reasoning = []
        self.proposed_actions = []
        self.selected_action = None

@dataclass
class LongTermMemory:
    """Base class for long-term memory modules."""
    # Placeholder for common methods like save, load, search
    pass

@dataclass
class EpisodicMemory(LongTermMemory):
    """Stores sequences of agent experiences (observations, actions, outcomes)."""
    episodes: List[Dict[str, Any]] = field(default_factory=list)

    def add_episode(self, episode_data: Dict[str, Any]):
        self.episodes.append(episode_data)

    def retrieve_relevant(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        # Basic keyword search for now, replace with vector search later
        results = []
        for ep in reversed(self.episodes):
            if query.lower() in json.dumps(ep).lower():
                results.append(ep)
                if len(results) >= limit:
                    break
        return results

@dataclass
class SemanticMemory(LongTermMemory):
    """Stores factual knowledge about the world and the agent itself."""
    facts: Dict[str, Any] = field(default_factory=dict) # Simple key-value store

    def add_fact(self, key: str, value: Any):
        self.facts[key] = {"value": value, "timestamp": time.time()}

    def retrieve_fact(self, key: str) -> Optional[Any]:
        return self.facts.get(key)

@dataclass
class ProceduralMemory(LongTermMemory):
    """Stores the agent's skills and procedures (code, LLM prompts, etc.)."""
    # This includes the agent's own code and potentially learned skills/tools
    # For now, it's implicitly represented by the ToolRegistry and agent code
    pass

# --- Action Space ---

@dataclass
class Action:
    """Base class for all actions."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InternalAction(Action):
    """Actions operating on the agent's internal memory."""
    pass

@dataclass
class ExternalAction(Action):
    """Actions interacting with the external environment."""
    pass

# --- CoALA Agent Class ---

class CoALAAgent:
    """
    Cognitive Architecture for Language Agents (CoALA).
    Integrates memory, actions, and decision-making using an LLM.
    """
    def __init__(self, model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
        self.model = model
        self.api_key = os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set TOGETHER_API_KEY environment variable.")
        self.client = Together(api_key=self.api_key)

        # Memory Modules
        self.working_memory = WorkingMemory()
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        # Procedural memory is implicitly the agent's code + ToolRegistry

        # Initialize the Agent Orchestrator with configurable number of scouts
        num_scouts = 5  # Default value, can be made configurable via parameters
        self.agent_orchestrator = AgentOrchestrator(num_scouts=num_scouts, model=model)
        console.print(f"[green]Initialized Agent Orchestrator with {len(self.agent_orchestrator.scouts)} scouts.[/green]")

        # Use the full ToolRegistry from atomic_agent
        self.tool_registry = AtomicToolRegistry()
        # Pass self (CoALAAgent instance) to the registry if needed by tools
        self.tool_registry.agent_instance = self # Provide access for tools needing agent state
        
        # Initialize advanced capabilities from atomic_agent
        self.memory = VectorMemory(embedding_model, vector_store)
        self.task_processor = AsyncTaskProcessor()
        self.code_repository = CodeRepository(db_path="coala_code_artifacts.db")

        self.conversation_history = [] # Stores raw LLM interactions

        # Define the core system prompt based on CoALA principles with advanced function calling
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are a CoALA-based language agent designed for advanced computer usage and problem-solving. "
                "Your goal is to assist the user by understanding their requests, reasoning about the steps needed, "
                "and utilizing available tools (actions) to interact with the environment (files, code, web).\n\n"
                "Follow this enhanced decision cycle:\n"
                "1. **Observe:** Receive user input and environmental feedback. Analyze multimodal content if present.\n"
                "2. **Orient (Plan):** Analyze the current state (working memory), retrieve relevant knowledge (long-term memory), "
                "   and reason to propose potential actions (internal or external). For complex tasks, decompose into subtasks.\n"
                "3. **Decide:** Evaluate proposed actions and select the optimal sequence. Consider parallel execution when possible.\n"
                "4. **Act:** Execute the selected actions (e.g., call tools, update memory, respond to user). Monitor execution and adapt as needed.\n"
                "5. **Reflect:** After completing actions, reflect on the effectiveness and learn from the experience.\n\n"
                "When calling functions:\n"
                "- Use precise parameter names and types as defined in the function specifications\n"
                "- For complex tasks, use orchestrate_tasks to delegate to specialized scout agents\n"
                "- Chain multiple function calls when needed to solve multi-step problems\n"
                "- Use JSON format for structured reasoning and action selection\n\n"
                "You can execute Python code, search the web, process images, analyze data, and interact with files. "
                "Use your capabilities creatively to solve problems efficiently."
            )
        }
        self.add_log(self.system_prompt) # Add system prompt to history

    def add_log(self, message: Dict):
        """Adds a message to the conversation history and potentially episodic memory."""
        self.conversation_history.append(message)
        # Simple episodic logging
        self.episodic_memory.add_episode(message)

    def get_memory_context(self) -> str:
        """Synthesizes context from working and long-term memory."""
        context = "## Current Context\n"
        if self.working_memory.current_goal:
            context += f"- Goal: {self.working_memory.current_goal}\n"
        if self.working_memory.recent_observations:
            last_obs = self.working_memory.recent_observations[-1]
            context += f"- Last Observation ({last_obs['source']}): {str(last_obs['data'])[:200]}...\n" # Truncate obs
        # Add retrieval from episodic/semantic memory here later
        return context

    def plan_step(self) -> List[Dict[str, Any]]:
        """Uses LLM to reason and propose actions based on current state."""
        context = self.get_memory_context()
        # Escape curly braces for the example JSON in the f-string
        example_json = "[{'action_name': '...', 'arguments': {{...}}, 'reasoning': '...'}]"
        planning_prompt_content = (
            f"{context}\nBased on the current goal and recent observations, what are the next logical steps or actions? "
            f"Consider using tools like 'execute_python', 'read_file', 'web_search', or 'respond_to_user'. "
            f"For complex tasks, consider using 'orchestrate_tasks' or 'assign_specialized_task'. "
            f"Propose 1-3 actions using JSON format like: {example_json}"
        )
        planning_prompt = [
            *self.conversation_history,
            {"role": "user", "content": planning_prompt_content}
        ]

        # Define Pydantic model for expected JSON output
        class ProposedAction(BaseModel):
            action_name: str = Field(description="Name of the action/tool to call")
            arguments: Dict[str, Any] = Field(description="Arguments for the action")
            reasoning: str = Field(description="Why this action is proposed")

        class ActionProposal(BaseModel):
            proposed_actions: List[ProposedAction]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=planning_prompt,
                response_format={
                    "type": "json_object",
                    "schema": ActionProposal.model_json_schema()
                },
                max_tokens=1024
            )
            content = response.choices[0].message.content
            self.add_log({"role": "assistant", "content": content, "type": "planning"}) # Log planning step

            # Parse JSON
            proposal_data = json.loads(content)
            proposed_actions = proposal_data.get("proposed_actions", [])

            # Validate actions against registry
            valid_actions = []
            for action in proposed_actions:
                if action.get("action_name") in self.tool_registry.functions:
                    valid_actions.append(action)
                else:
                    console.print(f"[yellow]Warning: Proposed action '{action.get('action_name')}' not found in registry.[/yellow]")

            self.working_memory.proposed_actions = valid_actions
            self.working_memory.intermediate_reasoning.append(content) # Store raw reasoning
            return valid_actions
        except Exception as e:
            console.print(f"[red]Error during planning step: {e}[/red]")
            self.add_log({"role": "system", "content": f"Planning Error: {e}", "type": "error"})
            return []

    def decide_step(self) -> Optional[Dict[str, Any]]:
        """Selects the best action from proposed actions."""
        if not self.working_memory.proposed_actions:
            return None
        if len(self.working_memory.proposed_actions) == 1:
            self.working_memory.selected_action = self.working_memory.proposed_actions[0]
            return self.working_memory.selected_action

        # Simple strategy: pick the first valid proposed action
        # TODO: Implement more sophisticated evaluation/selection later
        self.working_memory.selected_action = self.working_memory.proposed_actions[0]
        self.add_log({"role": "system", "content": f"Selected action: {self.working_memory.selected_action['action_name']}", "type": "decision"})
        return self.working_memory.selected_action

    def act_step(self) -> Any:
        """Executes the selected action."""
        action = self.working_memory.selected_action
        if not action:
            return {"error": "No action selected"}

        action_name = action.get("action_name")
        arguments = action.get("arguments", {})

        # Check if the action is respond_to_user, which we'll handle directly
        if action_name == "respond_to_user":
            response_text = arguments.get("response_text", "I'm not sure how to respond.")
            self.add_log({"role": "assistant", "content": response_text})
            return response_text
            
        # Call tool using the registry
        result = self.tool_registry.call_function(action_name, arguments)

        # Log tool call and result
        self.add_log({
            "role": "tool", # Use standard 'tool' role
            "name": action_name,
            "content": json.dumps(result, indent=2), # Log result as content
            "tool_call_id": str(uuid.uuid4()) # Generate a dummy ID
        })
        # Add result as an observation in working memory
        self.working_memory.add_observation(result, source=f"action:{action_name}")
        
        # Special handling for web_search results to potentially trigger follow-up actions
        if action_name == "web_search" and isinstance(result, dict) and result.get("success"):
            return self._process_web_search_result(result, arguments.get("query", ""))
            
        return result # Return raw result for potential further processing

    def decision_cycle(self, user_input: Optional[str] = None):
        """Runs one cycle of the CoALA Observe-Orient-Decide-Act loop with enhanced capabilities."""
        self.working_memory.clear_cycle_state()
        cycle_start_time = time.time()

        # 1. Observe - Enhanced with multimodal processing and context awareness
        if user_input:
            # Process multimodal input with advanced detection
            processed_input = self._process_input(user_input)
            self.working_memory.add_observation(processed_input, source="user")
            self.add_log({"role": "user", "content": processed_input if isinstance(processed_input, str) else str(processed_input)})
            
            # Intelligent goal setting based on input analysis
            if not self.working_memory.current_goal:
                if isinstance(processed_input, str):
                    # Analyze the input to extract a meaningful goal
                    if "?" in processed_input:
                        self.working_memory.current_goal = f"Answer question: {processed_input}"
                    elif any(cmd in processed_input.lower() for cmd in ["create", "make", "build", "generate"]):
                        self.working_memory.current_goal = f"Create content: {processed_input}"
                    elif any(cmd in processed_input.lower() for cmd in ["analyze", "examine", "investigate"]):
                        self.working_memory.current_goal = f"Analyze information: {processed_input}"
                    else:
                        self.working_memory.current_goal = processed_input
                else:  # Multimodal content
                    # Extract text from multimodal content for goal setting
                    text_content = ""
                    has_image = False
                    for item in processed_input if isinstance(processed_input, list) else []:
                        if item.get("type") == "text":
                            text_content += item.get("text", "") + " "
                        elif item.get("type") == "image_url":
                            has_image = True
                    
                    if has_image and text_content:
                        self.working_memory.current_goal = f"Process image and text: {text_content.strip()}"
                    elif has_image:
                        self.working_memory.current_goal = "Analyze image content"
                    else:
                        self.working_memory.current_goal = "Process multimodal input"
            
            # Enhanced memory storage with metadata
            if hasattr(self, 'memory'):
                if isinstance(processed_input, str):
                    # Add contextual metadata
                    metadata = {
                        "type": "text", 
                        "role": "user",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "goal": self.working_memory.current_goal,
                        "sentiment": "neutral"  # Could be enhanced with sentiment analysis
                    }
                    self.memory.add_memory(processed_input, metadata)
                elif isinstance(processed_input, list):  # Multimodal content
                    for item in processed_input:
                        if item.get("type") == "text":
                            self.memory.add_memory(item.get("text", ""), {
                                "type": "text", 
                                "role": "user",
                                "part_of_multimodal": True
                            })
                        elif item.get("type") == "image_url" and "url" in item.get("image_url", {}):
                            self.memory.add_memory(item["image_url"]["url"], {
                                "type": "image", 
                                "role": "user",
                                "content_type": self._detect_image_type(item["image_url"]["url"])
                            })

        # Check for task complexity and use advanced capabilities when appropriate
        task_complexity = self._analyze_task_complexity(user_input)
        self.working_memory.variables["task_complexity"] = task_complexity
        
        if task_complexity == "high" or self._should_use_advanced_capabilities(user_input):
            console.print(f"[cyan]Using advanced capabilities for {task_complexity} complexity task[/cyan]")
            result = self._handle_with_advanced_capabilities(user_input)
            if result:
                # Track execution metrics
                cycle_duration = time.time() - cycle_start_time
                self.add_log({
                    "role": "system", 
                    "content": f"Advanced capabilities used. Decision cycle completed in {cycle_duration:.2f}s",
                    "type": "metrics"
                })
                return result

        # 2. Orient (Plan) - Enhanced with memory retrieval and context integration
        # Retrieve relevant memories before planning
        if hasattr(self, 'memory'):
            relevant_memories = self.memory.search_memory(
                self.working_memory.current_goal or user_input or "", 
                limit=3
            )
            if relevant_memories:
                memory_context = "Relevant past information:\n"
                for mem in relevant_memories:
                    memory_context += f"- {mem['content'][:100]}...\n"
                self.working_memory.variables["memory_context"] = memory_context
                console.print(f"[dim cyan]Retrieved {len(relevant_memories)} relevant memories[/dim cyan]")

        # Enhanced planning with context integration
        proposed_actions = self.plan_step()
        
        if not proposed_actions:
            # If planning fails, use fallback strategies based on task type
            console.print("[yellow]Primary planning failed, attempting alternative planning approach.[/yellow]")
            
            # Try different planning approach based on task type
            if task_complexity == "high":
                # For complex tasks, try breaking it down first
                self.working_memory.selected_action = {
                    "action_name": "orchestrate_tasks", 
                    "arguments": {
                        "main_task": self.working_memory.current_goal or "Process user request",
                        "subtasks": [
                            "Analyze the user request in detail",
                            "Identify the key components needed",
                            "Formulate a response strategy"
                        ]
                    }
                }
            else:
                # For simpler tasks, use direct response
                self.working_memory.selected_action = {
                    "action_name": "respond_to_user", 
                    "arguments": {
                        "response_text": "I'm not sure how to proceed with your request. Could you provide more details or clarify what you're looking for?"
                    }
                }
        else:
            # 3. Decide - Enhanced with multi-criteria evaluation
            self.decide_step()

        # 4. Act - Enhanced with execution monitoring and error recovery
        try:
            result = self.act_step()
            
            # Check if the action was a direct response to the user via the tool
            if self.working_memory.selected_action and self.working_memory.selected_action.get("action_name") == "respond_to_user":
                # Track execution metrics
                cycle_duration = time.time() - cycle_start_time
                self.add_log({
                    "role": "system", 
                    "content": f"Direct response. Decision cycle completed in {cycle_duration:.2f}s",
                    "type": "metrics"
                })
                return result
            else:
                # If another action was taken, generate a comprehensive summary
                console.print(f"[bold magenta]Action Result:[/bold magenta]\n{json.dumps(result, indent=2)}")
                
                # Check if this is a web search result with a pre-generated summary
                if (self.working_memory.selected_action.get('action_name') == "web_search" and 
                    isinstance(result, dict) and "summary" in result):
                    summary_response = result["summary"]
                else:
                    # Enhanced summarization with context awareness for other actions
                    summary_response = self._generate_enhanced_action_summary(result)
                
                # Store the result in memory for future reference
                if hasattr(self, 'memory'):
                    self.memory.add_memory(
                        f"Action: {self.working_memory.selected_action.get('action_name')} - Result summary: {summary_response[:100]}...",
                        {
                            "type": "action_result",
                            "action": self.working_memory.selected_action.get('action_name'),
                            "success": "error" not in result if isinstance(result, dict) else True
                        }
                    )

                self.add_log({"role": "assistant", "content": summary_response})
                
                # Track execution metrics
                cycle_duration = time.time() - cycle_start_time
                self.add_log({
                    "role": "system", 
                    "content": f"Action execution. Decision cycle completed in {cycle_duration:.2f}s",
                    "type": "metrics"
                })
                
                return summary_response
                
        except Exception as e:
            # Enhanced error recovery
            console.print(f"[red]Error during action execution: {str(e)}[/red]")
            console.print(traceback.format_exc())
            
            # Try to recover with a fallback action
            fallback_response = f"I encountered an issue while processing your request: {str(e)}. Let me try a different approach."
            
            # Log the error for future improvement
            self.add_log({
                "role": "system", 
                "content": f"Error in act_step: {str(e)}\n{traceback.format_exc()}",
                "type": "error"
            })
            
            return fallback_response
            
    def _analyze_task_complexity(self, user_input) -> str:
        """Analyze the complexity of the task based on the user input."""
        if not user_input or not isinstance(user_input, str):
            return "medium"
            
        # Count indicators of complexity
        complexity_indicators = {
            "high": ["complex", "multiple", "analyze", "compare", "create a", "build a", "implement", 
                    "design", "optimize", "improve", "automate", "integrate"],
            "medium": ["find", "search", "explain", "describe", "summarize", "calculate", "convert"],
            "low": ["what is", "who is", "when", "where", "simple", "quick", "help me"]
        }
        
        # Count words and sentences
        word_count = len(user_input.split())
        sentence_count = len(re.split(r'[.!?]+', user_input))
        
        # Count complexity indicators
        high_indicators = sum(1 for indicator in complexity_indicators["high"] 
                             if indicator in user_input.lower())
        medium_indicators = sum(1 for indicator in complexity_indicators["medium"] 
                               if indicator in user_input.lower())
        low_indicators = sum(1 for indicator in complexity_indicators["low"] 
                            if indicator in user_input.lower())
        
        # Determine complexity based on multiple factors
        if (word_count > 50 or sentence_count > 3 or high_indicators > 1 or
            "code" in user_input.lower() and word_count > 20):
            return "high"
        elif (word_count > 20 or sentence_count > 1 or medium_indicators > 0 or
             high_indicators > 0):
            return "medium"
        else:
            return "low"
            
    def _detect_image_type(self, image_url: str) -> str:
        """Detect the type of image based on URL or content."""
        if not image_url:
            return "unknown"
            
        # Check for common image types based on URL
        if "chart" in image_url.lower() or "graph" in image_url.lower():
            return "chart"
        elif "diagram" in image_url.lower() or "flow" in image_url.lower():
            return "diagram"
        elif "screenshot" in image_url.lower() or "screen" in image_url.lower():
            return "screenshot"
        elif "photo" in image_url.lower() or any(ext in image_url.lower() for ext in [".jpg", ".jpeg", ".png"]):
            return "photo"
        elif "code" in image_url.lower() or "snippet" in image_url.lower():
            return "code"
        else:
            return "image"
            
    def _generate_enhanced_action_summary(self, result):
        """Generate a more sophisticated summary of action results using the LLM with enhanced context."""
        if not result:
            return "No result was returned from the action."
            
        # For simple success/failure cases, use a template
        if isinstance(result, dict):
            if result.get('success'):
                summary = f"Action '{self.working_memory.selected_action.get('action_name', 'unknown')}' executed successfully."
                
                # Include stdout if available and not too long
                stdout = result.get('stdout')
                if stdout and len(stdout) < 200:
                    summary += f" Output: {stdout.strip()}"
                    
                # Include result if available and not too long
                result_value = result.get('result')
                if result_value and isinstance(result_value, str) and len(result_value) < 200:
                    summary += f" Result: {result_value}"
                    
                return summary
            elif 'error' in result:
                return f"Action '{self.working_memory.selected_action.get('action_name', 'unknown')}' failed. Error: {result['error']}"
                
        # For complex results, use the LLM to generate a summary with enhanced context
        try:
            # Truncate result if too large
            result_str = str(result)
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "... (truncated)"
                
            # Include context about the original goal and action
            action_context = f"Original goal: {self.working_memory.current_goal}\n"
            action_context += f"Selected action: {self.working_memory.selected_action.get('action_name')}\n"
            action_context += f"Action arguments: {json.dumps(self.working_memory.selected_action.get('arguments', {}))}\n\n"
            
            summary_prompt = [
                {"role": "system", "content": "You are an AI assistant that summarizes complex action results concisely and insightfully."},
                {"role": "user", "content": f"{action_context}Summarize this action result in 2-3 sentences, highlighting the most important information and explaining its significance to the original goal:\n\n{result_str}"}
            ]
            
            summary_response = self.client.chat.completions.create(
                model=self.model,
                messages=summary_prompt,
                max_tokens=300,
                temperature=0.3  # Lower temperature for more focused summary
            )
            
            return summary_response.choices[0].message.content
        except Exception as e:
            console.print(f"[yellow]Error generating enhanced summary: {e}[/yellow]")
            # Fallback to simple summary
            return f"Action '{self.working_memory.selected_action.get('action_name', 'unknown')}' executed. Result type: {type(result).__name__}"


    def _process_input(self, user_input):
        """Process user input, handling multimodal content if needed."""
        # Check for clipboard paste command
        if isinstance(user_input, str) and user_input.strip() == "/paste":
            return self._paste_from_clipboard()
            
        # Process image URLs in text
        if isinstance(user_input, str):
            import re
            image_urls = re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)', user_input)
            if not image_urls:
                return user_input
                
            # Create multimodal format
            multimodal = []
            text_content = user_input
            for url in image_urls:
                text_content = text_content.replace(url, '')
            
            text_content = text_content.strip()
            if text_content:
                multimodal.append({"type": "text", "text": text_content})
                
            for url in image_urls:
                multimodal.append({"type": "image_url", "image_url": {"url": url}})
                
            return multimodal
            
        return user_input
        
    def _paste_from_clipboard(self):
        """Paste image from clipboard and convert to base64 for the model"""
        try:
            from PIL import ImageGrab
            import pyperclip
            import base64
            from io import BytesIO
            
            # Try to get image from clipboard
            image = ImageGrab.grabclipboard()
            
            if image is None:
                # If no image, try to get text
                text = pyperclip.paste()
                if text:
                    return text
                else:
                    return "No image or text found in clipboard."
            
            # Process the image
            console.print("[cyan]Image found in clipboard[/cyan]")
            
            # Convert to base64 for the model
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Create multimodal message
            multimodal = [
                {"type": "text", "text": "Image pasted from clipboard:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
            
            return multimodal
        except Exception as e:
            console.print(f"[red]Error pasting from clipboard: {str(e)}[/red]")
            return f"Error pasting from clipboard: {str(e)}"
            
    def _should_use_advanced_capabilities(self, user_input):
        """Determine if we should use advanced capabilities from atomic_agent."""
        if not user_input:
            return False
            
        # Check for direct commands that should use advanced capabilities
        if isinstance(user_input, str):
            direct_commands = [
                "/search", "/code", "/execute", "/python", 
                "/weather", "/memory", "/scout", "/orchestrate",
                "search the web", "execute this code", "run this python",
                "generate multiple solutions", "analyze this image"
            ]
            
            for cmd in direct_commands:
                if cmd in user_input.lower():
                    return True
                    
        # Always use advanced capabilities for multimodal input
        if isinstance(user_input, list):
            return True
            
        # Check if the task seems complex
        if isinstance(user_input, str):
            complex_indicators = [
                "multiple", "several", "various", "different", "steps",
                "complex", "complicated", "detailed", "comprehensive",
                "analyze and", "research and", "implement and"
            ]
            
            word_count = len(user_input.split())
            sentence_count = len(re.split(r'[.!?]+', user_input))
            
            # Task is complex if it's long or has complexity indicators
            if (word_count > 50 or 
                sentence_count > 3 or 
                any(indicator in user_input.lower() for indicator in complex_indicators)):
                return True
                
        return False
        
    def _handle_with_advanced_capabilities(self, user_input):
        """Handle the request using advanced capabilities from atomic_agent."""
        try:
            # For multimodal content, use the agent orchestrator
            if isinstance(user_input, list):
                # Find a scout specialized in image processing if there's an image
                has_image = any(item.get("type") == "image_url" for item in user_input)
                
                if has_image:
                    # Extract text content
                    text_content = ""
                    for item in user_input:
                        if item.get("type") == "text":
                            text_content += item.get("text", "") + " "
                    
                    # Create a task for the creative scout
                    for scout_id, scout in self.agent_orchestrator.scouts.items():
                        if scout.specialization == "creative" and scout.is_available.is_set():
                            task_id = scout.add_task(scout.perform_task, 
                                                    f"Analyze this image and respond to: {text_content}",
                                                    {"multimodal": True, "image_urls": [
                                                        item["image_url"]["url"] for item in user_input 
                                                        if item.get("type") == "image_url"
                                                    ]})
                            
                            # Wait for completion
                            max_wait_time = 60  # seconds
                            start_time = time.time()
                            
                            while True:
                                result = scout.get_result(task_id)
                                if result["status"] in ["completed", "failed"]:
                                    break
                                    
                                if time.time() - start_time > max_wait_time:
                                    return "I'm still processing the image. This is taking longer than expected."
                                
                                time.sleep(0.5)
                            
                            if result["status"] == "completed":
                                return result.get("result", {}).get("solution", 
                                       "I've analyzed the image but couldn't generate a proper response.")
            
            # For text that indicates web search
            if isinstance(user_input, str) and ("search" in user_input.lower() or "find information" in user_input.lower()):
                # Use the research scout
                for scout_id, scout in self.agent_orchestrator.scouts.items():
                    if scout.specialization == "research" and scout.is_available.is_set():
                        task_id = scout.add_task(scout.perform_task, user_input, {"web_search": True})
                        
                        # Wait for completion
                        max_wait_time = 60  # seconds
                        start_time = time.time()
                        
                        while True:
                            result = scout.get_result(task_id)
                            if result["status"] in ["completed", "failed"]:
                                break
                                
                            if time.time() - start_time > max_wait_time:
                                return "I'm still researching. This is taking longer than expected."
                            
                            time.sleep(0.5)
                        
                        if result["status"] == "completed":
                            return result.get("result", {}).get("solution", 
                                   "I've completed the research but couldn't generate a proper response.")
            
            # For code execution requests
            if isinstance(user_input, str) and ("execute" in user_input.lower() or "run" in user_input.lower() or "python" in user_input.lower()):
                # Extract code if present
                code_blocks = extract_python_code(user_input)
                
                if code_blocks:
                    # Execute the first code block
                    result = self.tool_registry._execute_python(code_blocks[0])
                    
                    if result.get("success", False):
                        response = "Code executed successfully.\n\n"
                        if result.get("stdout"):
                            response += f"Output:\n{result['stdout']}\n\n"
                        if result.get("result"):
                            response += f"Result: {result['result']}"
                        return response
                    else:
                        return f"Error executing code: {result.get('error', 'Unknown error')}"
                        
            # For complex tasks, use task decomposition and parallel execution
            if self._is_complex_task(user_input):
                # Use the planning scout to decompose the task
                for scout_id, scout in self.agent_orchestrator.scouts.items():
                    if scout.specialization == "planning" and scout.is_available.is_set():
                        task_id = scout.add_task(scout.perform_task, f"Decompose this task into subtasks: {user_input}", 
                                               {"decomposition": True})
                        
                        # Wait for completion
                        max_wait_time = 30  # seconds
                        start_time = time.time()
                        
                        while True:
                            result = scout.get_result(task_id)
                            if result["status"] in ["completed", "failed"]:
                                break
                                
                            if time.time() - start_time > max_wait_time:
                                # If taking too long, proceed without decomposition
                                return None
                            
                            time.sleep(0.5)
                        
                        if result["status"] == "completed" and "solution" in result.get("result", {}):
                            # Try to extract subtasks from the solution
                            solution = result["result"]["solution"]
                            subtasks = self._extract_subtasks(solution)
                            
                            if subtasks and len(subtasks) > 1:
                                # Execute subtasks in parallel
                                orchestration_result = self.tool_registry._orchestrate_tasks(
                                    main_task=user_input,
                                    subtasks=subtasks,
                                    context={},
                                    agent=self
                                )
                                
                                if orchestration_result.get("success", False):
                                    # Generate a summary of the results
                                    summary = "I've completed the task by breaking it down into steps:\n\n"
                                    for i, result in enumerate(orchestration_result.get("successful_results", [])):
                                        summary += f"{i+1}. {result.get('subtask')}: Completed\n"
                                    
                                    summary += "\n" + self._generate_orchestration_summary(orchestration_result)
                                    return summary
            
            # If we reach here, no advanced handling was done
            return None
            
        except Exception as e:
            console.print(f"[red]Error in advanced capabilities: {str(e)}[/red]")
            console.print(traceback.format_exc())
            return None
            
    def _is_complex_task(self, task):
        """Determine if a task is complex and should be decomposed"""
        if not isinstance(task, str):
            return False
            
        # Simple heuristics for complexity
        word_count = len(task.split())
        sentence_count = len(re.split(r'[.!?]+', task))
        question_count = task.count('?')
        
        # Check for indicators of complexity
        complexity_indicators = [
            "multiple", "several", "various", "different", "steps",
            "complex", "complicated", "detailed", "comprehensive",
            "analyze and", "research and", "implement and"
        ]
        
        has_complexity_indicators = any(indicator in task.lower() for indicator in complexity_indicators)
        
        # Task is complex if it's long or has complexity indicators
        return (word_count > 50 or 
                sentence_count > 3 or 
                question_count > 1 or 
                has_complexity_indicators)
                
    def _extract_subtasks(self, text):
        """Extract subtasks from planning scout output"""
        subtasks = []
        
        # Look for numbered lists
        import re
        numbered_items = re.findall(r'\d+\.\s+(.*?)(?=\n\d+\.|\n\n|$)', text, re.DOTALL)
        if numbered_items and len(numbered_items) > 1:
            return [item.strip() for item in numbered_items]
            
        # Look for bullet points
        bullet_items = re.findall(r'[\*\-\•]\s+(.*?)(?=\n[\*\-\•]|\n\n|$)', text, re.DOTALL)
        if bullet_items and len(bullet_items) > 1:
            return [item.strip() for item in bullet_items]
            
        # Look for "Subtask X:" format
        subtask_items = re.findall(r'(?:Subtask|Task)\s+\d+:?\s+(.*?)(?=\n(?:Subtask|Task)|$)', text, re.DOTALL)
        if subtask_items and len(subtask_items) > 1:
            return [item.strip() for item in subtask_items]
            
        # If no structured format found, try to split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if sentences and len(sentences) > 2:
            # Filter out very short sentences and introductory/concluding sentences
            filtered_sentences = [s for s in sentences if len(s.split()) > 5]
            if len(filtered_sentences) > 1:
                return filtered_sentences
                
        return subtasks
        
    def _generate_action_summary(self, result):
        """Generate a more sophisticated summary of action results using the LLM."""
        if not result:
            return "No result was returned from the action."
            
        # For simple success/failure cases, use a template
        if isinstance(result, dict):
            if result.get('success'):
                summary = f"Action '{self.working_memory.selected_action.get('action_name', 'unknown')}' executed successfully."
                
                # Include stdout if available and not too long
                stdout = result.get('stdout')
                if stdout and len(stdout) < 200:
                    summary += f" Output: {stdout.strip()}"
                    
                # Include result if available and not too long
                result_value = result.get('result')
                if result_value and isinstance(result_value, str) and len(result_value) < 200:
                    summary += f" Result: {result_value}"
                    
                return summary
            elif 'error' in result:
                return f"Action '{self.working_memory.selected_action.get('action_name', 'unknown')}' failed. Error: {result['error']}"
                
        # For complex results, use the LLM to generate a summary
        try:
            # Truncate result if too large
            result_str = str(result)
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "... (truncated)"
                
            summary_prompt = [
                {"role": "system", "content": "You are an AI assistant that summarizes complex action results concisely."},
                {"role": "user", "content": f"Summarize this action result in 2-3 sentences, highlighting the most important information:\n\n{result_str}"}
            ]
            
            summary_response = self.client.chat.completions.create(
                model=self.model,
                messages=summary_prompt,
                max_tokens=200
            )
            
            return summary_response.choices[0].message.content
        except Exception as e:
            console.print(f"[yellow]Error generating summary: {e}[/yellow]")
            # Fallback to simple summary
            return f"Action '{self.working_memory.selected_action.get('action_name', 'unknown')}' executed. Result type: {type(result).__name__}"
            
    def _generate_orchestration_summary(self, orchestration_result):
        """Generate a summary of orchestration results."""
        try:
            # Extract key information
            successful_count = len(orchestration_result.get("successful_results", []))
            failed_count = len(orchestration_result.get("failed_results", []))
            total_count = successful_count + failed_count
            
            # Create a prompt for the LLM to generate a summary
            summary_prompt = [
                {"role": "system", "content": "You are an AI assistant that summarizes complex task results concisely."},
                {"role": "user", "content": f"Summarize the results of this orchestrated task execution. {successful_count} out of {total_count} subtasks completed successfully.\n\nResults: {json.dumps(orchestration_result, indent=2)[:1500]}"}
            ]
            
            summary_response = self.client.chat.completions.create(
                model=self.model,
                messages=summary_prompt,
                max_tokens=300
            )
            
            return summary_response.choices[0].message.content
        except Exception as e:
            console.print(f"[yellow]Error generating orchestration summary: {e}[/yellow]")
            # Fallback to simple summary
            return f"Completed {successful_count} out of {total_count} subtasks."
            
    def _process_web_search_result(self, result, original_query):
        """Process web search results, summarize them, and potentially trigger follow-up actions."""
        try:
            # Extract search results
            search_results = result.get("results", [])
            if not search_results:
                return {"success": True, "summary": "The web search did not return any results."}
                
            # Log the number of results found
            console.print(f"[cyan]Found {len(search_results)} search results for query: {original_query}[/cyan]")
            
            # Prepare data for summarization
            search_data = []
            for i, item in enumerate(search_results[:5]):  # Limit to first 5 results
                # Handle case where item might be a string instead of a dict
                if isinstance(item, dict):
                    title = item.get("title", "No title")
                    snippet = item.get("snippet", "No snippet")
                    url = item.get("link", "No URL")
                else:
                    # If the item is a string, use it as both title and snippet
                    title = "Result"
                    snippet = str(item)
                    url = "No URL"
                
                search_data.append(f"Result {i+1}:\nTitle: {title}\nSnippet: {snippet}\nURL: {url}\n")
                
            search_content = "\n".join(search_data)
            
            # Generate a comprehensive summary using the LLM
            summary_prompt = [
                {"role": "system", "content": "You are an AI research assistant that summarizes web search results accurately and comprehensively."},
                {"role": "user", "content": f"Summarize these web search results for the query: '{original_query}'\n\n{search_content}\n\nProvide a comprehensive summary that captures the key information from all results. If the results contain factual information, include it. If there are different perspectives, mention them. If additional research is needed, suggest specific follow-up queries."}
            ]
            
            summary_response = self.client.chat.completions.create(
                model=self.model,
                messages=summary_prompt,
                max_tokens=800,
                temperature=0.3  # Lower temperature for more factual summary
            )
            
            summary = summary_response.choices[0].message.content
            
            # Analyze if follow-up actions are needed
            followup_prompt = [
                {"role": "system", "content": "You are an AI research assistant that identifies necessary follow-up actions based on search results."},
                {"role": "user", "content": f"Based on these search results for '{original_query}':\n\n{search_content}\n\nDetermine if any follow-up actions are needed. Return a JSON object with these fields:\n- needs_followup: boolean\n- followup_type: string (one of: 'additional_search', 'fact_check', 'read_url', 'none')\n- followup_query: string (if applicable)\n- followup_url: string (if applicable)\n- reasoning: string (brief explanation)"}
            ]
            
            followup_response = self.client.chat.completions.create(
                model=self.model,
                messages=followup_prompt,
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.2
            )
            
            followup_data = json.loads(followup_response.choices[0].message.content)
            
            # Execute follow-up action if needed
            if followup_data.get("needs_followup", False) and followup_data.get("followup_type") != "none":
                followup_type = followup_data.get("followup_type")
                console.print(f"[cyan]Executing follow-up action: {followup_type}[/cyan]")
                
                if followup_type == "additional_search":
                    followup_query = followup_data.get("followup_query")
                    if followup_query:
                        followup_result = self.tool_registry.call_function("web_search", {"query": followup_query})
                        if followup_result.get("success"):
                            # Add a note about the follow-up search to the summary
                            additional_results = followup_result.get("results", [])
                            if additional_results:
                                additional_summary = f"\n\n**Additional Information from Follow-up Search**\nI performed a follow-up search for '{followup_query}' and found:\n"
                                for i, item in enumerate(additional_results[:3]):
                                    # Handle case where item might be a string instead of a dict
                                    if isinstance(item, dict):
                                        title = item.get("title", "No title")
                                        snippet = item.get("snippet", "No snippet")
                                    else:
                                        # If item is a string, use it as both title and snippet
                                        title = "Result"
                                        snippet = str(item)
                                    additional_summary += f"- {title}: {snippet}\n"
                                summary += additional_summary
                
                elif followup_type == "fact_check":
                    statement = followup_data.get("followup_query")
                    if statement:
                        fact_check_result = self.tool_registry.call_function("fact_check", {"query": statement})
                        if fact_check_result.get("success"):
                            summary += f"\n\n**Fact Check**\n{fact_check_result.get('result', 'No result')}"
                
                elif followup_type == "read_url":
                    url = followup_data.get("followup_url")
                    if url:
                        read_result = self.tool_registry.call_function("read", {"url": url})
                        if read_result.get("success"):
                            # Summarize the content from the URL
                            url_content = read_result.get("content", "")
                            if url_content:
                                url_summary_prompt = [
                                    {"role": "system", "content": "You summarize web content concisely."},
                                    {"role": "user", "content": f"Summarize this content from {url} in 2-3 sentences:\n\n{url_content[:5000]}"}
                                ]
                                url_summary_response = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=url_summary_prompt,
                                    max_tokens=300
                                )
                                summary += f"\n\n**Additional Details from {url}**\n{url_summary_response.choices[0].message.content}"
            
            # Store the comprehensive result in memory for future reference
            if hasattr(self, 'memory'):
                self.memory.add_memory(
                    f"Web search for '{original_query}': {summary[:200]}...",
                    {
                        "type": "web_search_result",
                        "query": original_query,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
                
            return {"success": True, "summary": summary, "followup_data": followup_data}
            
        except Exception as e:
            console.print(f"[red]Error processing web search result: {str(e)}[/red]")
            console.print(traceback.format_exc())
            return {"success": False, "error": str(e), "summary": "I encountered an error while processing the search results."}
    
    def run_interactive(self):
        """Starts the interactive command-line loop."""
        console.print(Panel.fit(
            "[bold blue]CoALA Agent CLI[/bold blue]\n"
            "Based on Cognitive Architectures for Language Agents.\n"
            "Type your requests or commands.\n"
            "Type [bold]'/paste'[/bold] to paste an image from clipboard.\n"
            "Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to end.",
            title="Welcome"
        ))

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
                if user_input.lower() in ["exit", "quit"]:
                    console.print("[yellow]Exiting...[/yellow]")
                    break

                with console.status("[bold blue]Agent Processing...[/bold blue]", spinner="dots"):
                    response = self.decision_cycle(user_input)

                console.print("\n[bold purple]Agent[/bold purple]:")
                console.print(Markdown(str(response))) # Ensure response is string

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Exiting...[/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]An error occurred: {str(e)}[/red]")
                console.print(traceback.format_exc())

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run the CoALA-based Language Agent")
    parser.add_argument("--model", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", help="LLM model to use")
    parser.add_argument("--scouts", type=int, default=5, help="Number of scout agents to initialize")
    parser.add_argument("--logprobs", action="store_true", help="Enable returning logprobs for confidence analysis")
    parser.add_argument("--advanced", action="store_true", help="Enable advanced capabilities", default=True)
    args = parser.parse_args()

    try:
        # Check for required dependencies
        try:
            import pyperclip
            import PIL
        except ImportError:
            console.print("[yellow]Installing required dependencies for advanced features...[/yellow]")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyperclip", "pillow"])
            
        # Initialize the agent with the specified number of scouts
        console.print(f"[cyan]Initializing CoALA Agent with {args.scouts} scout agents...[/cyan]")
        agent = CoALAAgent(model=args.model)
        
        # Enable advanced features if requested
        if args.advanced:
            console.print("[green]Advanced capabilities enabled[/green]")
            
        # Enable logprobs if requested
        if args.logprobs:
            console.print("[cyan]Logprobs enabled for confidence analysis[/cyan]")
            
        agent.run_interactive()
    except ValueError as e:
        console.print(f"[red]Initialization Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Runtime Error: {e}[/red]")
        console.print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
