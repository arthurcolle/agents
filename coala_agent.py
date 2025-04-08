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

        # Define the core system prompt based on CoALA principles
        self.system_prompt = {
            "role": "system",
            "content": (
                "You are a CoALA-based language agent designed for general computer usage. "
                "Your goal is to assist the user by understanding their requests, reasoning about the steps needed, "
                "and utilizing available tools (actions) to interact with the environment (files, code, web).\n"
                "Follow this decision cycle:\n"
                "1. **Observe:** Receive user input and environmental feedback.\n"
                "2. **Orient (Plan):** Analyze the current state (working memory), retrieve relevant knowledge (long-term memory), "
                "   and reason to propose potential actions (internal or external).\n"
                "3. **Decide:** Evaluate proposed actions and select the best one.\n"
                "4. **Act:** Execute the selected action (e.g., call a tool, update memory, respond to user).\n"
                "Use JSON format for reasoning steps and action selection when possible.\n"
                "Available tools can be listed. Use them precisely as defined."
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

        # Call tool using the registry, passing self for context if needed
        result = self.tool_registry.call_function(action_name, arguments, agent=self)

        # Log tool call and result
        # Check if the result indicates a direct response was intended
        if action_name == "respond_to_user" and result.get("success"):
            response_text = result.get("response_sent", "Action completed.")
            self.add_log({"role": "assistant", "content": response_text})
            # Return the text intended for the user
            return response_text
        else:
            # Log non-response tool calls
            self.add_log({
                "role": "tool", # Use standard 'tool' role
                "name": action_name,
                "content": json.dumps(result, indent=2), # Log result as content
                "tool_call_id": str(uuid.uuid4()) # Generate a dummy ID
            })
            # Add result as an observation in working memory
            self.working_memory.add_observation(result, source=f"action:{action_name}")
            return result # Return raw result for potential further processing

    def decision_cycle(self, user_input: Optional[str] = None):
        """Runs one cycle of the CoALA Observe-Orient-Decide-Act loop."""
        self.working_memory.clear_cycle_state()

        # 1. Observe
        if user_input:
            # Process multimodal input if needed
            processed_input = self._process_input(user_input)
            self.working_memory.add_observation(processed_input, source="user")
            self.add_log({"role": "user", "content": processed_input if isinstance(processed_input, str) else str(processed_input)})
            # Simple goal setting for now
            if not self.working_memory.current_goal:
                 self.working_memory.current_goal = processed_input if isinstance(processed_input, str) else "Process multimodal input"
            
            # Store in memory for future reference
            if hasattr(self, 'memory'):
                if isinstance(processed_input, str):
                    self.memory.add_memory(processed_input, {"type": "text", "role": "user"})
                elif isinstance(processed_input, list):  # Multimodal content
                    for item in processed_input:
                        if item.get("type") == "text":
                            self.memory.add_memory(item.get("text", ""), {"type": "text", "role": "user"})
                        elif item.get("type") == "image_url" and "url" in item.get("image_url", {}):
                            self.memory.add_memory(item["image_url"]["url"], {"type": "image", "role": "user"})

        # Check if we should use advanced capabilities from atomic_agent
        if self._should_use_advanced_capabilities(user_input):
            result = self._handle_with_advanced_capabilities(user_input)
            if result:
                return result

        # 2. Orient (Plan)
        proposed_actions = self.plan_step()
        if not proposed_actions:
            # If planning fails, maybe try a simple response
            console.print("[yellow]Planning failed, attempting simple response.[/yellow]")
            self.working_memory.selected_action = {"action_name": "respond_to_user", "arguments": {"response_text": "I'm unsure how to proceed. Can you clarify?"}}
        else:
            # 3. Decide
            self.decide_step()

        # 4. Act
        result = self.act_step()

        # Check if the action was a direct response to the user via the tool
        if self.working_memory.selected_action and self.working_memory.selected_action.get("action_name") == "respond_to_user":
            # The result itself is the response text in this case
            return result
        else:
            # If another action was taken, potentially loop again or generate a summary response
            console.print(f"[bold magenta]Action Result:[/bold magenta]\n{json.dumps(result, indent=2)}")
            
            # Use more sophisticated summarization with the LLM
            summary_response = self._generate_action_summary(result)

            self.add_log({"role": "assistant", "content": summary_response})
            return summary_response


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
