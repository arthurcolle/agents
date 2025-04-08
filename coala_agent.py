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
    from atomic_agent import AgentOrchestrator, ScoutAgent, ToolRegistry as AtomicToolRegistry
    # Rename imported ToolRegistry to avoid conflict if needed later
except ImportError as e:
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
console = Console() # Initialize console early

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

        # Initialize the Agent Orchestrator
        # TODO: Make num_scouts configurable
        self.agent_orchestrator = AgentOrchestrator(num_scouts=3, model=model)
        console.print(f"[green]Initialized Agent Orchestrator with {len(self.agent_orchestrator.scouts)} scouts.[/green]")

        # Use the full ToolRegistry from atomic_agent
        self.tool_registry = AtomicToolRegistry()
        # Pass self (CoALAAgent instance) to the registry if needed by tools
        # For now, tools using inspect.stack() might need adaptation or CoALAAgent needs necessary attributes
        self.tool_registry.agent_instance = self # Provide access for tools needing agent state

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
            self.working_memory.add_observation(user_input, source="user")
            self.add_log({"role": "user", "content": user_input})
            # Simple goal setting for now
            if not self.working_memory.current_goal:
                 self.working_memory.current_goal = user_input

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
            # We might need another LLM call here to summarize the result for the user
            # For simplicity now, we'll just indicate the action was done.
            summary_response = f"Action '{self.working_memory.selected_action.get('action_name', 'unknown')}' executed."
            if isinstance(result, dict) and result.get('success'):
                 summary_response += " Status: Success."
                 # Optionally include brief output if available
                 stdout = result.get('stdout')
                 if stdout and len(stdout) < 100:
                     summary_response += f" Output: {stdout.strip()}"
            elif isinstance(result, dict) and 'error' in result:
                 summary_response += f" Status: Failed. Error: {result['error']}"

            self.add_log({"role": "assistant", "content": summary_response})
            return summary_response


    def run_interactive(self):
        """Starts the interactive command-line loop."""
        console.print(Panel.fit(
            "[bold blue]CoALA Agent CLI[/bold blue]\n"
            "Based on Cognitive Architectures for Language Agents.\n"
            "Type your requests or commands.\n"
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
    args = parser.parse_args()

    try:
        agent = CoALAAgent(model=args.model)
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
