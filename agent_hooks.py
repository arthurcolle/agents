import asyncio
import random
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel

class CLIAgentHooks:
    """
    Hooks for monitoring and debugging the CLI agent lifecycle
    """
    def __init__(self, console: Console, display_name: str = "CLI Agent"):
        self.event_counter = 0
        self.display_name = display_name
        self.console = console
        self.events = []
    
    def on_start(self, agent_name: str) -> None:
        """Called when the agent starts processing a request"""
        self.event_counter += 1
        event = f"({self.display_name}) {self.event_counter}: Agent started"
        self.events.append(event)
        self.console.print(f"[dim blue]{event}[/dim blue]")
    
    def on_end(self, agent_name: str, output: Any) -> None:
        """Called when the agent completes processing"""
        self.event_counter += 1
        event = f"({self.display_name}) {self.event_counter}: Agent completed"
        self.events.append(event)
        self.console.print(f"[dim blue]{event}[/dim blue]")
    
    def on_tool_start(self, agent_name: str, tool_name: str, arguments: Dict) -> None:
        """Called when a tool is about to be executed"""
        self.event_counter += 1
        event = f"({self.display_name}) {self.event_counter}: Starting tool {tool_name}"
        self.events.append(event)
        self.console.print(f"[dim blue]{event}[/dim blue]")
    
    def on_tool_end(self, agent_name: str, tool_name: str, result: Dict) -> None:
        """Called when a tool has completed execution"""
        self.event_counter += 1
        success = result.get("success", False)
        status = "[green]success[/green]" if success else "[red]failure[/red]"
        event = f"({self.display_name}) {self.event_counter}: Tool {tool_name} completed with {status}"
        self.events.append(event)
        self.console.print(f"[dim blue]{event}[/dim blue]")
    
    def on_error(self, agent_name: str, error: Exception) -> None:
        """Called when an error occurs during agent execution"""
        self.event_counter += 1
        event = f"({self.display_name}) {self.event_counter}: Error: {str(error)}"
        self.events.append(event)
        self.console.print(f"[dim red]{event}[/dim red]")
    
    def get_summary(self) -> str:
        """Get a summary of all events that occurred during agent execution"""
        return "\n".join(self.events)
    
    def display_summary(self) -> None:
        """Display a summary panel of all events"""
        self.console.print(Panel.fit(
            "\n".join(self.events),
            title=f"{self.display_name} Execution Summary",
            border_style="blue"
        ))
