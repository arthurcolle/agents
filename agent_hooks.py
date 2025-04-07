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
        self.performance_metrics = {
            "tool_calls": 0,
            "errors": 0,
            "total_time": 0,
            "tool_times": {}
        }
        self.start_time = None
    
    def on_start(self, agent_name: str) -> None:
        """Called when the agent starts processing a request"""
        import time
        self.start_time = time.time()
        self.event_counter += 1
        event = f"({self.display_name}) {self.event_counter}: Agent started"
        self.events.append(event)
        self.console.print(f"[dim blue]{event}[/dim blue]")
    
    def on_end(self, agent_name: str, output: Any) -> None:
        """Called when the agent completes processing"""
        import time
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.performance_metrics["total_time"] = elapsed
            
        self.event_counter += 1
        event = f"({self.display_name}) {self.event_counter}: Agent completed in {self.performance_metrics['total_time']:.2f}s"
        self.events.append(event)
        self.console.print(f"[dim blue]{event}[/dim blue]")
    
    def on_tool_start(self, agent_name: str, tool_name: str, arguments: Dict) -> None:
        """Called when a tool is about to be executed"""
        import time
        self.performance_metrics["tool_calls"] += 1
        if tool_name not in self.performance_metrics["tool_times"]:
            self.performance_metrics["tool_times"][tool_name] = {"count": 0, "total_time": 0}
        self.performance_metrics["tool_times"][tool_name]["count"] += 1
        self.performance_metrics["tool_times"][tool_name]["start_time"] = time.time()
        
        self.event_counter += 1
        event = f"({self.display_name}) {self.event_counter}: Starting tool {tool_name}"
        self.events.append(event)
        self.console.print(f"[dim blue]{event}[/dim blue]")
    
    def on_tool_end(self, agent_name: str, tool_name: str, result: Dict) -> None:
        """Called when a tool has completed execution"""
        import time
        if tool_name in self.performance_metrics["tool_times"] and "start_time" in self.performance_metrics["tool_times"][tool_name]:
            elapsed = time.time() - self.performance_metrics["tool_times"][tool_name]["start_time"]
            self.performance_metrics["tool_times"][tool_name]["total_time"] += elapsed
            del self.performance_metrics["tool_times"][tool_name]["start_time"]
        
        self.event_counter += 1
        success = result.get("success", False)
        status = "[green]success[/green]" if success else "[red]failure[/red]"
        event = f"({self.display_name}) {self.event_counter}: Tool {tool_name} completed with {status}"
        self.events.append(event)
        self.console.print(f"[dim blue]{event}[/dim blue]")
    
    def on_error(self, agent_name: str, error: Exception) -> None:
        """Called when an error occurs during agent execution"""
        self.performance_metrics["errors"] += 1
        self.event_counter += 1
        event = f"({self.display_name}) {self.event_counter}: Error: {str(error)}"
        self.events.append(event)
        self.console.print(f"[dim red]{event}[/dim red]")
    
    def get_summary(self) -> str:
        """Get a summary of all events that occurred during agent execution"""
        return "\n".join(self.events)
    
    def display_summary(self) -> None:
        """Display a summary panel of all events and performance metrics"""
        from rich.table import Table
        
        # Create performance metrics table
        metrics_table = Table(title="Performance Metrics", show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="dim")
        metrics_table.add_column("Value")
        
        metrics_table.add_row("Total Time", f"{self.performance_metrics['total_time']:.2f}s")
        metrics_table.add_row("Tool Calls", str(self.performance_metrics['tool_calls']))
        metrics_table.add_row("Errors", str(self.performance_metrics['errors']))
        
        # Add tool-specific metrics
        if self.performance_metrics['tool_calls'] > 0:
            tool_table = Table(title="Tool Performance", show_header=True, header_style="bold cyan")
            tool_table.add_column("Tool")
            tool_table.add_column("Calls")
            tool_table.add_column("Total Time")
            tool_table.add_column("Avg Time")
            
            for tool, stats in self.performance_metrics["tool_times"].items():
                if "count" in stats and stats["count"] > 0:
                    avg_time = stats["total_time"] / stats["count"]
                    tool_table.add_row(
                        tool, 
                        str(stats["count"]), 
                        f"{stats['total_time']:.2f}s",
                        f"{avg_time:.2f}s"
                    )
        
        # Display events and metrics
        self.console.print(Panel.fit(
            "\n".join(self.events),
            title=f"{self.display_name} Execution Log",
            border_style="blue"
        ))
        
        self.console.print(metrics_table)
        
        if self.performance_metrics['tool_calls'] > 0:
            self.console.print(tool_table)
