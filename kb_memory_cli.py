#!/usr/bin/env python3
"""
Knowledge Base Memory CLI - A command-line interface for interacting with
the distributed memory system and knowledge base agents.
"""

import os
import sys
import logging
import argparse
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import print as rprint

# Import distributed memory system
from distributed_memory_system import DistributedMemorySystem, KnowledgeFragment
# We'll create our own simplified connectors since the imported ones require async event loop
# from knowledge_base_dispatcher import dispatcher 
# from kb_agent_connector import connector

# Simplified mock connector for demo purposes
class MockConnector:
    def __init__(self):
        self.knowledge_bases = {}
        
    def get_available_knowledge_bases(self):
        """Get list of available knowledge bases"""
        return [
            {"name": "astronomy", "agent_id": "kb_astronomy"},
            {"name": "physics", "agent_id": "kb_physics"},
            {"name": "computers", "agent_id": "kb_computers"},
            {"name": "history", "agent_id": "kb_history"},
            {"name": "mathematics", "agent_id": "kb_mathematics"},
        ]
    
    async def dispatch_to_kb_agent(self, kb_name, command):
        """Mock dispatch to KB agent"""
        return {"success": True, "data": f"Executed '{command}' on {kb_name}"}
    
    async def dispatch_query_to_all_kbs(self, query):
        """Mock query to all KBs"""
        return {
            "success": True,
            "results": [
                {"content": f"Result for '{query}' from astronomy KB", "source_kb": "astronomy"},
                {"content": f"Result for '{query}' from physics KB", "source_kb": "physics"},
                {"content": f"Result for '{query}' from computers KB", "source_kb": "computers"},
            ]
        }

# Create a mock connector
connector = MockConnector()

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("kb-memory-cli")

# Rich console for better formatting
console = Console()

class KnowledgeBaseMemoryCLI:
    """
    Command-line interface for interacting with the distributed memory system
    and knowledge base agents.
    """
    
    def __init__(self, agent_id: str = "memory-cli", 
                dimensions: int = 256,
                memory_path: Optional[str] = None,
                max_history: int = 100):
        """
        Initialize the KB Memory CLI.
        
        Args:
            agent_id: Unique identifier for this agent
            dimensions: Vector dimensionality for knowledge representation
            memory_path: Path for persisting memory
            max_history: Maximum command history to keep
        """
        self.agent_id = agent_id
        self.dimensions = dimensions
        self.memory_path = memory_path
        self.max_history = max_history
        
        # Initialize components
        self.memory_system = DistributedMemorySystem(
            agent_id=agent_id,
            dimensions=dimensions,
            memory_path=memory_path,
            auto_verification=True,
            knowledge_propagation_enabled=True
        )
        
        # Connect to KB agents
        self.kb_connector = connector
        
        # Command history
        self.command_history = []
        
        # Preload system agents
        self.system_agents = {}
        self._initialize_system_agents()
        
    def _initialize_system_agents(self):
        """Initialize system agents for various knowledge domains"""
        # Get the list of available knowledge bases
        kb_list = self.kb_connector.get_available_knowledge_bases()
        console.print(f"[bold green]Connected to {len(kb_list)} knowledge bases[/]")
        
        # Map domains to knowledge bases
        domains = set()
        for kb in kb_list:
            kb_name = kb["name"]
            # Extract domains from KB name
            domain = kb_name.split('_')[0] if '_' in kb_name else kb_name
            domains.add(domain)
            
            # Add to system agents by domain
            if domain not in self.system_agents:
                self.system_agents[domain] = []
            self.system_agents[domain].append(kb_name)
        
        console.print(f"[bold]Available knowledge domains:[/] {', '.join(domains)}")
        
    async def start(self):
        """Start the CLI interface"""
        console.print(Panel(
            "[bold blue]Knowledge Base Memory CLI[/]\n"
            "Type [bold green]help[/] to see available commands\n"
            "Type [bold green]exit[/] to quit",
            title="Welcome",
            expand=False
        ))
        
        # Main command loop
        running = True
        while running:
            try:
                command = Prompt.ask("[bold blue]kb-memory[/]")
                self.command_history.append(command)
                if len(self.command_history) > self.max_history:
                    self.command_history.pop(0)
                
                if command.lower() == "exit":
                    running = False
                    continue
                
                # Process the command
                await self.process_command(command)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Operation cancelled[/]")
            except Exception as e:
                console.print(f"[bold red]Error:[/] {str(e)}")
                logger.exception("Error processing command")
    
    async def process_command(self, command: str):
        """Process a user command"""
        cmd_parts = command.strip().split(" ", 1)
        cmd = cmd_parts[0].lower()
        args = cmd_parts[1] if len(cmd_parts) > 1 else ""
        
        if cmd == "help":
            self.show_help()
        elif cmd == "query" or cmd == "q":
            await self.query_knowledge(args)
        elif cmd == "add" or cmd == "a":
            await self.add_knowledge(args)
        elif cmd == "verify" or cmd == "v":
            await self.verify_knowledge(args)
        elif cmd == "dispute" or cmd == "d":
            await self.dispute_knowledge(args)
        elif cmd == "list" or cmd == "ls":
            await self.list_knowledge(args)
        elif cmd == "kb":
            await self.kb_command(args)
        elif cmd == "search" or cmd == "s":
            await self.search_kb(args)
        elif cmd == "consensus" or cmd == "c":
            await self.request_consensus(args)
        elif cmd == "stats":
            await self.show_stats()
        elif cmd == "save":
            self.save_memory()
        elif cmd == "load":
            self.load_memory()
        else:
            console.print("[yellow]Unknown command. Type 'help' for available commands.[/]")
    
    def show_help(self):
        """Show help information"""
        help_table = Table(title="Available Commands")
        help_table.add_column("Command", style="green")
        help_table.add_column("Description")
        help_table.add_column("Usage", style="blue")
        
        commands = [
            ("help", "Show this help message", "help"),
            ("query (q)", "Query distributed memory and knowledge bases", "query <question>"),
            ("add (a)", "Add new knowledge fragment", "add <content>"),
            ("verify (v)", "Verify a knowledge fragment", "verify <fragment_id>"),
            ("dispute (d)", "Dispute a knowledge fragment", "dispute <fragment_id> <reason>"),
            ("list (ls)", "List knowledge fragments", "list [verified|disputed|all]"),
            ("kb", "Execute command on knowledge base", "kb <kb_name> <command>"),
            ("search (s)", "Search across all knowledge bases", "search <query>"),
            ("consensus (c)", "Request consensus on knowledge", "consensus <fragment_id>"),
            ("stats", "Show system statistics", "stats"),
            ("save", "Save memory to disk", "save"),
            ("load", "Load memory from disk", "load"),
            ("exit", "Exit the CLI", "exit")
        ]
        
        for cmd, desc, usage in commands:
            help_table.add_row(cmd, desc, usage)
        
        console.print(help_table)
    
    async def query_knowledge(self, query: str):
        """Query knowledge across distributed memory and knowledge bases"""
        if not query:
            console.print("[yellow]Please provide a query[/]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Searching distributed memory and knowledge bases...[/]"),
            transient=True
        ) as progress:
            progress.add_task("search", total=None)
            
            # Search in distributed memory
            memory_results = self.memory_system.query_knowledge(
                query=query,
                top_k=5,
                threshold=0.1,
                require_verification=True
            )
            
            # Search in knowledge bases (async)
            kb_results_dict = await self.kb_connector.dispatch_query_to_all_kbs(query)
            kb_results = kb_results_dict.get("results", [])
        
        # Show results
        console.print(Panel(f"[bold]Query:[/] {query}", title="Search Results"))
        
        # Display memory results
        if memory_results:
            console.print("[bold green]Results from distributed memory:[/]")
            for i, (content, similarity, source) in enumerate(memory_results):
                console.print(f"[bold]{i+1}.[/] [blue]{content}[/]")
                console.print(f"   [dim]Similarity: {similarity:.3f} | Source: {source}[/]")
            console.print()
        
        # Display KB results
        if kb_results:
            console.print("[bold green]Results from knowledge bases:[/]")
            for i, result in enumerate(kb_results[:10]):  # Limit to 10 results
                if isinstance(result, dict):
                    content = result.get("content", result.get("text", str(result)))
                    source = result.get("source_kb", "unknown")
                    console.print(f"[bold]{i+1}.[/] [blue]{content}[/]")
                    console.print(f"   [dim]Source: {source}[/]")
                else:
                    console.print(f"[bold]{i+1}.[/] {result}")
        
        if not memory_results and not kb_results:
            console.print("[yellow]No results found[/]")
    
    async def add_knowledge(self, content: str):
        """Add a new knowledge fragment"""
        if not content:
            console.print("[yellow]Please provide content for the knowledge fragment[/]")
            return
        
        # Prompt for tags
        tags_input = Prompt.ask("[bold]Enter tags[/] (comma separated)")
        tags = {tag.strip() for tag in tags_input.split(",")} if tags_input else set()
        
        # Add the fragment
        fragment = self.memory_system.create_knowledge_fragment(
            content=content,
            tags=tags,
            confidence=0.9
        )
        
        console.print(f"[bold green]Knowledge fragment added with ID:[/] {fragment.fragment_id}")
        console.print(f"[bold]Content:[/] {fragment.content}")
        console.print(f"[bold]Tags:[/] {', '.join(fragment.tags)}")
    
    async def verify_knowledge(self, fragment_id: str):
        """Verify a knowledge fragment"""
        if not fragment_id:
            console.print("[yellow]Please provide a fragment ID[/]")
            return
        
        # Verify the fragment
        result = self.memory_system.verify_knowledge_fragment(
            fragment_id=fragment_id,
            verification_confidence=0.9
        )
        
        if result:
            console.print(f"[bold green]Fragment {fragment_id} successfully verified[/]")
        else:
            console.print(f"[bold red]Failed to verify fragment {fragment_id}[/]")
    
    async def dispute_knowledge(self, args: str):
        """Dispute a knowledge fragment"""
        if not args:
            console.print("[yellow]Please provide a fragment ID and reason[/]")
            return
        
        args_parts = args.split(" ", 1)
        fragment_id = args_parts[0]
        reason = args_parts[1] if len(args_parts) > 1 else "No reason provided"
        
        # Dispute the fragment
        result = self.memory_system.dispute_knowledge_fragment(
            fragment_id=fragment_id,
            dispute_reason=reason
        )
        
        if result:
            console.print(f"[bold green]Fragment {fragment_id} successfully disputed[/]")
            console.print(f"[bold]Reason:[/] {reason}")
        else:
            console.print(f"[bold red]Failed to dispute fragment {fragment_id}[/]")
    
    async def list_knowledge(self, filter_type: str = "all"):
        """List knowledge fragments"""
        # Determine filter based on argument
        filter_type = filter_type.lower() if filter_type else "all"
        if filter_type not in ["all", "verified", "disputed", "unverified"]:
            filter_type = "all"
        
        # Get all fragments
        fragments = self.memory_system.knowledge_fragments.values()
        
        # Apply filter
        if filter_type == "verified":
            fragments = [f for f in fragments if f.verification_status == "verified"]
        elif filter_type == "disputed":
            fragments = [f for f in fragments if f.verification_status == "disputed"]
        elif filter_type == "unverified":
            fragments = [f for f in fragments if f.verification_status == "unverified"]
        
        # Create table
        table = Table(title=f"{filter_type.capitalize()} Knowledge Fragments")
        table.add_column("ID", style="dim")
        table.add_column("Content")
        table.add_column("Status", style="bold")
        table.add_column("Source")
        table.add_column("Tags")
        
        # Add rows
        for fragment in fragments:
            # Truncate content if too long
            content = fragment.content[:50] + "..." if len(str(fragment.content)) > 50 else fragment.content
            
            # Status color
            status_style = {
                "verified": "green",
                "disputed": "red",
                "unverified": "yellow"
            }.get(fragment.verification_status, "")
            
            table.add_row(
                fragment.fragment_id[:8],
                str(content),
                f"[{status_style}]{fragment.verification_status}[/]",
                fragment.source_agent_id,
                ", ".join(fragment.tags)
            )
        
        console.print(table)
        console.print(f"[dim]Total: {len(fragments)} fragments[/]")
    
    async def kb_command(self, args: str):
        """Execute a command on a specific knowledge base"""
        if not args:
            console.print("[yellow]Please provide a knowledge base name and command[/]")
            return
        
        args_parts = args.split(" ", 1)
        if len(args_parts) < 2:
            console.print("[yellow]Please provide both a knowledge base name and command[/]")
            return
        
        kb_name = args_parts[0]
        command = args_parts[1]
        
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]Executing command on {kb_name}...[/]"),
            transient=True
        ) as progress:
            progress.add_task("execute", total=None)
            
            # Execute command on KB
            result = await self.kb_connector.dispatch_to_kb_agent(kb_name, command)
        
        # Display result
        if result.get("success", False):
            if "data" in result:
                data = result["data"]
                if isinstance(data, dict) or isinstance(data, list):
                    # Format as JSON
                    formatted_data = json.dumps(data, indent=2)
                    console.print(Panel(formatted_data, title=f"Result from {kb_name}"))
                else:
                    # Format as string
                    console.print(Panel(str(data), title=f"Result from {kb_name}"))
            else:
                console.print(f"[green]Command executed successfully on {kb_name}[/]")
        else:
            error = result.get("error", "Unknown error")
            console.print(f"[bold red]Error executing command on {kb_name}:[/] {error}")
    
    async def search_kb(self, query: str):
        """Search across all knowledge bases"""
        if not query:
            console.print("[yellow]Please provide a search query[/]")
            return
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Searching knowledge bases...[/]"),
            transient=True
        ) as progress:
            progress.add_task("search", total=None)
            
            # Search in knowledge bases
            results = await self.kb_connector.dispatch_query_to_all_kbs(query)
        
        # Display results
        if results.get("success", False) and "results" in results:
            kb_results = results["results"]
            console.print(Panel(f"[bold]Query:[/] {query}", title=f"Found {len(kb_results)} results"))
            
            for i, result in enumerate(kb_results[:20]):  # Limit to 20 results
                if isinstance(result, dict):
                    content = result.get("content", result.get("text", str(result)))
                    source = result.get("source_kb", "unknown")
                    console.print(f"[bold]{i+1}.[/] [blue]{content}[/]")
                    console.print(f"   [dim]Source: {source}[/]")
                else:
                    console.print(f"[bold]{i+1}.[/] {result}")
        else:
            error = results.get("error", "No results found")
            console.print(f"[yellow]{error}[/]")
    
    async def request_consensus(self, fragment_id: str):
        """Request consensus on a knowledge fragment from other agents"""
        if not fragment_id:
            console.print("[yellow]Please provide a fragment ID[/]")
            return
        
        fragment = self.memory_system.knowledge_fragments.get(fragment_id)
        if not fragment:
            console.print(f"[bold red]Fragment {fragment_id} not found[/]")
            return
        
        # Get available agents (from KB)
        kb_list = self.kb_connector.get_available_knowledge_bases()
        agent_ids = [kb["agent_id"] for kb in kb_list]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Requesting consensus...[/]"),
            transient=True
        ) as progress:
            progress.add_task("consensus", total=None)
            
            # Request consensus
            consensus_future = self.memory_system.request_consensus(
                fragment_id=fragment_id,
                agent_ids=agent_ids,
                timeout=10.0
            )
            
            # Wait for result
            result = await consensus_future
        
        # Display result
        if result.get("consensus", False):
            console.print(f"[bold green]Consensus reached for fragment {fragment_id}[/]")
        else:
            reason = result.get("reason", "No consensus")
            console.print(f"[bold red]No consensus for fragment {fragment_id}: {reason}[/]")
        
        # Show details
        console.print(f"[bold]Verification count:[/] {result.get('verification_count', 0)}")
        console.print(f"[bold]Dispute count:[/] {result.get('dispute_count', 0)}")
        console.print(f"[bold]Total agents asked:[/] {result.get('total_agents', 0)}")
    
    async def show_stats(self):
        """Show statistics about the distributed memory system"""
        # Get stats
        stats = self.memory_system.get_agent_stats()
        
        # Create table
        table = Table(title="Distributed Memory System Statistics")
        table.add_column("Metric", style="bold blue")
        table.add_column("Value")
        
        # Add rows
        table.add_row("Agent ID", stats["agent_id"])
        table.add_row("Total Knowledge Fragments", str(stats["knowledge_fragments"]))
        table.add_row("Verified Fragments", str(stats["verified_fragments"]))
        table.add_row("Disputed Fragments", str(stats["disputed_fragments"]))
        table.add_row("Local Memories", str(stats["local_memories"]))
        table.add_row("Known Agents", str(stats["known_agents"]))
        table.add_row("Trusted Agents", str(stats["trusted_agents"]))
        
        console.print(table)
        
        # Integration stats
        integration_stats = stats["integration_stats"]
        console.print("[bold]Integration Statistics:[/]")
        for key, value in integration_stats.items():
            console.print(f"[bold]{key.replace('_', ' ').title()}:[/] {value}")
    
    def save_memory(self):
        """Save the distributed memory system to disk"""
        if not self.memory_path:
            memory_path = Prompt.ask("[bold]Enter path to save memory[/]", default="./memory.pkl")
            self.memory_path = memory_path
        
        result = self.memory_system.save_state()
        
        if result:
            console.print(f"[bold green]Memory successfully saved to {self.memory_path}[/]")
        else:
            console.print("[bold red]Failed to save memory[/]")
    
    def load_memory(self):
        """Load the distributed memory system from disk"""
        if not self.memory_path:
            memory_path = Prompt.ask("[bold]Enter path to load memory from[/]", default="./memory.pkl")
            self.memory_path = memory_path
        
        result = self.memory_system.load_state()
        
        if result:
            console.print(f"[bold green]Memory successfully loaded from {self.memory_path}[/]")
        else:
            console.print("[bold red]Failed to load memory[/]")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Knowledge Base Memory CLI")
    parser.add_argument("--agent-id", type=str, default="memory-cli",
                       help="Unique identifier for this agent")
    parser.add_argument("--dimensions", type=int, default=256,
                       help="Vector dimensionality for knowledge representation")
    parser.add_argument("--memory-path", type=str, default=None,
                       help="Path for persisting memory")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    return parser.parse_args()

async def main():
    """Main function"""
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start the CLI
    cli = KnowledgeBaseMemoryCLI(
        agent_id=args.agent_id,
        dimensions=args.dimensions,
        memory_path=args.memory_path
    )
    
    await cli.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Exiting Knowledge Base Memory CLI[/]")
        sys.exit(0)