#!/usr/bin/env python3
"""
enhanced_atomic_agent.py
-----------------------
An enhanced version of the atomic agent with optimized local performance,
advanced context awareness, and additional capabilities for improved AI interactions.

This version integrates:
1. Optimized vector memory for better embedding and retrieval performance
2. Dynamic context management for adaptive context windows and relevance
3. Additional capabilities focused on execution performance and user experience
"""

import os
import sys
import json
import time
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_agent.log')
    ]
)
logger = logging.getLogger("enhanced_agent")

# Import optimized components
try:
    from optimized_vector_memory import OptimizedVectorMemory
    OPTIMIZED_MEMORY_AVAILABLE = True
    logger.info("Optimized vector memory module loaded")
except ImportError:
    OPTIMIZED_MEMORY_AVAILABLE = False
    logger.warning("Optimized vector memory not available, using standard memory")

try:
    from dynamic_context_manager import DynamicContextManager
    DYNAMIC_CONTEXT_AVAILABLE = True
    logger.info("Dynamic context manager module loaded")
except ImportError:
    DYNAMIC_CONTEXT_AVAILABLE = False
    logger.warning("Dynamic context manager not available, using standard context")

# Import base atomic agent components (specific classes/functions as needed)
try:
    from atomic_agent import TogetherAgent, VectorMemory, AgentOrchestrator
    BASE_AGENT_AVAILABLE = True
    logger.info("Base atomic agent modules loaded")
except ImportError:
    BASE_AGENT_AVAILABLE = False
    logger.error("Base atomic agent not found - this module requires atomic_agent.py")
    sys.exit(1)

# ===================================
# Enhanced Together Agent
# ===================================
class EnhancedTogetherAgent(TogetherAgent):
    """
    Enhanced version of the TogetherAgent with optimized local performance,
    context awareness, and additional capabilities
    """
    
    def __init__(self, model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", 
                num_scouts: int = 3, use_optimized_memory: bool = True,
                use_dynamic_context: bool = True, **kwargs):
        """
        Initialize the enhanced agent
        
        Args:
            model: LLM model to use
            num_scouts: Number of scout agents to use
            use_optimized_memory: Whether to use optimized vector memory
            use_dynamic_context: Whether to use dynamic context management
            **kwargs: Additional arguments to pass to parent constructor
        """
        # Call parent constructor
        super().__init__(model=model, num_scouts=num_scouts, **kwargs)
        
        # Store creation settings
        self.use_optimized_memory = use_optimized_memory
        self.use_dynamic_context = use_dynamic_context
        
        # Enhanced memory system
        if use_optimized_memory and OPTIMIZED_MEMORY_AVAILABLE:
            self.memory = OptimizedVectorMemory()
            logger.info("Using optimized vector memory")
        else:
            logger.info("Using standard vector memory")
        
        # Enhanced context management
        if use_dynamic_context and DYNAMIC_CONTEXT_AVAILABLE:
            self.context_manager = DynamicContextManager()
            logger.info("Using dynamic context manager")
        else:
            self.context_manager = None
            logger.info("Dynamic context manager not available")
        
        # Performance metrics tracking
        self.performance_metrics = {
            "requests": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "avg_response_time": 0,
            "total_response_time": 0,
            "peak_memory_mb": 0,
            "tool_calls": 0
        }
        
        # Additional configuration
        self.config = {
            "conversation_complexity": 0.5,  # Default complexity
            "task_complexity": 0.5,          # Default task complexity
            "use_local_processing": True,    # Prefer local processing when possible
            "enable_batching": True,         # Enable request batching
            "enable_caching": True,          # Enable response caching
            "max_token_limit": 16384         # Max token limit for context
        }
        
        # Response cache system
        self.response_cache = {}
        self.cache_hits = 0
        
        # Initialize custom extensions
        self._init_extensions()
    
    def _init_extensions(self):
        """Initialize any additional extensions or capabilities"""
        # Add any extension initialization code here
        pass
    
    def update_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the agent configuration
        
        Args:
            config_updates: Dictionary of configuration updates
            
        Returns:
            The updated configuration
        """
        self.config.update(config_updates)
        return self.config
    
    def generate_response(self, user_input: Union[str, List], 
                         system_prompt: str = None,
                         conversation_history: List = None,
                         temperature: float = None,
                         stream: bool = False) -> str:
        """
        Generate a response to the user input with enhanced context and performance
        
        Args:
            user_input: User input string or list of messages
            system_prompt: Optional system prompt to override default
            conversation_history: Optional conversation history
            temperature: Optional temperature parameter
            stream: Whether to stream the response
            
        Returns:
            Generated response
        """
        # Track metrics
        start_time = time.time()
        self.performance_metrics["requests"] += 1
        
        # Check cache if enabled
        if self.config["enable_caching"] and isinstance(user_input, str):
            cache_key = self._compute_cache_key(user_input, system_prompt, temperature)
            if cache_key in self.response_cache:
                self.cache_hits += 1
                logger.info(f"Cache hit for input: {user_input[:30]}...")
                return self.response_cache[cache_key]
        
        # Process with dynamic context if available
        if self.context_manager and self.use_dynamic_context:
            # Add user message to context
            if isinstance(user_input, str):
                self.context_manager.add_message(
                    content=user_input,
                    from_role="user",
                    to_role="assistant",
                    message_type="user_message"
                )
            
            # Get optimized context
            context_data = self.context_manager.get_dynamic_context(
                query=user_input if isinstance(user_input, str) else None,
                conversation_complexity=self.config["conversation_complexity"],
                task_complexity=self.config["task_complexity"]
            )
            
            # Extract relevant context items
            if conversation_history is None:
                conversation_history = []
                for item in context_data["items"]:
                    if item["item_type"] == "MessageItem":
                        conversation_history.append({
                            "role": item["metadata"].get("from_role", "user"),
                            "content": item["content"]
                        })
        
        # Call parent implementation with enhanced context and settings
        response = super().generate_response(
            user_input=user_input,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            temperature=temperature,
            stream=stream
        )
        
        # Add response to context if using dynamic context
        if self.context_manager and self.use_dynamic_context and isinstance(response, str):
            self.context_manager.add_message(
                content=response,
                from_role="assistant",
                to_role="user",
                message_type="assistant_message"
            )
        
        # Update cache if enabled
        if self.config["enable_caching"] and isinstance(user_input, str) and isinstance(response, str):
            cache_key = self._compute_cache_key(user_input, system_prompt, temperature)
            self.response_cache[cache_key] = response
            
            # Limit cache size
            if len(self.response_cache) > 1000:
                # Remove oldest 20% of entries
                keys = list(self.response_cache.keys())
                for key in keys[:len(keys)//5]:
                    del self.response_cache[key]
        
        # Update performance metrics
        response_time = time.time() - start_time
        self.performance_metrics["total_response_time"] += response_time
        self.performance_metrics["avg_response_time"] = (
            self.performance_metrics["total_response_time"] / 
            self.performance_metrics["requests"]
        )
        
        # Estimate tokens (very rough, 4 chars per token)
        if isinstance(user_input, str):
            self.performance_metrics["tokens_input"] += len(user_input) // 4
        if isinstance(response, str):
            self.performance_metrics["tokens_output"] += len(response) // 4
        
        # Track memory usage
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > self.performance_metrics["peak_memory_mb"]:
            self.performance_metrics["peak_memory_mb"] = memory_mb
        
        return response
    
    def execute_function(self, function_name: str, params: Dict[str, Any]) -> Any:
        """
        Enhanced function execution with performance tracking
        
        Args:
            function_name: Name of the function to execute
            params: Function parameters
            
        Returns:
            Function result
        """
        start_time = time.time()
        self.performance_metrics["tool_calls"] += 1
        
        # Execute the function
        result = super().execute_function(function_name, params)
        
        # Track execution time
        execution_time = time.time() - start_time
        if "function_performance" not in self.performance_metrics:
            self.performance_metrics["function_performance"] = {}
        
        if function_name not in self.performance_metrics["function_performance"]:
            self.performance_metrics["function_performance"][function_name] = {
                "calls": 0,
                "total_time": 0,
                "avg_time": 0
            }
        
        # Update function metrics
        func_metrics = self.performance_metrics["function_performance"][function_name]
        func_metrics["calls"] += 1
        func_metrics["total_time"] += execution_time
        func_metrics["avg_time"] = func_metrics["total_time"] / func_metrics["calls"]
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Add context metrics if available
        if self.context_manager:
            metrics["context"] = self.context_manager.get_metrics()
        
        # Add memory metrics if using optimized memory
        if self.use_optimized_memory and hasattr(self.memory, "get_stats"):
            metrics["memory"] = self.memory.get_stats()
        
        # Add cache metrics
        metrics["cache"] = {
            "size": len(self.response_cache),
            "hits": self.cache_hits,
            "hit_rate": self.cache_hits / max(1, self.performance_metrics["requests"]) * 100
        }
        
        return metrics
    
    def save_state(self, path: str = None) -> bool:
        """
        Save agent state to disk
        
        Args:
            path: Path to save to (default: ~/.cache/llama4_enhanced_agent)
            
        Returns:
            True if successful
        """
        if path is None:
            path = os.path.join(os.path.expanduser('~'), '.cache', 'llama4_enhanced_agent')
        
        os.makedirs(path, exist_ok=True)
        
        try:
            # Save performance metrics
            with open(os.path.join(path, 'performance_metrics.json'), 'w') as f:
                json.dump(self.performance_metrics, f)
            
            # Save configuration
            with open(os.path.join(path, 'config.json'), 'w') as f:
                json.dump(self.config, f)
            
            # Save context if available
            if self.context_manager:
                self.context_manager._save_context()
            
            # Save memory if optimized and has save method
            if self.use_optimized_memory and hasattr(self.memory, "save"):
                self.memory.save()
            
            logger.info(f"Agent state saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
            return False
    
    def load_state(self, path: str = None) -> bool:
        """
        Load agent state from disk
        
        Args:
            path: Path to load from (default: ~/.cache/llama4_enhanced_agent)
            
        Returns:
            True if successful
        """
        if path is None:
            path = os.path.join(os.path.expanduser('~'), '.cache', 'llama4_enhanced_agent')
        
        try:
            # Load performance metrics
            metrics_path = os.path.join(path, 'performance_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.performance_metrics = json.load(f)
            
            # Load configuration
            config_path = os.path.join(path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            
            # Context and memory are loaded in their own constructors
            
            logger.info(f"Agent state loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load agent state: {e}")
            return False
    
    def _compute_cache_key(self, input_text: str, system_prompt: str = None, 
                         temperature: float = None) -> str:
        """Compute a cache key for response caching"""
        import hashlib
        
        # Create a string with all relevant inputs
        key_parts = [input_text]
        if system_prompt:
            key_parts.append(f"sys:{system_prompt}")
        if temperature is not None:
            key_parts.append(f"temp:{temperature}")
        
        # Add model name for safety
        key_parts.append(f"model:{self.model}")
        
        # Compute hash
        key_string = "||".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

# ===================================
# Enhanced Agent Orchestrator
# ===================================
class EnhancedAgentOrchestrator(AgentOrchestrator):
    """
    Enhanced version of the AgentOrchestrator with improved performance,
    context awareness, and additional capabilities
    """
    
    def __init__(self, num_scouts=3, num_societal_agents=0, 
                model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", 
                initial_simulation=False, use_enhanced_agents=True,
                shared_context=True):
        """
        Initialize the enhanced orchestrator
        
        Args:
            num_scouts: Number of scout agents
            num_societal_agents: Number of societal agents
            model: LLM model to use
            initial_simulation: Whether to start a simulation initially
            use_enhanced_agents: Whether to use enhanced agents
            shared_context: Whether to use shared context between agents
        """
        # Call parent constructor first
        super().__init__(
            num_scouts=num_scouts,
            num_societal_agents=num_societal_agents,
            model=model,
            initial_simulation=initial_simulation
        )
        
        # Store whether to use enhanced agents
        self.use_enhanced_agents = use_enhanced_agents
        
        # Shared context for all agents if enabled
        self.shared_context = None
        if shared_context and DYNAMIC_CONTEXT_AVAILABLE:
            self.shared_context = DynamicContextManager(
                storage_path=os.path.join(os.path.expanduser('~'), 
                                        '.cache', 'llama4_shared_context')
            )
            logger.info("Using shared context between agents")
        
        # Enhanced performance tracking
        self.enhanced_metrics = {
            "orchestration_overhead_ms": [],
            "agent_response_times_ms": {},
            "total_function_calls": 0,
            "cached_responses": 0
        }
        
        # Response caching for faster repeated questions
        self.response_cache = {}
        
        # Replace scouts with enhanced agents if enabled
        if use_enhanced_agents:
            self._upgrade_scouts()
    
    def _upgrade_scouts(self):
        """Upgrade scouts to enhanced agents"""
        if not self.use_enhanced_agents:
            return
        
        # Create new enhanced scouts
        new_scouts = {}
        for scout_id, scout in self.scouts.items():
            # Extract scout's specialization
            specialization = getattr(scout, 'specialization', 'general')
            
            # Create an enhanced agent with the same specialization
            new_scout = EnhancedTogetherAgent(
                model=self.model,
                num_scouts=1,  # Each scout is its own agent
                use_optimized_memory=True,
                use_dynamic_context=True
            )
            
            # Set specialization
            new_scout.specialization = specialization
            
            # Add to new scouts dictionary
            new_scouts[scout_id] = new_scout
            
            logger.info(f"Upgraded scout {scout_id} to enhanced agent")
        
        # Replace scouts
        self.scouts = new_scouts
        logger.info(f"Upgraded {len(self.scouts)} scouts to enhanced agents")
    
    def _find_best_agent_for_task(self, task, required_skills=None, context=None):
        """
        Enhanced agent selection with context awareness and performance history
        
        Args:
            task: The task to assign
            required_skills: Skills required for the task
            context: Additional context for the task
            
        Returns:
            The best agent for the task
        """
        start_time = time.time()
        
        # Get agent from parent implementation
        agent = super()._find_best_agent_for_task(task, required_skills, context)
        
        # Record orchestration overhead
        overhead_ms = (time.time() - start_time) * 1000
        self.enhanced_metrics["orchestration_overhead_ms"].append(overhead_ms)
        
        # Trim history to last 100 entries
        if len(self.enhanced_metrics["orchestration_overhead_ms"]) > 100:
            self.enhanced_metrics["orchestration_overhead_ms"] = self.enhanced_metrics["orchestration_overhead_ms"][-100:]
        
        return agent
    
    def add_task(self, task, priority=1, context=None):
        """
        Add a task with enhanced caching and context sharing
        
        Args:
            task: Task description
            priority: Task priority
            context: Additional context
            
        Returns:
            Task ID
        """
        # Check cache for identical tasks
        if self.response_cache and isinstance(task, str):
            cache_key = self._compute_cache_key(task, context)
            if cache_key in self.response_cache:
                self.enhanced_metrics["cached_responses"] += 1
                logger.info(f"Cache hit for task: {task[:30]}...")
                return self.response_cache[cache_key]
        
        # Add to shared context if enabled
        if self.shared_context and isinstance(task, str):
            self.shared_context.add_custom_item(
                content=task,
                source="task_queue",
                metadata={"priority": priority},
                tags=["task"]
            )
        
        # Call parent implementation
        task_id = super().add_task(task, priority, context)
        
        return task_id
    
    def get_task_result(self, task_id):
        """
        Get task result with enhanced metrics
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result
        """
        start_time = time.time()
        
        # Get result from parent implementation
        result = super().get_task_result(task_id)
        
        # Cache result if it's a string task
        if result and "task" in result and "result" in result:
            task = result.get("task")
            if isinstance(task, str):
                cache_key = self._compute_cache_key(task, None)
                self.response_cache[cache_key] = result
                
                # Limit cache size
                if len(self.response_cache) > 100:
                    # Remove oldest entries
                    keys = list(self.response_cache.keys())
                    for key in keys[:20]:  # Remove oldest 20
                        del self.response_cache[key]
        
        # Track agent response time
        if result and "agent_id" in result:
            agent_id = result["agent_id"]
            response_time_ms = (time.time() - start_time) * 1000
            
            if agent_id not in self.enhanced_metrics["agent_response_times_ms"]:
                self.enhanced_metrics["agent_response_times_ms"][agent_id] = []
            
            self.enhanced_metrics["agent_response_times_ms"][agent_id].append(response_time_ms)
            
            # Trim history to last 20 entries per agent
            if len(self.enhanced_metrics["agent_response_times_ms"][agent_id]) > 20:
                self.enhanced_metrics["agent_response_times_ms"][agent_id] = self.enhanced_metrics["agent_response_times_ms"][agent_id][-20:]
        
        return result
    
    def simulate_society(self, params=None):
        """
        Enhanced society simulation with context awareness
        
        Args:
            params: Simulation parameters
            
        Returns:
            Simulation object
        """
        # Add simulation setup to shared context if enabled
        if self.shared_context and params:
            self.shared_context.add_custom_item(
                content=f"Starting societal simulation with parameters: {json.dumps(params)}",
                source="simulation",
                metadata=params,
                tags=["simulation", "start"]
            )
        
        # Call parent implementation
        simulation = super().simulate_society(params)
        
        return simulation
    
    def get_enhanced_metrics(self):
        """Get enhanced performance metrics"""
        metrics = self.enhanced_metrics.copy()
        
        # Calculate averages
        if self.enhanced_metrics["orchestration_overhead_ms"]:
            metrics["avg_orchestration_overhead_ms"] = sum(self.enhanced_metrics["orchestration_overhead_ms"]) / len(self.enhanced_metrics["orchestration_overhead_ms"])
        
        # Calculate average response time per agent
        avg_response_times = {}
        for agent_id, times in self.enhanced_metrics["agent_response_times_ms"].items():
            if times:
                avg_response_times[agent_id] = sum(times) / len(times)
        
        metrics["avg_agent_response_times_ms"] = avg_response_times
        
        # Add shared context metrics if available
        if self.shared_context:
            metrics["shared_context"] = self.shared_context.get_metrics()
        
        # Add enhanced agent metrics if applicable
        agent_metrics = {}
        for scout_id, scout in self.scouts.items():
            if isinstance(scout, EnhancedTogetherAgent):
                agent_metrics[scout_id] = scout.get_performance_metrics()
        
        if agent_metrics:
            metrics["agent_performance"] = agent_metrics
        
        # Add cache metrics
        metrics["cache"] = {
            "size": len(self.response_cache),
            "hits": self.enhanced_metrics["cached_responses"]
        }
        
        return metrics
    
    def _compute_cache_key(self, task: str, context: Any = None) -> str:
        """Compute a cache key for response caching"""
        import hashlib
        
        # Create a string with task and context
        key_parts = [task]
        if context:
            if isinstance(context, dict):
                context_str = json.dumps(context, sort_keys=True)
            else:
                context_str = str(context)
            key_parts.append(context_str)
        
        # Compute hash
        key_string = "||".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

# ===================================
# Main Function 
# ===================================
def main():
    """Main function to demonstrate enhanced agent capabilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Llama4 Agent System")
    parser.add_argument("--scouts", type=int, default=3, help="Number of scout agents to create")
    parser.add_argument("--societal-agents", type=int, default=12, help="Number of societal agents to create")
    parser.add_argument("--simulation", action="store_true", help="Start a societal simulation")
    parser.add_argument("--duration", type=int, default=30, help="Simulation duration in days")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-4-Turbo-17B-Instruct-FP8", 
                       help="LLM model to use")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--api-key", type=str, help="Together API key (if not in env)")
    parser.add_argument("--disable-enhanced", action="store_true", help="Disable enhanced features")
    parser.add_argument("--disable-shared-context", action="store_true", help="Disable shared context")
    parser.add_argument("--performance-mode", action="store_true", help="Optimize for performance")
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["TOGETHER_API_KEY"] = args.api_key
    
    # Print header
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    console.print(Panel(
        "[bold green]Enhanced Llama4 Agent System[/bold green]\n"
        "Optimized local performance, advanced context awareness, and improved capabilities",
        border_style="green"
    ))
    
    # Initialize the orchestrator with enhanced settings
    console.print("[bold green]Initializing Enhanced Agent Orchestrator[/bold green]")
    
    use_enhanced = not args.disable_enhanced
    use_shared_context = not args.disable_shared_context
    
    orchestrator = EnhancedAgentOrchestrator(
        num_scouts=args.scouts,
        num_societal_agents=args.societal_agents,
        model=args.model,
        initial_simulation=args.simulation,
        use_enhanced_agents=use_enhanced,
        shared_context=use_shared_context
    )
    
    # Apply performance mode settings if enabled
    if args.performance_mode:
        console.print("[bold cyan]Performance mode enabled - optimizing for speed[/bold cyan]")
        for scout_id, scout in orchestrator.scouts.items():
            if isinstance(scout, EnhancedTogetherAgent):
                scout.update_config({
                    "conversation_complexity": 0.3,  # Reduce complexity
                    "max_token_limit": 8192,        # Lower token limit
                    "enable_batching": True,        # Enable batching
                    "enable_caching": True          # Enable caching
                })
    
    if args.interactive:
        from rich.prompt import Prompt
        
        # Interactive console mode
        console.print("\n[bold cyan]Enhanced Llama4 Agent System - Interactive Mode[/bold cyan]")
        console.print("Type 'help' for available commands or 'exit' to quit")
        
        while True:
            try:
                command = Prompt.ask("[bold blue]> [/bold blue]")
                
                if command.lower() == 'exit':
                    break
                elif command.lower() == 'help':
                    console.print("[cyan]Available commands:[/cyan]")
                    console.print("  simulate - Start a new societal simulation")
                    console.print("  list - List active and historical simulations")
                    console.print("  stats - Show performance statistics")
                    console.print("  memory - Show memory usage statistics")
                    console.print("  context - Show context statistics")
                    console.print("  agents - List all agents")
                    console.print("  ask <question> - Ask a question to all agents")
                    console.print("  optimize - Run optimization routines")
                    console.print("  exit - Exit the program")
                elif command.lower() == 'simulate':
                    duration = Prompt.ask("Simulation duration in days", default=str(args.duration))
                    simulation = orchestrator.simulate_society({
                        "name": f"Enhanced Interactive Simulation {int(time.time())}",
                        "duration": int(duration)
                    })
                    console.print(f"[green]Started simulation {simulation.simulation_id}[/green]")
                elif command.lower() == 'list':
                    simulations = orchestrator.list_simulations()
                    if simulations["active"]:
                        console.print("[bold cyan]Active simulations:[/bold cyan]")
                        for sim in simulations["active"]:
                            console.print(f"  {sim['simulation_id']} - {sim['name']} ({sim['status']}, {sim['days_elapsed']} days)")
                    else:
                        console.print("[yellow]No active simulations[/yellow]")
                        
                    if simulations["historical"]:
                        console.print("\n[bold cyan]Historical simulations:[/bold cyan]")
                        for sim in simulations["historical"]:
                            console.print(f"  {sim['simulation_id']} - {sim['name']} ({sim['status']}, {sim['days_elapsed']} days)")
                    else:
                        console.print("[yellow]No historical simulations[/yellow]")
                elif command.lower() == 'stats':
                    # Show enhanced metrics if available
                    if hasattr(orchestrator, 'get_enhanced_metrics'):
                        metrics = orchestrator.get_enhanced_metrics()
                        console.print("[bold cyan]Enhanced Performance Metrics:[/bold cyan]")
                        
                        if "avg_orchestration_overhead_ms" in metrics:
                            console.print(f"  Avg Orchestration Overhead: {metrics['avg_orchestration_overhead_ms']:.2f} ms")
                        
                        if "avg_agent_response_times_ms" in metrics:
                            console.print("  Agent Response Times:")
                            for agent_id, time_ms in metrics["avg_agent_response_times_ms"].items():
                                console.print(f"    {agent_id}: {time_ms:.2f} ms")
                        
                        if "cache" in metrics:
                            console.print(f"  Cache Size: {metrics['cache']['size']} entries")
                            console.print(f"  Cache Hits: {metrics['cache']['hits']}")
                        
                        console.print(f"  Total Function Calls: {metrics['total_function_calls']}")
                    else:
                        console.print("[yellow]Enhanced metrics not available[/yellow]")
                        
                    # Show base metrics too
                    console.print("\n[bold cyan]Base Performance Metrics:[/bold cyan]")
                    console.print(f"  Active Scouts: {len(orchestrator.scouts)}")
                    console.print(f"  Active Societal Agents: {sum(len(sector_data['agents']) for sector_data in orchestrator.societal_sectors.values() if sector_data['agents'])}")
                    console.print(f"  Active Policies: {len(orchestrator.policy_register) if hasattr(orchestrator, 'policy_register') else 0}")
                    console.print(f"  Active Coalitions: {len(orchestrator.coalition_register) if hasattr(orchestrator, 'coalition_register') else 0}")
                elif command.lower() == 'memory':
                    console.print("[bold cyan]Memory Usage Statistics:[/bold cyan]")
                    
                    # Get memory usage of the process
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    
                    console.print(f"  Process RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
                    console.print(f"  Process VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
                    
                    # Show agent memory stats if available
                    for scout_id, scout in orchestrator.scouts.items():
                        if isinstance(scout, EnhancedTogetherAgent):
                            metrics = scout.get_performance_metrics()
                            console.print(f"  Scout {scout_id}:")
                            console.print(f"    Peak Memory: {metrics.get('peak_memory_mb', 'N/A')} MB")
                            
                            if "memory" in metrics:
                                memory_stats = metrics["memory"]
                                console.print(f"    Memory Items: {memory_stats.get('memory_count', 'N/A')}")
                                console.print(f"    Active Items: {memory_stats.get('active_count', 'N/A')}")
                                
                                if "embedding_cache" in memory_stats:
                                    cache = memory_stats["embedding_cache"]
                                    console.print(f"    Embedding Cache Hits: {cache.get('cache_hits', 0)}")
                                    console.print(f"    Embedding Cache Size: {cache.get('memory_cache_size', 0)} items")
                elif command.lower() == 'context':
                    console.print("[bold cyan]Context Statistics:[/bold cyan]")
                    
                    # Show shared context stats if available
                    if orchestrator.shared_context:
                        metrics = orchestrator.shared_context.get_metrics()
                        console.print("  Shared Context:")
                        console.print(f"    Item Count: {metrics.get('item_count', 0)}")
                        console.print(f"    Conversation Length: {metrics.get('conversation_length', 0)}")
                        console.print(f"    Average Context Size: {metrics.get('avg_context_size', 0):.2f} items")
                        
                        token_usage = metrics.get('token_usage', {})
                        console.print(f"    Average Tokens: {token_usage.get('avg_tokens', 0):.2f}")
                        console.print(f"    Max Tokens: {token_usage.get('max_tokens', 0)}")
                    else:
                        console.print("  Shared Context: Not enabled")
                    
                    # Show agent context stats
                    for scout_id, scout in orchestrator.scouts.items():
                        if isinstance(scout, EnhancedTogetherAgent) and scout.context_manager:
                            metrics = scout.context_manager.get_metrics()
                            console.print(f"  Agent {scout_id} Context:")
                            console.print(f"    Item Count: {metrics.get('item_count', 0)}")
                            console.print(f"    Conversation Length: {metrics.get('conversation_length', 0)}")
                elif command.lower() == 'agents':
                    console.print("[bold cyan]Scout Agents:[/bold cyan]")
                    for agent_id, agent in orchestrator.scouts.items():
                        agent_type = "Enhanced" if isinstance(agent, EnhancedTogetherAgent) else "Standard"
                        console.print(f"  {agent_id} - {agent_type} - {getattr(agent, 'specialization', 'unknown')}")
                        
                    console.print("\n[bold cyan]Societal Agents:[/bold cyan]")
                    for sector, sector_data in orchestrator.societal_sectors.items():
                        if sector_data["agents"]:
                            console.print(f"[cyan]{sector} sector:[/cyan]")
                            for agent_id, agent in sector_data["agents"].items():
                                console.print(f"  {agent_id} - {agent.role}")
                elif command.lower().startswith('ask '):
                    query = command[4:].strip()
                    if not query:
                        console.print("[red]Please provide a question to ask[/red]")
                        continue
                    
                    console.print(f"[cyan]Asking all agents: {query}[/cyan]")
                    
                    # Create tasks for all scouts
                    tasks = []
                    for scout_id in orchestrator.scouts.keys():
                        tasks.append({
                            "task": query,
                            "agent_id": scout_id
                        })
                    
                    # Execute tasks in parallel
                    results = orchestrator.execute_parallel_tasks(tasks)
                    
                    # Display results
                    console.print("\n[bold cyan]Agent Responses:[/bold cyan]")
                    for result in results:
                        agent_id = result.get("agent_id", "unknown")
                        agent_result = result.get("result", "No response")
                        console.print(f"[bold]{agent_id}:[/bold]")
                        console.print(f"  {agent_result}")
                        console.print()
                elif command.lower() == 'optimize':
                    console.print("[cyan]Running optimization routines...[/cyan]")
                    
                    # Optimize team structure
                    orchestrator.optimize_team_structure()
                    console.print("[green]Team structure optimized[/green]")
                    
                    # Save agent states
                    for scout_id, scout in orchestrator.scouts.items():
                        if isinstance(scout, EnhancedTogetherAgent):
                            scout.save_state()
                    console.print("[green]Agent states saved[/green]")
                    
                    # Garbage collection
                    import gc
                    gc.collect()
                    console.print("[green]Memory garbage collected[/green]")
                else:
                    console.print(f"[red]Unknown command: {command}[/red]")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    
    elif args.simulation:
        # Non-interactive simulation mode
        console.print(f"[cyan]Running enhanced societal simulation for {args.duration} days...[/cyan]")
        simulation = orchestrator.simulate_society({
            "duration": args.duration
        })
        
        # Wait until simulation is complete
        while simulation.status != "completed":
            time.sleep(1)
            
        # Show simulation results
        console.print(f"\n[bold green]Simulation {simulation.simulation_id} completed![/bold green]")
        console.print(f"Duration: {len(simulation.historical_events)} days")
        
        if simulation.metrics:
            console.print("\n[bold cyan]Final metrics:[/bold cyan]")
            for metric, values in simulation.metrics.items():
                if values:
                    console.print(f"  {metric}: {values[-1]['value']}")
        
        # Show performance stats
        if hasattr(orchestrator, 'get_enhanced_metrics'):
            console.print("\n[bold cyan]Performance Metrics:[/bold cyan]")
            metrics = orchestrator.get_enhanced_metrics()
            
            if "avg_orchestration_overhead_ms" in metrics:
                console.print(f"  Avg Orchestration Overhead: {metrics['avg_orchestration_overhead_ms']:.2f} ms")
            
            if "cache" in metrics:
                console.print(f"  Cache Size: {metrics['cache']['size']} entries")
                console.print(f"  Cache Hits: {metrics['cache']['hits']}")
    
    console.print("[bold green]Shutting down...[/bold green]")

if __name__ == "__main__":
    main()