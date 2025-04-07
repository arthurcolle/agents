#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import json
import requests
import asyncio
import time
import random
import uuid
from typing import Dict, List, Any, Optional, Tuple
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Import agent hooks and advanced visualizer
from agent_hooks import CLIAgentHooks
from data_visualizer import AdvancedDataVisualizer
from dynamic_agents import registry, AgentContext, execute_agent_command

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cli-agent")

# Rich console for better formatting
console = Console()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DataAnalysisTools:
    """
    Tools for data analysis that can be used by the agent
    """
    def __init__(self):
        logger.info("Initializing data analysis tools")
        self.visualizer = AdvancedDataVisualizer()
    
    def load_csv(self, filepath: str) -> Dict:
        """Load a CSV file and return basic statistics"""
        try:
            import pandas as pd
            
            # Load the data
            df = pd.read_csv(filepath)
            
            # Generate basic statistics
            stats = {
                "columns": list(df.columns),
                "shape": df.shape,
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "head": df.head(5).to_dict(orient="records"),
                "describe": df.describe().to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            }
            
            return {
                "success": True,
                "message": f"Successfully loaded CSV file with {df.shape[0]} rows and {df.shape[1]} columns",
                "data": stats
            }
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return {
                "success": False,
                "message": f"Error loading CSV file: {str(e)}",
                "data": None
            }
    
    def plot_data(self, data: Dict, plot_type: str = "histogram", 
                 x_column: str = None, y_column: str = None,
                 title: str = "Data Visualization") -> Dict:
        """Generate a plot from data and save it to a file"""
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create a dataframe from the data
            if isinstance(data, dict) and "head" in data:
                # If data is from load_csv
                df = pd.DataFrame(data["head"])
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                # If data is a list of dictionaries
                df = pd.DataFrame(data)
            else:
                return {
                    "success": False,
                    "message": "Invalid data format for plotting",
                    "data": None
                }
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            if plot_type == "histogram" and x_column:
                sns.histplot(data=df, x=x_column)
            elif plot_type == "scatter" and x_column and y_column:
                sns.scatterplot(data=df, x=x_column, y=y_column)
            elif plot_type == "bar" and x_column and y_column:
                sns.barplot(data=df, x=x_column, y=y_column)
            elif plot_type == "line" and x_column and y_column:
                sns.lineplot(data=df, x=x_column, y=y_column)
            elif plot_type == "heatmap":
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            else:
                return {
                    "success": False,
                    "message": f"Invalid plot type or missing required columns for {plot_type}",
                    "data": None
                }
            
            plt.title(title)
            plt.tight_layout()
            
            # Save the plot
            output_file = f"{plot_type}_{x_column}_{y_column if y_column else ''}.png"
            plt.savefig(output_file)
            plt.close()
            
            return {
                "success": True,
                "message": f"Successfully created {plot_type} plot and saved to {output_file}",
                "data": {
                    "file": output_file,
                    "plot_type": plot_type,
                    "x_column": x_column,
                    "y_column": y_column
                }
            }
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            return {
                "success": False,
                "message": f"Error creating plot: {str(e)}",
                "data": None
            }
    
    def analyze_text(self, text: str) -> Dict:
        """Perform basic text analysis"""
        try:
            # Basic text statistics
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            
            # Word frequency
            import re
            from collections import Counter
            
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq = Counter(words).most_common(10)
            
            return {
                "success": True,
                "message": "Successfully analyzed text",
                "data": {
                    "word_count": word_count,
                    "character_count": char_count,
                    "sentence_count": sentence_count,
                    "top_words": word_freq
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                "success": False,
                "message": f"Error analyzing text: {str(e)}",
                "data": None
            }

class ModalIntegration:
    """
    Integration with Modal for running functions in the cloud
    """
    def __init__(self, endpoint="https://arthurcolle--registry.modal.run"):
        self.endpoint = endpoint
        self.available = self._check_availability()
        
        if self.available:
            logger.info(f"Modal integration available at {endpoint}")
        else:
            logger.warning(f"Modal integration not available at {endpoint}")
    
    def _check_availability(self) -> bool:
        """Check if Modal endpoint is available"""
        try:
            response = requests.get(self.endpoint, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def list_functions(self) -> Dict:
        """List available functions in Modal"""
        if not self.available:
            return {
                "success": False,
                "message": "Modal integration not available",
                "data": None
            }
        
        try:
            response = requests.get(f"{self.endpoint}/functions")
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": "Successfully retrieved Modal functions",
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "message": f"Error retrieving Modal functions: {response.status_code}",
                    "data": None
                }
        except Exception as e:
            logger.error(f"Error listing Modal functions: {e}")
            return {
                "success": False,
                "message": f"Error listing Modal functions: {str(e)}",
                "data": None
            }
    
    def call_function(self, function_name: str, params: Dict) -> Dict:
        """Call a function in Modal"""
        if not self.available:
            return {
                "success": False,
                "message": "Modal integration not available",
                "data": None
            }
        
        try:
            response = requests.post(
                f"{self.endpoint}/functions/{function_name}",
                json=params
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "message": f"Successfully called Modal function {function_name}",
                    "data": response.json()
                }
            else:
                return {
                    "success": False,
                    "message": f"Error calling Modal function: {response.status_code}",
                    "data": None
                }
        except Exception as e:
            logger.error(f"Error calling Modal function: {e}")
            return {
                "success": False,
                "message": f"Error calling Modal function: {str(e)}",
                "data": None
            }

class CLIAgent:
    """
    CLI agent for conversational data analysis
    """
    def __init__(self, model="gpt-4o", console=None):
        self.model = model
        self.data_tools = DataAnalysisTools()
        self.modal = ModalIntegration()
        self.conversation_history = []
        self.console = console or Console()
        self.hooks = CLIAgentHooks(self.console, display_name="Data Analysis Agent")
        
        # Dynamic agent contexts
        self.agent_contexts = {}
        
        # Initialize dynamic agent tools
        self._init_dynamic_agent_tools()
        
    def _init_dynamic_agent_tools(self):
        """Initialize tools for dynamic agent management"""
        # Create default agents
        try:
            # Create a file agent
            registry.create_agent("file_agent", "file")
            # Create a data analysis agent
            registry.create_agent("data_analysis_agent", "data_analysis")
            
            # Create contexts for each agent
            self.agent_contexts["file_agent"] = AgentContext(agent_id="file_agent")
            self.agent_contexts["data_analysis_agent"] = AgentContext(agent_id="data_analysis_agent")
            
            self.console.print("[dim]Initialized dynamic agents: file_agent, data_analysis_agent[/dim]")
        except Exception as e:
            self.console.print(f"[bold red]Error initializing dynamic agents: {e}[/bold red]")
    
        # Define available tools
        # Add advanced visualization tool
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_agent",
                    "description": "Create a new specialized agent for a specific task",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "Unique identifier for the agent"
                            },
                            "agent_type": {
                                "type": "string",
                                "description": "Type of agent to create",
                                "enum": ["file", "data_analysis"]
                            }
                        },
                        "required": ["agent_id", "agent_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_agents",
                    "description": "List all available agents",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_agent_command",
                    "description": "Execute a command on a specific agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_id": {
                                "type": "string",
                                "description": "ID of the agent to execute the command on"
                            },
                            "command": {
                                "type": "string",
                                "description": "Command to execute on the agent"
                            }
                        },
                        "required": ["agent_id", "command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_advanced_visualization",
                    "description": "Create advanced data visualizations including correlation matrices, pairplots, and more",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "description": "Data to visualize"
                            },
                            "viz_type": {
                                "type": "string",
                                "description": "Type of visualization to create",
                                "enum": ["correlation_matrix", "pairplot", "distribution", "boxplot", "timeseries", "3d_scatter"]
                            },
                            "title": {
                                "type": "string",
                                "description": "Title for the visualization"
                            },
                            "columns": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "Specific columns to include in the visualization"
                            },
                            "date_column": {
                                "type": "string",
                                "description": "Column to use as date for time series plots"
                            },
                            "value_column": {
                                "type": "string",
                                "description": "Column to use as value for time series plots"
                            },
                            "x_column": {
                                "type": "string",
                                "description": "Column to use for x-axis in 3D scatter plots"
                            },
                            "y_column": {
                                "type": "string",
                                "description": "Column to use for y-axis in 3D scatter plots"
                            },
                            "z_column": {
                                "type": "string",
                                "description": "Column to use for z-axis in 3D scatter plots"
                            },
                            "color_column": {
                                "type": "string",
                                "description": "Column to use for color in 3D scatter plots"
                            }
                        },
                        "required": ["data", "viz_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "load_csv",
                    "description": "Load a CSV file and return basic statistics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filepath": {
                                "type": "string",
                                "description": "Path to the CSV file"
                            }
                        },
                        "required": ["filepath"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plot_data",
                    "description": "Generate a plot from data and save it to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "description": "Data to plot"
                            },
                            "plot_type": {
                                "type": "string",
                                "description": "Type of plot (histogram, scatter, bar, line, heatmap)",
                                "enum": ["histogram", "scatter", "bar", "line", "heatmap"]
                            },
                            "x_column": {
                                "type": "string",
                                "description": "Column to use for x-axis"
                            },
                            "y_column": {
                                "type": "string",
                                "description": "Column to use for y-axis (for scatter, bar, line plots)"
                            },
                            "title": {
                                "type": "string",
                                "description": "Title for the plot"
                            }
                        },
                        "required": ["data", "plot_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_text",
                    "description": "Perform basic text analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to analyze"
                            }
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_modal_functions",
                    "description": "List available functions in Modal",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "call_modal_function",
                    "description": "Call a function in Modal",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "function_name": {
                                "type": "string",
                                "description": "Name of the function to call"
                            },
                            "params": {
                                "type": "object",
                                "description": "Parameters for the function"
                            }
                        },
                        "required": ["function_name", "params"]
                    }
                }
            }
        ]
        
        logger.info(f"CLI agent initialized with model {self.model}")
    
    def _handle_tool_call(self, name: str, arguments: Dict) -> Dict:
        """Handle tool calls from the agent"""
        logger.info(f"Handling tool call: {name} with arguments {arguments}")
        
        # Notify hooks that a tool is starting
        self.hooks.on_tool_start("CLI Agent", name, arguments)
        
        result = None
        try:
            if name == "load_csv":
                result = self.data_tools.load_csv(**arguments)
            elif name == "plot_data":
                result = self.data_tools.plot_data(**arguments)
            elif name == "analyze_text":
                result = self.data_tools.analyze_text(**arguments)
            elif name == "list_modal_functions":
                result = self.modal.list_functions()
            elif name == "call_modal_function":
                result = self.modal.call_function(**arguments)
            elif name == "create_advanced_visualization":
                result = self.data_tools.visualizer.create_visualization(**arguments)
            # Dynamic agent tools
            elif name == "create_agent":
                result = self._create_agent(**arguments)
            elif name == "list_agents":
                result = self._list_agents()
            elif name == "execute_agent_command":
                result = asyncio.run(self._execute_agent_command(**arguments))
            else:
                result = {
                    "success": False,
                    "message": f"Unknown tool: {name}",
                    "data": None
                }
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            self.hooks.on_error("CLI Agent", e)
            result = {
                "success": False,
                "message": f"Error executing tool {name}: {str(e)}",
                "data": None
            }
        
        # Notify hooks that the tool has completed
        self.hooks.on_tool_end("CLI Agent", name, result)
        
        return result
    
    def _create_agent(self, agent_id: str, agent_type: str) -> Dict:
        """Create a new agent of the specified type"""
        try:
            # Generate a unique ID if not provided
            if not agent_id:
                agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
            
            # Create the agent
            agent = registry.create_agent(agent_id, agent_type)
            
            # Create a context for the agent
            self.agent_contexts[agent_id] = AgentContext(agent_id=agent_id)
            
            return {
                "success": True,
                "message": f"Created {agent_type} agent with ID: {agent_id}",
                "data": {
                    "agent_id": agent_id,
                    "agent_type": agent_type
                }
            }
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return {
                "success": False,
                "message": f"Error creating agent: {str(e)}",
                "data": None
            }
    
    def _list_agents(self) -> Dict:
        """List all available agents"""
        try:
            agents = registry.list_agents()
            agent_types = registry.list_agent_types()
            
            return {
                "success": True,
                "message": f"Found {len(agents)} agents",
                "data": {
                    "agents": agents,
                    "available_types": agent_types
                }
            }
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            return {
                "success": False,
                "message": f"Error listing agents: {str(e)}",
                "data": None
            }
    
    async def _execute_agent_command(self, agent_id: str, command: str) -> Dict:
        """Execute a command on a specific agent"""
        try:
            # Get the agent context
            context = self.agent_contexts.get(agent_id)
            if not context:
                # Create a new context if it doesn't exist
                context = AgentContext(agent_id=agent_id)
                self.agent_contexts[agent_id] = context
            
            # Execute the command
            result = await execute_agent_command(agent_id, command, context)
            
            return {
                "success": result.get("success", False),
                "message": f"Executed command on agent {agent_id}",
                "data": result
            }
        except Exception as e:
            logger.error(f"Error executing agent command: {e}")
            return {
                "success": False,
                "message": f"Error executing agent command: {str(e)}",
                "data": None
            }
    
    async def chat(self, message: str) -> str:
        """Chat with the agent"""
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Notify hooks that the agent is starting
        self.hooks.on_start("CLI Agent")
        
        # Run the agent
        with self.console.status("[bold green]Thinking..."):
            try:
                # Check if the message contains any code execution requests
                if "```python" in message or "```py" in message or "execute this code" in message.lower():
                    self.console.print("[bold yellow]Warning: Code execution detected in request. Proceeding with caution.[/bold yellow]")
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    tools=self.tools,
                    tool_choice="auto"
                )
                
                # Get the response
                assistant_message = response.choices[0].message
                
                # Process tool calls if any
                if assistant_message.tool_calls:
                    # Add the assistant's message to the conversation
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            } for tool_call in assistant_message.tool_calls
                        ]
                    })
                    
                    # Process each tool call
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # Execute the function
                        with self.console.status(f"[bold blue]Running tool: {function_name}..."):
                            function_response = self._handle_tool_call(function_name, function_args)
                        
                        # Add the function response to the conversation
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(function_response)
                        })
                    
                    # Get the final response after tool calls
                    with self.console.status("[bold green]Processing results..."):
                        second_response = client.chat.completions.create(
                            model=self.model,
                            messages=self.conversation_history
                        )
                        
                        final_response = second_response.choices[0].message.content
                        
                        # Add the final response to the conversation
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": final_response
                        })
                        
                        return final_response
                else:
                    # No tool calls, just return the response
                    content = assistant_message.content
                    
                    # Add the response to the conversation history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": content
                    })
                    
                    # Notify hooks that the agent has completed
                    self.hooks.on_end("CLI Agent", content)
                    return content
                    
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                self.hooks.on_error("CLI Agent", e)
                return f"Error: {str(e)}"

def main():
    """Main function for the CLI agent"""
    parser = argparse.ArgumentParser(description="CLI Agent for Data Analysis")
    parser.add_argument("--model", default="gpt-4o", help="Model to use for the agent")
    parser.add_argument("--trace", action="store_true", help="Enable tracing for debugging")
    parser.add_argument("--debug", action="store_true", help="Show debug information including agent hooks")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model generation (0.0-2.0)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens for model response")
    parser.add_argument("--list-agents", action="store_true", help="List available dynamic agents and exit")
    args = parser.parse_args()
    
    # Enable tracing if requested
    if args.trace:
        openai.debug.trace.enable()
    
    # Create console
    console = Console()
    
    # Create the agent
    agent = CLIAgent(model=args.model, console=console)
    
    # If --list-agents flag is provided, list agents and exit
    if args.list_agents:
        agents = registry.list_agents()
        agent_types = registry.list_agent_types()
        
        console.print(Panel.fit(
            f"[bold]Available Agent Types:[/bold] {', '.join(agent_types)}\n\n"
            f"[bold]Registered Agents:[/bold]",
            title="Dynamic Agents",
            border_style="green"
        ))
        
        if agents:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Agent ID")
            table.add_column("Type")
            
            for agent_info in agents:
                table.add_row(agent_info["id"], agent_info["type"])
            
            console.print(table)
        else:
            console.print("[yellow]No agents registered yet.[/yellow]")
        
        return
    
    # Configure OpenAI client with additional parameters
    client.temperature = args.temperature
    client.max_tokens = args.max_tokens
    
    # Welcome message
    console.print(Panel.fit(
        "[bold blue]Welcome to the Data Analysis CLI Agent![/bold blue]\n"
        "You can chat with me about data analysis tasks, and I'll help you analyze data, "
        "create visualizations, and more.\n"
        "[bold cyan]Dynamic Agents:[/bold cyan] You can create and use specialized agents for specific tasks.\n"
        "Type [bold green]'exit'[/bold green] to quit.",
        title="Data Analysis Assistant",
        border_style="blue"
    ))
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold green]You")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("[bold blue]Goodbye![/bold blue]")
                break
            
            # Process the input
            response = asyncio.run(agent.chat(user_input))
            
            # Display the response
            console.print("\n[bold blue]Assistant[/bold blue]")
            console.print(Markdown(response))
            
            # Show debug summary if requested
            if args.debug:
                agent.hooks.display_summary()
            
        except KeyboardInterrupt:
            console.print("\n[bold blue]Goodbye![/bold blue]")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
