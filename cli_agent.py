#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import json
import requests
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint
import openai
from openai import OpenAI
from openai.types.beta.threads import Run
from openai.types.beta.assistant import Assistant
from openai.types.beta.threads.thread_message import ThreadMessage
from dotenv import load_dotenv

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
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.data_tools = DataAnalysisTools()
        self.modal = ModalIntegration()
        self.conversation_history = []
        self.assistant = None
        self.thread = None
        
        # Define available tools
        self.tools = [
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
        
        # Initialize the assistant and thread
        self._initialize_assistant()
        
        logger.info(f"CLI agent initialized with model {model}")
    
    def _initialize_assistant(self):
        """Initialize the OpenAI Assistant and Thread"""
        try:
            # Create or retrieve the assistant
            assistants = client.beta.assistants.list(
                order="desc",
                limit=10
            )
            
            # Check if we already have an assistant with the same name
            for assistant in assistants.data:
                if assistant.name == "Data Analysis Assistant":
                    self.assistant = assistant
                    logger.info(f"Using existing assistant: {assistant.id}")
                    break
            
            # Create a new assistant if none exists
            if not self.assistant:
                self.assistant = client.beta.assistants.create(
                    name="Data Analysis Assistant",
                    description="A data analysis assistant that can help with analyzing data, creating visualizations, and more.",
                    model=self.model,
                    tools=self.tools,
                    instructions="""You are a helpful data analysis assistant.
You can help users analyze data, create visualizations, and perform various data-related tasks.
When users ask for data analysis, always think step by step and explain your reasoning.
You have access to tools for loading data, creating visualizations, and analyzing text."""
                )
                logger.info(f"Created new assistant: {self.assistant.id}")
            
            # Create a new thread for this conversation
            self.thread = client.beta.threads.create()
            logger.info(f"Created new thread: {self.thread.id}")
            
        except Exception as e:
            logger.error(f"Error initializing assistant: {e}")
            raise
    
    def _handle_tool_call(self, tool_call) -> Dict:
        """Handle tool calls from the assistant"""
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        logger.info(f"Handling tool call: {name} with arguments {arguments}")
        
        if name == "load_csv":
            return self.data_tools.load_csv(**arguments)
        elif name == "plot_data":
            return self.data_tools.plot_data(**arguments)
        elif name == "analyze_text":
            return self.data_tools.analyze_text(**arguments)
        elif name == "list_modal_functions":
            return self.modal.list_functions()
        elif name == "call_modal_function":
            return self.modal.call_function(**arguments)
        else:
            return {
                "success": False,
                "message": f"Unknown tool: {name}",
                "data": None
            }
    
    def _wait_for_run(self, run: Run) -> Run:
        """Wait for a run to complete"""
        while run.status in ["queued", "in_progress"]:
            with console.status(f"[bold green]Thinking... (status: {run.status})"):
                # Wait a bit before checking again
                time.sleep(1)
                # Get the updated run
                run = client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=run.id
                )
        return run
    
    def _process_tool_calls(self, run: Run) -> Run:
        """Process tool calls from the assistant"""
        tool_outputs = []
        
        # Get the required action
        required_action = run.required_action
        
        if required_action and required_action.type == "submit_tool_outputs":
            # Process each tool call
            for tool_call in required_action.submit_tool_outputs.tool_calls:
                with console.status(f"[bold blue]Running tool: {tool_call.function.name}..."):
                    # Execute the function
                    function_response = self._handle_tool_call(tool_call)
                    
                    # Add the function response to the tool outputs
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps(function_response)
                    })
            
            # Submit the tool outputs
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=self.thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            
            # Wait for the run to complete
            run = self._wait_for_run(run)
            
            # Check if there are more tool calls
            if run.status == "requires_action":
                run = self._process_tool_calls(run)
        
        return run
    
    async def chat(self, message: str) -> str:
        """Chat with the agent using the Assistants API"""
        try:
            # Add the user message to the thread
            client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=message
            )
            
            # Run the assistant
            with console.status("[bold green]Starting assistant..."):
                run = client.beta.threads.runs.create(
                    thread_id=self.thread.id,
                    assistant_id=self.assistant.id
                )
            
            # Wait for the run to complete or require action
            run = self._wait_for_run(run)
            
            # Process any tool calls
            if run.status == "requires_action":
                run = self._process_tool_calls(run)
            
            # Check if the run completed successfully
            if run.status == "completed":
                # Get the latest messages (the assistant's response)
                messages = client.beta.threads.messages.list(
                    thread_id=self.thread.id,
                    order="desc",
                    limit=1
                )
                
                if messages.data:
                    # Extract the content from the message
                    message_content = ""
                    for content_item in messages.data[0].content:
                        if content_item.type == "text":
                            message_content += content_item.text.value
                    
                    return message_content
                else:
                    return "No response from the assistant."
            else:
                return f"Run failed with status: {run.status}"
                
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error: {str(e)}"

def main():
    """Main function for the CLI agent"""
    parser = argparse.ArgumentParser(description="CLI Agent for Data Analysis")
    parser.add_argument("--model", default="gpt-4o", help="Model to use for the agent")
    parser.add_argument("--trace", action="store_true", help="Enable tracing for debugging")
    args = parser.parse_args()
    
    # Enable tracing if requested
    if args.trace:
        openai.debug.trace.enable()
    
    # Create the agent
    agent = CLIAgent(model=args.model)
    
    # Welcome message
    console.print(Panel.fit(
        "[bold blue]Welcome to the Data Analysis CLI Agent![/bold blue]\n"
        "You can chat with me about data analysis tasks, and I'll help you analyze data, "
        "create visualizations, and more.\n"
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
            
        except KeyboardInterrupt:
            console.print("\n[bold blue]Goodbye![/bold blue]")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

import time

if __name__ == "__main__":
    main()
