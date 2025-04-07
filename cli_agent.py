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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cli-agent")

# Rich console for better formatting
console = Console()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    def __init__(self, model="gpt-4"):
        self.model = model
        self.data_tools = DataAnalysisTools()
        self.modal = ModalIntegration()
        self.conversation_history = []
        
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
        
        logger.info(f"CLI agent initialized with model {model}")
    
    def _handle_tool_call(self, name: str, arguments: Dict) -> Dict:
        """Handle tool calls from the agent"""
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
    
    async def chat(self, message: str) -> str:
        """Chat with the agent"""
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Run the agent
        with console.status("[bold green]Thinking..."):
            try:
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
                        with console.status(f"[bold blue]Running tool: {function_name}..."):
                            function_response = self._handle_tool_call(function_name, function_args)
                        
                        # Add the function response to the conversation
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(function_response)
                        })
                    
                    # Get the final response after tool calls
                    with console.status("[bold green]Processing results..."):
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
                    
                    return content
                    
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                return f"Error: {str(e)}"

def main():
    """Main function for the CLI agent"""
    parser = argparse.ArgumentParser(description="CLI Agent for Data Analysis")
    parser.add_argument("--model", default="gpt-4", help="Model to use for the agent")
    args = parser.parse_args()
    
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

if __name__ == "__main__":
    main()
