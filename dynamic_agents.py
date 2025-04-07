import os
import sys
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dynamic_agents")

# Agent response type
AgentResponse = Dict[str, Any]

@dataclass
class AgentContext:
    """Base context for all agents"""
    agent_id: str
    working_dir: Path = field(default_factory=lambda: Path.cwd())
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_to_history(self, action: str, details: Dict[str, Any]) -> None:
        """Add an action to history with timestamp"""
        self.history.append({
            "action": action,
            "details": details,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable with fallback default"""
        return self.variables.get(name, default)
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the context"""
        self.variables[name] = value
        self.add_to_history("set_variable", {"name": name, "value": str(value)[:100]})

class DynamicAgent:
    """Base class for all dynamic agents"""
    
    def __init__(self, agent_id: str, agent_type: str, description: str = ""):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.description = description
        self.capabilities = {}
        self._register_capabilities()
    
    def _register_capabilities(self) -> None:
        """Register agent capabilities - override in subclasses"""
        # Register built-in capabilities
        self.register_capability(
            "help", 
            self.cmd_help,
            "Show available commands and capabilities"
        )
    
    def register_capability(self, name: str, func: Callable, description: str = "") -> None:
        """Register a new capability for this agent"""
        self.capabilities[name] = {
            "function": func,
            "description": description
        }
        logger.info(f"Agent {self.agent_id} registered capability: {name}")
    
    async def execute(self, command: str, context: AgentContext) -> AgentResponse:
        """Execute a command with the given context"""
        logger.info(f"Agent {self.agent_id} executing: {command}")
        
        try:
            command_parts = command.strip().split(" ", 1)
            action = command_parts[0].lower()
            args = command_parts[1] if len(command_parts) > 1 else ""
            
            # Check if capability exists
            if action not in self.capabilities:
                return {
                    "success": False,
                    "error": f"Unknown command '{action}' for agent {self.agent_id}"
                }
            
            # Call the appropriate capability function
            capability = self.capabilities[action]
            result = await capability["function"](args, context)
            
            # Log successful actions for auditing
            context.add_to_history(action, {
                "command": command,
                "success": result.get("success", False)
            })
                
            return result
            
        except Exception as e:
            logger.exception(f"Error executing command: {command}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Built-in help command
    async def cmd_help(self, args: str, context: AgentContext) -> AgentResponse:
        """Show available commands"""
        capabilities = []
        
        for name, capability in self.capabilities.items():
            capabilities.append({
                "name": name,
                "description": capability["description"]
            })
        
        return {
            "success": True,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "description": self.description,
            "capabilities": capabilities
        }

class FileAgent(DynamicAgent):
    """Agent for file system operations"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "file", "File system operations agent")
    
    def _register_capabilities(self) -> None:
        """Register file agent capabilities"""
        super()._register_capabilities()
        
        self.register_capability("ls", self.cmd_ls, "List directory contents")
        self.register_capability("read", self.cmd_read, "Read a file's contents")
        self.register_capability("write", self.cmd_write, "Write content to a file")
        self.register_capability("append", self.cmd_append, "Append content to a file")
        self.register_capability("mkdir", self.cmd_mkdir, "Create a directory")
    
    async def cmd_ls(self, args: str, context: AgentContext) -> AgentResponse:
        """List directory contents"""
        target_dir = context.working_dir
        if args:
            target_dir = target_dir / args
            
        if not target_dir.exists():
            return {
                "success": False,
                "error": f"Directory not found: {target_dir}"
            }
            
        files = []
        for item in target_dir.iterdir():
            files.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0
            })
            
        return {
            "success": True,
            "path": str(target_dir),
            "items": files
        }
    
    async def cmd_read(self, args: str, context: AgentContext) -> AgentResponse:
        """Read a file's contents"""
        if not args:
            return {
                "success": False,
                "error": "No file specified"
            }
            
        file_path = Path(args) if args.startswith('/') else context.working_dir / args
        
        if not file_path.exists() or not file_path.is_file():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }
            
        try:
            content = file_path.read_text()
            context.set_variable("current_file", str(file_path))
            context.set_variable(f"file_content_{file_path.name}", content)
            
            return {
                "success": True,
                "file": str(file_path),
                "content": content
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read file: {e}"
            }
    
    async def cmd_write(self, args: str, context: AgentContext) -> AgentResponse:
        """Write content to a file (format: filename content)"""
        parts = args.split(' ', 1)
        if len(parts) < 2:
            return {
                "success": False,
                "error": "Usage: write <filename> <content>"
            }
            
        filename, content = parts
        
        file_path = Path(filename) if filename.startswith('/') else context.working_dir / filename
            
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            context.set_variable("current_file", str(file_path))
            context.set_variable(f"file_content_{file_path.name}", content)
            
            return {
                "success": True,
                "file": str(file_path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write file: {e}"
            }
    
    async def cmd_append(self, args: str, context: AgentContext) -> AgentResponse:
        """Append content to a file (format: filename content)"""
        parts = args.split(' ', 1)
        if len(parts) < 2:
            return {
                "success": False,
                "error": "Usage: append <filename> <content>"
            }
            
        filename, content = parts
        
        file_path = Path(filename) if filename.startswith('/') else context.working_dir / filename
            
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If file exists, read it first, otherwise start with empty string
            existing_content = ""
            if file_path.exists():
                existing_content = file_path.read_text()
            
            # Append and write
            new_content = existing_content + content
            file_path.write_text(new_content)
            
            context.set_variable("current_file", str(file_path))
            context.set_variable(f"file_content_{file_path.name}", new_content)
            
            return {
                "success": True,
                "file": str(file_path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to append to file: {e}"
            }
    
    async def cmd_mkdir(self, args: str, context: AgentContext) -> AgentResponse:
        """Create a new directory"""
        if not args:
            return {
                "success": False,
                "error": "No directory name specified"
            }
            
        dir_path = Path(args) if args.startswith('/') else context.working_dir / args
            
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            return {
                "success": True,
                "directory": str(dir_path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create directory: {e}"
            }

class DataAnalysisAgent(DynamicAgent):
    """Agent for data analysis operations"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "data_analysis", "Data analysis and visualization agent")
    
    def _register_capabilities(self) -> None:
        """Register data analysis capabilities"""
        super()._register_capabilities()
        
        self.register_capability("analyze_csv", self.cmd_analyze_csv, "Analyze a CSV file")
        self.register_capability("visualize", self.cmd_visualize, "Create a visualization")
        self.register_capability("summarize", self.cmd_summarize, "Summarize data")
    
    async def cmd_analyze_csv(self, args: str, context: AgentContext) -> AgentResponse:
        """Analyze a CSV file and return statistics"""
        if not args:
            return {
                "success": False,
                "error": "No file specified"
            }
        
        try:
            import pandas as pd
            
            file_path = Path(args) if args.startswith('/') else context.working_dir / args
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }
            
            # Load the CSV
            df = pd.read_csv(file_path)
            
            # Generate statistics
            stats = {
                "rows": len(df),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "missing_values": df.isnull().sum().to_dict(),
                "summary": df.describe().to_dict()
            }
            
            # Store in context
            context.set_variable(f"dataframe_{file_path.stem}", df)
            context.set_variable(f"stats_{file_path.stem}", stats)
            
            return {
                "success": True,
                "file": str(file_path),
                "statistics": stats
            }
        except ImportError:
            return {
                "success": False,
                "error": "Required packages not installed: pandas"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to analyze CSV: {e}"
            }
    
    async def cmd_visualize(self, args: str, context: AgentContext) -> AgentResponse:
        """Create a visualization from data (format: data_source plot_type [options_json])"""
        parts = args.split(' ', 2)
        if len(parts) < 2:
            return {
                "success": False,
                "error": "Usage: visualize <data_source> <plot_type> [options_json]"
            }
        
        data_source = parts[0]
        plot_type = parts[1]
        options = {}
        
        if len(parts) > 2:
            try:
                options = json.loads(parts[2])
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Invalid JSON for options"
                }
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get the data
            df = context.get_variable(f"dataframe_{data_source}")
            if df is None:
                return {
                    "success": False,
                    "error": f"Data source not found: {data_source}"
                }
            
            # Create output directory
            output_dir = context.working_dir / "visualizations"
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename
            filename = f"{data_source}_{plot_type}.png"
            output_path = output_dir / filename
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            
            if plot_type == "histogram":
                column = options.get("column")
                if not column:
                    return {"success": False, "error": "Column name required for histogram"}
                sns.histplot(data=df, x=column, kde=options.get("kde", True))
                
            elif plot_type == "scatter":
                x = options.get("x")
                y = options.get("y")
                if not (x and y):
                    return {"success": False, "error": "X and Y columns required for scatter plot"}
                sns.scatterplot(data=df, x=x, y=y, hue=options.get("hue"))
                
            elif plot_type == "bar":
                x = options.get("x")
                y = options.get("y")
                if not (x and y):
                    return {"success": False, "error": "X and Y columns required for bar plot"}
                sns.barplot(data=df, x=x, y=y)
                
            elif plot_type == "heatmap":
                # Create correlation matrix
                corr = df.select_dtypes(include=['number']).corr()
                sns.heatmap(corr, annot=True, cmap=options.get("cmap", "coolwarm"))
                
            else:
                return {
                    "success": False,
                    "error": f"Unsupported plot type: {plot_type}"
                }
            
            # Add title if provided
            if "title" in options:
                plt.title(options["title"])
                
            # Save the plot
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return {
                "success": True,
                "plot_type": plot_type,
                "output_file": str(output_path)
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "Required packages not installed: pandas, matplotlib, seaborn"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create visualization: {e}"
            }
    
    async def cmd_summarize(self, args: str, context: AgentContext) -> AgentResponse:
        """Summarize data with descriptive statistics"""
        if not args:
            return {
                "success": False,
                "error": "No data source specified"
            }
        
        try:
            import pandas as pd
            
            # Get the data
            df = context.get_variable(f"dataframe_{args}")
            if df is None:
                return {
                    "success": False,
                    "error": f"Data source not found: {args}"
                }
            
            # Generate summary
            summary = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "head": df.head(5).to_dict(orient="records"),
                "describe": df.describe().to_dict(),
                "missing": df.isnull().sum().to_dict()
            }
            
            return {
                "success": True,
                "data_source": args,
                "summary": summary
            }
            
        except ImportError:
            return {
                "success": False,
                "error": "Required packages not installed: pandas"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to summarize data: {e}"
            }

class AgentRegistry:
    """Registry for managing dynamic agents"""
    
    def __init__(self):
        self.agents: Dict[str, DynamicAgent] = {}
        self.agent_types: Dict[str, Type[DynamicAgent]] = {
            "file": FileAgent,
            "data_analysis": DataAnalysisAgent
        }
    
    def register_agent_type(self, type_name: str, agent_class: Type[DynamicAgent]) -> None:
        """Register a new agent type"""
        self.agent_types[type_name] = agent_class
        logger.info(f"Registered agent type: {type_name}")
    
    def create_agent(self, agent_id: str, agent_type: str) -> DynamicAgent:
        """Create a new agent of the specified type"""
        if agent_type not in self.agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        if agent_id in self.agents:
            raise ValueError(f"Agent ID already exists: {agent_id}")
        
        agent_class = self.agent_types[agent_type]
        agent = agent_class(agent_id)
        self.agents[agent_id] = agent
        
        logger.info(f"Created agent: {agent_id} of type {agent_type}")
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[DynamicAgent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, str]]:
        """List all registered agents"""
        return [
            {"id": agent_id, "type": agent.agent_type}
            for agent_id, agent in self.agents.items()
        ]
    
    def list_agent_types(self) -> List[str]:
        """List all available agent types"""
        return list(self.agent_types.keys())

# Global registry instance
registry = AgentRegistry()

async def execute_agent_command(agent_id: str, command: str, context: Optional[AgentContext] = None) -> AgentResponse:
    """Execute a command on an agent"""
    agent = registry.get_agent(agent_id)
    if not agent:
        return {
            "success": False,
            "error": f"Agent not found: {agent_id}"
        }
    
    # Create context if not provided
    if context is None:
        context = AgentContext(agent_id=agent_id)
    
    return await agent.execute(command, context)
