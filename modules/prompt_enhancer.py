#!/usr/bin/env python3
"""
Prompt Enhancer - Dynamic system prompt management and enhancement

Provides tools for automatically enhancing system prompts based on:
- Available tools and capabilities
- Interaction history and context
- User preferences and feedback
- Task-specific optimizations

The module enables the agent to evolve its capabilities over time
by adapting its system prompt to better serve user needs.
"""

import os
import sys
import json
import re
import time
import inspect
import importlib
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable

# === PROMPT MANAGER ===
class PromptManager:
    """
    Manages and enhances system prompts dynamically
    """
    def __init__(self, 
                base_prompt: str,
                max_prompt_length: int = 4000,
                capability_scan_interval: int = 300,  # 5 minutes
                history_window: int = 10):
        """
        Initialize the prompt manager
        
        Args:
            base_prompt: Base system prompt to build upon
            max_prompt_length: Maximum length for the system prompt
            capability_scan_interval: How often to scan for new capabilities in seconds
            history_window: Number of recent interactions to consider
        """
        self.base_prompt = base_prompt
        self.max_prompt_length = max_prompt_length
        self.capability_scan_interval = capability_scan_interval
        self.history_window = history_window
        
        # Store different prompt components
        self.sections = {
            "base": base_prompt,
            "capabilities": "",
            "tools": {},
            "examples": {},
            "context": "",
            "feedback": {},
            "preferences": {}
        }
        
        # Track last capability scan
        self.last_capability_scan = 0
        
        # Initialize interaction history
        self.interaction_history = []
        
    def add_tool(self, name: str, description: str, usage_example: str) -> None:
        """
        Add a tool to the prompt
        
        Args:
            name: Tool name
            description: Tool description
            usage_example: Example of tool usage
        """
        self.sections["tools"][name] = {
            "description": description,
            "example": usage_example,
            "added_at": time.time()
        }
        
    def add_example(self, task: str, example: str) -> None:
        """
        Add an example to the prompt
        
        Args:
            task: Task description
            example: Example solution
        """
        self.sections["examples"][task] = {
            "example": example,
            "added_at": time.time(),
            "usage_count": 0
        }
        
    def add_user_preference(self, preference: str, value: Any) -> None:
        """
        Add a user preference
        
        Args:
            preference: Preference name
            value: Preference value
        """
        self.sections["preferences"][preference] = {
            "value": value,
            "added_at": time.time()
        }
        
    def add_feedback(self, feedback_type: str, feedback: str) -> None:
        """
        Add user feedback
        
        Args:
            feedback_type: Type of feedback
            feedback: Feedback content
        """
        if feedback_type not in self.sections["feedback"]:
            self.sections["feedback"][feedback_type] = []
            
        self.sections["feedback"][feedback_type].append({
            "content": feedback,
            "added_at": time.time()
        })
        
        # Remove old feedback if there are too many
        if len(self.sections["feedback"][feedback_type]) > 3:
            self.sections["feedback"][feedback_type].sort(key=lambda x: x["added_at"])
            self.sections["feedback"][feedback_type] = self.sections["feedback"][feedback_type][-3:]
            
    def add_interaction(self, user_message: str, assistant_response: str) -> None:
        """
        Add an interaction to history
        
        Args:
            user_message: User message
            assistant_response: Assistant response
        """
        self.interaction_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": time.time()
        })
        
        # Keep only the most recent interactions
        if len(self.interaction_history) > self.history_window:
            self.interaction_history = self.interaction_history[-self.history_window:]
            
    def scan_for_capabilities(self, modules_dir: str = "modules") -> None:
        """
        Scan for available capabilities in modules
        
        Args:
            modules_dir: Directory containing modules
        """
        # Check if we need to scan
        current_time = time.time()
        if current_time - self.last_capability_scan < self.capability_scan_interval:
            return
            
        self.last_capability_scan = current_time
        capabilities = []
        
        # Get the full path to modules directory
        if not os.path.isabs(modules_dir):
            modules_dir = os.path.join(os.getcwd(), modules_dir)
            
        # Check if directory exists
        if not os.path.exists(modules_dir) or not os.path.isdir(modules_dir):
            return
            
        # Iterate through Python files
        for filename in os.listdir(modules_dir):
            if not filename.endswith(".py") or filename.startswith("__"):
                continue
                
            module_name = filename[:-3]  # Remove .py extension
            module_path = os.path.join(modules_dir, filename)
            
            try:
                # Extract module docstring
                with open(module_path, 'r') as f:
                    content = f.read()
                    
                # Try to find the module docstring
                docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                if docstring_match:
                    docstring = docstring_match.group(1).strip()
                    # Extract the first line as a short description
                    short_desc = docstring.split('\n')[0].strip()
                    
                    # Find function definitions
                    function_matches = re.finditer(r'def\s+([a-zA-Z_0-9]+)\s*\((.*?)\)\s*->.*?:(?:\s*"""(.*?)""")?', 
                                                  content, re.DOTALL)
                    
                    functions = []
                    for match in function_matches:
                        func_name = match.group(1)
                        
                        # Skip private functions
                        if func_name.startswith('_'):
                            continue
                            
                        func_docstring = match.group(3) if match.group(3) else ""
                        func_desc = func_docstring.split('\n')[0].strip() if func_docstring else func_name
                        
                        functions.append({
                            "name": func_name,
                            "description": func_desc
                        })
                        
                    capabilities.append({
                        "module": module_name,
                        "description": short_desc,
                        "functions": functions
                    })
            except Exception as e:
                print(f"Error scanning module {module_name}: {e}")
                continue
                
        # Update capabilities section
        if capabilities:
            self.sections["capabilities"] = self._format_capabilities(capabilities)
            
    def _format_capabilities(self, capabilities: List[Dict]) -> str:
        """
        Format capabilities into a string
        
        Args:
            capabilities: List of capability dictionaries
            
        Returns:
            Formatted capabilities string
        """
        formatted = "## Available Capabilities\n\n"
        
        for cap in capabilities:
            formatted += f"### {cap['module']}\n"
            formatted += f"{cap['description']}\n\n"
            
            if cap['functions']:
                formatted += "Functions:\n"
                for func in cap['functions']:
                    formatted += f"- `{func['name']}`: {func['description']}\n"
                formatted += "\n"
                
        return formatted
        
    def generate_prompt(self, context: Optional[str] = None) -> str:
        """
        Generate the enhanced system prompt
        
        Args:
            context: Current conversation context
            
        Returns:
            Enhanced system prompt
        """
        # Start with the base prompt
        prompt_parts = [self.sections["base"]]
        
        # Add capabilities section if available
        if self.sections["capabilities"]:
            prompt_parts.append(self.sections["capabilities"])
            
        # Add tools section
        if self.sections["tools"]:
            tools_text = "## Available Tools\n\n"
            
            # Sort tools by recent addition
            sorted_tools = sorted(self.sections["tools"].items(), 
                                 key=lambda x: x[1]["added_at"], 
                                 reverse=True)
            
            # Add the most recent 5 tools
            for name, tool in sorted_tools[:5]:
                tools_text += f"### {name}\n"
                tools_text += f"{tool['description']}\n"
                tools_text += f"Example: ```\n{tool['example']}\n```\n\n"
                
            prompt_parts.append(tools_text)
            
        # Add examples section, prioritizing less-used examples
        if self.sections["examples"]:
            examples_text = "## Examples\n\n"
            
            # Sort examples by usage count, then recency
            sorted_examples = sorted(self.sections["examples"].items(), 
                                    key=lambda x: (x[1]["usage_count"], -x[1]["added_at"]))
            
            # Add up to 2 examples
            for task, example in sorted_examples[:2]:
                examples_text += f"### {task}\n"
                examples_text += f"```\n{example['example']}\n```\n\n"
                
                # Increment usage count
                self.sections["examples"][task]["usage_count"] += 1
                
            prompt_parts.append(examples_text)
            
        # Add user preferences
        if self.sections["preferences"]:
            prefs_text = "## User Preferences\n\n"
            
            for pref, details in self.sections["preferences"].items():
                value = details["value"]
                value_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                prefs_text += f"- {pref}: {value_str}\n"
                
            prompt_parts.append(prefs_text)
            
        # Add recent feedback
        if self.sections["feedback"]:
            feedback_text = "## Recent Feedback\n\n"
            
            for feedback_type, items in self.sections["feedback"].items():
                if items:
                    feedback_text += f"### {feedback_type}\n"
                    for item in items:
                        feedback_text += f"- {item['content']}\n"
                    feedback_text += "\n"
                    
            prompt_parts.append(feedback_text)
            
        # Add context if provided
        if context:
            # Use provided context
            context_text = f"## Current Context\n\n{context}\n"
        elif self.interaction_history:
            # Or generate from interaction history
            context_text = "## Recent Interactions\n\n"
            
            # Get the 3 most recent interactions
            recent = self.interaction_history[-3:]
            
            for interaction in recent:
                user_msg = interaction["user"]
                # Truncate if too long
                if len(user_msg) > 100:
                    user_msg = user_msg[:97] + "..."
                    
                context_text += f"User: {user_msg}\n"
                
                # We don't need the full assistant response, just a hint
                asst_msg = interaction["assistant"]
                if len(asst_msg) > 100:
                    asst_msg = asst_msg[:97] + "..."
                    
                context_text += f"Assistant: {asst_msg}\n\n"
                
            prompt_parts.append(context_text)
            
        # Combine all parts
        full_prompt = "\n".join(prompt_parts)
        
        # Ensure we don't exceed maximum length
        if len(full_prompt) > self.max_prompt_length:
            # Try to trim the prompt while keeping the base and capabilities
            essential_parts = [self.sections["base"], self.sections["capabilities"]]
            essential_prompt = "\n".join(part for part in essential_parts if part)
            
            # Calculate how much space we have for the rest
            remaining_space = self.max_prompt_length - len(essential_prompt) - 100  # 100 for separator
            
            if remaining_space > 500:  # Minimum reasonable size
                # Prioritize tool info and preferences over examples and history
                priority_parts = []
                
                # Add tools section (truncated if needed)
                if self.sections["tools"]:
                    tools_text = "## Available Tools\n\n"
                    
                    # Only include 2 most recent tools to save space
                    sorted_tools = sorted(self.sections["tools"].items(), 
                                         key=lambda x: x[1]["added_at"], 
                                         reverse=True)
                    
                    for name, tool in sorted_tools[:2]:
                        tools_text += f"### {name}\n"
                        tools_text += f"{tool['description']}\n"
                        tools_text += f"Example: ```\n{tool['example']}\n```\n\n"
                        
                    priority_parts.append(tools_text)
                    
                # Add user preferences (they're usually short)
                if self.sections["preferences"]:
                    prefs_text = "## User Preferences\n\n"
                    
                    for pref, details in self.sections["preferences"].items():
                        value = details["value"]
                        value_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                        prefs_text += f"- {pref}: {value_str}\n"
                        
                    priority_parts.append(prefs_text)
                    
                # Combine priority parts with essential parts
                priority_prompt = "\n".join(priority_parts)
                
                if len(priority_prompt) <= remaining_space:
                    full_prompt = essential_prompt + "\n\n" + priority_prompt
                else:
                    # If still too long, just use essential parts
                    full_prompt = essential_prompt
            else:
                # If we have very little space, just use the essential prompt
                full_prompt = essential_prompt
                
        return full_prompt
        
    def save_to_file(self, file_path: str) -> bool:
        """
        Save the current prompt configuration to a file
        
        Args:
            file_path: Path to save the configuration
            
        Returns:
            Success status
        """
        try:
            # Create a copy of sections with only serializable data
            serializable_sections = {
                "base": self.sections["base"],
                "capabilities": self.sections["capabilities"],
                "tools": self.sections["tools"],
                "examples": self.sections["examples"],
                "context": self.sections["context"],
                "feedback": self.sections["feedback"],
                "preferences": self.sections["preferences"]
            }
            
            # Create parent directory if needed
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump({
                    "sections": serializable_sections,
                    "last_capability_scan": self.last_capability_scan,
                    "interaction_history": self.interaction_history,
                    "max_prompt_length": self.max_prompt_length,
                    "capability_scan_interval": self.capability_scan_interval,
                    "history_window": self.history_window
                }, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving prompt configuration: {e}")
            return False
            
    def load_from_file(self, file_path: str) -> bool:
        """
        Load prompt configuration from a file
        
        Args:
            file_path: Path to load the configuration from
            
        Returns:
            Success status
        """
        try:
            if not os.path.exists(file_path):
                return False
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Load the data
            self.sections = data["sections"]
            self.last_capability_scan = data["last_capability_scan"]
            self.interaction_history = data["interaction_history"]
            self.max_prompt_length = data["max_prompt_length"]
            self.capability_scan_interval = data["capability_scan_interval"]
            self.history_window = data["history_window"]
            
            return True
        except Exception as e:
            print(f"Error loading prompt configuration: {e}")
            return False
            
    def reset_to_base(self) -> None:
        """Reset the prompt to the base prompt only"""
        self.sections = {
            "base": self.base_prompt,
            "capabilities": "",
            "tools": {},
            "examples": {},
            "context": "",
            "feedback": {},
            "preferences": {}
        }
        self.interaction_history = []
        self.last_capability_scan = 0

# === DYNAMIC CAPABILITY INFERENCE ===
class CapabilityInference:
    """
    Infers capabilities from module inspection and user interactions
    """
    def __init__(self, modules_dir: str = "modules"):
        """
        Initialize the capability inference
        
        Args:
            modules_dir: Directory containing modules
        """
        self.modules_dir = modules_dir
        self.capabilities = {}
        
    def scan_modules(self) -> Dict[str, Any]:
        """
        Scan modules for capabilities
        
        Returns:
            Dictionary of capabilities
        """
        capabilities = {}
        
        # Get the full path to modules directory
        if not os.path.isabs(self.modules_dir):
            modules_dir = os.path.join(os.getcwd(), self.modules_dir)
        else:
            modules_dir = self.modules_dir
            
        # Check if directory exists
        if not os.path.exists(modules_dir) or not os.path.isdir(modules_dir):
            return capabilities
            
        # Get a list of Python files
        module_files = [f for f in os.listdir(modules_dir) 
                      if f.endswith('.py') and not f.startswith('__')]
        
        for filename in module_files:
            module_name = filename[:-3]  # Remove .py extension
            
            try:
                # Import the module dynamically
                spec = importlib.util.spec_from_file_location(
                    module_name, 
                    os.path.join(modules_dir, filename)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Get module docstring
                module_doc = inspect.getdoc(module) or ""
                
                # Extract functions
                functions = []
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and not name.startswith('_'):
                        # Get function signature
                        sig = inspect.signature(obj)
                        
                        # Get function docstring
                        func_doc = inspect.getdoc(obj) or ""
                        short_desc = func_doc.split('\n')[0] if func_doc else name
                        
                        # Get parameters
                        params = []
                        for param_name, param in sig.parameters.items():
                            if param_name == 'self':
                                continue
                                
                            param_type = "Any"
                            if param.annotation != inspect.Parameter.empty:
                                param_type = str(param.annotation).replace("<class '", "").replace("'>", "")
                                
                            default_value = None
                            if param.default != inspect.Parameter.empty:
                                default_value = param.default
                                
                            params.append({
                                "name": param_name,
                                "type": param_type,
                                "default": default_value,
                                "optional": param.default != inspect.Parameter.empty
                            })
                            
                        # Get return type
                        return_type = "None"
                        if sig.return_annotation != inspect.Signature.empty:
                            return_type = str(sig.return_annotation).replace("<class '", "").replace("'>", "")
                            
                        functions.append({
                            "name": name,
                            "description": short_desc,
                            "parameters": params,
                            "return_type": return_type,
                            "docstring": func_doc
                        })
                        
                # Store module info
                capabilities[module_name] = {
                    "description": module_doc.split('\n')[0] if module_doc else module_name,
                    "docstring": module_doc,
                    "functions": functions
                }
                
            except Exception as e:
                print(f"Error scanning module {module_name}: {e}")
                continue
                
        self.capabilities = capabilities
        return capabilities
        
    def generate_function_examples(self, function_info: Dict) -> str:
        """
        Generate usage examples for a function
        
        Args:
            function_info: Function information
            
        Returns:
            Example code
        """
        function_name = function_info["name"]
        params = function_info["parameters"]
        
        # Generate example parameter values
        example_args = []
        for param in params:
            param_name = param["name"]
            param_type = param["type"]
            
            # Skip optional parameters for simplicity
            if param["optional"]:
                continue
                
            # Generate example value based on type
            if param_type == "str" or param_type == "string":
                if "path" in param_name.lower() or "file" in param_name.lower():
                    example_args.append(f'{param_name}="/path/to/file.txt"')
                elif "url" in param_name.lower():
                    example_args.append(f'{param_name}="https://example.com"')
                elif "name" in param_name.lower():
                    example_args.append(f'{param_name}="example_name"')
                elif "code" in param_name.lower():
                    example_args.append(f'{param_name}="print(\'Hello world\')"')
                else:
                    example_args.append(f'{param_name}="example"')
            elif param_type == "int" or param_type == "float":
                example_args.append(f'{param_name}=10')
            elif param_type == "bool":
                example_args.append(f'{param_name}=True')
            elif param_type == "list" or "List" in param_type:
                example_args.append(f'{param_name}=[]')
            elif param_type == "dict" or "Dict" in param_type:
                example_args.append(f'{param_name}={{}}')
            else:
                example_args.append(f'{param_name}=None')
                
        # Build the example code
        example = f"result = {function_name}({', '.join(example_args)})\n"
        
        # Add result handling
        if "dict" in function_info["return_type"].lower():
            example += "if result['status'] == 'success':\n"
            example += "    # Process result\n"
            example += "    print(result['output'])\n"
        else:
            example += "# Process result\n"
            example += "print(result)\n"
            
        return example
        
    def generate_capability_examples(self) -> Dict[str, str]:
        """
        Generate examples for all capabilities
        
        Returns:
            Dictionary of module names to examples
        """
        examples = {}
        
        for module_name, module_info in self.capabilities.items():
            if not module_info["functions"]:
                continue
                
            # Pick a representative function
            main_functions = [f for f in module_info["functions"] 
                             if not f["name"].startswith('_')]
            
            if not main_functions:
                continue
                
            # Sort by complexity (number of params) to find a good example
            main_functions.sort(key=lambda f: len(f["parameters"]))
            
            # Get a function in the middle of complexity
            function_idx = min(1, len(main_functions) - 1)
            function = main_functions[function_idx]
            
            # Generate example
            example = self.generate_function_examples(function)
            examples[module_name] = example
            
        return examples
        
    def get_tool_summaries(self) -> List[Dict]:
        """
        Get summaries of all available tools
        
        Returns:
            List of tool summary dictionaries
        """
        tools = []
        
        for module_name, module_info in self.capabilities.items():
            for function in module_info["functions"]:
                # Create a tool summary
                tool_summary = {
                    "name": function["name"],
                    "module": module_name,
                    "description": function["description"],
                    "example": self.generate_function_examples(function)
                }
                tools.append(tool_summary)
                
        return tools
        
    def enhance_prompt_manager(self, prompt_manager: PromptManager) -> None:
        """
        Enhance a prompt manager with inferred capabilities
        
        Args:
            prompt_manager: PromptManager instance to enhance
        """
        # Scan modules if needed
        if not self.capabilities:
            self.scan_modules()
            
        # Get tool summaries
        tools = self.get_tool_summaries()
        
        # Add tools to prompt manager
        for tool in tools:
            prompt_manager.add_tool(
                tool["name"], 
                tool["description"],
                tool["example"]
            )
            
        # Generate capability examples
        examples = self.generate_capability_examples()
        
        # Add examples to prompt manager
        for module_name, example in examples.items():
            if module_info := self.capabilities.get(module_name):
                task = f"Using {module_name}"
                prompt_manager.add_example(task, example)

# === UTILITY FUNCTIONS ===
def create_prompt_manager(base_prompt: str) -> PromptManager:
    """
    Create a prompt manager with base prompt
    
    Args:
        base_prompt: Base system prompt
        
    Returns:
        PromptManager instance
    """
    return PromptManager(base_prompt)

def infer_capabilities() -> CapabilityInference:
    """
    Create a capability inference instance
    
    Returns:
        CapabilityInference instance
    """
    inference = CapabilityInference()
    inference.scan_modules()
    return inference

def enhance_system_prompt(base_prompt: str, context: Optional[str] = None) -> str:
    """
    Enhance a system prompt with capabilities
    
    Args:
        base_prompt: Base system prompt
        context: Optional conversation context
        
    Returns:
        Enhanced system prompt
    """
    # Create prompt manager
    prompt_manager = create_prompt_manager(base_prompt)
    
    # Scan for capabilities
    prompt_manager.scan_for_capabilities()
    
    # Create capability inference
    inference = infer_capabilities()
    
    # Enhance prompt manager with inferred capabilities
    inference.enhance_prompt_manager(prompt_manager)
    
    # Generate enhanced prompt
    return prompt_manager.generate_prompt(context)

# === COMMAND-LINE INTERFACE ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prompt Enhancer")
    parser.add_argument("--base-prompt", help="Base system prompt file")
    parser.add_argument("--output", help="Output file for enhanced prompt")
    parser.add_argument("--scan-modules", action="store_true", help="Scan for modules")
    parser.add_argument("--context", help="Context file")
    
    args = parser.parse_args()
    
    if args.base_prompt:
        try:
            with open(args.base_prompt, 'r') as f:
                base_prompt = f.read()
        except Exception as e:
            print(f"Error reading base prompt: {e}")
            sys.exit(1)
    else:
        base_prompt = """You are an advanced AI assistant with various capabilities.
You can help users with a wide range of tasks using your tools and abilities.
Answer questions clearly and helpfully."""
        
    context = None
    if args.context and os.path.exists(args.context):
        try:
            with open(args.context, 'r') as f:
                context = f.read()
        except Exception as e:
            print(f"Error reading context: {e}")
            
    # Enhance the prompt
    enhanced_prompt = enhance_system_prompt(base_prompt, context)
    
    # Output the result
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(enhanced_prompt)
            print(f"Enhanced prompt saved to {args.output}")
        except Exception as e:
            print(f"Error writing output: {e}")
            print(enhanced_prompt)
    else:
        print(enhanced_prompt)