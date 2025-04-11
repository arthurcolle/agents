#!/usr/bin/env python3
"""
Self-Improving Agent - An agent that can improve its own capabilities
through reflection and learning from interactions.
"""

import os
import sys
import json
import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Union
import re
import ast
import inspect

from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

try:
    import anthropic
    import openai
    from agent_process_manager import AgentServer
except ImportError as e:
    logging.error(f"Error importing required modules: {e}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("self-improving-agent")

class ChatRequest(BaseModel):
    """Chat request from user or another agent"""
    message: str
    context: Optional[Dict[str, Any]] = None
    prompt_template: Optional[str] = None

class SelfImprovementRequest(BaseModel):
    """Request to improve a specific capability"""
    capability: str
    description: str
    code_sample: Optional[str] = None

class CodeAnalysisRequest(BaseModel):
    """Request to analyze code for improvement opportunities"""
    code: str
    context: Optional[str] = None

class SelfImprovingAgent(AgentServer):
    """Agent that can improve its own capabilities through learning and reflection"""
    def __init__(self, agent_id=None, agent_name=None, 
                host="127.0.0.1", port=8600, redis_url=None, model="gpt-4"):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name or "Self-Improving Agent",
            agent_type="self_improving",
            host=host,
            port=port,
            redis_url=redis_url,
            model=model
        )
        # Initialize with basic capabilities
        self.capabilities = [
            "chat",
            "self_improvement",
            "code_analysis",
            "function_extraction",
            "memory_management",
            "teach_agent",
            "get_agent_capabilities",
            "request_brain_dump",
            "learn_from_peers"
        ]
        
        # Track improvements made
        self.improvements = []
        
        # Knowledge memory
        self.memory: Dict[str, Any] = {
            "code_patterns": {},
            "learned_concepts": {},
            "interaction_history": [],
            "agent_knowledge": {}  # Store knowledge learned from other agents
        }
        
        # Learning rate - how quickly agent adopts new improvements
        self.learning_rate = 0.8
        
        # Maximum memory entries to keep
        self.max_memory_entries = 1000
        
        # Initialize the LLM client based on the model
        self.init_llm_client()
        
        # Extend FastAPI with additional routes
        self.setup_extended_api()
        
        logger.info(f"Self-Improving Agent {self.agent_id} initialized with model: {self.model}")
    
    def init_llm_client(self):
        """Initialize the appropriate LLM client based on the model name"""
        if self.model.startswith("gpt"):
            # OpenAI model
            self.llm_provider = "openai"
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning("OPENAI_API_KEY environment variable not set")
            self.llm_client = openai.OpenAI(api_key=openai_api_key)
        elif self.model.startswith("claude"):
            # Anthropic model
            self.llm_provider = "anthropic"
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                logger.warning("ANTHROPIC_API_KEY environment variable not set")
            self.llm_client = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            # Default to OpenAI
            self.llm_provider = "openai"
            self.llm_client = openai.OpenAI()
    
    def setup_extended_api(self):
        """Add additional API routes specific to self-improving agent"""
        # Chat endpoint
        @self.app.post("/chat")
        async def chat(request: ChatRequest):
            response = await self.chat(request.message, request.context, request.prompt_template)
            return response
        
        # Self-improvement endpoint
        @self.app.post("/improve")
        async def improve(request: SelfImprovementRequest, background_tasks: BackgroundTasks):
            background_tasks.add_task(
                self.improve_capability, 
                request.capability,
                request.description,
                request.code_sample
            )
            return {
                "status": "improving",
                "capability": request.capability,
                "message": "Self-improvement process started in the background"
            }
        
        # Code analysis endpoint
        @self.app.post("/analyze")
        async def analyze_code(request: CodeAnalysisRequest):
            analysis = await self.analyze_code(request.code, request.context)
            return analysis
        
        # Memory query endpoint
        @self.app.get("/memory")
        async def get_memory(category: Optional[str] = None, key: Optional[str] = None):
            if category and category in self.memory:
                if key and key in self.memory[category]:
                    return {category: {key: self.memory[category][key]}}
                return {category: self.memory[category]}
            return self.memory
        
        # Get capabilities with details
        @self.app.get("/capabilities/details")
        async def get_capabilities_details():
            # Get function implementations for each capability
            capability_details = {}
            for capability in self.capabilities:
                try:
                    # Try to find a method matching the capability name
                    if hasattr(self, capability):
                        func = getattr(self, capability)
                        if callable(func):
                            # Get source code and signature
                            source = inspect.getsource(func)
                            signature = str(inspect.signature(func))
                            docstring = inspect.getdoc(func) or ""
                            
                            capability_details[capability] = {
                                "signature": signature,
                                "docstring": docstring,
                                "source": source
                            }
                except Exception as e:
                    capability_details[capability] = {
                        "error": f"Could not get details: {str(e)}"
                    }
            
            return {
                "capabilities": self.capabilities,
                "details": capability_details
            }
        
        # Add route for uploading improvement code
        @self.app.post("/upload_improvement")
        async def upload_improvement(
            capability: str = Form(...),
            description: str = Form(...),
            code_file: UploadFile = File(...)
        ):
            code_content = await code_file.read()
            code_str = code_content.decode("utf-8")
            
            # Apply the improvement in background
            background_tasks = BackgroundTasks()
            background_tasks.add_task(
                self.improve_capability, 
                capability,
                description,
                code_str
            )
            
            return {
                "status": "processing",
                "message": f"Processing improvement for {capability}",
                "size": len(code_str)
            }
            
        # Learn from peers endpoint
        @self.app.post("/learn_from_peers")
        async def learn_from_peers_endpoint(background_tasks: BackgroundTasks, timeout: int = 30):
            background_tasks.add_task(
                self.learn_from_peers,
                timeout=timeout
            )
            
            return {
                "status": "learning",
                "message": f"Learning from peer agents (timeout: {timeout}s)",
                "started_at": time.time()
            }
        
        # Teach capability to another agent
        @self.app.post("/teach")
        async def teach_capability(
            target_agent_id: str,
            capability: str,
            description: str = "",
            code_implementation: Optional[str] = None
        ):
            result = await self.teach_agent(
                target_agent_id,
                capability,
                code_implementation,
                description
            )
            return result
        
        # Get another agent's capabilities
        @self.app.get("/agent/{agent_id}/capabilities")
        async def get_remote_capabilities(agent_id: str, timeout: int = 10):
            result = await self.get_agent_capabilities(agent_id, timeout)
            return result
        
        # Get brain dump from another agent
        @self.app.get("/agent/{agent_id}/brain_dump")
        async def get_brain_dump(agent_id: str, timeout: int = 20):
            result = await self.request_brain_dump(agent_id, timeout)
            return result
    
    async def process_command(self, command, data, sender):
        """Process custom commands from PubSub"""
        if command == "chat":
            # Handle chat command
            message = data.get("message", "")
            message_id = data.get("message_id")
            
            # Process the chat message
            response = await self.chat(message, context={"sender": sender})
            
            # Send response back
            await self.publish_event(f"agent:{self.agent_id}:responses", {
                "type": "chat_response",
                "message_id": message_id,
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "receiver": sender,
                "message": response.get("response"),
                "data": response,
                "timestamp": time.time()
            })
            
        elif command == "improve":
            # Handle improvement request
            capability = data.get("capability")
            description = data.get("description")
            code_sample = data.get("code_sample")
            
            if capability and description:
                # Start improvement in background
                asyncio.create_task(self.improve_capability(capability, description, code_sample))
                
                # Send acknowledgement
                await self.publish_event(f"agent:{self.agent_id}:responses", {
                    "type": "improvement_started",
                    "agent_id": self.agent_id,
                    "receiver": sender,
                    "capability": capability,
                    "timestamp": time.time()
                })
            else:
                # Send error response
                await self.publish_event(f"agent:{self.agent_id}:responses", {
                    "type": "error",
                    "agent_id": self.agent_id,
                    "receiver": sender,
                    "error": "Missing capability or description for improvement",
                    "timestamp": time.time()
                })
                
        elif command == "analyze_code":
            # Handle code analysis request
            code = data.get("code")
            context = data.get("context")
            
            if code:
                # Analyze the code
                analysis = await self.analyze_code(code, context)
                
                # Send response
                await self.publish_event(f"agent:{self.agent_id}:responses", {
                    "type": "code_analysis",
                    "agent_id": self.agent_id,
                    "receiver": sender,
                    "analysis": analysis,
                    "timestamp": time.time()
                })
            else:
                # Send error response
                await self.publish_event(f"agent:{self.agent_id}:responses", {
                    "type": "error",
                    "agent_id": self.agent_id,
                    "receiver": sender,
                    "error": "Missing code for analysis",
                    "timestamp": time.time()
                })
        
        else:
            # Fall back to parent implementation for unknown commands
            await super().process_command(command, data, sender)
    
    async def chat(self, message: str, context: Optional[Dict[str, Any]] = None, 
                 prompt_template: Optional[str] = None) -> Dict[str, Any]:
        """Process a chat message and generate a response"""
        try:
            # Add to interaction history
            self.add_to_memory("interaction_history", int(time.time()), {
                "role": "user",
                "message": message,
                "context": context
            })
            
            # Prepare prompt
            if not prompt_template:
                prompt_template = """
                You are a helpful AI assistant named {agent_name} with ID {agent_id}. 
                You have the following capabilities: {capabilities}.
                
                User message: {message}
                
                Please provide a helpful response. If you need to use one of your advanced capabilities,
                you can indicate that in your response.
                """
            
            prompt = prompt_template.format(
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                capabilities=", ".join(self.capabilities),
                message=message
            )
            
            # Call LLM API based on provider
            if self.llm_provider == "anthropic":
                response = await self.call_anthropic(prompt)
            else:
                response = await self.call_openai(prompt)
            
            # Check if response suggests a capability improvement
            improvement_opportunity = await self.detect_improvement_opportunity(message, response)
            
            # Add response to interaction history
            self.add_to_memory("interaction_history", int(time.time()), {
                "role": "assistant",
                "message": response,
                "context": context
            })
            
            # Return response with metadata
            result = {
                "response": response,
                "timestamp": time.time(),
                "improvement_detected": improvement_opportunity is not None,
                "agent_id": self.agent_id
            }
            
            # If improvement detected, include it
            if improvement_opportunity:
                result["improvement_opportunity"] = improvement_opportunity
            
            return result
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {
                "response": f"I encountered an error while processing your message: {str(e)}",
                "error": str(e),
                "timestamp": time.time(),
                "agent_id": self.agent_id
            }
    
    async def call_openai(self, prompt: str) -> str:
        """Call OpenAI API to generate a response"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    async def call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API to generate a response"""
        try:
            response = self.llm_client.completions.create(
                model=self.model,
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                temperature=0.7,
                max_tokens_to_sample=1024
            )
            return response.completion
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            raise
    
    async def detect_improvement_opportunity(self, message: str, response: str) -> Optional[Dict[str, Any]]:
        """Detect if a message exchange suggests an opportunity for self-improvement"""
        # Create a prompt to analyze the exchange
        analysis_prompt = f"""
        Analyze this message exchange between a user and an AI assistant.
        
        User message: {message}
        
        Assistant response: {response}
        
        Is there an opportunity for the assistant to improve its capabilities based on this exchange?
        If yes, describe:
        1. What capability could be improved or added
        2. Why this would be valuable
        3. A brief description of how to implement it
        
        Format your response as JSON with these fields:
        {{
            "improvement_needed": true/false,
            "capability": "name of capability",
            "description": "description of improvement",
            "implementation_hint": "hint for implementation"
        }}
        """
        
        # Call LLM API based on provider
        if self.llm_provider == "anthropic":
            analysis = await self.call_anthropic(analysis_prompt)
        else:
            analysis = await self.call_openai(analysis_prompt)
        
        # Extract JSON from response
        try:
            # Find JSON block in response
            json_match = re.search(r'```json\s*(.*?)\s*```', analysis, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = analysis
            
            result = json.loads(json_str)
            
            if result.get("improvement_needed", False):
                return {
                    "capability": result.get("capability"),
                    "description": result.get("description"),
                    "implementation_hint": result.get("implementation_hint")
                }
            
            return None
            
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Error parsing improvement detection result: {e}")
            return None
    
    async def improve_capability(self, capability: str, description: str, code_sample: Optional[str] = None) -> Dict[str, Any]:
        """Improve or add a capability to the agent"""
        try:
            logger.info(f"Starting improvement process for capability: {capability}")
            
            # Prepare prompt for code generation
            prompt = f"""
            I need to improve or add a capability to a self-improving agent.
            
            Capability name: {capability}
            Description: {description}
            
            Here is the current code structure (the agent extends AgentServer class):
            
            ```python
            class SelfImprovingAgent(AgentServer):
                def __init__(self, agent_id=None, agent_name=None, 
                        host="127.0.0.1", port=8600, redis_url=None, model="gpt-4"):
                    # Initialization
                    super().__init__(...)
                    self.capabilities = [...]  # List of capabilities
                    
                async def chat(self, message, context=None, prompt_template=None):
                    # Chat functionality
                    
                async def improve_capability(self, capability, description, code_sample=None):
                    # Self-improvement functionality
                    
                async def analyze_code(self, code, context=None):
                    # Code analysis functionality
            ```
            
            {"Here's some sample code to build upon:\n\n```python\n" + code_sample + "\n```" if code_sample else ""}
            
            Please generate a Python implementation for this capability as a method.
            The method should:
            1. Have appropriate type hints
            2. Include detailed docstring
            3. Include error handling
            4. Return results as a dictionary
            5. Be well-commented
            
            IMPORTANT: Only provide the new method definition, not the entire class.
            Format your response as a Python function that could be directly added to the class.
            """
            
            # Call LLM API based on provider
            if self.llm_provider == "anthropic":
                implementation = await self.call_anthropic(prompt)
            else:
                implementation = await self.call_openai(prompt)
            
            # Extract code from response
            code_match = re.search(r'```python\s*(.*?)\s*```', implementation, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                code = implementation
            
            # Validate the code
            try:
                # Parse the code to ensure it's valid Python
                ast.parse(code)
                
                # Extract method name
                method_match = re.search(r'async def (\w+)', code)
                if method_match:
                    method_name = method_match.group(1)
                else:
                    method_name = capability.lower().replace(" ", "_")
                
                # Extract docstring
                docstring_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
                docstring = docstring_match.group(1).strip() if docstring_match else "No docstring provided"
                
                # Add the new capability to the list if not already present
                if capability not in self.capabilities:
                    self.capabilities.append(capability)
                
                # Store the improvement
                improvement = {
                    "capability": capability,
                    "method_name": method_name,
                    "description": description,
                    "implementation": code,
                    "docstring": docstring,
                    "timestamp": time.time()
                }
                self.improvements.append(improvement)
                
                # Dynamically add the method to the class
                try:
                    logger.info(f"Adding new method '{method_name}' to agent")
                    
                    # Create a namespace for exec to use
                    namespace = {}
                    
                    # Execute the code in the namespace
                    exec(code, globals(), namespace)
                    
                    # Get the function from the namespace
                    new_method = namespace.get(method_name)
                    
                    if new_method:
                        # Add the method to the class
                        setattr(SelfImprovingAgent, method_name, new_method)
                        
                        # Report self-improvement
                        await self.self_improve(
                            improvement_description=f"Added new capability: {capability}",
                            code_changes={method_name: code},
                            new_capabilities=[capability] if capability not in self.capabilities else None
                        )
                        
                        return {
                            "success": True,
                            "message": f"Successfully added capability: {capability}",
                            "method_name": method_name,
                            "improvement": improvement
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Method {method_name} not found in generated code"
                        }
                
                except Exception as e:
                    logger.error(f"Error adding method dynamically: {e}")
                    return {
                        "success": False,
                        "error": f"Error adding method: {str(e)}"
                    }
            
            except SyntaxError as e:
                logger.error(f"Syntax error in generated code: {e}")
                return {
                    "success": False,
                    "error": f"Syntax error in generated code: {str(e)}",
                    "code": code
                }
            
        except Exception as e:
            logger.error(f"Error in improve_capability: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_code(self, code: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Analyze code for improvement opportunities"""
        try:
            # Prepare prompt for code analysis
            prompt = f"""
            Analyze the following Python code for improvement opportunities:
            
            ```python
            {code}
            ```
            
            {f"Context: {context}" if context else ""}
            
            Please identify:
            1. Potential bugs or errors
            2. Performance improvements
            3. Code style and readability improvements
            4. Security concerns
            5. Opportunities for adding new capabilities
            
            Format your response as JSON with these sections:
            {{
                "bugs": [{{
                    "description": "Description of bug",
                    "location": "Line number or function name",
                    "fix": "Suggested fix"
                }}],
                "performance": [...],
                "style": [...],
                "security": [...],
                "new_capabilities": [...]
            }}
            """
            
            # Call LLM API based on provider
            if self.llm_provider == "anthropic":
                analysis = await self.call_anthropic(prompt)
            else:
                analysis = await self.call_openai(prompt)
            
            # Extract JSON from response
            try:
                # Find JSON block in response
                json_match = re.search(r'```json\s*(.*?)\s*```', analysis, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = analysis
                
                result = json.loads(json_str)
                
                # Add code patterns to memory
                self.learn_from_analysis(code, result)
                
                return {
                    "success": True,
                    "analysis": result,
                    "timestamp": time.time()
                }
                
            except (json.JSONDecodeError, AttributeError) as e:
                logger.error(f"Error parsing code analysis result: {e}")
                return {
                    "success": False,
                    "error": f"Error parsing analysis result: {str(e)}",
                    "raw_analysis": analysis
                }
            
        except Exception as e:
            logger.error(f"Error in analyze_code: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def learn_from_analysis(self, code: str, analysis: Dict[str, Any]) -> None:
        """Learn from code analysis by storing patterns in memory"""
        try:
            # Extract function definitions
            functions = self.extract_functions(code)
            
            for func_name, func_code in functions.items():
                # Create a simplified representation of the function
                func_repr = {
                    "name": func_name,
                    "code": func_code,
                    "bugs": [bug for bug in analysis.get("bugs", []) if bug.get("location", "").lower().find(func_name.lower()) >= 0],
                    "performance": [perf for perf in analysis.get("performance", []) if perf.get("location", "").lower().find(func_name.lower()) >= 0],
                    "time_added": time.time()
                }
                
                # Add to memory
                self.add_to_memory("code_patterns", func_name, func_repr)
            
            # Extract any concepts mentioned in the analysis
            all_items = (
                analysis.get("bugs", []) + 
                analysis.get("performance", []) + 
                analysis.get("style", []) + 
                analysis.get("security", []) +
                analysis.get("new_capabilities", [])
            )
            
            for item in all_items:
                desc = item.get("description", "")
                if desc:
                    # Extract potential concepts from the description
                    words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', desc)
                    words += re.findall(r'\b[a-z]+_[a-z_]+\b', desc)
                    
                    for word in words:
                        if len(word) > 3:  # Filter out short words
                            self.add_to_memory("learned_concepts", word, {
                                "mentions": self.memory.get("learned_concepts", {}).get(word, {}).get("mentions", 0) + 1,
                                "last_seen": time.time(),
                                "context": desc
                            })
            
        except Exception as e:
            logger.error(f"Error learning from analysis: {e}")
    
    def extract_functions(self, code: str) -> Dict[str, str]:
        """Extract function definitions from code"""
        functions = {}
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Get the source code for the function
                    func_body = ast.get_source_segment(code, node)
                    if func_body:
                        functions[node.name] = func_body
        except SyntaxError:
            # If the code doesn't parse, try using regex as fallback
            function_pattern = r'(async\s+)?def\s+(\w+)\s*\(.*?\).*?:'
            matches = re.finditer(function_pattern, code, re.DOTALL)
            
            for match in matches:
                func_name = match.group(2)
                # Get approximate function body (this is imperfect)
                start_pos = match.start()
                next_def = re.search(r'(async\s+)?def\s+\w+', code[start_pos + 1:])
                end_pos = next_def.start() + start_pos + 1 if next_def else len(code)
                func_body = code[start_pos:end_pos].strip()
                functions[func_name] = func_body
        
        return functions
    
    def add_to_memory(self, category: str, key: Union[str, int], value: Any) -> None:
        """Add an entry to the agent's memory"""
        if category not in self.memory:
            self.memory[category] = {}
        
        # Add to memory
        self.memory[category][key] = value
        
        # Check if we need to prune memory
        if category == "interaction_history" and len(self.memory[category]) > self.max_memory_entries:
            # Remove oldest entries
            sorted_keys = sorted(self.memory[category].keys())
            keys_to_remove = sorted_keys[:-self.max_memory_entries]
            
            for k in keys_to_remove:
                del self.memory[category][k]
    
    function_extraction = extract_functions  # Alias for API capability
    
    async def memory_management(self, action: str, category: Optional[str] = None, 
                             key: Optional[str] = None, value: Optional[Any] = None) -> Dict[str, Any]:
        """
        Manage the agent's memory store
        
        Args:
            action: The action to perform (get, set, delete, clear)
            category: The memory category
            key: The specific memory key
            value: The value to set (for 'set' action)
            
        Returns:
            Dictionary with operation result
        """
        try:
            if action == "get":
                if category and category in self.memory:
                    if key and key in self.memory[category]:
                        return {
                            "success": True,
                            "data": self.memory[category][key]
                        }
                    return {
                        "success": True,
                        "data": self.memory[category]
                    }
                return {
                    "success": True,
                    "data": self.memory
                }
            
            elif action == "set":
                if not category or not key:
                    return {
                        "success": False,
                        "error": "Category and key are required for set action"
                    }
                
                if value is None:
                    return {
                        "success": False,
                        "error": "Value is required for set action"
                    }
                
                self.add_to_memory(category, key, value)
                return {
                    "success": True,
                    "message": f"Value set for {category}.{key}"
                }
            
            elif action == "delete":
                if not category:
                    return {
                        "success": False,
                        "error": "Category is required for delete action"
                    }
                
                if category in self.memory:
                    if key and key in self.memory[category]:
                        del self.memory[category][key]
                        return {
                            "success": True,
                            "message": f"Deleted {category}.{key}"
                        }
                    elif not key:
                        del self.memory[category]
                        return {
                            "success": True,
                            "message": f"Deleted category: {category}"
                        }
                
                return {
                    "success": False,
                    "error": f"Memory location {category}.{key} not found"
                }
            
            elif action == "clear":
                if category and category in self.memory:
                    self.memory[category] = {}
                    return {
                        "success": True,
                        "message": f"Cleared category: {category}"
                    }
                else:
                    # Clear all memory
                    self.memory = {
                        "code_patterns": {},
                        "learned_concepts": {},
                        "interaction_history": []
                    }
                    return {
                        "success": True,
                        "message": "Cleared all memory"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
            
        except Exception as e:
            logger.error(f"Error in memory_management: {e}")
            return {
                "success": False,
                "error": str(e)
            }
            
    async def teach_agent(self, target_agent_id, capability, code_implementation, description=""):
        """
        Teach a capability to another agent
        
        Args:
            target_agent_id: ID of the agent to teach
            capability: Name of the capability to share
            code_implementation: Source code of the capability implementation
            description: Description of the capability
            
        Returns:
            Dictionary with operation result
        """
        try:
            # First check if the agent has the requested capability
            if capability not in self.capabilities:
                return {
                    "success": False,
                    "error": f"This agent doesn't have the capability: {capability}"
                }
            
            # If no implementation provided, try to extract it
            if not code_implementation:
                # Try to get source code for the method
                if hasattr(self, capability) and callable(getattr(self, capability)):
                    try:
                        code_implementation = inspect.getsource(getattr(self, capability))
                    except Exception as e:
                        logger.error(f"Could not extract source for {capability}: {e}")
                        return {
                            "success": False,
                            "error": f"Could not extract implementation for {capability}"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Capability exists but no method found: {capability}"
                    }
            
            # Send the capability to the target agent
            await self.publish_event(f"agent:{target_agent_id}:commands", {
                "command": "teach",
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "capability": capability,
                "implementation": code_implementation,
                "description": description or f"Capability shared from {self.agent_name}",
                "timestamp": time.time()
            })
            
            return {
                "success": True,
                "message": f"Sent capability {capability} to agent {target_agent_id}"
            }
        
        except Exception as e:
            logger.error(f"Error teaching agent {target_agent_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_agent_capabilities(self, target_agent_id, timeout=10):
        """
        Get the capabilities of another agent
        
        Args:
            target_agent_id: ID of the agent to query
            timeout: Time to wait for response in seconds
            
        Returns:
            Dictionary with agent capabilities
        """
        try:
            # Generate a unique message ID
            message_id = str(uuid.uuid4())
            
            # Subscribe to response channel
            response_channel = f"agent:{self.agent_id}:responses"
            response_pubsub = self.redis_client.pubsub()
            await response_pubsub.subscribe(response_channel)
            
            # Create a future to be resolved when response is received
            loop = asyncio.get_running_loop()
            response_future = loop.create_future()
            
            # Start listener task
            async def listen_for_capabilities():
                try:
                    while not response_future.done():
                        message = await response_pubsub.get_message(timeout=1)
                        if message and message["type"] == "message":
                            try:
                                data = json.loads(message["data"])
                                if data.get("type") == "capabilities" and data.get("agent_id") == target_agent_id:
                                    if not response_future.done():
                                        response_future.set_result(data)
                                        break
                            except json.JSONDecodeError:
                                pass
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in capabilities listener: {e}")
                    if not response_future.done():
                        response_future.set_exception(e)
                finally:
                    await response_pubsub.unsubscribe(response_channel)
            
            # Start listener
            listener_task = asyncio.create_task(listen_for_capabilities())
            
            # Request capabilities
            await self.publish_event(f"agent:{target_agent_id}:commands", {
                "command": "get_capabilities",
                "message_id": message_id,
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "timestamp": time.time()
            })
            
            try:
                # Wait for response or timeout
                response = await asyncio.wait_for(response_future, timeout)
                return {
                    "success": True,
                    "capabilities": response.get("capabilities", []),
                    "agent_id": target_agent_id
                }
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for capabilities from agent {target_agent_id}")
                return {
                    "success": False,
                    "error": f"Timeout waiting for capabilities from agent {target_agent_id}"
                }
            finally:
                # Cancel listener task
                listener_task.cancel()
                try:
                    await listener_task
                except asyncio.CancelledError:
                    pass
        
        except Exception as e:
            logger.error(f"Error getting capabilities from agent {target_agent_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def request_brain_dump(self, target_agent_id, timeout=20):
        """
        Request a full brain dump from another agent
        
        Args:
            target_agent_id: ID of the agent to query
            timeout: Time to wait for response in seconds
            
        Returns:
            Dictionary with agent's knowledge
        """
        try:
            # Generate a unique message ID
            message_id = str(uuid.uuid4())
            
            # Subscribe to response channel
            response_channel = f"agent:{self.agent_id}:responses"
            response_pubsub = self.redis_client.pubsub()
            await response_pubsub.subscribe(response_channel)
            
            # Create a future to be resolved when response is received
            loop = asyncio.get_running_loop()
            response_future = loop.create_future()
            
            # Start listener task
            async def listen_for_brain_dump():
                try:
                    while not response_future.done():
                        message = await response_pubsub.get_message(timeout=1)
                        if message and message["type"] == "message":
                            try:
                                data = json.loads(message["data"])
                                if data.get("type") == "brain_dump" and data.get("agent_id") == target_agent_id:
                                    if not response_future.done():
                                        response_future.set_result(data)
                                        break
                            except json.JSONDecodeError:
                                pass
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in brain dump listener: {e}")
                    if not response_future.done():
                        response_future.set_exception(e)
                finally:
                    await response_pubsub.unsubscribe(response_channel)
            
            # Start listener
            listener_task = asyncio.create_task(listen_for_brain_dump())
            
            # Request brain dump
            await self.publish_event(f"agent:{target_agent_id}:commands", {
                "command": "brain_dump",
                "message_id": message_id,
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "timestamp": time.time()
            })
            
            try:
                # Wait for response or timeout
                response = await asyncio.wait_for(response_future, timeout)
                
                # Store knowledge in memory for learning
                knowledge = response.get("knowledge", {})
                if knowledge:
                    # Add to memory
                    self.add_to_memory("agent_knowledge", target_agent_id, {
                        "knowledge": knowledge,
                        "timestamp": time.time()
                    })
                    
                    # Look for capabilities we don't have
                    their_capabilities = knowledge.get("capabilities", [])
                    
                    # Find capabilities we might want to learn
                    new_capabilities = [cap for cap in their_capabilities if cap not in self.capabilities]
                    
                    # Return with learning opportunities
                    return {
                        "success": True,
                        "knowledge": knowledge,
                        "learning_opportunities": new_capabilities,
                        "agent_id": target_agent_id
                    }
                
                return {
                    "success": True,
                    "knowledge": knowledge,
                    "agent_id": target_agent_id
                }
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for brain dump from agent {target_agent_id}")
                return {
                    "success": False,
                    "error": f"Timeout waiting for brain dump from agent {target_agent_id}"
                }
            finally:
                # Cancel listener task
                listener_task.cancel()
                try:
                    await listener_task
                except asyncio.CancelledError:
                    pass
        
        except Exception as e:
            logger.error(f"Error getting brain dump from agent {target_agent_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def learn_from_peers(self, timeout=30):
        """
        Discover and learn from peers in the agent network
        
        Args:
            timeout: Maximum time to spend learning
            
        Returns:
            Dictionary with learning results
        """
        try:
            start_time = time.time()
            results = {
                "peers_found": 0,
                "capabilities_learned": 0,
                "capabilities_discovered": []
            }
            
            # Get list of agents from manager
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"http://localhost:8500/agents") as response:
                        if response.status == 200:
                            data = await response.json()
                            agents = data.get("agents", [])
                        else:
                            logger.warning(f"Could not get agent list: {response.status}")
                            agents = []
            except Exception as e:
                logger.error(f"Error getting agent list: {e}")
                agents = []
            
            # Filter out self
            peer_agents = [agent for agent in agents if agent.get("agent_id") != self.agent_id]
            results["peers_found"] = len(peer_agents)
            
            # For each agent, get capabilities
            for agent in peer_agents:
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.info(f"Learn from peers timeout reached after checking {results['peers_found']} peers")
                    break
                
                agent_id = agent.get("agent_id")
                agent_name = agent.get("agent_name", "Unknown Agent")
                
                # Get capabilities
                capabilities_result = await self.get_agent_capabilities(agent_id, timeout=5)
                
                if capabilities_result.get("success"):
                    their_capabilities = capabilities_result.get("capabilities", [])
                    
                    # Find capabilities we don't have
                    new_capabilities = [cap for cap in their_capabilities if cap not in self.capabilities]
                    
                    # If there are new capabilities, request a brain dump to learn more
                    if new_capabilities:
                        logger.info(f"Found {len(new_capabilities)} new capabilities from {agent_name}: {new_capabilities}")
                        results["capabilities_discovered"].extend(new_capabilities)
                        
                        # Get brain dump
                        brain_dump_result = await self.request_brain_dump(agent_id, timeout=10)
                        
                        if brain_dump_result.get("success"):
                            # For each new capability, try to learn from them
                            for capability in new_capabilities:
                                # Try to find source code for the capability
                                implementation = None
                                description = f"Capability '{capability}' learned from agent {agent_name}"
                                
                                # Request learning
                                try:
                                    # Try to learn the capability with what we know
                                    learn_result = await self.improve_capability(capability, description, implementation)
                                    
                                    if learn_result.get("success"):
                                        results["capabilities_learned"] += 1
                                        logger.info(f"Learned capability '{capability}' from agent {agent_name}")
                                except Exception as e:
                                    logger.error(f"Error learning capability '{capability}': {e}")
            
            # Return results
            results["elapsed_time"] = time.time() - start_time
            return results
        
        except Exception as e:
            logger.error(f"Error in learn_from_peers: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def shutdown(self):
        """Shutdown the agent server"""
        logger.info(f"Agent {self.agent_id} ({self.agent_name}) shutting down...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Publish agent stopped event
        await self.publish_event("agent_events", {
            "type": "agent_stopped",
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "timestamp": time.time()
        })
        
        # Cancel background tasks
        if self.pubsub_task:
            self.pubsub_task.cancel()
            try:
                await self.pubsub_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close Redis connection
        if self.pubsub:
            await self.pubsub.unsubscribe()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info(f"Agent {self.agent_id} ({self.agent_name}) shutdown complete")


def run_agent(agent_id=None, agent_name=None, host="127.0.0.1", port=8600, model="gpt-4"):
    """Run the self-improving agent"""
    async def main():
        agent = SelfImprovingAgent(
            agent_id=agent_id,
            agent_name=agent_name,
            host=host,
            port=port,
            model=model
        )
        await agent.start()
    
    asyncio.run(main())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Improving Agent")
    parser.add_argument("--agent-id", help="Agent ID")
    parser.add_argument("--agent-name", default="Self-Improving Agent", help="Agent name")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8600, help="Port to bind to")
    parser.add_argument("--model", default="gpt-4", help="Model to use (gpt-4, claude-2, etc.)")
    
    args = parser.parse_args()
    
    run_agent(
        agent_id=args.agent_id,
        agent_name=args.agent_name,
        host=args.host,
        port=args.port,
        model=args.model
    )