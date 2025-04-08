import os
import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
import aiohttp
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai-agent")

# Import Jina client (with fallback for import errors)
try:
    from jina_client import JinaClient
    jina_client = JinaClient()
    JINA_AVAILABLE = True
    logger.info("Jina client initialized successfully")
except (ImportError, Exception) as e:
    logger.warning(f"Jina client initialization failed: {e}")
    jina_client = None
    JINA_AVAILABLE = False

# Global configuration
SYSTEM_GOAL = """
Ultimate Goal: Build a cool web app with dynamic generative UI that lets users send prompts and get real work done.
Key Features:
- Distributed systems architecture for scalability
- Native ability to call external services and APIs
- Dynamic UI generation based on user needs
- Intelligent task decomposition and execution
"""

class ContextLoader:
    """
    Loads context from various sources to enhance agent capabilities
    """
    def __init__(self, sources_dir="context_sources"):
        self.sources_dir = sources_dir
        os.makedirs(sources_dir, exist_ok=True)
        logger.info(f"Context loader initialized with sources directory: {sources_dir}")
        
        # Create system goal context file if it doesn't exist
        system_goal_path = os.path.join(sources_dir, "system_goal.txt")
        if not os.path.exists(system_goal_path):
            with open(system_goal_path, 'w') as f:
                f.write(SYSTEM_GOAL)
            logger.info(f"Created system goal context file at {system_goal_path}")
    
    def load_from_file(self, filepath):
        """Load context from a text file"""
        try:
            with open(filepath, 'r') as file:
                content = file.read()
                logger.info(f"Loaded context from {filepath}")
                return content
        except Exception as e:
            logger.error(f"Error loading context from {filepath}: {e}")
            return None
    
    def load_from_directory(self, directory=None):
        """Load all text files from a directory"""
        directory = directory or self.sources_dir
        contexts = {}
        
        try:
            for filename in os.listdir(directory):
                if filename.endswith('.txt'):
                    filepath = os.path.join(directory, filename)
                    contexts[filename] = self.load_from_file(filepath)
            
            logger.info(f"Loaded {len(contexts)} context files from {directory}")
            return contexts
        except Exception as e:
            logger.error(f"Error loading contexts from directory {directory}: {e}")
            return {}

class AutonomousAgent:
    """
    Enhanced agent with autonomous capabilities and context awareness
    """
    def __init__(self, name="Assistant", instructions=None, model="gpt-4", tools=None):
        self.name = name
        self.model = model
        self.context_loader = ContextLoader()
        self.contexts = {}
        self.max_iterations = 5
        
        # Default instructions if none provided
        default_instructions = f"""You are an autonomous assistant capable of reasoning through complex problems.
You can break down tasks into steps and work through them methodically.
When you need more information, you'll identify what you need and seek it out.

{SYSTEM_GOAL}

Your primary mission is to help users build and interact with this web application.
When addressing user requests, consider how they align with this goal and how you can
leverage distributed systems and external services to accomplish tasks efficiently.
"""
        
        self.agent = Agent(
            name=name, 
            instructions=instructions or default_instructions,
            model=model,
            tools=tools
        )
        
        logger.info(f"Autonomous agent '{name}' initialized with model {model}")
    
    def load_context(self, source=None):
        """Load context from a file or directory"""
        if source and os.path.isfile(source):
            content = self.context_loader.load_from_file(source)
            if content:
                self.contexts[os.path.basename(source)] = content
        else:
            directory = source if source and os.path.isdir(source) else None
            self.contexts.update(self.context_loader.load_from_directory(directory))
        
        return self.contexts
    
    def run(self, query, autonomous_mode=True):
        """Run the agent with the given query"""
        if autonomous_mode:
            return self._run_autonomous(query)
        else:
            return Runner.run_sync(self.agent, query)
    
    async def run_async(self, query, autonomous_mode=True):
        """Run the agent asynchronously with the given query"""
        if autonomous_mode:
            return await self._run_autonomous_async(query)
        else:
            return await Runner.run(self.agent, query)
    
    def _run_autonomous(self, initial_query):
        """Run the agent in autonomous mode with reasoning steps (synchronous)"""
        logger.info(f"Starting autonomous reasoning for query: {initial_query}")
        
        # Prepare context-enhanced prompt
        context_text = ""
        for name, content in self.contexts.items():
            context_text += f"\n--- Context from {name} ---\n{content}\n"
        
        if context_text:
            enhanced_query = f"I'll provide you with some context information first, then my query.\n\n{context_text}\n\nNow, my query is: {initial_query}"
        else:
            enhanced_query = initial_query
        
        # Initial response
        result = Runner.run_sync(self.agent, enhanced_query)
        
        # Check if further reasoning is needed
        iterations = 1
        current_state = result.final_output
        
        while iterations < self.max_iterations:
            # Check if the agent needs more information or reasoning
            follow_up = f"""Based on your previous response:
            
{current_state}

Do you need to perform additional reasoning steps or gather more information? 
If so, describe what additional steps you would take and why.
If not, indicate that your response is complete."""
            
            follow_up_result = Runner.run_sync(self.agent, follow_up)
            follow_up_response = follow_up_result.final_output
            
            # Check if the agent considers the response complete
            if "complete" in follow_up_response.lower() and "additional" not in follow_up_response.lower():
                logger.info(f"Autonomous reasoning complete after {iterations} iterations")
                break
            
            # Otherwise, continue reasoning
            reasoning_prompt = f"""Let's continue our reasoning process.

Previous reasoning:
{current_state}

Your thoughts on next steps:
{follow_up_response}

Please continue working on the original query: {initial_query}"""
            
            next_result = Runner.run_sync(self.agent, reasoning_prompt)
            current_state = next_result.final_output
            iterations += 1
            
            logger.info(f"Completed reasoning iteration {iterations}")
        
        if iterations >= self.max_iterations:
            logger.warning(f"Reached maximum iterations ({self.max_iterations}) for autonomous reasoning")
        
        return current_state
    
    async def _run_autonomous_async(self, initial_query):
        """Run the agent in autonomous mode with reasoning steps (asynchronous)"""
        logger.info(f"Starting autonomous reasoning for query: {initial_query}")
        
        # Prepare context-enhanced prompt
        context_text = ""
        for name, content in self.contexts.items():
            context_text += f"\n--- Context from {name} ---\n{content}\n"
        
        if context_text:
            enhanced_query = f"I'll provide you with some context information first, then my query.\n\n{context_text}\n\nNow, my query is: {initial_query}"
        else:
            enhanced_query = initial_query
        
        # Initial response
        result = await Runner.run(self.agent, enhanced_query)
        
        # Check if further reasoning is needed
        iterations = 1
        current_state = result.final_output
        
        while iterations < self.max_iterations:
            # Check if the agent needs more information or reasoning
            follow_up = f"""Based on your previous response:
            
{current_state}

Do you need to perform additional reasoning steps or gather more information? 
If so, describe what additional steps you would take and why.
If not, indicate that your response is complete."""
            
            follow_up_result = await Runner.run(self.agent, follow_up)
            follow_up_response = follow_up_result.final_output
            
            # Check if the agent considers the response complete
            if "complete" in follow_up_response.lower() and "additional" not in follow_up_response.lower():
                logger.info(f"Autonomous reasoning complete after {iterations} iterations")
                break
            
            # Otherwise, continue reasoning
            reasoning_prompt = f"""Let's continue our reasoning process.

Previous reasoning:
{current_state}

Your thoughts on next steps:
{follow_up_response}

Please continue working on the original query: {initial_query}"""
            
            next_result = await Runner.run(self.agent, reasoning_prompt)
            current_state = next_result.final_output
            iterations += 1
            
            logger.info(f"Completed reasoning iteration {iterations}")
        
        if iterations >= self.max_iterations:
            logger.warning(f"Reached maximum iterations ({self.max_iterations}) for autonomous reasoning")
        
        return current_state

class TranslationOrchestrator:
    """
    Orchestrates translation capabilities using multiple specialized agents
    """
    def __init__(self, model="gpt-4"):
        self.model = model
        
        # Create specialized translation agents
        self.spanish_agent = Agent(
            name="spanish_agent",
            instructions="You translate the user's message to Spanish",
            handoff_description="An English to Spanish translator",
            model=model
        )
        
        self.french_agent = Agent(
            name="french_agent",
            instructions="You translate the user's message to French",
            handoff_description="An English to French translator",
            model=model
        )
        
        self.italian_agent = Agent(
            name="italian_agent",
            instructions="You translate the user's message to Italian",
            handoff_description="An English to Italian translator",
            model=model
        )
        
        self.german_agent = Agent(
            name="german_agent",
            instructions="You translate the user's message to German",
            handoff_description="An English to German translator",
            model=model
        )
        
        # Create orchestrator agent
        self.orchestrator_agent = Agent(
            name="orchestrator_agent",
            instructions=(
                "You are a translation agent. You use the tools given to you to translate. "
                "If asked for multiple translations, you call the relevant tools in order. "
                "You never translate on your own, you always use the provided tools."
            ),
            model=model,
            tools=[
                self.spanish_agent.as_tool(
                    tool_name="translate_to_spanish",
                    tool_description="Translate the user's message to Spanish",
                ),
                self.french_agent.as_tool(
                    tool_name="translate_to_french",
                    tool_description="Translate the user's message to French",
                ),
                self.italian_agent.as_tool(
                    tool_name="translate_to_italian",
                    tool_description="Translate the user's message to Italian",
                ),
                self.german_agent.as_tool(
                    tool_name="translate_to_german",
                    tool_description="Translate the user's message to German",
                ),
            ],
        )
        
        # Create synthesizer agent
        self.synthesizer_agent = Agent(
            name="synthesizer_agent",
            instructions="You inspect translations, correct them if needed, and produce a final concatenated response.",
            model=model
        )
        
        logger.info("Translation orchestrator initialized")
    
    async def translate(self, text, languages=None):
        """
        Translate text to specified languages
        If languages is None, the orchestrator will determine which languages to use
        """
        if languages:
            query = f"Translate the following text to {', '.join(languages)}: '{text}'"
        else:
            query = f"Translate the following text to appropriate languages: '{text}'"
        
        logger.info(f"Starting translation for: {text}")
        
        # Run the entire orchestration in a single trace
        with trace("Translation orchestration"):
            orchestrator_result = await Runner.run(self.orchestrator_agent, query)
            
            # Log intermediate steps
            for item in orchestrator_result.new_items:
                if isinstance(item, MessageOutputItem):
                    message_text = ItemHelpers.text_message_output(item)
                    if message_text:
                        logger.info(f"Translation step: {message_text}")
            
            # Synthesize final result
            synthesizer_result = await Runner.run(
                self.synthesizer_agent, orchestrator_result.to_input_list()
            )
        
        return synthesizer_result.final_output

# Example usage
async def main_async():
    # Create an autonomous agent with translation capabilities
    translator = TranslationOrchestrator(model="gpt-4")
    
    # Create a context-aware agent
    auto_agent = AutonomousAgent(name="Assistant", model="gpt-4")
    
    # Create a context directory and sample file for demonstration
    os.makedirs("context_sources", exist_ok=True)
    with open("context_sources/programming_concepts.txt", "w") as f:
        f.write("Recursion is a programming concept where a function calls itself to solve a problem.")
    
    # Load context
    auto_agent.load_context()
    
    # Example 1: Run the autonomous agent
    print("\n=== Autonomous Agent Example ===")
    result = await auto_agent.run_async("Write a haiku about recursion in programming and explain the concept.")
    print("\nFinal output:")
    print(result)
    
    # Example 2: Run the translation orchestrator
    print("\n=== Translation Example ===")
    text_to_translate = input("\nEnter text to translate: ")
    languages = input("Enter languages to translate to (comma-separated, or press Enter for auto-selection): ")
    
    language_list = [lang.strip() for lang in languages.split(',')] if languages else None
    translation_result = await translator.translate(text_to_translate, language_list)
    
    print("\nTranslation results:")
    print(translation_result)

# Jina search functions
async def search_web(query: str) -> Dict[str, Any]:
    """
    Search the web using Jina.ai
    Args:
        query: Search query
    Returns:
        Search results
    """
    if not JINA_AVAILABLE or not jina_client:
        logger.warning("Jina client not available, returning mock results")
        if jina_client:
            return jina_client.get_mock_search_results(query)
        return {
            "error": True,
            "message": "Jina client not available"
        }
    
    logger.info(f"Searching web for: {query}")
    return await jina_client.search(query)

async def fact_check(statement: str) -> Dict[str, Any]:
    """
    Fact check a statement using Jina.ai
    Args:
        statement: Statement to fact check
    Returns:
        Fact check results
    """
    if not JINA_AVAILABLE or not jina_client:
        logger.warning("Jina client not available, returning mock results")
        if jina_client:
            return jina_client.get_mock_fact_check(statement)
        return {
            "error": True,
            "message": "Jina client not available"
        }
    
    logger.info(f"Fact checking: {statement}")
    return await jina_client.fact_check(statement)

async def read_url(url: str) -> Dict[str, Any]:
    """
    Read and analyze content from a URL using Jina.ai
    Args:
        url: URL to read
    Returns:
        Content analysis
    """
    if not JINA_AVAILABLE or not jina_client:
        logger.warning("Jina client not available, returning mock results")
        if jina_client:
            return jina_client.get_mock_read_results(url)
        return {
            "error": True,
            "message": "Jina client not available"
        }
    
    logger.info(f"Reading URL: {url}")
    return await jina_client.read(url)

if __name__ == "__main__":
    # Create a simple menu to choose between examples
    print("Choose an example to run:")
    print("1. Autonomous reasoning with context")
    print("2. Translation orchestration")
    print("3. Run both examples")
    print("4. Start web app with dynamic generative UI")
    print("5. Test Jina.ai integration")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == "1":
        # Run the autonomous agent example synchronously
        auto_agent = AutonomousAgent(name="Assistant", model="gpt-4")
        os.makedirs("context_sources", exist_ok=True)
        with open("context_sources/programming_concepts.txt", "w") as f:
            f.write("Recursion is a programming concept where a function calls itself to solve a problem.")
        auto_agent.load_context()
        result = auto_agent.run("Write a haiku about recursion in programming and explain the concept.")
        print("\nFinal output:")
        print(result)
    elif choice == "2" or choice == "3":
        # Run the async examples
        asyncio.run(main_async())
    elif choice == "4":
        # Import and run the web app
        try:
            from web_app import run_app
            print("\nStarting web app with dynamic generative UI...")
            print("Open your browser and navigate to http://localhost:8000")
            run_app()
        except ImportError:
            print("Web app module not found. Make sure web_app.py is in the current directory.")
    elif choice == "5":
        # Test Jina.ai integration
        print("\n=== Testing Jina.ai Integration ===")
        
        if not JINA_AVAILABLE:
            print("Jina client is not available. Make sure jina_client.py is in the current directory.")
            print("Note: You'll need a Jina API key for real results. Set it as JINA_API_KEY environment variable.")
            print("Without an API key, mock results will be returned.")
        
        async def run_jina_tests():
            # Test search
            query = input("\nEnter a search query: ")
            print("Searching...")
            search_results = await search_web(query)
            print("\nSearch Results:")
            print(json.dumps(search_results, indent=2))
            
            # Test fact check
            statement = input("\nEnter a statement to fact check: ")
            print("Fact checking...")
            fact_check_results = await fact_check(statement)
            print("\nFact Check Results:")
            print(json.dumps(fact_check_results, indent=2))
            
            # Test URL reading
            url = input("\nEnter a URL to read: ")
            print("Reading URL...")
            read_results = await read_url(url)
            print("\nURL Reading Results:")
            print(json.dumps(read_results, indent=2))
        
        asyncio.run(run_jina_tests())
