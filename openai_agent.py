import os
import logging
from agents import Agent, Runner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai-agent")

class ContextLoader:
    """
    Loads context from various sources to enhance agent capabilities
    """
    def __init__(self, sources_dir="context_sources"):
        self.sources_dir = sources_dir
        os.makedirs(sources_dir, exist_ok=True)
        logger.info(f"Context loader initialized with sources directory: {sources_dir}")
    
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
    def __init__(self, name="Assistant", instructions=None, model="gpt-4"):
        self.name = name
        self.model = model
        self.context_loader = ContextLoader()
        self.contexts = {}
        self.max_iterations = 5
        
        # Default instructions if none provided
        default_instructions = """You are an autonomous assistant capable of reasoning through complex problems.
You can break down tasks into steps and work through them methodically.
When you need more information, you'll identify what you need and seek it out."""
        
        self.agent = Agent(
            name=name, 
            instructions=instructions or default_instructions,
            model=model
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
    
    def _run_autonomous(self, initial_query):
        """Run the agent in autonomous mode with reasoning steps"""
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

# Example usage
if __name__ == "__main__":
    # Create an autonomous agent
    auto_agent = AutonomousAgent(name="Assistant", model="gpt-4")
    
    # Create a context directory and sample file for demonstration
    os.makedirs("context_sources", exist_ok=True)
    with open("context_sources/programming_concepts.txt", "w") as f:
        f.write("Recursion is a programming concept where a function calls itself to solve a problem.")
    
    # Load context
    auto_agent.load_context()
    
    # Run the agent with a query
    result = auto_agent.run("Write a haiku about recursion in programming and explain the concept.")
    print("\nFinal output:")
    print(result)
