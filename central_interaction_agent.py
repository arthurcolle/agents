import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("central_interaction_agent")

class CentralInteractionAgent:
    """
    Central Interaction Agent that intermediates all interactions across the agents.
    """

    def __init__(self, dispatcher):
        """Initialize the Central Interaction Agent with a dispatcher."""
        self.dispatcher = dispatcher

    async def execute_command(self, kb_name: str, command: str) -> Dict[str, Any]:
        """
        Execute a command on a knowledge base agent through the dispatcher.

        Args:
            kb_name: Name of the knowledge base
            command: Command to execute

        Returns:
            Command execution result
        """
        try:
            result = await self.dispatcher.execute_kb_command(kb_name, command)
            return result
        except Exception as e:
            logger.error(f"Error executing command through CIA: {e}")
            return {
                "success": False,
                "error": f"Error executing command through CIA: {str(e)}"
            }
