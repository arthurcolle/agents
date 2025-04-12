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

    def validate_classification_level(self, level: str) -> bool:
        """
        Validate the classification level.

        Args:
            level: Classification level to validate

        Returns:
            True if valid, False otherwise
        """
        valid_levels = {"public", "confidential", "secret", "top_secret"}
        return level in valid_levels

    def set_classification_level(self, kb_name: str, level: str) -> None:
        """
        Set the classification level for a knowledge base.

        Args:
            kb_name: Name of the knowledge base
            level: Classification level to set
        """
        if not self.validate_classification_level(level):
            logger.error(f"Invalid classification level: {level}")
            return

        logger.info(f"Setting classification level for {kb_name} to {level}")
        # Additional logic to apply the classification level can be added here

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
            logger.error(f"Error executing command on {kb_name} through CIA: {e}")
            return {
                "success": False,
                "error": f"Error executing command through CIA: {str(e)}"
            }
