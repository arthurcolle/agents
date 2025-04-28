"""
asi_agent.py

Define an ASIAgent that loads a system prompt from asi_prompt.md and
uses it to build conversation context for chat interactions.
"""
import os
from openai import OpenAI

class ASIAgent:
    """
    Agent that loads a system prompt from a markdown file and maintains
    conversation history for chat-based interactions.
    """

    def __init__(self,
                 prompt_path: str = "asi_prompt.md",
                 model: str = "gpt-3.5-turbo",
                 api_key: str = None):
        """
        Initialize the ASIAgent.

        Args:
            prompt_path: Path to the system prompt markdown file.
            model: OpenAI chat completion model identifier.
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
        """
        # Set API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Load system prompt
        self.system_prompt = self._load_prompt(prompt_path)

        # Conversation history (list of dicts with roles 'user' and 'assistant')
        self.history = []
        self.model = model

    def _load_prompt(self, path: str) -> str:
        """
        Read the system prompt from a markdown file.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise FileNotFoundError(f"Could not load prompt file '{path}': {e}")

    def reset(self) -> None:
        """
        Clear the conversation history.
        """
        self.history = []

    def chat(self, user_message: str, **kwargs) -> str:
        """
        Send a message to the agent and receive a response.

        Args:
            user_message: The user's input message.
            **kwargs: Additional parameters to pass to client.chat.completions.create

        Returns:
            The assistant's response text.
        """
        # Append user message to history
        self.history.append({"role": "user", "content": user_message})

        # Construct full message list with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
        ] + self.history

        # Call OpenAI ChatCompletion with new API format
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        # Extract assistant reply
        assistant_message = response.choices[0].message.content
        # Append assistant reply to history
        self.history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def get_history(self) -> list:
        """
        Retrieve the current conversation history (excluding system prompt).
        """
        return self.history.copy()