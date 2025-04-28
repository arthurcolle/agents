#!/usr/bin/env python3
"""
ASI Agent CLI

A simple command-line interface for interacting with the ASIAgent.
"""
import argparse
import sys

from asi_agent import ASIAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Command-line interface for ASIAgent"
    )
    parser.add_argument(
        "--prompt-path",
        default="asi_prompt.md",
        help="Path to the system prompt markdown file (default: asi_prompt.md)",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI chat model to use (default: gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (default: read from OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Print the conversation history after each exchange",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        agent = ASIAgent(
            prompt_path=args.prompt_path,
            model=args.model,
            api_key=args.api_key,
        )
    except Exception as e:
        print(f"Error initializing ASIAgent: {e}", file=sys.stderr)
        sys.exit(1)

    print("ASI Agent CLI. Type your messages below. Type 'exit' or 'quit' to end.")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            response = agent.chat(user_input)
            print(response)

            if args.show_history:
                print("\n--- Conversation history ---")
                for msg in agent.get_history():
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    print(f"[{role}] {content}")
                print("--- End history ---\n")
        except Exception as err:
            print(f"Error during chat: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()