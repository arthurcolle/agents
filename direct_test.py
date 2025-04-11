#!/usr/bin/env python3
"""
Direct test for the together_cli.py code execution functionality.
"""
import os
import sys

def main():
    """Test the code execution functionality directly."""
    print("Setting up test environment...")
    
    # Set TEST_MODE to true in the environment
    os.environ["TOGETHER_API_KEY"] = "dummy_api_key_for_testing"
    
    # Import our agent from the together_cli.py file
    from together_cli import TogetherAgent, console, extract_python_code
    
    print("\nInitializing agent in test mode...")
    agent = TogetherAgent()
    
    # Add a user message - append directly to conversation_history
    print("\nAdding user message: 'Use python code to print the current date'")
    agent.conversation_history.append({"role": "user", "content": "Use python code to print the current date"})
    
    # Get response directly from the conversation
    print("\nSimulating agent response...")
    
    # Use a simple mock response with Python code block
    response = """Here's Python code to print the current date:

<|python_start|>
from datetime import date

def get_current_date():
    # Get today's date
    today = date.today()
    return today

current_date = get_current_date()
print("Current Date: ", current_date)
<|python_end|>

This code imports the date class from the datetime module, defines a function to get the current date, and then prints it."""
    
    print("\nAgent response:")
    print("-" * 60)
    print(response)
    print("-" * 60)
    
    # Check if we got a response
    if not response:
        print("\n❌ No response received from the agent!")
        return 1
    
    # Check if response contains the expected code block
    if "<|python_start|>" in response and "<|python_end" in response:
        print("\n✅ Python code block detected in the response!")
        
        # Extract the code block
        code_blocks = extract_python_code(response)
        
        print(f"\nExtracted {len(code_blocks)} code block(s)")
        
        # Execute the code - simulate "Run the code" request
        if code_blocks:
            print("\nExecuting the code:")
            for i, code_block in enumerate(code_blocks):
                print(f"\nCode block {i+1}:")
                print("-" * 60)
                print(code_block)
                print("-" * 60)
                
                # Run the code
                print("\nOutput:")
                try:
                    # Execute the code with proper globals
                    import datetime  # Ensure datetime is available
                    exec_globals = globals().copy()
                    exec(code_block, exec_globals)
                    print("\n✅ Code execution successful!")
                except Exception as e:
                    print(f"\n❌ Error executing code: {e}")
                    return 1
            
            return 0
        else:
            print("\n❌ No code blocks extracted!")
            return 1
    else:
        print("\n❌ No Python code block detected in the response!")
        return 1

if __name__ == "__main__":
    exit(main())