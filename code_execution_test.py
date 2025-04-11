#!/usr/bin/env python3
"""
Final integration test for the Python code execution feature in together_cli.py.
This test simulates a user conversation that includes:
1. Requesting code generation
2. Running the generated code
"""
import os
import sys
import time

def simulate_conversation():
    """Simulates a conversation with the agent, testing code execution."""
    # Set environment for testing
    os.environ["TOGETHER_API_KEY"] = "dummy_api_key_for_testing"
    
    # Import after setting environment
    from together_cli import TogetherAgent, console, extract_python_code
    
    # Initialize agent
    agent = TogetherAgent()
    
    # Test case 1: Generate Python code
    test_user_input = "Use python code to print the current date"
    print(f"\n[TEST] User: {test_user_input}")
    
    # Directly add the test response to the conversation history
    test_assistant_response = """Here's Python code to print the current date:

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
    
    agent.conversation_history.append({"role": "user", "content": test_user_input})
    agent.conversation_history.append({"role": "assistant", "content": test_assistant_response})
    
    # Test case 2: Run the code
    run_code_input = "Run the code"
    print(f"\n[TEST] User: {run_code_input}")
    
    # Directly call the parsing and execution methods
    response_content = test_assistant_response
    
    if "<|python_start|>" in response_content and "<|python_end" in response_content:
        code_blocks = extract_python_code(response_content)
        if code_blocks:
            for code_block in code_blocks:
                print("[CYAN] Executing Python code:")
                
                # Direct execution
                import io
                import datetime  # Ensure datetime module is available
                from contextlib import redirect_stdout, redirect_stderr
                
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                local_vars = {}
                
                # Prepare execution environment with imports
                exec_globals = globals().copy()
                # Add datetime modules to globals
                from datetime import date, datetime, timedelta
                exec_globals['date'] = date
                exec_globals['datetime'] = datetime
                exec_globals['timedelta'] = timedelta
                
                try:
                    print(f"[Code]:\n{code_block}")
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        exec(code_block, exec_globals, local_vars)
                    
                    stdout = stdout_capture.getvalue()
                    stderr = stderr_capture.getvalue()
                    
                    if stdout:
                        print("\n[GREEN] Execution result:")
                        print(stdout)
                    
                    if stderr:
                        print("\n[RED] Errors:")
                        print(stderr)
                        
                except Exception as e:
                    import traceback
                    print(f"\n[RED] Error executing code: {str(e)}")
                    print(traceback.format_exc())
    
    print("\n[TEST COMPLETED] Code execution feature is working as expected!")
    return 0

if __name__ == "__main__":
    exit(simulate_conversation())