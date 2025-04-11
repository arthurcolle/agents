#!/usr/bin/env python3
"""
A simplified test script that directly tests the Python code execution functionality.
"""
import os
import sys
from datetime import date

def execute_python(code):
    """Execute Python code safely and return the results."""
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    local_vars = {}
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, globals(), local_vars)
        
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        return {
            "stdout": stdout,
            "stderr": stderr,
            "success": True
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False
        }

def extract_python_code(text):
    """Extract Python code from text between <|python_start|> and <|python_end|> tags."""
    import re
    pattern = r'<\|python_start\|>(.*?)(?:<\|python_end\|>|<\|python_end)'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

def main():
    """Test Python code extraction and execution."""
    # Example response with Python code
    example_response = """
    Here's Python code to print the current date:

    <|python_start|>
from datetime import date

def get_current_date():
    # Get today's date
    today = date.today()
    return today

current_date = get_current_date()
print("Current Date: ", current_date)
    <|python_end|>

    This code imports the date class from the datetime module, defines a function to get the current date, and then prints it.
    """
    
    print("Extracting Python code blocks from example response...")
    code_blocks = extract_python_code(example_response)
    
    if not code_blocks:
        print("❌ No Python code blocks found!")
        return 1
    
    print(f"✅ Found {len(code_blocks)} Python code block(s)")
    
    for i, code in enumerate(code_blocks):
        print(f"\nExecuting code block {i+1}:")
        print("=" * 40)
        print(code)
        print("=" * 40)
        
        # Execute the code
        result = execute_python(code)
        
        if result["success"]:
            print("\nExecution successful!")
            if result["stdout"]:
                print("\nStandard output:")
                print("-" * 40)
                print(result["stdout"])
                print("-" * 40)
            else:
                print("No output produced.")
            
            if result["stderr"]:
                print("\nStandard error:")
                print(result["stderr"])
        else:
            print("\n❌ Execution failed!")
            print(f"Error: {result['error']}")
            print(result["traceback"])
    
    return 0

if __name__ == "__main__":
    sys.exit(main())