#!/usr/bin/env python3
import subprocess
import time
import os

def main():
    """Test the together_cli.py with automated input including python code execution."""
    
    # Start the together_cli.py process in test mode
    process = subprocess.Popen(
        ["python", "together_cli.py", "--test-mode"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/Users/agent/llama4"
    )
    
    # Wait for the CLI to initialize
    time.sleep(2)
    
    # Send a test message with Python code request
    print("Sending Python code request...")
    process.stdin.write("Use python code to print the current date\n")
    process.stdin.flush()
    
    # Wait for the response
    time.sleep(3)
    
    # Send command to run the code 
    print("Sending run code command...")
    process.stdin.write("Run the code\n")
    process.stdin.flush()
    
    # Wait for execution
    time.sleep(3)
    
    # Exit the CLI
    print("Exiting CLI...")
    process.stdin.write("exit\n")
    process.stdin.flush()
    
    # Get the output
    stdout, stderr = process.communicate(timeout=5)
    
    # Print the output
    print("\nSTDOUT:")
    print(stdout)
    
    if stderr:
        print("\nSTDERR:")
        print(stderr)
    
    # Check if test was successful
    if "<|python_start|>" in stdout and "Current Date:" in stdout:
        print("\n✅ TEST PASSED: Python code block was detected and executed successfully!")
        return 0
    else:
        print("\n❌ TEST FAILED: Python code block was not detected or executed")
        return 1

if __name__ == "__main__":
    exit(main())