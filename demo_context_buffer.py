#!/usr/bin/env python3
"""
Demo of Dynamic Context Buffer

This script demonstrates how to use the dynamic context buffer to:
1. Load existing modules into memory
2. Modify them without affecting disk
3. Execute code with hot-reloaded modules
4. Commit changes to disk when satisfied

Features demonstrated:
- Hot reloading modules
- In-memory code execution
- Snapshot management
- Code analysis and introspection
"""

import os
import sys
import time
from typing import Dict, Any, List
from dynamic_context_buffer import (
    create_dynamic_context,
    create_execution_context,
    DynamicContextBuffer,
    HotSwapExecutionContext
)

def print_header(text: str) -> None:
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def demo_load_and_modify():
    """Demonstrate loading a module and modifying it in memory"""
    print_header("DEMO: Loading and Modifying Modules")
    
    # Create a dynamic context buffer
    buffer = create_dynamic_context()
    
    # Create a simple test module in memory
    print("\n1. Creating module in memory...")
    code = """
def hello(name="World"):
    \"\"\"Say hello to someone\"\"\"
    return f"Hello, {name}!"
    
def add(a, b):
    \"\"\"Add two numbers\"\"\"
    return a + b
"""
    
    # Add the module to the buffer
    module = buffer.add_module_from_code("demo_module", code)
    print(f"   Module created: {module.name}")
    
    # Analyze the module
    summary = buffer.get_module_summary("demo_module")
    print("\n2. Module summary:")
    print(f"   Functions: {len(summary['functions'])}")
    for func in summary['functions']:
        print(f"   - {func['name']}({', '.join(func['args'])})")
        if 'docstring' in func:
            print(f"     {func['docstring']}")
    
    # Execute a function from the module
    print("\n3. Executing function from the module...")
    result = buffer.execute_function("demo_module", "hello", "Dynamic Context")
    print(f"   Result: {result['result']}")
    
    # Modify the module
    print("\n4. Modifying the module...")
    new_code = """
def hello(name="World"):
    \"\"\"Say hello to someone\"\"\"
    return f"Hello, {name}! Welcome to Dynamic Context."
    
def add(a, b):
    \"\"\"Add two numbers\"\"\"
    return a + b
    
def multiply(a, b):
    \"\"\"Multiply two numbers\"\"\"
    return a * b
"""
    
    buffer.update_module("demo_module", new_code)
    
    # Execute the modified function
    print("\n5. Executing modified function...")
    result = buffer.execute_function("demo_module", "hello", "Dynamic Context")
    print(f"   Result: {result['result']}")
    
    # Execute the new function
    print("\n6. Executing new function...")
    result = buffer.execute_function("demo_module", "multiply", 6, 7)
    print(f"   Result: {result['result']}")
    
    # Return the buffer for further demos
    return buffer

def demo_execution_context(buffer: DynamicContextBuffer):
    """Demonstrate using an execution context"""
    print_header("DEMO: Using Execution Context")
    
    # Create an execution context
    context = create_execution_context(buffer)
    
    # Import our module
    print("\n1. Importing module into context...")
    context.import_module("demo_module")
    
    # Execute code in the context
    print("\n2. Executing code in the context...")
    result = context.execute("""
result = demo_module.hello("Execution Context")
print(f"From context: {result}")

# Calculate something
result = demo_module.multiply(10, 20)
""")
    
    print(f"   Output: {result['stdout']}")
    print(f"   Result variable: {result['result']}")
    
    # Modify the module and hot reload
    print("\n3. Modifying module and hot reloading...")
    new_code = """
def hello(name="World"):
    \"\"\"Say hello to someone\"\"\"
    return f"Hello, {name}! Welcome to Hot-Reloaded Context."
    
def add(a, b):
    \"\"\"Add two numbers\"\"\"
    return a + b
    
def multiply(a, b):
    \"\"\"Multiply two numbers\"\"\"
    return a * b

def divide(a, b):
    \"\"\"Divide a by b\"\"\"
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
"""
    
    buffer.update_module("demo_module", new_code)
    context.hot_reload_module("demo_module")
    
    # Execute with the hot-reloaded module
    print("\n4. Executing with hot-reloaded module...")
    result = context.execute("""
result = demo_module.hello("Hot Swap")
print(f"After reload: {result}")

# Try the new function
result = demo_module.divide(100, 5)
print(f"100 / 5 = {result}")
""")
    
    print(f"   Output: {result['stdout']}")
    
    return context

def demo_snapshots(buffer: DynamicContextBuffer):
    """Demonstrate snapshot management"""
    print_header("DEMO: Snapshot Management")
    
    # Get snapshots
    snapshots = buffer.get_snapshots("demo_module")
    print(f"\n1. Current snapshots: {len(snapshots)}")
    
    for i, snapshot in enumerate(snapshots):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snapshot["timestamp"]))
        print(f"   {i}: {timestamp}")
    
    # Add another version with bug
    print("\n2. Adding version with bug...")
    buggy_code = """
def hello(name="World"):
    \"\"\"Say hello to someone\"\"\"
    return f"Hello, {name}! Welcome to Buggy Context."
    
def add(a, b):
    \"\"\"Add two numbers\"\"\"
    return a + b
    
def multiply(a, b):
    \"\"\"Multiply two numbers\"\"\"
    return a * b

def divide(a, b):
    \"\"\"Divide a by b\"\"\"
    # Bug: No zero check!
    return a / b
"""
    
    buffer.update_module("demo_module", buggy_code)
    
    # Test the buggy function
    print("\n3. Testing buggy function...")
    result = buffer.execute_function("demo_module", "divide", 10, 0)
    print(f"   Success: {result['success']}")
    print(f"   Error: {result['error']}")
    
    # Restore previous snapshot
    print("\n4. Restoring previous snapshot...")
    buffer.restore_snapshot("demo_module", -2)  # Second to last snapshot
    
    # Test fixed function
    print("\n5. Testing restored function...")
    result = buffer.execute_function("demo_module", "divide", 10, 0)
    if not result['success']:
        print(f"   Error correctly caught: {result['error']}")
    else:
        print(f"   Result: {result['result']}")
    
    result = buffer.execute_function("demo_module", "divide", 10, 2)
    print(f"   Normal division result: {result['result']}")

def demo_persistence(buffer: DynamicContextBuffer):
    """Demonstrate persistence and commit to disk"""
    print_header("DEMO: Persistence and Disk Commit")
    
    # Save state
    print("\n1. Saving buffer state...")
    buffer.save_state("demo_buffer_state.pkl")
    print("   State saved to demo_buffer_state.pkl")
    
    # Create new buffer and load state
    print("\n2. Creating new buffer and loading state...")
    new_buffer = create_dynamic_context()
    new_buffer.load_state("demo_buffer_state.pkl")
    
    # Test loaded module
    print("\n3. Testing module from loaded state...")
    result = new_buffer.execute_function("demo_module", "hello", "Persistence")
    print(f"   Result: {result['result']}")
    
    # Commit to disk
    print("\n4. Committing module to disk...")
    new_buffer.commit_to_disk("demo_module")
    print("   Module written to demo_module.py")
    
    # Import from disk to prove it works
    print("\n5. Importing module directly from disk...")
    try:
        sys.path.insert(0, os.getcwd())
        import demo_module
        
        result = demo_module.hello("Disk Import")
        print(f"   Result: {result}")
        
        # Clean up
        if os.path.exists("demo_buffer_state.pkl"):
            os.remove("demo_buffer_state.pkl")
    except Exception as e:
        print(f"   Error importing: {str(e)}")

def main():
    """Run the full demo"""
    print("\n" + "=" * 60)
    print("  DYNAMIC CONTEXT BUFFER DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo shows how to use the dynamic context buffer")
    print("to manage and execute code modules in memory with hot reloading.")
    
    # Part 1: Load and modify
    buffer = demo_load_and_modify()
    
    # Part 2: Execution context
    context = demo_execution_context(buffer)
    
    # Part 3: Snapshots
    demo_snapshots(buffer)
    
    # Part 4: Persistence
    demo_persistence(buffer)
    
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()