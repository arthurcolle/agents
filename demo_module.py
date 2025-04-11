
def hello(name="World"):
    """Say hello to someone"""
    return f"Hello, {name}! Welcome to Hot-Reloaded Context."
    
def add(a, b):
    """Add two numbers"""
    return a + b
    
def multiply(a, b):
    """Multiply two numbers"""
    return a * b

def divide(a, b):
    """Divide a by b"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
