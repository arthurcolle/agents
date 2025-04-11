"""
Basic mathematical utility functions.
TODO: Add more comprehensive error handling
"""

def add_numbers(a, b):
    # Add two numbers
    return a + b

def subtract_numbers(a, b):
    # Subtract b from a
    return a - b

def multiply_numbers(a, b):
    # Multiply two numbers
    return a * b

def divide_numbers(a, b):
    # Divide a by b
    return a / b

def calculate_average(numbers):
    # Calculate the average of a list of numbers
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

def fibonacci(n):
    # Return the nth Fibonacci number
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for i in range(2, n + 1):
            a, b = b, a + b
        return b

def factorial(n):
    # Calculate factorial
    if n < 0:
        return None  # Factorial is not defined for negative numbers
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def is_prime(n):
    # Check if a number is prime
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

class Geometry:
    # Simple geometry functions
    
    def __init__(self):
        pass
    
    def circle_area(self, radius):
        # Calculate the area of a circle
        import math
        return math.pi * radius * radius
    
    def rectangle_area(self, length, width):
        # Calculate the area of a rectangle
        return length * width
    
    def triangle_area(self, base, height):
        # Calculate the area of a triangle
        return 0.5 * base * height