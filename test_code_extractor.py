#!/usr/bin/env python3
"""
Test script for the CodeExtractor class
"""

from code_extractor import CodeExtractor

def main():
    # Create a sample text with embedded code
    sample_text = """
Here's a simple neural network implementation:

```python
import numpy as np

class SimpleNN:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(input_dim, output_dim)

    def forward(self, x):
        return np.dot(x, self.weights)

def simple_fitness(network, inputs, targets):
    predictions = network.forward(inputs)
    error = np.mean((predictions - targets) ** 2)
    return -error

def simple_neuroevolution(input_dim, output_dim, population_size, generations, inputs, targets):
    population = [SimpleNN(input_dim, output_dim) for _ in range(population_size)]

    for generation in range(generations):
        fitnesses = [simple_fitness(network, inputs, targets) for network in population]
        sorted_indices = np.argsort(fitnesses)[::-1]
        selected_indices = sorted_indices[:int(population_size/2)]
        selected_population = [population[i] for i in selected_indices]

        new_population = []
        for _ in range(population_size):
            parent1, parent2 = np.random.choice(selected_population, 2, replace=False)
            child_weights = (parent1.weights + parent2.weights) / 2
            child_weights += np.random.randn(*child_weights.shape) * 0.1
            child = SimpleNN(input_dim, output_dim)
            child.weights = child_weights
            new_population.append(child)

        population = new_population

    return population[0]
```

Let's test it with some sample data:

<|python_start|>
if __name__ == "__main__":
    input_dim = 2
    output_dim = 1
    population_size = 10
    generations = 5
    inputs = np.random.rand(10, input_dim)
    targets = np.random.rand(10, output_dim)

    best_network = simple_neuroevolution(input_dim, output_dim, population_size, generations, inputs, targets)
    print("Best network weights:", best_network.weights)
<|python_end|>
"""

    # Create the extractor
    extractor = CodeExtractor()
    
    # Extract code blocks
    print("Extracting code blocks...")
    blocks = extractor.extract_code_blocks(sample_text)
    
    print(f"Found {len(blocks)} code blocks:")
    for i, block in enumerate(blocks):
        print(f"\nBlock {i+1} ({block.language}):")
        print(f"Lines: {block.line_count} (from {block.start_line} to {block.end_line})")
        print("-" * 40)
        print(block.code[:100] + "..." if len(block.code) > 100 else block.code)
        print("-" * 40)
    
    # Extract and execute
    print("\nExtracting and executing code...")
    result = extractor.extract_and_execute(sample_text)
    
    if result["success"]:
        print("\nExecution successful!")
        if result["stdout"]:
            print("\nOutput:")
            print(result["stdout"])
        
        print("\nLocal variables:")
        for name, value in result["local_vars"].items():
            if name in ["SimpleNN", "simple_fitness", "simple_neuroevolution", "best_network"]:
                print(f"- {name}: {type(value).__name__}")
    else:
        print("\nExecution failed:")
        print(result["error"])
        print(result["traceback"])

if __name__ == "__main__":
    main()
