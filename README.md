# Llama4 Agent System

A sophisticated agent system built on the Llama 4 model with dynamic code generation and self-modification capabilities.

## Key Features

- **Protected Kernel Architecture**: Prevents critical system components from being modified at runtime
- **Dynamic Code Extension**: Add new capabilities without restarting
- **Self-modifying Agent**: Analyzes and improves its own code
- **Module Management**: Load, register, and validate code modules
- **Function Calling**: Robust function registry with parameter validation
- **Reasoning Engine**: Multi-step reasoning for complex problem-solving
- **Hybrid Search**: Combined semantic and keyword search using DuckDB and FlockMTL
- **Jupyter Integration**: Execute code and manage notebooks interactively
- **Code Execution Engine**: Safely run code with resource limits
- **Filesystem Tools**: Safe file operations with security controls

## Components

- `openrouter-llama4.py`: Main agent with protected kernel and dynamic code capabilities
- `openrouter_kernel.py`: Core kernel implementation for protected operations
- `agent_editor.py`: Code analysis and modification utilities
- `self_modify_agent.py`: Agent that can analyze and improve its own code
- `hybrid_search.py`: Semantic and keyword search using DuckDB and FlockMTL
- `advanced_hybrid_search.py`: Enhanced version with additional features
- `modules/`: Directory for loadable code modules
  - `jupyter_tools.py`: Jupyter notebook and kernel management
  - `code_execution.py`: Safe code execution with sandboxing
  - `filesystem.py`: File system operations with safety controls
  - `jina_tools.py`: Web search and content tools
  - `math_utils.py`: Mathematical utilities

## Jina Integration

The system includes integration with Jina.ai APIs for search, fact checking, and content ranking:

- `modules/jina_client.py`: Async client for Jina.ai endpoints
- `modules/jina_tools.py`: Functions for kernel integration
- `register_jina_tools.py`: Script to register Jina tools with the agent kernel

## Setup & Usage

1. Set up your environment:
   ```
   export OPENROUTER_API_KEY=your_api_key
   export JINA_API_KEY=your_jina_api_key  # For Jina integration
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the main agent:
   ```
   python openrouter-llama4.py --interactive
   ```

4. Run the self-modifying agent:
   ```
   python self_modify_agent.py --interactive
   ```

5. Register all tools:
   ```
   python register_tools.py
   ```

## Using Jupyter Integration

```python
# Execute code in a Jupyter kernel
result = jupyter_execute_code("import numpy as np\nprint(np.random.rand(3,3))")

# Create and save a notebook
cells = [
    {"type": "markdown", "content": "# My Notebook\nThis is a test."},
    {"type": "code", "content": "print('Hello, world!')"}
]
notebook = jupyter_create_notebook(cells)
jupyter_save_notebook(notebook["notebook"], "my_notebook.ipynb")

# Run a notebook
result = jupyter_run_notebook("my_notebook.ipynb")
```

## Using Code Execution

```python
# Execute Python code safely
result = execute_code("print('Hello, world!')")

# Run with resource limits
result = execute_code("import numpy as np\nprint(np.random.rand(1000,1000))", 
                     time_limit=5, memory_limit=500)

# Execute JavaScript
result = execute_code("console.log('Hello from Node.js')", language="javascript")

# Run a script file
result = run_script("my_script.py")
```

## Using Filesystem Tools

```python
# Read a file
result = read_file("/path/to/file.txt")

# Write to a file
result = write_file("/path/to/output.txt", "Hello, world!")

# List directory contents
result = list_directory("/path/to/directory", recursive=True)

# Search for files
result = search_files("/path/to/directory", "*.py")

# Get file information
result = get_file_info("/path/to/file.txt")
```

## Using Hybrid Search

The hybrid search module combines keyword-based full-text search with semantic vector similarity search using DuckDB and the FlockMTL extension.

### Running the search demo:

```bash
# Run with default query "vector similarity for search"
python hybrid_search.py

# Run with a custom query and number of results
python hybrid_search.py --query "database systems" --results 5

# Use with API (requires OpenAI API key)
export OPENAI_API_KEY=your_openai_api_key
python hybrid_search.py --mock false
```

### Integrating in Python code:

```python
from hybrid_search import HybridSearch

# Initialize with mock embeddings (no API key needed)
search = HybridSearch(use_mock_embeddings=True)

# Create a database and add documents
documents = [
    {"title": "Document 1", "content": "This is the content of document 1."},
    {"title": "Document 2", "content": "This is the content of document 2."}
]
search.insert_documents(documents)
search.create_indexes()

# Perform search
results = search.hybrid_search("document content", k=5)
for result in results:
    print(f"{result['title']}: {result['score']}")

# Close connection when done
search.close()
```

## Extending

Add new modules to the `modules/` directory. They will be dynamically loaded and can be registered with the agent kernel.