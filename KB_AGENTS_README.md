# Knowledge Base Agents System

This extension adds the ability to dispatch to specialized sub-agents that work with knowledge base files stored in the `knowledge_bases` directory.

## Components

The system consists of the following components:

1. **KnowledgeBaseDispatcher**: Creates and manages specialized agents for each knowledge base JSON file.
2. **KnowledgeBaseConnector**: Provides a high-level API for interacting with knowledge base agents.
3. **CLIAgent Integration**: Patches the main CLI agent to use the knowledge base dispatcher.
4. **Demo Script**: Demonstrates the capabilities of the knowledge base agents.

## Files

- `knowledge_base_dispatcher.py`: Core dispatcher that creates sub-agents for each knowledge base.
- `kb_agent_connector.py`: Connector for integrating with other agent systems.
- `kb_cli_integration.py`: Patches the CLIAgent class to use the knowledge base dispatcher.
- `kb_agents_demo.py`: Standalone script for demonstrating knowledge base agents.

## Usage

### Demo Script

The `kb_agents_demo.py` script provides a standalone interface for exploring knowledge base agents:

```bash
# Run in interactive mode
./kb_agents_demo.py --interactive

# List all available knowledge bases
./kb_agents_demo.py --list

# Search a specific knowledge base
./kb_agents_demo.py --search <kb_name> <query>

# Search all knowledge bases
./kb_agents_demo.py --search-all <query>

# Get information about a knowledge base
./kb_agents_demo.py --info <kb_name>

# List entries in a knowledge base
./kb_agents_demo.py --entries <kb_name> [limit]

# Get a specific entry from a knowledge base
./kb_agents_demo.py --entry <kb_name> <entry_id>
```

### Programmatic Usage

To use the knowledge base agents in your own code:

```python
import asyncio
from kb_agent_connector import connector

async def example():
    # List available knowledge bases
    kb_list = connector.get_available_knowledge_bases()
    
    # Search a knowledge base
    if kb_list:
        kb_name = kb_list[0]["name"]
        result = await connector.dispatch_to_kb_agent(kb_name, "search definition")
        print(result)
    
    # Search all knowledge bases
    all_results = await connector.dispatch_query_to_all_kbs("science")
    print(all_results)

asyncio.run(example())
```

### Integration with CLIAgent

To integrate with the main CLIAgent:

```python
from kb_cli_integration import apply_kb_integration_patches

# Apply patches to CLIAgent
apply_kb_integration_patches()

# Now the CLIAgent has enhanced knowledge base capabilities
```

## Features

The knowledge base agents provide the following capabilities:

1. **Search**: Search for information within knowledge bases using natural language queries.
2. **Metadata**: Get information about knowledge bases.
3. **Entry Listing**: List entries contained in knowledge bases.
4. **Entry Retrieval**: Get specific entries from knowledge bases.
5. **Aggregation**: Search across all knowledge bases and aggregate results.

## Architecture

The system uses a dispatcher-agent architecture:

1. The dispatcher scans the knowledge base directory and creates specialized agents
2. Each agent has a context containing knowledge base information
3. The connector provides a high-level API for interacting with these agents
4. The CLIAgent integration extends the main agent with the new capabilities

## Extending

To add new capabilities to knowledge base agents:

1. Extend the `KnowledgeBaseAgent` class in `dynamic_agents.py`
2. Register new capabilities in the `_register_capabilities` method
3. Create a new command function following the pattern `cmd_<capability_name>`