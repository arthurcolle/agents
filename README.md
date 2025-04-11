# Llama4 Agent System

A sophisticated agent system built on advanced language models with dynamic code generation, self-modification capabilities, and inter-agent communication.

## Key Features

- **Distributed Agent Architecture**: Multiple autonomous agents running as separate processes
- **FastAPI Integration**: Each agent exposes a REST API with WebSocket support
- **Redis PubSub Communication**: Realtime messaging between agents
- **Self-Improving Capabilities**: Agents can improve their own code and capabilities
- **Process Manager**: Centralized manager for monitoring and controlling agents
- **Fault Tolerance**: Automatic restart of crashed agents
- **Resource Monitoring**: Track memory and CPU usage of agent processes
- **Dynamic Code Extension**: Add new capabilities without restarting
- **Agent Collaboration**: Agents can communicate and collaborate on tasks
- **Hybrid Search**: Combined semantic and keyword search capabilities
- **CLI and Web Interfaces**: Multiple interaction options

## Components

- `agent_process_manager.py`: Main process manager for controlling agent lifecycle
- `self_improving_agent.py`: Self-improving agent with dynamic capability learning
- `start_agent_system.py`: Script to launch the entire agent system
- `agent_config.json`: Configuration for the agent system
- `pubsub_service.py`: Redis PubSub service for real-time communication
- `distributed_services.py`: Base services for distributed agent architecture
- `web_app.py`: Web interface for interacting with agents
- `cli_agent.py`: Command line interface agent

## Architecture

The system follows a distributed architecture with these key components:

1. **Agent Process Manager**: Central process that manages agent lifecycles, monitors health, and provides a management API

2. **Individual Agents**: Each agent runs as a separate process with:
   - FastAPI server for REST API and WebSocket
   - Redis PubSub for inter-agent messaging
   - Capabilities specific to the agent type
   - Self-improvement mechanisms

3. **Communication Layer**: Redis PubSub for real-time messaging between agents

4. **Web Interface**: Optional web UI for interacting with the agent system

## Agent Types

- **Self-Improving Agent**: Can learn new capabilities and improve its own code
- **CLI Agent**: Command line interface for user interaction
- **Web Agent**: Web interface for browser-based interaction
- **Custom Agents**: Define specialized agents for specific tasks

## Setup & Usage

1. Set up your environment:
   ```
   pip install -r requirements.txt
   
   # Start Redis server for PubSub communication
   redis-server
   
   # Set API keys as needed
   export OPENAI_API_KEY=your_api_key
   # OR
   export ANTHROPIC_API_KEY=your_api_key
   ```

2. Start the agent system:
   ```
   # Start with default configuration
   python start_agent_system.py
   
   # Start with custom config
   python start_agent_system.py --config agent_config.json
   ```

3. Or start components individually:
   ```
   # Start process manager
   python agent_process_manager.py --mode manager --port 8500
   
   # Start a self-improving agent
   python self_improving_agent.py --port 8600 --model gpt-4
   ```

## API Endpoints

### Agent Process Manager (default port 8500)

- `GET /agents`: List all running agents
- `GET /agents/{agent_id}`: Get details about a specific agent
- `POST /agents`: Create a new agent
- `DELETE /agents/{agent_id}`: Stop and remove an agent
- `POST /agents/{agent_id}/restart`: Restart an agent
- `WebSocket /ws`: Real-time updates about agents

### Self-Improving Agent (default port 8600)

- `GET /health`: Health check endpoint
- `GET /capabilities`: List agent capabilities
- `POST /chat`: Send a chat message
- `POST /improve`: Request capability improvement
- `POST /analyze`: Analyze code for improvements
- `GET /memory`: Query agent's memory
- `POST /upload_improvement`: Upload code to improve the agent

## Inter-Agent Communication

Agents can communicate with each other using Redis PubSub channels:

- `agent:{agent_id}:commands`: Send commands to a specific agent
- `agent:{agent_id}:responses`: Receive responses from an agent
- `agent_events`: System-wide events (agent started, agent stopped, etc.)

Example communication pattern:

```python
# Agent A sends a chat message to Agent B
await agent_a.publish_event(f"agent:{agent_b_id}:commands", {
    "command": "chat",
    "message_id": message_id,
    "agent_id": agent_a_id,
    "message": "Hello, can you help me with this task?",
    "timestamp": time.time()
})

# Agent B receives the message and responds
await agent_b.publish_event(f"agent:{agent_b_id}:responses", {
    "type": "chat_response",
    "message_id": message_id,
    "agent_id": agent_b_id,
    "receiver": agent_a_id,
    "message": "Yes, I can help with that task.",
    "timestamp": time.time()
})
```

## Self-Improvement Process

The self-improving agent can learn new capabilities through:

1. **User requests**: Direct API calls to improve specific capabilities
2. **Self-detection**: Automatically detecting improvement opportunities during conversations
3. **Code analysis**: Analyzing existing code for improvement opportunities
4. **Agent collaboration**: Learning from other agents in the system

When an improvement is made:

1. The agent generates code for the new capability using an LLM
2. The code is validated for syntax and security
3. The capability is dynamically added to the agent
4. The improvement is reported to the agent process manager
5. The new capability becomes immediately available

## Configuration

See `agent_config.json` for an example configuration:

```json
{
  "manager": {
    "host": "0.0.0.0",
    "port": 8500
  },
  "agents": [
    {
      "agent_type": "self_improving",
      "agent_name": "Learning Agent",
      "port": 8600,
      "model": "gpt-4",
      "env_vars": {
        "IMPROVEMENT_LOGGING": "detailed"
      }
    },
    {
      "agent_type": "cli",
      "agent_name": "CLI Helper",
      "port": 8700,
      "model": "gpt-4"
    }
  ]
}
```

## Extending

To create a new agent type:

1. Create a new Python file for your agent (e.g., `my_agent.py`)
2. Extend the `AgentServer` class from `agent_process_manager.py`
3. Implement custom methods and override `process_command` as needed
4. Add the agent to your configuration file

Example:

```python
from agent_process_manager import AgentServer

class MySpecialAgent(AgentServer):
    def __init__(self, agent_id=None, agent_name=None, 
                host="127.0.0.1", port=8600, redis_url=None, model="gpt-4"):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name or "My Special Agent",
            agent_type="special",
            host=host,
            port=port,
            redis_url=redis_url,
            model=model
        )
        self.capabilities = ["special_task", "another_capability"]
        
    async def process_command(self, command, data, sender):
        if command == "special_task":
            # Handle special task command
            result = await self.do_special_task(data)
            
            # Send response back
            await self.publish_event(f"agent:{self.agent_id}:responses", {
                "type": "special_task_result",
                "agent_id": self.agent_id,
                "receiver": sender,
                "result": result,
                "timestamp": time.time()
            })
        else:
            # Fall back to parent implementation for unknown commands
            await super().process_command(command, data, sender)
            
    async def do_special_task(self, data):
        # Implement your special task here
        return {"success": True, "message": "Special task completed"}
```

## Security Considerations

- Agents run as separate processes with proper isolation
- LLM-generated code is validated before execution
- Resource limits can be applied to prevent runaway processes
- API endpoints should be properly secured in production deployments