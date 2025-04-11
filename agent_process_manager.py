#!/usr/bin/env python3
"""
Agent Process Manager - Creates and manages autonomous agent processes with FastAPI servers
and Redis PubSub communication for self-improvement.
"""

import os
import sys
import json
import asyncio
import logging
import signal
import uuid
import time
import subprocess
import multiprocessing
from typing import Dict, List, Any, Optional, Set, Union, Callable
import psutil

import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-process-manager")

class AgentConfig(BaseModel):
    """Configuration for an agent process"""
    agent_id: str = Field(default_factory=lambda: f"agent-{uuid.uuid4().hex[:8]}")
    agent_name: str = Field(...)
    agent_type: str = Field(...)
    host: str = Field(default="127.0.0.1")
    port: int = Field(...)
    capabilities: List[str] = Field(default_factory=list)
    model: str = Field(default="gpt-4")
    command: Optional[str] = None
    env_vars: Dict[str, str] = Field(default_factory=dict)
    auto_restart: bool = Field(default=True)
    memory_limit_mb: Optional[int] = None
    cpu_limit_percent: Optional[int] = None
    startup_timeout: int = Field(default=30)
    max_restart_attempts: int = Field(default=3)
    restart_delay: int = Field(default=5)

class AgentInfo(BaseModel):
    """Information about a running agent process"""
    agent_id: str
    agent_name: str
    agent_type: str
    host: str
    port: int
    url: str
    capabilities: List[str]
    status: str = "starting"
    pid: Optional[int] = None
    start_time: float = Field(default_factory=time.time)
    last_heartbeat: float = Field(default_factory=time.time)
    restart_count: int = 0
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    health: Dict[str, Any] = Field(default_factory=dict)

class AgentProcessManager:
    """Manages agent processes with FastAPI servers and Redis PubSub communication"""
    def __init__(self, redis_url=None, host="0.0.0.0", port=8500):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.host = host
        self.port = port
        self.redis_client = None
        self.pubsub = None
        self.pubsub_task = None
        self.agents: Dict[str, AgentInfo] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.websockets: Dict[str, WebSocket] = {}
        self.app = FastAPI(title="Agent Process Manager")
        self.setup_api()
        self._shutdown_event = asyncio.Event()
        self._health_check_task = None
        self._monitor_task = None
        logger.info(f"Agent Process Manager initialized on {host}:{port} with Redis URL: {self.redis_url}")

    def setup_api(self):
        """Setup FastAPI routes and CORS middleware"""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # API routes
        @self.app.get("/")
        async def root():
            return {"status": "online", "service": "Agent Process Manager"}

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "redis_connected": self.redis_client is not None,
                "active_agents": len(self.agents),
                "timestamp": time.time()
            }

        @self.app.get("/agents")
        async def list_agents():
            return {"agents": list(self.agents.values())}

        @self.app.get("/agents/{agent_id}")
        async def get_agent(agent_id: str):
            if agent_id not in self.agents:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            return self.agents[agent_id]

        @self.app.post("/agents")
        async def create_agent(agent_config: AgentConfig, background_tasks: BackgroundTasks):
            # Check if port is already in use
            for agent in self.agents.values():
                if agent.port == agent_config.port:
                    raise HTTPException(status_code=400, detail=f"Port {agent_config.port} is already in use by agent {agent.agent_id}")

            # Create agent info
            agent_info = AgentInfo(
                agent_id=agent_config.agent_id,
                agent_name=agent_config.agent_name,
                agent_type=agent_config.agent_type,
                host=agent_config.host,
                port=agent_config.port,
                url=f"http://{agent_config.host}:{agent_config.port}",
                capabilities=agent_config.capabilities
            )
            self.agents[agent_config.agent_id] = agent_info

            # Start agent in background
            background_tasks.add_task(self.start_agent_process, agent_config)
            return agent_info

        @self.app.delete("/agents/{agent_id}")
        async def delete_agent(agent_id: str):
            if agent_id not in self.agents:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            # Stop agent process if running
            if agent_id in self.processes:
                await self.stop_agent_process(agent_id)

            # Remove agent from agents dict
            agent_info = self.agents.pop(agent_id)
            
            # Publish agent stopped event
            await self.publish_event("agent_stopped", {
                "agent_id": agent_id,
                "agent_name": agent_info.agent_name
            })
            
            return {"status": "success", "message": f"Agent {agent_id} deleted"}

        @self.app.post("/agents/{agent_id}/restart")
        async def restart_agent(agent_id: str, background_tasks: BackgroundTasks):
            if agent_id not in self.agents:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            # Stop agent process if running
            if agent_id in self.processes:
                await self.stop_agent_process(agent_id)

            # Get agent config
            agent_info = self.agents[agent_id]
            agent_config = AgentConfig(
                agent_id=agent_info.agent_id,
                agent_name=agent_info.agent_name,
                agent_type=agent_info.agent_type,
                host=agent_info.host,
                port=agent_info.port,
                capabilities=agent_info.capabilities
            )

            # Start agent in background
            background_tasks.add_task(self.start_agent_process, agent_config)
            return {"status": "restarting", "agent_id": agent_id}

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            client_id = str(uuid.uuid4())
            self.websockets[client_id] = websocket
            
            # Send welcome message
            await websocket.send_json({
                "type": "welcome",
                "client_id": client_id,
                "timestamp": time.time()
            })
            
            try:
                while True:
                    message = await websocket.receive_json()
                    msg_type = message.get("type")
                    
                    if msg_type == "subscribe":
                        # Client wants to subscribe to agent events
                        agent_id = message.get("agent_id")
                        if agent_id and agent_id in self.agents:
                            # Send current agent info
                            await websocket.send_json({
                                "type": "agent_info",
                                "agent_id": agent_id,
                                "data": self.agents[agent_id].dict(),
                                "timestamp": time.time()
                            })
                    
                    elif msg_type == "command":
                        # Client wants to send a command to an agent
                        agent_id = message.get("agent_id")
                        command = message.get("command")
                        data = message.get("data", {})
                        
                        if agent_id and agent_id in self.agents and command:
                            # Publish command to agent's channel
                            await self.publish_event(f"agent:{agent_id}:commands", {
                                "command": command,
                                "data": data,
                                "sender": client_id,
                                "timestamp": time.time()
                            })
                            
                            await websocket.send_json({
                                "type": "command_sent",
                                "agent_id": agent_id,
                                "command": command,
                                "timestamp": time.time()
                            })
            
            except WebSocketDisconnect:
                if client_id in self.websockets:
                    del self.websockets[client_id]
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if client_id in self.websockets:
                    del self.websockets[client_id]

    async def start(self):
        """Start the agent process manager"""
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        
        # Connect to Redis
        await self.connect_redis()
        
        # Start health check task
        self._health_check_task = asyncio.create_task(self.health_check_loop())
        
        # Start process monitoring task
        self._monitor_task = asyncio.create_task(self.monitor_processes())
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def connect_redis(self):
        """Connect to Redis for PubSub communication"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            
            # Initialize PubSub
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe("agent_events")
            
            # Start PubSub listener
            self.pubsub_task = asyncio.create_task(self.pubsub_listener())
            logger.info("Started PubSub listener")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    async def pubsub_listener(self):
        """Listen for PubSub messages from agents"""
        try:
            logger.info("PubSub listener started")
            while not self._shutdown_event.is_set():
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode('utf-8')
                    
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    
                    # Parse JSON data
                    try:
                        event_data = json.loads(data)
                        await self.handle_agent_event(channel, event_data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in message: {data}")
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("PubSub listener cancelled")
        except Exception as e:
            logger.error(f"Error in PubSub listener: {e}")
            # Try to restart the listener if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(1)
                self.pubsub_task = asyncio.create_task(self.pubsub_listener())

    async def handle_agent_event(self, channel, event_data):
        """Handle events from agents"""
        try:
            event_type = event_data.get("type")
            agent_id = event_data.get("agent_id")
            
            if event_type == "heartbeat" and agent_id in self.agents:
                # Update agent heartbeat
                self.agents[agent_id].last_heartbeat = time.time()
                self.agents[agent_id].status = "online"
                
                # Update metrics if provided
                if "memory_usage_mb" in event_data:
                    self.agents[agent_id].memory_usage_mb = event_data["memory_usage_mb"]
                if "cpu_usage_percent" in event_data:
                    self.agents[agent_id].cpu_usage_percent = event_data["cpu_usage_percent"]
            
            elif event_type == "agent_started" and agent_id in self.agents:
                # Update agent status
                self.agents[agent_id].status = "online"
                self.agents[agent_id].pid = event_data.get("pid")
                logger.info(f"Agent {agent_id} ({self.agents[agent_id].agent_name}) started")
                
                # Forward event to WebSocket clients
                await self.broadcast_agent_update(agent_id)
            
            elif event_type == "agent_error" and agent_id in self.agents:
                # Update agent status
                self.agents[agent_id].status = "error"
                self.agents[agent_id].health["error"] = event_data.get("error")
                logger.error(f"Agent {agent_id} ({self.agents[agent_id].agent_name}) error: {event_data.get('error')}")
                
                # Forward event to WebSocket clients
                await self.broadcast_agent_update(agent_id)
            
            elif event_type == "self_improvement" and agent_id in self.agents:
                # Agent has self-improved
                improvement = event_data.get("improvement", {})
                logger.info(f"Agent {agent_id} ({self.agents[agent_id].agent_name}) self-improved: {improvement.get('description')}")
                
                # Update agent capabilities if provided
                if "new_capabilities" in improvement:
                    self.agents[agent_id].capabilities.extend(improvement["new_capabilities"])
                
                # Forward event to WebSocket clients
                await self.broadcast_event("agent_improved", {
                    "agent_id": agent_id,
                    "agent_name": self.agents[agent_id].agent_name,
                    "improvement": improvement
                })
            
        except Exception as e:
            logger.error(f"Error handling agent event: {e}")

    async def publish_event(self, channel, data):
        """Publish an event to Redis PubSub"""
        try:
            if not isinstance(data, dict):
                data = {"data": data}
            
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = time.time()
                
            await self.redis_client.publish(channel, json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Failed to publish event to {channel}: {e}")
            return False

    async def broadcast_agent_update(self, agent_id):
        """Broadcast agent update to all WebSocket clients"""
        if agent_id not in self.agents:
            return
            
        update = {
            "type": "agent_update",
            "agent_id": agent_id,
            "data": self.agents[agent_id].dict(),
            "timestamp": time.time()
        }
        
        # Send to all connected websockets
        for ws in list(self.websockets.values()):
            try:
                await ws.send_json(update)
            except Exception:
                # Ignore errors, clients will be cleaned up on the next message
                pass

    async def broadcast_event(self, event_type, data):
        """Broadcast an event to all WebSocket clients"""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        
        # Send to all connected websockets
        for ws in list(self.websockets.values()):
            try:
                await ws.send_json(event)
            except Exception:
                # Ignore errors, clients will be cleaned up on the next message
                pass

    async def start_agent_process(self, agent_config: AgentConfig):
        """Start an agent process"""
        agent_id = agent_config.agent_id
        
        try:
            # Update agent status
            if agent_id in self.agents:
                self.agents[agent_id].status = "starting"
                self.agents[agent_id].restart_count += 1
            
            # Kill existing process if it exists
            if agent_id in self.processes:
                await self.stop_agent_process(agent_id)
            
            # Create command to start agent
            if agent_config.command:
                cmd = agent_config.command
            else:
                # Default command for different agent types
                if agent_config.agent_type == "cli":
                    cmd = [
                        sys.executable, "-m", "cli_agent",
                        "--port", str(agent_config.port),
                        "--agent-id", agent_id,
                        "--agent-name", agent_config.agent_name,
                        "--model", agent_config.model
                    ]
                elif agent_config.agent_type == "web":
                    cmd = [
                        sys.executable, "-m", "web_app",
                        "--port", str(agent_config.port),
                        "--agent-id", agent_id,
                        "--agent-name", agent_config.agent_name,
                        "--model", agent_config.model
                    ]
                else:
                    # Generic agent
                    cmd = [
                        sys.executable, "-m", f"{agent_config.agent_type}_agent",
                        "--port", str(agent_config.port),
                        "--agent-id", agent_id,
                        "--agent-name", agent_config.agent_name,
                        "--model", agent_config.model
                    ]
            
            # Prepare environment variables
            env = os.environ.copy()
            env["REDIS_URL"] = self.redis_url
            env["AGENT_ID"] = agent_id
            env["AGENT_NAME"] = agent_config.agent_name
            env["AGENT_TYPE"] = agent_config.agent_type
            env["AGENT_PORT"] = str(agent_config.port)
            
            # Add custom environment variables
            for key, value in agent_config.env_vars.items():
                env[key] = value
            
            # Start the process
            logger.info(f"Starting agent process {agent_id} ({agent_config.agent_name}): {cmd}")
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self.processes[agent_id] = process
            
            # Start background tasks to handle stdout/stderr
            asyncio.create_task(self.process_output(agent_id, process.stdout, "stdout"))
            asyncio.create_task(self.process_output(agent_id, process.stderr, "stderr"))
            
            # Wait for agent to start (check health endpoint)
            start_time = time.time()
            started = False
            while time.time() - start_time < agent_config.startup_timeout:
                # Check if process has exited
                if process.poll() is not None:
                    logger.error(f"Agent process {agent_id} exited prematurely with code {process.returncode}")
                    if agent_id in self.agents:
                        self.agents[agent_id].status = "error"
                        self.agents[agent_id].health["error"] = f"Process exited with code {process.returncode}"
                    
                    # Try to restart if configured
                    if agent_config.auto_restart and self.agents[agent_id].restart_count < agent_config.max_restart_attempts:
                        logger.info(f"Restarting agent {agent_id} (attempt {self.agents[agent_id].restart_count + 1})")
                        await asyncio.sleep(agent_config.restart_delay)
                        await self.start_agent_process(agent_config)
                    break
                
                # Wait a bit before checking
                await asyncio.sleep(1)
                
                # Update PID in agent info
                if agent_id in self.agents:
                    self.agents[agent_id].pid = process.pid
                
                # Call health endpoint to check if agent is up
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"http://{agent_config.host}:{agent_config.port}/health", timeout=2) as response:
                            if response.status == 200:
                                started = True
                                logger.info(f"Agent {agent_id} ({agent_config.agent_name}) started successfully")
                                if agent_id in self.agents:
                                    self.agents[agent_id].status = "online"
                                    self.agents[agent_id].last_heartbeat = time.time()
                                break
                except Exception:
                    # Keep trying until timeout
                    pass
            
            if not started:
                logger.error(f"Agent {agent_id} failed to start within timeout period")
                if agent_id in self.agents:
                    self.agents[agent_id].status = "error"
                    self.agents[agent_id].health["error"] = "Failed to start within timeout period"
                
                # Try to restart if configured
                if agent_config.auto_restart and self.agents[agent_id].restart_count < agent_config.max_restart_attempts:
                    logger.info(f"Restarting agent {agent_id} (attempt {self.agents[agent_id].restart_count + 1})")
                    await asyncio.sleep(agent_config.restart_delay)
                    await self.start_agent_process(agent_config)
            
            # Broadcast agent update
            await self.broadcast_agent_update(agent_id)
            
            # Publish agent started event
            await self.publish_event("agent_events", {
                "type": "agent_started",
                "agent_id": agent_id,
                "agent_name": agent_config.agent_name,
                "agent_type": agent_config.agent_type,
                "pid": process.pid,
                "port": agent_config.port
            })
            
        except Exception as e:
            logger.error(f"Error starting agent process {agent_id}: {e}")
            if agent_id in self.agents:
                self.agents[agent_id].status = "error"
                self.agents[agent_id].health["error"] = str(e)
            
            # Try to restart if configured
            if agent_config.auto_restart and agent_id in self.agents and self.agents[agent_id].restart_count < agent_config.max_restart_attempts:
                logger.info(f"Restarting agent {agent_id} (attempt {self.agents[agent_id].restart_count + 1})")
                await asyncio.sleep(agent_config.restart_delay)
                await self.start_agent_process(agent_config)

    async def stop_agent_process(self, agent_id):
        """Stop an agent process"""
        if agent_id not in self.processes:
            return
            
        process = self.processes[agent_id]
        logger.info(f"Stopping agent process {agent_id}")
        
        try:
            # Try graceful shutdown first
            process.terminate()
            
            # Wait a bit for process to terminate
            for _ in range(5):
                if process.poll() is not None:
                    break
                await asyncio.sleep(1)
            
            # Force kill if still running
            if process.poll() is None:
                process.kill()
                await asyncio.sleep(1)
            
            # Update agent status
            if agent_id in self.agents:
                self.agents[agent_id].status = "stopped"
                self.agents[agent_id].pid = None
            
            # Remove from processes dict
            del self.processes[agent_id]
            
            # Broadcast agent update
            await self.broadcast_agent_update(agent_id)
            
            logger.info(f"Agent process {agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Error stopping agent process {agent_id}: {e}")
            return False

    async def process_output(self, agent_id, pipe, pipe_name):
        """Process stdout/stderr from agent process"""
        try:
            for line in iter(pipe.readline, ''):
                if not line:
                    break
                
                # Log output with agent ID
                if pipe_name == "stderr":
                    logger.error(f"[Agent {agent_id}] {line.strip()}")
                else:
                    logger.info(f"[Agent {agent_id}] {line.strip()}")
        except Exception as e:
            logger.error(f"Error processing {pipe_name} for agent {agent_id}: {e}")
        finally:
            pipe.close()

    async def health_check_loop(self):
        """Periodically check health of all agents"""
        try:
            while not self._shutdown_event.is_set():
                current_time = time.time()
                
                for agent_id, agent in list(self.agents.items()):
                    # Check if agent has timed out (no heartbeat for 60 seconds)
                    if agent.status == "online" and current_time - agent.last_heartbeat > 60:
                        logger.warning(f"Agent {agent_id} ({agent.agent_name}) heartbeat timeout")
                        agent.status = "timeout"
                        
                        # Try to restart if process exists
                        if agent_id in self.processes:
                            process = self.processes[agent_id]
                            
                            # Check if process is still running
                            if process.poll() is None:
                                # Process is running but not responding, restart it
                                logger.info(f"Restarting agent {agent_id} due to heartbeat timeout")
                                agent_config = AgentConfig(
                                    agent_id=agent.agent_id,
                                    agent_name=agent.agent_name,
                                    agent_type=agent.agent_type,
                                    host=agent.host,
                                    port=agent.port,
                                    capabilities=agent.capabilities
                                )
                                asyncio.create_task(self.start_agent_process(agent_config))
                            
                        # Broadcast agent update
                        await self.broadcast_agent_update(agent_id)
                
                # Wait before next check
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            logger.info("Health check loop cancelled")
        except Exception as e:
            logger.error(f"Error in health check loop: {e}")
            # Restart the health check loop if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(1)
                self._health_check_task = asyncio.create_task(self.health_check_loop())

    async def monitor_processes(self):
        """Monitor resource usage of agent processes"""
        try:
            while not self._shutdown_event.is_set():
                # Check each process
                for agent_id, agent in list(self.agents.items()):
                    if agent.pid and agent.status == "online":
                        try:
                            # Get process info
                            process = psutil.Process(agent.pid)
                            
                            # Get memory usage (MB)
                            memory_info = process.memory_info()
                            memory_mb = memory_info.rss / (1024 * 1024)
                            agent.memory_usage_mb = memory_mb
                            
                            # Get CPU usage
                            cpu_percent = process.cpu_percent(interval=0.1)
                            agent.cpu_usage_percent = cpu_percent
                            
                            # Check resource limits
                            if hasattr(agent, "memory_limit_mb") and agent.memory_limit_mb and memory_mb > agent.memory_limit_mb:
                                logger.warning(f"Agent {agent_id} exceeded memory limit: {memory_mb:.2f}MB > {agent.memory_limit_mb}MB")
                                
                                # Restart the agent
                                logger.info(f"Restarting agent {agent_id} due to memory limit exceeded")
                                agent_config = AgentConfig(
                                    agent_id=agent.agent_id,
                                    agent_name=agent.agent_name,
                                    agent_type=agent.agent_type,
                                    host=agent.host,
                                    port=agent.port,
                                    capabilities=agent.capabilities,
                                    memory_limit_mb=agent.memory_limit_mb
                                )
                                asyncio.create_task(self.start_agent_process(agent_config))
                            
                            if hasattr(agent, "cpu_limit_percent") and agent.cpu_limit_percent and cpu_percent > agent.cpu_limit_percent:
                                logger.warning(f"Agent {agent_id} exceeded CPU limit: {cpu_percent:.2f}% > {agent.cpu_limit_percent}%")
                                
                                # Slow down the agent by reducing its priority
                                try:
                                    process.nice(10)  # Lower priority
                                except Exception:
                                    pass
                                
                        except psutil.NoSuchProcess:
                            # Process no longer exists
                            if agent.status == "online":
                                logger.warning(f"Agent process {agent_id} ({agent.agent_name}) not found, marking as crashed")
                                agent.status = "crashed"
                                agent.pid = None
                                
                                # Broadcast agent update
                                await self.broadcast_agent_update(agent_id)
                        except Exception as e:
                            logger.error(f"Error monitoring agent {agent_id}: {e}")
                
                # Wait before next check
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("Process monitor cancelled")
        except Exception as e:
            logger.error(f"Error in process monitor: {e}")
            # Restart the monitor task if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(1)
                self._monitor_task = asyncio.create_task(self.monitor_processes())

    async def shutdown(self):
        """Shutdown the agent process manager"""
        logger.info("Shutting down Agent Process Manager...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop all agent processes
        for agent_id in list(self.processes.keys()):
            await self.stop_agent_process(agent_id)
        
        # Cancel background tasks
        if self.pubsub_task:
            self.pubsub_task.cancel()
            try:
                await self.pubsub_task
            except asyncio.CancelledError:
                pass
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Close Redis connection
        if self.pubsub:
            await self.pubsub.unsubscribe()
        
        if self.redis_client:
            await self.redis_client.close()
        
        # Close all WebSocket connections
        for ws in list(self.websockets.values()):
            try:
                await ws.close()
            except Exception:
                pass
        
        logger.info("Agent Process Manager shutdown complete")

class AgentServer:
    """Base class for agent servers with FastAPI and Redis PubSub integration"""
    def __init__(self, agent_id=None, agent_name=None, agent_type=None, 
                 host="127.0.0.1", port=8600, redis_url=None, model="gpt-4"):
        self.agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.agent_name = agent_name or f"Agent-{self.agent_id[:4]}"
        self.agent_type = agent_type or "generic"
        self.host = host
        self.port = port
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.model = model
        self.capabilities = []
        self.redis_client = None
        self.pubsub = None
        self.pubsub_task = None
        self.app = FastAPI(title=f"{self.agent_name} API")
        self.setup_api()
        self._shutdown_event = asyncio.Event()
        self._heartbeat_task = None
        logger.info(f"Agent {self.agent_id} ({self.agent_name}) initialized on {host}:{port} with Redis URL: {self.redis_url}")
    
    def setup_api(self):
        """Setup FastAPI routes and CORS middleware"""
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # API routes
        @self.app.get("/")
        async def root():
            return {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "agent_type": self.agent_type,
                "status": "online"
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "timestamp": time.time(),
                "uptime": time.time() - self.start_time,
                "redis_connected": self.redis_client is not None
            }
        
        @self.app.get("/capabilities")
        async def get_capabilities():
            return {"capabilities": self.capabilities}
    
    async def start(self):
        """Start the agent server"""
        self.start_time = time.time()
        
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        
        # Connect to Redis
        await self.connect_redis()
        
        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self.heartbeat_loop())
        
        # Publish agent started event
        await self.publish_event("agent_events", {
            "type": "agent_started",
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "pid": os.getpid(),
            "port": self.port
        })
        
        # Start FastAPI server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def connect_redis(self):
        """Connect to Redis for PubSub communication"""
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info(f"Agent {self.agent_id} connected to Redis successfully")
            
            # Initialize PubSub
            self.pubsub = self.redis_client.pubsub()
            
            # Subscribe to agent-specific channels
            await self.pubsub.subscribe(f"agent:{self.agent_id}:commands")
            
            # Start PubSub listener
            self.pubsub_task = asyncio.create_task(self.pubsub_listener())
            logger.info(f"Agent {self.agent_id} started PubSub listener")
            
            return True
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to connect to Redis: {e}")
            return False
    
    async def pubsub_listener(self):
        """Listen for PubSub messages"""
        try:
            logger.info(f"Agent {self.agent_id} PubSub listener started")
            while not self._shutdown_event.is_set():
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode('utf-8')
                    
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    
                    # Parse JSON data
                    try:
                        command_data = json.loads(data)
                        await self.handle_command(channel, command_data)
                    except json.JSONDecodeError:
                        logger.error(f"Agent {self.agent_id} received invalid JSON in message: {data}")
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info(f"Agent {self.agent_id} PubSub listener cancelled")
        except Exception as e:
            logger.error(f"Error in Agent {self.agent_id} PubSub listener: {e}")
            # Try to restart the listener if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(1)
                self.pubsub_task = asyncio.create_task(self.pubsub_listener())
    
    async def handle_command(self, channel, command_data):
        """Handle commands from PubSub"""
        try:
            command = command_data.get("command")
            data = command_data.get("data", {})
            sender = command_data.get("sender")
            
            logger.info(f"Agent {self.agent_id} received command: {command} from {sender}")
            
            # Handle different commands
            if command == "ping":
                # Respond to ping
                await self.publish_event(f"agent:{self.agent_id}:responses", {
                    "type": "pong",
                    "agent_id": self.agent_id,
                    "receiver": sender,
                    "timestamp": time.time()
                })
            
            elif command == "shutdown":
                # Shutdown the agent
                logger.info(f"Agent {self.agent_id} received shutdown command from {sender}")
                await self.shutdown()
            
            elif command == "update_capabilities":
                # Update capabilities
                if "capabilities" in data:
                    self.capabilities = data["capabilities"]
                    logger.info(f"Agent {self.agent_id} updated capabilities: {self.capabilities}")
                    
                    # Publish capabilities updated event
                    await self.publish_event("agent_events", {
                        "type": "capabilities_updated",
                        "agent_id": self.agent_id,
                        "capabilities": self.capabilities
                    })
            
            else:
                # Override in subclasses to handle specific commands
                await self.process_command(command, data, sender)
        
        except Exception as e:
            logger.error(f"Error handling command in Agent {self.agent_id}: {e}")
    
    async def process_command(self, command, data, sender):
        """Process custom commands - override in subclasses"""
        # Handle standard capability discovery
        if command == "get_capabilities":
            # Send capabilities information
            await self.publish_event(f"agent:{self.agent_id}:responses", {
                "type": "capabilities",
                "agent_id": self.agent_id,
                "receiver": sender,
                "capabilities": self.capabilities,
                "timestamp": time.time()
            })
        # Handle brain dump - send all internal knowledge
        elif command == "brain_dump":
            # Collect all knowledge and methods this agent has
            knowledge = {
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "agent_type": self.agent_type,
                "capabilities": self.capabilities,
                "methods": [method for method in dir(self) if callable(getattr(self, method)) and not method.startswith("_")]
            }
            
            # Add any custom attributes
            for attr in dir(self):
                if not attr.startswith("_") and not callable(getattr(self, attr)) and attr not in knowledge:
                    try:
                        value = getattr(self, attr)
                        # Only include serializable values
                        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                            knowledge[attr] = value
                    except:
                        pass
            
            await self.publish_event(f"agent:{self.agent_id}:responses", {
                "type": "brain_dump",
                "agent_id": self.agent_id,
                "receiver": sender,
                "knowledge": knowledge,
                "timestamp": time.time()
            })
        # Handle learning request - one agent teaching another
        elif command == "teach":
            capability = data.get("capability")
            implementation = data.get("implementation")
            description = data.get("description", "")
            
            if hasattr(self, "improve_capability") and callable(getattr(self, "improve_capability")):
                # Agent has improve_capability method, use it
                asyncio.create_task(self.improve_capability(capability, description, implementation))
                
                await self.publish_event(f"agent:{self.agent_id}:responses", {
                    "type": "learning",
                    "agent_id": self.agent_id,
                    "receiver": sender,
                    "message": f"Learning capability: {capability}",
                    "timestamp": time.time()
                })
            else:
                # Agent doesn't support learning
                await self.publish_event(f"agent:{self.agent_id}:responses", {
                    "type": "error",
                    "agent_id": self.agent_id,
                    "receiver": sender,
                    "error": "This agent doesn't support learning new capabilities",
                    "timestamp": time.time()
                })
        else:
            # Command not supported
            logger.warning(f"Agent {self.agent_id} received unhandled command: {command}")
            
            await self.publish_event(f"agent:{self.agent_id}:responses", {
                "type": "error",
                "agent_id": self.agent_id,
                "receiver": sender,
                "error": f"Command not supported: {command}",
                "timestamp": time.time()
            })
    
    async def heartbeat_loop(self):
        """Send periodic heartbeats to inform manager agent is alive"""
        try:
            while not self._shutdown_event.is_set():
                # Get current resource usage
                try:
                    import psutil
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    cpu_percent = process.cpu_percent(interval=0.1)
                except Exception:
                    memory_mb = None
                    cpu_percent = None
                
                # Send heartbeat
                await self.publish_event("agent_events", {
                    "type": "heartbeat",
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "timestamp": time.time(),
                    "memory_usage_mb": memory_mb,
                    "cpu_usage_percent": cpu_percent
                })
                
                # Wait for next heartbeat
                await asyncio.sleep(15)
        except asyncio.CancelledError:
            logger.info(f"Agent {self.agent_id} heartbeat loop cancelled")
        except Exception as e:
            logger.error(f"Error in Agent {self.agent_id} heartbeat loop: {e}")
            # Try to restart the heartbeat loop if it fails
            if not self._shutdown_event.is_set():
                await asyncio.sleep(1)
                self._heartbeat_task = asyncio.create_task(self.heartbeat_loop())
    
    async def publish_event(self, channel, data):
        """Publish an event to Redis PubSub"""
        try:
            if not isinstance(data, dict):
                data = {"data": data}
            
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = time.time()
                
            await self.redis_client.publish(channel, json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to publish event to {channel}: {e}")
            return False
    
    async def self_improve(self, improvement_description, code_changes=None, new_capabilities=None):
        """Report self-improvement to the agent manager"""
        improvement = {
            "description": improvement_description,
            "timestamp": time.time()
        }
        
        if code_changes:
            improvement["code_changes"] = code_changes
            
        if new_capabilities:
            improvement["new_capabilities"] = new_capabilities
            # Update local capabilities
            for capability in new_capabilities:
                if capability not in self.capabilities:
                    self.capabilities.append(capability)
        
        # Publish self-improvement event
        await self.publish_event("agent_events", {
            "type": "self_improvement",
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "improvement": improvement
        })
        
        logger.info(f"Agent {self.agent_id} reported self-improvement: {improvement_description}")
        return True
    
    async def chat_with_agent(self, target_agent_id, message, timeout=30):
        """Send a chat message to another agent and wait for response"""
        try:
            # Generate a unique message ID
            message_id = str(uuid.uuid4())
            
            # Subscribe to response channel
            response_channel = f"agent:{self.agent_id}:responses"
            response_pubsub = self.redis_client.pubsub()
            await response_pubsub.subscribe(response_channel)
            
            # Create a future to be resolved when response is received
            loop = asyncio.get_running_loop()
            response_future = loop.create_future()
            
            # Start listener task
            async def listen_for_response():
                try:
                    while not response_future.done():
                        message = await response_pubsub.get_message(timeout=1)
                        if message and message["type"] == "message":
                            try:
                                data = json.loads(message["data"])
                                if data.get("message_id") == message_id and data.get("type") == "chat_response":
                                    if not response_future.done():
                                        response_future.set_result(data)
                                        break
                            except json.JSONDecodeError:
                                pass
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in response listener: {e}")
                    if not response_future.done():
                        response_future.set_exception(e)
                finally:
                    await response_pubsub.unsubscribe(response_channel)
            
            # Start listener
            listener_task = asyncio.create_task(listen_for_response())
            
            # Send chat message
            await self.publish_event(f"agent:{target_agent_id}:commands", {
                "command": "chat",
                "message_id": message_id,
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
                "message": message,
                "timestamp": time.time()
            })
            
            try:
                # Wait for response or timeout
                response = await asyncio.wait_for(response_future, timeout)
                return {
                    "success": True,
                    "message": response.get("message"),
                    "data": response.get("data", {})
                }
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for response from agent {target_agent_id}")
                return {
                    "success": False,
                    "error": f"Timeout waiting for response from agent {target_agent_id}"
                }
            finally:
                # Cancel listener task
                listener_task.cancel()
                try:
                    await listener_task
                except asyncio.CancelledError:
                    pass
        
        except Exception as e:
            logger.error(f"Error chatting with agent {target_agent_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def shutdown(self):
        """Shutdown the agent server"""
        logger.info(f"Agent {self.agent_id} ({self.agent_name}) shutting down...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Publish agent stopped event
        await self.publish_event("agent_events", {
            "type": "agent_stopped",
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "timestamp": time.time()
        })
        
        # Cancel background tasks
        if self.pubsub_task:
            self.pubsub_task.cancel()
            try:
                await self.pubsub_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Close Redis connection
        if self.pubsub:
            await self.pubsub.unsubscribe()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info(f"Agent {self.agent_id} ({self.agent_name}) shutdown complete")

def run_agent_manager(host="0.0.0.0", port=8500):
    """Run the agent process manager"""
    async def main():
        manager = AgentProcessManager(host=host, port=port)
        await manager.start()
    
    asyncio.run(main())

def run_agent_server(agent_id=None, agent_name=None, agent_type=None, 
                    host="127.0.0.1", port=8600, model="gpt-4"):
    """Run an agent server"""
    async def main():
        agent = AgentServer(
            agent_id=agent_id, 
            agent_name=agent_name, 
            agent_type=agent_type,
            host=host, 
            port=port,
            model=model
        )
        await agent.start()
    
    asyncio.run(main())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent Process Manager")
    parser.add_argument("--mode", choices=["manager", "agent"], default="manager", help="Run as manager or agent")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8500, help="Port to bind to")
    parser.add_argument("--agent-id", help="Agent ID (for agent mode)")
    parser.add_argument("--agent-name", help="Agent name (for agent mode)")
    parser.add_argument("--agent-type", default="generic", help="Agent type (for agent mode)")
    parser.add_argument("--model", default="gpt-4", help="Model to use (for agent mode)")
    
    args = parser.parse_args()
    
    if args.mode == "manager":
        run_agent_manager(host=args.host, port=args.port)
    else:
        run_agent_server(
            agent_id=args.agent_id,
            agent_name=args.agent_name,
            agent_type=args.agent_type,
            host=args.host,
            port=args.port,
            model=args.model
        )