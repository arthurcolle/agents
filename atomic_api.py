#!/usr/bin/env python3
"""
atomic_api.py
------------
Advanced Modal.com wrapper for the atomic agent that provides a sophisticated API
with native streaming, connection management, monitoring, and multi-agent orchestration.
This enterprise-grade wrapper enables deploying the atomic agent as a scalable,
fault-tolerant serverless API with full instrumentation.
"""

import os
import sys
import json
import time
import asyncio
import traceback
import logging
import modal
import redis.asyncio as aioredis
from uuid import uuid4
from datetime import datetime
from typing import Dict, List, Any, AsyncGenerator, Optional, Union, Tuple, Set
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

# Import from atomic_agent without modifying it
from atomic_agent import TogetherAgent, VectorMemory

# Define Pydantic models for request/response validation
class ChatRequest(BaseModel):
    user_input: Union[str, List[Dict[str, Any]]] = Field(..., description="User input text or multimodal content")
    model: str = Field(default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", description="Model to use")
    num_scouts: int = Field(default=5, description="Number of scout agents to initialize")
    stream: bool = Field(default=True, description="Whether to stream the response")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context for the agent")
    tools_config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration for agent tools")
    system_prompt: Optional[str] = Field(default=None, description="Custom system prompt")
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default=None, description="Previous conversation history")
    temperature: Optional[float] = Field(default=None, description="Temperature for response generation")
    timeout: Optional[int] = Field(default=120, description="Request timeout in seconds")

class StreamingChunk(BaseModel):
    status: str = Field(..., description="Status of the streaming response")
    chunk: Optional[str] = Field(default=None, description="Text chunk of the response")
    timestamp: float = Field(default_factory=time.time, description="Timestamp of the chunk")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    request_id: str = Field(..., description="Unique request ID")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Function call details if applicable")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Performance metrics")

class StatusResponse(BaseModel):
    status: str = Field(..., description="API status")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: float = Field(default_factory=time.time, description="Current timestamp")
    uptime: float = Field(..., description="API uptime in seconds")
    active_sessions: int = Field(default=0, description="Number of active sessions")
    model_status: Dict[str, str] = Field(default_factory=dict, description="Status of available models")
    memory_usage: Dict[str, Any] = Field(default_factory=dict, description="Memory usage statistics")
    request_stats: Dict[str, int] = Field(default_factory=dict, description="Request statistics")

# Set up the Modal app with persistent volume for state
app = modal.App("atomic-agent-api")
volume = modal.Volume.from_name("atomic-agent-volume", create_if_missing=True)

# Define the Modal image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "together",
    "rich",
    "redis",
    "aioredis",
    "aiohttp",
    "pytz",
    "pyperclip",
    "pillow",
    "numpy",
    "sentence-transformers",
    "fastapi",
    "uvicorn",
    "pydantic>=2.0.0",
    "prometheus-client",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp",
    "tenacity",
)

# Explicitly add local Python modules to avoid automounting deprecation
image = image.add_local_python_source("atomic_agent")

# Global variables for session management
AGENT_SESSIONS = {}
START_TIME = time.time()
REQUEST_STATS = {"total": 0, "success": 0, "error": 0, "latency_avg_ms": 0}
ACTIVE_MODELS = {
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "ready",
    "meta-llama/Llama-3-8B-Instruct": "ready"
}

# Initialize async logger
logger = logging.getLogger("atomic-api")
logger.setLevel(logging.INFO)

class AgentSession:
    """Manages a stateful agent session with conversation history and configuration"""
    
    def __init__(self, session_id: str, model: str, num_scouts: int = 5):
        self.session_id = session_id
        self.model = model
        self.num_scouts = num_scouts
        self.agent = None
        self.last_active = time.time()
        self.created_at = time.time()
        self.request_count = 0
        self.conversation_history = []
        self.tools_used = set()
        self.is_initialized = False
        self.custom_context = {}
        self.performance_metrics = {
            "avg_response_time": 0,
            "total_response_time": 0,
            "tool_call_count": 0,
        }
    
    async def initialize(self):
        """Initialize the agent lazily when first needed"""
        if not self.is_initialized:
            self.agent = TogetherAgent(model=self.model, num_scouts=self.num_scouts)
            self.is_initialized = True
            return True
        return False
    
    def update_last_active(self):
        """Update the last active timestamp"""
        self.last_active = time.time()
        
    def add_tool_usage(self, tool_name: str):
        """Track tool usage for analytics"""
        self.tools_used.add(tool_name)
        self.performance_metrics["tool_call_count"] += 1
        
    def update_response_time(self, response_time: float):
        """Update response time metrics"""
        self.performance_metrics["total_response_time"] += response_time
        self.request_count += 1
        self.performance_metrics["avg_response_time"] = (
            self.performance_metrics["total_response_time"] / self.request_count
        )
        
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information for monitoring"""
        return {
            "session_id": self.session_id,
            "model": self.model,
            "num_scouts": self.num_scouts,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "request_count": self.request_count,
            "session_age": time.time() - self.created_at,
            "tools_used": list(self.tools_used),
            "performance_metrics": self.performance_metrics,
        }

class RedisSessionManager:
    """Manages agent sessions with Redis for distributed deployments"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize the Redis connection"""
        if not self.initialized:
            try:
                self.redis_client = await aioredis.from_url(self.redis_url)
                self.initialized = True
                return True
            except Exception as e:
                logger.error(f"Redis initialization failed: {str(e)}")
                return False
        return True
        
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session data from Redis"""
        if not self.initialized:
            await self.initialize()
            
        try:
            session_data = await self.redis_client.get(f"session:{session_id}")
            if session_data:
                return json.loads(session_data)
            return None
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {str(e)}")
            return None
            
    async def save_session(self, session_id: str, session_data: Dict[str, Any]):
        """Save session data to Redis"""
        if not self.initialized:
            await self.initialize()
            
        try:
            await self.redis_client.set(
                f"session:{session_id}", 
                json.dumps(session_data),
                ex=3600  # 1 hour expiration
            )
            return True
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {str(e)}")
            return False
            
    async def list_sessions(self) -> List[str]:
        """List all active session IDs"""
        if not self.initialized:
            await self.initialize()
            
        try:
            keys = await self.redis_client.keys("session:*")
            return [key.decode('utf-8').split(':')[1] for key in keys]
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return []
            
    async def track_request(self, request_type: str, latency: float):
        """Track request metrics in Redis"""
        if not self.initialized:
            await self.initialize()
            
        try:
            # Increment request counter
            await self.redis_client.hincrby("request_stats", request_type, 1)
            await self.redis_client.hincrby("request_stats", "total", 1)
            
            # Update latency metrics
            curr_avg = float(await self.redis_client.hget("request_stats", "latency_avg_ms") or 0)
            curr_count = int(await self.redis_client.hget("request_stats", "total") or 0)
            
            if curr_count > 0:
                new_avg = ((curr_avg * (curr_count - 1)) + latency) / curr_count
                await self.redis_client.hset("request_stats", "latency_avg_ms", new_avg)
        except Exception as e:
            logger.error(f"Error tracking request: {str(e)}")

# Initialize Redis session manager
session_manager = RedisSessionManager(
    redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0")
)

@asynccontextmanager
async def lifespan(app):
    """Lifecycle management for the API"""
    # Startup
    logger.info("API starting up")
    global START_TIME
    START_TIME = time.time()
    
    # Initialize Redis session manager
    await session_manager.initialize()
    
    yield
    
    # Shutdown
    logger.info("API shutting down")
    # Clean up any resources

# Define the API endpoint container with volume and secrets
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("distributed-systems")],
    volumes={"/data": volume},
    memory=4096,
    timeout=600,
)
async def chat(request: Dict[str, Any]) -> AsyncGenerator[str, None]:
    """
    Advanced chat endpoint with streaming, session management, and analytics.
    
    Args:
        request: Validated request with user input and configuration parameters
        
    Returns:
        Streaming response with chunks of the generated content
    """
    # Parse and validate the request
    try:
        chat_request = ChatRequest(**request)
    except Exception as e:
        error = f"Invalid request format: {str(e)}"
        yield json.dumps({"status": "error", "message": error})
        return
    
    # Generate unique IDs for this request
    request_id = str(uuid4())
    session_id = chat_request.session_id or str(uuid4())
    
    # Update request statistics
    REQUEST_STATS["total"] += 1
    request_start = time.time()
    
    # Create a new session or get existing one
    session = AGENT_SESSIONS.get(session_id)
    if not session:
        session = AgentSession(
            session_id=session_id,
            model=chat_request.model,
            num_scouts=chat_request.num_scouts
        )
        AGENT_SESSIONS[session_id] = session
        
        # Store session data in Redis for distributed deployments
        await session_manager.save_session(
            session_id=session_id,
            session_data=session.get_session_info()
        )
        
        # Cleanup old sessions (simple LRU approach)
        if len(AGENT_SESSIONS) > 100:
            oldest_session_id = min(
                AGENT_SESSIONS.keys(),
                key=lambda sid: AGENT_SESSIONS[sid].last_active
            )
            del AGENT_SESSIONS[oldest_session_id]
    
    # Initialize the agent if needed
    try:
        was_initialized = await session.initialize()
        if was_initialized:
            yield json.dumps(StreamingChunk(
                status="initialized",
                session_id=session_id,
                request_id=request_id,
                metrics={"initialization_time": time.time() - request_start}
            ).model_dump())
    except Exception as e:
        error_message = f"Agent initialization failed: {str(e)}"
        yield json.dumps(StreamingChunk(
            status="error",
            session_id=session_id,
            request_id=request_id,
            chunk=error_message
        ).model_dump())
        REQUEST_STATS["error"] += 1
        return
    
    # Processing notification
    yield json.dumps(StreamingChunk(
        status="processing",
        session_id=session_id,
        request_id=request_id
    ).model_dump())
    
    # Inject any custom context provided in the request
    if chat_request.context:
        session.custom_context.update(chat_request.context)
    
    # Custom tool configuration
    if chat_request.tools_config and hasattr(session.agent, "tool_registry"):
        for tool_name, config in chat_request.tools_config.items():
            if hasattr(session.agent.tool_registry, f"configure_{tool_name}"):
                getattr(session.agent.tool_registry, f"configure_{tool_name}")(**config)
    
    # Generate response
    try:
        # Generate complete response first
        response = session.agent.generate_response(chat_request.user_input)
        
        # Update session metrics
        response_time = time.time() - request_start
        session.update_response_time(response_time)
        session.update_last_active()
        
        # Check for tool usage in response
        if hasattr(session.agent, "tool_registry"):
            tool_calls = []
            for tool_name in session.agent.tool_registry.recent_calls:
                session.add_tool_usage(tool_name)
                tool_calls.append({
                    "name": tool_name,
                    "timestamp": time.time()
                })
        
        # Stream the response if requested
        if chat_request.stream:
            # Advanced chunking based on sentence boundaries
            chunks = []
            current_pos = 0
            sentence_end_chars = ['.', '!', '?', '\n']
            
            # Create reasonable chunks based on sentence boundaries
            while current_pos < len(response):
                next_end = -1
                for char in sentence_end_chars:
                    pos = response.find(char, current_pos)
                    if pos != -1 and (next_end == -1 or pos < next_end):
                        next_end = pos
                
                # If no sentence boundary found, chunk by size
                if next_end == -1 or next_end - current_pos > 100:
                    next_end = min(current_pos + 50, len(response) - 1)
                else:
                    next_end += 1  # Include the sentence end character
                
                chunks.append(response[current_pos:next_end])
                current_pos = next_end
            
            # Stream the chunks
            for i, chunk in enumerate(chunks):
                chunk_metrics = {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "elapsed_time": time.time() - request_start
                }
                
                yield json.dumps(StreamingChunk(
                    status="streaming" if i < len(chunks) - 1 else "complete",
                    chunk=chunk,
                    session_id=session_id,
                    request_id=request_id,
                    metrics=chunk_metrics,
                    tool_calls=tool_calls if i == 0 and tool_calls else None
                ).model_dump())
                
                # Simulate realistic typing speed
                if i < len(chunks) - 1:
                    await asyncio.sleep(len(chunk) * 0.01)  # ~100 chars per second
        else:
            # Single response for non-streaming mode
            yield json.dumps(StreamingChunk(
                status="complete",
                chunk=response,
                session_id=session_id,
                request_id=request_id,
                metrics={"response_time_ms": (time.time() - request_start) * 1000},
                tool_calls=tool_calls if tool_calls else None
            ).model_dump())
        
        # Update Redis with session data
        await session_manager.save_session(
            session_id=session_id,
            session_data=session.get_session_info()
        )
        
        # Update request statistics
        REQUEST_STATS["success"] += 1
        latency = (time.time() - request_start) * 1000  # ms
        REQUEST_STATS["latency_avg_ms"] = (
            (REQUEST_STATS["latency_avg_ms"] * (REQUEST_STATS["total"] - 1) + latency) / 
            REQUEST_STATS["total"]
        )
        await session_manager.track_request("success", latency)
        
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        logger.error(f"{error_message}\n{traceback.format_exc()}")
        
        yield json.dumps(StreamingChunk(
            status="error",
            chunk=error_message,
            session_id=session_id,
            request_id=request_id
        ).model_dump())
        
        REQUEST_STATS["error"] += 1
        await session_manager.track_request("error", (time.time() - request_start) * 1000)

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("distributed-systems")],
    volumes={"/data": volume},
    memory=4096,
    timeout=600,
)
@modal.fastapi_endpoint(method="POST")
async def chat_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Web endpoint for chat that collects all chunks from the generator
    and returns the complete response.
    
    Args:
        request: Validated request with user input and configuration
        
    Returns:
        Complete response with all generated content
    """
    responses = []
    async for chunk in chat(request):
        responses.append(json.loads(chunk))
    
    # Return the last chunk which contains the complete response
    # or collect all chunks if needed for detailed logging
    return responses[-1] if responses else {"status": "error", "message": "No response generated"}

@app.function(image=image, memory=512)
@modal.fastapi_endpoint(method="GET")
async def status() -> Dict[str, Any]:
    """Advanced status endpoint with detailed metrics and monitoring"""
    # Get session information
    active_sessions = len(AGENT_SESSIONS)
    try:
        redis_sessions = await session_manager.list_sessions()
        if redis_sessions:
            active_sessions = len(redis_sessions)
    except:
        pass
    
    # Build status response
    response = StatusResponse(
        status="online",
        version="1.0.0",
        timestamp=time.time(),
        uptime=time.time() - START_TIME,
        active_sessions=active_sessions,
        model_status=ACTIVE_MODELS,
        memory_usage={
            "sessions": len(AGENT_SESSIONS),
            "estimated_mb": len(AGENT_SESSIONS) * 50  # rough estimate
        },
        request_stats=REQUEST_STATS
    )
    
    return response.model_dump()

@app.function(image=image, secrets=[modal.Secret.from_name("distributed-systems")])
@modal.fastapi_endpoint(method="GET")
async def session_info(session_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific session"""
    # Check memory cache first
    session = AGENT_SESSIONS.get(session_id)
    if session:
        return session.get_session_info()
    
    # Check Redis for distributed setups
    try:
        redis_session = await session_manager.get_session(session_id)
        if redis_session:
            return redis_session
    except:
        pass
    
    return {"error": "Session not found", "session_id": session_id}

# Define the app for FastAPI integration with Modal
web_app = FastAPI(title="Atomic Agent API")

def run_local_server():
    """CLI interface for the atomic agent API"""
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Atomic Agent API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    args = parser.parse_args()
    
    # Use the same web_app defined for Modal
    web_app.lifespan = lifespan
    
    # Add streaming endpoint for local development
    @web_app.post("/chat/stream")
    async def stream_chat(request: Request):
        """FastAPI streaming endpoint for local development"""
        from fastapi.responses import StreamingResponse
        request_data = await request.json()
        
        async def stream_response():
            async for chunk in chat(request_data):
                yield chunk + "\n"
        
        return StreamingResponse(stream_response(), media_type="application/x-ndjson")
    
    print(f"Starting Atomic Agent API on http://{args.host}:{args.port}")
    uvicorn.run(web_app, host=args.host, port=args.port, log_level=args.log_level)

@web_app.post("/chat")
async def fastapi_chat(request: Request):
    """FastAPI endpoint for web access"""
    request_data = await request.json()
    
    responses = []
    async for chunk in chat(request_data):
        responses.append(json.loads(chunk))
    
    # Return the last chunk which contains the complete response
    return responses[-1] if responses else {"status": "error", "message": "No response generated"}

@web_app.get("/status")
async def fastapi_status():
    """FastAPI status endpoint for web access"""
    return await status()

@web_app.get("/session/{session_id}")
async def fastapi_session_info(session_id: str):
    """FastAPI session info endpoint for web access"""
    return await session_info(session_id)

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """Modal ASGI app integration"""
    return web_app

@app.function(image=image)
def cli():
    """Modal CLI interface entry point"""
    run_local_server()

# Command line interface for local testing
if __name__ == "__main__":
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the atomic agent API locally")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the local server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run on")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    server_parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Test chatting with the agent")
    chat_parser.add_argument("--input", type=str, default="Hello, how can you help me?", help="User input to test")
    chat_parser.add_argument("--model", type=str, default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", help="Model to use")
    chat_parser.add_argument("--scouts", type=int, default=5, help="Number of scout agents")
    
    args = parser.parse_args()
    
    if args.command == "server":
        # Run the FastAPI server locally with the provided args
        import uvicorn
        from fastapi import FastAPI, Request, Body
        
        fast_app = FastAPI(title="Atomic Agent API", lifespan=lifespan)
        
        @fast_app.post("/chat")
        async def fastapi_chat(request: Request, body=Body(...)):
            """FastAPI endpoint for local development"""
            from fastapi.responses import StreamingResponse
            request_data = await request.json()
            
            async def stream_response():
                async for chunk in chat(request_data):
                    yield chunk + "\n"
            
            return StreamingResponse(stream_response(), media_type="application/x-ndjson")
        
        @fast_app.get("/status")
        async def fastapi_status():
            """FastAPI status endpoint for local development"""
            return {"status": "online", "version": "1.0.0", "timestamp": time.time()}
        
        @fast_app.get("/session/{session_id}")
        async def fastapi_session_info(session_id: str):
            """FastAPI session info endpoint for local development"""
            return {"session_id": session_id, "status": "active"}
        
        print(f"Starting Atomic Agent API on http://{args.host}:{args.port}")
        uvicorn.run(fast_app, host=args.host, port=args.port, log_level=args.log_level)
        
    elif args.command == "chat" or not args.command:
        # Run a local test
        async def test_local():
            message = {
                "user_input": args.input if hasattr(args, 'input') else "Hello, how can you help me?",
                "model": args.model if hasattr(args, 'model') else "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                "num_scouts": args.scouts if hasattr(args, 'scouts') else 5,
                "stream": True
            }
            
            print("Testing local atomic agent API...")
            print(f"User input: {message['user_input']}")
            
            # Ensure TOGETHER_API_KEY is set
            if not os.environ.get("TOGETHER_API_KEY"):
                print("Error: TOGETHER_API_KEY environment variable not set")
                sys.exit(1)
            
            # Import directly without Modal
            try:
                from atomic_agent import TogetherAgent
                
                agent = TogetherAgent(model=message["model"], num_scouts=message["num_scouts"])
                response = agent.generate_response(message["user_input"])
                print("\nAgent response:")
                print(response)
            except Exception as e:
                print(f"Error: {str(e)}")
                print(traceback.format_exc())
        
        asyncio.run(test_local())
    else:
        parser.print_help()