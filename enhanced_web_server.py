#!/usr/bin/env python3
"""
enhanced_web_server.py
----------------------
FastAPI server for the enhanced Llama4 agent system with advanced features:
- WebSocket streaming for real-time responses
- Optimized vector memory integration 
- Dynamic context management
- Performance monitoring and diagnostics
- SSE (Server-Sent Events) for progress updates
- Multi-agent orchestration API
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncGenerator
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

# FastAPI and web dependencies
import fastapi
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import enhanced agent components
try:
    from enhanced_atomic_agent import EnhancedTogetherAgent, EnhancedAgentOrchestrator
    ENHANCED_AGENT_AVAILABLE = True
except ImportError:
    ENHANCED_AGENT_AVAILABLE = False
    print("Warning: Enhanced agent components not found. Using base components instead.")

try:
    from atomic_agent import TogetherAgent, AgentOrchestrator
    BASE_AGENT_AVAILABLE = True
except ImportError:
    BASE_AGENT_AVAILABLE = False
    print("Error: Base agent components not found. Cannot continue.")
    sys.exit(1)

# Import optimized components if available
try:
    from optimized_vector_memory import OptimizedVectorMemory
    OPTIMIZED_MEMORY_AVAILABLE = True
except ImportError:
    OPTIMIZED_MEMORY_AVAILABLE = False

try:
    from dynamic_context_manager import DynamicContextManager
    DYNAMIC_CONTEXT_AVAILABLE = True
except ImportError:
    DYNAMIC_CONTEXT_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_web_server.log')
    ]
)
logger = logging.getLogger("enhanced_web_server")

# =============================================
# Data Models for API Request/Response
# =============================================
class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    model: Optional[str] = Field(None, description="Model to use for this request")
    stream: bool = Field(False, description="Whether to stream the response")
    context_items: Optional[List[Dict[str, Any]]] = Field(None, description="Additional context items")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    temperature: Optional[float] = Field(None, description="Temperature for generation")
    context_complexity: Optional[float] = Field(0.5, description="Conversation complexity (0.0-1.0)")
    task_complexity: Optional[float] = Field(0.5, description="Task complexity (0.0-1.0)")
    agent_id: Optional[str] = Field(None, description="Specific agent ID to use")

class ChatResponse(BaseModel):
    """Chat response model"""
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="Assistant response")
    agent_id: str = Field(..., description="Agent ID that generated the response")
    request_id: str = Field(..., description="Unique request ID")
    created_at: float = Field(..., description="Timestamp when response was created")
    model: str = Field(..., description="Model used for generation")
    tools_used: List[str] = Field(default_factory=list, description="Tools used in the response")
    performance: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    context_tokens: int = Field(0, description="Number of tokens in the context window")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    request_id: str = Field(..., description="Request ID")
    created_at: float = Field(..., description="Timestamp")

class StatusResponse(BaseModel):
    """Status response model"""
    status: str = Field(..., description="Service status")
    uptime: float = Field(..., description="Uptime in seconds")
    active_sessions: int = Field(..., description="Number of active sessions")
    available_models: List[str] = Field(..., description="Available models")
    memory_usage: Dict[str, Any] = Field(..., description="Memory usage stats")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    agent_count: int = Field(..., description="Number of active agents")

class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str = Field(..., description="Session ID")
    created_at: float = Field(..., description="Session creation timestamp")
    last_active: float = Field(..., description="Last activity timestamp")
    messages: int = Field(..., description="Number of messages in the session")
    model: str = Field(..., description="Model used in the session")
    agent_id: str = Field(..., description="Agent ID used in the session")
    memory_items: int = Field(0, description="Number of memory items")
    context_items: int = Field(0, description="Number of context items")

class SessionRequest(BaseModel):
    """Session creation request model"""
    model: Optional[str] = Field(None, description="Model to use for this session")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    initial_context: Optional[List[Dict[str, Any]]] = Field(None, description="Initial context items")
    use_enhanced_agent: bool = Field(True, description="Whether to use enhanced agent capabilities")

class OrchestrationRequest(BaseModel):
    """Multi-agent orchestration request model"""
    query: str = Field(..., description="Query to process")
    num_agents: Optional[int] = Field(3, description="Number of agents to use")
    agent_specializations: Optional[List[str]] = Field(None, description="Agent specializations")
    stream_results: bool = Field(False, description="Whether to stream results")
    timeout: Optional[int] = Field(None, description="Timeout in seconds")

# =============================================
# Session Management
# =============================================
class SessionManager:
    """Manages agent sessions and state"""
    
    def __init__(self):
        """Initialize session manager"""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.agents: Dict[str, Union[EnhancedTogetherAgent, TogetherAgent]] = {}
        self.start_time = time.time()
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "streaming_requests": 0,
            "avg_response_time_ms": 0,
            "total_response_time_ms": 0
        }
        
        # Available models
        self.available_models = [
            "meta-llama/Llama-4-Turbo-17B-Instruct-FP8",
            "meta-llama/Llama-4-8B-Instruct",
            "meta-llama/Llama-3-8B-Instruct",
            "meta-llama/Llama-3-70B-Instruct"
        ]
        
        # Default model
        self.default_model = "meta-llama/Llama-4-Turbo-17B-Instruct-FP8"
        
        # Orchestrator for multi-agent tasks
        if ENHANCED_AGENT_AVAILABLE:
            self.orchestrator = EnhancedAgentOrchestrator(
                num_scouts=3,
                model=self.default_model,
                use_enhanced_agents=True,
                shared_context=True
            )
        else:
            self.orchestrator = AgentOrchestrator(
                num_scouts=3,
                model=self.default_model
            )
    
    def create_session(self, request: SessionRequest) -> str:
        """Create a new session or return existing session ID"""
        session_id = str(uuid.uuid4())
        model = request.model or self.default_model
        
        # Create appropriate agent type
        if ENHANCED_AGENT_AVAILABLE and request.use_enhanced_agent:
            agent = EnhancedTogetherAgent(
                model=model,
                use_optimized_memory=OPTIMIZED_MEMORY_AVAILABLE,
                use_dynamic_context=DYNAMIC_CONTEXT_AVAILABLE
            )
            agent_type = "enhanced"
        else:
            agent = TogetherAgent(model=model)
            agent_type = "base"
        
        # Add agent to registry
        agent_id = f"agent-{str(uuid.uuid4())[:8]}"
        self.agents[agent_id] = agent
        
        # Create session
        self.sessions[session_id] = {
            "created_at": time.time(),
            "last_active": time.time(),
            "model": model,
            "agent_id": agent_id,
            "agent_type": agent_type,
            "messages": [],
            "system_prompt": request.system_prompt,
            "metrics": {
                "requests": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "avg_response_time_ms": 0
            }
        }
        
        # Add initial context if provided and using enhanced agent
        if request.initial_context and agent_type == "enhanced" and hasattr(agent, "context_manager"):
            for item in request.initial_context:
                if "content" in item and "type" in item:
                    if item["type"] == "message":
                        agent.context_manager.add_message(
                            content=item["content"],
                            from_role=item.get("from_role", "system"),
                            to_role=item.get("to_role", "user"),
                            message_type=item.get("message_type", "system_message"),
                            metadata=item.get("metadata", {})
                        )
                    elif item["type"] == "fact":
                        agent.context_manager.add_fact(
                            content=item["content"],
                            domain=item.get("domain", "general"),
                            confidence=item.get("confidence", 1.0),
                            tags=item.get("tags", []),
                            metadata=item.get("metadata", {})
                        )
                    else:
                        agent.context_manager.add_custom_item(
                            content=item["content"],
                            source=item.get("source", "api"),
                            tags=item.get("tags", []),
                            metadata=item.get("metadata", {})
                        )
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID"""
        return self.sessions.get(session_id)
    
    def get_agent(self, session_id: str) -> Optional[Union[EnhancedTogetherAgent, TogetherAgent]]:
        """Get the agent for a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        agent_id = session.get("agent_id")
        if not agent_id or agent_id not in self.agents:
            return None
        
        return self.agents[agent_id]
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Get detailed session information"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        agent = self.get_agent(session_id)
        memory_items = 0
        context_items = 0
        
        # Get memory and context counts if available
        if hasattr(agent, "memory") and hasattr(agent.memory, "get_stats"):
            memory_stats = agent.memory.get_stats()
            memory_items = memory_stats.get("memory_count", 0)
        
        if hasattr(agent, "context_manager") and hasattr(agent.context_manager, "get_metrics"):
            context_metrics = agent.context_manager.get_metrics()
            context_items = context_metrics.get("item_count", 0)
        
        return SessionInfo(
            session_id=session_id,
            created_at=session["created_at"],
            last_active=session["last_active"],
            messages=len(session["messages"]),
            model=session["model"],
            agent_id=session["agent_id"],
            memory_items=memory_items,
            context_items=context_items
        )
    
    def update_session_activity(self, session_id: str):
        """Update session last activity timestamp"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_active"] = time.time()
    
    def add_message_to_session(self, session_id: str, role: str, content: str):
        """Add a message to the session history"""
        if session_id in self.sessions:
            self.sessions[session_id]["messages"].append({
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
    
    def get_status(self) -> StatusResponse:
        """Get overall service status"""
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info()
        
        # Get available agents by type
        enhanced_agents = 0
        base_agents = 0
        for session in self.sessions.values():
            if session["agent_type"] == "enhanced":
                enhanced_agents += 1
            else:
                base_agents += 1
        
        return StatusResponse(
            status="online",
            uptime=time.time() - self.start_time,
            active_sessions=len(self.sessions),
            available_models=self.available_models,
            memory_usage={
                "rss_mb": memory_usage.rss / 1024 / 1024,
                "vms_mb": memory_usage.vms / 1024 / 1024,
                "sessions_mb": len(self.sessions) * 5  # Rough estimate
            },
            performance=self.metrics,
            agent_count={
                "enhanced": enhanced_agents,
                "base": base_agents,
                "total": len(self.agents)
            }
        )
    
    def update_metrics(self, success: bool, response_time_ms: float, streaming: bool = False):
        """Update performance metrics"""
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        if streaming:
            self.metrics["streaming_requests"] += 1
        
        self.metrics["total_response_time_ms"] += response_time_ms
        self.metrics["avg_response_time_ms"] = (
            self.metrics["total_response_time_ms"] / self.metrics["total_requests"]
        )
    
    def cleanup_stale_sessions(self, max_age_hours: int = 24):
        """Clean up sessions that haven't been active in the specified time"""
        now = time.time()
        stale_sessions = []
        
        for session_id, session in self.sessions.items():
            age_hours = (now - session["last_active"]) / 3600
            if age_hours > max_age_hours:
                stale_sessions.append(session_id)
                
                # Clean up associated agent
                agent_id = session.get("agent_id")
                if agent_id in self.agents:
                    # Save state if enhanced agent
                    agent = self.agents[agent_id]
                    if hasattr(agent, "save_state"):
                        agent.save_state()
                    
                    # Remove agent
                    del self.agents[agent_id]
        
        # Remove stale sessions
        for session_id in stale_sessions:
            del self.sessions[session_id]
        
        return len(stale_sessions)

# =============================================
# API Setup and Lifecycle
# =============================================
# Initialize session manager
session_manager = SessionManager()

# Application lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    # Check for API keys
    if not os.environ.get("TOGETHER_API_KEY"):
        logger.warning("TOGETHER_API_KEY not found in environment variables. API functionality will be limited.")
    
    # Startup message
    logger.info(f"Starting Enhanced Llama4 Web Server with {len(session_manager.available_models)} models available")
    logger.info(f"Enhanced Agent: {'Available' if ENHANCED_AGENT_AVAILABLE else 'Not Available'}")
    logger.info(f"Optimized Memory: {'Available' if OPTIMIZED_MEMORY_AVAILABLE else 'Not Available'}")
    logger.info(f"Dynamic Context: {'Available' if DYNAMIC_CONTEXT_AVAILABLE else 'Not Available'}")
    
    # Start background tasks
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    # Yield to keep the application running
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Enhanced Llama4 Web Server")
    cleanup_task.cancel()
    
    # Save all agent states
    for agent_id, agent in session_manager.agents.items():
        if hasattr(agent, "save_state"):
            agent.save_state()
    
    logger.info("Enhanced Llama4 Web Server shutdown complete")

# Background session cleanup task
async def periodic_cleanup():
    """Periodically clean up stale sessions"""
    while True:
        try:
            # Run cleanup every hour
            await asyncio.sleep(3600)
            cleaned = session_manager.cleanup_stale_sessions(max_age_hours=24)
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} stale sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

# Create FastAPI application
app = FastAPI(
    title="Enhanced Llama4 API",
    description="API for the enhanced Llama4 agent system with improved performance and context awareness",
    version="1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
    # Templates
    templates = Jinja2Templates(directory="templates")
except:
    logger.warning("Static files or templates directory not found. Web UI may not work properly.")

# =============================================
# API Routes
# =============================================
@app.get("/")
async def root(request: Request):
    """Render the main page"""
    try:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "title": "Enhanced Llama4 Agent"}
        )
    except:
        return HTMLResponse("""
        <html>
            <head><title>Enhanced Llama4 Agent</title></head>
            <body>
                <h1>Enhanced Llama4 Agent API</h1>
                <p>API is running. Templates not found for web UI.</p>
                <p><a href="/docs">View API documentation</a></p>
            </body>
        </html>
        """)

@app.post("/api/sessions")
async def create_session(request: SessionRequest) -> Dict[str, str]:
    """Create a new chat session"""
    session_id = session_manager.create_session(request)
    return {"session_id": session_id}

@app.get("/api/sessions/{session_id}", response_model=None)
async def get_session(session_id: str) -> Union[SessionInfo, JSONResponse]:
    """Get information about a session"""
    session_info = session_manager.get_session_info(session_id)
    if not session_info:
        return JSONResponse(
            status_code=404,
            content={"error": f"Session {session_id} not found"}
        )
    return session_info

@app.post("/api/chat", response_model=None)
async def chat(
    request: ChatRequest, background_tasks: BackgroundTasks
) -> Union[ChatResponse, JSONResponse]:
    """Process a chat request (non-streaming)"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    if not session_manager.get_session(session_id):
        # Create new session
        session_id = session_manager.create_session(SessionRequest(
            model=request.model,
            system_prompt=request.system_prompt,
            use_enhanced_agent=True
        ))
    
    # Get agent for session
    agent = session_manager.get_agent(session_id)
    if not agent:
        session_manager.update_metrics(success=False, response_time_ms=(time.time() - start_time) * 1000)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=f"Failed to get agent for session {session_id}",
                request_id=request_id,
                created_at=time.time()
            ).model_dump()
        )
    
    # Update session activity
    session_manager.update_session_activity(session_id)
    
    # Add message to session
    session_manager.add_message_to_session(session_id, "user", request.message)
    
    try:
        # Process with the enhanced agent
        session = session_manager.get_session(session_id)
        
        # Configure enhanced agent if available
        if hasattr(agent, "update_config"):
            agent.update_config({
                "conversation_complexity": request.context_complexity,
                "task_complexity": request.task_complexity
            })
        
        # Get agent response
        response = agent.generate_response(
            user_input=request.message,
            system_prompt=request.system_prompt or session.get("system_prompt"),
            temperature=request.temperature
        )
        
        # Add response to session
        session_manager.add_message_to_session(session_id, "assistant", response)
        
        # Get performance metrics if available
        performance_metrics = {}
        tools_used = []
        context_tokens = 0
        
        if hasattr(agent, "get_performance_metrics"):
            metrics = agent.get_performance_metrics()
            performance_metrics = {
                "response_time_ms": (time.time() - start_time) * 1000,
                "token_count": metrics.get("tokens_output", 0),
                "input_tokens": metrics.get("tokens_input", 0)
            }
            
            if "function_performance" in metrics:
                tools_used = list(metrics["function_performance"].keys())
            
            if "context" in metrics:
                context_tokens = metrics.get("context", {}).get("used_tokens", 0)
        
        # Create response
        chat_response = ChatResponse(
            session_id=session_id,
            message=response,
            agent_id=session["agent_id"],
            request_id=request_id,
            created_at=time.time(),
            model=session["model"],
            tools_used=tools_used,
            performance=performance_metrics,
            context_tokens=context_tokens
        )
        
        # Update metrics
        response_time_ms = (time.time() - start_time) * 1000
        session_manager.update_metrics(success=True, response_time_ms=response_time_ms)
        
        # Background task to save agent state periodically
        if hasattr(agent, "save_state"):
            background_tasks.add_task(agent.save_state)
        
        return chat_response
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        session_manager.update_metrics(success=False, response_time_ms=(time.time() - start_time) * 1000)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=f"Error generating response: {str(e)}",
                request_id=request_id,
                created_at=time.time()
            ).model_dump()
        )

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Process a chat request with streaming response"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    async def response_generator():
        """Generate streaming response chunks"""
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        if not session_manager.get_session(session_id):
            # Create new session
            session_id = session_manager.create_session(SessionRequest(
                model=request.model,
                system_prompt=request.system_prompt,
                use_enhanced_agent=True
            ))
            
            # First chunk is session creation confirmation
            yield json.dumps({
                "type": "session_created",
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": time.time()
            }) + "\n"
        
        # Get agent for session
        agent = session_manager.get_agent(session_id)
        if not agent:
            yield json.dumps({
                "type": "error",
                "error": f"Failed to get agent for session {session_id}",
                "request_id": request_id,
                "timestamp": time.time()
            }) + "\n"
            return
        
        # Update session activity
        session_manager.update_session_activity(session_id)
        
        # Add message to session
        session_manager.add_message_to_session(session_id, "user", request.message)
        
        try:
            # Process with the agent
            session = session_manager.get_session(session_id)
            
            # Configure enhanced agent if available
            if hasattr(agent, "update_config"):
                agent.update_config({
                    "conversation_complexity": request.context_complexity,
                    "task_complexity": request.task_complexity
                })
            
            # Start processing notification
            yield json.dumps({
                "type": "processing",
                "session_id": session_id,
                "request_id": request_id,
                "timestamp": time.time()
            }) + "\n"
            
            # Generate response
            if hasattr(agent, "generate_response_stream"):
                # Use native streaming if available
                response_text = ""
                async for chunk in agent.generate_response_stream(
                    user_input=request.message,
                    system_prompt=request.system_prompt or session.get("system_prompt"),
                    temperature=request.temperature
                ):
                    response_text += chunk
                    yield json.dumps({
                        "type": "chunk",
                        "session_id": session_id,
                        "request_id": request_id,
                        "chunk": chunk,
                        "timestamp": time.time()
                    }) + "\n"
                    
                    # Add a small delay to simulate natural typing
                    await asyncio.sleep(0.01)
            else:
                # Simulate streaming with chunks
                response = agent.generate_response(
                    user_input=request.message,
                    system_prompt=request.system_prompt or session.get("system_prompt"),
                    temperature=request.temperature
                )
                
                # Split into reasonably-sized chunks
                response_text = response
                words = response.split(" ")
                chunks = []
                current_chunk = []
                
                for word in words:
                    current_chunk.append(word)
                    if len(" ".join(current_chunk)) > 20:  # ~20 chars per chunk
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Send chunks with short delays
                for chunk in chunks:
                    yield json.dumps({
                        "type": "chunk",
                        "session_id": session_id,
                        "request_id": request_id,
                        "chunk": chunk,
                        "timestamp": time.time()
                    }) + "\n"
                    
                    # Add a small delay to simulate natural typing
                    await asyncio.sleep(len(chunk) * 0.01)  # ~100 chars per second
            
            # Add response to session
            session_manager.add_message_to_session(session_id, "assistant", response_text)
            
            # Get performance metrics if available
            performance_metrics = {}
            tools_used = []
            
            if hasattr(agent, "get_performance_metrics"):
                metrics = agent.get_performance_metrics()
                performance_metrics = {
                    "response_time_ms": (time.time() - start_time) * 1000,
                    "token_count": metrics.get("tokens_output", 0),
                    "input_tokens": metrics.get("tokens_input", 0)
                }
                
                if "function_performance" in metrics:
                    tools_used = list(metrics["function_performance"].keys())
            
            # Final completion message
            yield json.dumps({
                "type": "complete",
                "session_id": session_id,
                "request_id": request_id,
                "message": response_text,
                "agent_id": session["agent_id"],
                "model": session["model"],
                "tools_used": tools_used,
                "performance": performance_metrics,
                "timestamp": time.time()
            }) + "\n"
            
            # Update metrics
            response_time_ms = (time.time() - start_time) * 1000
            session_manager.update_metrics(success=True, response_time_ms=response_time_ms, streaming=True)
            
        except Exception as e:
            logger.error(f"Error processing streaming chat request: {e}")
            session_manager.update_metrics(success=False, response_time_ms=(time.time() - start_time) * 1000, streaming=True)
            yield json.dumps({
                "type": "error",
                "session_id": session_id,
                "request_id": request_id,
                "error": f"Error generating response: {str(e)}",
                "timestamp": time.time()
            }) + "\n"
    
    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream"
    )

@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    # Get or create session
    if not session_manager.get_session(session_id):
        # Create new session
        session_id = session_manager.create_session(SessionRequest(
            use_enhanced_agent=True
        ))
        
        # Send session creation confirmation
        await websocket.send_json({
            "type": "session_created",
            "session_id": session_id,
            "timestamp": time.time()
        })
    
    # Get agent for session
    agent = session_manager.get_agent(session_id)
    if not agent:
        await websocket.send_json({
            "type": "error",
            "error": f"Failed to get agent for session {session_id}",
            "timestamp": time.time()
        })
        await websocket.close()
        return
    
    # Update session activity
    session_manager.update_session_activity(session_id)
    
    try:
        # WebSocket chat loop
        while True:
            # Receive message
            data = await websocket.receive_json()
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            # Process message
            if "message" in data:
                message = data["message"]
                system_prompt = data.get("system_prompt")
                temperature = data.get("temperature")
                context_complexity = data.get("context_complexity", 0.5)
                task_complexity = data.get("task_complexity", 0.5)
                
                # Add message to session
                session_manager.add_message_to_session(session_id, "user", message)
                
                # Configure enhanced agent if available
                if hasattr(agent, "update_config"):
                    agent.update_config({
                        "conversation_complexity": context_complexity,
                        "task_complexity": task_complexity
                    })
                
                # Send processing notification
                await websocket.send_json({
                    "type": "processing",
                    "request_id": request_id,
                    "timestamp": time.time()
                })
                
                try:
                    # Get session info
                    session = session_manager.get_session(session_id)
                    
                    # Generate response with streaming if possible
                    if hasattr(agent, "generate_response_stream"):
                        # Use native streaming
                        response_text = ""
                        async for chunk in agent.generate_response_stream(
                            user_input=message,
                            system_prompt=system_prompt or session.get("system_prompt"),
                            temperature=temperature
                        ):
                            response_text += chunk
                            await websocket.send_json({
                                "type": "chunk",
                                "request_id": request_id,
                                "chunk": chunk,
                                "timestamp": time.time()
                            })
                    else:
                        # Generate full response
                        response = agent.generate_response(
                            user_input=message,
                            system_prompt=system_prompt or session.get("system_prompt"),
                            temperature=temperature
                        )
                        
                        # Simulate streaming
                        response_text = response
                        words = response.split(" ")
                        chunks = []
                        current_chunk = []
                        
                        for word in words:
                            current_chunk.append(word)
                            if len(" ".join(current_chunk)) > 20:
                                chunks.append(" ".join(current_chunk))
                                current_chunk = []
                        
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                        
                        # Send chunks
                        for chunk in chunks:
                            await websocket.send_json({
                                "type": "chunk",
                                "request_id": request_id,
                                "chunk": chunk,
                                "timestamp": time.time()
                            })
                            await asyncio.sleep(len(chunk) * 0.01)
                    
                    # Add response to session
                    session_manager.add_message_to_session(session_id, "assistant", response_text)
                    
                    # Get performance metrics if available
                    performance_metrics = {}
                    tools_used = []
                    
                    if hasattr(agent, "get_performance_metrics"):
                        metrics = agent.get_performance_metrics()
                        performance_metrics = {
                            "response_time_ms": (time.time() - start_time) * 1000,
                            "token_count": metrics.get("tokens_output", 0),
                            "input_tokens": metrics.get("tokens_input", 0)
                        }
                        
                        if "function_performance" in metrics:
                            tools_used = list(metrics["function_performance"].keys())
                    
                    # Send completion message
                    await websocket.send_json({
                        "type": "complete",
                        "request_id": request_id,
                        "message": response_text,
                        "agent_id": session["agent_id"],
                        "model": session["model"],
                        "tools_used": tools_used,
                        "performance": performance_metrics,
                        "timestamp": time.time()
                    })
                    
                    # Update metrics
                    response_time_ms = (time.time() - start_time) * 1000
                    session_manager.update_metrics(success=True, response_time_ms=response_time_ms, streaming=True)
                    
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    session_manager.update_metrics(success=False, response_time_ms=(time.time() - start_time) * 1000, streaming=True)
                    await websocket.send_json({
                        "type": "error",
                        "request_id": request_id,
                        "error": f"Error generating response: {str(e)}",
                        "timestamp": time.time()
                    })
            
            elif "ping" in data:
                # Simple ping-pong for connection keepalive
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": time.time()
                })
            
            else:
                # Unknown message type
                await websocket.send_json({
                    "type": "error",
                    "error": "Unknown message format",
                    "timestamp": time.time()
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.post("/api/orchestrate", response_model=None)
async def orchestrate(request: OrchestrationRequest) -> Union[Dict[str, Any], StreamingResponse]:
    """Orchestrate multiple agents to process a query"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Stream results if requested
    if request.stream_results:
        async def generate_orchestration_stream():
            try:
                # Create task for orchestrator
                task_id = session_manager.orchestrator.add_task(
                    task=request.query,
                    context={
                        "num_agents": request.num_agents,
                        "specializations": request.agent_specializations
                    }
                )
                
                # Send task created notification
                yield json.dumps({
                    "type": "task_created",
                    "task_id": task_id,
                    "request_id": request_id,
                    "timestamp": time.time()
                }) + "\n"
                
                # Wait for result with progress updates
                result = None
                timeout = request.timeout or 60  # Default to 60 seconds
                start = time.time()
                
                while time.time() - start < timeout:
                    # Check if task is complete
                    task_result = session_manager.orchestrator.get_task_result(task_id)
                    if task_result:
                        result = task_result
                        break
                    
                    # Send progress update
                    yield json.dumps({
                        "type": "progress",
                        "task_id": task_id,
                        "elapsed_seconds": time.time() - start,
                        "request_id": request_id,
                        "timestamp": time.time()
                    }) + "\n"
                    
                    # Wait before checking again
                    await asyncio.sleep(1)
                
                if result:
                    # Format and send the full result
                    yield json.dumps({
                        "type": "complete",
                        "task_id": task_id,
                        "request_id": request_id,
                        "result": result,
                        "elapsed_seconds": time.time() - start,
                        "timestamp": time.time()
                    }) + "\n"
                else:
                    # Timeout
                    yield json.dumps({
                        "type": "error",
                        "task_id": task_id,
                        "request_id": request_id,
                        "error": "Task timed out",
                        "elapsed_seconds": time.time() - start,
                        "timestamp": time.time()
                    }) + "\n"
                
                # Update metrics
                response_time_ms = (time.time() - start_time) * 1000
                session_manager.update_metrics(success=(result is not None), response_time_ms=response_time_ms, streaming=True)
                
            except Exception as e:
                logger.error(f"Error in orchestration: {e}")
                yield json.dumps({
                    "type": "error",
                    "request_id": request_id,
                    "error": f"Error in orchestration: {str(e)}",
                    "timestamp": time.time()
                }) + "\n"
                
                # Update metrics
                response_time_ms = (time.time() - start_time) * 1000
                session_manager.update_metrics(success=False, response_time_ms=response_time_ms, streaming=True)
        
        return StreamingResponse(
            generate_orchestration_stream(),
            media_type="text/event-stream"
        )
    
    else:
        # Non-streaming execution
        try:
            # Create tasks for all available scouts
            tasks = []
            
            if request.agent_specializations:
                # Use specified specializations
                for specialization in request.agent_specializations[:request.num_agents]:
                    tasks.append({
                        "task": request.query,
                        "specialization": specialization
                    })
            else:
                # Use default scouts
                for i in range(min(request.num_agents, len(session_manager.orchestrator.scouts))):
                    scout_id = list(session_manager.orchestrator.scouts.keys())[i]
                    tasks.append({
                        "task": request.query,
                        "agent_id": scout_id
                    })
            
            # Execute tasks in parallel
            results = session_manager.orchestrator.execute_parallel_tasks(tasks)
            
            # Return combined results
            response_time_ms = (time.time() - start_time) * 1000
            session_manager.update_metrics(success=True, response_time_ms=response_time_ms)
            
            return {
                "request_id": request_id,
                "query": request.query,
                "results": results,
                "timestamp": time.time(),
                "elapsed_ms": response_time_ms
            }
            
        except Exception as e:
            logger.error(f"Error in orchestration: {e}")
            response_time_ms = (time.time() - start_time) * 1000
            session_manager.update_metrics(success=False, response_time_ms=response_time_ms)
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"Error in orchestration: {str(e)}",
                    "request_id": request_id,
                    "timestamp": time.time()
                }
            )

@app.get("/api/status")
async def get_status() -> StatusResponse:
    """Get server status and metrics"""
    return session_manager.get_status()

# HTML template routes for the web interface
@app.get("/chat")
async def chat_ui(request: Request):
    """Render the chat interface"""
    try:
        return templates.TemplateResponse(
            "chat.html",
            {"request": request, "title": "Enhanced Llama4 Chat"}
        )
    except:
        return HTMLResponse("""
        <html>
            <head><title>Enhanced Llama4 Chat</title></head>
            <body>
                <h1>Enhanced Llama4 Chat</h1>
                <p>Templates not found for web UI.</p>
                <p><a href="/docs">View API documentation</a></p>
            </body>
        </html>
        """)

@app.get("/dashboard")
async def dashboard_ui(request: Request):
    """Render the admin dashboard"""
    try:
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "title": "Enhanced Llama4 Dashboard"}
        )
    except:
        return HTMLResponse("""
        <html>
            <head><title>Enhanced Llama4 Dashboard</title></head>
            <body>
                <h1>Enhanced Llama4 Dashboard</h1>
                <p>Templates not found for web UI.</p>
                <p><a href="/docs">View API documentation</a></p>
            </body>
        </html>
        """)

# =============================================
# Main Function
# =============================================
def main():
    """Run the server using uvicorn"""
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Llama4 Web Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--api-key", type=str, help="Together API key (if not in env)")
    parser.add_argument("--log-level", type=str, default="info", 
                       choices=["debug", "info", "warning", "error", "critical"],
                       help="Logging level")
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["TOGETHER_API_KEY"] = args.api_key
    
    # Configure logging
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level)
    
    # Start server
    print(f"Starting Enhanced Llama4 Web Server on http://{args.host}:{args.port}")
    print(f"Documentation available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "enhanced_web_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()