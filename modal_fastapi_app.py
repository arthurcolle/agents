# ---
# lambda-test: false
# ---

# # Deploy FastAPI app with Modal

# This example shows how you can deploy a [FastAPI](https://fastapi.tiangolo.com/) app with Modal.
# You can serve any app written in an ASGI-compatible web framework (like FastAPI) using this pattern or you can server WSGI-compatible frameworks like Flask with [`wsgi_app`](https://modal.com/docs/guide/webhooks#wsgi).

from typing import Optional, Dict, List, Any
import json
import logging
import asyncio
import time
from uuid import uuid4

import modal
from fastapi import FastAPI, Header, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Import from other modules with error handling
try:
    from openai_agent import (
        AutonomousAgent, TranslationOrchestrator, SYSTEM_GOAL,
        search_web, fact_check, read_url, JINA_AVAILABLE
    )
except ImportError as e:
    logging.error(f"Error importing from openai_agent: {e}")
    # Define fallbacks
    SYSTEM_GOAL = "Build a cool web app with dynamic generative UI"
    JINA_AVAILABLE = False
    
    async def search_web(query: str):
        return {"error": "Search functionality not available"}
    
    async def fact_check(statement: str):
        return {"error": "Fact check functionality not available"}
    
    async def read_url(url: str):
        return {"error": "URL reading functionality not available"}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("modal-fastapi-app")

# Define the Modal image and app
image = modal.Image.debian_slim().pip_install("fastapi[standard]", "pydantic", "uvicorn")
app = modal.App("modal-fastapi-app", image=image)

# Define Pydantic models for request/response
class Item(BaseModel):
    name: str

class ChatRequest(BaseModel):
    user_input: str = Field(..., description="User input text")
    model: str = Field(default="gpt-4", description="Model to use")
    stream: bool = Field(default=True, description="Whether to stream the response")

class StatusResponse(BaseModel):
    status: str = Field(..., description="API status")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: float = Field(default_factory=time.time, description="Current timestamp")

# Initialize FastAPI app
web_app = FastAPI(title="Modal FastAPI Demo")

# Basic endpoint handlers
@web_app.get("/")
async def handle_root(user_agent: Optional[str] = Header(None)):
    print(f"GET /     - received user_agent={user_agent}")
    return "Hello World"

@web_app.post("/foo")
async def handle_foo(item: Item, user_agent: Optional[str] = Header(None)):
    print(
        f"POST /foo - received user_agent={user_agent}, item.name={item.name}"
    )
    return item

@web_app.get("/status")
async def get_status():
    return StatusResponse(
        status="online",
        version="1.0.0",
        timestamp=time.time()
    ).model_dump()

@web_app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Process chat requests"""
    try:
        # Just a simple response for demo purposes
        response = f"Processed request with model {request.model}: {request.user_input}"
        return {"status": "success", "response": response}
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return {"status": "error", "message": str(e)}

# Define the Modal function that serves the FastAPI app
@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app

# Alternative pattern with direct FastAPI integration
@app.function()
@modal.fastapi_endpoint(method="POST")
def f(item: Item):
    return "Hello " + item.name

# Entry point: run locally or deploy to Modal
if __name__ == "__main__":
    # Choose to run locally or deploy based on CLI flag
    import argparse
    parser = argparse.ArgumentParser(description="Modal FastAPI App Runner")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy the app to Modal with the given name")
    parser.add_argument("--name", default="webapp",
                        help="Name of the Modal App to deploy or target for remote calls")
    args = parser.parse_args()
    if args.deploy:
        # Deploy the Modal App
        app.deploy(args.name)
    else:
        # Run the ASGI app locally via Modal
        fastapi_app.local()