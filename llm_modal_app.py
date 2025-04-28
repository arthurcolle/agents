import os
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union

import modal
from fastapi import FastAPI, Header, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, APIRouter, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm-modal-app")

# Define Modal image and dependencies
image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", 
    "pydantic", 
    "uvicorn", 
    "datasets", 
    "transformers", 
    "torch", 
    "accelerate", 
    "bitsandbytes", 
    "peft",
    "tqdm",
    "jinja2"
)

# Define Modal volumes for persistent storage
models_volume = modal.Volume.from_name("models-volume", create_if_missing=True)
datasets_volume = modal.Volume.from_name("datasets-volume", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("checkpoints-volume", create_if_missing=True)

# Define Modal app
app = modal.App("llm-modal-app", image=image)

# Pydantic models for API
class ModelInfo(BaseModel):
    id: str
    name: str
    size: str
    description: Optional[str] = None
    parameters: Optional[int] = None
    created_at: float = Field(default_factory=time.time)
    
class DatasetInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    records: int
    created_at: float = Field(default_factory=time.time)
    
class ConversationMessage(BaseModel):
    role: str
    content: str
    timestamp: float = Field(default_factory=time.time)
    
class Conversation(BaseModel):
    id: str
    title: str
    messages: List[ConversationMessage] = []
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    
class FinetuneRequest(BaseModel):
    model_id: str
    dataset_id: str
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 8
    max_length: int = 512
    
class InferenceRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

# Initialize FastAPI app
web_app = FastAPI(title="LLM Modal App")

# Add CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
web_app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory storage (will be replaced with proper persistence)
MODELS = {}
DATASETS = {}
CONVERSATIONS = {}
FINETUNE_JOBS = {}

# Basic routes
@web_app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main application UI"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "LLM Modal App"}
    )

@web_app.get("/api/status")
async def get_status():
    """Get API status"""
    return {
        "status": "online",
        "version": "1.0.0",
        "timestamp": time.time()
    }

# Model endpoints
@web_app.get("/api/models")
async def list_models():
    """List all available models"""
    return {"models": list(MODELS.values())}

@web_app.post("/api/models/upload")
async def upload_model(
    name: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(...),
):
    """Upload a new model"""
    model_id = str(uuid.uuid4())
    model_info = ModelInfo(
        id=model_id,
        name=name,
        size=str(file.size),
        description=description
    )
    MODELS[model_id] = model_info.model_dump()
    
    # Here we would save the model to the volume
    # For now just log it
    logger.info(f"Model {name} uploaded with ID {model_id}")
    
    return model_info

@web_app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get model details"""
    if model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    return MODELS[model_id]

# Dataset endpoints
@web_app.get("/api/datasets")
async def list_datasets():
    """List all available datasets"""
    return {"datasets": list(DATASETS.values())}

@web_app.post("/api/datasets/upload")
async def upload_dataset(
    name: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(...),
):
    """Upload a new dataset"""
    dataset_id = str(uuid.uuid4())
    # In a real implementation, we would count records
    dataset_info = DatasetInfo(
        id=dataset_id,
        name=name,
        description=description,
        records=1000  # Placeholder
    )
    DATASETS[dataset_id] = dataset_info.model_dump()
    
    # Here we would save the dataset to the volume
    logger.info(f"Dataset {name} uploaded with ID {dataset_id}")
    
    return dataset_info

@web_app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset details"""
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DATASETS[dataset_id]

# Conversation endpoints
@web_app.get("/api/conversations")
async def list_conversations():
    """List all conversations"""
    return {"conversations": list(CONVERSATIONS.values())}

@web_app.post("/api/conversations")
async def create_conversation(title: str = Form("New Conversation")):
    """Create a new conversation"""
    conversation_id = str(uuid.uuid4())
    conversation = Conversation(
        id=conversation_id,
        title=title
    )
    CONVERSATIONS[conversation_id] = conversation.model_dump()
    return conversation

@web_app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation details"""
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return CONVERSATIONS[conversation_id]

@web_app.post("/api/conversations/{conversation_id}/messages")
async def add_message(conversation_id: str, message: ConversationMessage):
    """Add a message to a conversation"""
    if conversation_id not in CONVERSATIONS:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    conversation = CONVERSATIONS[conversation_id]
    conversation["messages"].append(message.model_dump())
    conversation["updated_at"] = time.time()
    
    # If the message is from the user, generate a response
    if message.role == "user":
        # Here we would call the model for inference
        # For now, just add a simple response
        response = ConversationMessage(
            role="assistant",
            content=f"This is a response to: {message.content}"
        )
        conversation["messages"].append(response.model_dump())
    
    return {"status": "success"}

# Finetune endpoints
@web_app.post("/api/finetune")
async def start_finetune(request: FinetuneRequest):
    """Start a model finetuning job"""
    if request.model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if request.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id,
        "model_id": request.model_id,
        "dataset_id": request.dataset_id,
        "status": "pending",
        "created_at": time.time(),
        "config": request.model_dump()
    }
    FINETUNE_JOBS[job_id] = job
    
    # Here we would trigger the actual finetuning in Modal
    logger.info(f"Started finetuning job {job_id}")
    
    return {"job_id": job_id, "status": "pending"}

@web_app.get("/api/finetune/{job_id}")
async def get_finetune_job(job_id: str):
    """Get finetuning job status"""
    if job_id not in FINETUNE_JOBS:
        raise HTTPException(status_code=404, detail="Finetune job not found")
    
    return FINETUNE_JOBS[job_id]

# Inference endpoint
@web_app.post("/api/inference")
async def run_inference(request: InferenceRequest):
    """Run model inference"""
    if request.model_id not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # In a real implementation, we would load the model and run inference
    # For now, just return a simple response
    response = f"This is a generated response to: {request.prompt}"
    
    return {"response": response}

# Studio endpoints - for running code, transformations, etc.
@web_app.post("/api/studio/transform")
async def transform_dataset(dataset_id: str, transformation: str):
    """Apply a transformation to a dataset"""
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # In a real implementation, we would apply the transformation
    # For now, just log it
    logger.info(f"Applied transformation '{transformation}' to dataset {dataset_id}")
    
    return {"status": "success", "message": f"Transformation '{transformation}' applied"}

@web_app.post("/api/studio/code")
async def run_code(code: str, inputs: Dict[str, Any] = None):
    """Run arbitrary code in the studio"""
    # This would be implemented with proper sandboxing and security
    # For now, just log it
    logger.info(f"Running code in studio: {code[:100]}...")
    
    return {"status": "success", "result": "Code execution result would appear here"}

# Define Modal function to serve the app
@app.function(
    volumes={
        "/models": models_volume,
        "/datasets": datasets_volume,
        "/checkpoints": checkpoints_volume
    },
    cpu=1,
    memory=4096
)
@modal.asgi_app()
def serve():
    return web_app

# Define a separate Modal function for finetuning
@app.function(
    volumes={
        "/models": models_volume,
        "/datasets": datasets_volume,
        "/checkpoints": checkpoints_volume
    },
    gpu="A10G",
    memory=32768,
    timeout=3600
)
def finetune_model(model_id: str, dataset_id: str, config: Dict[str, Any]):
    """Finetune a model with the given configuration"""
    logger.info(f"Finetuning model {model_id} with dataset {dataset_id}")
    
    # Simulate finetuning
    time.sleep(10)
    
    return {"status": "success", "message": "Model finetuned successfully"}

# Entry point: run locally or deploy to Modal
if __name__ == "__main__":
    # Choose to run locally or deploy based on CLI flag
    import argparse
    parser = argparse.ArgumentParser(description="LLM Modal App Runner")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy the app to Modal with the given name")
    parser.add_argument("--name", default="llm-modal-app",
                        help="Name of the Modal App to deploy or target for remote calls")
    args = parser.parse_args()
    if args.deploy:
        # Deploy the Modal App
        app.deploy(args.name)
    else:
        # Run the ASGI app locally via Modal
        # This will start the FastAPI server locally
        serve.local()