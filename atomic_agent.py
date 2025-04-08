#!/usr/bin/env python3
"""
cli.py
------
A CLI tool to chat with an AI agent using the Together API with dynamic tools.
This version defines all functions and tools, and supports multiple function calls
within a single turn. The system prompt (for Llama‑4 models) now informs the model
that multiple function calls may be issued.
"""

import re
import os
import sys
import json
import argparse
import subprocess
import inspect
import importlib
import importlib.util
import math
import sqlite3
import uuid
import hashlib
import base64
import time
import queue
import threading
import tempfile
import urllib.parse
import asyncio
import traceback
import redis
import numpy as np
import random
from typing import Dict, List, Any, Callable, Optional, Union, Tuple, Set
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO, BytesIO
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional, Union
from pydantic import BaseModel

try:
    import aiohttp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp

try:
    import pytz
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytz"])
    import pytz
    
try:
    import pyperclip
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyperclip"])
    import pyperclip

try:
    from PIL import ImageGrab
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
    from PIL import ImageGrab

try:
    from together import Together
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "together"])
    from together import Together

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.panel import Panel
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.prompt import Prompt
    from rich.panel import Panel

# Initialize console for rich output
console = Console()

# Initialize Redis connection for pubsub
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    pubsub = redis_client.pubsub()
    pubsub.subscribe('agent_messages')
    console.print("[green]Redis PubSub initialized successfully[/green]")
    
    # Initialize Redis for vector storage
    vector_store = redis.Redis(host='localhost', port=6379, db=1)
    console.print("[green]Redis Vector Store initialized successfully[/green]")
except Exception as e:
    console.print(f"[yellow]Warning: Redis PubSub initialization failed: {e}[/yellow]")
    pubsub = None
    vector_store = None

# Try to import optional dependencies for advanced features
try:
    import torch
    import transformers
    import base64
    from io import BytesIO
    from PIL import Image
    import requests
    import pyperclip
    from PIL import ImageGrab
    ADVANCED_EMBEDDINGS_AVAILABLE = True
    # Initialize Jina embedding client
    try:
        JINA_API_KEY = os.environ.get("JINA_API_KEY")
        if JINA_API_KEY:
            console.print("[green]Jina API key found, will use jina-clip-v2 for embeddings[/green]")
            embedding_model = "jina-clip-v2"
        else:
            console.print("[yellow]Warning: JINA_API_KEY not found in environment variables. Using fallback embedding method.[/yellow]")
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            console.print("[green]Sentence Transformer model loaded as fallback[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not initialize embedding system: {e}[/yellow]")
        embedding_model = None
except ImportError:
    ADVANCED_EMBEDDINGS_AVAILABLE = False
    embedding_model = None
    console.print("[yellow]Advanced embedding features not available. Install with: pip install torch transformers requests pillow[/yellow]")

# =======================
# Vector Memory System
# =======================
class VectorMemory:
    """Advanced memory system using vector embeddings for semantic retrieval"""
    def __init__(self, embedding_model=None, vector_store=None, dimension=1024):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.dimension = dimension
        self.memory_items = []
        self.embeddings = []
        self.available = ADVANCED_EMBEDDINGS_AVAILABLE and embedding_model is not None
        self.jina_api_key = os.environ.get("JINA_API_KEY")
        self.image_embeddings = {}  # Store image embeddings by URL
        
    def _get_jina_embedding(self, content, is_image=False):
        """Get embedding from Jina API for text or image"""
        if not self.jina_api_key:
            return None
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.jina_api_key}"
        }
        
        if is_image:
            # Check if it's a URL or base64
            if content.startswith('http'):
                payload = {
                    "model": "jina-clip-v2",
                    "input": [{"image": content}]
                }
            else:
                # Assume it's base64
                payload = {
                    "model": "jina-clip-v2",
                    "input": [{"image": content}]
                }
        else:
            # Text embedding
            payload = {
                "model": "jina-clip-v2",
                "input": [{"text": content}]
            }
            
        try:
            response = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                if "data" in result and len(result["data"]) > 0:
                    return np.array(result["data"][0]["embedding"])
            
            console.print(f"[yellow]Warning: Failed to get Jina embedding. Status: {response.status_code}, Response: {response.text}[/yellow]")
            return None
        except Exception as e:
            console.print(f"[yellow]Warning: Error getting Jina embedding: {e}[/yellow]")
            return None
    
    def _get_embedding(self, content, is_image=False):
        """Get embedding using the appropriate method"""
        # Try Jina first if available
        if self.jina_api_key:
            embedding = self._get_jina_embedding(content, is_image)
            if embedding is not None:
                return embedding
                
        # Fallback to sentence-transformers for text
        if not is_image and isinstance(self.embedding_model, SentenceTransformer):
            return self.embedding_model.encode(content)
            
        # Return None if no embedding method worked
        return None
        
    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a memory item and generate its embedding"""
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        memory_item = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": timestamp,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "access_count": 0
        }
        
        self.memory_items.append(memory_item)
        
        # Check if this is an image URL
        is_image = False
        if metadata and metadata.get("type") == "image":
            is_image = True
        elif isinstance(content, str) and (content.startswith('http') and any(ext in content.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])):
            is_image = True
            memory_item["metadata"]["type"] = "image"
        
        # Generate and store embedding if available
        if self.available:
            embedding = self._get_embedding(content, is_image)
            
            if embedding is not None:
                self.embeddings.append(embedding)
                
                # If it's an image, store in image embeddings dictionary
                if is_image:
                    self.image_embeddings[content] = embedding
                
                # Store in Redis if available
                if self.vector_store:
                    try:
                        # Store as JSON with embedding
                        memory_with_embedding = memory_item.copy()
                        memory_with_embedding["embedding"] = embedding.tolist()
                        memory_with_embedding["is_image"] = is_image
                        self.vector_store.set(f"memory:{memory_id}", json.dumps(memory_with_embedding))
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to store memory in Redis: {e}[/yellow]")
        
        return memory_id
    
    def search_memory(self, query: str, limit: int = 5, include_images: bool = True) -> List[Dict[str, Any]]:
        """Search memory using vector similarity"""
        if not self.available or not self.memory_items:
            # Fallback to keyword search if embeddings not available
            return self._keyword_search(query, limit)
        
        # Check if query might be an image URL
        is_image_query = query.startswith('http') and any(ext in query.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])
        
        # Get query embedding
        query_embedding = self._get_embedding(query, is_image=is_image_query)
        
        if query_embedding is None:
            # Fallback to keyword search if embedding fails
            return self._keyword_search(query, limit)
            
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            if embedding is not None:  # Skip items without embeddings
                similarity = self._cosine_similarity(query_embedding, embedding)
                similarities.append((similarity, i))
        
        # Sort by similarity (highest first)
        similarities.sort(reverse=True)
        
        # Return top matches
        results = []
        for similarity, idx in similarities[:limit]:
            memory_item = self.memory_items[idx].copy()
            memory_item["relevance_score"] = float(similarity)
            memory_item["access_count"] += 1
            
            # Skip images if not requested
            if not include_images and memory_item.get("metadata", {}).get("type") == "image":
                continue
                
            results.append(memory_item)
            
        # If we don't have enough results after filtering, get more
        if len(results) < limit and not include_images:
            more_results = self.search_memory(query, limit=limit*2, include_images=include_images)
            results.extend([r for r in more_results if r["id"] not in [item["id"] for item in results]])
            results = results[:limit]
            
        return results
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _keyword_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fallback keyword-based search"""
        query_terms = query.lower().split()
        results = []
        
        for item in self.memory_items:
            content = item["content"].lower()
            # Calculate a simple relevance score based on term frequency
            score = sum(content.count(term) for term in query_terms)
            if score > 0:
                result = item.copy()
                result["relevance_score"] = score
                result["access_count"] += 1
                results.append(result)
                
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:limit]
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID"""
        for item in self.memory_items:
            if item["id"] == memory_id:
                item["access_count"] += 1
                return item
        return None
    
    def forget_memory(self, memory_id: str) -> bool:
        """Remove a memory item"""
        for i, item in enumerate(self.memory_items):
            if item["id"] == memory_id:
                self.memory_items.pop(i)
                if self.available and i < len(self.embeddings):
                    self.embeddings.pop(i)
                if self.vector_store:
                    try:
                        self.vector_store.delete(f"memory:{memory_id}")
                    except Exception:
                        pass
                return True
        return False
    
    def summarize_memories(self, category: str = None) -> Dict[str, Any]:
        """Generate a summary of stored memories"""
        if not self.memory_items:
            return {"count": 0, "categories": [], "oldest": None, "newest": None}
            
        categories = set()
        timestamps = []
        
        for item in self.memory_items:
            if "category" in item.get("metadata", {}):
                categories.add(item["metadata"]["category"])
            timestamps.append(item["timestamp"])
            
        filtered_items = self.memory_items
        if category:
            filtered_items = [item for item in self.memory_items 
                             if item.get("metadata", {}).get("category") == category]
            
        return {
            "count": len(self.memory_items),
            "filtered_count": len(filtered_items),
            "categories": list(categories),
            "oldest": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(min(timestamps))) if timestamps else None,
            "newest": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(max(timestamps))) if timestamps else None,
            "total_tokens": sum(len(item["content"].split()) for item in filtered_items),
            "category_filter": category
        }

# =======================
# MultiPrompt Class
# =======================
class MultiPrompt(BaseModel):
    original_prompt: str
    steps: List[str]
    chain_of_thought: List[str]
    reasoning: str
    next_actions: List[str]
# Data Classes & Helpers
# =======================
@dataclass
class CodeArtifact:
    artifact_id: str
    name: str
    code: str
    description: str
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_count: int = 0
    last_result: Optional[Dict[str, Any]] = None

@dataclass
class FunctionSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    source_code: Optional[str] = None

@dataclass
class PlanningSession:
    session_id: str
    task: str
    created_at: float
    steps: List[Dict[str, Any]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    active: bool = True
    completion_status: str = "in_progress"  # "in_progress", "completed", "failed"

@dataclass
class StructuredOutput:
    source: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class URLExtraction(StructuredOutput):
    urls: List[str] = field(default_factory=list)

@dataclass
class KnowledgeItem:
    content: str
    source_url: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

# ================================
# Scout Agent
# ================================
class ScoutAgent:
    """Scout agent that can autonomously perform specialized tasks with advanced capabilities."""
    def __init__(self, agent_id, specialization, model="meta-llama/Llama-4-Turbo-17B-Instruct-FP8"):
        self.agent_id = agent_id
        self.specialization = specialization
        self.model = model
        self.conversation_history = []
        self.task_queue = queue.Queue()
        self.results = {}
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.knowledge_base = []
        self.processed_urls = set()
        self.url_pattern = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')
        self.image_processing_limit = 5
        self.status = "idle"  # idle, working, completed, error
        self.chains_of_thought = []
        self.rollouts = []  # Store different solution paths/rollouts
        self.max_rollouts = 3  # Maximum number of rollouts to generate
        self.is_available = threading.Event()
        self.is_available.set()  # Initially available
        
        # Advanced capabilities
        self.memory = VectorMemory(embedding_model, vector_store)
        self.skills = set([specialization])  # Start with base specialization
        self.performance_metrics = {
            "tasks_completed": 0,
            "success_rate": 1.0,
            "avg_execution_time": 0,
            "total_execution_time": 0,
            "feedback_scores": []
        }
        self.learning_rate = 0.1  # For skill acquisition
        self.skill_proficiency = {specialization: 1.0}  # Base skill starts at 100%
        self.collaboration_history = {}  # Track collaborations with other agents
        self.reinforcement_learning = {
            "exploration_rate": 0.2,  # Probability of trying new approaches
            "learning_enabled": True,
            "reward_history": []
        }
        
        # Initialize Redis pubsub for agent communication
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.pubsub = self.redis_client.pubsub()
            self.pubsub.subscribe(f'agent_{self.agent_id}')
            self.pubsub_thread = threading.Thread(target=self._listen_for_messages, daemon=True)
            self.pubsub_thread.start()
            print(f"[green]Agent {self.agent_id} subscribed to Redis channel agent_{self.agent_id}[/green]")
        except Exception as e:
            print(f"[yellow]Warning: Redis PubSub initialization failed for agent {self.agent_id}: {e}[/yellow]")
            self.redis_client = None
            self.pubsub = None
            
        self.worker_thread.start()
        
        # Initialize system prompt based on specialization
        specialization_prompts = {
            "research": "You are a research-focused scout agent. Focus on finding, analyzing, and summarizing information from various sources. Be thorough and analytical.",
            "code": "You are a code-focused scout agent. Your primary role is to write, optimize, and debug code. Focus on technical excellence and clean, efficient solutions.",
            "planning": "You are a planning-focused scout agent. Your role is to break down complex tasks, create step-by-step approaches, and identify potential issues.",
            "creative": "You are a creative-focused scout agent. Your role is to generate innovative ideas, create original content, and think outside the box.",
            "critical": "You are a critical-focused scout agent. Your role is to analyze proposals, identify weak points, and suggest improvements."
        }
        
        self.system_message = {"role": "system", "content": specialization_prompts.get(
            specialization, "You are a specialized scout agent working under a central orchestrating agent.")}
        self.conversation_history.append(self.system_message)
        
    def _listen_for_messages(self):
        """Listen for messages from other agents via Redis pubsub"""
        if not self.pubsub:
            return
            
        while not self.stop_event.is_set():
            try:
                message = self.pubsub.get_message(timeout=1)
                if message and message['type'] == 'message':
                    data = json.loads(message['data'].decode('utf-8'))
                    print(f"[cyan]Agent {self.agent_id} received message: {data}[/cyan]")
                    
                    # Handle different message types
                    if data.get('type') == 'task_request':
                        # Another agent is requesting help with a task
                        self.add_task(self._process_task_request, data)
                    elif data.get('type') == 'knowledge_share':
                        # Another agent is sharing knowledge
                        self.knowledge_base.append({
                            'content': data.get('content'),
                            'source': f"agent_{data.get('sender_id')}",
                            'timestamp': time.time()
                        })
            except Exception as e:
                print(f"[yellow]Error in pubsub listener for agent {self.agent_id}: {e}[/yellow]")
                time.sleep(1)
                
    def send_message(self, recipient_id, message_type, content):
        """Send a message to another agent via Redis pubsub"""
        if not hasattr(self, 'redis_client') or not self.redis_client:
            print(f"[yellow]Warning: Redis client not available for agent {self.agent_id}[/yellow]")
            return False
            
        try:
            message = {
                'sender_id': self.agent_id,
                'sender_specialization': self.specialization,
                'type': message_type,
                'content': content,
                'timestamp': time.time()
            }
            self.redis_client.publish(f'agent_{recipient_id}', json.dumps(message))
            return True
        except Exception as e:
            print(f"[yellow]Error sending message from agent {self.agent_id} to {recipient_id}: {e}[/yellow]")
            return False
            
    def broadcast_message(self, message_type, content):
        """Broadcast a message to all agents via Redis pubsub"""
        if not hasattr(self, 'redis_client') or not self.redis_client:
            print(f"[yellow]Warning: Redis client not available for agent {self.agent_id}[/yellow]")
            return False
            
        try:
            message = {
                'sender_id': self.agent_id,
                'sender_specialization': self.specialization,
                'type': message_type,
                'content': content,
                'timestamp': time.time()
            }
            self.redis_client.publish('agent_broadcast', json.dumps(message))
            return True
        except Exception as e:
            print(f"[yellow]Error broadcasting message from agent {self.agent_id}: {e}[/yellow]")
            return False
            
    def _process_task_request(self, data):
        """Process a task request from another agent"""
        task = data.get('task')
        sender_id = data.get('sender_id')
        
        # Check if this is a task we can handle based on specialization
        if self.specialization == data.get('requested_specialization'):
            result = self.perform_task(task, data.get('context'))
            # Send the result back to the requesting agent
            self.send_message(sender_id, 'task_result', result)
            return result
        else:
            # We're not the right specialization for this task
            self.send_message(sender_id, 'task_rejected', {
                'reason': f"Agent {self.agent_id} is {self.specialization}, not {data.get('requested_specialization')}"
            })
            return None
            
    def _worker(self):
        while not self.stop_event.is_set():
            try:
                task_id, task_func, args, kwargs = self.task_queue.get(timeout=1)
                try:
                    self.status = "working"
                    self.is_available.clear()  # Mark as unavailable while working
                    if asyncio.iscoroutinefunction(task_func):
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(task_func(*args, **kwargs))
                    else:
                        result = task_func(*args, **kwargs)
                    self.results[task_id] = {"status": "completed", "result": result}
                    self.status = "completed"
                except Exception as e:
                    self.results[task_id] = {
                        "status": "failed",
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                    self.status = "error"
                finally:
                    self.task_queue.task_done()
                    self.is_available.set()  # Mark as available again
            except queue.Empty:
                continue
                
    def add_task(self, task_func, *args, **kwargs):
        task_id = str(uuid.uuid4())
        self.results[task_id] = {"status": "pending"}
        self.task_queue.put((task_id, task_func, args, kwargs))
        return task_id
        
    def get_result(self, task_id):
        return self.results.get(task_id, {"status": "not_found"})
        
    def add_chain_of_thought(self, thought):
        timestamp = time.time()
        self.chains_of_thought.append({
            "timestamp": timestamp,
            "thought": thought,
            "agent_id": self.agent_id,
            "specialization": self.specialization
        })
        
    def perform_task(self, task, context=None):
        """Main method to process a task with advanced reasoning, learning, and multiple rollouts"""
        start_time = time.time()
        
        # Add the task to the conversation history
        if context:
            task_with_context = f"Task: {task}\n\nContext: {context}"
        else:
            task_with_context = f"Task: {task}"
            
        self.conversation_history.append({"role": "user", "content": task_with_context})
        
        # Store task in memory for future reference
        memory_metadata = {
            "type": "task",
            "category": self.specialization,
            "context": context
        }
        memory_id = self.memory.add_memory(task, memory_metadata)
        
        # Check if we've done similar tasks before
        similar_experiences = self.memory.search_memory(task, limit=3)
        
        # Determine if we should explore new approaches based on RL exploration rate
        should_explore = (random.random() < self.reinforcement_learning["exploration_rate"]) and self.reinforcement_learning["learning_enabled"]
        
        from together import Together
        together = Together()
        
        # Generate multiple rollouts/solution paths
        rollout_results = []
        
        # Enhance prompt with similar past experiences if available
        experience_context = ""
        if similar_experiences and not should_explore:
            experience_context = "\n\nRelevant past experiences:\n"
            for i, exp in enumerate(similar_experiences):
                experience_context += f"{i+1}. {exp['content']}\n"
                if "solution" in exp.get("metadata", {}):
                    experience_context += f"   Previous solution: {exp['metadata']['solution']}\n"
        
        # First generate a chain of thought with enhanced context
        cot_prompt = {
            "role": "user", 
            "content": f"For this task: '{task}', please use chain-of-thought reasoning. First, break down the task into steps. Then think through each step carefully before providing your final answer or solution.{experience_context}"
        }
        
        # Generate the chain of thought
        cot_response = together.chat.completions.create(
            model=self.model,
            messages=self.conversation_history + [cot_prompt],
                            
        )
        
        # Extract and store the chain of thought
        cot_content = cot_response.choices[0].message.content
        self.add_chain_of_thought(cot_content)
        
        # Now generate multiple solution rollouts with different approaches
        self.conversation_history.append({"role": "assistant", "content": cot_content})
        
        for i in range(self.max_rollouts):
            rollout_prompt = {
                "role": "user",
                "content": f"Based on your thinking above, provide solution approach #{i+1} to this task. Try to use a different strategy or perspective than your previous approaches."
            }
            
            if i > 0:
                # Add previous rollouts to the context to ensure diversity
                rollout_prompt["content"] += f"\n\nYour previous solution approaches were:\n" + "\n".join([
                    f"Approach #{j+1}: {r['solution'][:100]}..." for j, r in enumerate(rollout_results)
                ])
            
            rollout_response = together.chat.completions.create(
                model=self.model,
                messages=self.conversation_history + [rollout_prompt],
                max_tokens=4096,
                temperature=0.7 + (i * 0.1)  # Increase temperature for more diversity in later rollouts
            )
            
            rollout_content = rollout_response.choices[0].message.content
            
            # Store this rollout
            rollout_result = {
                "rollout_id": i+1,
                "solution": rollout_content,
                "timestamp": time.time()
            }
            rollout_results.append(rollout_result)
            
            # Add a brief evaluation of this rollout
            eval_prompt = {
                "role": "user",
                "content": f"Evaluate the strengths and weaknesses of solution approach #{i+1}."
            }
            
            eval_response = together.chat.completions.create(
                model=self.model,
                messages=self.conversation_history + [{"role": "assistant", "content": rollout_content}] + [eval_prompt],
                max_tokens=512
            )
            
            eval_content = eval_response.choices[0].message.content
            rollout_result["evaluation"] = eval_content
        
        # Store all rollouts
        self.rollouts.extend(rollout_results)
        
        # Select the best rollout based on evaluations
        best_rollout = self._select_best_rollout(rollout_results)
        
        # Add the final selected solution to conversation history
        self.conversation_history.append({
            "role": "user", 
            "content": "Based on all the approaches you've considered, what is your final recommended solution?"
        })
        
        final_response = together.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            max_tokens=1024
        )
        
        final_content = final_response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": final_content})
        
        # Update performance metrics
        execution_time = time.time() - start_time
        self.performance_metrics["tasks_completed"] += 1
        self.performance_metrics["total_execution_time"] += execution_time
        self.performance_metrics["avg_execution_time"] = (
            self.performance_metrics["total_execution_time"] / 
            self.performance_metrics["tasks_completed"]
        )
        
        # Store the solution in memory for future reference
        solution_metadata = {
            "type": "solution",
            "task_id": memory_id,
            "category": self.specialization,
            "execution_time": execution_time,
            "solution": final_content,
            "best_rollout_id": best_rollout.get("rollout_id") if best_rollout else None
        }
        self.memory.add_memory(final_content, solution_metadata)
        
        # Apply reinforcement learning if enabled
        if self.reinforcement_learning["learning_enabled"]:
            # Calculate a reward based on solution quality (using best_rollout evaluation)
            if best_rollout and "evaluation" in best_rollout:
                # Extract sentiment from evaluation to estimate quality
                evaluation = best_rollout["evaluation"].lower()
                positive_terms = ["excellent", "good", "effective", "efficient", "optimal", "best"]
                negative_terms = ["limitation", "drawback", "issue", "problem", "concern", "weakness"]
                
                positive_score = sum(1 for term in positive_terms if term in evaluation)
                negative_score = sum(1 for term in negative_terms if term in evaluation)
                
                # Calculate reward between -1 and 1
                reward = (positive_score - negative_score) / max(1, positive_score + negative_score)
                
                # Store reward
                self.reinforcement_learning["reward_history"].append(reward)
                
                # Adjust exploration rate based on recent performance
                if len(self.reinforcement_learning["reward_history"]) >= 5:
                    recent_rewards = self.reinforcement_learning["reward_history"][-5:]
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                    
                    # If we're doing well, reduce exploration (exploit more)
                    # If we're doing poorly, increase exploration
                    if avg_reward > 0.5:
                        self.reinforcement_learning["exploration_rate"] = max(
                            0.05, self.reinforcement_learning["exploration_rate"] * 0.9
                        )
                    elif avg_reward < 0:
                        self.reinforcement_learning["exploration_rate"] = min(
                            0.5, self.reinforcement_learning["exploration_rate"] * 1.1
                        )
        
        # Check if we should acquire a new skill based on this task
        self._consider_skill_acquisition(task, final_content, best_rollout)
        
        return {
            "task": task,
            "chain_of_thought": cot_content,
            "solution": final_content,
            "agent_id": self.agent_id,
            "specialization": self.specialization,
            "rollouts": rollout_results,
            "best_rollout": best_rollout,
            "execution_time": execution_time,
            "memory_id": memory_id,
            "similar_past_experiences": [exp["id"] for exp in similar_experiences] if similar_experiences else []
        }
        
    def _consider_skill_acquisition(self, task, solution, best_rollout):
        """Consider acquiring a new skill based on task execution"""
        # Potential new skills to detect in the task/solution
        potential_skills = {
            "research": ["research", "analyze", "investigate", "study", "examine", "review"],
            "code": ["code", "program", "develop", "implement", "script", "function", "class"],
            "planning": ["plan", "strategy", "roadmap", "timeline", "schedule", "organize"],
            "creative": ["create", "design", "innovate", "imagine", "novel", "artistic"],
            "critical": ["evaluate", "assess", "critique", "review", "analyze", "judge"],
            "data_analysis": ["data", "statistics", "analyze", "dataset", "correlation", "regression"],
            "summarization": ["summarize", "condense", "digest", "synopsis", "overview"],
            "translation": ["translate", "language", "conversion", "interpret"],
            "mathematical": ["math", "equation", "calculate", "formula", "computation"]
        }
        
        # Check if task/solution contains indicators of skills we don't have
        combined_text = f"{task} {solution}".lower()
        
        for skill, indicators in potential_skills.items():
            # Skip skills we already have
            if skill in self.skills:
                continue
                
            # Count occurrences of skill indicators
            indicator_count = sum(1 for indicator in indicators if indicator in combined_text)
            
            # If enough indicators are present, consider acquiring the skill
            if indicator_count >= 3:
                # Probability of acquiring skill depends on learning rate and indicator count
                acquisition_probability = min(0.8, self.learning_rate * indicator_count / len(indicators))
                
                if random.random() < acquisition_probability:
                    # Acquire the new skill
                    self.skills.add(skill)
                    # Start with partial proficiency
                    self.skill_proficiency[skill] = 0.3
                    console.print(f"[green]Agent {self.agent_id} acquired new skill: {skill}[/green]")
    
    def _select_best_rollout(self, rollouts):
        """Select the best rollout based on evaluations"""
        if not rollouts:
            return None
            
        # If we only have one rollout, return it
        if len(rollouts) == 1:
            return rollouts[0]
            
        # Create a prompt to compare all rollouts
        comparison_prompt = {
            "role": "user",
            "content": "Compare all the solution approaches and select the best one. Explain your reasoning."
        }
        
        # Add all rollouts to the conversation history temporarily
        temp_history = self.conversation_history.copy()
        for i, rollout in enumerate(rollouts):
            temp_history.append({
                "role": "assistant", 
                "content": f"Solution Approach #{i+1}:\n{rollout['solution']}\n\nEvaluation: {rollout['evaluation']}"
            })
        
        from together import Together
        together = Together()
        
        # Generate comparison
        comparison_response = together.chat.completions.create(
            model=self.model,
            messages=temp_history + [comparison_prompt],
            max_tokens=512
        )
        
        comparison_content = comparison_response.choices[0].message.content
        
        # Try to extract the best rollout number
        best_id = 1  # Default to first rollout
        match = re.search(r"(?:approach|solution|rollout)\s*#?\s*(\d+)", comparison_content.lower())
        if match:
            try:
                best_id = int(match.group(1))
                if best_id < 1 or best_id > len(rollouts):
                    best_id = 1
            except:
                best_id = 1
                
        best_rollout = rollouts[best_id-1]
        best_rollout["comparison"] = comparison_content
        
        return best_rollout
    
    def stop(self):
        self.stop_event.set()
        self.worker_thread.join(timeout=2)
        if hasattr(self, 'pubsub_thread') and self.pubsub_thread:
            self.pubsub_thread.join(timeout=2)
        if hasattr(self, 'pubsub') and self.pubsub:
            self.pubsub.unsubscribe()
        
# ================================
# Central Agent Orchestrator
# ================================
class AgentOrchestrator:
    """Advanced orchestrator that coordinates multiple scout agents with dynamic team formation."""
    def __init__(self, num_scouts=3, model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"):
        self.model = model
        self.scouts = {}
        self.task_queue = queue.Queue()
        self.results = {}
        self.stop_event = threading.Event()
        self.orchestrator_thread = threading.Thread(target=self._orchestrator, daemon=True)
        self.knowledge_base = []
        self.processed_urls = set()
        self.url_pattern = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')
        self.image_processing_limit = 5
        
        # Advanced orchestration capabilities
        self.memory = VectorMemory(embedding_model, vector_store)
        self.team_history = {}  # Track team formations and their performance
        self.task_decomposition_cache = {}  # Cache task decompositions
        self.agent_performance_history = {}  # Track agent performance by task type
        self.collaboration_graph = {}  # Graph of agent collaborations
        self.skill_registry = {  # Registry of skills and which agents have them
            "research": set(),
            "code": set(),
            "planning": set(),
            "creative": set(),
            "critical": set(),
            "data_analysis": set(),
            "summarization": set(),
            "translation": set(),
            "mathematical": set()
        }
        self.adaptive_scheduling = True  # Enable adaptive task scheduling
        self.learning_enabled = True  # Enable learning from task execution
        
        # Initialize Redis for orchestrator
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.pubsub = self.redis_client.pubsub()
            self.pubsub.subscribe('agent_broadcast')
            self.pubsub.subscribe('orchestrator')
            self.pubsub_thread = threading.Thread(target=self._listen_for_messages, daemon=True)
            self.pubsub_thread.start()
            print(f"[green]Orchestrator subscribed to Redis channels[/green]")
        except Exception as e:
            print(f"[yellow]Warning: Redis PubSub initialization failed for orchestrator: {e}[/yellow]")
            self.redis_client = None
            self.pubsub = None
        
        # Create scouts with different specializations
        specializations = ["research", "code", "planning", "creative", "critical"]
        for i in range(min(num_scouts, len(specializations))):
            agent_id = f"scout_{i+1}"
            specialization = specializations[i]
            self.scouts[agent_id] = ScoutAgent(agent_id, specialization, model)
            
        # Initialize additional scouts if needed
        for i in range(len(specializations), num_scouts):
            agent_id = f"scout_{i+1}"
            specialization = "general"
            self.scouts[agent_id] = ScoutAgent(agent_id, specialization, model)
            
        self.orchestrator_thread.start()
        
    def _orchestrator(self):
        """Advanced orchestrator loop with dynamic team formation and adaptive scheduling"""
        while not self.stop_event.is_set():
            try:
                task_id, task, priority, context = self.task_queue.get(timeout=1)
                
                # Store task in memory
                memory_metadata = {
                    "type": "orchestrator_task",
                    "priority": priority,
                    "context": context
                }
                memory_id = self.memory.add_memory(task, memory_metadata)
                
                # Check if we've seen similar tasks before
                similar_tasks = self.memory.search_memory(task, limit=3)
                
                # Analyze task to determine required skills
                required_skills = self._analyze_task_skills(task)
                
                # Determine if this is a complex task that needs decomposition
                is_complex = self._is_complex_task(task)
                
                if is_complex and self.adaptive_scheduling:
                    # Decompose complex task into subtasks
                    subtasks = self._decompose_task(task, context)
                    
                    # Create a team of agents for this complex task
                    team = self._form_optimal_team(required_skills, len(subtasks))
                    
                    # Assign subtasks to team members
                    subtask_assignments = self._assign_subtasks_to_team(subtasks, team)
                    
                    # Track this team formation
                    team_id = str(uuid.uuid4())
                    self.team_history[team_id] = {
                        "task_id": task_id,
                        "members": team,
                        "subtasks": subtask_assignments,
                        "formation_time": time.time(),
                        "status": "working"
                    }
                    
                    # Assign subtasks to team members
                    for subtask_id, assignment in subtask_assignments.items():
                        subtask = assignment["subtask"]
                        agent_id = assignment["agent_id"]
                        
                        # Add team context to subtask
                        subtask_context = context.copy() if context else {}
                        subtask_context["team_id"] = team_id
                        subtask_context["parent_task_id"] = task_id
                        
                        # Assign to agent
                        scout = self.scouts[agent_id]
                        scout_task_id = scout.add_task(scout.perform_task, subtask, subtask_context)
                        
                        # Update assignment with task ID
                        assignment["scout_task_id"] = scout_task_id
                    
                    # Update task status
                    self.results[task_id] = {
                        "status": "decomposed",
                        "team_id": team_id,
                        "subtask_count": len(subtasks),
                        "memory_id": memory_id
                    }
                    
                else:
                    # Standard task assignment with skill matching
                    assigned = False
                    
                    # Find the best agent based on skills and performance history
                    best_agent_id = self._find_best_agent_for_task(task, required_skills, context)
                    
                    if best_agent_id:
                        scout = self.scouts[best_agent_id]
                        scout_task_id = scout.add_task(scout.perform_task, task, context)
                        self.results[task_id] = {
                            "status": "assigned", 
                            "scout_id": best_agent_id,
                            "scout_task_id": scout_task_id,
                            "memory_id": memory_id
                        }
                        assigned = True
                    
                    # If no optimal assignment found, try specialization matching
                    if not assigned and "specialization" in context:
                        target_specialization = context["specialization"]
                        for agent_id, scout in self.scouts.items():
                            if scout.specialization == target_specialization and scout.is_available.is_set():
                                scout_task_id = scout.add_task(scout.perform_task, task, context)
                                self.results[task_id] = {
                                    "status": "assigned", 
                                    "scout_id": agent_id,
                                    "scout_task_id": scout_task_id,
                                    "memory_id": memory_id
                                }
                                assigned = True
                                break
                    
                    # If still not assigned, use any available scout
                    if not assigned:
                        for agent_id, scout in self.scouts.items():
                            if scout.is_available.is_set():
                                scout_task_id = scout.add_task(scout.perform_task, task, context)
                                self.results[task_id] = {
                                    "status": "assigned", 
                                    "scout_id": agent_id,
                                    "scout_task_id": scout_task_id,
                                    "memory_id": memory_id
                                }
                                assigned = True
                                break
                                
                    # If all scouts are busy, prioritize based on task importance
                    if not assigned:
                        if priority > 1:  # High priority task
                            # Find the scout working on the lowest priority task
                            lowest_priority_scout = None
                            lowest_priority = float('inf')
                            
                            for agent_id, scout in self.scouts.items():
                                # Check current task priority
                                current_tasks = self._get_agent_current_tasks(agent_id)
                                if current_tasks:
                                    current_priority = min(task.get("priority", 1) for task in current_tasks)
                                    if current_priority < lowest_priority:
                                        lowest_priority = current_priority
                                        lowest_priority_scout = agent_id
                            
                            if lowest_priority_scout and lowest_priority < priority:
                                # Preempt the lower priority task
                                scout = self.scouts[lowest_priority_scout]
                                scout_task_id = scout.add_task(scout.perform_task, task, context)
                                self.results[task_id] = {
                                    "status": "assigned_preemptive", 
                                    "scout_id": lowest_priority_scout,
                                    "scout_task_id": scout_task_id,
                                    "memory_id": memory_id
                                }
                                assigned = True
                        
                        # If still not assigned, put back in queue
                        if not assigned:
                            # Put the task back in the queue with its original priority
                            self.task_queue.put((task_id, task, priority, context))
                            time.sleep(0.5)  # Wait a bit before retrying
                    
            except queue.Empty:
                continue
                
    def _analyze_task_skills(self, task):
        """Analyze a task to determine required skills"""
        required_skills = set()
        task_lower = task.lower()
        
        # Simple keyword-based skill detection
        skill_keywords = {
            "research": ["research", "find", "search", "investigate", "explore", "analyze"],
            "code": ["code", "program", "develop", "implement", "function", "class", "algorithm"],
            "planning": ["plan", "schedule", "organize", "strategy", "roadmap", "timeline"],
            "creative": ["create", "design", "generate", "imagine", "creative", "novel"],
            "critical": ["evaluate", "assess", "critique", "review", "analyze", "critical"],
            "data_analysis": ["data", "analyze", "statistics", "dataset", "correlation", "visualization"],
            "summarization": ["summarize", "summary", "condense", "overview", "digest"],
            "translation": ["translate", "language", "conversion"],
            "mathematical": ["math", "calculate", "equation", "formula", "computation"]
        }
        
        for skill, keywords in skill_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                required_skills.add(skill)
        
        # If no skills detected, add a default
        if not required_skills:
            required_skills.add("general")
            
        return required_skills
        
    def _is_complex_task(self, task):
        """Determine if a task is complex and should be decomposed"""
        # Simple heuristics for complexity
        word_count = len(task.split())
        sentence_count = len(re.split(r'[.!?]+', task))
        question_count = task.count('?')
        
        # Check for indicators of complexity
        complexity_indicators = [
            "multiple", "several", "various", "different", "steps",
            "complex", "complicated", "detailed", "comprehensive",
            "analyze and", "research and", "implement and"
        ]
        
        has_complexity_indicators = any(indicator in task.lower() for indicator in complexity_indicators)
        
        # Task is complex if it's long or has complexity indicators
        return (word_count > 50 or 
                sentence_count > 3 or 
                question_count > 1 or 
                has_complexity_indicators)
                
    def _decompose_task(self, task, context=None):
        """Decompose a complex task into subtasks"""
        # Check cache first
        cache_key = hashlib.md5(task.encode()).hexdigest()
        if cache_key in self.task_decomposition_cache:
            return self.task_decomposition_cache[cache_key]
            
        # Use LLM to decompose the task
        from together import Together
        together = Together()
        
        decomposition_prompt = [
            {"role": "system", "content": "You are an expert at breaking down complex tasks into smaller, manageable subtasks."},
            {"role": "user", "content": f"Break down the following task into 2-5 subtasks that can be worked on independently. Return ONLY a JSON array of subtask descriptions.\n\nTask: {task}"}
        ]
        
        try:
            response = together.chat.completions.create(
                model=self.model,
                messages=decomposition_prompt,
                                    
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Extract subtasks from response
            if isinstance(result, dict) and "subtasks" in result:
                subtasks = result["subtasks"]
            elif isinstance(result, list):
                subtasks = result
            else:
                # Fallback if format is unexpected
                subtasks = [task]
                
            # Cache the decomposition
            self.task_decomposition_cache[cache_key] = subtasks
            
            return subtasks
            
        except Exception as e:
            console.print(f"[yellow]Error decomposing task: {e}[/yellow]")
            # Fallback to simple decomposition
            return [task]
            
    def _form_optimal_team(self, required_skills, team_size):
        """Form an optimal team based on required skills and agent capabilities"""
        team = []
        
        # First, add agents with the required skills
        for skill in required_skills:
            if skill in self.skill_registry:
                skilled_agents = self.skill_registry[skill]
                available_skilled_agents = [
                    agent_id for agent_id in skilled_agents 
                    if agent_id not in team and self.scouts[agent_id].is_available.is_set()
                ]
                
                if available_skilled_agents:
                    # Choose the agent with highest proficiency in this skill
                    best_agent = max(
                        available_skilled_agents,
                        key=lambda a: self.scouts[a].skill_proficiency.get(skill, 0)
                    )
                    team.append(best_agent)
                
                # Stop if we have enough team members
                if len(team) >= team_size:
                    break
        
        # If we need more team members, add available agents
        if len(team) < team_size:
            for agent_id, scout in self.scouts.items():
                if agent_id not in team and scout.is_available.is_set():
                    team.append(agent_id)
                    if len(team) >= team_size:
                        break
        
        return team
        
    def _assign_subtasks_to_team(self, subtasks, team):
        """Assign subtasks to team members optimally"""
        assignments = {}
        
        # If team is empty, can't assign
        if not team:
            return assignments
            
        # Analyze subtasks to determine required skills
        subtask_skills = {}
        for i, subtask in enumerate(subtasks):
            subtask_id = f"subtask_{i+1}"
            subtask_skills[subtask_id] = self._analyze_task_skills(subtask)
            
        # Create a mapping of agents to their skills
        agent_skills = {}
        for agent_id in team:
            scout = self.scouts[agent_id]
            agent_skills[agent_id] = scout.skills
            
        # Assign subtasks to agents based on skill matching
        assigned_agents = set()
        
        # First pass: assign subtasks to agents with matching skills
        for subtask_id, skills in subtask_skills.items():
            best_agent = None
            best_match = 0
            
            for agent_id in team:
                if agent_id in assigned_agents:
                    continue
                    
                # Count matching skills
                match_count = len(skills.intersection(agent_skills[agent_id]))
                
                # Consider agent's proficiency in these skills
                proficiency = 0
                for skill in skills:
                    proficiency += self.scouts[agent_id].skill_proficiency.get(skill, 0)
                
                # Combined score
                score = match_count + proficiency
                
                if score > best_match:
                    best_match = score
                    best_agent = agent_id
            
            if best_agent:
                assignments[subtask_id] = {
                    "subtask": subtasks[int(subtask_id.split('_')[1]) - 1],
                    "agent_id": best_agent,
                    "skills_required": list(skills)
                }
                assigned_agents.add(best_agent)
        
        # Second pass: assign remaining subtasks to any available team members
        unassigned_subtasks = [
            (subtask_id, subtasks[int(subtask_id.split('_')[1]) - 1])
            for subtask_id in subtask_skills
            if subtask_id not in assignments
        ]
        
        available_agents = [agent_id for agent_id in team if agent_id not in assigned_agents]
        
        for i, (subtask_id, subtask) in enumerate(unassigned_subtasks):
            if i < len(available_agents):
                agent_id = available_agents[i]
                assignments[subtask_id] = {
                    "subtask": subtask,
                    "agent_id": agent_id,
                    "skills_required": list(subtask_skills[subtask_id])
                }
            else:
                # If we run out of agents, assign to agents that already have tasks
                # (round-robin assignment)
                agent_id = team[i % len(team)]
                assignments[subtask_id] = {
                    "subtask": subtask,
                    "agent_id": agent_id,
                    "skills_required": list(subtask_skills[subtask_id])
                }
        
        return assignments
        
    def _get_agent_current_tasks(self, agent_id):
        """Get the current tasks assigned to an agent"""
        current_tasks = []
        
        for task_id, result in self.results.items():
            if result.get("status") in ["assigned", "assigned_preemptive"] and result.get("scout_id") == agent_id:
                scout = self.scouts[agent_id]
                scout_task_id = result.get("scout_task_id")
                
                if scout_task_id:
                    task_result = scout.get_result(scout_task_id)
                    if task_result.get("status") == "pending":
                        current_tasks.append({
                            "task_id": task_id,
                            "scout_task_id": scout_task_id,
                            "priority": self._get_task_priority(task_id)
                        })
        
        return current_tasks
        
    def _get_task_priority(self, task_id):
        """Get the priority of a task"""
        # Default priority is 1
        priority = 1
        
        # Look through the task queue for this task
        for item in list(self.task_queue.queue):
            if item[0] == task_id:  # item[0] is the task_id
                priority = item[2]  # item[2] is the priority
                break
                
        return priority
        
    def _find_best_agent_for_task(self, task, required_skills, context=None):
        """Find the best agent for a task based on skills and performance history"""
        candidates = []
        
        for agent_id, scout in self.scouts.items():
            if not scout.is_available.is_set():
                continue
                
            # Calculate skill match score
            skill_match = len(required_skills.intersection(scout.skills))
            
            # Calculate proficiency score
            proficiency = sum(scout.skill_proficiency.get(skill, 0) for skill in required_skills)
            
            # Calculate performance score based on history
            performance_score = 0
            if agent_id in self.agent_performance_history:
                history = self.agent_performance_history[agent_id]
                if history["tasks_completed"] > 0:
                    performance_score = history["success_rate"] * 2  # Weight success rate highly
                    
                    # Bonus for specialization match
                    if "specialization" in context and scout.specialization == context["specialization"]:
                        performance_score += 1
            
            # Combined score
            total_score = skill_match + proficiency + performance_score
            
            candidates.append((agent_id, total_score))
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best candidate, or None if no candidates
        return candidates[0][0] if candidates else None
                
    def add_task(self, task, priority=1, context=None):
        """Add a task to be processed by a scout agent"""
        if context is None:
            context = {}
        task_id = str(uuid.uuid4())
        self.results[task_id] = {"status": "pending"}
        self.task_queue.put((task_id, task, priority, context))
        return task_id
        
    def get_task_result(self, task_id):
        """Get the result of a task by its ID"""
        task_info = self.results.get(task_id)
        if not task_info:
            return {"status": "not_found"}
            
        # If the task has been assigned to a scout, get the result from the scout
        if task_info["status"] == "assigned":
            scout_id = task_info["scout_id"]
            scout_task_id = task_info["scout_task_id"]
            scout_result = self.scouts[scout_id].get_result(scout_task_id)
            
            # If the scout has completed the task, update our records
            if scout_result["status"] in ["completed", "failed"]:
                self.results[task_id] = {
                    "status": scout_result["status"],
                    "result": scout_result.get("result"),
                    "scout_id": scout_id
                }
                
        return self.results[task_id]
        
    def get_all_chains_of_thought(self):
        """Get all chains of thought from all scouts"""
        all_thoughts = []
        for scout_id, scout in self.scouts.items():
            all_thoughts.extend(scout.chains_of_thought)
        return sorted(all_thoughts, key=lambda x: x["timestamp"])
        
    def get_all_rollouts(self):
        """Get all solution rollouts from all scouts"""
        all_rollouts = []
        for scout_id, scout in self.scouts.items():
            for rollout in scout.rollouts:
                # Add scout information to each rollout
                rollout_copy = rollout.copy()
                rollout_copy["scout_id"] = scout_id
                rollout_copy["specialization"] = scout.specialization
                all_rollouts.append(rollout_copy)
        return sorted(all_rollouts, key=lambda x: x["timestamp"], reverse=True)
        
    def update_skill_registry(self):
        """Update the skill registry based on current scout skills"""
        # Reset registry
        for skill in self.skill_registry:
            self.skill_registry[skill] = set()
            
        # Update with current skills
        for agent_id, scout in self.scouts.items():
            for skill in scout.skills:
                if skill in self.skill_registry:
                    self.skill_registry[skill].add(agent_id)
                    
    def update_collaboration_graph(self, team_id):
        """Update the collaboration graph based on team performance"""
        if team_id not in self.team_history:
            return
            
        team_info = self.team_history[team_id]
        team_members = team_info.get("members", [])
        
        # Update collaboration links between team members
        for i, agent1 in enumerate(team_members):
            if agent1 not in self.collaboration_graph:
                self.collaboration_graph[agent1] = {}
                
            for agent2 in team_members[i+1:]:
                if agent2 not in self.collaboration_graph:
                    self.collaboration_graph[agent2] = {}
                    
                # Increment collaboration count
                self.collaboration_graph[agent1][agent2] = self.collaboration_graph[agent1].get(agent2, 0) + 1
                self.collaboration_graph[agent2][agent1] = self.collaboration_graph[agent2].get(agent1, 0) + 1
                
    def get_collaboration_network(self):
        """Get the collaboration network statistics"""
        if not self.collaboration_graph:
            return {"nodes": [], "edges": [], "stats": {"density": 0, "avg_collaborations": 0}}
            
        nodes = []
        edges = []
        total_collaborations = 0
        collaboration_count = 0
        
        for agent1, collaborators in self.collaboration_graph.items():
            nodes.append({"id": agent1, "specialization": self.scouts[agent1].specialization})
            
            for agent2, count in collaborators.items():
                edges.append({"source": agent1, "target": agent2, "count": count})
                total_collaborations += count
                collaboration_count += 1
                
        # Calculate network statistics
        agent_count = len(self.scouts)
        max_possible_edges = (agent_count * (agent_count - 1)) / 2
        density = collaboration_count / max_possible_edges if max_possible_edges > 0 else 0
        avg_collaborations = total_collaborations / collaboration_count if collaboration_count > 0 else 0
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "density": density,
                "avg_collaborations": avg_collaborations,
                "total_collaborations": total_collaborations
            }
        }
        
    def get_agent_performance_metrics(self):
        """Get performance metrics for all agents"""
        metrics = {}
        
        for agent_id, scout in self.scouts.items():
            metrics[agent_id] = {
                "specialization": scout.specialization,
                "tasks_completed": scout.performance_metrics["tasks_completed"],
                "success_rate": scout.performance_metrics["success_rate"],
                "avg_execution_time": scout.performance_metrics["avg_execution_time"],
                "skills": list(scout.skills),
                "skill_proficiency": scout.skill_proficiency
            }
            
        return metrics
        
    def optimize_team_structure(self):
        """Optimize the team structure based on performance history"""
        if not self.learning_enabled:
            return {"status": "learning_disabled"}
            
        # Update skill registry
        self.update_skill_registry()
        
        # Analyze skill gaps
        skill_coverage = {}
        for skill, agents in self.skill_registry.items():
            skill_coverage[skill] = len(agents) / len(self.scouts) if self.scouts else 0
            
        # Identify skills with low coverage
        low_coverage_skills = [skill for skill, coverage in skill_coverage.items() if coverage < 0.3]
        
        # Encourage skill acquisition for underrepresented skills
        for agent_id, scout in self.scouts.items():
            for skill in low_coverage_skills:
                if skill not in scout.skills:
                    # Increase learning rate for this agent
                    scout.learning_rate = min(0.5, scout.learning_rate * 1.2)
                    break
        
        # Analyze collaboration patterns
        collaboration_network = self.get_collaboration_network()
        
        # Identify isolated agents (low collaboration)
        agent_collaboration_counts = {}
        for agent_id in self.scouts:
            agent_collaboration_counts[agent_id] = 0
            
        for edge in collaboration_network.get("edges", []):
            agent_collaboration_counts[edge["source"]] += edge["count"]
            agent_collaboration_counts[edge["target"]] += edge["count"]
            
        isolated_agents = [
            agent_id for agent_id, count in agent_collaboration_counts.items()
            if count < collaboration_network["stats"].get("avg_collaborations", 0) / 2
        ]
        
        # Encourage isolated agents to collaborate more
        for agent_id in isolated_agents:
            scout = self.scouts[agent_id]
            # Increase exploration rate to try new approaches
            scout.reinforcement_learning["exploration_rate"] = min(
                0.5, scout.reinforcement_learning["exploration_rate"] * 1.2
            )
            
        return {
            "skill_coverage": skill_coverage,
            "low_coverage_skills": low_coverage_skills,
            "isolated_agents": isolated_agents,
            "optimization_applied": True
        }
        
    def execute_parallel_tasks(self, tasks, context=None):
        """Execute multiple tasks in parallel and wait for all results"""
        task_ids = []
        for task in tasks:
            task_id = self.add_task(task, context=context)
            task_ids.append(task_id)
            
        results = []
        for task_id in task_ids:
            while True:
                result = self.get_task_result(task_id)
                if result["status"] in ["completed", "failed"]:
                    results.append(result)
                    break
                time.sleep(0.5)
                
        return results
        
    def _listen_for_messages(self):
        """Listen for messages from agents via Redis pubsub"""
        if not self.pubsub:
            return
            
        while not self.stop_event.is_set():
            try:
                message = self.pubsub.get_message(timeout=1)
                if message and message['type'] == 'message':
                    data = json.loads(message['data'].decode('utf-8'))
                    print(f"[cyan]Orchestrator received message on channel {message['channel'].decode('utf-8')}: {data}[/cyan]")
                    
                    # Handle different message types
                    if data.get('type') == 'task_result':
                        # An agent has completed a task
                        task_id = data.get('task_id')
                        if task_id in self.results:
                            self.results[task_id].update({
                                "status": "completed",
                                "result": data.get('content'),
                                "scout_id": data.get('sender_id')
                            })
                    elif data.get('type') == 'knowledge_share':
                        # An agent is sharing knowledge with everyone
                        self.knowledge_base.append({
                            'content': data.get('content'),
                            'source': f"agent_{data.get('sender_id')}",
                            'timestamp': time.time()
                        })
            except Exception as e:
                print(f"[yellow]Error in pubsub listener for orchestrator: {e}[/yellow]")
                time.sleep(1)
                
    def send_message_to_agent(self, agent_id, message_type, content):
        """Send a message to a specific agent via Redis pubsub"""
        if not hasattr(self, 'redis_client') or not self.redis_client:
            print(f"[yellow]Warning: Redis client not available for orchestrator[/yellow]")
            return False
            
        try:
            message = {
                'sender_id': 'orchestrator',
                'type': message_type,
                'content': content,
                'timestamp': time.time()
            }
            self.redis_client.publish(f'agent_{agent_id}', json.dumps(message))
            return True
        except Exception as e:
            print(f"[yellow]Error sending message from orchestrator to agent {agent_id}: {e}[/yellow]")
            return False
            
    def broadcast_to_agents(self, message_type, content):
        """Broadcast a message to all agents via Redis pubsub"""
        if not hasattr(self, 'redis_client') or not self.redis_client:
            print(f"[yellow]Warning: Redis client not available for orchestrator[/yellow]")
            return False
            
        try:
            message = {
                'sender_id': 'orchestrator',
                'type': message_type,
                'content': content,
                'timestamp': time.time()
            }
            self.redis_client.publish('agent_broadcast', json.dumps(message))
            return True
        except Exception as e:
            print(f"[yellow]Error broadcasting message from orchestrator: {e}[/yellow]")
            return False
    
    def stop(self):
        """Stop the orchestrator and all scouts"""
        self.stop_event.set()
        for scout in self.scouts.values():
            scout.stop()
        self.orchestrator_thread.join(timeout=2)
        if hasattr(self, 'pubsub_thread') and self.pubsub_thread:
            self.pubsub_thread.join(timeout=2)
        if hasattr(self, 'pubsub') and self.pubsub:
            self.pubsub.unsubscribe()

# ================================
# Async Task Processor
# ================================
class AsyncTaskProcessor:
    """Processes tasks asynchronously in a background thread."""
    def __init__(self):
        self.task_queue = queue.Queue()
        self.results = {}
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.knowledge_base = []
        self.processed_urls = set()
        self.url_pattern = re.compile(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+')
        self.image_processing_limit = 5
        self.worker_thread.start()

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                task_id, task_func, args, kwargs = self.task_queue.get(timeout=1)
                try:
                    if asyncio.iscoroutinefunction(task_func):
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(task_func(*args, **kwargs))
                    else:
                        result = task_func(*args, **kwargs)
                    self.results[task_id] = {"status": "completed", "result": result}
                except Exception as e:
                    self.results[task_id] = {
                        "status": "failed",
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
                finally:
                    self.task_queue.task_done()
            except queue.Empty:
                continue

    def add_task(self, task_func, *args, **kwargs):
        task_id = str(uuid.uuid4())
        self.results[task_id] = {"status": "pending"}
        self.task_queue.put((task_id, task_func, args, kwargs))
        return task_id

    def get_result(self, task_id):
        return self.results.get(task_id, {"status": "not_found"})

    def extract_urls(self, text):
        urls = [url for url in self.url_pattern.findall(text)]
        normalized_urls = []
        for url in urls:
            if url.startswith('www.'):
                url = 'https://' + url
            normalized_urls.append(url)
        return URLExtraction(source="text_extraction", urls=normalized_urls)

    def add_urls_to_process(self, urls, process_images=False):
        new_urls = [url for url in urls if url not in self.processed_urls]
        if process_images:
            image_urls = [url for url in new_urls if url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
            for image_url in image_urls[:self.image_processing_limit]:
                self.processed_urls.add(image_url)
                self.add_task(self._process_image, image_url)
            new_urls = [url for url in new_urls if url not in image_urls]

        for url in new_urls:
            if url not in self.processed_urls:
                self.processed_urls.add(url)
                self.add_task(self._process_url, url)
        return len(new_urls)

    async def _process_image(self, image_url):
        url = image_url  # Define the url variable
        try:
            # Simulate image processing
            console.print(f"[blue]Processing image: {image_url}[/blue]")
            await asyncio.sleep(1)  # Simulate processing time
            return {"success": True, "image_url": image_url, "processed": True}
        except Exception as e:
            return {"success": False, "image_url": image_url, "error": str(e)}
        try:
            jina_client = JinaClient(token=os.environ.get("JINA_API_KEY"))
            result = await jina_client.read(url)
            if isinstance(result, dict) and "results" in result:
                content = result["results"]
                knowledge_item = KnowledgeItem(content=content, source_url=url)
                self.knowledge_base.append(knowledge_item)
                urls_extraction = self.extract_urls(content)
                self.add_urls_to_process(urls_extraction.urls)
                return {
                    "success": True,
                    "url": url,
                    "knowledge_extracted": True,
                    "further_urls_found": len(urls_extraction.urls)
                }
        except Exception as e:
            return {"success": False, "url": url, "error": str(e)}
            
    async def _process_url(self, url):
        """Process a URL to extract knowledge"""
        try:
            console.print(f"[blue]Processing URL: {url}[/blue]")
            
            # Check if it's a valid URL
            if not url.startswith(('http://', 'https://')):
                return {"success": False, "url": url, "error": "Invalid URL format"}
                
            # Try to fetch content from the URL
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()
                            
                            # Add to knowledge base
                            knowledge_item = KnowledgeItem(content=content, source_url=url)
                            self.knowledge_base.append(knowledge_item)
                            
                            # Extract further URLs
                            urls_extraction = self.extract_urls(content)
                            if urls_extraction.urls:
                                self.add_urls_to_process(urls_extraction.urls)
                            
                            return {
                                "success": True,
                                "url": url,
                                "content_length": len(content),
                                "knowledge_extracted": True,
                                "further_urls_found": len(urls_extraction.urls) if urls_extraction else 0
                            }
                        else:
                            return {"success": False, "url": url, "error": f"HTTP status {response.status}"}
            except Exception as e:
                return {"success": False, "url": url, "error": f"Request failed: {str(e)}"}
                
        except Exception as e:
            return {"success": False, "url": url, "error": str(e)}

    def get_knowledge_summary(self):
        return {
            "total_items": len(self.knowledge_base),
            "total_urls_processed": len(self.processed_urls),
            "total_urls_pending": self.task_queue.qsize(),
            "recent_items": [
                {"source": item.source_url, "timestamp": item.timestamp}
                for item in sorted(self.knowledge_base, key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }

    def search_knowledge(self, query):
        results = []
        for item in self.knowledge_base:
            if query.lower() in item.content.lower():
                results.append({
                    "source": item.source_url,
                    "timestamp": item.timestamp,
                    "relevance": "high" if query.lower() in item.content.lower()[:500] else "medium"
                })
        return results

    def stop(self):
        self.stop_event.set()
        self.worker_thread.join(timeout=2)

# =======================
# Jina API Client
# =======================
class JinaClient:
    """Client for interacting with Jina.ai endpoints"""
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("JINA_API_KEY")
        if not self.token:
            raise ValueError("JINA_API_KEY environment variable or token must be provided")
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    async def search(self, query: str) -> dict:
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/{encoded_query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response_text = await response.text()
                return {"results": response_text}

    async def fact_check(self, query: str) -> str:
        encoded_query = urllib.parse.quote(query)
        url = f"https://g.jina.ai/{encoded_query}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                return await response.text()

    async def read(self, url: str) -> dict:
        encoded_url = urllib.parse.quote(url)
        rank_url = f"https://r.jina.ai/{encoded_url}"
        async with aiohttp.ClientSession() as session:
            async with session.get(rank_url, headers=self.headers) as response:
                response_text = await response.text()
                return {"results": response_text}

# =======================
# Code Repository
# =======================
class CodeRepository:
    """Repository for storing and managing code artifacts."""
    def __init__(self, db_path=":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._initialize_db()
        
    def _initialize_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            code TEXT NOT NULL,
            description TEXT,
            created_at REAL NOT NULL,
            execution_count INTEGER DEFAULT 0,
            metadata TEXT,
            last_result TEXT
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS execution_logs (
            log_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL,
            executed_at REAL NOT NULL,
            success INTEGER NOT NULL,
            stdout TEXT,
            stderr TEXT,
            result TEXT,
            execution_time REAL,
            FOREIGN KEY (artifact_id) REFERENCES artifacts (artifact_id)
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS modules (
            module_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            code TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_updated_at REAL NOT NULL,
            description TEXT
        )
        ''')
        self.conn.commit()

    def add_artifact(self, name: str, code: str, description: str = "", metadata: Dict[str, Any] = None) -> str:
        artifact_id = str(uuid.uuid4())
        created_at = time.time()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO artifacts (artifact_id, name, code, description, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (artifact_id, name, code, description, created_at, json.dumps(metadata or {}))
        )
        self.conn.commit()
        return artifact_id

    def update_artifact(self, artifact_id: str, code: str = None, description: str = None, metadata: Dict[str, Any] = None) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("SELECT code, description, metadata FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        row = cursor.fetchone()
        if not row:
            return False
        current_code, current_description, current_metadata_str = row
        current_metadata = json.loads(current_metadata_str) if current_metadata_str else {}
        updated_code = code if code is not None else current_code
        updated_description = description if description is not None else current_description
        if metadata:
            current_metadata.update(metadata)
        updated_metadata = json.dumps(current_metadata)
        cursor.execute(
            "UPDATE artifacts SET code = ?, description = ?, metadata = ? WHERE artifact_id = ?",
            (updated_code, updated_description, updated_metadata, artifact_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_artifact(self, artifact_id: str) -> Optional[CodeArtifact]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        row = cursor.fetchone()
        if not row:
            return None
        column_names = [desc[0] for desc in cursor.description]
        artifact_data = dict(zip(column_names, row))
        return CodeArtifact(
            artifact_id=artifact_data["artifact_id"],
            name=artifact_data["name"],
            code=artifact_data["code"],
            description=artifact_data["description"],
            created_at=artifact_data["created_at"],
            execution_count=artifact_data["execution_count"],
            metadata=json.loads(artifact_data["metadata"]) if artifact_data["metadata"] else {},
            last_result=json.loads(artifact_data["last_result"]) if artifact_data["last_result"] else None
        )

    def find_artifacts(self, query: str = None, limit: int = 10) -> List[CodeArtifact]:
        cursor = self.conn.cursor()
        if query:
            sql = """
            SELECT * FROM artifacts 
            WHERE name LIKE ? OR description LIKE ? OR code LIKE ?
            ORDER BY created_at DESC LIMIT ?
            """
            params = (f"%{query}%", f"%{query}%", f"%{query}%", limit)
        else:
            sql = "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT ?"
            params = (limit,)
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        artifacts = []
        col_names = [desc[0] for desc in cursor.description]
        for row in rows:
            artifact_data = dict(zip(col_names, row))
            artifacts.append(CodeArtifact(
                artifact_id=artifact_data["artifact_id"],
                name=artifact_data["name"],
                code=artifact_data["code"],
                description=artifact_data["description"],
                created_at=artifact_data["created_at"],
                execution_count=artifact_data["execution_count"],
                metadata=json.loads(artifact_data["metadata"]) if artifact_data["metadata"] else {},
                last_result=json.loads(artifact_data["last_result"]) if artifact_data["last_result"] else None
            ))
        return artifacts

    def log_execution(self, artifact_id: str, success: bool, stdout: str = "", stderr: str = "", 
                      result: str = None, execution_time: float = None) -> str:
        log_id = str(uuid.uuid4())
        executed_at = time.time()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO execution_logs (log_id, artifact_id, executed_at, success, stdout, stderr, result, execution_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (log_id, artifact_id, executed_at, 1 if success else 0, stdout, stderr, result, execution_time)
        )
        cursor.execute(
            "UPDATE artifacts SET execution_count = execution_count + 1, last_result = ? WHERE artifact_id = ?",
            (json.dumps({"success": success, "result": result, "executed_at": executed_at}), artifact_id)
        )
        self.conn.commit()
        return log_id

    def get_execution_logs(self, artifact_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM execution_logs WHERE artifact_id = ? ORDER BY executed_at DESC LIMIT ?", (artifact_id, limit))
        rows = cursor.fetchall()
        logs = []
        col_names = [desc[0] for desc in cursor.description]
        for row in rows:
            logs.append(dict(zip(col_names, row)))
        return logs

    def add_module(self, name: str, code: str, description: str = "") -> str:
        module_id = str(uuid.uuid4())
        timestamp = time.time()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO modules (module_id, name, code, created_at, last_updated_at, description) VALUES (?, ?, ?, ?, ?, ?)",
            (module_id, name, code, timestamp, timestamp, description)
        )
        self.conn.commit()
        return module_id

    def update_module(self, name: str, code: str, description: str = None) -> bool:
        timestamp = time.time()
        cursor = self.conn.cursor()
        if description is not None:
            cursor.execute(
                "UPDATE modules SET code = ?, last_updated_at = ?, description = ? WHERE name = ?",
                (code, timestamp, description, name)
            )
        else:
            cursor.execute(
                "UPDATE modules SET code = ?, last_updated_at = ? WHERE name = ?",
                (code, timestamp, name)
            )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_module(self, name: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM modules WHERE name = ?", (name,))
        row = cursor.fetchone()
        if not row:
            return None
        col_names = [desc[0] for desc in cursor.description]
        return dict(zip(col_names, row))

    def list_modules(self) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM modules ORDER BY name")
        rows = cursor.fetchall()
        modules = []
        col_names = [desc[0] for desc in cursor.description]
        for row in rows:
            modules.append(dict(zip(col_names, row)))
        return modules

    def delete_module(self, name: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM modules WHERE name = ?", (name,))
        self.conn.commit()
        return cursor.rowcount > 0

    def execute_module(self, name: str, globals_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        module = self.get_module(name)
        if not module:
            return {"success": False, "error": f"Module '{name}' not found"}
        try:
            if globals_dict is None:
                globals_dict = globals().copy()
            locals_dict = {}
            start_time = time.time()
            exec(module["code"], globals_dict, locals_dict)
            execution_time = time.time() - start_time
            return {"success": True, "locals": locals_dict, "execution_time": execution_time}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def close(self):
        if self.conn:
            self.conn.close()

# =======================
# Tool Registry
# =======================
class ToolRegistry:
    def __init__(self):
        self.functions: Dict[str, FunctionSpec] = {}
        self.code_repo = CodeRepository(db_path="code_artifacts.db")
        self.jina_client = None
        try:
            self.jina_client = JinaClient()
            console.print("[green]Jina client initialized successfully[/green]")
        except ValueError:
            console.print("[yellow]Warning: JINA_API_KEY not found. Jina tools will not be available.[/yellow]")
            console.print("[yellow]Set the JINA_API_KEY environment variable to enable web search functionality.[/yellow]")
        self._register_default_tools()
        self.together = Together()

    def _register_default_tools(self):
        # Basic response tool
        self.register_function(
            name="respond_to_user",
            description="Respond directly to the user with a text message",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The text response to send to the user"}
                },
                "required": ["message"]
            },
            function=self._respond_to_user
        )
        
        # Weather function for direct location queries
        self.register_function(
            name="weather_for_location",
            description="Get current weather information for a specific location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Location (e.g., city name, city and state)"}
                },
                "required": ["location"]
            },
            function=self._weather_for_location
        )
        
        # Scout Agent Orchestration tools
        self.register_function(
            name="orchestrate_tasks",
            description="Split a complex task into subtasks and assign them to specialized scout agents to run in parallel for maximum efficiency",
            parameters={
                "type": "object",
                "properties": {
                    "main_task": {"type": "string", "description": "The main task description"},
                    "subtasks": {"type": "array", "items": {"type": "string"}, "description": "List of subtasks to run in parallel"},
                    "priority": {"type": "integer", "description": "Task priority (1-5, higher is more important)", "default": 3},
                    "context": {"type": "object", "description": "Additional context to provide for all subtasks", "default": {}}
                },
                "required": ["main_task", "subtasks"]
            },
            function=self._orchestrate_tasks
        )
        
        # Code extraction tool
        self.register_function(
            name="extract_code",
            description="Extract code blocks from text using advanced structured output parsing with language detection",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text containing code blocks to extract"},
                    "language": {"type": "string", "description": "Specific language to extract (e.g., 'python', 'javascript', 'html', 'css')", "default": ""},
                    "execute": {"type": "boolean", "description": "Whether to automatically execute extracted Python code", "default": False}
                },
                "required": ["text"]
            },
            function=self._extract_code
        )
        
        # Extract and execute code
        self.register_function(
            name="extract_and_execute_code",
            description="Extract code blocks from text and execute them",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text containing code blocks to extract and execute"}
                },
                "required": ["text"]
            },
            function=self._extract_and_execute_code
        )
        
        self.register_function(
            name="generate_solution_rollouts",
            description="Generate multiple solution approaches (rollouts) for a given task",
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task to generate solutions for"},
                    "num_rollouts": {"type": "integer", "description": "Number of different solution approaches to generate", "default": 3},
                    "specialization": {"type": "string", "description": "Type of scout agent to use (research, code, planning, creative, critical)", "default": ""}
                },
                "required": ["task"]
            },
            function=self._generate_solution_rollouts
        )
        
        # Advanced agent management tools
        self.register_function(
            name="optimize_agent_team",
            description="Optimize the agent team structure based on performance history",
            parameters={
                "type": "object",
                "properties": {}
            },
            function=self._optimize_agent_team
        )

        self.register_function(
            name="get_agent_performance",
            description="Get performance metrics for all agents",
            parameters={
                "type": "object",
                "properties": {}
            },
            function=self._get_agent_performance
        )

        self.register_function(
            name="get_collaboration_network",
            description="Get the collaboration network between agents",
            parameters={
                "type": "object",
                "properties": {}
            },
            function=self._get_collaboration_network
        )

        # Advanced memory management tools
        self.register_function(
            name="search_agent_memory",
            description="Search the agent's semantic memory",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum number of results", "default": 5}
                },
                "required": ["query"]
            },
            function=self._search_agent_memory
        )

        self.register_function(
            name="add_to_memory",
            description="Add an item to the agent's semantic memory",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to remember"},
                    "category": {"type": "string", "description": "Memory category", "default": "general"}
                },
                "required": ["content"]
            },
            function=self._add_to_memory
        )

        self.register_function(
            name="get_memory_summary",
            description="Get a summary of the agent's memory",
            parameters={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter by category", "default": ""}
                }
            },
            function=self._get_memory_summary
        )

        self.register_function(
            name="assign_specialized_task",
            description="Assign a task to a specific type of specialized scout agent",
            parameters={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task description"},
                    "specialization": {"type": "string", "description": "The required specialization (research, code, planning, creative, critical)"},
                    "context": {"type": "object", "description": "Additional context for the task", "default": {}}
                },
                "required": ["task", "specialization"]
            },
            function=self._assign_specialized_task
        )

        self.register_function(
            name="get_scout_status",
            description="Get the status of all scout agents",
            parameters={
                "type": "object",
                "properties": {}
            },
            function=self._get_scout_status
        )

        self.register_function(
            name="get_chains_of_thought",
            description="Get all chains of thought from scout agents' reasoning",
            parameters={
                "type": "object",
                "properties": {}
            },
            function=self._get_chains_of_thought
        )

        self.register_function(
            name="get_solution_rollouts",
            description="Get all solution rollouts from scout agents",
            parameters={
                "type": "object",
                "properties": {}
            },
            function=self._get_solution_rollouts
        )

        # Date and time tools
        self.register_function(
            name="create_pydantic_model",
            description="Dynamically create a Pydantic model",
            parameters={
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Name of the Pydantic model"},
                    "fields": {"type": "object", "description": "Fields for the Pydantic model"}
                },
                "required": ["model_name", "fields"]
            },
            function=self._create_pydantic_model
        )

        self.register_function(
            name="add_numbers",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
            function=self._add_numbers
        )
        self.register_function(
            name="subtract_numbers",
            description="Subtract two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
            function=self._subtract_numbers
        )
        self.register_function(
            name="multiply_numbers",
            description="Multiply two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            },
            function=self._multiply_numbers
        )
        self.register_function(
            name="divide_numbers",
            description="Divide two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Numerator"},
                    "b": {"type": "number", "description": "Denominator"}
                },
                "required": ["a", "b"]
            },
            function=self._divide_numbers
        )
        self.register_function(
            name="get_current_datetime",
            description="Get the current date and time",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Optional timezone (e.g., 'UTC', 'US/Eastern')", "default": "local"}
                },
                "required": []
            },
            function=self._get_current_datetime
        )

        # Web search tools
        self.register_function(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            },
            function=self._web_search
        )

        # Python code execution
        self.register_function(
            name="execute_python",
            description="Execute Python code and return the result",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "save_artifact": {"type": "boolean", "description": "Whether to save this code as an artifact", "default": False},
                    "artifact_name": {"type": "string", "description": "Name for the artifact if saving", "default": ""},
                    "description": {"type": "string", "description": "Artifact description", "default": ""}
                },
                "required": ["code"]
            },
            function=self._execute_python
        )
        # Save code to module
        self.register_function(
            name="save_module",
            description="Save Python code as a reusable module",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the module (valid Python identifier)"},
                    "code": {"type": "string", "description": "Python code for the module"},
                    "description": {"type": "string", "description": "Module description", "default": ""}
                },
                "required": ["name", "code"]
            },
            function=self._save_module
        )
        # Execute a saved module
        self.register_function(
            name="execute_module",
            description="Execute a previously saved module",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the module"},
                    "args": {"type": "object", "description": "Arguments to pass (if any)", "default": {}}
                },
                "required": ["name"]
            },
            function=self._execute_saved_module
        )
        # List modules
        self.register_function(
            name="list_modules",
            description="List all available Python modules",
            parameters={"type": "object", "properties": {}},
            function=self._list_modules
        )
        # Get a module
        self.register_function(
            name="get_module",
            description="Get a specific Python module by name",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Module name"}},
                "required": ["name"]
            },
            function=self._get_module
        )
        # Run a Python script file
        self.register_function(
            name="run_script",
            description="Run a Python script file and return the result",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the script"},
                    "args": {"type": "array", "items": {"type": "string"}, "description": "Command-line arguments", "default": []}
                },
                "required": ["file_path"]
            },
            function=self._run_script
        )
        # Code search
        self.register_function(
            name="search_code",
            description="Search for code artifacts by name or content",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results", "default": 10}
                },
                "required": ["query"]
            },
            function=self._search_code
        )
        # Weather functions
        self.register_function(
            name="get_weather",
            description="Get current weather information for a location",
            parameters={"type": "object", "properties": {"location": {"type": "string", "description": "Location (e.g., city)"}}, "required": ["location"]},
            function=self._get_weather
        )
        self.register_function(
            name="parse_weather_response",
            description="Parse weather data into a readable format",
            parameters={"type": "object", "properties": {"response": {"type": "string", "description": "Weather API response"}}, "required": ["response"]},
            function=self._parse_weather_response
        )
        # System prompt management
        self.register_function(
            name="update_system_prompt",
            description="Update the system prompt for the agent",
            parameters={"type": "object", "properties": {
                "system_prompt": {"type": "string", "description": "New system prompt"},
                "append": {"type": "boolean", "description": "Append rather than replace", "default": False}
            }, "required": ["system_prompt"]},
            function=self._update_system_prompt
        )
        self.register_function(
            name="get_system_prompt",
            description="Retrieve the current system prompt",
            parameters={"type": "object", "properties": {}},
            function=self._get_system_prompt
        )
        self.register_function(
            name="add_reflection_note",
            description="Add a self-reflection note",
            parameters={"type": "object", "properties": {
                "note": {"type": "string", "description": "Reflection content"},
                "category": {"type": "string", "description": "Note category", "default": "general"}
            }, "required": ["note"]},
            function=self._add_reflection_note
        )
        self.register_function(
            name="get_reflection_notes",
            description="Retrieve self-reflection notes",
            parameters={"type": "object", "properties": {"category": {"type": "string", "description": "Category filter", "default": "all"}}},
            function=self._get_reflection_notes
        )
        # Environmental adaptation
        self.register_function(
            name="analyze_environment",
            description="Analyze the current environment and context",
            parameters={"type": "object", "properties": {"aspect": {"type": "string", "description": "Aspect to analyze", "enum": ["system", "user_behavior", "conversation_context", "task_complexity", "all"], "default": "all"}}},
            function=self._analyze_environment
        )
        self.register_function(
            name="adapt_to_environment",
            description="Adapt agent behavior based on environmental analysis",
            parameters={"type": "object", "properties": {
                "adaptation_strategy": {"type": "string", "description": "Adaptation strategy"},
                "reason": {"type": "string", "description": "Reason for adaptation"},
                "system_prompt_update": {"type": "string", "description": "Optional prompt update"}
            }, "required": ["adaptation_strategy", "reason"]},
            function=self._adapt_to_environment
        )
        # File operations (read, write, list_directory, delete_file)
        self.register_function(
            name="install_package",
            description="Installs a Python package using pip.",
            parameters={"type": "object", "properties": {"package_name": {"type": "string", "description": "The name of the package to install."}}, "required": ["package_name"]},
            function=self._install_package
        )
        self.register_function(
            name="create_or_update_tool",
            description="Creates or updates a tool with the specified name, code, description, and parameters.",
            parameters={"type": "object", "properties": {
                "name": {"type": "string", "description": "The tool name."},
                "code": {"type": "string", "description": "The Python code for the tool."},
                "description": {"type": "string", "description": "A description of the tool."},
                "parameters": {"type": "object", "description": "A dictionary defining the parameters for the tool.", "additionalProperties": {"type": "object", "properties": {"type": {"type": "string", "description": "Data type of the parameter."}, "description": {"type": "string", "description": "Description of the parameter."}}, "required": ["type", "description"]}}
            }, "required": ["name", "code", "description", "parameters"]},
            function=self._create_or_update_tool
        )
        self.register_function(
            name="serialize_tool_result",
            description="Serializes the result of a tool call, truncating if necessary.",
            parameters={"type": "object", "properties": {"tool_result": {"type": "string", "description": "The result of the tool call."}, "max_length": {"type": "integer", "description": "Maximum length of the serialized result.", "default": 5000}}, "required": ["tool_result"]},
            function=self._serialize_tool_result
        )
        self.register_function(
            name="task_completed",
            description="Marks the current task as completed.",
            parameters={"type": "object", "properties": {}},
            function=self._task_completed
        )
        self.register_function(
            name="read_file",
            description="Read file contents",
            parameters={"type": "object", "properties": {"path": {"type": "string", "description": "File path"}}, "required": ["path"]},
            function=self._read_file
        )
        self.register_function(
            name="write_file",
            description="Write content to a file",
            parameters={"type": "object", "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content"},
                "append": {"type": "boolean", "description": "Append flag", "default": False}
            }, "required": ["path", "content"]},
            function=self._write_file
        )
        self.register_function(
            name="list_directory",
            description="List files/directories in a directory",
            parameters={"type": "object", "properties": {"path": {"type": "string", "description": "Directory path"}}, "required": ["path"]},
            function=self._list_directory
        )
        self.register_function(
            name="delete_file",
            description="Delete a file",
            parameters={"type": "object", "properties": {"path": {"type": "string", "description": "File path"}}, "required": ["path"]},
            function=self._delete_file
        )

        # Self-modification functions
        self.register_function(
            name="create_dynamic_function",
            description="Create a new function at runtime",
            parameters={"type": "object", "properties": {
                "name": {"type": "string", "description": "Name of the function to create"},
                "code": {"type": "string", "description": "Python code defining the function"},
                "description": {"type": "string", "description": "Description of the function"},
                "parameters": {"type": "object", "description": "Parameters schema in JSON Schema format"},
                "category": {"type": "string", "description": "Function category for organization", "default": "dynamic"}
            }, "required": ["name", "code", "description", "parameters"]},
            function=self._create_dynamic_function
        )
        self.register_function(
            name="list_dynamic_functions",
            description="List all dynamically created functions",
            parameters={"type": "object", "properties": {
                "category": {"type": "string", "description": "Filter by category", "default": "all"}
            }},
            function=self._list_dynamic_functions
        )

    # --- Tool Implementations ---
    # Note: These methods now accept an optional 'agent' parameter

    def _respond_to_user(self, message: str) -> Dict[str, Any]:
        """Simple tool to respond directly to the user with text"""
        return {
            "success": True,
            "response_sent": message
        }
        
    def _orchestrate_tasks(self, main_task: str, subtasks: List[str] = None, priority: int = 1, context: Dict[str, Any] = None, agent=None) -> Dict[str, Any]:
        """Orchestrate multiple parallel tasks using scout agents"""
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    # Check only for TogetherAgent or rely on duck typing (presence of agent_orchestrator)
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator'):
                return {"error": "Could not access Agent Orchestrator instance", "success": False}

            if context is None:
                context = {}
            
            # Add main task to context
            context["main_task"] = main_task
            
            # If subtasks not provided, create a single task for the main task
            if subtasks is None or not subtasks:
                subtasks = [f"Get {main_task}"]
            
            # Convert string to list if needed
            if isinstance(subtasks, str):
                try:
                    # Try to parse as JSON
                    if subtasks.startswith('[') and subtasks.endswith(']'):
                        subtasks = json.loads(subtasks)
                    else:
                        # Split by commas or create a single item list
                        subtasks = [s.strip() for s in subtasks.split(',') if s.strip()]
                except:
                    subtasks = [subtasks]
            
            start_time = time.time()
            results = agent.agent_orchestrator.execute_parallel_tasks(subtasks, context)
            execution_time = time.time() - start_time
            
            # Organize results
            successful_results = []
            failed_results = []
            
            for i, result in enumerate(results):
                task_result = {
                    "subtask": subtasks[i] if i < len(subtasks) else f"Task {i+1}",
                    "status": result["status"]
                }
                
                if result["status"] == "completed":
                    task_result["result"] = result.get("result", {}).get("solution", "")
                    task_result["chain_of_thought"] = result.get("result", {}).get("chain_of_thought", "")
                    task_result["scout_id"] = result.get("scout_id", "unknown")
                    successful_results.append(task_result)
                else:
                    task_result["error"] = result.get("error", "Unknown error")
                    failed_results.append(task_result)
            
            # Collect all chains of thought
            chains_of_thought = agent.agent_orchestrator.get_all_chains_of_thought()
            recent_thoughts = sorted(chains_of_thought, key=lambda x: x["timestamp"], reverse=True)[:min(len(chains_of_thought), 3)]
            
            return {
                "success": len(failed_results) == 0,
                "main_task": main_task,
                "total_subtasks": len(subtasks),
                "completed_subtasks": len(successful_results),
                "failed_subtasks": len(failed_results),
                "execution_time": execution_time,
                "successful_results": successful_results,
                "failed_results": failed_results,
                "recent_chains_of_thought": recent_thoughts
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _assign_specialized_task(self, task: str, specialization: str, context: Dict[str, Any] = None, location: str = None, agent=None) -> Dict[str, Any]:
        """Assign a task to a specific type of specialized scout agent"""
        # Handle special case for weather tasks
        if "weather" in task.lower() and location:
            # Redirect to weather function
            return self._get_weather(location=location)
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator'):
                return {"error": "Could not access Agent Orchestrator instance", "success": False}

            if context is None:
                context = {}
            
            # Add specialization to context
            context["specialization"] = specialization
            
            # Find suitable scout agent
            suitable_scout = None
            for scout_id, scout in agent.agent_orchestrator.scouts.items():
                if scout.specialization == specialization:
                    suitable_scout = scout
                    break
            
            if not suitable_scout:
                return {"error": f"No scout agent with specialization '{specialization}' found", "success": False}
            
            # Assign task
            task_id = agent.agent_orchestrator.add_task(task, context=context)
            
            # Wait for completion
            start_time = time.time()
            max_wait_time = 60  # Maximum wait time in seconds
            
            while True:
                result = agent.agent_orchestrator.get_task_result(task_id)
                if result["status"] in ["completed", "failed"]:
                    break
                    
                if time.time() - start_time > max_wait_time:
                    return {"error": "Timed out waiting for scout agent to complete task", "success": False}
                
                time.sleep(0.5)
            
            execution_time = time.time() - start_time
            
            if result["status"] == "completed":
                return {
                    "success": True,
                    "task": task,
                    "specialization": specialization,
                    "execution_time": execution_time,
                    "scout_id": result.get("scout_id", "unknown"),
                    "solution": result.get("result", {}).get("solution", ""),
                    "chain_of_thought": result.get("result", {}).get("chain_of_thought", "")
                }
            else:
                return {
                    "success": False,
                    "task": task,
                    "specialization": specialization,
                    "error": result.get("error", "Unknown error"),
                    "execution_time": execution_time
                }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_scout_status(self, agent=None) -> Dict[str, Any]:
        """Get status of all scout agents"""
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator'):
                return {"error": "Could not access Agent Orchestrator instance", "success": False}

            scout_statuses = []
            for scout_id, scout in agent.agent_orchestrator.scouts.items():
                scout_statuses.append({
                    "scout_id": scout_id,
                    "specialization": scout.specialization,
                    "status": scout.status,
                    "is_available": scout.is_available.is_set(),
                    "pending_tasks": scout.task_queue.qsize(),
                    "completed_chains": len(scout.chains_of_thought),
                    "completed_rollouts": len(scout.rollouts)
                })
            
            return {
                "success": True,
                "scout_count": len(scout_statuses),
                "scouts": scout_statuses,
                "busy_scouts": len([s for s in scout_statuses if not s["is_available"]]),
                "available_scouts": len([s for s in scout_statuses if s["is_available"]])
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_solution_rollouts(self, agent=None) -> Dict[str, Any]:
        """Get all solution rollouts from scout agents"""
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator'):
                return {"error": "Could not access Agent Orchestrator instance", "success": False}

            all_rollouts = agent.agent_orchestrator.get_all_rollouts()
            
            # Group by scout agent
            scout_rollouts = {}
            for rollout in all_rollouts:
                scout_id = rollout.get("scout_id", "unknown")
                if scout_id not in scout_rollouts:
                    scout_rollouts[scout_id] = []
                scout_rollouts[scout_id].append(rollout)
            
            # Get most recent rollouts
            recent_rollouts = all_rollouts[:min(len(all_rollouts), 5)]
            
            return {
                "success": True,
                "total_rollouts": len(all_rollouts),
                "scouts_with_rollouts": len(scout_rollouts),
                "recent_rollouts": recent_rollouts,
                "rollouts_by_scout": scout_rollouts
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _extract_code(self, text: str, language: str = "", agent=None) -> Dict[str, Any]:
        """Extract code blocks from text using structured output parsing"""
        try:
            # Import the CodeExtractor class
            from code_extractor import CodeExtractor
            
            extractor = CodeExtractor()
            blocks = extractor.extract_code_blocks(text)
            
            # Filter by language if specified
            if language:
                blocks = [block for block in blocks if block.language.lower() == language.lower()]
            
            # Format the results
            formatted_blocks = []
            for i, block in enumerate(blocks):
                formatted_blocks.append({
                    "id": i+1,
                    "language": block.language,
                    "code": block.code,
                    "line_count": block.line_count,
                    "start_line": block.start_line,
                    "end_line": block.end_line
                })
            
            return {
                "success": True,
                "total_blocks": len(blocks),
                "blocks": formatted_blocks
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _extract_and_execute_code(self, text: str, agent=None) -> Dict[str, Any]:
        """Extract code blocks from text and execute them"""
        try:
            # Import the CodeExtractor class
            from code_extractor import CodeExtractor
            
            extractor = CodeExtractor()
            result = extractor.extract_and_execute(text)
            
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _optimize_agent_team(self, agent=None) -> Dict[str, Any]:
        """Optimize the agent team structure based on performance history"""
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator'):
                return {"error": "Could not access Agent Orchestrator instance", "success": False}

            result = agent.agent_orchestrator.optimize_team_structure()
            result["success"] = True
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_agent_performance(self, agent=None) -> Dict[str, Any]:
        """Get performance metrics for all agents"""
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator'):
                return {"error": "Could not access Agent Orchestrator instance", "success": False}

            metrics = agent.agent_orchestrator.get_agent_performance_metrics()
            
            # Calculate overall statistics
            total_tasks = sum(m["tasks_completed"] for m in metrics.values())
            avg_success_rate = sum(m["success_rate"] for m in metrics.values()) / len(metrics) if metrics else 0
            avg_execution_time = sum(m["avg_execution_time"] for m in metrics.values()) / len(metrics) if metrics else 0
            
            # Count skills across agents
            skill_counts = {}
            for agent_metrics in metrics.values():
                for skill in agent_metrics["skills"]:
                    skill_counts[skill] = skill_counts.get(skill, 0) + 1
            
            return {
                "success": True,
                "agent_metrics": metrics,
                "overall_stats": {
                    "total_agents": len(metrics),
                    "total_tasks_completed": total_tasks,
                    "avg_success_rate": avg_success_rate,
                    "avg_execution_time": avg_execution_time,
                    "skill_distribution": skill_counts
                }
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_collaboration_network(self, agent=None) -> Dict[str, Any]:
        """Get the collaboration network between agents"""
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator'):
                return {"error": "Could not access Agent Orchestrator instance", "success": False}

            network = agent.agent_orchestrator.get_collaboration_network()
            network["success"] = True
            return network
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _search_agent_memory(self, query: str, limit: int = 5, agent=None) -> Dict[str, Any]:
        """Search the agent's semantic memory"""
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator') or not hasattr(agent.agent_orchestrator, 'memory'):
                 return {"error": "Could not access Agent Orchestrator memory", "success": False}

            # Search both orchestrator memory and individual agent memories
            orchestrator_results = agent.agent_orchestrator.memory.search_memory(query, limit)
            
            # Also search individual agent memories
            agent_results = {}
            for agent_id, scout in agent.agent_orchestrator.scouts.items():
                scout_results = scout.memory.search_memory(query, limit=3)
                if scout_results:
                    agent_results[agent_id] = scout_results
            
            return {
                "success": True,
                "query": query,
                "orchestrator_results": orchestrator_results,
                "agent_results": agent_results,
                "total_results": len(orchestrator_results) + sum(len(results) for results in agent_results.values())
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _add_to_memory(self, content: str, category: str = "general", agent=None) -> Dict[str, Any]:
        """Add an item to the agent's semantic memory"""
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator') or not hasattr(agent.agent_orchestrator, 'memory'):
                 return {"error": "Could not access Agent Orchestrator memory", "success": False}

            metadata = {
                "category": category,
                "source": "user_input",
                "timestamp": time.time()
            }
            
            memory_id = agent.agent_orchestrator.memory.add_memory(content, metadata)
            
            return {
                "success": True,
                "memory_id": memory_id,
                "content": content,
                "category": category,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_memory_summary(self, category: str = "", agent=None) -> Dict[str, Any]:
        """Get a summary of the agent's memory"""
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator') or not hasattr(agent.agent_orchestrator, 'memory'):
                 return {"error": "Could not access Agent Orchestrator memory", "success": False}

            orchestrator_summary = agent.agent_orchestrator.memory.summarize_memories(category)
            
            # Also get summaries from individual agents
            agent_summaries = {}
            for agent_id, scout in agent.agent_orchestrator.scouts.items():
                agent_summaries[agent_id] = scout.memory.summarize_memories(category)
            
            return {
                "success": True,
                "orchestrator_memory": orchestrator_summary,
                "agent_memories": agent_summaries,
                "total_memories": orchestrator_summary.get("count", 0) + sum(
                    summary.get("count", 0) for summary in agent_summaries.values()
                )
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_chains_of_thought(self, agent=None) -> Dict[str, Any]:
        """Get all chains of thought from scout agents"""
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator'):
                return {"error": "Could not access Agent Orchestrator instance", "success": False}

            all_thoughts = agent.agent_orchestrator.get_all_chains_of_thought()
            
            # Group by scout agent
            scout_thoughts = {}
            for thought in all_thoughts:
                scout_id = thought["agent_id"]
                if scout_id not in scout_thoughts:
                    scout_thoughts[scout_id] = []
                scout_thoughts[scout_id].append(thought)
            
            # Get most recent thoughts
            recent_thoughts = sorted(all_thoughts, key=lambda x: x["timestamp"], reverse=True)[:min(len(all_thoughts), 5)]
            
            return {
                "success": True,
                "total_chains": len(all_thoughts),
                "scouts_with_chains": len(scout_thoughts),
                "recent_chains": recent_thoughts,
                "chains_by_scout": scout_thoughts
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _generate_solution_rollouts(self, task: str, num_rollouts: int = 3, specialization: str = "", agent=None) -> Dict[str, Any]:
        """Generate multiple solution approaches (rollouts) for a given task"""
        try:
            # Use passed agent instance if available, otherwise try inspect.stack()
            if agent is None:
                for frame in inspect.stack():
                    if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                        agent = frame.frame.f_locals['self']
                        break

            if not agent or not hasattr(agent, 'agent_orchestrator'):
                return {"error": "Could not access Agent Orchestrator instance", "success": False}

            # Set the maximum number of rollouts
            for scout_id, scout in agent.agent_orchestrator.scouts.items():
                scout.max_rollouts = min(num_rollouts, 5)  # Cap at 5 to prevent excessive API calls
                
            # Find a suitable scout agent based on specialization
            suitable_scout = None
            if specialization:
                for scout_id, scout in agent.agent_orchestrator.scouts.items():
                    if scout.specialization == specialization and scout.is_available.is_set():
                        suitable_scout = scout
                        break
            
            # If no specific specialization or no suitable scout found, use any available scout
            if not suitable_scout:
                for scout_id, scout in agent.agent_orchestrator.scouts.items():
                    if scout.is_available.is_set():
                        suitable_scout = scout
                        break
            
            if not suitable_scout:
                return {"error": "No available scout agents to perform the task", "success": False}
            
            # Assign the task to the selected scout
            task_id = suitable_scout.add_task(suitable_scout.perform_task, task)
            
            # Wait for completion
            start_time = time.time()
            max_wait_time = 120  # Maximum wait time in seconds (longer for multiple rollouts)
            
            while True:
                result = suitable_scout.get_result(task_id)
                if result["status"] in ["completed", "failed"]:
                    break
                    
                if time.time() - start_time > max_wait_time:
                    return {"error": "Timed out waiting for scout agent to complete rollouts", "success": False}
                
                time.sleep(0.5)
            
            if result["status"] == "failed":
                return {"error": result.get("error", "Unknown error during rollout generation"), "success": False}
            
            # Extract rollouts from the result
            rollouts = result.get("result", {}).get("rollouts", [])
            best_rollout = result.get("result", {}).get("best_rollout", {})
            
            # Format the response
            formatted_rollouts = []
            for rollout in rollouts:
                formatted_rollouts.append({
                    "rollout_id": rollout.get("rollout_id", 0),
                    "solution": rollout.get("solution", ""),
                    "evaluation": rollout.get("evaluation", "")
                })
            
            return {
                "success": True,
                "task": task,
                "num_rollouts": len(formatted_rollouts),
                "rollouts": formatted_rollouts,
                "best_rollout_id": best_rollout.get("rollout_id", 1) if best_rollout else 1,
                "best_solution": best_rollout.get("solution", "") if best_rollout else "",
                "comparison": best_rollout.get("comparison", "") if best_rollout else "",
                "scout_id": suitable_scout.agent_id,
                "specialization": suitable_scout.specialization,
                "execution_time": time.time() - start_time
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _add_numbers(self, a: float, b: float) -> Dict[str, Any]:
        return {"result": a + b, "success": True}

    def _subtract_numbers(self, a: float, b: float) -> Dict[str, Any]:
        return {"result": a - b, "success": True}

    def _multiply_numbers(self, a: float, b: float) -> Dict[str, Any]:
        return {"result": a * b, "success": True}
        
    def _divide_numbers(self, a: float, b: float) -> Dict[str, Any]:
        if b == 0:
            return {"error": "Division by zero", "success": False}
        return {"result": a / b, "success": True}

    def _decompose_prompt(self, transcript: str) -> Dict[str, Any]:
        # Call the LLM with the JSON schema for MultiPrompt
        extract = self.together.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You will decompose a prompt into a specific structure that will be provided. Only answer in JSON.",
                },
                {
                    "role": "user",
                    "content": transcript,
                },
            ],
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            response_format={
                "type": "json_object",
                "schema": MultiPrompt.model_json_schema(),
            },
        )

        output = json.loads(extract.choices[0].message.content)
        print(json.dumps(output, indent=2))
        return output

    def _delete_file(self, path: str) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser()
            if not path.exists():
                return {"error": f"File '{path}' does not exist", "success": False}
            path.unlink()
            return {"success": True, "message": f"File '{path}' deleted successfully"}
        except Exception as e:
            return {"error": str(e), "success": False}
        self.register_function(
            name="execute_command",
            description="Execute a shell command",
            parameters={"type": "object", "properties": {"command": {"type": "string", "description": "Shell command"}}, "required": ["command"]},
            function=self._execute_command
        )
        # Function creation
        self.register_function(
            name="create_python_function",
            description="Create a new Python function for later calls",
            parameters={"type": "object", "properties": {
                "name": {"type": "string", "description": "Function name"},
                "description": {"type": "string", "description": "Function description"},
                "parameters_schema": {"type": "object", "description": "JSON Schema for parameters"},
                "source_code": {"type": "string", "description": "Source code"}
            }, "required": ["name", "description", "parameters_schema", "source_code"]},
            function=self._create_python_function
        )
        self.register_function(
            name="list_available_functions",
            description="List all available functions",
            parameters={"type": "object", "properties": {}},
            function=self._list_available_functions
        )
        # Together API models & completions
        self.register_function(
            name="list_together_models",
            description="List models available on Together API",
            parameters={"type": "object", "properties": {"filter": {"type": "string", "description": "Filter", "default": ""}}},
            function=self._list_together_models
        )
        self.register_function(
            name="generate_completion",
            description="Generate a text completion using Together API",
            parameters={"type": "object", "properties": {
                "model": {"type": "string", "description": "Model to use"},
                "prompt": {"type": "string", "description": "Prompt"},
                "max_tokens": {"type": "integer", "description": "Max tokens", "default": 256},
                "temperature": {"type": "number", "description": "Temperature", "default": 0.7},
                "logprobs": {"type": "integer", "description": "Number of logprobs", "default": 0},
                "echo": {"type": "boolean", "description": "Echo prompt tokens", "default": False}
            }, "required": ["model", "prompt"]},
            function=self._generate_completion
        )
        # Create/update assistant
        self.register_function(
            name="create_or_update_assistant",
            description="Create or update an assistant in Together platform",
            parameters={"type": "object", "properties": {
                "assistant_id": {"type": "string", "description": "Assistant ID (if updating)"},
                "name": {"type": "string", "description": "Assistant name"},
                "description": {"type": "string", "description": "Assistant description"},
                "model": {"type": "string", "description": "Model to use"},
                "system_prompt": {"type": "string", "description": "System prompt"}
            }, "required": ["name", "model"]},
            function=self._create_or_update_assistant
        )
        # Create thread
        self.register_function(
            name="create_thread",
            description="Create a new conversation thread",
            parameters={"type": "object", "properties": {"metadata": {"type": "object", "description": "Optional metadata"}}},
            function=self._create_thread
        )
        # Add message to thread
        self.register_function(
            name="add_message_to_thread",
            description="Add a message to an existing thread",
            parameters={"type": "object", "properties": {
                "thread_id": {"type": "string", "description": "Thread ID"},
                "role": {"type": "string", "description": "Sender role", "enum": ["user", "assistant"]},
                "content": {"type": "string", "description": "Message content"}
            }, "required": ["thread_id", "role", "content"]},
            function=self._add_message_to_thread
        )
        # Run assistant on thread
        self.register_function(
            name="run_assistant",
            description="Run an assistant on a thread to generate a response",
            parameters={"type": "object", "properties": {
                "assistant_id": {"type": "string", "description": "Assistant ID"},
                "thread_id": {"type": "string", "description": "Thread ID"}
            }, "required": ["assistant_id", "thread_id"]},
            function=self._run_assistant
        )
        # Jina tools
        # Register the new search, read, fact_check, and weather functions
        self.register_function(
            name="search",
            description="Search the web for information",
            parameters={
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                }, 
                "required": ["query"]
            },
            function=self.search
        )
        
        self.register_function(
            name="search_similar_images",
            description="Search for similar images in the conversation history",
            parameters={
                "type": "object", 
                "properties": {
                    "image_url": {"type": "string", "description": "URL of the image to find similar images for"},
                    "limit": {"type": "integer", "description": "Maximum number of results to return", "default": 3}
                }, 
                "required": ["image_url"]
            },
            function=self._search_similar_images
        )
        
        # Register a more robust weather function with multiple parameter options
        self.register_function(
            name="get_weather",
            description="Get current weather information for a location",
            parameters={
                "type": "object", 
                "properties": {
                    "location": {"type": "string", "description": "Location (e.g., city name)"},
                    "city": {"type": "string", "description": "City name (alternative to location)"}
                }, 
                "required": []
            },
            function=self._get_weather
        )
        
        # Register a weather_api alias for compatibility
        self.register_function(
            name="weather_api",
            description="Get weather information for a location (alias for get_weather)",
            parameters={
                "type": "object", 
                "properties": {
                    "location": {"type": "string", "description": "Location (e.g., city name)"},
                    "city": {"type": "string", "description": "City name (alternative to location)"}
                }, 
                "required": []
            },
            function=self._get_weather
        )
        
        # Register a function for getting weather in multiple locations
        self.register_function(
            name="get_multiple_weather",
            description="Get current weather information for multiple locations",
            parameters={
                "type": "object", 
                "properties": {
                    "locations": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "List of locations (e.g., city names)"
                    }
                }, 
                "required": ["locations"]
            },
            function=self._get_multiple_weather
        )
        
        self.register_function(
            name="read",
            description="Read and extract content from a web page",
            parameters={
                "type": "object", 
                "properties": {
                    "url": {"type": "string", "description": "URL of the web page to read"}
                }, 
                "required": ["url"]
            },
            function=self.read
        )
        
        self.register_function(
            name="fact_check",
            description="Verify a statement or claim for factual accuracy",
            parameters={
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "Statement to fact check"}
                }, 
                "required": ["query"]
            },
            function=self.fact_check
        )
        
        # Keep the legacy functions for backward compatibility
        if self.jina_client:
            self.register_function(
                name="web_search",
                description="Search the web using Jina's search API",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                },
                function=self._web_search
            )
            self.register_function(
                name="web_read",
                description="Read web page content using Jina's reader API",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to read content from"
                        }
                    },
                    "required": ["url"]
                },
                function=self._web_read
            )
            self.register_function(
                name="fact_check",
                description="Verify a statement using Jina's fact checking API",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The statement to fact check"
                        }
                    },
                    "required": ["query"]
                },
                function=self._fact_check
            )
        # Planning tools
        self.register_function(
            name="create_planning_session",
            description="Start a new planning session for multi-turn tasks",
            parameters={"type": "object", "properties": {"task": {"type": "string", "description": "Task description"}}, "required": ["task"]},
            function=self._create_planning_session
        )
        self.register_function(
            name="add_plan_step",
            description="Add a planning step with optional tool execution",
            parameters={"type": "object", "properties": {
                "plan": {"type": "string", "description": "Step description"},
                "tool_name": {"type": "string", "description": "Tool name (optional)"},
                "tool_args": {"type": "object", "description": "Tool arguments (optional)"}
            }, "required": ["plan"]},
            function=self._add_plan_step
        )
        self.register_function(
            name="get_planning_status",
            description="Get the status of the current planning session",
            parameters={"type": "object", "properties": {}},
            function=self._get_planning_status
        )
        self.register_function(
            name="complete_planning_session",
            description="Complete the current planning session with a summary",
            parameters={"type": "object", "properties": {
                "summary": {"type": "string", "description": "Session summary"},
                "success": {"type": "boolean", "description": "Was the session successful", "default": True}
            }, "required": ["summary"]},
            function=self._complete_planning_session
        )
        # Knowledge tools
        self.register_function(
            name="extract_urls",
            description="Extract URLs from text",
            parameters={"type": "object", "properties": {"text": {"type": "string", "description": "Text input"}}, "required": ["text"]},
            function=self._extract_urls
        )
        self.register_function(
            name="process_urls",
            description="Process URLs to extract knowledge",
            parameters={"type": "object", "properties": {"urls": {"type": "array", "items": {"type": "string"}, "description": "List of URLs"}}, "required": ["urls"]},
            function=self._process_urls
        )
        self.register_function(
            name="get_knowledge_summary",
            description="Get a summary of the knowledge base",
            parameters={"type": "object", "properties": {}},
            function=self._get_knowledge_summary
        )
        self.register_function(
            name="search_knowledge",
            description="Search the knowledge base for relevant information",
            parameters={"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]},
            function=self._search_knowledge
        )
        
        self.register_function(
            name="search_conversation_history",
            description="Search the conversation history for relevant content",
            parameters={
                "type": "object", 
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum number of results to return", "default": 5},
                    "include_images": {"type": "boolean", "description": "Whether to include images in the search", "default": True}
                }, 
                "required": ["query"]
            },
            function=self._search_conversation_history
        )
        self.register_function(
            name="monitor_task",
            description="Monitor the status of a background task",
            parameters={"type": "object", "properties": {"task_id": {"type": "string", "description": "Task ID"}}, "required": ["task_id"]},
            function=self._monitor_task
        )

    def _categorize_function(self, name: str) -> str:
        """Categorize functions based on their names or purposes"""
        web_tools = ["web_search", "web_read", "fact_check", "extract_urls", "process_urls"]
        code_tools = ["execute_python", "save_module", "execute_module", "list_modules", "get_module", 
                     "run_script", "search_code", "create_python_function"]
        file_tools = ["read_file", "write_file", "list_directory"]
        system_tools = ["execute_command", "update_system_prompt", "get_system_prompt", 
                       "add_reflection_note", "get_reflection_notes"]
        planning_tools = ["create_planning_session", "add_plan_step", "get_planning_status", 
                         "complete_planning_session"]
        knowledge_tools = ["get_knowledge_summary", "search_knowledge", "monitor_task"]
        weather_tools = ["get_weather", "parse_weather_response"]
        together_tools = ["list_together_models", "generate_completion", "create_or_update_assistant",
                         "create_thread", "add_message_to_thread", "run_assistant"]
        
        if name in web_tools:
            return "Web & Search"
        elif name in code_tools:
            return "Code & Development"
        elif name in file_tools:
            return "File Operations"
        elif name in system_tools:
            return "System & Configuration"
        elif name in planning_tools:
            return "Planning & Task Management"
        elif name in knowledge_tools:
            return "Knowledge Management"
        elif name in weather_tools:
            return "Weather"
        elif name in together_tools:
            return "Together API"
        elif "analyze" in name or "adapt" in name:
            return "Environment & Adaptation"
        else:
            return "Miscellaneous"
    
    def register_function(self, name: str, description: str, parameters: Dict[str, Any], function: Callable, source_code: Optional[str] = None):
        if name in self.functions:
            console.print(f"[yellow]Warning: Overwriting existing function '{name}'[/yellow]")
        self.functions[name] = FunctionSpec(name=name, description=description, parameters=parameters, function=function, source_code=source_code)

    def get_openai_tools_format(self) -> List[Dict[str, Any]]:
        tools = []
        for name, spec in self.functions.items():
            tools.append({
                "type": "function",
                "function": {"name": spec.name, "description": spec.description, "parameters": spec.parameters}
            })
        return tools

    def call_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self.functions:
            return {"error": f"Function '{name}' not found in registry"}
        try:
            return self.functions[name].function(**arguments)
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}

    # Extra helper methods to support parallel execution
    def has_tool(self, name: str) -> bool:
        return name in self.functions

    def get_tool(self, name: str):
        return self.functions[name].function if name in self.functions else None
        
    def get_available_tools(self) -> List[str]:
        """Return a list of all available tool names"""
        return list(self.functions.keys())

    # ===============================
    # Default tool implementations
    # ===============================
    def _read_file(self, path: str) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser()
            if not path.exists():
                return {"error": f"File '{path}' does not exist"}
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            return {"content": content, "size_bytes": path.stat().st_size, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _write_file(self, path: str, content: str, append: bool = False) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as file:
                file.write(content)
            return {"path": str(path), "size_bytes": path.stat().st_size, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _list_directory(self, path: str) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser()
            if not path.exists():
                return {"error": f"Path '{path}' does not exist"}
            if not path.is_dir():
                return {"error": f"Path '{path}' is not a directory"}
            items = []
            for item in path.iterdir():
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size_bytes": item.stat().st_size if item.is_file() else None,
                    "last_modified": time.ctime(item.stat().st_mtime)
                })
            return {"path": str(path), "items": items, "count": len(items), "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _install_package(self, package_name: str) -> Dict[str, Any]:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return {"success": True, "message": f"Package '{package_name}' installed successfully."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_or_update_tool(self, name: str, code: str, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            exec(code, globals())
            self.register_function(name, globals()[name], description, parameters)
            return {"success": True, "message": f"Tool '{name}' created/updated successfully."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _serialize_tool_result(self, tool_result: Any, max_length: int = 5000) -> str:
        try:
            serialized_result = json.dumps(tool_result)
        except TypeError:
            serialized_result = str(tool_result)
        if len(serialized_result) > max_length:
            return serialized_result[:max_length] + f"\n\n(Note: Result was truncated to {max_length} characters out of {len(serialized_result)} total characters.)"
        else:
            return serialized_result

    def _task_completed(self) -> str:
        return "Task marked as completed."

    def _create_python_function(self, name: str, description: str, parameters_schema: Dict[str, Any], source_code: str) -> Dict[str, Any]:
        try:
            temp_module_path = Path(tempfile.gettempdir()) / f"dynamic_func_{name}_{int(time.time())}.py"
            with open(temp_module_path, 'w', encoding='utf-8') as f:
                f.write(source_code)
            spec = importlib.util.spec_from_file_location(f"dynamic_func_{name}", temp_module_path)
            if spec is None or spec.loader is None:
                return {"error": "Failed to create module specification", "success": False}
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if name not in dir(module):
                return {"error": f"Function '{name}' not found in the provided source code", "success": False}
            function = getattr(module, name)
            if not callable(function):
                return {"error": f"'{name}' is not a callable function", "success": False}
            self.register_function(name=name, description=description, parameters=parameters_schema, function=function, source_code=source_code)
            return {"name": name, "description": description, "success": True, "message": f"Function '{name}' successfully created and registered"}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _delete_file(self, path: str) -> Dict[str, Any]:
        try:
            path = Path(path).expanduser()
            if not path.exists():
                return {"error": f"File '{path}' does not exist"}
            if not path.is_file():
                return {"error": f"'{path}' is not a file"}
            path.unlink()
            return {"success": True, "message": f"File '{path}' deleted"}
        except Exception as e:
            return {"error": str(e), "success": False}
            
    def _create_dynamic_function(self, name: str, code: str, description: str, parameters: Dict[str, Any], category: str = "dynamic") -> Dict[str, Any]:
        """Create a new function at runtime and register it with the tool registry
        
        This allows for dynamic expansion of the agent's capabilities during operation.
        """
        try:
            # Safety checks
            if self.self_modification["safety_checks"]:
                # Check for potentially harmful operations
                unsafe_patterns = [
                    "os.system", "subprocess.call", "subprocess.Popen", 
                    "eval(", "exec(", "__import__", "open(", "shutil.rmtree"
                ]
                for pattern in unsafe_patterns:
                    if pattern in code:
                        return {
                            "success": False, 
                            "error": f"Unsafe code pattern detected: {pattern}",
                            "recommendation": "Remove potentially dangerous operations and try again"
                        }
            
            # Create function namespace and compile code
            namespace = {}
            exec(code, globals(), namespace)
            
            # Find the function in the namespace (assuming it's defined at the top level)
            if name not in namespace:
                return {
                    "success": False, 
                    "error": f"Function '{name}' not found in the provided code",
                    "recommendation": "Ensure the function name matches the name defined in the code"
                }
            
            function = namespace[name]
            
            # Register the function in our dynamic registry
            self.dynamic_functions[name] = {
                "function": function,
                "description": description,
                "parameters": parameters,
                "code": code,
                "category": category,
                "created_at": time.time()
            }
            
            # Also register it with the tool registry
            self.tool_registry.register_function(
                name=name,
                description=description,
                parameters=parameters,
                function=function,
                source_code=code
            )
            
            # Update modification history
            self.self_modification["modification_history"].append({
                "type": "function_creation",
                "name": name,
                "timestamp": time.time(),
                "category": category
            })
            self.self_modification["modification_count"] += 1
            self.self_modification["last_modification_time"] = time.time()
            
            return {
                "success": True,
                "message": f"Function '{name}' successfully created and registered",
                "function_name": name,
                "category": category
            }
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
    def _list_dynamic_functions(self, category: str = "all") -> Dict[str, Any]:
        """List all dynamically created functions, optionally filtered by category"""
        try:
            results = []
            
            for name, function_data in self.dynamic_functions.items():
                if category == "all" or function_data["category"] == category:
                    results.append({
                        "name": name,
                        "description": function_data["description"],
                        "category": function_data["category"],
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(function_data["created_at"]))
                    })
            
            return {
                "success": True,
                "functions": results,
                "count": len(results),
                "category_filter": category
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    def _list_available_functions(self) -> Dict[str, Any]:
        function_list = []
        for name, spec in self.functions.items():
            # Extract required parameters for easier reference
            required_params = spec.parameters.get("required", [])
            properties = spec.parameters.get("properties", {})
            
            # Create a simplified parameter description
            param_desc = []
            for param_name, param_info in properties.items():
                is_required = param_name in required_params
                param_type = param_info.get("type", "any")
                description = param_info.get("description", "")
                default = f", default: {param_info.get('default')}" if "default" in param_info else ""
                req_marker = "*" if is_required else ""
                param_desc.append(f"{param_name}{req_marker} ({param_type}{default}): {description}")
            
            # Create a simplified function description
            simplified_desc = {
                "name": name,
                "description": spec.description,
                "parameters": param_desc,
                "has_source_code": spec.source_code is not None,
                "category": self._categorize_function(name)
            }
            function_list.append(simplified_desc)
        
        # Group functions by category
        categories = {}
        for func in function_list:
            category = func["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(func)
        
        return {
            "functions": function_list, 
            "count": len(function_list), 
            "categories": categories,
            "success": True
        }

    def _list_together_models(self, filter: str = "") -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            models_data = agent.client.models.list()
            models = []
            for model in models_data.data:
                model_dict = {"id": model.id, "name": model.name, "context_length": model.context_length, "capabilities": model.capabilities}
                if not filter or filter.lower() in model.name.lower():
                    models.append(model_dict)
            return {"models": models, "count": len(models), "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _generate_completion(self, model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.7, logprobs: int = 0, echo: bool = False) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            params = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
            agent_logprobs_enabled = False
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    agent_logprobs_enabled = getattr(agent, 'enable_logprobs', False)
                    break
            if (logprobs > 0 or agent_logprobs_enabled) and logprobs >= 0:
                actual_logprobs = logprobs if logprobs > 0 else 1
                params["logprobs"] = actual_logprobs
                if echo:
                    params["echo"] = echo
            response = agent.client.completions.create(**params)
            result = {
                "model": model,
                "completion": response.choices[0].text,
                "finish_reason": response.choices[0].finish_reason,
                "tokens_used": {"prompt": response.usage.prompt_tokens, "completion": response.usage.completion_tokens, "total": response.usage.total_tokens},
                "success": True
            }
            if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                logprobs_data = response.choices[0].logprobs
                result["logprobs"] = {"tokens": logprobs_data.tokens, "token_logprobs": logprobs_data.token_logprobs}
                if hasattr(logprobs_data, "top_logprobs") and logprobs_data.top_logprobs:
                    result["logprobs"]["top_logprobs"] = logprobs_data.top_logprobs
            if echo and hasattr(response, "prompt"):
                result["prompt_tokens"] = []
                for prompt_item in response.prompt:
                    if hasattr(prompt_item, "logprobs") and prompt_item.logprobs:
                        result["prompt_tokens"].append({
                            "text": prompt_item.text,
                            "tokens": prompt_item.logprobs.tokens,
                            "token_logprobs": prompt_item.logprobs.token_logprobs
                        })
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _create_or_update_assistant(self, name: str, model: str, assistant_id: str = None, description: str = None, system_prompt: str = None) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            if not assistant_id:
                assistant_id = f"asst_{int(time.time())}"
                action = "Created"
            else:
                action = "Updated"
            if not hasattr(agent, 'assistants'):
                agent.assistants = {}
            agent.assistants[assistant_id] = {
                "id": assistant_id,
                "name": name,
                "model": model,
                "description": description or "",
                "system_prompt": system_prompt or "",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            return {"assistant_id": assistant_id, "name": name, "model": model, "description": description or "", "action": action, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _create_pydantic_model(self, model_name: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from pydantic import create_model
            model = create_model(model_name, **fields)
            globals()[model_name] = model
            return {"success": True, "model_name": model_name, "fields": fields}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            thread_id = f"thread_{int(time.time())}"
            return {"thread_id": thread_id, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "metadata": {}, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _add_message_to_thread(self, thread_id: str, role: str, content: str) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            if thread_id.startswith("thread_"):
                message_id = f"msg_{int(time.time())}_{hash(content) % 10000}"
                agent.add_message(role, content)
                return {"message_id": message_id, "thread_id": thread_id, "role": role, "content": content, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "success": True}
            else:
                return {"error": "Invalid thread ID format", "success": False}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_system_prompt(self) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            return {"system_prompt": agent.system_message["content"], "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _add_reflection_note(self, note: str, category: str = "general") -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            reflection_note = {"timestamp": time.time(), "datetime": time.strftime("%Y-%m-%d %H:%M:%S"), "category": category, "note": note}
            agent.reflection_notes.append(reflection_note)
            return {"note_id": len(agent.reflection_notes) - 1, "timestamp": reflection_note["datetime"], "category": category, "note": note, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_reflection_notes(self, category: str = "all") -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            if category.lower() == "all":
                filtered_notes = agent.reflection_notes
            else:
                filtered_notes = [note for note in agent.reflection_notes if note["category"].lower() == category.lower()]
            formatted_notes = []
            for i, note in enumerate(filtered_notes):
                formatted_notes.append({"id": i, "timestamp": note["datetime"], "category": note["category"], "note": note["note"]})
            return {"notes": formatted_notes, "count": len(formatted_notes), "category_filter": category, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _analyze_environment(self, aspect: str = "all") -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            agent.environment_state["last_analysis_time"] = time.time()
            if aspect in ["conversation_context", "all"] and len(agent.conversation_history) > 2:
                user_messages = [msg["content"] for msg in agent.conversation_history if msg.get("role") == "user"]
                positive_words = ["good", "great", "excellent", "amazing", "helpful", "thanks", "thank", "appreciate"]
                negative_words = ["bad", "wrong", "incorrect", "error", "issue", "problem", "not", "doesn't", "don't"]
                positive_count = sum(sum(1 for word in positive_words if word.lower() in msg.lower()) for msg in user_messages)
                negative_count = sum(sum(1 for word in negative_words if word.lower() in msg.lower()) for msg in user_messages)
                if positive_count > negative_count * 2:
                    agent.environment_state["conversation_context"]["sentiment"] = "positive"
                elif negative_count > positive_count:
                    agent.environment_state["conversation_context"]["sentiment"] = "negative"
                else:
                    agent.environment_state["conversation_context"]["sentiment"] = "neutral"
                task_words = ["do", "make", "create", "implement", "build", "fix", "solve", "how", "help"]
                task_count = sum(sum(1 for word in task_words if word.lower() in msg.lower()) for msg in user_messages)
                agent.environment_state["conversation_context"]["task_oriented"] = task_count > len(user_messages) / 2
            if aspect in ["task_complexity", "all"] and len(agent.conversation_history) > 2:
                recent_msgs = [msg["content"] for msg in agent.conversation_history[-min(5, len(agent.conversation_history)):] if msg.get("role") == "user"]
                complexity_indicators = {"high": ["complex", "advanced", "detailed", "comprehensive", "integrate", "optimize", "scale"],
                                         "low": ["simple", "basic", "easy", "quick", "just", "help me", "show me"]}
                high_complexity = sum(sum(1 for word in complexity_indicators["high"] if word.lower() in msg.lower()) for msg in recent_msgs)
                low_complexity = sum(sum(1 for word in complexity_indicators["low"] if word.lower() in msg.lower()) for msg in recent_msgs)
                if high_complexity > low_complexity * 2:
                    agent.environment_state["task_complexity"]["current_level"] = "high"
                elif low_complexity > high_complexity:
                    agent.environment_state["task_complexity"]["current_level"] = "low"
                else:
                    agent.environment_state["task_complexity"]["current_level"] = "medium"
            if aspect == "all":
                return {"environment_state": agent.environment_state, "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S"), "message_count": len([msg for msg in agent.conversation_history if msg.get("role") == "user"]), "success": True}
            else:
                return {aspect: agent.environment_state[aspect], "analysis_time": time.strftime("%Y-%m-%d %H:%M:%S"), "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _adapt_to_environment(self, adaptation_strategy: str, reason: str, system_prompt_update: str = None) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            adaptation = {"timestamp": time.time(), "datetime": time.strftime("%Y-%m-%d %H:%M:%S"), "strategy": adaptation_strategy, "reason": reason}
            agent.environment_state["task_complexity"]["adaptations_made"].append(adaptation)
            self._add_reflection_note(note=f"Adapted using {adaptation_strategy}: {reason}", category="adaptation")
            if system_prompt_update:
                self._update_system_prompt(system_prompt=system_prompt_update, append=True)
                adaptation["system_prompt_updated"] = True
            else:
                adaptation["system_prompt_updated"] = False
            return {"adaptation": adaptation, "current_environment": agent.environment_state, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _update_system_prompt(self, system_prompt: str, append: bool = False) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            old_prompt = agent.system_message["content"]
            new_prompt = old_prompt + "\n\n" + system_prompt if append else system_prompt
            for i, message in enumerate(agent.conversation_history):
                if message.get("role") == "system":
                    agent.conversation_history[i]["content"] = new_prompt
                    agent.system_message["content"] = new_prompt
                    break
            else:
                agent.conversation_history.insert(0, {"role": "system", "content": new_prompt})
                agent.system_message["content"] = new_prompt
            return {"old_system_prompt": old_prompt, "new_system_prompt": new_prompt, "append": append, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_weather(self, location: str = None, city: str = None) -> Dict[str, Any]:
        # Use city parameter if provided, otherwise use location
        location = city or location or "New York"
        try:
            import random
            conditions = ["sunny", "partly cloudy", "cloudy", "rainy", "stormy", "snowy", "windy", "foggy"]
            condition = random.choice(conditions)
            if condition == "sunny":
                temp_f = random.randint(70, 95)
            elif condition in ["partly cloudy", "cloudy"]:
                temp_f = random.randint(60, 80)
            elif condition in ["rainy", "stormy"]:
                temp_f = random.randint(50, 70)
            elif condition == "snowy":
                temp_f = random.randint(20, 35)
            else:
                temp_f = random.randint(40, 75)
            temp_c = round((temp_f - 32) * 5 / 9, 1)
            humidity = random.randint(30, 90)
            wind_speed = random.randint(0, 20)
            forecast = []
            current_temp = temp_f
            for i in range(5):
                forecast_temp = current_temp + random.randint(-10, 10)
                forecast_condition = random.choice(conditions)
                forecast.append({
                    "day": i + 1,
                    "condition": forecast_condition,
                    "high_f": forecast_temp,
                    "low_f": forecast_temp - random.randint(10, 20),
                    "precipitation_chance": random.randint(0, 100)
                })
            return {"location": location, "current_condition": condition, "temperature_f": temp_f, "temperature_c": temp_c, "humidity": humidity, "wind_speed_mph": wind_speed, "forecast": forecast, "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"), "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
            
    def _weather_for_location(self, location: str) -> Dict[str, Any]:
        """Get weather for a specific location - wrapper around _get_weather"""
        return self._get_weather(location=location)

    def _execute_python(self, code: str, save_artifact: bool = False, artifact_name: str = "", description: str = "") -> Dict[str, Any]:
        try:
            import io
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            local_vars = {}
            start_time = time.time()
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, globals(), local_vars)
            execution_time = time.time() - start_time
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            result = local_vars.get('_', None)
            try:
                result_str = json.dumps(result) if isinstance(result, (dict, list, tuple, set)) else str(result)
            except:
                result_str = str(result)
            response = {"stdout": stdout, "stderr": stderr, "result": result_str, "execution_time": execution_time, "success": True}
            if save_artifact:
                if not artifact_name:
                    first_line = code.strip().split('\n')[0]
                    if len(first_line) > 30:
                        first_line = first_line[:27] + "..."
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    artifact_name = f"code_{timestamp}_{first_line.replace(' ', '_')}"
                metadata = {"execution_time": execution_time, "has_output": bool(stdout), "has_error": bool(stderr), "has_result": result is not None}
                artifact_id = self.code_repo.add_artifact(name=artifact_name, code=code, description=description, metadata=metadata)
                self.code_repo.log_execution(artifact_id=artifact_id, success=True, stdout=stdout, stderr=stderr, result=result_str, execution_time=execution_time)
                response["artifact_id"] = artifact_id
                response["artifact_name"] = artifact_name
            return response
        except Exception as e:
            error_result = {"error": str(e), "traceback": traceback.format_exc(), "success": False}
            if save_artifact:
                if not artifact_name:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    artifact_name = f"failed_code_{timestamp}"
                metadata = {"error_type": type(e).__name__, "error_message": str(e)}
                artifact_id = self.code_repo.add_artifact(name=artifact_name, code=code, description=description or f"Failed execution: {str(e)}", metadata=metadata)
                self.code_repo.log_execution(artifact_id=artifact_id, success=False, stderr=traceback.format_exc(), result=str(e))
                error_result["artifact_id"] = artifact_id
                error_result["artifact_name"] = artifact_name
            return error_result

    def _save_module(self, name: str, code: str, description: str = "") -> Dict[str, Any]:
        try:
            if not name.isidentifier():
                return {"success": False, "error": f"Invalid module name: '{name}'. Must be a valid Python identifier."}
            existing_module = self.code_repo.get_module(name)
            if existing_module:
                success = self.code_repo.update_module(name, code, description)
                action = "updated"
            else:
                self.code_repo.add_module(name, code, description)
                success = True
                action = "created"
            if not success:
                return {"success": False, "error": f"Failed to {action} module '{name}'"}
            try:
                compile(code, f"<module:{name}>", "exec")
            except Exception as e:
                return {"success": True, "warning": f"Module saved but contains syntax errors: {str(e)}", "module_name": name, "action": action}
            return {"success": True, "module_name": name, "description": description, "action": action}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _execute_saved_module(self, name: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            module_data = self.code_repo.get_module(name)
            if not module_data:
                return {"success": False, "error": f"Module '{name}' not found"}
            globals_dict = globals().copy()
            if args:
                globals_dict.update(args)
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            result = None
            start_time = time.time()
            locals_dict = {}
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(module_data["code"], globals_dict, locals_dict)
                if "main" in locals_dict and callable(locals_dict["main"]):
                    result = locals_dict["main"](**args) if args else locals_dict["main"]()
            execution_time = time.time() - start_time
            try:
                result_str = json.dumps(result) if isinstance(result, (dict, list, tuple, set)) else str(result)
            except:
                result_str = str(result)
            return {"success": True, "module_name": name, "stdout": stdout_capture.getvalue(), "stderr": stderr_capture.getvalue(), "result": result_str, "execution_time": execution_time, "returned_main_function": "main" in locals_dict and callable(locals_dict["main"])}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _list_modules(self) -> Dict[str, Any]:
        try:
            modules = self.code_repo.list_modules()
            formatted_modules = []
            for module in modules:
                code = module["code"]
                line_count = len(code.split("\n"))
                code_size = len(code.encode("utf-8"))
                formatted_modules.append({
                    "name": module["name"],
                    "description": module["description"] or "No description",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(module["created_at"])),
                    "last_updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(module["last_updated_at"])),
                    "line_count": line_count,
                    "code_size_bytes": code_size
                })
            return {"success": True, "modules": formatted_modules, "count": len(formatted_modules)}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _get_module(self, name: str) -> Dict[str, Any]:
        try:
            module = self.code_repo.get_module(name)
            if not module:
                return {"success": False, "error": f"Module '{name}' not found"}
            formatted_module = {
                "name": module["name"],
                "description": module["description"] or "No description",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(module["created_at"])),
                "last_updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(module["last_updated_at"])),
                "code": module["code"],
                "line_count": len(module["code"].split("\n")),
                "code_size_bytes": len(module["code"].encode("utf-8"))
            }
            return {"success": True, "module": formatted_module}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _run_script(self, file_path: str, args: List[str] = None) -> Dict[str, Any]:
        try:
            path = Path(file_path).expanduser()
            if not path.exists():
                return {"success": False, "error": f"Script file '{file_path}' does not exist"}
            if not path.is_file():
                return {"success": False, "error": f"'{file_path}' is not a file"}
            cmd = [sys.executable, str(path)]
            if args:
                cmd.extend(args)
            start_time = time.time()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            execution_time = time.time() - start_time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            script_name = path.name
            with open(path, "r", encoding="utf-8") as f:
                script_content = f.read()
            artifact_id = self.code_repo.add_artifact(name=f"script_{script_name}_{timestamp}", code=script_content, description=f"Execution of script {script_name}", metadata={"file_path": str(path), "args": args or [], "exit_code": process.returncode, "execution_time": execution_time})
            self.code_repo.log_execution(artifact_id=artifact_id, success=process.returncode == 0, stdout=stdout, stderr=stderr, execution_time=execution_time)
            return {"success": process.returncode == 0, "exit_code": process.returncode, "stdout": stdout, "stderr": stderr, "execution_time": execution_time, "file_path": str(path), "args": args or [], "artifact_id": artifact_id}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _search_code(self, query: str, limit: int = 10) -> Dict[str, Any]:
        try:
            artifacts = self.code_repo.find_artifacts(query, limit)
            formatted_artifacts = []
            for artifact in artifacts:
                code_snippets = []
                lines = artifact.code.split("\n")
                for i, line in enumerate(lines):
                    if query.lower() in line.lower():
                        start = max(0, i - 2)
                        end = min(len(lines), i + 3)
                        snippet = "\n".join(lines[start:end])
                        line_number = i + 1
                        code_snippets.append({"line": line_number, "snippet": snippet})
                formatted_artifacts.append({
                    "artifact_id": artifact.artifact_id,
                    "name": artifact.name,
                    "description": artifact.description or "No description",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(artifact.created_at)),
                    "execution_count": artifact.execution_count,
                    "line_count": len(artifact.code.split("\n")),
                    "code_size_bytes": len(artifact.code.encode("utf-8")),
                    "snippets": code_snippets[:3]
                })
            return {"success": True, "query": query, "artifacts": formatted_artifacts, "count": len(formatted_artifacts)}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

    def _get_multiple_weather(self, locations: List[str]) -> Dict[str, Any]:
        """Get weather information for multiple locations in parallel"""
        try:
            # Create a task processor for parallel execution
            task_processor = AsyncTaskProcessor()
            tasks = []
            
            # Add tasks for each location
            for location in locations:
                task_id = task_processor.add_task(self._get_weather, location)
                tasks.append((task_id, location))
            
            # Wait for all tasks to complete
            results = {}
            for task_id, location in tasks:
                while True:
                    task_result = task_processor.get_result(task_id)
                    if task_result["status"] in ["completed", "failed"]:
                        if task_result["status"] == "completed":
                            results[location] = task_result["result"]
                        else:
                            results[location] = {"error": task_result.get("error", "Unknown error"), "success": False}
                        break
                    time.sleep(0.1)
            
            # Clean up
            task_processor.stop()
            
            return {
                "success": True,
                "locations_count": len(locations),
                "results": results
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _parse_weather_response(self, response: str) -> Dict[str, Any]:
        try:
            if isinstance(response, dict):
                weather_data = response
            else:
                try:
                    weather_data = json.loads(response)
                except json.JSONDecodeError:
                    return {"error": "Invalid weather data format", "success": False}
            location = weather_data.get("location", "Unknown location")
            condition = weather_data.get("current_condition", "unknown")
            temp_f = weather_data.get("temperature_f", 0)
            temp_c = weather_data.get("temperature_c", 0)
            humidity = weather_data.get("humidity", 0)
            wind_speed = weather_data.get("wind_speed_mph", 0)
            forecast = weather_data.get("forecast", [])
            summary = f"Current weather in {location}: {condition.capitalize()}, {temp_f}°F ({temp_c}°C), humidity {humidity}%, wind {wind_speed} mph."
            forecast_summary = ""
            if forecast and len(forecast) > 0:
                tomorrow = forecast[0]
                forecast_summary = f"\n\nTomorrow's forecast: {tomorrow.get('condition', 'unknown').capitalize()}, high of {tomorrow.get('high_f', 0)}°F, low of {tomorrow.get('low_f', 0)}°F, {tomorrow.get('precipitation_chance', 0)}% chance of precipitation."
            return {"location": location, "summary": summary + forecast_summary, "condition": condition, "temperature_f": temp_f, "temperature_c": temp_c, "humidity": humidity, "wind_speed": wind_speed, "forecast": forecast, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _run_assistant(self, assistant_id: str, thread_id: str) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access the TogetherAgent instance", "success": False}
            system_prompt = f"You are an assistant (ID: {assistant_id}) helping with this conversation. Please analyze the conversation history and provide a helpful response."
            messages = [{"role": "system", "content": system_prompt}]
            for msg in agent.conversation_history:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            response = agent.client.chat.completions.create(model=agent.model, messages=messages, temperature=0.7, max_tokens=500)
            response_content = response.choices[0].message.content
            agent.add_message("assistant", response_content)
            return {"run_id": f"run_{int(time.time())}", "thread_id": thread_id, "assistant_id": assistant_id, "status": "completed", "response": response_content, "success": True}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def search(self, query: str) -> Dict[str, Any]:
        """
        Search the web for information using Jina's search API.
        
        Args:
            query: The search query string
            
        Returns:
            Dictionary containing search results or error information
        """
        if not self.jina_client:
            try:
                self.jina_client = JinaClient()
                console.print("[green]Successfully initialized Jina client[/green]")
            except ValueError:
                return {"error": "Jina client not initialized. Please set JINA_API_KEY environment variable.", "success": False}
        
        try:
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            result = loop.run_until_complete(self.jina_client.search(query))
            
            # Process the results to make them more readable
            processed_results = result["results"]
            
            # Extract relevant information if possible
            try:
                # Try to parse as JSON if it looks like JSON
                if processed_results.strip().startswith('{') and processed_results.strip().endswith('}'):
                    processed_results = json.loads(processed_results)
            except:
                # If parsing fails, keep as string
                pass
                
            return {
                "success": True, 
                "query": query, 
                "results": processed_results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _web_search(self, query: str) -> Dict[str, Any]:
        """Legacy wrapper for the search function"""
        return self.search(query)

    def read(self, url: str) -> Dict[str, Any]:
        """
        Read and extract content from a web page using Jina's reader API.
        
        Args:
            url: The URL of the web page to read
            
        Returns:
            Dictionary containing the extracted content or error information
        """
        if not self.jina_client:
            try:
                self.jina_client = JinaClient()
                console.print("[green]Successfully initialized Jina client[/green]")
            except ValueError:
                return {"error": "Jina client not initialized. Please set JINA_API_KEY environment variable.", "success": False}
        
        try:
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            result = loop.run_until_complete(self.jina_client.read(url))
            
            # Process the content to make it more readable
            content = result["results"]
            
            # Extract metadata about the page
            metadata = {
                "url": url,
                "retrieved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "content_length": len(content)
            }
            
            return {
                "success": True, 
                "url": url, 
                "content": content,
                "metadata": metadata
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _web_read(self, url: str) -> Dict[str, Any]:
        """Legacy wrapper for the read function"""
        return self.read(url)

    def fact_check(self, query: str) -> Dict[str, Any]:
        """
        Verify a statement using Jina's fact checking API.
        
        Args:
            query: The statement to fact check
            
        Returns:
            Dictionary containing fact check results or error information
        """
        if not self.jina_client:
            try:
                self.jina_client = JinaClient()
                console.print("[green]Successfully initialized Jina client[/green]")
            except ValueError:
                return {"error": "Jina client not initialized. Please set JINA_API_KEY environment variable.", "success": False}
        
        try:
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            result = loop.run_until_complete(self.jina_client.fact_check(query))
            
            # Process the result to provide a more structured response
            return {
                "success": True, 
                "query": query, 
                "fact_check_result": result,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _search_similar_images(self, image_url: str, limit: int = 3) -> Dict[str, Any]:
        """Search for similar images in the conversation history"""
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            
            # Try to import the image processor
            try:
                from image_processor import ImageProcessor
                image_processor = ImageProcessor()
            except ImportError:
                # Fall back to the old method if image_processor.py is not available
                results = agent.search_similar_images(image_url, limit)
                
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "content": result["content"],
                        "type": result.get("metadata", {}).get("type", "unknown"),
                        "relevance_score": result.get("relevance_score", 0),
                        "timestamp": result.get("created_at", "unknown")
                    })
                
                return {
                    "success": True,
                    "query_image": image_url,
                    "results_count": len(formatted_results),
                    "results": formatted_results
                }
            
            # Use the new image processor for more advanced image comparison
            results = agent.search_similar_images(image_url, limit)
            
            # Process the query image
            try:
                query_image_features = image_processor.extract_image_features(
                    image_processor.process_image(image_url)
                )
                
                # Enhanced results with similarity scores
                enhanced_results = []
                for result in results:
                    try:
                        # Only process image results
                        if result.get("metadata", {}).get("type") == "image":
                            result_image_url = result["content"]
                            similarity = image_processor.compare_images(image_url, result_image_url)
                            
                            enhanced_results.append({
                                "content": result["content"],
                                "type": "image",
                                "relevance_score": float(similarity),
                                "timestamp": result.get("created_at", "unknown")
                            })
                    except Exception as img_err:
                        # If processing a specific image fails, still include it but with original score
                        enhanced_results.append({
                            "content": result["content"],
                            "type": result.get("metadata", {}).get("type", "unknown"),
                            "relevance_score": result.get("relevance_score", 0),
                            "timestamp": result.get("created_at", "unknown"),
                            "processing_error": str(img_err)
                        })
                
                # Sort by relevance score
                enhanced_results.sort(key=lambda x: x["relevance_score"], reverse=True)
                
                return {
                    "success": True,
                    "query_image": image_url,
                    "results_count": len(enhanced_results),
                    "results": enhanced_results[:limit],
                    "using_advanced_processor": True
                }
            
            except Exception as proc_err:
                # Fall back to basic results if advanced processing fails
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "content": result["content"],
                        "type": result.get("metadata", {}).get("type", "unknown"),
                        "relevance_score": result.get("relevance_score", 0),
                        "timestamp": result.get("created_at", "unknown")
                    })
                
                return {
                    "success": True,
                    "query_image": image_url,
                    "results_count": len(formatted_results),
                    "results": formatted_results,
                    "processing_error": str(proc_err)
                }
                
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _check_multi_location_weather_query(self, query: str) -> Optional[str]:
        """Check if the query is asking for weather in multiple locations"""
        # Common patterns for multi-location weather queries
        patterns = [
            r"weather\s+in\s+([A-Za-z\s,]+)\s+and\s+([A-Za-z\s,]+)",
            r"weather\s+for\s+([A-Za-z\s,]+)\s+and\s+([A-Za-z\s,]+)",
            r"weather\s+(?:in|for|at)\s+([A-Za-z\s,]+)(?:\s*,\s*|\s+and\s+)([A-Za-z\s,]+)"
        ]
        
        locations = []
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                for match in matches:
                    locations.extend([loc.strip() for loc in match if loc.strip()])
                break
        
        # If we found multiple locations, process them
        if len(locations) > 1:
            console.print(f"[cyan]Detected weather query for multiple locations: {locations}[/cyan]")
            
            # Call the multiple weather function
            result = self._get_multiple_weather(locations)
            
            if result.get("success", False):
                # Format the results into a nice response
                response = f"Here's the current weather for the locations you asked about:\n\n"
                
                for location, weather_data in result.get("results", {}).items():
                    if weather_data.get("success", False):
                        response += f"**{location}**: {weather_data.get('current_condition', 'Unknown').capitalize()}, " \
                                   f"{weather_data.get('temperature_f', 'N/A')}°F ({weather_data.get('temperature_c', 'N/A')}°C), " \
                                   f"humidity {weather_data.get('humidity', 'N/A')}%, " \
                                   f"wind speed {weather_data.get('wind_speed_mph', 'N/A')} mph.\n\n"
                        
                        # Add tomorrow's forecast
                        forecast = weather_data.get("forecast", [])
                        if forecast and len(forecast) > 0:
                            tomorrow = forecast[0]
                            response += f"Tomorrow in {location}: {tomorrow.get('condition', 'unknown').capitalize()}, " \
                                       f"high of {tomorrow.get('high_f', 'N/A')}°F, " \
                                       f"low of {tomorrow.get('low_f', 'N/A')}°F, " \
                                       f"{tomorrow.get('precipitation_chance', 'N/A')}% chance of precipitation.\n\n"
                    else:
                        response += f"**{location}**: Could not retrieve weather data. Error: {weather_data.get('error', 'Unknown error')}\n\n"
                
                # Add the response to conversation history
                self.add_message("assistant", response)
                
                return response
        
        return None
        
    def _fact_check(self, query: str) -> Dict[str, Any]:
        """Legacy wrapper for the fact_check function"""
        return self.fact_check(query)

    def _create_planning_session(self, task: str) -> Dict[str, Any]:
        try:
            session_id = str(uuid.uuid4())
            created_at = time.time()
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if agent:
                session = PlanningSession(session_id=session_id, task=task, created_at=created_at)
                agent.planning_session = session
                return {"success": True, "session_id": session_id, "task": task, "message": "Planning session created successfully"}
            else:
                return {"error": "Could not access TogetherAgent instance", "success": False}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _add_plan_step(self, plan: str, tool_name: str = None, tool_args: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent or not agent.planning_session:
                return {"error": "No active planning session", "success": False}
            step = {"step_id": len(agent.planning_session.steps) + 1, "timestamp": time.time(), "plan": plan}
            if tool_name and tool_name in self.functions:
                step["tool"] = {"name": tool_name, "arguments": tool_args or {}}
                if tool_args:
                    result = self.call_function(tool_name, tool_args)
                    step["result"] = result
            agent.planning_session.steps.append(step)
            return {"success": True, "step_id": step["step_id"], "message": "Plan step added successfully"}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_planning_status(self) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent or not agent.planning_session:
                return {"error": "No active planning session", "success": False}
            return {"success": True, "session_id": agent.planning_session.session_id, "task": agent.planning_session.task, "steps_count": len(agent.planning_session.steps), "steps": agent.planning_session.steps, "state": agent.planning_session.state, "active": agent.planning_session.active, "completion_status": agent.planning_session.completion_status, "created_at": agent.planning_session.created_at, "elapsed_time": time.time() - agent.planning_session.created_at}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _complete_planning_session(self, summary: str, success: bool = True) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent or not agent.planning_session:
                return {"error": "No active planning session", "success": False}
            agent.planning_session.active = False
            agent.planning_session.completion_status = "completed" if success else "failed"
            agent.planning_session.steps.append({"step_id": len(agent.planning_session.steps) + 1, "timestamp": time.time(), "summary": summary, "success": success})
            if not hasattr(agent, "completed_planning_sessions"):
                agent.completed_planning_sessions = []
            agent.completed_planning_sessions.append(agent.planning_session)
            completed_session = agent.planning_session
            agent.planning_session = None
            return {"success": True, "session_id": completed_session.session_id, "task": completed_session.task, "steps_count": len(completed_session.steps), "completion_status": completed_session.completion_status, "elapsed_time": time.time() - completed_session.created_at, "summary": summary}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _extract_urls(self, text: str) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            extraction = agent.task_processor.extract_urls(text)
            return {"success": True, "urls": extraction.urls, "count": len(extraction.urls), "source": extraction.source, "timestamp": extraction.timestamp}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _process_urls(self, urls: List[str]) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            new_urls_count = agent.task_processor.add_urls_to_process(urls)
            return {"success": True, "urls_added": new_urls_count, "total_urls_pending": agent.task_processor.task_queue.qsize(), "total_urls_processed": len(agent.task_processor.processed_urls), "message": f"Added {new_urls_count} new URLs to the processing queue"}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_knowledge_summary(self) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            summary = agent.task_processor.get_knowledge_summary()
            summary["success"] = True
            return summary
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _search_conversation_history(self, query: str, limit: int = 5, include_images: bool = True) -> Dict[str, Any]:
        """Search the conversation history for relevant content"""
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            
            if not agent or not hasattr(agent, 'memory'):
                return {"error": "Could not access TogetherAgent memory", "success": False}
            
            results = agent.memory.search_memory(query, limit=limit, include_images=include_images)
            
            formatted_results = []
            for result in results:
                formatted_result = {
                    "content": result["content"],
                    "type": result.get("metadata", {}).get("type", "text"),
                    "role": result.get("metadata", {}).get("role", "unknown"),
                    "relevance_score": result.get("relevance_score", 0),
                    "timestamp": result.get("created_at", "unknown")
                }
                formatted_results.append(formatted_result)
            
            return {
                "success": True,
                "query": query,
                "results_count": len(formatted_results),
                "results": formatted_results
            }
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _search_knowledge(self, query: str) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            results = agent.task_processor.search_knowledge(query)
            return {"success": True, "query": query, "results_count": len(results), "results": results}
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

    def _get_current_datetime(self, timezone: str = "local") -> Dict[str, Any]:
        """Get the current date and time, optionally in a specific timezone."""
        try:
            import datetime
            import pytz
            from zoneinfo import ZoneInfo, available_timezones
            
            now = datetime.datetime.now()
            local_time = now
            
            if timezone and timezone.lower() != "local":
                try:
                    # Try using ZoneInfo (Python 3.9+)
                    local_time = now.astimezone(ZoneInfo(timezone))
                except (ImportError, KeyError):
                    try:
                        # Fall back to pytz
                        tz = pytz.timezone(timezone)
                        local_time = now.astimezone(tz)
                    except (pytz.exceptions.UnknownTimeZoneError, ImportError):
                        return {
                            "error": f"Unknown timezone: {timezone}",
                            "available_timezones": "Use standard timezone names like 'UTC', 'US/Eastern', 'Europe/London'",
                            "success": False
                        }
            
            result = {
                "current_datetime": local_time.strftime("%Y-%m-%d %H:%M:%S"),
                "date": local_time.strftime("%Y-%m-%d"),
                "time": local_time.strftime("%H:%M:%S"),
                "timezone": timezone if timezone != "local" else "local system timezone",
                "timestamp": time.time(),
                "iso_format": local_time.isoformat(),
                "success": True
            }
            
            # Add day of week, month name, etc.
            result["day_of_week"] = local_time.strftime("%A")
            result["month"] = local_time.strftime("%B")
            result["year"] = local_time.year
            result["day"] = local_time.day
            
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}
    
    def _monitor_task(self, task_id: str) -> Dict[str, Any]:
        try:
            agent = None
            for frame in inspect.stack():
                if 'self' in frame.frame.f_locals and isinstance(frame.frame.f_locals['self'], TogetherAgent):
                    agent = frame.frame.f_locals['self']
                    break
            if not agent:
                return {"error": "Could not access TogetherAgent instance", "success": False}
            result = agent.task_processor.get_result(task_id)
            result["task_id"] = task_id
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc(), "success": False}

# =======================
# Helper Functions
# =======================
def extract_python_code(text: str) -> list:
    """Extract Python code blocks with better handling of formatting issues"""
    pattern = r'<\|python_start\|>(.*?)(?:<\|python_end\|>|<\|python_end)'
    code_blocks = [match.strip() for match in re.findall(pattern, text, re.DOTALL)]
    
    # Process each code block to fix common formatting issues
    fixed_blocks = []
    for block in code_blocks:
        # Fix compressed one-liners for if statements, function definitions, etc.
        lines = []
        for line in block.split('\n'):
            # Skip purely decorative lines
            if all(c in '┏━┓┃┗┛ ' for c in line):
                continue
                
            # Fix compressed if statements
            if ' if ' in line and ': ' in line and 'return' in line and not line.strip().startswith('if'):
                parts = line.split(' if ')
                pre = parts[0].strip()
                rest = parts[1].strip()
                if_parts = rest.split(': ')
                if len(if_parts) == 2:
                    lines.append(f"{pre} if {if_parts[0]}:")
                    lines.append(f"    {if_parts[1]}")
                    continue
            
            # Fix compressed function definitions
            if 'def ' in line and '(' in line and ')' in line and ': ' in line and not line.strip().startswith('def'):
                parts = line.split('def ')
                pre = parts[0].strip()
                rest = 'def ' + parts[1].strip()
                if pre:
                    lines.append(pre)
                lines.append(rest)
                continue
                
            lines.append(line)
        
        fixed_blocks.append('\n'.join(lines))
    
    return fixed_blocks

def parse_function_calls(text: str) -> List[Dict[str, Any]]:
    """
    Advanced function call parser that handles multiple formats and provides better error recovery.
    Supports:
    1. OpenAI-style JSON format: {"name": "func_name", "arguments": {...}}
    2. Bracket format: [func_name(arg1=val1, arg2=val2)]
    3. Function tag format: <function=func_name>{"arg1": "val1"}</function>
    4. Python code block format: <|python_start|><function=func_name>...</|python_end|>
    5. Tool calls format: {"type": "function", "function": {"name": "func_name", "arguments": "{...}"}}
    6. Multiple locations in weather queries: "weather in X and Y" -> get_multiple_weather
    7. Direct weather queries: "weather in X" -> get_weather(location="X")
    """
    function_calls = []
    
    # Track if we found any potential function calls for better error reporting
    potential_function_call_found = False
    
    # Try to find OpenAI-style tool calls first (most reliable format)
    tool_call_pattern = r'\{"type"\s*:\s*"function"\s*,\s*"function"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*"(.+?)"\s*\}\s*\}'
    tool_matches = re.findall(tool_call_pattern, text.replace('\n', ' '))
    for func_name, args_str in tool_matches:
        potential_function_call_found = True
        try:
            # Handle escaped JSON in the arguments string
            args_str = args_str.replace('\\"', '"').replace('\\\\', '\\')
            args = json.loads(args_str)
            function_calls.append({"name": func_name, "arguments": args})
            continue
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Could not parse tool call arguments for {func_name}, trying fallback methods[/yellow]")
    
    # Try to find JSON-formatted function calls
    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    potential_matches = re.findall(json_pattern, text)
    for potential_json in potential_matches:
        try:
            data = json.loads(potential_json)
            # Handle direct function call format
            if isinstance(data, dict) and "name" in data and "arguments" in data:
                potential_function_call_found = True
                function_calls.append({"name": data["name"], "arguments": data["arguments"]})
                continue
                
            # Handle OpenAI-style tool calls format
            if isinstance(data, dict) and data.get("type") == "function" and "function" in data:
                potential_function_call_found = True
                func_data = data["function"]
                func_name = func_data.get("name")
                if func_name:
                    # Arguments might be a JSON string or already parsed
                    args_data = func_data.get("arguments", {})
                    if isinstance(args_data, str):
                        try:
                            args = json.loads(args_data)
                        except json.JSONDecodeError:
                            args = {"raw_arguments": args_data}
                    else:
                        args = args_data
                    function_calls.append({"name": func_name, "arguments": args})
                    continue
        except json.JSONDecodeError:
            pass
    
    # If we already found function calls in the preferred formats, return them
    if function_calls:
        return function_calls
    
    # Try to find function calls in the format [function_name(arg1=val1, arg2=val2)]
    pattern = r'\[(\w+)\((.*?)\)\]'
    matches = re.findall(pattern, text)
    for match in matches:
        potential_function_call_found = True
        func_name, args_str = match
        args = {}
        try:
            args_str = args_str.strip()
            if args_str.startswith("{") and args_str.endswith("}"):
                args = json.loads(args_str)
            else:
                # Handle both key=value and key="value with spaces"
                arg_pairs = re.findall(r'(\w+)=("[^"]*"|\'[^\']*\'|\S+)', args_str)
                for arg_name, arg_value in arg_pairs:
                    if (arg_value.startswith('"') and arg_value.endswith('"')) or (arg_value.startswith("'") and arg_value.endswith("'")):
                        arg_value = arg_value[1:-1]
                    args[arg_name] = arg_value
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Error parsing arguments for {func_name}: {args_str}[/yellow]")
            # Still add with empty args for recovery
            args = {}
        function_calls.append({"name": func_name, "arguments": args})
    
    # Try to find function calls in the format <function=function_name>{"arg1": "val1"}</function>
    function_pattern = r'<function=(\w+)>(.*?)</function>'
    fn_matches = re.findall(function_pattern, text)
    for m in fn_matches:
        potential_function_call_found = True
        func_name = m[0]
        args_str = m[1].strip()
        try:
            # If args_str is empty or not valid JSON, use empty dict
            if not args_str or args_str == "{}":
                args = {}
            else:
                args = json.loads(args_str)
            function_calls.append({"name": func_name, "arguments": args})
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Error parsing arguments for {func_name}: {m[1]}[/yellow]")
            # Still add the function call with empty arguments as a fallback
            function_calls.append({"name": func_name, "arguments": {}})
    
    # Special case for <|python_start|><function=X>...<|python_end pattern
    python_function_pattern = r'<\|python_start\|><function=(\w+)>'
    python_fn_matches = re.findall(python_function_pattern, text)
    for func_name in python_fn_matches:
        potential_function_call_found = True
        function_calls.append({"name": func_name, "arguments": {}})
    
    # Special case for <|python_start|>{"type": "function", "name": "X", "parameters": {}}
    python_json_pattern = r'<\|python_start\|>(\{.*?\})<\|python_end'
    python_json_matches = re.findall(python_json_pattern, text, re.DOTALL)
    for json_str in python_json_matches:
        potential_function_call_found = True
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and "name" in data:
                # Handle both "parameters" and "arguments" keys
                args = data.get("parameters", data.get("arguments", {}))
                function_calls.append({"name": data["name"], "arguments": args})
        except json.JSONDecodeError:
            console.print(f"[yellow]Warning: Could not parse JSON in Python block: {json_str[:50]}...[/yellow]")
    
    # Special case for "list_functions" or similar common function names if no other matches
    if not function_calls and "list_functions" in text.lower():
        function_calls.append({"name": "list_available_functions", "arguments": {}})
    elif not function_calls and "list tools" in text.lower():
        function_calls.append({"name": "list_available_functions", "arguments": {}})
    # Special case for direct weather queries
    elif not function_calls and "weather" in text.lower():
        # Try to extract location from "weather in X" pattern
        weather_match = re.search(r"weather\s+(?:in|for|at)\s+([A-Za-z\s,]+)", text.lower())
        if weather_match:
            location = weather_match.group(1).strip()
            function_calls.append({"name": "weather_for_location", "arguments": {"location": location}})
    
    # If we found potential function calls but couldn't parse any, log a warning
    if potential_function_call_found and not function_calls:
        console.print("[red]Warning: Detected potential function calls but failed to parse them[/red]")
        console.print(f"[dim]Text snippet: {text[:100]}...[/dim]")
    
    return function_calls

def get_structured_function_call(messages: List[Dict[str, str]], tools: List[Dict[str, Any]], model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo") -> List[Dict[str, Any]]:
    from pydantic import BaseModel, Field
    class FunctionCall(BaseModel):
        name: str = Field(description="Name of the function to call")
        arguments: Dict[str, Any] = Field(description="Arguments for the function call")
    class ParallelFunctionCalls(BaseModel):
        calls: List[FunctionCall] = Field(description="List of function calls to execute in parallel")
    system_prefix = "You must respond with a valid JSON object containing one or more function calls. "
    system_found = False
    for msg in messages:
        if msg["role"] == "system":
            msg["content"] = system_prefix + msg["content"]
            system_found = True
            break
    if not system_found:
        messages.insert(0, {"role": "system", "content": system_prefix + "Use the available tools to respond to the user's request."})
    from together import Together
    together = Together()
    response = together.chat.completions.create(
        messages=messages,
        model=model,
        response_format={"type": "json_object", "schema": ParallelFunctionCalls.model_json_schema()},
        tools=tools
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        if "calls" in data and isinstance(data["calls"], list):
            return [{"name": call["name"], "arguments": call["arguments"]} for call in data["calls"] if "name" in call and "arguments" in call]
        elif "name" in data and "arguments" in data:
            return [{"name": data["name"], "arguments": data["arguments"]}]
        return []
    except json.JSONDecodeError:
        console.print("[yellow]Warning: JSON parsing failed, falling back to regex[/yellow]")
        return parse_function_calls(content)

def get_parallel_function_calls(messages: List[Dict[str, str]], tools: List[Dict[str, Any]], model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo") -> List[Dict[str, Any]]:
    from pydantic import BaseModel, Field
    class FunctionCall(BaseModel):
        name: str = Field(description="Name of the function to call")
        arguments: Dict[str, Any] = Field(description="Arguments for the function call")
        reasoning: str = Field(description="Reasoning for why this function should be called", default="")
        priority: int = Field(description="Priority of this function call (1-5, higher is more important)", default=3)
        
    class ParallelFunctionCalls(BaseModel):
        calls: List[FunctionCall] = Field(description="List of function calls to execute in parallel")
        execution_strategy: str = Field(description="Strategy for executing these calls (e.g., 'parallel', 'sequential', 'priority-based')", default="parallel")
        
    system_prefix = """You must respond with a valid JSON object containing multiple function calls to execute in parallel.
For each function call, provide:
1. The exact function name from the available tools
2. The correct arguments with proper types
3. A brief reasoning for why this function is needed
4. A priority level (1-5) to indicate importance

ALL functions can be executed in parallel, including weather queries for multiple locations, web searches, and other operations.
Group related operations that can be executed in parallel, and specify an execution strategy.
"""
    system_found = False
    for msg in messages:
        if msg["role"] == "system":
            msg["content"] = system_prefix + msg["content"]
            system_found = True
            break
    if not system_found:
        messages.insert(0, {"role": "system", "content": system_prefix + "Identify all operations that can be executed in parallel and return them as a structured list."})
    
    from together import Together
    together = Together()
    
    # First try with structured JSON format
    try:
        response = together.chat.completions.create(
            messages=messages,
            model=model,
            response_format={"type": "json_object", "schema": ParallelFunctionCalls.model_json_schema()},
            tools=tools,
            temperature=0.2  # Lower temperature for more precise function calling
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        if "calls" in data and isinstance(data["calls"], list):
            # Sort by priority if provided
            sorted_calls = sorted(data["calls"], key=lambda x: x.get("priority", 3), reverse=True)
            return [{"name": call["name"], "arguments": call["arguments"]} for call in sorted_calls if "name" in call and "arguments" in call]
    except (json.JSONDecodeError, Exception) as e:
        console.print(f"[yellow]Warning: Structured JSON parsing failed: {str(e)}, trying alternative approach[/yellow]")
    
    # Fallback to simpler approach with more direct instructions
    fallback_messages = messages.copy()
    fallback_messages.append({
        "role": "user", 
        "content": "Identify the specific functions needed to complete this task. For each function, provide the exact name and arguments in JSON format."
    })
    
    try:
        response = together.chat.completions.create(
            messages=fallback_messages,
            model=model,
            tools=tools,
            temperature=0.3
        )
        content = response.choices[0].message.content
        return parse_function_calls(content)
    except Exception as e:
        console.print(f"[red]Error in fallback approach: {str(e)}[/red]")
        return []

def execute_parallel_functions(function_calls: List[Dict[str, Any]], tool_registry):
    """Execute multiple function calls in parallel with advanced error handling and retry logic
    
    This function supports all types of functions, including weather queries for multiple locations,
    web searches, and other operations that can be executed in parallel.
    """
    task_processor = AsyncTaskProcessor()
    tasks = []
    
    # First, validate all function calls and try to fix any issues
    validated_calls = []
    for func_call in function_calls:
        function_name = func_call.get("name")
        arguments = func_call.get("arguments", {})
        
        # Try to find similar function if exact match not found
        if not tool_registry.has_tool(function_name):
            similar_tools = []
            for tool_name in tool_registry.get_available_tools():
                # Check for similarity using various metrics
                if function_name.lower() in tool_name.lower() or tool_name.lower() in function_name.lower():
                    similar_tools.append(tool_name)
                elif function_name.replace("_", "").lower() == tool_name.replace("_", "").lower():
                    similar_tools.append(tool_name)
            
            if similar_tools:
                # Use the most similar tool
                corrected_name = similar_tools[0]
                console.print(f"[yellow]Function '{function_name}' not found, using similar function '{corrected_name}'[/yellow]")
                function_name = corrected_name
            else:
                console.print(f"[red]Function '{function_name}' not found in tool registry and no similar functions available[/red]")
                continue
        
        # Validate arguments against function parameters
        if function_name in tool_registry.functions:
            func_spec = tool_registry.functions[function_name]
            required_params = func_spec.parameters.get("required", [])
            properties = func_spec.parameters.get("properties", {})
            
            # Check for missing required parameters
            missing_params = [param for param in required_params if param not in arguments]
            if missing_params:
                console.print(f"[yellow]Missing required parameters for '{function_name}': {missing_params}[/yellow]")
                # Try to provide default values for missing parameters
                for param in missing_params:
                    if param in properties and "default" in properties[param]:
                        arguments[param] = properties[param]["default"]
                        console.print(f"[yellow]Using default value for '{param}'[/yellow]")
            
            # Check for invalid parameters
            invalid_params = [param for param in arguments if param not in properties]
            if invalid_params:
                console.print(f"[yellow]Removing invalid parameters for '{function_name}': {invalid_params}[/yellow]")
                for param in invalid_params:
                    arguments.pop(param)
        
        validated_calls.append((function_name, arguments))
    
    # Add validated calls to the task processor
    for function_name, arguments in validated_calls:
        if tool_registry.has_tool(function_name):
            func = tool_registry.get_tool(function_name)
            task_id = task_processor.add_task(func, **arguments)
            tasks.append((task_id, function_name, arguments))
    
    # Process results with timeout and retry logic
    results = []
    max_wait_time = 30  # seconds
    retry_count = 0
    max_retries = 1
    
    for task_id, function_name, arguments in tasks:
        start_time = time.time()
        while True:
            task_result = task_processor.get_result(task_id)
            
            # Check if task completed or failed
            if task_result["status"] in ["completed", "failed"]:
                # If failed and we have retries left, try again with adjusted parameters
                if task_result["status"] == "failed" and retry_count < max_retries:
                    retry_count += 1
                    console.print(f"[yellow]Retrying '{function_name}' after failure: {task_result.get('error')}[/yellow]")
                    # Try to adjust arguments based on error message
                    error_msg = str(task_result.get('error', ''))
                    if "missing" in error_msg.lower() and "required" in error_msg.lower():
                        # Try to extract parameter name from error
                        import re
                        param_match = re.search(r"'([^']+)'", error_msg)
                        if param_match and param_match.group(1) not in arguments:
                            arguments[param_match.group(1)] = None  # Add placeholder
                    
                    # Add new task with adjusted arguments
                    func = tool_registry.get_tool(function_name)
                    new_task_id = task_processor.add_task(func, **arguments)
                    task_id = new_task_id  # Update task_id for the while loop
                    continue
                
                # Add result to the list
                results.append({
                    "function_name": function_name, 
                    "arguments": arguments, 
                    "status": task_result["status"], 
                    "result": task_result.get("result") if task_result["status"] == "completed" else task_result.get("error"),
                    "execution_time": time.time() - start_time
                })
                break
                
            # Check for timeout
            if time.time() - start_time > max_wait_time:
                console.print(f"[yellow]Timeout waiting for '{function_name}' to complete[/yellow]")
                results.append({
                    "function_name": function_name,
                    "arguments": arguments,
                    "status": "timeout",
                    "error": f"Function execution timed out after {max_wait_time} seconds"
                })
                break
                
            time.sleep(0.1)
    
    task_processor.stop()
    
    # Sort results by execution time for reporting
    results.sort(key=lambda x: x.get("execution_time", float('inf')))
    
    return results

def batch_process_function_calls(agent, user_message, max_batch_size=5):
    """
    Process multiple function calls in batches with enhanced orchestration, 
    dependency tracking, and error recovery.
    
    Args:
        agent: The TogetherAgent instance
        user_message: The original user message
        max_batch_size: Maximum number of operations to process in parallel
        
    Returns:
        Final response after all batches completed
    """
    # First, analyze the task to determine if it needs decomposition
    task_analysis_prompt = [
        {"role": "system", "content": """You are an expert task analyzer. 
        Examine the user's request and determine:
        1. If it should be broken into multiple subtasks
        2. If there are dependencies between operations
        3. The optimal execution strategy (parallel, sequential, or hybrid)
        
        Return your analysis as a structured JSON object."""},
        {"role": "user", "content": user_message}
    ]
    
    from together import Together
    together = Together()
    
    try:
        analysis_response = together.chat.completions.create(
            messages=task_analysis_prompt,
            model=agent.model,
            temperature=0.2,
                            
        )
        
        # Extract all function calls with dependency information if possible
        all_function_calls = get_parallel_function_calls(
            messages=[
                {"role": "system", "content": """Identify all operations needed to complete this task. 
                For each operation, specify:
                1. The exact function name
                2. The precise arguments with correct types
                3. Any dependencies on other operations
                4. Priority level (1-5, higher is more important)
                
                Return them as a structured JSON array."""},
                {"role": "user", "content": user_message}
            ],
            tools=agent.tool_registry.get_openai_tools_format(),
            model=agent.model
        )
        
        if not all_function_calls:
            console.print("[yellow]No function calls identified, falling back to standard chat[/yellow]")
            return agent.chat(user_message)
            
        console.print(f"[cyan]Identified {len(all_function_calls)} operations to process[/cyan]")
        
        # Group operations by priority and dependencies
        operation_groups = []
        current_group = []
        
        # Sort by priority if available
        sorted_calls = sorted(
            all_function_calls, 
            key=lambda x: x.get("priority", 3) if isinstance(x, dict) and "priority" in x else 3,
            reverse=True
        )
        
        for func_call in sorted_calls:
            current_group.append(func_call)
            if len(current_group) >= max_batch_size:
                operation_groups.append(current_group)
                current_group = []
                
        # Add any remaining operations
        if current_group:
            operation_groups.append(current_group)
            
        # Process each group of operations
        all_results = []
        successful_operations = set()
        failed_operations = []
        
        for group_idx, operation_group in enumerate(operation_groups):
            console.print(f"[cyan]Processing operation group {group_idx + 1} with {len(operation_group)} operations[/cyan]")
            
            # Execute the batch
            batch_results = execute_parallel_functions(operation_group, agent.tool_registry)
            
            # Track results
            for result in batch_results:
                all_results.append(result)
                
                # Track successful and failed operations for potential retry
                if result.get("status") == "completed":
                    successful_operations.add(result.get("function_name"))
                elif result.get("status") in ["failed", "timeout"]:
                    failed_operations.append(result)
            
            # Add results to conversation history
            result_message = {"role": "function", "content": json.dumps(batch_results, indent=2)}
            agent.conversation_history.append(result_message)
            
        # Retry failed operations if needed
        if failed_operations and len(successful_operations) > 0:
            console.print(f"[yellow]Attempting to retry {len(failed_operations)} failed operations[/yellow]")
            
            # Generate recovery strategy using successful operations as context
            recovery_prompt = [
                {"role": "system", "content": "You are an expert at recovering from failed operations. Analyze the failures and suggest fixes."},
                {"role": "user", "content": f"""Some operations failed during task execution. 
                Successful operations: {list(successful_operations)}
                
                Failed operations:
                {json.dumps(failed_operations, indent=2)}
                
                Suggest how to fix these operations or alternative approaches to achieve the same goal."""}
            ]
            
            recovery_response = together.chat.completions.create(
                messages=recovery_prompt,
                model=agent.model,
                temperature=0.3,
                            
            )
            
            # Extract any function calls from the recovery suggestion
            recovery_calls = parse_function_calls(recovery_response.choices[0].message.content)
            
            if recovery_calls:
                console.print(f"[cyan]Executing {len(recovery_calls)} recovery operations[/cyan]")
                recovery_results = execute_parallel_functions(recovery_calls, agent.tool_registry)
                all_results.extend(recovery_results)
                
                # Add recovery results to conversation history
                recovery_message = {"role": "function", "content": json.dumps(recovery_results, indent=2)}
                agent.conversation_history.append(recovery_message)
        
        # Generate a comprehensive final summary with analysis
        success_count = sum(1 for result in all_results if result.get("status") == "completed")
        failure_count = len(all_results) - success_count
        
        # Create a more structured summary prompt
        final_prompt = f"""I've completed the operations you requested. Here's a summary:

Operations completed: {success_count}/{len(all_results)} successful

Results:
{json.dumps(all_results, indent=2)}

Please provide:
1. A comprehensive summary of all work completed
2. Analysis of any failures or issues encountered
3. The final answer or solution to the original request
4. Any recommendations for follow-up actions"""

        agent.last_user_message = final_prompt
        agent.add_message("user", final_prompt)
        
        # Generate the final response with a focus on completeness
        response = together.chat.completions.create(
            messages=agent.conversation_history, 
            model=agent.model, 
            tool_choice="none",
            temperature=0.3,  # Lower temperature for more focused summary
            max_tokens=2048   # Allow for a comprehensive response
        )
        
        final_message = response.choices[0].message
        agent.conversation_history.append(final_message.model_dump())
        
        # Add execution metrics to memory if available
        if hasattr(agent, 'memory'):
            agent.memory.add_memory(
                f"Batch processed {len(all_results)} operations with {success_count} successes and {failure_count} failures.",
                {"type": "execution_metrics", "success_rate": success_count/len(all_results) if all_results else 0}
            )
            
        return final_message.content
        
    except Exception as e:
        console.print(f"[red]Error in batch processing: {str(e)}[/red]")
        console.print(traceback.format_exc())
        # Fallback to standard chat
        return agent.chat(user_message)

# =======================
# Together Agent
# =======================
class TogetherAgent:
    def __init__(self, model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", num_scouts=5):
        """Main TogetherAgent class with advanced capabilities including self-modification
        
        This agent can:
        1. Orchestrate specialized scout agents for parallel task execution
        2. Use vector memory for semantic recall
        3. Apply reinforcement learning for optimization
        4. Dynamically add/modify tools and capabilities at runtime
        5. Perform meta-cognitive reflection on its performance
        """
        self.api_key = os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set TOGETHER_API_KEY environment variable.")
        self.use_mock = "dummy_api_key_for_testing" in self.api_key
        if not self.use_mock:
            self.client = Together(api_key=self.api_key)
            
            # Initialize advanced memory system
            self.memory = VectorMemory(embedding_model, vector_store)
        else:
            from types import SimpleNamespace
            def mock_completion(**kwargs):
                if "python" in kwargs.get("prompt", "").lower() or (isinstance(kwargs.get("messages", []), list) and any("python" in str(m.get("content", "")).lower() for m in kwargs.get("messages", []))):
                    mock_text = """Here's Python code to perform the requested task:

<|python_start|>
print("Hello, world!")
<|python_end|>"""
                    if "messages" in kwargs:
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=mock_text, model_dump=lambda: {"role": "assistant", "content": mock_text}))])
                    else:
                        return SimpleNamespace(choices=[SimpleNamespace(text=mock_text)])
                else:
                    mock_text = "I'm a mock response from the agent."
                    if "messages" in kwargs:
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=mock_text, model_dump=lambda: {"role": "assistant", "content": mock_text}))])
                    else:
                        return SimpleNamespace(choices=[SimpleNamespace(text=mock_text)])
            self.client = SimpleNamespace(
                completions=SimpleNamespace(create=mock_completion),
                chat=SimpleNamespace(completions=SimpleNamespace(create=mock_completion))
            )
            console.print("[yellow]Running in mock mode with test responses[/yellow]")
        self.model = model
        self.tool_registry = ToolRegistry()
        self.conversation_history = []
        self.reflection_notes = []
        self.enable_logprobs = False
        self.enable_planning = True
        self.planning_session = None
        
        # Initialize the agent orchestrator with scout agents
        console.print(f"[cyan]Initializing Agent Orchestrator with {num_scouts} scout agents...[/cyan]")
        self.agent_orchestrator = AgentOrchestrator(num_scouts=num_scouts, model=model)
        console.print(f"[green]Agent Orchestrator initialized with {len(self.agent_orchestrator.scouts)} scout agents[/green]")
        
        # Initialize advanced capabilities
        self._initialize_advanced_capabilities()
        
        # Initialize self-modification capabilities
        self.self_modification = {
            "enabled": True,
            "safety_checks": True,
            "modification_history": [],
            "performance_metrics": {},
            "hot_reload_enabled": True,
            "last_modification_time": time.time(),
            "modification_count": 0,
            "pending_modifications": [],
            "allowed_modules": [
                "core", "tools", "memory", "planning", "execution"
            ]
        }
        
        # Create function registry for runtime modifications
        self.dynamic_functions = {}
        
        # Set up code analyzer for runtime analysis
        try:
            import ast
            self.code_analyzer_available = True
        except ImportError:
            self.code_analyzer_available = False
            console.print("[yellow]Warning: AST module not available, code analysis capabilities will be limited[/yellow]")
        
        # Keep task_processor for backward compatibility
        self.task_processor = AsyncTaskProcessor()
        self.interaction_memory = []
        self.environment_state = {
            "system": self._detect_system_info(),
            "user_behavior": {"message_count": 0, "avg_message_length": 0, "technical_level": "medium", "detected_preferences": []},
            "conversation_context": {"topic_clusters": [], "sentiment": "neutral", "task_oriented": True},
            "task_complexity": {"current_level": "medium", "adaptations_made": []},
            "last_analysis_time": time.time()
        }
        
    def _initialize_advanced_capabilities(self):
        """Initialize advanced capabilities for the agent"""
        # Set up continuous learning
        self.continuous_learning = {
            "enabled": True,
            "learning_rate": 0.1,
            "adaptation_threshold": 0.7,
            "performance_history": [],
            "last_optimization_time": time.time()
        }
        
        # Set up advanced planning
        self.planning_system = {
            "enabled": True,
            "planning_horizon": 5,  # Number of steps to plan ahead
            "execution_monitoring": True,
            "plan_adaptation_threshold": 0.5,
            "current_plans": []
        }
        
        # Set up meta-cognition
        self.meta_cognition = {
            "enabled": True,
            "reflection_frequency": 5,  # Reflect every N interactions
            "last_reflection_time": time.time(),
            "insights": [],
            "self_improvement_goals": [
                "Improve task decomposition accuracy",
                "Enhance collaboration between agents",
                "Optimize memory retrieval relevance"
            ]
        }
        
        # Set up dynamic reprompting for premium experience
        self.reprompting = {
            "enabled": True,
            "max_reprompt_attempts": 2,
            "partial_response_threshold": 0.7,  # Threshold for detecting partial responses
            "premium_experience": True,  # Flag for premium users ($10,000/day)
            "reprompt_stats": {
                "total_reprompts": 0,
                "successful_fixes": 0,
                "last_reprompt_time": None
            }
        }
        
        # Register advanced capabilities with scouts
        for scout_id, scout in self.agent_orchestrator.scouts.items():
            # Enable reinforcement learning for all scouts
            scout.reinforcement_learning["learning_enabled"] = True
            
            # Set initial exploration rates based on specialization
            if scout.specialization == "creative":
                scout.reinforcement_learning["exploration_rate"] = 0.3  # Higher exploration for creative agents
            elif scout.specialization == "research":
                scout.reinforcement_learning["exploration_rate"] = 0.2  # Medium exploration for research
            else:
                scout.reinforcement_learning["exploration_rate"] = 0.1  # Lower for other specializations
                
        # Initialize the skill registry
        self.agent_orchestrator.update_skill_registry()
        
        # Store initialization in memory
        self.memory.add_memory(
            f"Agent initialized with {len(self.agent_orchestrator.scouts)} scout agents and advanced capabilities enabled.",
            {"category": "system", "type": "initialization"}
        )
        # For Llama-4 models, add a system prompt with explicit instructions
        self.is_llama4 = "llama-4" in self.model.lower()
        if self.is_llama4:
            system_content = (
                "You are a helpful AI assistant that can dynamically use tools to accomplish tasks and "
                "orchestrate a team of specialized scout agents to solve complex problems in parallel. "
                "You can issue multiple function calls in parallel to efficiently solve complex tasks. "
                "When you need to make function calls, you can return them in the format: "
                "[function_name1(param1=value1, param2=value2), function_name2(param1=value1)]"
            )
        else:
            system_content = (
                "You are a helpful AI assistant that can dynamically use tools to accomplish tasks and "
                "orchestrate a team of specialized scout agents to solve complex problems in parallel. "
                "You can issue function calls using the defined tools."
            )
        self.system_message = {"role": "system", "content": system_content}
        self.conversation_history.append(self.system_message)

    def _detect_system_info(self):
        return {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "os_name": os.name,
            "cpu_count": os.cpu_count() or "unknown",
            "terminal_size": {"columns": 80, "lines": 24},
            "environment_variables": {"PATH_exists": "PATH" in os.environ, "HOME_exists": "HOME" in os.environ, "LANG": os.environ.get("LANG", "unknown")}
        }

    def update_environment_user_behavior(self, message):
        message_text = ""
        if isinstance(message, list):
            for item in message:
                if item.get('type') == 'text':
                    message_text += item.get('text', '')
        else:
            message_text = message
        self.environment_state["user_behavior"]["message_count"] += 1
        current_count = self.environment_state["user_behavior"]["message_count"]
        current_avg = self.environment_state["user_behavior"]["avg_message_length"]
        new_length = len(message_text)
        new_avg = ((current_avg * (current_count - 1)) + new_length) / current_count
        self.environment_state["user_behavior"]["avg_message_length"] = new_avg
        technical_terms = ["code", "function", "class", "method", "algorithm", "implementation", "api", "database", "query", "json", "xml", "http", "rest", "async", "thread", "concurrency", "memory", "cpu", "processor", "git", "repository", "commit", "merge", "branch"]
        technical_count = sum(1 for term in technical_terms if term.lower() in message_text.lower())
        if technical_count > 5 or (technical_count / max(1, len(message_text.split())) > 0.1):
            self.environment_state["user_behavior"]["technical_level"] = "high"
        elif technical_count > 2:
            self.environment_state["user_behavior"]["technical_level"] = "medium"
        else:
            if current_count > 3:
                self.environment_state["user_behavior"]["technical_level"] = "low"
        has_images = isinstance(message, list) and any(item.get('type') == 'image_url' for item in message)
        if has_images and "multimodal" not in self.environment_state["user_behavior"]["detected_preferences"]:
            self.environment_state["user_behavior"]["detected_preferences"].append("multimodal")

    def _generate_tools_list(self) -> str:
        """Generate a nicely formatted list of available tools"""
        tools_info = self.tool_registry._list_available_functions()
        categories = tools_info.get("categories", {})
        
        response = "# Available Tools\n\n"
        
        for category, functions in sorted(categories.items()):
            response += f"## {category}\n\n"
            
            for func in sorted(functions, key=lambda x: x["name"]):
                name = func["name"]
                description = func["description"]
                params = func["parameters"]
                
                response += f"### {name}\n"
                response += f"{description}\n\n"
                
                if params:
                    response += "**Parameters:**\n"
                    for param in params:
                        response += f"- {param}\n"
                    response += "\n"
            
            response += "\n"
        
        response += "You can use these tools by asking me to perform specific tasks. For example:\n"
        response += "- \"Search the web for the latest news about AI\"\n"
        response += "- \"Get the weather in New York\"\n"
        response += "- \"Create a Python function to calculate Fibonacci numbers\"\n"
        
        return response
    
    def _find_similar_tool(self, name: str) -> Optional[str]:
        """Find a similar tool name in the registry using fuzzy matching."""
        available_tools = self.tool_registry.get_available_tools()
        
        # Check for common aliases
        aliases = {
            "list_functions": "list_available_functions",
            "get_time": "get_current_datetime",
            "get_date": "get_current_datetime",
            "datetime": "get_current_datetime",
            "search": "web_search",
            "weather": "get_weather",
            "weather_api": "get_weather",
            "get_weather_data": "get_weather",
            "weather_data": "get_weather",
            "check_weather": "get_weather",
            "execute": "execute_python",
            "run_python": "execute_python",
            "python": "execute_python",
            "read": "read_file",
            "write": "write_file",
            "ls": "list_directory",
            "dir": "list_directory",
        }
        
        if name.lower() in aliases:
            alias = aliases[name.lower()]
            if alias in available_tools:
                return alias
        
        # Try exact match with different casing
        for tool in available_tools:
            if tool.lower() == name.lower():
                return tool
        
        # Try prefix match
        for tool in available_tools:
            if tool.lower().startswith(name.lower()) or name.lower().startswith(tool.lower()):
                return tool
        
        # Try substring match
        for tool in available_tools:
            if name.lower() in tool.lower() or tool.lower() in name.lower():
                return tool
        
        return None
    
    def search_similar_images(self, image_url: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search for similar images in the conversation history"""
        if not hasattr(self, 'memory') or not self.memory.available:
            return []
        
        # Try to use the image processor for enhanced image similarity search
        try:
            from image_processor import ImageProcessor
            image_processor = ImageProcessor()
            
            # Get all image memories
            all_images = [item for item in self.memory.memory_items 
                         if item.get("metadata", {}).get("type") == "image"]
            
            if not all_images:
                return []
            
            # Process the query image
            try:
                query_image_features = image_processor.extract_image_features(
                    image_processor.process_image(image_url)
                )
                
                # Calculate similarity for each image in memory
                similarities = []
                for img_item in all_images:
                    try:
                        # If we already have features stored, use them
                        if "features" in img_item.get("metadata", {}):
                            stored_features = img_item["metadata"]["features"]
                            # Convert to numpy array if needed
                            if isinstance(stored_features, list):
                                stored_features = np.array(stored_features)
                            
                            # Calculate similarity
                            similarity = np.dot(query_image_features, stored_features)
                            
                        # Otherwise, process the image and calculate similarity
                        else:
                            img_content = img_item["content"]
                            similarity = image_processor.compare_images(image_url, img_content)
                            
                        similarities.append((similarity, img_item))
                    except Exception:
                        # Skip images that can't be processed
                        continue
                
                # Sort by similarity (highest first)
                similarities.sort(reverse=True, key=lambda x: x[0])
                
                # Return top matches with similarity scores
                results = []
                for similarity, item in similarities[:limit]:
                    result = item.copy()
                    result["relevance_score"] = float(similarity)
                    result["access_count"] = result.get("access_count", 0) + 1
                    results.append(result)
                
                return results
                
            except Exception as e:
                console.print(f"[yellow]Advanced image similarity failed: {e}. Using basic search.[/yellow]")
                # Fall back to basic search
                return self.memory.search_memory(image_url, limit=limit, include_images=True)
                
        except ImportError:
            # Fall back to the original method if image_processor.py is not available
            return self.memory.search_memory(image_url, limit=limit, include_images=True)
    
    def _get_function_suggestion(self, function_name: str, arguments: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Generate a helpful suggestion for fixing a function call error."""
        if not self.tool_registry.has_tool(function_name):
            similar_tool = self._find_similar_tool(function_name)
            if similar_tool:
                return f"Use '{similar_tool}' instead of '{function_name}'"
            else:
                available = ", ".join(self.tool_registry.get_available_tools()[:5])
                return f"Function '{function_name}' not found. Available functions include: {available}..."
        
        # Get the function spec to check parameters
        function_spec = None
        for name, spec in self.tool_registry.functions.items():
            if name == function_name:
                function_spec = spec
                break
        
        if not function_spec:
            return "Unknown function"
        
        # Check for missing required parameters
        required_params = function_spec.parameters.get("required", [])
        missing_params = [param for param in required_params if param not in arguments]
        
        if missing_params:
            return f"Missing required parameters: {', '.join(missing_params)}"
        
        # Check for invalid parameters
        properties = function_spec.parameters.get("properties", {})
        invalid_params = [param for param in arguments if param not in properties]
        
        if invalid_params:
            valid_params = list(properties.keys())
            return f"Invalid parameters: {', '.join(invalid_params)}. Valid parameters are: {', '.join(valid_params)}"
        
        # If we have an error message, return it
        if "error" in result:
            return f"Error: {result['error']}"
        
        return "Unknown error"
    
    def extract_python_code(self, text: str) -> List[str]:
        """Extract Python code blocks from text with better handling of formatting issues"""
        try:
            # Try to use the new CodeExtractor if available
            from code_extractor import CodeExtractor
            extractor = CodeExtractor()
            return extractor.extract_python_code(text)
        except ImportError:
            # Fall back to the old method if CodeExtractor is not available
            code_blocks = extract_python_code(text)
            
            # Process each code block to fix common formatting issues
            fixed_blocks = []
            for block in code_blocks:
                lines = block.split('\n')
                fixed_lines = []
                i = 0
                while i < len(lines):
                    line = lines[i]
                    # Skip purely decorative lines
                    if all(c in '┏━┓┃┗┛ ' for c in line):
                        i += 1
                        continue
                    
                    # Fix compressed if statements
                    if ' if ' in line and ': ' in line and not line.strip().startswith('if'):
                        parts = line.split(' if ')
                        pre = parts[0].strip()
                        rest = parts[1].strip()
                        if_parts = rest.split(': ')
                        if len(if_parts) == 2:
                            fixed_lines.append(f"{pre} if {if_parts[0]}:")
                            fixed_lines.append(f"    {if_parts[1]}")
                            i += 1
                            continue
                    
                    # Fix compressed function definitions
                    if 'def ' in line and '(' in line and ')' in line and ': ' in line and not line.strip().startswith('def'):
                        parts = line.split('def ')
                        pre = parts[0].strip()
                        rest = 'def ' + parts[1].strip()
                        if pre:
                            fixed_lines.append(pre)
                        fixed_lines.append(rest)
                        i += 1
                        continue
                    
                    # Add the line as is if no fixes needed
                    fixed_lines.append(line)
                    i += 1
                
                fixed_blocks.append('\n'.join(fixed_lines))
            
            return fixed_blocks
            
    def _check_and_fix_partial_response(self, response: str) -> str:
        """
        Check if a response appears to be cut off or incomplete and fix it by reprompting.
        This ensures premium users ($10,000/day) receive complete, high-quality responses.
        
        Args:
            response: The original response from the model
            
        Returns:
            The complete response, either original or fixed through reprompting
        """
        if not hasattr(self, 'reprompting') or not self.reprompting.get("enabled", False):
            return response
            
        # Skip empty responses
        if not response or len(response.strip()) < 10:
            return response
            
        # Patterns that suggest a response is incomplete
        incomplete_patterns = [
            # Sentences that end abruptly
            r'(?<!\.)$',                           # Doesn't end with a period
            r'(?<=[a-zA-Z0-9])[,;:]$',             # Ends with comma, semicolon, or colon
            r'\.\.\.$',                            # Ends with ellipsis
            r'(?<=[a-zA-Z0-9])\s*$',               # Ends with a word without punctuation
            
            # Structural incompleteness
            r'```[^`]*$',                          # Unclosed code block
            r'\([^)]*$',                           # Unclosed parenthesis
            r'\[[^\]]*$',                          # Unclosed bracket
            r'"{1}[^"]*$',                         # Unclosed quote
            
            # Semantic incompleteness
            r'(?i)for example[,:]?\s*$',           # "For example" without the example
            r'(?i)such as[,:]?\s*$',               # "Such as" without examples
            r'(?i)these (?:include|are)[,:]?\s*$', # "These include" without items
            r'(?i)steps?[,:]?\s*$',                # "Steps:" without steps
            r'(?i)here\'s[,:]?\s*$',               # "Here's" without what follows
            
            # Numbered lists that end abruptly
            r'(?i)\d+\.\s*[^.]*$',                 # Numbered item without completion
            r'(?i)\d+\.\s*[^.]*\n$',               # Numbered item ending with newline
            
            # Incomplete function calls
            r'(?i)function[^(]*\([^)]*$',          # Function call with unclosed parenthesis
            r'(?i)tool[^(]*\([^)]*$',              # Tool call with unclosed parenthesis
        ]
        
        # Check if response matches any incomplete pattern
        is_incomplete = False
        for pattern in incomplete_patterns:
            if re.search(pattern, response):
                is_incomplete = True
                break
                
        # Additional heuristics for incompleteness
        sentence_count = len(re.split(r'[.!?]+', response))
        word_count = len(response.split())
        
        # Very short responses with few sentences might be incomplete
        if sentence_count < 2 and word_count < 30:
            is_incomplete = True
            
        # Check for abrupt ending in the middle of a list
        list_items = re.findall(r'(?:^|\n)\s*(?:\d+\.|[\*\-•])\s+[^\n]+', response)
        if list_items and len(list_items) < 3 and list_items[-1].strip()[-1] not in '.!?':
            is_incomplete = True
            
        # If response seems complete, return it as is
        if not is_incomplete:
            return response
            
        # Log the detection of an incomplete response
        console.print("[yellow]Detected incomplete response. Reprompting for premium user experience...[/yellow]")
        
        # Update reprompt statistics
        if hasattr(self, 'reprompting') and 'reprompt_stats' in self.reprompting:
            self.reprompting['reprompt_stats']['total_reprompts'] += 1
            self.reprompting['reprompt_stats']['last_reprompt_time'] = time.time()
        
        # Create a reprompt to complete the response
        max_attempts = self.reprompting.get("max_reprompt_attempts", 2)
        
        for attempt in range(max_attempts):
            try:
                # Remove the incomplete response from conversation history
                if self.conversation_history and self.conversation_history[-1].get("role") == "assistant":
                    self.conversation_history.pop()
                
                # Create a reprompt that asks for completion
                reprompt_message = {
                    "role": "user", 
                    "content": f"Your previous response was cut off. Please provide a complete response to my original question. Make sure to include all relevant details and finish your thoughts completely."
                }
                
                # Add the reprompt to conversation history
                self.conversation_history.append(reprompt_message)
                
                # Generate a new, complete response
                if self.model.startswith("meta-llama/Llama-4"):
                    new_response = self.client.completions.create(
                        model=self.model,
                        prompt=self.format_llama4_prompt() + "\nAssistant: ",
                        max_tokens=4096,
                        stop=["<|eot|>"]
                    )
                    new_content = new_response.choices[0].text + "<|eot|>"
                    new_content = new_content.rstrip("<|eot|>")
                else:
                    new_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        max_tokens=4096
                    )
                    new_content = new_response.choices[0].message.content
                
                # Add the new response to conversation history
                self.add_message("assistant", new_content)
                
                # Check if the new response is also incomplete
                still_incomplete = False
                for pattern in incomplete_patterns:
                    if re.search(pattern, new_content):
                        still_incomplete = True
                        break
                
                if not still_incomplete:
                    # Update success statistics
                    if hasattr(self, 'reprompting') and 'reprompt_stats' in self.reprompting:
                        self.reprompting['reprompt_stats']['successful_fixes'] += 1
                    
                    console.print("[green]Successfully generated complete response after reprompting[/green]")
                    return new_content
                
                # If we're on the last attempt and it's still incomplete, combine the responses
                if attempt == max_attempts - 1:
                    combined_response = response + "\n\n" + new_content
                    return combined_response
                    
            except Exception as e:
                console.print(f"[red]Error during reprompting (attempt {attempt+1}): {str(e)}[/red]")
                # If reprompting fails, return the original response
                return response
        
        # If all reprompt attempts fail, return the original response
        return response

    def process_tool_calls(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
                console.print(f"[red]Error parsing arguments for {function_name}: {tool_call.function.arguments}[/red]")
            console.print(f"[cyan]Calling function: [bold]{function_name}[/bold][/cyan]")
            console.print(f"[cyan]With arguments:[/cyan] {json.dumps(arguments, indent=2)}")
            result = self.tool_registry.call_function(function_name, arguments)
            self.conversation_history.append({
                "role": "tool",
                "tool_call_id": getattr(tool_call, "id", "n/a"),
                "name": function_name,
                "content": json.dumps(result, indent=2)
            })
            results.append({"function_name": function_name, "arguments": arguments, "result": result})
        return results

    def format_llama4_prompt(self) -> str:
        # Basic formatting: concatenate conversation history
        prompt = "<|begin_of_text|>"
        for msg in self.conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role != "tool":
                prompt += f"<|header_start|>{role}<|header_end|>\n{content}"
                if role != "system" and not content.endswith("<|eot|>"):
                    prompt += "<|eot|>"
            else:
                prompt += f"\nTool output: {content}\n"
        prompt += "<|header_start|>assistant<|header_end|>"
        return prompt

    def process_image_in_message(self, message):
        if isinstance(message, str):
            # Check for clipboard paste command
            if message.strip() == "/paste":
                return self._paste_from_clipboard()
                
            # Process image URLs
            image_urls = re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)', message)
            if not image_urls:
                return message
            for url in image_urls:
                message = message.replace(url, '')
            text_content = message.strip()
            multimodal = []
            if text_content:
                multimodal.append({"type": "text", "text": text_content})
            
            # Try to use the image processor for enhanced processing
            try:
                from image_processor import ImageProcessor
                image_processor = ImageProcessor()
                
                for url in image_urls:
                    # Process the image to extract features
                    try:
                        # Process image and store features in memory
                        image_tensor = image_processor.process_image(url)
                        features = image_processor.extract_image_features(image_tensor)
                        
                        # Add to multimodal message
                        multimodal.append({"type": "image_url", "image_url": {"url": url}})
                        
                        # Add to interaction memory with enhanced metadata
                        self.interaction_memory.append({
                            "type": "image", 
                            "url": url, 
                            "timestamp": time.time(),
                            "features": features.tolist() if hasattr(features, "tolist") else features,
                            "processed": True
                        })
                        
                        # Add image to memory for retrieval with enhanced metadata
                        if hasattr(self, 'memory'):
                            self.memory.add_memory(url, {
                                "type": "image", 
                                "source": "user_input",
                                "features": features.tolist() if hasattr(features, "tolist") else features,
                                "processed_with": "ImageProcessor"
                            })
                    except Exception as e:
                        # Fall back to basic processing if advanced processing fails
                        console.print(f"[yellow]Warning: Advanced image processing failed: {e}. Using basic processing.[/yellow]")
                        multimodal.append({"type": "image_url", "image_url": {"url": url}})
                        self.interaction_memory.append({"type": "image", "url": url, "timestamp": time.time()})
                        
                        # Add image to memory for retrieval
                        if hasattr(self, 'memory'):
                            self.memory.add_memory(url, {"type": "image", "source": "user_input"})
            except ImportError:
                # Fall back to the original method if image_processor.py is not available
                for url in image_urls:
                    multimodal.append({"type": "image_url", "image_url": {"url": url}})
                    self.interaction_memory.append({"type": "image", "url": url, "timestamp": time.time()})
                    
                    # Add image to memory for retrieval
                    if hasattr(self, 'memory'):
                        self.memory.add_memory(url, {"type": "image", "source": "user_input"})
                
            return multimodal
        return message
        
    def _paste_from_clipboard(self):
        """Paste image from clipboard and convert to base64 for the model"""
        try:
            # Try to use the image processor for enhanced clipboard handling
            try:
                from image_processor import ImageProcessor
                image_processor = ImageProcessor()
                
                # Use the image processor to save the clipboard image
                img_path, img_base64 = image_processor.save_image_from_clipboard()
                
                console.print(f"[green]Image saved to {img_path}[/green]")
                
                # Process the image to extract features
                try:
                    # Process image and store features in memory
                    image_tensor = image_processor.process_image(img_base64)
                    features = image_processor.extract_image_features(image_tensor)
                    
                    # Create multimodal message
                    multimodal = [
                        {"type": "text", "text": "Image pasted from clipboard:"},
                        {"type": "image_url", "image_url": {"url": img_base64}}
                    ]
                    
                    # Add to memory with enhanced metadata
                    if hasattr(self, 'memory'):
                        self.memory.add_memory(
                            f"{img_base64[:30]}...", 
                            {
                                "type": "image", 
                                "source": "clipboard", 
                                "local_path": str(img_path),
                                "features": features.tolist() if hasattr(features, "tolist") else features,
                                "processed_with": "ImageProcessor"
                            }
                        )
                    
                    self.interaction_memory.append({
                        "type": "image", 
                        "source": "clipboard", 
                        "timestamp": time.time(),
                        "local_path": str(img_path),
                        "features": features.tolist() if hasattr(features, "tolist") else features,
                        "processed": True
                    })
                    
                    return multimodal
                    
                except Exception as proc_err:
                    console.print(f"[yellow]Warning: Advanced image processing failed: {proc_err}. Using basic processing.[/yellow]")
                    # Fall back to basic processing if feature extraction fails
                    multimodal = [
                        {"type": "text", "text": "Image pasted from clipboard:"},
                        {"type": "image_url", "image_url": {"url": img_base64}}
                    ]
                    
                    # Add to memory
                    if hasattr(self, 'memory'):
                        self.memory.add_memory(
                            f"{img_base64[:30]}...", 
                            {"type": "image", "source": "clipboard", "local_path": str(img_path)}
                        )
                    
                    self.interaction_memory.append({
                        "type": "image", 
                        "source": "clipboard", 
                        "timestamp": time.time(),
                        "local_path": str(img_path)
                    })
                    
                    return multimodal
                
            except ImportError:
                # Fall back to the original method if image_processor.py is not available
                # Try to get image from clipboard
                image = ImageGrab.grabclipboard()
                
                if image is None:
                    # If no image, try to get text
                    text = pyperclip.paste()
                    if text:
                        return text
                    else:
                        return "No image or text found in clipboard."
                
                # Process the image
                console.print("[cyan]Image found in clipboard[/cyan]")
                
                # Save image to a temporary file with a unique name
                temp_dir = Path(tempfile.gettempdir())
                timestamp = int(time.time())
                img_path = temp_dir / f"clipboard_image_{timestamp}.png"
                image.save(img_path)
                
                console.print(f"[green]Image saved to {img_path}[/green]")
                
                # Convert to base64 for the model
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                # Create multimodal message
                multimodal = [
                    {"type": "text", "text": "Image pasted from clipboard:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
                
                # Add to memory
                if hasattr(self, 'memory'):
                    self.memory.add_memory(
                        f"data:image/png;base64,{img_base64[:20]}...", 
                        {"type": "image", "source": "clipboard", "local_path": str(img_path)}
                    )
                
                self.interaction_memory.append({
                    "type": "image", 
                    "source": "clipboard", 
                    "timestamp": time.time(),
                    "local_path": str(img_path)
                })
                
                return multimodal
                
        except Exception as e:
            console.print(f"[red]Error pasting from clipboard: {str(e)}[/red]")
            return f"Error pasting from clipboard: {str(e)}"

    def add_message(self, role: str, content):
        if hasattr(self, "is_llama4") and self.is_llama4 and role == "user" and isinstance(content, str):
            content = self.process_image_in_message(content)
        if role in ["user", "assistant"] and isinstance(content, str) and hasattr(self, "task_processor"):
            extracted = self.task_processor.extract_urls(content)
            if extracted.urls:
                self.task_processor.add_urls_to_process(extracted.urls)
                console.print(f"[dim]Extracted {len(extracted.urls)} URLs for background processing[/dim]")
        message = {"role": role}
        if isinstance(content, list):
            message["content"] = content
            
            # For multimodal content, add each part to memory
            if hasattr(self, 'memory'):
                for item in content:
                    if item.get("type") == "text":
                        self.memory.add_memory(item.get("text", ""), {"type": "text", "role": role})
                    elif item.get("type") == "image_url" and "url" in item.get("image_url", {}):
                        self.memory.add_memory(item["image_url"]["url"], {"type": "image", "role": role})
        else:
            if hasattr(self, "is_llama4") and self.is_llama4 and role != "system" and not content.endswith("<|eot|>"):
                content += "<|eot|>"
            message["content"] = content
            if role == "user":
                self.update_environment_user_behavior(content)
                
            # Add text content to memory
            if hasattr(self, 'memory') and content and role in ["user", "assistant"]:
                self.memory.add_memory(content, {"type": "text", "role": role})
                
        self.conversation_history.append(message)

    def _check_multi_location_weather_query(self, query: str) -> Optional[str]:
        """Check if the query is asking for weather in multiple locations"""
        # Common patterns for multi-location weather queries
        patterns = [
            r"weather\s+in\s+([A-Za-z\s,]+)\s+and\s+([A-Za-z\s,]+)",
            r"weather\s+for\s+([A-Za-z\s,]+)\s+and\s+([A-Za-z\s,]+)",
            r"weather\s+(?:in|for|at)\s+([A-Za-z\s,]+)(?:\s*,\s*|\s+and\s+)([A-Za-z\s,]+)"
        ]
        
        locations = []
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                for match in matches:
                    locations.extend([loc.strip() for loc in match if loc.strip()])
                break
        
        # If we found multiple locations, process them
        if len(locations) > 1:
            console.print(f"[cyan]Detected weather query for multiple locations: {locations}[/cyan]")
            
            # Call the multiple weather function
            result = self.tool_registry._get_multiple_weather(locations)
            
            if result.get("success", False):
                # Format the results into a nice response
                response = f"Here's the current weather for the locations you asked about:\n\n"
                
                for location, weather_data in result.get("results", {}).items():
                    if weather_data.get("success", False):
                        response += f"**{location}**: {weather_data.get('current_condition', 'Unknown').capitalize()}, " \
                                   f"{weather_data.get('temperature_f', 'N/A')}°F ({weather_data.get('temperature_c', 'N/A')}°C), " \
                                   f"humidity {weather_data.get('humidity', 'N/A')}%, " \
                                   f"wind speed {weather_data.get('wind_speed_mph', 'N/A')} mph.\n\n"
                        
                        # Add tomorrow's forecast
                        forecast = weather_data.get("forecast", [])
                        if forecast and len(forecast) > 0:
                            tomorrow = forecast[0]
                            response += f"Tomorrow in {location}: {tomorrow.get('condition', 'unknown').capitalize()}, " \
                                       f"high of {tomorrow.get('high_f', 'N/A')}°F, " \
                                       f"low of {tomorrow.get('low_f', 'N/A')}°F, " \
                                       f"{tomorrow.get('precipitation_chance', 'N/A')}% chance of precipitation.\n\n"
                    else:
                        response += f"**{location}**: Could not retrieve weather data. Error: {weather_data.get('error', 'Unknown error')}\n\n"
                
                # Add the response to conversation history
                self.add_message("assistant", response)
                
                return response
        
        return None
        
    def generate_response(self, user_input):
        # Store last user message and add to conversation history
        if isinstance(user_input, list):
            for item in user_input:
                if item.get("type") == "image_url":
                    self.interaction_memory.append({"type": "image", "url": item["image_url"]["url"], "timestamp": time.time()})
        self.last_user_message = user_input
        self.add_message("user", user_input)
        
        # Check for multi-location weather queries
        if isinstance(user_input, str) and "weather" in user_input.lower():
            multi_location_result = self._check_multi_location_weather_query(user_input)
            if multi_location_result:
                return multi_location_result
        
        # Note: We don't need to explicitly store in memory here anymore since add_message now handles it
        
        # Both continuous learning and meta-cognitive reflection are optional features
        # that may not be implemented in all agent versions
        
        # Special handling for tool listing requests
        if isinstance(user_input, str) and user_input.lower().strip() in [
            "list tools", "list your tools", "what tools do you have", 
            "show tools", "show available tools", "what can you do",
            "list functions", "list available functions", "what functions do you have"
        ]:
            return self._generate_tools_list()
            
        tools = self.tool_registry.get_openai_tools_format()
        # Periodic environment analysis
        user_msg_count = len([msg for msg in self.conversation_history if msg.get("role") == "user"])
        if user_msg_count == 1 or user_msg_count % 3 == 0:
            self.tool_registry._analyze_environment(aspect="all")
            env = self.environment_state
            if env["user_behavior"]["technical_level"] == "high" and env["task_complexity"]["current_level"] == "high":
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="increase_technical_detail",
                    reason="High technical level and complex task",
                    system_prompt_update="Provide detailed technical explanations and comprehensive code examples. Multiple function calls may be issued within one turn."
                )
            elif env["user_behavior"]["technical_level"] == "low" and env["conversation_context"]["sentiment"] == "negative":
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="simplify_explanations",
                    reason="User struggling with technical content",
                    system_prompt_update="Simplify explanations and offer step-by-step guidance."
                )
            elif env["conversation_context"]["task_oriented"] and env["task_complexity"]["current_level"] == "medium":
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="focus_on_practical_solutions",
                    reason="Task-focused user with moderate complexity",
                    system_prompt_update="Focus on practical solutions and step-by-step procedures."
                )
            elif "multimodal" in env["user_behavior"]["detected_preferences"]:
                self.tool_registry._adapt_to_environment(
                    adaptation_strategy="optimize_multimodal",
                    reason="User uses multimodal features",
                    system_prompt_update="Pay special attention to visual content; provide detailed analysis of images."
                )
        while True:
            try:
                if self.model.startswith("meta-llama/Llama-4"):
                    has_multimodal = any(isinstance(msg.get("content"), list) for msg in self.conversation_history)
                    if has_multimodal:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=self.conversation_history,
                            tools=tools,
                            tool_choice="auto",
                            
                        )
                        assistant_message = response.choices[0].message
                        self.conversation_history.append(assistant_message.model_dump())
                        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                            results = self.process_tool_calls(assistant_message.tool_calls)
                            for res in results:
                                console.print(f"[green]Result from {res['function_name']}:[/green]")
                                console.print(json.dumps(res['result'], indent=2))
                            final_response = self.client.chat.completions.create(
                                model=self.model,
                                messages=self.conversation_history,
                                tools=tools,
                                tool_choice="none",
                                max_tokens=4096
                            )
                            final_message = final_response.choices[0].message
                            self.conversation_history.append(final_message.model_dump())
                            return final_message.content
                        return assistant_message.content
                    else:
                        formatted_prompt = self.format_llama4_prompt()
                        params = {"model": self.model, "prompt": formatted_prompt, "stop": ["<|eot|>"]}
                        if self.is_llama4 and self.enable_logprobs:
                            params["logprobs"] = 1
                        response = self.client.completions.create(**params)
                        assistant_content = response.choices[0].text + "<|eot|>"
                        assistant_message = {"role": "assistant", "content": assistant_content}
                        if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                            logprobs_data = response.choices[0].logprobs
                            assistant_message["logprobs"] = {"tokens": logprobs_data.tokens, "token_logprobs": logprobs_data.token_logprobs}
                            if len(logprobs_data.token_logprobs) > 0:
                                avg_logprob = sum(lp for lp in logprobs_data.token_logprobs if lp is not None) / len(logprobs_data.token_logprobs)
                                assistant_message["avg_logprob"] = avg_logprob
                        self.conversation_history.append(assistant_message)
                        # Check for potential function calls using a more comprehensive detection
                        has_function_call = (
                            ("[" in assistant_content and "(" in assistant_content and ")" in assistant_content) or 
                            "<function=" in assistant_content or 
                            "\"type\": \"function\"" in assistant_content or
                            "{\"name\":" in assistant_content or
                            "<|python_start|>" in assistant_content
                        )
                        
                        if has_function_call:
                            function_calls = parse_function_calls(assistant_content)
                            if function_calls:
                                results = []
                                all_successful = True
                                max_retries = 2  # Allow up to 2 retries for failed function calls
                                
                                # First pass: try to execute all functions
                                for fc in function_calls:
                                    name = fc["name"]
                                    args = fc["arguments"]
                                    
                                    # Try to find similar function if exact match not found
                                    if not self.tool_registry.has_tool(name):
                                        similar_tool = self._find_similar_tool(name)
                                        if similar_tool:
                                            console.print(f"[yellow]Function '{name}' not found, using similar function '{similar_tool}'[/yellow]")
                                            name = similar_tool
                                        else:
                                            error_msg = f"Function '{name}' not found in registry"
                                            console.print(f"[red]{error_msg}[/red]")
                                            result = {"error": error_msg, "success": False, "available_tools": self.tool_registry.get_available_tools()[:5]}
                                            all_successful = False
                                            self.conversation_history.append({"role": "tool", "name": name, "content": json.dumps(result, indent=2)})
                                            results.append({"function_name": name, "arguments": args, "result": result})
                                            continue
                                    
                                    # Execute the function
                                    console.print(f"[cyan]Calling function: [bold]{name}[/bold][/cyan]")
                                    console.print(f"[cyan]With arguments:[/cyan] {json.dumps(args, indent=2)}")
                                    result = self.tool_registry.call_function(name, args)
                                    
                                    # Check if there was an error
                                    if "error" in result:
                                        all_successful = False
                                        console.print(f"[red]Error in {name}: {result.get('error')}[/red]")
                                    else:
                                        console.print(f"[green]Result from {name}:[/green]")
                                        console.print(json.dumps(result, indent=2))
                                    
                                    self.conversation_history.append({"role": "tool", "name": name, "content": json.dumps(result, indent=2)})
                                    results.append({"function_name": name, "arguments": args, "result": result})
                                
                                # If any function call failed, try to recover with increasingly specific guidance
                                retry_count = 0
                                while not all_successful and retry_count < max_retries:
                                    retry_count += 1
                                    
                                    # Create a detailed error report
                                    error_details = []
                                    for res in results:
                                        if "error" in res["result"]:
                                            error_details.append({
                                                "function": res["function_name"],
                                                "arguments": res["arguments"],
                                                "error": res["result"].get("error"),
                                                "suggestion": self._get_function_suggestion(res["function_name"], res["arguments"], res["result"])
                                            })
                                    
                                    # Generate a correction prompt based on retry count
                                    if retry_count == 1:
                                        # First retry: general guidance
                                        correction_prompt = (
                                            f"There were errors in your function calls. Please correct your approach and try again with valid function calls.\n\n"
                                            f"Error details: {json.dumps(error_details, indent=2)}"
                                        )
                                    else:
                                        # Second retry: more specific guidance with available tools
                                        available_tools = self.tool_registry.get_available_tools()
                                        correction_prompt = (
                                            f"Your function calls still have errors. Here are the available tools you can use: "
                                            f"{', '.join(available_tools[:10])}.\n\n"
                                            f"Please use one of these tools with the correct parameters. Error details: {json.dumps(error_details, indent=2)}"
                                        )
                                    
                                    self.add_message("user", correction_prompt)
                                    corrected_response = self.client.completions.create(
                                        model=self.model,
                                        prompt=self.format_llama4_prompt() + "\nAssistant: ",
                                        max_tokens=1024,
                                        stop=["<|eot|>"]
                                    )
                                    corrected_content = corrected_response.choices[0].text + "<|eot|>"
                                    corrected_message = {"role": "assistant", "content": corrected_content}
                                    self.conversation_history.append(corrected_message)
                                    
                                    # Try to parse function calls from the corrected response
                                    corrected_function_calls = parse_function_calls(corrected_content)
                                    
                                    # Check if any of the function calls are to non-existent functions
                                    invalid_functions = [fc["name"] for fc in corrected_function_calls 
                                                        if not self.tool_registry.has_tool(fc["name"]) 
                                                        and not self._find_similar_tool(fc["name"])]
                                    
                                    # If there are invalid functions and this is the last retry, 
                                    # generate a direct response instead
                                    if invalid_functions and retry_count >= max_retries:
                                        console.print(f"[yellow]Invalid functions detected after max retries: {invalid_functions}. Generating direct response.[/yellow]")
                                        direct_prompt = "Please provide a direct response without using any functions."
                                        self.add_message("user", direct_prompt)
                                        direct_response = self.client.completions.create(
                                            model=self.model,
                                            prompt=self.format_llama4_prompt() + "\nAssistant: ",
                                            max_tokens=1024,
                                            stop=["<|eot|>"]
                                        )
                                        direct_content = direct_response.choices[0].text + "<|eot|>"
                                        direct_message = {"role": "assistant", "content": direct_content}
                                        self.conversation_history.append(direct_message)
                                        return direct_content.rstrip("<|eot|>")
                                    
                                    if corrected_function_calls:
                                        # Process the corrected function calls
                                        corrected_results = []
                                        all_successful = True  # Reset success flag for this retry
                                        
                                        for fc in corrected_function_calls:
                                            name = fc["name"]
                                            args = fc["arguments"]
                                            
                                            # Try to find similar function if exact match not found
                                            if not self.tool_registry.has_tool(name):
                                                similar_tool = self._find_similar_tool(name)
                                                if similar_tool:
                                                    console.print(f"[yellow]Function '{name}' not found, using similar function '{similar_tool}'[/yellow]")
                                                    name = similar_tool
                                                else:
                                                    error_msg = f"Function '{name}' not found in registry"
                                                    console.print(f"[red]{error_msg}[/red]")
                                                    result = {"error": error_msg, "success": False}
                                                    all_successful = False
                                                    self.conversation_history.append({"role": "tool", "name": name, "content": json.dumps(result, indent=2)})
                                                    corrected_results.append({"function_name": name, "arguments": args, "result": result})
                                                    continue
                                            
                                            console.print(f"[cyan]Retrying function: [bold]{name}[/bold][/cyan]")
                                            console.print(f"[cyan]With arguments:[/cyan] {json.dumps(args, indent=2)}")
                                            result = self.tool_registry.call_function(name, args)
                                            
                                            if "error" in result:
                                                all_successful = False
                                                console.print(f"[red]Error in retry {retry_count} for {name}: {result.get('error')}[/red]")
                                            else:
                                                console.print(f"[green]Result from {name} (retry {retry_count}):[/green]")
                                                console.print(json.dumps(result, indent=2))
                                                
                                            self.conversation_history.append({"role": "tool", "name": name, "content": json.dumps(result, indent=2)})
                                            corrected_results.append({"function_name": name, "arguments": args, "result": result})
                                        
                                        if corrected_results:
                                            # If we got successful results in this retry, use them
                                            # Otherwise, keep the original results to avoid losing information
                                            successful_results = [r for r in corrected_results if "error" not in r["result"]]
                                            if successful_results:
                                                results = successful_results
                                            else:
                                                # Combine original successful results with any new information
                                                successful_original = [r for r in results if "error" not in r["result"]]
                                                results = successful_original + corrected_results
                                    
                                    # If all functions succeeded in this retry, break the loop
                                    if all_successful:
                                        break
                                
                                final_prompt = f"Based on these function results: {json.dumps(results)}, provide a direct answer."
                                final_response = self.client.completions.create(
                                    model=self.model,
                                    prompt=self.format_llama4_prompt() + "\n\nUser: " + final_prompt + "<|eot|>\nAssistant: ",
                                    
                                    stop=["<|eot|>"]
                                )
                                final_content = final_response.choices[0].text + "<|eot|>"
                                final_message = {"role": "assistant", "content": final_content}
                                self.conversation_history.append(final_message)
                                return final_content.rstrip("<|eot|>")
                        response_content = assistant_content.rstrip("<|eot|>")
                        if "<|python_start|>" in response_content and "<|python_end" in response_content:
                            code_blocks = self.extract_python_code(response_content)
                            if code_blocks:
                                for code_block in code_blocks:
                                    if hasattr(self, 'last_user_message') and ("Run the code" in self.last_user_message or "run the code" in self.last_user_message):
                                        console.print("[cyan]Executing Python code:[/cyan]")
                                        import io, datetime
                                        stdout_capture = io.StringIO()
                                        stderr_capture = io.StringIO()
                                        local_vars = {}
                                        exec_globals = globals().copy()
                                        from datetime import date, datetime, timedelta
                                        exec_globals['date'] = date
                                        exec_globals['datetime'] = datetime
                                        exec_globals['timedelta'] = timedelta
                                        try:
                                            # Clean up the code block by removing any decorative lines and joining multiline statements
                                            clean_code = []
                                            for line in code_block.split('\n'):
                                                # Skip purely decorative lines with box drawing characters
                                                if all(c in '┏━┓┃┗┛ ' for c in line):
                                                    continue
                                                # Fix lines that have been compressed without proper newlines
                                                if ' if ' in line and ':' in line and 'return' in line:
                                                    parts = line.split(' if ')
                                                    pre = parts[0].strip()
                                                    rest = parts[1].strip()
                                                    if_parts = rest.split(': return')
                                                    if len(if_parts) == 2:
                                                        clean_code.append(f"{pre} if {if_parts[0]}:")
                                                        clean_code.append(f"    return{if_parts[1]}")
                                                        continue
                                                # Fix other compressed lines
                                                if ' return ' in line and line.count('return') == 1:
                                                    parts = line.split(' return ')
                                                    if parts[0].strip() and parts[1].strip():
                                                        clean_code.append(parts[0].strip())
                                                        clean_code.append(f"return {parts[1].strip()}")
                                                        continue
                                                clean_code.append(line)
                                            
                                            # Join the cleaned code
                                            cleaned_code_block = '\n'.join(clean_code)
                                            console.print(f"[dim]{cleaned_code_block}[/dim]")
                                            
                                            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                                                exec(cleaned_code_block, exec_globals, local_vars)
                                            stdout = stdout_capture.getvalue()
                                            stderr = stderr_capture.getvalue()
                                            if stdout:
                                                console.print("[green]Execution result:[/green]")
                                                console.print(stdout)
                                            if stderr:
                                                console.print("[red]Errors:[/red]")
                                                console.print(stderr)
                                        except Exception as e:
                                            console.print(f"[red]Error executing code: {str(e)}[/red]")
                                            console.print(traceback.format_exc())
                        # Check for common patterns that might indicate the model is trying to call a non-existent function
                        if "<|python_start|>" in response_content and any(func in response_content for func in ["get_user_info", "get_name", "user_info"]):
                            # Replace with a direct response for basic questions
                            self.conversation_history.pop()  # Remove the problematic response
                            direct_response = self.client.completions.create(
                                model=self.model,
                                prompt=self.format_llama4_prompt() + "\n\nAssistant: I don't have access to personal information about you unless you share it with me. How can I help you today?",
                                max_tokens=1024,
                                stop=["<|eot|>"]
                            )
                            direct_content = direct_response.choices[0].text + "<|eot|>"
                            direct_message = {"role": "assistant", "content": direct_content}
                            self.conversation_history.append(direct_message)
                            return direct_content.rstrip("<|eot|>")
                        
                        # Check for partial responses and reprompt if needed
                        response_content = self._check_and_fix_partial_response(response_content)
                        return response_content
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        tools=tools,
                        tool_choice="auto"
                    )
                    assistant_message = response.choices[0].message
                    self.conversation_history.append(assistant_message.model_dump())
                    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                        results = self.process_tool_calls(assistant_message.tool_calls)
                        for res in results:
                            console.print(f"[green]Result from {res['function_name']}:[/green]")
                            console.print(json.dumps(res['result'], indent=2))
                        final_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=self.conversation_history,
                            tools=tools,
                            tool_choice="none",
                            
                        )
                        final_message = final_response.choices[0].message
                        self.conversation_history.append(final_message.model_dump())
                        content = final_message.content
                        # Check for partial responses and reprompt if needed
                        content = self._check_and_fix_partial_response(content)
                        return content
                    content = assistant_message.content
                    # Check for partial responses and reprompt if needed
                    content = self._check_and_fix_partial_response(content)
                    return content
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
                console.print(traceback.format_exc())
                return f"Error: {str(e)}"

# =======================
# Main CLI Loop
# =======================
def main():
    # Make sure re module is available in this function's scope
    import re
    
    # Check if redis is installed, if not, install it
    try:
        import redis
    except ImportError:
        print("[yellow]Redis package not found. Installing...[/yellow]")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "redis"])
        import redis
        
    # Check for Jina API key
    jina_api_key = os.environ.get("JINA_API_KEY")
    if jina_api_key:
        console.print("[green]JINA_API_KEY found. Will use jina-clip-v2 for embeddings.[/green]")
    else:
        console.print("[yellow]Warning: JINA_API_KEY not found in environment variables.[/yellow]")
        console.print("[yellow]Set JINA_API_KEY to enable advanced image and text embeddings.[/yellow]")
    
    parser = argparse.ArgumentParser(description="Chat with an orchestrator AI agent with specialized scout agents using Together API")
    parser.add_argument("--model", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", help="Model to use (default: meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8)")
    parser.add_argument("--scouts", type=int, default=5, help="Number of scout agents to initialize (default: 5)")
    parser.add_argument("--logprobs", action="store_true", help="Enable returning logprobs for confidence analysis")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with mock API responses")
    parser.add_argument("--install-deps", action="store_true", help="Install required dependencies")
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        console.print("[cyan]Installing required dependencies...[/cyan]")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "pillow", "numpy", "redis"])
            console.print("[green]Dependencies installed successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error installing dependencies: {e}[/red]")
    
    welcome = (
        "[bold blue]Enhanced Together Agent CLI with Scout Orchestration[/bold blue]\n"
        "Chat with an AI agent that uses multiple specialized scout agents to solve complex tasks in parallel.\n"
        "The main agent coordinates a team of specialized scouts (research, code, planning, creative, critical).\n"
        "Each scout uses extensive chain-of-thought reasoning to solve its assigned tasks.\n"
        "The system supports autonomous task decomposition and parallel execution.\n"
        "Type [bold]'/paste'[/bold] to paste an image from your clipboard.\n"
        "Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to exit."
    )
    console.print(Panel.fit(welcome, title="Welcome"))
    
    if args.test_mode and "TOGETHER_API_KEY" not in os.environ:
        os.environ["TOGETHER_API_KEY"] = "dummy_api_key_for_testing"
    
    try:
        console.print(f"[cyan]Initializing TogetherAgent with {args.scouts} scout agents...[/cyan]")
        agent = TogetherAgent(model=args.model, num_scouts=args.scouts)
        agent.enable_logprobs = args.logprobs
        console.print(f"[green]Successfully initialized agent with {len(agent.agent_orchestrator.scouts)} scout agents[/green]")
    except ValueError as e:
        if "API key is required" in str(e) and not args.test_mode:
            console.print("[red]Error: Together API key is required. Set TOGETHER_API_KEY environment variable.[/red]")
            return 1
        raise
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]Exiting...[/yellow]")
                break
                
            # Check for direct Python execution
            if user_input.strip().startswith("execute_python(") or user_input.strip().startswith("python "):
                try:
                    if user_input.strip().startswith("execute_python("):
                        # Extract the code from execute_python call by parsing parameters
                        code_match = re.search(r'execute_python\(code="([^"]+)"', user_input)
                        if code_match:
                            code = code_match.group(1).replace("\\n", "\n")
                            console.print("[cyan]Executing Python code:[/cyan]")
                            console.print(f"[dim]{code}[/dim]")
                            result = agent.tool_registry._execute_python(code)
                            if result.get("success", False):
                                if result.get("stdout"):
                                    console.print("[green]Execution result:[/green]")
                                    console.print(result["stdout"])
                                if result.get("stderr"):
                                    console.print("[red]Errors:[/red]")
                                    console.print(result["stderr"])
                            else:
                                console.print(f"[red]Error executing code: {result.get('error', 'Unknown error')}[/red]")
                            continue
                    elif user_input.strip().startswith("python "):
                        # Extract code that follows "python "
                        code = user_input[7:].strip()
                        console.print("[cyan]Executing Python code:[/cyan]")
                        console.print(f"[dim]{code}[/dim]")
                        result = agent.tool_registry._execute_python(code)
                        if result.get("success", False):
                            if result.get("stdout"):
                                console.print("[green]Execution result:[/green]")
                                console.print(result["stdout"])
                            if result.get("stderr"):
                                console.print("[red]Errors:[/red]")
                                console.print(result["stderr"])
                        else:
                            console.print(f"[red]Error executing code: {result.get('error', 'Unknown error')}[/red]")
                        continue
                except Exception as e:
                    console.print(f"[red]Error processing Python command: {str(e)}[/red]")
                    console.print(traceback.format_exc())
                    continue
                
            # Check for paste command
            if user_input.strip() == "/paste":
                with console.status("[bold blue]Pasting from clipboard...[/bold blue]", spinner="dots"):
                    clipboard_content = agent.process_image_in_message("/paste")
                    if isinstance(clipboard_content, list):
                        console.print("[green]Image pasted from clipboard[/green]")
                        response = agent.generate_response(clipboard_content)
                    else:
                        console.print(f"[yellow]{clipboard_content}[/yellow]")
                        continue
            # Check for image URLs
            elif re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)', user_input):
                multimodal = []
                text_content = user_input
                image_urls = re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)', user_input)
                for url in image_urls:
                    text_content = text_content.replace(url, '')
                    console.print(f"[cyan]Including image: {url}[/cyan]")
                text_content = text_content.strip()
                if text_content:
                    multimodal.append({"type": "text", "text": text_content})
                for url in image_urls:
                    multimodal.append({"type": "image_url", "image_url": {"url": url}})
                with console.status("[bold blue]Processing images...[/bold blue]", spinner="dots"):
                    response = agent.generate_response(multimodal)
            else:
                with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
                    response = agent.generate_response(user_input)
            for msg in reversed(agent.conversation_history):
                if isinstance(msg, dict) and msg.get("role") == "assistant" and 'logprobs' in msg and msg.get('avg_logprob') is not None:
                    avg_conf = msg.get('avg_logprob')
                    level = "high" if avg_conf > -1.0 else "medium" if avg_conf > -2.0 else "low"
                    console.print(f"[cyan]Model confidence: {level} (avg logprob: {avg_conf:.2f})[/cyan]")
                    break
            console.print("\n[bold purple]Assistant[/bold purple]:")
            console.print(Markdown(response))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]")
            console.print(traceback.format_exc())
    return 0

if __name__ == "__main__":
    sys.exit(main())

    def check_learning(self):
        """Check if we should perform continuous learning and optimization"""
        if not hasattr(self, 'continuous_learning') or not self.continuous_learning.get("enabled", False):
            return
            
        current_time = time.time()
        time_since_last_optimization = current_time - self.continuous_learning.get("last_optimization_time", 0)
        
        # Perform optimization every 10 minutes or after 5 user interactions
        user_message_count = len([msg for msg in self.conversation_history if msg.get("role") == "user"])
        
        if (time_since_last_optimization > 600 or  # 10 minutes
            (user_message_count > 0 and user_message_count % 5 == 0)):
            
            if hasattr(self, 'agent_orchestrator') and hasattr(self.agent_orchestrator, 'optimize_team_structure'):
                # Optimize agent team
                optimization_result = self.agent_orchestrator.optimize_team_structure()
                
                # Update last optimization time
                self.continuous_learning["last_optimization_time"] = current_time
                
                # Store optimization result in memory
                self.memory.add_memory(
                    f"Performed agent team optimization. Results: {json.dumps(optimization_result)}",
                    {"category": "system", "type": "optimization"}
                )
            
    def check_meta_cognitive_reflection(self):
        """Check if we should perform meta-cognitive reflection"""
        if not hasattr(self, 'meta_cognition') or not self.meta_cognition.get("enabled", False):
            return
            
        current_time = time.time()
        time_since_last_reflection = current_time - self.meta_cognition.get("last_reflection_time", 0)
        user_message_count = len([msg for msg in self.conversation_history if msg.get("role") == "user"])
        
        # Perform reflection based on frequency setting
        if (user_message_count > 0 and 
            user_message_count % self.meta_cognition.get("reflection_frequency", 10) == 0):
            
            if hasattr(self, '_generate_meta_cognitive_reflection'):
                # Perform reflection
                reflection = self._generate_meta_cognitive_reflection()
                
                # Update last reflection time
                self.meta_cognition["last_reflection_time"] = current_time
                
                # Store reflection in memory and insights
                self.memory.add_memory(
                    reflection,
                    {"category": "meta_cognition", "type": "reflection"}
                )
                
                self.meta_cognition["insights"].append({
                    "timestamp": current_time,
                    "reflection": reflection
                })
            
    def _generate_meta_cognitive_reflection(self) -> str:
        """Generate a meta-cognitive reflection on recent performance"""
        # Get recent conversation history
        recent_messages = self.conversation_history[-min(10, len(self.conversation_history)):]
        
        # Get recent agent performance
        agent_metrics = self.agent_orchestrator.get_agent_performance_metrics()
        
        # Get recent memory items
        recent_memories = self.memory.search_memory("recent performance", limit=5)
        
        # Create reflection prompt
        reflection_prompt = [
            {"role": "system", "content": "You are an advanced AI performing meta-cognitive reflection on your own performance. Analyze recent interactions and agent performance to identify strengths, weaknesses, and areas for improvement."},
            {"role": "user", "content": f"Generate a thoughtful reflection on recent performance. Consider conversation quality, task success rates, and collaboration patterns. Recent conversation: {json.dumps(recent_messages[-3:])}\n\nAgent metrics: {json.dumps(agent_metrics)}"}
        ]
        
        # Generate reflection
        reflection_response = self.client.chat.completions.create(
            model=self.model,
            messages=reflection_prompt,
            
        )
        
        reflection = reflection_response.choices[0].message.content
        
        return reflection
