#!/usr/bin/env python3
"""
dynamic_context_manager.py
-------------------------
Advanced context management system for the Llama4 agent architecture, providing
dynamic, adaptive context windows, automatic relevance filtering, and improved
conversation understanding capabilities.

This module enables agents to maintain better awareness of conversation history,
domain-specific knowledge, and user preferences while efficiently managing
context length for optimal performance.
"""

import os
import json
import time
import uuid
import hashlib
import numpy as np
import sqlite3
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import logging
from collections import defaultdict, deque
import re
import heapq

# Import optimized vector memory for semantic search
try:
    from optimized_vector_memory import OptimizedVectorMemory, OptimizedEmbeddingProvider
    VECTOR_MEMORY_AVAILABLE = True
except ImportError:
    VECTOR_MEMORY_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dynamic_context")

# =====================================
# Data Models
# =====================================
@dataclass
class ContextItem:
    """Base class for all context items"""
    item_id: str
    content: str
    timestamp: float
    source: str
    relevance: float = 1.0  # Default relevance score
    expiration: Optional[float] = None  # Expiration time (None = never expires)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if this context item has expired"""
        if self.expiration is None:
            return False
        return time.time() > self.expiration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "item_id": self.item_id,
            "content": self.content,
            "timestamp": self.timestamp,
            "source": self.source,
            "relevance": self.relevance,
            "expiration": self.expiration,
            "tags": self.tags,
            "metadata": self.metadata,
            "item_type": self.__class__.__name__
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextItem':
        """Create from dictionary after deserialization"""
        item_type = data.pop("item_type", "ContextItem")
        
        # Handle specific item types
        if item_type == "MessageItem" and item_type != cls.__name__:
            from_role = data.get("metadata", {}).get("from_role", "")
            to_role = data.get("metadata", {}).get("to_role", "")
            message_type = data.get("metadata", {}).get("message_type", "")
            return MessageItem(
                item_id=data["item_id"],
                content=data["content"],
                timestamp=data["timestamp"],
                source=data["source"],
                from_role=from_role,
                to_role=to_role,
                message_type=message_type,
                relevance=data.get("relevance", 1.0),
                expiration=data.get("expiration"),
                tags=data.get("tags", []),
                metadata=data.get("metadata", {})
            )
        elif item_type == "FactItem" and item_type != cls.__name__:
            domain = data.get("metadata", {}).get("domain", "")
            confidence = data.get("metadata", {}).get("confidence", 1.0)
            return FactItem(
                item_id=data["item_id"],
                content=data["content"],
                timestamp=data["timestamp"],
                source=data["source"],
                domain=domain,
                confidence=confidence,
                relevance=data.get("relevance", 1.0),
                expiration=data.get("expiration"),
                tags=data.get("tags", []),
                metadata=data.get("metadata", {})
            )
        
        # Default case - create base ContextItem
        return cls(
            item_id=data["item_id"],
            content=data["content"],
            timestamp=data["timestamp"],
            source=data["source"],
            relevance=data.get("relevance", 1.0),
            expiration=data.get("expiration"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )

@dataclass
class MessageItem(ContextItem):
    """Context item representing a message in the conversation"""
    from_role: str = field(default="")
    to_role: str = field(default="")
    message_type: str = field(default="")  # "user_message", "agent_message", "system_message", etc.
    
    def __post_init__(self):
        """After initialization, update metadata"""
        if "from_role" not in self.metadata:
            self.metadata["from_role"] = self.from_role
        if "to_role" not in self.metadata:
            self.metadata["to_role"] = self.to_role
        if "message_type" not in self.metadata:
            self.metadata["message_type"] = self.message_type

@dataclass
class FactItem(ContextItem):
    """Context item representing a factual statement or knowledge"""
    domain: str = field(default="")  # Knowledge domain
    confidence: float = 1.0  # Confidence in this fact (0.0-1.0)
    
    def __post_init__(self):
        """After initialization, update metadata"""
        if "domain" not in self.metadata:
            self.metadata["domain"] = self.domain
        if "confidence" not in self.metadata:
            self.metadata["confidence"] = self.confidence

# =====================================
# Dynamic Window Calculation
# =====================================
class DynamicWindowCalculator:
    """Calculates optimal context window size based on various factors"""
    
    def __init__(self, base_tokens: int = 4096, max_tokens: int = 16384):
        self.base_tokens = base_tokens
        self.max_tokens = max_tokens
        self.recent_completions = deque(maxlen=10)  # Store recent completion times
        self.token_usage_history = deque(maxlen=50)  # Store token usage history
    
    def record_completion(self, tokens_used: int, completion_time: float):
        """Record a completion for performance tracking"""
        self.recent_completions.append((tokens_used, completion_time))
        self.token_usage_history.append(tokens_used)
    
    def get_optimal_window(self, 
                        conversation_complexity: float = 0.5,
                        task_complexity: float = 0.5,
                        performance_sensitivity: float = 0.5) -> int:
        """
        Calculate optimal context window size
        
        Args:
            conversation_complexity: How complex the conversation is (0.0-1.0)
            task_complexity: How complex the current task is (0.0-1.0) 
            performance_sensitivity: How sensitive to performance we are (0.0-1.0)
            
        Returns:
            Optimal token count for context window
        """
        # Start with base window size
        window_size = self.base_tokens
        
        # Adjust for conversation complexity
        # More complex conversations need more context
        complexity_factor = 0.5 + (conversation_complexity * 0.5)  # 0.5-1.0
        window_size *= complexity_factor
        
        # Adjust for task complexity
        # More complex tasks need more context
        if task_complexity > 0.7:  # High complexity tasks
            window_size *= 1.0 + (task_complexity - 0.7) * 2.0  # Up to 1.6x for very complex tasks
        
        # Performance adjustment based on history
        if self.recent_completions:
            # Calculate average time per token
            total_tokens = sum(tokens for tokens, _ in self.recent_completions)
            total_time = sum(time for _, time in self.recent_completions)
            
            if total_tokens > 0:
                avg_time_per_token = total_time / total_tokens
                
                # If we're sensitive to performance and completion is slow,
                # reduce window size
                if performance_sensitivity > 0.5 and avg_time_per_token > 0.001:  # 1ms per token threshold
                    reduction_factor = min(0.8, max(0.4, 1.0 - (avg_time_per_token * 1000 - 1) * 0.1))
                    window_size *= reduction_factor
        
        # Ensure window is within min and max bounds
        window_size = max(self.base_tokens * 0.5, min(window_size, self.max_tokens))
        
        return int(window_size)
    
    def get_token_usage_stats(self) -> Dict[str, Any]:
        """Get statistics about token usage"""
        if not self.token_usage_history:
            return {
                "avg_tokens": 0,
                "max_tokens": 0,
                "min_tokens": 0
            }
        
        return {
            "avg_tokens": sum(self.token_usage_history) / len(self.token_usage_history),
            "max_tokens": max(self.token_usage_history),
            "min_tokens": min(self.token_usage_history)
        }

# =====================================
# Context Relevance Scoring
# =====================================
class RelevanceScorer:
    """Scores context items for relevance to current query/conversation"""
    
    def __init__(self, embedding_provider=None):
        self.embedding_provider = embedding_provider
        
        # Decay parameters
        self.time_decay_factor = 0.05  # How much to decay per hour
        self.usage_boost_factor = 0.2  # How much to boost per usage
        
        # Feature weights
        self.weights = {
            "semantic_similarity": 0.4,
            "recency": 0.2,
            "usage_count": 0.1,
            "explicit_relevance": 0.3
        }
    
    def score_item(self, item: ContextItem, query: str, 
                conversation_context: List[str] = None) -> float:
        """
        Score an item's relevance to the current query and conversation
        
        Args:
            item: The context item to score
            query: The current query or focus
            conversation_context: Recent conversation messages for context
            
        Returns:
            Relevance score (0.0-1.0)
        """
        features = {}
        
        # Calculate semantic similarity if embedding provider available
        if self.embedding_provider:
            query_embedding = self.embedding_provider.get_embedding(query)
            item_embedding = self.embedding_provider.get_embedding(item.content)
            
            from numpy import dot
            from numpy.linalg import norm
            
            features["semantic_similarity"] = max(0.0, float(
                dot(query_embedding, item_embedding) / 
                (norm(query_embedding) * norm(item_embedding))
            ))
        else:
            # Fallback: simple keyword matching
            query_words = set(query.lower().split())
            content_words = set(item.content.lower().split())
            overlap = len(query_words.intersection(content_words))
            features["semantic_similarity"] = min(1.0, overlap / max(1, len(query_words)))
        
        # Calculate recency (newer = more relevant)
        hours_old = (time.time() - item.timestamp) / 3600
        features["recency"] = max(0.0, 1.0 - (hours_old * self.time_decay_factor))
        
        # Usage count (from metadata)
        usage_count = item.metadata.get("usage_count", 0)
        features["usage_count"] = min(1.0, usage_count * self.usage_boost_factor)
        
        # Any explicit relevance assigned to the item
        features["explicit_relevance"] = item.relevance
        
        # Calculate weighted score
        score = sum(self.weights[feature] * value for feature, value in features.items())
        
        # Normalize to 0.0-1.0
        score = max(0.0, min(1.0, score))
        
        return score
    
    def adjust_weights(self, new_weights: Dict[str, float]):
        """Adjust the feature weights"""
        self.weights.update(new_weights)
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total

# =====================================
# Dynamic Context Manager
# =====================================
class DynamicContextManager:
    """
    Advanced context management system with dynamically adjusted windows,
    relevance scoring, and progressive disclosure of context
    """
    
    def __init__(self, 
                 storage_path: str = None,
                 base_tokens: int = 4096,
                 max_tokens: int = 16384,
                 enable_semantic_search: bool = True):
        self.storage_path = storage_path or os.path.join(
            os.path.expanduser('~'), '.cache', 'llama4_context')
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # All context items
        self.items: Dict[str, ContextItem] = {}
        
        # Indexed for fast retrieval
        self.item_by_source: Dict[str, List[str]] = defaultdict(list)  # source -> [item_ids]
        self.item_by_tag: Dict[str, List[str]] = defaultdict(list)     # tag -> [item_ids]
        
        # Conversation tracking
        self.conversation_history: List[str] = []  # item_ids in time order
        self.conversation_focus: List[str] = []    # Current topics of focus
        
        # User preferences and profile
        self.user_profile: Dict[str, Any] = {}
        self.attention_map: Dict[str, float] = {}  # topic -> attention_score
        
        # Initialize window calculator
        self.window_calculator = DynamicWindowCalculator(
            base_tokens=base_tokens,
            max_tokens=max_tokens
        )
        
        # Initialize vector memory if available
        self.vector_memory = None
        self.embedding_provider = None
        
        if enable_semantic_search and VECTOR_MEMORY_AVAILABLE:
            try:
                self.embedding_provider = OptimizedEmbeddingProvider(
                    cache_dir=os.path.join(self.storage_path, 'embedding_cache')
                )
                
                self.vector_memory = OptimizedVectorMemory(
                    storage_path=os.path.join(self.storage_path, 'vector_memory')
                )
                
                logger.info("Semantic search enabled with optimized vector memory")
            except Exception as e:
                logger.warning(f"Failed to initialize vector memory: {e}")
        
        # Initialize relevance scorer
        self.relevance_scorer = RelevanceScorer(embedding_provider=self.embedding_provider)
        
        # Performance tracking
        self.metrics = {
            "items_added": 0,
            "window_calculations": 0,
            "context_retrievals": 0,
            "avg_context_size": 0
        }
        
        # Load existing context if available
        self._load_context()
    
    def add_message(self, content: str, from_role: str, to_role: str, 
                  message_type: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a message to the context
        
        Args:
            content: Message content
            from_role: Role that sent the message
            to_role: Role that received the message
            message_type: Type of message
            metadata: Additional metadata
            
        Returns:
            Item ID of the added message
        """
        item_id = str(uuid.uuid4())
        timestamp = time.time()
        
        message = MessageItem(
            item_id=item_id,
            content=content,
            timestamp=timestamp,
            source="conversation",
            from_role=from_role,
            to_role=to_role,
            message_type=message_type,
            metadata=metadata or {}
        )
        
        # Add to context
        self.items[item_id] = message
        self.item_by_source["conversation"].append(item_id)
        
        # Add to conversation history
        self.conversation_history.append(item_id)
        
        # Add conversation focus topics (if user message)
        if from_role == "user":
            self._update_conversation_focus(content)
        
        # Add to vector memory if available
        if self.vector_memory:
            self.vector_memory.add_memory(content, {
                "item_id": item_id,
                "from_role": from_role,
                "to_role": to_role,
                "message_type": message_type,
                "timestamp": timestamp
            })
        
        # Update metrics
        self.metrics["items_added"] += 1
        
        # Periodically save context
        if self.metrics["items_added"] % 20 == 0:
            self._save_context()
        
        return item_id
    
    def add_fact(self, content: str, domain: str, confidence: float = 1.0,
               source: str = "knowledge_base", metadata: Dict[str, Any] = None,
               tags: List[str] = None) -> str:
        """
        Add a factual statement to the context
        
        Args:
            content: Fact content
            domain: Knowledge domain
            confidence: Confidence in this fact (0.0-1.0)
            source: Source of the fact
            metadata: Additional metadata
            tags: Tags for categorization
            
        Returns:
            Item ID of the added fact
        """
        item_id = str(uuid.uuid4())
        timestamp = time.time()
        
        fact = FactItem(
            item_id=item_id,
            content=content,
            timestamp=timestamp,
            source=source,
            domain=domain,
            confidence=confidence,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Add to context
        self.items[item_id] = fact
        self.item_by_source[source].append(item_id)
        
        # Add tag indexing
        for tag in fact.tags:
            self.item_by_tag[tag].append(item_id)
        
        # Add to vector memory if available
        if self.vector_memory:
            self.vector_memory.add_memory(content, {
                "item_id": item_id,
                "domain": domain,
                "confidence": confidence,
                "source": source,
                "tags": tags or [],
                "timestamp": timestamp
            })
        
        # Update metrics
        self.metrics["items_added"] += 1
        
        return item_id
    
    def add_custom_item(self, content: str, source: str, metadata: Dict[str, Any] = None,
                       tags: List[str] = None, expiration: Optional[float] = None) -> str:
        """
        Add a custom context item
        
        Args:
            content: Item content
            source: Source identifier
            metadata: Additional metadata
            tags: Tags for categorization
            expiration: Expiration timestamp
            
        Returns:
            Item ID of the added item
        """
        item_id = str(uuid.uuid4())
        timestamp = time.time()
        
        item = ContextItem(
            item_id=item_id,
            content=content,
            timestamp=timestamp,
            source=source,
            tags=tags or [],
            metadata=metadata or {},
            expiration=expiration
        )
        
        # Add to context
        self.items[item_id] = item
        self.item_by_source[source].append(item_id)
        
        # Add tag indexing
        for tag in item.tags:
            self.item_by_tag[tag].append(item_id)
        
        # Add to vector memory if available
        if self.vector_memory:
            vector_metadata = {
                "item_id": item_id,
                "source": source,
                "tags": tags or [],
                "timestamp": timestamp
            }
            if metadata:
                vector_metadata.update(metadata)
                
            self.vector_memory.add_memory(content, vector_metadata)
        
        # Update metrics
        self.metrics["items_added"] += 1
        
        return item_id
    
    def get_dynamic_context(self, 
                          query: str = None,
                          token_budget: int = None,
                          conversation_complexity: float = 0.5,
                          task_complexity: float = 0.5) -> Dict[str, Any]:
        """
        Get a dynamically constructed context optimized for the current query
        
        Args:
            query: Current query or focus (if None, uses last user message)
            token_budget: Maximum tokens to use (if None, calculated dynamically)
            conversation_complexity: Complexity of the conversation (0.0-1.0)
            task_complexity: Complexity of the current task (0.0-1.0)
            
        Returns:
            Dict with optimized context data:
            - items: List of context items
            - token_budget: Token budget used
            - stats: Stats about the context selection
        """
        # Track this operation
        self.metrics["context_retrievals"] += 1
        start_time = time.time()
        
        # If no query provided, use the last user message
        if query is None and self.conversation_history:
            for item_id in reversed(self.conversation_history):
                item = self.items.get(item_id)
                if isinstance(item, MessageItem) and item.from_role == "user":
                    query = item.content
                    break
        
        # If still no query, use a generic one
        if not query:
            query = "general context"
        
        # Get or calculate token budget
        if token_budget is None:
            token_budget = self.window_calculator.get_optimal_window(
                conversation_complexity=conversation_complexity,
                task_complexity=task_complexity
            )
            self.metrics["window_calculations"] += 1
        
        # Score all items for relevance to the query
        relevant_items = []
        
        if self.vector_memory and len(query) > 3:
            # Use vector search for efficiency
            vector_results = self.vector_memory.search_memory(query, limit=50)
            for result in vector_results:
                item_id = result.get("item_id")
                if item_id in self.items:
                    item = self.items[item_id]
                    
                    # Skip expired items
                    if item.is_expired():
                        continue
                    
                    # Use the relevance score from vector search
                    relevant_items.append((item, result.get("relevance_score", 0.5)))
        else:
            # Score each item directly
            for item_id, item in list(self.items.items()):
                # Skip expired items
                if item.is_expired():
                    continue
                
                # Get relevance score
                score = self.relevance_scorer.score_item(
                    item, 
                    query,
                    # Use last 5 messages as additional context
                    conversation_context=[
                        self.items[msg_id].content 
                        for msg_id in self.conversation_history[-5:]
                        if msg_id in self.items
                    ]
                )
                
                relevant_items.append((item, score))
        
        # Sort by relevance score
        relevant_items.sort(key=lambda x: x[1], reverse=True)
        
        # Estimate token count (very approximate, 4 chars per token)
        def estimate_tokens(text: str) -> int:
            return len(text) // 4 + 1
        
        # Build the context within token budget
        selected_items = []
        used_tokens = 0
        
        # First, add recent conversation history (most recent first)
        recent_history = []
        for item_id in reversed(self.conversation_history[-10:]):  # Last 10 messages
            if item_id in self.items:
                item = self.items[item_id]
                token_count = estimate_tokens(item.content)
                
                if used_tokens + token_count <= token_budget * 0.6:  # Use up to 60% for history
                    recent_history.append(item)
                    used_tokens += token_count
        
        # Then add other relevant items
        for item, score in relevant_items:
            token_count = estimate_tokens(item.content)
            
            # Skip if already in recent history
            if item in recent_history:
                continue
            
            # Only add if it meets minimum relevance and fits in budget
            if score >= 0.3 and used_tokens + token_count <= token_budget:
                selected_items.append(item)
                used_tokens += token_count
                
                # Update usage count in metadata
                item.metadata["usage_count"] = item.metadata.get("usage_count", 0) + 1
        
        # Add in the recent history at the beginning
        selected_items = recent_history + selected_items
        
        # Get actual context data
        context_data = {
            "items": [item.to_dict() for item in selected_items],
            "token_budget": token_budget,
            "used_tokens": used_tokens,
            "stats": {
                "elapsed_ms": (time.time() - start_time) * 1000,
                "history_items": len(recent_history),
                "relevant_items": len(selected_items) - len(recent_history),
                "total_items": len(selected_items),
                "token_usage_percent": (used_tokens / token_budget) * 100 if token_budget > 0 else 0
            }
        }
        
        # Update running average context size
        total_retrievals = self.metrics["context_retrievals"]
        self.metrics["avg_context_size"] = (
            (self.metrics["avg_context_size"] * (total_retrievals - 1) + len(selected_items)) / 
            total_retrievals
        )
        
        return context_data
    
    def get_items_by_source(self, source: str) -> List[ContextItem]:
        """Get all context items from a specific source"""
        return [self.items[item_id] for item_id in self.item_by_source.get(source, [])
                if item_id in self.items and not self.items[item_id].is_expired()]
    
    def get_items_by_tag(self, tag: str) -> List[ContextItem]:
        """Get all context items with a specific tag"""
        return [self.items[item_id] for item_id in self.item_by_tag.get(tag, [])
                if item_id in self.items and not self.items[item_id].is_expired()]
    
    def get_conversation_history(self, limit: int = None) -> List[ContextItem]:
        """Get conversation history items in chronological order"""
        # Get only non-expired items that still exist
        history = [
            self.items[item_id] for item_id in self.conversation_history
            if item_id in self.items and not self.items[item_id].is_expired()
        ]
        
        # Apply limit if provided
        if limit is not None:
            history = history[-limit:]
            
        return history
    
    def search_context(self, query: str, limit: int = 10) -> List[Tuple[ContextItem, float]]:
        """
        Search context items semantically
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of (item, score) tuples
        """
        if self.vector_memory:
            # Use vector memory for efficient search
            results = self.vector_memory.search_memory(query, limit=limit)
            return [
                (self.items[result["id"]], result["relevance_score"])
                for result in results
                if result["id"] in self.items and not self.items[result["id"]].is_expired()
            ]
        else:
            # Fallback to relevance scorer
            items = [(item, self.relevance_scorer.score_item(item, query))
                    for item in self.items.values() if not item.is_expired()]
            
            # Sort by relevance
            items.sort(key=lambda x: x[1], reverse=True)
            return items[:limit]
    
    def update_user_profile(self, data: Dict[str, Any]):
        """Update user profile data"""
        self.user_profile.update(data)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "item_count": len(self.items),
            "conversation_length": len(self.conversation_history),
            "token_usage": self.window_calculator.get_token_usage_stats()
        }
    
    def forget_item(self, item_id: str) -> bool:
        """Remove an item from the context"""
        if item_id not in self.items:
            return False
        
        # Get the item for tag and source removal
        item = self.items[item_id]
        
        # Remove from source index
        if item_id in self.item_by_source.get(item.source, []):
            self.item_by_source[item.source].remove(item_id)
        
        # Remove from tag index
        for tag in item.tags:
            if item_id in self.item_by_tag.get(tag, []):
                self.item_by_tag[tag].remove(item_id)
        
        # Remove from conversation history if present
        if item_id in self.conversation_history:
            self.conversation_history.remove(item_id)
        
        # Remove from items
        del self.items[item_id]
        
        return True
    
    def _update_conversation_focus(self, message: str):
        """Update the conversation focus based on a message"""
        # Simple keyword extraction for focus
        words = re.findall(r'\b[a-zA-Z]{4,}\b', message.lower())
        word_counts = defaultdict(int)
        
        for word in words:
            word_counts[word] += 1
        
        # Update attention map
        for word, count in word_counts.items():
            self.attention_map[word] = self.attention_map.get(word, 0) + count
        
        # Decay old topics
        for topic in list(self.attention_map.keys()):
            self.attention_map[topic] *= 0.9
            if self.attention_map[topic] < 0.5:
                del self.attention_map[topic]
        
        # Update focus topics (top 5)
        self.conversation_focus = heapq.nlargest(
            5, 
            self.attention_map.keys(),
            key=lambda k: self.attention_map[k]
        )
    
    def _save_context(self):
        """Save context to disk"""
        try:
            # Save main context data
            context_data = {
                "items": {item_id: item.to_dict() for item_id, item in self.items.items()},
                "conversation_history": self.conversation_history,
                "user_profile": self.user_profile,
                "attention_map": self.attention_map,
                "conversation_focus": self.conversation_focus,
                "metrics": self.metrics,
                "saved_at": time.time()
            }
            
            with open(os.path.join(self.storage_path, 'context_data.json'), 'w') as f:
                json.dump(context_data, f)
            
            # Save vector memory if available
            if self.vector_memory:
                self.vector_memory.save()
                
            return True
        except Exception as e:
            logger.error(f"Failed to save context: {e}")
            return False
    
    def _load_context(self):
        """Load context from disk"""
        try:
            context_path = os.path.join(self.storage_path, 'context_data.json')
            if not os.path.exists(context_path):
                return False
            
            with open(context_path, 'r') as f:
                context_data = json.load(f)
            
            # Load items
            self.items = {}
            for item_id, item_data in context_data.get("items", {}).items():
                self.items[item_id] = ContextItem.from_dict(item_data)
            
            # Rebuild indices
            self.item_by_source = defaultdict(list)
            self.item_by_tag = defaultdict(list)
            
            for item_id, item in self.items.items():
                # Skip expired items
                if item.is_expired():
                    continue
                    
                self.item_by_source[item.source].append(item_id)
                for tag in item.tags:
                    self.item_by_tag[tag].append(item_id)
            
            # Load other data
            self.conversation_history = context_data.get("conversation_history", [])
            self.user_profile = context_data.get("user_profile", {})
            self.attention_map = context_data.get("attention_map", {})
            self.conversation_focus = context_data.get("conversation_focus", [])
            self.metrics = context_data.get("metrics", self.metrics)
            
            logger.info(f"Loaded {len(self.items)} context items from disk")
            return True
        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize context manager
    context_manager = DynamicContextManager()
    
    # Add some context items
    context_manager.add_message(
        content="How can I make my agents more context-aware?",
        from_role="user",
        to_role="assistant",
        message_type="user_message"
    )
    
    context_manager.add_message(
        content="You can use the DynamicContextManager to improve context awareness.",
        from_role="assistant",
        to_role="user",
        message_type="assistant_message"
    )
    
    # Add some facts
    context_manager.add_fact(
        content="Context awareness in AI systems improves response relevance by 30-45%.",
        domain="ai_research",
        confidence=0.85,
        tags=["context", "performance", "ai"]
    )
    
    context_manager.add_fact(
        content="Vector embeddings enable semantic search capabilities in context management.",
        domain="nlp",
        confidence=0.95,
        tags=["embeddings", "vectors", "search"]
    )
    
    # Get dynamic context
    context = context_manager.get_dynamic_context(
        query="How do I implement context-aware agents?",
        conversation_complexity=0.7
    )
    
    # Print results
    print("Dynamic Context Results:")
    print(f"Token budget: {context['token_budget']}")
    print(f"Used tokens: {context['used_tokens']}")
    print(f"Items selected: {len(context['items'])}")
    
    print("\nSelected items:")
    for i, item in enumerate(context['items']):
        print(f"{i+1}. Type: {item['item_type']}, Source: {item['source']}")
        print(f"   Content: {item['content'][:60]}..." if len(item['content']) > 60 else f"   Content: {item['content']}")
        print(f"   Relevance: {item.get('relevance', 'N/A')}")
        print()
    
    # Get metrics
    print("\nPerformance Metrics:")
    metrics = context_manager.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Save context
    context_manager._save_context()