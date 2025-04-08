#!/usr/bin/env python3
"""
Realtime PubSub Service for distributed communication
Provides a high-performance, fault-tolerant publish/subscribe system
"""

import os
import json
import logging
import asyncio
import time
import signal
import uuid
import threading
from typing import Dict, List, Set, Any, Callable, Optional, Union
import redis.asyncio as redis
import aiohttp
from aiohttp import web

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pubsub-service")

class PubSubService:
    """
    Realtime PubSub Service with Redis backend and WebSocket frontend
    Provides both Redis PubSub and WebSocket interfaces for realtime communication
    """
    def __init__(self, redis_url=None, fallback_redis_urls=None, host="0.0.0.0", port=8004):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.fallback_redis_urls = fallback_redis_urls or [
            url.strip() for url in os.getenv("FALLBACK_REDIS_URLS", "").split(",") if url.strip()
        ]
        self.host = host
        self.port = port
        self.redis_client = None
        self.fallback_clients = []
        self.pubsub = None
        self.channels = {}  # channel -> set of subscribers
        self.websockets = {}  # client_id -> websocket
        self.client_channels = {}  # client_id -> set of subscribed channels
        self.app = None
        self.runner = None
        self.site = None
        self.listener_task = None
        self._shutdown_event = asyncio.Event()
        logger.info(f"PubSub service initialized on {host}:{port} with Redis URL: {self.redis_url}")
    
    async def start(self):
        """Start the PubSub service"""
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        
        # Connect to Redis
        await self.connect_redis()
        
        # Setup web application
        self.app = web.Application()
        self.app.add_routes([
            web.get('/ws', self.websocket_handler),
            web.get('/health', self.health_handler),
            web.post('/publish', self.publish_handler),
            web.get('/channels', self.channels_handler)
        ])
        
        # Start web server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        logger.info(f"PubSub service started on http://{self.host}:{self.port}")
        
        # Start Redis PubSub listener
        self.listener_task = asyncio.create_task(self.redis_listener())
        
        # Wait for shutdown signal
        await self._shutdown_event.wait()
    
    async def connect_redis(self):
        """Connect to Redis with fallback support"""
        # Try primary Redis first
        try:
            self.redis_client = await redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to primary Redis successfully")
            
            # Initialize PubSub
            self.pubsub = self.redis_client.pubsub()
            
            # Connect to fallback Redis servers
            for url in self.fallback_redis_urls:
                try:
                    client = await redis.from_url(url)
                    await client.ping()
                    self.fallback_clients.append(client)
                    logger.info(f"Connected to fallback Redis at {url}")
                except Exception as e:
                    logger.warning(f"Failed to connect to fallback Redis at {url}: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to primary Redis: {e}")
            
            # Try fallbacks if primary fails
            for url in self.fallback_redis_urls:
                try:
                    self.redis_client = await redis.from_url(url)
                    await self.redis_client.ping()
                    logger.info(f"Connected to fallback Redis at {url} as primary")
                    
                    # Initialize PubSub
                    self.pubsub = self.redis_client.pubsub()
                    
                    return True
                except Exception as fallback_e:
                    logger.error(f"Failed to connect to fallback Redis at {url}: {fallback_e}")
            
            return False
    
    async def shutdown(self):
        """Shutdown the PubSub service gracefully"""
        logger.info("Shutting down PubSub service...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel listener task
        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
        
        # Close all WebSocket connections
        close_tasks = []
        for ws in self.websockets.values():
            close_tasks.append(ws.close(code=1001, message=b"Server shutting down"))
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # Unsubscribe from all Redis channels
        if self.pubsub:
            channels = list(self.channels.keys())
            if channels:
                await self.pubsub.unsubscribe(*channels)
            await self.pubsub.close()
        
        # Close Redis connections
        if self.redis_client:
            await self.redis_client.close()
        
        for client in self.fallback_clients:
            await client.close()
        
        # Shutdown web server
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("PubSub service shutdown complete")
    
    async def redis_listener(self):
        """Listen for Redis PubSub messages and forward to WebSocket clients"""
        try:
            logger.info("Started Redis PubSub listener")
            while not self._shutdown_event.is_set():
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    channel = message["channel"]
                    if isinstance(channel, bytes):
                        channel = channel.decode('utf-8')
                    
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    
                    # Forward message to all WebSocket subscribers
                    if channel in self.channels:
                        await self.broadcast_to_channel(channel, data)
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            logger.info("Redis PubSub listener cancelled")
        except Exception as e:
            logger.error(f"Error in Redis PubSub listener: {e}")
            
            # Try to reconnect
            try:
                if not self._shutdown_event.is_set():
                    logger.info("Attempting to reconnect to Redis...")
                    await self.connect_redis()
                    
                    # Resubscribe to channels
                    channels = list(self.channels.keys())
                    if channels and self.pubsub:
                        await self.pubsub.subscribe(*channels)
                        logger.info(f"Resubscribed to {len(channels)} channels")
                        
                    # Restart listener
                    if not self._shutdown_event.is_set():
                        self.listener_task = asyncio.create_task(self.redis_listener())
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect to Redis: {reconnect_error}")
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Generate client ID
        client_id = str(uuid.uuid4())
        self.websockets[client_id] = ws
        self.client_channels[client_id] = set()
        
        # Send welcome message with client ID
        await ws.send_json({
            "type": "welcome",
            "client_id": client_id
        })
        
        logger.info(f"Client {client_id} connected")
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_ws_message(client_id, data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON from client {client_id}: {msg.data}")
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error from client {client_id}: {ws.exception()}")
        finally:
            # Clean up when client disconnects
            await self.handle_client_disconnect(client_id)
        
        return ws
    
    async def handle_ws_message(self, client_id: str, data: Dict):
        """Handle a message from a WebSocket client"""
        msg_type = data.get("type")
        
        if msg_type == "subscribe":
            # Client wants to subscribe to a channel
            channel = data.get("channel")
            if channel:
                await self.subscribe_client(client_id, channel)
                
        elif msg_type == "unsubscribe":
            # Client wants to unsubscribe from a channel
            channel = data.get("channel")
            if channel:
                await self.unsubscribe_client(client_id, channel)
                
        elif msg_type == "publish":
            # Client wants to publish a message to a channel
            channel = data.get("channel")
            message = data.get("message")
            if channel and message is not None:
                await self.publish_message(channel, message)
                
        elif msg_type == "ping":
            # Client is sending a ping
            await self.websockets[client_id].send_json({
                "type": "pong",
                "timestamp": time.time()
            })
            
        else:
            logger.warning(f"Unknown message type from client {client_id}: {msg_type}")
    
    async def subscribe_client(self, client_id: str, channel: str):
        """Subscribe a client to a channel"""
        # Add client to channel subscribers
        if channel not in self.channels:
            self.channels[channel] = set()
            
            # Subscribe to Redis channel if this is the first subscriber
            if self.pubsub:
                await self.pubsub.subscribe(channel)
                logger.info(f"Subscribed to Redis channel: {channel}")
        
        self.channels[channel].add(client_id)
        self.client_channels[client_id].add(channel)
        
        logger.info(f"Client {client_id} subscribed to channel: {channel}")
        
        # Confirm subscription to client
        if client_id in self.websockets:
            await self.websockets[client_id].send_json({
                "type": "subscribed",
                "channel": channel
            })
    
    async def unsubscribe_client(self, client_id: str, channel: str):
        """Unsubscribe a client from a channel"""
        if channel in self.channels and client_id in self.channels[channel]:
            self.channels[channel].remove(client_id)
            
            # If no more subscribers, unsubscribe from Redis
            if not self.channels[channel] and self.pubsub:
                await self.pubsub.unsubscribe(channel)
                del self.channels[channel]
                logger.info(f"Unsubscribed from Redis channel: {channel}")
        
        if client_id in self.client_channels and channel in self.client_channels[client_id]:
            self.client_channels[client_id].remove(channel)
        
        logger.info(f"Client {client_id} unsubscribed from channel: {channel}")
        
        # Confirm unsubscription to client
        if client_id in self.websockets:
            await self.websockets[client_id].send_json({
                "type": "unsubscribed",
                "channel": channel
            })
    
    async def handle_client_disconnect(self, client_id: str):
        """Handle a client disconnection"""
        # Unsubscribe from all channels
        if client_id in self.client_channels:
            channels = list(self.client_channels[client_id])
            for channel in channels:
                await self.unsubscribe_client(client_id, channel)
            del self.client_channels[client_id]
        
        # Remove from websockets
        if client_id in self.websockets:
            del self.websockets[client_id]
            
        logger.info(f"Client {client_id} disconnected")
    
    async def broadcast_to_channel(self, channel: str, message: str):
        """Broadcast a message to all WebSocket subscribers of a channel"""
        if channel not in self.channels:
            return
        
        # Try to parse as JSON
        try:
            data = json.loads(message)
            message_obj = {
                "type": "message",
                "channel": channel,
                "data": data,
                "timestamp": time.time()
            }
        except json.JSONDecodeError:
            # Not valid JSON, send as string
            message_obj = {
                "type": "message",
                "channel": channel,
                "data": message,
                "timestamp": time.time()
            }
        
        # Send to all subscribers
        send_tasks = []
        for client_id in list(self.channels[channel]):
            if client_id in self.websockets:
                try:
                    send_tasks.append(self.websockets[client_id].send_json(message_obj))
                except Exception as e:
                    logger.error(f"Error preparing message for client {client_id}: {e}")
        
        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)
    
    async def publish_message(self, channel: str, message: Any):
        """Publish a message to a Redis channel"""
        try:
            # Convert message to JSON string if it's not already a string
            if not isinstance(message, str):
                message = json.dumps(message)
                
            # Publish to Redis
            await self.redis_client.publish(channel, message)
            logger.debug(f"Published message to channel: {channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to channel {channel}: {e}")
            
            # Try fallbacks
            for client in self.fallback_clients:
                try:
                    await client.publish(channel, message)
                    logger.debug(f"Published message to channel {channel} via fallback")
                    return True
                except Exception:
                    continue
                    
            return False
    
    async def health_handler(self, request):
        """Handle health check requests"""
        # Check Redis connection
        redis_healthy = False
        try:
            if self.redis_client:
                await self.redis_client.ping()
                redis_healthy = True
        except Exception:
            pass
        
        health_status = {
            "status": "healthy" if redis_healthy else "degraded",
            "redis_connected": redis_healthy,
            "fallbacks_available": len(self.fallback_clients),
            "active_channels": len(self.channels),
            "connected_clients": len(self.websockets),
            "timestamp": time.time()
        }
        
        return web.json_response(health_status)
    
    async def publish_handler(self, request):
        """Handle HTTP publish requests"""
        try:
            data = await request.json()
            channel = data.get("channel")
            message = data.get("message")
            
            if not channel:
                return web.json_response({"error": "Channel is required"}, status=400)
            
            if message is None:
                return web.json_response({"error": "Message is required"}, status=400)
            
            success = await self.publish_message(channel, message)
            
            if success:
                return web.json_response({"status": "success"})
            else:
                return web.json_response({"error": "Failed to publish message"}, status=500)
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            logger.error(f"Error in publish handler: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def channels_handler(self, request):
        """Handle requests for active channels"""
        channels_info = {}
        for channel, subscribers in self.channels.items():
            channels_info[channel] = len(subscribers)
        
        return web.json_response({
            "channels": channels_info,
            "total": len(self.channels)
        })

async def main():
    """Main entry point for the PubSub service"""
    # Get configuration from environment variables
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    fallback_redis_urls = [url.strip() for url in os.getenv("FALLBACK_REDIS_URLS", "").split(",") if url.strip()]
    host = os.getenv("PUBSUB_HOST", "0.0.0.0")
    port = int(os.getenv("PUBSUB_PORT", "8004"))
    
    # Create and start the PubSub service
    service = PubSubService(redis_url, fallback_redis_urls, host, port)
    await service.start()

if __name__ == "__main__":
    asyncio.run(main())
