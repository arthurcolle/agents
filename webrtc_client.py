#!/usr/bin/env python3
"""
WebRTC Client for peer-to-peer communication
Provides a Python client for WebRTC connections
"""

import os
import json
import logging
import asyncio
import time
import uuid
import threading
from typing import Dict, List, Set, Optional, Any, Callable
import websockets
from websockets.exceptions import ConnectionClosed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-client")

class WebRTCClient:
    """
    WebRTC Client for peer-to-peer communication
    Connects to a signaling server and establishes WebRTC connections
    """
    def __init__(self, signaling_url="ws://localhost:8003"):
        self.signaling_url = signaling_url
        self.websocket = None
        self.peer_id = None
        self.room_id = None
        self.peers = set()
        self.event_handlers = {}
        self.connected = False
        self.reconnect_task = None
        self.listener_task = None
        self._shutdown_event = asyncio.Event()
        logger.info(f"WebRTC client initialized with signaling URL: {signaling_url}")
    
    async def connect(self):
        """Connect to the signaling server"""
        try:
            self.websocket = await websockets.connect(self.signaling_url)
            
            # Start message listener
            self.listener_task = asyncio.create_task(self._message_listener())
            
            # Wait for peer ID
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "peer_id":
                        self.peer_id = data.get("peer_id")
                        self.connected = True
                        logger.info(f"Connected to signaling server with peer ID: {self.peer_id}")
                        
                        # Trigger connected event
                        await self._trigger_event("connected", {"peer_id": self.peer_id})
                        break
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from signaling server: {message}")
            
            return self.peer_id
        except Exception as e:
            logger.error(f"Failed to connect to signaling server: {e}")
            self.connected = False
            
            # Start reconnection task
            if not self.reconnect_task or self.reconnect_task.done():
                self.reconnect_task = asyncio.create_task(self._reconnect_loop())
                
            return None
    
    async def _reconnect_loop(self):
        """Attempt to reconnect to the signaling server"""
        retry_count = 0
        max_retries = 10
        base_delay = 1.0
        max_delay = 30.0
        
        while not self._shutdown_event.is_set() and retry_count < max_retries:
            retry_count += 1
            delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)
            
            logger.info(f"Reconnecting to signaling server in {delay:.1f} seconds (attempt {retry_count}/{max_retries})...")
            await asyncio.sleep(delay)
            
            try:
                await self.connect()
                if self.connected:
                    logger.info("Reconnected to signaling server")
                    
                    # Rejoin room if we were in one
                    if self.room_id:
                        await self.join_room(self.room_id)
                        
                    retry_count = 0
                    break
            except Exception as e:
                logger.error(f"Reconnection attempt failed: {e}")
        
        if retry_count >= max_retries and not self.connected:
            logger.error("Failed to reconnect after maximum retries")
            await self._trigger_event("connection_failed", {"max_retries": max_retries})
    
    async def _message_listener(self):
        """Listen for messages from the signaling server"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from signaling server: {message}")
        except ConnectionClosed:
            logger.info("Connection to signaling server closed")
            self.connected = False
            
            # Trigger disconnected event
            await self._trigger_event("disconnected", {})
            
            # Start reconnection task
            if not self.reconnect_task or self.reconnect_task.done():
                self.reconnect_task = asyncio.create_task(self._reconnect_loop())
        except Exception as e:
            logger.error(f"Error in message listener: {e}")
            self.connected = False
    
    async def _handle_message(self, data):
        """Handle a message from the signaling server"""
        msg_type = data.get("type")
        
        if msg_type == "room_joined":
            # We joined a room
            self.room_id = data.get("room_id")
            logger.info(f"Joined room: {self.room_id}")
            
            # Trigger room_joined event
            await self._trigger_event("room_joined", {"room_id": self.room_id})
            
            # Request peers list
            await self.get_peers()
            
        elif msg_type == "room_left":
            # We left a room
            old_room_id = self.room_id
            self.room_id = None
            self.peers.clear()
            logger.info(f"Left room: {old_room_id}")
            
            # Trigger room_left event
            await self._trigger_event("room_left", {"room_id": old_room_id})
            
        elif msg_type == "peer_joined":
            # A new peer joined the room
            peer_id = data.get("peer_id")
            if peer_id and peer_id != self.peer_id:
                self.peers.add(peer_id)
                logger.info(f"Peer joined: {peer_id}")
                
                # Trigger peer_joined event
                await self._trigger_event("peer_joined", {"peer_id": peer_id})
                
        elif msg_type == "peer_left":
            # A peer left the room
            peer_id = data.get("peer_id")
            if peer_id and peer_id in self.peers:
                self.peers.remove(peer_id)
                logger.info(f"Peer left: {peer_id}")
                
                # Trigger peer_left event
                await self._trigger_event("peer_left", {"peer_id": peer_id})
                
        elif msg_type == "peers_list":
            # Received list of peers in the room
            peers_list = data.get("peers", [])
            self.peers = set(peers_list)
            if self.peer_id in self.peers:
                self.peers.remove(self.peer_id)
            
            logger.info(f"Peers in room: {len(self.peers)}")
            
            # Trigger peers_list event
            await self._trigger_event("peers_list", {"peers": list(self.peers)})
            
        elif msg_type == "offer":
            # Received an offer from a peer
            from_peer_id = data.get("from_peer_id")
            sdp = data.get("sdp")
            
            logger.info(f"Received offer from peer: {from_peer_id}")
            
            # Trigger offer event
            await self._trigger_event("offer", {
                "from_peer_id": from_peer_id,
                "sdp": sdp
            })
            
        elif msg_type == "answer":
            # Received an answer from a peer
            from_peer_id = data.get("from_peer_id")
            sdp = data.get("sdp")
            
            logger.info(f"Received answer from peer: {from_peer_id}")
            
            # Trigger answer event
            await self._trigger_event("answer", {
                "from_peer_id": from_peer_id,
                "sdp": sdp
            })
            
        elif msg_type == "ice_candidate":
            # Received an ICE candidate from a peer
            from_peer_id = data.get("from_peer_id")
            candidate = data.get("candidate")
            
            logger.debug(f"Received ICE candidate from peer: {from_peer_id}")
            
            # Trigger ice_candidate event
            await self._trigger_event("ice_candidate", {
                "from_peer_id": from_peer_id,
                "candidate": candidate
            })
            
        else:
            logger.debug(f"Received unknown message type: {msg_type}")
            
            # Trigger generic message event
            await self._trigger_event("message", data)
    
    async def _trigger_event(self, event_type, data):
        """Trigger an event handler"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(data))
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
    
    def on(self, event_type, handler):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        return self
    
    async def send(self, data):
        """Send a message to the signaling server"""
        if not self.connected or not self.websocket:
            logger.error("Not connected to signaling server")
            return False
            
        try:
            if isinstance(data, dict):
                data = json.dumps(data)
            await self.websocket.send(data)
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.connected = False
            return False
    
    async def join_room(self, room_id=None):
        """Join a room"""
        return await self.send({
            "type": "join_room",
            "room_id": room_id
        })
    
    async def leave_room(self):
        """Leave the current room"""
        if not self.room_id:
            return False
            
        return await self.send({
            "type": "leave_room"
        })
    
    async def get_peers(self):
        """Get the list of peers in the current room"""
        return await self.send({
            "type": "get_peers"
        })
    
    async def send_offer(self, target_peer_id, sdp):
        """Send an offer to a peer"""
        return await self.send({
            "type": "offer",
            "target_peer_id": target_peer_id,
            "sdp": sdp
        })
    
    async def send_answer(self, target_peer_id, sdp):
        """Send an answer to a peer"""
        return await self.send({
            "type": "answer",
            "target_peer_id": target_peer_id,
            "sdp": sdp
        })
    
    async def send_ice_candidate(self, target_peer_id, candidate):
        """Send an ICE candidate to a peer"""
        return await self.send({
            "type": "ice_candidate",
            "target_peer_id": target_peer_id,
            "candidate": candidate
        })
    
    async def close(self):
        """Close the connection to the signaling server"""
        logger.info("Closing WebRTC client")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Leave room if in one
        if self.room_id:
            await self.leave_room()
        
        # Cancel tasks
        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
                
        if self.reconnect_task:
            self.reconnect_task.cancel()
            try:
                await self.reconnect_task
            except asyncio.CancelledError:
                pass
        
        # Close websocket
        if self.websocket:
            await self.websocket.close()
            
        self.connected = False
        logger.info("WebRTC client closed")

async def example():
    """Example usage of WebRTC client"""
    # Create client
    client = WebRTCClient()
    
    # Register event handlers
    client.on("connected", lambda data: print(f"Connected with peer ID: {data['peer_id']}"))
    client.on("room_joined", lambda data: print(f"Joined room: {data['room_id']}"))
    client.on("peers_list", lambda data: print(f"Peers in room: {data['peers']}"))
    
    # Connect to signaling server
    await client.connect()
    
    # Join a room
    await client.join_room("test-room")
    
    # Wait for a while
    await asyncio.sleep(30)
    
    # Leave the room
    await client.leave_room()
    
    # Close the client
    await client.close()

if __name__ == "__main__":
    asyncio.run(example())
