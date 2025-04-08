#!/usr/bin/env python3
"""
WebRTC Signaling Server for peer-to-peer communication
Provides WebSocket-based signaling for WebRTC peers
"""

import os
import json
import logging
import asyncio
import uuid
import time
import signal
from typing import Dict, List, Set, Optional, Any
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-signaling")

class SignalingServer:
    """
    WebRTC Signaling Server using WebSockets
    Handles peer discovery, ICE candidate exchange, and SDP negotiation
    """
    def __init__(self, host="0.0.0.0", port=8003):
        self.host = host
        self.port = port
        self.peers: Dict[str, WebSocketServerProtocol] = {}
        self.rooms: Dict[str, Set[str]] = {}
        self.peer_to_room: Dict[str, str] = {}
        self.server = None
        self._shutdown_event = asyncio.Event()
        logger.info(f"Signaling server initialized on {host}:{port}")
    
    async def start(self):
        """Start the signaling server"""
        # Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        
        # Start WebSocket server
        self.server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
        )
        
        logger.info(f"Signaling server started on ws://{self.host}:{self.port}")
        
        # Wait for shutdown signal
        await self._shutdown_event.wait()
    
    async def shutdown(self):
        """Shutdown the signaling server gracefully"""
        logger.info("Shutting down signaling server...")
        
        # Close all peer connections
        close_coroutines = [ws.close(1001, "Server shutting down") 
                           for ws in self.peers.values()]
        if close_coroutines:
            await asyncio.gather(*close_coroutines, return_exceptions=True)
        
        # Stop the server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Signal shutdown complete
        self._shutdown_event.set()
        
        logger.info("Signaling server shutdown complete")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket connection"""
        # Generate a unique peer ID
        peer_id = str(uuid.uuid4())
        
        try:
            # Register the new peer
            self.peers[peer_id] = websocket
            
            # Send peer ID to the client
            await websocket.send(json.dumps({
                "type": "peer_id",
                "peer_id": peer_id
            }))
            
            logger.info(f"Peer {peer_id} connected")
            
            # Handle messages from this peer
            async for message in websocket:
                await self.handle_message(peer_id, message)
                
        except ConnectionClosed:
            logger.info(f"Peer {peer_id} connection closed")
        except Exception as e:
            logger.error(f"Error handling connection for peer {peer_id}: {e}")
        finally:
            # Clean up when the connection is closed
            await self.handle_peer_disconnect(peer_id)
    
    async def handle_message(self, peer_id: str, message: str):
        """Handle a message from a peer"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "join_room":
                # Peer wants to join a room
                room_id = data.get("room_id")
                if not room_id:
                    # Create a new room if none specified
                    room_id = str(uuid.uuid4())
                
                await self.join_room(peer_id, room_id)
                
            elif msg_type == "leave_room":
                # Peer wants to leave their current room
                await self.leave_room(peer_id)
                
            elif msg_type == "offer":
                # Peer is sending an offer to another peer
                target_peer_id = data.get("target_peer_id")
                if target_peer_id in self.peers:
                    # Forward the offer to the target peer
                    await self.peers[target_peer_id].send(json.dumps({
                        "type": "offer",
                        "sdp": data.get("sdp"),
                        "from_peer_id": peer_id
                    }))
                    
            elif msg_type == "answer":
                # Peer is sending an answer to another peer
                target_peer_id = data.get("target_peer_id")
                if target_peer_id in self.peers:
                    # Forward the answer to the target peer
                    await self.peers[target_peer_id].send(json.dumps({
                        "type": "answer",
                        "sdp": data.get("sdp"),
                        "from_peer_id": peer_id
                    }))
                    
            elif msg_type == "ice_candidate":
                # Peer is sending an ICE candidate to another peer
                target_peer_id = data.get("target_peer_id")
                if target_peer_id in self.peers:
                    # Forward the ICE candidate to the target peer
                    await self.peers[target_peer_id].send(json.dumps({
                        "type": "ice_candidate",
                        "candidate": data.get("candidate"),
                        "from_peer_id": peer_id
                    }))
                    
            elif msg_type == "get_peers":
                # Peer wants to get a list of peers in their room
                await self.send_peers_list(peer_id)
                
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from peer {peer_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling message from peer {peer_id}: {e}")
    
    async def join_room(self, peer_id: str, room_id: str):
        """Add a peer to a room"""
        # Leave current room if in one
        if peer_id in self.peer_to_room:
            await self.leave_room(peer_id)
        
        # Create room if it doesn't exist
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
            logger.info(f"Created new room: {room_id}")
        
        # Add peer to room
        self.rooms[room_id].add(peer_id)
        self.peer_to_room[peer_id] = room_id
        
        logger.info(f"Peer {peer_id} joined room {room_id}")
        
        # Notify peer they've joined the room
        if peer_id in self.peers:
            await self.peers[peer_id].send(json.dumps({
                "type": "room_joined",
                "room_id": room_id
            }))
        
        # Notify all peers in the room about the new peer
        await self.broadcast_to_room(room_id, {
            "type": "peer_joined",
            "peer_id": peer_id
        }, exclude_peer=peer_id)
        
        # Send the list of peers to the new peer
        await self.send_peers_list(peer_id)
    
    async def leave_room(self, peer_id: str):
        """Remove a peer from their current room"""
        if peer_id not in self.peer_to_room:
            return
        
        room_id = self.peer_to_room[peer_id]
        
        # Remove peer from room
        if room_id in self.rooms:
            self.rooms[room_id].discard(peer_id)
            
            # Remove room if empty
            if not self.rooms[room_id]:
                del self.rooms[room_id]
                logger.info(f"Room {room_id} removed (empty)")
        
        # Remove room reference
        del self.peer_to_room[peer_id]
        
        logger.info(f"Peer {peer_id} left room {room_id}")
        
        # Notify peer they've left the room
        if peer_id in self.peers:
            await self.peers[peer_id].send(json.dumps({
                "type": "room_left",
                "room_id": room_id
            }))
        
        # Notify all peers in the room about the peer leaving
        await self.broadcast_to_room(room_id, {
            "type": "peer_left",
            "peer_id": peer_id
        })
    
    async def handle_peer_disconnect(self, peer_id: str):
        """Handle a peer disconnection"""
        # Remove peer from their room
        await self.leave_room(peer_id)
        
        # Remove peer from peers list
        if peer_id in self.peers:
            del self.peers[peer_id]
            
        logger.info(f"Peer {peer_id} disconnected")
    
    async def broadcast_to_room(self, room_id: str, message: Dict, exclude_peer: str = None):
        """Broadcast a message to all peers in a room"""
        if room_id not in self.rooms:
            return
        
        # Convert message to JSON string
        message_str = json.dumps(message)
        
        # Send to all peers in the room except excluded peer
        for peer_id in self.rooms[room_id]:
            if peer_id != exclude_peer and peer_id in self.peers:
                try:
                    await self.peers[peer_id].send(message_str)
                except Exception as e:
                    logger.error(f"Error sending to peer {peer_id}: {e}")
    
    async def send_peers_list(self, peer_id: str):
        """Send a list of peers in the room to a specific peer"""
        if peer_id not in self.peer_to_room or peer_id not in self.peers:
            return
        
        room_id = self.peer_to_room[peer_id]
        peers_list = list(self.rooms[room_id])
        
        try:
            await self.peers[peer_id].send(json.dumps({
                "type": "peers_list",
                "peers": peers_list,
                "room_id": room_id
            }))
        except Exception as e:
            logger.error(f"Error sending peers list to peer {peer_id}: {e}")

async def main():
    """Main entry point for the signaling server"""
    # Get configuration from environment variables
    host = os.getenv("SIGNALING_HOST", "0.0.0.0")
    port = int(os.getenv("SIGNALING_PORT", "8003"))
    
    # Create and start the signaling server
    server = SignalingServer(host, port)
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())
