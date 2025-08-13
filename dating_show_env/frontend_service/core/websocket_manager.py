"""
WebSocket connection manager for real-time communication
"""

from fastapi import WebSocket
from typing import List, Dict, Any
import json
import logging
from datetime import datetime

from .models import WebSocketMessage, SimulationUpdate

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            "connected_at": datetime.now(),
            "subscriptions": set()
        }
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_metadata.pop(websocket, None)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific connection"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connections"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_to_subscribers(self, topic: str, message: str):
        """Broadcast to connections subscribed to specific topic"""
        disconnected = []
        for connection in self.active_connections:
            metadata = self.connection_metadata.get(connection, {})
            if topic in metadata.get("subscriptions", set()):
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to subscriber: {e}")
                    disconnected.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def handle_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        try:
            msg_type = message.get("type")
            
            if msg_type == "subscribe":
                topic = message.get("topic")
                if topic and websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["subscriptions"].add(topic)
                    await self.send_personal_message(
                        json.dumps({"type": "subscribed", "topic": topic}),
                        websocket
                    )
            
            elif msg_type == "unsubscribe":
                topic = message.get("topic")
                if topic and websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["subscriptions"].discard(topic)
                    await self.send_personal_message(
                        json.dumps({"type": "unsubscribed", "topic": topic}),
                        websocket
                    )
            
            elif msg_type == "ping":
                await self.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def broadcast_simulation_update(self, update: SimulationUpdate):
        """Broadcast simulation update to all subscribers"""
        message = WebSocketMessage(
            type="simulation_update",
            data=update.dict()
        )
        await self.broadcast_to_subscribers(
            "simulation_updates",
            message.json()
        )
    
    async def broadcast_agent_update(self, agent_name: str, agent_data: Dict[str, Any]):
        """Broadcast agent-specific update"""
        message = WebSocketMessage(
            type="agent_update",
            data={"agent_name": agent_name, "agent_data": agent_data}
        )
        await self.broadcast_to_subscribers(
            f"agent_{agent_name}",
            message.json()
        )
        # Also broadcast to general agent updates
        await self.broadcast_to_subscribers(
            "agent_updates",
            message.json()
        )
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_subscriber_count(self, topic: str) -> int:
        """Get number of subscribers to a topic"""
        count = 0
        for metadata in self.connection_metadata.values():
            if topic in metadata.get("subscriptions", set()):
                count += 1
        return count