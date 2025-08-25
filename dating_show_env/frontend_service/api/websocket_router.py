"""
FastAPI WebSocket Router for Unified Architecture Integration
Provides WebSocket endpoints for real-time agent state updates.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Set, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.websockets import WebSocketState

# Configure logging
logger = logging.getLogger(__name__)

# Global connection manager
class WebSocketConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        # Active connections by connection ID
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Room-based connections for agent updates
        self.agent_rooms: Dict[str, Set[str]] = {}
        
        # System monitoring connections
        self.system_connections: Set[str] = set()
        
        # Agent subscriptions per connection
        self.agent_subscriptions: Dict[str, Set[str]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info("WebSocket connection manager initialized")
    
    async def connect_agent_room(self, websocket: WebSocket, room_name: str = "general") -> str:
        """Connect to agent updates room."""
        await websocket.accept()
        
        connection_id = f"agent_{datetime.now().timestamp()}"
        self.active_connections[connection_id] = websocket
        
        # Add to room
        if room_name not in self.agent_rooms:
            self.agent_rooms[room_name] = set()
        self.agent_rooms[room_name].add(connection_id)
        
        # Initialize metadata
        self.connection_metadata[connection_id] = {
            "type": "agent_room",
            "room": room_name,
            "connected_at": datetime.now(timezone.utc),
            "subscriptions": set()
        }
        
        logger.info(f"WebSocket connected to agent room '{room_name}': {connection_id}")
        
        # Send connection confirmation
        await self.send_personal_message(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "room": room_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return connection_id
    
    async def connect_system_monitoring(self, websocket: WebSocket) -> str:
        """Connect to system monitoring."""
        await websocket.accept()
        
        connection_id = f"system_{datetime.now().timestamp()}"
        self.active_connections[connection_id] = websocket
        self.system_connections.add(connection_id)
        
        # Initialize metadata
        self.connection_metadata[connection_id] = {
            "type": "system_monitoring",
            "connected_at": datetime.now(timezone.utc)
        }
        
        logger.info(f"WebSocket connected to system monitoring: {connection_id}")
        
        # Send system status
        await self.send_system_status(connection_id)
        
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Disconnect a WebSocket connection."""
        if connection_id in self.active_connections:
            # Remove from rooms
            for room_connections in self.agent_rooms.values():
                room_connections.discard(connection_id)
            
            # Remove from system connections
            self.system_connections.discard(connection_id)
            
            # Clean up subscriptions
            if connection_id in self.agent_subscriptions:
                del self.agent_subscriptions[connection_id]
            
            # Remove connection
            del self.active_connections[connection_id]
            del self.connection_metadata[connection_id]
            
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def broadcast_to_room(self, room_name: str, message: Dict[str, Any]):
        """Broadcast message to all connections in a room."""
        if room_name in self.agent_rooms:
            disconnected = []
            for connection_id in self.agent_rooms[room_name]:
                if connection_id in self.active_connections:
                    try:
                        websocket = self.active_connections[connection_id]
                        await websocket.send_text(json.dumps(message, default=str))
                    except Exception as e:
                        logger.warning(f"Failed to broadcast to {connection_id}: {e}")
                        disconnected.append(connection_id)
            
            # Clean up disconnected connections
            for connection_id in disconnected:
                self.disconnect(connection_id)
    
    async def broadcast_to_system(self, message: Dict[str, Any]):
        """Broadcast message to system monitoring connections."""
        disconnected = []
        for connection_id in self.system_connections:
            if connection_id in self.active_connections:
                try:
                    websocket = self.active_connections[connection_id]
                    await websocket.send_text(json.dumps(message, default=str))
                except Exception as e:
                    logger.warning(f"Failed to broadcast to system {connection_id}: {e}")
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    async def handle_agent_subscription(self, connection_id: str, agent_id: str):
        """Handle agent subscription."""
        if connection_id not in self.agent_subscriptions:
            self.agent_subscriptions[connection_id] = set()
        
        self.agent_subscriptions[connection_id].add(agent_id)
        
        # Update metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscriptions"].add(agent_id)
        
        await self.send_personal_message(connection_id, {
            "type": "subscription_confirmed",
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        logger.debug(f"Agent subscription added: {connection_id} -> {agent_id}")
    
    async def handle_agent_unsubscription(self, connection_id: str, agent_id: str):
        """Handle agent unsubscription."""
        if connection_id in self.agent_subscriptions:
            self.agent_subscriptions[connection_id].discard(agent_id)
        
        # Update metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscriptions"].discard(agent_id)
        
        await self.send_personal_message(connection_id, {
            "type": "unsubscription_confirmed",
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        logger.debug(f"Agent subscription removed: {connection_id} -> {agent_id}")
    
    async def send_agent_state(self, connection_id: str, agent_id: str):
        """Send current agent state to connection."""
        try:
            # Try to get agent state from unified architecture
            agent_state = await self.get_agent_state_from_unified_architecture(agent_id)
            
            if agent_state:
                await self.send_personal_message(connection_id, {
                    "type": "agent_state_response",
                    "agent_id": agent_id,
                    "data": agent_state,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            else:
                await self.send_personal_message(connection_id, {
                    "type": "error",
                    "message": f"Agent {agent_id} not found",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        except Exception as e:
            logger.error(f"Error getting agent state for {agent_id}: {e}")
            await self.send_personal_message(connection_id, {
                "type": "error",
                "message": f"Failed to get agent state: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    async def send_social_network(self, connection_id: str):
        """Send social network data to connection."""
        try:
            # Try to get social network from unified architecture
            social_data = await self.get_social_network_from_unified_architecture()
            
            await self.send_personal_message(connection_id, {
                "type": "social_network_response",
                "data": social_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting social network data: {e}")
            await self.send_personal_message(connection_id, {
                "type": "error",
                "message": f"Failed to get social network: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    async def send_system_status(self, connection_id: str):
        """Send system status to connection."""
        try:
            # Get system status
            system_status = await self.get_system_status()
            
            await self.send_personal_message(connection_id, {
                "type": "system_status",
                "data": system_status,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            await self.send_personal_message(connection_id, {
                "type": "error",
                "message": f"Failed to get system status: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    async def send_performance_metrics(self, connection_id: str):
        """Send performance metrics to connection."""
        try:
            # Get performance metrics
            metrics = await self.get_performance_metrics()
            
            await self.send_personal_message(connection_id, {
                "type": "performance_metrics",
                "data": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            await self.send_personal_message(connection_id, {
                "type": "error",
                "message": f"Failed to get metrics: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    # Integration methods with unified architecture
    
    async def get_agent_state_from_unified_architecture(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent state from unified architecture."""
        try:
            # Import here to avoid circular imports
            from ...dating_show.services.frontend_state_adapter import get_frontend_state_adapter
            
            adapter = get_frontend_state_adapter()
            return adapter.get_agent_for_frontend(agent_id)
        except ImportError:
            logger.warning("Unified architecture not available, using fallback")
            return None
        except Exception as e:
            logger.error(f"Error accessing unified architecture: {e}")
            return None
    
    async def get_social_network_from_unified_architecture(self) -> Dict[str, Any]:
        """Get social network data from unified architecture."""
        try:
            # Import here to avoid circular imports
            from ...dating_show.services.frontend_state_adapter import get_frontend_state_adapter
            
            adapter = get_frontend_state_adapter()
            return adapter.get_social_network_data()
        except ImportError:
            logger.warning("Unified architecture not available, using fallback")
            return {"agents": {}, "relationships": []}
        except Exception as e:
            logger.error(f"Error accessing unified architecture: {e}")
            return {"agents": {}, "relationships": []}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        try:
            # Import here to avoid circular imports
            from ...dating_show.services.unified_agent_manager import get_unified_agent_manager
            
            unified_manager = get_unified_agent_manager()
            
            return {
                "unified_architecture_available": True,
                "total_agents": len(unified_manager.agents),
                "cached_states": len(unified_manager.state_cache),
                "active_websockets": len(self.active_connections),
                "agent_rooms": len(self.agent_rooms),
                "system_connections": len(self.system_connections),
                "total_subscriptions": sum(len(subs) for subs in self.agent_subscriptions.values())
            }
        except ImportError:
            return {
                "unified_architecture_available": False,
                "active_websockets": len(self.active_connections),
                "agent_rooms": len(self.agent_rooms),
                "system_connections": len(self.system_connections)
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            # Import here to avoid circular imports
            from ...dating_show.services.update_pipeline import get_update_pipeline
            
            pipeline = get_update_pipeline()
            return pipeline.get_performance_metrics()
        except ImportError:
            logger.warning("Update pipeline not available")
            return {"error": "Update pipeline not available"}
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}

# Global connection manager instance
connection_manager = WebSocketConnectionManager()

# Create router
router = APIRouter()

@router.websocket("/ws/agents/{room_name}")
async def websocket_agent_room(websocket: WebSocket, room_name: str = "general"):
    """WebSocket endpoint for agent state updates."""
    connection_id = None
    try:
        connection_id = await connection_manager.connect_agent_room(websocket, room_name)
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "ping":
                await connection_manager.send_personal_message(connection_id, {
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            elif message_type == "subscribe_agent":
                agent_id = message.get("agent_id")
                if agent_id:
                    await connection_manager.handle_agent_subscription(connection_id, agent_id)
                else:
                    await connection_manager.send_personal_message(connection_id, {
                        "type": "error",
                        "message": "agent_id required for subscription"
                    })
            
            elif message_type == "unsubscribe_agent":
                agent_id = message.get("agent_id")
                if agent_id:
                    await connection_manager.handle_agent_unsubscription(connection_id, agent_id)
                else:
                    await connection_manager.send_personal_message(connection_id, {
                        "type": "error",
                        "message": "agent_id required for unsubscription"
                    })
            
            elif message_type == "request_agent_state":
                agent_id = message.get("agent_id")
                if agent_id:
                    await connection_manager.send_agent_state(connection_id, agent_id)
                else:
                    await connection_manager.send_personal_message(connection_id, {
                        "type": "error",
                        "message": "agent_id required for state request"
                    })
            
            elif message_type == "request_social_network":
                await connection_manager.send_social_network(connection_id)
            
            else:
                await connection_manager.send_personal_message(connection_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        if connection_id:
            connection_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error in agent room: {e}")
        if connection_id:
            connection_manager.disconnect(connection_id)

@router.websocket("/ws/agents")
async def websocket_agent_default(websocket: WebSocket):
    """WebSocket endpoint for agent state updates (default room)."""
    await websocket_agent_room(websocket, "general")

@router.websocket("/ws/system")
async def websocket_system_monitoring(websocket: WebSocket):
    """WebSocket endpoint for system monitoring."""
    connection_id = None
    try:
        connection_id = await connection_manager.connect_system_monitoring(websocket)
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "request_system_status":
                await connection_manager.send_system_status(connection_id)
            
            elif message_type == "request_performance_metrics":
                await connection_manager.send_performance_metrics(connection_id)
            
            else:
                await connection_manager.send_personal_message(connection_id, {
                    "type": "error",
                    "message": f"Unknown system message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        if connection_id:
            connection_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error in system monitoring: {e}")
        if connection_id:
            connection_manager.disconnect(connection_id)

# Export connection manager for use by other modules
__all__ = ['router', 'connection_manager']