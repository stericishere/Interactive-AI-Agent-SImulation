"""
WebSocket Consumer for Django Channels integration with UpdatePipeline.

This module provides Django Channels WebSocket consumers that integrate
with the UpdatePipeline service for real-time agent state broadcasting.

Optional dependency: Only used if Django Channels is available.
"""

import json
import logging
from typing import Dict, Any, Optional

try:
    from channels.generic.websocket import AsyncWebsocketConsumer
    from channels.exceptions import DenyConnection
    CHANNELS_AVAILABLE = True
except ImportError:
    # Fallback base class if Channels not available
    class AsyncWebsocketConsumer:
        """Fallback base class when Django Channels is not available"""
        pass
    CHANNELS_AVAILABLE = False

from .update_pipeline import get_update_pipeline, UpdateMessage, UpdatePriority

logger = logging.getLogger(__name__)


class DatingShowConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time dating show updates.
    
    Handles WebSocket connections from the frontend and integrates
    with the UpdatePipeline service for broadcasting agent state changes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_name = "dating_show_updates"
        self.connection_id = None
        self.update_pipeline = None
        self.subscriptions = set()
    
    async def connect(self):
        """Handle WebSocket connection"""
        if not CHANNELS_AVAILABLE:
            logger.error("Django Channels not available - WebSocket connection denied")
            await self.close()
            return
        
        try:
            # Extract connection info
            self.connection_id = f"ws_{id(self)}_{self.scope.get('client', ['unknown'])[0]}"
            
            # Get update pipeline
            self.update_pipeline = get_update_pipeline()
            
            # Join the general updates group
            await self.channel_layer.group_add(
                self.group_name,
                self.channel_name
            )
            
            # Register with update pipeline
            success = await self.update_pipeline.register_websocket(
                self.connection_id,
                self,
                {self.group_name}
            )
            
            if success:
                await self.accept()
                logger.info(f"WebSocket connected: {self.connection_id}")
                
                # Send initial connection confirmation
                await self.send(text_data=json.dumps({
                    'type': 'connection_established',
                    'connection_id': self.connection_id,
                    'group': self.group_name,
                    'timestamp': self.update_pipeline.performance_metrics.total_updates
                }))
            else:
                logger.warning(f"UpdatePipeline rejected connection: {self.connection_id}")
                await self.close()
                
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self.close()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        try:
            # Leave the general updates group
            if hasattr(self, 'channel_layer'):
                await self.channel_layer.group_discard(
                    self.group_name,
                    self.channel_name
                )
            
            # Unregister from update pipeline
            if self.update_pipeline and self.connection_id:
                await self.update_pipeline.unregister_websocket(self.connection_id)
            
            logger.info(f"WebSocket disconnected: {self.connection_id} (code: {close_code})")
            
        except Exception as e:
            logger.error(f"WebSocket disconnect error: {e}")
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(text_data)
            message_type = data.get('type', 'unknown')
            
            if message_type == 'subscribe':
                await self._handle_subscription(data)
            elif message_type == 'unsubscribe':
                await self._handle_unsubscription(data)
            elif message_type == 'ping':
                await self._handle_ping(data)
            elif message_type == 'agent_action':
                await self._handle_agent_action(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Unknown message type: {message_type}'
                }))
                
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from WebSocket {self.connection_id}: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Internal server error'
            }))
    
    async def _handle_subscription(self, data: Dict[str, Any]):
        """Handle subscription requests"""
        channels = data.get('channels', [])
        
        for channel in channels:
            if channel not in self.subscriptions:
                # Join the channel group
                await self.channel_layer.group_add(channel, self.channel_name)
                self.subscriptions.add(channel)
                
                logger.debug(f"WebSocket {self.connection_id} subscribed to {channel}")
        
        await self.send(text_data=json.dumps({
            'type': 'subscription_confirmed',
            'channels': list(self.subscriptions)
        }))
    
    async def _handle_unsubscription(self, data: Dict[str, Any]):
        """Handle unsubscription requests"""
        channels = data.get('channels', [])
        
        for channel in channels:
            if channel in self.subscriptions:
                # Leave the channel group
                await self.channel_layer.group_discard(channel, self.channel_name)
                self.subscriptions.discard(channel)
                
                logger.debug(f"WebSocket {self.connection_id} unsubscribed from {channel}")
        
        await self.send(text_data=json.dumps({
            'type': 'unsubscription_confirmed',
            'channels': list(self.subscriptions)
        }))
    
    async def _handle_ping(self, data: Dict[str, Any]):
        """Handle ping requests"""
        await self.send(text_data=json.dumps({
            'type': 'pong',
            'timestamp': data.get('timestamp'),
            'server_time': self.update_pipeline.performance_metrics.total_updates
        }))
    
    async def _handle_agent_action(self, data: Dict[str, Any]):
        """Handle agent action requests from frontend"""
        try:
            agent_id = data.get('agent_id')
            action = data.get('action')
            params = data.get('params', {})
            
            if not agent_id or not action:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'agent_id and action are required'
                }))
                return
            
            # Queue the action as an update
            if self.update_pipeline:
                self.update_pipeline.queue_agent_update(
                    agent_id=agent_id,
                    update_type='frontend_action',
                    data={
                        'action': action,
                        'params': params,
                        'source': 'frontend',
                        'connection_id': self.connection_id
                    },
                    priority=UpdatePriority.HIGH
                )
                
                await self.send(text_data=json.dumps({
                    'type': 'action_queued',
                    'agent_id': agent_id,
                    'action': action
                }))
            else:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': 'UpdatePipeline not available'
                }))
                
        except Exception as e:
            logger.error(f"Error handling agent action: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Failed to process agent action'
            }))
    
    # Channel layer message handlers
    
    async def broadcast_message(self, event):
        """Handle broadcast messages from the UpdatePipeline"""
        try:
            # Extract the actual data from the event
            message_data = event.get('data', {})
            
            # Send to WebSocket
            await self.send(text_data=json.dumps(message_data))
            
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
    
    async def agent_update(self, event):
        """Handle agent update messages"""
        try:
            await self.send(text_data=json.dumps(event))
        except Exception as e:
            logger.error(f"Error sending agent update: {e}")
    
    async def system_notification(self, event):
        """Handle system notification messages"""
        try:
            await self.send(text_data=json.dumps({
                'type': 'system_notification',
                'level': event.get('level', 'info'),
                'message': event.get('message', ''),
                'timestamp': event.get('timestamp')
            }))
        except Exception as e:
            logger.error(f"Error sending system notification: {e}")


class DatingShowAdminConsumer(AsyncWebsocketConsumer):
    """
    Administrative WebSocket consumer for monitoring and control.
    
    Provides enhanced monitoring capabilities and system control
    for administrators and developers.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_name = "dating_show_admin"
        self.connection_id = None
        self.update_pipeline = None
        self.is_admin = False
    
    async def connect(self):
        """Handle admin WebSocket connection"""
        if not CHANNELS_AVAILABLE:
            await self.close()
            return
        
        try:
            # Check admin permissions (implement your own logic)
            self.is_admin = await self._check_admin_permissions()
            
            if not self.is_admin:
                logger.warning("Non-admin attempted admin WebSocket connection")
                await self.close()
                return
            
            # Setup connection
            self.connection_id = f"admin_ws_{id(self)}"
            self.update_pipeline = get_update_pipeline()
            
            # Join admin group
            await self.channel_layer.group_add(
                self.group_name,
                self.channel_name
            )
            
            await self.accept()
            logger.info(f"Admin WebSocket connected: {self.connection_id}")
            
            # Send initial status
            health_status = self.update_pipeline.get_health_status()
            await self.send(text_data=json.dumps({
                'type': 'admin_connected',
                'connection_id': self.connection_id,
                'health_status': health_status
            }))
            
        except Exception as e:
            logger.error(f"Admin WebSocket connection failed: {e}")
            await self.close()
    
    async def disconnect(self, close_code):
        """Handle admin WebSocket disconnection"""
        try:
            if hasattr(self, 'channel_layer'):
                await self.channel_layer.group_discard(
                    self.group_name,
                    self.channel_name
                )
            
            logger.info(f"Admin WebSocket disconnected: {self.connection_id}")
            
        except Exception as e:
            logger.error(f"Admin WebSocket disconnect error: {e}")
    
    async def receive(self, text_data):
        """Handle admin WebSocket messages"""
        if not self.is_admin:
            await self.close()
            return
        
        try:
            data = json.loads(text_data)
            command = data.get('command', 'unknown')
            
            if command == 'get_health':
                await self._handle_get_health()
            elif command == 'get_metrics':
                await self._handle_get_metrics()
            elif command == 'reset_circuit_breaker':
                await self._handle_reset_circuit_breaker()
            elif command == 'get_connections':
                await self._handle_get_connections()
            else:
                await self.send(text_data=json.dumps({
                    'type': 'error',
                    'message': f'Unknown admin command: {command}'
                }))
                
        except Exception as e:
            logger.error(f"Error processing admin command: {e}")
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Internal server error'
            }))
    
    async def _check_admin_permissions(self) -> bool:
        """Check if connection has admin permissions"""
        # Implement your own admin check logic here
        # For now, return True for development
        # In production, check user authentication, IP whitelist, etc.
        return True
    
    async def _handle_get_health(self):
        """Handle health status request"""
        health_status = self.update_pipeline.get_health_status()
        await self.send(text_data=json.dumps({
            'type': 'health_status',
            'data': health_status
        }))
    
    async def _handle_get_metrics(self):
        """Handle metrics request"""
        metrics = self.update_pipeline.get_performance_metrics()
        await self.send(text_data=json.dumps({
            'type': 'performance_metrics',
            'data': metrics
        }))
    
    async def _handle_reset_circuit_breaker(self):
        """Handle circuit breaker reset"""
        try:
            # Reset circuit breaker
            self.update_pipeline.circuit_breaker.metrics.state = \
                self.update_pipeline.circuit_breaker.metrics.state.CLOSED
            self.update_pipeline.circuit_breaker.metrics.failure_count = 0
            
            await self.send(text_data=json.dumps({
                'type': 'circuit_breaker_reset',
                'message': 'Circuit breaker reset successfully'
            }))
            
        except Exception as e:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': f'Failed to reset circuit breaker: {e}'
            }))
    
    async def _handle_get_connections(self):
        """Handle get connections request"""
        connection_count = self.update_pipeline.websocket_manager.get_connection_count()
        await self.send(text_data=json.dumps({
            'type': 'connection_info',
            'data': {
                'active_connections': connection_count,
                'total_connections': len(self.update_pipeline.websocket_manager.connections)
            }
        }))


# Routing configuration for Django Channels
def get_websocket_routing():
    """Get WebSocket routing configuration for Django Channels"""
    if not CHANNELS_AVAILABLE:
        return []
    
    try:
        from django.urls import re_path
        
        return [
            re_path(r'ws/dating_show/$', DatingShowConsumer.as_asgi()),
            re_path(r'ws/dating_show/admin/$', DatingShowAdminConsumer.as_asgi()),
        ]
    except ImportError:
        return []


# Helper function for manual WebSocket setup (non-Channels)
class SimpleWebSocketHandler:
    """
    Simple WebSocket handler for non-Django Channels environments.
    
    Provides basic WebSocket functionality using standard WebSocket libraries.
    """
    
    def __init__(self, websocket, path):
        self.websocket = websocket
        self.path = path
        self.connection_id = f"simple_ws_{id(self)}"
        self.update_pipeline = get_update_pipeline()
    
    async def handle_connection(self):
        """Handle WebSocket connection lifecycle"""
        try:
            # Register with update pipeline
            success = await self.update_pipeline.register_websocket(
                self.connection_id,
                self.websocket
            )
            
            if not success:
                await self.websocket.close()
                return
            
            logger.info(f"Simple WebSocket connected: {self.connection_id}")
            
            # Send connection confirmation
            await self.websocket.send(json.dumps({
                'type': 'connection_established',
                'connection_id': self.connection_id
            }))
            
            # Handle messages
            async for message in self.websocket:
                await self._handle_message(message)
                
        except Exception as e:
            logger.error(f"Simple WebSocket error: {e}")
        finally:
            # Cleanup
            await self.update_pipeline.unregister_websocket(self.connection_id)
            logger.info(f"Simple WebSocket disconnected: {self.connection_id}")
    
    async def _handle_message(self, message):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            message_type = data.get('type', 'unknown')
            
            if message_type == 'ping':
                await self.websocket.send(json.dumps({
                    'type': 'pong',
                    'timestamp': data.get('timestamp')
                }))
            else:
                logger.debug(f"Received message type: {message_type}")
                
        except Exception as e:
            logger.warning(f"Error handling WebSocket message: {e}")


# Integration with popular WebSocket servers
async def setup_websocket_server(host='localhost', port=8765):
    """
    Setup simple WebSocket server using websockets library.
    
    This is a fallback for environments without Django Channels.
    """
    try:
        import websockets
        
        async def handler(websocket, path):
            handler_instance = SimpleWebSocketHandler(websocket, path)
            await handler_instance.handle_connection()
        
        server = await websockets.serve(handler, host, port)
        logger.info(f"WebSocket server started on ws://{host}:{port}")
        return server
        
    except ImportError:
        logger.error("websockets library not available - install with: pip install websockets")
        return None
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {e}")
        return None