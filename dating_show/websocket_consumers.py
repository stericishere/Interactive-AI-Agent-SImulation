"""
WebSocket Consumers for Real-time Agent State Updates
Integrates with UpdatePipeline for live frontend synchronization.
"""

import json
import logging
from typing import Dict, Any
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.exceptions import DenyConnection
from datetime import datetime, timezone

from .services.update_pipeline import get_update_pipeline, UpdateType

# Configure logging
logger = logging.getLogger(__name__)


class AgentStateConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time agent state updates.
    
    Provides live synchronization of agent states, social networks,
    and memory updates with the frontend applications.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.room_group_name = None
        self.user_groups = []
        self.update_pipeline = None
        self.connection_id = None
        
    async def connect(self):
        """Handle WebSocket connection."""
        try:
            # Get room name from URL route
            self.room_name = self.scope['url_route']['kwargs'].get('room_name', 'general')
            self.room_group_name = f'agents_{self.room_name}'
            
            # Generate connection ID
            self.connection_id = f"conn_{datetime.now().timestamp()}"
            
            # Join room group
            await self.channel_layer.group_add(
                self.room_group_name,
                self.channel_name
            )
            
            # Initialize update pipeline integration
            self.update_pipeline = get_update_pipeline()
            
            # Register WebSocket with update pipeline
            self.update_pipeline.register_websocket(self, groups=[self.room_name])
            
            # Accept the WebSocket connection
            await self.accept()
            
            # Send connection confirmation
            await self.send(text_data=json.dumps({
                'type': 'connection_established',
                'connection_id': self.connection_id,
                'room': self.room_name,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }))
            
            logger.info(f"WebSocket connected: {self.connection_id} to room {self.room_name}")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            await self.close()
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        try:
            # Unregister from update pipeline
            if self.update_pipeline:
                self.update_pipeline.unregister_websocket(self)
            
            # Leave room group
            if self.room_group_name:
                await self.channel_layer.group_discard(
                    self.room_group_name,
                    self.channel_name
                )
            
            logger.info(f"WebSocket disconnected: {self.connection_id} from room {self.room_name}")
            
        except Exception as e:
            logger.error(f"WebSocket disconnect error: {e}")
    
    async def receive(self, text_data):
        """Handle messages from WebSocket."""
        try:
            data = json.loads(text_data)
            message_type = data.get('type')
            
            if message_type == 'ping':
                await self.send_pong()
            elif message_type == 'subscribe_agent':
                await self.handle_agent_subscription(data)
            elif message_type == 'unsubscribe_agent':
                await self.handle_agent_unsubscription(data)
            elif message_type == 'request_agent_state':
                await self.handle_agent_state_request(data)
            elif message_type == 'request_social_network':
                await self.handle_social_network_request(data)
            elif message_type == 'performance_metrics':
                await self.handle_performance_metrics_request()
            else:
                await self.send_error(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self.send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
            await self.send_error(f"Internal error: {str(e)}")
    
    async def send_pong(self):
        """Send pong response to ping."""
        await self.send(text_data=json.dumps({
            'type': 'pong',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }))
    
    async def handle_agent_subscription(self, data: Dict[str, Any]):
        """Handle agent-specific subscription."""
        agent_id = data.get('agent_id')
        if not agent_id:
            await self.send_error("agent_id required for subscription")
            return
        
        # Add to agent-specific group
        agent_group = f"agent_{agent_id}"
        await self.channel_layer.group_add(
            agent_group,
            self.channel_name
        )
        
        if agent_group not in self.user_groups:
            self.user_groups.append(agent_group)
        
        await self.send(text_data=json.dumps({
            'type': 'subscription_confirmed',
            'agent_id': agent_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }))
        
        logger.debug(f"Subscribed {self.connection_id} to agent {agent_id}")
    
    async def handle_agent_unsubscription(self, data: Dict[str, Any]):
        """Handle agent-specific unsubscription."""
        agent_id = data.get('agent_id')
        if not agent_id:
            await self.send_error("agent_id required for unsubscription")
            return
        
        # Remove from agent-specific group
        agent_group = f"agent_{agent_id}"
        await self.channel_layer.group_discard(
            agent_group,
            self.channel_name
        )
        
        if agent_group in self.user_groups:
            self.user_groups.remove(agent_group)
        
        await self.send(text_data=json.dumps({
            'type': 'unsubscription_confirmed',
            'agent_id': agent_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }))
        
        logger.debug(f"Unsubscribed {self.connection_id} from agent {agent_id}")
    
    async def handle_agent_state_request(self, data: Dict[str, Any]):
        """Handle request for current agent state."""
        agent_id = data.get('agent_id')
        if not agent_id:
            await self.send_error("agent_id required for state request")
            return
        
        try:
            # Get state from unified architecture
            from .services.frontend_state_adapter import get_frontend_state_adapter
            frontend_adapter = get_frontend_state_adapter()
            
            agent_state = frontend_adapter.get_agent_for_frontend(agent_id)
            
            if agent_state:
                await self.send(text_data=json.dumps({
                    'type': 'agent_state_response',
                    'agent_id': agent_id,
                    'data': agent_state,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }))
            else:
                await self.send_error(f"Agent {agent_id} not found")
                
        except Exception as e:
            logger.error(f"Error getting agent state for {agent_id}: {e}")
            await self.send_error(f"Failed to get agent state: {str(e)}")
    
    async def handle_social_network_request(self, data: Dict[str, Any]):
        """Handle request for social network data."""
        try:
            # Get social network from unified architecture
            from .services.frontend_state_adapter import get_frontend_state_adapter
            frontend_adapter = get_frontend_state_adapter()
            
            social_data = frontend_adapter.get_social_network_data()\n            \n            await self.send(text_data=json.dumps({\n                'type': 'social_network_response',\n                'data': social_data,\n                'timestamp': datetime.now(timezone.utc).isoformat()\n            }))\n            \n        except Exception as e:\n            logger.error(f\"Error getting social network data: {e}\")\n            await self.send_error(f\"Failed to get social network: {str(e)}\")\n    \n    async def handle_performance_metrics_request(self):\n        \"\"\"Handle request for update pipeline performance metrics.\"\"\"\n        try:\n            if self.update_pipeline:\n                metrics = self.update_pipeline.get_performance_metrics()\n                \n                await self.send(text_data=json.dumps({\n                    'type': 'performance_metrics_response',\n                    'data': metrics,\n                    'timestamp': datetime.now(timezone.utc).isoformat()\n                }))\n            else:\n                await self.send_error(\"Update pipeline not available\")\n                \n        except Exception as e:\n            logger.error(f\"Error getting performance metrics: {e}\")\n            await self.send_error(f\"Failed to get metrics: {str(e)}\")\n    \n    async def send_error(self, message: str):\n        \"\"\"Send error message to client.\"\"\"\n        await self.send(text_data=json.dumps({\n            'type': 'error',\n            'message': message,\n            'timestamp': datetime.now(timezone.utc).isoformat()\n        }))\n    \n    # Channel layer message handlers\n    \n    async def agent_update(self, event):\n        \"\"\"Handle agent update broadcast.\"\"\"\n        await self.send(text_data=json.dumps(event))\n    \n    async def batch_update(self, event):\n        \"\"\"Handle batch update broadcast.\"\"\"\n        await self.send(text_data=json.dumps(event))\n    \n    async def social_network_update(self, event):\n        \"\"\"Handle social network update broadcast.\"\"\"\n        await self.send(text_data=json.dumps(event))\n    \n    async def memory_update(self, event):\n        \"\"\"Handle memory update broadcast.\"\"\"\n        await self.send(text_data=json.dumps(event))\n\n\nclass AgentSystemConsumer(AsyncWebsocketConsumer):\n    \"\"\"\n    WebSocket consumer for system-level updates and monitoring.\n    \n    Provides real-time system metrics, performance data,\n    and administrative updates.\n    \"\"\"\n    \n    def __init__(self, *args, **kwargs):\n        super().__init__(*args, **kwargs)\n        self.system_group = \"system_updates\"\n        self.connection_id = None\n        self.update_pipeline = None\n    \n    async def connect(self):\n        \"\"\"Handle system WebSocket connection.\"\"\"\n        try:\n            # Check if user has admin privileges (implement your auth logic)\n            # For now, we'll allow all connections\n            \n            self.connection_id = f\"system_conn_{datetime.now().timestamp()}\"\n            \n            # Join system updates group\n            await self.channel_layer.group_add(\n                self.system_group,\n                self.channel_name\n            )\n            \n            # Initialize update pipeline\n            self.update_pipeline = get_update_pipeline()\n            self.update_pipeline.register_websocket(self, groups=[\"system\"])\n            \n            await self.accept()\n            \n            # Send system status\n            await self.send_system_status()\n            \n            logger.info(f\"System WebSocket connected: {self.connection_id}\")\n            \n        except Exception as e:\n            logger.error(f\"System WebSocket connection failed: {e}\")\n            await self.close()\n    \n    async def disconnect(self, close_code):\n        \"\"\"Handle system WebSocket disconnection.\"\"\"\n        try:\n            if self.update_pipeline:\n                self.update_pipeline.unregister_websocket(self)\n            \n            await self.channel_layer.group_discard(\n                self.system_group,\n                self.channel_name\n            )\n            \n            logger.info(f\"System WebSocket disconnected: {self.connection_id}\")\n            \n        except Exception as e:\n            logger.error(f\"System WebSocket disconnect error: {e}\")\n    \n    async def receive(self, text_data):\n        \"\"\"Handle system messages.\"\"\"\n        try:\n            data = json.loads(text_data)\n            message_type = data.get('type')\n            \n            if message_type == 'request_system_status':\n                await self.send_system_status()\n            elif message_type == 'request_performance_metrics':\n                await self.send_performance_metrics()\n            elif message_type == 'reset_metrics':\n                await self.handle_metrics_reset()\n            else:\n                await self.send_error(f\"Unknown system message type: {message_type}\")\n                \n        except json.JSONDecodeError:\n            await self.send_error(\"Invalid JSON format\")\n        except Exception as e:\n            logger.error(f\"System WebSocket receive error: {e}\")\n            await self.send_error(f\"Internal error: {str(e)}\")\n    \n    async def send_system_status(self):\n        \"\"\"Send current system status.\"\"\"\n        try:\n            # Get unified architecture status\n            from .services.unified_agent_manager import get_unified_agent_manager\n            from .services.frontend_state_adapter import get_frontend_state_adapter\n            \n            unified_manager = get_unified_agent_manager()\n            frontend_adapter = get_frontend_state_adapter()\n            \n            status = {\n                'unified_architecture_available': True,\n                'total_agents': len(unified_manager.agents),\n                'cached_states': len(unified_manager.state_cache),\n                'update_pipeline_active': self.update_pipeline is not None,\n                'websocket_connections': len(self.update_pipeline.websocket_connections) if self.update_pipeline else 0,\n                'system_uptime': 'active'\n            }\n            \n            await self.send(text_data=json.dumps({\n                'type': 'system_status',\n                'data': status,\n                'timestamp': datetime.now(timezone.utc).isoformat()\n            }))\n            \n        except Exception as e:\n            logger.error(f\"Error getting system status: {e}\")\n            await self.send_error(f\"Failed to get system status: {str(e)}\")\n    \n    async def send_performance_metrics(self):\n        \"\"\"Send current performance metrics.\"\"\"\n        try:\n            if self.update_pipeline:\n                metrics = self.update_pipeline.get_performance_metrics()\n                \n                await self.send(text_data=json.dumps({\n                    'type': 'performance_metrics',\n                    'data': metrics,\n                    'timestamp': datetime.now(timezone.utc).isoformat()\n                }))\n            else:\n                await self.send_error(\"Update pipeline not available\")\n                \n        except Exception as e:\n            logger.error(f\"Error getting performance metrics: {e}\")\n            await self.send_error(f\"Failed to get metrics: {str(e)}\")\n    \n    async def handle_metrics_reset(self):\n        \"\"\"Handle metrics reset request.\"\"\"\n        try:\n            if self.update_pipeline:\n                self.update_pipeline.reset_metrics()\n                \n                await self.send(text_data=json.dumps({\n                    'type': 'metrics_reset_confirmed',\n                    'timestamp': datetime.now(timezone.utc).isoformat()\n                }))\n            else:\n                await self.send_error(\"Update pipeline not available\")\n                \n        except Exception as e:\n            logger.error(f\"Error resetting metrics: {e}\")\n            await self.send_error(f\"Failed to reset metrics: {str(e)}\")\n    \n    async def send_error(self, message: str):\n        \"\"\"Send error message to client.\"\"\"\n        await self.send(text_data=json.dumps({\n            'type': 'error',\n            'message': message,\n            'timestamp': datetime.now(timezone.utc).isoformat()\n        }))\n    \n    # System event handlers\n    \n    async def system_alert(self, event):\n        \"\"\"Handle system alert broadcast.\"\"\"\n        await self.send(text_data=json.dumps(event))\n    \n    async def performance_update(self, event):\n        \"\"\"Handle performance update broadcast.\"\"\"\n        await self.send(text_data=json.dumps(event))