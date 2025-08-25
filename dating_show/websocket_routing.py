"""
WebSocket URL Routing for Real-time Updates
Configures WebSocket endpoints for agent state synchronization.
"""

from django.urls import re_path
from . import websocket_consumers

websocket_urlpatterns = [
    # Agent state updates WebSocket
    re_path(r'ws/agents/(?P<room_name>\w+)/$', websocket_consumers.AgentStateConsumer.as_asgi()),
    
    # Default agent state updates (general room)
    re_path(r'ws/agents/$', websocket_consumers.AgentStateConsumer.as_asgi()),
    
    # System monitoring WebSocket
    re_path(r'ws/system/$', websocket_consumers.AgentSystemConsumer.as_asgi()),
]