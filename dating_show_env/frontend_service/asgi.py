"""
ASGI config for Dating Show frontend service.
Configures WebSocket support for real-time agent state updates.
"""

import os
import django
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

# Set Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'frontend_service.settings')

# Initialize Django
django.setup()

# Import WebSocket routing after Django setup
try:
    from ..dating_show.websocket_routing import websocket_urlpatterns
    WEBSOCKET_PATTERNS_AVAILABLE = True
except ImportError:
    # Fallback for when unified architecture is not available
    websocket_urlpatterns = []
    WEBSOCKET_PATTERNS_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning("WebSocket routing not available, unified architecture disabled")

# ASGI application configuration
application = ProtocolTypeRouter({
    # HTTP protocol for regular Django views
    "http": get_asgi_application(),
    
    # WebSocket protocol for real-time updates
    "websocket": AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            URLRouter(websocket_urlpatterns)
        )
    ) if WEBSOCKET_PATTERNS_AVAILABLE else None,
})

# Log ASGI configuration
import logging
logger = logging.getLogger(__name__)
logger.info(f"ASGI application configured with WebSocket support: {WEBSOCKET_PATTERNS_AVAILABLE}")
if WEBSOCKET_PATTERNS_AVAILABLE:
    logger.info(f"WebSocket URL patterns: {len(websocket_urlpatterns)} endpoints")
else:
    logger.warning("WebSocket support disabled - unified architecture not available")