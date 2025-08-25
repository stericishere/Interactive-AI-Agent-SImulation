#!/usr/bin/env python3
"""
ASGI Production Configuration for Dating Show Frontend Service
Optimized for production deployment with WebSocket support
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set production environment
os.environ.setdefault('DATING_SHOW_ENV', 'production')

from main import app

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/asgi.log') if os.path.exists('logs') else logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ASGI application for production deployment
application = app

# Log startup information
logger.info("ASGI Production Configuration Loaded")
logger.info(f"Python Path: {sys.path[0]}")
logger.info(f"Working Directory: {os.getcwd()}")
logger.info("WebSocket routes available at /api/ws/")

# Health check for deployment validation
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy",
        "service": "dating-show-frontend",
        "websocket_support": True,
        "unified_architecture": True,
        "timestamp": "2024-08-22T00:00:00Z"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Production server configuration
    uvicorn.run(
        "asgi_production:application",
        host="0.0.0.0",
        port=8001,
        workers=1,  # Single worker for WebSocket consistency
        loop="uvloop",  # High-performance event loop
        http="httptools",  # High-performance HTTP parser
        ws="websockets",  # WebSocket support
        log_level="info",
        access_log=True,
        server_header=False,  # Security: hide server info
        date_header=False,    # Performance: disable date header
    )