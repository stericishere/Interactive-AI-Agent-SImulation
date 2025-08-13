"""
Configuration settings for the Dating Show Frontend Service
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # Server configuration
    host: str = "localhost"
    port: int = 8001
    debug: bool = True
    
    # CORS settings
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]
    
    # Dating show integration
    dating_show_backend_url: str = "http://localhost:8000"
    simulation_data_path: str = "../../environment/frontend_server"
    
    # Performance settings
    max_connections: int = 100
    connection_timeout: int = 30
    
    # Simulation settings
    auto_refresh_interval: int = 5  # seconds
    
    class Config:
        env_file = ".env"