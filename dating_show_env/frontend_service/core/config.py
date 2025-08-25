"""
Configuration settings for the Dating Show Frontend Service
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Environment detection
    environment: str = os.getenv('DATING_SHOW_ENV', 'development')
    
    # Server configuration
    host: str = "localhost" if os.getenv('DATING_SHOW_ENV') != 'production' else "0.0.0.0"
    port: int = int(os.getenv('PORT', 8001))
    debug: bool = os.getenv('DATING_SHOW_ENV') != 'production'
    
    # Production security settings
    secret_key: str = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    ssl_redirect: bool = os.getenv('DATING_SHOW_ENV') == 'production'
    
    # Dating show integration
    dating_show_backend_url: str = os.getenv('DATING_SHOW_BACKEND_URL', "http://localhost:8001")
    simulation_data_path: str = os.getenv('SIMULATION_DATA_PATH', "../../environment/frontend_server")
    
    # Performance settings (production optimized)
    max_connections: int = int(os.getenv('MAX_CONNECTIONS', 500 if os.getenv('DATING_SHOW_ENV') == 'production' else 100))
    connection_timeout: int = int(os.getenv('CONNECTION_TIMEOUT', 60 if os.getenv('DATING_SHOW_ENV') == 'production' else 30))
    
    # Production logging
    log_level: str = os.getenv('LOG_LEVEL', 'INFO' if os.getenv('DATING_SHOW_ENV') == 'production' else 'DEBUG')
    log_file: str = os.getenv('LOG_FILE', 'logs/dating_show_frontend.log')
    
    # Simulation settings
    auto_refresh_interval: int = 5  # seconds
    
    # Unified Architecture settings
    enable_unified_architecture: bool = True
    update_pipeline_enabled: bool = True
    websocket_heartbeat_interval: int = 30  # seconds
    websocket_max_message_size: int = 1024 * 1024  # 1MB
    
    # Performance monitoring
    performance_monitoring_enabled: bool = True
    metrics_retention_hours: int = 24
    alert_thresholds: dict = {
        "processing_time_warning": 80,
        "processing_time_critical": 100,
        "success_rate_warning": 95,
        "success_rate_critical": 90,
        "queue_size_warning": 50,
        "queue_size_critical": 100
    }
    
    @property
    def allowed_origins(self) -> List[str]:
        """Get CORS allowed origins from environment or defaults"""
        env_origins = os.getenv('ALLOWED_ORIGINS')
        if env_origins:
            return [origin.strip() for origin in env_origins.split(',') if origin.strip()]
        
        if self.environment == 'production':
            # Production: Restrict CORS to specific domains
            return [
                "https://dating-show.example.com",
                "https://api.dating-show.example.com"
            ]
        else:
            # Development: Allow local origins
            return [
                "http://localhost:3000",
                "http://localhost:8001", 
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8001",
            ]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == 'production'
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == 'development'
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from environment