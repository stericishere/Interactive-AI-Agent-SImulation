"""
Dating Show Orchestration Service
Main orchestration service for complete integration of PIANO agents with Django frontend
"""

import os
import sys
import asyncio
import logging
import signal
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

from .database_service import DatabaseService, DatabaseConfig, create_database_service
from .enhanced_bridge import EnhancedFrontendBridge, create_enhanced_bridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class OrchestrationConfig:
    """Configuration for orchestration service"""
    # Database configuration
    database_url: str = "postgresql://localhost/dating_show"
    frontend_server_path: str = "./environment/frontend_server"
    
    # Bridge configuration
    frontend_url: str = "http://localhost:8001"
    bridge_update_interval: float = 1.0
    bridge_batch_size: int = 10
    
    # PIANO configuration
    piano_config_path: str = "./dating_show/config"
    max_agents: int = 50
    simulation_steps: int = 1000
    
    # Service configuration
    health_check_interval: float = 30.0
    auto_cleanup: bool = True
    cleanup_interval: timedelta = timedelta(hours=24)
    log_level: str = "INFO"
    
    # Safety configuration
    max_startup_time: float = 120.0  # 2 minutes
    graceful_shutdown_timeout: float = 30.0
    
    @classmethod
    def from_file(cls, config_path: str) -> 'OrchestrationConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return cls()  # Return default config
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(asdict(self), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")


@dataclass
class ServiceHealth:
    """Service health status"""
    database_service: bool = False
    frontend_bridge: bool = False
    piano_agents: bool = False
    overall_status: ServiceStatus = ServiceStatus.STOPPED
    last_health_check: datetime = None
    error_message: Optional[str] = None


class DatingShowOrchestrator:
    """
    Main orchestration service for dating show integration.
    
    Manages the complete lifecycle of:
    - Database service initialization
    - Enhanced frontend bridge
    - PIANO agent registration and synchronization
    - Health monitoring and error recovery
    """
    
    def __init__(self, config: OrchestrationConfig):
        """Initialize orchestrator with configuration"""
        self.config = config
        self.status = ServiceStatus.STOPPED
        self.start_time: Optional[datetime] = None
        
        # Service instances
        self.database_service: Optional[DatabaseService] = None
        self.frontend_bridge: Optional[EnhancedFrontendBridge] = None
        self.piano_agents: List[Any] = []
        
        # Health monitoring
        self.service_health = ServiceHealth()
        self.last_cleanup = datetime.now(timezone.utc)
        
        # Background tasks
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Setup logging
        logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
        
        logger.info("Dating Show Orchestrator initialized")
    
    async def initialize_database(self) -> None:
        """Setup and validate database configuration"""
        try:
            logger.info("Initializing database service...")
            
            # Create database service
            self.database_service = create_database_service(
                database_url=self.config.database_url,
                frontend_server_path=self.config.frontend_server_path,
                migration_timeout=300,
                auto_migrate=True
            )
            
            # Ensure migrations are applied
            migration_result = await self.database_service.ensure_migrations()
            if not migration_result.success:
                raise Exception(f"Database migration failed: {migration_result.error_message}")
            
            if migration_result.applied_migrations:
                logger.info(f"Applied {len(migration_result.applied_migrations)} database migrations")
            else:
                logger.info("Database is up to date")
            
            # Perform health check
            health_metrics = await self.database_service.health_check()
            if health_metrics.status.value in ['failed', 'critical']:
                raise Exception(f"Database health check failed: {health_metrics.status}")
            
            self.service_health.database_service = True
            logger.info("Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.service_health.database_service = False
            self.service_health.error_message = str(e)
            raise
    
    async def start_frontend_bridge(self) -> None:
        """Initialize and start enhanced bridge service"""
        try:
            logger.info("Starting frontend bridge service...")
            
            # Create enhanced bridge
            self.frontend_bridge = create_enhanced_bridge(
                frontend_url=self.config.frontend_url,
                update_interval=self.config.bridge_update_interval,
                batch_size=self.config.bridge_batch_size,
                discovery_interval=30.0,
                health_check_interval=5.0
            )
            
            # Start bridge service
            self.frontend_bridge.start_bridge()
            
            # Verify bridge is running
            await asyncio.sleep(1.0)  # Give it time to start
            health_metrics = self.frontend_bridge.get_health_metrics()
            if health_metrics.status.value in ['error', 'disconnected']:
                raise Exception(f"Bridge service failed to start: {health_metrics.status}")
            
            self.service_health.frontend_bridge = True
            logger.info("Frontend bridge service started successfully")
            
        except Exception as e:
            logger.error(f"Frontend bridge startup failed: {e}")
            self.service_health.frontend_bridge = False
            self.service_health.error_message = str(e)
            raise
    
    async def register_piano_agents(self, agents: List[Any]) -> None:
        """Register all PIANO agents with frontend integration"""
        try:
            logger.info(f"Registering {len(agents)} PIANO agents...")
            
            if not self.database_service:
                raise Exception("Database service not initialized")
            
            if not self.frontend_bridge:
                raise Exception("Frontend bridge not started")
            
            # Store agent references
            self.piano_agents = agents
            
            # Synchronize agents with database
            await self.database_service.sync_agent_models(agents)
            
            # Auto-discover and register agents with bridge
            discovery_result = await self.frontend_bridge.auto_discover_agents()
            logger.info(f"Agent discovery: {discovery_result.total_agents} total, "
                       f"{len(discovery_result.new_agents)} new")
            
            # Queue initial agent updates
            for agent in agents:
                agent_data = self._extract_agent_data(agent)
                self.frontend_bridge.queue_agent_update(agent_data['agent_id'], agent_data)
            
            self.service_health.piano_agents = True
            logger.info(f"Successfully registered {len(agents)} PIANO agents")
            
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            self.service_health.piano_agents = False
            self.service_health.error_message = str(e)
            raise
    
    async def start_simulation_loop(self) -> None:
        """Begin main simulation with frontend synchronization"""
        try:
            logger.info("Starting simulation loop...")
            
            if not all([self.database_service, self.frontend_bridge, self.piano_agents]):
                raise Exception("Services not properly initialized")
            
            self.status = ServiceStatus.RUNNING
            self.start_time = datetime.now(timezone.utc)
            
            # Start background monitoring tasks
            self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            if self.config.auto_cleanup:
                self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Simulation loop started - orchestrator is now running")
            
            # Main simulation loop
            step = 0
            while not self.shutdown_event.is_set() and step < self.config.simulation_steps:
                try:
                    # This is where the main PIANO simulation step would occur
                    # For now, we'll simulate by updating agent states
                    await self._simulation_step(step)
                    
                    step += 1
                    await asyncio.sleep(0.1)  # 100ms per step
                    
                except Exception as e:
                    logger.error(f"Error in simulation step {step}: {e}")
                    # Continue simulation unless it's a critical error
                    if "critical" in str(e).lower():
                        break
            
            logger.info(f"Simulation completed after {step} steps")
            
        except Exception as e:
            logger.error(f"Simulation loop failed: {e}")
            self.status = ServiceStatus.ERROR
            self.service_health.error_message = str(e)
            raise
    
    async def handle_shutdown(self) -> None:
        """Graceful service shutdown with cleanup"""
        try:
            logger.info("Initiating graceful shutdown...")
            self.status = ServiceStatus.STOPPING
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Stop background tasks
            if self.health_monitor_task:
                self.health_monitor_task.cancel()
                try:
                    await asyncio.wait_for(self.health_monitor_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await asyncio.wait_for(self.cleanup_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Stop frontend bridge
            if self.frontend_bridge:
                self.frontend_bridge.stop_bridge()
                logger.info("Frontend bridge stopped")
            
            # Database service doesn't need explicit shutdown
            if self.database_service:
                logger.info("Database service connections closed")
            
            self.status = ServiceStatus.STOPPED
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self.status = ServiceStatus.ERROR
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        return {
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0,
            'service_health': asdict(self.service_health),
            'config_summary': {
                'max_agents': self.config.max_agents,
                'simulation_steps': self.config.simulation_steps,
                'frontend_url': self.config.frontend_url,
                'auto_cleanup': self.config.auto_cleanup
            },
            'services': {
                'database': self.database_service.get_service_status() if self.database_service else None,
                'bridge': self.frontend_bridge.get_performance_summary() if self.frontend_bridge else None,
                'piano_agents': len(self.piano_agents)
            }
        }
    
    # Private helper methods
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                if datetime.now(timezone.utc) - self.last_cleanup > self.config.cleanup_interval:
                    await self._perform_cleanup()
                    self.last_cleanup = datetime.now(timezone.utc)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            self.service_health.last_health_check = datetime.now(timezone.utc)
            
            # Check database service
            if self.database_service:
                db_health = await self.database_service.health_check()
                self.service_health.database_service = db_health.status.value != 'failed'
            
            # Check frontend bridge
            if self.frontend_bridge:
                bridge_health = self.frontend_bridge.get_health_metrics()
                self.service_health.frontend_bridge = bridge_health.status.value not in ['error', 'disconnected']
            
            # Check PIANO agents
            self.service_health.piano_agents = len(self.piano_agents) > 0
            
            # Determine overall status
            if all([self.service_health.database_service, 
                   self.service_health.frontend_bridge, 
                   self.service_health.piano_agents]):
                self.service_health.overall_status = ServiceStatus.RUNNING
            else:
                self.service_health.overall_status = ServiceStatus.ERROR
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.service_health.overall_status = ServiceStatus.ERROR
            self.service_health.error_message = str(e)
    
    async def _perform_cleanup(self):
        """Perform system cleanup"""
        try:
            logger.info("Performing system cleanup...")
            
            if self.database_service:
                cleanup_count = await self.database_service.cleanup_stale_data(
                    max_age=timedelta(days=7)
                )
                logger.info(f"Cleaned up {cleanup_count} stale database records")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def _simulation_step(self, step: int):
        """Execute one simulation step"""
        # This would integrate with the actual PIANO simulation
        # For now, we'll simulate agent activity
        
        if step % 100 == 0:  # Log every 100 steps
            logger.info(f"Simulation step {step}/{self.config.simulation_steps}")
        
        # Simulate agent updates
        for i, agent in enumerate(self.piano_agents[:5]):  # Update first 5 agents per step
            if (step + i) % 10 == 0:  # Update each agent every 10 steps
                agent_data = self._extract_agent_data(agent)
                agent_data['current_action'] = f"Step {step} action"
                self.frontend_bridge.queue_agent_update(agent_data['agent_id'], agent_data)
    
    def _extract_agent_data(self, agent: Any) -> Dict[str, Any]:
        """Extract agent data for synchronization"""
        try:
            return {
                'agent_id': getattr(agent, 'agent_id', f'agent_{id(agent)}'),
                'name': getattr(agent, 'name', f'Agent {id(agent)}'),
                'current_role': getattr(agent, 'current_role', 'participant'),
                'specialization': getattr(agent, 'specialization', {}),
                'skills': getattr(agent, 'skills', {}),
                'memory': getattr(agent, 'memory', {}),
                'location': getattr(agent, 'location', {}),
                'current_action': getattr(agent, 'current_action', 'idle')
            }
        except Exception as e:
            logger.warning(f"Error extracting agent data: {e}")
            return {
                'agent_id': f'agent_{id(agent)}',
                'name': f'Agent {id(agent)}',
                'current_role': 'participant',
                'specialization': {},
                'skills': {},
                'memory': {},
                'location': {},
                'current_action': 'unknown'
            }


# Factory function
async def create_orchestrator(config_path: Optional[str] = None, **kwargs) -> DatingShowOrchestrator:
    """
    Create and configure orchestration service.
    
    Args:
        config_path: Optional path to configuration file
        **kwargs: Configuration overrides
        
    Returns:
        Configured DatingShowOrchestrator instance
    """
    if config_path and os.path.exists(config_path):
        config = OrchestrationConfig.from_file(config_path)
    else:
        config = OrchestrationConfig()
    
    # Apply any configuration overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return DatingShowOrchestrator(config)