
"""
Database Service for Dating Show Integration
Enterprise database management for seamless Django integration with health monitoring and migration management
"""

import os
import sys
from pathlib import Path
import asyncio
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import django
from django.core.management import execute_from_command_line
from django.db import connection, IntegrityError
from django.conf import settings
from asgiref.sync import sync_to_async


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class DatabaseConfig:
    """Configuration for database service"""
    database_url: str
    frontend_server_path: str
    migration_timeout: int = 300  # 5 minutes
    health_check_interval: int = 30  # seconds
    auto_migrate: bool = True
    backup_enabled: bool = True


@dataclass
class MigrationResult:
    """Result of migration operation"""
    success: bool
    applied_migrations: List[str]
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class HealthMetrics:
    """Database health metrics"""
    status: HealthStatus
    connection_pool_size: int
    active_connections: int
    query_time_ms: float
    last_check: datetime
    error_count: int
    uptime_seconds: float


class DatabaseService:
    """
    Enterprise database management for dating show integration.
    
    Provides comprehensive database management including Django migration detection,
    health monitoring, automatic cleanup, and performance optimization.
    """
    
    def __init__(self, config: DatabaseConfig):
        """Initialize database service with configuration"""
        self.config = config
        self.health_status = HealthStatus.UNKNOWN
        self.last_health_check = datetime.now(timezone.utc)
        self.error_count = 0
        self.start_time = datetime.now(timezone.utc)
        self.actual_frontend_path = None  # Will be set by _setup_django
        self._setup_django()
        
    def _setup_django(self):
        """Setup Django environment for database operations"""
        try:
            # Robust path detection - same logic as main.py
            from pathlib import Path
            
            # Try to find the frontend server using multiple strategies
            current_dir = Path.cwd()
            project_root_candidates = [
                current_dir,
                current_dir.parent,
                Path(__file__).parent.parent.parent  # dating_show/services -> project_root
            ]
            
            frontend_server_path = None
            for root_candidate in project_root_candidates:
                potential_path = root_candidate / "environment" / "frontend_server"
                if potential_path.exists() and (potential_path / "manage.py").exists():
                    frontend_server_path = str(potential_path)
                    break
            
            # Use configured path if auto-detection fails
            if not frontend_server_path:
                frontend_server_path = self.config.frontend_server_path
            
            # Verify the path exists
            if not Path(frontend_server_path).exists():
                raise FileNotFoundError(f"Frontend server path not found: {frontend_server_path}")
            
            # Store the actual path for use in other methods
            self.actual_frontend_path = frontend_server_path
            
            # Add frontend server to Python path
            if frontend_server_path not in sys.path:
                sys.path.insert(0, frontend_server_path)
                logger.info(f"Added to Python path: {frontend_server_path}")
            
            # Configure Django settings
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'environment.frontend_server.frontend_server.settings')
            
            # Initialize Django
            if not hasattr(django.conf.settings, 'configured') or not django.conf.settings.configured:
                django.setup()
                logger.info("Django settings import successful!")
            else:
                logger.info("Django already configured")
                
            logger.info("Django environment initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Django environment: {e}")
            logger.error(f"Current working directory: {Path.cwd()}")
            logger.error(f"Configured frontend path: {self.config.frontend_server_path}")
            logger.error(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
            self.health_status = HealthStatus.FAILED
            raise
    
    async def ensure_migrations(self) -> MigrationResult:
        """
        Detect and apply pending Django migrations.
        
        Returns:
            MigrationResult with success status and applied migrations
        """
        start_time = datetime.now()
        applied_migrations = []
        
        try:
            logger.info("Checking for pending migrations...")
            
            # Check for pending migrations
            pending_result = await self._run_django_command(['showmigrations', '--plan'])
            if not pending_result['success']:
                return MigrationResult(
                    success=False,
                    applied_migrations=[],
                    error_message=f"Failed to check migrations: {pending_result['error']}",
                    duration_seconds=(datetime.now() - start_time).total_seconds()
                )
            
            # Parse pending migrations
            pending_migrations = self._parse_pending_migrations(pending_result['output'])
            
            if not pending_migrations:
                logger.info("No pending migrations found")
                return MigrationResult(
                    success=True,
                    applied_migrations=[],
                    duration_seconds=(datetime.now() - start_time).total_seconds()
                )
            
            logger.info(f"Found {len(pending_migrations)} pending migrations")
            
            # Apply migrations
            if self.config.auto_migrate:
                migrate_result = await self._run_django_command(['migrate'], timeout=self.config.migration_timeout)
                if migrate_result['success']:
                    applied_migrations = pending_migrations
                    logger.info(f"Successfully applied {len(applied_migrations)} migrations")
                else:
                    return MigrationResult(
                        success=False,
                        applied_migrations=[],
                        error_message=f"Migration failed: {migrate_result['error']}",
                        duration_seconds=(datetime.now() - start_time).total_seconds()
                    )
            else:
                logger.warning("Auto-migration disabled, skipping migration application")
            
            return MigrationResult(
                success=True,
                applied_migrations=applied_migrations,
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Error in ensure_migrations: {e}")
            return MigrationResult(
                success=False,
                applied_migrations=[],
                error_message=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )
    
    async def sync_agent_models(self, agents: List[Any]) -> None:
        """
        Synchronize PIANO agents with Django models.
        
        Args:
            agents: List of PIANO agent instances to synchronize
        """
        # Use sync_to_async for Django ORM operations
        await sync_to_async(self._sync_agent_models_sync)(agents)
    
    def _sync_agent_models_sync(self, agents: List[Any]) -> None:
        """Synchronous agent model synchronization implementation"""
        try:
            from dating_show_api.models import Agent
            
            logger.info(f"Synchronizing {len(agents)} agents with database...")
            
            synced_count = 0
            error_count = 0
            
            for agent in agents:
                try:
                    # Extract agent data
                    agent_data = self._extract_agent_data(agent)
                    
                    # Update or create agent in database
                    agent_obj, created = Agent.objects.update_or_create(
                        agent_id=agent_data['agent_id'],
                        defaults=agent_data
                    )
                    
                    if created:
                        logger.debug(f"Created new agent: {agent_data['agent_id']}")
                    else:
                        logger.debug(f"Updated existing agent: {agent_data['agent_id']}")
                    
                    synced_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to sync agent {getattr(agent, 'agent_id', 'unknown')}: {e}")
                    error_count += 1
            
            logger.info(f"Agent synchronization complete: {synced_count} synced, {error_count} errors")
            
        except Exception as e:
            logger.error(f"Error in sync_agent_models: {e}")
            raise
    
    async def health_check(self) -> HealthMetrics:
        """
        Comprehensive database health validation.
        
        Returns:
            HealthMetrics with current database health status
        """
        # Use sync_to_async for Django database operations
        return await sync_to_async(self._sync_health_check)()
    
    def _sync_health_check(self) -> HealthMetrics:
        """Synchronous health check implementation"""
        start_time = datetime.now()
        
        try:
            # Test database connection
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                query_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Get connection pool metrics
            connection_info = self._get_connection_info()
            
            # Determine health status
            if query_time_ms > 1000:  # >1 second is critical
                status = HealthStatus.CRITICAL
            elif query_time_ms > 500:  # >500ms is warning
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            self.health_status = status
            self.last_health_check = datetime.now(timezone.utc)
            
            # Reset error count on successful check
            if status != HealthStatus.CRITICAL:
                self.error_count = 0
            
            return HealthMetrics(
                status=status,
                connection_pool_size=connection_info.get('pool_size', 0),
                active_connections=connection_info.get('active_connections', 0),
                query_time_ms=query_time_ms,
                last_check=self.last_health_check,
                error_count=self.error_count,
                uptime_seconds=(datetime.now(timezone.utc) - self.start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.error_count += 1
            self.health_status = HealthStatus.FAILED
            
            return HealthMetrics(
                status=HealthStatus.FAILED,
                connection_pool_size=0,
                active_connections=0,
                query_time_ms=0.0,
                last_check=datetime.now(timezone.utc),
                error_count=self.error_count,
                uptime_seconds=(datetime.now(timezone.utc) - self.start_time).total_seconds()
            )
    
    async def cleanup_stale_data(self, max_age: timedelta) -> int:
        """
        Remove outdated simulation data.
        
        Args:
            max_age: Maximum age for data retention
            
        Returns:
            Number of records cleaned up
        """
        try:
            from environment.frontend_server.dating_show_api.models import (
                AgentMemorySnapshot, SimulationState, ComplianceViolation
            )
            
            cutoff_time = datetime.now(timezone.utc) - max_age
            cleanup_count = 0
            
            logger.info(f"Cleaning up data older than {max_age}")
            
            # Clean up old memory snapshots with low importance
            old_memories = AgentMemorySnapshot.objects.filter(
                created_at__lt=cutoff_time,
                importance_score__lt=1.0
            )
            memory_count = old_memories.count()
            old_memories.delete()
            cleanup_count += memory_count
            logger.info(f"Cleaned up {memory_count} old memory snapshots")
            
            # Clean up completed simulations older than max_age
            old_simulations = SimulationState.objects.filter(
                updated_at__lt=cutoff_time,
                status__in=['completed', 'stopped', 'error']
            )
            sim_count = old_simulations.count()
            old_simulations.delete()
            cleanup_count += sim_count
            logger.info(f"Cleaned up {sim_count} old simulations")
            
            # Clean up resolved violations older than max_age
            old_violations = ComplianceViolation.objects.filter(
                resolved_at__lt=cutoff_time,
                resolved=True
            )
            violation_count = old_violations.count()
            old_violations.delete()
            cleanup_count += violation_count
            logger.info(f"Cleaned up {violation_count} resolved violations")
            
            logger.info(f"Total cleanup: {cleanup_count} records removed")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Error in cleanup_stale_data: {e}")
            raise
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            'health_status': self.health_status.value,
            'last_health_check': self.last_health_check.isoformat(),
            'error_count': self.error_count,
            'uptime_seconds': (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            'config': {
                'database_url': self.config.database_url[:20] + "..." if len(self.config.database_url) > 20 else self.config.database_url,
                'auto_migrate': self.config.auto_migrate,
                'backup_enabled': self.config.backup_enabled,
                'migration_timeout': self.config.migration_timeout
            }
        }
    
    # Private helper methods
    
    async def _run_django_command(self, command: List[str], timeout: int = 60) -> Dict[str, Any]:
        """Run Django management command"""
        try:
            project_root = Path(self.actual_frontend_path).parent.parent
            manage_py_path = Path(self.actual_frontend_path) / 'manage.py'

            full_command = [sys.executable, str(manage_py_path)] + command
            
            process = await asyncio.create_subprocess_exec(
                *full_command,
                cwd=str(project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            
            return {
                'success': process.returncode == 0,
                'output': stdout.decode('utf-8'),
                'error': stderr.decode('utf-8') if process.returncode != 0 else None
            }
            
        except asyncio.TimeoutError:
            return {
                'success': False,
                'output': '',
                'error': f'Command timed out after {timeout} seconds'
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e)
            }
    
    def _parse_pending_migrations(self, migration_output: str) -> List[str]:
        """Parse pending migrations from showmigrations output"""
        pending = []
        
        for line in migration_output.split('\n'):
            line = line.strip()
            if line.startswith('[ ]'):  # Unapplied migration
                # Extract migration name
                parts = line.split()
                if len(parts) >= 2:
                    pending.append(parts[1])
        
        return pending
    
    def _get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information"""
        try:
            # This is a simplified version - in production you might want
            # to use more sophisticated connection pool monitoring
            return {
                'pool_size': getattr(connection, 'pool_size', 1),
                'active_connections': 1  # Simplified for Django
            }
        except Exception:
            return {'pool_size': 0, 'active_connections': 0}
    
    def _extract_agent_data(self, agent: Any) -> Dict[str, Any]:
        """Extract agent data for database synchronization"""
        try:
            # Extract basic agent information
            agent_data = {
                'agent_id': getattr(agent, 'agent_id', str(id(agent))),
                'name': getattr(agent, 'name', f'Agent_{id(agent)}'),
                'current_role': getattr(agent, 'current_role', 'participant'),
                'is_active': True,
                'last_activity': datetime.now(timezone.utc)
            }
            
            # Extract specialization if available
            if hasattr(agent, 'specialization'):
                specialization = getattr(agent, 'specialization')
                if isinstance(specialization, dict):
                    agent_data['specialization'] = specialization
                else:
                    agent_data['specialization'] = {'type': str(specialization)}
            
            # Extract performance rating if available
            if hasattr(agent, 'performance_rating'):
                rating = getattr(agent, 'performance_rating', 0.0)
                agent_data['performance_rating'] = max(0.0, min(10.0, float(rating)))
            
            return agent_data
            
        except Exception as e:
            logger.warning(f"Error extracting agent data: {e}")
            # Return minimal agent data
            return {
                'agent_id': str(id(agent)),
                'name': f'Agent_{id(agent)}',
                'current_role': 'participant',
                'is_active': True,
                'last_activity': datetime.now(timezone.utc),
                'specialization': {},
                'performance_rating': 0.0
            }


# Service factory function
def create_database_service(
    database_url: str,
    frontend_server_path: str,
    **kwargs
) -> DatabaseService:
    """
    Create and configure database service instance.
    
    Args:
        database_url: Database connection URL
        frontend_server_path: Path to Django frontend server
        **kwargs: Additional configuration options
        
    Returns:
        Configured DatabaseService instance
    """
    config = DatabaseConfig(
        database_url=database_url,
        frontend_server_path=frontend_server_path,
        **kwargs
    )
    
    return DatabaseService(config)
