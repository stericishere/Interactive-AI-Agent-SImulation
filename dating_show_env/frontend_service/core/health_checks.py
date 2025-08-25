"""
Comprehensive System Health Checks for Dating Show Frontend Service
Epic 4: Production Deployment & Validation
"""

import asyncio
import aiohttp
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheckResult:
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    duration_ms: float
    timestamp: datetime

class HealthCheckManager:
    """Comprehensive health check management for production deployment"""
    
    def __init__(self, settings):
        self.settings = settings
        self.health_checks: Dict[str, callable] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.register_default_checks()
    
    def register_default_checks(self):
        """Register default health checks for the dating show system"""
        self.register_check("database_connection", self.check_database_connection)
        self.register_check("dating_show_backend", self.check_dating_show_backend)
        self.register_check("unified_architecture", self.check_unified_architecture)
        self.register_check("websocket_support", self.check_websocket_support)
        self.register_check("file_system", self.check_file_system)
        self.register_check("memory_usage", self.check_memory_usage)
        self.register_check("disk_space", self.check_disk_space)
        self.register_check("update_pipeline", self.check_update_pipeline)
        self.register_check("circuit_breaker", self.check_circuit_breaker)
    
    def register_check(self, name: str, check_func: callable):
        """Register a health check function"""
        self.health_checks[name] = check_func
        logger.debug(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self.health_checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
                details={},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        
        start_time = time.perf_counter()
        try:
            result = await self.health_checks[name]()
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            if isinstance(result, HealthCheckResult):
                result.duration_ms = duration_ms
                result.timestamp = datetime.now(timezone.utc)
                self.last_results[name] = result
                return result
            else:
                # Handle simple boolean/dict returns
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                result = HealthCheckResult(
                    name=name,
                    status=status,
                    message="Check completed" if result else "Check failed",
                    details=result if isinstance(result, dict) else {},
                    duration_ms=duration_ms,
                    timestamp=datetime.now(timezone.utc)
                )
                self.last_results[name] = result
                return result
        
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                duration_ms=duration_ms,
                timestamp=datetime.now(timezone.utc)
            )
            self.last_results[name] = result
            logger.error(f"Health check '{name}' failed: {e}")
            return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks concurrently"""
        tasks = [self.run_check(name) for name in self.health_checks.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            result.name: result for result in results 
            if isinstance(result, HealthCheckResult)
        }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        results = await self.run_all_checks()
        
        # Calculate overall health
        healthy_count = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)
        total_count = len(results)
        
        if unhealthy_count > total_count * 0.3:  # More than 30% unhealthy
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count + unhealthy_count > total_count * 0.2:  # More than 20% degraded/unhealthy
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "dating-show-frontend",
            "version": "1.0.0",
            "environment": self.settings.environment,
            "summary": {
                "total_checks": total_count,
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "success_rate": round((healthy_count / total_count) * 100, 2) if total_count > 0 else 0
            },
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "details": result.details
                }
                for name, result in results.items()
            }
        }
    
    # Individual Health Check Implementations
    
    async def check_database_connection(self) -> HealthCheckResult:
        """Check database connectivity (if using database)"""
        # For now, this is a placeholder as the current system doesn't use a persistent database
        return HealthCheckResult(
            name="database_connection",
            status=HealthStatus.HEALTHY,
            message="No database configured - using in-memory state",
            details={"type": "in_memory", "persistent": False},
            duration_ms=0.0,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def check_dating_show_backend(self) -> HealthCheckResult:
        """Check connectivity to the dating show backend"""
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{self.settings.dating_show_backend_url}/health"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return HealthCheckResult(
                            name="dating_show_backend",
                            status=HealthStatus.HEALTHY,
                            message="Backend connection successful",
                            details={"backend_url": self.settings.dating_show_backend_url, "response": data},
                            duration_ms=0.0,
                            timestamp=datetime.now(timezone.utc)
                        )
                    else:
                        return HealthCheckResult(
                            name="dating_show_backend",
                            status=HealthStatus.DEGRADED,
                            message=f"Backend returned status {response.status}",
                            details={"backend_url": self.settings.dating_show_backend_url, "status_code": response.status},
                            duration_ms=0.0,
                            timestamp=datetime.now(timezone.utc)
                        )
        except Exception as e:
            return HealthCheckResult(
                name="dating_show_backend",
                status=HealthStatus.UNHEALTHY,
                message=f"Backend connection failed: {str(e)}",
                details={"backend_url": self.settings.dating_show_backend_url, "error": str(e)},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def check_unified_architecture(self) -> HealthCheckResult:
        """Check unified architecture components availability"""
        try:
            # Try to import unified architecture components
            from ...dating_show.services.unified_agent_manager import get_unified_agent_manager
            from ...dating_show.services.update_pipeline import get_update_pipeline
            from ...dating_show.services.frontend_state_adapter import get_frontend_state_adapter
            
            # Test component initialization
            unified_manager = get_unified_agent_manager()
            update_pipeline = get_update_pipeline()
            frontend_adapter = get_frontend_state_adapter()
            
            return HealthCheckResult(
                name="unified_architecture",
                status=HealthStatus.HEALTHY,
                message="Unified architecture components available and initialized",
                details={
                    "unified_manager_agents": len(unified_manager.agents),
                    "cached_states": len(unified_manager.state_cache),
                    "update_pipeline_available": update_pipeline is not None,
                    "frontend_adapter_available": frontend_adapter is not None
                },
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        except ImportError as e:
            return HealthCheckResult(
                name="unified_architecture",
                status=HealthStatus.DEGRADED,
                message="Unified architecture not available, using fallback mode",
                details={"error": str(e), "fallback_mode": True},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            return HealthCheckResult(
                name="unified_architecture",
                status=HealthStatus.UNHEALTHY,
                message=f"Unified architecture error: {str(e)}",
                details={"error": str(e)},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def check_websocket_support(self) -> HealthCheckResult:
        """Check WebSocket functionality"""
        try:
            # Import WebSocket components
            from ..api.websocket_router import connection_manager
            
            active_connections = len(connection_manager.active_connections)
            agent_rooms = len(connection_manager.agent_rooms)
            system_connections = len(connection_manager.system_connections)
            
            return HealthCheckResult(
                name="websocket_support",
                status=HealthStatus.HEALTHY,
                message="WebSocket support available and operational",
                details={
                    "active_connections": active_connections,
                    "agent_rooms": agent_rooms,
                    "system_connections": system_connections,
                    "websocket_enabled": True
                },
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            return HealthCheckResult(
                name="websocket_support",
                status=HealthStatus.UNHEALTHY,
                message=f"WebSocket support unavailable: {str(e)}",
                details={"error": str(e), "websocket_enabled": False},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def check_file_system(self) -> HealthCheckResult:
        """Check file system access and permissions"""
        import os
        from pathlib import Path
        
        try:
            # Check static files directory
            static_dir = Path("static")
            static_js_dir = Path("static/js")
            templates_dir = Path("templates")
            
            checks = {
                "static_dir_exists": static_dir.exists(),
                "static_js_dir_exists": static_js_dir.exists(),
                "templates_dir_exists": templates_dir.exists(),
                "static_readable": os.access(static_dir, os.R_OK) if static_dir.exists() else False,
                "templates_readable": os.access(templates_dir, os.R_OK) if templates_dir.exists() else False
            }
            
            if all(checks.values()):
                status = HealthStatus.HEALTHY
                message = "File system access OK"
            elif any(checks.values()):
                status = HealthStatus.DEGRADED
                message = "Some file system issues detected"
            else:
                status = HealthStatus.UNHEALTHY
                message = "File system access problems"
            
            return HealthCheckResult(
                name="file_system",
                status=status,
                message=message,
                details=checks,
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            return HealthCheckResult(
                name="file_system",
                status=HealthStatus.UNHEALTHY,
                message=f"File system check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            memory_percent = memory.percent
            if memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High memory usage: {memory_percent:.1f}%"
            elif memory_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Elevated memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                details={
                    "percent": memory_percent,
                    "available_mb": memory.available // 1024 // 1024,
                    "total_mb": memory.total // 1024 // 1024,
                    "used_mb": memory.used // 1024 // 1024
                },
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        except ImportError:
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.DEGRADED,
                message="psutil not available for memory monitoring",
                details={"psutil_available": False},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            
            free_percent = (free / total) * 100
            if free_percent < 5:
                status = HealthStatus.UNHEALTHY
                message = f"Low disk space: {free_percent:.1f}% free"
            elif free_percent < 15:
                status = HealthStatus.DEGRADED
                message = f"Limited disk space: {free_percent:.1f}% free"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {free_percent:.1f}% free"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                details={
                    "free_percent": round(free_percent, 2),
                    "free_gb": round(free / 1024 / 1024 / 1024, 2),
                    "total_gb": round(total / 1024 / 1024 / 1024, 2),
                    "used_gb": round(used / 1024 / 1024 / 1024, 2)
                },
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk space check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def check_update_pipeline(self) -> HealthCheckResult:
        """Check update pipeline performance and health"""
        try:
            from ...dating_show.services.update_pipeline import get_update_pipeline
            
            pipeline = get_update_pipeline()
            metrics = pipeline.get_performance_metrics()
            
            # Analyze performance metrics
            avg_processing_time = metrics.get('average_processing_time_ms', 0)
            success_rate = metrics.get('success_rate_percent', 100)
            queue_size = metrics.get('current_queue_size', 0)
            
            if (avg_processing_time > self.settings.alert_thresholds['processing_time_critical'] or
                success_rate < self.settings.alert_thresholds['success_rate_critical'] or
                queue_size > self.settings.alert_thresholds['queue_size_critical']):
                status = HealthStatus.UNHEALTHY
                message = "Update pipeline performance critical"
            elif (avg_processing_time > self.settings.alert_thresholds['processing_time_warning'] or
                  success_rate < self.settings.alert_thresholds['success_rate_warning'] or
                  queue_size > self.settings.alert_thresholds['queue_size_warning']):
                status = HealthStatus.DEGRADED
                message = "Update pipeline performance degraded"
            else:
                status = HealthStatus.HEALTHY
                message = "Update pipeline performing well"
            
            return HealthCheckResult(
                name="update_pipeline",
                status=status,
                message=message,
                details=metrics,
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        except ImportError:
            return HealthCheckResult(
                name="update_pipeline",
                status=HealthStatus.DEGRADED,
                message="Update pipeline not available, using fallback mode",
                details={"fallback_mode": True},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            return HealthCheckResult(
                name="update_pipeline",
                status=HealthStatus.UNHEALTHY,
                message=f"Update pipeline check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
    
    async def check_circuit_breaker(self) -> HealthCheckResult:
        """Check circuit breaker status"""
        try:
            from ...dating_show.services.update_pipeline import get_update_pipeline
            
            pipeline = get_update_pipeline()
            circuit_breaker = pipeline.circuit_breaker
            
            if circuit_breaker.state == "CLOSED":
                status = HealthStatus.HEALTHY
                message = "Circuit breaker closed - system operational"
            elif circuit_breaker.state == "HALF_OPEN":
                status = HealthStatus.DEGRADED
                message = "Circuit breaker half-open - testing recovery"
            else:  # OPEN
                status = HealthStatus.UNHEALTHY
                message = "Circuit breaker open - system protection active"
            
            return HealthCheckResult(
                name="circuit_breaker",
                status=status,
                message=message,
                details={
                    "state": circuit_breaker.state,
                    "failure_count": circuit_breaker.failure_count,
                    "failure_threshold": circuit_breaker.failure_threshold,
                    "timeout_seconds": circuit_breaker.timeout
                },
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        except ImportError:
            return HealthCheckResult(
                name="circuit_breaker",
                status=HealthStatus.DEGRADED,
                message="Circuit breaker not available",
                details={"available": False},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            return HealthCheckResult(
                name="circuit_breaker",
                status=HealthStatus.UNHEALTHY,
                message=f"Circuit breaker check failed: {str(e)}",
                details={"error": str(e)},
                duration_ms=0.0,
                timestamp=datetime.now(timezone.utc)
            )

# Global health check manager instance
_health_manager: Optional[HealthCheckManager] = None

def get_health_manager(settings=None) -> HealthCheckManager:
    """Get or create the global health check manager"""
    global _health_manager
    if _health_manager is None:
        from .config import Settings
        if settings is None:
            settings = Settings()
        _health_manager = HealthCheckManager(settings)
    return _health_manager