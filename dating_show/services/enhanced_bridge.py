"""
Enhanced Frontend Bridge Service
Advanced bridge service with auto-discovery, batch optimization, health monitoring, and error recovery
"""

import asyncio
import json
import time
import threading
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import requests
from urllib.parse import urljoin
from ..api.frontend_bridge import FrontendBridge, AgentUpdate, GovernanceUpdate, SocialUpdate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BridgeStatus(Enum):
    """Bridge service status enumeration"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    DISCONNECTED = "disconnected"


@dataclass
class HealthMetrics:
    """Bridge health metrics"""
    status: BridgeStatus
    last_successful_sync: datetime
    total_syncs: int
    failed_syncs: int
    average_sync_time_ms: float
    queue_depths: Dict[str, int]
    error_rate: float
    uptime_seconds: float


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    sync_latency_ms: float
    throughput_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    network_errors: int
    recovery_attempts: int


@dataclass
class AutoDiscoveryResult:
    """Result of agent auto-discovery"""
    discovered_agents: List[str]
    new_agents: List[str]
    removed_agents: List[str]
    total_agents: int


class EnhancedFrontendBridge(FrontendBridge):
    """
    Enhanced bridge service with advanced features.
    
    Extends the base FrontendBridge with:
    - Auto-discovery of PIANO agents
    - Batch optimization for performance
    - Health monitoring and metrics
    - Intelligent error recovery
    - Performance optimization
    """
    
    def __init__(self, frontend_url: str = "http://localhost:8001", 
                 update_interval: float = 1.0, **kwargs):
        """Initialize enhanced bridge with additional capabilities"""
        super().__init__(frontend_url, update_interval)
        
        # Enhanced configuration
        self.batch_size = kwargs.get('batch_size', 10)
        self.discovery_interval = kwargs.get('discovery_interval', 30.0)  # seconds
        self.health_check_interval = kwargs.get('health_check_interval', 5.0)  # seconds
        self.max_retry_attempts = kwargs.get('max_retry_attempts', 3)
        self.circuit_breaker_threshold = kwargs.get('circuit_breaker_threshold', 5)
        
        # Enhanced state tracking
        self.discovered_agents: Set[str] = set()
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.health_metrics = HealthMetrics(
            status=BridgeStatus.INITIALIZING,
            last_successful_sync=datetime.now(timezone.utc),
            total_syncs=0,
            failed_syncs=0,
            average_sync_time_ms=0.0,
            queue_depths={},
            error_rate=0.0,
            uptime_seconds=0.0
        )
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.sync_times: List[float] = []
        self.start_time = time.time()
        
        # Error recovery
        self.consecutive_failures = 0
        self.circuit_breaker_open = False
        self.last_circuit_breaker_reset = time.time()
        
        # Background task management
        self.discovery_task: Optional[asyncio.Task] = None
        self.health_monitor_task: Optional[asyncio.Task] = None
        
        logger.info("Enhanced frontend bridge initialized")
    
    def start_bridge(self):
        """Start the enhanced bridge service with all background tasks"""
        super().start_bridge()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.health_metrics.status = BridgeStatus.HEALTHY
        logger.info("Enhanced frontend bridge service started with background tasks")
    
    def stop_bridge(self):
        """Stop the enhanced bridge service and cleanup background tasks"""
        self._stop_background_tasks()
        super().stop_bridge()
        logger.info("Enhanced frontend bridge service stopped")
    
    async def auto_discover_agents(self) -> AutoDiscoveryResult:
        """
        Automatically discover active PIANO agents.
        
        Returns:
            AutoDiscoveryResult with discovery statistics
        """
        try:
            logger.debug("Starting agent auto-discovery...")
            
            # This would be implemented to scan the PIANO system for active agents
            # For now, we'll simulate discovery by checking the agent cache
            current_agents = set(self.agent_state_cache.keys())
            
            # Compare with previously discovered agents
            new_agents = list(current_agents - self.discovered_agents)
            removed_agents = list(self.discovered_agents - current_agents)
            
            # Update discovered agents set
            self.discovered_agents = current_agents.copy()
            
            # Register new agents
            for agent_id in new_agents:
                await self._register_agent(agent_id)
            
            # Deregister removed agents
            for agent_id in removed_agents:
                await self._deregister_agent(agent_id)
            
            result = AutoDiscoveryResult(
                discovered_agents=list(current_agents),
                new_agents=new_agents,
                removed_agents=removed_agents,
                total_agents=len(current_agents)
            )
            
            if new_agents or removed_agents:
                logger.info(f"Agent discovery: {len(new_agents)} new, {len(removed_agents)} removed, {len(current_agents)} total")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in auto_discover_agents: {e}")
            raise
    
    async def batch_sync_optimization(self) -> None:
        """
        Optimize sync performance through intelligent batching.
        """
        try:
            start_time = time.time()
            
            # Collect updates for batching
            agent_updates = []
            governance_updates = []
            social_updates = []
            
            # Collect agent updates
            while len(agent_updates) < self.batch_size:
                try:
                    update = self.agent_updates.get_nowait()
                    agent_updates.append(update)
                except asyncio.QueueEmpty:
                    break
            
            # Collect governance updates
            while len(governance_updates) < self.batch_size:
                try:
                    update = self.governance_updates.get_nowait()
                    governance_updates.append(update)
                except asyncio.QueueEmpty:
                    break
            
            # Collect social updates
            while len(social_updates) < self.batch_size:
                try:
                    update = self.social_updates.get_nowait()
                    social_updates.append(update)
                except asyncio.QueueEmpty:
                    break
            
            # Process batches if we have updates
            if agent_updates or governance_updates or social_updates:
                await self._send_batch_updates_optimized(
                    agent_updates, governance_updates, social_updates
                )
            
            # Track performance
            sync_time = (time.time() - start_time) * 1000  # Convert to ms
            self.sync_times.append(sync_time)
            
            # Keep only recent sync times for average calculation
            if len(self.sync_times) > 100:
                self.sync_times = self.sync_times[-100:]
            
            self.health_metrics.average_sync_time_ms = sum(self.sync_times) / len(self.sync_times)
            
        except Exception as e:
            logger.error(f"Error in batch_sync_optimization: {e}")
            await self._handle_sync_error(e)
    
    def get_health_metrics(self) -> HealthMetrics:
        """
        Comprehensive bridge health monitoring.
        
        Returns:
            HealthMetrics with current bridge health status
        """
        try:
            # Update queue depths
            self.health_metrics.queue_depths = {
                'agent_updates': self.agent_updates.qsize(),
                'governance_updates': self.governance_updates.qsize(),
                'social_updates': self.social_updates.qsize()
            }
            
            # Calculate error rate
            if self.health_metrics.total_syncs > 0:
                self.health_metrics.error_rate = self.health_metrics.failed_syncs / self.health_metrics.total_syncs
            else:
                self.health_metrics.error_rate = 0.0
            
            # Update uptime
            self.health_metrics.uptime_seconds = time.time() - self.start_time
            
            # Determine status based on metrics
            if self.circuit_breaker_open:
                self.health_metrics.status = BridgeStatus.ERROR
            elif self.health_metrics.error_rate > 0.1:  # >10% error rate
                self.health_metrics.status = BridgeStatus.WARNING
            elif max(self.health_metrics.queue_depths.values()) > 100:  # Queue backup
                self.health_metrics.status = BridgeStatus.WARNING
            else:
                self.health_metrics.status = BridgeStatus.HEALTHY
            
            return self.health_metrics
            
        except Exception as e:
            logger.error(f"Error getting health metrics: {e}")
            self.health_metrics.status = BridgeStatus.ERROR
            return self.health_metrics
    
    async def recover_from_error(self, error: Exception) -> bool:
        """
        Intelligent error recovery and retry logic.
        
        Args:
            error: The error to recover from
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            self.consecutive_failures += 1
            logger.warning(f"Attempting recovery from error (attempt {self.consecutive_failures}): {error}")
            
            # Check circuit breaker
            if self.consecutive_failures >= self.circuit_breaker_threshold:
                self.circuit_breaker_open = True
                self.last_circuit_breaker_reset = time.time()
                logger.error("Circuit breaker opened due to consecutive failures")
                return False
            
            # Implement exponential backoff
            backoff_delay = min(2 ** self.consecutive_failures, 30)  # Max 30 seconds
            await asyncio.sleep(backoff_delay)
            
            # Attempt recovery actions
            recovery_successful = await self._attempt_recovery()
            
            if recovery_successful:
                self.consecutive_failures = 0
                self.circuit_breaker_open = False
                logger.info("Error recovery successful")
                return True
            else:
                logger.warning("Error recovery failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in recovery process: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'health_metrics': asdict(self.get_health_metrics()),
            'discovered_agents': len(self.discovered_agents),
            'registered_agents': len(self.agent_registry),
            'batch_size': self.batch_size,
            'discovery_interval': self.discovery_interval,
            'circuit_breaker_open': self.circuit_breaker_open,
            'consecutive_failures': self.consecutive_failures,
            'average_sync_time_ms': self.health_metrics.average_sync_time_ms,
            'bridge_status': super().get_bridge_status()
        }
    
    # Private helper methods
    
    def _start_background_tasks(self):
        """Start background tasks for discovery and health monitoring"""
        try:
            # Create event loop for background tasks if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Note: In a real implementation, these would be proper async tasks
            # For this bridge service, we'll use threading for background tasks
            self._discovery_thread = threading.Thread(target=self._discovery_loop, daemon=True)
            self._health_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
            
            self._discovery_thread.start()
            self._health_thread.start()
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    def _stop_background_tasks(self):
        """Stop background tasks"""
        # Background threads will stop when the main thread stops (daemon=True)
        logger.info("Background tasks stopped")
    
    def _discovery_loop(self):
        """Background discovery loop"""
        while self.running:
            try:
                # Run discovery in a synchronous context
                asyncio.run(self.auto_discover_agents())
                time.sleep(self.discovery_interval)
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                time.sleep(self.discovery_interval)
    
    def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self.running:
            try:
                self.get_health_metrics()
                
                # Check circuit breaker reset
                if self.circuit_breaker_open:
                    if time.time() - self.last_circuit_breaker_reset > 60:  # Reset after 1 minute
                        self.circuit_breaker_open = False
                        self.consecutive_failures = 0
                        logger.info("Circuit breaker reset")
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                time.sleep(self.health_check_interval)
    
    async def _register_agent(self, agent_id: str):
        """Register a newly discovered agent"""
        try:
            # Get agent information from cache if available
            agent_info = self.agent_state_cache.get(agent_id)
            if agent_info:
                self.agent_registry[agent_id] = {
                    'registered_at': datetime.now(timezone.utc),
                    'last_seen': datetime.now(timezone.utc),
                    'status': 'active',
                    'agent_info': asdict(agent_info)
                }
                logger.debug(f"Registered agent: {agent_id}")
        except Exception as e:
            logger.error(f"Error registering agent {agent_id}: {e}")
    
    async def _deregister_agent(self, agent_id: str):
        """Deregister a removed agent"""
        try:
            if agent_id in self.agent_registry:
                self.agent_registry[agent_id]['status'] = 'removed'
                self.agent_registry[agent_id]['deregistered_at'] = datetime.now(timezone.utc)
                logger.debug(f"Deregistered agent: {agent_id}")
        except Exception as e:
            logger.error(f"Error deregistering agent {agent_id}: {e}")
    
    async def _send_batch_updates_optimized(self, agent_updates: List[AgentUpdate], 
                                          governance_updates: List[GovernanceUpdate],
                                          social_updates: List[SocialUpdate]):
        """Send optimized batch updates to frontend"""
        try:
            if self.circuit_breaker_open:
                logger.warning("Circuit breaker open, skipping batch update")
                return
            
            # Prepare batch payload
            batch_payload = {
                'batch_id': f"batch_{int(time.time())}",
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'agent_updates': [asdict(update) for update in agent_updates],
                'governance_updates': [asdict(update) for update in governance_updates],
                'social_updates': [asdict(update) for update in social_updates],
                'metrics': {
                    'total_updates': len(agent_updates) + len(governance_updates) + len(social_updates),
                    'batch_size': self.batch_size
                }
            }
            
            # Send batch update
            url = urljoin(self.frontend_url, "/dating_show/api/batch/update/")
            response = self.session.post(url, json=batch_payload, timeout=10.0)
            
            if response.status_code == 200:
                self.health_metrics.total_syncs += 1
                self.health_metrics.last_successful_sync = datetime.now(timezone.utc)
                self.consecutive_failures = 0
                logger.debug(f"Batch update successful: {len(agent_updates)} agents, {len(governance_updates)} governance, {len(social_updates)} social")
            else:
                raise Exception(f"Batch update failed with status {response.status_code}")
                
        except Exception as e:
            self.health_metrics.failed_syncs += 1
            await self._handle_sync_error(e)
    
    async def _handle_sync_error(self, error: Exception):
        """Handle synchronization errors"""
        logger.error(f"Sync error: {error}")
        
        # Attempt recovery if not too many failures
        if self.consecutive_failures < self.max_retry_attempts:
            recovery_success = await self.recover_from_error(error)
            if not recovery_success:
                logger.error("Recovery failed, continuing with errors")
    
    async def _attempt_recovery(self) -> bool:
        """Attempt to recover from errors"""
        try:
            # Test basic connectivity
            test_url = urljoin(self.frontend_url, "/dating_show/api/health/")
            response = self.session.get(test_url, timeout=5.0)
            
            if response.status_code == 200:
                logger.info("Connectivity test successful")
                return True
            else:
                logger.warning(f"Connectivity test failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False


# Factory function for creating enhanced bridge
def create_enhanced_bridge(frontend_url: str = "http://localhost:8001", **kwargs) -> EnhancedFrontendBridge:
    """
    Create and configure enhanced frontend bridge.
    
    Args:
        frontend_url: Frontend server URL
        **kwargs: Additional configuration options
        
    Returns:
        Configured EnhancedFrontendBridge instance
    """
    return EnhancedFrontendBridge(frontend_url, **kwargs)