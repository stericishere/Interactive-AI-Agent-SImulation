"""
UpdatePipeline Service
Real-time state synchronization with WebSocket broadcasting and batch processing.
Provides <100ms performance targets with circuit breaker reliability patterns.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import asdict, dataclass
from collections import defaultdict, deque
import threading
from enum import Enum

from .unified_agent_manager import get_unified_agent_manager
from .frontend_state_adapter import get_frontend_state_adapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UpdateType(Enum):
    """Types of state updates for pipeline processing."""
    AGENT_STATE = "agent_state"
    SOCIAL_NETWORK = "social_network"
    MEMORY_UPDATE = "memory_update"
    LOCATION_CHANGE = "location_change"
    ACTIVITY_CHANGE = "activity_change"
    RELATIONSHIP_UPDATE = "relationship_update"
    BATCH_UPDATE = "batch_update"


@dataclass
class UpdateEvent:
    """Represents a single state update event."""
    event_id: str
    event_type: UpdateType
    agent_id: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=high, 2=medium, 3=low
    retries: int = 0
    max_retries: int = 3


@dataclass
class PerformanceMetrics:
    """Performance tracking for update pipeline."""
    total_updates: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    average_processing_time_ms: float = 0.0
    batch_processing_time_ms: float = 0.0
    websocket_broadcast_time_ms: float = 0.0
    circuit_breaker_trips: int = 0
    last_reset_time: datetime = None


class CircuitBreakerState(Enum):
    """Circuit breaker states for reliability."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for update pipeline reliability."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if (datetime.now() - self.last_failure_time).seconds > self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN - blocking request")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.reset()
                    logger.info("Circuit breaker reset to CLOSED state")
                
                return result
                
            except Exception as e:
                self.record_failure()
                raise e
    
    def record_failure(self):
        """Record a failure and potentially trip the circuit breaker."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED


class UpdatePipeline:
    """
    Real-time state synchronization pipeline with WebSocket broadcasting.
    
    Provides high-performance state updates with <100ms targets,
    batch processing, and circuit breaker reliability patterns.
    """
    
    def __init__(self):
        self.unified_manager = get_unified_agent_manager()
        self.frontend_adapter = get_frontend_state_adapter()
        
        # Update processing
        self.update_queue: asyncio.Queue = None
        self.batch_queue: deque = deque()
        self.processing_task: Optional[asyncio.Task] = None
        self.batch_task: Optional[asyncio.Task] = None
        
        # WebSocket management
        self.websocket_connections: Set[Any] = set()
        self.websocket_groups: Dict[str, Set[Any]] = defaultdict(set)
        
        # Performance and reliability
        self.metrics = PerformanceMetrics(last_reset_time=datetime.now(timezone.utc))
        self.circuit_breaker = CircuitBreaker()
        
        # Configuration
        self.batch_size = 10
        self.batch_timeout_ms = 50  # 50ms batch window
        self.max_processing_time_ms = 100  # <100ms target
        self.enable_websockets = True
        
        # Event subscribers
        self.event_subscribers: Dict[UpdateType, List[Callable]] = defaultdict(list)
        
        logger.info("UpdatePipeline initialized with real-time synchronization")
    
    async def initialize(self):
        """Initialize async components of the pipeline."""
        self.update_queue = asyncio.Queue()
        
        # Start processing tasks
        self.processing_task = asyncio.create_task(self._process_updates())
        self.batch_task = asyncio.create_task(self._process_batches())
        
        logger.info("UpdatePipeline async components initialized")
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline."""
        if self.processing_task:
            self.processing_task.cancel()
        if self.batch_task:
            self.batch_task.cancel()
        
        # Close WebSocket connections
        for ws in self.websocket_connections.copy():
            try:
                await ws.close()
            except Exception:
                pass
        
        logger.info("UpdatePipeline shutdown completed")
    
    # WebSocket Management
    
    def register_websocket(self, websocket, groups: List[str] = None):
        """Register WebSocket connection for real-time updates."""
        self.websocket_connections.add(websocket)
        
        if groups:
            for group in groups:
                self.websocket_groups[group].add(websocket)
        
        logger.debug(f"Registered WebSocket connection, total: {len(self.websocket_connections)}")
    
    def unregister_websocket(self, websocket):
        """Unregister WebSocket connection."""
        self.websocket_connections.discard(websocket)
        
        # Remove from all groups
        for group_connections in self.websocket_groups.values():
            group_connections.discard(websocket)
        
        logger.debug(f"Unregistered WebSocket connection, total: {len(self.websocket_connections)}")
    
    async def broadcast_to_websockets(self, message: Dict[str, Any], groups: List[str] = None):
        """Broadcast message to WebSocket connections."""
        if not self.enable_websockets:
            return
        
        start_time = time.time()
        
        target_connections = set()
        if groups:
            for group in groups:
                target_connections.update(self.websocket_groups.get(group, set()))
        else:
            target_connections = self.websocket_connections.copy()
        
        if not target_connections:
            return
        
        message_json = json.dumps(message, default=str)
        failed_connections = []
        
        for ws in target_connections:
            try:
                await ws.send(message_json)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                failed_connections.append(ws)
        
        # Clean up failed connections
        for ws in failed_connections:
            self.unregister_websocket(ws)
        
        broadcast_time = (time.time() - start_time) * 1000
        self.metrics.websocket_broadcast_time_ms = self._update_average(
            self.metrics.websocket_broadcast_time_ms, broadcast_time, self.metrics.total_updates
        )
        
        logger.debug(f"WebSocket broadcast completed in {broadcast_time:.2f}ms to {len(target_connections)} connections")
    
    # Update Processing
    
    async def queue_update(self, event: UpdateEvent):
        """Queue an update event for processing."""
        if self.update_queue:
            await self.update_queue.put(event)
            logger.debug(f"Queued update event: {event.event_type.value} for agent {event.agent_id}")
    
    async def update_agent_state(self, agent_id: str, updates: Dict[str, Any], 
                                event_type: UpdateType = UpdateType.AGENT_STATE,
                                priority: int = 1) -> bool:
        """Update agent state with real-time synchronization."""
        try:
            event = UpdateEvent(
                event_id=f"{agent_id}_{int(time.time() * 1000)}",
                event_type=event_type,
                agent_id=agent_id,
                data=updates,
                timestamp=datetime.now(timezone.utc),
                priority=priority
            )
            
            await self.queue_update(event)
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue agent state update for {agent_id}: {e}")
            return False
    
    async def batch_update_agents(self, updates: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """Process multiple agent updates as a batch."""
        start_time = time.time()
        results = {}
        
        try:
            # Create batch event
            batch_event = UpdateEvent(
                event_id=f"batch_{int(time.time() * 1000)}",
                event_type=UpdateType.BATCH_UPDATE,
                agent_id="batch",
                data={"updates": updates},
                timestamp=datetime.now(timezone.utc),
                priority=1
            )
            
            await self.queue_update(batch_event)
            
            # Return optimistic results
            for agent_id in updates.keys():
                results[agent_id] = True
                
        except Exception as e:
            logger.error(f"Failed to process batch update: {e}")
            for agent_id in updates.keys():
                results[agent_id] = False
        
        batch_time = (time.time() - start_time) * 1000
        self.metrics.batch_processing_time_ms = self._update_average(
            self.metrics.batch_processing_time_ms, batch_time, self.metrics.total_updates
        )
        
        return results
    
    async def _process_updates(self):
        """Main update processing loop."""
        while True:
            try:
                # Get update event with timeout
                event = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)
                
                start_time = time.time()
                success = await self._execute_update(event)
                processing_time = (time.time() - start_time) * 1000
                
                # Update metrics
                self.metrics.total_updates += 1
                if success:
                    self.metrics.successful_updates += 1
                else:
                    self.metrics.failed_updates += 1
                
                self.metrics.average_processing_time_ms = self._update_average(
                    self.metrics.average_processing_time_ms, processing_time, self.metrics.total_updates
                )
                
                # Check performance target
                if processing_time > self.max_processing_time_ms:
                    logger.warning(f"Update processing exceeded target: {processing_time:.2f}ms > {self.max_processing_time_ms}ms")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in update processing loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_update(self, event: UpdateEvent) -> bool:
        """Execute individual update event."""
        try:
            return await self.circuit_breaker.call(self._process_single_update, event)
        except Exception as e:
            logger.error(f"Circuit breaker blocked update or update failed: {e}")
            
            # Retry logic
            if event.retries < event.max_retries:
                event.retries += 1
                await asyncio.sleep(0.1 * event.retries)  # Exponential backoff
                await self.queue_update(event)
                return False
            
            return False
    
    async def _process_single_update(self, event: UpdateEvent) -> bool:
        """Process single update event with state management."""
        try:
            if event.event_type == UpdateType.BATCH_UPDATE:
                return await self._process_batch_event(event)
            
            # Apply update to unified manager
            success = self.unified_manager.update_agent_state(
                event.agent_id, event.data, batch_mode=False
            )
            
            if not success:
                logger.error(f"Failed to apply update to unified manager for agent {event.agent_id}")
                return False
            
            # Get updated state for frontend
            frontend_state = self.frontend_adapter.get_agent_for_frontend(event.agent_id)
            if not frontend_state:
                logger.warning(f"Could not get frontend state for agent {event.agent_id}")
                return False
            
            # Broadcast to WebSockets
            await self.broadcast_to_websockets({
                "type": "agent_update",
                "event_type": event.event_type.value,
                "agent_id": event.agent_id,
                "data": frontend_state,
                "timestamp": event.timestamp.isoformat()
            })
            
            # Notify event subscribers
            await self._notify_subscribers(event.event_type, event)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process update event {event.event_id}: {e}")
            return False
    
    async def _process_batch_event(self, event: UpdateEvent) -> bool:
        """Process batch update event."""
        updates = event.data.get("updates", {})
        success_count = 0
        
        for agent_id, agent_updates in updates.items():
            try:
                success = self.unified_manager.update_agent_state(
                    agent_id, agent_updates, batch_mode=True
                )
                if success:
                    success_count += 1
                else:
                    logger.warning(f"Failed to update agent {agent_id} in batch")
            except Exception as e:
                logger.error(f"Error updating agent {agent_id} in batch: {e}")
        
        # Broadcast batch completion
        await self.broadcast_to_websockets({
            "type": "batch_update",
            "total_agents": len(updates),
            "successful_updates": success_count,
            "timestamp": event.timestamp.isoformat()
        })
        
        return success_count == len(updates)
    
    async def _process_batches(self):
        """Process batched operations for efficiency."""
        while True:
            try:
                await asyncio.sleep(self.batch_timeout_ms / 1000)
                
                if len(self.batch_queue) > 0:
                    batch_events = []
                    while self.batch_queue and len(batch_events) < self.batch_size:
                        batch_events.append(self.batch_queue.popleft())
                    
                    if batch_events:
                        await self._execute_batch(batch_events)
                        
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
    
    async def _execute_batch(self, events: List[UpdateEvent]):
        """Execute batch of events efficiently."""
        start_time = time.time()
        
        try:
            # Group events by agent for efficiency
            agent_updates = defaultdict(dict)
            
            for event in events:
                if event.event_type != UpdateType.BATCH_UPDATE:
                    agent_updates[event.agent_id].update(event.data)
            
            # Apply batch updates
            for agent_id, updates in agent_updates.items():
                self.unified_manager.update_agent_state(agent_id, updates, batch_mode=True)
            
            # Get frontend states for all updated agents
            frontend_states = {}
            for agent_id in agent_updates.keys():
                state = self.frontend_adapter.get_agent_for_frontend(agent_id)
                if state:
                    frontend_states[agent_id] = state
            
            # Single WebSocket broadcast for entire batch
            await self.broadcast_to_websockets({
                "type": "batch_state_update",
                "agents": frontend_states,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to execute batch: {e}")
        
        batch_time = (time.time() - start_time) * 1000
        logger.debug(f"Executed batch of {len(events)} events in {batch_time:.2f}ms")
    
    # Event Subscription
    
    def subscribe_to_updates(self, event_type: UpdateType, callback: Callable):
        """Subscribe to specific update events."""
        self.event_subscribers[event_type].append(callback)
        logger.debug(f"Added subscriber for {event_type.value}")
    
    def unsubscribe_from_updates(self, event_type: UpdateType, callback: Callable):
        """Unsubscribe from update events."""
        if callback in self.event_subscribers[event_type]:
            self.event_subscribers[event_type].remove(callback)
            logger.debug(f"Removed subscriber for {event_type.value}")
    
    async def _notify_subscribers(self, event_type: UpdateType, event: UpdateEvent):
        """Notify event subscribers."""
        for callback in self.event_subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    # Performance Monitoring
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        uptime = (datetime.now(timezone.utc) - self.metrics.last_reset_time).total_seconds()
        
        return {
            "total_updates": self.metrics.total_updates,
            "successful_updates": self.metrics.successful_updates,
            "failed_updates": self.metrics.failed_updates,
            "success_rate": (self.metrics.successful_updates / max(self.metrics.total_updates, 1)) * 100,
            "average_processing_time_ms": self.metrics.average_processing_time_ms,
            "batch_processing_time_ms": self.metrics.batch_processing_time_ms,
            "websocket_broadcast_time_ms": self.metrics.websocket_broadcast_time_ms,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
            "active_websockets": len(self.websocket_connections),
            "websocket_groups": len(self.websocket_groups),
            "queue_size": self.update_queue.qsize() if self.update_queue else 0,
            "batch_queue_size": len(self.batch_queue),
            "uptime_seconds": uptime,
            "performance_target_met": self.metrics.average_processing_time_ms < self.max_processing_time_ms
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = PerformanceMetrics(last_reset_time=datetime.now(timezone.utc))
        logger.info("Performance metrics reset")
    
    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update rolling average."""
        if count <= 1:
            return new_value
        return ((current_avg * (count - 1)) + new_value) / count


# Global instance
_update_pipeline: Optional[UpdatePipeline] = None


def get_update_pipeline() -> UpdatePipeline:
    """Get or create global UpdatePipeline instance."""
    global _update_pipeline
    if _update_pipeline is None:
        _update_pipeline = UpdatePipeline()
    return _update_pipeline


async def initialize_update_pipeline():
    """Initialize the global update pipeline."""
    pipeline = get_update_pipeline()
    await pipeline.initialize()
    return pipeline


def reset_update_pipeline():
    """Reset global pipeline (mainly for testing)."""
    global _update_pipeline
    if _update_pipeline:
        asyncio.create_task(_update_pipeline.shutdown())
    _update_pipeline = None