"""
File: resource_coordinator.py
Description: ResourceCoordinator - Manages shared memory access and resource allocation
Enhanced PIANO architecture with concurrent processing capabilities
"""

from typing import Dict, List, Optional, Any, Set, Union, ContextManager
from datetime import datetime, timedelta
import threading
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import weakref
from contextlib import contextmanager

from ..memory_structures.episodic_memory import EpisodicMemory
from ..memory_structures.semantic_memory import SemanticMemory
from ..memory_structures.temporal_memory import TemporalMemory
from ..memory_structures.circular_buffer import CircularBuffer
from ..memory_structures.security_utils import SecurityValidator, SecurityError


class ResourceType(Enum):
    """Types of resources managed by the coordinator."""
    EPISODIC_MEMORY = "episodic_memory"
    SEMANTIC_MEMORY = "semantic_memory"
    TEMPORAL_MEMORY = "temporal_memory"
    CIRCULAR_BUFFER = "circular_buffer"
    AGENT_STATE = "agent_state"
    COMPUTATION = "computation"
    NETWORK = "network"
    FILE_IO = "file_io"


class AccessMode(Enum):
    """Access modes for resource locks."""
    READ = "read"
    WRITE = "write"
    EXCLUSIVE = "exclusive"


@dataclass
class ResourceLock:
    """Represents a lock on a resource."""
    resource_id: str
    resource_type: ResourceType
    access_mode: AccessMode
    holder_id: str
    acquired_at: datetime
    expires_at: Optional[datetime] = None
    lock_count: int = 1  # For reentrant locks


@dataclass
class ResourceRequest:
    """Represents a request for resource access."""
    request_id: str
    resource_id: str
    resource_type: ResourceType
    access_mode: AccessMode
    requester_id: str
    requested_at: datetime
    timeout: Optional[float] = None
    priority: int = 0


@dataclass
class ResourceMetrics:
    """Metrics for resource usage and performance."""
    total_requests: int = 0
    successful_acquisitions: int = 0
    failed_acquisitions: int = 0
    timeout_count: int = 0
    contention_count: int = 0
    average_wait_time: float = 0.0
    peak_concurrent_locks: int = 0
    memory_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.memory_usage is None:
            self.memory_usage = {}


class ResourceCoordinator:
    """
    Manages shared memory access and resource allocation for concurrent modules.
    Provides thread-safe access to memory structures with deadlock prevention.
    """
    
    def __init__(self, max_concurrent_locks: int = 50, default_timeout: float = 30.0,
                 enable_deadlock_detection: bool = True, cleanup_interval: int = 60):
        """
        Initialize ResourceCoordinator.
        
        Args:
            max_concurrent_locks: Maximum number of concurrent locks
            default_timeout: Default timeout for resource acquisition
            enable_deadlock_detection: Enable deadlock detection and prevention
            cleanup_interval: Cleanup interval for expired locks (seconds)
        """
        self.max_concurrent_locks = max_concurrent_locks
        self.default_timeout = default_timeout
        self.enable_deadlock_detection = enable_deadlock_detection
        self.cleanup_interval = cleanup_interval
        
        # Resource registry
        self.resources: Dict[str, Any] = {}
        self.resource_types: Dict[str, ResourceType] = {}
        
        # Lock management
        self.active_locks: Dict[str, ResourceLock] = {}
        self.pending_requests: deque[ResourceRequest] = deque()
        self.lock_holders: Dict[str, Set[str]] = defaultdict(set)  # holder_id -> lock_ids
        
        # Reader-writer lock implementation
        self.read_locks: Dict[str, Set[str]] = defaultdict(set)  # resource_id -> reader_ids
        self.write_locks: Dict[str, Optional[str]] = {}  # resource_id -> writer_id
        self.wait_queues: Dict[str, deque[ResourceRequest]] = defaultdict(deque)
        
        # Deadlock detection
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # holder -> waiting_for
        
        # Metrics
        self.metrics = ResourceMetrics()
        self.request_history: List[ResourceRequest] = []
        
        # Synchronization
        self._global_lock = threading.RLock()
        self._resource_locks: Dict[str, threading.RLock] = {}
        self._condition_vars: Dict[str, threading.Condition] = {}
        
        # Background threads
        self._cleanup_thread: Optional[threading.Thread] = None
        self._deadlock_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()
        
        # Start background threads
        self._start_background_threads()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _start_background_threads(self) -> None:
        """Start background maintenance threads."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_expired_locks,
            name="ResourceCleanup",
            daemon=True
        )
        self._cleanup_thread.start()
        
        if self.enable_deadlock_detection:
            self._deadlock_thread = threading.Thread(
                target=self._deadlock_detector,
                name="DeadlockDetector",
                daemon=True
            )
            self._deadlock_thread.start()
    
    def register_resource(self, resource_id: str, resource: Any, 
                         resource_type: ResourceType) -> None:
        """
        Register a resource for managed access.
        
        Args:
            resource_id: Unique identifier for the resource
            resource: Resource object to manage
            resource_type: Type of resource
        """
        with self._global_lock:
            if resource_id in self.resources:
                raise ValueError(f"Resource {resource_id} is already registered")
            
            # Security validation
            self._validate_resource(resource, resource_type)
            
            self.resources[resource_id] = resource
            self.resource_types[resource_id] = resource_type
            self._resource_locks[resource_id] = threading.RLock()
            self._condition_vars[resource_id] = threading.Condition(self._resource_locks[resource_id])
            
            self.logger.info(f"Registered resource: {resource_id} ({resource_type.value})")
    
    def unregister_resource(self, resource_id: str) -> None:
        """
        Unregister a resource and release all associated locks.
        
        Args:
            resource_id: ID of resource to unregister
        """
        with self._global_lock:
            if resource_id not in self.resources:
                raise KeyError(f"Resource {resource_id} is not registered")
            
            # Release all locks for this resource
            locks_to_remove = [
                lock_id for lock_id, lock in self.active_locks.items()
                if lock.resource_id == resource_id
            ]
            
            for lock_id in locks_to_remove:
                self._release_lock_internal(lock_id)
            
            # Clean up
            del self.resources[resource_id]
            del self.resource_types[resource_id]
            del self._resource_locks[resource_id]
            del self._condition_vars[resource_id]
            self.read_locks.pop(resource_id, None)
            self.write_locks.pop(resource_id, None)
            self.wait_queues.pop(resource_id, None)
            
            self.logger.info(f"Unregistered resource: {resource_id}")
    
    def acquire_resource(self, resource_id: str, access_mode: AccessMode,
                        holder_id: str, timeout: Optional[float] = None) -> Optional[str]:
        """
        Acquire access to a resource.
        
        Args:
            resource_id: ID of resource to acquire
            access_mode: Access mode (read/write/exclusive)
            holder_id: ID of the requester
            timeout: Timeout in seconds
            
        Returns:
            Lock ID if successful, None if failed
        """
        if resource_id not in self.resources:
            raise KeyError(f"Resource {resource_id} is not registered")
        
        if timeout is None:
            timeout = self.default_timeout
        
        request_id = f"{holder_id}_{resource_id}_{int(time.time() * 1000000)}"
        request = ResourceRequest(
            request_id=request_id,
            resource_id=resource_id,
            resource_type=self.resource_types[resource_id],
            access_mode=access_mode,
            requester_id=holder_id,
            requested_at=datetime.now(),
            timeout=timeout
        )
        
        # Update metrics
        with self._global_lock:
            self.metrics.total_requests += 1
            if len(self.request_history) >= 1000:
                self.request_history = self.request_history[-500:]
            self.request_history.append(request)
        
        # Try to acquire the lock
        lock_id = self._try_acquire_lock(request)
        
        if lock_id:
            self.logger.debug(f"Acquired lock {lock_id} for resource {resource_id}")
            return lock_id
        
        self.logger.warning(f"Failed to acquire lock for resource {resource_id}")
        return None
    
    def _try_acquire_lock(self, request: ResourceRequest) -> Optional[str]:
        """Try to acquire a lock for a resource request."""
        resource_id = request.resource_id
        access_mode = request.access_mode
        holder_id = request.requester_id
        
        with self._condition_vars[resource_id]:
            start_time = time.time()
            
            while True:
                # Check if we can acquire the lock
                if self._can_acquire_lock(resource_id, access_mode, holder_id):
                    lock_id = self._grant_lock(request)
                    return lock_id
                
                # Check timeout
                if request.timeout:
                    elapsed = time.time() - start_time
                    if elapsed >= request.timeout:
                        with self._global_lock:
                            self.metrics.timeout_count += 1
                        break
                    
                    remaining_timeout = request.timeout - elapsed
                else:
                    remaining_timeout = None
                
                # Add to wait queue if not already there
                if request not in self.wait_queues[resource_id]:
                    self.wait_queues[resource_id].append(request)
                    with self._global_lock:
                        self.metrics.contention_count += 1
                
                # Wait for resource to become available
                if not self._condition_vars[resource_id].wait(timeout=remaining_timeout):
                    # Timeout occurred
                    if request in self.wait_queues[resource_id]:
                        self.wait_queues[resource_id].remove(request)
                    with self._global_lock:
                        self.metrics.timeout_count += 1
                    break
        
        return None
    
    def _can_acquire_lock(self, resource_id: str, access_mode: AccessMode, 
                         holder_id: str) -> bool:
        """Check if a lock can be acquired for the given parameters."""
        # Check for existing locks
        existing_readers = self.read_locks.get(resource_id, set())
        existing_writer = self.write_locks.get(resource_id)
        
        if access_mode == AccessMode.READ:
            # Can read if no writer or we are the writer
            return existing_writer is None or existing_writer == holder_id
        
        elif access_mode == AccessMode.WRITE:
            # Can write if no other readers/writers or we hold all locks
            return (not existing_readers or existing_readers == {holder_id}) and \
                   (existing_writer is None or existing_writer == holder_id)
        
        elif access_mode == AccessMode.EXCLUSIVE:
            # Can acquire exclusive if no other locks
            return not existing_readers and existing_writer is None
        
        return False
    
    def _grant_lock(self, request: ResourceRequest) -> str:
        """Grant a lock for a resource request."""
        resource_id = request.resource_id
        access_mode = request.access_mode
        holder_id = request.requester_id
        
        # Generate lock ID
        lock_id = f"lock_{holder_id}_{resource_id}_{int(time.time() * 1000000)}"
        
        # Create lock record
        lock = ResourceLock(
            resource_id=resource_id,
            resource_type=request.resource_type,
            access_mode=access_mode,
            holder_id=holder_id,
            acquired_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=request.timeout) if request.timeout else None
        )
        
        # Update lock tracking
        with self._global_lock:
            self.active_locks[lock_id] = lock
            self.lock_holders[holder_id].add(lock_id)
            
            # Update specific lock trackers
            if access_mode == AccessMode.READ:
                self.read_locks[resource_id].add(holder_id)
            elif access_mode in [AccessMode.WRITE, AccessMode.EXCLUSIVE]:
                self.write_locks[resource_id] = holder_id
            
            # Update metrics
            self.metrics.successful_acquisitions += 1
            current_locks = len(self.active_locks)
            if current_locks > self.metrics.peak_concurrent_locks:
                self.metrics.peak_concurrent_locks = current_locks
            
            # Calculate wait time
            wait_time = (datetime.now() - request.requested_at).total_seconds()
            total_wait = self.metrics.average_wait_time * (self.metrics.successful_acquisitions - 1)
            self.metrics.average_wait_time = (total_wait + wait_time) / self.metrics.successful_acquisitions
        
        # Remove from wait queue if present
        if request in self.wait_queues[resource_id]:
            self.wait_queues[resource_id].remove(request)
        
        return lock_id
    
    def release_resource(self, lock_id: str) -> bool:
        """
        Release a resource lock.
        
        Args:
            lock_id: ID of lock to release
            
        Returns:
            True if lock was released, False if not found
        """
        return self._release_lock_internal(lock_id)
    
    def _release_lock_internal(self, lock_id: str) -> bool:
        """Internal method to release a lock."""
        if lock_id not in self.active_locks:
            return False
        
        lock = self.active_locks[lock_id]
        resource_id = lock.resource_id
        holder_id = lock.holder_id
        
        with self._condition_vars[resource_id]:
            # Remove from tracking
            with self._global_lock:
                del self.active_locks[lock_id]
                self.lock_holders[holder_id].discard(lock_id)
                
                # Update specific lock trackers
                if lock.access_mode == AccessMode.READ:
                    self.read_locks[resource_id].discard(holder_id)
                elif lock.access_mode in [AccessMode.WRITE, AccessMode.EXCLUSIVE]:
                    if self.write_locks.get(resource_id) == holder_id:
                        self.write_locks[resource_id] = None
            
            # Notify waiting requests
            self._condition_vars[resource_id].notify_all()
        
        self.logger.debug(f"Released lock {lock_id} for resource {resource_id}")
        return True
    
    @contextmanager
    def acquire_context(self, resource_id: str, access_mode: AccessMode,
                       holder_id: str, timeout: Optional[float] = None):
        """
        Context manager for resource acquisition.
        
        Args:
            resource_id: ID of resource to acquire
            access_mode: Access mode
            holder_id: ID of the requester
            timeout: Timeout in seconds
        
        Yields:
            Resource object if acquisition successful
            
        Raises:
            RuntimeError: If resource acquisition fails
        """
        lock_id = self.acquire_resource(resource_id, access_mode, holder_id, timeout)
        
        if not lock_id:
            raise RuntimeError(f"Failed to acquire resource {resource_id}")
        
        try:
            yield self.resources[resource_id]
        finally:
            self.release_resource(lock_id)
    
    def get_resource_status(self, resource_id: str) -> Dict[str, Any]:
        """
        Get status information for a resource.
        
        Args:
            resource_id: ID of resource
            
        Returns:
            Dictionary with resource status
        """
        if resource_id not in self.resources:
            raise KeyError(f"Resource {resource_id} is not registered")
        
        with self._global_lock:
            active_locks = [
                lock for lock in self.active_locks.values()
                if lock.resource_id == resource_id
            ]
            
            readers = list(self.read_locks.get(resource_id, set()))
            writer = self.write_locks.get(resource_id)
            waiting = len(self.wait_queues.get(resource_id, []))
            
            return {
                'resource_id': resource_id,
                'resource_type': self.resource_types[resource_id].value,
                'active_locks': len(active_locks),
                'readers': readers,
                'writer': writer,
                'waiting_requests': waiting,
                'locks': [
                    {
                        'lock_id': lock.resource_id,
                        'holder': lock.holder_id,
                        'mode': lock.access_mode.value,
                        'acquired_at': lock.acquired_at.isoformat(),
                        'expires_at': lock.expires_at.isoformat() if lock.expires_at else None
                    }
                    for lock in active_locks
                ]
            }
    
    def _validate_resource(self, resource: Any, resource_type: ResourceType) -> None:
        """Validate that a resource is of the expected type."""
        type_mapping = {
            ResourceType.EPISODIC_MEMORY: EpisodicMemory,
            ResourceType.SEMANTIC_MEMORY: SemanticMemory,
            ResourceType.TEMPORAL_MEMORY: TemporalMemory,
            ResourceType.CIRCULAR_BUFFER: CircularBuffer,
        }
        
        if resource_type in type_mapping:
            expected_type = type_mapping[resource_type]
            if not isinstance(resource, expected_type):
                raise TypeError(f"Resource must be of type {expected_type.__name__}")
    
    def _cleanup_expired_locks(self) -> None:
        """Background thread to clean up expired locks."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.now()
                expired_locks = []
                
                with self._global_lock:
                    for lock_id, lock in self.active_locks.items():
                        if lock.expires_at and current_time >= lock.expires_at:
                            expired_locks.append(lock_id)
                
                # Release expired locks
                for lock_id in expired_locks:
                    self._release_lock_internal(lock_id)
                    self.logger.info(f"Released expired lock: {lock_id}")
                
                if expired_locks:
                    self.logger.info(f"Cleaned up {len(expired_locks)} expired locks")
                
            except Exception as e:
                self.logger.error(f"Error in lock cleanup: {e}")
            
            time.sleep(self.cleanup_interval)
    
    def _deadlock_detector(self) -> None:
        """Background thread for deadlock detection."""
        while not self._shutdown_event.is_set():
            try:
                if self._detect_deadlock():
                    self.logger.warning("Potential deadlock detected, resolving...")
                    self._resolve_deadlock()
                
            except Exception as e:
                self.logger.error(f"Error in deadlock detection: {e}")
            
            time.sleep(10)  # Check every 10 seconds
    
    def _detect_deadlock(self) -> bool:
        """Simple deadlock detection using dependency graph."""
        with self._global_lock:
            # Build dependency graph
            self.dependency_graph.clear()
            
            for request in self.pending_requests:
                holder_id = request.requester_id
                resource_id = request.resource_id
                
                # Find who currently holds locks on this resource
                current_holders = set()
                for lock in self.active_locks.values():
                    if lock.resource_id == resource_id:
                        current_holders.add(lock.holder_id)
                
                # Add dependencies
                for current_holder in current_holders:
                    if current_holder != holder_id:
                        self.dependency_graph[holder_id].add(current_holder)
            
            # Check for cycles using DFS
            visited = set()
            rec_stack = set()
            
            def has_cycle(node):
                if node in rec_stack:
                    return True
                if node in visited:
                    return False
                
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in self.dependency_graph[node]:
                    if has_cycle(neighbor):
                        return True
                
                rec_stack.remove(node)
                return False
            
            for node in self.dependency_graph:
                if node not in visited:
                    if has_cycle(node):
                        return True
        
        return False
    
    def _resolve_deadlock(self) -> None:
        """Simple deadlock resolution by releasing some locks."""
        # Find the holder with the most locks and release one
        with self._global_lock:
            if not self.lock_holders:
                return
            
            # Find holder with most locks
            max_locks = 0
            victim_holder = None
            
            for holder_id, lock_ids in self.lock_holders.items():
                if len(lock_ids) > max_locks:
                    max_locks = len(lock_ids)
                    victim_holder = holder_id
            
            if victim_holder and self.lock_holders[victim_holder]:
                # Release one lock from the victim
                victim_lock = list(self.lock_holders[victim_holder])[0]
                self._release_lock_internal(victim_lock)
                self.logger.warning(f"Released lock {victim_lock} from {victim_holder} to resolve deadlock")
    
    def get_coordinator_metrics(self) -> ResourceMetrics:
        """Get resource coordinator metrics."""
        with self._global_lock:
            return self.metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._global_lock:
            return {
                'registered_resources': len(self.resources),
                'active_locks': len(self.active_locks),
                'pending_requests': len(self.pending_requests),
                'lock_holders': len(self.lock_holders),
                'total_wait_queues': len(self.wait_queues),
                'max_concurrent_locks': self.max_concurrent_locks,
                'deadlock_detection_enabled': self.enable_deadlock_detection,
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'successful_acquisitions': self.metrics.successful_acquisitions,
                    'failed_acquisitions': self.metrics.failed_acquisitions,
                    'timeout_count': self.metrics.timeout_count,
                    'contention_count': self.metrics.contention_count,
                    'average_wait_time': self.metrics.average_wait_time,
                    'peak_concurrent_locks': self.metrics.peak_concurrent_locks
                }
            }
    
    def shutdown(self) -> None:
        """Shutdown the resource coordinator."""
        self.logger.info("Shutting down resource coordinator...")
        
        # Signal shutdown to background threads
        self._shutdown_event.set()
        
        # Release all active locks
        with self._global_lock:
            lock_ids = list(self.active_locks.keys())
        
        for lock_id in lock_ids:
            self._release_lock_internal(lock_id)
        
        # Wait for background threads
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        if self._deadlock_thread and self._deadlock_thread.is_alive():
            self._deadlock_thread.join(timeout=5.0)
        
        # Clear all data structures
        with self._global_lock:
            self.resources.clear()
            self.resource_types.clear()
            self.active_locks.clear()
            self.pending_requests.clear()
            self.lock_holders.clear()
            self.read_locks.clear()
            self.write_locks.clear()
            self.wait_queues.clear()
            self.dependency_graph.clear()
        
        self.logger.info("Resource coordinator shutdown complete")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ResourceCoordinator(resources={len(self.resources)}, "
                f"active_locks={len(self.active_locks)}, "
                f"pending={len(self.pending_requests)})")