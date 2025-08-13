"""
File: state_coordinator.py
Description: StateCoordinator - Cross-module state synchronization and consistency management
Enhanced PIANO architecture with concurrent processing capabilities
"""

from typing import Dict, List, Optional, Any, Set, Union, Callable
from datetime import datetime, timedelta
import threading
import time
import logging
import json
import copy
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import weakref
from concurrent.futures import Future

from ..memory_structures.security_utils import SecurityValidator, SecurityError


class StateChangeType(Enum):
    """Types of state changes."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    RESET = "reset"


class SyncPolicy(Enum):
    """Synchronization policies for state coordination."""
    IMMEDIATE = "immediate"
    BATCHED = "batched"
    PERIODIC = "periodic"
    ON_DEMAND = "on_demand"
    CONFLICT_RESOLUTION = "conflict_resolution"


@dataclass
class StateChange:
    """Represents a change to the agent state."""
    change_id: str
    module_id: str
    change_type: StateChangeType
    path: str  # Dot-separated path to the state field
    old_value: Any = None
    new_value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    applied: bool = False


@dataclass
class StateSnapshot:
    """Snapshot of agent state at a specific time."""
    snapshot_id: str
    timestamp: datetime
    state_data: Dict[str, Any]
    module_versions: Dict[str, int]
    change_sequence: int


@dataclass
class ConflictResolution:
    """Configuration for resolving state conflicts."""
    field_path: str
    resolution_strategy: str  # 'merge', 'latest_wins', 'priority', 'custom'
    priority_order: Optional[List[str]] = None  # Module priority order
    custom_resolver: Optional[Callable] = None


@dataclass
class SyncMetrics:
    """Metrics for state synchronization."""
    total_changes: int = 0
    conflicts_resolved: int = 0
    sync_operations: int = 0
    rollbacks_performed: int = 0
    average_sync_time: float = 0.0
    max_conflict_resolution_time: float = 0.0
    state_size_bytes: int = 0
    snapshot_count: int = 0


class StateCoordinator:
    """
    Manages cross-module state synchronization and consistency for concurrent execution.
    Provides conflict resolution, state versioning, and transactional state updates.
    """
    
    def __init__(self, agent_state: Any, sync_policy: SyncPolicy = SyncPolicy.IMMEDIATE,
                 max_snapshots: int = 50, batch_size: int = 10, sync_interval: float = 1.0):
        """
        Initialize StateCoordinator.
        
        Args:
            agent_state: The shared agent state object
            sync_policy: Synchronization policy to use
            max_snapshots: Maximum number of state snapshots to keep
            batch_size: Batch size for batched sync policy
            sync_interval: Sync interval for periodic policy (seconds)
        """
        self.agent_state = agent_state
        self.sync_policy = sync_policy
        self.max_snapshots = max_snapshots
        self.batch_size = batch_size
        self.sync_interval = sync_interval
        
        # State management
        self.state_version: int = 0
        self.module_versions: Dict[str, int] = defaultdict(int)
        self.pending_changes: deque[StateChange] = deque()
        self.change_history: List[StateChange] = []
        self.state_snapshots: Dict[str, StateSnapshot] = {}
        
        # Conflict resolution
        self.conflict_resolvers: Dict[str, ConflictResolution] = {}
        self.conflict_queue: deque[List[StateChange]] = deque()
        
        # Module registration
        self.registered_modules: Dict[str, weakref.ref] = {}
        self.module_subscriptions: Dict[str, Set[str]] = defaultdict(set)  # module -> paths
        self.path_subscribers: Dict[str, Set[str]] = defaultdict(set)  # path -> modules
        
        # Transaction support
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        self.transaction_changes: Dict[str, List[StateChange]] = defaultdict(list)
        
        # Synchronization
        self._state_lock = threading.RLock()
        self._change_lock = threading.RLock()
        self._condition = threading.Condition(self._state_lock)
        
        # Background processing
        self._sync_thread: Optional[threading.Thread] = None
        self._conflict_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Metrics
        self.metrics = SyncMetrics()
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()
        
        # Start background threads
        self._start_background_threads()
        
        # Create initial snapshot
        self._create_snapshot("initial")
    
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
        """Start background processing threads."""
        if self.sync_policy in [SyncPolicy.BATCHED, SyncPolicy.PERIODIC]:
            self._sync_thread = threading.Thread(
                target=self._sync_loop,
                name="StateSync",
                daemon=True
            )
            self._sync_thread.start()
        
        self._conflict_thread = threading.Thread(
            target=self._conflict_resolution_loop,
            name="ConflictResolver",
            daemon=True
        )
        self._conflict_thread.start()
    
    def register_module(self, module_id: str, module_ref: Any) -> None:
        """
        Register a module for state coordination.
        
        Args:
            module_id: Unique identifier for the module
            module_ref: Weak reference to the module
        """
        with self._state_lock:
            if hasattr(module_ref, '__weakref__'):
                self.registered_modules[module_id] = weakref.ref(module_ref)
            else:
                # For objects that don't support weak references, store directly
                self.registered_modules[module_id] = lambda: module_ref
            
            self.module_versions[module_id] = 0
            
        self.logger.info(f"Registered module for state coordination: {module_id}")
    
    def unregister_module(self, module_id: str) -> None:
        """
        Unregister a module from state coordination.
        
        Args:
            module_id: ID of module to unregister
        """
        with self._state_lock:
            self.registered_modules.pop(module_id, None)
            self.module_versions.pop(module_id, None)
            self.module_subscriptions.pop(module_id, None)
            
            # Remove from path subscribers
            for path, subscribers in self.path_subscribers.items():
                subscribers.discard(module_id)
        
        self.logger.info(f"Unregistered module: {module_id}")
    
    def subscribe_to_path(self, module_id: str, path: str) -> None:
        """
        Subscribe a module to changes in a specific state path.
        
        Args:
            module_id: ID of the module
            path: Dot-separated path to monitor
        """
        with self._state_lock:
            if module_id not in self.registered_modules:
                raise ValueError(f"Module {module_id} is not registered")
            
            self.module_subscriptions[module_id].add(path)
            self.path_subscribers[path].add(module_id)
        
        self.logger.debug(f"Module {module_id} subscribed to path: {path}")
    
    def unsubscribe_from_path(self, module_id: str, path: str) -> None:
        """
        Unsubscribe a module from a state path.
        
        Args:
            module_id: ID of the module
            path: Path to unsubscribe from
        """
        with self._state_lock:
            self.module_subscriptions[module_id].discard(path)
            self.path_subscribers[path].discard(module_id)
        
        self.logger.debug(f"Module {module_id} unsubscribed from path: {path}")
    
    def propose_change(self, module_id: str, path: str, new_value: Any,
                      change_type: StateChangeType = StateChangeType.UPDATE,
                      priority: int = 0, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Propose a state change from a module.
        
        Args:
            module_id: ID of the module proposing the change
            path: State path to change
            new_value: New value for the path
            change_type: Type of change
            priority: Change priority (higher = more important)
            metadata: Additional metadata
            
        Returns:
            Change ID for tracking
        """
        if module_id not in self.registered_modules:
            raise ValueError(f"Module {module_id} is not registered")
        
        # Security validation
        SecurityValidator.validate_state_path(path)
        SecurityValidator.validate_state_value(new_value)
        
        # Get current value
        old_value = self._get_state_value(path)
        
        # Generate change ID
        change_id = f"{module_id}_{path}_{int(time.time() * 1000000)}"
        
        # Create state change
        change = StateChange(
            change_id=change_id,
            module_id=module_id,
            change_type=change_type,
            path=path,
            old_value=old_value,
            new_value=new_value,
            priority=priority,
            metadata=metadata or {}
        )
        
        with self._change_lock:
            self.pending_changes.append(change)
            self.change_history.append(change)
            
            # Keep history bounded
            if len(self.change_history) > 10000:
                self.change_history = self.change_history[-5000:]
            
            self.metrics.total_changes += 1
        
        # Apply change based on sync policy
        if self.sync_policy == SyncPolicy.IMMEDIATE:
            self._apply_changes([change])
        elif self.sync_policy == SyncPolicy.ON_DEMAND:
            # Changes are applied when explicitly requested
            pass
        
        # Notify sync thread
        with self._condition:
            self._condition.notify()
        
        self.logger.debug(f"Proposed change {change_id}: {path} = {new_value}")
        return change_id
    
    def apply_changes(self, change_ids: Optional[List[str]] = None) -> bool:
        """
        Manually apply pending changes (for on-demand policy).
        
        Args:
            change_ids: Specific change IDs to apply, or None for all pending
            
        Returns:
            True if all changes applied successfully
        """
        with self._change_lock:
            if change_ids:
                # Apply specific changes
                changes_to_apply = [
                    change for change in self.pending_changes
                    if change.change_id in change_ids
                ]
            else:
                # Apply all pending changes
                changes_to_apply = list(self.pending_changes)
        
        return self._apply_changes(changes_to_apply)
    
    def _apply_changes(self, changes: List[StateChange]) -> bool:
        """
        Apply a list of state changes with conflict resolution.
        
        Args:
            changes: List of changes to apply
            
        Returns:
            True if all changes applied successfully
        """
        if not changes:
            return True
        
        start_time = time.time()
        
        with self._state_lock:
            try:
                # Group changes by path for conflict detection
                path_changes = defaultdict(list)
                for change in changes:
                    path_changes[change.path].append(change)
                
                # Check for conflicts
                conflicts = []
                for path, path_change_list in path_changes.items():
                    if len(path_change_list) > 1:
                        conflicts.append(path_change_list)
                
                # Resolve conflicts
                if conflicts:
                    resolved_changes = self._resolve_conflicts(conflicts)
                    if not resolved_changes:
                        return False
                    changes = resolved_changes
                
                # Apply changes to state
                for change in changes:
                    self._apply_single_change(change)
                    change.applied = True
                    
                    # Remove from pending changes
                    if change in self.pending_changes:
                        self.pending_changes.remove(change)
                
                # Update version numbers
                self.state_version += 1
                for change in changes:
                    self.module_versions[change.module_id] += 1
                
                # Notify subscribers
                self._notify_subscribers(changes)
                
                # Update metrics
                sync_time = time.time() - start_time
                self.metrics.sync_operations += 1
                total_sync_time = self.metrics.average_sync_time * (self.metrics.sync_operations - 1)
                self.metrics.average_sync_time = (total_sync_time + sync_time) / self.metrics.sync_operations
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error applying changes: {e}")
                return False
    
    def _apply_single_change(self, change: StateChange) -> None:
        """Apply a single state change."""
        path_parts = change.path.split('.')
        
        # Navigate to the parent object
        current = self.agent_state
        for part in path_parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
            else:
                raise ValueError(f"Invalid path: {change.path}")
        
        # Apply the change
        final_part = path_parts[-1]
        
        if change.change_type == StateChangeType.CREATE:
            if hasattr(current, final_part):
                setattr(current, final_part, change.new_value)
            elif isinstance(current, dict):
                current[final_part] = change.new_value
            else:
                raise ValueError(f"Cannot create field {final_part} on {type(current)}")
        
        elif change.change_type == StateChangeType.UPDATE:
            if hasattr(current, final_part):
                setattr(current, final_part, change.new_value)
            elif isinstance(current, dict):
                current[final_part] = change.new_value
            else:
                raise ValueError(f"Cannot update field {final_part} on {type(current)}")
        
        elif change.change_type == StateChangeType.DELETE:
            if hasattr(current, final_part):
                delattr(current, final_part)
            elif isinstance(current, dict):
                current.pop(final_part, None)
        
        elif change.change_type == StateChangeType.MERGE:
            if hasattr(current, final_part):
                existing = getattr(current, final_part)
            elif isinstance(current, dict):
                existing = current.get(final_part)
            else:
                existing = None
            
            if isinstance(existing, dict) and isinstance(change.new_value, dict):
                merged = {**existing, **change.new_value}
                if hasattr(current, final_part):
                    setattr(current, final_part, merged)
                else:
                    current[final_part] = merged
            else:
                # Fallback to update
                if hasattr(current, final_part):
                    setattr(current, final_part, change.new_value)
                elif isinstance(current, dict):
                    current[final_part] = change.new_value
    
    def _get_state_value(self, path: str) -> Any:
        """Get current value at a state path."""
        path_parts = path.split('.')
        current = self.agent_state
        
        try:
            for part in path_parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict):
                    current = current[part]
                else:
                    return None
            return current
        except (KeyError, AttributeError):
            return None
    
    def _resolve_conflicts(self, conflicts: List[List[StateChange]]) -> List[StateChange]:
        """
        Resolve conflicts between concurrent state changes.
        
        Args:
            conflicts: List of conflicting change groups
            
        Returns:
            List of resolved changes
        """
        resolved_changes = []
        
        for conflict_group in conflicts:
            path = conflict_group[0].path
            
            # Check for registered conflict resolver
            resolver = self.conflict_resolvers.get(path)
            
            if resolver:
                resolved = self._apply_conflict_resolution(conflict_group, resolver)
            else:
                # Default resolution: priority then timestamp
                resolved = self._default_conflict_resolution(conflict_group)
            
            resolved_changes.extend(resolved)
            self.metrics.conflicts_resolved += 1
        
        return resolved_changes
    
    def _apply_conflict_resolution(self, conflicts: List[StateChange],
                                 resolver: ConflictResolution) -> List[StateChange]:
        """Apply a specific conflict resolution strategy."""
        if resolver.resolution_strategy == 'latest_wins':
            # Keep the most recent change
            latest = max(conflicts, key=lambda x: x.timestamp)
            return [latest]
        
        elif resolver.resolution_strategy == 'priority':
            # Use priority order if available
            if resolver.priority_order:
                for module_id in resolver.priority_order:
                    for change in conflicts:
                        if change.module_id == module_id:
                            return [change]
            # Fallback to highest priority value
            highest = max(conflicts, key=lambda x: x.priority)
            return [highest]
        
        elif resolver.resolution_strategy == 'merge':
            # Merge all values if possible
            try:
                base_value = conflicts[0].old_value or {}
                merged_value = copy.deepcopy(base_value)
                
                for change in conflicts:
                    if isinstance(change.new_value, dict) and isinstance(merged_value, dict):
                        merged_value.update(change.new_value)
                    else:
                        # Can't merge non-dict values, use latest
                        merged_value = change.new_value
                
                # Create merged change
                merged_change = StateChange(
                    change_id=f"merged_{int(time.time() * 1000000)}",
                    module_id="system_merge",
                    change_type=StateChangeType.MERGE,
                    path=conflicts[0].path,
                    old_value=conflicts[0].old_value,
                    new_value=merged_value,
                    metadata={'merged_from': [c.change_id for c in conflicts]}
                )
                return [merged_change]
                
            except Exception as e:
                self.logger.error(f"Merge failed: {e}")
                return self._default_conflict_resolution(conflicts)
        
        elif resolver.resolution_strategy == 'custom' and resolver.custom_resolver:
            # Use custom resolver function
            try:
                resolved = resolver.custom_resolver(conflicts)
                return resolved if isinstance(resolved, list) else [resolved]
            except Exception as e:
                self.logger.error(f"Custom resolver failed: {e}")
                return self._default_conflict_resolution(conflicts)
        
        return self._default_conflict_resolution(conflicts)
    
    def _default_conflict_resolution(self, conflicts: List[StateChange]) -> List[StateChange]:
        """Default conflict resolution: highest priority, then most recent."""
        # Sort by priority (desc) then timestamp (desc)
        sorted_conflicts = sorted(
            conflicts,
            key=lambda x: (x.priority, x.timestamp),
            reverse=True
        )
        return [sorted_conflicts[0]]
    
    def _notify_subscribers(self, changes: List[StateChange]) -> None:
        """Notify subscribed modules about state changes."""
        notifications = defaultdict(list)
        
        # Group changes by subscribers
        for change in changes:
            for subscriber in self.path_subscribers.get(change.path, []):
                notifications[subscriber].append(change)
        
        # Send notifications
        for module_id, module_changes in notifications.items():
            module_ref = self.registered_modules.get(module_id)
            if module_ref:
                module = module_ref()
                if module and hasattr(module, 'on_state_change'):
                    try:
                        module.on_state_change(module_changes)
                    except Exception as e:
                        self.logger.error(f"Error notifying module {module_id}: {e}")
    
    def register_conflict_resolver(self, path: str, resolution: ConflictResolution) -> None:
        """
        Register a conflict resolution strategy for a state path.
        
        Args:
            path: State path to handle conflicts for
            resolution: Conflict resolution configuration
        """
        self.conflict_resolvers[path] = resolution
        self.logger.info(f"Registered conflict resolver for path: {path}")
    
    def create_snapshot(self, snapshot_id: Optional[str] = None) -> str:
        """
        Create a snapshot of the current state.
        
        Args:
            snapshot_id: Optional custom snapshot ID
            
        Returns:
            Snapshot ID
        """
        if not snapshot_id:
            snapshot_id = f"snapshot_{int(time.time() * 1000000)}"
        
        return self._create_snapshot(snapshot_id)
    
    def _create_snapshot(self, snapshot_id: str) -> str:
        """Internal method to create a state snapshot."""
        with self._state_lock:
            try:
                # Deep copy the state
                state_copy = copy.deepcopy(self.agent_state.__dict__)
                
                snapshot = StateSnapshot(
                    snapshot_id=snapshot_id,
                    timestamp=datetime.now(),
                    state_data=state_copy,
                    module_versions=self.module_versions.copy(),
                    change_sequence=self.state_version
                )
                
                self.state_snapshots[snapshot_id] = snapshot
                
                # Cleanup old snapshots
                if len(self.state_snapshots) > self.max_snapshots:
                    oldest_id = min(
                        self.state_snapshots.keys(),
                        key=lambda x: self.state_snapshots[x].timestamp
                    )
                    del self.state_snapshots[oldest_id]
                
                self.metrics.snapshot_count = len(self.state_snapshots)
                
                self.logger.debug(f"Created state snapshot: {snapshot_id}")
                return snapshot_id
                
            except Exception as e:
                self.logger.error(f"Failed to create snapshot: {e}")
                raise
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        Restore state from a snapshot.
        
        Args:
            snapshot_id: ID of snapshot to restore
            
        Returns:
            True if restore successful
        """
        if snapshot_id not in self.state_snapshots:
            raise KeyError(f"Snapshot {snapshot_id} not found")
        
        snapshot = self.state_snapshots[snapshot_id]
        
        with self._state_lock:
            try:
                # Restore state data
                for key, value in snapshot.state_data.items():
                    setattr(self.agent_state, key, copy.deepcopy(value))
                
                # Restore version numbers
                self.state_version = snapshot.change_sequence
                self.module_versions = snapshot.module_versions.copy()
                
                # Clear pending changes (they're now invalid)
                self.pending_changes.clear()
                
                self.metrics.rollbacks_performed += 1
                
                self.logger.info(f"Restored state from snapshot: {snapshot_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to restore snapshot {snapshot_id}: {e}")
                return False
    
    def begin_transaction(self, transaction_id: str) -> None:
        """
        Begin a state transaction for atomic updates.
        
        Args:
            transaction_id: Unique transaction identifier
        """
        with self._state_lock:
            if transaction_id in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} already active")
            
            # Create transaction snapshot
            snapshot_id = f"tx_{transaction_id}"
            self._create_snapshot(snapshot_id)
            
            self.active_transactions[transaction_id] = {
                'snapshot_id': snapshot_id,
                'started_at': datetime.now()
            }
            
        self.logger.debug(f"Started transaction: {transaction_id}")
    
    def commit_transaction(self, transaction_id: str) -> bool:
        """
        Commit a state transaction.
        
        Args:
            transaction_id: Transaction to commit
            
        Returns:
            True if commit successful
        """
        if transaction_id not in self.active_transactions:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        with self._state_lock:
            try:
                # Apply all changes from this transaction
                tx_changes = self.transaction_changes.get(transaction_id, [])
                success = self._apply_changes(tx_changes)
                
                if success:
                    # Clean up transaction
                    snapshot_id = self.active_transactions[transaction_id]['snapshot_id']
                    self.state_snapshots.pop(snapshot_id, None)
                    del self.active_transactions[transaction_id]
                    self.transaction_changes.pop(transaction_id, None)
                    
                    self.logger.debug(f"Committed transaction: {transaction_id}")
                    return True
                else:
                    self.logger.error(f"Failed to commit transaction: {transaction_id}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error committing transaction {transaction_id}: {e}")
                return False
    
    def rollback_transaction(self, transaction_id: str) -> bool:
        """
        Rollback a state transaction.
        
        Args:
            transaction_id: Transaction to rollback
            
        Returns:
            True if rollback successful
        """
        if transaction_id not in self.active_transactions:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        with self._state_lock:
            try:
                # Restore from transaction snapshot
                snapshot_id = self.active_transactions[transaction_id]['snapshot_id']
                success = self.restore_snapshot(snapshot_id)
                
                # Clean up transaction
                del self.active_transactions[transaction_id]
                self.transaction_changes.pop(transaction_id, None)
                
                self.logger.debug(f"Rolled back transaction: {transaction_id}")
                return success
                
            except Exception as e:
                self.logger.error(f"Error rolling back transaction {transaction_id}: {e}")
                return False
    
    def _sync_loop(self) -> None:
        """Background sync loop for batched and periodic policies."""
        last_sync = time.time()
        
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                should_sync = False
                
                if self.sync_policy == SyncPolicy.BATCHED:
                    should_sync = len(self.pending_changes) >= self.batch_size
                elif self.sync_policy == SyncPolicy.PERIODIC:
                    should_sync = (current_time - last_sync) >= self.sync_interval
                
                if should_sync:
                    with self._change_lock:
                        changes_to_apply = list(self.pending_changes)
                    
                    if changes_to_apply:
                        self._apply_changes(changes_to_apply)
                        last_sync = current_time
                
                # Wait for next sync or notification
                with self._condition:
                    self._condition.wait(timeout=self.sync_interval)
                    
            except Exception as e:
                self.logger.error(f"Error in sync loop: {e}")
                time.sleep(1)
    
    def _conflict_resolution_loop(self) -> None:
        """Background loop for processing conflicts."""
        while not self._shutdown_event.is_set():
            try:
                if self.conflict_queue:
                    with self._change_lock:
                        if self.conflict_queue:
                            conflicts = self.conflict_queue.popleft()
                            resolved = self._resolve_conflicts(conflicts)
                            
                            # Add resolved changes back to pending
                            self.pending_changes.extend(resolved)
                
                time.sleep(0.1)  # Process conflicts frequently
                
            except Exception as e:
                self.logger.error(f"Error in conflict resolution loop: {e}")
                time.sleep(1)
    
    def get_state_metrics(self) -> SyncMetrics:
        """Get state coordination metrics."""
        with self._state_lock:
            # Update current state size
            try:
                state_json = json.dumps(self.agent_state.__dict__, default=str)
                self.metrics.state_size_bytes = len(state_json.encode('utf-8'))
            except Exception:
                pass
            
            return self.metrics
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status."""
        with self._state_lock:
            return {
                'state_version': self.state_version,
                'registered_modules': len(self.registered_modules),
                'pending_changes': len(self.pending_changes),
                'active_transactions': len(self.active_transactions),
                'snapshots': len(self.state_snapshots),
                'conflict_resolvers': len(self.conflict_resolvers),
                'sync_policy': self.sync_policy.value,
                'module_versions': dict(self.module_versions),
                'subscriptions': {
                    module: list(paths) 
                    for module, paths in self.module_subscriptions.items()
                }
            }
    
    def shutdown(self) -> None:
        """Shutdown the state coordinator."""
        self.logger.info("Shutting down state coordinator...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Notify sync thread
        with self._condition:
            self._condition.notify_all()
        
        # Wait for background threads
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        
        if self._conflict_thread and self._conflict_thread.is_alive():
            self._conflict_thread.join(timeout=5.0)
        
        # Apply any remaining changes
        with self._change_lock:
            if self.pending_changes:
                remaining_changes = list(self.pending_changes)
                self._apply_changes(remaining_changes)
        
        self.logger.info("State coordinator shutdown complete")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"StateCoordinator(version={self.state_version}, "
                f"modules={len(self.registered_modules)}, "
                f"pending={len(self.pending_changes)}, "
                f"policy={self.sync_policy.value})")