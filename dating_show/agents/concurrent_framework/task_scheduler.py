"""
File: task_scheduler.py
Description: TaskScheduler - Priority-based task scheduling and dependency management
Enhanced PIANO architecture with concurrent processing capabilities
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import threading
import time
import heapq
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from .concurrent_module_manager import ModuleTask, ModuleState, Priority
from ..memory_structures.security_utils import SecurityValidator


class SchedulingPolicy(Enum):
    """Task scheduling policies."""
    PRIORITY_FIRST = "priority_first"
    ROUND_ROBIN = "round_robin"
    FAIR_SHARE = "fair_share"
    SHORTEST_JOB_FIRST = "shortest_job_first"
    WEIGHTED_PRIORITY = "weighted_priority"


@dataclass
class SchedulingMetrics:
    """Metrics for task scheduling performance."""
    total_scheduled: int = 0
    total_completed: int = 0
    average_wait_time: float = 0.0
    average_execution_time: float = 0.0
    priority_distribution: Dict[Priority, int] = field(default_factory=dict)
    dependency_violations: int = 0
    scheduler_overheads: List[float] = field(default_factory=list)
    last_scheduling_time: float = 0.0


@dataclass
class TaskNode:
    """Node in the task dependency graph."""
    task: ModuleTask
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    priority_score: float = 0.0
    scheduled_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    def __lt__(self, other):
        """Comparison for priority queue (higher priority = lower number)."""
        return self.priority_score < other.priority_score


class TaskScheduler:
    """
    Priority-based task scheduler with dependency management and fair resource allocation.
    Supports multiple scheduling policies and adaptive load balancing.
    """
    
    def __init__(self, scheduling_policy: SchedulingPolicy = SchedulingPolicy.WEIGHTED_PRIORITY,
                 max_pending_tasks: int = 1000, enable_fairness: bool = True,
                 dependency_timeout: float = 300.0):
        """
        Initialize TaskScheduler.
        
        Args:
            scheduling_policy: Task scheduling policy to use
            max_pending_tasks: Maximum number of pending tasks
            enable_fairness: Enable fair scheduling across modules
            dependency_timeout: Maximum time to wait for dependencies
        """
        self.scheduling_policy = scheduling_policy
        self.max_pending_tasks = max_pending_tasks
        self.enable_fairness = enable_fairness
        self.dependency_timeout = dependency_timeout
        
        # Task queues and storage
        self.priority_queue: List[TaskNode] = []  # Min-heap for priority scheduling
        self.pending_tasks: Dict[str, TaskNode] = {}
        self.ready_queue: deque[TaskNode] = deque()
        self.blocked_tasks: Dict[str, TaskNode] = {}  # Tasks waiting for dependencies
        
        # Dependency management
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # task -> dependencies
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)  # task -> dependents
        
        # Fair scheduling
        self.module_quotas: Dict[str, int] = {}
        self.module_usage: Dict[str, int] = defaultdict(int)
        self.last_scheduled_module: Optional[str] = None
        
        # Metrics and monitoring
        self.metrics = SchedulingMetrics()
        self.task_wait_times: Dict[str, float] = {}
        
        # Synchronization
        self._lock = threading.RLock()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()
        
        # Start scheduler thread
        self._start_scheduler()
    
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
    
    def _start_scheduler(self) -> None:
        """Start the background scheduler thread."""
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="TaskScheduler",
            daemon=True
        )
        self._scheduler_thread.start()
    
    def schedule_task(self, task: ModuleTask, dependencies: Optional[Set[str]] = None) -> bool:
        """
        Schedule a task for execution.
        
        Args:
            task: Task to schedule
            dependencies: Set of task IDs this task depends on
            
        Returns:
            True if task was scheduled, False if rejected
        """
        with self._lock:
            if len(self.pending_tasks) >= self.max_pending_tasks:
                self.logger.warning(f"Task queue full, rejecting task {task.task_id}")
                return False
            
            # Create task node
            task_node = TaskNode(
                task=task,
                dependencies=dependencies or set(),
                priority_score=self._calculate_priority_score(task),
                scheduled_at=datetime.now()
            )
            
            # Add to pending tasks
            self.pending_tasks[task.task_id] = task_node
            
            # Update dependency graph
            if dependencies:
                self.dependency_graph[task.task_id] = dependencies.copy()
                for dep_id in dependencies:
                    self.reverse_dependencies[dep_id].add(task.task_id)
            
            # Check if task is ready to execute
            if self._are_dependencies_satisfied(task.task_id):
                self._make_task_ready(task_node)
            else:
                self.blocked_tasks[task.task_id] = task_node
                task.state = ModuleState.QUEUED
            
            # Update metrics
            self.metrics.total_scheduled += 1
            priority = task.priority
            self.metrics.priority_distribution[priority] = (
                self.metrics.priority_distribution.get(priority, 0) + 1
            )
            
            self.logger.debug(f"Scheduled task {task.task_id} with priority {task.priority}")
            return True
    
    def get_next_task(self) -> Optional[ModuleTask]:
        """
        Get the next task to execute based on scheduling policy.
        
        Returns:
            Next task to execute or None if no tasks available
        """
        with self._lock:
            if not self.ready_queue and not self.priority_queue:
                return None
            
            start_time = time.time()
            
            # Select task based on scheduling policy
            task_node = self._select_next_task()
            
            if task_node:
                # Remove from ready structures
                if task_node in self.priority_queue:
                    self.priority_queue.remove(task_node)
                    heapq.heapify(self.priority_queue)
                elif task_node in self.ready_queue:
                    self.ready_queue.remove(task_node)
                
                # Update metrics
                self.metrics.last_scheduling_time = time.time() - start_time
                self.metrics.scheduler_overheads.append(self.metrics.last_scheduling_time)
                
                # Keep only recent overhead measurements
                if len(self.metrics.scheduler_overheads) > 1000:
                    self.metrics.scheduler_overheads = self.metrics.scheduler_overheads[-500:]
                
                # Track wait time
                if task_node.task.created_at:
                    wait_time = (datetime.now() - task_node.task.created_at).total_seconds()
                    self.task_wait_times[task_node.task.task_id] = wait_time
                
                # Update module usage for fair scheduling
                if self.enable_fairness:
                    self.module_usage[task_node.task.module_name] += 1
                    self.last_scheduled_module = task_node.task.module_name
                
                self.logger.debug(f"Selected task {task_node.task.task_id} for execution")
                return task_node.task
            
            return None
    
    def _select_next_task(self) -> Optional[TaskNode]:
        """Select the next task based on the current scheduling policy."""
        if self.scheduling_policy == SchedulingPolicy.PRIORITY_FIRST:
            return self._select_priority_first()
        elif self.scheduling_policy == SchedulingPolicy.ROUND_ROBIN:
            return self._select_round_robin()
        elif self.scheduling_policy == SchedulingPolicy.FAIR_SHARE:
            return self._select_fair_share()
        elif self.scheduling_policy == SchedulingPolicy.WEIGHTED_PRIORITY:
            return self._select_weighted_priority()
        else:
            return self._select_priority_first()  # Default fallback
    
    def _select_priority_first(self) -> Optional[TaskNode]:
        """Select highest priority task."""
        if self.priority_queue:
            return heapq.heappop(self.priority_queue)
        elif self.ready_queue:
            return self.ready_queue.popleft()
        return None
    
    def _select_round_robin(self) -> Optional[TaskNode]:
        """Select next task using round-robin among modules."""
        if not self.ready_queue and not self.priority_queue:
            return None
        
        # Combine available tasks
        all_ready_tasks = list(self.ready_queue) + self.priority_queue
        if not all_ready_tasks:
            return None
        
        # Group by module
        module_tasks = defaultdict(list)
        for task_node in all_ready_tasks:
            module_tasks[task_node.task.module_name].append(task_node)
        
        # Select next module in round-robin fashion
        module_names = sorted(module_tasks.keys())
        if not module_names:
            return None
        
        # Find next module after last scheduled
        start_idx = 0
        if self.last_scheduled_module in module_names:
            start_idx = (module_names.index(self.last_scheduled_module) + 1) % len(module_names)
        
        selected_module = module_names[start_idx]
        selected_task = module_tasks[selected_module][0]
        
        # Remove from appropriate queue
        if selected_task in self.ready_queue:
            self.ready_queue.remove(selected_task)
        elif selected_task in self.priority_queue:
            self.priority_queue.remove(selected_task)
            heapq.heapify(self.priority_queue)
        
        return selected_task
    
    def _select_fair_share(self) -> Optional[TaskNode]:
        """Select task ensuring fair resource allocation among modules."""
        if not self.ready_queue and not self.priority_queue:
            return None
        
        # Combine all ready tasks
        all_ready_tasks = list(self.ready_queue) + self.priority_queue
        if not all_ready_tasks:
            return None
        
        # Group by module and calculate usage ratios
        module_tasks = defaultdict(list)
        for task_node in all_ready_tasks:
            module_tasks[task_node.task.module_name].append(task_node)
        
        # Find module with lowest usage ratio
        min_usage_ratio = float('inf')
        selected_module = None
        
        for module_name, tasks in module_tasks.items():
            quota = self.module_quotas.get(module_name, 1)
            current_usage = self.module_usage[module_name]
            usage_ratio = current_usage / quota
            
            if usage_ratio < min_usage_ratio:
                min_usage_ratio = usage_ratio
                selected_module = module_name
        
        if selected_module and module_tasks[selected_module]:
            # Select highest priority task from selected module
            selected_task = min(module_tasks[selected_module], key=lambda x: x.priority_score)
            
            # Remove from appropriate queue
            if selected_task in self.ready_queue:
                self.ready_queue.remove(selected_task)
            elif selected_task in self.priority_queue:
                self.priority_queue.remove(selected_task)
                heapq.heapify(self.priority_queue)
            
            return selected_task
        
        return self._select_priority_first()  # Fallback
    
    def _select_weighted_priority(self) -> Optional[TaskNode]:
        """Select task using weighted priority considering fairness."""
        if not self.ready_queue and not self.priority_queue:
            return None
        
        # Combine all ready tasks
        all_ready_tasks = list(self.ready_queue) + self.priority_queue
        if not all_ready_tasks:
            return None
        
        # Calculate weighted scores
        best_task = None
        best_score = float('-inf')
        
        for task_node in all_ready_tasks:
            # Base priority score
            score = -task_node.priority_score  # Negative because lower priority value = higher priority
            
            # Fairness adjustment
            if self.enable_fairness:
                module_name = task_node.task.module_name
                quota = self.module_quotas.get(module_name, 1)
                usage = self.module_usage[module_name]
                fairness_boost = max(0, quota - usage) * 0.1  # Boost underutilized modules
                score += fairness_boost
            
            # Age adjustment (avoid starvation)
            if task_node.scheduled_at:
                age_seconds = (datetime.now() - task_node.scheduled_at).total_seconds()
                age_boost = min(age_seconds / 300.0, 2.0)  # Up to 2 points for 5+ minutes
                score += age_boost
            
            if score > best_score:
                best_score = score
                best_task = task_node
        
        # Remove selected task from queues
        if best_task:
            if best_task in self.ready_queue:
                self.ready_queue.remove(best_task)
            elif best_task in self.priority_queue:
                self.priority_queue.remove(best_task)
                heapq.heapify(self.priority_queue)
        
        return best_task
    
    def task_completed(self, task_id: str, success: bool = True) -> None:
        """
        Mark a task as completed and update dependent tasks.
        
        Args:
            task_id: ID of completed task
            success: Whether task completed successfully
        """
        with self._lock:
            # Update metrics
            self.metrics.total_completed += 1
            if task_id in self.task_wait_times:
                wait_time = self.task_wait_times.pop(task_id)
                # Update average wait time
                total_wait = self.metrics.average_wait_time * (self.metrics.total_completed - 1)
                self.metrics.average_wait_time = (total_wait + wait_time) / self.metrics.total_completed
            
            # Remove from pending tasks
            self.pending_tasks.pop(task_id, None)
            
            # Check dependent tasks
            if task_id in self.reverse_dependencies:
                dependent_task_ids = self.reverse_dependencies[task_id].copy()
                
                for dependent_id in dependent_task_ids:
                    if dependent_id in self.blocked_tasks:
                        # Remove this dependency
                        if dependent_id in self.dependency_graph:
                            self.dependency_graph[dependent_id].discard(task_id)
                        
                        # Check if all dependencies are now satisfied
                        if self._are_dependencies_satisfied(dependent_id):
                            task_node = self.blocked_tasks.pop(dependent_id)
                            self._make_task_ready(task_node)
                
                # Clean up reverse dependencies
                del self.reverse_dependencies[task_id]
            
            # Clean up dependency graph
            self.dependency_graph.pop(task_id, None)
            
            self.logger.debug(f"Task {task_id} marked as completed")
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled, False if not found
        """
        with self._lock:
            cancelled = False
            
            # Remove from pending tasks
            if task_id in self.pending_tasks:
                task_node = self.pending_tasks.pop(task_id)
                
                # Remove from queues
                if task_node in self.priority_queue:
                    self.priority_queue.remove(task_node)
                    heapq.heapify(self.priority_queue)
                    cancelled = True
                elif task_node in self.ready_queue:
                    self.ready_queue.remove(task_node)
                    cancelled = True
                elif task_id in self.blocked_tasks:
                    self.blocked_tasks.pop(task_id)
                    cancelled = True
            
            # Clean up dependencies
            if task_id in self.dependency_graph:
                del self.dependency_graph[task_id]
            
            if task_id in self.reverse_dependencies:
                # Notify dependent tasks that this dependency failed
                for dependent_id in self.reverse_dependencies[task_id]:
                    if dependent_id in self.blocked_tasks:
                        self.metrics.dependency_violations += 1
                del self.reverse_dependencies[task_id]
            
            self.logger.debug(f"Task {task_id} cancellation: {cancelled}")
            return cancelled
    
    def set_module_quota(self, module_name: str, quota: int) -> None:
        """
        Set execution quota for a module (for fair scheduling).
        
        Args:
            module_name: Name of the module
            quota: Execution quota (tasks per scheduling cycle)
        """
        with self._lock:
            self.module_quotas[module_name] = max(1, quota)
            self.logger.info(f"Set quota for module {module_name}: {quota}")
    
    def _calculate_priority_score(self, task: ModuleTask) -> float:
        """
        Calculate priority score for a task.
        
        Args:
            task: Task to calculate priority for
            
        Returns:
            Priority score (lower = higher priority)
        """
        # Base priority from enum value
        base_score = task.priority.value
        
        # Adjust for deadline if present
        if hasattr(task, 'deadline') and task.deadline:
            time_to_deadline = (task.deadline - datetime.now()).total_seconds()
            if time_to_deadline > 0:
                # Closer deadline = higher priority (lower score)
                deadline_factor = max(0.1, time_to_deadline / 3600)  # Hours to deadline
                base_score *= deadline_factor
        
        # Adjust for estimated duration
        if task.max_duration:
            # Shorter tasks get slight priority boost
            duration_factor = min(2.0, task.max_duration / 60.0)  # Minutes
            base_score *= (1 + duration_factor * 0.1)
        
        return base_score
    
    def _are_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied."""
        dependencies = self.dependency_graph.get(task_id, set())
        
        # Check for timeout
        if task_id in self.blocked_tasks:
            task_node = self.blocked_tasks[task_id]
            if task_node.scheduled_at:
                wait_time = (datetime.now() - task_node.scheduled_at).total_seconds()
                if wait_time > self.dependency_timeout:
                    self.logger.warning(f"Dependency timeout for task {task_id}")
                    self.metrics.dependency_violations += 1
                    return True  # Force execution
        
        # All dependencies must be completed (not in pending or blocked)
        for dep_id in dependencies:
            if dep_id in self.pending_tasks or dep_id in self.blocked_tasks:
                return False
        
        return True
    
    def _make_task_ready(self, task_node: TaskNode) -> None:
        """Move a task to ready state for execution."""
        task_node.task.state = ModuleState.QUEUED
        
        if task_node.task.priority in [Priority.CRITICAL, Priority.HIGH]:
            heapq.heappush(self.priority_queue, task_node)
        else:
            self.ready_queue.append(task_node)
    
    def _scheduler_loop(self) -> None:
        """Background scheduler loop for periodic maintenance."""
        while not self._shutdown_event.is_set():
            try:
                with self._lock:
                    # Check for expired dependencies
                    expired_tasks = []
                    current_time = datetime.now()
                    
                    for task_id, task_node in self.blocked_tasks.items():
                        if (task_node.scheduled_at and 
                            (current_time - task_node.scheduled_at).total_seconds() > self.dependency_timeout):
                            expired_tasks.append(task_id)
                    
                    # Move expired tasks to ready queue
                    for task_id in expired_tasks:
                        task_node = self.blocked_tasks.pop(task_id)
                        self._make_task_ready(task_node)
                        self.logger.warning(f"Moved task {task_id} to ready due to dependency timeout")
                
                # Reset module usage counters periodically
                if len(self.metrics.scheduler_overheads) % 100 == 0:
                    with self._lock:
                        self.module_usage.clear()
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
            
            time.sleep(5)  # Run every 5 seconds
    
    def get_scheduling_metrics(self) -> SchedulingMetrics:
        """Get current scheduling metrics."""
        with self._lock:
            return self.metrics
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status information.
        
        Returns:
            Dictionary with queue status details
        """
        with self._lock:
            return {
                'priority_queue_size': len(self.priority_queue),
                'ready_queue_size': len(self.ready_queue),
                'blocked_tasks': len(self.blocked_tasks),
                'pending_tasks': len(self.pending_tasks),
                'total_dependencies': sum(len(deps) for deps in self.dependency_graph.values()),
                'scheduling_policy': self.scheduling_policy.value,
                'average_scheduler_overhead': (
                    sum(self.metrics.scheduler_overheads) / len(self.metrics.scheduler_overheads)
                    if self.metrics.scheduler_overheads else 0.0
                )
            }
    
    def shutdown(self) -> None:
        """Shutdown the task scheduler."""
        self.logger.info("Shutting down task scheduler...")
        self._shutdown_event.set()
        
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)
        
        with self._lock:
            self.priority_queue.clear()
            self.ready_queue.clear()
            self.blocked_tasks.clear()
            self.pending_tasks.clear()
            self.dependency_graph.clear()
            self.reverse_dependencies.clear()
        
        self.logger.info("Task scheduler shutdown complete")
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"TaskScheduler(policy={self.scheduling_policy.value}, "
                f"pending={len(self.pending_tasks)}, "
                f"ready={len(self.ready_queue) + len(self.priority_queue)}, "
                f"blocked={len(self.blocked_tasks)})")