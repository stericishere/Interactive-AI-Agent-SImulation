"""
File: concurrent_module_manager.py
Description: ConcurrentModuleManager - Central coordination hub for concurrent module execution
Enhanced PIANO architecture with concurrent processing capabilities
"""

from typing import Dict, List, Optional, Any, Callable, Union, Set
from datetime import datetime, timedelta
import threading
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import time
import traceback

from ..modules.base_module import BaseModule
from ..memory_structures.security_utils import SecurityValidator, SecurityError


class ModuleState(Enum):
    """States that a module can be in during concurrent execution."""
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class Priority(Enum):
    """Task priority levels for concurrent execution."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ModuleTask:
    """Represents a task to be executed by a concurrent module."""
    task_id: str
    module_name: str
    method_name: str
    args: tuple = ()
    kwargs: Dict[str, Any] = None
    priority: Priority = Priority.NORMAL
    max_duration: Optional[float] = None
    dependencies: Set[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    state: ModuleState = ModuleState.QUEUED
    result: Any = None
    error: Optional[Exception] = None
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = set()
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ModuleMetrics:
    """Performance metrics for concurrent modules."""
    execution_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    error_count: int = 0
    last_execution: Optional[datetime] = None
    peak_memory_usage: float = 0.0
    concurrent_executions: int = 0


class ConcurrentModuleManager:
    """
    Central coordination hub for concurrent module execution in the PIANO architecture.
    Manages task scheduling, resource allocation, and inter-module communication.
    """
    
    def __init__(self, agent_state, max_workers: int = 4, max_queue_size: int = 100,
                 enable_monitoring: bool = True, task_timeout: float = 30.0):
        """
        Initialize ConcurrentModuleManager.
        
        Args:
            agent_state: Shared agent state object
            max_workers: Maximum number of concurrent worker threads
            max_queue_size: Maximum size of task queue
            enable_monitoring: Enable performance monitoring
            task_timeout: Default task timeout in seconds
        """
        self.agent_state = agent_state
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.enable_monitoring = enable_monitoring
        self.task_timeout = task_timeout
        
        # Core components (to be initialized separately)
        self.module_executor = None
        self.task_scheduler = None 
        self.resource_coordinator = None
        self.state_coordinator = None
        
        # Module registry
        self.modules: Dict[str, BaseModule] = {}
        self.module_configs: Dict[str, Dict[str, Any]] = {}
        self.module_states: Dict[str, ModuleState] = {}
        self.module_metrics: Dict[str, ModuleMetrics] = {}
        
        # Task management
        self.active_tasks: Dict[str, ModuleTask] = {}
        self.task_history: List[ModuleTask] = []
        self.task_dependencies: Dict[str, Set[str]] = {}
        
        # Synchronization
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._paused_event = threading.Event()
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'module_started': [],
            'module_completed': [],
            'module_error': [],
            'task_queued': [],
            'task_completed': [],
            'system_overload': []
        }
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        # Execution loop thread
        self._execution_thread = None
        
    def _setup_logging(self) -> None:
        """Setup logging configuration for the manager."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _initialize_components(self) -> None:
        """Initialize core components."""
        from .module_executor import ModuleExecutor
        from .task_scheduler import TaskScheduler, SchedulingPolicy
        from .resource_coordinator import ResourceCoordinator
        from .state_coordinator import StateCoordinator, SyncPolicy
        
        # Initialize components
        self.module_executor = ModuleExecutor(
            max_workers=self.max_workers,
            max_queue_size=self.max_queue_size,
            enable_monitoring=self.enable_monitoring
        )
        
        self.task_scheduler = TaskScheduler(
            scheduling_policy=SchedulingPolicy.WEIGHTED_PRIORITY,
            max_pending_tasks=self.max_queue_size
        )
        
        self.resource_coordinator = ResourceCoordinator(
            max_concurrent_locks=self.max_workers * 2,
            default_timeout=self.task_timeout
        )
        
        self.state_coordinator = StateCoordinator(
            agent_state=self.agent_state,
            sync_policy=SyncPolicy.IMMEDIATE
        )
        
        self.logger.info("ConcurrentModuleManager initialized with all components")
        
    def register_module(self, name: str, module: BaseModule, 
                       config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a module for concurrent execution.
        
        Args:
            name: Unique name for the module
            module: Module instance extending BaseModule
            config: Optional configuration dictionary
        """
        with self._lock:
            if name in self.modules:
                raise ValueError(f"Module '{name}' is already registered")
            
            # Security validation
            if not isinstance(module, BaseModule):
                raise TypeError(f"Module must extend BaseModule, got {type(module)}")
            
            self.modules[name] = module
            self.module_configs[name] = config or {}
            self.module_states[name] = ModuleState.IDLE
            self.module_metrics[name] = ModuleMetrics()
            
            self.logger.info(f"Registered module: {name}")
    
    def unregister_module(self, name: str) -> None:
        """
        Unregister a module and cancel any pending tasks.
        
        Args:
            name: Name of module to unregister
        """
        with self._lock:
            if name not in self.modules:
                raise KeyError(f"Module '{name}' is not registered")
            
            # Cancel pending tasks for this module
            self._cancel_module_tasks(name)
            
            # Remove from registries
            del self.modules[name]
            del self.module_configs[name]
            del self.module_states[name] 
            del self.module_metrics[name]
            
            self.logger.info(f"Unregistered module: {name}")
    
    def submit_task(self, module_name: str, method_name: str = "run",
                   args: tuple = (), kwargs: Optional[Dict[str, Any]] = None,
                   priority: Priority = Priority.NORMAL,
                   max_duration: Optional[float] = None,
                   dependencies: Optional[Set[str]] = None,
                   callback: Optional[Callable] = None) -> str:
        """
        Submit a task for concurrent execution.
        
        Args:
            module_name: Name of registered module
            method_name: Method to execute on module
            args: Positional arguments
            kwargs: Keyword arguments
            priority: Task priority
            max_duration: Maximum execution time in seconds
            dependencies: Set of task IDs this task depends on
            callback: Optional callback function for completion
        
        Returns:
            Task ID string
        """
        with self._lock:
            if module_name not in self.modules:
                raise KeyError(f"Module '{module_name}' is not registered")
            
            if self._shutdown_event.is_set():
                raise RuntimeError("Manager is shutting down")
            
            # Generate unique task ID
            task_id = f"{module_name}_{method_name}_{int(time.time() * 1000000)}"
            
            # Create task
            task = ModuleTask(
                task_id=task_id,
                module_name=module_name,
                method_name=method_name,
                args=args,
                kwargs=kwargs or {},
                priority=priority,
                max_duration=max_duration or self.task_timeout,
                dependencies=dependencies or set(),
                callback=callback
            )
            
            # Validate dependencies exist
            for dep_id in task.dependencies:
                if dep_id not in self.active_tasks and not self._task_completed(dep_id):
                    raise ValueError(f"Dependency task '{dep_id}' does not exist")
            
            # Add to active tasks
            self.active_tasks[task_id] = task
            
            # Trigger callbacks
            self._trigger_event('task_queued', task)
            
            self.logger.info(f"Submitted task {task_id} for module {module_name}")
            
            # Schedule the task
            self.task_scheduler.schedule_task(task, dependencies)
            
            # Start task execution loop if not already running
            self._ensure_execution_loop()
            
            return task_id
    
    def get_task_status(self, task_id: str) -> Optional[ModuleState]:
        """
        Get the current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Current task state or None if not found
        """
        with self._lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].state
            
            # Check task history
            for task in reversed(self.task_history):
                if task.task_id == task_id:
                    return task.state
            
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled, False otherwise
        """
        with self._lock:
            if task_id not in self.active_tasks:
                return False
            
            task = self.active_tasks[task_id]
            
            if task.state in [ModuleState.COMPLETED, ModuleState.ERROR, ModuleState.CANCELLED]:
                return False
            
            task.state = ModuleState.CANCELLED
            task.completed_at = datetime.now()
            
            # Move to history
            self.task_history.append(task)
            del self.active_tasks[task_id]
            
            self.logger.info(f"Cancelled task: {task_id}")
            return True
    
    def pause_module(self, module_name: str) -> None:
        """
        Pause execution of tasks for a specific module.
        
        Args:
            module_name: Name of module to pause
        """
        with self._lock:
            if module_name not in self.modules:
                raise KeyError(f"Module '{module_name}' is not registered")
            
            self.module_states[module_name] = ModuleState.PAUSED
            self.logger.info(f"Paused module: {module_name}")
    
    def resume_module(self, module_name: str) -> None:
        """
        Resume execution of tasks for a specific module.
        
        Args:
            module_name: Name of module to resume
        """
        with self._lock:
            if module_name not in self.modules:
                raise KeyError(f"Module '{module_name}' is not registered")
            
            self.module_states[module_name] = ModuleState.IDLE
            self.logger.info(f"Resumed module: {module_name}")
    
    def get_module_metrics(self, module_name: str) -> Optional[ModuleMetrics]:
        """
        Get performance metrics for a module.
        
        Args:
            module_name: Name of module
            
        Returns:
            ModuleMetrics object or None if module not found
        """
        with self._lock:
            return self.module_metrics.get(module_name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status information.
        
        Returns:
            Dictionary with system status details
        """
        with self._lock:
            active_count = len(self.active_tasks)
            running_count = sum(1 for task in self.active_tasks.values() 
                              if task.state == ModuleState.RUNNING)
            queued_count = sum(1 for task in self.active_tasks.values()
                             if task.state == ModuleState.QUEUED)
            
            return {
                'registered_modules': len(self.modules),
                'active_tasks': active_count,
                'running_tasks': running_count,
                'queued_tasks': queued_count,
                'completed_tasks': len(self.task_history),
                'is_paused': self._paused_event.is_set(),
                'is_shutdown': self._shutdown_event.is_set(),
                'uptime_seconds': (datetime.now() - getattr(self, '_start_time', datetime.now())).total_seconds()
            }
    
    def add_event_callback(self, event_type: str, callback: Callable) -> None:
        """
        Add callback for system events.
        
        Args:
            event_type: Type of event to listen for
            callback: Callback function to execute
        """
        if event_type not in self.event_callbacks:
            raise ValueError(f"Unknown event type: {event_type}")
        
        self.event_callbacks[event_type].append(callback)
    
    def remove_event_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Remove event callback.
        
        Args:
            event_type: Type of event
            callback: Callback function to remove
            
        Returns:
            True if callback was removed, False if not found
        """
        if event_type not in self.event_callbacks:
            return False
        
        try:
            self.event_callbacks[event_type].remove(callback)
            return True
        except ValueError:
            return False
    
    def shutdown(self, timeout: float = 10.0) -> None:
        """
        Shutdown the concurrent module manager.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        self.logger.info("Initiating shutdown...")
        
        with self._lock:
            self._shutdown_event.set()
            
            # Cancel all active tasks
            for task_id in list(self.active_tasks.keys()):
                self.cancel_task(task_id)
        
        # Shutdown components
        if self.module_executor:
            self.module_executor.shutdown(timeout)
        if self.task_scheduler:
            self.task_scheduler.shutdown()
        if self.resource_coordinator:
            self.resource_coordinator.shutdown()
        if self.state_coordinator:
            self.state_coordinator.shutdown()
        
        # Wait for execution thread
        if self._execution_thread and self._execution_thread.is_alive():
            self._execution_thread.join(timeout=5.0)
        
        self.logger.info("Shutdown complete")
    
    def _cancel_module_tasks(self, module_name: str) -> None:
        """Cancel all tasks for a specific module."""
        tasks_to_cancel = [
            task_id for task_id, task in self.active_tasks.items()
            if task.module_name == module_name
        ]
        
        for task_id in tasks_to_cancel:
            self.cancel_task(task_id)
    
    def _task_completed(self, task_id: str) -> bool:
        """Check if a task has been completed."""
        for task in self.task_history:
            if task.task_id == task_id:
                return task.state in [ModuleState.COMPLETED, ModuleState.ERROR, ModuleState.CANCELLED]
        return False
    
    def _trigger_event(self, event_type: str, *args, **kwargs) -> None:
        """Trigger event callbacks."""
        for callback in self.event_callbacks.get(event_type, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in event callback for {event_type}: {e}")
    
    def _update_metrics(self, module_name: str, execution_time: float, 
                       success: bool = True) -> None:
        """Update performance metrics for a module."""
        if not self.enable_monitoring:
            return
            
        metrics = self.module_metrics[module_name]
        metrics.execution_count += 1
        metrics.total_execution_time += execution_time
        metrics.average_execution_time = metrics.total_execution_time / metrics.execution_count
        metrics.last_execution = datetime.now()
        
        if not success:
            metrics.error_count += 1
    
    def __enter__(self):
        """Context manager entry."""
        self._start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.shutdown()
    
    def _ensure_execution_loop(self) -> None:
        """Ensure the task execution loop is running."""
        if self._execution_thread is None or not self._execution_thread.is_alive():
            self._execution_thread = threading.Thread(
                target=self._execution_loop,
                name="TaskExecutionLoop",
                daemon=True
            )
            self._execution_thread.start()
    
    def _execution_loop(self) -> None:
        """Main execution loop that gets tasks from scheduler and executes them."""
        while not self._shutdown_event.is_set():
            try:
                # Get next task from scheduler
                next_task = self.task_scheduler.get_next_task()
                
                if next_task:
                    # Get the module for this task
                    module = self.modules.get(next_task.module_name)
                    if module:
                        # Execute the task
                        future = self.module_executor.execute_task(next_task, module)
                        
                        # Set up completion callback
                        def task_completion_callback(task_id, success=True):
                            self.task_scheduler.task_completed(task_id, success)
                            self._trigger_event('task_completed', task_id, success)
                        
                        # Add callback to future
                        def completion_wrapper(fut):
                            try:
                                fut.result()  # This will raise exception if task failed
                                task_completion_callback(next_task.task_id, True)
                            except Exception as e:
                                self.logger.error(f"Task {next_task.task_id} failed: {e}")
                                task_completion_callback(next_task.task_id, False)
                        
                        future.add_done_callback(completion_wrapper)
                    else:
                        self.logger.error(f"Module {next_task.module_name} not found for task {next_task.task_id}")
                else:
                    # No tasks available, wait a bit
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                time.sleep(1)
    
    def __repr__(self) -> str:
        """String representation of the manager."""
        return (f"ConcurrentModuleManager(modules={len(self.modules)}, "
                f"active_tasks={len(self.active_tasks)}, "
                f"max_workers={self.max_workers})")