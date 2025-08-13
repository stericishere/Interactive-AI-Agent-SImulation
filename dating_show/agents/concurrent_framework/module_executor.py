"""
File: module_executor.py
Description: ModuleExecutor - Thread pool management and task execution for concurrent modules
Enhanced PIANO architecture with concurrent processing capabilities
"""

from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
import threading
import asyncio
import logging
import time
import traceback
import resource
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, Future, as_completed, TimeoutError
from queue import Queue, Empty, Full
from dataclasses import dataclass
from enum import Enum

from .concurrent_module_manager import ModuleTask, ModuleState, Priority
from ..modules.base_module import BaseModule
from ..memory_structures.security_utils import SecurityValidator


@dataclass
class ExecutorMetrics:
    """Performance metrics for the module executor."""
    tasks_executed: int = 0
    tasks_failed: int = 0
    tasks_timed_out: int = 0
    total_execution_time: float = 0.0
    peak_memory_usage: int = 0
    peak_thread_count: int = 0
    current_load: float = 0.0
    last_cleanup: Optional[datetime] = None


@dataclass
class ThreadInfo:
    """Information about an executor thread."""
    thread_id: str
    thread_name: str
    task_id: Optional[str] = None
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    memory_usage: int = 0
    is_busy: bool = False


class ModuleExecutor:
    """
    Thread pool management and task execution system for concurrent modules.
    Provides resource monitoring, load balancing, and secure task execution.
    """
    
    def __init__(self, max_workers: int = 4, max_queue_size: int = 100,
                 enable_monitoring: bool = True, cleanup_interval: int = 300):
        """
        Initialize ModuleExecutor.
        
        Args:
            max_workers: Maximum number of worker threads
            max_queue_size: Maximum size of execution queue
            enable_monitoring: Enable performance monitoring
            cleanup_interval: Cleanup interval in seconds
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.enable_monitoring = enable_monitoring
        self.cleanup_interval = cleanup_interval
        
        # Core components
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ModuleExecutor"
        )
        self.execution_queue: Queue[ModuleTask] = Queue(maxsize=max_queue_size)
        
        # State management
        self.active_futures: Dict[str, Future] = {}
        self.thread_info: Dict[str, ThreadInfo] = {}
        self.metrics = ExecutorMetrics()
        
        # Control flow
        self._shutdown_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.RLock()
        
        # Resource monitoring
        self.process = psutil.Process(os.getpid())
        self._monitor_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        
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
        """Start monitoring and cleanup threads."""
        if self.enable_monitoring:
            self._monitor_thread = threading.Thread(
                target=self._monitor_resources,
                name="ExecutorMonitor",
                daemon=True
            )
            self._monitor_thread.start()
        
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_completed_tasks,
            name="ExecutorCleanup", 
            daemon=True
        )
        self._cleanup_thread.start()
    
    def execute_task(self, task: ModuleTask, module: BaseModule) -> Future:
        """
        Execute a task using the thread pool.
        
        Args:
            task: Task to execute
            module: Module instance to run task on
            
        Returns:
            Future object for the task execution
            
        Raises:
            RuntimeError: If executor is shutdown or queue is full
        """
        if self._shutdown_event.is_set():
            raise RuntimeError("Executor is shutdown")
        
        if self.execution_queue.full():
            raise RuntimeError("Execution queue is full")
        
        # Security validation
        self._validate_task_security(task, module)
        
        # Update task state
        task.state = ModuleState.QUEUED
        task.started_at = datetime.now()
        
        # Submit to thread pool
        future = self.thread_pool.submit(self._execute_task_safe, task, module)
        
        with self._lock:
            self.active_futures[task.task_id] = future
            self.execution_queue.put(task)
        
        self.logger.info(f"Submitted task {task.task_id} for execution")
        
        # Add completion callback
        future.add_done_callback(lambda f: self._task_completed_callback(task.task_id, f))
        
        return future
    
    def _execute_task_safe(self, task: ModuleTask, module: BaseModule) -> Any:
        """
        Safely execute a task with error handling and monitoring.
        
        Args:
            task: Task to execute
            module: Module instance
            
        Returns:
            Task result
        """
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        
        # Update thread info
        with self._lock:
            self.thread_info[str(thread_id)] = ThreadInfo(
                thread_id=str(thread_id),
                thread_name=thread_name,
                task_id=task.task_id,
                started_at=datetime.now(),
                last_activity=datetime.now(),
                is_busy=True
            )
        
        # Update task state
        task.state = ModuleState.RUNNING
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting execution of task {task.task_id}")
            
            # Check if paused
            while self._pause_event.is_set() and not self._shutdown_event.is_set():
                time.sleep(0.1)
            
            if self._shutdown_event.is_set():
                raise RuntimeError("Executor shutdown during task execution")
            
            # Execute the task method
            if hasattr(module, task.method_name):
                method = getattr(module, task.method_name)
                
                # Execute with timeout if specified
                if task.max_duration:
                    result = self._execute_with_timeout(
                        method, task.args, task.kwargs, task.max_duration
                    )
                else:
                    result = method(*task.args, **task.kwargs)
                
                task.result = result
                task.state = ModuleState.COMPLETED
                task.completed_at = datetime.now()
                
                self.logger.info(f"Task {task.task_id} completed successfully")
                return result
                
            else:
                raise AttributeError(f"Module {module.__class__.__name__} has no method '{task.method_name}'")
        
        except Exception as e:
            execution_time = time.time() - start_time
            task.error = e
            task.state = ModuleState.ERROR
            task.completed_at = datetime.now()
            
            # Update metrics
            with self._lock:
                self.metrics.tasks_failed += 1
                if isinstance(e, TimeoutError):
                    self.metrics.tasks_timed_out += 1
            
            self.logger.error(f"Task {task.task_id} failed: {str(e)}\n{traceback.format_exc()}")
            raise
        
        finally:
            execution_time = time.time() - start_time
            
            # Update metrics
            with self._lock:
                self.metrics.tasks_executed += 1
                self.metrics.total_execution_time += execution_time
                
                # Update thread info
                if str(thread_id) in self.thread_info:
                    self.thread_info[str(thread_id)].is_busy = False
                    self.thread_info[str(thread_id)].last_activity = datetime.now()
    
    def _execute_with_timeout(self, method: Callable, args: tuple, 
                            kwargs: Dict[str, Any], timeout: float) -> Any:
        """
        Execute a method with timeout using a separate thread.
        
        Args:
            method: Method to execute
            args: Method arguments
            kwargs: Method keyword arguments
            timeout: Timeout in seconds
            
        Returns:
            Method result
            
        Raises:
            TimeoutError: If execution exceeds timeout
        """
        result_queue = Queue()
        exception_queue = Queue()
        
        def target():
            try:
                result = method(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred - we can't actually kill the thread in Python
            # But we can mark it and continue
            self.logger.warning(f"Task execution exceeded timeout of {timeout}s")
            raise TimeoutError(f"Task execution timed out after {timeout}s")
        
        # Check for exceptions
        if not exception_queue.empty():
            raise exception_queue.get()
        
        # Return result
        if not result_queue.empty():
            return result_queue.get()
        
        return None
    
    def _validate_task_security(self, task: ModuleTask, module: BaseModule) -> None:
        """
        Validate task security before execution.
        
        Args:
            task: Task to validate
            module: Module to validate
            
        Raises:
            SecurityError: If validation fails
        """
        # Validate method name
        if not hasattr(module, task.method_name):
            raise SecurityError(f"Module does not have method: {task.method_name}")
        
        # Validate arguments (basic sanitization)
        SecurityValidator.validate_execution_args(task.args, task.kwargs)
        
        # Check resource limits
        current_memory = self.process.memory_info().rss
        if current_memory > 1024 * 1024 * 1024:  # 1GB limit
            raise SecurityError("Memory usage too high for task execution")
    
    def _task_completed_callback(self, task_id: str, future: Future) -> None:
        """
        Callback executed when a task completes.
        
        Args:
            task_id: ID of completed task
            future: Completed future object
        """
        with self._lock:
            if task_id in self.active_futures:
                del self.active_futures[task_id]
        
        # Remove from execution queue (find and remove)
        try:
            temp_queue = Queue()
            while not self.execution_queue.empty():
                task = self.execution_queue.get_nowait()
                if task.task_id != task_id:
                    temp_queue.put(task)
            
            # Put remaining tasks back
            while not temp_queue.empty():
                self.execution_queue.put(temp_queue.get_nowait())
                
        except Empty:
            pass
    
    def _monitor_resources(self) -> None:
        """Background thread for monitoring system resources."""
        while not self._shutdown_event.is_set():
            try:
                # Update memory usage
                memory_info = self.process.memory_info()
                with self._lock:
                    if memory_info.rss > self.metrics.peak_memory_usage:
                        self.metrics.peak_memory_usage = memory_info.rss
                    
                    # Update thread count
                    current_threads = threading.active_count()
                    if current_threads > self.metrics.peak_thread_count:
                        self.metrics.peak_thread_count = current_threads
                    
                    # Calculate current load
                    active_tasks = len(self.active_futures)
                    self.metrics.current_load = active_tasks / self.max_workers
                
                # Log warnings for high resource usage
                memory_mb = memory_info.rss / 1024 / 1024
                if memory_mb > 512:  # 512MB warning threshold
                    self.logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                
                if self.metrics.current_load > 0.8:  # 80% load warning
                    self.logger.warning(f"High executor load: {self.metrics.current_load:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
            
            time.sleep(10)  # Monitor every 10 seconds
    
    def _cleanup_completed_tasks(self) -> None:
        """Background thread for cleaning up completed tasks."""
        while not self._shutdown_event.is_set():
            try:
                with self._lock:
                    # Clean up old thread info
                    current_time = datetime.now()
                    threads_to_remove = []
                    
                    for thread_id, info in self.thread_info.items():
                        if (not info.is_busy and info.last_activity and
                            (current_time - info.last_activity).total_seconds() > 300):
                            threads_to_remove.append(thread_id)
                    
                    for thread_id in threads_to_remove:
                        del self.thread_info[thread_id]
                    
                    self.metrics.last_cleanup = current_time
                
            except Exception as e:
                self.logger.error(f"Error in cleanup: {e}")
            
            time.sleep(self.cleanup_interval)
    
    def get_queue_size(self) -> int:
        """Get current execution queue size."""
        return self.execution_queue.qsize()
    
    def get_active_task_count(self) -> int:
        """Get number of currently active tasks."""
        with self._lock:
            return len(self.active_futures)
    
    def get_metrics(self) -> ExecutorMetrics:
        """Get executor performance metrics."""
        with self._lock:
            return self.metrics
    
    def get_thread_info(self) -> Dict[str, ThreadInfo]:
        """Get information about executor threads."""
        with self._lock:
            return self.thread_info.copy()
    
    def pause(self) -> None:
        """Pause task execution."""
        self._pause_event.set()
        self.logger.info("Executor paused")
    
    def resume(self) -> None:
        """Resume task execution."""
        self._pause_event.clear()
        self.logger.info("Executor resumed")
    
    def is_paused(self) -> bool:
        """Check if executor is paused."""
        return self._pause_event.is_set()
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled, False if not found
        """
        with self._lock:
            if task_id in self.active_futures:
                future = self.active_futures[task_id]
                success = future.cancel()
                if success:
                    del self.active_futures[task_id]
                self.logger.info(f"Task {task_id} cancellation: {success}")
                return success
        return False
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> List[str]:
        """
        Wait for all active tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            List of task IDs that completed within timeout
        """
        completed_tasks = []
        
        with self._lock:
            active_futures = list(self.active_futures.items())
        
        if not active_futures:
            return completed_tasks
        
        try:
            for task_id, future in as_completed(
                [f for _, f in active_futures], 
                timeout=timeout
            ):
                # Find task_id for this future
                for tid, f in active_futures:
                    if f is future:
                        completed_tasks.append(tid)
                        break
        except TimeoutError:
            self.logger.warning(f"Timeout waiting for task completion")
        
        return completed_tasks
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive status summary.
        
        Returns:
            Dictionary with status information
        """
        with self._lock:
            return {
                'max_workers': self.max_workers,
                'active_tasks': len(self.active_futures),
                'queue_size': self.execution_queue.qsize(),
                'is_paused': self._pause_event.is_set(),
                'is_shutdown': self._shutdown_event.is_set(),
                'thread_count': len(self.thread_info),
                'metrics': {
                    'tasks_executed': self.metrics.tasks_executed,
                    'tasks_failed': self.metrics.tasks_failed,
                    'tasks_timed_out': self.metrics.tasks_timed_out,
                    'current_load': self.metrics.current_load,
                    'peak_memory_mb': self.metrics.peak_memory_usage / 1024 / 1024,
                    'peak_threads': self.metrics.peak_thread_count
                }
            }
    
    def shutdown(self, timeout: float = 30.0) -> None:
        """
        Shutdown the executor and wait for tasks to complete.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        self.logger.info("Initiating executor shutdown...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel all active tasks
        with self._lock:
            active_task_ids = list(self.active_futures.keys())
        
        for task_id in active_task_ids:
            self.cancel_task(task_id)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True, timeout=timeout)
        
        # Wait for background threads
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        self.logger.info("Executor shutdown complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ModuleExecutor(max_workers={self.max_workers}, "
                f"active_tasks={len(self.active_futures)}, "
                f"queue_size={self.execution_queue.qsize()})")