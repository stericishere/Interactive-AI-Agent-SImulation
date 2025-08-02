"""
File: langgraph_base_module.py
Description: Enhanced Base Module with LangGraph StateGraph integration for PIANO architecture Phase 1.
Provides StateGraph node interface for cognitive modules with concurrent execution patterns.
"""

from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future

# Import enhanced agent state
from ..enhanced_agent_state import EnhancedAgentState, EnhancedAgentStateManager


class ModulePriority(Enum):
    """Priority levels for module execution."""
    CRITICAL = "critical"    # Must complete: perception, safety checks
    HIGH = "high"           # Core functions: planning, execution
    MEDIUM = "medium"       # Important: social, specialization
    LOW = "low"            # Background: reflection, cleanup


class ExecutionTimeScale(Enum):
    """Execution time scales for different module types."""
    FAST = 100      # 100ms - perception, working memory
    MEDIUM = 500    # 500ms - planning, social interactions
    SLOW = 5000     # 5000ms - reflection, specialization


@dataclass
class ModuleExecutionConfig:
    """Configuration for module execution parameters."""
    time_scale: ExecutionTimeScale
    priority: ModulePriority
    can_run_parallel: bool = True
    requires_completion: bool = False  # Must complete before next cycle
    max_execution_time: float = 10.0  # Maximum execution time in seconds
    retry_count: int = 3
    fallback_enabled: bool = True


@dataclass
class ModuleExecutionResult:
    """Result of module execution."""
    module_name: str
    success: bool
    execution_time_ms: float
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    state_changes: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None


class LangGraphBaseModule(ABC):
    """
    Enhanced base class for cognitive modules with LangGraph StateGraph integration.
    Provides concurrent execution patterns and state management protocols.
    """
    
    def __init__(self, module_name: str, config: ModuleExecutionConfig,
                 state_manager: Optional[EnhancedAgentStateManager] = None):
        """
        Initialize LangGraph base module.
        
        Args:
            module_name: Name of the cognitive module
            config: Execution configuration
            state_manager: Enhanced agent state manager
        """
        self.module_name = module_name
        self.config = config
        self.state_manager = state_manager
        
        # Execution tracking
        self.last_execution_time = None
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0
        self.success_rate = 1.0
        
        # Performance metrics
        self.performance_history: List[float] = []
        self.max_history_size = 100
        
        # State access control
        self.state_locks: Set[str] = set()
        self.state_dependencies: Set[str] = set()
        
        # Logging
        self.logger = logging.getLogger(f"LangGraphModule.{module_name}")
        
        # Thread pool for concurrent operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"{module_name}")
    
    @abstractmethod
    def process_state(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Abstract method for processing agent state.
        
        Args:
            state: Current enhanced agent state
        
        Returns:
            Dictionary with state updates and outputs
        """
        pass
    
    def __call__(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """
        LangGraph node interface - called by StateGraph execution.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        execution_start = time.time()
        
        try:
            # Check if module should run based on time scale
            if not self._should_execute(state):
                return state
            
            # Validate state dependencies
            if not self._validate_dependencies(state):
                self.logger.warning(f"Dependencies not met for {self.module_name}")
                return state
            
            # Acquire state locks if needed
            if not self._acquire_locks(state):
                self.logger.warning(f"Could not acquire locks for {self.module_name}")
                return state
            
            try:
                # Execute the module processing
                result = self._execute_with_timeout(state)
                
                if result.success:
                    # Apply state changes
                    updated_state = self._apply_state_changes(state, result.state_changes)
                    
                    # Update performance metrics
                    self._update_performance_metrics(result)
                    
                    self.logger.debug(f"{self.module_name} executed successfully in {result.execution_time_ms:.2f}ms")
                    return updated_state
                else:
                    self.error_count += 1
                    self.logger.error(f"{self.module_name} execution failed: {result.error_message}")
                    return state
            
            finally:
                # Always release locks
                self._release_locks()
        
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Unexpected error in {self.module_name}: {str(e)}")
            return state
        
        finally:
            # Update execution statistics
            execution_time = (time.time() - execution_start) * 1000  # Convert to ms
            self.execution_count += 1
            self.total_execution_time += execution_time
            self.last_execution_time = datetime.now()
    
    def _should_execute(self, state: EnhancedAgentState) -> bool:
        """
        Determine if module should execute based on time scale and state conditions.
        
        Args:
            state: Current agent state
        
        Returns:
            True if module should execute
        """
        if not self.last_execution_time:
            return True
        
        # Calculate time since last execution
        time_since_last = (datetime.now() - self.last_execution_time).total_seconds() * 1000
        
        # Check if enough time has passed based on time scale
        if time_since_last < self.config.time_scale.value:
            return False
        
        # Additional state-based conditions can be added here
        return True
    
    def _validate_dependencies(self, state: EnhancedAgentState) -> bool:
        """
        Validate that state dependencies are met.
        
        Args:
            state: Current agent state
        
        Returns:
            True if dependencies are satisfied
        """
        for dependency in self.state_dependencies:
            if not self._check_dependency(state, dependency):
                return False
        return True
    
    def _check_dependency(self, state: EnhancedAgentState, dependency: str) -> bool:
        """
        Check if a specific dependency is satisfied.
        
        Args:
            state: Current agent state
            dependency: Dependency key to check
        
        Returns:
            True if dependency is satisfied
        """
        # Basic dependency checking - can be extended
        return dependency in state and state[dependency] is not None
    
    def _acquire_locks(self, state: EnhancedAgentState) -> bool:
        """
        Acquire state locks for thread-safe execution.
        
        Args:
            state: Current agent state
        
        Returns:
            True if all locks acquired successfully
        """
        # Simple implementation - in production would use actual locking mechanism
        # For now, just track what we're locking
        return True
    
    def _release_locks(self) -> None:
        """Release all acquired state locks."""
        self.state_locks.clear()
    
    def _execute_with_timeout(self, state: EnhancedAgentState) -> ModuleExecutionResult:
        """
        Execute module processing with timeout protection.
        
        Args:
            state: Current agent state
        
        Returns:
            Module execution result
        """
        start_time = time.time()
        
        try:
            # Submit task to executor with timeout
            future = self._executor.submit(self._safe_process_state, state)
            
            try:
                result = future.result(timeout=self.config.max_execution_time)
                execution_time = (time.time() - start_time) * 1000
                
                return ModuleExecutionResult(
                    module_name=self.module_name,
                    success=True,
                    execution_time_ms=execution_time,
                    output_data=result.get("output_data"),
                    state_changes=result.get("state_changes"),
                    performance_metrics=result.get("performance_metrics")
                )
            
            except asyncio.TimeoutError:
                future.cancel()
                return ModuleExecutionResult(
                    module_name=self.module_name,
                    success=False,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    error_message=f"Execution timeout after {self.config.max_execution_time}s"
                )
        
        except Exception as e:
            return ModuleExecutionResult(
                module_name=self.module_name,
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def _safe_process_state(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Safely execute the process_state method with error handling.
        
        Args:
            state: Current agent state
        
        Returns:
            Processing result dictionary
        """
        try:
            return self.process_state(state)
        except Exception as e:
            self.logger.error(f"Error in {self.module_name}.process_state: {str(e)}")
            raise
    
    def _apply_state_changes(self, state: EnhancedAgentState, 
                           state_changes: Optional[Dict[str, Any]]) -> EnhancedAgentState:
        """
        Apply state changes to the agent state.
        
        Args:
            state: Current agent state
            state_changes: Dictionary of changes to apply
        
        Returns:
            Updated agent state
        """
        if not state_changes:
            return state
        
        # Create a copy of the state for modification
        updated_state = state.copy()
        
        # Apply changes
        for key, value in state_changes.items():
            if key in updated_state:
                updated_state[key] = value
            else:
                self.logger.warning(f"Attempting to set unknown state key: {key}")
        
        return updated_state
    
    def _update_performance_metrics(self, result: ModuleExecutionResult) -> None:
        """
        Update performance tracking metrics.
        
        Args:
            result: Execution result with timing information
        """
        # Track execution time history
        self.performance_history.append(result.execution_time_ms)
        if len(self.performance_history) > self.max_history_size:
            self.performance_history.pop(0)
        
        # Update success rate
        total_executions = self.execution_count + 1
        success_count = total_executions - self.error_count
        self.success_rate = success_count / total_executions if total_executions > 0 else 1.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for this module.
        
        Returns:
            Performance statistics dictionary
        """
        if not self.performance_history:
            return {
                "module_name": self.module_name,
                "execution_count": 0,
                "avg_execution_time_ms": 0.0,
                "success_rate": 1.0,
                "last_execution": None
            }
        
        return {
            "module_name": self.module_name,
            "execution_count": self.execution_count,
            "avg_execution_time_ms": sum(self.performance_history) / len(self.performance_history),
            "min_execution_time_ms": min(self.performance_history),
            "max_execution_time_ms": max(self.performance_history),
            "success_rate": self.success_rate,
            "error_count": self.error_count,
            "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "time_scale": self.config.time_scale.value,
            "priority": self.config.priority.value
        }
    
    def add_state_dependency(self, dependency: str) -> None:
        """
        Add a state dependency for this module.
        
        Args:
            dependency: State key that must be present
        """
        self.state_dependencies.add(dependency)
    
    def remove_state_dependency(self, dependency: str) -> None:
        """
        Remove a state dependency.
        
        Args:
            dependency: State key to remove from dependencies
        """
        self.state_dependencies.discard(dependency)
    
    def reset_performance_metrics(self) -> None:
        """Reset all performance tracking metrics."""
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0
        self.success_rate = 1.0
        self.performance_history.clear()
        self.last_execution_time = None
    
    def shutdown(self) -> None:
        """Shutdown the module and cleanup resources."""
        self._executor.shutdown(wait=True)
        self.logger.info(f"Module {self.module_name} shutdown complete")


class ModuleCoordinator:
    """
    Coordinates execution of multiple LangGraph modules with dependency management.
    """
    
    def __init__(self):
        """Initialize module coordinator."""
        self.modules: Dict[str, LangGraphBaseModule] = {}
        self.module_dependencies: Dict[str, Set[str]] = {}
        self.execution_order: List[str] = []
        self.parallel_groups: List[Set[str]] = []
        
        self.logger = logging.getLogger("ModuleCoordinator")
    
    def register_module(self, module: LangGraphBaseModule, 
                       dependencies: Optional[List[str]] = None) -> None:
        """
        Register a module with the coordinator.
        
        Args:
            module: Module to register
            dependencies: List of module names this module depends on
        """
        self.modules[module.module_name] = module
        self.module_dependencies[module.module_name] = set(dependencies or [])
        
        # Recalculate execution order
        self._calculate_execution_order()
    
    def _calculate_execution_order(self) -> None:
        """Calculate optimal execution order based on dependencies and parallelization."""
        # Topological sort for dependency order
        visited = set()
        temp_visited = set()
        self.execution_order = []
        
        def visit(module_name: str):
            if module_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {module_name}")
            if module_name in visited:
                return
            
            temp_visited.add(module_name)
            for dependency in self.module_dependencies.get(module_name, set()):
                if dependency in self.modules:
                    visit(dependency)
            
            temp_visited.remove(module_name)
            visited.add(module_name)
            self.execution_order.append(module_name)
        
        for module_name in self.modules:
            if module_name not in visited:
                visit(module_name)
        
        # Calculate parallel execution groups
        self._calculate_parallel_groups()
    
    def _calculate_parallel_groups(self) -> None:
        """Calculate which modules can execute in parallel."""
        self.parallel_groups = []
        remaining_modules = set(self.execution_order)
        
        while remaining_modules:
            # Find modules with no dependencies in remaining set
            parallel_group = set()
            
            for module_name in list(remaining_modules):
                dependencies = self.module_dependencies.get(module_name, set())
                if not (dependencies & remaining_modules):  # No dependencies in remaining modules
                    parallel_group.add(module_name)
                    remaining_modules.remove(module_name)
            
            if parallel_group:
                self.parallel_groups.append(parallel_group)
            else:
                # Break circular dependency by taking one module
                module_name = remaining_modules.pop()
                self.parallel_groups.append({module_name})
    
    def get_execution_plan(self) -> Dict[str, Any]:
        """
        Get the current execution plan.
        
        Returns:
            Dictionary describing the execution plan
        """
        return {
            "execution_order": self.execution_order,
            "parallel_groups": [list(group) for group in self.parallel_groups],
            "total_modules": len(self.modules),
            "dependencies": {k: list(v) for k, v in self.module_dependencies.items()}
        }
    
    def get_coordinator_summary(self) -> Dict[str, Any]:
        """
        Get summary of all registered modules.
        
        Returns:
            Coordinator summary with module performance
        """
        return {
            "total_modules": len(self.modules),
            "execution_plan": self.get_execution_plan(),
            "module_performance": {
                name: module.get_performance_summary() 
                for name, module in self.modules.items()
            }
        }


# Helper functions for creating common module configurations

def create_perception_config() -> ModuleExecutionConfig:
    """Create configuration for perception modules."""
    return ModuleExecutionConfig(
        time_scale=ExecutionTimeScale.FAST,
        priority=ModulePriority.CRITICAL,
        can_run_parallel=True,
        requires_completion=True,
        max_execution_time=0.5
    )

def create_planning_config() -> ModuleExecutionConfig:
    """Create configuration for planning modules."""
    return ModuleExecutionConfig(
        time_scale=ExecutionTimeScale.MEDIUM,
        priority=ModulePriority.HIGH,
        can_run_parallel=False,
        requires_completion=True,
        max_execution_time=2.0
    )

def create_social_config() -> ModuleExecutionConfig:
    """Create configuration for social modules."""
    return ModuleExecutionConfig(
        time_scale=ExecutionTimeScale.MEDIUM,
        priority=ModulePriority.MEDIUM,
        can_run_parallel=True,
        requires_completion=False,
        max_execution_time=1.5
    )

def create_reflection_config() -> ModuleExecutionConfig:
    """Create configuration for reflection modules."""
    return ModuleExecutionConfig(
        time_scale=ExecutionTimeScale.SLOW,
        priority=ModulePriority.LOW,
        can_run_parallel=True,
        requires_completion=False,
        max_execution_time=5.0
    )


# Example usage and testing
if __name__ == "__main__":
    # Example implementation of a concrete module
    class ExamplePerceptionModule(LangGraphBaseModule):
        def __init__(self):
            config = create_perception_config()
            super().__init__("perception", config)
        
        def process_state(self, state: EnhancedAgentState) -> Dict[str, Any]:
            # Example perception processing
            current_time = datetime.now()
            
            return {
                "state_changes": {
                    "current_time": current_time,
                    "current_activity": "perceiving environment"
                },
                "output_data": {
                    "perception_result": "environment scanned"
                },
                "performance_metrics": {
                    "objects_detected": 5,
                    "confidence_score": 0.85
                }
            }
    
    # Example usage
    print("LangGraph Base Module Example")
    perception_module = ExamplePerceptionModule()
    
    # Create mock state
    from ..enhanced_agent_state import create_enhanced_agent_state
    
    state_manager = create_enhanced_agent_state(
        "test_agent", "Test Agent", {"confidence": 0.8}
    )
    
    # Execute module
    updated_state = perception_module(state_manager.state)
    
    print("Performance Summary:", perception_module.get_performance_summary())
    print("Module executed successfully!")