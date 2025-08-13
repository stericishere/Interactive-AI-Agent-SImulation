# Concurrent Module Framework for Enhanced PIANO Architecture

The Concurrent Module Framework extends the existing PIANO (Perception, Introspection, Action, Navigation, Organization) architecture with advanced concurrent processing capabilities, enabling multiple cognitive modules to execute simultaneously while maintaining thread safety and state consistency.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ConcurrentModuleManager                       │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐ │
│  │  ModuleExecutor │  TaskScheduler  │    ResourceCoordinator  │ │  
│  │                 │                 │                         │ │
│  │  • Thread Pool  │  • Priority     │    • Memory Locks       │ │
│  │  • Task Exec    │  • Dependencies │    • Deadlock Detect   │ │
│  │  • Monitoring   │  • Fair Share   │    • Reader/Writer      │ │
│  └─────────────────┴─────────────────┴─────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              StateCoordinator                               │ │
│  │  • Cross-Module Sync  • Conflict Resolution  • Snapshots   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼───────┐     ┌────────▼────────┐     ┌───────▼───────┐
│   Enhanced    │     │   Concurrent    │     │  Distributed  │
│  Perception   │     │   Planning      │     │  Reflection   │
│               │     │                 │     │               │
│ • Parallel    │     │ • Multi-Goal    │     │ • Background  │
│ • Attention   │     │ • State Coord   │     │ • Insight Gen │
│ • Caching     │     │ • Scheduling    │     │ • Pattern Rec │
└───────────────┘     └─────────────────┘     └───────────────┘
```

## 🚀 Quick Start

### Basic Usage

```python
from concurrent_framework import ConcurrentModuleManager, Priority
from modules.base_module import BaseModule

# Initialize the concurrent framework
manager = ConcurrentModuleManager(
    agent_state=your_agent_state,
    max_workers=4,
    max_queue_size=20
)

# Register your modules
class MyModule(BaseModule):
    def run(self):
        return "Module executed successfully"

my_module = MyModule(agent_state)
manager.register_module("my_module", my_module)

# Submit concurrent tasks
task_id = manager.submit_task(
    module_name="my_module",
    method_name="run", 
    priority=Priority.HIGH
)

# Check task status
status = manager.get_task_status(task_id)
```

### Enhanced Perception with Parallel Processing

```python
from concurrent_framework.enhanced_modules import ParallelPerceptionModule

# Create enhanced perception module
perception = ParallelPerceptionModule(
    agent_state=agent_state,
    max_workers=3,
    attention_bandwidth=5,
    vision_radius=4
)

# Register and use
manager.register_module("perception", perception)
task_id = manager.submit_task("perception", "run", priority=Priority.CRITICAL)
```

## 📚 Core Components

### 1. ConcurrentModuleManager

The central coordination hub that manages all concurrent module execution.

**Key Features:**
- Module registration and lifecycle management
- Task submission with priority handling
- Resource coordination across modules
- Event callbacks and monitoring
- Graceful shutdown and error handling

**API Reference:**

```python
manager = ConcurrentModuleManager(
    agent_state,              # Shared agent state object
    max_workers=4,            # Maximum concurrent threads  
    max_queue_size=100,       # Task queue capacity
    enable_monitoring=True,   # Performance monitoring
    task_timeout=30.0         # Default task timeout (seconds)
)

# Module Management
manager.register_module(name, module, config=None)
manager.unregister_module(name)

# Task Submission
task_id = manager.submit_task(
    module_name,              # Registered module name
    method_name="run",        # Method to execute
    args=(),                  # Positional arguments
    kwargs={},                # Keyword arguments
    priority=Priority.NORMAL, # Task priority
    dependencies=set(),       # Dependency task IDs
    callback=None            # Completion callback
)

# Status and Control
status = manager.get_task_status(task_id)
manager.cancel_task(task_id)
manager.pause_module(module_name)
manager.resume_module(module_name)
system_status = manager.get_system_status()
```

### 2. ModuleExecutor

Manages thread pool execution and resource monitoring.

**Key Features:**
- Configurable thread pool with resource limits
- Task timeout handling and cancellation
- Memory and CPU usage monitoring
- Thread safety and error isolation

**Configuration:**

```python
executor = ModuleExecutor(
    max_workers=4,            # Thread pool size
    max_queue_size=100,       # Execution queue capacity
    enable_monitoring=True,   # Resource monitoring
    cleanup_interval=300      # Cleanup interval (seconds)
)

# Metrics
metrics = executor.get_metrics()
print(f"Tasks executed: {metrics.tasks_executed}")
print(f"Average execution time: {metrics.average_execution_time}")
print(f"Peak memory usage: {metrics.peak_memory_usage}")
```

### 3. TaskScheduler

Intelligent task scheduling with multiple policies and dependency management.

**Scheduling Policies:**
- `PRIORITY_FIRST`: Highest priority tasks first
- `ROUND_ROBIN`: Fair rotation among modules
- `FAIR_SHARE`: Resource quotas per module
- `WEIGHTED_PRIORITY`: Priority with fairness adjustment

**Usage:**

```python
from concurrent_framework import TaskScheduler, SchedulingPolicy

scheduler = TaskScheduler(
    scheduling_policy=SchedulingPolicy.WEIGHTED_PRIORITY,
    max_pending_tasks=1000,
    enable_fairness=True
)

# Set module quotas for fair scheduling
scheduler.set_module_quota("perception", 3)
scheduler.set_module_quota("planning", 2) 
scheduler.set_module_quota("retrieval", 1)

# Task scheduling with dependencies
task1_id = scheduler.schedule_task(task1)
task2_id = scheduler.schedule_task(task2, dependencies={task1_id})
```

### 4. ResourceCoordinator

Thread-safe resource access with deadlock prevention.

**Resource Types:**
- `EPISODIC_MEMORY`: Episodic memory system
- `SEMANTIC_MEMORY`: Semantic memory system  
- `TEMPORAL_MEMORY`: Temporal memory system
- `CIRCULAR_BUFFER`: Working memory buffer
- `AGENT_STATE`: Agent state object

**Access Modes:**
- `READ`: Shared read access
- `WRITE`: Exclusive write access
- `EXCLUSIVE`: Complete exclusive access

**Usage:**

```python
from concurrent_framework import ResourceCoordinator, ResourceType, AccessMode

coordinator = ResourceCoordinator(
    max_concurrent_locks=10,
    default_timeout=30.0,
    enable_deadlock_detection=True
)

# Register resources
coordinator.register_resource(
    "episodic_memory", 
    agent_state.episodic_memory,
    ResourceType.EPISODIC_MEMORY
)

# Acquire resource locks
lock_id = coordinator.acquire_resource(
    "episodic_memory",
    AccessMode.WRITE,
    "my_module_id"
)

# Use context manager (recommended)
with coordinator.acquire_context(
    "episodic_memory", AccessMode.READ, "my_module"
) as memory:
    # Safe concurrent access
    episodes = memory.get_recent_episodes()

# Resource status
status = coordinator.get_resource_status("episodic_memory")
```

### 5. StateCoordinator

Cross-module state synchronization with conflict resolution.

**Sync Policies:**
- `IMMEDIATE`: Apply changes immediately
- `BATCHED`: Batch changes together
- `PERIODIC`: Apply changes periodically
- `ON_DEMAND`: Manual application

**Usage:**

```python
from concurrent_framework import StateCoordinator, SyncPolicy, StateChangeType

coordinator = StateCoordinator(
    agent_state=agent_state,
    sync_policy=SyncPolicy.IMMEDIATE
)

# Register modules
coordinator.register_module("module1", module1)
coordinator.register_module("module2", module2)

# Subscribe to state changes
coordinator.subscribe_to_path("module2", "emotional_state.happiness")

# Propose state changes
change_id = coordinator.propose_change(
    module_id="module1",
    path="emotional_state.happiness",
    new_value=0.8,
    change_type=StateChangeType.UPDATE,
    priority=1
)

# Transaction support
coordinator.begin_transaction("my_transaction")
coordinator.propose_change("module1", "goals", new_goals)
coordinator.commit_transaction("my_transaction")

# Snapshots and rollback
snapshot_id = coordinator.create_snapshot("checkpoint")
coordinator.restore_snapshot("checkpoint")
```

## 🧠 Enhanced Cognitive Modules

### ParallelPerceptionModule

Advanced perception with concurrent sensory processing.

**Features:**
- Parallel processing of multiple sensor inputs
- Attention-based filtering and prioritization
- Intelligent caching with TTL
- Performance metrics and monitoring

**Sensor Types:**
- `VISUAL`: Spatial and object perception
- `SOCIAL`: Agent interaction detection
- `ENVIRONMENTAL`: Event analysis
- `TEMPORAL`: Time-based context
- `SPATIAL`: Navigation and mapping
- `AUDITORY`: Sound processing (future)

**Usage:**

```python
from concurrent_framework.enhanced_modules import ParallelPerceptionModule

perception = ParallelPerceptionModule(
    agent_state=agent_state,
    max_workers=3,              # Concurrent processing threads
    attention_bandwidth=5,      # Max simultaneous perceptions
    vision_radius=4,           # Spatial perception range
    enable_caching=True        # Result caching
)

# Automatic parallel processing
results = perception.run()

# Results contain processed data from multiple sensors
for result in results:
    print(f"Sensor: {result.sensor_type}")
    print(f"Confidence: {result.confidence}")
    print(f"Data: {result.processed_data}")

# Performance metrics
metrics = perception.get_perception_metrics()
print(f"Total tasks: {metrics.total_tasks_processed}")
print(f"Avg processing time: {metrics.average_processing_time}")
```

## 🔧 Configuration and Tuning

### Performance Tuning

```python
# For CPU-intensive workloads
manager = ConcurrentModuleManager(
    agent_state=agent_state,
    max_workers=8,              # More threads for CPU work
    max_queue_size=200,         # Larger queue
    task_timeout=60.0           # Longer timeout
)

# For memory-constrained environments  
manager = ConcurrentModuleManager(
    agent_state=agent_state,
    max_workers=2,              # Fewer threads
    max_queue_size=50,          # Smaller queue
    enable_monitoring=False     # Reduce overhead
)
```

### Resource Limits

```python
# Memory system limits
episodic_mem = EpisodicMemory(max_episodes=200)  # Larger capacity
semantic_mem = SemanticMemory(max_concepts=1000)
temporal_mem = TemporalMemory(retention_hours=4)

# Resource coordinator limits
coordinator = ResourceCoordinator(
    max_concurrent_locks=20,    # More concurrent access
    default_timeout=45.0,       # Longer wait times
    enable_deadlock_detection=True
)
```

### Scheduling Configuration

```python
# Fair scheduling with quotas
scheduler = TaskScheduler(
    scheduling_policy=SchedulingPolicy.FAIR_SHARE,
    enable_fairness=True
)

scheduler.set_module_quota("perception", 4)    # High priority
scheduler.set_module_quota("planning", 2)      # Medium priority  
scheduler.set_module_quota("reflection", 1)    # Background
```

## 📊 Monitoring and Metrics

### System Metrics

```python
# Overall system status
status = manager.get_system_status()
print(f"Active tasks: {status['active_tasks']}")
print(f"Completed tasks: {status['completed_tasks']}")  
print(f"Uptime: {status['uptime_seconds']}s")

# Executor metrics
exec_metrics = manager.module_executor.get_metrics()
print(f"Tasks executed: {exec_metrics.tasks_executed}")
print(f"Failure rate: {exec_metrics.tasks_failed / exec_metrics.tasks_executed}")
print(f"Peak memory: {exec_metrics.peak_memory_usage / 1024 / 1024:.1f} MB")

# Resource metrics
resource_metrics = manager.resource_coordinator.get_coordinator_metrics()
print(f"Lock contentions: {resource_metrics.contention_count}")
print(f"Average wait time: {resource_metrics.average_wait_time:.3f}s")

# State coordination metrics
state_metrics = manager.state_coordinator.get_state_metrics()
print(f"State changes: {state_metrics.total_changes}")
print(f"Conflicts resolved: {state_metrics.conflicts_resolved}")
```

### Performance Monitoring

```python
# Set up monitoring callbacks
def on_task_completed(task_id, success):
    if not success:
        print(f"Task {task_id} failed")

def on_system_overload():
    print("System overload detected!")

manager.add_event_callback('task_completed', on_task_completed)
manager.add_event_callback('system_overload', on_system_overload)
```

## 🔒 Security and Safety

### Input Validation

All inputs are automatically validated using the security utilities:

```python
# Content sanitization
safe_content = SecurityValidator.sanitize_memory_data(content, type, importance, metadata)

# Path validation for state changes
SecurityValidator.validate_state_path("agent.emotional_state.happiness")

# Execution argument validation
SecurityValidator.validate_execution_args(args, kwargs)
```

### Resource Protection

```python
# Resource access is automatically managed
with coordinator.acquire_context(resource_id, AccessMode.WRITE, holder_id) as resource:
    # Thread-safe access guaranteed
    # Automatic deadlock prevention
    # Timeout protection
    resource.modify_data()
```

### Error Handling

```python
try:
    task_id = manager.submit_task("module", "method")
except ValueError as e:
    print(f"Invalid module or configuration: {e}")
except RuntimeError as e:
    print(f"System error: {e}")

# Graceful degradation
if not task_id:
    # Fallback to synchronous execution
    result = module.method()
```

## 🧪 Testing

### Unit Tests

```python
# Run comprehensive test suite
python -m concurrent_framework.tests.test_concurrent_framework

# Run specific test class
python -m unittest concurrent_framework.tests.test_concurrent_framework.TestConcurrentModuleManager
```

### Integration Testing

```python
# Run integration example
python concurrent_framework/examples/piano_integration_example.py

# Expected output:
# 🎹 PIANO Concurrent Framework Integration Demo
# ============================================================
# ✓ All memory systems registered for coordinated access
# ✓ Perception Module - parallel sensory processing
# ✓ Planning Module - state-coordinated goal management
# ✓ Retrieval Module - concurrent memory access
```

## 🚦 Best Practices

### Module Design

1. **Extend BaseModule**: Always inherit from the base module class
2. **Stateless Operations**: Keep module methods stateless when possible
3. **Resource Cleanup**: Implement proper cleanup in module destructors
4. **Error Handling**: Handle exceptions gracefully

```python
class MyModule(BaseModule):
    def __init__(self, agent_state):
        super().__init__(agent_state)
        self.cleanup_needed = []
    
    def run(self):
        try:
            # Your module logic here
            result = self.process_data()
            return result
        except Exception as e:
            self.logger.error(f"Module error: {e}")
            raise
    
    def __del__(self):
        # Cleanup resources
        for resource in self.cleanup_needed:
            resource.close()
```

### Task Submission

1. **Use Appropriate Priorities**: Reserve HIGH/CRITICAL for urgent tasks
2. **Manage Dependencies**: Minimize dependency chains
3. **Handle Timeouts**: Set reasonable timeouts for long-running tasks
4. **Batch Related Tasks**: Submit related tasks together for better scheduling

```python
# Good: Appropriate priority usage
perception_task = manager.submit_task("perception", "run", priority=Priority.HIGH)
planning_task = manager.submit_task("planning", "run", priority=Priority.NORMAL)
reflection_task = manager.submit_task("reflection", "run", priority=Priority.LOW)

# Good: Proper dependency management
task1 = manager.submit_task("module1", "gather_data")
task2 = manager.submit_task("module2", "process_data", dependencies={task1})
```

### Resource Management

1. **Use Context Managers**: Always use `acquire_context()` when possible
2. **Minimize Lock Duration**: Keep resource locks as short as possible
3. **Choose Appropriate Access Modes**: Use READ when possible
4. **Register All Shared Resources**: Don't access memory directly

```python
# Good: Context manager with minimal lock duration
with coordinator.acquire_context("episodic_memory", AccessMode.READ, "my_module") as memory:
    relevant_episodes = memory.get_recent_episodes(hours_back=1)

# Process episodes outside the lock
processed_results = process_episodes(relevant_episodes)
```

### State Coordination

1. **Subscribe Judiciously**: Only subscribe to relevant state changes
2. **Use Transactions**: Group related state changes
3. **Handle Conflicts**: Implement conflict resolution strategies
4. **Create Snapshots**: Use snapshots before major state changes

```python
# Good: Transaction for related changes
coordinator.begin_transaction("goal_update")
try:
    coordinator.propose_change("module", "current_goal", new_goal)
    coordinator.propose_change("module", "goal_progress", 0.0)
    coordinator.commit_transaction("goal_update")
except Exception:
    coordinator.rollback_transaction("goal_update")
```

## 📋 Migration Guide

### From Legacy PIANO

1. **Replace Direct Memory Access**:
```python
# Before
persona.a_mem.add_event(content, importance)

# After  
with manager.resource_coordinator.acquire_context(
    "associative_memory", AccessMode.WRITE, "module_id"
) as memory:
    memory.add_event(content, importance)
```

2. **Convert Modules to Concurrent**:
```python
# Before
class PerceptionModule:
    def run(self):
        return self.perceive()

# After
class EnhancedPerceptionModule(BaseModule):
    def __init__(self, agent_state, concurrent_manager):
        super().__init__(agent_state)
        self.concurrent_manager = concurrent_manager
    
    def run(self):
        # Use concurrent framework features
        return self.parallel_perceive()
```

3. **Update Task Execution**:
```python
# Before
perception_result = perception_module.run()
planning_result = planning_module.run()

# After
perception_task = manager.submit_task("perception", "run", priority=Priority.HIGH)
planning_task = manager.submit_task("planning", "run", priority=Priority.NORMAL)
```

## 🔍 Troubleshooting

### Common Issues

1. **Task Timeouts**:
   - Increase task timeout in manager configuration
   - Check for blocking operations in module code
   - Verify resource availability

2. **Resource Deadlocks**:
   - Enable deadlock detection
   - Review resource acquisition order
   - Minimize lock duration

3. **Memory Issues**:
   - Monitor peak memory usage
   - Reduce max_workers if memory-constrained
   - Enable cleanup intervals

4. **Performance Degradation**:
   - Check thread pool utilization
   - Monitor task queue sizes
   - Verify fair scheduling configuration

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger('concurrent_framework').setLevel(logging.DEBUG)

# Or set environment variable
import os
os.environ['CONCURRENT_FRAMEWORK_LOG_LEVEL'] = 'DEBUG'
```

## 🛣️ Roadmap

### Week 2 Enhancements (Current Phase)

- ✅ Core concurrent infrastructure
- ✅ Resource coordination system  
- ✅ State synchronization
- ✅ Parallel perception module
- 🔄 Additional enhanced cognitive modules
- 🔄 Memory coordination system
- 🔄 Advanced scheduling algorithms

### Future Phases

- **Week 3**: Advanced cognitive architectures
- **Week 4**: Multi-agent coordination
- **Week 5**: Learning and adaptation
- **Week 6**: Performance optimization

## 📞 Support

For issues, questions, or contributions:

1. Check the test suite for usage examples
2. Review the integration example in `examples/piano_integration_example.py`
3. Examine module source code for detailed implementation
4. Run the comprehensive test suite to validate setup

---

**Built for enhanced cognitive processing in generative agent simulations** 🤖🧠