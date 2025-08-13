"""
File: test_concurrent_framework.py
Description: Comprehensive test suite for concurrent module framework
Enhanced PIANO architecture testing
"""

import unittest
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from concurrent.futures import Future

# Import the modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent_module_manager import (
    ConcurrentModuleManager, ModuleState, Priority, ModuleTask
)
from module_executor import ModuleExecutor
from task_scheduler import TaskScheduler, SchedulingPolicy
from resource_coordinator import ResourceCoordinator, ResourceType, AccessMode
from state_coordinator import StateCoordinator, SyncPolicy, StateChangeType
from enhanced_modules.parallel_perception_module import ParallelPerceptionModule, SensorType


class MockModule:
    """Mock module for testing."""
    
    def __init__(self, agent_state=None, processing_time=0.1):
        self.agent_state = agent_state
        self.processing_time = processing_time
        self.run_count = 0
        self.last_args = None
        self.last_kwargs = None
    
    def run(self, *args, **kwargs):
        """Mock run method."""
        self.run_count += 1
        self.last_args = args
        self.last_kwargs = kwargs
        time.sleep(self.processing_time)
        return f"result_{self.run_count}"
    
    def process_data(self, data):
        """Mock data processing method."""
        time.sleep(self.processing_time)
        return f"processed_{data}"
    
    def on_state_change(self, changes):
        """Mock state change handler."""
        pass


class MockAgentState:
    """Mock agent state for testing."""
    
    def __init__(self):
        self.test_value = "initial"
        self.counter = 0
        self.nested = {"value": 42}


class TestConcurrentModuleManager(unittest.TestCase):
    """Test suite for ConcurrentModuleManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent_state = MockAgentState()
        self.manager = ConcurrentModuleManager(
            agent_state=self.agent_state,
            max_workers=2,
            max_queue_size=10
        )
        self.mock_module = MockModule(self.agent_state)
    
    def tearDown(self):
        """Clean up after tests."""
        self.manager.shutdown(timeout=5.0)
    
    def test_module_registration(self):
        """Test module registration and unregistration."""
        # Test registration
        self.manager.register_module("test_module", self.mock_module)
        self.assertIn("test_module", self.manager.modules)
        self.assertEqual(self.manager.module_states["test_module"], ModuleState.IDLE)
        
        # Test duplicate registration
        with self.assertRaises(ValueError):
            self.manager.register_module("test_module", self.mock_module)
        
        # Test unregistration
        self.manager.unregister_module("test_module")
        self.assertNotIn("test_module", self.manager.modules)
    
    def test_task_submission(self):
        """Test task submission and execution."""
        self.manager.register_module("test_module", self.mock_module)
        
        # Submit a task
        task_id = self.manager.submit_task(
            module_name="test_module",
            method_name="run",
            priority=Priority.HIGH
        )
        
        self.assertIsInstance(task_id, str)
        self.assertIn(task_id, self.manager.active_tasks)
        
        # Wait for task completion
        time.sleep(1.0)
        
        # Check task was executed
        self.assertEqual(self.mock_module.run_count, 1)
    
    def test_task_with_dependencies(self):
        """Test task execution with dependencies."""
        self.manager.register_module("test_module", self.mock_module)
        
        # Submit first task
        task1_id = self.manager.submit_task("test_module", "run")
        
        # Submit dependent task
        task2_id = self.manager.submit_task(
            "test_module", "run", 
            dependencies={task1_id}
        )
        
        # Wait for completion
        time.sleep(2.0)
        
        # Both tasks should have executed
        self.assertGreaterEqual(self.mock_module.run_count, 2)
    
    def test_task_cancellation(self):
        """Test task cancellation."""
        self.manager.register_module("test_module", MockModule(processing_time=2.0))
        
        # Submit a long-running task
        task_id = self.manager.submit_task("test_module", "run")
        
        # Cancel the task
        success = self.manager.cancel_task(task_id)
        
        # Task should be cancelled
        self.assertTrue(success)
        self.assertNotIn(task_id, self.manager.active_tasks)
    
    def test_system_status(self):
        """Test system status reporting."""
        self.manager.register_module("test_module", self.mock_module)
        
        status = self.manager.get_system_status()
        
        self.assertIn('registered_modules', status)
        self.assertIn('active_tasks', status)
        self.assertIn('is_paused', status)
        self.assertEqual(status['registered_modules'], 1)


class TestModuleExecutor(unittest.TestCase):
    """Test suite for ModuleExecutor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.executor = ModuleExecutor(max_workers=2)
        self.mock_module = MockModule()
    
    def tearDown(self):
        """Clean up after tests."""
        self.executor.shutdown()
    
    def test_task_execution(self):
        """Test basic task execution."""
        task = ModuleTask(
            task_id="test_task",
            module_name="test_module",
            method_name="run"
        )
        
        future = self.executor.execute_task(task, self.mock_module)
        result = future.result(timeout=5.0)
        
        self.assertEqual(result, "result_1")
        self.assertEqual(self.mock_module.run_count, 1)
    
    def test_concurrent_execution(self):
        """Test concurrent task execution."""
        tasks = []
        futures = []
        
        for i in range(4):
            task = ModuleTask(
                task_id=f"task_{i}",
                module_name="test_module",
                method_name="run"
            )
            tasks.append(task)
        
        # Submit all tasks
        for task in tasks:
            future = self.executor.execute_task(task, MockModule())
            futures.append(future)
        
        # Wait for all to complete
        results = [f.result(timeout=5.0) for f in futures]
        
        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.startswith("result_") for r in results))
    
    def test_timeout_handling(self):
        """Test task timeout handling."""
        task = ModuleTask(
            task_id="timeout_task",
            module_name="test_module", 
            method_name="run",
            max_duration=0.1  # Very short timeout
        )
        
        slow_module = MockModule(processing_time=1.0)  # Slow processing
        
        future = self.executor.execute_task(task, slow_module)
        
        with self.assertRaises(Exception):  # Should timeout
            future.result(timeout=2.0)
    
    def test_resource_monitoring(self):
        """Test resource monitoring."""
        metrics = self.executor.get_metrics()
        
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics.tasks_executed, 0)
        self.assertGreaterEqual(metrics.peak_memory_usage, 0)
    
    def test_pause_resume(self):
        """Test executor pause and resume."""
        self.assertFalse(self.executor.is_paused())
        
        # Pause executor
        self.executor.pause()
        self.assertTrue(self.executor.is_paused())
        
        # Resume executor
        self.executor.resume()
        self.assertFalse(self.executor.is_paused())


class TestTaskScheduler(unittest.TestCase):
    """Test suite for TaskScheduler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = TaskScheduler(
            scheduling_policy=SchedulingPolicy.PRIORITY_FIRST,
            max_pending_tasks=10
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.scheduler.shutdown()
    
    def test_task_scheduling(self):
        """Test basic task scheduling."""
        task = ModuleTask(
            task_id="test_task",
            module_name="test_module",
            method_name="run",
            priority=Priority.HIGH
        )
        
        success = self.scheduler.schedule_task(task)
        self.assertTrue(success)
        
        # Get next task
        next_task = self.scheduler.get_next_task()
        self.assertIsNotNone(next_task)
        self.assertEqual(next_task.task_id, "test_task")
    
    def test_priority_ordering(self):
        """Test priority-based task ordering."""
        # Create tasks with different priorities
        high_task = ModuleTask("high", "module", "run", priority=Priority.HIGH)
        normal_task = ModuleTask("normal", "module", "run", priority=Priority.NORMAL)
        critical_task = ModuleTask("critical", "module", "run", priority=Priority.CRITICAL)
        
        # Schedule in random order
        self.scheduler.schedule_task(normal_task)
        self.scheduler.schedule_task(high_task)
        self.scheduler.schedule_task(critical_task)
        
        # Should get critical first
        next_task = self.scheduler.get_next_task()
        self.assertEqual(next_task.task_id, "critical")
        
        # Then high
        next_task = self.scheduler.get_next_task()
        self.assertEqual(next_task.task_id, "high")
    
    def test_dependency_handling(self):
        """Test task dependency handling."""
        task1 = ModuleTask("task1", "module", "run")
        task2 = ModuleTask("task2", "module", "run")
        
        # Schedule task1
        self.scheduler.schedule_task(task1)
        
        # Schedule task2 with dependency on task1
        self.scheduler.schedule_task(task2, dependencies={"task1"})
        
        # Should get task1 first
        next_task = self.scheduler.get_next_task()
        self.assertEqual(next_task.task_id, "task1")
        
        # Mark task1 as completed
        self.scheduler.task_completed("task1")
        
        # Now should get task2
        next_task = self.scheduler.get_next_task()
        self.assertEqual(next_task.task_id, "task2")
    
    def test_scheduling_metrics(self):
        """Test scheduling metrics collection."""
        task = ModuleTask("test", "module", "run")
        self.scheduler.schedule_task(task)
        
        metrics = self.scheduler.get_scheduling_metrics()
        
        self.assertGreaterEqual(metrics.total_scheduled, 1)
        self.assertIsInstance(metrics.priority_distribution, dict)
    
    def test_fair_scheduling(self):
        """Test fair scheduling across modules."""
        scheduler = TaskScheduler(
            scheduling_policy=SchedulingPolicy.FAIR_SHARE,
            enable_fairness=True
        )
        
        try:
            # Set quotas
            scheduler.set_module_quota("module1", 2)
            scheduler.set_module_quota("module2", 1)
            
            # Create tasks
            for i in range(3):
                task1 = ModuleTask(f"m1_task_{i}", "module1", "run")
                task2 = ModuleTask(f"m2_task_{i}", "module2", "run")
                scheduler.schedule_task(task1)
                scheduler.schedule_task(task2)
            
            # Should balance between modules based on quotas
            selected_modules = []
            for _ in range(6):
                task = scheduler.get_next_task()
                if task:
                    selected_modules.append(task.module_name)
            
            # Check fair distribution
            module1_count = selected_modules.count("module1")
            module2_count = selected_modules.count("module2")
            
            # module1 should get more tasks due to higher quota
            self.assertGreaterEqual(module1_count, module2_count)
        
        finally:
            scheduler.shutdown()


class TestResourceCoordinator(unittest.TestCase):
    """Test suite for ResourceCoordinator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.coordinator = ResourceCoordinator(
            max_concurrent_locks=5,
            default_timeout=5.0
        )
        self.test_resource = {"data": "test"}
    
    def tearDown(self):
        """Clean up after tests."""
        self.coordinator.shutdown()
    
    def test_resource_registration(self):
        """Test resource registration and access."""
        # Register resource
        self.coordinator.register_resource(
            "test_resource",
            self.test_resource,
            ResourceType.AGENT_STATE
        )
        
        # Check registration
        self.assertIn("test_resource", self.coordinator.resources)
        self.assertEqual(
            self.coordinator.resource_types["test_resource"],
            ResourceType.AGENT_STATE
        )
    
    def test_lock_acquisition(self):
        """Test resource lock acquisition and release."""
        self.coordinator.register_resource(
            "test_resource",
            self.test_resource,
            ResourceType.AGENT_STATE
        )
        
        # Acquire lock
        lock_id = self.coordinator.acquire_resource(
            "test_resource",
            AccessMode.READ,
            "test_holder"
        )
        
        self.assertIsNotNone(lock_id)
        self.assertIn(lock_id, self.coordinator.active_locks)
        
        # Release lock
        success = self.coordinator.release_resource(lock_id)
        self.assertTrue(success)
        self.assertNotIn(lock_id, self.coordinator.active_locks)
    
    def test_concurrent_read_access(self):
        """Test concurrent read access to resources."""
        self.coordinator.register_resource(
            "test_resource",
            self.test_resource,
            ResourceType.AGENT_STATE
        )
        
        # Acquire multiple read locks
        lock1 = self.coordinator.acquire_resource(
            "test_resource", AccessMode.READ, "reader1"
        )
        lock2 = self.coordinator.acquire_resource(
            "test_resource", AccessMode.READ, "reader2"
        )
        
        self.assertIsNotNone(lock1)
        self.assertIsNotNone(lock2)
        
        # Both should succeed
        self.assertNotEqual(lock1, lock2)
    
    def test_write_lock_exclusivity(self):
        """Test write lock exclusivity."""
        self.coordinator.register_resource(
            "test_resource",
            self.test_resource,
            ResourceType.AGENT_STATE
        )
        
        # Acquire write lock
        write_lock = self.coordinator.acquire_resource(
            "test_resource", AccessMode.WRITE, "writer"
        )
        self.assertIsNotNone(write_lock)
        
        # Try to acquire another write lock (should fail due to timeout)
        start_time = time.time()
        read_lock = self.coordinator.acquire_resource(
            "test_resource", AccessMode.READ, "reader", timeout=0.5
        )
        elapsed_time = time.time() - start_time
        
        self.assertIsNone(read_lock)
        self.assertGreaterEqual(elapsed_time, 0.4)  # Should have waited for timeout
    
    def test_context_manager(self):
        """Test resource access context manager."""
        self.coordinator.register_resource(
            "test_resource",
            self.test_resource,
            ResourceType.AGENT_STATE
        )
        
        # Use context manager
        with self.coordinator.acquire_context(
            "test_resource", AccessMode.READ, "test_user"
        ) as resource:
            self.assertEqual(resource, self.test_resource)
        
        # Lock should be automatically released
        status = self.coordinator.get_resource_status("test_resource")
        self.assertEqual(len(status['locks']), 0)
    
    def test_resource_metrics(self):
        """Test resource usage metrics."""
        self.coordinator.register_resource(
            "test_resource",
            self.test_resource,
            ResourceType.AGENT_STATE
        )
        
        # Perform some operations
        lock = self.coordinator.acquire_resource(
            "test_resource", AccessMode.READ, "test_user"
        )
        self.coordinator.release_resource(lock)
        
        # Check metrics
        metrics = self.coordinator.get_coordinator_metrics()
        self.assertGreaterEqual(metrics.total_requests, 1)
        self.assertGreaterEqual(metrics.successful_acquisitions, 1)


class TestStateCoordinator(unittest.TestCase):
    """Test suite for StateCoordinator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent_state = MockAgentState()
        self.coordinator = StateCoordinator(
            agent_state=self.agent_state,
            sync_policy=SyncPolicy.IMMEDIATE
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.coordinator.shutdown()
    
    def test_module_registration(self):
        """Test module registration for state coordination."""
        mock_module = MockModule()
        
        self.coordinator.register_module("test_module", mock_module)
        
        self.assertIn("test_module", self.coordinator.registered_modules)
        self.assertEqual(self.coordinator.module_versions["test_module"], 0)
    
    def test_state_change_proposal(self):
        """Test state change proposal and application."""
        self.coordinator.register_module("test_module", MockModule())
        
        # Propose a state change
        change_id = self.coordinator.propose_change(
            module_id="test_module",
            path="test_value",
            new_value="updated",
            change_type=StateChangeType.UPDATE
        )
        
        self.assertIsNotNone(change_id)
        
        # Check that change was applied (immediate sync policy)
        self.assertEqual(self.agent_state.test_value, "updated")
    
    def test_state_subscriptions(self):
        """Test state change subscriptions."""
        mock_module = MockModule()
        self.coordinator.register_module("test_module", mock_module)
        
        # Subscribe to path changes
        self.coordinator.subscribe_to_path("test_module", "test_value")
        
        # Make a change
        self.coordinator.propose_change(
            module_id="test_module",
            path="test_value",
            new_value="notified"
        )
        
        # Module should have been notified (in real implementation)
        self.assertIn("test_module", self.coordinator.path_subscribers["test_value"])
    
    def test_conflict_resolution(self):
        """Test conflict resolution between concurrent changes."""
        # Register two modules
        self.coordinator.register_module("module1", MockModule())
        self.coordinator.register_module("module2", MockModule())
        
        # Switch to batched sync to create conflicts
        self.coordinator.sync_policy = SyncPolicy.BATCHED
        
        # Propose conflicting changes
        change1_id = self.coordinator.propose_change(
            module_id="module1",
            path="test_value",
            new_value="value1",
            priority=1
        )
        
        change2_id = self.coordinator.propose_change(
            module_id="module2",
            path="test_value", 
            new_value="value2",
            priority=2  # Higher priority
        )
        
        # Apply changes manually
        self.coordinator.apply_changes()
        
        # Higher priority change should win
        self.assertEqual(self.agent_state.test_value, "value2")
    
    def test_snapshot_creation(self):
        """Test state snapshot creation and restoration."""
        # Create initial state
        self.agent_state.test_value = "snapshot_test"
        
        # Create snapshot
        snapshot_id = self.coordinator.create_snapshot("test_snapshot")
        
        self.assertIn("test_snapshot", self.coordinator.state_snapshots)
        
        # Modify state
        self.agent_state.test_value = "modified"
        
        # Restore snapshot
        success = self.coordinator.restore_snapshot("test_snapshot")
        self.assertTrue(success)
        self.assertEqual(self.agent_state.test_value, "snapshot_test")
    
    def test_transaction_support(self):
        """Test transactional state updates."""
        self.coordinator.register_module("test_module", MockModule())
        
        # Begin transaction
        self.coordinator.begin_transaction("test_tx")
        
        # Make changes in transaction
        self.coordinator.propose_change(
            module_id="test_module",
            path="test_value",
            new_value="tx_value"
        )
        
        # Commit transaction
        success = self.coordinator.commit_transaction("test_tx")
        self.assertTrue(success)
        
        # Change should be applied
        self.assertEqual(self.agent_state.test_value, "tx_value")
    
    def test_transaction_rollback(self):
        """Test transaction rollback."""
        original_value = self.agent_state.test_value
        self.coordinator.register_module("test_module", MockModule())
        
        # Begin transaction
        self.coordinator.begin_transaction("rollback_tx")
        
        # Make changes
        self.coordinator.propose_change(
            module_id="test_module",
            path="test_value",
            new_value="rollback_value"
        )
        
        # Rollback transaction
        success = self.coordinator.rollback_transaction("rollback_tx")
        self.assertTrue(success)
        
        # State should be restored
        self.assertEqual(self.agent_state.test_value, original_value)


class TestParallelPerceptionModule(unittest.TestCase):
    """Test suite for ParallelPerceptionModule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent_state = MockAgentState()
        # Add maze-like attributes
        self.agent_state.curr_tile = (5, 5)
        self.agent_state.curr_time = datetime.now()
        self.agent_state.maze = Mock()
        self.agent_state.maze.access_tile = Mock(return_value={
            'world': 'test_world',
            'sector': 'test_sector', 
            'arena': 'test_arena',
            'game_object': 'test_object',
            'events': []
        })
        
        self.perception_module = ParallelPerceptionModule(
            agent_state=self.agent_state,
            max_workers=2,
            attention_bandwidth=3,
            vision_radius=2
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.perception_module.shutdown()
    
    def test_environmental_data_gathering(self):
        """Test environmental data gathering."""
        env_data = self.perception_module._gather_environmental_data()
        
        self.assertIn('spatial', env_data)
        self.assertIn('temporal', env_data)
        self.assertIn('agent_state', env_data)
        
        spatial_data = env_data['spatial']
        self.assertEqual(spatial_data['current_location'], (5, 5))
        self.assertEqual(spatial_data['vision_radius'], 2)
    
    def test_perception_task_creation(self):
        """Test perception task creation from environmental data."""
        env_data = {
            'spatial': {'current_location': (5, 5), 'nearby_tiles': []},
            'social': {'other_agents': {'agent1'}, 'social_interactions': []},
            'events': [],
            'temporal': {'current_time': datetime.now()},
            'agent_state': {'current_action': 'test_action'}
        }
        
        tasks = self.perception_module._create_perception_tasks(env_data)
        
        # Should create tasks for different sensor types
        sensor_types = [task.sensor_type for task in tasks]
        self.assertIn(SensorType.VISUAL, sensor_types)
        self.assertIn(SensorType.SOCIAL, sensor_types)
        self.assertIn(SensorType.TEMPORAL, sensor_types)
    
    def test_visual_processing(self):
        """Test visual input processing."""
        visual_data = {
            'current_location': (5, 5),
            'nearby_tiles': [
                {
                    'coordinates': (4, 5),
                    'distance': 1.0,
                    'info': {
                        'game_object': 'chair',
                        'world': 'test_world',
                        'sector': 'living_room'
                    }
                }
            ]
        }
        
        processed, confidence = self.perception_module._process_visual_input(visual_data)
        
        self.assertIn('current_location', processed)
        self.assertIn('visible_objects', processed)
        self.assertIn('scene_layout', processed)
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_social_processing(self):
        """Test social input processing.""" 
        social_data = {
            'other_agents': {'agent1', 'agent2'},
            'social_interactions': [
                {
                    'type': 'conversation',
                    'participants': ['agent1'],
                    'location': (4, 5)
                }
            ]
        }
        
        processed, confidence = self.perception_module._process_social_input(social_data)
        
        self.assertIn('present_agents', processed)
        self.assertIn('social_density', processed)
        self.assertIn('active_interactions', processed)
        self.assertEqual(processed['social_density'], 2)
        self.assertGreater(confidence, 0.0)
    
    def test_attention_filtering(self):
        """Test attention-based result filtering."""
        # Create more results than attention bandwidth
        results = []
        for i in range(5):
            result = Mock()
            result.sensor_type = SensorType.VISUAL if i % 2 == 0 else SensorType.SOCIAL
            result.confidence = 0.5 + (i * 0.1)  # Varying confidence
            results.append(result)
        
        filtered = self.perception_module._apply_attention_filter(results)
        
        # Should be limited by attention bandwidth
        self.assertLessEqual(len(filtered), self.perception_module.attention_bandwidth)
        
        # Should be ordered by priority
        if len(filtered) > 1:
            for i in range(len(filtered) - 1):
                curr_priority = (
                    self.perception_module.attention_weights[filtered[i].sensor_type] * 
                    filtered[i].confidence
                )
                next_priority = (
                    self.perception_module.attention_weights[filtered[i+1].sensor_type] * 
                    filtered[i+1].confidence
                )
                self.assertGreaterEqual(curr_priority, next_priority)
    
    def test_caching(self):
        """Test perception result caching."""
        if not self.perception_module.enable_caching:
            self.skipTest("Caching is disabled")
        
        task = Mock()
        task.sensor_type = SensorType.VISUAL
        task.input_data = {'test': 'data'}
        
        # Should not be in cache initially
        cached_result = self.perception_module._check_cache(task)
        self.assertIsNone(cached_result)
        
        # Create and cache a result
        result = Mock()
        result.task_id = "test_task"
        result.sensor_type = SensorType.VISUAL
        result.processed_data = {'result': 'data'}
        
        self.perception_module._cache_result(result)
        
        # Should now find in cache (if cache key generation works)
        # Note: This test is simplified - actual cache key generation is more complex
        self.assertGreater(len(self.perception_module.perception_cache), 0)
    
    def test_metrics_tracking(self):
        """Test perception metrics tracking."""
        initial_metrics = self.perception_module.get_perception_metrics()
        initial_count = initial_metrics.total_tasks_processed
        
        # Create mock results
        results = [Mock() for _ in range(3)]
        for i, result in enumerate(results):
            result.processing_time = 0.1
            result.sensor_type = SensorType.VISUAL
        
        # Update metrics
        self.perception_module._update_metrics(results)
        
        # Check metrics were updated
        updated_metrics = self.perception_module.get_perception_metrics()
        self.assertEqual(
            updated_metrics.total_tasks_processed, 
            initial_count + 3
        )
        self.assertIn(SensorType.VISUAL, updated_metrics.tasks_by_sensor_type)
    
    def test_full_perception_run(self):
        """Test complete perception processing run."""
        # Set up agent state with more realistic data
        self.agent_state.maze.access_tile.return_value = {
            'world': 'villa',
            'sector': 'living_room',
            'arena': 'main_area', 
            'game_object': 'sofa',
            'events': [
                ('agent1:person', 'is', 'sitting', 'Agent1 is sitting on sofa')
            ]
        }
        
        # Run perception
        results = self.perception_module.run()
        
        # Should return perception results
        self.assertIsInstance(results, list)
        
        # Results should be filtered by attention bandwidth
        self.assertLessEqual(len(results), self.perception_module.attention_bandwidth)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete concurrent framework."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.agent_state = MockAgentState()
        self.manager = ConcurrentModuleManager(
            agent_state=self.agent_state,
            max_workers=2,
            max_queue_size=5
        )
    
    def tearDown(self):
        """Clean up integration tests."""
        self.manager.shutdown(timeout=10.0)
    
    def test_end_to_end_task_execution(self):
        """Test complete task execution pipeline."""
        # Register multiple modules
        module1 = MockModule(processing_time=0.1)
        module2 = MockModule(processing_time=0.2)
        
        self.manager.register_module("module1", module1)
        self.manager.register_module("module2", module2)
        
        # Submit tasks with different priorities
        task_ids = []
        
        task_ids.append(self.manager.submit_task(
            "module1", "run", priority=Priority.HIGH
        ))
        task_ids.append(self.manager.submit_task(
            "module2", "process_data", args=("test_data",), priority=Priority.NORMAL
        ))
        task_ids.append(self.manager.submit_task(
            "module1", "run", priority=Priority.LOW
        ))
        
        # Wait for all tasks to complete
        time.sleep(3.0)
        
        # Check that tasks were executed
        self.assertGreaterEqual(module1.run_count, 2)
        self.assertEqual(module2.last_args, ("test_data",))
        
        # Check system status
        status = self.manager.get_system_status()
        self.assertEqual(status['registered_modules'], 2)
        self.assertGreaterEqual(status['completed_tasks'], 3)
    
    def test_resource_coordination_integration(self):
        """Test integration with resource coordinator."""
        # Register a resource
        test_resource = {"shared_data": 0}
        self.manager.resource_coordinator.register_resource(
            "shared_counter",
            test_resource,
            ResourceType.AGENT_STATE
        )
        
        class CounterModule(MockModule):
            def increment_counter(self, resource_coordinator):
                # Acquire resource lock
                with resource_coordinator.acquire_context(
                    "shared_counter", AccessMode.WRITE, "counter_module"
                ) as resource:
                    current = resource["shared_data"]
                    time.sleep(0.1)  # Simulate processing
                    resource["shared_data"] = current + 1
                
                return resource["shared_data"]
        
        # Register counter module
        counter_module = CounterModule()
        self.manager.register_module("counter", counter_module)
        
        # Submit concurrent increment tasks
        task_ids = []
        for i in range(5):
            task_id = self.manager.submit_task(
                "counter", 
                "increment_counter",
                args=(self.manager.resource_coordinator,)
            )
            task_ids.append(task_id)
        
        # Wait for completion
        time.sleep(5.0)
        
        # Counter should be incremented correctly despite concurrent access
        self.assertEqual(test_resource["shared_data"], 5)
    
    def test_state_coordination_integration(self):
        """Test integration with state coordinator."""
        # Register modules with state coordinator
        module1 = MockModule()
        module2 = MockModule()
        
        self.manager.state_coordinator.register_module("module1", module1)
        self.manager.state_coordinator.register_module("module2", module2)
        
        # Subscribe modules to state changes
        self.manager.state_coordinator.subscribe_to_path("module2", "test_value")
        
        # Propose state change from module1
        change_id = self.manager.state_coordinator.propose_change(
            module_id="module1",
            path="test_value",
            new_value="coordinated_value"
        )
        
        # Check that state was updated
        self.assertEqual(self.agent_state.test_value, "coordinated_value")
        
        # Check that module2 was notified (subscription exists)
        self.assertIn("module2", self.manager.state_coordinator.path_subscribers["test_value"])
    
    def test_concurrent_perception_module(self):
        """Test concurrent perception module integration."""
        # Set up agent state for perception
        self.agent_state.curr_tile = (10, 10)
        self.agent_state.curr_time = datetime.now()
        self.agent_state.maze = Mock()
        self.agent_state.maze.access_tile = Mock(return_value={
            'world': 'test_villa',
            'sector': 'garden',
            'arena': 'outdoor_area',
            'game_object': 'tree',
            'events': [
                ('agent2:person', 'is', 'walking', 'Agent2 is walking in garden')
            ]
        })
        
        # Create and register perception module
        perception_module = ParallelPerceptionModule(
            agent_state=self.agent_state,
            max_workers=2,
            attention_bandwidth=4,
            vision_radius=3
        )
        
        try:
            self.manager.register_module("perception", perception_module)
            
            # Submit perception task
            task_id = self.manager.submit_task(
                "perception", "run", 
                priority=Priority.HIGH
            )
            
            # Wait for completion
            time.sleep(2.0)
            
            # Check that perception ran
            status = perception_module.get_processing_status()
            self.assertIsInstance(status, dict)
            self.assertIn('attention_bandwidth', status)
            
        finally:
            perception_module.shutdown()
    
    def test_system_performance_under_load(self):
        """Test system performance under concurrent load."""
        # Register multiple modules
        modules = []
        for i in range(3):
            module = MockModule(processing_time=0.05)
            modules.append(module)
            self.manager.register_module(f"module_{i}", module)
        
        # Submit many concurrent tasks
        task_ids = []
        start_time = time.time()
        
        for i in range(20):
            module_name = f"module_{i % 3}"
            task_id = self.manager.submit_task(
                module_name, "run",
                priority=Priority.NORMAL if i % 2 == 0 else Priority.HIGH
            )
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        time.sleep(10.0)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check that all tasks were processed
        total_executions = sum(module.run_count for module in modules)
        self.assertEqual(total_executions, 20)
        
        # Check system status
        status = self.manager.get_system_status()
        self.assertEqual(status['completed_tasks'], 20)
        
        # Performance should be reasonable (concurrent processing should be faster than serial)
        self.assertLess(total_time, 20 * 0.05)  # Should be much faster than serial execution
        
        print(f"Processed {total_executions} tasks in {total_time:.2f} seconds")
        print(f"Average throughput: {total_executions / total_time:.2f} tasks/second")


def run_tests():
    """Run all test suites."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestConcurrentModuleManager,
        TestModuleExecutor,
        TestTaskScheduler,
        TestResourceCoordinator,
        TestStateCoordinator,
        TestParallelPerceptionModule,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    print("Running Concurrent Module Framework Test Suite...")
    print("=" * 60)
    
    result = run_tests()
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"\nTest suite {'PASSED' if exit_code == 0 else 'FAILED'}")
    exit(exit_code)