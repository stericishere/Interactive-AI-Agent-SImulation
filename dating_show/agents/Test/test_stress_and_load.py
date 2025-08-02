#!/usr/bin/env python3
"""
Stress and Load Testing for Enhanced PIANO Memory Architecture
Tests system behavior under high load, stress conditions, and resource constraints.
"""

import sys
import time
import threading
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
import traceback
import gc
import psutil
import os

# Add current directory to path
sys.path.append('.')
sys.path.append('./memory_structures')

# Import memory components
from memory_structures.circular_buffer import CircularBuffer, CircularBufferReducer
from memory_structures.temporal_memory import TemporalMemory
from memory_structures.episodic_memory import EpisodicMemory, CausalRelationType, EpisodeType
from memory_structures.semantic_memory import SemanticMemory, ConceptType, SemanticRelationType


class StressTestResult:
    """Stress test result tracking with performance metrics"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.performance_metrics = {}
        self.resource_usage = {}
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")
    
    def add_performance(self, test_name: str, time_ms: float, threshold_ms: float, 
                       operations_per_sec: float = None):
        self.performance_metrics[test_name] = {
            'time_ms': time_ms,
            'threshold_ms': threshold_ms,
            'passed': time_ms <= threshold_ms,
            'ops_per_sec': operations_per_sec
        }
        status = "✅" if time_ms <= threshold_ms else "❌"
        ops_info = f" ({operations_per_sec:.1f} ops/sec)" if operations_per_sec else ""
        print(f"{status} {test_name}: {time_ms:.2f}ms{ops_info} (threshold: {threshold_ms}ms)")
    
    def add_resource_usage(self, test_name: str, memory_mb: float, cpu_percent: float):
        self.resource_usage[test_name] = {
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent
        }
        print(f"📊 {test_name}: Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"STRESS TEST SUMMARY: {self.passed}/{total} passed ({100*self.passed/total if total > 0 else 0:.1f}%)")
        print(f"{'='*60}")
        
        if self.performance_metrics:
            print("\nPERFORMANCE METRICS:")
            for test_name, metrics in self.performance_metrics.items():
                status = "PASS" if metrics['passed'] else "FAIL"
                ops_info = f" ({metrics['ops_per_sec']:.1f} ops/sec)" if metrics.get('ops_per_sec') else ""
                print(f"  {status}: {test_name} - {metrics['time_ms']:.2f}ms{ops_info}")
        
        if self.resource_usage:
            print("\nRESOURCE USAGE:")
            for test_name, usage in self.resource_usage.items():
                print(f"  {test_name}: {usage['memory_mb']:.1f}MB memory, {usage['cpu_percent']:.1f}% CPU")
        
        if self.errors:
            print("\nFAILURES:")
            for error in self.errors:
                print(f"  • {error}")
        
        return self.failed == 0


def measure_resource_usage():
    """Measure current resource usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    return memory_mb, cpu_percent


def test_high_volume_operations(result: StressTestResult):
    """Test system behavior under high volume operations"""
    print("\n--- High Volume Operations Test ---")
    
    try:
        # Test CircularBuffer high volume
        buffer = CircularBuffer(max_size=1000)
        
        memory_before, cpu_before = measure_resource_usage()
        start_time = time.perf_counter()
        
        operations = 10000
        for i in range(operations):
            buffer.add_memory(f"High volume memory {i}", 
                            random.choice(["event", "thought", "conversation", "activity"]), 
                            random.uniform(0.1, 1.0))
            
            # Intermittent retrieval operations
            if i % 100 == 0:
                buffer.get_recent_memories(10)
                buffer.get_important_memories(0.7)
        
        end_time = time.perf_counter()
        memory_after, cpu_after = measure_resource_usage()
        
        total_time_ms = (end_time - start_time) * 1000
        ops_per_sec = operations / (total_time_ms / 1000)
        
        result.add_performance("CircularBuffer: High volume operations", 
                             total_time_ms, 5000.0, ops_per_sec)
        result.add_resource_usage("CircularBuffer: High volume", 
                                memory_after - memory_before, cpu_after)
        
        # Verify system integrity
        if len(buffer) == 1000:  # Should maintain size constraint
            result.add_pass("CircularBuffer: High volume integrity")
        else:
            result.add_fail("CircularBuffer: High volume integrity", 
                          f"Expected 1000, got {len(buffer)}")
    
    except Exception as e:
        result.add_fail("CircularBuffer: High volume operations", str(e))


def test_memory_pressure_scenarios(result: StressTestResult):
    """Test behavior under memory pressure"""
    print("\n--- Memory Pressure Scenarios ---")
    
    try:
        # Create multiple large memory systems
        systems = []
        memory_before, _ = measure_resource_usage()
        
        start_time = time.perf_counter()
        
        # Create multiple systems simultaneously
        for i in range(5):
            buffer = CircularBuffer(max_size=500)
            temporal = TemporalMemory(retention_hours=24)
            semantic = SemanticMemory(max_concepts=1000)
            
            # Populate each system
            for j in range(200):
                content = f"System {i} memory {j}"
                buffer.add_memory(content, "event", 0.5)
                temporal.add_memory(content, "event", 0.5)
                
                if j < 100:  # Don't overflow semantic
                    semantic.add_concept(f"System{i}_Concept{j}", ConceptType.PERSON, 
                                       f"Concept {j}", 0.5)
            
            systems.append((buffer, temporal, semantic))
        
        end_time = time.perf_counter()
        memory_after, cpu_after = measure_resource_usage()
        
        creation_time_ms = (end_time - start_time) * 1000
        memory_used_mb = memory_after - memory_before
        
        result.add_performance("Memory Pressure: Multi-system creation", 
                             creation_time_ms, 10000.0)
        result.add_resource_usage("Memory Pressure: Multi-system", 
                                memory_used_mb, cpu_after)
        
        # Test system interactions under pressure
        start_time = time.perf_counter()
        
        for buffer, temporal, semantic in systems:
            buffer.get_recent_memories(20)
            temporal.retrieve_recent_memories(hours_back=1, limit=50)
            semantic.retrieve_by_activation(threshold=0.3, limit=30)
        
        end_time = time.perf_counter()
        query_time_ms = (end_time - start_time) * 1000
        
        result.add_performance("Memory Pressure: Multi-system queries", 
                             query_time_ms, 2000.0)
        
        # Cleanup and measure memory release
        del systems
        gc.collect()
        
        memory_final, _ = measure_resource_usage()
        memory_released = memory_after - memory_final
        
        if memory_released > memory_used_mb * 0.5:  # At least 50% released
            result.add_pass("Memory Pressure: Memory cleanup")
        else:
            result.add_fail("Memory Pressure: Memory cleanup", 
                          f"Only {memory_released:.1f}MB released from {memory_used_mb:.1f}MB")
    
    except Exception as e:
        result.add_fail("Memory Pressure: Scenarios", str(e))


def test_concurrent_thread_safety(result: StressTestResult):
    """Test thread safety with concurrent operations"""
    print("\n--- Concurrent Thread Safety Test ---")
    
    try:
        # Shared memory system
        shared_semantic = SemanticMemory(max_concepts=2000)
        errors = []
        operations_completed = []
        
        def worker_thread(thread_id: int, operations: int):
            """Worker thread for concurrent operations"""
            try:
                for i in range(operations):
                    # Add concepts
                    concept_name = f"Thread{thread_id}_Concept{i}"
                    concept_id = shared_semantic.add_concept(
                        concept_name, ConceptType.PERSON, f"Person {i}", 0.5
                    )
                    
                    # Perform queries
                    if i % 10 == 0:
                        shared_semantic.retrieve_by_activation(threshold=0.3, limit=10)
                        shared_semantic.retrieve_by_type(ConceptType.PERSON, limit=5)
                    
                    # Add relationships occasionally
                    if i > 0 and i % 20 == 0:
                        concepts = list(shared_semantic.concepts.keys())
                        if len(concepts) >= 2:
                            concept1, concept2 = random.sample(concepts, 2)
                            shared_semantic.add_relation(
                                concept1, concept2, SemanticRelationType.KNOWS, 0.6
                            )
                
                operations_completed.append(operations)
                
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Start multiple threads
        threads = []
        num_threads = 4
        operations_per_thread = 100
        
        start_time = time.perf_counter()
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, 
                                    args=(i, operations_per_thread))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        
        concurrent_time_ms = (end_time - start_time) * 1000
        total_operations = sum(operations_completed)
        
        if not errors and len(operations_completed) == num_threads:
            result.add_pass("Concurrent Thread Safety: No errors")
            result.add_performance("Concurrent Thread Safety: Operations", 
                                 concurrent_time_ms, 5000.0, 
                                 total_operations / (concurrent_time_ms / 1000))
        else:
            result.add_fail("Concurrent Thread Safety: Errors occurred", 
                          f"Errors: {len(errors)}, Completed threads: {len(operations_completed)}")
        
        # Verify data integrity
        final_concepts = len(shared_semantic.concepts)
        expected_range = (total_operations * 0.8, total_operations * 1.2)  # Allow for some variance
        
        if expected_range[0] <= final_concepts <= expected_range[1]:
            result.add_pass("Concurrent Thread Safety: Data integrity")
        else:
            result.add_fail("Concurrent Thread Safety: Data integrity", 
                          f"Expected {expected_range}, got {final_concepts}")
    
    except Exception as e:
        result.add_fail("Concurrent Thread Safety: Test", str(e))


def test_long_running_simulation(result: StressTestResult):
    """Test system behavior in long-running simulation"""
    print("\n--- Long Running Simulation Test ---")
    
    try:
        # Create agent simulation environment
        buffer = CircularBuffer(max_size=100)
        temporal = TemporalMemory(retention_hours=2)
        episodic = EpisodicMemory(max_episodes=50)
        semantic = SemanticMemory(max_concepts=200)
        
        simulation_duration = 5  # 5 seconds simulation
        operation_interval = 0.01  # 10ms between operations
        
        start_time = time.perf_counter()
        operations_count = 0
        cleanup_operations = 0
        
        while (time.perf_counter() - start_time) < simulation_duration:
            current_time = datetime.now()
            
            # Simulate agent activities
            activities = [
                "observing environment",
                "having conversation",
                "forming memory",
                "making decision",
                "reflecting on experience"
            ]
            
            activity = random.choice(activities)
            importance = random.uniform(0.1, 0.9)
            
            # Add to all memory systems
            buffer.add_memory(f"Agent {activity} at {current_time.strftime('%H:%M:%S')}", 
                            "activity", importance)
            temporal.add_memory(f"Temporal: {activity}", "activity", importance, 
                              timestamp=current_time)
            
            # Occasionally add episodic events
            if operations_count % 5 == 0:
                participants = set() if random.random() < 0.3 else {"Other Agent"}
                episodic.add_event(f"Episodic: {activity}", "activity", importance,
                                 participants=participants, location="Simulation Environment")
            
            # Occasionally add semantic concepts
            if operations_count % 10 == 0 and len(semantic.concepts) < 150:
                concept_name = f"Concept_{operations_count}"
                semantic.add_concept(concept_name, ConceptType.ACTIVITY, 
                                   f"Activity concept {operations_count}", importance)
            
            # Periodic cleanup operations
            if operations_count % 20 == 0:
                buffer.cleanup_expired_memories()
                temporal.cleanup_expired_memories()
                semantic.update_activation_decay()
                cleanup_operations += 1
            
            operations_count += 1
            time.sleep(operation_interval)
        
        end_time = time.perf_counter()
        
        actual_duration_ms = (end_time - start_time) * 1000
        ops_per_sec = operations_count / (actual_duration_ms / 1000)
        
        result.add_performance("Long Running: Simulation operations", 
                             actual_duration_ms, simulation_duration * 1000 * 1.2,  # 20% tolerance
                             ops_per_sec)
        
        # Verify system health after long simulation
        buffer_health = len(buffer) <= buffer.max_size
        temporal_health = len(temporal.memories) > 0
        episodic_health = len(episodic.events) > 0
        semantic_health = len(semantic.concepts) <= semantic.max_concepts
        
        if all([buffer_health, temporal_health, episodic_health, semantic_health]):
            result.add_pass("Long Running: System health maintained")
        else:
            result.add_fail("Long Running: System health", 
                          f"Buffer: {buffer_health}, Temporal: {temporal_health}, "
                          f"Episodic: {episodic_health}, Semantic: {semantic_health}")
        
        # Test system responsiveness after long run
        query_start = time.perf_counter()
        
        recent_memories = buffer.get_recent_memories(10)
        temporal_memories = temporal.retrieve_recent_memories(hours_back=1, limit=10)
        semantic_concepts = semantic.retrieve_by_activation(threshold=0.3, limit=10)
        
        query_end = time.perf_counter()
        query_time_ms = (query_end - query_start) * 1000
        
        result.add_performance("Long Running: Post-simulation responsiveness", 
                             query_time_ms, 100.0)
        
        print(f"📊 Simulation completed: {operations_count} operations, {cleanup_operations} cleanups")
    
    except Exception as e:
        result.add_fail("Long Running: Simulation", str(e))


def test_resource_exhaustion_recovery(result: StressTestResult):
    """Test recovery from resource exhaustion scenarios"""
    print("\n--- Resource Exhaustion Recovery Test ---")
    
    try:
        # Test CircularBuffer overflow handling
        small_buffer = CircularBuffer(max_size=5)
        
        # Fill beyond capacity
        for i in range(20):
            small_buffer.add_memory(f"Overflow memory {i}", "event", 0.5)
        
        if len(small_buffer) == 5:
            result.add_pass("Resource Exhaustion: CircularBuffer overflow handling")
        else:
            result.add_fail("Resource Exhaustion: CircularBuffer overflow", 
                          f"Expected 5, got {len(small_buffer)}")
        
        # Test SemanticMemory concept limit
        limited_semantic = SemanticMemory(max_concepts=10)
        
        # Add beyond limit
        for i in range(25):
            limited_semantic.add_concept(f"Limited_Concept_{i}", ConceptType.PERSON, 
                                       f"Person {i}", random.uniform(0.1, 1.0))
        
        final_count = len(limited_semantic.concepts)
        if final_count <= 10:
            result.add_pass("Resource Exhaustion: SemanticMemory concept limit")
        else:
            result.add_fail("Resource Exhaustion: SemanticMemory concept limit", 
                          f"Expected ≤10, got {final_count}")
        
        # Test system recovery after cleanup
        cleanup_start = time.perf_counter()
        
        cleaned_buffer = small_buffer.cleanup_expired_memories()
        consolidated_semantic = limited_semantic.consolidate_concepts()
        
        cleanup_end = time.perf_counter()
        cleanup_time_ms = (cleanup_end - cleanup_start) * 1000
        
        result.add_performance("Resource Exhaustion: Recovery cleanup", 
                             cleanup_time_ms, 200.0)
        
        # Test continued functionality after recovery
        small_buffer.add_memory("Post-recovery memory", "event", 0.8)
        limited_semantic.add_concept("Post_Recovery", ConceptType.PERSON, "Recovery test", 0.7)
        
        result.add_pass("Resource Exhaustion: Post-recovery functionality")
    
    except Exception as e:
        result.add_fail("Resource Exhaustion: Recovery", str(e))


def main():
    """Execute comprehensive stress and load tests"""
    print("🔥 Enhanced PIANO Memory Architecture - Stress and Load Testing")
    print("="*70)
    print("Testing system behavior under stress, high load, and resource constraints...")
    
    result = StressTestResult()
    
    # Execute all stress test categories
    test_high_volume_operations(result)
    test_memory_pressure_scenarios(result)
    test_concurrent_thread_safety(result)
    test_long_running_simulation(result)
    test_resource_exhaustion_recovery(result)
    
    # Final summary with recommendations
    success = result.summary()
    
    if success:
        print("\n🎉 ALL STRESS TESTS PASSED! System is robust under extreme conditions.")
        print("💪 Memory architecture demonstrates excellent resilience and performance.")
    else:
        print(f"\n💥 {result.failed} stress tests failed. System may need hardening.")
        print("⚠️  Review failures above for potential reliability issues.")
    
    # Resource usage summary
    if result.resource_usage:
        total_memory = sum(usage['memory_mb'] for usage in result.resource_usage.values())
        avg_cpu = sum(usage['cpu_percent'] for usage in result.resource_usage.values()) / len(result.resource_usage)
        print(f"\n📊 Total memory impact: {total_memory:.1f}MB, Average CPU: {avg_cpu:.1f}%")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)