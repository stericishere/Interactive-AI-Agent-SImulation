#!/usr/bin/env python3
"""
Simplified Integration tests for Enhanced PIANO Memory Architecture
Focus on core integration patterns and performance requirements.
"""

import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add current directory to path
sys.path.append('.')
sys.path.append('./memory_structures')

# Import memory components
from memory_structures.circular_buffer import CircularBuffer, CircularBufferReducer
from memory_structures.temporal_memory import TemporalMemory
from memory_structures.episodic_memory import EpisodicMemory
from memory_structures.semantic_memory import SemanticMemory, ConceptType, SemanticRelationType


class IntegrationTestResult:
    """Integration test result tracking"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.performance_metrics = {}
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")
    
    def add_performance(self, test_name: str, time_ms: float, threshold_ms: float):
        self.performance_metrics[test_name] = {
            'time_ms': time_ms,
            'threshold_ms': threshold_ms,
            'passed': time_ms <= threshold_ms
        }
        status = "✅" if time_ms <= threshold_ms else "❌"
        print(f"{status} {test_name}: {time_ms:.2f}ms (threshold: {threshold_ms}ms)")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"INTEGRATION TEST SUMMARY: {self.passed}/{total} passed ({100*self.passed/total if total > 0 else 0:.1f}%)")
        print(f"{'='*60}")
        if self.errors:
            print("\nFAILURES:")
            for error in self.errors:
                print(f"  • {error}")
        return self.failed == 0


def test_memory_system_integration(result: IntegrationTestResult):
    """Test basic integration between memory systems"""
    print("\n--- Memory System Integration Tests ---")
    
    try:
        # Create all memory systems
        buffer = CircularBuffer(max_size=20)
        temporal = TemporalMemory(retention_hours=2)
        episodic = EpisodicMemory(max_episodes=50)
        semantic = SemanticMemory(max_concepts=500)
        
        # Add related content to each system
        content = "Had conversation with Maria about hiking preferences"
        
        # Add to buffer
        buffer.add_memory(content, "conversation", 0.8)
        
        # Add to temporal
        temporal.add_memory(content, "conversation", 0.8)
        
        # Add to episodic (simplified)
        event_id = episodic.add_event(content, "conversation", 0.8)
        
        # Add concepts to semantic
        maria_id = semantic.add_concept("Maria", ConceptType.PERSON, "Fellow contestant", 0.7)
        hiking_id = semantic.add_concept("Hiking", ConceptType.ACTIVITY, "Outdoor activity", 0.6)
        
        # Verify all systems have content
        if (len(buffer) > 0 and len(temporal.memories) > 0 and 
            len(episodic.events) > 0 and len(semantic.concepts) > 0):
            result.add_pass("Memory Integration: All systems populated")
        else:
            result.add_fail("Memory Integration: All systems populated", 
                          f"Buffer:{len(buffer)} Temporal:{len(temporal.memories)} Episodic:{len(episodic.events)} Semantic:{len(semantic.concepts)}")
        
        # Test cross-system retrieval
        buffer_memories = buffer.search_memories("Maria")
        temporal_memories = temporal.retrieve_recent_memories(hours_back=1)
        semantic_concepts = semantic.retrieve_by_type(ConceptType.PERSON)
        
        if len(buffer_memories) > 0 and len(temporal_memories) > 0 and len(semantic_concepts) > 0:
            result.add_pass("Memory Integration: Cross-system retrieval")
        else:
            result.add_fail("Memory Integration: Cross-system retrieval", "Retrieval failed")
            
    except Exception as e:
        result.add_fail("Memory Integration: Tests", str(e))


def test_circular_buffer_reducer(result: IntegrationTestResult):
    """Test CircularBufferReducer for LangGraph integration"""
    print("\n--- CircularBufferReducer Tests ---")
    
    try:
        reducer = CircularBufferReducer(max_size=3)
        
        # Test reducer behavior
        current_state = []
        
        # Add first batch
        update1 = [{"content": "Memory 1", "timestamp": datetime.now().isoformat()}]
        state1 = reducer(current_state, update1)
        
        # Add second batch  
        update2 = [{"content": "Memory 2", "timestamp": datetime.now().isoformat()}]
        state2 = reducer(state1, update2)
        
        if len(state2) == 2:
            result.add_pass("CircularBufferReducer: Sequential updates")
        else:
            result.add_fail("CircularBufferReducer: Sequential updates", f"Expected 2, got {len(state2)}")
        
        # Test size limit
        for i in range(5):
            update = [{"content": f"Memory {i+3}", "timestamp": datetime.now().isoformat()}]
            state2 = reducer(state2, update)
        
        if len(state2) == 3:  # Should be limited to max_size
            result.add_pass("CircularBufferReducer: Size limit enforcement")
        else:
            result.add_fail("CircularBufferReducer: Size limit enforcement", f"Expected 3, got {len(state2)}")
            
    except Exception as e:
        result.add_fail("CircularBufferReducer: Tests", str(e))


def test_performance_under_load(result: IntegrationTestResult):
    """Test performance requirements under simulated load"""
    print("\n--- Performance Under Load Tests ---")
    
    try:
        # Create memory systems
        buffer = CircularBuffer(max_size=20)
        temporal = TemporalMemory(retention_hours=1)
        semantic = SemanticMemory(max_concepts=100)
        
        # Test batch operations performance
        start_time = time.perf_counter()
        
        # Simulate 50 memory operations
        for i in range(50):
            content = f"Memory operation {i}"
            buffer.add_memory(content, "event", 0.5)
            temporal.add_memory(content, "event", 0.5)
            
            if i % 10 == 0:  # Add concepts periodically
                semantic.add_concept(f"Concept_{i}", ConceptType.PERSON, f"Test concept {i}", 0.5)
        
        end_time = time.perf_counter()
        batch_time_ms = (end_time - start_time) * 1000
        
        result.add_performance("Performance Under Load: Batch operations (50 items)", 
                             batch_time_ms, 100.0)
        
        # Test retrieval performance
        start_time = time.perf_counter()
        
        # Multiple retrieval operations
        for i in range(20):
            buffer.get_recent_memories(5)
            temporal.retrieve_recent_memories(hours_back=1, limit=10)
            semantic.retrieve_by_activation(threshold=0.3, limit=5)
        
        end_time = time.perf_counter()
        retrieval_time_ms = (end_time - start_time) * 1000
        avg_retrieval_ms = retrieval_time_ms / 20
        
        result.add_performance("Performance Under Load: Retrieval operations", 
                             avg_retrieval_ms, 50.0)
        
        # Test memory cleanup performance
        start_time = time.perf_counter()
        
        buffer.cleanup_expired_memories()
        temporal.cleanup_expired_memories()
        semantic.update_activation_decay()
        
        end_time = time.perf_counter()
        cleanup_time_ms = (end_time - start_time) * 1000
        
        result.add_performance("Performance Under Load: Memory cleanup", 
                             cleanup_time_ms, 50.0)
        
    except Exception as e:
        result.add_fail("Performance Under Load: Tests", str(e))


def test_concurrent_agent_simulation(result: IntegrationTestResult):
    """Test concurrent operations simulating multiple agents"""
    print("\n--- Concurrent Agent Simulation Tests ---")
    
    try:
        # Create separate memory systems for 3 simulated agents
        agents = []
        for i in range(3):
            agent_memories = {
                'buffer': CircularBuffer(max_size=15),
                'temporal': TemporalMemory(retention_hours=1),
                'semantic': SemanticMemory(max_concepts=50)
            }
            agents.append(agent_memories)
        
        start_time = time.perf_counter()
        
        # Simulate concurrent activities
        for agent_id, agent in enumerate(agents):
            # Each agent performs different activities
            for j in range(10):
                content = f"Agent {agent_id} activity {j}"
                agent['buffer'].add_memory(content, "activity", 0.5)
                agent['temporal'].add_memory(content, "activity", 0.5)
                
                # Add unique concepts
                concept_name = f"Agent_{agent_id}_Concept_{j}"
                agent['semantic'].add_concept(concept_name, ConceptType.PERSON, 
                                            f"Concept for agent {agent_id}", 0.5)
        
        end_time = time.perf_counter()
        concurrent_time_ms = (end_time - start_time) * 1000
        
        result.add_performance("Concurrent Simulation: 3 agents, 10 operations each", 
                             concurrent_time_ms, 200.0)
        
        # Verify independent state
        unique_concepts = set()
        for agent in agents:
            for concept in agent['semantic'].concepts.values():
                unique_concepts.add(concept.name)
        
        expected_concepts = 3 * 10  # 3 agents × 10 concepts each
        if len(unique_concepts) == expected_concepts:
            result.add_pass("Concurrent Simulation: Independent agent state")
        else:
            result.add_fail("Concurrent Simulation: Independent agent state", 
                          f"Expected {expected_concepts}, got {len(unique_concepts)}")
        
        # Test cross-agent memory retrieval doesn't interfere
        agent_0_memories = agents[0]['buffer'].get_recent_memories(5)
        agent_1_memories = agents[1]['buffer'].get_recent_memories(5)
        
        # Check memories are agent-specific
        agent_0_content = [m['content'] for m in agent_0_memories]
        agent_1_content = [m['content'] for m in agent_1_memories]
        
        overlap = set(agent_0_content) & set(agent_1_content)
        if len(overlap) == 0:
            result.add_pass("Concurrent Simulation: Memory isolation")
        else:
            result.add_fail("Concurrent Simulation: Memory isolation", 
                          f"Found {len(overlap)} overlapping memories")
            
    except Exception as e:
        result.add_fail("Concurrent Simulation: Tests", str(e))


def test_decision_latency_simulation(result: IntegrationTestResult):
    """Test decision-making latency simulation"""
    print("\n--- Decision Latency Simulation Tests ---")
    
    try:
        # Create agent memory systems
        buffer = CircularBuffer(max_size=20)
        temporal = TemporalMemory(retention_hours=2)
        semantic = SemanticMemory(max_concepts=100)
        
        # Populate with base memories
        for i in range(15):
            buffer.add_memory(f"Background memory {i}", "background", 0.3)
            temporal.add_memory(f"Background memory {i}", "background", 0.3)
        
        # Add some concepts
        for i in range(10):
            semantic.add_concept(f"Person_{i}", ConceptType.PERSON, f"Person {i}", 0.5)
        
        decision_times = []
        
        # Simulate 10 decision-making cycles
        for decision_num in range(10):
            start_time = time.perf_counter()
            
            # 1. Perceive (add new memory)
            perception = f"New situation {decision_num} encountered"
            buffer.add_memory(perception, "perception", 0.6)
            
            # 2. Retrieve relevant context
            recent_memories = buffer.get_recent_memories(5)
            temporal_context = temporal.retrieve_recent_memories(hours_back=1, limit=5)
            semantic_context = semantic.retrieve_by_activation(threshold=0.3, limit=5)
            
            # 3. Make decision (simple logic)
            context_size = len(recent_memories) + len(temporal_context) + len(semantic_context)
            decision = f"action_{context_size % 3}"  # Simple decision logic
            
            # 4. Store decision
            buffer.add_memory(f"Decided on {decision}", "decision", 0.7)
            temporal.add_memory(f"Decided on {decision}", "decision", 0.7)
            
            end_time = time.perf_counter()
            decision_time_ms = (end_time - start_time) * 1000
            decision_times.append(decision_time_ms)
        
        # Calculate average decision time
        avg_decision_time = sum(decision_times) / len(decision_times)
        max_decision_time = max(decision_times)
        
        result.add_performance("Decision Latency: Average decision time", 
                             avg_decision_time, 100.0)
        result.add_performance("Decision Latency: Maximum decision time", 
                             max_decision_time, 150.0)
        
        # Verify decision consistency
        decision_memories = buffer.search_memories("Decided on")
        if len(decision_memories) == 10:
            result.add_pass("Decision Latency: Decision consistency")
        else:
            result.add_fail("Decision Latency: Decision consistency", 
                          f"Expected 10 decisions, found {len(decision_memories)}")
            
    except Exception as e:
        result.add_fail("Decision Latency: Tests", str(e))


def main():
    """Execute simplified integration test suite"""
    print("🔗 Enhanced PIANO Memory Architecture - Simplified Integration Tests")
    print("="*75)
    
    result = IntegrationTestResult()
    
    # Execute integration test categories
    test_memory_system_integration(result)
    test_circular_buffer_reducer(result)
    test_performance_under_load(result)
    test_concurrent_agent_simulation(result)
    test_decision_latency_simulation(result)
    
    # Generate performance report
    print(f"\n--- Performance Report ---")
    for test_name, metrics in result.performance_metrics.items():
        status = "PASS" if metrics['passed'] else "FAIL"
        print(f"{status}: {test_name}")
        print(f"  Time: {metrics['time_ms']:.2f}ms (threshold: {metrics['threshold_ms']}ms)")
    
    # Final summary
    success = result.summary()
    
    if success:
        print("\n🎉 ALL INTEGRATION TESTS PASSED! System ready for production.")
        return 0
    else:
        print(f"\n💥 {result.failed} integration tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)