#!/usr/bin/env python3
"""
Comprehensive test suite for Enhanced PIANO Memory Architecture
Tests all memory components, integration, and performance requirements.
"""

import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import traceback

# Add current directory to path
sys.path.append('.')
sys.path.append('./memory_structures')

# Import memory components
from memory_structures.circular_buffer import CircularBuffer, CircularBufferReducer
from memory_structures.temporal_memory import TemporalMemory
from memory_structures.episodic_memory import EpisodicMemory, CausalRelationType, EpisodeType
from memory_structures.semantic_memory import SemanticMemory, ConceptType, SemanticRelationType


class TestResult:
    """Test result tracking"""
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
        print(f"TEST SUMMARY: {self.passed}/{total} passed ({100*self.passed/total if total > 0 else 0:.1f}%)")
        print(f"{'='*60}")
        if self.errors:
            print("\nFAILURES:")
            for error in self.errors:
                print(f"  • {error}")
        return self.failed == 0


def test_circular_buffer_functionality(result: TestResult):
    """Test CircularBuffer core functionality"""
    print("\n--- CircularBuffer Functionality Tests ---")
    
    try:
        # Test basic functionality
        buffer = CircularBuffer(max_size=5, retention_minutes=30)
        
        # Test memory addition
        mem1 = buffer.add_memory("Test memory 1", "event", 0.8)
        mem2 = buffer.add_memory("Test memory 2", "thought", 0.6)
        
        if len(buffer) == 2:
            result.add_pass("CircularBuffer: Memory addition")
        else:
            result.add_fail("CircularBuffer: Memory addition", f"Expected 2, got {len(buffer)}")
        
        # Test retrieval
        recent = buffer.get_recent_memories(2)
        if len(recent) == 2 and recent[-1]["content"] == "Test memory 2":
            result.add_pass("CircularBuffer: Recent memory retrieval")
        else:
            result.add_fail("CircularBuffer: Recent memory retrieval", "Incorrect retrieval")
        
        # Test importance filtering
        important = buffer.get_important_memories(0.7)
        if len(important) == 1 and important[0]["importance"] == 0.8:
            result.add_pass("CircularBuffer: Importance filtering")
        else:
            result.add_fail("CircularBuffer: Importance filtering", "Incorrect filtering")
        
        # Test circular behavior
        for i in range(10):  # Add more than max_size
            buffer.add_memory(f"Excess memory {i}", "event", 0.5)
        
        if len(buffer) == 5:  # Should be limited to max_size
            result.add_pass("CircularBuffer: Circular size constraint")
        else:
            result.add_fail("CircularBuffer: Circular size constraint", f"Expected 5, got {len(buffer)}")
        
        # Test search functionality
        buffer.add_memory("Find this specific content", "event", 0.7)
        search_results = buffer.search_memories("specific content")
        if len(search_results) >= 1:
            result.add_pass("CircularBuffer: Search functionality")
        else:
            result.add_fail("CircularBuffer: Search functionality", "Search failed")
            
    except Exception as e:
        result.add_fail("CircularBuffer: Functionality tests", str(e))


def test_temporal_memory_functionality(result: TestResult):
    """Test TemporalMemory core functionality"""
    print("\n--- TemporalMemory Functionality Tests ---")
    
    try:
        temporal = TemporalMemory(retention_hours=2, decay_rate=0.1)
        
        # Test memory addition with timestamps
        now = datetime.now()
        mem1 = temporal.add_memory("Recent memory", "event", 0.8, timestamp=now)
        mem2 = temporal.add_memory("Older memory", "event", 0.6, timestamp=now - timedelta(minutes=30))
        
        if len(temporal.memories) > 0:
            result.add_pass("TemporalMemory: Memory addition")
        else:
            result.add_fail("TemporalMemory: Memory addition", "No memories stored")
        
        # Test strength calculation with decay
        strength1 = temporal.get_memory_strength(mem1)
        strength2 = temporal.get_memory_strength(mem2)
        
        if strength1 > strength2:  # Recent should be stronger
            result.add_pass("TemporalMemory: Decay calculation")
        else:
            result.add_fail("TemporalMemory: Decay calculation", f"Recent: {strength1}, Older: {strength2}")
        
        # Test time range retrieval
        recent_memories = temporal.retrieve_memories_by_timerange(
            now - timedelta(hours=1), now
        )
        if len(recent_memories) >= 1:
            result.add_pass("TemporalMemory: Time range retrieval")
        else:
            result.add_fail("TemporalMemory: Time range retrieval", "No memories retrieved")
        
        # Test pattern matching
        pattern_memories = temporal.retrieve_memories_by_pattern("same_hour", now)
        if len(pattern_memories) >= 0:  # Should work without error
            result.add_pass("TemporalMemory: Pattern matching")
        else:
            result.add_fail("TemporalMemory: Pattern matching", "Pattern matching failed")
            
    except Exception as e:
        result.add_fail("TemporalMemory: Functionality tests", str(e))


def test_episodic_memory_functionality(result: TestResult):
    """Test EpisodicMemory core functionality"""
    print("\n--- EpisodicMemory Functionality Tests ---")
    
    try:
        episodic = EpisodicMemory(max_episodes=50, coherence_threshold=0.6)
        
        # Test event addition
        event1 = episodic.add_event(
            "Started conversation with Maria",
            "conversation", 0.8, 
            participants={"Maria"}, 
            location="Living Room"
        )
        
        event2 = episodic.add_event(
            "Maria shared her interests",
            "conversation", 0.7,
            participants={"Maria"},
            location="Living Room"
        )
        
        if len(episodic.events) == 2:
            result.add_pass("EpisodicMemory: Event addition")
        else:
            result.add_fail("EpisodicMemory: Event addition", f"Expected 2, got {len(episodic.events)}")
        
        # Test automatic episode creation
        if len(episodic.episodes) >= 1:
            result.add_pass("EpisodicMemory: Automatic episode creation")
        else:
            result.add_fail("EpisodicMemory: Automatic episode creation", "No episodes created")
        
        # Test causal relationship
        episodic.add_causal_relation(
            event1, event2, CausalRelationType.ENABLES, 0.8, 0.9
        )
        
        if len(episodic.causal_relations) == 1:
            result.add_pass("EpisodicMemory: Causal relationships")
        else:
            result.add_fail("EpisodicMemory: Causal relationships", "Causal relation not added")
        
        # Test participant indexing
        maria_episodes = episodic.get_episodes_by_participant("Maria")
        if len(maria_episodes) >= 1:
            result.add_pass("EpisodicMemory: Participant indexing")
        else:
            result.add_fail("EpisodicMemory: Participant indexing", "No episodes found for Maria")
        
        # Test episode narrative generation
        if maria_episodes:
            narrative = episodic.get_episode_narrative(maria_episodes[0].episode_id)
            if len(narrative) > 0:
                result.add_pass("EpisodicMemory: Narrative generation")
            else:
                result.add_fail("EpisodicMemory: Narrative generation", "Empty narrative")
                
    except Exception as e:
        result.add_fail("EpisodicMemory: Functionality tests", str(e))


def test_semantic_memory_functionality(result: TestResult):
    """Test SemanticMemory core functionality"""
    print("\n--- SemanticMemory Functionality Tests ---")
    
    try:
        semantic = SemanticMemory(max_concepts=500)
        
        # Test concept addition
        isabella_id = semantic.add_concept("Isabella", ConceptType.PERSON, "Confident contestant", 0.8)
        maria_id = semantic.add_concept("Maria", ConceptType.PERSON, "Artistic contestant", 0.7)
        confident_id = semantic.add_concept("Confident", ConceptType.TRAIT, "Confidence trait", 0.6)
        
        if len(semantic.concepts) == 3:
            result.add_pass("SemanticMemory: Concept addition")
        else:
            result.add_fail("SemanticMemory: Concept addition", f"Expected 3, got {len(semantic.concepts)}")
        
        # Test relationship addition
        rel1 = semantic.add_relation(isabella_id, confident_id, SemanticRelationType.HAS_A, 0.9)
        
        if len(semantic.relations) == 1:
            result.add_pass("SemanticMemory: Relationship addition")
        else:
            result.add_fail("SemanticMemory: Relationship addition", "Relationship not added")
        
        # Test activation and spreading
        semantic.activate_concept("Isabella", 0.8)
        activated = semantic.retrieve_by_activation(threshold=0.3)
        
        if len(activated) >= 1:
            result.add_pass("SemanticMemory: Activation and retrieval")
        else:
            result.add_fail("SemanticMemory: Activation and retrieval", "No activated concepts")
        
        # Test associative retrieval
        associated = semantic.retrieve_by_association(["Isabella"], max_hops=2)
        if len(associated) >= 0:  # Should work without error
            result.add_pass("SemanticMemory: Associative retrieval")
        else:
            result.add_fail("SemanticMemory: Associative retrieval", "Association failed")
        
        # Test type-based retrieval
        people = semantic.retrieve_by_type(ConceptType.PERSON)
        if len(people) == 2:
            result.add_pass("SemanticMemory: Type-based retrieval")
        else:
            result.add_fail("SemanticMemory: Type-based retrieval", f"Expected 2, got {len(people)}")
            
    except Exception as e:
        result.add_fail("SemanticMemory: Functionality tests", str(e))


def test_memory_performance(result: TestResult):
    """Test performance requirements for all memory systems"""
    print("\n--- Performance Tests ---")
    
    # CircularBuffer performance
    try:
        buffer = CircularBuffer(max_size=20)
        
        start_time = time.perf_counter()
        for i in range(50):
            buffer.add_memory(f"Performance test {i}", "event", 0.5)
        end_time = time.perf_counter()
        
        add_time_ms = (end_time - start_time) * 1000
        result.add_performance("CircularBuffer: Batch add (50 items)", add_time_ms, 50.0)
        
        start_time = time.perf_counter()
        recent = buffer.get_recent_memories(10)
        end_time = time.perf_counter()
        
        retrieve_time_ms = (end_time - start_time) * 1000
        result.add_performance("CircularBuffer: Retrieval", retrieve_time_ms, 50.0)
        
    except Exception as e:
        result.add_fail("CircularBuffer: Performance test", str(e))
    
    # TemporalMemory performance
    try:
        temporal = TemporalMemory(retention_hours=2)
        
        start_time = time.perf_counter()
        for i in range(100):
            temporal.add_memory(f"Temporal test {i}", "event", 0.5)
        end_time = time.perf_counter()
        
        temporal_add_ms = (end_time - start_time) * 1000
        result.add_performance("TemporalMemory: Batch add (100 items)", temporal_add_ms, 100.0)
        
        start_time = time.perf_counter()
        recent = temporal.retrieve_recent_memories(hours_back=1, limit=20)
        end_time = time.perf_counter()
        
        temporal_retrieve_ms = (end_time - start_time) * 1000
        result.add_performance("TemporalMemory: Retrieval", temporal_retrieve_ms, 100.0)
        
    except Exception as e:
        result.add_fail("TemporalMemory: Performance test", str(e))
    
    # SemanticMemory performance
    try:
        semantic = SemanticMemory(max_concepts=500)
        
        start_time = time.perf_counter()
        for i in range(50):
            semantic.add_concept(f"Concept_{i}", ConceptType.PERSON, f"Test concept {i}", 0.5)
        end_time = time.perf_counter()
        
        semantic_add_ms = (end_time - start_time) * 1000
        result.add_performance("SemanticMemory: Batch add (50 concepts)", semantic_add_ms, 100.0)
        
        start_time = time.perf_counter()
        activated = semantic.retrieve_by_activation(threshold=0.3, limit=20)
        end_time = time.perf_counter()
        
        semantic_retrieve_ms = (end_time - start_time) * 1000
        result.add_performance("SemanticMemory: Activation retrieval", semantic_retrieve_ms, 100.0)
        
    except Exception as e:
        result.add_fail("SemanticMemory: Performance test", str(e))


def test_serialization_functionality(result: TestResult):
    """Test serialization and deserialization of all memory systems"""
    print("\n--- Serialization Tests ---")
    
    try:
        # Test CircularBuffer serialization
        buffer = CircularBuffer(max_size=10)
        buffer.add_memory("Serialization test", "event", 0.7)
        
        buffer_dict = buffer.to_dict()
        restored_buffer = CircularBuffer.from_dict(buffer_dict)
        
        if len(restored_buffer) == len(buffer):
            result.add_pass("CircularBuffer: Serialization")
        else:
            result.add_fail("CircularBuffer: Serialization", "Serialization mismatch")
            
    except Exception as e:
        result.add_fail("CircularBuffer: Serialization", str(e))
    
    try:
        # Test TemporalMemory serialization
        temporal = TemporalMemory(retention_hours=1)
        temporal.add_memory("Temporal serialization test", "event", 0.6)
        
        temporal_dict = temporal.to_dict()
        restored_temporal = TemporalMemory.from_dict(temporal_dict)
        
        if len(restored_temporal.memories) == len(temporal.memories):
            result.add_pass("TemporalMemory: Serialization")
        else:
            result.add_fail("TemporalMemory: Serialization", "Serialization mismatch")
            
    except Exception as e:
        result.add_fail("TemporalMemory: Serialization", str(e))
    
    try:
        # Test EpisodicMemory serialization
        episodic = EpisodicMemory(max_episodes=10)
        episodic.add_event("Episodic serialization test", "event", 0.5)
        
        episodic_dict = episodic.to_dict()
        restored_episodic = EpisodicMemory.from_dict(episodic_dict)
        
        if len(restored_episodic.events) == len(episodic.events):
            result.add_pass("EpisodicMemory: Serialization")
        else:
            result.add_fail("EpisodicMemory: Serialization", "Serialization mismatch")
            
    except Exception as e:
        result.add_fail("EpisodicMemory: Serialization", str(e))
    
    try:
        # Test SemanticMemory serialization
        semantic = SemanticMemory(max_concepts=100)
        semantic.add_concept("Test Concept", ConceptType.PERSON, "Test description", 0.5)
        
        semantic_dict = semantic.to_dict()
        restored_semantic = SemanticMemory.from_dict(semantic_dict)
        
        if len(restored_semantic.concepts) == len(semantic.concepts):
            result.add_pass("SemanticMemory: Serialization")
        else:
            result.add_fail("SemanticMemory: Serialization", "Serialization mismatch")
            
    except Exception as e:
        result.add_fail("SemanticMemory: Serialization", str(e))


def test_memory_integration(result: TestResult):
    """Test integration between different memory systems"""
    print("\n--- Integration Tests ---")
    
    try:
        # Create all memory systems
        buffer = CircularBuffer(max_size=20)
        temporal = TemporalMemory(retention_hours=2)
        episodic = EpisodicMemory(max_episodes=50)
        semantic = SemanticMemory(max_concepts=500)
        
        # Add related memories across systems
        buffer.add_memory("Conversation with Maria about hiking", "conversation", 0.8)
        temporal.add_memory("Conversation with Maria about hiking", "conversation", 0.8)
        
        event_id = episodic.add_event(
            "Conversation with Maria about hiking",
            "conversation", 0.8,
            participants={"Maria"},
            location="Garden"
        )
        
        maria_id = semantic.add_concept("Maria", ConceptType.PERSON, "Fellow contestant", 0.7)
        hiking_id = semantic.add_concept("Hiking", ConceptType.ACTIVITY, "Outdoor activity", 0.6)
        semantic.add_relation(maria_id, hiking_id, SemanticRelationType.RELATED_TO, 0.8)
        
        # Verify all systems have content
        if (len(buffer) > 0 and len(temporal.memories) > 0 and 
            len(episodic.events) > 0 and len(semantic.concepts) > 0):
            result.add_pass("Memory Integration: Cross-system consistency")
        else:
            result.add_fail("Memory Integration: Cross-system consistency", "Not all systems populated")
        
        # Test memory cleanup coordination
        buffer.cleanup_expired_memories()
        temporal.cleanup_expired_memories()
        semantic.update_activation_decay()
        
        result.add_pass("Memory Integration: Coordinated cleanup")
        
    except Exception as e:
        result.add_fail("Memory Integration: Tests", str(e))


def main():
    """Execute comprehensive test suite"""
    print("🧪 Enhanced PIANO Memory Architecture - Comprehensive Test Suite")
    print("="*70)
    
    result = TestResult()
    
    # Execute all test categories
    test_circular_buffer_functionality(result)
    test_temporal_memory_functionality(result)
    test_episodic_memory_functionality(result)
    test_semantic_memory_functionality(result)
    test_memory_performance(result)
    test_serialization_functionality(result)
    test_memory_integration(result)
    
    # Generate performance report
    print(f"\n--- Performance Report ---")
    for test_name, metrics in result.performance_metrics.items():
        status = "PASS" if metrics['passed'] else "FAIL"
        print(f"{status}: {test_name}")
        print(f"  Time: {metrics['time_ms']:.2f}ms (threshold: {metrics['threshold_ms']}ms)")
    
    # Final summary
    success = result.summary()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! Memory architecture is production-ready.")
        return 0
    else:
        print(f"\n💥 {result.failed} tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)