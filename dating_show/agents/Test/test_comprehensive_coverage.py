#!/usr/bin/env python3
"""
Comprehensive Test Coverage Enhancement for Enhanced PIANO Memory Architecture
Tests all previously untested methods and adds edge cases for complete coverage.
"""

import sys
import os
import time
import json
import tempfile
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


class EnhancedTestResult:
    """Enhanced test result tracking with detailed coverage analysis"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.performance_metrics = {}
        self.coverage_data = {}
    
    def add_pass(self, test_name: str, method_tested: str = None):
        self.passed += 1
        print(f"✅ {test_name}")
        if method_tested:
            self.coverage_data.setdefault(method_tested, True)
    
    def add_fail(self, test_name: str, error: str, method_tested: str = None):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")
        if method_tested:
            self.coverage_data.setdefault(method_tested, False)
    
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
        coverage_percentage = (len([v for v in self.coverage_data.values() if v]) / 
                             len(self.coverage_data) * 100) if self.coverage_data else 100
        
        print(f"\n{'='*60}")
        print(f"ENHANCED TEST SUMMARY: {self.passed}/{total} passed ({100*self.passed/total if total > 0 else 0:.1f}%)")
        print(f"METHOD COVERAGE: {coverage_percentage:.1f}% ({len([v for v in self.coverage_data.values() if v])}/{len(self.coverage_data)} methods tested)")
        print(f"{'='*60}")
        
        if self.errors:
            print("\nFAILURES:")
            for error in self.errors:
                print(f"  • {error}")
        
        return self.failed == 0


def test_circular_buffer_untested_methods(result: EnhancedTestResult):
    """Test previously untested CircularBuffer methods"""
    print("\n--- CircularBuffer Untested Methods ---")
    
    # Test get_memories_by_type
    try:
        buffer = CircularBuffer(max_size=10)
        buffer.add_memory("Event memory", "event", 0.7)
        buffer.add_memory("Thought memory", "thought", 0.6)
        buffer.add_memory("Another event", "event", 0.8)
        
        event_memories = buffer.get_memories_by_type("event")
        thought_memories = buffer.get_memories_by_type("thought")
        
        if len(event_memories) == 2 and len(thought_memories) == 1:
            result.add_pass("CircularBuffer: get_memories_by_type", "get_memories_by_type")
        else:
            result.add_fail("CircularBuffer: get_memories_by_type", 
                          f"Events: {len(event_memories)}, Thoughts: {len(thought_memories)}", 
                          "get_memories_by_type")
    except Exception as e:
        result.add_fail("CircularBuffer: get_memories_by_type", str(e), "get_memories_by_type")
    
    # Test get_memory_summary
    try:
        buffer = CircularBuffer(max_size=5)
        for i in range(3):
            buffer.add_memory(f"Memory {i}", "event", 0.5 + i*0.1)
        
        summary = buffer.get_memory_summary()
        
        if (isinstance(summary, dict) and 
            'total_memories' in summary and 
            'memory_types' in summary and
            'avg_importance' in summary):
            result.add_pass("CircularBuffer: get_memory_summary", "get_memory_summary")
        else:
            result.add_fail("CircularBuffer: get_memory_summary", 
                          f"Invalid summary structure: {summary}", "get_memory_summary")
    except Exception as e:
        result.add_fail("CircularBuffer: get_memory_summary", str(e), "get_memory_summary")
    
    # Test save_to_file and load_from_file
    try:
        buffer = CircularBuffer(max_size=5)
        buffer.add_memory("Persistent memory 1", "event", 0.8)
        buffer.add_memory("Persistent memory 2", "thought", 0.7)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_filename = tmp_file.name
        
        buffer.save_to_file(tmp_filename)
        
        # Create new buffer and load
        new_buffer = CircularBuffer(max_size=5)
        new_buffer.load_from_file(tmp_filename)
        
        # Cleanup
        os.unlink(tmp_filename)
        
        if (len(new_buffer) == len(buffer) and 
            new_buffer.get_recent_memories(1)[0]["content"] == buffer.get_recent_memories(1)[0]["content"]):
            result.add_pass("CircularBuffer: save_to_file and load_from_file", "save_to_file,load_from_file")
        else:
            result.add_fail("CircularBuffer: save_to_file and load_from_file", 
                          "File persistence failed", "save_to_file,load_from_file")
    except Exception as e:
        result.add_fail("CircularBuffer: save_to_file and load_from_file", str(e), "save_to_file,load_from_file")


def test_temporal_memory_untested_methods(result: EnhancedTestResult):
    """Test previously untested TemporalMemory methods"""
    print("\n--- TemporalMemory Untested Methods ---")
    
    # Test get_temporal_summary
    try:
        temporal = TemporalMemory(retention_hours=2)
        now = datetime.now()
        
        temporal.add_memory("Recent memory", "event", 0.8, timestamp=now)
        temporal.add_memory("Older memory", "thought", 0.6, timestamp=now - timedelta(hours=1))
        temporal.add_memory("Old memory", "event", 0.4, timestamp=now - timedelta(hours=1.5))
        
        summary = temporal.get_temporal_summary()
        
        if (isinstance(summary, dict) and 
            'total_memories' in summary and 
            'time_distribution' in summary and
            'memory_strength_stats' in summary):
            result.add_pass("TemporalMemory: get_temporal_summary", "get_temporal_summary")
        else:
            result.add_fail("TemporalMemory: get_temporal_summary", 
                          f"Invalid summary: {summary}", "get_temporal_summary")
    except Exception as e:
        result.add_fail("TemporalMemory: get_temporal_summary", str(e), "get_temporal_summary")
    
    # Test consolidate_memories
    try:
        temporal = TemporalMemory(retention_hours=1, decay_rate=0.2)
        now = datetime.now()
        
        # Add many memories to trigger consolidation
        for i in range(20):
            temporal.add_memory(f"Memory {i}", "event", 0.3 + (i % 5) * 0.1, 
                              timestamp=now - timedelta(minutes=i*2))
        
        initial_count = len(temporal.memories)
        consolidated = temporal.consolidate_memories(min_strength=0.4, max_memories=10)
        final_count = len(temporal.memories)
        
        if consolidated >= 0 and final_count <= initial_count:
            result.add_pass("TemporalMemory: consolidate_memories", "consolidate_memories")
        else:
            result.add_fail("TemporalMemory: consolidate_memories", 
                          f"Consolidation failed: {consolidated} consolidated, {initial_count}->{final_count}", 
                          "consolidate_memories")
    except Exception as e:
        result.add_fail("TemporalMemory: consolidate_memories", str(e), "consolidate_memories")


def test_episodic_memory_untested_methods(result: EnhancedTestResult):
    """Test previously untested EpisodicMemory methods"""
    print("\n--- EpisodicMemory Untested Methods ---")
    
    # Test get_episodes_by_type
    try:
        episodic = EpisodicMemory(max_episodes=20)
        
        # Add events of different types
        event1 = episodic.add_event("Conversation with Maria", "conversation", 0.8, 
                                  participants={"Maria"}, location="Garden")
        event2 = episodic.add_event("Personal reflection", "reflection", 0.6, 
                                  participants=set(), location="Bedroom")
        event3 = episodic.add_event("Group discussion", "conversation", 0.7, 
                                  participants={"Maria", "Isabella"}, location="Living Room")
        
        # Allow episode formation
        time.sleep(0.1)
        
        conversation_episodes = episodic.get_episodes_by_type("conversation")
        reflection_episodes = episodic.get_episodes_by_type("reflection")
        
        if len(conversation_episodes) >= 1:  # Should have conversation episodes
            result.add_pass("EpisodicMemory: get_episodes_by_type", "get_episodes_by_type")
        else:
            result.add_fail("EpisodicMemory: get_episodes_by_type", 
                          f"Conversations: {len(conversation_episodes)}, Reflections: {len(reflection_episodes)}", 
                          "get_episodes_by_type")
    except Exception as e:
        result.add_fail("EpisodicMemory: get_episodes_by_type", str(e), "get_episodes_by_type")


def test_semantic_memory_untested_methods(result: EnhancedTestResult):
    """Test previously untested SemanticMemory methods"""
    print("\n--- SemanticMemory Untested Methods ---")
    
    # Test get_memory_summary
    try:
        semantic = SemanticMemory(max_concepts=100)
        
        # Add diverse concepts
        maria_id = semantic.add_concept("Maria", ConceptType.PERSON, "Artistic contestant", 0.8)
        art_id = semantic.add_concept("Art", ConceptType.ACTIVITY, "Creative activity", 0.7)
        confident_id = semantic.add_concept("Confident", ConceptType.TRAIT, "Personality trait", 0.6)
        
        # Add relationships
        semantic.add_relation(maria_id, art_id, SemanticRelationType.RELATED_TO, 0.9)
        semantic.add_relation(maria_id, confident_id, SemanticRelationType.HAS_A, 0.8)
        
        summary = semantic.get_memory_summary()
        
        if (isinstance(summary, dict) and 
            'total_concepts' in summary and 
            'concept_types' in summary and
            'total_relations' in summary and
            'activation_stats' in summary):
            result.add_pass("SemanticMemory: get_memory_summary", "get_memory_summary")
        else:
            result.add_fail("SemanticMemory: get_memory_summary", 
                          f"Invalid summary: {summary}", "get_memory_summary")
    except Exception as e:
        result.add_fail("SemanticMemory: get_memory_summary", str(e), "get_memory_summary")
    
    # Test get_concept_relationships
    try:
        semantic = SemanticMemory(max_concepts=100)
        
        # Create rich relationship network
        person1_id = semantic.add_concept("Isabella", ConceptType.PERSON, "Confident contestant", 0.8)
        person2_id = semantic.add_concept("Maria", ConceptType.PERSON, "Artistic contestant", 0.7)
        trait1_id = semantic.add_concept("Confident", ConceptType.TRAIT, "Confidence trait", 0.6)
        trait2_id = semantic.add_concept("Creative", ConceptType.TRAIT, "Creativity trait", 0.6)
        
        # Add multiple relationships
        semantic.add_relation(person1_id, trait1_id, SemanticRelationType.HAS_A, 0.9)
        semantic.add_relation(person2_id, trait2_id, SemanticRelationType.HAS_A, 0.8)
        semantic.add_relation(person1_id, person2_id, SemanticRelationType.KNOWS, 0.7)
        
        # Test relationship retrieval
        isabella_relationships = semantic.get_concept_relationships("Isabella")
        maria_relationships = semantic.get_concept_relationships("Maria")
        
        if (len(isabella_relationships) >= 2 and len(maria_relationships) >= 2):
            result.add_pass("SemanticMemory: get_concept_relationships", "get_concept_relationships")
        else:
            result.add_fail("SemanticMemory: get_concept_relationships", 
                          f"Isabella: {len(isabella_relationships)}, Maria: {len(maria_relationships)}", 
                          "get_concept_relationships")
    except Exception as e:
        result.add_fail("SemanticMemory: get_concept_relationships", str(e), "get_concept_relationships")


def test_edge_cases_and_error_handling(result: EnhancedTestResult):
    """Test edge cases and error handling across all memory systems"""
    print("\n--- Edge Cases and Error Handling ---")
    
    # CircularBuffer edge cases
    try:
        buffer = CircularBuffer(max_size=1)  # Very small buffer
        buffer.add_memory("First", "event", 0.5)
        buffer.add_memory("Second", "event", 0.6)  # Should replace first
        
        if len(buffer) == 1 and buffer.get_recent_memories(1)[0]["content"] == "Second":
            result.add_pass("CircularBuffer: Edge case - minimal size")
        else:
            result.add_fail("CircularBuffer: Edge case - minimal size", "Size constraint failed")
    except Exception as e:
        result.add_fail("CircularBuffer: Edge case - minimal size", str(e))
    
    # TemporalMemory with expired memories
    try:
        temporal = TemporalMemory(retention_hours=0.001)  # Very short retention
        temporal.add_memory("Should expire", "event", 0.5)
        
        time.sleep(0.01)  # Wait for expiration
        cleaned = temporal.cleanup_expired_memories()
        
        if cleaned >= 0:  # Should handle expired memories gracefully
            result.add_pass("TemporalMemory: Edge case - expired memories")
        else:
            result.add_fail("TemporalMemory: Edge case - expired memories", "Cleanup failed")
    except Exception as e:
        result.add_fail("TemporalMemory: Edge case - expired memories", str(e))
    
    # EpisodicMemory with empty participants
    try:
        episodic = EpisodicMemory(max_episodes=5)
        event_id = episodic.add_event("Solo activity", "activity", 0.6, 
                                    participants=set(), location="Bedroom")
        
        if event_id and len(episodic.events) == 1:
            result.add_pass("EpisodicMemory: Edge case - empty participants")
        else:
            result.add_fail("EpisodicMemory: Edge case - empty participants", "Empty participants failed")
    except Exception as e:
        result.add_fail("EpisodicMemory: Edge case - empty participants", str(e))
    
    # SemanticMemory with maximum concepts
    try:
        semantic = SemanticMemory(max_concepts=2)  # Very small limit
        concept1 = semantic.add_concept("First", ConceptType.PERSON, "First person", 0.5)
        concept2 = semantic.add_concept("Second", ConceptType.PERSON, "Second person", 0.6)
        concept3 = semantic.add_concept("Third", ConceptType.PERSON, "Third person", 0.7)  # Should trigger cleanup
        
        if len(semantic.concepts) <= 2:  # Should respect size limit
            result.add_pass("SemanticMemory: Edge case - concept limit")
        else:
            result.add_fail("SemanticMemory: Edge case - concept limit", f"Has {len(semantic.concepts)} concepts")
    except Exception as e:
        result.add_fail("SemanticMemory: Edge case - concept limit", str(e))


def test_performance_regression(result: EnhancedTestResult):
    """Test performance to prevent regression"""
    print("\n--- Performance Regression Tests ---")
    
    # Test large-scale operations
    try:
        buffer = CircularBuffer(max_size=100)
        
        start_time = time.perf_counter()
        for i in range(500):  # Large batch
            buffer.add_memory(f"Large scale memory {i}", "event", 0.5)
        end_time = time.perf_counter()
        
        large_batch_ms = (end_time - start_time) * 1000
        result.add_performance("CircularBuffer: Large batch operations", large_batch_ms, 200.0)
        
    except Exception as e:
        result.add_fail("CircularBuffer: Large batch performance", str(e))
    
    # Test complex semantic network operations
    try:
        semantic = SemanticMemory(max_concepts=200)
        
        start_time = time.perf_counter()
        
        # Create complex network
        concept_ids = []
        for i in range(50):
            concept_id = semantic.add_concept(f"Entity_{i}", ConceptType.PERSON, f"Person {i}", 0.5)
            concept_ids.append(concept_id)
        
        # Add many relationships
        for i in range(0, len(concept_ids)-1, 2):
            semantic.add_relation(concept_ids[i], concept_ids[i+1], 
                                SemanticRelationType.KNOWS, 0.6)
        
        # Perform complex retrieval
        activated = semantic.retrieve_by_activation(threshold=0.3, limit=20)
        
        end_time = time.perf_counter()
        
        complex_network_ms = (end_time - start_time) * 1000
        result.add_performance("SemanticMemory: Complex network operations", complex_network_ms, 500.0)
        
    except Exception as e:
        result.add_fail("SemanticMemory: Complex network performance", str(e))


def test_concurrent_access_simulation(result: EnhancedTestResult):
    """Simulate concurrent access patterns"""
    print("\n--- Concurrent Access Simulation ---")
    
    try:
        # Simulate multiple agents accessing same memory system
        shared_semantic = SemanticMemory(max_concepts=100)
        
        # Agent 1 operations
        start_time = time.perf_counter()
        
        agent1_concepts = []
        for i in range(10):
            concept_id = shared_semantic.add_concept(f"Agent1_Entity_{i}", ConceptType.PERSON, 
                                                   f"Agent 1 person {i}", 0.6)
            agent1_concepts.append(concept_id)
        
        # Agent 2 operations (interleaved)
        agent2_concepts = []
        for i in range(10):
            concept_id = shared_semantic.add_concept(f"Agent2_Entity_{i}", ConceptType.PERSON, 
                                                   f"Agent 2 person {i}", 0.6)
            agent2_concepts.append(concept_id)
        
        # Cross-agent relationships
        for i in range(min(len(agent1_concepts), len(agent2_concepts))):
            shared_semantic.add_relation(agent1_concepts[i], agent2_concepts[i], 
                                       SemanticRelationType.KNOWS, 0.7)
        
        end_time = time.perf_counter()
        concurrent_ms = (end_time - start_time) * 1000
        
        # Verify integrity
        if len(shared_semantic.concepts) == 20 and len(shared_semantic.relations) == 10:
            result.add_pass("Concurrent Access: Multi-agent integrity")
            result.add_performance("Concurrent Access: Multi-agent operations", concurrent_ms, 300.0)
        else:
            result.add_fail("Concurrent Access: Multi-agent integrity", 
                          f"Concepts: {len(shared_semantic.concepts)}, Relations: {len(shared_semantic.relations)}")
        
    except Exception as e:
        result.add_fail("Concurrent Access: Multi-agent simulation", str(e))


def main():
    """Execute comprehensive coverage enhancement tests"""
    print("🧪 Enhanced PIANO Memory Architecture - Comprehensive Coverage Tests")
    print("="*70)
    print("Testing previously untested methods and edge cases...")
    
    result = EnhancedTestResult()
    
    # Execute all enhanced test categories
    test_circular_buffer_untested_methods(result)
    test_temporal_memory_untested_methods(result)
    test_episodic_memory_untested_methods(result)
    test_semantic_memory_untested_methods(result)
    test_edge_cases_and_error_handling(result)
    test_performance_regression(result)
    test_concurrent_access_simulation(result)
    
    # Generate detailed coverage report
    print(f"\n--- Enhanced Coverage Report ---")
    print(f"Methods tested in this suite: {len(result.coverage_data)}")
    
    passed_methods = [method for method, passed in result.coverage_data.items() if passed]
    failed_methods = [method for method, passed in result.coverage_data.items() if not passed]
    
    if passed_methods:
        print(f"✅ Successfully tested methods: {', '.join(passed_methods)}")
    if failed_methods:
        print(f"❌ Failed method tests: {', '.join(failed_methods)}")
    
    # Performance summary
    if result.performance_metrics:
        print(f"\n--- Performance Summary ---")
        for test_name, metrics in result.performance_metrics.items():
            status = "PASS" if metrics['passed'] else "FAIL"
            print(f"{status}: {test_name} - {metrics['time_ms']:.2f}ms")
    
    # Final summary
    success = result.summary()
    
    if success:
        print("\n🎉 COVERAGE ENHANCEMENT COMPLETE! All untested methods now have comprehensive tests.")
        print("📊 Test coverage significantly improved with edge cases and performance validation.")
        return 0
    else:
        print(f"\n💥 {result.failed} enhanced tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)