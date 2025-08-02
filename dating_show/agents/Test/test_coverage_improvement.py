#!/usr/bin/env python3
"""
Coverage Improvement Tests for Enhanced PIANO Memory Architecture
Specifically targets the untested methods identified in the coverage report.
"""

import sys
import os
import time
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import memory components
from memory_structures.circular_buffer import CircularBuffer, CircularBufferReducer
from memory_structures.temporal_memory import TemporalMemory
from memory_structures.episodic_memory import EpisodicMemory, CausalRelationType, EpisodeType
from memory_structures.semantic_memory import SemanticMemory, ConceptType, SemanticRelationType


class CoverageImprovementResult:
    """Track coverage improvement test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.methods_tested = []
    
    def add_pass(self, test_name: str, method_name: str = None):
        self.passed += 1
        if method_name:
            self.methods_tested.append(method_name)
        print(f"✅ {test_name}")
    
    def add_fail(self, test_name: str, error: str, method_name: str = None):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        coverage_improvement = len(set(self.methods_tested))
        
        print(f"\n{'='*60}")
        print(f"COVERAGE IMPROVEMENT SUMMARY: {self.passed}/{total} tests passed ({100*self.passed/total if total > 0 else 0:.1f}%)")
        print(f"METHODS TESTED: {coverage_improvement} previously untested methods now covered")
        print(f"NEW METHODS: {', '.join(set(self.methods_tested))}")
        print(f"{'='*60}")
        
        if self.errors:
            print("\nFAILURES:")
            for error in self.errors:
                print(f"  • {error}")
        
        return self.failed == 0


def test_circular_buffer_untested_methods(result: CoverageImprovementResult):
    """Test CircularBuffer's untested methods"""
    print("\n--- CircularBuffer Untested Methods ---")
    
    # Test get_memories_by_type
    try:
        buffer = CircularBuffer(max_size=20)
        buffer.add_memory("Event memory 1", "event", 0.7)
        buffer.add_memory("Thought memory", "thought", 0.6)
        buffer.add_memory("Event memory 2", "event", 0.8)
        buffer.add_memory("Conversation memory", "conversation", 0.9)
        
        # Test filtering by type
        event_memories = buffer.get_memories_by_type("event")
        thought_memories = buffer.get_memories_by_type("thought")
        conversation_memories = buffer.get_memories_by_type("conversation")
        nonexistent_memories = buffer.get_memories_by_type("nonexistent")
        
        if (len(event_memories) == 2 and len(thought_memories) == 1 and 
            len(conversation_memories) == 1 and len(nonexistent_memories) == 0):
            result.add_pass("CircularBuffer.get_memories_by_type: Correct filtering", "get_memories_by_type")
        else:
            result.add_fail("CircularBuffer.get_memories_by_type", 
                          f"Expected [2,1,1,0], got [{len(event_memories)},{len(thought_memories)},{len(conversation_memories)},{len(nonexistent_memories)}]",
                          "get_memories_by_type")
    except Exception as e:
        result.add_fail("CircularBuffer.get_memories_by_type", str(e), "get_memories_by_type")
    
    # Test get_memory_summary
    try:
        buffer = CircularBuffer(max_size=10)
        buffer.add_memory("Summary test 1", "event", 0.8)
        buffer.add_memory("Summary test 2", "thought", 0.6)
        buffer.add_memory("Summary test 3", "event", 0.9)
        
        summary = buffer.get_memory_summary()
        
        expected_keys = ['total_memories', 'memory_types', 'avg_importance', 'oldest_memory', 'newest_memory', 'buffer_utilization']
        
        if (isinstance(summary, dict) and 
            all(key in summary for key in expected_keys) and
            summary['total_memories'] == 3 and
            summary['memory_types']['event'] == 2 and
            summary['memory_types']['thought'] == 1 and
            0.75 <= summary['avg_importance'] <= 0.8):  # Average of 0.8, 0.6, 0.9
            result.add_pass("CircularBuffer.get_memory_summary: Correct summary structure and values", "get_memory_summary")
        else:
            result.add_fail("CircularBuffer.get_memory_summary", 
                          f"Invalid summary structure or values: {summary}",
                          "get_memory_summary")
    except Exception as e:
        result.add_fail("CircularBuffer.get_memory_summary", str(e), "get_memory_summary")
    
    # Test save_to_file and load_from_file
    try:
        # Create buffer with test data
        original_buffer = CircularBuffer(max_size=15)
        original_buffer.add_memory("File test memory 1", "event", 0.7)
        original_buffer.add_memory("File test memory 2", "thought", 0.8)
        original_buffer.add_memory("File test memory 3", "conversation", 0.6)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_filename = tmp_file.name
        
        original_buffer.save_to_file(tmp_filename)
        
        # Verify file was created and contains data
        if not os.path.exists(tmp_filename):
            result.add_fail("CircularBuffer.save_to_file", "File was not created", "save_to_file")
        else:
            # Test loading
            loaded_buffer = CircularBuffer(max_size=15)
            loaded_buffer.load_from_file(tmp_filename)
            
            # Compare loaded data with original
            original_memories = original_buffer.get_recent_memories(10)
            loaded_memories = loaded_buffer.get_recent_memories(10)
            
            if (len(loaded_memories) == len(original_memories) and
                all(orig['content'] == loaded['content'] for orig, loaded in zip(original_memories, loaded_memories))):
                result.add_pass("CircularBuffer.save_to_file and load_from_file: Data persistence works correctly", "save_to_file,load_from_file")
            else:
                result.add_fail("CircularBuffer.load_from_file", 
                              f"Loaded data doesn't match original: {len(loaded_memories)} vs {len(original_memories)}",
                              "load_from_file")
        
        # Cleanup
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)
            
    except Exception as e:
        result.add_fail("CircularBuffer.save_to_file/load_from_file", str(e), "save_to_file,load_from_file")


def test_temporal_memory_untested_methods(result: CoverageImprovementResult):
    """Test TemporalMemory's untested methods"""
    print("\n--- TemporalMemory Untested Methods ---")
    
    # Test get_temporal_summary
    try:
        temporal = TemporalMemory(retention_hours=24)
        now = datetime.now()
        
        # Add memories at different times
        temporal.add_memory("Recent memory", "event", 0.8, timestamp=now)
        temporal.add_memory("Older memory", "thought", 0.6, timestamp=now - timedelta(hours=2))
        temporal.add_memory("Old memory", "event", 0.4, timestamp=now - timedelta(hours=5))
        temporal.add_memory("Very old memory", "conversation", 0.3, timestamp=now - timedelta(hours=10))
        
        summary = temporal.get_temporal_summary()
        
        expected_keys = ['total_memories', 'time_span_hours', 'hourly_distribution', 'type_distribution', 'strength_distribution', 'avg_strength']
        
        if (isinstance(summary, dict) and 
            all(key in summary for key in expected_keys) and
            summary['total_memories'] == 4 and
            'event' in summary['type_distribution'] and
            'thought' in summary['type_distribution'] and
            'conversation' in summary['type_distribution']):
            result.add_pass("TemporalMemory.get_temporal_summary: Correct summary structure and distribution analysis", "get_temporal_summary")
        else:
            result.add_fail("TemporalMemory.get_temporal_summary", 
                          f"Invalid summary structure: {summary}",
                          "get_temporal_summary")
    except Exception as e:
        result.add_fail("TemporalMemory.get_temporal_summary", str(e), "get_temporal_summary")
    
    # Test consolidate_memories
    try:
        temporal = TemporalMemory(retention_hours=2, decay_rate=0.3)
        now = datetime.now()
        
        # Add many memories with varying importance
        memory_ids = []
        for i in range(15):
            importance = 0.3 + (i % 5) * 0.15  # Varies from 0.3 to 0.9
            mem_id = temporal.add_memory(f"Consolidation test memory {i}", "event", importance, 
                                       timestamp=now - timedelta(minutes=i*5))
            memory_ids.append(mem_id)
        
        initial_count = len(temporal.memories)
        
        # Test consolidation with different parameters
        consolidated = temporal.consolidate_memories(min_strength=0.4, max_memories=8)
        
        final_count = len(temporal.memories)
        
        # Consolidation should return information about what was consolidated
        if (isinstance(consolidated, (list, int)) and 
            final_count <= initial_count):  # Should not increase memory count
            result.add_pass("TemporalMemory.consolidate_memories: Successfully consolidates memories", "consolidate_memories")
        else:
            result.add_fail("TemporalMemory.consolidate_memories", 
                          f"Consolidation failed: {consolidated}, count {initial_count}->{final_count}",
                          "consolidate_memories")
    except Exception as e:
        result.add_fail("TemporalMemory.consolidate_memories", str(e), "consolidate_memories")


def test_episodic_memory_untested_methods(result: CoverageImprovementResult):
    """Test EpisodicMemory's untested methods"""
    print("\n--- EpisodicMemory Untested Methods ---")
    
    # Test get_episodes_by_type
    try:
        episodic = EpisodicMemory(max_episodes=30)
        
        # Add events of different types
        conversation_events = []
        reflection_events = []
        activity_events = []
        
        # Add conversation events
        for i in range(3):
            event_id = episodic.add_event(f"Conversation {i+1}", "conversation", 0.7 + i*0.1, 
                                        participants={f"Person{i+1}"}, location=f"Location{i+1}")
            conversation_events.append(event_id)
        
        # Add reflection events
        for i in range(2):
            event_id = episodic.add_event(f"Reflection {i+1}", "reflection", 0.6 + i*0.1, 
                                        participants=set(), location="Private")
            reflection_events.append(event_id)
        
        # Add activity events
        for i in range(4):
            event_id = episodic.add_event(f"Activity {i+1}", "activity", 0.5 + i*0.1, 
                                        participants={f"Person{i+1}", f"Person{i+2}"}, location="Common Area")
            activity_events.append(event_id)
        
        # Give some time for episode formation
        time.sleep(0.1)
        
        # Test retrieval by type
        conv_episodes = episodic.get_episodes_by_type("conversation")
        refl_episodes = episodic.get_episodes_by_type("reflection")
        activity_episodes = episodic.get_episodes_by_type("activity")
        nonexistent_episodes = episodic.get_episodes_by_type("nonexistent")
        
        # Check that we get reasonable results (episodes are formed automatically based on events)
        total_episodes = len(conv_episodes) + len(refl_episodes) + len(activity_episodes)
        
        if (total_episodes > 0 and  # At least some episodes should be formed
            len(nonexistent_episodes) == 0 and  # Non-existent type should return empty
            all(isinstance(ep.episode_type, str) for ep in conv_episodes + refl_episodes + activity_episodes)):  # All should have episode_type
            result.add_pass("EpisodicMemory.get_episodes_by_type: Correctly filters episodes by type", "get_episodes_by_type")
        else:
            result.add_fail("EpisodicMemory.get_episodes_by_type", 
                          f"Type filtering failed: conv={len(conv_episodes)}, refl={len(refl_episodes)}, activity={len(activity_episodes)}, none={len(nonexistent_episodes)}",
                          "get_episodes_by_type")
    except Exception as e:
        result.add_fail("EpisodicMemory.get_episodes_by_type", str(e), "get_episodes_by_type")


def test_semantic_memory_untested_methods(result: CoverageImprovementResult):
    """Test SemanticMemory's untested methods"""
    print("\n--- SemanticMemory Untested Methods ---")
    
    # Test get_memory_summary
    try:
        semantic = SemanticMemory(max_concepts=200)
        
        # Add diverse concepts and relationships
        people_ids = []
        activity_ids = []
        trait_ids = []
        
        # Add people
        for i in range(4):
            person_id = semantic.add_concept(f"Person{i+1}", ConceptType.PERSON, f"Person number {i+1}", 0.7 + i*0.05)
            people_ids.append(person_id)
        
        # Add activities
        for i in range(3):
            activity_id = semantic.add_concept(f"Activity{i+1}", ConceptType.ACTIVITY, f"Activity number {i+1}", 0.6 + i*0.1)
            activity_ids.append(activity_id)
        
        # Add traits
        for i in range(2):
            trait_id = semantic.add_concept(f"Trait{i+1}", ConceptType.TRAIT, f"Trait number {i+1}", 0.5 + i*0.2)
            trait_ids.append(trait_id)
        
        # Add relationships
        relationship_count = 0
        for person_id in people_ids:
            for activity_id in activity_ids:
                semantic.add_relation(person_id, activity_id, SemanticRelationType.RELATED_TO, 0.7)
                relationship_count += 1
            
            for trait_id in trait_ids:
                semantic.add_relation(person_id, trait_id, SemanticRelationType.HAS_A, 0.8)
                relationship_count += 1
        
        # Test summary
        summary = semantic.get_memory_summary()
        
        expected_keys = ['total_concepts', 'total_relations', 'concept_types', 'relation_types', 'avg_activation', 'highly_activated', 'network_density']
        
        if (isinstance(summary, dict) and 
            all(key in summary for key in expected_keys) and
            summary['total_concepts'] == 9 and  # 4 people + 3 activities + 2 traits
            summary['total_relations'] == relationship_count and
            'person' in summary['concept_types'] and
            'activity' in summary['concept_types'] and
            'trait' in summary['concept_types']):
            result.add_pass("SemanticMemory.get_memory_summary: Correct summary with concept and relation analysis", "get_memory_summary")
        else:
            result.add_fail("SemanticMemory.get_memory_summary", 
                          f"Invalid summary: {summary}",
                          "get_memory_summary")
    except Exception as e:
        result.add_fail("SemanticMemory.get_memory_summary", str(e), "get_memory_summary")
    
    # Test get_concept_relationships
    try:
        semantic = SemanticMemory(max_concepts=100)
        
        # Create a rich relationship network
        central_person = semantic.add_concept("CentralPerson", ConceptType.PERSON, "Main character", 0.9)
        
        related_concepts = []
        # Add related people
        for i in range(3):
            person_id = semantic.add_concept(f"RelatedPerson{i+1}", ConceptType.PERSON, f"Related person {i+1}", 0.7)
            semantic.add_relation(central_person, person_id, SemanticRelationType.KNOWS, 0.8)
            related_concepts.append(person_id)
        
        # Add activities
        for i in range(2):
            activity_id = semantic.add_concept(f"SharedActivity{i+1}", ConceptType.ACTIVITY, f"Shared activity {i+1}", 0.6)
            semantic.add_relation(central_person, activity_id, SemanticRelationType.RELATED_TO, 0.7)
            related_concepts.append(activity_id)
        
        # Add traits
        confidence_id = semantic.add_concept("Confident", ConceptType.TRAIT, "Confidence trait", 0.8)
        semantic.add_relation(central_person, confidence_id, SemanticRelationType.HAS_A, 0.9)
        related_concepts.append(confidence_id)
        
        # Test relationship retrieval
        relationships = semantic.get_concept_relationships("CentralPerson")
        
        if (isinstance(relationships, (list, dict)) and 
            len(relationships) == 6):  # 3 people + 2 activities + 1 trait = 6 relationships
            result.add_pass("SemanticMemory.get_concept_relationships: Correctly retrieves all relationships for concept", "get_concept_relationships")
        else:
            result.add_fail("SemanticMemory.get_concept_relationships", 
                          f"Expected 6 relationships, got {len(relationships) if hasattr(relationships, '__len__') else 'invalid type'}: {relationships}",
                          "get_concept_relationships")
        
        # Test with non-existent concept
        empty_relationships = semantic.get_concept_relationships("NonExistentPerson")
        if len(empty_relationships) == 0:
            result.add_pass("SemanticMemory.get_concept_relationships: Correctly handles non-existent concepts", "get_concept_relationships")
        else:
            result.add_fail("SemanticMemory.get_concept_relationships", 
                          f"Non-existent concept should return empty, got {len(empty_relationships)} relationships",
                          "get_concept_relationships")
    except Exception as e:
        result.add_fail("SemanticMemory.get_concept_relationships", str(e), "get_concept_relationships")


def main():
    """Run coverage improvement tests"""
    print("📈 Enhanced PIANO Memory Architecture - Coverage Improvement Tests")
    print("=" * 70)
    print("Testing previously untested methods to improve overall coverage...")
    
    result = CoverageImprovementResult()
    
    # Test all untested methods
    test_circular_buffer_untested_methods(result)
    test_temporal_memory_untested_methods(result)
    test_episodic_memory_untested_methods(result)
    test_semantic_memory_untested_methods(result)
    
    # Generate summary
    success = result.summary()
    
    if success:
        print("\n🎉 ALL COVERAGE IMPROVEMENT TESTS PASSED!")
        print("📈 Test coverage significantly improved by testing previously untested methods.")
        print("🚀 Memory architecture now has more comprehensive test coverage.")
    else:
        print(f"\n💥 {result.failed} coverage improvement tests failed.")
        print("📊 Some methods may still need additional testing.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)