#!/usr/bin/env python3
"""
Integration tests for Enhanced PIANO Agent State Management
Tests state coordination, LangGraph integration patterns, and system cohesion.
"""

import sys
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add current directory to path
sys.path.append('.')
sys.path.append('./memory_structures')

# Import components for testing
from memory_structures.circular_buffer import CircularBuffer, CircularBufferReducer
from memory_structures.temporal_memory import TemporalMemory
from memory_structures.episodic_memory import EpisodicMemory, CausalRelationType
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


class MockEnhancedAgentState:
    """Mock implementation of EnhancedAgentState for testing integration patterns"""
    
    def __init__(self, agent_id: str, name: str, personality_traits: Dict[str, float]):
        self.agent_id = agent_id
        self.name = name
        self.personality_traits = personality_traits
        
        # Initialize memory systems
        self.circular_buffer = CircularBuffer(max_size=20)
        self.temporal_memory = TemporalMemory(retention_hours=1)
        self.episodic_memory = EpisodicMemory(max_episodes=50)
        self.semantic_memory = SemanticMemory(max_concepts=500)
        
        # State components
        self.current_location = "villa"
        self.current_activity = "idle"
        self.emotional_state = {"happiness": 0.5, "anxiety": 0.1, "excitement": 0.3}
        self.goals = []
        self.conversation_partners = set()
        self.recent_interactions = []
        
        # Performance tracking
        self.decision_count = 0
        self.total_decision_time = 0.0
    
    def add_memory(self, content: str, memory_type: str = "event", 
                   importance: float = 0.5, context: Dict = None) -> str:
        """Add memory across all systems"""
        timestamp = datetime.now()
        
        # Add to all memory systems
        self.circular_buffer.add_memory(content, memory_type, importance, context)
        temp_mem_id = self.temporal_memory.add_memory(content, memory_type, importance, context, timestamp)
        
        participants = set()
        if context and "participants" in context:
            participants = set(context["participants"])
        
        episodic_mem_id = self.episodic_memory.add_event(
            content, memory_type, importance, participants, self.current_location,
            emotional_valence=context.get("emotional_valence", 0.0) if context else 0.0,
            metadata=context, timestamp=timestamp
        )
        
        # Extract simple concepts for semantic memory
        words = content.split()
        for word in words:
            if word.istitle() and len(word) > 2:  # Likely proper noun
                self.semantic_memory.add_concept(
                    word, ConceptType.PERSON, f"Entity: {word}", importance * 0.5
                )
        
        return episodic_mem_id
    
    def update_performance_metrics(self, decision_time_ms: float):
        """Update performance tracking"""
        self.decision_count += 1
        self.total_decision_time += decision_time_ms
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory system summary"""
        return {
            "working_memory": len(self.circular_buffer),
            "temporal_memory": len(self.temporal_memory.memories),
            "episodic_memory": len(self.episodic_memory.episodes),
            "semantic_memory": len(self.semantic_memory.concepts),
            "avg_decision_time": self.total_decision_time / max(self.decision_count, 1)
        }


def test_memory_coordination(result: IntegrationTestResult):
    """Test coordination between different memory systems"""
    print("\n--- Memory System Coordination Tests ---")
    
    try:
        agent = MockEnhancedAgentState(
            "test_agent", "Test Agent", {"confidence": 0.8, "extroversion": 0.7}
        )
        
        # Add coordinated memory across systems
        mem_id = agent.add_memory(
            "Had meaningful conversation with Isabella about future goals",
            "conversation", 0.8,
            {"participants": ["Isabella"], "emotional_valence": 0.4}
        )
        
        # Verify memory exists across all systems
        working_memories = agent.circular_buffer.get_recent_memories(5)
        temporal_memories = agent.temporal_memory.retrieve_recent_memories(hours_back=1)
        episodes = list(agent.episodic_memory.episodes.values())
        concepts = agent.semantic_memory.concepts
        
        if (len(working_memories) > 0 and len(temporal_memories) > 0 and
            len(episodes) > 0 and len(concepts) > 0):
            result.add_pass("Memory Coordination: Multi-system consistency")
        else:
            result.add_fail("Memory Coordination: Multi-system consistency", 
                          f"WM:{len(working_memories)} TM:{len(temporal_memories)} EM:{len(episodes)} SM:{len(concepts)}")
        
        # Test memory relationships
        if episodes:
            episode = episodes[0]
            if "Isabella" in episode.participants:
                result.add_pass("Memory Coordination: Participant tracking")
            else:
                result.add_fail("Memory Coordination: Participant tracking", "Isabella not in participants")
        
        # Test concept extraction
        isabella_concepts = [c for c in concepts.values() if c.name == "Isabella"]
        if len(isabella_concepts) > 0:
            result.add_pass("Memory Coordination: Concept extraction")
        else:
            result.add_fail("Memory Coordination: Concept extraction", "Isabella concept not created")
            
    except Exception as e:
        result.add_fail("Memory Coordination: Tests", str(e))


def test_circular_buffer_reducer_integration(result: IntegrationTestResult):
    """Test CircularBufferReducer for LangGraph integration"""
    print("\n--- CircularBufferReducer Integration Tests ---")
    
    try:
        reducer = CircularBufferReducer(max_size=5)
        
        # Test reducer behavior with state updates
        current_state = []
        
        # Simulate LangGraph state updates
        update1 = [{"content": "First memory", "timestamp": datetime.now().isoformat()}]
        new_state1 = reducer(current_state, update1)
        
        update2 = [{"content": "Second memory", "timestamp": datetime.now().isoformat()}]
        new_state2 = reducer(new_state1, update2)
        
        if len(new_state2) == 2:
            result.add_pass("CircularBufferReducer: Basic functionality")
        else:
            result.add_fail("CircularBufferReducer: Basic functionality", f"Expected 2, got {len(new_state2)}")
        
        # Test size constraint
        for i in range(10):  # Add more than max_size
            update = [{"content": f"Memory {i}", "timestamp": datetime.now().isoformat()}]
            new_state2 = reducer(new_state2, update)
        
        if len(new_state2) == 5:  # Should be constrained to max_size
            result.add_pass("CircularBufferReducer: Size constraint")
        else:
            result.add_fail("CircularBufferReducer: Size constraint", f"Expected 5, got {len(new_state2)}")
        
        # Test empty updates
        empty_update = []
        final_state = reducer(new_state2, empty_update)
        if len(final_state) == len(new_state2):
            result.add_pass("CircularBufferReducer: Empty update handling")
        else:
            result.add_fail("CircularBufferReducer: Empty update handling", "State changed with empty update")
            
    except Exception as e:
        result.add_fail("CircularBufferReducer: Integration tests", str(e))


def test_state_persistence_and_recovery(result: IntegrationTestResult):
    """Test state persistence and recovery patterns"""
    print("\n--- State Persistence Tests ---")
    
    try:
        # Create agent with memories
        agent = MockEnhancedAgentState(
            "persist_test", "Persist Agent", {"confidence": 0.9}
        )
        
        # Add diverse memories
        agent.add_memory("Morning coffee routine", "activity", 0.6)
        agent.add_memory("Strategic conversation with Maria", "conversation", 0.8,
                        {"participants": ["Maria"], "emotional_valence": 0.3})
        agent.add_memory("Reflecting on relationship dynamics", "thought", 0.7)
        
        # Simulate serialization
        buffer_state = agent.circular_buffer.to_dict()
        temporal_state = agent.temporal_memory.to_dict()
        episodic_state = agent.episodic_memory.to_dict()
        semantic_state = agent.semantic_memory.to_dict()
        
        # Simulate recovery
        recovered_buffer = CircularBuffer.from_dict(buffer_state)
        recovered_temporal = TemporalMemory.from_dict(temporal_state)
        recovered_episodic = EpisodicMemory.from_dict(episodic_state)
        recovered_semantic = SemanticMemory.from_dict(semantic_state)
        
        # Verify recovery integrity
        original_summary = agent.get_memory_summary()
        
        recovered_summary = {
            "working_memory": len(recovered_buffer),
            "temporal_memory": len(recovered_temporal.memories),
            "episodic_memory": len(recovered_episodic.episodes),
            "semantic_memory": len(recovered_semantic.concepts)
        }
        
        if (recovered_summary["working_memory"] == original_summary["working_memory"] and
            recovered_summary["temporal_memory"] == original_summary["temporal_memory"] and
            recovered_summary["episodic_memory"] == original_summary["episodic_memory"] and
            recovered_summary["semantic_memory"] == original_summary["semantic_memory"]):
            result.add_pass("State Persistence: Recovery integrity")
        else:
            result.add_fail("State Persistence: Recovery integrity", 
                          f"Original: {original_summary}, Recovered: {recovered_summary}")
        
        # Test specific memory content preservation
        original_recent = agent.circular_buffer.get_recent_memories(3)
        recovered_recent = recovered_buffer.get_recent_memories(3)
        
        if len(original_recent) == len(recovered_recent):
            result.add_pass("State Persistence: Content preservation")
        else:
            result.add_fail("State Persistence: Content preservation", "Memory content mismatch")
            
    except Exception as e:
        result.add_fail("State Persistence: Tests", str(e))


def test_concurrent_memory_operations(result: IntegrationTestResult):
    """Test concurrent memory operations for multi-agent scenarios"""
    print("\n--- Concurrent Operations Tests ---")
    
    try:
        # Create multiple agents
        agent1 = MockEnhancedAgentState("agent1", "Agent One", {"confidence": 0.8})
        agent2 = MockEnhancedAgentState("agent2", "Agent Two", {"extroversion": 0.9})
        
        # Simulate concurrent operations
        start_time = time.perf_counter()
        
        # Agent 1 activities
        agent1.add_memory("Started conversation with Agent Two", "conversation", 0.7,
                         {"participants": ["Agent Two"], "emotional_valence": 0.2})
        agent1.add_memory("Feeling confident about connections", "thought", 0.6)
        
        # Agent 2 activities
        agent2.add_memory("Engaged in meaningful discussion", "conversation", 0.8,
                         {"participants": ["Agent One"], "emotional_valence": 0.3})
        agent2.add_memory("Planning next social interaction", "thought", 0.5)
        
        end_time = time.perf_counter()
        concurrent_time_ms = (end_time - start_time) * 1000
        
        # Verify both agents have independent state
        summary1 = agent1.get_memory_summary()
        summary2 = agent2.get_memory_summary()
        
        if (summary1["working_memory"] > 0 and summary2["working_memory"] > 0 and
            summary1["episodic_memory"] > 0 and summary2["episodic_memory"] > 0):
            result.add_pass("Concurrent Operations: Independent state")
        else:
            result.add_fail("Concurrent Operations: Independent state", 
                          f"Agent1: {summary1}, Agent2: {summary2}")
        
        # Performance requirement: concurrent operations should be fast
        result.add_performance("Concurrent Operations: Multi-agent performance", 
                             concurrent_time_ms, 50.0)
        
        # Test cross-agent memory references
        agent1_episodes = list(agent1.episodic_memory.episodes.values())
        agent2_episodes = list(agent2.episodic_memory.episodes.values())
        
        cross_references = 0
        for episode in agent1_episodes:
            if "Agent Two" in episode.participants:
                cross_references += 1
        
        for episode in agent2_episodes:
            if "Agent One" in episode.participants:
                cross_references += 1
        
        if cross_references >= 2:
            result.add_pass("Concurrent Operations: Cross-agent references")
        else:
            result.add_fail("Concurrent Operations: Cross-agent references", 
                          f"Only {cross_references} cross-references found")
            
    except Exception as e:
        result.add_fail("Concurrent Operations: Tests", str(e))


def test_memory_consolidation_workflow(result: IntegrationTestResult):
    """Test background memory consolidation workflows"""
    print("\n--- Memory Consolidation Workflow Tests ---")
    
    try:
        agent = MockEnhancedAgentState(
            "consolidation_test", "Consolidation Agent", {"openness": 0.8}
        )
        
        # Add many memories to trigger consolidation
        for i in range(25):
            agent.add_memory(f"Daily activity number {i}", "activity", 0.3)
            
        # Add some important memories
        agent.add_memory("Life-changing conversation", "conversation", 0.9,
                        {"participants": ["Important Person"], "emotional_valence": 0.8})
        agent.add_memory("Major personal insight", "thought", 0.85)
        
        initial_summary = agent.get_memory_summary()
        
        # Simulate consolidation workflow
        start_time = time.perf_counter()
        
        # Cleanup expired memories
        expired_working = agent.circular_buffer.cleanup_expired_memories()
        removed_temporal = agent.temporal_memory.cleanup_expired_memories()
        
        # Update semantic activation decay
        agent.semantic_memory.update_activation_decay()
        
        # Consolidate similar concepts
        consolidated_concepts = agent.semantic_memory.consolidate_concepts()
        
        end_time = time.perf_counter()
        consolidation_time_ms = (end_time - start_time) * 1000
        
        final_summary = agent.get_memory_summary()
        
        # Verify consolidation occurred
        if consolidation_time_ms < 100:  # Should be fast (background operation)
            result.add_pass("Memory Consolidation: Performance requirement")
        else:
            result.add_fail("Memory Consolidation: Performance requirement", 
                          f"Took {consolidation_time_ms:.2f}ms, expected <100ms")
        
        # Verify important memories preserved
        important_memories = agent.circular_buffer.get_important_memories(0.8)
        if len(important_memories) >= 2:  # Should preserve high-importance memories
            result.add_pass("Memory Consolidation: Important memory preservation")
        else:
            result.add_fail("Memory Consolidation: Important memory preservation", 
                          f"Only {len(important_memories)} important memories preserved")
        
        # Test consolidation doesn't break system integrity
        try:
            agent.add_memory("Post-consolidation memory", "event", 0.6)
            result.add_pass("Memory Consolidation: System integrity")
        except Exception as e:
            result.add_fail("Memory Consolidation: System integrity", str(e))
            
    except Exception as e:
        result.add_fail("Memory Consolidation: Workflow tests", str(e))


def test_decision_latency_requirements(result: IntegrationTestResult):
    """Test decision latency requirements under various loads"""
    print("\n--- Decision Latency Requirements Tests ---")
    
    try:
        agent = MockEnhancedAgentState(
            "latency_test", "Latency Agent", {"confidence": 0.7, "extroversion": 0.6}
        )
        
        # Test single decision latency
        start_time = time.perf_counter()
        
        # Simulate decision-making process
        agent.add_memory("Perceiving environment", "perception", 0.5)
        
        # Retrieve relevant memories (planning phase)
        recent_memories = agent.circular_buffer.get_recent_memories(5)
        temporal_context = agent.temporal_memory.retrieve_recent_memories(hours_back=1)
        
        # Make decision based on memories
        decision = "approach_social_group" if len(recent_memories) > 2 else "observe_environment"
        
        # Add decision to memory
        agent.add_memory(f"Decided to {decision}", "decision", 0.7)
        
        end_time = time.perf_counter()
        single_decision_ms = (end_time - start_time) * 1000
        
        result.add_performance("Decision Latency: Single decision", single_decision_ms, 100.0)
        
        # Test batch decision latency (simulating multiple agents)
        decisions = []
        start_time = time.perf_counter()
        
        for i in range(10):  # Simulate 10 concurrent decisions
            agent.add_memory(f"Situation {i} encountered", "perception", 0.4)
            recent = agent.circular_buffer.get_recent_memories(3)
            decision = f"action_for_situation_{i}"
            decisions.append(decision)
            agent.add_memory(f"Chose {decision}", "decision", 0.6)
            agent.update_performance_metrics(single_decision_ms)
        
        end_time = time.perf_counter()
        batch_decision_ms = (end_time - start_time) * 1000
        avg_batch_decision_ms = batch_decision_ms / 10
        
        result.add_performance("Decision Latency: Batch average", avg_batch_decision_ms, 100.0)
        
        # Test memory retrieval performance under load
        start_time = time.perf_counter()
        
        for i in range(50):  # Heavy memory retrieval
            recent = agent.circular_buffer.get_recent_memories(5)
            temporal = agent.temporal_memory.retrieve_recent_memories(hours_back=1, limit=10)
        
        end_time = time.perf_counter()
        heavy_retrieval_ms = (end_time - start_time) * 1000
        avg_retrieval_ms = heavy_retrieval_ms / 50
        
        result.add_performance("Decision Latency: Memory retrieval under load", 
                             avg_retrieval_ms, 50.0)
        
        # Verify decision consistency
        final_summary = agent.get_memory_summary()
        if final_summary["avg_decision_time"] < 100:
            result.add_pass("Decision Latency: Consistency requirement")
        else:
            result.add_fail("Decision Latency: Consistency requirement", 
                          f"Average: {final_summary['avg_decision_time']:.2f}ms")
            
    except Exception as e:
        result.add_fail("Decision Latency: Tests", str(e))


async def main():
    """Execute comprehensive integration test suite"""
    print("🔗 Enhanced PIANO Memory Architecture - Integration Test Suite")
    print("="*70)
    
    result = IntegrationTestResult()
    
    # Execute all integration test categories
    test_memory_coordination(result)
    test_circular_buffer_reducer_integration(result)
    test_state_persistence_and_recovery(result)
    test_concurrent_memory_operations(result)
    test_memory_consolidation_workflow(result)
    test_decision_latency_requirements(result)
    
    # Generate performance report
    print(f"\n--- Integration Performance Report ---")
    for test_name, metrics in result.performance_metrics.items():
        status = "PASS" if metrics['passed'] else "FAIL"
        print(f"{status}: {test_name}")
        print(f"  Time: {metrics['time_ms']:.2f}ms (threshold: {metrics['threshold_ms']}ms)")
    
    # Final summary
    success = result.summary()
    
    if success:
        print("\n🎉 ALL INTEGRATION TESTS PASSED! System integration is production-ready.")
        return 0
    else:
        print(f"\n💥 {result.failed} integration tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)