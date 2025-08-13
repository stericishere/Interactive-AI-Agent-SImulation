#!/usr/bin/env python3
"""
Comprehensive Test Suite for Goal Consistency System - Enhanced PIANO Architecture
Test-Driven Development for Task 3.1.2: Goal Consistency Measurement

Tests for dating_show/agents/specialization/goal_consistency.py
- Goal tracking and consistency scoring
- Role-goal alignment validation  
- Consistency-based role reinforcement
- Real-time performance <50ms processing
"""

import pytest
import asyncio
import time
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Import modules we're testing
from dating_show.agents.specialization.goal_consistency import (
    GoalConsistencyMeasurement,
    GoalTracker,
    ConsistencyAnalyzer,
    GoalConsistencyResult
)


class TestGoalTracker:
    """Test suite for GoalTracker class"""
    
    @pytest.fixture
    def goal_tracker(self):
        """Create GoalTracker instance for testing"""
        return GoalTracker(max_goal_history=50)
    
    @pytest.fixture
    def sample_actions(self):
        """Sample action data for testing"""
        return [
            {"action": "plan_event", "timestamp": time.time(), "success": True},
            {"action": "coordinate_team", "timestamp": time.time(), "success": True},
            {"action": "manage_resources", "timestamp": time.time(), "success": True},
            {"action": "socialize", "timestamp": time.time(), "success": True},
            {"action": "analyze_data", "timestamp": time.time(), "success": True},
            {"action": "optimize_process", "timestamp": time.time(), "success": True},
        ]
    
    def test_goal_tracker_initialization(self, goal_tracker):
        """Test GoalTracker initializes with correct parameters"""
        assert hasattr(goal_tracker, 'max_goal_history')
        assert hasattr(goal_tracker, 'goal_categories')
        assert goal_tracker.max_goal_history == 50
        assert len(goal_tracker.goal_categories) >= 5  # Should have multiple goal categories
    
    def test_extract_goals_from_text(self, goal_tracker):
        """Test goal extraction from text descriptions"""
        text = "I want to lead the team and manage our resources to achieve success"
        
        goals = goal_tracker.extract_goals_from_text(text)
        
        assert isinstance(goals, list)
        assert len(goals) > 0
        
        # Should extract management-related goals
        goal_types = [goal["goal_type"] for goal in goals]
        assert "management" in goal_types
        assert "achievement" in goal_types
        
        # Check goal structure
        for goal in goals:
            assert "goal_type" in goal
            assert "relevance_score" in goal
            assert "matched_keywords" in goal
            assert goal["relevance_score"] > 0
    
    def test_extract_goals_empty_text(self, goal_tracker):
        """Test goal extraction with empty or None text"""
        assert goal_tracker.extract_goals_from_text("") == []
        assert goal_tracker.extract_goals_from_text(None) == []
    
    def test_infer_goals_from_actions(self, goal_tracker, sample_actions):
        """Test goal inference from action patterns"""
        inferred_goals = goal_tracker.infer_goals_from_actions(sample_actions)
        
        assert isinstance(inferred_goals, list)
        assert len(inferred_goals) > 0
        
        # Should infer goals based on action patterns
        goal_types = [goal["goal_type"] for goal in inferred_goals]
        assert "management" in goal_types  # From plan_event, coordinate_team, manage_resources
        assert "social" in goal_types      # From socialize
        assert "analytical" in goal_types  # From analyze_data
        
        # Check goal structure
        for goal in inferred_goals:
            assert "goal_type" in goal
            assert "relevance_score" in goal
            assert "supporting_actions" in goal
            assert "extracted_from" in goal
            assert goal["extracted_from"] == "action_analysis"
    
    def test_track_goal_evolution(self, goal_tracker):
        """Test tracking of goal evolution over time"""
        goal_history = [
            {"goal_type": "management", "timestamp": time.time() - 3600, "relevance_score": 0.8},
            {"goal_type": "management", "timestamp": time.time() - 2400, "relevance_score": 0.7},
            {"goal_type": "social", "timestamp": time.time() - 1200, "relevance_score": 0.6},
            {"goal_type": "management", "timestamp": time.time() - 600, "relevance_score": 0.9},
            {"goal_type": "management", "timestamp": time.time(), "relevance_score": 0.8},
        ]
        
        evolution = goal_tracker.track_goal_evolution(goal_history)
        
        assert isinstance(evolution, dict)
        assert "evolution_patterns" in evolution
        assert "stability_score" in evolution
        assert "goal_changes" in evolution
        assert "dominant_goals" in evolution
        
        # Stability score should be between 0 and 1
        assert 0.0 <= evolution["stability_score"] <= 1.0
        
        # Should identify management as dominant goal
        dominant_goals = evolution["dominant_goals"]
        assert len(dominant_goals) > 0
        assert dominant_goals[0]["goal_type"] == "management"


class TestConsistencyAnalyzer:
    """Test suite for ConsistencyAnalyzer class"""
    
    @pytest.fixture
    def consistency_analyzer(self):
        """Create ConsistencyAnalyzer instance for testing"""
        return ConsistencyAnalyzer(consistency_window=20)
    
    @pytest.fixture
    def sample_goals(self):
        """Sample goals for testing"""
        return [
            {
                "goal_type": "management",
                "relevance_score": 0.8,
                "description": "Lead team effectively"
            },
            {
                "goal_type": "social",
                "relevance_score": 0.6,
                "description": "Build good relationships"
            }
        ]
    
    @pytest.fixture
    def sample_actions(self):
        """Sample actions for testing"""
        return [
            {"action": "plan_event", "timestamp": time.time(), "success": True},
            {"action": "coordinate_team", "timestamp": time.time(), "success": True},
            {"action": "socialize", "timestamp": time.time(), "success": True},
            {"action": "manage_resources", "timestamp": time.time(), "success": True},
        ]
    
    def test_consistency_analyzer_initialization(self, consistency_analyzer):
        """Test ConsistencyAnalyzer initializes correctly"""
        assert hasattr(consistency_analyzer, 'consistency_window')
        assert hasattr(consistency_analyzer, 'goal_tracker')
        assert consistency_analyzer.consistency_window == 20
    
    def test_calculate_goal_action_alignment(self, consistency_analyzer, sample_goals, sample_actions):
        """Test calculation of goal-action alignment scores"""
        alignment_scores = consistency_analyzer.calculate_goal_action_alignment(sample_goals, sample_actions)
        
        assert isinstance(alignment_scores, dict)
        assert "management" in alignment_scores
        assert "social" in alignment_scores
        
        # Management should have high alignment due to plan_event, coordinate_team, manage_resources
        assert alignment_scores["management"] > 0.5
        
        # Social should have some alignment due to socialize
        assert alignment_scores["social"] > 0.0
        
        # All scores should be between 0 and 1
        for score in alignment_scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_identify_inconsistent_actions(self, consistency_analyzer, sample_goals):
        """Test identification of actions that don't align with goals"""
        actions_with_inconsistent = [
            {"action": "plan_event", "timestamp": time.time(), "success": True},      # Consistent with management
            {"action": "socialize", "timestamp": time.time(), "success": True},       # Consistent with social
            {"action": "random_unrelated_action", "timestamp": time.time(), "success": True},  # Inconsistent
        ]
        
        inconsistent_actions = consistency_analyzer.identify_inconsistent_actions(sample_goals, actions_with_inconsistent)
        
        assert isinstance(inconsistent_actions, list)
        # Should identify the random action as inconsistent
        inconsistent_action_types = [action["action"] for action in inconsistent_actions]
        assert "random_unrelated_action" in inconsistent_action_types
        
        # Should not identify aligned actions as inconsistent
        assert "plan_event" not in inconsistent_action_types
        assert "socialize" not in inconsistent_action_types
    
    def test_calculate_temporal_consistency(self, consistency_analyzer):
        """Test calculation of temporal consistency"""
        agent_history = [
            {"type": "goal", "goal_type": "management", "timestamp": time.time() - 3600, "relevance_score": 0.8},
            {"type": "action", "action": "plan_event", "timestamp": time.time() - 3000, "success": True},
            {"type": "action", "action": "coordinate_team", "timestamp": time.time() - 2400, "success": True},
            {"type": "goal", "goal_type": "management", "timestamp": time.time() - 1800, "relevance_score": 0.7},
            {"type": "action", "action": "manage_resources", "timestamp": time.time() - 1200, "success": True},
        ]
        
        temporal_consistency = consistency_analyzer.calculate_temporal_consistency(agent_history)
        
        assert isinstance(temporal_consistency, float)
        assert 0.0 <= temporal_consistency <= 1.0
    
    def test_analyze_goal_pattern_strength(self, consistency_analyzer, sample_goals, sample_actions):
        """Test analysis of goal pattern strengths"""
        pattern_strengths = consistency_analyzer.analyze_goal_pattern_strength(sample_goals, sample_actions)
        
        assert isinstance(pattern_strengths, list)
        assert len(pattern_strengths) > 0
        
        for pattern in pattern_strengths:
            assert "goal_type" in pattern
            assert "pattern_strength" in pattern
            assert "goal_frequency" in pattern
            assert "combined_score" in pattern
            assert "matching_actions" in pattern
            assert "expected_actions" in pattern
            
            # Scores should be valid
            assert 0.0 <= pattern["pattern_strength"] <= 1.0
            assert pattern["goal_frequency"] >= 0.0


class TestGoalConsistencyMeasurement:
    """Test suite for main GoalConsistencyMeasurement class"""
    
    @pytest.fixture
    def consistency_measurer(self):
        """Create GoalConsistencyMeasurement instance for testing"""
        return GoalConsistencyMeasurement(
            consistency_threshold=0.7,
            temporal_window=20
        )
    
    @pytest.fixture
    def sample_agent_data(self):
        """Sample agent data for testing"""
        return {
            "agent_id": "test_agent_001",
            "goals": [
                {
                    "goal_type": "management",
                    "relevance_score": 0.8,
                    "description": "Lead team projects effectively"
                }
            ],
            "goal_description": "I want to organize and coordinate team activities",
            "action_history": [
                {"action": "plan_event", "timestamp": time.time() - 3600, "success": True},
                {"action": "coordinate_team", "timestamp": time.time() - 3000, "success": True},
                {"action": "delegate_task", "timestamp": time.time() - 2400, "success": True},
                {"action": "manage_resources", "timestamp": time.time() - 1800, "success": True},
                {"action": "socialize", "timestamp": time.time() - 1200, "success": True},  # Less consistent
                {"action": "evaluate_performance", "timestamp": time.time() - 600, "success": True},
            ]
        }
    
    def test_consistency_measurer_initialization(self, consistency_measurer):
        """Test GoalConsistencyMeasurement initializes correctly"""
        assert hasattr(consistency_measurer, 'consistency_threshold')
        assert hasattr(consistency_measurer, 'temporal_window')
        assert hasattr(consistency_measurer, 'goal_tracker')
        assert hasattr(consistency_measurer, 'consistency_analyzer')
        assert consistency_measurer.consistency_threshold == 0.7
    
    def test_measure_consistency_performance(self, consistency_measurer, sample_agent_data):
        """Test consistency measurement meets performance target <50ms"""
        start_time = time.time()
        
        result = consistency_measurer.measure_consistency(sample_agent_data)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        assert processing_time < 50, f"Consistency measurement took {processing_time}ms, exceeds 50ms target"
        assert isinstance(result, GoalConsistencyResult)
    
    def test_measure_consistency_comprehensive(self, consistency_measurer, sample_agent_data):
        """Test comprehensive consistency measurement"""
        result = consistency_measurer.measure_consistency(sample_agent_data)
        
        # Verify result structure
        assert result.agent_id == "test_agent_001"
        assert isinstance(result.consistency_score, float)
        assert 0.0 <= result.consistency_score <= 1.0
        assert isinstance(result.goal_alignment_scores, dict)
        assert isinstance(result.inconsistent_actions, list)
        assert isinstance(result.dominant_goal_patterns, list)
        assert isinstance(result.temporal_consistency, float)
        
        # Should have reasonable consistency due to management-focused actions
        assert result.consistency_score > 0.3
        
        # Should have management alignment
        if "management" in result.goal_alignment_scores:
            assert result.goal_alignment_scores["management"] > 0.5
    
    def test_measure_consistency_with_no_goals(self, consistency_measurer):
        """Test consistency measurement with agent having no clear goals"""
        agent_data = {
            "agent_id": "no_goals_agent",
            "goals": [],
            "goal_description": "",
            "action_history": [
                {"action": "random_action", "timestamp": time.time(), "success": True},
            ]
        }
        
        result = consistency_measurer.measure_consistency(agent_data)
        
        assert result.consistency_score == 0.0
        assert len(result.goal_alignment_scores) == 0
        assert result.agent_id == "no_goals_agent"
    
    def test_batch_measure_consistency(self, consistency_measurer):
        """Test batch consistency measurement for multiple agents"""
        agents_data = [
            {
                "agent_id": f"batch_agent_{i:02d}",
                "goals": [{"goal_type": "management", "relevance_score": 0.8}],
                "action_history": [
                    {"action": "plan_event", "timestamp": time.time(), "success": True},
                    {"action": "coordinate_team", "timestamp": time.time(), "success": True},
                ]
            }
            for i in range(5)
        ]
        
        results = consistency_measurer.batch_measure_consistency(agents_data)
        
        assert len(results) == 5
        assert all(isinstance(result, GoalConsistencyResult) for result in results)
        assert all(result.agent_id.startswith("batch_agent_") for result in results)
    
    def test_get_consistency_statistics(self, consistency_measurer):
        """Test calculation of consistency statistics"""
        results = [
            GoalConsistencyResult(
                agent_id=f"stats_agent_{i}",
                consistency_score=0.8 if i % 2 == 0 else 0.5,  # Mixed consistency
                goal_alignment_scores={"management": 0.7},
                inconsistent_actions=[],
                dominant_goal_patterns=[],
                temporal_consistency=0.6
            )
            for i in range(10)
        ]
        
        stats = consistency_measurer.get_consistency_statistics(results)
        
        assert isinstance(stats, dict)
        assert "total_agents" in stats
        assert "consistent_agents" in stats
        assert "consistency_rate" in stats
        assert "avg_consistency_score" in stats
        assert "avg_temporal_consistency" in stats
        assert "consistency_distribution" in stats
        
        assert stats["total_agents"] == 10
        # With threshold 0.7, only agents with 0.8 score should be consistent
        assert stats["consistent_agents"] == 5
        assert stats["consistency_rate"] == 0.5


class TestGoalConsistencyResult:
    """Test suite for GoalConsistencyResult data structure"""
    
    def test_result_creation(self):
        """Test creation of GoalConsistencyResult"""
        result = GoalConsistencyResult(
            agent_id="test_agent",
            consistency_score=0.85,
            goal_alignment_scores={"management": 0.8, "social": 0.6},
            inconsistent_actions=[],
            dominant_goal_patterns=[{"goal_type": "management", "pattern_strength": 0.9}],
            temporal_consistency=0.7
        )
        
        assert result.agent_id == "test_agent"
        assert result.consistency_score == 0.85
        assert result.goal_alignment_scores["management"] == 0.8
        assert result.temporal_consistency == 0.7
        assert result.timestamp is not None
    
    def test_is_consistent_method(self):
        """Test is_consistent method with different thresholds"""
        result = GoalConsistencyResult(
            agent_id="test_agent",
            consistency_score=0.75,
            goal_alignment_scores={},
            inconsistent_actions=[],
            dominant_goal_patterns=[],
            temporal_consistency=0.6
        )
        
        assert result.is_consistent(threshold=0.7)  # Should be consistent
        assert not result.is_consistent(threshold=0.8)  # Should not be consistent
    
    def test_result_serialization(self):
        """Test serialization of GoalConsistencyResult"""
        result = GoalConsistencyResult(
            agent_id="test_agent",
            consistency_score=0.75,
            goal_alignment_scores={"management": 0.8},
            inconsistent_actions=[{"action": "socialize", "inconsistency_reason": "not_aligned"}],
            dominant_goal_patterns=[],
            temporal_consistency=0.6
        )
        
        # Test JSON serialization
        json_str = result.to_json()
        assert isinstance(json_str, str)
        
        # Verify JSON can be parsed back
        parsed_data = json.loads(json_str)
        assert parsed_data["agent_id"] == "test_agent"
        assert parsed_data["consistency_score"] == 0.75
        
        # Test dict conversion
        dict_data = result.to_dict()
        assert isinstance(dict_data, dict)
        assert dict_data["agent_id"] == "test_agent"


class TestIntegrationScenarios:
    """Integration tests for complete goal consistency workflow"""
    
    @pytest.fixture
    def complete_system(self):
        """Set up complete goal consistency measurement system"""
        return GoalConsistencyMeasurement(
            consistency_threshold=0.7,
            temporal_window=15
        )
    
    def test_end_to_end_consistency_measurement(self, complete_system):
        """Test complete goal consistency workflow from raw data to measurement"""
        agent_data = {
            "agent_id": "integration_agent",
            "goals": [
                {"goal_type": "management", "relevance_score": 0.9, "description": "Lead projects"},
                {"goal_type": "social", "relevance_score": 0.6, "description": "Build relationships"}
            ],
            "goal_description": "I want to be an effective leader who also maintains good team relationships",
            "action_history": [
                {"action": "plan_event", "timestamp": time.time() - 7200, "success": True},
                {"action": "coordinate_team", "timestamp": time.time() - 6600, "success": True},
                {"action": "socialize", "timestamp": time.time() - 6000, "success": True},
                {"action": "delegate_task", "timestamp": time.time() - 5400, "success": True},
                {"action": "build_relationship", "timestamp": time.time() - 4800, "success": True},
                {"action": "manage_resources", "timestamp": time.time() - 4200, "success": True},
                {"action": "evaluate_performance", "timestamp": time.time() - 3600, "success": True},
                {"action": "organize_social_event", "timestamp": time.time() - 3000, "success": True},
            ],
            "history": [
                {"type": "goal", "goal_type": "management", "timestamp": time.time() - 8000, "relevance_score": 0.9},
                {"type": "action", "action": "plan_event", "timestamp": time.time() - 7200, "success": True},
                {"type": "action", "action": "coordinate_team", "timestamp": time.time() - 6600, "success": True},
                {"type": "goal", "goal_type": "social", "timestamp": time.time() - 6300, "relevance_score": 0.6},
                {"type": "action", "action": "socialize", "timestamp": time.time() - 6000, "success": True},
            ]
        }
        
        result = complete_system.measure_consistency(agent_data)
        
        # Comprehensive verification
        assert result.agent_id == "integration_agent"
        assert result.consistency_score > 0.2  # Should have some consistency
        assert len(result.goal_alignment_scores) >= 2  # Should measure both management and social
        assert isinstance(result.inconsistent_actions, list)
        assert isinstance(result.dominant_goal_patterns, list)
        assert result.temporal_consistency >= 0.0
        
        # Verify management alignment exists
        if "management" in result.goal_alignment_scores:
            assert result.goal_alignment_scores["management"] > 0.0
    
    def test_stress_testing_with_large_data(self, complete_system):
        """Test performance with large amounts of data"""
        # Create agent with large action history
        large_action_history = []
        for i in range(100):
            large_action_history.append({
                "action": f"action_{i % 10}",
                "timestamp": time.time() - (i * 60),
                "success": True
            })
        
        agent_data = {
            "agent_id": "stress_test_agent",
            "goals": [{"goal_type": "management", "relevance_score": 0.8}],
            "action_history": large_action_history
        }
        
        start_time = time.time()
        result = complete_system.measure_consistency(agent_data)
        processing_time = (time.time() - start_time) * 1000
        
        # Should still perform well with large data
        assert processing_time < 100  # Allow more time for large data
        assert result.agent_id == "stress_test_agent"
        assert isinstance(result.consistency_score, float)


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])