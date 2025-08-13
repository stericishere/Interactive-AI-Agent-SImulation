#!/usr/bin/env python3
"""
Test Suite for Role Detection Algorithm - Enhanced PIANO Architecture
Test-Driven Development for Week 3: Specialization System Implementation

Tests for dating_show/agents/specialization/role_detector.py
- Statistical analysis of agent action frequencies
- Pattern recognition for professional behaviors
- Role classification algorithms with >80% accuracy target
- Real-time performance <50ms processing
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import numpy as np
from datetime import datetime, timedelta
import json

# Import the module we're testing (will be implemented after tests)
try:
    from dating_show.agents.specialization.role_detector import (
        RoleDetector, 
        ActionPatternAnalyzer,
        ProfessionalBehaviorClassifier,
        RoleClassificationResult
    )
except ImportError:
    # Mock classes for initial test development
    class RoleDetector:
        pass
    class ActionPatternAnalyzer:
        pass
    class ProfessionalBehaviorClassifier:
        pass
    class RoleClassificationResult:
        pass


class TestRoleDetector:
    """Test suite for the main RoleDetector class"""
    
    @pytest.fixture
    def sample_agent_data(self):
        """Sample agent data for testing"""
        return {
            "agent_id": "agent_001",
            "name": "TestAgent",
            "action_history": [
                {"action": "plan_event", "timestamp": time.time(), "success": True},
                {"action": "coordinate_team", "timestamp": time.time(), "success": True},
                {"action": "manage_resources", "timestamp": time.time(), "success": True},
                {"action": "plan_event", "timestamp": time.time(), "success": True},
                {"action": "delegate_task", "timestamp": time.time(), "success": True},
                {"action": "socialize", "timestamp": time.time(), "success": True},
                {"action": "plan_event", "timestamp": time.time(), "success": True},
                {"action": "coordinate_team", "timestamp": time.time(), "success": True},
            ],
            "specialization": {
                "current_role": None,
                "role_history": [],
                "skills": {},
                "expertise_level": 0.0
            }
        }
    
    @pytest.fixture
    def role_detector(self):
        """Create RoleDetector instance for testing"""
        return RoleDetector(
            min_actions_for_detection=5,
            accuracy_threshold=0.8,
            confidence_threshold=0.7
        )
    
    def test_role_detector_initialization(self, role_detector):
        """Test RoleDetector initializes with correct parameters"""
        assert hasattr(role_detector, 'min_actions_for_detection')
        assert hasattr(role_detector, 'accuracy_threshold')
        assert hasattr(role_detector, 'confidence_threshold')
        assert role_detector.accuracy_threshold == 0.8
        assert role_detector.confidence_threshold == 0.7
        
    def test_detect_role_with_insufficient_data(self, role_detector):
        """Test role detection fails gracefully with insufficient action data"""
        sparse_data = {
            "agent_id": "sparse_agent",
            "action_history": [
                {"action": "socialize", "timestamp": time.time(), "success": True}
            ]
        }
        
        result = role_detector.detect_role(sparse_data)
        assert result.confidence < role_detector.confidence_threshold
        assert result.detected_role is None or result.detected_role == "undetermined"
        
    def test_detect_role_performance_benchmark(self, role_detector, sample_agent_data):
        """Test role detection meets <50ms performance target"""
        start_time = time.time()
        
        result = role_detector.detect_role(sample_agent_data)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        assert processing_time < 50, f"Role detection took {processing_time}ms, exceeds 50ms target"
        
    def test_detect_role_manager_pattern(self, role_detector):
        """Test detection of manager role from action patterns"""
        manager_data = {
            "agent_id": "manager_001",
            "action_history": [
                {"action": "plan_event", "timestamp": time.time(), "success": True},
                {"action": "coordinate_team", "timestamp": time.time(), "success": True},
                {"action": "delegate_task", "timestamp": time.time(), "success": True},
                {"action": "manage_resources", "timestamp": time.time(), "success": True},
                {"action": "plan_event", "timestamp": time.time(), "success": True},
                {"action": "coordinate_team", "timestamp": time.time(), "success": True},
                {"action": "delegate_task", "timestamp": time.time(), "success": True},
                {"action": "evaluate_performance", "timestamp": time.time(), "success": True},
            ]
        }
        
        result = role_detector.detect_role(manager_data)
        
        assert result.detected_role == "manager"
        assert result.confidence >= 0.7
        assert "leadership" in result.supporting_evidence
        assert "coordination" in result.supporting_evidence
        
    def test_detect_role_socializer_pattern(self, role_detector):
        """Test detection of socializer role from action patterns"""
        socializer_data = {
            "agent_id": "socializer_001", 
            "action_history": [
                {"action": "socialize", "timestamp": time.time(), "success": True},
                {"action": "build_relationship", "timestamp": time.time(), "success": True},
                {"action": "socialize", "timestamp": time.time(), "success": True},
                {"action": "mediate_conflict", "timestamp": time.time(), "success": True},
                {"action": "socialize", "timestamp": time.time(), "success": True},
                {"action": "organize_social_event", "timestamp": time.time(), "success": True},
                {"action": "socialize", "timestamp": time.time(), "success": True},
                {"action": "build_relationship", "timestamp": time.time(), "success": True},
            ]
        }
        
        result = role_detector.detect_role(socializer_data)
        
        assert result.detected_role == "socializer"
        assert result.confidence >= 0.7
        assert "social_interaction" in result.supporting_evidence
        
    def test_role_classification_accuracy_target(self, role_detector):
        """Test role classification meets >80% accuracy target across multiple role types"""
        test_cases = [
            # Manager patterns
            {
                "expected_role": "manager",
                "actions": ["plan_event", "coordinate_team", "delegate_task", "manage_resources", "evaluate_performance"] * 2
            },
            # Socializer patterns
            {
                "expected_role": "socializer", 
                "actions": ["socialize", "build_relationship", "mediate_conflict", "organize_social_event"] * 3
            },
            # Resource Manager patterns
            {
                "expected_role": "resource_manager",
                "actions": ["manage_resources", "allocate_budget", "optimize_efficiency", "track_inventory"] * 3
            },
            # Mediator patterns
            {
                "expected_role": "mediator",
                "actions": ["mediate_conflict", "negotiate", "facilitate_discussion", "build_consensus"] * 3
            },
        ]
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for test_case in test_cases:
            agent_data = {
                "agent_id": f"test_agent_{test_case['expected_role']}",
                "action_history": [
                    {"action": action, "timestamp": time.time(), "success": True}
                    for action in test_case["actions"]
                ]
            }
            
            result = role_detector.detect_role(agent_data)
            
            if result.detected_role == test_case["expected_role"] and result.confidence >= 0.7:
                correct_predictions += 1
                
        accuracy = correct_predictions / total_predictions
        assert accuracy >= 0.8, f"Role detection accuracy {accuracy:.2%} is below 80% target"


class TestActionPatternAnalyzer:
    """Test suite for ActionPatternAnalyzer statistical analysis"""
    
    @pytest.fixture
    def pattern_analyzer(self):
        """Create ActionPatternAnalyzer instance"""
        return ActionPatternAnalyzer()
        
    @pytest.fixture
    def sample_actions(self):
        """Sample action data for pattern analysis"""
        return [
            {"action": "plan_event", "timestamp": time.time() - 3600, "success": True},
            {"action": "coordinate_team", "timestamp": time.time() - 3000, "success": True},
            {"action": "plan_event", "timestamp": time.time() - 2400, "success": True},
            {"action": "delegate_task", "timestamp": time.time() - 1800, "success": True},
            {"action": "plan_event", "timestamp": time.time() - 1200, "success": True},
            {"action": "socialize", "timestamp": time.time() - 600, "success": False},
            {"action": "coordinate_team", "timestamp": time.time() - 300, "success": True},
            {"action": "plan_event", "timestamp": time.time(), "success": True},
        ]
    
    def test_calculate_action_frequencies(self, pattern_analyzer, sample_actions):
        """Test statistical analysis of action frequencies"""
        frequencies = pattern_analyzer.calculate_action_frequencies(sample_actions)
        
        assert isinstance(frequencies, dict)
        assert frequencies["plan_event"] == 4  # Most frequent action
        assert frequencies["coordinate_team"] == 2
        assert frequencies["delegate_task"] == 1
        assert frequencies["socialize"] == 1
        
    def test_calculate_success_rates(self, pattern_analyzer, sample_actions):
        """Test calculation of action success rates"""
        success_rates = pattern_analyzer.calculate_success_rates(sample_actions)
        
        assert isinstance(success_rates, dict)
        assert success_rates["plan_event"] == 1.0  # 100% success rate
        assert success_rates["coordinate_team"] == 1.0  # 100% success rate  
        assert success_rates["socialize"] == 0.0  # 0% success rate (1 failure out of 1)
        
    def test_temporal_pattern_analysis(self, pattern_analyzer, sample_actions):
        """Test temporal pattern analysis for action sequences"""
        temporal_patterns = pattern_analyzer.analyze_temporal_patterns(sample_actions)
        
        assert isinstance(temporal_patterns, dict)
        assert "sequence_patterns" in temporal_patterns
        assert "time_distribution" in temporal_patterns
        assert "action_clustering" in temporal_patterns
        
    def test_identify_dominant_patterns(self, pattern_analyzer, sample_actions):
        """Test identification of dominant action patterns"""
        dominant_patterns = pattern_analyzer.identify_dominant_patterns(sample_actions)
        
        assert isinstance(dominant_patterns, list)
        assert len(dominant_patterns) > 0
        
        # The most dominant pattern should be planning-related
        top_pattern = dominant_patterns[0]
        assert "plan" in top_pattern["pattern_name"].lower() or "planning" in top_pattern["pattern_name"].lower()
        assert top_pattern["strength"] > 0.0


class TestProfessionalBehaviorClassifier:
    """Test suite for professional behavior classification"""
    
    @pytest.fixture 
    def behavior_classifier(self):
        """Create ProfessionalBehaviorClassifier instance"""
        return ProfessionalBehaviorClassifier()
        
    def test_classify_management_behavior(self, behavior_classifier):
        """Test classification of management professional behavior"""
        management_actions = [
            "plan_event", "coordinate_team", "delegate_task", "manage_resources", 
            "evaluate_performance", "set_goals", "allocate_budget"
        ]
        
        classification = behavior_classifier.classify_behavior_pattern(management_actions)
        
        assert classification["primary_behavior"] == "management"
        assert classification["confidence"] >= 0.7
        assert "leadership" in classification["behavioral_traits"]
        assert "organization" in classification["behavioral_traits"]
        
    def test_classify_social_behavior(self, behavior_classifier):
        """Test classification of social professional behavior"""
        social_actions = [
            "socialize", "build_relationship", "mediate_conflict", "organize_social_event",
            "facilitate_discussion", "network", "collaborate"
        ]
        
        classification = behavior_classifier.classify_behavior_pattern(social_actions)
        
        assert classification["primary_behavior"] == "social_coordination"
        assert classification["confidence"] >= 0.7
        assert "interpersonal" in classification["behavioral_traits"]
        assert "communication" in classification["behavioral_traits"]
        
    def test_classify_analytical_behavior(self, behavior_classifier):
        """Test classification of analytical professional behavior"""
        analytical_actions = [
            "analyze_data", "research", "evaluate_options", "optimize_process",
            "create_report", "investigate", "assess_risk"
        ]
        
        classification = behavior_classifier.classify_behavior_pattern(analytical_actions)
        
        assert classification["primary_behavior"] == "analytical"
        assert classification["confidence"] >= 0.7
        assert "analysis" in classification["behavioral_traits"]
        assert "problem_solving" in classification["behavioral_traits"]
        
    def test_mixed_behavior_classification(self, behavior_classifier):
        """Test classification of mixed professional behaviors"""
        mixed_actions = [
            "plan_event", "socialize", "analyze_data", "coordinate_team", 
            "build_relationship", "research", "delegate_task"
        ]
        
        classification = behavior_classifier.classify_behavior_pattern(mixed_actions)
        
        # Should identify primary behavior with lower confidence
        assert classification["confidence"] < 0.9  # Mixed patterns have lower confidence
        assert classification["confidence"] >= 0.5  # But still confident enough
        assert len(classification["behavioral_traits"]) >= 2  # Multiple traits identified


class TestRoleClassificationResult:
    """Test suite for RoleClassificationResult data structure"""
    
    def test_result_creation(self):
        """Test creation of RoleClassificationResult"""
        result = RoleClassificationResult(
            detected_role="manager",
            confidence=0.85,
            supporting_evidence=["leadership", "coordination", "planning"],
            behavioral_patterns={"management": 0.8, "social": 0.3},
            agent_id="test_agent"
        )
        
        assert result.detected_role == "manager"
        assert result.confidence == 0.85
        assert "leadership" in result.supporting_evidence
        assert result.behavioral_patterns["management"] == 0.8
        assert result.agent_id == "test_agent"
        
    def test_result_serialization(self):
        """Test JSON serialization of classification results"""
        result = RoleClassificationResult(
            detected_role="socializer",
            confidence=0.78,
            supporting_evidence=["interpersonal", "communication"],
            behavioral_patterns={"social": 0.9, "management": 0.2},
            agent_id="social_agent"
        )
        
        json_data = result.to_json()
        assert isinstance(json_data, str)
        
        # Verify JSON can be parsed back
        parsed_data = json.loads(json_data)
        assert parsed_data["detected_role"] == "socializer"
        assert parsed_data["confidence"] == 0.78
        
    def test_result_validation(self):
        """Test validation of classification result integrity"""
        result = RoleClassificationResult(
            detected_role="manager",
            confidence=0.85,
            supporting_evidence=["leadership", "coordination"],
            behavioral_patterns={"management": 0.8, "social": 0.3},
            agent_id="test_agent"
        )
        
        assert result.is_valid()
        assert result.confidence >= 0.0 and result.confidence <= 1.0
        assert result.detected_role is not None
        assert len(result.supporting_evidence) > 0


class TestIntegrationScenarios:
    """Integration tests for complete role detection workflow"""
    
    @pytest.fixture
    def complete_system(self):
        """Set up complete role detection system"""
        return RoleDetector(
            min_actions_for_detection=8,
            accuracy_threshold=0.8,
            confidence_threshold=0.7
        )
    
    def test_end_to_end_role_detection(self, complete_system):
        """Test complete role detection workflow from raw data to classification"""
        agent_data = {
            "agent_id": "integration_test_agent",
            "name": "IntegrationAgent",
            "action_history": [
                {"action": "plan_event", "timestamp": time.time() - 7200, "success": True},
                {"action": "coordinate_team", "timestamp": time.time() - 6600, "success": True},
                {"action": "delegate_task", "timestamp": time.time() - 6000, "success": True},
                {"action": "manage_resources", "timestamp": time.time() - 5400, "success": True},
                {"action": "plan_event", "timestamp": time.time() - 4800, "success": True},
                {"action": "evaluate_performance", "timestamp": time.time() - 4200, "success": True},
                {"action": "coordinate_team", "timestamp": time.time() - 3600, "success": True},
                {"action": "set_goals", "timestamp": time.time() - 3000, "success": True},
                {"action": "plan_event", "timestamp": time.time() - 2400, "success": True},
                {"action": "delegate_task", "timestamp": time.time() - 1800, "success": True},
            ],
            "specialization": {
                "current_role": None,
                "role_history": [],
                "skills": {},
                "expertise_level": 0.0
            }
        }
        
        # Execute complete detection workflow
        result = complete_system.detect_role(agent_data)
        
        # Verify comprehensive result
        assert result.detected_role == "manager"
        assert result.confidence >= 0.7
        assert result.agent_id == "integration_test_agent"
        assert len(result.supporting_evidence) >= 2
        assert "management" in result.behavioral_patterns
        assert result.behavioral_patterns["management"] > 0.5
        
    def test_batch_role_detection_performance(self, complete_system):
        """Test performance with batch processing of multiple agents"""
        agents_data = []
        
        # Create 20 test agents with different role patterns
        role_patterns = {
            "manager": ["plan_event", "coordinate_team", "delegate_task", "manage_resources"],
            "socializer": ["socialize", "build_relationship", "mediate_conflict", "organize_social_event"],
            "analyst": ["analyze_data", "research", "evaluate_options", "create_report"],
            "coordinator": ["coordinate_team", "schedule_meeting", "facilitate_discussion", "track_progress"],
        }
        
        for i in range(20):
            role_type = list(role_patterns.keys())[i % 4]
            actions = role_patterns[role_type] * 3  # 12 actions per agent
            
            agents_data.append({
                "agent_id": f"batch_agent_{i:02d}",
                "action_history": [
                    {"action": action, "timestamp": time.time() - (j * 300), "success": True}
                    for j, action in enumerate(actions)
                ]
            })
        
        # Measure batch processing time
        start_time = time.time()
        
        results = []
        for agent_data in agents_data:
            result = complete_system.detect_role(agent_data)
            results.append(result)
            
        end_time = time.time()
        
        # Verify performance and accuracy
        total_time = (end_time - start_time) * 1000  # Convert to ms
        avg_time_per_agent = total_time / len(agents_data)
        
        assert avg_time_per_agent < 50, f"Average processing time {avg_time_per_agent}ms exceeds 50ms target"
        
        # Verify classification accuracy
        confident_results = [r for r in results if r.confidence >= 0.7]
        accuracy_rate = len(confident_results) / len(results)
        
        assert accuracy_rate >= 0.8, f"Batch accuracy rate {accuracy_rate:.2%} is below 80% target"


# Performance and Load Testing
class TestPerformanceMetrics:
    """Performance-focused tests for role detection system"""
    
    def test_memory_usage_efficiency(self):
        """Test memory efficiency with large action histories"""
        import tracemalloc
        
        tracemalloc.start()
        
        # Create agent with large action history (1000 actions)
        large_action_history = []
        for i in range(1000):
            large_action_history.append({
                "action": f"action_{i % 10}",
                "timestamp": time.time() - (i * 60),
                "success": True
            })
            
        agent_data = {
            "agent_id": "memory_test_agent",
            "action_history": large_action_history
        }
        
        detector = RoleDetector()
        result = detector.detect_role(agent_data)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Memory usage should be reasonable (less than 10MB for this test)
        assert peak < 10 * 1024 * 1024, f"Peak memory usage {peak / 1024 / 1024:.2f}MB is too high"
        
    def test_concurrent_detection_safety(self):
        """Test thread safety with concurrent role detections"""
        import threading
        import concurrent.futures
        
        detector = RoleDetector()
        
        def detect_role_worker(agent_id):
            agent_data = {
                "agent_id": f"concurrent_agent_{agent_id}",
                "action_history": [
                    {"action": "plan_event", "timestamp": time.time(), "success": True},
                    {"action": "coordinate_team", "timestamp": time.time(), "success": True},
                    {"action": "delegate_task", "timestamp": time.time(), "success": True},
                    {"action": "manage_resources", "timestamp": time.time(), "success": True},
                    {"action": "plan_event", "timestamp": time.time(), "success": True},
                    {"action": "coordinate_team", "timestamp": time.time(), "success": True},
                    {"action": "evaluate_performance", "timestamp": time.time(), "success": True},
                    {"action": "set_goals", "timestamp": time.time(), "success": True},
                ]
            }
            return detector.detect_role(agent_data)
        
        # Run 10 concurrent detections
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(detect_role_worker, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All results should be valid
        assert len(results) == 10
        for result in results:
            assert result.detected_role is not None
            assert result.confidence >= 0.0


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])