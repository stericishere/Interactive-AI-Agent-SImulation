#!/usr/bin/env python3
"""
Comprehensive Test Suite for Social Goal Interpretation System - Enhanced PIANO Architecture
Test-Driven Development for Task 3.1.3: Social Goal Interpretation

Tests for dating_show/agents/specialization/social_goals.py
- Community role recognition
- Social expectation alignment
- Collective goal contribution measurement
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
from dating_show.agents.specialization.social_goals import (
    SocialGoalInterpreter,
    CommunityRoleRecognizer,
    SocialExpectationAnalyzer,
    CollectiveGoalAnalyzer,
    SocialGoalResult
)


class TestCommunityRoleRecognizer:
    """Test suite for CommunityRoleRecognizer class"""
    
    @pytest.fixture
    def role_recognizer(self):
        """Create CommunityRoleRecognizer instance for testing"""
        return CommunityRoleRecognizer()
    
    @pytest.fixture
    def sample_interactions(self):
        """Sample social interaction data"""
        return [
            {
                "type": "group_discussion",
                "participants": ["agent_001", "agent_002", "agent_003"],
                "initiator": "agent_001",
                "timestamp": time.time() - 3600,
                "agent_id": "agent_001"
            },
            {
                "type": "one_on_one",
                "participants": ["agent_001", "agent_002"],
                "initiator": "agent_002",
                "timestamp": time.time() - 2400,
                "agent_id": "agent_001"
            },
            {
                "type": "introduction",
                "participants": ["agent_002", "agent_004"],
                "initiator": "agent_001",
                "timestamp": time.time() - 1200,
                "agent_id": "agent_001"
            }
        ]
    
    def test_role_recognizer_initialization(self, role_recognizer):
        """Test CommunityRoleRecognizer initializes with correct role definitions"""
        assert hasattr(role_recognizer, 'community_roles')
        assert len(role_recognizer.community_roles) >= 6  # Should have multiple role types
        
        # Verify essential roles are defined
        essential_roles = ["leader", "mediator", "connector", "supporter", "innovator", "observer"]
        for role in essential_roles:
            assert role in role_recognizer.community_roles
            
            role_config = role_recognizer.community_roles[role]
            assert "keywords" in role_config
            assert "actions" in role_config
            assert "social_traits" in role_config
            assert "network_position" in role_config
            assert "interaction_patterns" in role_config
            assert "weight" in role_config
    
    def test_analyze_social_interactions(self, role_recognizer, sample_interactions):
        """Test analysis of social interaction patterns"""
        analysis = role_recognizer.analyze_social_interactions(sample_interactions)
        
        assert isinstance(analysis, dict)
        assert "interaction_frequency" in analysis
        assert "interaction_types" in analysis
        assert "network_metrics" in analysis
        assert "unique_partners" in analysis
        
        assert analysis["interaction_frequency"] == 3
        assert "group_discussion" in analysis["interaction_types"]
        assert "one_on_one" in analysis["interaction_types"]
        
        network_metrics = analysis["network_metrics"]
        assert "connectivity" in network_metrics
        assert "interaction_diversity" in network_metrics
        assert "initiative_ratio" in network_metrics
        assert "response_ratio" in network_metrics
        
        # Verify metrics are reasonable
        assert network_metrics["initiative_ratio"] + network_metrics["response_ratio"] == 1.0
        assert analysis["unique_partners"] >= 0
    
    def test_recognize_leader_role(self, role_recognizer):
        """Test recognition of leader role from behavior patterns"""
        leader_agent_data = {
            "agent_id": "leader_001",
            "action_history": [
                {"action": "coordinate_team", "timestamp": time.time(), "success": True},
                {"action": "lead_meeting", "timestamp": time.time(), "success": True},
                {"action": "delegate_task", "timestamp": time.time(), "success": True},
                {"action": "make_decision", "timestamp": time.time(), "success": True},
                {"action": "coordinate_team", "timestamp": time.time(), "success": True},
                {"action": "set_direction", "timestamp": time.time(), "success": True},
            ],
            "social_interactions": [
                {
                    "type": "team_meeting",
                    "participants": ["leader_001", "agent_002", "agent_003", "agent_004"],
                    "initiator": "leader_001",
                    "agent_id": "leader_001"
                },
                {
                    "type": "directive",
                    "participants": ["leader_001", "agent_002"],
                    "initiator": "leader_001",
                    "agent_id": "leader_001"
                }
            ]
        }
        
        role, confidence, evidence = role_recognizer.recognize_community_role(leader_agent_data)
        
        assert role == "leader"
        assert confidence > 0.5  # Should have reasonable confidence
        assert isinstance(evidence, dict)
        assert len(evidence) > 0
    
    def test_recognize_connector_role(self, role_recognizer):
        """Test recognition of connector role from behavior patterns"""
        connector_agent_data = {
            "agent_id": "connector_001",
            "action_history": [
                {"action": "introduce_people", "timestamp": time.time(), "success": True},
                {"action": "organize_social_event", "timestamp": time.time(), "success": True},
                {"action": "facilitate_connections", "timestamp": time.time(), "success": True},
                {"action": "network", "timestamp": time.time(), "success": True},
                {"action": "collaborate", "timestamp": time.time(), "success": True},
                {"action": "introduce_people", "timestamp": time.time(), "success": True},
            ],
            "social_interactions": [
                {
                    "type": "introduction",
                    "participants": ["connector_001", "agent_002", "agent_003"],
                    "initiator": "connector_001",
                    "agent_id": "connector_001"
                },
                {
                    "type": "social_event",
                    "participants": ["connector_001", "agent_002", "agent_003", "agent_004", "agent_005", "agent_006"],
                    "initiator": "connector_001",
                    "agent_id": "connector_001"
                }
            ]
        }
        
        role, confidence, evidence = role_recognizer.recognize_community_role(connector_agent_data)
        
        assert role == "connector"
        assert confidence > 0.5
        assert "high_connectivity" in evidence.get("connector", [])
    
    def test_recognize_mediator_role(self, role_recognizer):
        """Test recognition of mediator role from behavior patterns"""
        mediator_agent_data = {
            "agent_id": "mediator_001",
            "action_history": [
                {"action": "mediate_conflict", "timestamp": time.time(), "success": True},
                {"action": "facilitate_discussion", "timestamp": time.time(), "success": True},
                {"action": "build_consensus", "timestamp": time.time(), "success": True},
                {"action": "negotiate", "timestamp": time.time(), "success": True},
                {"action": "resolve_dispute", "timestamp": time.time(), "success": True},
            ],
            "social_interactions": [
                {
                    "type": "conflict_mediation",
                    "participants": ["mediator_001", "agent_002", "agent_003"],
                    "initiator": "mediator_001",
                    "agent_id": "mediator_001"
                }
            ]
        }
        
        role, confidence, evidence = role_recognizer.recognize_community_role(mediator_agent_data)
        
        assert role == "mediator"
        assert confidence > 0.5
        assert "mediation_actions" in evidence.get("mediator", [])


class TestSocialExpectationAnalyzer:
    """Test suite for SocialExpectationAnalyzer class"""
    
    @pytest.fixture
    def expectation_analyzer(self):
        """Create SocialExpectationAnalyzer instance for testing"""
        return SocialExpectationAnalyzer()
    
    def test_expectation_analyzer_initialization(self, expectation_analyzer):
        """Test SocialExpectationAnalyzer initializes with role expectations"""
        assert hasattr(expectation_analyzer, 'role_expectations')
        assert len(expectation_analyzer.role_expectations) >= 6
        
        # Verify structure of role expectations
        for role_name, expectations in expectation_analyzer.role_expectations.items():
            assert "expected_behaviors" in expectations
            assert "communication_style" in expectations
            assert "interaction_frequency" in expectations
            assert "network_position" in expectations
            
            assert isinstance(expectations["expected_behaviors"], list)
            assert isinstance(expectations["communication_style"], list)
            assert expectations["interaction_frequency"] in ["low", "medium", "high", "selective"]
    
    def test_analyze_leader_expectation_alignment(self, expectation_analyzer):
        """Test expectation alignment analysis for leader role"""
        leader_agent_data = {
            "action_history": [
                {"action": "make_decision", "timestamp": time.time(), "success": True},
                {"action": "provide_direction", "timestamp": time.time(), "success": True},
                {"action": "take_responsibility", "timestamp": time.time(), "success": True},
                {"action": "inspire_others", "timestamp": time.time(), "success": True},
            ],
            "communication_style": {
                "clear": 0.9,
                "confident": 0.8,
                "directive": 0.9,
                "supportive": 0.7
            },
            "social_interactions": [
                {"type": "meeting", "participants": ["leader", "agent1", "agent2", "agent3"]},
                {"type": "decision", "participants": ["leader", "agent1"]},
                {"type": "direction", "participants": ["leader", "agent2"]},
            ]
        }
        
        alignment = expectation_analyzer.analyze_expectation_alignment(leader_agent_data, "leader")
        
        assert isinstance(alignment, dict)
        assert "behavior_alignment" in alignment
        assert "communication_alignment" in alignment
        assert "interaction_alignment" in alignment
        assert "overall_alignment" in alignment
        
        # Should have high alignment for leader behaviors
        assert alignment["behavior_alignment"] > 0.5
        assert alignment["communication_alignment"] > 0.5
        assert 0.0 <= alignment["overall_alignment"] <= 1.0
    
    def test_analyze_observer_expectation_alignment(self, expectation_analyzer):
        """Test expectation alignment analysis for observer role"""
        observer_agent_data = {
            "action_history": [
                {"action": "monitor_situation", "timestamp": time.time(), "success": True},
                {"action": "analyze_pattern", "timestamp": time.time(), "success": True},
                {"action": "provide_insight", "timestamp": time.time(), "success": True},
            ],
            "communication_style": {
                "thoughtful": 0.9,
                "analytical": 0.8,
                "precise": 0.8,
                "quiet": 0.9
            },
            "social_interactions": [
                {"type": "observation", "participants": ["observer", "agent1"]},
            ]
        }
        
        alignment = expectation_analyzer.analyze_expectation_alignment(observer_agent_data, "observer")
        
        assert alignment["behavior_alignment"] > 0.3  # Some alignment with observer behaviors
        assert alignment["communication_alignment"] > 0.7  # High alignment with observer communication style
        assert alignment["interaction_alignment"] > 0.5  # Low interaction frequency expected for observer
    
    def test_behavior_alignment_calculation(self, expectation_analyzer):
        """Test behavior alignment calculation logic"""
        actions = [
            {"action": "make_decisions", "timestamp": time.time(), "success": True},
            {"action": "provide_direction", "timestamp": time.time(), "success": True},
            {"action": "unrelated_action", "timestamp": time.time(), "success": True},
        ]
        
        expected_behaviors = ["make_decisions", "provide_direction", "take_responsibility", "inspire_others"]
        
        alignment = expectation_analyzer._analyze_behavior_alignment(actions, expected_behaviors)
        
        # Should have 50% alignment (2 out of 4 expected behaviors matched)
        assert 0.4 <= alignment <= 0.6


class TestCollectiveGoalAnalyzer:
    """Test suite for CollectiveGoalAnalyzer class"""
    
    @pytest.fixture
    def collective_analyzer(self):
        """Create CollectiveGoalAnalyzer instance for testing"""
        return CollectiveGoalAnalyzer()
    
    def test_collective_analyzer_initialization(self, collective_analyzer):
        """Test CollectiveGoalAnalyzer initializes with collective goal types"""
        assert hasattr(collective_analyzer, 'collective_goal_types')
        assert len(collective_analyzer.collective_goal_types) >= 5
        
        # Verify essential collective goal types
        essential_goals = ["community_harmony", "community_growth", "knowledge_sharing", "innovation", "resource_optimization"]
        for goal_type in essential_goals:
            assert goal_type in collective_analyzer.collective_goal_types
            
            goal_config = collective_analyzer.collective_goal_types[goal_type]
            assert "keywords" in goal_config
            assert "contributing_actions" in goal_config
            assert "weight" in goal_config
    
    def test_measure_collective_contribution(self, collective_analyzer):
        """Test measurement of collective goal contributions"""
        agent_data = {
            "action_history": [
                {"action": "mediate_conflict", "timestamp": time.time(), "success": True},      # community_harmony
                {"action": "organize_event", "timestamp": time.time(), "success": True},        # community_growth
                {"action": "share_information", "timestamp": time.time(), "success": True},     # knowledge_sharing
                {"action": "propose_idea", "timestamp": time.time(), "success": True},          # innovation
                {"action": "optimize_process", "timestamp": time.time(), "success": True},      # resource_optimization
            ]
        }
        
        contributions = collective_analyzer.measure_collective_contribution(agent_data)
        
        assert isinstance(contributions, dict)
        assert len(contributions) >= 5  # Should measure all collective goal types
        
        # All contribution scores should be between 0 and 1
        for goal_type, score in contributions.items():
            assert 0.0 <= score <= 1.0
        
        # Should have contributions to multiple goal types
        positive_contributions = [score for score in contributions.values() if score > 0]
        assert len(positive_contributions) >= 4  # Should contribute to most goals
    
    def test_analyze_collaboration_patterns(self, collective_analyzer):
        """Test analysis of collaboration patterns"""
        agent_data = {
            "social_interactions": [
                {
                    "type": "group_project",
                    "participants": ["agent_001", "agent_002", "agent_003", "agent_004"],
                    "timestamp": time.time()
                },
                {
                    "type": "team_meeting",
                    "participants": ["agent_001", "agent_002", "agent_003"],
                    "timestamp": time.time()
                }
            ],
            "action_history": [
                {"action": "coordinate_team", "timestamp": time.time(), "success": True},
                {"action": "help_colleague", "timestamp": time.time(), "success": True},
                {"action": "collaborate_on_project", "timestamp": time.time(), "success": True},
            ]
        }
        
        patterns = collective_analyzer.analyze_collaboration_patterns(agent_data)
        
        assert isinstance(patterns, list)
        
        for pattern in patterns:
            assert "pattern_type" in pattern
            assert "frequency" in pattern
            assert "strength" in pattern
            assert "description" in pattern
            
            assert pattern["frequency"] >= 0
            assert 0.0 <= pattern["strength"] <= 1.0
        
        # Should identify multiple collaboration patterns
        pattern_types = [p["pattern_type"] for p in patterns]
        expected_patterns = ["group_collaboration", "coordination", "supportive_collaboration"]
        
        # Should find at least one collaboration pattern
        assert any(pattern_type in expected_patterns for pattern_type in pattern_types)


class TestSocialGoalInterpreter:
    """Test suite for main SocialGoalInterpreter class"""
    
    @pytest.fixture
    def social_interpreter(self):
        """Create SocialGoalInterpreter instance for testing"""
        return SocialGoalInterpreter()
    
    @pytest.fixture
    def sample_social_agent(self):
        """Sample social agent data for testing"""
        return {
            "agent_id": "social_test_agent",
            "action_history": [
                {"action": "organize_social_event", "timestamp": time.time() - 3600, "success": True},
                {"action": "introduce_people", "timestamp": time.time() - 3000, "success": True},
                {"action": "facilitate_connections", "timestamp": time.time() - 2400, "success": True},
                {"action": "network", "timestamp": time.time() - 1800, "success": True},
                {"action": "collaborate", "timestamp": time.time() - 1200, "success": True},
                {"action": "organize_social_event", "timestamp": time.time() - 600, "success": True},
            ],
            "social_interactions": [
                {
                    "type": "social_event",
                    "participants": ["social_test_agent", "agent_002", "agent_003", "agent_004"],
                    "initiator": "social_test_agent",
                    "timestamp": time.time() - 2000
                },
                {
                    "type": "introduction",
                    "participants": ["agent_002", "agent_005"],
                    "initiator": "social_test_agent",
                    "timestamp": time.time() - 1000
                }
            ],
            "communication_style": {
                "friendly": 0.9,
                "enthusiastic": 0.8,
                "inclusive": 0.9,
                "energetic": 0.7
            }
        }
    
    def test_social_interpreter_initialization(self, social_interpreter):
        """Test SocialGoalInterpreter initializes correctly"""
        assert hasattr(social_interpreter, 'community_context')
        assert hasattr(social_interpreter, 'role_recognizer')
        assert hasattr(social_interpreter, 'expectation_analyzer')
        assert hasattr(social_interpreter, 'collective_analyzer')
        
        assert isinstance(social_interpreter.role_recognizer, CommunityRoleRecognizer)
        assert isinstance(social_interpreter.expectation_analyzer, SocialExpectationAnalyzer)
        assert isinstance(social_interpreter.collective_analyzer, CollectiveGoalAnalyzer)
    
    def test_interpret_social_goals_performance(self, social_interpreter, sample_social_agent):
        """Test social goal interpretation meets performance target <50ms"""
        start_time = time.time()
        
        result = social_interpreter.interpret_social_goals(sample_social_agent)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000
        
        assert processing_time < 50, f"Social goal interpretation took {processing_time}ms, exceeds 50ms target"
        assert isinstance(result, SocialGoalResult)
    
    def test_interpret_social_goals_comprehensive(self, social_interpreter, sample_social_agent):
        """Test comprehensive social goal interpretation"""
        result = social_interpreter.interpret_social_goals(sample_social_agent)
        
        # Verify result structure
        assert result.agent_id == "social_test_agent"
        assert result.community_role == "connector"  # Should identify as connector based on actions
        assert isinstance(result.social_contribution_score, float)
        assert 0.0 <= result.social_contribution_score <= 1.0
        assert isinstance(result.collective_goal_alignment, dict)
        assert isinstance(result.social_expectations, list)
        assert isinstance(result.community_impact_metrics, dict)
        assert isinstance(result.social_network_position, dict)
        assert isinstance(result.collaboration_patterns, list)
        
        # Should have reasonable social contribution score for connector
        assert result.social_contribution_score > 0.2
        
        # Should have collective goal contributions
        assert len(result.collective_goal_alignment) > 0
        
        # Should have social expectations for connector role
        assert len(result.social_expectations) > 0
        
        # Verify community impact metrics structure
        impact_metrics = result.community_impact_metrics
        expected_metrics = ["role_fulfillment", "expectation_alignment", "collective_contribution", 
                          "collaboration_effectiveness", "social_influence"]
        for metric in expected_metrics:
            assert metric in impact_metrics
            assert 0.0 <= impact_metrics[metric] <= 1.0
    
    def test_batch_interpret_social_goals(self, social_interpreter):
        """Test batch social goal interpretation for multiple agents"""
        agents_data = [
            {
                "agent_id": f"batch_social_agent_{i:02d}",
                "action_history": [
                    {"action": "organize_social_event", "timestamp": time.time(), "success": True},
                    {"action": "network", "timestamp": time.time(), "success": True},
                ],
                "social_interactions": [
                    {
                        "type": "networking",
                        "participants": [f"batch_social_agent_{i:02d}", "agent_other"],
                        "initiator": f"batch_social_agent_{i:02d}"
                    }
                ],
                "communication_style": {"friendly": 0.8}
            }
            for i in range(5)
        ]
        
        results = social_interpreter.batch_interpret_social_goals(agents_data)
        
        assert len(results) == 5
        assert all(isinstance(result, SocialGoalResult) for result in results)
        assert all(result.agent_id.startswith("batch_social_agent_") for result in results)
    
    def test_get_social_statistics(self, social_interpreter):
        """Test calculation of social interpretation statistics"""
        results = [
            SocialGoalResult(
                agent_id=f"stats_agent_{i}",
                community_role="connector" if i % 2 == 0 else "supporter",
                social_contribution_score=0.8 if i % 2 == 0 else 0.6,
                collective_goal_alignment={"community_growth": 0.7, "knowledge_sharing": 0.5},
                social_expectations=[],
                community_impact_metrics={
                    "role_fulfillment": 0.8,
                    "expectation_alignment": 0.7,
                    "collective_contribution": 0.6,
                    "collaboration_effectiveness": 0.7,
                    "social_influence": 0.5
                },
                social_network_position={},
                collaboration_patterns=[]
            )
            for i in range(10)
        ]
        
        stats = social_interpreter.get_social_statistics(results)
        
        assert isinstance(stats, dict)
        assert "total_agents" in stats
        assert "socially_aligned_agents" in stats
        assert "social_alignment_rate" in stats
        assert "role_distribution" in stats
        assert "avg_social_contribution" in stats
        assert "community_impact_summary" in stats
        
        assert stats["total_agents"] == 10
        # With threshold 0.7, only agents with 0.8 score should be socially aligned
        assert stats["socially_aligned_agents"] == 5
        assert stats["social_alignment_rate"] == 0.5
        
        # Verify role distribution
        role_dist = stats["role_distribution"]
        assert "connector" in role_dist
        assert "supporter" in role_dist
        assert role_dist["connector"] == 5
        assert role_dist["supporter"] == 5


class TestSocialGoalResult:
    """Test suite for SocialGoalResult data structure"""
    
    def test_result_creation(self):
        """Test creation of SocialGoalResult"""
        result = SocialGoalResult(
            agent_id="test_social_agent",
            community_role="connector",
            social_contribution_score=0.85,
            collective_goal_alignment={"community_growth": 0.7, "knowledge_sharing": 0.8},
            social_expectations=[{"expectation_type": "behavior", "expected_behaviors": ["connect"], "alignment_score": 0.9}],
            community_impact_metrics={"role_fulfillment": 0.8, "social_influence": 0.6},
            social_network_position={"centrality": 5, "activity_level": 8},
            collaboration_patterns=[{"pattern_type": "group_collaboration", "frequency": 3, "strength": 0.7}]
        )
        
        assert result.agent_id == "test_social_agent"
        assert result.community_role == "connector"
        assert result.social_contribution_score == 0.85
        assert result.collective_goal_alignment["community_growth"] == 0.7
        assert result.timestamp is not None
    
    def test_is_socially_aligned_method(self):
        """Test is_socially_aligned method with different thresholds"""
        result = SocialGoalResult(
            agent_id="test_agent",
            community_role="leader",
            social_contribution_score=0.75,
            collective_goal_alignment={},
            social_expectations=[],
            community_impact_metrics={},
            social_network_position={},
            collaboration_patterns=[]
        )
        
        assert result.is_socially_aligned(threshold=0.7)  # Should be aligned
        assert not result.is_socially_aligned(threshold=0.8)  # Should not be aligned
    
    def test_result_serialization(self):
        """Test serialization of SocialGoalResult"""
        result = SocialGoalResult(
            agent_id="test_agent",
            community_role="mediator",
            social_contribution_score=0.72,
            collective_goal_alignment={"community_harmony": 0.9},
            social_expectations=[{"expectation_type": "behavior", "alignment_score": 0.8}],
            community_impact_metrics={"role_fulfillment": 0.7},
            social_network_position={"centrality": 3},
            collaboration_patterns=[]
        )
        
        # Test JSON serialization
        json_str = result.to_json()
        assert isinstance(json_str, str)
        
        # Verify JSON can be parsed back
        parsed_data = json.loads(json_str)
        assert parsed_data["agent_id"] == "test_agent"
        assert parsed_data["community_role"] == "mediator"
        assert parsed_data["social_contribution_score"] == 0.72
        
        # Test dict conversion
        dict_data = result.to_dict()
        assert isinstance(dict_data, dict)
        assert dict_data["agent_id"] == "test_agent"


class TestIntegrationScenarios:
    """Integration tests for complete social goal interpretation workflow"""
    
    @pytest.fixture
    def complete_system(self):
        """Set up complete social goal interpretation system"""
        community_context = {
            "collective_goals": [
                {"goal_type": "community_harmony"},
                {"goal_type": "community_growth"},
                {"goal_type": "knowledge_sharing"}
            ]
        }
        return SocialGoalInterpreter(community_context=community_context)
    
    def test_end_to_end_social_interpretation(self, complete_system):
        """Test complete social goal interpretation workflow"""
        agent_data = {
            "agent_id": "integration_social_agent",
            "action_history": [
                {"action": "organize_social_event", "timestamp": time.time() - 7200, "success": True},
                {"action": "introduce_people", "timestamp": time.time() - 6600, "success": True},
                {"action": "facilitate_connections", "timestamp": time.time() - 6000, "success": True},
                {"action": "share_information", "timestamp": time.time() - 5400, "success": True},
                {"action": "collaborate", "timestamp": time.time() - 4800, "success": True},
                {"action": "organize_event", "timestamp": time.time() - 4200, "success": True},
                {"action": "network", "timestamp": time.time() - 3600, "success": True},
                {"action": "mediate_conflict", "timestamp": time.time() - 3000, "success": True},
            ],
            "social_interactions": [
                {
                    "type": "social_event",
                    "participants": ["integration_social_agent", "agent_002", "agent_003", "agent_004", "agent_005"],
                    "initiator": "integration_social_agent",
                    "timestamp": time.time() - 6000
                },
                {
                    "type": "introduction",
                    "participants": ["agent_002", "agent_006"],
                    "initiator": "integration_social_agent",
                    "timestamp": time.time() - 4800
                },
                {
                    "type": "conflict_mediation",
                    "participants": ["integration_social_agent", "agent_003", "agent_004"],
                    "initiator": "integration_social_agent",
                    "timestamp": time.time() - 3000
                }
            ],
            "communication_style": {
                "friendly": 0.9,
                "enthusiastic": 0.8,
                "inclusive": 0.9,
                "diplomatic": 0.7,
                "energetic": 0.8
            }
        }
        
        result = complete_system.interpret_social_goals(agent_data)
        
        # Comprehensive verification
        assert result.agent_id == "integration_social_agent"
        assert result.community_role in ["connector", "mediator"]  # Could be either based on mixed actions
        assert result.social_contribution_score > 0.3  # Should have reasonable social contribution
        assert len(result.collective_goal_alignment) >= 3  # Should align with community goals
        assert len(result.social_expectations) >= 2  # Should have behavior and communication expectations
        assert len(result.collaboration_patterns) > 0  # Should identify collaboration patterns
        
        # Verify community impact metrics
        impact_metrics = result.community_impact_metrics
        assert impact_metrics["role_fulfillment"] > 0.5
        assert impact_metrics["collective_contribution"] >= 0.0
        assert impact_metrics["collaboration_effectiveness"] >= 0.0
        
        # Verify social network position analysis
        network_position = result.social_network_position
        assert "centrality" in network_position
        assert "activity_level" in network_position
        assert network_position["activity_level"] >= 3  # Should have high activity
    
    def test_mixed_role_detection_accuracy(self, complete_system):
        """Test accuracy with agents having mixed role characteristics"""
        # Create agent with mixed leader and connector traits
        mixed_agent_data = {
            "agent_id": "mixed_role_agent",
            "action_history": [
                # Leader actions
                {"action": "coordinate_team", "timestamp": time.time(), "success": True},
                {"action": "make_decision", "timestamp": time.time(), "success": True},
                # Connector actions
                {"action": "introduce_people", "timestamp": time.time(), "success": True},
                {"action": "organize_social_event", "timestamp": time.time(), "success": True},
                # More leader actions to make it dominant
                {"action": "delegate_task", "timestamp": time.time(), "success": True},
                {"action": "set_direction", "timestamp": time.time(), "success": True},
                {"action": "lead_meeting", "timestamp": time.time(), "success": True},
            ],
            "social_interactions": [
                {
                    "type": "team_meeting",
                    "participants": ["mixed_role_agent", "agent_002", "agent_003"],
                    "initiator": "mixed_role_agent"
                },
                {
                    "type": "social_event", 
                    "participants": ["mixed_role_agent", "agent_002", "agent_003", "agent_004"],
                    "initiator": "mixed_role_agent"
                }
            ],
            "communication_style": {
                "confident": 0.8,  # Leader trait
                "directive": 0.7,  # Leader trait
                "friendly": 0.8,   # Connector trait
                "inclusive": 0.6   # Connector trait
            }
        }
        
        result = complete_system.interpret_social_goals(mixed_agent_data)
        
        # Should identify primary role based on dominant pattern
        assert result.community_role == "leader"  # More leader actions
        assert result.social_contribution_score > 0.3
        
        # Should still have some collective goal contributions
        assert len(result.collective_goal_alignment) > 0


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v", "--tb=short"])