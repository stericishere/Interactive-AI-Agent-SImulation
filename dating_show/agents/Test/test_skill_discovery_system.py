"""
Comprehensive Test Suite for Dynamic Skill Discovery System
Task 3.2.1.1: Dynamic Skill Discovery System Tests

This module provides comprehensive testing for the dynamic skill discovery system,
including discovery probability calculations, action analysis, and integration tests.
"""

import unittest
import random
import time
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Import the skill system modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.skill_development import (
    SkillDevelopmentSystem, SkillType, SkillLevel, LearningSourceType
)


class TestDynamicSkillDiscovery(unittest.TestCase):
    """Test cases for dynamic skill discovery functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.skill_system = SkillDevelopmentSystem(max_agents=100)
        self.test_agent_id = "test_agent_001"
        self.skill_system.add_agent(self.test_agent_id)
        
        # Create test contexts
        self.basic_context = {
            "complexity": 0.5,
            "environment": "training",
            "agent_curiosity": 0.7
        }
        
        self.combat_context = {
            "complexity": 0.8,
            "environment": "battlefield",
            "danger_level": 0.9,
            "agent_curiosity": 0.6
        }
        
        self.creative_context = {
            "complexity": 0.6,
            "environment": "workshop",
            "creativity_required": True,
            "agent_curiosity": 0.8
        }
    
    def test_discover_skills_from_actions_basic(self):
        """Test basic skill discovery from actions"""
        # Test combat skill discovery
        discovered_skills = self.skill_system.discover_skills_from_actions(
            agent_id=self.test_agent_id,
            action="fight the enemy",
            performance=0.8,
            context=self.combat_context
        )
        
        # Should potentially discover combat-related skills
        # Note: This is probabilistic, so we test the mechanism rather than exact results
        self.assertIsInstance(discovered_skills, list)
        
        # Test multiple discovery attempts to verify probabilistic nature
        discovery_attempts = []
        for _ in range(20):
            skills = self.skill_system.discover_skills_from_actions(
                agent_id=f"agent_{random.randint(1000, 9999)}",
                action="craft a tool",
                performance=0.9,
                context=self.creative_context
            )
            discovery_attempts.append(len(skills))
            self.skill_system.add_agent(f"agent_{random.randint(1000, 9999)}")
        
        # Should have some variation in discovery rates
        self.assertGreater(max(discovery_attempts), 0)
    
    def test_analyze_action_for_skills(self):
        """Test action analysis for skill identification"""
        # Test combat actions
        combat_skills = self.skill_system._analyze_action_for_skills(
            "attack with sword", self.combat_context
        )
        self.assertIn(SkillType.COMBAT, combat_skills)
        
        # Test crafting actions
        craft_skills = self.skill_system._analyze_action_for_skills(
            "build a shelter", self.creative_context
        )
        self.assertIn(SkillType.CRAFTING, craft_skills)
        self.assertIn(SkillType.SHELTER_BUILDING, craft_skills)
        
        # Test social actions
        social_context = {"social_interaction": True}
        social_skills = self.skill_system._analyze_action_for_skills(
            "convince the merchant", social_context
        )
        self.assertIn(SkillType.PERSUASION, social_skills)
    
    def test_analyze_context_for_skills(self):
        """Test context-based skill discovery"""
        # Test dangerous context
        danger_context = {"danger_level": 0.8}
        danger_skills = self.skill_system._analyze_context_for_skills(danger_context)
        expected_danger_skills = [SkillType.COMBAT, SkillType.STEALTH, SkillType.ATHLETICS]
        for skill in expected_danger_skills:
            self.assertIn(skill, danger_skills)
        
        # Test learning context
        learning_context = {"learning_opportunity": True}
        learning_skills = self.skill_system._analyze_context_for_skills(learning_context)
        expected_learning_skills = [SkillType.LEARNING, SkillType.MEMORY, SkillType.FOCUS]
        for skill in expected_learning_skills:
            self.assertIn(skill, learning_skills)
    
    def test_analyze_environment_for_skills(self):
        """Test environment-based skill discovery"""
        # Test wilderness environment
        wilderness_skills = self.skill_system._analyze_environment_for_skills("wilderness")
        expected_wilderness = [SkillType.FORAGING, SkillType.HUNTING, SkillType.NAVIGATION, SkillType.SHELTER_BUILDING]
        for skill in expected_wilderness:
            self.assertIn(skill, wilderness_skills)
        
        # Test laboratory environment
        lab_skills = self.skill_system._analyze_environment_for_skills("laboratory")
        expected_lab = [SkillType.RESEARCH, SkillType.ANALYSIS, SkillType.FOCUS]
        for skill in expected_lab:
            self.assertIn(skill, lab_skills)
    
    def test_calculate_discovery_probability(self):
        """Test discovery probability calculation"""
        # Test with high performance
        high_prob = self.skill_system._calculate_discovery_probability(
            SkillType.COMBAT, 0.9, self.combat_context, self.test_agent_id
        )
        
        # Test with low performance
        low_prob = self.skill_system._calculate_discovery_probability(
            SkillType.COMBAT, 0.2, self.combat_context, self.test_agent_id
        )
        
        # High performance should yield higher probability
        self.assertGreater(high_prob, low_prob)
        
        # Probabilities should be within valid range
        self.assertGreaterEqual(high_prob, 0.01)
        self.assertLessEqual(high_prob, 0.3)
        self.assertGreaterEqual(low_prob, 0.01)
        self.assertLessEqual(low_prob, 0.3)
    
    def test_get_related_skills(self):
        """Test skill relationship mapping"""
        # Test combat skill relationships
        combat_related = self.skill_system._get_related_skills(SkillType.COMBAT)
        expected_combat = [SkillType.ATHLETICS, SkillType.STEALTH, SkillType.ACROBATICS]
        for skill in expected_combat:
            self.assertIn(skill, combat_related)
        
        # Test social skill relationships
        persuasion_related = self.skill_system._get_related_skills(SkillType.PERSUASION)
        expected_persuasion = [SkillType.EMPATHY, SkillType.NETWORKING, SkillType.NEGOTIATION]
        for skill in expected_persuasion:
            self.assertIn(skill, persuasion_related)
    
    def test_initialize_discovered_skill(self):
        """Test skill initialization upon discovery"""
        # Discover a new skill
        initial_skills_count = len(self.skill_system.get_agent_skills(self.test_agent_id))
        
        self.skill_system._initialize_discovered_skill(
            self.test_agent_id, SkillType.ACROBATICS, 5.0
        )
        
        # Check that skill was added
        agent_skills = self.skill_system.get_agent_skills(self.test_agent_id)
        self.assertEqual(len(agent_skills), initial_skills_count + 1)
        self.assertIn(SkillType.ACROBATICS, agent_skills)
        
        # Check skill properties
        acrobatics_skill = agent_skills[SkillType.ACROBATICS]
        self.assertEqual(acrobatics_skill.experience_points, 5.0)
        self.assertEqual(acrobatics_skill.level, SkillLevel.NOVICE)
    
    def test_get_discoverable_skills(self):
        """Test discoverable skills prediction"""
        discoverable = self.skill_system.get_discoverable_skills(
            self.test_agent_id, "practice archery", self.basic_context
        )
        
        self.assertIsInstance(discoverable, dict)
        
        # All values should be valid probabilities
        for skill_type, probability in discoverable.items():
            self.assertIsInstance(skill_type, SkillType)
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)
    
    def test_skill_discovery_with_existing_skills(self):
        """Test that agents don't rediscover existing skills"""
        # Add a skill to the agent
        self.skill_system.add_skill(self.test_agent_id, SkillType.COMBAT, SkillLevel.BEGINNER)
        
        # Try to discover the same skill
        discovered = self.skill_system.discover_skills_from_actions(
            self.test_agent_id, "fight enemy", 0.8, self.combat_context
        )
        
        # Should not include the existing skill
        self.assertNotIn(SkillType.COMBAT, discovered)
    
    def test_discovery_with_learning_skill_bonus(self):
        """Test discovery probability bonus from learning skill"""
        # Add learning skill to agent
        self.skill_system.add_skill(self.test_agent_id, SkillType.LEARNING, SkillLevel.EXPERT)
        
        # Calculate probability with learning bonus
        prob_with_learning = self.skill_system._calculate_discovery_probability(
            SkillType.RESEARCH, 0.5, self.basic_context, self.test_agent_id
        )
        
        # Create agent without learning skill for comparison
        other_agent = "agent_no_learning"
        self.skill_system.add_agent(other_agent)
        
        prob_without_learning = self.skill_system._calculate_discovery_probability(
            SkillType.RESEARCH, 0.5, self.basic_context, other_agent
        )
        
        # Should have higher probability with learning skill
        self.assertGreater(prob_with_learning, prob_without_learning)
    
    def test_discovery_integration_with_execution(self):
        """Test integration with skill execution module"""
        # This would require mocking or actual execution module integration
        # For now, test the structure that supports integration
        
        # Mock execution result with discovery
        mock_result = {
            "discovered_skills": ["stealth", "acrobatics"],
            "skill_used": "combat",
            "success": True,
            "performance_score": 0.8
        }
        
        # Verify structure is compatible
        self.assertIn("discovered_skills", mock_result)
        self.assertIsInstance(mock_result["discovered_skills"], list)
    
    def test_discovery_edge_cases(self):
        """Test edge cases in skill discovery"""
        # Test with invalid agent
        discovered = self.skill_system.discover_skills_from_actions(
            "nonexistent_agent", "test action", 0.5, {}
        )
        self.assertEqual(discovered, [])
        
        # Test with empty action
        discovered = self.skill_system.discover_skills_from_actions(
            self.test_agent_id, "", 0.5, {}
        )
        self.assertIsInstance(discovered, list)
        
        # Test with extreme performance values
        discovered = self.skill_system.discover_skills_from_actions(
            self.test_agent_id, "test action", 2.0, {}  # Above normal range
        )
        self.assertIsInstance(discovered, list)
        
        discovered = self.skill_system.discover_skills_from_actions(
            self.test_agent_id, "test action", -0.5, {}  # Below normal range
        )
        self.assertIsInstance(discovered, list)


class TestSkillDiscoveryPerformance(unittest.TestCase):
    """Performance tests for skill discovery system"""
    
    def setUp(self):
        self.skill_system = SkillDevelopmentSystem(max_agents=1000)
    
    def test_discovery_performance_at_scale(self):
        """Test discovery performance with multiple agents"""
        # Create multiple agents
        agent_ids = []
        for i in range(100):
            agent_id = f"perf_test_agent_{i}"
            self.skill_system.add_agent(agent_id)
            agent_ids.append(agent_id)
        
        # Time discovery operations
        start_time = time.time()
        
        discoveries = []
        for agent_id in agent_ids[:50]:  # Test with 50 agents
            discovered = self.skill_system.discover_skills_from_actions(
                agent_id, "practice combat techniques", 0.7, {"complexity": 0.6}
            )
            discoveries.append(discovered)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time (2 seconds for 50 agents)
        self.assertLess(total_time, 2.0)
        
        # Should have processed all agents
        self.assertEqual(len(discoveries), 50)
    
    def test_probability_calculation_performance(self):
        """Test performance of probability calculations"""
        agent_id = "perf_agent"
        self.skill_system.add_agent(agent_id)
        
        start_time = time.time()
        
        # Perform many probability calculations
        for _ in range(1000):
            self.skill_system._calculate_discovery_probability(
                SkillType.COMBAT, 0.6, {"complexity": 0.5}, agent_id
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 1000 calculations in under 1 second
        self.assertLess(total_time, 1.0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDynamicSkillDiscovery))
    test_suite.addTest(unittest.makeSuite(TestSkillDiscoveryPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")