"""
Integration Tests for Enhanced Skill Development System
Task 3.2: Integration testing for skill discovery, optimization, and execution

This module provides comprehensive integration testing for the complete skill
development system, including discovery, optimization, and execution integration.
"""

import unittest
import time
import random
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Import the skill system modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.skill_development import (
    SkillDevelopmentSystem, SkillType, SkillLevel, LearningSourceType
)
from modules.skill_execution import SkillExecutionModule


class MockAgentState:
    """Mock agent state for testing"""
    
    def __init__(self, name: str):
        self.name = name
        self.proprioception = {}
        self.memory_log = []
    
    def add_to_memory(self, message: str):
        """Mock memory addition"""
        self.memory_log.append(message)


class TestSkillSystemIntegration(unittest.TestCase):
    """Integration tests for the complete skill system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.skill_system = SkillDevelopmentSystem(max_agents=100)
        
        # Create mock agent states
        self.agent_states = {}
        self.skill_executors = {}
        
        for i in range(5):
            agent_name = f"integration_agent_{i:02d}"
            agent_state = MockAgentState(agent_name)
            
            self.agent_states[agent_name] = agent_state
            self.skill_executors[agent_name] = SkillExecutionModule(
                agent_state, self.skill_system
            )
    
    def test_discovery_execution_integration(self):
        """Test integration between skill discovery and execution"""
        agent_name = "integration_agent_00"
        executor = self.skill_executors[agent_name]
        
        # Set up a decision that should trigger skill discovery
        executor.agent_state.proprioception["current_decision"] = "practice sword fighting techniques"
        
        # Execute the action
        executor.run()
        
        # Check execution result
        result = executor.agent_state.proprioception.get("executed_action")
        self.assertIsNotNone(result)
        
        # Check if any skills were discovered
        if "discovered_skills" in result:
            self.assertIsInstance(result["discovered_skills"], list)
            
            # If skills were discovered, check memory log
            if result["discovered_skills"]:
                self.assertGreater(len(executor.agent_state.memory_log), 0)
    
    def test_optimization_discovery_integration(self):
        """Test integration between optimization and discovery systems"""
        agent_name = "integration_agent_01"
        
        # Perform many discovery operations to test optimization
        start_time = time.time()
        
        total_discovered = []
        for i in range(100):
            discovered = self.skill_system.discover_skills_from_actions(
                agent_id=agent_name,
                action=f"practice technique {i}",
                performance=0.5 + (i % 10) * 0.05,
                context={"complexity": 0.6, "learning_opportunity": True}
            )
            total_discovered.extend(discovered)
        
        total_time = time.time() - start_time
        
        # Should complete efficiently
        self.assertLess(total_time, 2.0)
        
        # Check optimization metrics
        metrics = self.skill_system.get_performance_metrics()
        self.assertGreater(metrics["cache_hit_rate"], 0.0)
    
    def test_full_skill_lifecycle_integration(self):
        """Test complete skill lifecycle: discovery -> practice -> optimization"""
        agent_name = "integration_agent_02"
        
        # Step 1: Discover skills through actions
        discovered_skills = self.skill_system.discover_skills_from_actions(
            agent_id=agent_name,
            action="craft a wooden sword",
            performance=0.8,
            context={
                "environment": "workshop",
                "creativity_required": True,
                "complexity": 0.7
            }
        )
        
        # Step 2: Practice discovered skills
        agent_skills = self.skill_system.get_agent_skills(agent_name)
        initial_skill_count = len(agent_skills)
        
        if discovered_skills:
            for skill_type in discovered_skills:
                # Practice the skill
                practice_result = self.skill_system.practice_skill(
                    agent_id=agent_name,
                    skill_type=skill_type,
                    hours=2.0,
                    focus_level=0.8
                )
                
                self.assertTrue(practice_result["success"])
                self.assertGreater(practice_result["experience_gained"], 0)
        
        # Step 3: Test optimization with batch operations
        updates = []
        for skill_type in SkillType:
            updates.append({
                "agent_id": agent_name,
                "skill_type": skill_type,
                "experience_gain": 10.0
            })
        
        batch_results = self.skill_system.batch_process_experience_updates(updates)
        self.assertEqual(len(batch_results), len(SkillType))
        
        # Check final state
        final_skills = self.skill_system.get_agent_skills(agent_name)
        final_skill_count = len(final_skills)
        
        # Should have more skills after the lifecycle
        self.assertGreaterEqual(final_skill_count, initial_skill_count)
    
    def test_concurrent_agent_operations(self):
        """Test concurrent operations across multiple agents"""
        import threading
        
        results = {}
        errors = []
        
        def agent_worker(agent_name):
            try:
                executor = self.skill_executors[agent_name]
                
                # Perform various operations
                operations = [
                    "practice combat techniques",
                    "study ancient texts",
                    "craft healing potions",
                    "negotiate trade deals",
                    "explore wilderness"
                ]
                
                agent_results = []
                for operation in operations:
                    # Discovery attempt
                    discovered = self.skill_system.discover_skills_from_actions(
                        agent_id=agent_name,
                        action=operation,
                        performance=random.uniform(0.4, 0.9),
                        context={"complexity": random.uniform(0.3, 0.8)}
                    )
                    
                    # Execution
                    executor.agent_state.proprioception["current_decision"] = operation
                    executor.run()
                    
                    result = executor.agent_state.proprioception.get("executed_action")
                    agent_results.append({
                        "operation": operation,
                        "discovered": len(discovered),
                        "executed": result is not None
                    })
                
                results[agent_name] = agent_results
                
            except Exception as e:
                errors.append((agent_name, str(e)))
        
        # Start concurrent operations
        threads = []
        for agent_name in list(self.agent_states.keys())[:3]:  # Use 3 agents
            thread = threading.Thread(target=agent_worker, args=(agent_name,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")
        self.assertEqual(len(results), 3)
        
        # Verify all agents completed operations
        for agent_name, agent_results in results.items():
            self.assertEqual(len(agent_results), 5)
            for result in agent_results:
                self.assertTrue(result["executed"])
    
    def test_skill_system_state_consistency(self):
        """Test state consistency across system components"""
        agent_name = "integration_agent_03"
        
        # Add skills through different pathways
        
        # 1. Direct skill addition
        self.skill_system.add_skill(agent_name, SkillType.COMBAT, SkillLevel.BEGINNER)
        
        # 2. Skill discovery
        self.skill_system._initialize_discovered_skill(
            agent_name, SkillType.STEALTH, 15.0
        )
        
        # 3. Experience addition
        self.skill_system.add_experience(agent_name, SkillType.COMBAT, 50.0)
        
        # Verify consistency
        agent_skills = self.skill_system.get_agent_skills(agent_name)
        
        # Check combat skill
        self.assertIn(SkillType.COMBAT, agent_skills)
        combat_skill = agent_skills[SkillType.COMBAT]
        self.assertGreaterEqual(combat_skill.experience_points, 150.0)  # Initial + added
        
        # Check stealth skill
        self.assertIn(SkillType.STEALTH, agent_skills)
        stealth_skill = agent_skills[SkillType.STEALTH]
        self.assertEqual(stealth_skill.experience_points, 15.0)
        
        # Check skill status
        status = self.skill_system.get_skill_status(agent_name)
        self.assertEqual(status["total_skills"], 2)
        
        # Check performance calculations still work
        performance = self.skill_system.calculate_skill_performance(
            agent_name, SkillType.COMBAT, 0.5
        )
        self.assertIsInstance(performance, dict)
        self.assertIn("success_probability", performance)
    
    def test_memory_and_performance_integration(self):
        """Test integration between memory management and performance optimization"""
        # Create scenario with many agents and operations
        agent_names = []
        for i in range(20):
            agent_name = f"memory_test_agent_{i:02d}"
            self.skill_system.add_agent(agent_name)
            agent_names.append(agent_name)
        
        # Perform operations that will stress memory and cache systems
        start_time = time.time()
        
        for iteration in range(5):
            # Discovery operations
            for agent_name in agent_names:
                self.skill_system.discover_skills_from_actions(
                    agent_id=agent_name,
                    action=f"practice skill {iteration}",
                    performance=0.6,
                    context={"complexity": 0.5}
                )
            
            # Optimization operations
            updates = []
            for agent_name in agent_names[:10]:  # Batch for half the agents
                updates.append({
                    "agent_id": agent_name,
                    "skill_type": SkillType.LEARNING,
                    "experience_gain": 5.0
                })
            
            self.skill_system.batch_process_experience_updates(updates)
        
        total_time = time.time() - start_time
        
        # Should complete within reasonable time despite scale
        self.assertLess(total_time, 5.0)
        
        # Check memory usage is reasonable
        metrics = self.skill_system.get_performance_metrics()
        
        # Caches should not be excessively large
        for cache_name, cache_size in metrics["cache_sizes"].items():
            self.assertLess(cache_size, 2000, f"{cache_name} cache too large: {cache_size}")
        
        # Should have good cache performance
        self.assertGreater(metrics["cache_hit_rate"], 0.1)
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components"""
        # Test with invalid agent
        discovered = self.skill_system.discover_skills_from_actions(
            agent_id="nonexistent_agent",
            action="test action",
            performance=0.5,
            context={}
        )
        self.assertEqual(discovered, [])
        
        # Test with invalid skill execution
        invalid_executor = SkillExecutionModule(
            MockAgentState("invalid_agent"), 
            self.skill_system
        )
        
        # Should not crash
        invalid_executor.run()
        
        # Test batch processing with invalid updates
        invalid_updates = [
            {"agent_id": "nonexistent", "skill_type": SkillType.COMBAT, "experience_gain": 10.0},
            {"skill_type": SkillType.ATHLETICS, "experience_gain": 5.0},  # Missing agent_id
            {"agent_id": "integration_agent_00", "experience_gain": 7.0}  # Missing skill_type
        ]
        
        results = self.skill_system.batch_process_experience_updates(invalid_updates)
        self.assertEqual(len(results), len(invalid_updates))
        
        # Some should fail gracefully
        success_count = sum(1 for r in results if r.get("success", False))
        self.assertLess(success_count, len(invalid_updates))  # Some should fail
        
        # But system should remain stable
        metrics = self.skill_system.get_performance_metrics()
        self.assertIsInstance(metrics, dict)
    
    def test_system_performance_under_load(self):
        """Test system performance under high load"""
        # Create many agents
        agent_names = []
        for i in range(50):
            agent_name = f"load_test_agent_{i:03d}"
            self.skill_system.add_agent(agent_name)
            agent_names.append(agent_name)
        
        # Optimize for this scale
        self.skill_system.optimize_for_agent_count(len(agent_names))
        
        # Perform high-load operations
        start_time = time.time()
        
        operations_completed = 0
        
        # Discovery operations
        for agent_name in agent_names:
            for action in ["combat training", "study books", "craft items"]:
                discovered = self.skill_system.discover_skills_from_actions(
                    agent_id=agent_name,
                    action=action,
                    performance=random.uniform(0.4, 0.9),
                    context={"complexity": random.uniform(0.3, 0.8)}
                )
                operations_completed += 1
        
        # Batch experience updates
        updates = []
        for agent_name in agent_names:
            for skill_type in [SkillType.COMBAT, SkillType.LEARNING, SkillType.CRAFTING]:
                updates.append({
                    "agent_id": agent_name,
                    "skill_type": skill_type,
                    "experience_gain": random.uniform(1.0, 10.0)
                })
        
        batch_results = self.skill_system.batch_process_experience_updates(updates)
        operations_completed += len(batch_results)
        
        total_time = time.time() - start_time
        
        print(f"\nLoad Test Results:")
        print(f"Agents: {len(agent_names)}")
        print(f"Operations completed: {operations_completed}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Operations per second: {operations_completed / total_time:.1f}")
        
        # Performance requirements
        self.assertLess(total_time, 10.0)  # Should complete within 10 seconds
        self.assertGreater(operations_completed / total_time, 50)  # At least 50 ops/sec
        
        # Check system stability
        metrics = self.skill_system.get_performance_metrics()
        self.assertGreater(metrics["cache_hit_rate"], 0.2)  # Good cache performance
        self.assertLess(metrics["average_calculation_time_ms"], 50)  # Fast calculations


class TestSkillSystemRegressionTests(unittest.TestCase):
    """Regression tests for skill system stability"""
    
    def setUp(self):
        self.skill_system = SkillDevelopmentSystem()
    
    def test_skill_data_integrity(self):
        """Test that skill data remains consistent over operations"""
        agent_id = "regression_agent"
        self.skill_system.add_agent(agent_id)
        
        # Add skill and record initial state
        self.skill_system.add_skill(agent_id, SkillType.COMBAT, SkillLevel.COMPETENT)
        initial_skill = self.skill_system.get_skill(agent_id, SkillType.COMBAT)
        initial_exp = initial_skill.experience_points
        
        # Perform various operations
        for i in range(10):
            self.skill_system.add_experience(agent_id, SkillType.COMBAT, 5.0)
            
            # Discovery attempts (shouldn't affect existing skills)
            self.skill_system.discover_skills_from_actions(
                agent_id, f"action {i}", 0.5, {}
            )
            
            # Optimization operations
            self.skill_system.calculate_experience_gain_optimized(
                SkillType.COMBAT, LearningSourceType.PRACTICE, 0.6, 0.5, SkillLevel.COMPETENT
            )
        
        # Check data integrity
        final_skill = self.skill_system.get_skill(agent_id, SkillType.COMBAT)
        final_exp = final_skill.experience_points
        
        # Experience should have increased by expected amount
        expected_exp = initial_exp + 50.0  # 10 * 5.0
        self.assertAlmostEqual(final_exp, expected_exp, delta=0.1)
        
        # Skill should still exist and be valid
        self.assertIsNotNone(final_skill)
        self.assertEqual(final_skill.skill_type, SkillType.COMBAT)
    
    def test_system_state_after_cache_clears(self):
        """Test system behavior after cache operations"""
        agent_id = "cache_test_agent"
        self.skill_system.add_agent(agent_id)
        
        # Build up some state
        self.skill_system.add_skill(agent_id, SkillType.ATHLETICS, SkillLevel.BEGINNER)
        
        # Perform operations to populate caches
        for i in range(20):
            self.skill_system.calculate_experience_gain_optimized(
                SkillType.ATHLETICS, LearningSourceType.PRACTICE, 0.5, 0.5, SkillLevel.BEGINNER
            )
        
        # Clear caches
        self.skill_system.clear_optimization_caches()
        
        # System should still function correctly
        skill = self.skill_system.get_skill(agent_id, SkillType.ATHLETICS)
        self.assertIsNotNone(skill)
        
        # Operations should still work
        exp_gain = self.skill_system.calculate_experience_gain_optimized(
            SkillType.ATHLETICS, LearningSourceType.PRACTICE, 0.5, 0.5, SkillLevel.BEGINNER
        )
        self.assertGreater(exp_gain, 0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSkillSystemIntegration))
    test_suite.addTest(unittest.makeSuite(TestSkillSystemRegressionTests))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nIntegration Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split(chr(10))[0] if 'AssertionError:' in traceback else 'Unknown failure'}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2] if traceback else 'Unknown error'}")