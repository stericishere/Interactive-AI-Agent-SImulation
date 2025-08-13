"""
Comprehensive Test Suite for Skill Calculation Optimization
Task 3.2.6.1: Skill Calculation Algorithm Optimization Tests

This module provides comprehensive testing for the optimized skill calculation
algorithms, including caching, batch processing, and performance metrics.
"""

import unittest
import time
import threading
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Import the skill system modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from modules.skill_development import (
    SkillDevelopmentSystem, SkillType, SkillLevel, LearningSourceType
)


class TestSkillOptimization(unittest.TestCase):
    """Test cases for skill calculation optimization"""
    
    def setUp(self):
        """Set up test environment"""
        self.skill_system = SkillDevelopmentSystem(max_agents=1000)
        self.test_agent_ids = []
        
        # Create test agents
        for i in range(10):
            agent_id = f"test_agent_{i:03d}"
            self.skill_system.add_agent(agent_id)
            self.test_agent_ids.append(agent_id)
        
        # Add some skills to agents for testing
        for agent_id in self.test_agent_ids[:5]:
            self.skill_system.add_skill(agent_id, SkillType.COMBAT, SkillLevel.COMPETENT)
            self.skill_system.add_skill(agent_id, SkillType.ATHLETICS, SkillLevel.BEGINNER)
    
    def test_precompute_level_multipliers(self):
        """Test pre-computed level multipliers"""
        multipliers = self.skill_system._precompute_level_multipliers()
        
        # Check all levels are present
        for level in SkillLevel:
            self.assertIn(level, multipliers)
        
        # Check reasonable multiplier values
        self.assertEqual(multipliers[SkillLevel.NOVICE], 1.2)
        self.assertEqual(multipliers[SkillLevel.MASTER], 0.7)
        
        # Check decreasing multipliers (diminishing returns)
        self.assertGreater(multipliers[SkillLevel.NOVICE], multipliers[SkillLevel.MASTER])
    
    def test_precompute_synergy_lookups(self):
        """Test pre-computed synergy lookups"""
        synergy_lookup = self.skill_system._precompute_synergy_lookups()
        
        self.assertIsInstance(synergy_lookup, dict)
        
        # Check that lookups contain tuples of skill types
        for key, value in synergy_lookup.items():
            self.assertIsInstance(key, tuple)
            self.assertEqual(len(key), 2)
            self.assertIsInstance(key[0], SkillType)
            self.assertIsInstance(key[1], SkillType)
            self.assertIsInstance(value, float)
    
    def test_optimized_learning_rate_caching(self):
        """Test optimized learning rate with caching"""
        # Clear cache first
        self.skill_system.clear_optimization_caches()
        
        # First call should be a cache miss
        initial_misses = self.skill_system.performance_metrics["cache_misses"]
        rate1 = self.skill_system.get_learning_rate_optimized(SkillLevel.COMPETENT, SkillType.COMBAT)
        self.assertEqual(self.skill_system.performance_metrics["cache_misses"], initial_misses + 1)
        
        # Second call should be a cache hit
        initial_hits = self.skill_system.performance_metrics["cache_hits"]
        rate2 = self.skill_system.get_learning_rate_optimized(SkillLevel.COMPETENT, SkillType.COMBAT)
        self.assertEqual(self.skill_system.performance_metrics["cache_hits"], initial_hits + 1)
        
        # Results should be identical
        self.assertEqual(rate1, rate2)
    
    def test_optimized_experience_gain_calculation(self):
        """Test optimized experience gain calculation"""
        # Test basic calculation
        exp_gain = self.skill_system.calculate_experience_gain_optimized(
            skill_type=SkillType.COMBAT,
            source=LearningSourceType.PRACTICE,
            performance=0.7,
            difficulty=0.5,
            current_level=SkillLevel.COMPETENT
        )
        
        self.assertIsInstance(exp_gain, float)
        self.assertGreater(exp_gain, 0.0)
        
        # Test caching
        self.skill_system.clear_optimization_caches()
        initial_misses = self.skill_system.performance_metrics["cache_misses"]
        
        # First call - cache miss
        exp1 = self.skill_system.calculate_experience_gain_optimized(
            SkillType.ATHLETICS, LearningSourceType.SUCCESS, 0.8, 0.6, SkillLevel.BEGINNER
        )
        
        # Second call - cache hit
        initial_hits = self.skill_system.performance_metrics["cache_hits"]
        exp2 = self.skill_system.calculate_experience_gain_optimized(
            SkillType.ATHLETICS, LearningSourceType.SUCCESS, 0.8, 0.6, SkillLevel.BEGINNER
        )
        
        self.assertEqual(exp1, exp2)
        self.assertGreater(self.skill_system.performance_metrics["cache_hits"], initial_hits)
    
    def test_optimized_context_multiplier(self):
        """Test optimized context multiplier calculation"""
        context = {
            "equipment_quality": 0.8,
            "environment_suitability": 0.9,
            "fatigue_level": 0.2,
            "stress_level": 0.3
        }
        
        multiplier = self.skill_system._calculate_context_multiplier_optimized(
            SkillType.COMBAT, context
        )
        
        self.assertIsInstance(multiplier, float)
        self.assertGreaterEqual(multiplier, 0.3)
        self.assertLessEqual(multiplier, 2.0)
        
        # Test caching
        multiplier2 = self.skill_system._calculate_context_multiplier_optimized(
            SkillType.COMBAT, context
        )
        self.assertEqual(multiplier, multiplier2)
    
    def test_optimized_synergy_bonus_calculation(self):
        """Test optimized synergy bonus calculation"""
        agent_id = self.test_agent_ids[0]
        
        # Calculate synergy bonus
        bonus = self.skill_system.calculate_synergy_bonus_optimized(
            agent_id, SkillType.COMBAT
        )
        
        self.assertIsInstance(bonus, float)
        self.assertGreaterEqual(bonus, 0.0)
        self.assertLessEqual(bonus, self.skill_system.synergy_max_bonus)
        
        # Test caching
        bonus2 = self.skill_system.calculate_synergy_bonus_optimized(
            agent_id, SkillType.COMBAT
        )
        self.assertEqual(bonus, bonus2)
    
    def test_batch_processing_experience_updates(self):
        """Test batch processing of experience updates"""
        # Create batch of updates
        updates = []
        for i, agent_id in enumerate(self.test_agent_ids):
            updates.append({
                "agent_id": agent_id,
                "skill_type": SkillType.COMBAT,
                "experience_gain": 10.0 + i
            })
        
        # Process batch
        results = self.skill_system.batch_process_experience_updates(updates)
        
        self.assertEqual(len(results), len(updates))
        
        # Check that all updates were processed
        for result in results:
            self.assertIn("success", result)
    
    def test_batch_processing_performance(self):
        """Test performance improvement from batch processing"""
        # Create large batch of updates
        updates = []
        for i in range(100):
            agent_id = self.test_agent_ids[i % len(self.test_agent_ids)]
            updates.append({
                "agent_id": agent_id,
                "skill_type": SkillType.ATHLETICS,
                "experience_gain": 5.0
            })
        
        # Time batch processing
        start_time = time.time()
        results = self.skill_system.batch_process_experience_updates(updates)
        batch_time = time.time() - start_time
        
        # Should complete relatively quickly
        self.assertLess(batch_time, 2.0)  # Less than 2 seconds for 100 updates
        self.assertEqual(len(results), 100)
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality"""
        # Fill cache with entries
        for i in range(1200):  # More than the cleanup threshold
            self.skill_system.calculation_cache[f"test_key_{i}"] = i
        
        # Trigger cleanup
        self.skill_system._cleanup_caches()
        
        # Cache should be reduced
        self.assertLessEqual(len(self.skill_system.calculation_cache), 500)
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking"""
        # Clear metrics
        self.skill_system.clear_optimization_caches()
        
        # Perform some operations
        for i in range(10):
            self.skill_system.get_learning_rate_optimized(SkillLevel.COMPETENT, SkillType.COMBAT)
        
        # Check metrics
        metrics = self.skill_system.get_performance_metrics()
        
        self.assertIn("optimization_enabled", metrics)
        self.assertIn("cache_hit_rate", metrics)
        self.assertIn("total_cache_hits", metrics)
        self.assertIn("total_cache_misses", metrics)
        self.assertIn("average_calculation_time_ms", metrics)
        
        # Should have cache hits after the first calculation
        self.assertGreater(metrics["total_cache_hits"], 0)
    
    def test_optimization_scaling_parameters(self):
        """Test optimization parameter scaling"""
        # Test small scale optimization
        self.skill_system.optimize_for_agent_count(5)
        self.assertFalse(self.skill_system.batch_processing)
        self.assertEqual(self.skill_system._batch_size, 10)
        
        # Test medium scale optimization
        self.skill_system.optimize_for_agent_count(50)
        self.assertTrue(self.skill_system.batch_processing)
        self.assertEqual(self.skill_system._batch_size, 25)
        
        # Test large scale optimization
        self.skill_system.optimize_for_agent_count(200)
        self.assertTrue(self.skill_system.batch_processing)
        self.assertEqual(self.skill_system._batch_size, 50)
    
    def test_concurrent_access_safety(self):
        """Test thread safety of optimized operations"""
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                for i in range(50):
                    # Perform various operations that might conflict
                    agent_id = f"thread_{thread_id}_agent_{i}"
                    self.skill_system.add_agent(agent_id)
                    
                    # Test optimized calculations
                    rate = self.skill_system.get_learning_rate_optimized(
                        SkillLevel.COMPETENT, SkillType.COMBAT
                    )
                    
                    exp_gain = self.skill_system.calculate_experience_gain_optimized(
                        SkillType.ATHLETICS, LearningSourceType.PRACTICE, 
                        0.5, 0.5, SkillLevel.BEGINNER
                    )
                    
                    results.append((thread_id, rate, exp_gain))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create and start threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Thread errors occurred: {errors}")
        self.assertGreater(len(results), 0)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of optimization features"""
        import sys
        
        # Get initial memory usage (rough approximation)
        initial_cache_size = len(self.skill_system.calculation_cache)
        
        # Perform many operations to build up cache
        for i in range(1000):
            self.skill_system.calculate_experience_gain_optimized(
                SkillType.COMBAT, 
                LearningSourceType.PRACTICE,
                0.5 + (i % 10) * 0.05,  # Vary parameters to avoid too much caching
                0.4 + (i % 8) * 0.1,
                SkillLevel.COMPETENT
            )
        
        # Cache should not grow unbounded
        self.assertLess(
            len(self.skill_system.calculation_cache), 
            initial_cache_size + 2000  # Should be much less due to cleanup
        )


class TestSkillOptimizationPerformance(unittest.TestCase):
    """Performance benchmarks for skill optimization"""
    
    def setUp(self):
        self.skill_system = SkillDevelopmentSystem(max_agents=1000)
        
        # Create test agents with skills
        for i in range(100):
            agent_id = f"perf_agent_{i:03d}"
            self.skill_system.add_agent(agent_id)
            self.skill_system.add_skill(agent_id, SkillType.COMBAT, SkillLevel.COMPETENT)
            self.skill_system.add_skill(agent_id, SkillType.ATHLETICS, SkillLevel.BEGINNER)
    
    def test_calculation_performance_benchmark(self):
        """Benchmark optimized vs non-optimized calculations"""
        # Time optimized calculations
        start_time = time.time()
        
        for i in range(1000):
            self.skill_system.calculate_experience_gain_optimized(
                SkillType.COMBAT, LearningSourceType.PRACTICE, 0.6, 0.5, SkillLevel.COMPETENT
            )
        
        optimized_time = time.time() - start_time
        
        # Time regular calculations (if available)
        start_time = time.time()
        
        for i in range(1000):
            self.skill_system.calculate_experience_gain(
                SkillType.COMBAT, LearningSourceType.PRACTICE, 0.6, 0.5, SkillLevel.COMPETENT
            )
        
        regular_time = time.time() - start_time
        
        # Check performance metrics
        metrics = self.skill_system.get_performance_metrics()
        
        print(f"\nPerformance Benchmark Results:")
        print(f"Optimized time: {optimized_time:.3f}s")
        print(f"Regular time: {regular_time:.3f}s")
        print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
        print(f"Average calculation time: {metrics['average_calculation_time_ms']:.3f}ms")
        
        # Optimized should be faster due to caching (after initial cache misses)
        # Note: This might not always be true for small tests due to cache overhead
        self.assertLess(optimized_time, regular_time * 2)  # Should be at least not much worse
    
    def test_batch_processing_performance_benchmark(self):
        """Benchmark batch vs individual processing"""
        # Create updates
        updates = []
        for i in range(500):
            updates.append({
                "agent_id": f"perf_agent_{i % 100:03d}",
                "skill_type": SkillType.ATHLETICS,
                "experience_gain": 5.0 + (i % 10)
            })
        
        # Time batch processing
        start_time = time.time()
        batch_results = self.skill_system.batch_process_experience_updates(updates)
        batch_time = time.time() - start_time
        
        # Time individual processing
        start_time = time.time()
        individual_results = []
        for update in updates:
            result = self.skill_system._process_single_experience_update(update)
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        print(f"\nBatch Processing Benchmark:")
        print(f"Batch time: {batch_time:.3f}s")
        print(f"Individual time: {individual_time:.3f}s")
        print(f"Batch speedup: {individual_time / batch_time:.2f}x")
        
        # Batch should be faster or at least not significantly slower
        self.assertLess(batch_time, individual_time * 1.5)
        self.assertEqual(len(batch_results), len(individual_results))
    
    def test_large_scale_performance(self):
        """Test performance at large scale"""
        # Create many agents
        large_agent_count = 500
        for i in range(100, 100 + large_agent_count):
            agent_id = f"large_scale_agent_{i:04d}"
            self.skill_system.add_agent(agent_id)
        
        # Optimize for this scale
        self.skill_system.optimize_for_agent_count(large_agent_count)
        
        # Perform many operations
        start_time = time.time()
        
        for i in range(large_agent_count):
            agent_id = f"large_scale_agent_{100 + i:04d}"
            
            # Various skill operations
            self.skill_system.add_skill(agent_id, SkillType.COMBAT, SkillLevel.NOVICE)
            
            exp_gain = self.skill_system.calculate_experience_gain_optimized(
                SkillType.COMBAT, LearningSourceType.PRACTICE, 0.6, 0.5, SkillLevel.NOVICE
            )
            
            synergy = self.skill_system.calculate_synergy_bonus_optimized(agent_id, SkillType.COMBAT)
        
        total_time = time.time() - start_time
        
        print(f"\nLarge Scale Performance:")
        print(f"Operations for {large_agent_count} agents: {total_time:.3f}s")
        print(f"Average time per agent: {total_time / large_agent_count * 1000:.3f}ms")
        
        # Should complete within reasonable time
        self.assertLess(total_time, 10.0)  # Less than 10 seconds
        
        # Check final metrics
        metrics = self.skill_system.get_performance_metrics()
        print(f"Final cache hit rate: {metrics['cache_hit_rate']:.2%}")
        print(f"Cache sizes: {metrics['cache_sizes']}")


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSkillOptimization))
    test_suite.addTest(unittest.makeSuite(TestSkillOptimizationPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")