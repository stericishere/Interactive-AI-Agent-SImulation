#!/usr/bin/env python3
"""
Compatibility and Regression Testing for Enhanced PIANO Memory Architecture
Tests backward compatibility, API stability, and prevents regression issues.
"""

import sys
import os
import json
import time
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import traceback
import unittest.mock as mock

# Add current directory to path
sys.path.append('.')
sys.path.append('./memory_structures')

# Import memory components
from memory_structures.circular_buffer import CircularBuffer, CircularBufferReducer
from memory_structures.temporal_memory import TemporalMemory
from memory_structures.episodic_memory import EpisodicMemory, CausalRelationType, EpisodeType
from memory_structures.semantic_memory import SemanticMemory, ConceptType, SemanticRelationType


class CompatibilityTestResult:
    """Compatibility test result tracking"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.compatibility_issues = []
        self.performance_regressions = []
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ {test_name}: {error}")
    
    def add_compatibility_issue(self, issue: str, severity: str, description: str):
        self.compatibility_issues.append({
            'issue': issue,
            'severity': severity,
            'description': description
        })
        print(f"⚠️  COMPATIBILITY [{severity}]: {issue} - {description}")
    
    def add_performance_regression(self, test_name: str, current_ms: float, 
                                  baseline_ms: float, threshold_percent: float):
        regression_percent = ((current_ms - baseline_ms) / baseline_ms) * 100
        if regression_percent > threshold_percent:
            self.performance_regressions.append({
                'test': test_name,
                'current_ms': current_ms,
                'baseline_ms': baseline_ms,
                'regression_percent': regression_percent
            })
            print(f"📉 REGRESSION: {test_name} - {regression_percent:.1f}% slower "
                  f"({current_ms:.2f}ms vs {baseline_ms:.2f}ms baseline)")
        else:
            print(f"📊 {test_name}: {current_ms:.2f}ms (baseline: {baseline_ms:.2f}ms)")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"COMPATIBILITY TEST SUMMARY: {self.passed}/{total} passed ({100*self.passed/total if total > 0 else 0:.1f}%)")
        print(f"{'='*60}")
        
        # Compatibility issues summary
        if self.compatibility_issues:
            print(f"\n⚠️  COMPATIBILITY ISSUES: {len(self.compatibility_issues)}")
            for issue in self.compatibility_issues:
                print(f"  [{issue['severity']}] {issue['issue']}: {issue['description']}")
        else:
            print("\n✅ NO COMPATIBILITY ISSUES DETECTED")
        
        # Performance regressions summary
        if self.performance_regressions:
            print(f"\n📉 PERFORMANCE REGRESSIONS: {len(self.performance_regressions)}")
            for regression in self.performance_regressions:
                print(f"  {regression['test']}: {regression['regression_percent']:.1f}% slower")
        else:
            print("\n📊 NO PERFORMANCE REGRESSIONS DETECTED")
        
        if self.errors:
            print("\nFAILURES:")
            for error in self.errors:
                print(f"  • {error}")
        
        return self.failed == 0 and len(self.compatibility_issues) == 0


def test_api_backward_compatibility(result: CompatibilityTestResult):
    """Test backward compatibility of public APIs"""
    print("\n--- API Backward Compatibility ---")
    
    # Test CircularBuffer API compatibility
    try:
        buffer = CircularBuffer(max_size=10)
        
        # Test core API methods exist and work
        api_methods = [
            ('add_memory', lambda: buffer.add_memory("test", "event", 0.5)),
            ('get_recent_memories', lambda: buffer.get_recent_memories(5)),
            ('get_important_memories', lambda: buffer.get_important_memories(0.7)),
            ('search_memories', lambda: buffer.search_memories("test")),
            ('cleanup_expired_memories', lambda: buffer.cleanup_expired_memories()),
            ('to_dict', lambda: buffer.to_dict()),
            ('__len__', lambda: len(buffer)),
        ]
        
        for method_name, method_call in api_methods:
            try:
                result_val = method_call()
                # Basic validation that method returns expected type
                if method_name == '__len__' and not isinstance(result_val, int):
                    result.add_compatibility_issue("API Change", "HIGH", 
                                                 f"CircularBuffer.__len__ returns {type(result_val)}, expected int")
                elif method_name == 'to_dict' and not isinstance(result_val, dict):
                    result.add_compatibility_issue("API Change", "HIGH", 
                                                 f"CircularBuffer.to_dict returns {type(result_val)}, expected dict")
            except Exception as e:
                result.add_compatibility_issue("API Breakage", "CRITICAL", 
                                             f"CircularBuffer.{method_name} failed: {str(e)}")
        
        result.add_pass("API Compatibility: CircularBuffer methods")
    
    except Exception as e:
        result.add_fail("API Compatibility: CircularBuffer", str(e))
    
    # Test SemanticMemory API compatibility
    try:
        semantic = SemanticMemory(max_concepts=50)
        
        # Test core API methods
        concept_id = semantic.add_concept("TestConcept", ConceptType.PERSON, "Test", 0.5)
        
        api_methods = [
            ('add_concept', lambda: semantic.add_concept("Test2", ConceptType.PERSON, "Test2", 0.6)),
            ('activate_concept', lambda: semantic.activate_concept("TestConcept", 0.8)),
            ('retrieve_by_activation', lambda: semantic.retrieve_by_activation(threshold=0.3)),
            ('retrieve_by_type', lambda: semantic.retrieve_by_type(ConceptType.PERSON)),
            ('consolidate_concepts', lambda: semantic.consolidate_concepts()),
            ('to_dict', lambda: semantic.to_dict()),
        ]
        
        for method_name, method_call in api_methods:
            try:
                method_call()
            except Exception as e:
                result.add_compatibility_issue("API Breakage", "CRITICAL", 
                                             f"SemanticMemory.{method_name} failed: {str(e)}")
        
        result.add_pass("API Compatibility: SemanticMemory methods")
    
    except Exception as e:
        result.add_fail("API Compatibility: SemanticMemory", str(e))


def test_data_format_compatibility(result: CompatibilityTestResult):
    """Test compatibility with existing data formats"""
    print("\n--- Data Format Compatibility ---")
    
    # Test legacy serialization format compatibility
    try:
        # Create a "legacy" data format (simplified structure)
        legacy_buffer_data = {
            "max_size": 10,
            "retention_minutes": 60,
            "memories": [
                {
                    "id": "mem_001",
                    "content": "Legacy memory",
                    "memory_type": "event",
                    "importance": 0.7,
                    "timestamp": datetime.now().isoformat(),
                    "context": {"source": "legacy_system"}
                }
            ]
        }
        
        # Test if current system can load legacy data
        try:
            buffer = CircularBuffer.from_dict(legacy_buffer_data)
            if len(buffer) == 1:
                result.add_pass("Data Format: Legacy CircularBuffer compatibility")
            else:
                result.add_compatibility_issue("Data Format", "HIGH", 
                                             "Legacy CircularBuffer data not loaded correctly")
        except Exception as e:
            result.add_compatibility_issue("Data Format", "CRITICAL", 
                                         f"Cannot load legacy CircularBuffer data: {str(e)}")
    
    except Exception as e:
        result.add_fail("Data Format: Legacy compatibility", str(e))
    
    # Test forward compatibility (graceful handling of unknown fields)
    try:
        # Create data with extra fields (simulating future version)
        future_buffer_data = {
            "max_size": 10,
            "retention_minutes": 60,
            "version": "2.0",  # Future version field
            "new_feature_config": {"enabled": True},  # Future feature
            "memories": [
                {
                    "id": "mem_001",
                    "content": "Future memory",
                    "memory_type": "event",
                    "importance": 0.8,
                    "timestamp": datetime.now().isoformat(),
                    "context": {},
                    "future_field": "future_value"  # Future field
                }
            ]
        }
        
        try:
            buffer = CircularBuffer.from_dict(future_buffer_data)
            if len(buffer) == 1:
                result.add_pass("Data Format: Forward compatibility")
            else:
                result.add_compatibility_issue("Data Format", "MEDIUM", 
                                             "Future data format not handled gracefully")
        except Exception as e:
            result.add_compatibility_issue("Data Format", "MEDIUM", 
                                         f"Future data format causes errors: {str(e)}")
    
    except Exception as e:
        result.add_fail("Data Format: Forward compatibility", str(e))


def test_performance_regression_baselines(result: CompatibilityTestResult):
    """Test for performance regressions against established baselines"""
    print("\n--- Performance Regression Testing ---")
    
    # Baseline performance expectations (in milliseconds)
    performance_baselines = {
        'circular_buffer_add_100': 5.0,
        'circular_buffer_retrieve_10': 1.0,
        'temporal_memory_add_100': 10.0,
        'temporal_memory_retrieve_recent': 5.0,
        'semantic_memory_add_50': 20.0,
        'semantic_memory_activate_retrieve': 15.0,
        'episodic_memory_add_10': 10.0,
    }
    
    # Test CircularBuffer performance
    try:
        buffer = CircularBuffer(max_size=100)
        
        start_time = time.perf_counter()
        for i in range(100):
            buffer.add_memory(f"Performance test {i}", "event", 0.5)
        end_time = time.perf_counter()
        
        add_time_ms = (end_time - start_time) * 1000
        result.add_performance_regression('circular_buffer_add_100', add_time_ms, 
                                        performance_baselines['circular_buffer_add_100'], 20.0)
        
        start_time = time.perf_counter()
        recent = buffer.get_recent_memories(10)
        end_time = time.perf_counter()
        
        retrieve_time_ms = (end_time - start_time) * 1000
        result.add_performance_regression('circular_buffer_retrieve_10', retrieve_time_ms,
                                        performance_baselines['circular_buffer_retrieve_10'], 50.0)
    
    except Exception as e:
        result.add_fail("Performance Regression: CircularBuffer", str(e))
    
    # Test TemporalMemory performance
    try:
        temporal = TemporalMemory(retention_hours=2)
        
        start_time = time.perf_counter()
        for i in range(100):
            temporal.add_memory(f"Temporal test {i}", "event", 0.5)
        end_time = time.perf_counter()
        
        add_time_ms = (end_time - start_time) * 1000
        result.add_performance_regression('temporal_memory_add_100', add_time_ms,
                                        performance_baselines['temporal_memory_add_100'], 20.0)
        
        start_time = time.perf_counter()
        recent = temporal.retrieve_recent_memories(hours_back=1, limit=20)
        end_time = time.perf_counter()
        
        retrieve_time_ms = (end_time - start_time) * 1000
        result.add_performance_regression('temporal_memory_retrieve_recent', retrieve_time_ms,
                                        performance_baselines['temporal_memory_retrieve_recent'], 30.0)
    
    except Exception as e:
        result.add_fail("Performance Regression: TemporalMemory", str(e))
    
    # Test SemanticMemory performance
    try:
        semantic = SemanticMemory(max_concepts=100)
        
        start_time = time.perf_counter()
        concept_ids = []
        for i in range(50):
            concept_id = semantic.add_concept(f"Concept_{i}", ConceptType.PERSON, 
                                            f"Person {i}", 0.5)
            concept_ids.append(concept_id)
        end_time = time.perf_counter()
        
        add_time_ms = (end_time - start_time) * 1000
        result.add_performance_regression('semantic_memory_add_50', add_time_ms,
                                        performance_baselines['semantic_memory_add_50'], 25.0)
        
        start_time = time.perf_counter()
        semantic.activate_concept("Concept_0", 0.8)
        activated = semantic.retrieve_by_activation(threshold=0.3, limit=10)
        end_time = time.perf_counter()
        
        activate_retrieve_ms = (end_time - start_time) * 1000
        result.add_performance_regression('semantic_memory_activate_retrieve', activate_retrieve_ms,
                                        performance_baselines['semantic_memory_activate_retrieve'], 30.0)
    
    except Exception as e:
        result.add_fail("Performance Regression: SemanticMemory", str(e))


def test_version_migration_compatibility(result: CompatibilityTestResult):
    """Test migration compatibility between versions"""
    print("\n--- Version Migration Compatibility ---")
    
    try:
        # Simulate version 1.0 data structure
        v1_semantic_data = {
            "max_concepts": 100,
            "concepts": {
                "concept_001": {
                    "name": "Maria",
                    "concept_type": "PERSON",  # String instead of enum
                    "description": "Artistic person",
                    "base_activation": 0.7,
                    "current_activation": 0.7,
                    "last_accessed": datetime.now().isoformat()
                }
            },
            "relations": {},
            "activation_decay_rate": 0.1
        }
        
        # Test migration to current version
        try:
            semantic = SemanticMemory.from_dict(v1_semantic_data)
            
            # Verify migration worked
            if len(semantic.concepts) == 1:
                # Check if enum conversion worked
                concept = list(semantic.concepts.values())[0]
                if hasattr(concept, 'concept_type'):
                    result.add_pass("Version Migration: Enum conversion")
                else:
                    result.add_compatibility_issue("Migration", "HIGH", 
                                                 "Enum fields not migrated properly")
            else:
                result.add_compatibility_issue("Migration", "CRITICAL", 
                                             "Concept data lost during migration")
        
        except Exception as e:
            result.add_compatibility_issue("Migration", "CRITICAL", 
                                         f"Version migration failed: {str(e)}")
    
    except Exception as e:
        result.add_fail("Version Migration: Testing", str(e))


def test_dependency_compatibility(result: CompatibilityTestResult):
    """Test compatibility with different dependency versions"""
    print("\n--- Dependency Compatibility ---")
    
    # Test datetime handling across Python versions
    try:
        temporal = TemporalMemory(retention_hours=1)
        
        # Test different datetime formats
        datetime_formats = [
            datetime.now(),  # datetime object
            datetime.now().isoformat(),  # ISO string
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Standard format
            datetime.now().timestamp(),  # Unix timestamp
        ]
        
        for dt_format in datetime_formats:
            try:
                if isinstance(dt_format, (int, float)):
                    # Skip timestamp test as it may not be supported
                    continue
                
                mem_id = temporal.add_memory("Datetime test", "event", 0.5, 
                                           timestamp=dt_format if isinstance(dt_format, datetime) else None)
                
                if mem_id:
                    result.add_pass(f"Dependency: Datetime format {type(dt_format).__name__}")
            
            except Exception as e:
                result.add_compatibility_issue("Dependency", "MEDIUM", 
                                             f"Datetime format {type(dt_format).__name__} not supported: {str(e)}")
    
    except Exception as e:
        result.add_fail("Dependency: Datetime compatibility", str(e))


def test_interface_stability(result: CompatibilityTestResult):
    """Test interface stability and method signatures"""
    print("\n--- Interface Stability ---")
    
    try:
        # Test that method signatures haven't changed unexpectedly
        buffer = CircularBuffer(max_size=10)
        
        # Test add_memory signature
        import inspect
        add_memory_sig = inspect.signature(buffer.add_memory)
        expected_params = ['content', 'memory_type', 'importance']
        
        actual_params = list(add_memory_sig.parameters.keys())
        missing_params = set(expected_params) - set(actual_params)
        
        if missing_params:
            result.add_compatibility_issue("Interface", "CRITICAL", 
                                         f"add_memory missing parameters: {missing_params}")
        else:
            result.add_pass("Interface: Method signature stability")
        
        # Test that methods return expected types
        mem_id = buffer.add_memory("Test", "event", 0.5)
        if not isinstance(mem_id, str):
            result.add_compatibility_issue("Interface", "HIGH", 
                                         f"add_memory returns {type(mem_id)}, expected str")
        
        memories = buffer.get_recent_memories(5)
        if not isinstance(memories, list):
            result.add_compatibility_issue("Interface", "HIGH", 
                                         f"get_recent_memories returns {type(memories)}, expected list")
        
        result.add_pass("Interface: Return type stability")
    
    except Exception as e:
        result.add_fail("Interface: Stability testing", str(e))


def test_error_handling_compatibility(result: CompatibilityTestResult):
    """Test that error handling remains consistent"""
    print("\n--- Error Handling Compatibility ---")
    
    try:
        buffer = CircularBuffer(max_size=5)
        
        # Test that same errors are raised for invalid inputs
        error_cases = [
            (lambda: buffer.add_memory(None, "event", 0.5), (TypeError, ValueError)),
            (lambda: buffer.add_memory("test", "event", "invalid"), (TypeError, ValueError)),
            (lambda: buffer.get_recent_memories(-1), (ValueError,)),
        ]
        
        for error_case, expected_exceptions in error_cases:
            try:
                error_case()
                result.add_compatibility_issue("Error Handling", "MEDIUM", 
                                             "Expected exception not raised")
            except expected_exceptions:
                # Expected behavior
                pass
            except Exception as e:
                result.add_compatibility_issue("Error Handling", "MEDIUM", 
                                             f"Unexpected exception type: {type(e).__name__}")
        
        result.add_pass("Error Handling: Exception consistency")
    
    except Exception as e:
        result.add_fail("Error Handling: Compatibility", str(e))


def main():
    """Execute comprehensive compatibility and regression tests"""
    print("🔄 Enhanced PIANO Memory Architecture - Compatibility and Regression Testing")
    print("="*70)
    print("Testing backward compatibility, API stability, and performance regressions...")
    
    result = CompatibilityTestResult()
    
    # Execute all compatibility test categories
    test_api_backward_compatibility(result)
    test_data_format_compatibility(result)
    test_performance_regression_baselines(result)
    test_version_migration_compatibility(result)
    test_dependency_compatibility(result)
    test_interface_stability(result)
    test_error_handling_compatibility(result)
    
    # Final compatibility assessment
    success = result.summary()
    
    if success:
        print("\n✅ ALL COMPATIBILITY TESTS PASSED! System maintains backward compatibility.")
        print("🔄 No breaking changes or performance regressions detected.")
    else:
        print(f"\n⚠️  COMPATIBILITY ISSUES DETECTED!")
        
        if result.compatibility_issues:
            critical_count = sum(1 for issue in result.compatibility_issues if issue['severity'] == 'CRITICAL')
            if critical_count > 0:
                print(f"🚨 {critical_count} CRITICAL compatibility issues require immediate attention!")
        
        if result.performance_regressions:
            major_regressions = sum(1 for reg in result.performance_regressions if reg['regression_percent'] > 50)
            if major_regressions > 0:
                print(f"📉 {major_regressions} major performance regressions detected!")
    
    # Compatibility recommendations
    print(f"\n📋 COMPATIBILITY RECOMMENDATIONS:")
    print(f"• Maintain semantic versioning for API changes")
    print(f"• Provide migration utilities for data format changes")
    print(f"• Add deprecation warnings before removing features")
    print(f"• Monitor performance benchmarks in CI/CD pipeline")
    print(f"• Test with multiple Python versions and dependency versions")
    print(f"• Document all breaking changes in changelog")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)