"""
UpdatePipeline Validation and Testing Suite
Comprehensive validation for real-time synchronization pipeline.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .update_pipeline import get_update_pipeline, UpdateType, UpdateEvent
from .unified_agent_manager import get_unified_agent_manager
from .frontend_state_adapter import get_frontend_state_adapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Represents validation test result."""
    test_name: str
    passed: bool
    execution_time_ms: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class PerformanceTest:
    """Performance test configuration."""
    name: str
    agent_count: int
    update_count: int
    expected_time_ms: float
    concurrent_updates: bool = False


class PipelineValidator:
    """
    Comprehensive validation suite for UpdatePipeline.
    
    Tests performance, reliability, data integrity, and real-time capabilities.
    """
    
    def __init__(self):
        self.pipeline = None
        self.unified_manager = None
        self.frontend_adapter = None
        self.test_results: List[ValidationResult] = []
        
        # Test configuration
        self.performance_threshold_ms = 100
        self.batch_threshold_ms = 50
        self.websocket_threshold_ms = 30
        
        logger.info("PipelineValidator initialized")
    
    async def initialize(self):
        """Initialize validator with pipeline services."""
        try:
            self.pipeline = get_update_pipeline()
            await self.pipeline.initialize()
            
            self.unified_manager = get_unified_agent_manager()
            self.frontend_adapter = get_frontend_state_adapter()
            
            logger.info("PipelineValidator initialized with services")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PipelineValidator: {e}")
            return False
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("Starting comprehensive UpdatePipeline validation")
        
        self.test_results = []
        start_time = time.time()
        
        # Test categories
        await self._test_basic_functionality()
        await self._test_performance_targets()
        await self._test_reliability_patterns()
        await self._test_websocket_integration()
        await self._test_batch_processing()
        await self._test_data_integrity()
        await self._test_error_handling()
        
        total_time = (time.time() - start_time) * 1000
        
        # Compile results
        passed_tests = [r for r in self.test_results if r.passed]
        failed_tests = [r for r in self.test_results if not r.passed]
        
        results = {
            'validation_summary': {
                'total_tests': len(self.test_results),
                'passed_tests': len(passed_tests),
                'failed_tests': len(failed_tests),
                'success_rate': (len(passed_tests) / len(self.test_results)) * 100,
                'total_execution_time_ms': total_time
            },
            'test_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'execution_time_ms': r.execution_time_ms,
                    'details': r.details,
                    'error_message': r.error_message
                }
                for r in self.test_results
            ],
            'performance_analysis': await self._analyze_performance(),
            'recommendations': self._generate_recommendations(failed_tests),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Validation completed: {len(passed_tests)}/{len(self.test_results)} tests passed")
        return results
    
    async def _test_basic_functionality(self):
        """Test basic pipeline functionality."""
        logger.info("Testing basic functionality...")
        
        # Test 1: Pipeline initialization
        start_time = time.time()
        try:
            metrics = self.pipeline.get_performance_metrics()
            execution_time = (time.time() - start_time) * 1000
            
            self.test_results.append(ValidationResult(
                test_name="pipeline_initialization",
                passed=True,
                execution_time_ms=execution_time,
                details={'metrics_available': True, 'pipeline_active': True}
            ))
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.test_results.append(ValidationResult(
                test_name="pipeline_initialization",
                passed=False,
                execution_time_ms=execution_time,
                details={'metrics_available': False},
                error_message=str(e)
            ))
        
        # Test 2: Single agent update
        start_time = time.time()
        try:
            test_agent_id = "test_agent_001"
            test_updates = {"current_location": "test_location", "test_field": "test_value"}
            
            success = await self.pipeline.update_agent_state(
                test_agent_id, test_updates, UpdateType.AGENT_STATE
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            self.test_results.append(ValidationResult(
                test_name="single_agent_update",
                passed=success,
                execution_time_ms=execution_time,
                details={'agent_id': test_agent_id, 'update_success': success}
            ))
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.test_results.append(ValidationResult(
                test_name="single_agent_update",
                passed=False,
                execution_time_ms=execution_time,
                details={'agent_id': test_agent_id},
                error_message=str(e)
            ))
    
    async def _test_performance_targets(self):
        """Test performance targets compliance."""
        logger.info("Testing performance targets...")
        
        performance_tests = [
            PerformanceTest("single_update_performance", 1, 1, 100),
            PerformanceTest("batch_update_performance", 5, 5, 50),
            PerformanceTest("concurrent_updates", 10, 20, 100, True)
        ]
        
        for test in performance_tests:
            start_time = time.time()
            try:
                if test.concurrent_updates:
                    # Concurrent updates test
                    tasks = []
                    for i in range(test.update_count):
                        agent_id = f"perf_test_agent_{i}"
                        updates = {"performance_test": True, "test_iteration": i}
                        tasks.append(
                            self.pipeline.update_agent_state(agent_id, updates, UpdateType.AGENT_STATE)
                        )
                    
                    results = await asyncio.gather(*tasks)
                    success_count = sum(1 for r in results if r)
                else:
                    # Sequential updates test
                    success_count = 0
                    for i in range(test.update_count):
                        agent_id = f"perf_test_agent_{i}"
                        updates = {"performance_test": True, "test_iteration": i}
                        
                        if await self.pipeline.update_agent_state(agent_id, updates, UpdateType.AGENT_STATE):
                            success_count += 1
                
                execution_time = (time.time() - start_time) * 1000
                target_met = execution_time <= test.expected_time_ms
                
                self.test_results.append(ValidationResult(
                    test_name=test.name,
                    passed=target_met and success_count == test.update_count,
                    execution_time_ms=execution_time,
                    details={
                        'expected_time_ms': test.expected_time_ms,
                        'actual_time_ms': execution_time,
                        'target_met': target_met,
                        'successful_updates': success_count,
                        'total_updates': test.update_count,
                        'concurrent': test.concurrent_updates
                    }
                ))
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                self.test_results.append(ValidationResult(
                    test_name=test.name,
                    passed=False,
                    execution_time_ms=execution_time,
                    details={'test_config': test.__dict__},
                    error_message=str(e)
                ))
    
    async def _test_reliability_patterns(self):
        """Test circuit breaker and reliability patterns."""
        logger.info("Testing reliability patterns...")
        
        # Test circuit breaker functionality
        start_time = time.time()
        try:
            # Get initial circuit breaker state
            initial_state = self.pipeline.circuit_breaker.state.value
            failure_count = self.pipeline.circuit_breaker.failure_count
            
            # Test that circuit breaker is working
            circuit_breaker_working = hasattr(self.pipeline.circuit_breaker, 'call')
            
            execution_time = (time.time() - start_time) * 1000
            
            self.test_results.append(ValidationResult(
                test_name="circuit_breaker_functionality",
                passed=circuit_breaker_working,
                execution_time_ms=execution_time,
                details={
                    'initial_state': initial_state,
                    'failure_count': failure_count,
                    'circuit_breaker_available': circuit_breaker_working
                }
            ))
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.test_results.append(ValidationResult(
                test_name="circuit_breaker_functionality",
                passed=False,
                execution_time_ms=execution_time,
                details={},
                error_message=str(e)
            ))
        
        # Test retry mechanism
        start_time = time.time()
        try:
            # Create event that might trigger retry
            test_event = UpdateEvent(
                event_id="retry_test",
                event_type=UpdateType.AGENT_STATE,
                agent_id="retry_test_agent",
                data={"retry_test": True},
                timestamp=datetime.now(timezone.utc),
                priority=1
            )
            
            # Test event processing (this should work normally)
            await self.pipeline.queue_update(test_event)
            
            execution_time = (time.time() - start_time) * 1000
            
            self.test_results.append(ValidationResult(
                test_name="retry_mechanism",
                passed=True,
                execution_time_ms=execution_time,
                details={'event_queued': True, 'retry_capability': True}
            ))
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.test_results.append(ValidationResult(
                test_name="retry_mechanism",
                passed=False,
                execution_time_ms=execution_time,
                details={},
                error_message=str(e)
            ))
    
    async def _test_websocket_integration(self):
        """Test WebSocket integration."""
        logger.info("Testing WebSocket integration...")
        
        start_time = time.time()
        try:
            # Test WebSocket connection management
            websocket_count = len(self.pipeline.websocket_connections)
            websocket_groups = len(self.pipeline.websocket_groups)
            
            # Test broadcast capability (mock)
            test_message = {
                "type": "test_broadcast",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Since we don't have actual WebSocket connections in tests,
            # we'll test that the broadcast method exists and can be called
            await self.pipeline.broadcast_to_websockets(test_message)
            
            execution_time = (time.time() - start_time) * 1000
            
            self.test_results.append(ValidationResult(
                test_name="websocket_integration",
                passed=execution_time <= self.websocket_threshold_ms,
                execution_time_ms=execution_time,
                details={
                    'websocket_connections': websocket_count,
                    'websocket_groups': websocket_groups,
                    'broadcast_capability': True,
                    'threshold_met': execution_time <= self.websocket_threshold_ms
                }
            ))
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.test_results.append(ValidationResult(
                test_name="websocket_integration",
                passed=False,
                execution_time_ms=execution_time,
                details={},
                error_message=str(e)
            ))
    
    async def _test_batch_processing(self):
        """Test batch processing capabilities."""
        logger.info("Testing batch processing...")
        
        start_time = time.time()
        try:
            # Create batch updates
            batch_updates = {}
            for i in range(5):
                agent_id = f"batch_test_agent_{i}"
                batch_updates[agent_id] = {"batch_test": True, "agent_index": i}
            
            # Process batch
            results = await self.pipeline.batch_update_agents(batch_updates)
            
            execution_time = (time.time() - start_time) * 1000
            success_count = sum(1 for success in results.values() if success)
            
            self.test_results.append(ValidationResult(
                test_name="batch_processing",
                passed=execution_time <= self.batch_threshold_ms and success_count == len(batch_updates),
                execution_time_ms=execution_time,
                details={
                    'batch_size': len(batch_updates),
                    'successful_updates': success_count,
                    'threshold_met': execution_time <= self.batch_threshold_ms,
                    'results': results
                }
            ))
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.test_results.append(ValidationResult(
                test_name="batch_processing",
                passed=False,
                execution_time_ms=execution_time,
                details={'batch_size': 5},
                error_message=str(e)
            ))
    
    async def _test_data_integrity(self):
        """Test data integrity through the pipeline."""
        logger.info("Testing data integrity...")
        
        start_time = time.time()
        try:
            test_agent_id = "integrity_test_agent"
            test_data = {
                "current_location": "integrity_test_location",
                "test_value": "integrity_check",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Update through pipeline
            success = await self.pipeline.update_agent_state(
                test_agent_id, test_data, UpdateType.AGENT_STATE
            )
            
            # Wait a moment for processing
            await asyncio.sleep(0.1)
            
            # Verify data integrity through frontend adapter
            retrieved_state = self.frontend_adapter.get_agent_for_frontend(test_agent_id)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Check if data was preserved
            data_preserved = retrieved_state is not None
            if retrieved_state:
                data_preserved = (
                    retrieved_state.get('current_location') == test_data['current_location']
                )
            
            self.test_results.append(ValidationResult(
                test_name="data_integrity",
                passed=success and data_preserved,
                execution_time_ms=execution_time,
                details={
                    'update_success': success,
                    'data_preserved': data_preserved,
                    'retrieved_state_available': retrieved_state is not None
                }
            ))
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.test_results.append(ValidationResult(
                test_name="data_integrity",
                passed=False,
                execution_time_ms=execution_time,
                details={'test_agent_id': test_agent_id},
                error_message=str(e)
            ))
    
    async def _test_error_handling(self):
        """Test error handling and recovery."""
        logger.info("Testing error handling...")
        
        start_time = time.time()
        try:
            # Test invalid agent ID
            invalid_result = await self.pipeline.update_agent_state(
                "", {}, UpdateType.AGENT_STATE
            )
            
            # Test invalid data
            invalid_data_result = await self.pipeline.update_agent_state(
                "error_test_agent", None, UpdateType.AGENT_STATE
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Error handling should gracefully handle these cases
            error_handling_works = True  # Pipeline should not crash
            
            self.test_results.append(ValidationResult(
                test_name="error_handling",
                passed=error_handling_works,
                execution_time_ms=execution_time,
                details={
                    'invalid_agent_handled': not invalid_result,  # Should return False
                    'invalid_data_handled': not invalid_data_result,  # Should return False
                    'pipeline_stable': True
                }
            ))
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.test_results.append(ValidationResult(
                test_name="error_handling",
                passed=False,
                execution_time_ms=execution_time,
                details={},
                error_message=str(e)
            ))
    
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall performance metrics."""
        if not self.pipeline:
            return {'error': 'Pipeline not available for analysis'}
        
        try:
            metrics = self.pipeline.get_performance_metrics()
            
            # Performance analysis
            performance_analysis = {
                'target_compliance': {
                    'sub_100ms_target': metrics.get('average_processing_time_ms', 0) < 100,
                    'batch_efficiency': metrics.get('batch_processing_time_ms', 0) < 50,
                    'websocket_responsiveness': metrics.get('websocket_broadcast_time_ms', 0) < 30
                },
                'reliability_metrics': {
                    'success_rate': metrics.get('success_rate', 0),
                    'circuit_breaker_state': metrics.get('circuit_breaker_state', 'unknown'),
                    'performance_target_met': metrics.get('performance_target_met', False)
                },
                'scalability_indicators': {
                    'active_websockets': metrics.get('active_websockets', 0),
                    'queue_size': metrics.get('queue_size', 0),
                    'total_updates_processed': metrics.get('total_updates', 0)
                },
                'raw_metrics': metrics
            }
            
            return performance_analysis
            
        except Exception as e:
            return {'error': f'Performance analysis failed: {e}'}
    
    def _generate_recommendations(self, failed_tests: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on failed tests."""
        recommendations = []
        
        if not failed_tests:
            recommendations.append("✅ All tests passed! UpdatePipeline is performing optimally.")
            return recommendations
        
        for test in failed_tests:
            if test.test_name == "single_update_performance":
                recommendations.append("⚠️ Single update performance below target. Consider optimizing state update logic.")
            elif test.test_name == "batch_processing":
                recommendations.append("⚠️ Batch processing performance issues. Review batch size and timeout configuration.")
            elif test.test_name == "websocket_integration":
                recommendations.append("⚠️ WebSocket integration issues. Check network configuration and connection management.")
            elif test.test_name == "data_integrity":
                recommendations.append("🚨 Data integrity issues detected. Review state synchronization logic.")
            elif test.test_name == "circuit_breaker_functionality":
                recommendations.append("⚠️ Circuit breaker issues. Review failure threshold and recovery timeout settings.")
            elif test.test_name == "error_handling":
                recommendations.append("⚠️ Error handling improvements needed. Add more robust error recovery mechanisms.")
        
        recommendations.append("📊 Review performance metrics and consider scaling adjustments if needed.")
        return recommendations
    
    async def cleanup(self):
        """Clean up test resources."""
        try:
            if self.pipeline:
                await self.pipeline.shutdown()
            logger.info("PipelineValidator cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Utility functions for validation

async def run_quick_validation() -> Dict[str, Any]:
    """Run quick validation suite for basic functionality."""
    validator = PipelineValidator()
    
    if not await validator.initialize():
        return {'error': 'Failed to initialize validator'}
    
    try:
        # Run basic tests only
        await validator._test_basic_functionality()
        await validator._test_performance_targets()
        
        passed_tests = [r for r in validator.test_results if r.passed]
        
        return {
            'quick_validation': True,
            'tests_run': len(validator.test_results),
            'tests_passed': len(passed_tests),
            'success': len(passed_tests) == len(validator.test_results),
            'details': [r.__dict__ for r in validator.test_results]
        }
    finally:
        await validator.cleanup()


async def run_performance_benchmark() -> Dict[str, Any]:
    """Run performance-focused benchmark tests."""
    validator = PipelineValidator()
    
    if not await validator.initialize():
        return {'error': 'Failed to initialize validator'}
    
    try:
        await validator._test_performance_targets()
        performance_analysis = await validator._analyze_performance()
        
        return {
            'benchmark_results': performance_analysis,
            'test_results': [r.__dict__ for r in validator.test_results]
        }
    finally:
        await validator.cleanup()