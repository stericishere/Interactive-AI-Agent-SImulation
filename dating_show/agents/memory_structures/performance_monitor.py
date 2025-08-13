"""
File: performance_monitor.py
Description: Performance monitoring and optimization for Enhanced PIANO memory systems.
Ensures <50ms working memory and <100ms long-term memory response times for 50+ concurrent agents.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import statistics
import json
from collections import deque, defaultdict


class PerformanceThreshold(Enum):
    """Performance threshold definitions."""
    WORKING_MEMORY = 50.0  # ms
    TEMPORAL_MEMORY = 100.0  # ms
    EPISODIC_MEMORY = 100.0  # ms
    SEMANTIC_MEMORY = 100.0  # ms
    STORE_API = 200.0  # ms for cultural propagation
    DATABASE_OPERATION = 100.0  # ms
    DECISION_LATENCY = 100.0  # ms


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    operation_name: str
    duration_ms: float
    timestamp: datetime
    agent_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class PerformanceAlert:
    """Performance alert for threshold violations."""
    operation_name: str
    threshold_ms: float
    actual_ms: float
    agent_id: Optional[str]
    timestamp: datetime
    severity: str  # 'warning', 'error', 'critical'
    context: Dict[str, Any] = field(default_factory=dict)


class PerformanceCache:
    """High-performance LRU cache with TTL."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize performance cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time to live for cached items
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_order = deque()
        self._expiry_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        if key not in self._cache:
            return None
        
        # Check expiry
        if datetime.now() > self._expiry_times[key]:
            self.delete(key)
            return None
        
        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        
        return self._cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with expiry time."""
        # Remove if already exists
        if key in self._cache:
            self.delete(key)
        
        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = self._access_order.popleft()
            self.delete(oldest_key)
        
        # Add new item
        self._cache[key] = value
        self._access_order.append(key)
        self._expiry_times[key] = datetime.now() + timedelta(seconds=self.ttl_seconds)
    
    def delete(self, key: str) -> None:
        """Delete item from cache."""
        self._cache.pop(key, None)
        self._expiry_times.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)
    
    def clear_expired(self) -> int:
        """Clear expired items and return count."""
        now = datetime.now()
        expired_keys = [key for key, expiry in self._expiry_times.items() if now > expiry]
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)


class MemoryPerformanceMonitor:
    """
    Comprehensive performance monitoring for Enhanced PIANO memory systems.
    Tracks response times, detects bottlenecks, and provides optimization recommendations.
    """
    
    def __init__(self, enable_alerts: bool = True, enable_caching: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            enable_alerts: Enable performance threshold alerts
            enable_caching: Enable performance caching
        """
        self.enable_alerts = enable_alerts
        self.enable_caching = enable_caching
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
        self.alerts: List[PerformanceAlert] = []
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "error_count": 0,
            "success_rate": 1.0
        })
        
        # Performance caching
        self.cache = PerformanceCache() if enable_caching else None
        
        # Alert thresholds
        self.thresholds = {
            "working_memory": PerformanceThreshold.WORKING_MEMORY.value,
            "temporal_memory": PerformanceThreshold.TEMPORAL_MEMORY.value,
            "episodic_memory": PerformanceThreshold.EPISODIC_MEMORY.value,
            "semantic_memory": PerformanceThreshold.SEMANTIC_MEMORY.value,
            "store_api": PerformanceThreshold.STORE_API.value,
            "database_operation": PerformanceThreshold.DATABASE_OPERATION.value,
            "decision_latency": PerformanceThreshold.DECISION_LATENCY.value
        }
        
        # Keep only last N metrics to prevent memory bloat
        self.max_metrics = 10000
        
        self.logger = logging.getLogger(f"{__name__}.MemoryPerformanceMonitor")
        self.logger.info("Memory performance monitor initialized")
    
    def track_operation(self, operation_name: str, agent_id: str = None) -> 'PerformanceTracker':
        """
        Create a performance tracker for an operation.
        
        Args:
            operation_name: Name of the operation
            agent_id: Agent ID (optional)
        
        Returns:
            Performance tracker context manager
        """
        return PerformanceTracker(self, operation_name, agent_id)
    
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric."""
        self.metrics.append(metric)
        
        # Update operation statistics
        stats = self.operation_stats[metric.operation_name]
        stats["count"] += 1
        stats["total_time"] += metric.duration_ms
        stats["min_time"] = min(stats["min_time"], metric.duration_ms)
        stats["max_time"] = max(stats["max_time"], metric.duration_ms)
        
        if not metric.success:
            stats["error_count"] += 1
        
        stats["success_rate"] = 1.0 - (stats["error_count"] / stats["count"])
        
        # Check for threshold violations
        if self.enable_alerts and metric.success:
            self._check_performance_threshold(metric)
        
        # Cleanup old metrics
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics // 2:]
    
    def _check_performance_threshold(self, metric: PerformanceMetric) -> None:
        """Check if metric violates performance thresholds."""
        # Determine appropriate threshold
        threshold_key = None
        
        if "working_memory" in metric.operation_name.lower():
            threshold_key = "working_memory"
        elif "temporal_memory" in metric.operation_name.lower():
            threshold_key = "temporal_memory"
        elif "episodic" in metric.operation_name.lower():
            threshold_key = "episodic_memory"
        elif "semantic" in metric.operation_name.lower():
            threshold_key = "semantic_memory"
        elif "store" in metric.operation_name.lower() or "meme" in metric.operation_name.lower():
            threshold_key = "store_api"
        elif "database" in metric.operation_name.lower() or "postgres" in metric.operation_name.lower():
            threshold_key = "database_operation"
        elif "decision" in metric.operation_name.lower():
            threshold_key = "decision_latency"
        
        if threshold_key:
            threshold = self.thresholds[threshold_key]
            
            if metric.duration_ms > threshold:
                severity = "warning"
                if metric.duration_ms > threshold * 2:
                    severity = "error"
                if metric.duration_ms > threshold * 5:
                    severity = "critical"
                
                alert = PerformanceAlert(
                    operation_name=metric.operation_name,
                    threshold_ms=threshold,
                    actual_ms=metric.duration_ms,
                    agent_id=metric.agent_id,
                    timestamp=metric.timestamp,
                    severity=severity,
                    context=metric.context
                )
                
                self.alerts.append(alert)
                self.logger.warning(f"Performance alert ({severity}): {metric.operation_name} "
                                  f"took {metric.duration_ms:.2f}ms (threshold: {threshold}ms)")
    
    def get_performance_summary(self, last_minutes: int = 60) -> Dict[str, Any]:
        """
        Get performance summary for the last N minutes.
        
        Args:
            last_minutes: Minutes to look back
        
        Returns:
            Performance summary
        """
        cutoff_time = datetime.now() - timedelta(minutes=last_minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics in the specified time period"}
        
        # Overall statistics
        total_operations = len(recent_metrics)
        successful_operations = sum(1 for m in recent_metrics if m.success)
        success_rate = successful_operations / total_operations if total_operations > 0 else 0.0
        
        # Performance statistics
        durations = [m.duration_ms for m in recent_metrics if m.success]
        if durations:
            avg_duration = statistics.mean(durations)
            median_duration = statistics.median(durations)
            p95_duration = durations[int(len(durations) * 0.95)] if len(durations) > 20 else max(durations)
            p99_duration = durations[int(len(durations) * 0.99)] if len(durations) > 100 else max(durations)
        else:
            avg_duration = median_duration = p95_duration = p99_duration = 0.0
        
        # Operation breakdown
        operation_breakdown = {}
        operations_by_type = defaultdict(list)
        
        for metric in recent_metrics:
            operations_by_type[metric.operation_name].append(metric)
        
        for operation, metrics in operations_by_type.items():
            successful_metrics = [m for m in metrics if m.success]
            if successful_metrics:
                durations = [m.duration_ms for m in successful_metrics]
                operation_breakdown[operation] = {
                    "count": len(metrics),
                    "success_count": len(successful_metrics),
                    "success_rate": len(successful_metrics) / len(metrics),
                    "avg_duration_ms": statistics.mean(durations),
                    "median_duration_ms": statistics.median(durations),
                    "max_duration_ms": max(durations),
                    "min_duration_ms": min(durations)
                }
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if a.timestamp >= cutoff_time]
        alert_summary = {
            "total_alerts": len(recent_alerts),
            "critical_alerts": sum(1 for a in recent_alerts if a.severity == "critical"),
            "error_alerts": sum(1 for a in recent_alerts if a.severity == "error"),
            "warning_alerts": sum(1 for a in recent_alerts if a.severity == "warning")
        }
        
        # Threshold compliance
        threshold_compliance = {}
        for operation, breakdown in operation_breakdown.items():
            threshold_key = None
            
            if "working_memory" in operation.lower():
                threshold_key = "working_memory"
            elif "temporal_memory" in operation.lower():
                threshold_key = "temporal_memory"
            elif "episodic" in operation.lower():
                threshold_key = "episodic_memory"
            elif "semantic" in operation.lower():
                threshold_key = "semantic_memory"
            elif "store" in operation.lower():
                threshold_key = "store_api"
            elif "database" in operation.lower():
                threshold_key = "database_operation"
            elif "decision" in operation.lower():
                threshold_key = "decision_latency"
            
            if threshold_key:
                threshold = self.thresholds[threshold_key]
                compliant_operations = sum(1 for m in operations_by_type[operation] 
                                        if m.success and m.duration_ms <= threshold)
                compliance_rate = compliant_operations / breakdown["success_count"] if breakdown["success_count"] > 0 else 0.0
                
                threshold_compliance[operation] = {
                    "threshold_ms": threshold,
                    "compliance_rate": compliance_rate,
                    "avg_duration_ms": breakdown["avg_duration_ms"],
                    "exceeds_threshold": breakdown["avg_duration_ms"] > threshold
                }
        
        return {
            "time_period_minutes": last_minutes,
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "success_rate": success_rate,
            "performance": {
                "avg_duration_ms": avg_duration,
                "median_duration_ms": median_duration,
                "p95_duration_ms": p95_duration,
                "p99_duration_ms": p99_duration
            },
            "operation_breakdown": operation_breakdown,
            "alerts": alert_summary,
            "threshold_compliance": threshold_compliance,
            "cache_stats": self.get_cache_stats() if self.cache else None
        }
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on performance data."""
        recommendations = []
        summary = self.get_performance_summary(60)
        
        # Check overall performance
        if summary["performance"]["avg_duration_ms"] > 100:
            recommendations.append({
                "priority": "high",
                "category": "overall_performance",
                "issue": f"Average operation time ({summary['performance']['avg_duration_ms']:.1f}ms) exceeds 100ms target",
                "recommendation": "Consider enabling caching, optimizing database queries, or scaling infrastructure",
                "impact": "high"
            })
        
        # Check specific operation performance
        if "threshold_compliance" in summary:
            for operation, compliance in summary["threshold_compliance"].items():
                if compliance["exceeds_threshold"]:
                    severity = "critical" if compliance["avg_duration_ms"] > compliance["threshold_ms"] * 2 else "high"
                    
                    recommendations.append({
                        "priority": severity,
                        "category": "operation_performance",
                        "operation": operation,
                        "issue": f"{operation} averaging {compliance['avg_duration_ms']:.1f}ms "
                               f"(threshold: {compliance['threshold_ms']}ms)",
                        "recommendation": self._get_operation_optimization_advice(operation),
                        "impact": "high"
                    })
        
        # Check success rates
        if summary["success_rate"] < 0.95:
            recommendations.append({
                "priority": "critical",
                "category": "reliability",
                "issue": f"Success rate ({summary['success_rate']:.1%}) below 95% target",
                "recommendation": "Investigate error patterns and implement better error handling",
                "impact": "critical"
            })
        
        # Check alert patterns
        if summary["alerts"]["critical_alerts"] > 0:
            recommendations.append({
                "priority": "critical",
                "category": "alerts",
                "issue": f"{summary['alerts']['critical_alerts']} critical performance alerts in last hour",
                "recommendation": "Immediate investigation required for critical performance issues",
                "impact": "critical"
            })
        
        # Cache recommendations
        if self.cache and "cache_stats" in summary and summary["cache_stats"]:
            cache_stats = summary["cache_stats"]
            if cache_stats["hit_rate"] < 0.8:
                recommendations.append({
                    "priority": "medium",
                    "category": "caching",
                    "issue": f"Cache hit rate ({cache_stats['hit_rate']:.1%}) below optimal 80%",
                    "recommendation": "Consider increasing cache size or TTL, or review caching strategy",
                    "impact": "medium"
                })
        
        return sorted(recommendations, key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["priority"], 3))
    
    def _get_operation_optimization_advice(self, operation: str) -> str:
        """Get specific optimization advice for an operation."""
        if "working_memory" in operation.lower():
            return "Optimize circular buffer operations, consider in-memory caching, reduce serialization overhead"
        elif "temporal_memory" in operation.lower():
            return "Add database indexes for temporal queries, implement query result caching"
        elif "episodic" in operation.lower():
            return "Optimize episode retrieval queries, implement lazy loading for large episodes"
        elif "semantic" in operation.lower():
            return "Add concept search indexes, implement concept activation caching"
        elif "store" in operation.lower():
            return "Optimize Store API batch operations, implement local caching for frequently accessed data"
        elif "database" in operation.lower():
            return "Optimize database queries, add appropriate indexes, consider connection pooling"
        else:
            return "Review operation logic, add caching where appropriate, optimize data structures"
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache performance statistics."""
        if not self.cache:
            return None
        
        # This is a simplified version - in practice, you'd track hits/misses
        expired_count = self.cache.clear_expired()
        
        return {
            "size": len(self.cache._cache),
            "max_size": self.cache.max_size,
            "utilization": len(self.cache._cache) / self.cache.max_size,
            "ttl_seconds": self.cache.ttl_seconds,
            "expired_cleaned": expired_count,
            "hit_rate": 0.85  # Placeholder - would be tracked in practice
        }
    
    def export_metrics(self, format: str = "json", last_minutes: int = 60) -> str:
        """
        Export performance metrics in specified format.
        
        Args:
            format: Export format ('json', 'csv')
            last_minutes: Minutes to look back
        
        Returns:
            Exported data as string
        """
        cutoff_time = datetime.now() - timedelta(minutes=last_minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
        
        if format.lower() == "json":
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "time_period_minutes": last_minutes,
                "metrics": [
                    {
                        "operation_name": m.operation_name,
                        "duration_ms": m.duration_ms,
                        "timestamp": m.timestamp.isoformat(),
                        "agent_id": m.agent_id,
                        "success": m.success,
                        "error_message": m.error_message,
                        "context": m.context
                    }
                    for m in recent_metrics
                ],
                "summary": self.get_performance_summary(last_minutes)
            }
            return json.dumps(export_data, indent=2)
        
        elif format.lower() == "csv":
            # Simple CSV export
            lines = ["operation_name,duration_ms,timestamp,agent_id,success,error_message"]
            for m in recent_metrics:
                lines.append(f"{m.operation_name},{m.duration_ms},{m.timestamp.isoformat()},"
                           f"{m.agent_id or ''},{m.success},{m.error_message or ''}")
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


class PerformanceTracker:
    """Context manager for tracking operation performance."""
    
    def __init__(self, monitor: MemoryPerformanceMonitor, operation_name: str, agent_id: str = None):
        """
        Initialize performance tracker.
        
        Args:
            monitor: Performance monitor instance
            operation_name: Name of the operation to track
            agent_id: Agent ID (optional)
        """
        self.monitor = monitor
        self.operation_name = operation_name
        self.agent_id = agent_id
        self.start_time = None
        self.context = {}
        self.success = True
        self.error_message = None
    
    def __enter__(self) -> 'PerformanceTracker':
        """Start performance tracking."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """End performance tracking and record metric."""
        if self.start_time is None:
            return
        
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val) if exc_val else "Unknown error"
        
        metric = PerformanceMetric(
            operation_name=self.operation_name,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            context=self.context,
            success=self.success,
            error_message=self.error_message
        )
        
        self.monitor.record_metric(metric)
    
    def add_context(self, **kwargs) -> None:
        """Add context information to the performance metric."""
        self.context.update(kwargs)
    
    def mark_error(self, error_message: str) -> None:
        """Mark operation as failed with error message."""
        self.success = False
        self.error_message = error_message


# Helper functions

def create_performance_monitor(enable_alerts: bool = True, enable_caching: bool = True) -> MemoryPerformanceMonitor:
    """Create a performance monitor instance."""
    return MemoryPerformanceMonitor(enable_alerts, enable_caching)


# Global performance monitor instance (singleton pattern)
_global_monitor: Optional[MemoryPerformanceMonitor] = None


def get_global_monitor() -> MemoryPerformanceMonitor:
    """Get or create global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = create_performance_monitor()
    return _global_monitor


# Decorator for easy performance tracking
def track_performance(operation_name: str = None, monitor: MemoryPerformanceMonitor = None):
    """
    Decorator to track function performance.
    
    Args:
        operation_name: Custom operation name (defaults to function name)
        monitor: Performance monitor instance (defaults to global monitor)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal operation_name, monitor
            if operation_name is None:
                operation_name = f"{func.__module__}.{func.__name__}"
            if monitor is None:
                monitor = get_global_monitor()
            
            with monitor.track_operation(operation_name) as tracker:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    tracker.mark_error(str(e))
                    raise
        
        async def async_wrapper(*args, **kwargs):
            nonlocal operation_name, monitor
            if operation_name is None:
                operation_name = f"{func.__module__}.{func.__name__}"
            if monitor is None:
                monitor = get_global_monitor()
            
            with monitor.track_operation(operation_name) as tracker:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    tracker.mark_error(str(e))
                    raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    async def test_performance_monitor():
        """Test the performance monitoring system."""
        
        # Create performance monitor
        monitor = create_performance_monitor()
        
        # Test operation tracking
        with monitor.track_operation("test_working_memory", "test_agent") as tracker:
            await asyncio.sleep(0.03)  # Simulate 30ms operation
            tracker.add_context(memory_size=100, operation_type="retrieval")
        
        # Test threshold violation
        with monitor.track_operation("test_working_memory_slow", "test_agent") as tracker:
            await asyncio.sleep(0.08)  # Simulate 80ms operation (exceeds 50ms threshold)
            tracker.add_context(memory_size=500, operation_type="retrieval")
        
        # Test error tracking
        with monitor.track_operation("test_error", "test_agent") as tracker:
            tracker.mark_error("Simulated error for testing")
        
        # Get performance summary
        summary = monitor.get_performance_summary(1)
        print("Performance Summary:")
        print(f"  Total operations: {summary['total_operations']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        print(f"  Average duration: {summary['performance']['avg_duration_ms']:.1f}ms")
        print(f"  Alerts: {summary['alerts']['total_alerts']}")
        
        # Get recommendations
        recommendations = monitor.get_optimization_recommendations()
        print(f"\nOptimization Recommendations ({len(recommendations)} found):")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. [{rec['priority']}] {rec['issue']}")
            print(f"     Recommendation: {rec['recommendation']}")
        
        print("\nPerformance monitoring test completed successfully!")
    
    # Run test
    asyncio.run(test_performance_monitor())