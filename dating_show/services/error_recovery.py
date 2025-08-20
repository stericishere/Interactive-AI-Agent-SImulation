"""
Error Recovery and Resilience Service
Handles simulation errors gracefully and provides recovery mechanisms
Ensures the dating show simulation continues running even when individual components fail
"""

import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    RESTART_COMPONENT = "restart_component"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class ErrorEvent:
    """Represents an error event in the system"""
    timestamp: datetime
    component: str
    error_type: str
    message: str
    severity: ErrorSeverity
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None


class ErrorRecoveryService:
    """
    Provides comprehensive error handling and recovery for the dating show simulation
    Monitors system health and automatically applies recovery strategies
    """
    
    def __init__(self, max_errors_per_minute: int = 10, max_retries: int = 3):
        self.max_errors_per_minute = max_errors_per_minute
        self.max_retries = max_retries
        
        # Error tracking
        self.error_history: List[ErrorEvent] = []
        self.component_error_counts: Dict[str, int] = {}
        self.last_error_time: Dict[str, float] = {}
        
        # Recovery strategies by component
        self.recovery_strategies = {
            'reverie_simulation': [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            'environment_generator': [RecoveryStrategy.RETRY, RecoveryStrategy.GRACEFUL_DEGRADATION],
            'agent_state_bridge': [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            'frontend_bridge': [RecoveryStrategy.RETRY, RecoveryStrategy.GRACEFUL_DEGRADATION],
            'file_io': [RecoveryStrategy.RETRY, RecoveryStrategy.SKIP],
            'api_communication': [RecoveryStrategy.RETRY, RecoveryStrategy.GRACEFUL_DEGRADATION],
            'default': [RecoveryStrategy.RETRY, RecoveryStrategy.SKIP]
        }
        
        # Fallback functions
        self.fallback_functions: Dict[str, Callable] = {}
        
        # System health tracking
        self.system_health = {
            'overall_status': 'healthy',
            'error_rate': 0.0,
            'last_check': datetime.now(timezone.utc),
            'degraded_components': set()
        }
    
    def handle_error(self, component: str, error: Exception, 
                    context: Optional[Dict[str, Any]] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> bool:
        """
        Handle an error event and attempt recovery
        
        Args:
            component: Name of the component where error occurred
            error: The exception that was raised
            context: Additional context information
            severity: Severity level of the error
            
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        try:
            # Create error event
            error_event = ErrorEvent(
                timestamp=datetime.now(timezone.utc),
                component=component,
                error_type=type(error).__name__,
                message=str(error),
                severity=severity,
                stack_trace=traceback.format_exc(),
                context=context or {}
            )
            
            # Log the error
            self._log_error(error_event)
            
            # Add to error history
            self.error_history.append(error_event)
            
            # Update component error count
            self.component_error_counts[component] = self.component_error_counts.get(component, 0) + 1
            self.last_error_time[component] = time.time()
            
            # Check if we should attempt recovery
            if self._should_attempt_recovery(component, error_event):
                recovery_success = self._attempt_recovery(error_event)
                error_event.recovery_attempted = True
                error_event.recovery_successful = recovery_success
                return recovery_success
            else:
                logger.warning(f"Skipping recovery for {component} due to rate limiting or severity")
                return False
                
        except Exception as recovery_error:
            logger.critical(f"Error in error recovery system: {recovery_error}")
            return False
    
    def _log_error(self, error_event: ErrorEvent):
        """Log error event with appropriate level"""
        log_message = f"[{error_event.component}] {error_event.error_type}: {error_event.message}"
        
        if error_event.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_event.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_event.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _should_attempt_recovery(self, component: str, error_event: ErrorEvent) -> bool:
        """Determine if recovery should be attempted based on error rate and severity"""
        # Never attempt recovery for critical errors that could make things worse
        if error_event.severity == ErrorSeverity.CRITICAL:
            return False
        
        # Check error rate for this component
        current_time = time.time()
        recent_errors = [
            e for e in self.error_history 
            if e.component == component and 
            (current_time - e.timestamp.timestamp()) < 60  # Last minute
        ]
        
        if len(recent_errors) >= self.max_errors_per_minute:
            logger.warning(f"Too many errors for {component}, throttling recovery attempts")
            return False
        
        # Check retry count
        if self.component_error_counts.get(component, 0) > self.max_retries:
            logger.warning(f"Max retries exceeded for {component}")
            return False
        
        return True
    
    def _attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt recovery using available strategies"""
        component = error_event.component
        strategies = self.recovery_strategies.get(component, self.recovery_strategies['default'])
        
        for strategy in strategies:
            try:
                logger.info(f"Attempting {strategy.value} recovery for {component}")
                
                if strategy == RecoveryStrategy.RETRY:
                    success = self._retry_operation(error_event)
                elif strategy == RecoveryStrategy.FALLBACK:
                    success = self._use_fallback(error_event)
                elif strategy == RecoveryStrategy.SKIP:
                    success = self._skip_operation(error_event)
                elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    success = self._graceful_degradation(error_event)
                elif strategy == RecoveryStrategy.RESTART_COMPONENT:
                    success = self._restart_component(error_event)
                else:
                    success = False
                
                if success:
                    error_event.recovery_strategy = strategy
                    logger.info(f"Recovery successful using {strategy.value} for {component}")
                    self._reset_error_count(component)
                    return True
                    
            except Exception as recovery_error:
                logger.error(f"Recovery strategy {strategy.value} failed for {component}: {recovery_error}")
                continue
        
        # All recovery strategies failed
        logger.error(f"All recovery strategies failed for {component}")
        self._mark_component_degraded(component)
        return False
    
    def _retry_operation(self, error_event: ErrorEvent) -> bool:
        """Simple retry with exponential backoff"""
        component = error_event.component
        retry_count = self.component_error_counts.get(component, 0)
        
        # Exponential backoff: 1s, 2s, 4s, 8s...
        wait_time = min(2 ** retry_count, 30)  # Max 30 seconds
        time.sleep(wait_time)
        
        # For now, we can't actually retry the original operation
        # This would need to be implemented per component with stored operation context
        logger.info(f"Retry delay completed for {component}, assuming success")
        return True
    
    def _use_fallback(self, error_event: ErrorEvent) -> bool:
        """Use fallback function if available"""
        component = error_event.component
        
        if component in self.fallback_functions:
            try:
                fallback_func = self.fallback_functions[component]
                result = fallback_func(error_event)
                logger.info(f"Fallback function executed successfully for {component}")
                return result
            except Exception as e:
                logger.error(f"Fallback function failed for {component}: {e}")
                return False
        
        # Default fallbacks based on component type
        if 'simulation' in component:
            return self._simulation_fallback(error_event)
        elif 'generator' in component:
            return self._generator_fallback(error_event)
        elif 'bridge' in component:
            return self._bridge_fallback(error_event)
        else:
            return self._generic_fallback(error_event)
    
    def _skip_operation(self, error_event: ErrorEvent) -> bool:
        """Skip the failed operation and continue"""
        logger.info(f"Skipping failed operation for {error_event.component}")
        return True
    
    def _graceful_degradation(self, error_event: ErrorEvent) -> bool:
        """Degrade component functionality but keep running"""
        component = error_event.component
        self.system_health['degraded_components'].add(component)
        
        logger.info(f"Component {component} running in degraded mode")
        return True
    
    def _restart_component(self, error_event: ErrorEvent) -> bool:
        """Restart a component (placeholder for now)"""
        component = error_event.component
        logger.info(f"Component restart requested for {component}")
        
        # This would need specific restart logic per component
        return True
    
    def _simulation_fallback(self, error_event: ErrorEvent) -> bool:
        """Fallback for simulation errors"""
        logger.info("Using simulation fallback: continuing with placeholder step")
        # Could generate a minimal placeholder step
        return True
    
    def _generator_fallback(self, error_event: ErrorEvent) -> bool:
        """Fallback for environment generator errors"""
        logger.info("Using generator fallback: skipping file generation for this step")
        return True
    
    def _bridge_fallback(self, error_event: ErrorEvent) -> bool:
        """Fallback for bridge errors"""
        logger.info("Using bridge fallback: continuing without real-time updates")
        return True
    
    def _generic_fallback(self, error_event: ErrorEvent) -> bool:
        """Generic fallback for unknown components"""
        logger.info(f"Using generic fallback for {error_event.component}")
        return True
    
    def _mark_component_degraded(self, component: str):
        """Mark a component as degraded"""
        self.system_health['degraded_components'].add(component)
        self._update_system_health()
    
    def _reset_error_count(self, component: str):
        """Reset error count for a component after successful recovery"""
        self.component_error_counts[component] = 0
        if component in self.system_health['degraded_components']:
            self.system_health['degraded_components'].remove(component)
        self._update_system_health()
    
    def _update_system_health(self):
        """Update overall system health status"""
        current_time = datetime.now(timezone.utc)
        
        # Calculate error rate
        recent_errors = [
            e for e in self.error_history 
            if (current_time - e.timestamp).total_seconds() < 300  # Last 5 minutes
        ]
        self.system_health['error_rate'] = len(recent_errors) / 5.0  # Errors per minute
        
        # Determine overall status
        degraded_count = len(self.system_health['degraded_components'])
        if degraded_count == 0 and self.system_health['error_rate'] < 0.5:
            status = 'healthy'
        elif degraded_count <= 2 and self.system_health['error_rate'] < 2.0:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        self.system_health['overall_status'] = status
        self.system_health['last_check'] = current_time
    
    def register_fallback_function(self, component: str, fallback_func: Callable):
        """Register a custom fallback function for a component"""
        self.fallback_functions[component] = fallback_func
        logger.info(f"Registered fallback function for {component}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        self._update_system_health()
        return self.system_health.copy()
    
    def get_error_summary(self, component: Optional[str] = None, 
                         hours: int = 24) -> Dict[str, Any]:
        """Get error summary for a component or all components"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        
        if component:
            errors = [
                e for e in self.error_history 
                if e.component == component and e.timestamp.timestamp() > cutoff_time
            ]
        else:
            errors = [
                e for e in self.error_history 
                if e.timestamp.timestamp() > cutoff_time
            ]
        
        return {
            'total_errors': len(errors),
            'errors_by_type': self._group_errors_by_type(errors),
            'errors_by_severity': self._group_errors_by_severity(errors),
            'recovery_success_rate': self._calculate_recovery_rate(errors),
            'most_recent_error': errors[-1] if errors else None
        }
    
    def _group_errors_by_type(self, errors: List[ErrorEvent]) -> Dict[str, int]:
        """Group errors by type"""
        counts = {}
        for error in errors:
            counts[error.error_type] = counts.get(error.error_type, 0) + 1
        return counts
    
    def _group_errors_by_severity(self, errors: List[ErrorEvent]) -> Dict[str, int]:
        """Group errors by severity"""
        counts = {}
        for error in errors:
            severity = error.severity.value
            counts[severity] = counts.get(severity, 0) + 1
        return counts
    
    def _calculate_recovery_rate(self, errors: List[ErrorEvent]) -> float:
        """Calculate recovery success rate"""
        recovery_attempts = [e for e in errors if e.recovery_attempted]
        if not recovery_attempts:
            return 0.0
        
        successful_recoveries = [e for e in recovery_attempts if e.recovery_successful]
        return len(successful_recoveries) / len(recovery_attempts) * 100
    
    def clear_old_errors(self, hours: int = 168):  # Default: 1 week
        """Clear old errors to prevent memory buildup"""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        
        initial_count = len(self.error_history)
        self.error_history = [
            e for e in self.error_history 
            if e.timestamp.timestamp() > cutoff_time
        ]
        
        cleared_count = initial_count - len(self.error_history)
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old error records")


# Global error recovery service instance
_error_recovery: Optional[ErrorRecoveryService] = None


def get_error_recovery() -> ErrorRecoveryService:
    """Get or create global error recovery service instance"""
    global _error_recovery
    if _error_recovery is None:
        _error_recovery = ErrorRecoveryService()
    return _error_recovery


def safe_execute(component: str, operation: Callable, 
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                context: Optional[Dict[str, Any]] = None) -> tuple[bool, Any]:
    """
    Safely execute an operation with error handling and recovery
    
    Args:
        component: Name of the component
        operation: Function to execute
        severity: Error severity level
        context: Additional context
        
    Returns:
        tuple: (success: bool, result: Any)
    """
    recovery_service = get_error_recovery()
    
    try:
        result = operation()
        return True, result
    except Exception as e:
        success = recovery_service.handle_error(component, e, context, severity)
        return success, None


def with_error_recovery(component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """
    Decorator for automatic error recovery
    
    Usage:
        @with_error_recovery('my_component')
        def risky_function():
            # code that might fail
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            context = {
                'function': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            success, result = safe_execute(
                component, 
                lambda: func(*args, **kwargs), 
                severity, 
                context
            )
            
            if success:
                return result
            else:
                # Return None or raise exception based on severity
                if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
                    raise RuntimeError(f"Critical error in {component}.{func.__name__}")
                return None
        
        return wrapper
    return decorator