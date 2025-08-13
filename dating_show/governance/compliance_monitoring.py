"""
File: compliance_monitoring.py
Description: Rule Compliance Monitoring System for real-time rule validation.
Monitors agent actions against established rules, tracks compliance scores,
detects violations, and logs infractions for the democratic governance system.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
import re

# Import governance and agent components
from ..agents.memory_structures.store_integration import (
    MemoryStoreIntegration, StoreNamespace, GovernanceRule
)
from ..agents.enhanced_agent_state import GovernanceData


class RuleCategory(Enum):
    """Categories of rules for compliance monitoring."""
    CONSTITUTIONAL = "constitutional"  # Fundamental governance rules
    BEHAVIORAL = "behavioral"  # Agent behavior guidelines
    SOCIAL = "social"  # Social interaction rules
    ECONOMIC = "economic"  # Resource and economic rules
    PROCEDURAL = "procedural"  # Process and procedure rules
    EMERGENCY = "emergency"  # Emergency protocols


class ViolationType(Enum):
    """Types of rule violations."""
    MINOR = "minor"  # Minor infractions
    MODERATE = "moderate"  # Moderate violations
    SEVERE = "severe"  # Severe violations
    CRITICAL = "critical"  # Critical system violations


class ComplianceStatus(Enum):
    """Compliance status for agents."""
    COMPLIANT = "compliant"  # Good compliance
    WARNING = "warning"  # Some violations, needs attention
    PROBATION = "probation"  # Multiple violations, on probation
    SUSPENDED = "suspended"  # Suspended due to violations
    BANNED = "banned"  # Permanently banned


@dataclass
class Rule:
    """Structure for governance rules."""
    rule_id: str
    rule_name: str
    rule_text: str
    category: RuleCategory
    priority: int  # 1-10 scale
    enforcement_level: float  # How strictly enforced (0.0 to 1.0)
    scope: str  # Who the rule applies to
    conditions: Dict[str, Any]  # Conditions for rule application
    violation_penalties: Dict[ViolationType, float]  # Penalty weights
    is_active: bool
    created_at: datetime
    last_updated: datetime
    version: int
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


@dataclass
class Violation:
    """Structure for rule violations."""
    violation_id: str
    agent_id: str
    rule_id: str
    violation_type: ViolationType
    description: str
    action_that_violated: Dict[str, Any]  # The action that caused violation
    severity_score: float  # 0.0 to 1.0
    detected_at: datetime
    context: Dict[str, Any]  # Contextual information
    witnesses: Set[str]  # Other agents who witnessed
    evidence: List[Dict[str, Any]]  # Evidence of violation
    is_disputed: bool
    resolution_status: str  # 'pending', 'acknowledged', 'resolved', 'appealed'
    penalty_applied: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.witnesses:
            self.witnesses = set()
        if not self.evidence:
            self.evidence = []
        if not self.metadata:
            self.metadata = {}


@dataclass
class ComplianceRecord:
    """Compliance tracking record for an agent."""
    agent_id: str
    overall_compliance_score: float  # 0.0 to 1.0
    category_scores: Dict[RuleCategory, float]  # Per-category compliance
    violation_count: Dict[ViolationType, int]  # Count by violation type
    recent_violations: List[str]  # Recent violation IDs
    compliance_trend: List[float]  # Historical scores
    status: ComplianceStatus
    status_since: datetime
    probation_until: Optional[datetime]
    strikes: int  # Accumulated strikes
    commendations: int  # Positive compliance actions
    last_updated: datetime
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}


class ComplianceMonitor:
    """
    Real-time Rule Compliance Monitoring System.
    Monitors agent actions, validates against rules, and tracks compliance.
    """

    def __init__(self, store_integration: MemoryStoreIntegration,
                 postgres_persistence=None, community_size: int = 50):
        """
        Initialize the Compliance Monitor.
        
        Args:
            store_integration: Store API integration
            postgres_persistence: PostgreSQL persistence layer
            community_size: Expected community size
        """
        self.store_integration = store_integration
        self.postgres_persistence = postgres_persistence
        self.community_size = community_size
        self.logger = logging.getLogger(f"{__name__}.ComplianceMonitor")
        
        # Rule storage and compliance tracking
        self.active_rules = {}  # rule_id -> Rule
        self.compliance_records = {}  # agent_id -> ComplianceRecord
        self.recent_violations = {}  # violation_id -> Violation
        self.rule_validators = {}  # rule_id -> validation_function
        
        # Configuration
        self.config = {
            "compliance_score_decay": 0.01,  # Daily decay of good behavior
            "violation_memory_days": 90,  # How long violations affect score
            "strikes_threshold": {
                ViolationType.MINOR: 10,
                ViolationType.MODERATE: 5,
                ViolationType.SEVERE: 3,
                ViolationType.CRITICAL: 1
            },
            "probation_period_days": 30,
            "suspension_period_days": 7,
            "monitoring_frequency_seconds": 10,
            "batch_processing_size": 100
        }
        
        # Metrics
        self.metrics = {
            "total_rules": 0,
            "active_rules": 0,
            "violations_detected": 0,
            "average_compliance": 0.0,
            "monitoring_efficiency": 0.0
        }
        
        # Start monitoring task
        self._monitoring_task = None

    # =====================================================
    # Rule Management
    # =====================================================

    async def add_rule(self, rule_name: str, rule_text: str, category: RuleCategory,
                      priority: int = 5, enforcement_level: float = 1.0,
                      scope: str = "all", conditions: Dict[str, Any] = None,
                      violation_penalties: Dict[ViolationType, float] = None) -> str:
        """
        Add a new rule to the compliance monitoring system.
        
        Args:
            rule_name: Human-readable rule name
            rule_text: Full text description of the rule
            category: Rule category
            priority: Priority level (1-10)
            enforcement_level: How strictly enforced (0.0-1.0)
            scope: Who the rule applies to
            conditions: Conditions for rule application
            violation_penalties: Penalty weights by violation type
        
        Returns:
            rule_id: Unique identifier for the rule
        """
        try:
            rule_id = str(uuid.uuid4())
            
            # Default penalties if not provided
            if violation_penalties is None:
                violation_penalties = {
                    ViolationType.MINOR: 0.05,
                    ViolationType.MODERATE: 0.15,
                    ViolationType.SEVERE: 0.30,
                    ViolationType.CRITICAL: 0.50
                }
            
            rule = Rule(
                rule_id=rule_id,
                rule_name=rule_name,
                rule_text=rule_text,
                category=category,
                priority=max(1, min(10, priority)),
                enforcement_level=max(0.0, min(1.0, enforcement_level)),
                scope=scope,
                conditions=conditions or {},
                violation_penalties=violation_penalties,
                is_active=True,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                version=1,
                metadata={"created_by": "system"}
            )
            
            # Store rule
            self.active_rules[rule_id] = rule
            
            # Create validator function
            await self._create_rule_validator(rule)
            
            # Store in Store API
            await self._store_rule(rule)
            
            # Store in PostgreSQL
            if self.postgres_persistence:
                await self._store_rule_in_db(rule)
            
            self.metrics["total_rules"] += 1
            self.metrics["active_rules"] += 1
            
            self.logger.info(f"Added rule '{rule_name}' ({rule_id}) in category {category.value}")
            return rule_id
            
        except Exception as e:
            self.logger.error(f"Failed to add rule: {str(e)}")
            raise

    async def update_rule(self, rule_id: str, **updates) -> bool:
        """Update an existing rule."""
        try:
            if rule_id not in self.active_rules:
                return False
            
            rule = self.active_rules[rule_id]
            old_version = rule.version
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(rule, field):
                    setattr(rule, field, value)
            
            rule.last_updated = datetime.now()
            rule.version = old_version + 1
            
            # Update validator if rule text changed
            if "rule_text" in updates or "conditions" in updates:
                await self._create_rule_validator(rule)
            
            # Update storage
            await self._store_rule(rule)
            
            self.logger.info(f"Updated rule {rule_id} to version {rule.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update rule {rule_id}: {str(e)}")
            return False

    async def deactivate_rule(self, rule_id: str, reason: str = "manual_deactivation") -> bool:
        """Deactivate a rule."""
        try:
            if rule_id not in self.active_rules:
                return False
            
            rule = self.active_rules[rule_id]
            rule.is_active = False
            rule.last_updated = datetime.now()
            rule.metadata["deactivation_reason"] = reason
            rule.metadata["deactivated_at"] = datetime.now().isoformat()
            
            # Remove validator
            if rule_id in self.rule_validators:
                del self.rule_validators[rule_id]
            
            await self._store_rule(rule)
            
            self.metrics["active_rules"] -= 1
            
            self.logger.info(f"Deactivated rule {rule_id}: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deactivate rule {rule_id}: {str(e)}")
            return False

    # =====================================================
    # Compliance Monitoring
    # =====================================================

    async def start_monitoring(self) -> None:
        """Start the compliance monitoring system."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Started compliance monitoring")

    async def stop_monitoring(self) -> None:
        """Stop the compliance monitoring system."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            self.logger.info("Stopped compliance monitoring")

    async def monitor_action(self, agent_id: str, action: Dict[str, Any]) -> List[str]:
        """
        Monitor a specific agent action against all applicable rules.
        
        Args:
            agent_id: ID of the agent performing the action
            action: Action data to monitor
        
        Returns:
            List of violation IDs detected
        """
        try:
            violations_detected = []
            
            # Get agent's compliance record
            if agent_id not in self.compliance_records:
                await self._initialize_compliance_record(agent_id)
            
            # Check action against all applicable rules
            for rule_id, rule in self.active_rules.items():
                if not rule.is_active:
                    continue
                
                # Check if rule applies to this agent
                if not await self._rule_applies_to_agent(rule, agent_id):
                    continue
                
                # Validate action against rule
                violation = await self._validate_action_against_rule(agent_id, action, rule)
                if violation:
                    violations_detected.append(violation.violation_id)
                    await self._process_violation(violation)
            
            return violations_detected
            
        except Exception as e:
            self.logger.error(f"Failed to monitor action for {agent_id}: {str(e)}")
            return []

    async def get_compliance_score(self, agent_id: str) -> float:
        """Get current compliance score for an agent."""
        try:
            if agent_id not in self.compliance_records:
                await self._initialize_compliance_record(agent_id)
                return 1.0  # New agents start with perfect compliance
            
            record = self.compliance_records[agent_id]
            
            # Update score with decay
            await self._update_compliance_score(agent_id)
            
            return record.overall_compliance_score
            
        except Exception as e:
            self.logger.error(f"Failed to get compliance score for {agent_id}: {str(e)}")
            return 0.5  # Default to neutral score on error

    async def get_agent_violations(self, agent_id: str, days_back: int = 30) -> List[Violation]:
        """Get recent violations for an agent."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            agent_violations = []
            
            for violation in self.recent_violations.values():
                if (violation.agent_id == agent_id and 
                    violation.detected_at >= cutoff_date):
                    agent_violations.append(violation)
            
            # Sort by detection date, most recent first
            agent_violations.sort(key=lambda x: x.detected_at, reverse=True)
            
            return agent_violations
            
        except Exception as e:
            self.logger.error(f"Failed to get violations for {agent_id}: {str(e)}")
            return []

    # =====================================================
    # Violation Processing
    # =====================================================

    async def _validate_action_against_rule(self, agent_id: str, action: Dict[str, Any], 
                                          rule: Rule) -> Optional[Violation]:
        """Validate an action against a specific rule."""
        try:
            # Use custom validator if available
            if rule.rule_id in self.rule_validators:
                validator = self.rule_validators[rule.rule_id]
                violation_result = await validator(agent_id, action, rule)
                if violation_result:
                    return await self._create_violation_record(
                        agent_id, rule.rule_id, violation_result["type"],
                        violation_result["description"], action,
                        violation_result.get("severity", 0.5),
                        violation_result.get("context", {})
                    )
            else:
                # Use generic text-based validation
                violation_result = await self._generic_rule_validation(action, rule)
                if violation_result:
                    return await self._create_violation_record(
                        agent_id, rule.rule_id, violation_result["type"],
                        violation_result["description"], action,
                        violation_result.get("severity", 0.5),
                        violation_result.get("context", {})
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to validate action against rule {rule.rule_id}: {str(e)}")
            return None

    async def _create_violation_record(self, agent_id: str, rule_id: str, 
                                     violation_type: ViolationType, description: str,
                                     action: Dict[str, Any], severity: float,
                                     context: Dict[str, Any] = None) -> Violation:
        """Create a violation record."""
        violation_id = str(uuid.uuid4())
        
        violation = Violation(
            violation_id=violation_id,
            agent_id=agent_id,
            rule_id=rule_id,
            violation_type=violation_type,
            description=description,
            action_that_violated=action.copy(),
            severity_score=max(0.0, min(1.0, severity)),
            detected_at=datetime.now(),
            context=context or {},
            witnesses=set(),
            evidence=[{"type": "action_log", "data": action}],
            is_disputed=False,
            resolution_status="pending",
            penalty_applied=None,
            metadata={"detected_by": "compliance_monitor"}
        )
        
        self.recent_violations[violation_id] = violation
        
        return violation

    async def _process_violation(self, violation: Violation) -> None:
        """Process a detected violation."""
        try:
            # Store violation
            await self._store_violation(violation)
            
            # Update agent's compliance record
            await self._update_agent_compliance_for_violation(violation)
            
            # Apply immediate penalties if configured
            await self._apply_violation_penalty(violation)
            
            # Notify relevant parties
            await self._notify_violation(violation)
            
            self.metrics["violations_detected"] += 1
            
            self.logger.warning(f"Processed violation {violation.violation_id} for agent {violation.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process violation {violation.violation_id}: {str(e)}")

    async def _update_agent_compliance_for_violation(self, violation: Violation) -> None:
        """Update agent's compliance record after a violation."""
        agent_id = violation.agent_id
        record = self.compliance_records[agent_id]
        
        # Get rule to determine penalty
        rule = self.active_rules.get(violation.rule_id)
        if not rule:
            return
        
        # Calculate compliance score penalty
        penalty = rule.violation_penalties.get(violation.violation_type, 0.1)
        penalty *= violation.severity_score
        penalty *= rule.enforcement_level
        
        # Apply penalty to overall score
        record.overall_compliance_score = max(0.0, record.overall_compliance_score - penalty)
        
        # Update category-specific score
        if rule.category not in record.category_scores:
            record.category_scores[rule.category] = 1.0
        
        record.category_scores[rule.category] = max(0.0, 
            record.category_scores[rule.category] - penalty)
        
        # Update violation counts
        if violation.violation_type not in record.violation_count:
            record.violation_count[violation.violation_type] = 0
        record.violation_count[violation.violation_type] += 1
        
        # Add to recent violations
        record.recent_violations.append(violation.violation_id)
        if len(record.recent_violations) > 20:  # Keep only 20 most recent
            record.recent_violations = record.recent_violations[-20:]
        
        # Update compliance trend
        record.compliance_trend.append(record.overall_compliance_score)
        if len(record.compliance_trend) > 100:  # Keep 100 data points
            record.compliance_trend = record.compliance_trend[-100:]
        
        # Check if status change needed
        await self._check_compliance_status_change(agent_id)
        
        record.last_updated = datetime.now()

    async def _check_compliance_status_change(self, agent_id: str) -> None:
        """Check if agent's compliance status needs to change."""
        record = self.compliance_records[agent_id]
        old_status = record.status
        
        # Calculate total strikes
        total_strikes = 0
        for violation_type, count in record.violation_count.items():
            threshold = self.config["strikes_threshold"][violation_type]
            if count >= threshold:
                total_strikes += count // threshold
        
        # Determine new status based on score and strikes
        if record.overall_compliance_score >= 0.8 and total_strikes == 0:
            record.status = ComplianceStatus.COMPLIANT
        elif record.overall_compliance_score >= 0.6 and total_strikes <= 1:
            record.status = ComplianceStatus.WARNING
        elif record.overall_compliance_score >= 0.4 and total_strikes <= 3:
            record.status = ComplianceStatus.PROBATION
            if old_status != ComplianceStatus.PROBATION:
                record.probation_until = datetime.now() + timedelta(
                    days=self.config["probation_period_days"])
        elif total_strikes <= 5:
            record.status = ComplianceStatus.SUSPENDED
        else:
            record.status = ComplianceStatus.BANNED
        
        if old_status != record.status:
            record.status_since = datetime.now()
            await self._notify_status_change(agent_id, old_status, record.status)

    # =====================================================
    # Rule Validation Functions
    # =====================================================

    async def _create_rule_validator(self, rule: Rule) -> None:
        """Create a validator function for a rule."""
        # This would create custom validation logic based on rule content
        # For now, using generic text-based validation
        
        async def generic_validator(agent_id: str, action: Dict[str, Any], rule: Rule):
            return await self._generic_rule_validation(action, rule)
        
        self.rule_validators[rule.rule_id] = generic_validator

    async def _generic_rule_validation(self, action: Dict[str, Any], rule: Rule) -> Optional[Dict[str, Any]]:
        """Generic rule validation based on text patterns."""
        try:
            action_text = json.dumps(action).lower()
            rule_text = rule.rule_text.lower()
            
            # Simple keyword-based violation detection
            violation_indicators = {
                "harassment": ["harass", "bully", "intimidate", "threaten"],
                "spam": ["spam", "repeat", "flood"],
                "inappropriate": ["inappropriate", "offensive", "rude"],
                "resource_abuse": ["hoard", "monopolize", "abuse", "waste"]
            }
            
            for violation_type, keywords in violation_indicators.items():
                if any(keyword in action_text for keyword in keywords):
                    # Check if rule prohibits this behavior
                    if any(keyword in rule_text for keyword in ["prohibit", "forbidden", "not allowed"]):
                        return {
                            "type": ViolationType.MODERATE,
                            "description": f"Potential {violation_type} behavior detected",
                            "severity": 0.6,
                            "context": {"detected_keywords": [k for k in keywords if k in action_text]}
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Generic validation failed: {str(e)}")
            return None

    async def _rule_applies_to_agent(self, rule: Rule, agent_id: str) -> bool:
        """Check if a rule applies to a specific agent."""
        # Simple scope checking - could be enhanced with role-based logic
        if rule.scope == "all":
            return True
        elif rule.scope == "members":
            return agent_id in self.compliance_records
        elif rule.scope.startswith("role:"):
            # Would check agent's role here
            return True
        else:
            return False

    # =====================================================
    # Helper Methods
    # =====================================================

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while True:
                await asyncio.sleep(self.config["monitoring_frequency_seconds"])
                
                # Perform periodic maintenance
                await self._decay_compliance_scores()
                await self._cleanup_old_violations()
                await self._update_metrics()
                
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error(f"Monitoring loop error: {str(e)}")

    async def _initialize_compliance_record(self, agent_id: str) -> None:
        """Initialize compliance record for a new agent."""
        record = ComplianceRecord(
            agent_id=agent_id,
            overall_compliance_score=1.0,
            category_scores={},
            violation_count={},
            recent_violations=[],
            compliance_trend=[1.0],
            status=ComplianceStatus.COMPLIANT,
            status_since=datetime.now(),
            probation_until=None,
            strikes=0,
            commendations=0,
            last_updated=datetime.now(),
            metadata={"initialized_at": datetime.now().isoformat()}
        )
        
        self.compliance_records[agent_id] = record

    async def _decay_compliance_scores(self) -> None:
        """Apply daily decay to compliance scores."""
        decay_rate = self.config["compliance_score_decay"]
        
        for record in self.compliance_records.values():
            # Slowly improve scores over time (reward good behavior)
            if record.overall_compliance_score < 1.0:
                record.overall_compliance_score = min(1.0, 
                    record.overall_compliance_score + decay_rate)
            
            for category in record.category_scores:
                if record.category_scores[category] < 1.0:
                    record.category_scores[category] = min(1.0,
                        record.category_scores[category] + decay_rate)

    async def _cleanup_old_violations(self) -> None:
        """Remove old violations from memory."""
        cutoff_date = datetime.now() - timedelta(days=self.config["violation_memory_days"])
        
        old_violations = [v_id for v_id, violation in self.recent_violations.items()
                         if violation.detected_at < cutoff_date]
        
        for v_id in old_violations:
            del self.recent_violations[v_id]

    async def _update_compliance_score(self, agent_id: str) -> None:
        """Update compliance score for an agent."""
        record = self.compliance_records[agent_id]
        
        # Simple update - could be enhanced with more sophisticated algorithms
        record.last_updated = datetime.now()

    async def _apply_violation_penalty(self, violation: Violation) -> None:
        """Apply immediate penalties for violations."""
        # This would apply immediate consequences like temporary restrictions
        pass

    async def _notify_violation(self, violation: Violation) -> None:
        """Notify relevant parties about a violation."""
        await self.store_integration._broadcast_community_event("violation_detected", {
            "agent_id": violation.agent_id,
            "rule_id": violation.rule_id,
            "violation_type": violation.violation_type.value,
            "severity": violation.severity_score
        })

    async def _notify_status_change(self, agent_id: str, old_status: ComplianceStatus, 
                                  new_status: ComplianceStatus) -> None:
        """Notify about compliance status changes."""
        await self.store_integration._broadcast_community_event("compliance_status_change", {
            "agent_id": agent_id,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "timestamp": datetime.now().isoformat()
        })

    async def _update_metrics(self) -> None:
        """Update system metrics."""
        if self.compliance_records:
            total_compliance = sum(r.overall_compliance_score for r in self.compliance_records.values())
            self.metrics["average_compliance"] = total_compliance / len(self.compliance_records)

    # =====================================================
    # Storage Methods
    # =====================================================

    async def _store_rule(self, rule: Rule) -> None:
        """Store rule in Store API."""
        if self.store_integration.store:
            rule_data = asdict(rule)
            rule_data["category"] = rule.category.value
            rule_data["created_at"] = rule.created_at.isoformat()
            rule_data["last_updated"] = rule.last_updated.isoformat()
            rule_data["violation_penalties"] = {k.value: v for k, v in rule.violation_penalties.items()}
            
            await self.store_integration.store.aput(
                StoreNamespace.GOVERNANCE.value,
                f"rule_{rule.rule_id}",
                rule_data
            )

    async def _store_violation(self, violation: Violation) -> None:
        """Store violation in Store API."""
        if self.store_integration.store:
            violation_data = asdict(violation)
            violation_data["violation_type"] = violation.violation_type.value
            violation_data["detected_at"] = violation.detected_at.isoformat()
            violation_data["witnesses"] = list(violation.witnesses)
            
            await self.store_integration.store.aput(
                StoreNamespace.GOVERNANCE.value,
                f"violation_{violation.violation_id}",
                violation_data
            )

    async def _store_rule_in_db(self, rule: Rule) -> None:
        """Store rule in PostgreSQL."""
        if self.postgres_persistence:
            async with self.postgres_persistence.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO compliance_rules (rule_id, rule_name, rule_text, category, priority,
                                                enforcement_level, scope, conditions, violation_penalties,
                                                is_active, created_at, last_updated, version, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (rule_id) DO UPDATE SET
                        rule_name = EXCLUDED.rule_name,
                        rule_text = EXCLUDED.rule_text,
                        category = EXCLUDED.category,
                        priority = EXCLUDED.priority,
                        enforcement_level = EXCLUDED.enforcement_level,
                        scope = EXCLUDED.scope,
                        conditions = EXCLUDED.conditions,
                        violation_penalties = EXCLUDED.violation_penalties,
                        is_active = EXCLUDED.is_active,
                        last_updated = EXCLUDED.last_updated,
                        version = EXCLUDED.version,
                        metadata = EXCLUDED.metadata
                """, rule.rule_id, rule.rule_name, rule.rule_text, rule.category.value,
                rule.priority, rule.enforcement_level, rule.scope, json.dumps(rule.conditions),
                json.dumps({k.value: v for k, v in rule.violation_penalties.items()}),
                rule.is_active, rule.created_at, rule.last_updated, rule.version,
                json.dumps(rule.metadata))

    # =====================================================
    # System Interface
    # =====================================================

    async def get_compliance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive compliance monitoring metrics."""
        try:
            active_rules = len([r for r in self.active_rules.values() if r.is_active])
            
            # Calculate compliance distribution
            status_distribution = {}
            for status in ComplianceStatus:
                status_distribution[status.value] = len([r for r in self.compliance_records.values()
                                                        if r.status == status])
            
            recent_violations = len([v for v in self.recent_violations.values()
                                   if v.detected_at >= datetime.now() - timedelta(days=7)])
            
            return {
                "total_rules": len(self.active_rules),
                "active_rules": active_rules,
                "monitored_agents": len(self.compliance_records),
                "average_compliance_score": self.metrics["average_compliance"],
                "compliance_status_distribution": status_distribution,
                "recent_violations": recent_violations,
                "total_violations": len(self.recent_violations),
                "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get compliance metrics: {str(e)}")
            return {}


# Helper functions

def create_compliance_monitor(store_integration: MemoryStoreIntegration,
                            postgres_persistence=None, community_size: int = 50) -> ComplianceMonitor:
    """Create a ComplianceMonitor instance."""
    return ComplianceMonitor(store_integration, postgres_persistence, community_size)


# Example usage
if __name__ == "__main__":
    async def test_compliance_monitor():
        """Test the Compliance Monitor."""
        print("Compliance Monitor loaded successfully")
        
    asyncio.run(test_compliance_monitor())