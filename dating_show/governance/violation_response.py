"""
File: violation_response.py
Description: Violation Response System for community response to rule violations.
Handles community responses, punishment and rehabilitation mechanisms,
social pressure, reputation effects, and restorative justice processes.
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import random

# Import governance and agent components
try:
    from .compliance_monitoring import ComplianceMonitor, Rule, RuleCategory, ViolationType, ComplianceViolation
    from ..agents.memory_structures.store_integration import MemoryStoreIntegration, StoreNamespace
    from ..agents.enhanced_agent_state import GovernanceData
except ImportError:
    # Mock classes for testing
    class ComplianceMonitor:
        pass
    class Rule:
        pass
    class RuleCategory:
        pass
    class ViolationType:
        pass
    class ComplianceViolation:
        pass
    class MemoryStoreIntegration:
        pass
    class StoreNamespace:
        GOVERNANCE = "governance"
    class GovernanceData:
        pass


class ResponseType(Enum):
    """Types of community responses to violations."""
    WARNING = "warning"  # Formal warning
    MEDIATION = "mediation"  # Mediated discussion
    COMMUNITY_SERVICE = "community_service"  # Service to community
    TEMPORARY_RESTRICTION = "temporary_restriction"  # Limited privileges
    REPUTATION_PENALTY = "reputation_penalty"  # Social reputation impact
    REHABILITATION = "rehabilitation"  # Behavioral rehabilitation
    RESTITUTION = "restitution"  # Make amends to affected parties
    SUSPENSION = "suspension"  # Temporary exclusion
    EXPULSION = "expulsion"  # Permanent removal
    RESTORATIVE_JUSTICE = "restorative_justice"  # Focus on repair and healing


class ResponseSeverity(Enum):
    """Severity levels for violation responses."""
    ADVISORY = "advisory"  # Gentle guidance
    CORRECTIVE = "corrective"  # Corrective action required
    PUNITIVE = "punitive"  # Punishment-focused
    REHABILITATIVE = "rehabilitative"  # Rehabilitation-focused
    TRANSFORMATIVE = "transformative"  # Community transformation


class ResponseStatus(Enum):
    """Status of violation response."""
    PROPOSED = "proposed"  # Response proposed
    DELIBERATING = "deliberating"  # Community discussing
    APPROVED = "approved"  # Community approved response
    ACTIVE = "active"  # Response being implemented
    COMPLETED = "completed"  # Response completed
    APPEALED = "appealed"  # Under appeal
    OVERTURNED = "overturned"  # Appeal successful
    FAILED = "failed"  # Response implementation failed


@dataclass
class ViolationResponse:
    """Represents a community response to a rule violation."""
    response_id: str
    violation_id: str
    violator_agent_id: str
    response_type: ResponseType
    severity: ResponseSeverity
    status: ResponseStatus
    description: str
    duration_days: Optional[int] = None
    conditions: List[str] = field(default_factory=list)
    community_support: float = 0.0  # 0.0 to 1.0
    proposed_by: str = "community"
    proposed_at: datetime = field(default_factory=datetime.now)
    implemented_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    effectiveness_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.conditions:
            self.conditions = []
        if not self.metadata:
            self.metadata = {}


@dataclass
class RestorativeProcess:
    """Represents a restorative justice process."""
    process_id: str
    violation_id: str
    violator_agent_id: str
    affected_parties: List[str]
    facilitator_agent_id: Optional[str]
    process_type: str  # "circle", "mediation", "conference"
    status: str  # "scheduled", "active", "completed", "cancelled"
    scheduled_at: datetime
    objectives: List[str]
    agreements_reached: List[str] = field(default_factory=list)
    satisfaction_scores: Dict[str, float] = field(default_factory=dict)
    follow_up_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RehabilitationPlan:
    """Represents a rehabilitation plan for repeat offenders."""
    plan_id: str
    agent_id: str
    violation_history: List[str]
    goals: List[str]
    interventions: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    support_agents: List[str]
    progress_tracking: Dict[str, Any]
    completion_criteria: List[str]
    success_metrics: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ViolationResponseSystem:
    """
    Violation Response System for community-driven violation handling.
    Implements various response mechanisms from warnings to restorative justice.
    """

    def __init__(self, store_integration: MemoryStoreIntegration,
                 compliance_monitor: ComplianceMonitor = None):
        """
        Initialize the Violation Response System.
        
        Args:
            store_integration: Store API integration
            compliance_monitor: Compliance monitoring system
        """
        self.store_integration = store_integration
        self.compliance_monitor = compliance_monitor
        self.logger = logging.getLogger(f"{__name__}.ViolationResponseSystem")
        
        # Active responses and processes
        self.active_responses = {}  # response_id -> ViolationResponse
        self.restorative_processes = {}  # process_id -> RestorativeProcess
        self.rehabilitation_plans = {}  # agent_id -> RehabilitationPlan
        
        # Response configuration
        self.response_config = {
            "community_input_weight": 0.7,  # Weight of community input in response decisions
            "repeat_offender_threshold": 3,  # Number of violations for repeat offender status
            "rehabilitation_trigger_threshold": 5,  # Violations triggering rehabilitation
            "community_service_hours": {"minor": 2, "moderate": 8, "severe": 20, "critical": 40},
            "suspension_durations": {"minor": 1, "moderate": 7, "severe": 30, "critical": 90},
            "mediation_preference": True,  # Prefer mediation over punitive measures
            "restorative_justice_threshold": 0.6  # Community support threshold for restorative processes
        }
        
        # Response effectiveness tracking
        self.effectiveness_metrics = {
            "responses_implemented": 0,
            "recidivism_rate": 0.0,
            "community_satisfaction": 0.0,
            "rehabilitation_success_rate": 0.0
        }

    # =====================================================
    # Response Generation and Approval
    # =====================================================

    async def generate_response_proposals(self, violation: ComplianceViolation) -> List[ViolationResponse]:
        """
        Generate appropriate response proposals for a violation.
        
        Args:
            violation: The compliance violation to respond to
        
        Returns:
            List of proposed responses
        """
        try:
            proposals = []
            
            # Get violation context and history
            agent_history = await self._get_agent_violation_history(violation.agent_id)
            community_sentiment = await self._assess_community_sentiment(violation)
            
            # Generate responses based on violation severity and history
            if violation.violation_type == ViolationType.MINOR:
                proposals.extend(await self._generate_minor_violation_responses(violation, agent_history))
            elif violation.violation_type == ViolationType.MODERATE:
                proposals.extend(await self._generate_moderate_violation_responses(violation, agent_history))
            elif violation.violation_type == ViolationType.SEVERE:
                proposals.extend(await self._generate_severe_violation_responses(violation, agent_history))
            elif violation.violation_type == ViolationType.CRITICAL:
                proposals.extend(await self._generate_critical_violation_responses(violation, agent_history))
            
            # Add restorative justice options if appropriate
            if community_sentiment.get("restorative_preference", 0.0) > self.response_config["restorative_justice_threshold"]:
                restorative_proposal = await self._generate_restorative_response(violation, agent_history)
                if restorative_proposal:
                    proposals.append(restorative_proposal)
            
            # Rank proposals by appropriateness
            proposals = await self._rank_response_proposals(proposals, violation, community_sentiment)
            
            self.logger.info(f"Generated {len(proposals)} response proposals for violation {violation.violation_id}")
            return proposals
            
        except Exception as e:
            self.logger.error(f"Failed to generate response proposals: {str(e)}")
            return []

    async def submit_community_response_vote(self, violation_id: str, 
                                           proposed_responses: List[ViolationResponse]) -> str:
        """
        Submit response proposals to the community for voting.
        
        Args:
            violation_id: ID of the violation
            proposed_responses: List of proposed responses
        
        Returns:
            voting_session_id if successful
        """
        try:
            from .voting_system import create_democratic_voting_system, VotingMechanism
            
            # Create voting session for response selection
            voting_system = create_democratic_voting_system(self.store_integration)
            
            # Get eligible voters (community members)
            eligible_voters = await self._get_eligible_response_voters()
            
            # Create voting session
            voting_session_id = await voting_system.create_voting_session(
                proposal_id=f"violation_response_{violation_id}",
                voting_mechanism=VotingMechanism.SIMPLE_MAJORITY,
                eligible_voters=eligible_voters,
                voting_duration_hours=24  # 24 hour voting period
            )
            
            # Store proposed responses
            for response in proposed_responses:
                await self._store_proposed_response(response, voting_session_id)
            
            # Notify community
            await self._notify_community_of_response_vote(violation_id, voting_session_id)
            
            self.logger.info(f"Community voting initiated for violation {violation_id}")
            return voting_session_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit community response vote: {str(e)}")
            return ""

    async def implement_approved_response(self, response: ViolationResponse) -> bool:
        """
        Implement an approved community response.
        
        Args:
            response: The approved violation response
        
        Returns:
            bool: True if implementation successful
        """
        try:
            response.status = ResponseStatus.ACTIVE
            response.implemented_at = datetime.now()
            
            # Implement specific response type
            success = False
            if response.response_type == ResponseType.WARNING:
                success = await self._implement_warning(response)
            elif response.response_type == ResponseType.MEDIATION:
                success = await self._implement_mediation(response)
            elif response.response_type == ResponseType.COMMUNITY_SERVICE:
                success = await self._implement_community_service(response)
            elif response.response_type == ResponseType.TEMPORARY_RESTRICTION:
                success = await self._implement_temporary_restriction(response)
            elif response.response_type == ResponseType.REPUTATION_PENALTY:
                success = await self._implement_reputation_penalty(response)
            elif response.response_type == ResponseType.REHABILITATION:
                success = await self._implement_rehabilitation(response)
            elif response.response_type == ResponseType.RESTITUTION:
                success = await self._implement_restitution(response)
            elif response.response_type == ResponseType.RESTORATIVE_JUSTICE:
                success = await self._implement_restorative_justice(response)
            elif response.response_type == ResponseType.SUSPENSION:
                success = await self._implement_suspension(response)
            
            if success:
                response.status = ResponseStatus.COMPLETED
                response.completed_at = datetime.now()
                self.effectiveness_metrics["responses_implemented"] += 1
            else:
                response.status = ResponseStatus.FAILED
            
            # Store updated response
            await self._store_response_update(response)
            
            # Notify stakeholders
            await self._notify_response_implementation(response, success)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to implement response {response.response_id}: {str(e)}")
            return False

    # =====================================================
    # Restorative Justice Processes
    # =====================================================

    async def initiate_restorative_process(self, violation: ComplianceViolation,
                                         process_type: str = "circle") -> Optional[RestorativeProcess]:
        """
        Initiate a restorative justice process.
        
        Args:
            violation: The violation to address
            process_type: Type of restorative process
        
        Returns:
            RestorativeProcess if successful
        """
        try:
            # Identify affected parties
            affected_parties = await self._identify_affected_parties(violation)
            
            # Find suitable facilitator
            facilitator = await self._find_restorative_facilitator()
            
            # Create restorative process
            process = RestorativeProcess(
                process_id=str(uuid.uuid4())[:8],
                violation_id=violation.violation_id,
                violator_agent_id=violation.agent_id,
                affected_parties=affected_parties,
                facilitator_agent_id=facilitator,
                process_type=process_type,
                status="scheduled",
                scheduled_at=datetime.now() + timedelta(days=2),
                objectives=[
                    "Understanding impact of violation",
                    "Taking responsibility",
                    "Making amends",
                    "Preventing future violations",
                    "Healing community relationships"
                ]
            )
            
            # Store process
            self.restorative_processes[process.process_id] = process
            await self._store_restorative_process(process)
            
            # Invite participants
            await self._invite_restorative_participants(process)
            
            self.logger.info(f"Initiated restorative process {process.process_id} for violation {violation.violation_id}")
            return process
            
        except Exception as e:
            self.logger.error(f"Failed to initiate restorative process: {str(e)}")
            return None

    async def conduct_restorative_session(self, process_id: str) -> bool:
        """
        Conduct a restorative justice session.
        
        Args:
            process_id: ID of the restorative process
        
        Returns:
            bool: True if session successful
        """
        try:
            process = self.restorative_processes.get(process_id)
            if not process:
                return False
            
            process.status = "active"
            
            # Simulate restorative session phases
            session_phases = [
                "opening_and_introductions",
                "sharing_impact_stories",
                "violator_takes_responsibility", 
                "exploring_needs_and_harms",
                "developing_agreements",
                "closing_and_commitments"
            ]
            
            session_outcomes = {}
            
            for phase in session_phases:
                outcome = await self._conduct_restorative_phase(process, phase)
                session_outcomes[phase] = outcome
            
            # Develop agreements
            agreements = await self._develop_restorative_agreements(process, session_outcomes)
            process.agreements_reached = agreements
            
            # Collect satisfaction scores
            process.satisfaction_scores = await self._collect_satisfaction_scores(process)
            
            # Determine follow-up needs
            process.follow_up_required = any(score < 0.7 for score in process.satisfaction_scores.values())
            
            process.status = "completed"
            
            # Store updated process
            await self._store_restorative_process(process)
            
            self.logger.info(f"Completed restorative session {process_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to conduct restorative session: {str(e)}")
            return False

    # =====================================================
    # Rehabilitation System
    # =====================================================

    async def create_rehabilitation_plan(self, agent_id: str) -> Optional[RehabilitationPlan]:
        """
        Create a rehabilitation plan for repeat offender.
        
        Args:
            agent_id: ID of the agent needing rehabilitation
        
        Returns:
            RehabilitationPlan if successful
        """
        try:
            # Get violation history
            violation_history = await self._get_agent_violation_history(agent_id)
            
            if len(violation_history) < self.response_config["rehabilitation_trigger_threshold"]:
                return None
            
            # Analyze violation patterns
            patterns = await self._analyze_violation_patterns(violation_history)
            
            # Create rehabilitation plan
            plan = RehabilitationPlan(
                plan_id=str(uuid.uuid4())[:8],
                agent_id=agent_id,
                violation_history=[v.violation_id for v in violation_history],
                goals=await self._generate_rehabilitation_goals(patterns),
                interventions=await self._design_rehabilitation_interventions(patterns),
                milestones=await self._create_rehabilitation_milestones(patterns),
                support_agents=await self._assign_support_agents(agent_id),
                progress_tracking={"violations": 0, "compliance_score": 0.0, "community_integration": 0.0},
                completion_criteria=["60 days violation-free", "compliance_score > 0.8", "community_acceptance > 0.6"]
            )
            
            # Store plan
            self.rehabilitation_plans[agent_id] = plan
            await self._store_rehabilitation_plan(plan)
            
            # Notify stakeholders
            await self._notify_rehabilitation_plan_created(plan)
            
            self.logger.info(f"Created rehabilitation plan {plan.plan_id} for agent {agent_id}")
            return plan
            
        except Exception as e:
            self.logger.error(f"Failed to create rehabilitation plan: {str(e)}")
            return None

    async def monitor_rehabilitation_progress(self, agent_id: str) -> Dict[str, Any]:
        """
        Monitor progress of an agent's rehabilitation.
        
        Args:
            agent_id: ID of the agent in rehabilitation
        
        Returns:
            Progress report
        """
        try:
            plan = self.rehabilitation_plans.get(agent_id)
            if not plan:
                return {}
            
            # Assess current progress
            progress = {
                "plan_id": plan.plan_id,
                "days_in_program": (datetime.now() - plan.created_at).days,
                "goals_achieved": 0,
                "milestones_reached": 0,
                "current_compliance_score": 0.0,
                "recent_violations": 0,
                "community_feedback": {},
                "recommendation": "continue"
            }
            
            # Check milestone progress
            for milestone in plan.milestones:
                if await self._check_milestone_completion(milestone, agent_id):
                    progress["milestones_reached"] += 1
            
            # Update progress tracking
            plan.progress_tracking.update({
                "last_updated": datetime.now().isoformat(),
                "milestones_reached": progress["milestones_reached"],
                "total_milestones": len(plan.milestones)
            })
            
            # Store updated plan
            await self._store_rehabilitation_plan(plan)
            
            return progress
            
        except Exception as e:
            self.logger.error(f"Failed to monitor rehabilitation progress: {str(e)}")
            return {}

    # =====================================================
    # Response Implementation Methods
    # =====================================================

    async def _implement_warning(self, response: ViolationResponse) -> bool:
        """Implement a formal warning."""
        try:
            warning_message = {
                "type": "formal_warning",
                "recipient": response.violator_agent_id,
                "violation_id": response.violation_id,
                "message": response.description,
                "conditions": response.conditions,
                "issued_at": datetime.now().isoformat()
            }
            
            # Store warning in agent's record
            if self.store_integration.store:
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"warning_{response.violator_agent_id}_{response.response_id}",
                    warning_message
                )
            
            # Notify agent
            await self._notify_agent_of_response(response.violator_agent_id, warning_message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to implement warning: {str(e)}")
            return False

    async def _implement_reputation_penalty(self, response: ViolationResponse) -> bool:
        """Implement reputation penalty."""
        try:
            penalty_amount = response.metadata.get("penalty_amount", 0.1)
            
            # Apply reputation penalty (would integrate with reputation system)
            penalty_record = {
                "type": "reputation_penalty",
                "agent_id": response.violator_agent_id,
                "penalty_amount": penalty_amount,
                "reason": response.description,
                "applied_at": datetime.now().isoformat()
            }
            
            if self.store_integration.store:
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"reputation_penalty_{response.violator_agent_id}_{response.response_id}",
                    penalty_record
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to implement reputation penalty: {str(e)}")
            return False

    async def _implement_community_service(self, response: ViolationResponse) -> bool:
        """Implement community service requirement."""
        try:
            service_hours = response.metadata.get("service_hours", 8)
            
            service_assignment = {
                "type": "community_service",
                "agent_id": response.violator_agent_id,
                "hours_required": service_hours,
                "tasks": ["community_cleanup", "mentor_new_agents", "assist_with_events"],
                "supervisor": response.metadata.get("supervisor"),
                "deadline": (datetime.now() + timedelta(days=30)).isoformat(),
                "status": "assigned"
            }
            
            if self.store_integration.store:
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"community_service_{response.violator_agent_id}_{response.response_id}",
                    service_assignment
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to implement community service: {str(e)}")
            return False

    # =====================================================
    # Helper Methods
    # =====================================================

    async def _get_agent_violation_history(self, agent_id: str) -> List[ComplianceViolation]:
        """Get violation history for an agent."""
        # Placeholder - would integrate with compliance monitoring
        return []

    async def _assess_community_sentiment(self, violation: ComplianceViolation) -> Dict[str, Any]:
        """Assess community sentiment about a violation."""
        # Simplified sentiment analysis
        return {
            "severity_perception": random.uniform(0.3, 1.0),
            "restorative_preference": random.uniform(0.4, 0.8),
            "punitive_preference": random.uniform(0.2, 0.6),
            "community_concern": random.uniform(0.3, 0.9)
        }

    async def _generate_minor_violation_responses(self, violation: ComplianceViolation, 
                                               history: List) -> List[ViolationResponse]:
        """Generate responses for minor violations."""
        responses = []
        
        # Warning response
        responses.append(ViolationResponse(
            response_id=str(uuid.uuid4())[:8],
            violation_id=violation.violation_id,
            violator_agent_id=violation.agent_id,
            response_type=ResponseType.WARNING,
            severity=ResponseSeverity.ADVISORY,
            status=ResponseStatus.PROPOSED,
            description="Formal warning with guidance on rule compliance",
            conditions=["Review community guidelines", "Acknowledge understanding"]
        ))
        
        # Mediation if repeat offender
        if len(history) > 1:
            responses.append(ViolationResponse(
                response_id=str(uuid.uuid4())[:8],
                violation_id=violation.violation_id,
                violator_agent_id=violation.agent_id,
                response_type=ResponseType.MEDIATION,
                severity=ResponseSeverity.CORRECTIVE,
                status=ResponseStatus.PROPOSED,
                description="Mediated discussion to address underlying issues"
            ))
        
        return responses

    # Additional helper methods would be implemented here for:
    # - _generate_moderate_violation_responses
    # - _generate_severe_violation_responses  
    # - _generate_critical_violation_responses
    # - _generate_restorative_response
    # - Various other implementation details

    async def _store_response_update(self, response: ViolationResponse) -> None:
        """Store response update in Store API."""
        try:
            if self.store_integration.store:
                response_data = asdict(response)
                response_data["proposed_at"] = response.proposed_at.isoformat()
                response_data["implemented_at"] = response.implemented_at.isoformat() if response.implemented_at else None
                response_data["completed_at"] = response.completed_at.isoformat() if response.completed_at else None
                response_data["response_type"] = response.response_type.value
                response_data["severity"] = response.severity.value
                response_data["status"] = response.status.value
                
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"violation_response_{response.response_id}",
                    response_data
                )
                
        except Exception as e:
            self.logger.error(f"Failed to store response update: {str(e)}")

    # Placeholder implementations for remaining methods
    async def _generate_moderate_violation_responses(self, violation, history):
        return []
    async def _generate_severe_violation_responses(self, violation, history):
        return []
    async def _generate_critical_violation_responses(self, violation, history):
        return []
    async def _generate_restorative_response(self, violation, history):
        return None
    async def _rank_response_proposals(self, proposals, violation, sentiment):
        return proposals
    async def _get_eligible_response_voters(self):
        return {f"agent_{i}" for i in range(20)}
    async def _implement_mediation(self, response):
        return True
    async def _implement_temporary_restriction(self, response):
        return True
    async def _implement_rehabilitation(self, response):
        return True
    async def _implement_restitution(self, response):
        return True
    async def _implement_restorative_justice(self, response):
        return True
    async def _implement_suspension(self, response):
        return True
    async def _notify_agent_of_response(self, agent_id, message):
        pass
    async def _notify_community_of_response_vote(self, violation_id, session_id):
        pass
    async def _notify_response_implementation(self, response, success):
        pass
    async def _store_proposed_response(self, response, session_id):
        pass
    async def _identify_affected_parties(self, violation):
        return ["affected_agent_1", "affected_agent_2"]
    async def _find_restorative_facilitator(self):
        return "facilitator_agent"
    async def _store_restorative_process(self, process):
        pass
    async def _invite_restorative_participants(self, process):
        pass
    async def _conduct_restorative_phase(self, process, phase):
        return {"success": True, "outcome": f"Phase {phase} completed"}
    async def _develop_restorative_agreements(self, process, outcomes):
        return ["Make public apology", "Contribute to community project"]
    async def _collect_satisfaction_scores(self, process):
        return {agent: random.uniform(0.6, 1.0) for agent in process.affected_parties}
    async def _analyze_violation_patterns(self, history):
        return {"pattern": "social_conflict"}
    async def _generate_rehabilitation_goals(self, patterns):
        return ["Improve social skills", "Learn conflict resolution"]
    async def _design_rehabilitation_interventions(self, patterns):
        return [{"type": "counseling", "frequency": "weekly"}]
    async def _create_rehabilitation_milestones(self, patterns):
        return [{"milestone": "30_days_clean", "achieved": False}]
    async def _assign_support_agents(self, agent_id):
        return ["mentor_agent_1", "mentor_agent_2"]
    async def _store_rehabilitation_plan(self, plan):
        pass
    async def _notify_rehabilitation_plan_created(self, plan):
        pass
    async def _check_milestone_completion(self, milestone, agent_id):
        return random.choice([True, False])
    async def _notify_plan_created(self, plan):
        pass


# Helper functions
def create_violation_response_system(store_integration: MemoryStoreIntegration,
                                   compliance_monitor: ComplianceMonitor = None) -> ViolationResponseSystem:
    """Create a ViolationResponseSystem instance."""
    return ViolationResponseSystem(store_integration, compliance_monitor)


# Example usage
if __name__ == "__main__":
    async def test_violation_response():
        """Test the Violation Response System."""
        print("Testing Violation Response System...")
        
        # This would integrate with actual systems in production
        print("Violation Response System loaded successfully")
    
    asyncio.run(test_violation_response())