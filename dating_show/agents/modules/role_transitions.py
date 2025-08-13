"""
File: role_transitions.py
Description: Role Transition Management system for Task 3.3.
Handles smooth role changes, transition periods, identity crisis patterns, and resolution mechanisms.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid
import statistics
import time
import math

from .langgraph_base_module import LangGraphBaseModule, ModuleExecutionConfig, ExecutionTimeScale, ModulePriority
from ..enhanced_agent_state import EnhancedAgentState, EnhancedAgentStateManager, SpecializationData
from .identity_persistence import ProfessionalIdentity, IdentityDevelopmentPhase, IdentityStrength


class TransitionType(Enum):
    """Types of role transitions."""
    NATURAL_EVOLUTION = "natural_evolution"    # Gradual development into new role
    CRISIS_DRIVEN = "crisis_driven"           # Transition due to identity crisis
    OPPORTUNITY_BASED = "opportunity_based"    # Transition to seize new opportunities
    SOCIAL_PRESSURE = "social_pressure"       # Transition due to social expectations
    PERFORMANCE_DRIVEN = "performance_driven" # Transition due to poor role performance
    EXPLORATION = "exploration"               # Experimental role trying


class TransitionPhase(Enum):
    """Phases of role transition process."""
    PRE_TRANSITION = "pre_transition"         # Building up to transition
    INITIATION = "initiation"                # Starting the transition
    EXPLORATION = "exploration"              # Exploring new role possibilities
    COMMITMENT = "commitment"                # Committing to new role
    STABILIZATION = "stabilization"          # Stabilizing in new role
    INTEGRATION = "integration"              # Fully integrated into new role


class TransitionChallenge(Enum):
    """Challenges agents face during role transitions."""
    IDENTITY_CONFUSION = "identity_confusion"
    SKILL_GAPS = "skill_gaps"
    SOCIAL_RESISTANCE = "social_resistance"
    CONFIDENCE_LOSS = "confidence_loss"
    ROLE_CONFLICT = "role_conflict"
    PERFORMANCE_ANXIETY = "performance_anxiety"
    ISOLATION = "isolation"
    UNCERTAINTY = "uncertainty"


@dataclass
class TransitionTrigger:
    """Event or condition that triggers a role transition."""
    trigger_id: str
    trigger_type: str  # 'performance_decline', 'social_feedback', 'opportunity', 'crisis'
    description: str
    strength: float  # 0.0-1.0, how strong the trigger is
    detected_at: datetime
    related_events: List[str]
    agent_awareness: float  # How aware the agent is of this trigger
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionSupport:
    """Support mechanisms available during transition."""
    support_id: str
    support_type: str  # 'mentorship', 'social', 'skill_development', 'emotional'
    provider_agent_id: Optional[str]
    description: str
    effectiveness: float  # 0.0-1.0
    availability_period: Tuple[datetime, datetime]
    utilization_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoleTransitionRecord:
    """Complete record of a role transition process."""
    transition_id: str
    agent_id: str
    
    # Transition Context
    from_role: str
    to_role: str
    transition_type: TransitionType
    current_phase: TransitionPhase
    
    # Timeline
    initiated_at: datetime
    expected_duration: timedelta
    actual_duration: Optional[timedelta]
    completed_at: Optional[datetime]
    
    # Triggers and Catalysts
    primary_trigger: TransitionTrigger
    secondary_triggers: List[TransitionTrigger]
    catalyzing_events: List[str]
    
    # Transition Process
    confidence_trajectory: List[Tuple[datetime, float]]
    performance_trajectory: List[Tuple[datetime, float]]
    challenges_encountered: List[TransitionChallenge]
    support_systems: List[TransitionSupport]
    
    # Skills and Learning
    skills_before: Dict[str, float]
    skills_after: Dict[str, float]
    skill_transfer_rate: float
    learning_milestones: List[Dict[str, Any]]
    
    # Social Dynamics
    social_acceptance_rate: float
    relationship_changes: Dict[str, float]
    community_support_level: float
    resistance_encountered: List[str]
    
    # Outcomes
    transition_success_rate: float
    identity_coherence_change: float
    well_being_impact: float
    performance_improvement: float
    
    # Metadata
    created_at: datetime
    last_updated: datetime
    is_complete: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoleTransitionManager(LangGraphBaseModule):
    """
    Role Transition Management system for Task 3.3.
    Manages smooth role changes, transition periods, and identity crisis resolution.
    """
    
    def __init__(self, state_manager: Optional[EnhancedAgentStateManager] = None):
        """
        Initialize Role Transition Manager.
        
        Args:
            state_manager: Enhanced agent state manager
        """
        config = ModuleExecutionConfig(
            time_scale=ExecutionTimeScale.MEDIUM,  # Transitions need regular monitoring
            priority=ModulePriority.HIGH,          # Important for identity stability
            can_run_parallel=True,
            requires_completion=False,
            max_execution_time=3.0
        )
        
        super().__init__("role_transition_manager", config, state_manager)
        
        # Transition tracking
        self.active_transitions: Dict[str, RoleTransitionRecord] = {}
        self.completed_transitions: List[RoleTransitionRecord] = []
        self.transition_history_limit = 50
        
        # Transition settings
        self.transition_threshold = 0.6  # Confidence threshold to trigger transition
        self.crisis_threshold = 0.3      # Confidence threshold for crisis detection
        self.stabilization_period = timedelta(days=7)  # Time to stabilize in new role
        self.support_effectiveness_threshold = 0.5
        
        # Challenge detection and resolution
        self.challenge_detectors: Dict[TransitionChallenge, Callable] = {
            TransitionChallenge.IDENTITY_CONFUSION: self._detect_identity_confusion,
            TransitionChallenge.SKILL_GAPS: self._detect_skill_gaps,
            TransitionChallenge.SOCIAL_RESISTANCE: self._detect_social_resistance,
            TransitionChallenge.CONFIDENCE_LOSS: self._detect_confidence_loss,
            TransitionChallenge.ROLE_CONFLICT: self._detect_role_conflict
        }
        
        self.challenge_resolvers: Dict[TransitionChallenge, Callable] = {
            TransitionChallenge.IDENTITY_CONFUSION: self._resolve_identity_confusion,
            TransitionChallenge.SKILL_GAPS: self._resolve_skill_gaps,
            TransitionChallenge.SOCIAL_RESISTANCE: self._resolve_social_resistance,
            TransitionChallenge.CONFIDENCE_LOSS: self._resolve_confidence_loss,
            TransitionChallenge.ROLE_CONFLICT: self._resolve_role_conflict
        }
        
        # Performance tracking
        self.transition_stats = {
            "active_transitions": 0,
            "completed_transitions": 0,
            "avg_transition_duration_days": 0.0,
            "success_rate": 0.0,
            "challenge_resolution_rate": 0.0,
            "avg_confidence_improvement": 0.0
        }
        
        # State dependencies
        self.add_state_dependency("specialization")
        self.add_state_dependency("professional_identity")
        
        self.logger = logging.getLogger("RoleTransitionManager")
    
    def process_state(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Process agent state for role transition management.
        
        Args:
            state: Current enhanced agent state
        
        Returns:
            Dictionary with transition management results
        """
        start_time = time.time()
        
        try:
            agent_id = state.get("agent_id", "unknown")
            
            # Check for new transition triggers
            new_triggers = self._detect_transition_triggers(state)
            
            # Update active transitions
            transition_updates = self._update_active_transitions(state)
            
            # Detect and resolve transition challenges
            challenge_resolutions = self._manage_transition_challenges(state)
            
            # Check for transition completions
            completions = self._check_transition_completions(state)
            
            # Provide transition support
            support_actions = self._provide_transition_support(state)
            
            # Update performance statistics
            self._update_performance_stats()
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "state_changes": {
                    "transition_status": self._get_current_transition_status(agent_id)
                },
                "output_data": {
                    "transition_detection": {
                        "new_triggers": len(new_triggers),
                        "trigger_types": [t.trigger_type for t in new_triggers],
                        "strongest_trigger": max(new_triggers, key=lambda t: t.strength).trigger_type if new_triggers else None
                    },
                    "active_transitions": {
                        "count": len(self.active_transitions),
                        "phases": [t.current_phase.value for t in self.active_transitions.values()],
                        "avg_progress": self._calculate_avg_transition_progress()
                    },
                    "challenge_management": {
                        "challenges_detected": sum(len(r.challenges_encountered) for r in challenge_resolutions),
                        "resolutions_attempted": len(challenge_resolutions),
                        "support_actions": len(support_actions)
                    },
                    "completions": {
                        "transitions_completed": len(completions),
                        "success_rate": sum(1 for c in completions if c.transition_success_rate > 0.7) / len(completions) if completions else 0.0
                    }
                },
                "performance_metrics": {
                    "processing_time_ms": processing_time,
                    "transition_management_efficiency": self._calculate_management_efficiency(),
                    "active_transition_health": self._calculate_transition_health()
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error in role transition management: {str(e)}")
            return {
                "output_data": {"error": str(e)},
                "performance_metrics": {"processing_time_ms": (time.time() - start_time) * 1000}
            }
    
    def _detect_transition_triggers(self, state: EnhancedAgentState) -> List[TransitionTrigger]:
        """Detect conditions that might trigger a role transition."""
        triggers = []
        agent_id = state.get("agent_id", "unknown")
        specialization = state.get("specialization")
        professional_identity = state.get("professional_identity")
        
        if not specialization or not professional_identity:
            return triggers
        
        # Performance decline trigger
        if hasattr(specialization, 'role_consistency_score'):
            if specialization.role_consistency_score < self.crisis_threshold:
                trigger = TransitionTrigger(
                    trigger_id=str(uuid.uuid4()),
                    trigger_type="performance_decline",
                    description=f"Role performance declined to {specialization.role_consistency_score:.2f}",
                    strength=1.0 - specialization.role_consistency_score,
                    detected_at=datetime.now(),
                    related_events=["low_performance_metrics"],
                    agent_awareness=0.8,
                    metadata={"performance_score": specialization.role_consistency_score}
                )
                triggers.append(trigger)
        
        # Role confidence crisis
        if hasattr(professional_identity, 'role_confidence'):
            if professional_identity.role_confidence < self.crisis_threshold:
                trigger = TransitionTrigger(
                    trigger_id=str(uuid.uuid4()),
                    trigger_type="confidence_crisis",
                    description=f"Role confidence dropped to {professional_identity.role_confidence:.2f}",
                    strength=1.0 - professional_identity.role_confidence,
                    detected_at=datetime.now(),
                    related_events=["confidence_decline"],
                    agent_awareness=0.9,
                    metadata={"confidence_level": professional_identity.role_confidence}
                )
                triggers.append(trigger)
        
        # Identity development phase changes
        if hasattr(professional_identity, 'development_phase'):
            if professional_identity.development_phase == IdentityDevelopmentPhase.RENEWAL:
                trigger = TransitionTrigger(
                    trigger_id=str(uuid.uuid4()),
                    trigger_type="identity_renewal",
                    description="Identity development reached renewal phase",
                    strength=0.7,
                    detected_at=datetime.now(),
                    related_events=["development_phase_change"],
                    agent_awareness=0.6,
                    metadata={"development_phase": professional_identity.development_phase.value}
                )
                triggers.append(trigger)
        
        # Opportunity-based triggers (simplified)
        # In real implementation, would analyze social context for opportunities
        if len(triggers) == 0 and specialization.role_consistency_score > 0.8:
            # High performance might indicate readiness for growth
            if hasattr(professional_identity, 'role_confidence') and professional_identity.role_confidence > 0.8:
                trigger = TransitionTrigger(
                    trigger_id=str(uuid.uuid4()),
                    trigger_type="growth_opportunity",
                    description="High performance suggests readiness for role expansion",
                    strength=0.5,
                    detected_at=datetime.now(),
                    related_events=["high_performance"],
                    agent_awareness=0.4,
                    metadata={"performance_level": "high"}
                )
                triggers.append(trigger)
        
        return triggers
    
    def _update_active_transitions(self, state: EnhancedAgentState) -> List[Dict[str, Any]]:
        """Update progress of active transitions."""
        updates = []
        agent_id = state.get("agent_id", "unknown")
        
        # Check if agent should start a new transition
        if agent_id not in self.active_transitions:
            # Check for transition triggers
            triggers = self._detect_transition_triggers(state)
            strong_triggers = [t for t in triggers if t.strength > self.transition_threshold]
            
            if strong_triggers:
                # Start new transition
                strongest_trigger = max(strong_triggers, key=lambda t: t.strength)
                self._initiate_transition(state, strongest_trigger)
                updates.append({"action": "transition_initiated", "trigger": strongest_trigger.trigger_type})
        
        # Update existing transitions
        for transition_id, transition in self.active_transitions.items():
            if transition.agent_id == agent_id:
                # Update transition phase
                old_phase = transition.current_phase
                new_phase = self._determine_transition_phase(transition, state)
                
                if new_phase != old_phase:
                    transition.current_phase = new_phase
                    transition.last_updated = datetime.now()
                    updates.append({
                        "action": "phase_change",
                        "from_phase": old_phase.value,
                        "to_phase": new_phase.value
                    })
                
                # Update trajectories
                current_time = datetime.now()
                specialization = state.get("specialization")
                professional_identity = state.get("professional_identity")
                
                if specialization and hasattr(specialization, 'role_consistency_score'):
                    transition.performance_trajectory.append((current_time, specialization.role_consistency_score))
                
                if professional_identity and hasattr(professional_identity, 'role_confidence'):
                    transition.confidence_trajectory.append((current_time, professional_identity.role_confidence))
                
                # Limit trajectory length
                if len(transition.performance_trajectory) > 50:
                    transition.performance_trajectory = transition.performance_trajectory[-50:]
                if len(transition.confidence_trajectory) > 50:
                    transition.confidence_trajectory = transition.confidence_trajectory[-50:]
        
        return updates
    
    def _initiate_transition(self, state: EnhancedAgentState, trigger: TransitionTrigger) -> RoleTransitionRecord:
        """Initiate a new role transition."""
        agent_id = state.get("agent_id", "unknown")
        specialization = state.get("specialization")
        professional_identity = state.get("professional_identity")
        
        # Determine from and to roles
        from_role = "contestant"  # Default
        if professional_identity and hasattr(professional_identity, 'primary_role'):
            from_role = professional_identity.primary_role
        elif specialization and hasattr(specialization, 'current_role'):
            from_role = specialization.current_role
        
        # Determine target role based on trigger type and current performance
        to_role = self._determine_target_role(trigger, from_role, state)
        
        # Determine transition type
        transition_type_mapping = {
            "performance_decline": TransitionType.PERFORMANCE_DRIVEN,
            "confidence_crisis": TransitionType.CRISIS_DRIVEN,
            "identity_renewal": TransitionType.NATURAL_EVOLUTION,
            "growth_opportunity": TransitionType.OPPORTUNITY_BASED
        }
        transition_type = transition_type_mapping.get(trigger.trigger_type, TransitionType.NATURAL_EVOLUTION)
        
        # Create transition record
        transition = RoleTransitionRecord(
            transition_id=str(uuid.uuid4()),
            agent_id=agent_id,
            
            # Transition Context
            from_role=from_role,
            to_role=to_role,
            transition_type=transition_type,
            current_phase=TransitionPhase.INITIATION,
            
            # Timeline
            initiated_at=datetime.now(),
            expected_duration=self._estimate_transition_duration(transition_type, from_role, to_role),
            actual_duration=None,
            completed_at=None,
            
            # Triggers and Catalysts
            primary_trigger=trigger,
            secondary_triggers=[],
            catalyzing_events=[],
            
            # Transition Process
            confidence_trajectory=[(datetime.now(), professional_identity.role_confidence if professional_identity else 0.5)],
            performance_trajectory=[(datetime.now(), specialization.role_consistency_score if specialization else 0.5)],
            challenges_encountered=[],
            support_systems=[],
            
            # Skills and Learning
            skills_before=specialization.skills.copy() if specialization and hasattr(specialization, 'skills') else {},
            skills_after={},
            skill_transfer_rate=0.0,
            learning_milestones=[],
            
            # Social Dynamics
            social_acceptance_rate=0.5,
            relationship_changes={},
            community_support_level=0.5,
            resistance_encountered=[],
            
            # Outcomes
            transition_success_rate=0.0,
            identity_coherence_change=0.0,
            well_being_impact=0.0,
            performance_improvement=0.0,
            
            # Metadata
            created_at=datetime.now(),
            last_updated=datetime.now(),
            is_complete=False,
            metadata={"initial_trigger": trigger.trigger_type}
        )
        
        # Add to active transitions
        self.active_transitions[agent_id] = transition
        
        self.logger.info(f"Initiated transition for {agent_id}: {from_role} -> {to_role} ({transition_type.value})")
        
        return transition
    
    def _determine_target_role(self, trigger: TransitionTrigger, from_role: str, 
                              state: EnhancedAgentState) -> str:
        """Determine target role for transition based on trigger and context."""
        
        # Role progression paths (simplified)
        role_progressions = {
            "contestant": ["social_connector", "entertainment_leader", "romantic_pursuer"],
            "social_connector": ["conflict_resolver", "emotional_support", "entertainment_leader"],
            "conflict_resolver": ["strategist", "independent", "emotional_support"],
            "emotional_support": ["confidant", "social_connector", "conflict_resolver"],
            "entertainment_leader": ["social_connector", "romantic_pursuer", "independent"],
            "strategist": ["observer", "independent", "conflict_resolver"],
            "confidant": ["emotional_support", "observer", "social_connector"],
            "romantic_pursuer": ["social_connector", "independent", "entertainment_leader"],
            "observer": ["strategist", "independent", "conflict_resolver"],
            "independent": ["observer", "strategist", "contestant"]
        }
        
        possible_roles = role_progressions.get(from_role, ["social_connector"])
        
        # Choose based on trigger type
        if trigger.trigger_type == "performance_decline":
            # Move to easier or more supportive role
            supportive_roles = ["emotional_support", "social_connector", "confidant"]
            candidates = [role for role in possible_roles if role in supportive_roles]
            return candidates[0] if candidates else possible_roles[0]
        
        elif trigger.trigger_type == "growth_opportunity":
            # Move to more challenging role
            challenging_roles = ["strategist", "conflict_resolver", "entertainment_leader"]
            candidates = [role for role in possible_roles if role in challenging_roles]
            return candidates[0] if candidates else possible_roles[0]
        
        else:
            # Natural progression - choose first available
            return possible_roles[0]
    
    def _estimate_transition_duration(self, transition_type: TransitionType, 
                                     from_role: str, to_role: str) -> timedelta:
        """Estimate how long a transition should take."""
        
        base_duration_days = {
            TransitionType.NATURAL_EVOLUTION: 14,
            TransitionType.CRISIS_DRIVEN: 21,
            TransitionType.OPPORTUNITY_BASED: 10,
            TransitionType.SOCIAL_PRESSURE: 7,
            TransitionType.PERFORMANCE_DRIVEN: 21,
            TransitionType.EXPLORATION: 3
        }
        
        # Role similarity affects transition time
        similar_roles = {
            "social_connector": ["entertainment_leader", "conflict_resolver"],
            "emotional_support": ["confidant", "social_connector"],
            "strategist": ["observer", "conflict_resolver"],
            "romantic_pursuer": ["social_connector", "entertainment_leader"]
        }
        
        base_days = base_duration_days.get(transition_type, 14)
        
        # Reduce time for similar roles
        if to_role in similar_roles.get(from_role, []):
            base_days = int(base_days * 0.7)
        
        return timedelta(days=base_days)
    
    def _determine_transition_phase(self, transition: RoleTransitionRecord, 
                                   state: EnhancedAgentState) -> TransitionPhase:
        """Determine current phase of transition."""
        
        days_since_start = (datetime.now() - transition.initiated_at).days
        expected_days = transition.expected_duration.days
        progress_ratio = days_since_start / max(expected_days, 1)
        
        # Get recent confidence trend
        recent_confidence = [point[1] for point in transition.confidence_trajectory[-5:]]
        confidence_trend = "stable"
        if len(recent_confidence) >= 3:
            if recent_confidence[-1] > recent_confidence[0] + 0.1:
                confidence_trend = "improving"
            elif recent_confidence[-1] < recent_confidence[0] - 0.1:
                confidence_trend = "declining"
        
        # Phase determination logic
        if progress_ratio < 0.2:
            return TransitionPhase.INITIATION
        elif progress_ratio < 0.4:
            return TransitionPhase.EXPLORATION
        elif progress_ratio < 0.6:
            if confidence_trend == "improving":
                return TransitionPhase.COMMITMENT
            else:
                return TransitionPhase.EXPLORATION  # Stay in exploration if uncertain
        elif progress_ratio < 0.8:
            return TransitionPhase.STABILIZATION
        else:
            return TransitionPhase.INTEGRATION
    
    def _manage_transition_challenges(self, state: EnhancedAgentState) -> List[RoleTransitionRecord]:
        """Detect and resolve transition challenges."""
        resolved_transitions = []
        agent_id = state.get("agent_id", "unknown")
        
        if agent_id in self.active_transitions:
            transition = self.active_transitions[agent_id]
            
            # Detect challenges
            for challenge_type, detector in self.challenge_detectors.items():
                if detector(transition, state):
                    if challenge_type not in transition.challenges_encountered:
                        transition.challenges_encountered.append(challenge_type)
                        self.logger.info(f"Challenge detected for {agent_id}: {challenge_type.value}")
            
            # Resolve challenges
            for challenge in transition.challenges_encountered:
                if challenge in self.challenge_resolvers:
                    resolution_success = self.challenge_resolvers[challenge](transition, state)
                    if resolution_success:
                        resolved_transitions.append(transition)
        
        return resolved_transitions
    
    def _detect_identity_confusion(self, transition: RoleTransitionRecord, 
                                  state: EnhancedAgentState) -> bool:
        """Detect identity confusion challenge."""
        professional_identity = state.get("professional_identity")
        if professional_identity and hasattr(professional_identity, 'internal_coherence'):
            return professional_identity.internal_coherence < 0.4
        return False
    
    def _detect_skill_gaps(self, transition: RoleTransitionRecord, 
                          state: EnhancedAgentState) -> bool:
        """Detect skill gaps challenge."""
        specialization = state.get("specialization")
        if specialization and hasattr(specialization, 'expertise_level'):
            return specialization.expertise_level < 0.3
        return False
    
    def _detect_social_resistance(self, transition: RoleTransitionRecord, 
                                 state: EnhancedAgentState) -> bool:
        """Detect social resistance challenge."""
        # Simplified detection - would analyze social feedback
        return transition.social_acceptance_rate < 0.4
    
    def _detect_confidence_loss(self, transition: RoleTransitionRecord, 
                               state: EnhancedAgentState) -> bool:
        """Detect confidence loss challenge."""
        if len(transition.confidence_trajectory) >= 3:
            recent_confidence = [point[1] for point in transition.confidence_trajectory[-3:]]
            return recent_confidence[-1] < recent_confidence[0] - 0.2
        return False
    
    def _detect_role_conflict(self, transition: RoleTransitionRecord, 
                             state: EnhancedAgentState) -> bool:
        """Detect role conflict challenge."""
        professional_identity = state.get("professional_identity")
        if professional_identity and hasattr(professional_identity, 'value_alignment'):
            return professional_identity.value_alignment < 0.4
        return False
    
    def _resolve_identity_confusion(self, transition: RoleTransitionRecord, 
                                   state: EnhancedAgentState) -> bool:
        """Resolve identity confusion through clarification."""
        # Add identity clarification support
        support = TransitionSupport(
            support_id=str(uuid.uuid4()),
            support_type="identity_clarification",
            provider_agent_id=None,
            description="Self-reflection and role clarity exercises",
            effectiveness=0.7,
            availability_period=(datetime.now(), datetime.now() + timedelta(days=7)),
            utilization_rate=0.0,
            metadata={"intervention_type": "self_guided"}
        )
        transition.support_systems.append(support)
        return True
    
    def _resolve_skill_gaps(self, transition: RoleTransitionRecord, 
                           state: EnhancedAgentState) -> bool:
        """Resolve skill gaps through development."""
        support = TransitionSupport(
            support_id=str(uuid.uuid4()),
            support_type="skill_development",
            provider_agent_id=None,
            description="Targeted skill building for new role",
            effectiveness=0.8,
            availability_period=(datetime.now(), datetime.now() + timedelta(days=14)),
            utilization_rate=0.0,
            metadata={"focus_area": "role_specific_skills"}
        )
        transition.support_systems.append(support)
        return True
    
    def _resolve_social_resistance(self, transition: RoleTransitionRecord, 
                                  state: EnhancedAgentState) -> bool:
        """Resolve social resistance through community engagement."""
        support = TransitionSupport(
            support_id=str(uuid.uuid4()),
            support_type="social_integration",
            provider_agent_id=None,
            description="Community relationship building",
            effectiveness=0.6,
            availability_period=(datetime.now(), datetime.now() + timedelta(days=10)),
            utilization_rate=0.0,
            metadata={"strategy": "gradual_acceptance"}
        )
        transition.support_systems.append(support)
        transition.social_acceptance_rate = min(1.0, transition.social_acceptance_rate + 0.2)
        return True
    
    def _resolve_confidence_loss(self, transition: RoleTransitionRecord, 
                                state: EnhancedAgentState) -> bool:
        """Resolve confidence loss through encouragement."""
        support = TransitionSupport(
            support_id=str(uuid.uuid4()),
            support_type="emotional_support",
            provider_agent_id=None,
            description="Confidence building and encouragement",
            effectiveness=0.7,
            availability_period=(datetime.now(), datetime.now() + timedelta(days=5)),
            utilization_rate=0.0,
            metadata={"focus": "self_confidence"}
        )
        transition.support_systems.append(support)
        return True
    
    def _resolve_role_conflict(self, transition: RoleTransitionRecord, 
                              state: EnhancedAgentState) -> bool:
        """Resolve role conflict through value alignment."""
        support = TransitionSupport(
            support_id=str(uuid.uuid4()),
            support_type="value_alignment",
            provider_agent_id=None,
            description="Role-value integration work",
            effectiveness=0.8,
            availability_period=(datetime.now(), datetime.now() + timedelta(days=7)),
            utilization_rate=0.0,
            metadata={"approach": "values_clarification"}
        )
        transition.support_systems.append(support)
        return True
    
    def _provide_transition_support(self, state: EnhancedAgentState) -> List[Dict[str, Any]]:
        """Provide ongoing support for active transitions."""
        support_actions = []
        agent_id = state.get("agent_id", "unknown")
        
        if agent_id in self.active_transitions:
            transition = self.active_transitions[agent_id]
            
            # Update support utilization
            for support in transition.support_systems:
                if support.utilization_rate < 1.0:
                    support.utilization_rate = min(1.0, support.utilization_rate + 0.1)
                    
                    # Apply support effects
                    if support.support_type == "skill_development":
                        support_actions.append({"type": "skill_boost", "effectiveness": support.effectiveness})
                    elif support.support_type == "emotional_support":
                        support_actions.append({"type": "confidence_boost", "effectiveness": support.effectiveness})
        
        return support_actions
    
    def _check_transition_completions(self, state: EnhancedAgentState) -> List[RoleTransitionRecord]:
        """Check if any transitions have completed."""
        completed = []
        agent_id = state.get("agent_id", "unknown")
        
        if agent_id in self.active_transitions:
            transition = self.active_transitions[agent_id]
            
            # Check completion criteria
            if transition.current_phase == TransitionPhase.INTEGRATION:
                days_in_integration = (datetime.now() - transition.initiated_at).days
                
                if days_in_integration >= transition.expected_duration.days:
                    # Complete transition
                    transition.completed_at = datetime.now()
                    transition.actual_duration = transition.completed_at - transition.initiated_at
                    transition.is_complete = True
                    
                    # Calculate final metrics
                    self._calculate_transition_outcomes(transition, state)
                    
                    # Move to completed list
                    completed.append(transition)
                    self.completed_transitions.append(transition)
                    del self.active_transitions[agent_id]
                    
                    # Limit completed history
                    if len(self.completed_transitions) > self.transition_history_limit:
                        self.completed_transitions = self.completed_transitions[-self.transition_history_limit:]
                    
                    self.logger.info(f"Completed transition for {agent_id}: {transition.from_role} -> {transition.to_role}")
        
        return completed
    
    def _calculate_transition_outcomes(self, transition: RoleTransitionRecord, 
                                      state: EnhancedAgentState) -> None:
        """Calculate final outcomes for completed transition."""
        
        # Success rate based on confidence improvement
        if transition.confidence_trajectory:
            initial_confidence = transition.confidence_trajectory[0][1]
            final_confidence = transition.confidence_trajectory[-1][1]
            confidence_improvement = final_confidence - initial_confidence
            transition.transition_success_rate = max(0.0, min(1.0, 0.5 + confidence_improvement))
        
        # Performance improvement
        if transition.performance_trajectory:
            initial_performance = transition.performance_trajectory[0][1]
            final_performance = transition.performance_trajectory[-1][1]
            transition.performance_improvement = final_performance - initial_performance
        
        # Identity coherence change
        professional_identity = state.get("professional_identity")
        if professional_identity and hasattr(professional_identity, 'internal_coherence'):
            transition.identity_coherence_change = professional_identity.internal_coherence - 0.5  # Baseline
        
        # Well-being impact (simplified)
        challenge_count = len(transition.challenges_encountered)
        support_count = len(transition.support_systems)
        transition.well_being_impact = max(-0.5, min(0.5, (support_count - challenge_count) * 0.1))
    
    def _get_current_transition_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get current transition status for agent."""
        if agent_id in self.active_transitions:
            transition = self.active_transitions[agent_id]
            return {
                "transition_id": transition.transition_id,
                "from_role": transition.from_role,
                "to_role": transition.to_role,
                "current_phase": transition.current_phase.value,
                "progress": (datetime.now() - transition.initiated_at).days / transition.expected_duration.days,
                "challenges": [c.value for c in transition.challenges_encountered],
                "support_systems": len(transition.support_systems)
            }
        return None
    
    def _calculate_avg_transition_progress(self) -> float:
        """Calculate average progress across all active transitions."""
        if not self.active_transitions:
            return 0.0
        
        total_progress = 0.0
        for transition in self.active_transitions.values():
            progress = (datetime.now() - transition.initiated_at).days / transition.expected_duration.days
            total_progress += min(1.0, progress)
        
        return total_progress / len(self.active_transitions)
    
    def _calculate_management_efficiency(self) -> float:
        """Calculate transition management efficiency."""
        total_transitions = len(self.completed_transitions)
        if total_transitions == 0:
            return 1.0
        
        successful_transitions = sum(1 for t in self.completed_transitions if t.transition_success_rate > 0.7)
        return successful_transitions / total_transitions
    
    def _calculate_transition_health(self) -> float:
        """Calculate health score of active transitions."""
        if not self.active_transitions:
            return 1.0
        
        health_scores = []
        for transition in self.active_transitions.values():
            # Fewer challenges = better health
            challenge_penalty = len(transition.challenges_encountered) * 0.1
            support_bonus = len(transition.support_systems) * 0.05
            health_score = max(0.0, 1.0 - challenge_penalty + support_bonus)
            health_scores.append(health_score)
        
        return statistics.mean(health_scores)
    
    def _update_performance_stats(self) -> None:
        """Update performance statistics."""
        self.transition_stats["active_transitions"] = len(self.active_transitions)
        self.transition_stats["completed_transitions"] = len(self.completed_transitions)
        
        if self.completed_transitions:
            # Average duration
            durations = [t.actual_duration.days for t in self.completed_transitions if t.actual_duration]
            if durations:
                self.transition_stats["avg_transition_duration_days"] = statistics.mean(durations)
            
            # Success rate
            successful = sum(1 for t in self.completed_transitions if t.transition_success_rate > 0.7)
            self.transition_stats["success_rate"] = successful / len(self.completed_transitions)
            
            # Confidence improvement
            improvements = [t.performance_improvement for t in self.completed_transitions if hasattr(t, 'performance_improvement')]
            if improvements:
                self.transition_stats["avg_confidence_improvement"] = statistics.mean(improvements)
    
    def get_transition_summary(self) -> Dict[str, Any]:
        """
        Get summary of role transition management.
        
        Returns:
            Transition management summary with metrics
        """
        return {
            "module_name": self.module_name,
            "transition_stats": self.transition_stats.copy(),
            "active_transitions": {
                transition_id: {
                    "agent_id": transition.agent_id,
                    "from_role": transition.from_role,
                    "to_role": transition.to_role,
                    "current_phase": transition.current_phase.value,
                    "progress": min(1.0, (datetime.now() - transition.initiated_at).days / transition.expected_duration.days),
                    "challenges": [c.value for c in transition.challenges_encountered]
                }
                for transition_id, transition in self.active_transitions.items()
            },
            "recent_completions": [
                {
                    "from_role": t.from_role,
                    "to_role": t.to_role,
                    "success_rate": t.transition_success_rate,
                    "duration_days": t.actual_duration.days if t.actual_duration else 0
                }
                for t in self.completed_transitions[-5:]  # Last 5 completions
            ]
        }


# Example usage and testing
if __name__ == "__main__":
    # Example of role transition manager usage
    from ..enhanced_agent_state import create_enhanced_agent_state
    
    # Create state manager with test data
    state_manager = create_enhanced_agent_state(
        "test_agent", "Test Agent", {"confidence": 0.3, "empathy": 0.9}  # Low confidence triggers transition
    )
    
    # Create role transition manager
    transition_manager = RoleTransitionManager(state_manager)
    
    print("Testing role transition management...")
    
    # Process state to manage transitions
    result = transition_manager(state_manager.state)
    
    print(f"Transition result: {result}")
    
    # Get transition summary
    summary = transition_manager.get_transition_summary()
    print(f"\nRole transition summary:")
    print(f"- Stats: {summary['transition_stats']}")
    print(f"- Active transitions: {len(summary['active_transitions'])}")
    print(f"- Recent completions: {len(summary['recent_completions'])}")
    
    print("Role transition management example completed!")