"""
File: identity_consistency.py
Description: Identity-Action Consistency validation module for Task 3.3.
Validates actions against professional identity, provides consistency scoring and feedback,
and implements identity-driven decision biasing for enhanced agent coherence.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
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


class ConsistencyLevel(Enum):
    """Levels of identity-action consistency."""
    HIGHLY_CONSISTENT = "highly_consistent"      # 0.8-1.0: Actions strongly align with identity
    MODERATELY_CONSISTENT = "moderately_consistent"  # 0.6-0.8: Actions mostly align with identity  
    SOMEWHAT_CONSISTENT = "somewhat_consistent"  # 0.4-0.6: Actions partially align with identity
    INCONSISTENT = "inconsistent"               # 0.2-0.4: Actions poorly align with identity
    HIGHLY_INCONSISTENT = "highly_inconsistent" # 0.0-0.2: Actions contradict identity


class InconsistencyType(Enum):
    """Types of identity-action inconsistencies."""
    ROLE_VIOLATION = "role_violation"           # Action contradicts primary role expectations
    VALUE_CONFLICT = "value_conflict"           # Action conflicts with agent's values
    SKILL_MISMATCH = "skill_mismatch"          # Action requires skills agent lacks
    CONTEXT_INAPPROPRIATE = "context_inappropriate"  # Action inappropriate for current context
    DEVELOPMENTAL_REGRESSION = "developmental_regression"  # Action regresses from development
    SOCIAL_EXPECTATION_BREACH = "social_expectation_breach"  # Action violates social expectations


class ActionCategory(Enum):
    """Categories of actions for consistency analysis."""
    COMMUNICATION = "communication"     # Talking, messaging, expressing
    SOCIAL_INTERACTION = "social_interaction"  # Group activities, bonding
    PROBLEM_SOLVING = "problem_solving"  # Decision making, conflict resolution
    EMOTIONAL_EXPRESSION = "emotional_expression"  # Showing emotions, empathy
    SKILL_DEMONSTRATION = "skill_demonstration"  # Using professional skills
    LEADERSHIP = "leadership"           # Leading, organizing, directing
    SUPPORT_GIVING = "support_giving"   # Helping, comforting, advising
    INDEPENDENCE = "independence"       # Solo activities, self-reliance
    ROMANTIC = "romantic"              # Dating, flirting, intimacy
    ENTERTAINMENT = "entertainment"     # Fun activities, jokes, games


@dataclass
class ActionEvaluation:
    """Evaluation of a single action against identity."""
    evaluation_id: str
    action_id: str
    action_description: str
    action_category: ActionCategory
    
    # Identity Context
    evaluated_against_role: str
    agent_development_phase: IdentityDevelopmentPhase
    agent_identity_strength: IdentityStrength
    
    # Consistency Analysis
    consistency_score: float  # 0.0-1.0
    consistency_level: ConsistencyLevel
    alignment_factors: Dict[str, float]  # What contributed to score
    
    # Inconsistency Detection
    inconsistencies_found: List[InconsistencyType]
    inconsistency_severity: float  # 0.0-1.0, how severe inconsistencies are
    
    # Context Analysis
    context_appropriateness: float  # 0.0-1.0
    social_expectations_met: bool
    skill_requirements_satisfied: bool
    
    # Learning and Feedback
    feedback_messages: List[str]
    improvement_suggestions: List[str]
    reinforcement_recommendations: List[str]
    
    # Metadata
    evaluated_at: datetime
    confidence_in_evaluation: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsistencyProfile:
    """Overall consistency profile for an agent."""
    agent_id: str
    profile_period: Tuple[datetime, datetime]
    
    # Overall Metrics
    overall_consistency_score: float
    consistency_trend: float  # -1.0 to 1.0, improving or declining
    consistency_stability: float  # 0.0-1.0, how stable consistency is
    
    # Category Breakdown
    category_consistency: Dict[ActionCategory, float]
    strongest_consistency_areas: List[ActionCategory]
    weakest_consistency_areas: List[ActionCategory]
    
    # Inconsistency Analysis
    common_inconsistencies: Dict[InconsistencyType, int]
    inconsistency_patterns: List[Dict[str, Any]]
    recurring_issues: List[str]
    
    # Development Tracking
    consistency_milestones: List[Dict[str, Any]]
    improvement_areas: List[str]
    reinforcement_successes: List[str]
    
    # Contextual Analysis
    context_adaptability: float  # How well agent adapts to different contexts
    social_calibration: float    # How well agent meets social expectations
    skill_application_accuracy: float  # How accurately agent applies skills
    
    # Metadata
    evaluation_count: int
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionBias:
    """Bias applied to decisions based on identity."""
    bias_id: str
    bias_type: str  # 'role_preference', 'value_alignment', 'skill_emphasis', 'social_expectation'
    bias_strength: float  # 0.0-1.0, how strongly to apply bias
    
    # Decision Modification
    preferred_actions: List[str]
    discouraged_actions: List[str]
    action_modifications: Dict[str, float]  # Action -> modification weight
    
    # Context Sensitivity  
    applies_in_contexts: List[str]
    context_sensitivity: float
    
    # Learning and Adaptation
    reinforcement_history: List[Tuple[datetime, float]]
    effectiveness_score: float
    last_applied: Optional[datetime]
    
    # Metadata
    created_at: datetime
    is_active: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class IdentityConsistencyModule(LangGraphBaseModule):
    """
    Identity-Action Consistency validation module for Task 3.3.
    Validates actions against professional identity and provides consistency feedback.
    """
    
    def __init__(self, state_manager: Optional[EnhancedAgentStateManager] = None):
        """
        Initialize Identity Consistency Module.
        
        Args:
            state_manager: Enhanced agent state manager
        """
        config = ModuleExecutionConfig(
            time_scale=ExecutionTimeScale.FAST,   # Needs to evaluate actions quickly
            priority=ModulePriority.HIGH,         # Important for agent coherence
            can_run_parallel=True,
            requires_completion=True,             # Consistency checks should complete
            max_execution_time=1.0
        )
        
        super().__init__("identity_consistency", config, state_manager)
        
        # Evaluation settings
        self.consistency_threshold = 0.6  # Minimum acceptable consistency
        self.evaluation_history_limit = 100
        self.profile_update_interval = timedelta(hours=6)
        self.bias_adaptation_rate = 0.1
        
        # Action evaluation data
        self.recent_evaluations: List[ActionEvaluation] = []
        self.consistency_profiles: Dict[str, ConsistencyProfile] = {}
        self.decision_biases: Dict[str, List[DecisionBias]] = {}
        
        # Role-specific expectations
        self.role_expectations = {
            "social_connector": {
                "preferred_actions": ["facilitate_conversation", "introduce_people", "organize_group_activity"],
                "discouraged_actions": ["isolate_self", "ignore_others", "create_conflict"],
                "required_skills": ["communication", "empathy", "social_awareness"],
                "context_adaptability": 0.8
            },
            "conflict_resolver": {
                "preferred_actions": ["mediate_dispute", "find_compromise", "calm_tensions"],
                "discouraged_actions": ["escalate_conflict", "take_sides", "avoid_problems"],
                "required_skills": ["negotiation", "diplomacy", "problem_solving"],
                "context_adaptability": 0.9
            },
            "emotional_support": {
                "preferred_actions": ["listen_actively", "offer_comfort", "provide_advice"],
                "discouraged_actions": ["dismiss_feelings", "be_judgmental", "avoid_emotional_topics"],
                "required_skills": ["empathy", "patience", "emotional_intelligence"],
                "context_adaptability": 0.7
            },
            "entertainment_leader": {
                "preferred_actions": ["organize_fun", "tell_jokes", "create_positive_atmosphere"],
                "discouraged_actions": ["be_serious_constantly", "dampen_mood", "refuse_participation"],
                "required_skills": ["creativity", "humor", "energy"],
                "context_adaptability": 0.6
            },
            "strategist": {
                "preferred_actions": ["analyze_situation", "make_plans", "think_ahead"],
                "discouraged_actions": ["act_impulsively", "ignore_consequences", "be_reactive"],
                "required_skills": ["analytical_thinking", "planning", "foresight"],
                "context_adaptability": 0.8
            },
            "confidant": {
                "preferred_actions": ["keep_secrets", "offer_private_counsel", "build_trust"],
                "discouraged_actions": ["gossip", "betray_trust", "be_unreliable"],
                "required_skills": ["trustworthiness", "discretion", "wisdom"],
                "context_adaptability": 0.7
            },
            "romantic_pursuer": {
                "preferred_actions": ["flirt_appropriately", "show_romantic_interest", "create_intimacy"],
                "discouraged_actions": ["be_unromantic", "ignore_attraction", "remain_platonic"],
                "required_skills": ["charm", "emotional_intelligence", "social_calibration"],
                "context_adaptability": 0.5
            },
            "observer": {
                "preferred_actions": ["watch_carefully", "analyze_behavior", "stay_objective"],
                "discouraged_actions": ["jump_to_conclusions", "be_overly_involved", "ignore_details"],
                "required_skills": ["observation", "analysis", "patience"],
                "context_adaptability": 0.9
            },
            "independent": {
                "preferred_actions": ["self_reliance", "individual_activities", "autonomous_decisions"],
                "discouraged_actions": ["depend_heavily_on_others", "always_follow_crowd", "seek_constant_approval"],
                "required_skills": ["self_sufficiency", "decision_making", "confidence"],
                "context_adaptability": 0.4
            }
        }
        
        # Performance tracking
        self.consistency_stats = {
            "evaluations_performed": 0,
            "avg_consistency_score": 0.0,
            "consistency_improvement_rate": 0.0,
            "bias_applications": 0,
            "feedback_effectiveness": 0.0,
            "last_profile_update": None
        }
        
        # State dependencies
        self.add_state_dependency("specialization")
        self.add_state_dependency("professional_identity")
        
        self.logger = logging.getLogger("IdentityConsistency")
    
    def process_state(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Process agent state for identity-action consistency validation.
        
        Args:
            state: Current enhanced agent state
        
        Returns:
            Dictionary with consistency validation results
        """
        start_time = time.time()
        
        try:
            agent_id = state.get("agent_id", "unknown")
            
            # Get recent actions for evaluation
            recent_actions = self._get_recent_actions(state)
            
            # Evaluate actions against identity
            action_evaluations = self._evaluate_actions_consistency(recent_actions, state)
            
            # Update consistency profile
            profile_updates = self._update_consistency_profile(agent_id, action_evaluations)
            
            # Generate feedback and recommendations
            feedback = self._generate_consistency_feedback(action_evaluations, state)
            
            # Update decision biases
            bias_updates = self._update_decision_biases(agent_id, action_evaluations, state)
            
            # Apply identity-driven decision biasing
            decision_modifications = self._apply_decision_biasing(state)
            
            # Update performance statistics
            self._update_performance_stats(action_evaluations)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "state_changes": {
                    "consistency_profile": self.consistency_profiles.get(agent_id),
                    "decision_biases": decision_modifications
                },
                "output_data": {
                    "consistency_evaluation": {
                        "actions_evaluated": len(action_evaluations),
                        "avg_consistency_score": statistics.mean([e.consistency_score for e in action_evaluations]) if action_evaluations else 0.0,
                        "consistency_level": self._determine_overall_consistency_level(action_evaluations),
                        "inconsistencies_found": sum(len(e.inconsistencies_found) for e in action_evaluations)
                    },
                    "feedback": {
                        "improvement_suggestions": feedback.get("improvements", []),
                        "reinforcement_recommendations": feedback.get("reinforcements", []),
                        "consistency_strengths": feedback.get("strengths", []),
                        "areas_for_development": feedback.get("development_areas", [])
                    },
                    "decision_biasing": {
                        "biases_active": len(decision_modifications.get("active_biases", [])),
                        "decision_modifications": len(decision_modifications.get("modifications", {})),
                        "bias_strength": decision_modifications.get("overall_bias_strength", 0.0)
                    }
                },
                "performance_metrics": {
                    "processing_time_ms": processing_time,
                    "consistency_validation_efficiency": self._calculate_validation_efficiency(),
                    "identity_coherence_score": self._calculate_identity_coherence_score(action_evaluations)
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error in identity consistency validation: {str(e)}")
            return {
                "output_data": {"error": str(e)},
                "performance_metrics": {"processing_time_ms": (time.time() - start_time) * 1000}
            }
    
    def _get_recent_actions(self, state: EnhancedAgentState) -> List[Dict[str, Any]]:
        """Get recent actions from agent memory systems."""
        recent_actions = []
        cutoff_time = datetime.now() - timedelta(hours=2)  # Last 2 hours
        
        if not self.state_manager:
            return recent_actions
        
        try:
            # From working memory
            working_memories = self.state_manager.circular_buffer.get_recent_memories(10)
            for memory in working_memories:
                if (memory.get("type") == "action" and 
                    memory.get("timestamp", datetime.now()) > cutoff_time):
                    recent_actions.append({
                        "id": str(uuid.uuid4()),
                        "description": memory.get("content", ""),
                        "category": self._categorize_action(memory.get("content", "")),
                        "timestamp": memory.get("timestamp"),
                        "importance": memory.get("importance", 0.5),
                        "context": memory.get("metadata", {}),
                        "source": "working_memory"
                    })
            
            # From temporal memory
            temporal_memories = self.state_manager.temporal_memory.retrieve_recent_memories(
                hours_back=2, memory_type="action", limit=20
            )
            for memory in temporal_memories:
                if memory.get("timestamp", datetime.now()) > cutoff_time:
                    recent_actions.append({
                        "id": str(uuid.uuid4()),
                        "description": memory.get("content", ""),
                        "category": self._categorize_action(memory.get("content", "")),
                        "timestamp": memory.get("timestamp"),
                        "importance": memory.get("importance", 0.5),
                        "context": memory.get("context", {}),
                        "source": "temporal_memory"
                    })
        
        except Exception as e:
            self.logger.error(f"Error getting recent actions: {str(e)}")
        
        return recent_actions
    
    def _categorize_action(self, action_description: str) -> ActionCategory:
        """Categorize action based on its description."""
        description_lower = action_description.lower()
        
        # Keyword mapping for action categories
        category_keywords = {
            ActionCategory.COMMUNICATION: ["talk", "say", "tell", "communicate", "speak", "discuss"],
            ActionCategory.SOCIAL_INTERACTION: ["group", "together", "socialize", "mingle", "interact"],
            ActionCategory.PROBLEM_SOLVING: ["solve", "resolve", "decide", "analyze", "figure out"],
            ActionCategory.EMOTIONAL_EXPRESSION: ["feel", "emotion", "happy", "sad", "angry", "express"],
            ActionCategory.SKILL_DEMONSTRATION: ["demonstrate", "show", "skill", "expertise", "ability"],
            ActionCategory.LEADERSHIP: ["lead", "direct", "organize", "manage", "guide"],
            ActionCategory.SUPPORT_GIVING: ["help", "support", "assist", "comfort", "encourage"],
            ActionCategory.INDEPENDENCE: ["alone", "solo", "independent", "self", "individual"],
            ActionCategory.ROMANTIC: ["flirt", "romantic", "date", "kiss", "intimate"],
            ActionCategory.ENTERTAINMENT: ["fun", "joke", "play", "entertain", "laugh"]
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            category_scores[category] = score
        
        # Return category with highest score, default to COMMUNICATION
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                return best_category[0]
        
        return ActionCategory.COMMUNICATION  # Default category
    
    def _evaluate_actions_consistency(self, actions: List[Dict[str, Any]], 
                                     state: EnhancedAgentState) -> List[ActionEvaluation]:
        """Evaluate consistency of actions against professional identity."""
        evaluations = []
        
        professional_identity = state.get("professional_identity")
        specialization = state.get("specialization")
        
        if not professional_identity:
            return evaluations
        
        current_role = getattr(professional_identity, 'primary_role', 'contestant')
        development_phase = getattr(professional_identity, 'development_phase', IdentityDevelopmentPhase.EXPLORATION)
        identity_strength = getattr(professional_identity, 'identity_strength', IdentityStrength.WEAK)
        
        role_expectations = self.role_expectations.get(current_role, {})
        
        for action in actions:
            evaluation = self._evaluate_single_action(
                action, current_role, development_phase, identity_strength, 
                role_expectations, specialization
            )
            evaluations.append(evaluation)
            self.recent_evaluations.append(evaluation)
        
        # Limit evaluation history
        if len(self.recent_evaluations) > self.evaluation_history_limit:
            self.recent_evaluations = self.recent_evaluations[-self.evaluation_history_limit:]
        
        return evaluations
    
    def _evaluate_single_action(self, action: Dict[str, Any], role: str, 
                               development_phase: IdentityDevelopmentPhase,
                               identity_strength: IdentityStrength,
                               role_expectations: Dict[str, Any],
                               specialization: Optional[SpecializationData]) -> ActionEvaluation:
        """Evaluate consistency of a single action."""
        
        action_description = action.get("description", "")
        action_category = action.get("category", ActionCategory.COMMUNICATION)
        
        # Calculate base consistency score
        consistency_score = self._calculate_base_consistency_score(
            action_description, action_category, role_expectations
        )
        
        # Adjust for development phase
        consistency_score = self._adjust_for_development_phase(
            consistency_score, development_phase
        )
        
        # Adjust for identity strength
        consistency_score = self._adjust_for_identity_strength(
            consistency_score, identity_strength
        )
        
        # Detect inconsistencies
        inconsistencies = self._detect_inconsistencies(
            action, role_expectations, specialization
        )
        
        # Calculate alignment factors
        alignment_factors = self._analyze_alignment_factors(
            action, role_expectations, specialization
        )
        
        # Generate feedback
        feedback_messages = self._generate_action_feedback(
            action, consistency_score, inconsistencies, role
        )
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            action, inconsistencies, role_expectations
        )
        
        # Generate reinforcement recommendations
        reinforcement_recommendations = self._generate_reinforcement_recommendations(
            action, consistency_score, alignment_factors
        )
        
        return ActionEvaluation(
            evaluation_id=str(uuid.uuid4()),
            action_id=action.get("id", str(uuid.uuid4())),
            action_description=action_description,
            action_category=action_category,
            
            # Identity Context
            evaluated_against_role=role,
            agent_development_phase=development_phase,
            agent_identity_strength=identity_strength,
            
            # Consistency Analysis
            consistency_score=consistency_score,
            consistency_level=self._determine_consistency_level(consistency_score),
            alignment_factors=alignment_factors,
            
            # Inconsistency Detection
            inconsistencies_found=inconsistencies,
            inconsistency_severity=self._calculate_inconsistency_severity(inconsistencies),
            
            # Context Analysis
            context_appropriateness=self._evaluate_context_appropriateness(action, role_expectations),
            social_expectations_met=self._check_social_expectations(action, role),
            skill_requirements_satisfied=self._check_skill_requirements(action, specialization),
            
            # Learning and Feedback
            feedback_messages=feedback_messages,
            improvement_suggestions=improvement_suggestions,
            reinforcement_recommendations=reinforcement_recommendations,
            
            # Metadata
            evaluated_at=datetime.now(),
            confidence_in_evaluation=0.8,  # Could be calculated based on data quality
            metadata={
                "role_expectations_used": bool(role_expectations),
                "specialization_available": specialization is not None
            }
        )
    
    def _calculate_base_consistency_score(self, action_description: str, 
                                         action_category: ActionCategory,
                                         role_expectations: Dict[str, Any]) -> float:
        """Calculate base consistency score for action."""
        score = 0.5  # Neutral baseline
        
        if not role_expectations:
            return score
        
        preferred_actions = role_expectations.get("preferred_actions", [])
        discouraged_actions = role_expectations.get("discouraged_actions", [])
        
        # Check for preferred action matches
        for preferred in preferred_actions:
            if any(word in action_description.lower() for word in preferred.split("_")):
                score += 0.2
        
        # Check for discouraged action matches
        for discouraged in discouraged_actions:
            if any(word in action_description.lower() for word in discouraged.split("_")):
                score -= 0.3
        
        # Category-specific adjustments
        category_bonuses = {
            ActionCategory.SOCIAL_INTERACTION: 0.1 if "social" in str(preferred_actions) else 0.0,
            ActionCategory.PROBLEM_SOLVING: 0.1 if "resolve" in str(preferred_actions) else 0.0,
            ActionCategory.SUPPORT_GIVING: 0.1 if "support" in str(preferred_actions) else 0.0,
            ActionCategory.LEADERSHIP: 0.1 if "lead" in str(preferred_actions) else 0.0
        }
        
        score += category_bonuses.get(action_category, 0.0)
        
        return max(0.0, min(1.0, score))
    
    def _adjust_for_development_phase(self, score: float, 
                                     phase: IdentityDevelopmentPhase) -> float:
        """Adjust consistency score based on development phase."""
        
        # More lenient scoring for early development phases
        phase_adjustments = {
            IdentityDevelopmentPhase.EXPLORATION: 0.1,    # More forgiving
            IdentityDevelopmentPhase.COMMITMENT: 0.0,     # Neutral
            IdentityDevelopmentPhase.SYNTHESIS: -0.05,    # Slightly more strict
            IdentityDevelopmentPhase.MASTERY: -0.1,       # More strict expectation
            IdentityDevelopmentPhase.RENEWAL: 0.05        # Slightly more forgiving during transition
        }
        
        adjustment = phase_adjustments.get(phase, 0.0)
        return max(0.0, min(1.0, score + adjustment))
    
    def _adjust_for_identity_strength(self, score: float, 
                                     strength: IdentityStrength) -> float:
        """Adjust consistency score based on identity strength."""
        
        # Stronger identities should have higher consistency expectations
        strength_adjustments = {
            IdentityStrength.WEAK: 0.15,          # Very forgiving
            IdentityStrength.MODERATE: 0.05,      # Slightly forgiving
            IdentityStrength.STRONG: -0.05,       # Slightly more strict
            IdentityStrength.VERY_STRONG: -0.1    # More strict
        }
        
        adjustment = strength_adjustments.get(strength, 0.0)
        return max(0.0, min(1.0, score + adjustment))
    
    def _detect_inconsistencies(self, action: Dict[str, Any], 
                               role_expectations: Dict[str, Any],
                               specialization: Optional[SpecializationData]) -> List[InconsistencyType]:
        """Detect specific types of inconsistencies."""
        inconsistencies = []
        action_description = action.get("description", "").lower()
        
        # Role violation detection
        discouraged_actions = role_expectations.get("discouraged_actions", [])
        for discouraged in discouraged_actions:
            if any(word in action_description for word in discouraged.split("_")):
                inconsistencies.append(InconsistencyType.ROLE_VIOLATION)
                break
        
        # Skill mismatch detection
        required_skills = role_expectations.get("required_skills", [])
        if specialization and hasattr(specialization, 'skills'):
            agent_skills = specialization.skills
            for required_skill in required_skills:
                if agent_skills.get(required_skill, 0.0) < 0.3:  # Low skill level
                    if any(word in action_description for word in required_skill.split("_")):
                        inconsistencies.append(InconsistencyType.SKILL_MISMATCH)
                        break
        
        # Context inappropriateness (simplified check)
        context = action.get("context", {})
        if context.get("formal_setting", False) and "joke" in action_description:
            inconsistencies.append(InconsistencyType.CONTEXT_INAPPROPRIATE)
        
        # Remove duplicates
        return list(set(inconsistencies))
    
    def _analyze_alignment_factors(self, action: Dict[str, Any],
                                  role_expectations: Dict[str, Any],
                                  specialization: Optional[SpecializationData]) -> Dict[str, float]:
        """Analyze factors that contribute to or detract from alignment."""
        
        factors = {
            "role_preference_alignment": 0.0,
            "skill_utilization": 0.0,
            "context_appropriateness": 0.0,
            "social_expectation_fulfillment": 0.0,
            "value_consistency": 0.0
        }
        
        action_description = action.get("description", "").lower()
        
        # Role preference alignment
        preferred_actions = role_expectations.get("preferred_actions", [])
        for preferred in preferred_actions:
            if any(word in action_description for word in preferred.split("_")):
                factors["role_preference_alignment"] = 0.8
                break
        
        # Skill utilization
        if specialization and hasattr(specialization, 'skills'):
            relevant_skills = []
            for skill, level in specialization.skills.items():
                if any(word in action_description for word in skill.split("_")):
                    relevant_skills.append(level)
            
            if relevant_skills:
                factors["skill_utilization"] = statistics.mean(relevant_skills)
        
        # Context appropriateness
        factors["context_appropriateness"] = self._evaluate_context_appropriateness(action, role_expectations)
        
        # Social expectation fulfillment
        factors["social_expectation_fulfillment"] = 0.7 if self._check_social_expectations(action, role_expectations.get("role", "")) else 0.3
        
        # Value consistency (simplified)
        factors["value_consistency"] = 0.6  # Would need value system to calculate properly
        
        return factors
    
    def _evaluate_context_appropriateness(self, action: Dict[str, Any],
                                         role_expectations: Dict[str, Any]) -> float:
        """Evaluate how appropriate action is for current context."""
        
        context = action.get("context", {})
        context_adaptability = role_expectations.get("context_adaptability", 0.5)
        
        # Simple context appropriateness check
        # In real implementation, would analyze specific contextual factors
        
        base_appropriateness = 0.7
        
        # Adjust based on role's context adaptability
        appropriateness = base_appropriateness * context_adaptability + (1 - context_adaptability) * 0.5
        
        return appropriateness
    
    def _check_social_expectations(self, action: Dict[str, Any], role: str) -> bool:
        """Check if action meets social expectations for the role."""
        
        # Simplified social expectation checking
        # In real implementation, would analyze community standards and feedback
        
        action_description = action.get("description", "").lower()
        
        # Basic social expectations by role
        social_expectations = {
            "social_connector": ["friendly", "inclusive", "engaging"],
            "conflict_resolver": ["fair", "calm", "diplomatic"],
            "emotional_support": ["caring", "understanding", "patient"],
            "entertainment_leader": ["fun", "energetic", "positive"],
            "strategist": ["thoughtful", "analytical", "careful"],
            "confidant": ["trustworthy", "discreet", "reliable"],
            "romantic_pursuer": ["charming", "attentive", "romantic"],
            "observer": ["quiet", "attentive", "analytical"],
            "independent": ["self_reliant", "decisive", "autonomous"]
        }
        
        expected_qualities = social_expectations.get(role, [])
        return any(quality in action_description for quality in expected_qualities)
    
    def _check_skill_requirements(self, action: Dict[str, Any],
                                 specialization: Optional[SpecializationData]) -> bool:
        """Check if agent has skills required for action."""
        
        if not specialization or not hasattr(specialization, 'skills'):
            return True  # Can't verify, assume satisfied
        
        action_description = action.get("description", "").lower()
        agent_skills = specialization.skills
        
        # Map actions to required skills (simplified)
        skill_requirements = {
            "mediate": ["negotiation", "diplomacy"],
            "comfort": ["empathy", "emotional_intelligence"],
            "analyze": ["analytical_thinking", "problem_solving"],
            "organize": ["leadership", "planning"],
            "joke": ["humor", "social_awareness"]
        }
        
        for action_word, required_skills in skill_requirements.items():
            if action_word in action_description:
                return all(agent_skills.get(skill, 0.0) >= 0.3 for skill in required_skills)
        
        return True  # No specific requirements identified
    
    def _calculate_inconsistency_severity(self, inconsistencies: List[InconsistencyType]) -> float:
        """Calculate overall severity of inconsistencies."""
        
        if not inconsistencies:
            return 0.0
        
        # Severity weights for different inconsistency types
        severity_weights = {
            InconsistencyType.ROLE_VIOLATION: 0.9,
            InconsistencyType.VALUE_CONFLICT: 0.8,
            InconsistencyType.SKILL_MISMATCH: 0.6,
            InconsistencyType.CONTEXT_INAPPROPRIATE: 0.5,
            InconsistencyType.DEVELOPMENTAL_REGRESSION: 0.7,
            InconsistencyType.SOCIAL_EXPECTATION_BREACH: 0.4
        }
        
        total_severity = sum(severity_weights.get(inc, 0.5) for inc in inconsistencies)
        return min(1.0, total_severity / 2.0)  # Normalize
    
    def _determine_consistency_level(self, score: float) -> ConsistencyLevel:
        """Determine consistency level based on score."""
        
        if score >= 0.8:
            return ConsistencyLevel.HIGHLY_CONSISTENT
        elif score >= 0.6:
            return ConsistencyLevel.MODERATELY_CONSISTENT
        elif score >= 0.4:
            return ConsistencyLevel.SOMEWHAT_CONSISTENT
        elif score >= 0.2:
            return ConsistencyLevel.INCONSISTENT
        else:
            return ConsistencyLevel.HIGHLY_INCONSISTENT
    
    def _generate_action_feedback(self, action: Dict[str, Any], 
                                 consistency_score: float,
                                 inconsistencies: List[InconsistencyType],
                                 role: str) -> List[str]:
        """Generate feedback messages for action."""
        
        feedback = []
        action_description = action.get("description", "")
        
        if consistency_score >= 0.8:
            feedback.append(f"Excellent! This action strongly aligns with your {role} role.")
        elif consistency_score >= 0.6:
            feedback.append(f"Good job! This action fits well with your {role} identity.")
        elif consistency_score >= 0.4:
            feedback.append(f"This action partially aligns with your {role} role, but could be improved.")
        else:
            feedback.append(f"This action doesn't align well with your {role} role.")
        
        # Specific inconsistency feedback
        for inconsistency in inconsistencies:
            if inconsistency == InconsistencyType.ROLE_VIOLATION:
                feedback.append(f"This action goes against typical {role} behavior expectations.")
            elif inconsistency == InconsistencyType.SKILL_MISMATCH:
                feedback.append("This action requires skills you haven't fully developed yet.")
            elif inconsistency == InconsistencyType.CONTEXT_INAPPROPRIATE:
                feedback.append("This action doesn't seem appropriate for the current situation.")
        
        return feedback
    
    def _generate_improvement_suggestions(self, action: Dict[str, Any],
                                         inconsistencies: List[InconsistencyType],
                                         role_expectations: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving action consistency."""
        
        suggestions = []
        preferred_actions = role_expectations.get("preferred_actions", [])
        
        if InconsistencyType.ROLE_VIOLATION in inconsistencies:
            if preferred_actions:
                suggestion = f"Try actions more aligned with your role, such as: {', '.join(preferred_actions[:2])}"
                suggestions.append(suggestion)
        
        if InconsistencyType.SKILL_MISMATCH in inconsistencies:
            suggestions.append("Consider developing the skills required for this type of action.")
        
        if InconsistencyType.CONTEXT_INAPPROPRIATE in inconsistencies:
            suggestions.append("Pay attention to the context and adjust your actions accordingly.")
        
        # General improvements
        if not suggestions:
            if preferred_actions:
                suggestions.append(f"Consider incorporating more {preferred_actions[0].replace('_', ' ')} into your interactions.")
        
        return suggestions
    
    def _generate_reinforcement_recommendations(self, action: Dict[str, Any],
                                               consistency_score: float,
                                               alignment_factors: Dict[str, float]) -> List[str]:
        """Generate recommendations for reinforcing positive behaviors."""
        
        recommendations = []
        
        if consistency_score >= 0.7:
            recommendations.append("Continue with this type of behavior - it suits your role well!")
            
            # Identify strongest alignment factors
            strong_factors = [(factor, score) for factor, score in alignment_factors.items() if score >= 0.7]
            
            for factor, score in strong_factors:
                if factor == "role_preference_alignment":
                    recommendations.append("Your role-appropriate choices are excellent.")
                elif factor == "skill_utilization":
                    recommendations.append("Great use of your skills in this situation.")
                elif factor == "social_expectation_fulfillment":
                    recommendations.append("You're meeting social expectations very well.")
        
        return recommendations
    
    def _update_consistency_profile(self, agent_id: str, 
                                   evaluations: List[ActionEvaluation]) -> Dict[str, Any]:
        """Update agent's consistency profile."""
        
        if not evaluations:
            return {}
        
        current_time = datetime.now()
        
        # Get or create profile
        if agent_id in self.consistency_profiles:
            profile = self.consistency_profiles[agent_id]
        else:
            profile = ConsistencyProfile(
                agent_id=agent_id,
                profile_period=(current_time - timedelta(days=7), current_time),
                overall_consistency_score=0.5,
                consistency_trend=0.0,
                consistency_stability=0.5,
                category_consistency={},
                strongest_consistency_areas=[],
                weakest_consistency_areas=[],
                common_inconsistencies={},
                inconsistency_patterns=[],
                recurring_issues=[],
                consistency_milestones=[],
                improvement_areas=[],
                reinforcement_successes=[],
                context_adaptability=0.5,
                social_calibration=0.5,
                skill_application_accuracy=0.5,
                evaluation_count=0,
                last_updated=current_time,
                metadata={}
            )
        
        # Update with new evaluations
        all_scores = [e.consistency_score for e in evaluations]
        profile.overall_consistency_score = statistics.mean(all_scores)
        
        # Update category consistency
        category_scores = {}
        for evaluation in evaluations:
            category = evaluation.action_category
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(evaluation.consistency_score)
        
        profile.category_consistency = {
            category: statistics.mean(scores) 
            for category, scores in category_scores.items()
        }
        
        # Identify strongest and weakest areas
        if profile.category_consistency:
            sorted_categories = sorted(profile.category_consistency.items(), key=lambda x: x[1], reverse=True)
            profile.strongest_consistency_areas = [cat for cat, score in sorted_categories[:3] if score > 0.6]
            profile.weakest_consistency_areas = [cat for cat, score in sorted_categories[-3:] if score < 0.5]
        
        # Update inconsistency tracking
        all_inconsistencies = []
        for evaluation in evaluations:
            all_inconsistencies.extend(evaluation.inconsistencies_found)
        
        inconsistency_counts = {}
        for inconsistency in all_inconsistencies:
            inconsistency_counts[inconsistency] = inconsistency_counts.get(inconsistency, 0) + 1
        
        profile.common_inconsistencies = inconsistency_counts
        
        # Update metadata
        profile.evaluation_count += len(evaluations)
        profile.last_updated = current_time
        
        self.consistency_profiles[agent_id] = profile
        
        return {"profile_updated": True, "new_evaluations": len(evaluations)}
    
    def _generate_consistency_feedback(self, evaluations: List[ActionEvaluation],
                                      state: EnhancedAgentState) -> Dict[str, List[str]]:
        """Generate comprehensive consistency feedback."""
        
        if not evaluations:
            return {"improvements": [], "reinforcements": [], "strengths": [], "development_areas": []}
        
        # Aggregate feedback from all evaluations
        all_improvements = []
        all_reinforcements = []
        
        for evaluation in evaluations:
            all_improvements.extend(evaluation.improvement_suggestions)
            all_reinforcements.extend(evaluation.reinforcement_recommendations)
        
        # Remove duplicates and limit
        unique_improvements = list(set(all_improvements))[:5]
        unique_reinforcements = list(set(all_reinforcements))[:5]
        
        # Identify consistency strengths
        strengths = []
        high_consistency_evaluations = [e for e in evaluations if e.consistency_score >= 0.8]
        if high_consistency_evaluations:
            strengths.append("You're showing strong identity-action alignment!")
            common_categories = [e.action_category for e in high_consistency_evaluations]
            most_common = max(set(common_categories), key=common_categories.count)
            strengths.append(f"Particularly strong in {most_common.value} actions")
        
        # Identify development areas
        development_areas = []
        low_consistency_evaluations = [e for e in evaluations if e.consistency_score < 0.4]
        if low_consistency_evaluations:
            common_inconsistencies = []
            for evaluation in low_consistency_evaluations:
                common_inconsistencies.extend(evaluation.inconsistencies_found)
            
            if common_inconsistencies:
                most_common_issue = max(set(common_inconsistencies), key=common_inconsistencies.count)
                development_areas.append(f"Focus on addressing {most_common_issue.value}")
        
        return {
            "improvements": unique_improvements,
            "reinforcements": unique_reinforcements,
            "strengths": strengths,
            "development_areas": development_areas
        }
    
    def _update_decision_biases(self, agent_id: str, 
                               evaluations: List[ActionEvaluation],
                               state: EnhancedAgentState) -> Dict[str, Any]:
        """Update decision biases based on consistency patterns."""
        
        if agent_id not in self.decision_biases:
            self.decision_biases[agent_id] = []
        
        professional_identity = state.get("professional_identity")
        if not professional_identity:
            return {"biases_updated": 0}
        
        current_role = getattr(professional_identity, 'primary_role', 'contestant')
        role_expectations = self.role_expectations.get(current_role, {})
        
        # Create or update role preference bias
        role_bias = self._find_or_create_bias(agent_id, "role_preference")
        
        # Update based on evaluations
        consistent_actions = [e for e in evaluations if e.consistency_score >= 0.7]
        inconsistent_actions = [e for e in evaluations if e.consistency_score < 0.4]
        
        # Strengthen bias toward consistent actions
        for evaluation in consistent_actions:
            if evaluation.action_description not in role_bias.preferred_actions:
                role_bias.preferred_actions.append(evaluation.action_description)
        
        # Strengthen bias against inconsistent actions  
        for evaluation in inconsistent_actions:
            if evaluation.action_description not in role_bias.discouraged_actions:
                role_bias.discouraged_actions.append(evaluation.action_description)
        
        # Limit list sizes
        role_bias.preferred_actions = role_bias.preferred_actions[-10:]
        role_bias.discouraged_actions = role_bias.discouraged_actions[-10:]
        
        # Update bias strength based on consistency success
        if evaluations:
            avg_consistency = statistics.mean([e.consistency_score for e in evaluations])
            role_bias.bias_strength = min(1.0, role_bias.bias_strength + self.bias_adaptation_rate * (avg_consistency - 0.5))
        
        role_bias.last_applied = datetime.now()
        
        return {"biases_updated": 1}
    
    def _find_or_create_bias(self, agent_id: str, bias_type: str) -> DecisionBias:
        """Find existing bias or create new one."""
        
        # Look for existing bias of this type
        for bias in self.decision_biases[agent_id]:
            if bias.bias_type == bias_type:
                return bias
        
        # Create new bias
        new_bias = DecisionBias(
            bias_id=str(uuid.uuid4()),
            bias_type=bias_type,
            bias_strength=0.5,
            preferred_actions=[],
            discouraged_actions=[],
            action_modifications={},
            applies_in_contexts=["all"],
            context_sensitivity=0.5,
            reinforcement_history=[],
            effectiveness_score=0.5,
            last_applied=None,
            created_at=datetime.now(),
            is_active=True,
            metadata={}
        )
        
        self.decision_biases[agent_id].append(new_bias)
        return new_bias
    
    def _apply_decision_biasing(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """Apply identity-driven biasing to decision making."""
        
        agent_id = state.get("agent_id", "unknown")
        
        if agent_id not in self.decision_biases:
            return {"active_biases": [], "modifications": {}, "overall_bias_strength": 0.0}
        
        active_biases = [b for b in self.decision_biases[agent_id] if b.is_active]
        modifications = {}
        
        for bias in active_biases:
            # Apply action modifications
            for action, modification in bias.action_modifications.items():
                if action not in modifications:
                    modifications[action] = 0.0
                modifications[action] += modification * bias.bias_strength
        
        # Calculate overall bias strength
        overall_strength = statistics.mean([b.bias_strength for b in active_biases]) if active_biases else 0.0
        
        return {
            "active_biases": [{"type": b.bias_type, "strength": b.bias_strength} for b in active_biases],
            "modifications": modifications,
            "overall_bias_strength": overall_strength
        }
    
    def _determine_overall_consistency_level(self, evaluations: List[ActionEvaluation]) -> str:
        """Determine overall consistency level from evaluations."""
        
        if not evaluations:
            return ConsistencyLevel.SOMEWHAT_CONSISTENT.value
        
        avg_score = statistics.mean([e.consistency_score for e in evaluations])
        return self._determine_consistency_level(avg_score).value
    
    def _calculate_validation_efficiency(self) -> float:
        """Calculate efficiency of consistency validation process."""
        
        if not self.recent_evaluations:
            return 1.0
        
        # Efficiency based on evaluation speed and accuracy
        recent_count = len([e for e in self.recent_evaluations if (datetime.now() - e.evaluated_at).seconds < 3600])
        total_count = len(self.recent_evaluations)
        
        recency_factor = recent_count / max(total_count, 1)
        
        # Confidence factor
        avg_confidence = statistics.mean([e.confidence_in_evaluation for e in self.recent_evaluations])
        
        return (recency_factor * 0.4 + avg_confidence * 0.6)
    
    def _calculate_identity_coherence_score(self, evaluations: List[ActionEvaluation]) -> float:
        """Calculate overall identity coherence score."""
        
        if not evaluations:
            return 0.5
        
        # Coherence based on consistency and alignment
        consistency_scores = [e.consistency_score for e in evaluations]
        avg_consistency = statistics.mean(consistency_scores)
        
        # Factor in alignment factors
        all_alignment_scores = []
        for evaluation in evaluations:
            alignment_values = list(evaluation.alignment_factors.values())
            if alignment_values:
                all_alignment_scores.extend(alignment_values)
        
        avg_alignment = statistics.mean(all_alignment_scores) if all_alignment_scores else 0.5
        
        # Combine scores
        coherence_score = avg_consistency * 0.6 + avg_alignment * 0.4
        
        return coherence_score
    
    def _update_performance_stats(self, evaluations: List[ActionEvaluation]) -> None:
        """Update performance statistics."""
        
        self.consistency_stats["evaluations_performed"] += len(evaluations)
        
        if evaluations:
            # Update average consistency score
            new_scores = [e.consistency_score for e in evaluations]
            current_avg = self.consistency_stats["avg_consistency_score"]
            total_evaluations = self.consistency_stats["evaluations_performed"]
            
            self.consistency_stats["avg_consistency_score"] = (
                (current_avg * (total_evaluations - len(evaluations)) + sum(new_scores)) / total_evaluations
            )
        
        # Update other stats as needed
        self.consistency_stats["last_profile_update"] = datetime.now().isoformat()
    
    def get_consistency_summary(self) -> Dict[str, Any]:
        """
        Get summary of identity consistency module performance.
        
        Returns:
            Consistency validation summary with metrics
        """
        
        return {
            "module_name": self.module_name,
            "performance_stats": self.consistency_stats.copy(),
            "active_profiles": len(self.consistency_profiles),
            "recent_evaluations": len([e for e in self.recent_evaluations if (datetime.now() - e.evaluated_at).seconds < 3600]),
            "consistency_distribution": {
                level.value: len([e for e in self.recent_evaluations if e.consistency_level == level])
                for level in ConsistencyLevel
            },
            "common_inconsistencies": {
                inconsistency.value: sum(1 for e in self.recent_evaluations for i in e.inconsistencies_found if i == inconsistency)
                for inconsistency in InconsistencyType
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Example of identity consistency module usage
    from ..enhanced_agent_state import create_enhanced_agent_state
    
    # Create state manager with test data
    state_manager = create_enhanced_agent_state(
        "test_agent", "Test Agent", {"confidence": 0.8, "empathy": 0.9}
    )
    
    # Add some test actions
    test_actions = [
        ("helped comfort a crying contestant", "action", 0.9),
        ("organized fun group activity", "action", 0.7),
        ("made jokes during serious conversation", "action", 0.3),
        ("listened carefully to someone's problems", "action", 0.85)
    ]
    
    for content, mem_type, importance in test_actions:
        state_manager.add_memory(content, mem_type, importance)
    
    # Create identity consistency module
    consistency_module = IdentityConsistencyModule(state_manager)
    
    print("Testing identity consistency validation...")
    
    # Process state to validate consistency
    result = consistency_module(state_manager.state)
    
    print(f"Consistency result: {result}")
    
    # Get consistency summary
    summary = consistency_module.get_consistency_summary()
    print(f"\nIdentity consistency summary:")
    print(f"- Performance stats: {summary['performance_stats']}")
    print(f"- Recent evaluations: {summary['recent_evaluations']}")
    print(f"- Consistency distribution: {summary['consistency_distribution']}")
    
    print("Identity consistency validation example completed!")