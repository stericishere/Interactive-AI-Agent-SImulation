"""
File: specialization_detection.py
Description: Role Emergence Detection module for enhanced PIANO architecture.
Handles action pattern analysis for role identification, goal consistency measurement,
and social role interpretation for professional identity formation.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import time
import math
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import statistics

from .langgraph_base_module import LangGraphBaseModule, ModuleExecutionConfig, ExecutionTimeScale, ModulePriority
from ..enhanced_agent_state import EnhancedAgentState, EnhancedAgentStateManager, SpecializationData


class ProfessionalRole(Enum):
    """Professional roles that agents can develop."""
    CONTESTANT = "contestant"                    # Default role - dating show participant
    SOCIAL_CONNECTOR = "social_connector"        # Facilitates group interactions
    CONFLICT_RESOLVER = "conflict_resolver"      # Mediates disputes and tensions
    EMOTIONAL_SUPPORT = "emotional_support"      # Provides comfort and guidance
    ENTERTAINMENT_LEADER = "entertainment_leader" # Organizes activities and fun
    STRATEGIST = "strategist"                   # Plans and analyzes situations
    CONFIDANT = "confidant"                     # Trusted advisor and secret keeper
    ROMANTIC_PURSUER = "romantic_pursuer"       # Actively seeks romantic connections
    OBSERVER = "observer"                       # Watches and analyzes others
    INDEPENDENT = "independent"                 # Self-reliant and autonomous


class ActionCategory(Enum):
    """Categories of actions for role analysis."""
    SOCIAL_INTERACTION = "social_interaction"    # Talking, bonding, group activities
    CONFLICT_MANAGEMENT = "conflict_management"  # Resolving disputes, mediating
    EMOTIONAL_CARE = "emotional_care"           # Comforting, supporting others
    ENTERTAINMENT = "entertainment"             # Organizing fun, being playful
    STRATEGIC_PLANNING = "strategic_planning"   # Planning, analyzing, deciding
    CONFIDENTIAL_SHARING = "confidential_sharing" # Private conversations, secrets  
    ROMANTIC_PURSUIT = "romantic_pursuit"       # Flirting, dating, intimacy
    OBSERVATION = "observation"                 # Watching, listening, analyzing
    INDEPENDENCE = "independence"               # Solo activities, self-care


@dataclass
class ActionPattern:
    """Represents a pattern of actions indicating role tendencies."""
    pattern_id: str
    action_category: ActionCategory
    frequency: float  # Actions per day
    consistency: float  # 0.0-1.0, how consistent over time
    effectiveness: float  # 0.0-1.0, success rate of actions
    social_impact: float  # 0.0-1.0, impact on others
    recent_trend: float  # -1.0 to 1.0, increasing or decreasing
    evidence_actions: List[str]  # Recent actions supporting this pattern


@dataclass
class RoleAssessment:
    """Assessment of how well an agent fits a professional role."""
    role: ProfessionalRole
    fit_score: float  # 0.0-1.0, how well agent fits this role
    confidence: float  # 0.0-1.0, confidence in assessment
    supporting_patterns: List[ActionPattern]
    key_behaviors: List[str]
    development_trend: float  # -1.0 to 1.0, improving or declining fit
    consistency_score: float  # 0.0-1.0, behavioral consistency
    social_validation: float  # 0.0-1.0, how others respond to this role


@dataclass
class GoalConsistencyAnalysis:
    """Analysis of goal consistency for role development."""
    consistency_score: float  # 0.0-1.0, overall consistency
    conflicting_goals: List[Tuple[str, str]]  # Pairs of conflicting goals
    supporting_goals: List[str]  # Goals that support current role
    goal_achievement_rate: float  # Success rate in achieving goals
    goal_stability: float  # How stable goals are over time
    social_alignment: float  # How well goals align with social expectations


class SpecializationDetectionModule(LangGraphBaseModule):
    """
    Role emergence detection module that analyzes action patterns,
    measures goal consistency, and interprets social roles.
    """
    
    def __init__(self, state_manager: Optional[EnhancedAgentStateManager] = None):
        """
        Initialize Specialization Detection Module.
        
        Args:
            state_manager: Enhanced agent state manager
        """
        config = ModuleExecutionConfig(
            time_scale=ExecutionTimeScale.SLOW,
            priority=ModulePriority.MEDIUM,
            can_run_parallel=True,
            requires_completion=False,
            max_execution_time=3.0
        )
        
        super().__init__("specialization_detection", config, state_manager)
        
        # Detection settings
        self.min_role_confidence = 0.6
        self.pattern_analysis_days = 7
        self.consistency_threshold = 0.7
        self.role_transition_threshold = 0.8
        
        # Action pattern tracking
        self.action_patterns: Dict[ActionCategory, ActionPattern] = {}
        self.role_assessments: Dict[ProfessionalRole, RoleAssessment] = {}
        self.goal_consistency: Optional[GoalConsistencyAnalysis] = None
        
        # Historical tracking
        self.action_history: List[Dict[str, Any]] = []
        self.role_history: List[Tuple[datetime, ProfessionalRole, float]] = []
        self.goal_history: List[Tuple[datetime, List[str]]] = []
        
        # Role transition tracking
        self.role_transitions: List[Dict[str, Any]] = []
        self.current_role_stability = 1.0
        self.role_development_phase = "exploration"  # exploration, specialization, mastery
        
        # Performance tracking
        self.detection_stats = {
            "total_analyses": 0,
            "role_transitions": 0,
            "avg_role_confidence": 0.0,
            "pattern_stability": 0.0,
            "last_role_change": None
        }
        
        self.logger = logging.getLogger("SpecializationDetection")
    
    def process_state(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Process agent state for role emergence detection.
        
        Args:
            state: Current enhanced agent state
        
        Returns:
            Dictionary with specialization analysis results
        """
        start_time = time.time()
        
        try:
            # Analyze recent action patterns
            pattern_analysis = self._analyze_action_patterns(state)
            
            # Assess fit for different professional roles
            role_assessments = self._assess_professional_roles(pattern_analysis)
            
            # Analyze goal consistency
            goal_analysis = self._analyze_goal_consistency(state)
            
            # Detect role emergence or transitions
            role_changes = self._detect_role_transitions(role_assessments)
            
            # Update specialization data in state
            updated_specialization = self._update_specialization_data(state, role_assessments, goal_analysis)
            
            # Update historical tracking
            self._update_historical_tracking(state, role_assessments)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "state_changes": {
                    "specialization": updated_specialization
                },
                "output_data": {
                    "pattern_analysis": {
                        "detected_patterns": len(pattern_analysis),
                        "strongest_pattern": self._get_strongest_pattern(pattern_analysis),
                        "pattern_diversity": self._calculate_pattern_diversity(pattern_analysis)
                    },
                    "role_assessment": {
                        "current_role": updated_specialization.current_role,
                        "role_confidence": updated_specialization.role_consistency_score,
                        "emerging_roles": self._get_emerging_roles(role_assessments),
                        "role_transitions": len(role_changes)
                    },
                    "goal_consistency": {
                        "consistency_score": goal_analysis.consistency_score if goal_analysis else 0.5,
                        "achievement_rate": goal_analysis.goal_achievement_rate if goal_analysis else 0.5,
                        "conflicting_goals": len(goal_analysis.conflicting_goals) if goal_analysis else 0
                    }
                },
                "performance_metrics": {
                    "processing_time_ms": processing_time,
                    "role_stability": self.current_role_stability,
                    "development_phase": self.role_development_phase
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error in specialization detection: {str(e)}")
            return {
                "output_data": {"error": str(e)},
                "performance_metrics": {"processing_time_ms": (time.time() - start_time) * 1000}
            }
    
    def _analyze_action_patterns(self, state: EnhancedAgentState) -> Dict[ActionCategory, ActionPattern]:
        """
        Analyze recent actions to identify behavioral patterns.
        
        Args:
            state: Current agent state
        
        Returns:
            Dictionary of detected action patterns
        """
        patterns = {}
        
        if not self.state_manager:
            return patterns
        
        try:
            # Get recent actions from memory systems
            recent_actions = self._collect_recent_actions()
            
            # Categorize actions
            categorized_actions = self._categorize_actions(recent_actions)
            
            # Analyze patterns for each category
            for category, actions in categorized_actions.items():
                if actions:  # Only analyze categories with actions
                    pattern = self._analyze_category_pattern(category, actions)
                    patterns[category] = pattern
                    self.action_patterns[category] = pattern
        
        except Exception as e:
            self.logger.error(f"Error analyzing action patterns: {str(e)}")
        
        return patterns
    
    def _collect_recent_actions(self) -> List[Dict[str, Any]]:
        """Collect recent actions from memory systems."""
        recent_actions = []
        cutoff_date = datetime.now() - timedelta(days=self.pattern_analysis_days)
        
        try:
            # From working memory
            working_memories = self.state_manager.circular_buffer.get_recent_memories(20)
            for memory in working_memories:
                if memory.get("type") == "action" and memory.get("timestamp", datetime.now()) > cutoff_date:
                    recent_actions.append({
                        "content": memory.get("content", ""),
                        "timestamp": memory.get("timestamp"),
                        "importance": memory.get("importance", 0.5),
                        "metadata": memory.get("metadata", {}),
                        "source": "working"
                    })
            
            # From temporal memory
            temporal_memories = self.state_manager.temporal_memory.retrieve_recent_memories(
                hours_back=self.pattern_analysis_days * 24, memory_type="action", limit=50
            )
            for memory in temporal_memories:
                if memory.get("timestamp", datetime.now()) > cutoff_date:
                    recent_actions.append({
                        "content": memory.get("content", ""),
                        "timestamp": memory.get("timestamp"),
                        "importance": memory.get("importance", 0.5),
                        "metadata": memory.get("context", {}),
                        "source": "temporal"
                    })
            
            # From episodic memory
            for episode in self.state_manager.episodic_memory.episodes.values():
                if episode.end_time > cutoff_date:
                    # Extract actions from episode events
                    for event_id in episode.event_ids:
                        if event_id in self.state_manager.episodic_memory.events:
                            event = self.state_manager.episodic_memory.events[event_id]
                            if event.get("type") == "action":
                                recent_actions.append({
                                    "content": event.get("content", ""),
                                    "timestamp": event.get("timestamp"),
                                    "importance": event.get("importance", 0.5),
                                    "metadata": {
                                        **event.get("metadata", {}),
                                        "episode_id": episode.episode_id,
                                        "participants": list(episode.participants)
                                    },
                                    "source": "episodic"
                                })
        
        except Exception as e:
            self.logger.error(f"Error collecting recent actions: {str(e)}")
        
        return recent_actions
    
    def _categorize_actions(self, actions: List[Dict[str, Any]]) -> Dict[ActionCategory, List[Dict[str, Any]]]:
        """Categorize actions by type."""
        categorized = defaultdict(list)
        
        # Keywords for each category
        category_keywords = {
            ActionCategory.SOCIAL_INTERACTION: ["talk", "chat", "conversation", "discuss", "share", "bond", "group", "socialize"],
            ActionCategory.CONFLICT_MANAGEMENT: ["resolve", "mediate", "calm", "peacekeep", "negotiate", "compromise", "intervene"],
            ActionCategory.EMOTIONAL_CARE: ["comfort", "support", "console", "help", "care", "empathize", "listen", "encourage"],
            ActionCategory.ENTERTAINMENT: ["fun", "play", "joke", "laugh", "organize", "activity", "game", "party", "celebrate"],
            ActionCategory.STRATEGIC_PLANNING: ["plan", "strategy", "analyze", "think", "decide", "calculate", "consider", "evaluate"],
            ActionCategory.CONFIDENTIAL_SHARING: ["secret", "private", "confide", "trust", "personal", "intimate", "whisper"],
            ActionCategory.ROMANTIC_PURSUIT: ["flirt", "romance", "date", "kiss", "attract", "seduce", "court", "intimate"],
            ActionCategory.OBSERVATION: ["watch", "observe", "notice", "study", "monitor", "analyze", "assess", "evaluate"],
            ActionCategory.INDEPENDENCE: ["alone", "solo", "self", "independent", "individual", "personal", "private", "autonomous"]
        }
        
        for action in actions:
            content = action.get("content", "").lower()
            metadata = action.get("metadata", {})
            
            # Score action against each category
            category_scores = {}
            for category, keywords in category_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in content:
                        score += 1
                
                # Additional scoring based on metadata
                if category == ActionCategory.SOCIAL_INTERACTION:
                    participants = metadata.get("participants", [])
                    if len(participants) > 1:
                        score += 2
                
                elif category == ActionCategory.ROMANTIC_PURSUIT:
                    if metadata.get("emotional_valence", 0) > 0.5:
                        score += 1
                
                elif category == ActionCategory.CONFLICT_MANAGEMENT:
                    if "conflict" in metadata or "tension" in metadata:
                        score += 2
                
                category_scores[category] = score
            
            # Assign to category with highest score
            if category_scores:
                best_category = max(category_scores.items(), key=lambda x: x[1])
                if best_category[1] > 0:  # Only if there's some match
                    categorized[best_category[0]].append(action)
                else:
                    # Default to social interaction if no clear category
                    categorized[ActionCategory.SOCIAL_INTERACTION].append(action)
        
        return dict(categorized)
    
    def _analyze_category_pattern(self, category: ActionCategory, actions: List[Dict[str, Any]]) -> ActionPattern:
        """Analyze pattern for a specific action category."""
        if not actions:
            return ActionPattern(
                pattern_id=f"pattern_{category.value}",
                action_category=category,
                frequency=0.0,
                consistency=0.0,
                effectiveness=0.0,
                social_impact=0.0,
                recent_trend=0.0,
                evidence_actions=[]
            )
        
        # Calculate frequency (actions per day)
        total_days = self.pattern_analysis_days
        frequency = len(actions) / total_days
        
        # Calculate consistency (how evenly distributed over time)
        action_dates = [action.get("timestamp", datetime.now()).date() for action in actions]
        date_counts = Counter(action_dates)
        consistency = self._calculate_temporal_consistency(date_counts, total_days)
        
        # Calculate effectiveness (based on importance scores)
        importance_scores = [action.get("importance", 0.5) for action in actions]
        effectiveness = statistics.mean(importance_scores) if importance_scores else 0.5
        
        # Calculate social impact (based on participant count and emotional valence)
        social_impact = self._calculate_social_impact(actions)
        
        # Calculate recent trend (increasing or decreasing frequency)
        recent_trend = self._calculate_recent_trend(actions)
        
        # Select evidence actions (most important/recent)
        evidence_actions = self._select_evidence_actions(actions, 5)
        
        return ActionPattern(
            pattern_id=f"pattern_{category.value}_{int(time.time())}",
            action_category=category,
            frequency=frequency,
            consistency=consistency,
            effectiveness=effectiveness,
            social_impact=social_impact,
            recent_trend=recent_trend,
            evidence_actions=evidence_actions
        )
    
    def _calculate_temporal_consistency(self, date_counts: Counter, total_days: int) -> float:
        """Calculate how consistently actions are distributed over time."""
        if not date_counts:
            return 0.0
        
        # Expected frequency per day
        total_actions = sum(date_counts.values())
        expected_per_day = total_actions / total_days
        
        # Calculate variance from expected
        variances = []
        for i in range(total_days):
            date = datetime.now().date() - timedelta(days=i)
            actual = date_counts.get(date, 0)
            variance = (actual - expected_per_day) ** 2
            variances.append(variance)
        
        # Convert variance to consistency score (lower variance = higher consistency)
        avg_variance = statistics.mean(variances) if variances else 0
        max_possible_variance = (total_actions - expected_per_day) ** 2
        
        if max_possible_variance > 0:
            consistency = 1.0 - (avg_variance / max_possible_variance)
            return max(0.0, min(1.0, consistency))
        
        return 1.0  # Perfect consistency if no variance possible
    
    def _calculate_social_impact(self, actions: List[Dict[str, Any]]) -> float:
        """Calculate social impact of actions."""
        if not actions:
            return 0.0
        
        impact_scores = []
        
        for action in actions:
            metadata = action.get("metadata", {})
            impact = 0.0
            
            # Number of participants involved
            participants = metadata.get("participants", [])
            if participants:
                impact += min(len(participants) * 0.2, 0.8)  # Up to 0.8 for many participants
            
            # Emotional valence (positive impact)
            emotional_valence = metadata.get("emotional_valence", 0.0)
            if emotional_valence > 0:
                impact += emotional_valence * 0.3
            
            # Importance of the action
            importance = action.get("importance", 0.5)
            impact += importance * 0.2
            
            impact_scores.append(min(impact, 1.0))
        
        return statistics.mean(impact_scores) if impact_scores else 0.0
    
    def _calculate_recent_trend(self, actions: List[Dict[str, Any]]) -> float:
        """Calculate whether actions are increasing or decreasing recently."""
        if len(actions) < 4:  # Need minimum actions for trend
            return 0.0
        
        # Sort actions by timestamp
        sorted_actions = sorted(actions, key=lambda x: x.get("timestamp", datetime.now()))
        
        # Split into first half and second half
        mid_point = len(sorted_actions) // 2
        first_half = sorted_actions[:mid_point]
        second_half = sorted_actions[mid_point:]
        
        # Calculate frequency in each half
        first_days = (sorted_actions[mid_point-1].get("timestamp") - sorted_actions[0].get("timestamp")).days + 1
        second_days = (sorted_actions[-1].get("timestamp") - sorted_actions[mid_point].get("timestamp")).days + 1
        
        first_frequency = len(first_half) / max(first_days, 1)
        second_frequency = len(second_half) / max(second_days, 1)
        
        # Calculate trend (-1.0 to 1.0)
        if first_frequency == 0:
            return 1.0 if second_frequency > 0 else 0.0
        
        trend = (second_frequency - first_frequency) / first_frequency
        return max(-1.0, min(1.0, trend))
    
    def _select_evidence_actions(self, actions: List[Dict[str, Any]], count: int) -> List[str]:
        """Select most relevant actions as evidence."""
        # Sort by importance and recency
        def action_score(action):
            importance = action.get("importance", 0.5)
            recency = (datetime.now() - action.get("timestamp", datetime.now())).days
            recency_score = max(0, 1.0 - (recency / 7))  # Decay over 7 days
            return importance * 0.7 + recency_score * 0.3
        
        sorted_actions = sorted(actions, key=action_score, reverse=True)
        return [action.get("content", "") for action in sorted_actions[:count]]
    
    def _assess_professional_roles(self, patterns: Dict[ActionCategory, ActionPattern]) -> Dict[ProfessionalRole, RoleAssessment]:
        """Assess how well agent fits different professional roles."""
        role_assessments = {}
        
        # Role-pattern mappings with weights
        role_patterns = {
            ProfessionalRole.SOCIAL_CONNECTOR: {
                ActionCategory.SOCIAL_INTERACTION: 0.8,
                ActionCategory.ENTERTAINMENT: 0.6,
                ActionCategory.EMOTIONAL_CARE: 0.4
            },
            ProfessionalRole.CONFLICT_RESOLVER: {
                ActionCategory.CONFLICT_MANAGEMENT: 0.9,
                ActionCategory.EMOTIONAL_CARE: 0.5,
                ActionCategory.SOCIAL_INTERACTION: 0.3
            },
            ProfessionalRole.EMOTIONAL_SUPPORT: {
                ActionCategory.EMOTIONAL_CARE: 0.9,
                ActionCategory.CONFIDENTIAL_SHARING: 0.6,
                ActionCategory.SOCIAL_INTERACTION: 0.4
            },
            ProfessionalRole.ENTERTAINMENT_LEADER: {
                ActionCategory.ENTERTAINMENT: 0.9,
                ActionCategory.SOCIAL_INTERACTION: 0.7,
                ActionCategory.INDEPENDENCE: -0.3  # Negative weight
            },
            ProfessionalRole.STRATEGIST: {
                ActionCategory.STRATEGIC_PLANNING: 0.9,
                ActionCategory.OBSERVATION: 0.7,
                ActionCategory.INDEPENDENCE: 0.4
            },
            ProfessionalRole.CONFIDANT: {
                ActionCategory.CONFIDENTIAL_SHARING: 0.9,
                ActionCategory.EMOTIONAL_CARE: 0.6,
                ActionCategory.SOCIAL_INTERACTION: 0.5
            },
            ProfessionalRole.ROMANTIC_PURSUER: {
                ActionCategory.ROMANTIC_PURSUIT: 0.9,
                ActionCategory.SOCIAL_INTERACTION: 0.5,
                ActionCategory.ENTERTAINMENT: 0.3
            },
            ProfessionalRole.OBSERVER: {
                ActionCategory.OBSERVATION: 0.9,
                ActionCategory.INDEPENDENCE: 0.6,
                ActionCategory.STRATEGIC_PLANNING: 0.4
            },
            ProfessionalRole.INDEPENDENT: {
                ActionCategory.INDEPENDENCE: 0.9,
                ActionCategory.SOCIAL_INTERACTION: -0.4,  # Negative weight
                ActionCategory.ENTERTAINMENT: -0.2
            }
        }
        
        for role, role_weights in role_patterns.items():
            fit_score = self._calculate_role_fit_score(patterns, role_weights)
            confidence = self._calculate_role_confidence(patterns, role_weights)
            supporting_patterns = self._get_supporting_patterns(patterns, role_weights)
            key_behaviors = self._extract_key_behaviors(supporting_patterns)
            development_trend = self._calculate_development_trend(patterns, role_weights)
            consistency_score = self._calculate_role_consistency(patterns, role_weights)
            social_validation = self._calculate_social_validation(patterns, role)
            
            assessment = RoleAssessment(
                role=role,
                fit_score=fit_score,
                confidence=confidence,
                supporting_patterns=supporting_patterns,
                key_behaviors=key_behaviors,
                development_trend=development_trend,
                consistency_score=consistency_score,
                social_validation=social_validation
            )
            
            role_assessments[role] = assessment
            self.role_assessments[role] = assessment
        
        return role_assessments
    
    def _calculate_role_fit_score(self, patterns: Dict[ActionCategory, ActionPattern], 
                                 role_weights: Dict[ActionCategory, float]) -> float:
        """Calculate how well patterns fit a role."""
        weighted_scores = []
        
        for category, weight in role_weights.items():
            if category in patterns:
                pattern = patterns[category]
                # Combine frequency, consistency, and effectiveness
                pattern_strength = (
                    min(pattern.frequency / 2.0, 1.0) * 0.4 +  # Normalize frequency to max 2/day
                    pattern.consistency * 0.3 +
                    pattern.effectiveness * 0.3
                )
                weighted_scores.append(pattern_strength * abs(weight))
            else:
                # No pattern for this category
                if weight > 0:
                    weighted_scores.append(0.0)  # Missing positive pattern
                else:
                    weighted_scores.append(abs(weight))  # Good to not have negative pattern
        
        return statistics.mean(weighted_scores) if weighted_scores else 0.0
    
    def _calculate_role_confidence(self, patterns: Dict[ActionCategory, ActionPattern],
                                  role_weights: Dict[ActionCategory, float]) -> float:
        """Calculate confidence in role assessment."""
        confidence_factors = []
        
        # Factor 1: Pattern consistency
        relevant_patterns = [patterns[cat] for cat in role_weights.keys() if cat in patterns]
        if relevant_patterns:
            avg_consistency = statistics.mean([p.consistency for p in relevant_patterns])
            confidence_factors.append(avg_consistency)
        
        # Factor 2: Pattern strength
        if relevant_patterns:
            avg_frequency = statistics.mean([min(p.frequency / 2.0, 1.0) for p in relevant_patterns])
            confidence_factors.append(avg_frequency)
        
        # Factor 3: Coverage (how many required patterns we have)
        positive_weights = [cat for cat, weight in role_weights.items() if weight > 0]
        coverage = len([cat for cat in positive_weights if cat in patterns]) / len(positive_weights)
        confidence_factors.append(coverage)
        
        return statistics.mean(confidence_factors) if confidence_factors else 0.0
    
    def _get_supporting_patterns(self, patterns: Dict[ActionCategory, ActionPattern],
                               role_weights: Dict[ActionCategory, float]) -> List[ActionPattern]:
        """Get patterns that support this role."""
        supporting = []
        
        for category, weight in role_weights.items():
            if weight > 0 and category in patterns:
                pattern = patterns[category]
                if pattern.frequency > 0.5:  # At least some activity
                    supporting.append(pattern)
        
        return supporting
    
    def _extract_key_behaviors(self, patterns: List[ActionPattern]) -> List[str]:
        """Extract key behaviors from supporting patterns."""
        behaviors = []
        
        for pattern in patterns:
            # Add most important evidence actions
            behaviors.extend(pattern.evidence_actions[:2])  # Top 2 per pattern
        
        return behaviors[:10]  # Limit to top 10 behaviors
    
    def _calculate_development_trend(self, patterns: Dict[ActionCategory, ActionPattern],
                                   role_weights: Dict[ActionCategory, float]) -> float:
        """Calculate if agent is developing toward this role."""
        trends = []
        
        for category, weight in role_weights.items():
            if weight > 0 and category in patterns:
                pattern = patterns[category]
                trends.append(pattern.recent_trend * weight)
        
        return statistics.mean(trends) if trends else 0.0
    
    def _calculate_role_consistency(self, patterns: Dict[ActionCategory, ActionPattern],
                                  role_weights: Dict[ActionCategory, float]) -> float:
        """Calculate behavioral consistency for this role."""
        consistency_scores = []
        
        for category, weight in role_weights.items():
            if weight > 0 and category in patterns:
                pattern = patterns[category]
                consistency_scores.append(pattern.consistency * abs(weight))
        
        return statistics.mean(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_social_validation(self, patterns: Dict[ActionCategory, ActionPattern],
                                   role: ProfessionalRole) -> float:
        """Calculate social validation for this role (simplified)."""
        # Use social impact of relevant patterns as proxy for social validation
        social_impacts = []
        
        role_categories = {
            ProfessionalRole.SOCIAL_CONNECTOR: [ActionCategory.SOCIAL_INTERACTION, ActionCategory.ENTERTAINMENT],
            ProfessionalRole.CONFLICT_RESOLVER: [ActionCategory.CONFLICT_MANAGEMENT],
            ProfessionalRole.EMOTIONAL_SUPPORT: [ActionCategory.EMOTIONAL_CARE],
            ProfessionalRole.ENTERTAINMENT_LEADER: [ActionCategory.ENTERTAINMENT],
            ProfessionalRole.STRATEGIST: [ActionCategory.STRATEGIC_PLANNING],
            ProfessionalRole.CONFIDANT: [ActionCategory.CONFIDENTIAL_SHARING],
            ProfessionalRole.ROMANTIC_PURSUER: [ActionCategory.ROMANTIC_PURSUIT],
            ProfessionalRole.OBSERVER: [ActionCategory.OBSERVATION],
            ProfessionalRole.INDEPENDENT: [ActionCategory.INDEPENDENCE]
        }
        
        relevant_categories = role_categories.get(role, [])
        for category in relevant_categories:
            if category in patterns:
                social_impacts.append(patterns[category].social_impact)
        
        return statistics.mean(social_impacts) if social_impacts else 0.5
    
    def _analyze_goal_consistency(self, state: EnhancedAgentState) -> GoalConsistencyAnalysis:
        """Analyze consistency of agent's goals with their emerging role."""
        goals = state.get("goals", [])
        
        if not goals:
            return GoalConsistencyAnalysis(
                consistency_score=1.0,  # No goals = no inconsistency
                conflicting_goals=[],
                supporting_goals=[],
                goal_achievement_rate=0.5,
                goal_stability=1.0,
                social_alignment=0.5
            )
        
        # Get current specialization
        specialization = state.get("specialization")
        current_role = specialization.current_role if specialization else "contestant"
        
        # Analyze goals for consistency
        supporting_goals = []
        conflicting_goals = []
        
        # Define goal-role compatibility
        role_compatible_goals = {
            "social_connector": ["make friends", "help others", "organize", "connect people"],
            "conflict_resolver": ["resolve conflict", "mediate", "peacekeep", "calm situation"],
            "emotional_support": ["support others", "be there", "help", "comfort"],
            "entertainment_leader": ["have fun", "entertain", "organize party", "make laugh"],
            "strategist": ["plan ahead", "analyze", "think through", "strategize"],
            "confidant": ["be trustworthy", "keep secrets", "listen", "advise"],
            "romantic_pursuer": ["find love", "romantic", "date", "connection"],
            "observer": ["understand", "watch", "learn", "analyze"],
            "independent": ["self-reliant", "autonomous", "independent", "alone"]
        }
        
        compatible_keywords = role_compatible_goals.get(current_role, [])
        
        for goal in goals:
            goal_lower = goal.lower()
            is_supporting = any(keyword in goal_lower for keyword in compatible_keywords)
            
            if is_supporting:
                supporting_goals.append(goal)
            else:
                # Check for conflicts with other roles
                for other_role, other_keywords in role_compatible_goals.items():
                    if other_role != current_role:
                        if any(keyword in goal_lower for keyword in other_keywords):
                            conflicting_goals.append((goal, other_role))
                            break
        
        # Calculate consistency score
        total_goals = len(goals)
        consistency_score = len(supporting_goals) / total_goals if total_goals > 0 else 1.0
        
        # Calculate other metrics (simplified)
        goal_achievement_rate = 0.7  # Would need historical data
        goal_stability = 0.8  # Would need goal history
        social_alignment = 0.6  # Would need social feedback
        
        analysis = GoalConsistencyAnalysis(
            consistency_score=consistency_score,
            conflicting_goals=conflicting_goals,
            supporting_goals=supporting_goals,
            goal_achievement_rate=goal_achievement_rate,
            goal_stability=goal_stability,
            social_alignment=social_alignment
        )
        
        self.goal_consistency = analysis
        return analysis
    
    def _detect_role_transitions(self, role_assessments: Dict[ProfessionalRole, RoleAssessment]) -> List[Dict[str, Any]]:
        """Detect if agent is transitioning to a new role."""
        role_changes = []
        
        if not self.state_manager:
            return role_changes
        
        current_specialization = self.state_manager.state.get("specialization")
        if not current_specialization:
            return role_changes
        
        current_role = ProfessionalRole(current_specialization.current_role)
        current_assessment = role_assessments.get(current_role)
        
        # Check if current role is still strong
        if current_assessment and current_assessment.fit_score > self.role_transition_threshold:
            # Current role is still strong, check stability
            self.current_role_stability = min(self.current_role_stability + 0.1, 1.0)
            return role_changes
        
        # Look for stronger alternative roles
        for role, assessment in role_assessments.items():
            if (role != current_role and 
                assessment.fit_score > self.role_transition_threshold and
                assessment.confidence > self.min_role_confidence):
                
                # Potential role transition
                transition = {
                    "from_role": current_role.value,
                    "to_role": role.value,
                    "fit_score": assessment.fit_score,
                    "confidence": assessment.confidence,
                    "supporting_evidence": assessment.key_behaviors[:3],
                    "detected_at": datetime.now().isoformat()
                }
                
                role_changes.append(transition)
                self.role_transitions.append(transition)
                self.current_role_stability = 0.5  # Reset stability
                
                # Update development phase
                if self.role_development_phase == "exploration":
                    self.role_development_phase = "specialization"
                elif assessment.fit_score > 0.9:
                    self.role_development_phase = "mastery"
        
        return role_changes
    
    def _update_specialization_data(self, state: EnhancedAgentState, 
                                  role_assessments: Dict[ProfessionalRole, RoleAssessment],
                                  goal_analysis: GoalConsistencyAnalysis) -> SpecializationData:
        """Update specialization data based on analysis."""
        current_specialization = state.get("specialization")
        
        if not current_specialization:
            # Create initial specialization
            return SpecializationData(
                current_role="contestant",
                role_history=["contestant"],
                skills={},
                expertise_level=0.1,
                role_consistency_score=0.5
            )
        
        # Find best role
        best_role_assessment = max(role_assessments.values(), key=lambda x: x.fit_score)
        
        # Update role if confidence is high enough
        if (best_role_assessment.confidence > self.min_role_confidence and
            best_role_assessment.fit_score > self.role_transition_threshold):
            
            new_role = best_role_assessment.role.value
            
            # Update role history if role changed
            if new_role != current_specialization.current_role:
                if new_role not in current_specialization.role_history:
                    current_specialization.role_history.append(new_role)
                current_specialization.current_role = new_role
                current_specialization.last_role_change = datetime.now()
        
        # Update skills based on action patterns
        for category, pattern in self.action_patterns.items():
            skill_name = category.value
            
            # Calculate skill level based on pattern strength
            skill_level = (pattern.frequency / 3.0) * 0.4 + pattern.consistency * 0.3 + pattern.effectiveness * 0.3
            skill_level = min(skill_level, 1.0)
            
            if skill_level > 0.1:  # Only track meaningful skills
                current_specialization.skills[skill_name] = skill_level
        
        # Update expertise level
        if current_specialization.skills:
            current_specialization.expertise_level = statistics.mean(current_specialization.skills.values())
        
        # Update consistency score
        if goal_analysis:
            current_specialization.role_consistency_score = (
                best_role_assessment.consistency_score * 0.6 +
                goal_analysis.consistency_score * 0.4
            )
        else:
            current_specialization.role_consistency_score = best_role_assessment.consistency_score
        
        return current_specialization
    
    def _update_historical_tracking(self, state: EnhancedAgentState,
                                  role_assessments: Dict[ProfessionalRole, RoleAssessment]) -> None:
        """Update historical tracking data."""
        current_time = datetime.now()
        
        # Update role history
        best_role = max(role_assessments.values(), key=lambda x: x.fit_score)
        self.role_history.append((current_time, best_role.role, best_role.fit_score))
        
        # Keep last 30 entries
        if len(self.role_history) > 30:
            self.role_history = self.role_history[-30:]
        
        # Update goal history
        goals = state.get("goals", [])
        self.goal_history.append((current_time, goals.copy()))
        
        # Keep last 20 entries
        if len(self.goal_history) > 20:
            self.goal_history = self.goal_history[-20:]
        
        # Update statistics
        self.detection_stats["total_analyses"] += 1
        if self.role_history and len(self.role_history) > 1:
            if self.role_history[-1][1] != self.role_history[-2][1]:
                self.detection_stats["role_transitions"] += 1
                self.detection_stats["last_role_change"] = current_time.isoformat()
        
        # Update average role confidence
        current_avg = self.detection_stats["avg_role_confidence"]
        total_analyses = self.detection_stats["total_analyses"]
        self.detection_stats["avg_role_confidence"] = (
            (current_avg * (total_analyses - 1) + best_role.confidence) / total_analyses
        )
    
    def _get_strongest_pattern(self, patterns: Dict[ActionCategory, ActionPattern]) -> Optional[str]:
        """Get the strongest action pattern."""
        if not patterns:
            return None
        
        strongest = max(patterns.values(), key=lambda p: p.frequency * p.consistency * p.effectiveness)
        return strongest.action_category.value
    
    def _calculate_pattern_diversity(self, patterns: Dict[ActionCategory, ActionPattern]) -> float:
        """Calculate diversity of action patterns."""
        if not patterns:
            return 0.0
        
        # Shannon diversity index
        total_frequency = sum(p.frequency for p in patterns.values())
        if total_frequency == 0:
            return 0.0
        
        diversity = 0.0
        for pattern in patterns.values():
            if pattern.frequency > 0:
                proportion = pattern.frequency / total_frequency
                diversity -= proportion * math.log2(proportion)
        
        # Normalize to 0-1 scale
        max_diversity = math.log2(len(patterns))
        return diversity / max_diversity if max_diversity > 0 else 0.0
    
    def _get_emerging_roles(self, role_assessments: Dict[ProfessionalRole, RoleAssessment]) -> List[str]:
        """Get roles that are emerging (high development trend)."""
        emerging = []
        
        for role, assessment in role_assessments.items():
            if (assessment.development_trend > 0.3 and 
                assessment.fit_score > 0.4 and
                assessment.confidence > 0.5):
                emerging.append(role.value)
        
        return emerging
    
    def get_specialization_summary(self) -> Dict[str, Any]:
        """
        Get summary of specialization detection results.
        
        Returns:
            Specialization summary with analysis
        """
        return {
            "module_name": self.module_name,
            "current_role_stability": self.current_role_stability,
            "development_phase": self.role_development_phase,
            "detection_stats": self.detection_stats.copy(),
            "action_patterns": {
                category.value: {
                    "frequency": pattern.frequency,
                    "consistency": pattern.consistency,
                    "effectiveness": pattern.effectiveness,
                    "social_impact": pattern.social_impact,
                    "recent_trend": pattern.recent_trend
                }
                for category, pattern in self.action_patterns.items()
            },
            "role_assessments": {
                role.value: {
                    "fit_score": assessment.fit_score,
                    "confidence": assessment.confidence,
                    "consistency_score": assessment.consistency_score,
                    "development_trend": assessment.development_trend,
                    "social_validation": assessment.social_validation
                }
                for role, assessment in self.role_assessments.items()
            },
            "goal_consistency": {
                "consistency_score": self.goal_consistency.consistency_score if self.goal_consistency else 0.5,
                "supporting_goals": len(self.goal_consistency.supporting_goals) if self.goal_consistency else 0,
                "conflicting_goals": len(self.goal_consistency.conflicting_goals) if self.goal_consistency else 0
            } if self.goal_consistency else None,
            "recent_transitions": self.role_transitions[-3:] if self.role_transitions else []
        }


# Example usage and testing
if __name__ == "__main__":
    # Example of specialization detection module usage
    from ..enhanced_agent_state import create_enhanced_agent_state
    
    # Create state manager with test data
    state_manager = create_enhanced_agent_state(
        "test_agent", "Test Agent", {"confidence": 0.8, "empathy": 0.9}
    )
    
    # Add test actions that suggest emotional support role
    test_actions = [
        ("Comforted Maria when she was upset", "action", 0.9),
        ("Listened to Klaus talk about his problems", "action", 0.8),
        ("Helped resolve conflict between two contestants", "action", 0.85),
        ("Organized group therapy session", "action", 0.7),
        ("Gave advice to struggling contestant", "action", 0.75)
    ]
    
    for content, mem_type, importance in test_actions:
        state_manager.add_memory(content, mem_type, importance)
    
    # Set some goals that align with emotional support
    state_manager.state["goals"] = [
        "Help others feel better",
        "Be a good listener",
        "Support my fellow contestants"
    ]
    
    # Create specialization detection module
    specialization_module = SpecializationDetectionModule(state_manager)
    
    print("Testing specialization detection...")
    
    # Process state to detect role emergence
    result = specialization_module(state_manager.state)
    
    print(f"Detection result: {result}")
    
    # Get specialization summary
    summary = specialization_module.get_specialization_summary()
    print(f"\nSpecialization summary:")
    print(f"- Development phase: {summary['development_phase']}")
    print(f"- Role stability: {summary['current_role_stability']:.3f}")
    
    print(f"\nAction patterns:")
    for pattern, data in summary['action_patterns'].items():
        print(f"- {pattern}: freq={data['frequency']:.2f}, consistency={data['consistency']:.2f}")
    
    print(f"\nRole assessments:")
    for role, assessment in summary['role_assessments'].items():
        print(f"- {role}: fit={assessment['fit_score']:.3f}, confidence={assessment['confidence']:.3f}")
    
    print("Specialization detection module example completed!")