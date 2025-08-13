"""
File: behavioral_adaptation.py
Description: Behavioral Adaptation System for rule influence on agent decision-making.
Implements rule internalization, behavioral adaptation learning algorithms,
and integration with agent cognitive processes to ensure rule compliance.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
import math
import random

# Import governance and agent components
from .compliance_monitoring import ComplianceMonitor, Rule, RuleCategory, ViolationType
from ..agents.memory_structures.store_integration import MemoryStoreIntegration, StoreNamespace
from ..agents.enhanced_agent_state import GovernanceData


class AdaptationStrategy(Enum):
    """Strategies for behavioral adaptation."""
    REINFORCEMENT_LEARNING = "reinforcement_learning"  # Learn from rewards/penalties
    SOCIAL_LEARNING = "social_learning"  # Learn from observing others
    RULE_INTERNALIZATION = "rule_internalization"  # Internalize rules into values
    HABIT_FORMATION = "habit_formation"  # Form habitual compliance behaviors
    COGNITIVE_DISSONANCE = "cognitive_dissonance"  # Reduce dissonance between actions and rules


class AdaptationPhase(Enum):
    """Phases of behavioral adaptation."""
    AWARENESS = "awareness"  # Becoming aware of rules
    UNDERSTANDING = "understanding"  # Understanding rule implications
    ACCEPTANCE = "acceptance"  # Accepting rule validity
    INTEGRATION = "integration"  # Integrating rules into decision-making
    INTERNALIZATION = "internalization"  # Rules become personal values
    AUTOMATIZATION = "automatization"  # Compliance becomes automatic


@dataclass
class BehavioralPattern:
    """Represents a learned behavioral pattern."""
    pattern_id: str
    agent_id: str
    pattern_name: str
    trigger_conditions: Dict[str, Any]  # When this pattern activates
    behavioral_response: Dict[str, Any]  # What behavior to exhibit
    associated_rules: Set[str]  # Rule IDs this pattern helps comply with
    strength: float  # How strong this pattern is (0.0 to 1.0)
    success_rate: float  # How often this pattern achieves compliance
    usage_count: int  # How many times this pattern has been used
    last_used: Optional[datetime]
    created_at: datetime
    adaptation_source: AdaptationStrategy  # How this pattern was learned
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.associated_rules:
            self.associated_rules = set()
        if not self.metadata:
            self.metadata = {}


@dataclass
class RuleInternalization:
    """Tracks how well an agent has internalized a rule."""
    agent_id: str
    rule_id: str
    internalization_level: float  # 0.0 to 1.0
    adaptation_phase: AdaptationPhase
    understanding_score: float  # How well agent understands the rule
    acceptance_score: float  # How much agent accepts the rule
    compliance_motivation: float  # Intrinsic motivation to comply
    behavioral_patterns: Set[str]  # Pattern IDs related to this rule
    violation_sensitivity: float  # How sensitive to violations
    learning_rate: float  # How fast agent adapts to this rule
    last_updated: datetime
    adaptation_history: List[Dict[str, Any]]  # History of adaptation changes
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.behavioral_patterns:
            self.behavioral_patterns = set()
        if not self.adaptation_history:
            self.adaptation_history = []
        if not self.metadata:
            self.metadata = {}


@dataclass
class AdaptationContext:
    """Context for behavioral adaptation decisions."""
    agent_id: str
    situation: Dict[str, Any]  # Current situation
    relevant_rules: List[str]  # Applicable rule IDs
    social_pressure: float  # Social pressure to comply
    personal_values: Dict[str, float]  # Agent's personal values
    recent_violations: List[str]  # Recent violation IDs
    peer_behaviors: Dict[str, Any]  # Observed behaviors of peers
    reward_history: List[float]  # Recent rewards/penalties
    timestamp: datetime


class BehavioralAdaptationSystem:
    """
    Behavioral Adaptation System for rule-influenced decision making.
    Helps agents learn to comply with rules through various adaptation strategies.
    """

    def __init__(self, compliance_monitor: ComplianceMonitor,
                 store_integration: MemoryStoreIntegration,
                 postgres_persistence=None, community_size: int = 50):
        """
        Initialize the Behavioral Adaptation System.
        
        Args:
            compliance_monitor: Compliance monitoring system
            store_integration: Store API integration
            postgres_persistence: PostgreSQL persistence layer
            community_size: Expected community size
        """
        self.compliance_monitor = compliance_monitor
        self.store_integration = store_integration
        self.postgres_persistence = postgres_persistence
        self.community_size = community_size
        self.logger = logging.getLogger(f"{__name__}.BehavioralAdaptationSystem")
        
        # Adaptation tracking
        self.agent_patterns = {}  # agent_id -> List[BehavioralPattern]
        self.rule_internalization = {}  # (agent_id, rule_id) -> RuleInternalization
        self.adaptation_contexts = {}  # agent_id -> AdaptationContext
        
        # Configuration
        self.config = {
            "learning_rate": 0.1,
            "pattern_decay_rate": 0.05,  # Daily decay for unused patterns
            "success_threshold": 0.7,  # Success rate needed for pattern reinforcement
            "internalization_threshold": 0.8,  # Level needed for full internalization
            "social_influence_weight": 0.3,  # Weight of social learning
            "adaptation_update_frequency": 3600,  # Update frequency in seconds
            "pattern_strength_threshold": 0.1,  # Minimum strength to keep pattern
            "max_patterns_per_agent": 50,  # Maximum behavioral patterns per agent
            "violation_sensitivity_decay": 0.02  # Daily decay of violation sensitivity
        }
        
        # Learning strategies
        self.adaptation_strategies = {
            AdaptationStrategy.REINFORCEMENT_LEARNING: self._reinforcement_learning,
            AdaptationStrategy.SOCIAL_LEARNING: self._social_learning,
            AdaptationStrategy.RULE_INTERNALIZATION: self._rule_internalization,
            AdaptationStrategy.HABIT_FORMATION: self._habit_formation,
            AdaptationStrategy.COGNITIVE_DISSONANCE: self._cognitive_dissonance_reduction
        }
        
        # Metrics
        self.metrics = {
            "total_patterns": 0,
            "successful_adaptations": 0,
            "average_internalization": 0.0,
            "adaptation_efficiency": 0.0
        }

    # =====================================================
    # Agent Behavior Adaptation
    # =====================================================

    async def initialize_agent_adaptation(self, agent_id: str, 
                                        initial_values: Dict[str, float] = None) -> None:
        """Initialize behavioral adaptation for an agent."""
        try:
            if agent_id not in self.agent_patterns:
                self.agent_patterns[agent_id] = []
            
            # Initialize rule internalization for all active rules
            for rule_id, rule in self.compliance_monitor.active_rules.items():
                if not rule.is_active:
                    continue
                
                internalization = RuleInternalization(
                    agent_id=agent_id,
                    rule_id=rule_id,
                    internalization_level=0.1,  # Start with minimal internalization
                    adaptation_phase=AdaptationPhase.AWARENESS,
                    understanding_score=0.0,
                    acceptance_score=0.0,
                    compliance_motivation=0.5,  # Neutral motivation
                    behavioral_patterns=set(),
                    violation_sensitivity=1.0,  # High sensitivity initially
                    learning_rate=self.config["learning_rate"],
                    last_updated=datetime.now(),
                    adaptation_history=[],
                    metadata={"initialized_at": datetime.now().isoformat()}
                )
                
                self.rule_internalization[(agent_id, rule_id)] = internalization
            
            # Create initial adaptation context
            self.adaptation_contexts[agent_id] = AdaptationContext(
                agent_id=agent_id,
                situation={},
                relevant_rules=[],
                social_pressure=0.5,
                personal_values=initial_values or {},
                recent_violations=[],
                peer_behaviors={},
                reward_history=[],
                timestamp=datetime.now()
            )
            
            self.logger.info(f"Initialized behavioral adaptation for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize adaptation for {agent_id}: {str(e)}")

    async def adapt_behavior_for_action(self, agent_id: str, proposed_action: Dict[str, Any],
                                      situation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Adapt agent behavior for a proposed action based on learned rules.
        
        Args:
            agent_id: Agent identifier
            proposed_action: Action the agent wants to take
            situation_context: Current situation context
        
        Returns:
            Dict with adapted action and adaptation reasoning
        """
        try:
            if agent_id not in self.agent_patterns:
                await self.initialize_agent_adaptation(agent_id)
            
            # Update adaptation context
            await self._update_adaptation_context(agent_id, situation_context or {})
            
            # Get relevant rules for this action
            relevant_rules = await self._get_relevant_rules_for_action(proposed_action)
            
            # Check current internalization levels
            adaptation_factors = await self._calculate_adaptation_factors(agent_id, relevant_rules)
            
            # Apply behavioral patterns
            adapted_action = await self._apply_behavioral_patterns(agent_id, proposed_action, relevant_rules)
            
            # Calculate adaptation confidence
            adaptation_confidence = await self._calculate_adaptation_confidence(agent_id, relevant_rules)
            
            return {
                "adapted_action": adapted_action,
                "original_action": proposed_action,
                "adaptation_factors": adaptation_factors,
                "relevant_rules": relevant_rules,
                "adaptation_confidence": adaptation_confidence,
                "patterns_applied": [p.pattern_id for p in self.agent_patterns[agent_id] 
                                   if any(r in relevant_rules for r in p.associated_rules)]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to adapt behavior for {agent_id}: {str(e)}")
            return {
                "adapted_action": proposed_action,
                "original_action": proposed_action,
                "adaptation_factors": {},
                "relevant_rules": [],
                "adaptation_confidence": 0.0,
                "patterns_applied": []
            }

    async def learn_from_action_outcome(self, agent_id: str, action: Dict[str, Any],
                                      outcome: Dict[str, Any], violations: List[str] = None) -> None:
        """Learn from the outcome of an action."""
        try:
            if agent_id not in self.agent_patterns:
                await self.initialize_agent_adaptation(agent_id)
            
            # Calculate reward/penalty from outcome
            reward = await self._calculate_reward_from_outcome(outcome, violations or [])
            
            # Update adaptation context
            context = self.adaptation_contexts[agent_id]
            context.reward_history.append(reward)
            if len(context.reward_history) > 20:  # Keep only recent history
                context.reward_history = context.reward_history[-20:]
            
            # Update violation history
            if violations:
                context.recent_violations.extend(violations)
                context.recent_violations = context.recent_violations[-10:]  # Keep 10 most recent
            
            # Apply learning strategies
            for strategy in AdaptationStrategy:
                if strategy in self.adaptation_strategies:
                    await self.adaptation_strategies[strategy](agent_id, action, outcome, reward)
            
            # Update rule internalization based on outcome
            relevant_rules = await self._get_relevant_rules_for_action(action)
            for rule_id in relevant_rules:
                await self._update_rule_internalization(agent_id, rule_id, action, outcome, reward)
            
            # Create or reinforce behavioral patterns
            await self._update_behavioral_patterns(agent_id, action, outcome, reward)
            
            self.logger.debug(f"Agent {agent_id} learned from action outcome with reward {reward}")
            
        except Exception as e:
            self.logger.error(f"Failed to process learning for {agent_id}: {str(e)}")

    async def get_compliance_motivation(self, agent_id: str, rule_id: str) -> float:
        """Get agent's intrinsic motivation to comply with a specific rule."""
        try:
            key = (agent_id, rule_id)
            if key not in self.rule_internalization:
                return 0.5  # Default neutral motivation
            
            internalization = self.rule_internalization[key]
            
            # Motivation is combination of acceptance and internalization
            motivation = (internalization.acceptance_score * 0.4 + 
                         internalization.internalization_level * 0.6)
            
            # Adjust for recent violations (reduce motivation temporarily)
            if agent_id in self.adaptation_contexts:
                context = self.adaptation_contexts[agent_id]
                recent_rule_violations = [v for v in context.recent_violations 
                                        if v in [r.rule_id for r in self.compliance_monitor.recent_violations.values()
                                                if r.rule_id == rule_id]]
                
                violation_penalty = len(recent_rule_violations) * 0.1
                motivation = max(0.0, motivation - violation_penalty)
            
            return min(1.0, motivation)
            
        except Exception as e:
            self.logger.error(f"Failed to get compliance motivation: {str(e)}")
            return 0.5

    # =====================================================
    # Adaptation Strategies
    # =====================================================

    async def _reinforcement_learning(self, agent_id: str, action: Dict[str, Any],
                                    outcome: Dict[str, Any], reward: float) -> None:
        """Apply reinforcement learning adaptation."""
        try:
            # Strengthen patterns that led to positive outcomes
            for pattern in self.agent_patterns[agent_id]:
                if await self._pattern_matches_action(pattern, action):
                    # Update pattern strength based on reward
                    old_strength = pattern.strength
                    pattern.strength += self.config["learning_rate"] * reward
                    pattern.strength = max(0.0, min(1.0, pattern.strength))
                    
                    # Update success rate
                    pattern.usage_count += 1
                    if reward > 0:
                        pattern.success_rate = ((pattern.success_rate * (pattern.usage_count - 1)) + 1) / pattern.usage_count
                    else:
                        pattern.success_rate = ((pattern.success_rate * (pattern.usage_count - 1)) + 0) / pattern.usage_count
                    
                    pattern.last_used = datetime.now()
                    
                    self.logger.debug(f"Reinforcement learning: pattern {pattern.pattern_name} strength: {old_strength:.3f} -> {pattern.strength:.3f}")
                    
        except Exception as e:
            self.logger.error(f"Reinforcement learning failed: {str(e)}")

    async def _social_learning(self, agent_id: str, action: Dict[str, Any],
                             outcome: Dict[str, Any], reward: float) -> None:
        """Apply social learning adaptation."""
        try:
            # Learn from observing peer behaviors and their outcomes
            context = self.adaptation_contexts.get(agent_id)
            if not context:
                return
            
            # Simulate observing peer behaviors (simplified)
            successful_peer_actions = []
            for peer_id, peer_behaviors in context.peer_behaviors.items():
                if peer_id != agent_id:
                    # Check peer's recent compliance score
                    peer_score = await self.compliance_monitor.get_compliance_score(peer_id)
                    if peer_score > 0.8:  # High compliance peer
                        successful_peer_actions.extend(peer_behaviors.get("recent_actions", []))
            
            # Create patterns based on successful peer behaviors
            if successful_peer_actions and reward < 0:  # Only if current action was unsuccessful
                for peer_action in successful_peer_actions[-3:]:  # Consider last 3 successful actions
                    await self._create_behavioral_pattern_from_observation(agent_id, peer_action, 
                                                                         AdaptationStrategy.SOCIAL_LEARNING)
                    
        except Exception as e:
            self.logger.error(f"Social learning failed: {str(e)}")

    async def _rule_internalization(self, agent_id: str, action: Dict[str, Any],
                                  outcome: Dict[str, Any], reward: float) -> None:
        """Apply rule internalization adaptation."""
        try:
            relevant_rules = await self._get_relevant_rules_for_action(action)
            
            for rule_id in relevant_rules:
                key = (agent_id, rule_id)
                if key not in self.rule_internalization:
                    continue
                
                internalization = self.rule_internalization[key]
                
                # Positive outcomes increase understanding and acceptance
                if reward > 0:
                    internalization.understanding_score = min(1.0, 
                        internalization.understanding_score + self.config["learning_rate"] * 0.5)
                    internalization.acceptance_score = min(1.0,
                        internalization.acceptance_score + self.config["learning_rate"] * 0.3)
                else:
                    # Negative outcomes might reduce acceptance but increase understanding
                    internalization.understanding_score = min(1.0,
                        internalization.understanding_score + self.config["learning_rate"] * 0.2)
                    internalization.acceptance_score = max(0.0,
                        internalization.acceptance_score - self.config["learning_rate"] * 0.1)
                
                # Update internalization level
                internalization.internalization_level = (
                    internalization.understanding_score * 0.4 +
                    internalization.acceptance_score * 0.6
                )
                
                # Update adaptation phase
                await self._update_adaptation_phase(internalization)
                
                internalization.last_updated = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Rule internalization failed: {str(e)}")

    async def _habit_formation(self, agent_id: str, action: Dict[str, Any],
                             outcome: Dict[str, Any], reward: float) -> None:
        """Apply habit formation adaptation."""
        try:
            # Strengthen patterns through repetition
            for pattern in self.agent_patterns[agent_id]:
                if await self._pattern_matches_action(pattern, action):
                    # Habit strength increases with usage frequency
                    habit_bonus = min(0.1, pattern.usage_count * 0.01)
                    pattern.strength += habit_bonus
                    pattern.strength = min(1.0, pattern.strength)
                    
                    # Mark as habit-based pattern
                    if pattern.usage_count > 10:
                        pattern.metadata["is_habit"] = True
                        pattern.metadata["habit_strength"] = habit_bonus
                        
        except Exception as e:
            self.logger.error(f"Habit formation failed: {str(e)}")

    async def _cognitive_dissonance_reduction(self, agent_id: str, action: Dict[str, Any],
                                            outcome: Dict[str, Any], reward: float) -> None:
        """Apply cognitive dissonance reduction adaptation."""
        try:
            # If action violated rules but agent has high rule internalization,
            # create dissonance that motivates behavior change
            relevant_rules = await self._get_relevant_rules_for_action(action)
            
            for rule_id in relevant_rules:
                key = (agent_id, rule_id)
                if key not in self.rule_internalization:
                    continue
                
                internalization = self.rule_internalization[key]
                
                # High internalization + rule violation = high dissonance
                if reward < 0 and internalization.internalization_level > 0.6:
                    # Increase motivation to comply (reduce dissonance)
                    internalization.compliance_motivation = min(1.0,
                        internalization.compliance_motivation + 0.2)
                    
                    # Create avoidance pattern
                    await self._create_avoidance_pattern(agent_id, action, rule_id)
                    
        except Exception as e:
            self.logger.error(f"Cognitive dissonance reduction failed: {str(e)}")

    # =====================================================
    # Behavioral Pattern Management
    # =====================================================

    async def _apply_behavioral_patterns(self, agent_id: str, proposed_action: Dict[str, Any],
                                       relevant_rules: List[str]) -> Dict[str, Any]:
        """Apply relevant behavioral patterns to modify an action."""
        try:
            modified_action = proposed_action.copy()
            
            # Get patterns relevant to current situation
            relevant_patterns = []
            for pattern in self.agent_patterns[agent_id]:
                if (pattern.strength > self.config["pattern_strength_threshold"] and
                    any(rule_id in relevant_rules for rule_id in pattern.associated_rules)):
                    relevant_patterns.append(pattern)
            
            # Sort by strength (strongest patterns applied first)
            relevant_patterns.sort(key=lambda x: x.strength, reverse=True)
            
            # Apply pattern modifications
            for pattern in relevant_patterns:
                if await self._pattern_applies_to_situation(pattern, modified_action):
                    # Apply pattern's behavioral response
                    modifications = pattern.behavioral_response
                    
                    # Apply modifications with strength-based weighting
                    for key, value in modifications.items():
                        if key in modified_action:
                            if isinstance(value, (int, float)) and isinstance(modified_action[key], (int, float)):
                                # Weighted average based on pattern strength
                                modified_action[key] = (
                                    modified_action[key] * (1 - pattern.strength) +
                                    value * pattern.strength
                                )
                            elif isinstance(value, str):
                                # For string values, use pattern if strong enough
                                if pattern.strength > 0.7:
                                    modified_action[key] = value
                        else:
                            # Add new attributes from pattern
                            modified_action[key] = value
                    
                    # Record pattern usage
                    pattern.last_used = datetime.now()
            
            return modified_action
            
        except Exception as e:
            self.logger.error(f"Failed to apply behavioral patterns: {str(e)}")
            return proposed_action

    async def _create_behavioral_pattern_from_observation(self, agent_id: str, 
                                                        observed_action: Dict[str, Any],
                                                        source: AdaptationStrategy) -> str:
        """Create a behavioral pattern from observing a successful action."""
        try:
            pattern_id = str(uuid.uuid4())
            
            # Extract trigger conditions from the situation
            trigger_conditions = {
                "action_type": observed_action.get("type", "unknown"),
                "context_similarity": 0.7  # How similar situation needs to be
            }
            
            # Create behavioral response (simplified)
            behavioral_response = {
                k: v for k, v in observed_action.items() 
                if k not in ["timestamp", "agent_id"]
            }
            
            # Determine associated rules (simplified)
            associated_rules = set(await self._get_relevant_rules_for_action(observed_action))
            
            pattern = BehavioralPattern(
                pattern_id=pattern_id,
                agent_id=agent_id,
                pattern_name=f"social_learned_{observed_action.get('type', 'action')}",
                trigger_conditions=trigger_conditions,
                behavioral_response=behavioral_response,
                associated_rules=associated_rules,
                strength=0.3,  # Start with moderate strength for observed patterns
                success_rate=0.8,  # Assume observed successful behavior has high success rate
                usage_count=0,
                last_used=None,
                created_at=datetime.now(),
                adaptation_source=source,
                metadata={"learned_from": "peer_observation"}
            )
            
            self.agent_patterns[agent_id].append(pattern)
            self.metrics["total_patterns"] += 1
            
            return pattern_id
            
        except Exception as e:
            self.logger.error(f"Failed to create pattern from observation: {str(e)}")
            return ""

    async def _create_avoidance_pattern(self, agent_id: str, violating_action: Dict[str, Any], 
                                      rule_id: str) -> str:
        """Create a pattern to avoid actions that violate rules."""
        try:
            pattern_id = str(uuid.uuid4())
            
            # Create trigger conditions that match the violating situation
            trigger_conditions = {
                "action_type": violating_action.get("type", "unknown"),
                "avoid": True  # This is an avoidance pattern
            }
            
            # Create alternative behavioral response
            behavioral_response = {}
            if "intensity" in violating_action:
                behavioral_response["intensity"] = max(0.1, violating_action["intensity"] * 0.5)
            if "target" in violating_action:
                behavioral_response["target"] = "alternative_target"
            
            # Add "think before acting" behavior
            behavioral_response["deliberation_time"] = 2.0  # Seconds to consider
            behavioral_response["rule_check"] = True
            
            pattern = BehavioralPattern(
                pattern_id=pattern_id,
                agent_id=agent_id,
                pattern_name=f"avoid_{violating_action.get('type', 'violation')}",
                trigger_conditions=trigger_conditions,
                behavioral_response=behavioral_response,
                associated_rules={rule_id},
                strength=0.8,  # High strength for avoidance patterns
                success_rate=0.9,  # Avoiding violations is usually successful
                usage_count=0,
                last_used=None,
                created_at=datetime.now(),
                adaptation_source=AdaptationStrategy.COGNITIVE_DISSONANCE,
                metadata={"avoidance_pattern": True, "triggered_by_rule": rule_id}
            )
            
            self.agent_patterns[agent_id].append(pattern)
            self.metrics["total_patterns"] += 1
            
            return pattern_id
            
        except Exception as e:
            self.logger.error(f"Failed to create avoidance pattern: {str(e)}")
            return ""

    # =====================================================
    # Helper Methods
    # =====================================================

    async def _update_adaptation_context(self, agent_id: str, situation: Dict[str, Any]) -> None:
        """Update adaptation context for an agent."""
        if agent_id not in self.adaptation_contexts:
            await self.initialize_agent_adaptation(agent_id)
        
        context = self.adaptation_contexts[agent_id]
        context.situation = situation
        context.timestamp = datetime.now()
        
        # Update social pressure based on community compliance
        community_compliance = self.metrics.get("average_compliance", 0.5)
        context.social_pressure = community_compliance * 0.8 + 0.2  # 0.2 to 1.0 range

    async def _get_relevant_rules_for_action(self, action: Dict[str, Any]) -> List[str]:
        """Get rules that are relevant to a specific action."""
        relevant_rules = []
        
        for rule_id, rule in self.compliance_monitor.active_rules.items():
            if not rule.is_active:
                continue
            
            # Simple relevance check based on action type and rule category
            action_type = action.get("type", "").lower()
            
            # Map action types to rule categories (simplified)
            category_mapping = {
                "social": [RuleCategory.SOCIAL, RuleCategory.BEHAVIORAL],
                "economic": [RuleCategory.ECONOMIC],
                "communication": [RuleCategory.SOCIAL, RuleCategory.BEHAVIORAL],
                "resource": [RuleCategory.ECONOMIC],
                "governance": [RuleCategory.PROCEDURAL, RuleCategory.CONSTITUTIONAL]
            }
            
            relevant_categories = category_mapping.get(action_type, [RuleCategory.BEHAVIORAL])
            
            if rule.category in relevant_categories:
                relevant_rules.append(rule_id)
        
        return relevant_rules

    async def _calculate_adaptation_factors(self, agent_id: str, relevant_rules: List[str]) -> Dict[str, Any]:
        """Calculate factors influencing behavioral adaptation."""
        factors = {
            "rule_internalization": {},
            "pattern_strength": 0.0,
            "social_influence": 0.0,
            "violation_sensitivity": 0.0
        }
        
        # Calculate average internalization for relevant rules
        total_internalization = 0.0
        for rule_id in relevant_rules:
            key = (agent_id, rule_id)
            if key in self.rule_internalization:
                internalization = self.rule_internalization[key]
                factors["rule_internalization"][rule_id] = internalization.internalization_level
                total_internalization += internalization.internalization_level
        
        if relevant_rules:
            factors["average_internalization"] = total_internalization / len(relevant_rules)
        
        # Calculate pattern strength
        relevant_patterns = [p for p in self.agent_patterns[agent_id] 
                           if any(r in relevant_rules for r in p.associated_rules)]
        if relevant_patterns:
            factors["pattern_strength"] = sum(p.strength for p in relevant_patterns) / len(relevant_patterns)
        
        # Get social influence
        if agent_id in self.adaptation_contexts:
            factors["social_influence"] = self.adaptation_contexts[agent_id].social_pressure
        
        return factors

    async def _calculate_adaptation_confidence(self, agent_id: str, relevant_rules: List[str]) -> float:
        """Calculate confidence in adaptation for current situation."""
        if not relevant_rules:
            return 1.0  # High confidence if no rules apply
        
        # Base confidence on internalization levels
        total_internalization = 0.0
        for rule_id in relevant_rules:
            key = (agent_id, rule_id)
            if key in self.rule_internalization:
                total_internalization += self.rule_internalization[key].internalization_level
        
        base_confidence = total_internalization / len(relevant_rules) if relevant_rules else 1.0
        
        # Adjust for pattern availability
        relevant_patterns = [p for p in self.agent_patterns[agent_id] 
                           if any(r in relevant_rules for r in p.associated_rules)]
        
        if relevant_patterns:
            pattern_confidence = sum(p.strength * p.success_rate for p in relevant_patterns) / len(relevant_patterns)
            base_confidence = (base_confidence + pattern_confidence) / 2
        
        return min(1.0, base_confidence)

    async def _calculate_reward_from_outcome(self, outcome: Dict[str, Any], violations: List[str]) -> float:
        """Calculate reward/penalty from action outcome."""
        reward = 0.0
        
        # Penalty for violations
        reward -= len(violations) * 0.3
        
        # Reward for successful outcomes
        if outcome.get("success", False):
            reward += 0.5
        
        # Additional rewards/penalties based on outcome details
        if "social_approval" in outcome:
            reward += outcome["social_approval"] * 0.2
        
        if "resource_gain" in outcome:
            reward += outcome["resource_gain"] * 0.1
        
        return max(-1.0, min(1.0, reward))  # Clamp to [-1, 1]

    async def _pattern_matches_action(self, pattern: BehavioralPattern, action: Dict[str, Any]) -> bool:
        """Check if a behavioral pattern matches an action."""
        # Simple matching based on action type
        pattern_type = pattern.trigger_conditions.get("action_type")
        action_type = action.get("type")
        
        return pattern_type == action_type

    async def _pattern_applies_to_situation(self, pattern: BehavioralPattern, action: Dict[str, Any]) -> bool:
        """Check if a behavioral pattern applies to current situation."""
        # Check if it's an avoidance pattern
        if pattern.trigger_conditions.get("avoid", False):
            # Apply avoidance patterns when action type matches
            return pattern.trigger_conditions.get("action_type") == action.get("type")
        
        # For regular patterns, check trigger conditions
        return await self._pattern_matches_action(pattern, action)

    async def _update_rule_internalization(self, agent_id: str, rule_id: str, 
                                         action: Dict[str, Any], outcome: Dict[str, Any], 
                                         reward: float) -> None:
        """Update rule internalization based on action outcome."""
        key = (agent_id, rule_id)
        if key not in self.rule_internalization:
            return
        
        internalization = self.rule_internalization[key]
        
        # Record the learning event
        learning_event = {
            "action": action.get("type", "unknown"),
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
            "outcome_success": outcome.get("success", False)
        }
        
        internalization.adaptation_history.append(learning_event)
        if len(internalization.adaptation_history) > 50:  # Keep only recent history
            internalization.adaptation_history = internalization.adaptation_history[-50:]

    async def _update_adaptation_phase(self, internalization: RuleInternalization) -> None:
        """Update the adaptation phase based on internalization level."""
        level = internalization.internalization_level
        
        if level < 0.2:
            internalization.adaptation_phase = AdaptationPhase.AWARENESS
        elif level < 0.4:
            internalization.adaptation_phase = AdaptationPhase.UNDERSTANDING
        elif level < 0.6:
            internalization.adaptation_phase = AdaptationPhase.ACCEPTANCE
        elif level < 0.8:
            internalization.adaptation_phase = AdaptationPhase.INTEGRATION
        elif level < 0.95:
            internalization.adaptation_phase = AdaptationPhase.INTERNALIZATION
        else:
            internalization.adaptation_phase = AdaptationPhase.AUTOMATIZATION

    async def _update_behavioral_patterns(self, agent_id: str, action: Dict[str, Any],
                                        outcome: Dict[str, Any], reward: float) -> None:
        """Update or create behavioral patterns based on action outcome."""
        # If action was successful and no violations, potentially create new pattern
        if reward > 0 and outcome.get("success", False):
            # Check if we already have a pattern for this type of action
            action_type = action.get("type", "unknown")
            existing_patterns = [p for p in self.agent_patterns[agent_id] 
                               if p.trigger_conditions.get("action_type") == action_type]
            
            if not existing_patterns and len(self.agent_patterns[agent_id]) < self.config["max_patterns_per_agent"]:
                # Create new successful pattern
                await self._create_behavioral_pattern_from_observation(
                    agent_id, action, AdaptationStrategy.REINFORCEMENT_LEARNING)

    # =====================================================
    # System Interface
    # =====================================================

    async def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive behavioral adaptation metrics."""
        try:
            total_agents = len(self.agent_patterns)
            total_internalizations = len(self.rule_internalization)
            
            # Calculate average internalization level
            if total_internalizations > 0:
                avg_internalization = sum(r.internalization_level for r in self.rule_internalization.values()) / total_internalizations
            else:
                avg_internalization = 0.0
            
            # Calculate adaptation phase distribution
            phase_distribution = {}
            for phase in AdaptationPhase:
                phase_distribution[phase.value] = len([r for r in self.rule_internalization.values() 
                                                     if r.adaptation_phase == phase])
            
            # Calculate pattern effectiveness
            total_patterns = sum(len(patterns) for patterns in self.agent_patterns.values())
            if total_patterns > 0:
                avg_pattern_strength = sum(sum(p.strength for p in patterns) for patterns in self.agent_patterns.values()) / total_patterns
                avg_success_rate = sum(sum(p.success_rate for p in patterns) for patterns in self.agent_patterns.values()) / total_patterns
            else:
                avg_pattern_strength = 0.0
                avg_success_rate = 0.0
            
            return {
                "total_agents_with_adaptation": total_agents,
                "total_rule_internalizations": total_internalizations,
                "average_internalization_level": avg_internalization,
                "adaptation_phase_distribution": phase_distribution,
                "total_behavioral_patterns": total_patterns,
                "average_pattern_strength": avg_pattern_strength,
                "average_pattern_success_rate": avg_success_rate,
                "successful_adaptations": self.metrics["successful_adaptations"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get adaptation metrics: {str(e)}")
            return {}


# Helper functions

def create_behavioral_adaptation_system(compliance_monitor: ComplianceMonitor,
                                      store_integration: MemoryStoreIntegration,
                                      postgres_persistence=None, 
                                      community_size: int = 50) -> BehavioralAdaptationSystem:
    """Create a BehavioralAdaptationSystem instance."""
    return BehavioralAdaptationSystem(compliance_monitor, store_integration, 
                                    postgres_persistence, community_size)


# Example usage
if __name__ == "__main__":
    async def test_behavioral_adaptation():
        """Test the Behavioral Adaptation System."""
        print("Behavioral Adaptation System loaded successfully")
        
    asyncio.run(test_behavioral_adaptation())