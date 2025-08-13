"""
File: rule_interpretation.py
Description: Natural language rule interpretation engine with contextual application.
Handles rule interpretation, contextual application, and conflict resolution for the constitutional system.
"""

import asyncio
import uuid
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

# Import constitutional storage
try:
    from .constitution_storage import (
        ConstitutionalStorage, ConstitutionalRule, RuleType, RuleStatus
    )
except ImportError:
    from dating_show.governance.constitution_storage import (
        ConstitutionalStorage, ConstitutionalRule, RuleType, RuleStatus
    )

try:
    from ..agents.memory_structures.store_integration import (
        MemoryStoreIntegration, StoreNamespace
    )
except ImportError:
    try:
        from dating_show.agents.memory_structures.store_integration import (
            MemoryStoreIntegration, StoreNamespace
        )
    except ImportError:
        class MemoryStoreIntegration:
            pass
        class StoreNamespace:
            GOVERNANCE = "governance"


class InterpretationContext(Enum):
    """Different contexts for rule interpretation."""
    AGENT_ACTION = "agent_action"  # Interpreting rules for agent actions
    DISPUTE_RESOLUTION = "dispute_resolution"  # Resolving conflicts between agents
    COMMUNITY_DECISION = "community_decision"  # Community-level decisions
    RESOURCE_ALLOCATION = "resource_allocation"  # Resource distribution decisions
    BEHAVIORAL_ASSESSMENT = "behavioral_assessment"  # Assessing agent behavior
    META_GOVERNANCE = "meta_governance"  # Rules about rules


class InterpretationResult(Enum):
    """Results of rule interpretation."""
    COMPLIANT = "compliant"  # Action/behavior is compliant with rules
    VIOLATION = "violation"  # Clear violation of rules
    AMBIGUOUS = "ambiguous"  # Rule application is unclear
    CONFLICTED = "conflicted"  # Multiple conflicting rules apply
    NOT_APPLICABLE = "not_applicable"  # Rules don't apply to this situation


@dataclass
class RuleInterpretation:
    """Result of interpreting a rule in a specific context."""
    interpretation_id: str
    rule_id: str
    context: InterpretationContext
    situation_description: str
    interpretation_result: InterpretationResult
    confidence_score: float  # 0.0 to 1.0
    reasoning: str
    applicable_clauses: List[str]
    exceptions_applied: List[str]
    contextual_factors: Dict[str, Any]
    precedent_cases: List[str]  # Similar past interpretations
    created_at: datetime
    interpreted_by: str  # Agent or system that made interpretation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleConflictResolution:
    """Resolution of conflicts between multiple rules."""
    resolution_id: str
    conflicting_rules: List[str]
    conflict_description: str
    resolution_strategy: str
    winning_rule: Optional[str]
    resolution_reasoning: str
    precedence_applied: bool
    temporal_factors: Dict[str, Any]
    stakeholder_impact: Dict[str, float]
    resolved_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuleInterpreter:
    """
    Natural language rule interpretation engine.
    Handles contextual rule application and conflict resolution.
    """

    def __init__(self, constitutional_storage: ConstitutionalStorage,
                 store_integration: MemoryStoreIntegration):
        """
        Initialize the Rule Interpreter.
        
        Args:
            constitutional_storage: Constitutional storage system
            store_integration: Store API integration
        """
        self.constitutional_storage = constitutional_storage
        self.store_integration = store_integration
        self.logger = logging.getLogger(f"{__name__}.RuleInterpreter")
        
        # Interpretation cache
        self.interpretation_cache = {}  # situation_hash -> RuleInterpretation
        self.conflict_resolutions = {}  # conflict_hash -> RuleConflictResolution
        
        # Precedent storage
        self.precedent_cases = {}  # rule_id -> List[interpretation_id]
        
        # Interpretation patterns and keywords
        self.interpretation_patterns = {
            # Action-related patterns
            "prohibitive": ["shall not", "must not", "prohibited", "forbidden", "banned"],
            "mandatory": ["shall", "must", "required", "mandatory", "obligated"],
            "permissive": ["may", "can", "allowed", "permitted", "optional"],
            "conditional": ["if", "when", "unless", "provided that", "subject to"],
            
            # Temporal patterns
            "temporal": ["during", "before", "after", "within", "by", "until"],
            "frequency": ["daily", "weekly", "always", "never", "once", "repeatedly"],
            
            # Scope patterns
            "universal": ["all", "every", "each", "any"],
            "specific": ["the", "this", "that", "such"],
            "quantitative": ["more than", "less than", "at least", "at most", "exactly"]
        }
        
        # Conflict resolution strategies
        self.resolution_strategies = {
            "precedence": self._resolve_by_precedence,
            "temporal": self._resolve_by_temporal_order,
            "specificity": self._resolve_by_specificity,
            "community_benefit": self._resolve_by_community_benefit,
            "least_harm": self._resolve_by_least_harm
        }
        
        # Metrics
        self.metrics = {
            "interpretations_performed": 0,
            "conflicts_resolved": 0,
            "cache_hit_rate": 0.0,
            "average_confidence": 0.0,
            "ambiguous_cases": 0
        }

    # =====================================================
    # Rule Interpretation
    # =====================================================

    async def interpret_rule_for_situation(self, rule_id: str, context: InterpretationContext,
                                         situation_description: str,
                                         contextual_factors: Dict[str, Any] = None,
                                         agent_id: str = "system") -> Optional[RuleInterpretation]:
        """
        Interpret a specific rule for a given situation.
        
        Args:
            rule_id: ID of the rule to interpret
            context: Context for interpretation
            situation_description: Description of the situation
            contextual_factors: Additional context factors
            agent_id: Agent requesting interpretation
        
        Returns:
            RuleInterpretation if successful, None otherwise
        """
        try:
            # Check cache first
            situation_hash = self._hash_situation(rule_id, context, situation_description)
            if situation_hash in self.interpretation_cache:
                cached_interpretation = self.interpretation_cache[situation_hash]
                # Check if cached interpretation is still valid (less than 1 hour old)
                if (datetime.now() - cached_interpretation.created_at).seconds < 3600:
                    self.metrics["cache_hit_rate"] = (
                        self.metrics["cache_hit_rate"] * self.metrics["interpretations_performed"] + 1.0
                    ) / (self.metrics["interpretations_performed"] + 1)
                    self.metrics["interpretations_performed"] += 1
                    return cached_interpretation
            
            # Retrieve the rule
            rule = await self.constitutional_storage.retrieve_rule(rule_id)
            if not rule:
                return None
            
            # Perform interpretation
            interpretation = await self._perform_rule_interpretation(
                rule, context, situation_description, contextual_factors or {}, agent_id
            )
            
            if interpretation:
                # Cache the interpretation
                self.interpretation_cache[situation_hash] = interpretation
                
                # Store in precedent database
                await self._store_precedent_case(interpretation)
                
                # Update metrics
                self.metrics["interpretations_performed"] += 1
                self.metrics["average_confidence"] = (
                    self.metrics["average_confidence"] * (self.metrics["interpretations_performed"] - 1) +
                    interpretation.confidence_score
                ) / self.metrics["interpretations_performed"]
                
                if interpretation.interpretation_result == InterpretationResult.AMBIGUOUS:
                    self.metrics["ambiguous_cases"] += 1
                
                self.logger.info(f"Interpreted rule {rule_id} for situation with confidence {interpretation.confidence_score:.2f}")
            
            return interpretation
            
        except Exception as e:
            self.logger.error(f"Failed to interpret rule {rule_id}: {str(e)}")
            return None

    async def interpret_multiple_rules(self, rule_ids: List[str], context: InterpretationContext,
                                     situation_description: str,
                                     contextual_factors: Dict[str, Any] = None,
                                     agent_id: str = "system") -> List[RuleInterpretation]:
        """
        Interpret multiple rules for a situation and detect conflicts.
        
        Args:
            rule_ids: List of rule IDs to interpret
            context: Context for interpretation
            situation_description: Description of the situation
            contextual_factors: Additional context factors
            agent_id: Agent requesting interpretation
        
        Returns:
            List of RuleInterpretation objects
        """
        try:
            interpretations = []
            
            # Interpret each rule
            for rule_id in rule_ids:
                interpretation = await self.interpret_rule_for_situation(
                    rule_id, context, situation_description, contextual_factors, agent_id
                )
                if interpretation:
                    interpretations.append(interpretation)
            
            # Check for conflicts
            conflicts = await self._detect_rule_conflicts(interpretations)
            if conflicts:
                # Resolve conflicts
                resolutions = await self._resolve_rule_conflicts(conflicts, context, situation_description)
                
                # Update interpretations based on conflict resolution
                for resolution in resolutions:
                    await self._apply_conflict_resolution(interpretations, resolution)
            
            return interpretations
            
        except Exception as e:
            self.logger.error(f"Failed to interpret multiple rules: {str(e)}")
            return []

    async def evaluate_compliance(self, agent_action: Dict[str, Any], 
                                context: InterpretationContext = InterpretationContext.AGENT_ACTION,
                                agent_id: str = "system") -> Dict[str, Any]:
        """
        Evaluate whether an agent action complies with constitutional rules.
        
        Args:
            agent_action: Dictionary describing the agent action
            context: Context for evaluation
            agent_id: Agent being evaluated
        
        Returns:
            Dictionary with compliance evaluation results
        """
        try:
            # Get all applicable rules
            applicable_rules = await self._get_applicable_rules(agent_action, context)
            
            # Interpret each applicable rule
            interpretations = []
            for rule in applicable_rules:
                interpretation = await self.interpret_rule_for_situation(
                    rule.rule_id,
                    context,
                    self._action_to_description(agent_action),
                    {"agent_id": agent_id, "action": agent_action},
                    agent_id
                )
                if interpretation:
                    interpretations.append(interpretation)
            
            # Aggregate compliance results
            compliance_result = self._aggregate_compliance_results(interpretations)
            
            # Store compliance evaluation
            await self._store_compliance_evaluation(agent_id, agent_action, compliance_result)
            
            return compliance_result
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate compliance: {str(e)}")
            return {"compliant": False, "error": str(e)}

    # =====================================================
    # Rule Conflict Resolution
    # =====================================================

    async def resolve_rule_conflict(self, conflicting_rules: List[str],
                                  conflict_description: str,
                                  context: InterpretationContext,
                                  stakeholders: List[str] = None) -> Optional[RuleConflictResolution]:
        """
        Resolve conflicts between multiple rules.
        
        Args:
            conflicting_rules: List of rule IDs in conflict
            conflict_description: Description of the conflict
            context: Context of the conflict
            stakeholders: Affected stakeholders
        
        Returns:
            RuleConflictResolution if successful, None otherwise
        """
        try:
            # Get the conflicting rules
            rules = []
            for rule_id in conflicting_rules:
                rule = await self.constitutional_storage.retrieve_rule(rule_id)
                if rule:
                    rules.append(rule)
            
            if len(rules) < 2:
                return None
            
            # Determine resolution strategy
            resolution_strategy = await self._determine_resolution_strategy(rules, context)
            
            # Apply resolution strategy
            resolution_func = self.resolution_strategies.get(resolution_strategy)
            if not resolution_func:
                resolution_func = self.resolution_strategies["precedence"]
            
            resolution = await resolution_func(rules, conflict_description, context, stakeholders or [])
            
            if resolution:
                # Store the resolution
                conflict_hash = self._hash_conflict(conflicting_rules, conflict_description)
                self.conflict_resolutions[conflict_hash] = resolution
                
                await self._store_conflict_resolution(resolution)
                
                self.metrics["conflicts_resolved"] += 1
                self.logger.info(f"Resolved rule conflict between {len(conflicting_rules)} rules using {resolution_strategy}")
            
            return resolution
            
        except Exception as e:
            self.logger.error(f"Failed to resolve rule conflict: {str(e)}")
            return None

    # =====================================================
    # Private Helper Methods
    # =====================================================

    async def _perform_rule_interpretation(self, rule: ConstitutionalRule,
                                         context: InterpretationContext,
                                         situation_description: str,
                                         contextual_factors: Dict[str, Any],
                                         agent_id: str) -> Optional[RuleInterpretation]:
        """Perform the actual rule interpretation logic."""
        try:
            # Analyze rule content for interpretation patterns
            rule_analysis = self._analyze_rule_content(rule.content)
            
            # Determine applicability
            applicability = self._determine_rule_applicability(
                rule, situation_description, contextual_factors, context
            )
            
            if not applicability["applicable"]:
                return RuleInterpretation(
                    interpretation_id=str(uuid.uuid4())[:8],
                    rule_id=rule.rule_id,
                    context=context,
                    situation_description=situation_description,
                    interpretation_result=InterpretationResult.NOT_APPLICABLE,
                    confidence_score=applicability["confidence"],
                    reasoning=applicability["reasoning"],
                    applicable_clauses=[],
                    exceptions_applied=[],
                    contextual_factors=contextual_factors,
                    precedent_cases=[],
                    created_at=datetime.now(),
                    interpreted_by=agent_id
                )
            
            # Determine interpretation result
            interpretation_result, confidence, reasoning = self._determine_interpretation_result(
                rule, rule_analysis, situation_description, contextual_factors, context
            )
            
            # Find similar precedent cases
            precedent_cases = await self._find_precedent_cases(rule.rule_id, context, situation_description)
            
            # Identify applicable clauses and exceptions
            applicable_clauses = self._identify_applicable_clauses(rule.content, situation_description)
            exceptions_applied = self._identify_exceptions(rule.content, situation_description, contextual_factors)
            
            return RuleInterpretation(
                interpretation_id=str(uuid.uuid4())[:8],
                rule_id=rule.rule_id,
                context=context,
                situation_description=situation_description,
                interpretation_result=interpretation_result,
                confidence_score=confidence,
                reasoning=reasoning,
                applicable_clauses=applicable_clauses,
                exceptions_applied=exceptions_applied,
                contextual_factors=contextual_factors,
                precedent_cases=precedent_cases,
                created_at=datetime.now(),
                interpreted_by=agent_id
            )
            
        except Exception as e:
            self.logger.error(f"Failed to perform rule interpretation: {str(e)}")
            return None

    def _analyze_rule_content(self, rule_content: str) -> Dict[str, Any]:
        """Analyze rule content for linguistic patterns and structure."""
        analysis = {
            "patterns_found": [],
            "rule_type_indicators": [],
            "conditional_clauses": [],
            "exception_clauses": [],
            "temporal_constraints": [],
            "scope_indicators": []
        }
        
        content_lower = rule_content.lower()
        
        # Check for different pattern types
        for pattern_type, patterns in self.interpretation_patterns.items():
            for pattern in patterns:
                if pattern in content_lower:
                    analysis["patterns_found"].append({
                        "type": pattern_type,
                        "pattern": pattern,
                        "position": content_lower.find(pattern)
                    })
        
        # Identify conditional clauses
        conditional_markers = ["if", "when", "unless", "provided that", "subject to"]
        for marker in conditional_markers:
            if marker in content_lower:
                # Find the clause containing this marker
                sentences = rule_content.split('.')
                for sentence in sentences:
                    if marker in sentence.lower():
                        analysis["conditional_clauses"].append(sentence.strip())
        
        # Identify exception clauses
        exception_markers = ["except", "unless", "excluding", "but not", "however"]
        for marker in exception_markers:
            if marker in content_lower:
                sentences = rule_content.split('.')
                for sentence in sentences:
                    if marker in sentence.lower():
                        analysis["exception_clauses"].append(sentence.strip())
        
        return analysis

    def _determine_rule_applicability(self, rule: ConstitutionalRule,
                                    situation_description: str,
                                    contextual_factors: Dict[str, Any],
                                    context: InterpretationContext) -> Dict[str, Any]:
        """Determine if a rule applies to the given situation."""
        # Simplified applicability logic
        situation_lower = situation_description.lower()
        rule_content_lower = rule.content.lower()
        
        # Check for keyword overlap
        situation_words = set(situation_lower.split())
        rule_words = set(rule_content_lower.split())
        
        common_words = situation_words.intersection(rule_words)
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        meaningful_overlap = common_words - stop_words
        
        overlap_score = len(meaningful_overlap) / max(len(situation_words), len(rule_words))
        
        # Context matching
        context_match_score = 0.0
        context_keywords = {
            InterpretationContext.AGENT_ACTION: ["action", "behavior", "act", "do"],
            InterpretationContext.DISPUTE_RESOLUTION: ["dispute", "conflict", "disagreement"],
            InterpretationContext.RESOURCE_ALLOCATION: ["resource", "allocation", "distribution"],
            # Add more context mappings as needed
        }
        
        if context in context_keywords:
            for keyword in context_keywords[context]:
                if keyword in rule_content_lower:
                    context_match_score += 0.2
        
        # Rule type matching
        type_match_score = 0.0
        if context == InterpretationContext.AGENT_ACTION and rule.rule_type == RuleType.BEHAVIORAL_NORM:
            type_match_score = 0.3
        elif context == InterpretationContext.RESOURCE_ALLOCATION and rule.rule_type == RuleType.RESOURCE_ALLOCATION:
            type_match_score = 0.3
        
        overall_score = min(1.0, overlap_score + context_match_score + type_match_score)
        
        applicable = overall_score > 0.3  # Threshold for applicability
        
        reasoning = f"Keyword overlap: {overlap_score:.2f}, Context match: {context_match_score:.2f}, Type match: {type_match_score:.2f}"
        
        return {
            "applicable": applicable,
            "confidence": overall_score,
            "reasoning": reasoning,
            "overlap_score": overlap_score
        }

    def _determine_interpretation_result(self, rule: ConstitutionalRule,
                                       rule_analysis: Dict[str, Any],
                                       situation_description: str,
                                       contextual_factors: Dict[str, Any],
                                       context: InterpretationContext) -> Tuple[InterpretationResult, float, str]:
        """Determine the interpretation result for a rule."""
        # Simplified interpretation logic
        prohibitive_patterns = [p for p in rule_analysis["patterns_found"] if p["type"] == "prohibitive"]
        mandatory_patterns = [p for p in rule_analysis["patterns_found"] if p["type"] == "mandatory"]
        permissive_patterns = [p for p in rule_analysis["patterns_found"] if p["type"] == "permissive"]
        conditional_patterns = [p for p in rule_analysis["patterns_found"] if p["type"] == "conditional"]
        
        confidence = 0.8  # Default confidence
        
        # Determine result based on pattern analysis
        if prohibitive_patterns and not conditional_patterns:
            # Clear prohibition
            return InterpretationResult.VIOLATION, confidence, "Rule contains clear prohibitive language"
        
        elif mandatory_patterns and not conditional_patterns:
            # Clear requirement
            return InterpretationResult.COMPLIANT, confidence, "Rule contains mandatory requirements that appear to be met"
        
        elif permissive_patterns:
            # Permissive rule
            return InterpretationResult.COMPLIANT, confidence * 0.9, "Rule is permissive in nature"
        
        elif conditional_patterns:
            # Conditional rule - needs more analysis
            return InterpretationResult.AMBIGUOUS, confidence * 0.6, "Rule contains conditional clauses requiring further analysis"
        
        else:
            # Unclear
            return InterpretationResult.AMBIGUOUS, confidence * 0.5, "Rule language is ambiguous in this context"

    def _identify_applicable_clauses(self, rule_content: str, situation_description: str) -> List[str]:
        """Identify which clauses of a rule apply to the situation."""
        # Split rule into sentences/clauses
        sentences = [s.strip() for s in rule_content.split('.') if s.strip()]
        
        applicable_clauses = []
        situation_words = set(situation_description.lower().split())
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(situation_words.intersection(sentence_words))
            
            if overlap > 2:  # At least 3 common words
                applicable_clauses.append(sentence)
        
        return applicable_clauses

    def _identify_exceptions(self, rule_content: str, situation_description: str,
                           contextual_factors: Dict[str, Any]) -> List[str]:
        """Identify any exceptions that apply to the situation."""
        # Look for exception clauses in the rule
        exception_markers = ["except", "unless", "excluding", "but not", "however", "provided that"]
        exceptions_applied = []
        
        for marker in exception_markers:
            if marker in rule_content.lower():
                # Extract the exception clause
                sentences = rule_content.split('.')
                for sentence in sentences:
                    if marker in sentence.lower():
                        exceptions_applied.append(sentence.strip())
        
        return exceptions_applied

    async def _detect_rule_conflicts(self, interpretations: List[RuleInterpretation]) -> List[List[str]]:
        """Detect conflicts between rule interpretations."""
        conflicts = []
        
        # Look for contradictory results
        for i, interp1 in enumerate(interpretations):
            for j, interp2 in enumerate(interpretations[i+1:], i+1):
                if self._interpretations_conflict(interp1, interp2):
                    conflicts.append([interp1.rule_id, interp2.rule_id])
        
        return conflicts

    def _interpretations_conflict(self, interp1: RuleInterpretation, interp2: RuleInterpretation) -> bool:
        """Check if two interpretations conflict with each other."""
        # Simplified conflict detection
        conflicting_pairs = [
            (InterpretationResult.COMPLIANT, InterpretationResult.VIOLATION),
            (InterpretationResult.VIOLATION, InterpretationResult.COMPLIANT)
        ]
        
        result_pair = (interp1.interpretation_result, interp2.interpretation_result)
        return result_pair in conflicting_pairs or result_pair[::-1] in conflicting_pairs

    async def _resolve_rule_conflicts(self, conflicts: List[List[str]], 
                                    context: InterpretationContext,
                                    situation_description: str) -> List[RuleConflictResolution]:
        """Resolve detected rule conflicts."""
        resolutions = []
        
        for conflict in conflicts:
            resolution = await self.resolve_rule_conflict(
                conflict, f"Conflict in {context.value} context", context
            )
            if resolution:
                resolutions.append(resolution)
        
        return resolutions

    # Conflict resolution strategy implementations
    async def _resolve_by_precedence(self, rules: List[ConstitutionalRule],
                                   conflict_description: str,
                                   context: InterpretationContext,
                                   stakeholders: List[str]) -> Optional[RuleConflictResolution]:
        """Resolve conflict by rule precedence level."""
        # Find rule with highest precedence
        winning_rule = max(rules, key=lambda r: r.precedence_level)
        
        return RuleConflictResolution(
            resolution_id=str(uuid.uuid4())[:8],
            conflicting_rules=[r.rule_id for r in rules],
            conflict_description=conflict_description,
            resolution_strategy="precedence",
            winning_rule=winning_rule.rule_id,
            resolution_reasoning=f"Rule {winning_rule.rule_id} takes precedence with level {winning_rule.precedence_level}",
            precedence_applied=True,
            temporal_factors={},
            stakeholder_impact={},
            resolved_at=datetime.now()
        )

    async def _resolve_by_temporal_order(self, rules: List[ConstitutionalRule],
                                       conflict_description: str,
                                       context: InterpretationContext,
                                       stakeholders: List[str]) -> Optional[RuleConflictResolution]:
        """Resolve conflict by temporal order (newer rules take precedence)."""
        # Find most recent rule
        winning_rule = max(rules, key=lambda r: r.effective_date)
        
        return RuleConflictResolution(
            resolution_id=str(uuid.uuid4())[:8],
            conflicting_rules=[r.rule_id for r in rules],
            conflict_description=conflict_description,
            resolution_strategy="temporal",
            winning_rule=winning_rule.rule_id,
            resolution_reasoning=f"Rule {winning_rule.rule_id} is more recent (effective {winning_rule.effective_date})",
            precedence_applied=False,
            temporal_factors={"winning_rule_date": winning_rule.effective_date.isoformat()},
            stakeholder_impact={},
            resolved_at=datetime.now()
        )

    async def _resolve_by_specificity(self, rules: List[ConstitutionalRule],
                                    conflict_description: str,
                                    context: InterpretationContext,
                                    stakeholders: List[str]) -> Optional[RuleConflictResolution]:
        """Resolve conflict by rule specificity (more specific rules win)."""
        # Simplified specificity calculation based on content length and detail
        winning_rule = max(rules, key=lambda r: len(r.content.split()))
        
        return RuleConflictResolution(
            resolution_id=str(uuid.uuid4())[:8],
            conflicting_rules=[r.rule_id for r in rules],
            conflict_description=conflict_description,
            resolution_strategy="specificity",
            winning_rule=winning_rule.rule_id,
            resolution_reasoning=f"Rule {winning_rule.rule_id} is more specific",
            precedence_applied=False,
            temporal_factors={},
            stakeholder_impact={},
            resolved_at=datetime.now()
        )

    # Additional helper methods
    async def _determine_resolution_strategy(self, rules: List[ConstitutionalRule],
                                           context: InterpretationContext) -> str:
        """Determine the best resolution strategy for conflicting rules."""
        # Check if rules have different precedence levels
        precedence_levels = [r.precedence_level for r in rules]
        if len(set(precedence_levels)) > 1:
            return "precedence"
        
        # Check if rules have different effective dates
        effective_dates = [r.effective_date for r in rules]
        if len(set(effective_dates)) > 1:
            return "temporal"
        
        # Default to specificity
        return "specificity"

    def _hash_situation(self, rule_id: str, context: InterpretationContext, situation: str) -> str:
        """Create a hash for caching situation interpretations."""
        content = f"{rule_id}_{context.value}_{situation[:100]}"
        return str(hash(content))

    def _hash_conflict(self, rule_ids: List[str], description: str) -> str:
        """Create a hash for caching conflict resolutions."""
        content = f"{'_'.join(sorted(rule_ids))}_{description[:50]}"
        return str(hash(content))

    def _action_to_description(self, action: Dict[str, Any]) -> str:
        """Convert an action dictionary to a description string."""
        return f"Agent performs {action.get('type', 'action')} involving {action.get('target', 'unknown')}"

    # Additional placeholder methods for completeness
    async def _get_applicable_rules(self, action: Dict[str, Any], context: InterpretationContext) -> List[ConstitutionalRule]:
        """Get rules applicable to a specific action."""
        # Placeholder - would implement logic to find relevant rules
        all_rules = await self.constitutional_storage.get_all_active_rules()
        return list(all_rules.values())[:5]  # Return first 5 for testing

    def _aggregate_compliance_results(self, interpretations: List[RuleInterpretation]) -> Dict[str, Any]:
        """Aggregate multiple rule interpretation results."""
        if not interpretations:
            return {"compliant": True, "confidence": 0.0, "details": []}
        
        violations = [i for i in interpretations if i.interpretation_result == InterpretationResult.VIOLATION]
        compliant = [i for i in interpretations if i.interpretation_result == InterpretationResult.COMPLIANT]
        
        overall_compliant = len(violations) == 0
        avg_confidence = sum(i.confidence_score for i in interpretations) / len(interpretations)
        
        return {
            "compliant": overall_compliant,
            "confidence": avg_confidence,
            "violations": len(violations),
            "compliant_rules": len(compliant),
            "details": [{"rule_id": i.rule_id, "result": i.interpretation_result.value, 
                        "confidence": i.confidence_score} for i in interpretations]
        }

    # Storage and retrieval methods
    async def _store_precedent_case(self, interpretation: RuleInterpretation) -> None:
        """Store interpretation as precedent case."""
        try:
            if self.store_integration.store:
                precedent_data = asdict(interpretation)
                precedent_data["created_at"] = interpretation.created_at.isoformat()
                precedent_data["context"] = interpretation.context.value
                precedent_data["interpretation_result"] = interpretation.interpretation_result.value
                
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"precedent_{interpretation.interpretation_id}",
                    precedent_data
                )
        except Exception as e:
            self.logger.error(f"Failed to store precedent case: {str(e)}")

    async def _find_precedent_cases(self, rule_id: str, context: InterpretationContext,
                                  situation: str) -> List[str]:
        """Find similar precedent cases."""
        # Placeholder - would implement similarity search
        return []

    async def _store_conflict_resolution(self, resolution: RuleConflictResolution) -> None:
        """Store conflict resolution."""
        try:
            if self.store_integration.store:
                resolution_data = asdict(resolution)
                resolution_data["resolved_at"] = resolution.resolved_at.isoformat()
                
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"conflict_resolution_{resolution.resolution_id}",
                    resolution_data
                )
        except Exception as e:
            self.logger.error(f"Failed to store conflict resolution: {str(e)}")

    async def _store_compliance_evaluation(self, agent_id: str, action: Dict[str, Any],
                                         result: Dict[str, Any]) -> None:
        """Store compliance evaluation result."""
        try:
            if self.store_integration.store:
                evaluation_data = {
                    "agent_id": agent_id,
                    "action": action,
                    "result": result,
                    "evaluated_at": datetime.now().isoformat()
                }
                
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"compliance_evaluation_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    evaluation_data
                )
        except Exception as e:
            self.logger.error(f"Failed to store compliance evaluation: {str(e)}")

    async def _apply_conflict_resolution(self, interpretations: List[RuleInterpretation],
                                       resolution: RuleConflictResolution) -> None:
        """Apply conflict resolution to interpretations."""
        # Update interpretations based on resolution
        for interpretation in interpretations:
            if interpretation.rule_id in resolution.conflicting_rules:
                if interpretation.rule_id != resolution.winning_rule:
                    # Mark as superseded by conflict resolution
                    interpretation.metadata["superseded_by_resolution"] = resolution.resolution_id
                    interpretation.metadata["resolution_reasoning"] = resolution.resolution_reasoning

    # Placeholder methods for strategies not yet implemented
    async def _resolve_by_community_benefit(self, rules: List[ConstitutionalRule],
                                          conflict_description: str,
                                          context: InterpretationContext,
                                          stakeholders: List[str]) -> Optional[RuleConflictResolution]:
        """Resolve by community benefit (placeholder)."""
        return await self._resolve_by_precedence(rules, conflict_description, context, stakeholders)

    async def _resolve_by_least_harm(self, rules: List[ConstitutionalRule],
                                   conflict_description: str,
                                   context: InterpretationContext,
                                   stakeholders: List[str]) -> Optional[RuleConflictResolution]:
        """Resolve by least harm principle (placeholder)."""
        return await self._resolve_by_precedence(rules, conflict_description, context, stakeholders)


# Helper functions
def create_rule_interpreter(constitutional_storage: ConstitutionalStorage,
                          store_integration: MemoryStoreIntegration) -> RuleInterpreter:
    """Create a RuleInterpreter instance."""
    return RuleInterpreter(constitutional_storage, store_integration)