"""
Reputation System for Enhanced PIANO Architecture
Phase 3: Advanced Features - Week 9: Complex Social Dynamics

This module implements a sophisticated reputation management system that tracks
and calculates agent reputations across multiple dimensions, enabling reputation-based
decision making and social capital tracking.
"""

import json
import math
import time
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from statistics import mean, stdev


class ReputationDimension(Enum):
    """Different dimensions of reputation"""
    TRUSTWORTHINESS = "trustworthiness"
    COMPETENCE = "competence"
    LIKABILITY = "likability"
    LEADERSHIP = "leadership"
    COOPERATION = "cooperation"
    RELIABILITY = "reliability"
    INNOVATION = "innovation"
    HONESTY = "honesty"
    SOCIAL_INFLUENCE = "social_influence"
    CONFLICT_RESOLUTION = "conflict_resolution"


class ReputationEvent(Enum):
    """Types of events that affect reputation"""
    PROMISE_KEPT = "promise_kept"
    PROMISE_BROKEN = "promise_broken"
    SUCCESSFUL_COLLABORATION = "successful_collaboration"
    FAILED_COLLABORATION = "failed_collaboration"
    HELP_PROVIDED = "help_provided"
    HELP_REFUSED = "help_refused"
    LEADERSHIP_SUCCESS = "leadership_success"
    LEADERSHIP_FAILURE = "leadership_failure"
    INNOVATION_SUCCESS = "innovation_success"
    INNOVATION_FAILURE = "innovation_failure"
    HONEST_DISCLOSURE = "honest_disclosure"
    DECEPTION_DETECTED = "deception_detected"
    CONFLICT_RESOLVED = "conflict_resolved"
    CONFLICT_ESCALATED = "conflict_escalated"
    RESOURCE_SHARED = "resource_shared"
    RESOURCE_HOARDED = "resource_hoarded"


@dataclass
class ReputationRecord:
    """Record of a reputation-affecting event"""
    event_id: str
    agent_id: str
    event_type: ReputationEvent
    dimensions_affected: Dict[ReputationDimension, float]  # dimension -> impact (-1.0 to 1.0)
    observer_id: str  # Who observed/reported this event
    witness_ids: Set[str]  # Other witnesses to the event
    context: Dict[str, Any]  # Additional context about the event
    timestamp: float
    verified: bool  # Whether the event has been verified by multiple sources
    weight: float  # How much this event should impact reputation (0.0 to 1.0)
    decay_rate: float  # How quickly this event's impact should decay over time
    
    def __post_init__(self):
        """Validate reputation record data"""
        self.weight = max(0.0, min(1.0, self.weight))
        self.decay_rate = max(0.0, min(1.0, self.decay_rate))
        for dimension, impact in self.dimensions_affected.items():
            self.dimensions_affected[dimension] = max(-1.0, min(1.0, impact))


@dataclass
class ReputationScore:
    """Reputation score for an agent in a specific dimension"""
    agent_id: str
    dimension: ReputationDimension
    score: float  # -1.0 to 1.0 (negative to positive reputation)
    confidence: float  # 0.0 to 1.0 (how confident we are in this score)
    num_observations: int
    last_updated: float
    trend: float  # Recent trend (-1.0 to 1.0, negative = declining, positive = improving)
    
    def __post_init__(self):
        """Validate reputation score data"""
        self.score = max(-1.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.trend = max(-1.0, min(1.0, self.trend))


class ReputationSystem:
    """
    Comprehensive reputation management system for tracking and calculating
    agent reputations across multiple dimensions.
    
    Features:
    - Multi-dimensional reputation tracking
    - Event-based reputation updates with decay
    - Observer credibility weighting
    - Reputation-based decision making support
    - Social capital calculation
    - Reputation network effects
    - Fraud detection and verification
    """
    
    def __init__(self, max_agents: int = 1000, max_records_per_agent: int = 1000):
        self.max_agents = max_agents
        self.max_records_per_agent = max_records_per_agent
        
        # Core data structures
        self.agents: Set[str] = set()
        self.reputation_scores: Dict[str, Dict[ReputationDimension, ReputationScore]] = defaultdict(dict)
        self.reputation_records: Dict[str, List[ReputationRecord]] = defaultdict(list)
        self.observer_credibility: Dict[str, float] = defaultdict(lambda: 0.5)  # Default credibility
        
        # Social capital tracking
        self.social_capital: Dict[str, float] = defaultdict(float)
        self.influence_network: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Caching and performance
        self.lock = threading.RLock()
        self._reputation_cache: Dict[Tuple[str, str], float] = {}  # (agent, observer) -> reputation
        self._social_capital_cache: Dict[str, float] = {}
        self._cache_timestamp = 0.0
        self._cache_ttl = 300.0  # 5 minutes
        
        # Configuration
        self.dimension_weights = {
            ReputationDimension.TRUSTWORTHINESS: 1.0,
            ReputationDimension.COMPETENCE: 0.9,
            ReputationDimension.LIKABILITY: 0.7,
            ReputationDimension.LEADERSHIP: 0.8,
            ReputationDimension.COOPERATION: 0.9,
            ReputationDimension.RELIABILITY: 1.0,
            ReputationDimension.INNOVATION: 0.6,
            ReputationDimension.HONESTY: 1.0,
            ReputationDimension.SOCIAL_INFLUENCE: 0.5,
            ReputationDimension.CONFLICT_RESOLUTION: 0.8
        }
        
        self.event_impact_map = self._initialize_event_impact_map()
    
    def add_agent(self, agent_id: str) -> bool:
        """Add an agent to the reputation system"""
        with self.lock:
            if len(self.agents) >= self.max_agents:
                return False
            
            self.agents.add(agent_id)
            
            # Initialize reputation scores for all dimensions
            for dimension in ReputationDimension:
                self.reputation_scores[agent_id][dimension] = ReputationScore(
                    agent_id=agent_id,
                    dimension=dimension,
                    score=0.0,
                    confidence=0.0,
                    num_observations=0,
                    last_updated=time.time(),
                    trend=0.0
                )
            
            # Initialize social capital
            self.social_capital[agent_id] = 0.0
            self.observer_credibility[agent_id] = 0.5
            
            self._invalidate_cache()
            return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the reputation system"""
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            # Remove agent's reputation data
            self.agents.remove(agent_id)
            self.reputation_scores.pop(agent_id, None)
            self.reputation_records.pop(agent_id, None)
            self.social_capital.pop(agent_id, None)
            self.observer_credibility.pop(agent_id, None)
            
            # Remove agent from influence networks
            self.influence_network.pop(agent_id, None)
            for other_agent in self.influence_network:
                self.influence_network[other_agent].pop(agent_id, None)
            
            # Remove agent from reputation records as observer/witness
            for other_agent in self.reputation_records:
                records_to_remove = []
                for i, record in enumerate(self.reputation_records[other_agent]):
                    if record.observer_id == agent_id or agent_id in record.witness_ids:
                        records_to_remove.append(i)
                
                # Remove in reverse order to maintain indices
                for i in reversed(records_to_remove):
                    del self.reputation_records[other_agent][i]
            
            self._invalidate_cache()
            return True
    
    def record_reputation_event(
        self,
        agent_id: str,
        event_type: ReputationEvent,
        observer_id: str,
        witness_ids: Optional[Set[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        custom_impacts: Optional[Dict[ReputationDimension, float]] = None,
        weight: float = 1.0
    ) -> str:
        """Record a reputation-affecting event"""
        # Input validation
        if not isinstance(agent_id, str) or not agent_id.strip():
            return ""
        if not isinstance(observer_id, str) or not observer_id.strip():
            return ""
        if not isinstance(event_type, ReputationEvent):
            return ""
        if weight < 0.0 or weight > 1.0:
            weight = max(0.0, min(1.0, weight))
        
        with self.lock:
            if agent_id not in self.agents or observer_id not in self.agents:
                return ""
            
            witness_ids = witness_ids or set()
            context = context or {}
            
            # Get impact on different reputation dimensions
            if custom_impacts:
                dimensions_affected = custom_impacts.copy()
            else:
                dimensions_affected = self.event_impact_map.get(event_type, {}).copy()
            
            # Create reputation record
            event_id = f"{agent_id}_{event_type.value}_{int(time.time() * 1000)}"
            record = ReputationRecord(
                event_id=event_id,
                agent_id=agent_id,
                event_type=event_type,
                dimensions_affected=dimensions_affected,
                observer_id=observer_id,
                witness_ids=witness_ids,
                context=context,
                timestamp=time.time(),
                verified=len(witness_ids) > 0,  # Verified if there are witnesses
                weight=weight,
                decay_rate=0.1  # Default decay rate
            )
            
            # Add to reputation records
            self.reputation_records[agent_id].append(record)
            
            # Maintain record limit per agent
            if len(self.reputation_records[agent_id]) > self.max_records_per_agent:
                # Remove oldest records
                self.reputation_records[agent_id] = self.reputation_records[agent_id][-self.max_records_per_agent:]
            
            # Update reputation scores
            self._update_reputation_scores(agent_id)
            
            # Update observer credibility based on witness verification
            self._update_observer_credibility(observer_id, record)
            
            # Update social capital
            self._update_social_capital(agent_id)
            
            self._invalidate_cache()
            return event_id
    
    def get_reputation_score(
        self, 
        agent_id: str, 
        dimension: Optional[ReputationDimension] = None,
        observer_perspective: Optional[str] = None
    ) -> Union[float, Dict[ReputationDimension, float]]:
        """Get reputation score(s) for an agent"""
        if agent_id not in self.agents:
            return 0.0 if dimension else {}
        
        # Check cache if observer perspective is provided
        if observer_perspective:
            cache_key = (agent_id, observer_perspective)
            if self._is_cache_valid() and cache_key in self._reputation_cache:
                cached_score = self._reputation_cache[cache_key]
                if dimension:
                    return cached_score
                else:
                    # For overall reputation, return all dimensions
                    return {dim: cached_score for dim in ReputationDimension}
        
        if dimension:
            # Single dimension
            score_obj = self.reputation_scores[agent_id].get(dimension)
            if not score_obj:
                return 0.0
            
            base_score = score_obj.score
            
            # Adjust based on observer perspective if provided
            if observer_perspective and observer_perspective in self.agents:
                adjusted_score = self._adjust_score_for_observer(agent_id, dimension, observer_perspective)
                if observer_perspective:
                    self._reputation_cache[(agent_id, observer_perspective)] = adjusted_score
                return adjusted_score
            
            return base_score
        else:
            # All dimensions
            scores = {}
            for dim in ReputationDimension:
                scores[dim] = self.get_reputation_score(agent_id, dim, observer_perspective)
            return scores
    
    def get_overall_reputation(self, agent_id: str, observer_perspective: Optional[str] = None) -> float:
        """Calculate overall reputation score weighted by dimension importance"""
        if agent_id not in self.agents:
            return 0.0
        
        dimension_scores = self.get_reputation_score(agent_id, observer_perspective=observer_perspective)
        if not isinstance(dimension_scores, dict):
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = self.dimension_weights.get(dimension, 0.5)
            confidence = self.reputation_scores[agent_id][dimension].confidence
            
            # Weight by both importance and confidence
            effective_weight = weight * confidence
            weighted_sum += score * effective_weight
            total_weight += effective_weight
        
        return weighted_sum / max(total_weight, 0.01)  # Avoid division by zero
    
    def get_social_capital(self, agent_id: str) -> float:
        """Get social capital score for an agent"""
        if agent_id not in self.agents:
            return 0.0
        
        if self._is_cache_valid() and agent_id in self._social_capital_cache:
            return self._social_capital_cache[agent_id]
        
        capital = self.social_capital[agent_id]
        self._social_capital_cache[agent_id] = capital
        return capital
    
    def calculate_trust_score(self, agent_a: str, agent_b: str) -> float:
        """Calculate how much agent_a should trust agent_b based on reputation"""
        if agent_a not in self.agents or agent_b not in self.agents:
            return 0.0
        
        # Get reputation from A's perspective
        trustworthiness = self.get_reputation_score(agent_b, ReputationDimension.TRUSTWORTHINESS, agent_a)
        reliability = self.get_reputation_score(agent_b, ReputationDimension.RELIABILITY, agent_a)
        honesty = self.get_reputation_score(agent_b, ReputationDimension.HONESTY, agent_a)
        
        # Weight the components
        trust_score = (trustworthiness * 0.4 + reliability * 0.3 + honesty * 0.3)
        
        # Adjust based on direct interaction history
        interaction_bonus = self._calculate_interaction_bonus(agent_a, agent_b)
        trust_score += interaction_bonus
        
        return max(-1.0, min(1.0, trust_score))
    
    def recommend_collaboration_partners(
        self, 
        agent_id: str, 
        task_requirements: Dict[ReputationDimension, float],
        exclude_agents: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """Recommend agents for collaboration based on reputation and requirements"""
        if agent_id not in self.agents:
            return []
        
        exclude_agents = exclude_agents or set()
        exclude_agents.add(agent_id)  # Don't recommend self
        
        candidates = []
        
        for other_agent in self.agents:
            if other_agent in exclude_agents:
                continue
            
            # Calculate match score based on requirements
            match_score = 0.0
            total_weight = 0.0
            
            for dimension, required_level in task_requirements.items():
                agent_score = self.get_reputation_score(other_agent, dimension, agent_id)
                confidence = self.reputation_scores[other_agent][dimension].confidence
                
                # How well does this agent meet the requirement?
                if required_level > 0:
                    # We need someone good at this
                    dimension_match = min(1.0, agent_score / required_level) if required_level > 0 else 1.0
                else:
                    # We need someone who isn't bad at this
                    dimension_match = 1.0 if agent_score >= required_level else 0.0
                
                weight = confidence
                match_score += dimension_match * weight
                total_weight += weight
            
            if total_weight > 0:
                final_score = match_score / total_weight
                
                # Bonus for high social capital
                social_bonus = self.get_social_capital(other_agent) * 0.1
                final_score += social_bonus
                
                # Bonus for existing positive relationship
                trust_bonus = self.calculate_trust_score(agent_id, other_agent) * 0.2
                final_score += trust_bonus
                
                candidates.append((other_agent, final_score))
        
        # Sort by match score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def detect_reputation_fraud(self, agent_id: str, threshold: float = 0.8) -> List[str]:
        """Detect potential reputation fraud or manipulation"""
        if agent_id not in self.agents:
            return []
        
        suspicious_events = []
        records = self.reputation_records[agent_id]
        
        # Check for patterns that might indicate fraud
        recent_records = [r for r in records if time.time() - r.timestamp < 86400]  # Last 24 hours
        
        if len(recent_records) > 20:  # Too many events in short time
            suspicious_events.append("excessive_recent_activity")
        
        # Check for observer bias
        observer_counts = defaultdict(int)
        for record in recent_records:
            observer_counts[record.observer_id] += 1
        
        for observer, count in observer_counts.items():
            if count > 10:  # Same observer reporting too many events
                suspicious_events.append(f"observer_bias_{observer}")
        
        # Check for unusual positive score increases
        recent_positive = [r for r in recent_records if any(impact > 0.5 for impact in r.dimensions_affected.values())]
        if len(recent_positive) > 5:
            suspicious_events.append("suspicious_positive_spike")
        
        # Check for low credibility observers
        low_credibility_events = [
            r for r in recent_records 
            if self.observer_credibility[r.observer_id] < 0.3
        ]
        if len(low_credibility_events) > 3:
            suspicious_events.append("low_credibility_observers")
        
        return suspicious_events
    
    def get_reputation_analytics(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive reputation analytics for an agent"""
        if agent_id not in self.agents:
            return {}
        
        records = self.reputation_records[agent_id]
        scores = self.reputation_scores[agent_id]
        
        analytics = {
            'agent_id': agent_id,
            'overall_reputation': self.get_overall_reputation(agent_id),
            'social_capital': self.get_social_capital(agent_id),
            'total_events': len(records),
            'recent_events': len([r for r in records if time.time() - r.timestamp < 86400]),
            'dimension_scores': {dim.value: score.score for dim, score in scores.items()},
            'dimension_confidence': {dim.value: score.confidence for dim, score in scores.items()},
            'dimension_trends': {dim.value: score.trend for dim, score in scores.items()},
            'observer_credibility': self.observer_credibility[agent_id],
            'fraud_indicators': self.detect_reputation_fraud(agent_id),
            'top_dimensions': self._get_top_dimensions(agent_id),
            'improvement_areas': self._get_improvement_areas(agent_id),
            'reputation_trajectory': self._calculate_reputation_trajectory(agent_id)
        }
        
        return analytics
    
    def export_reputation_data(self) -> Dict[str, Any]:
        """Export all reputation data for analysis or backup"""
        export_data = {
            'agents': list(self.agents),
            'reputation_scores': {},
            'reputation_records': {},
            'social_capital': dict(self.social_capital),
            'observer_credibility': dict(self.observer_credibility),
            'influence_network': {k: dict(v) for k, v in self.influence_network.items()},
            'system_statistics': self._get_system_statistics()
        }
        
        # Convert reputation scores to serializable format
        for agent_id, scores in self.reputation_scores.items():
            export_data['reputation_scores'][agent_id] = {}
            for dimension, score in scores.items():
                export_data['reputation_scores'][agent_id][dimension.value] = asdict(score)
        
        # Convert reputation records to serializable format
        for agent_id, records in self.reputation_records.items():
            export_data['reputation_records'][agent_id] = []
            for record in records:
                record_dict = asdict(record)
                record_dict['event_type'] = record.event_type.value
                record_dict['dimensions_affected'] = {
                    dim.value: impact for dim, impact in record.dimensions_affected.items()
                }
                record_dict['witness_ids'] = list(record.witness_ids)
                export_data['reputation_records'][agent_id].append(record_dict)
        
        return export_data
    
    # Private helper methods
    
    def _initialize_event_impact_map(self) -> Dict[ReputationEvent, Dict[ReputationDimension, float]]:
        """Initialize mapping of events to reputation dimension impacts"""
        return {
            ReputationEvent.PROMISE_KEPT: {
                ReputationDimension.TRUSTWORTHINESS: 0.2,
                ReputationDimension.RELIABILITY: 0.3,
                ReputationDimension.HONESTY: 0.1
            },
            ReputationEvent.PROMISE_BROKEN: {
                ReputationDimension.TRUSTWORTHINESS: -0.3,
                ReputationDimension.RELIABILITY: -0.4,
                ReputationDimension.HONESTY: -0.2
            },
            ReputationEvent.SUCCESSFUL_COLLABORATION: {
                ReputationDimension.COOPERATION: 0.3,
                ReputationDimension.COMPETENCE: 0.2,
                ReputationDimension.LIKABILITY: 0.1
            },
            ReputationEvent.FAILED_COLLABORATION: {
                ReputationDimension.COOPERATION: -0.2,
                ReputationDimension.COMPETENCE: -0.1,
                ReputationDimension.RELIABILITY: -0.1
            },
            ReputationEvent.HELP_PROVIDED: {
                ReputationDimension.COOPERATION: 0.2,
                ReputationDimension.LIKABILITY: 0.3,
                ReputationDimension.SOCIAL_INFLUENCE: 0.1
            },
            ReputationEvent.HELP_REFUSED: {
                ReputationDimension.COOPERATION: -0.2,
                ReputationDimension.LIKABILITY: -0.1
            },
            ReputationEvent.LEADERSHIP_SUCCESS: {
                ReputationDimension.LEADERSHIP: 0.4,
                ReputationDimension.COMPETENCE: 0.2,
                ReputationDimension.SOCIAL_INFLUENCE: 0.2
            },
            ReputationEvent.LEADERSHIP_FAILURE: {
                ReputationDimension.LEADERSHIP: -0.3,
                ReputationDimension.COMPETENCE: -0.1
            },
            ReputationEvent.INNOVATION_SUCCESS: {
                ReputationDimension.INNOVATION: 0.4,
                ReputationDimension.COMPETENCE: 0.2,
                ReputationDimension.LEADERSHIP: 0.1
            },
            ReputationEvent.INNOVATION_FAILURE: {
                ReputationDimension.INNOVATION: -0.2,
                ReputationDimension.COMPETENCE: -0.1
            },
            ReputationEvent.HONEST_DISCLOSURE: {
                ReputationDimension.HONESTY: 0.3,
                ReputationDimension.TRUSTWORTHINESS: 0.2
            },
            ReputationEvent.DECEPTION_DETECTED: {
                ReputationDimension.HONESTY: -0.5,
                ReputationDimension.TRUSTWORTHINESS: -0.4,
                ReputationDimension.SOCIAL_INFLUENCE: -0.2
            },
            ReputationEvent.CONFLICT_RESOLVED: {
                ReputationDimension.CONFLICT_RESOLUTION: 0.4,
                ReputationDimension.LEADERSHIP: 0.2,
                ReputationDimension.SOCIAL_INFLUENCE: 0.1
            },
            ReputationEvent.CONFLICT_ESCALATED: {
                ReputationDimension.CONFLICT_RESOLUTION: -0.3,
                ReputationDimension.COOPERATION: -0.2
            },
            ReputationEvent.RESOURCE_SHARED: {
                ReputationDimension.COOPERATION: 0.3,
                ReputationDimension.LIKABILITY: 0.2
            },
            ReputationEvent.RESOURCE_HOARDED: {
                ReputationDimension.COOPERATION: -0.3,
                ReputationDimension.LIKABILITY: -0.2,
                ReputationDimension.SOCIAL_INFLUENCE: -0.1
            }
        }
    
    def _update_reputation_scores(self, agent_id: str):
        """Update reputation scores based on recent events"""
        if agent_id not in self.agents:
            return
        
        records = self.reputation_records[agent_id]
        current_time = time.time()
        
        # Calculate scores for each dimension
        for dimension in ReputationDimension:
            relevant_records = [
                r for r in records 
                if dimension in r.dimensions_affected
            ]
            
            if not relevant_records:
                continue
            
            # Calculate weighted score with decay
            weighted_sum = 0.0
            total_weight = 0.0
            recent_impacts = []
            
            for record in relevant_records:
                impact = record.dimensions_affected[dimension]
                
                # Apply time decay
                age_hours = (current_time - record.timestamp) / 3600.0
                decay_factor = math.exp(-record.decay_rate * age_hours)
                
                # Weight by observer credibility and record weight
                observer_credibility = self.observer_credibility[record.observer_id]
                verification_bonus = 1.2 if record.verified else 1.0
                
                effective_weight = record.weight * observer_credibility * verification_bonus * decay_factor
                
                weighted_sum += impact * effective_weight
                total_weight += effective_weight
                
                # Track recent impacts for trend calculation
                if age_hours < 168:  # Last week
                    recent_impacts.append(impact)
            
            if total_weight > 0:
                score = weighted_sum / total_weight
                confidence = min(1.0, total_weight / 10.0)  # Higher weight = higher confidence
                
                # Calculate trend
                trend = 0.0
                if len(recent_impacts) > 1:
                    if len(recent_impacts) >= 3:
                        trend = mean(recent_impacts[-3:]) - mean(recent_impacts[:-3]) if len(recent_impacts) > 3 else 0.0
                    trend = max(-1.0, min(1.0, trend))
                
                # Update score object
                score_obj = self.reputation_scores[agent_id][dimension]
                score_obj.score = max(-1.0, min(1.0, score))
                score_obj.confidence = confidence
                score_obj.num_observations = len(relevant_records)
                score_obj.last_updated = current_time
                score_obj.trend = trend
    
    def _update_observer_credibility(self, observer_id: str, record: ReputationRecord):
        """Update observer credibility based on record verification"""
        if observer_id not in self.agents:
            return
        
        current_credibility = self.observer_credibility[observer_id]
        
        # Increase credibility if record is verified by witnesses
        if record.verified and len(record.witness_ids) > 0:
            credibility_boost = 0.05 * len(record.witness_ids)
            self.observer_credibility[observer_id] = min(1.0, current_credibility + credibility_boost)
        
        # Credibility slowly decays towards 0.5 (neutral) over time
        time_decay = 0.001
        if current_credibility > 0.5:
            self.observer_credibility[observer_id] = max(0.5, current_credibility - time_decay)
        elif current_credibility < 0.5:
            self.observer_credibility[observer_id] = min(0.5, current_credibility + time_decay)
    
    def _update_social_capital(self, agent_id: str):
        """Update social capital based on reputation and network effects"""
        if agent_id not in self.agents:
            return
        
        # Base social capital from reputation
        overall_rep = self.get_overall_reputation(agent_id)
        base_capital = overall_rep * 50.0  # Scale to 0-50
        
        # Network effects - agents with high reputation endorsing you increases your capital
        network_bonus = 0.0
        endorsement_count = 0
        
        for record in self.reputation_records[agent_id]:
            if any(impact > 0 for impact in record.dimensions_affected.values()):
                observer_rep = self.get_overall_reputation(record.observer_id)
                if observer_rep > 0.5:  # Only count endorsements from well-regarded agents
                    network_bonus += observer_rep * 5.0
                    endorsement_count += 1
        
        # Diminishing returns on endorsements
        if endorsement_count > 0:
            network_bonus = network_bonus * (1.0 - math.exp(-endorsement_count / 10.0))
        
        self.social_capital[agent_id] = base_capital + network_bonus
    
    def _adjust_score_for_observer(self, agent_id: str, dimension: ReputationDimension, observer_id: str) -> float:
        """Adjust reputation score based on observer's perspective"""
        base_score = self.reputation_scores[agent_id][dimension].score
        
        # If observer has direct interaction records with the agent, weight those more heavily
        direct_records = [
            r for r in self.reputation_records[agent_id]
            if r.observer_id == observer_id and dimension in r.dimensions_affected
        ]
        
        if direct_records:
            # Calculate score based on direct observations
            direct_sum = sum(r.dimensions_affected[dimension] for r in direct_records)
            direct_weight = len(direct_records) / max(len(self.reputation_records[agent_id]), 1)
            
            # Blend direct and overall scores
            adjusted_score = base_score * (1.0 - direct_weight) + (direct_sum / len(direct_records)) * direct_weight
            return max(-1.0, min(1.0, adjusted_score))
        
        return base_score
    
    def _calculate_interaction_bonus(self, agent_a: str, agent_b: str) -> float:
        """Calculate trust bonus based on direct interaction history"""
        # Count positive vs negative interactions
        positive_interactions = 0
        negative_interactions = 0
        
        for record in self.reputation_records[agent_b]:
            if record.observer_id == agent_a:
                avg_impact = mean(record.dimensions_affected.values()) if record.dimensions_affected else 0.0
                if avg_impact > 0.1:
                    positive_interactions += 1
                elif avg_impact < -0.1:
                    negative_interactions += 1
        
        total_interactions = positive_interactions + negative_interactions
        if total_interactions == 0:
            return 0.0
        
        # Calculate bonus/penalty based on interaction ratio
        positive_ratio = positive_interactions / total_interactions
        interaction_bonus = (positive_ratio - 0.5) * 0.3  # -0.15 to +0.15 bonus
        
        # Scale by number of interactions (more interactions = more reliable)
        reliability_factor = min(1.0, total_interactions / 10.0)
        
        return interaction_bonus * reliability_factor
    
    def _get_top_dimensions(self, agent_id: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """Get top reputation dimensions for an agent"""
        scores = [(dim.value, score.score) for dim, score in self.reputation_scores[agent_id].items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    def _get_improvement_areas(self, agent_id: str, bottom_n: int = 3) -> List[Tuple[str, float]]:
        """Get dimensions where agent needs improvement"""
        scores = [(dim.value, score.score) for dim, score in self.reputation_scores[agent_id].items()]
        scores.sort(key=lambda x: x[1])
        return scores[:bottom_n]
    
    def _calculate_reputation_trajectory(self, agent_id: str) -> Dict[str, float]:
        """Calculate reputation trajectory (improving, declining, stable)"""
        trends = [score.trend for score in self.reputation_scores[agent_id].values()]
        avg_trend = mean(trends) if trends else 0.0
        
        return {
            'overall_trend': avg_trend,
            'improving_dimensions': len([t for t in trends if t > 0.1]),
            'declining_dimensions': len([t for t in trends if t < -0.1]),
            'stable_dimensions': len([t for t in trends if -0.1 <= t <= 0.1])
        }
    
    def _get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide reputation statistics"""
        if not self.agents:
            return {}
        
        all_scores = []
        for agent_scores in self.reputation_scores.values():
            for score in agent_scores.values():
                all_scores.append(score.score)
        
        return {
            'total_agents': len(self.agents),
            'total_records': sum(len(records) for records in self.reputation_records.values()),
            'average_reputation': mean(all_scores) if all_scores else 0.0,
            'reputation_std_dev': stdev(all_scores) if len(all_scores) > 1 else 0.0,
            'average_social_capital': mean(self.social_capital.values()) if self.social_capital else 0.0,
            'average_observer_credibility': mean(self.observer_credibility.values()) if self.observer_credibility else 0.5
        }
    
    def _invalidate_cache(self):
        """Invalidate cached values"""
        self._reputation_cache.clear()
        self._social_capital_cache.clear()
        self._cache_timestamp = 0.0
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        return (time.time() - self._cache_timestamp) < self._cache_ttl