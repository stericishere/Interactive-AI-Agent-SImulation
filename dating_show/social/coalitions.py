"""
Coalition Formation System for Enhanced PIANO Architecture
Phase 3: Advanced Features - Week 9: Complex Social Dynamics

This module implements a sophisticated coalition formation system that enables
interest-based grouping, coalition stability analysis, and group decision mechanisms
for multi-agent coordination and collaboration.
"""

import json
import math
import time
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from statistics import mean, median
import itertools


class CoalitionType(Enum):
    """Types of coalitions that can form"""
    TASK_BASED = "task_based"
    INTEREST_BASED = "interest_based"
    SURVIVAL_BASED = "survival_based"
    RESOURCE_BASED = "resource_based"
    IDEOLOGICAL = "ideological"
    TEMPORARY = "temporary"
    STRATEGIC = "strategic"
    SOCIAL = "social"
    PROFESSIONAL = "professional"
    EMERGENCY = "emergency"


class CoalitionStatus(Enum):
    """Status of a coalition"""
    FORMING = "forming"
    ACTIVE = "active"
    STABLE = "stable"
    DECLINING = "declining"
    DISSOLVED = "dissolved"
    MERGING = "merging"
    SPLITTING = "splitting"


class MemberRole(Enum):
    """Roles within a coalition"""
    LEADER = "leader"
    CO_LEADER = "co_leader"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    CONTRIBUTOR = "contributor"
    OBSERVER = "observer"
    CANDIDATE = "candidate"


@dataclass
class CoalitionMember:
    """Represents a member of a coalition"""
    agent_id: str
    role: MemberRole
    contribution_score: float  # 0.0 to 1.0
    commitment_level: float  # 0.0 to 1.0
    satisfaction: float  # 0.0 to 1.0
    join_time: float
    last_activity: float
    skills_contributed: Set[str]
    resources_contributed: Dict[str, float]
    influence_within_coalition: float  # 0.0 to 1.0
    exit_probability: float  # 0.0 to 1.0 (likelihood of leaving)
    
    def __post_init__(self):
        """Validate member data"""
        self.contribution_score = max(0.0, min(1.0, self.contribution_score))
        self.commitment_level = max(0.0, min(1.0, self.commitment_level))
        self.satisfaction = max(0.0, min(1.0, self.satisfaction))
        self.influence_within_coalition = max(0.0, min(1.0, self.influence_within_coalition))
        self.exit_probability = max(0.0, min(1.0, self.exit_probability))


@dataclass
class CoalitionGoal:
    """Represents a goal of a coalition"""
    goal_id: str
    description: str
    priority: float  # 0.0 to 1.0
    target_completion: float  # timestamp
    required_skills: Set[str]
    required_resources: Dict[str, float]
    progress: float  # 0.0 to 1.0
    assigned_members: Set[str]
    dependencies: Set[str]  # Other goal IDs this depends on
    success_criteria: Dict[str, Any]
    
    def __post_init__(self):
        """Validate goal data"""
        self.priority = max(0.0, min(1.0, self.priority))
        self.progress = max(0.0, min(1.0, self.progress))


@dataclass
class Coalition:
    """Represents a coalition of agents"""
    coalition_id: str
    name: str
    coalition_type: CoalitionType
    status: CoalitionStatus
    created_time: float
    last_updated: float
    
    # Membership
    members: Dict[str, CoalitionMember]
    max_size: int
    min_size: int
    
    # Goals and objectives
    goals: Dict[str, CoalitionGoal]
    shared_values: Dict[str, float]
    common_interests: Set[str]
    
    # Resources and capabilities
    shared_resources: Dict[str, float]
    collective_skills: Dict[str, float]
    
    # Performance metrics
    cohesion_score: float  # 0.0 to 1.0
    effectiveness_score: float  # 0.0 to 1.0
    stability_score: float  # 0.0 to 1.0
    success_rate: float  # 0.0 to 1.0
    
    # Decision making
    decision_making_method: str  # "consensus", "majority", "leader", "weighted"
    voting_threshold: float  # For majority decisions
    
    # Dynamics
    internal_conflicts: List[Dict[str, Any]]
    recent_decisions: List[Dict[str, Any]]
    performance_history: List[Dict[str, Any]]
    
    def __post_init__(self):
        """Validate coalition data"""
        self.cohesion_score = max(0.0, min(1.0, self.cohesion_score))
        self.effectiveness_score = max(0.0, min(1.0, self.effectiveness_score))
        self.stability_score = max(0.0, min(1.0, self.stability_score))
        self.success_rate = max(0.0, min(1.0, self.success_rate))
        self.voting_threshold = max(0.0, min(1.0, self.voting_threshold))


class CoalitionFormationSystem:
    """
    Comprehensive coalition formation and management system for multi-agent coordination.
    
    Features:
    - Interest-based coalition formation
    - Dynamic membership management
    - Coalition stability analysis
    - Group decision mechanisms
    - Resource and skill pooling
    - Coalition lifecycle management
    - Performance tracking and optimization
    """
    
    def __init__(self, max_coalitions: int = 100, max_agents: int = 1000):
        self.max_coalitions = max_coalitions
        self.max_agents = max_agents
        
        # Core data structures
        self.agents: Set[str] = set()
        self.coalitions: Dict[str, Coalition] = {}
        self.agent_memberships: Dict[str, Set[str]] = defaultdict(set)  # agent -> coalition IDs
        
        # Agent capabilities and interests
        self.agent_skills: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.agent_interests: Dict[str, Set[str]] = defaultdict(set)
        self.agent_values: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.agent_resources: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Formation algorithms
        self.formation_algorithms = {
            'interest_similarity': self._form_by_interest_similarity,
            'skill_complementarity': self._form_by_skill_complementarity,
            'value_alignment': self._form_by_value_alignment,
            'resource_synergy': self._form_by_resource_synergy,
            'social_network': self._form_by_social_network
        }
        
        # Performance tracking
        self.coalition_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.formation_success_rate: float = 0.0
        self.average_coalition_lifespan: float = 0.0
        
        # Concurrency control
        self.lock = threading.RLock()
        
        # Caching
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        self._stability_cache: Dict[str, float] = {}
        self._cache_timestamp = 0.0
        self._cache_ttl = 300.0  # 5 minutes
    
    def add_agent(
        self, 
        agent_id: str, 
        skills: Optional[Dict[str, float]] = None,
        interests: Optional[Set[str]] = None,
        values: Optional[Dict[str, float]] = None,
        resources: Optional[Dict[str, float]] = None
    ) -> bool:
        """Add an agent to the coalition formation system"""
        with self.lock:
            if len(self.agents) >= self.max_agents:
                return False
            
            self.agents.add(agent_id)
            
            # Initialize agent data
            self.agent_skills[agent_id] = skills or {}
            self.agent_interests[agent_id] = interests or set()
            self.agent_values[agent_id] = values or {}
            self.agent_resources[agent_id] = resources or {}
            self.agent_memberships[agent_id] = set()
            
            self._invalidate_cache()
            return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the system and all their coalitions"""
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            # Remove from all coalitions
            coalition_ids = list(self.agent_memberships[agent_id])
            for coalition_id in coalition_ids:
                self.leave_coalition(agent_id, coalition_id)
            
            # Remove agent data
            self.agents.remove(agent_id)
            del self.agent_skills[agent_id]
            del self.agent_interests[agent_id]
            del self.agent_values[agent_id]
            del self.agent_resources[agent_id]
            del self.agent_memberships[agent_id]
            
            self._invalidate_cache()
            return True
    
    def update_agent_profile(
        self,
        agent_id: str,
        skills: Optional[Dict[str, float]] = None,
        interests: Optional[Set[str]] = None,
        values: Optional[Dict[str, float]] = None,
        resources: Optional[Dict[str, float]] = None
    ) -> bool:
        """Update an agent's profile"""
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            if skills is not None:
                self.agent_skills[agent_id].update(skills)
            if interests is not None:
                self.agent_interests[agent_id].update(interests)
            if values is not None:
                self.agent_values[agent_id].update(values)
            if resources is not None:
                self.agent_resources[agent_id].update(resources)
            
            # Update relevant coalitions
            for coalition_id in self.agent_memberships[agent_id]:
                self._update_coalition_capabilities(coalition_id)
            
            self._invalidate_cache()
            return True
    
    def suggest_coalitions(
        self,
        agent_id: str,
        task_requirements: Optional[Dict[str, Any]] = None,
        algorithm: str = 'interest_similarity',
        max_suggestions: int = 5
    ) -> List[Tuple[List[str], float]]:
        """Suggest potential coalitions for an agent"""
        # Input validation
        if not isinstance(agent_id, str) or not agent_id.strip():
            return []
        if agent_id not in self.agents:
            return []
        if max_suggestions <= 0:
            max_suggestions = 5
        max_suggestions = min(max_suggestions, 20)  # Reasonable upper limit
        
        if algorithm not in self.formation_algorithms:
            algorithm = 'interest_similarity'
        
        formation_func = self.formation_algorithms[algorithm]
        potential_coalitions = formation_func(agent_id, task_requirements)
        
        # Score and rank suggestions
        scored_suggestions = []
        for potential_members in potential_coalitions:
            if len(potential_members) < 2:  # Need at least 2 members
                continue
            
            score = self._calculate_coalition_potential(potential_members, task_requirements)
            scored_suggestions.append((potential_members, score))
        
        # Sort by score and return top suggestions
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        return scored_suggestions[:max_suggestions]
    
    def create_coalition(
        self,
        initiator_id: str,
        members: List[str],
        coalition_type: CoalitionType,
        name: Optional[str] = None,
        goals: Optional[List[Dict[str, Any]]] = None,
        max_size: int = 20,
        min_size: int = 2,
        decision_method: str = "majority"
    ) -> Optional[str]:
        """Create a new coalition"""
        # Input validation
        if not isinstance(initiator_id, str) or not initiator_id.strip():
            return None
        if not isinstance(members, list) or len(members) < 1:
            return None
        if not isinstance(coalition_type, CoalitionType):
            return None
        if max_size < min_size or min_size < 1:
            return None
        if decision_method not in ["majority", "consensus", "leader", "weighted"]:
            decision_method = "majority"
        
        with self.lock:
            if len(self.coalitions) >= self.max_coalitions:
                return None
            
            if initiator_id not in self.agents:
                return None
            
            # Validate members
            valid_members = [m for m in members if m in self.agents]
            if len(valid_members) < min_size:
                return None
            
            # Generate coalition ID
            coalition_id = f"coalition_{coalition_type.value}_{int(time.time() * 1000)}"
            
            # Create coalition
            coalition = Coalition(
                coalition_id=coalition_id,
                name=name or f"Coalition {coalition_id[-8:]}",
                coalition_type=coalition_type,
                status=CoalitionStatus.FORMING,
                created_time=time.time(),
                last_updated=time.time(),
                members={},
                max_size=max_size,
                min_size=min_size,
                goals={},
                shared_values={},
                common_interests=set(),
                shared_resources={},
                collective_skills={},
                cohesion_score=0.0,
                effectiveness_score=0.0,
                stability_score=0.0,
                success_rate=0.0,
                decision_making_method=decision_method,
                voting_threshold=0.5,
                internal_conflicts=[],
                recent_decisions=[],
                performance_history=[]
            )
            
            # Add members
            for i, member_id in enumerate(valid_members):
                role = MemberRole.LEADER if member_id == initiator_id else MemberRole.CONTRIBUTOR
                
                member = CoalitionMember(
                    agent_id=member_id,
                    role=role,
                    contribution_score=0.5,
                    commitment_level=0.7,
                    satisfaction=0.5,
                    join_time=time.time(),
                    last_activity=time.time(),
                    skills_contributed=set(),
                    resources_contributed={},
                    influence_within_coalition=1.0 if role == MemberRole.LEADER else 0.5,
                    exit_probability=0.1
                )
                
                coalition.members[member_id] = member
                self.agent_memberships[member_id].add(coalition_id)
            
            # Add goals if provided
            if goals:
                for goal_data in goals:
                    self._add_goal_to_coalition(coalition_id, goal_data)
            
            # Calculate initial metrics
            self._update_coalition_capabilities(coalition_id)
            self._update_coalition_metrics(coalition_id)
            
            self.coalitions[coalition_id] = coalition
            
            # Update status if enough members
            if len(coalition.members) >= min_size:
                coalition.status = CoalitionStatus.ACTIVE
            
            self._invalidate_cache()
            return coalition_id
    
    def join_coalition(self, agent_id: str, coalition_id: str) -> bool:
        """Add an agent to an existing coalition"""
        with self.lock:
            if agent_id not in self.agents or coalition_id not in self.coalitions:
                return False
            
            coalition = self.coalitions[coalition_id]
            
            # Check if already a member
            if agent_id in coalition.members:
                return False
            
            # Check size limits
            if len(coalition.members) >= coalition.max_size:
                return False
            
            # Check if coalition is accepting new members
            if coalition.status in [CoalitionStatus.DISSOLVED, CoalitionStatus.SPLITTING]:
                return False
            
            # Calculate compatibility
            compatibility = self._calculate_agent_coalition_compatibility(agent_id, coalition_id)
            if compatibility < 0.3:  # Minimum compatibility threshold
                return False
            
            # Add member
            member = CoalitionMember(
                agent_id=agent_id,
                role=MemberRole.CONTRIBUTOR,
                contribution_score=0.5,
                commitment_level=0.7,
                satisfaction=0.5,
                join_time=time.time(),
                last_activity=time.time(),
                skills_contributed=set(),
                resources_contributed={},
                influence_within_coalition=0.3,
                exit_probability=0.2
            )
            
            coalition.members[agent_id] = member
            self.agent_memberships[agent_id].add(coalition_id)
            
            # Update coalition
            coalition.last_updated = time.time()
            self._update_coalition_capabilities(coalition_id)
            self._update_coalition_metrics(coalition_id)
            
            self._invalidate_cache()
            return True
    
    def leave_coalition(self, agent_id: str, coalition_id: str) -> bool:
        """Remove an agent from a coalition"""
        with self.lock:
            if agent_id not in self.agents or coalition_id not in self.coalitions:
                return False
            
            coalition = self.coalitions[coalition_id]
            
            if agent_id not in coalition.members:
                return False
            
            # Remove member
            del coalition.members[agent_id]
            self.agent_memberships[agent_id].discard(coalition_id)
            
            # Update coalition
            coalition.last_updated = time.time()
            
            # Check if coalition should be dissolved
            if len(coalition.members) < coalition.min_size:
                self._dissolve_coalition(coalition_id)
            else:
                # Reassign leadership if leader left
                if coalition.members and all(m.role != MemberRole.LEADER for m in coalition.members.values()):
                    self._elect_new_leader(coalition_id)
                
                self._update_coalition_capabilities(coalition_id)
                self._update_coalition_metrics(coalition_id)
            
            self._invalidate_cache()
            return True
    
    def make_coalition_decision(
        self,
        coalition_id: str,
        decision_type: str,
        options: List[str],
        proposer_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Make a decision within a coalition"""
        if coalition_id not in self.coalitions:
            return None
        
        coalition = self.coalitions[coalition_id]
        
        if proposer_id not in coalition.members:
            return None
        
        # Get votes from members based on decision making method
        if coalition.decision_making_method == "leader":
            # Leader decides
            leader = next((m for m in coalition.members.values() if m.role == MemberRole.LEADER), None)
            if leader:
                chosen_option = self._simulate_agent_choice(leader.agent_id, options, context)
            else:
                chosen_option = options[0] if options else None
        
        elif coalition.decision_making_method == "consensus":
            # All members must agree
            votes = {}
            for member in coalition.members.values():
                vote = self._simulate_agent_choice(member.agent_id, options, context)
                votes[member.agent_id] = vote
            
            # Check if all votes are the same
            unique_votes = set(votes.values())
            if len(unique_votes) == 1:
                chosen_option = list(unique_votes)[0]
            else:
                chosen_option = None  # No consensus
        
        elif coalition.decision_making_method == "majority":
            # Majority vote
            votes = {}
            for member in coalition.members.values():
                vote = self._simulate_agent_choice(member.agent_id, options, context)
                votes[member.agent_id] = vote
            
            # Count votes
            vote_counts = defaultdict(int)
            for vote in votes.values():
                vote_counts[vote] += 1
            
            # Find majority
            total_votes = len(votes)
            threshold = int(total_votes * coalition.voting_threshold)
            
            majority_option = None
            max_votes = 0
            for option, count in vote_counts.items():
                if count > max_votes and count >= threshold:
                    max_votes = count
                    majority_option = option
            
            chosen_option = majority_option
        
        elif coalition.decision_making_method == "weighted":
            # Weighted vote based on influence
            weighted_votes = defaultdict(float)
            total_influence = sum(m.influence_within_coalition for m in coalition.members.values())
            
            for member in coalition.members.values():
                vote = self._simulate_agent_choice(member.agent_id, options, context)
                weight = member.influence_within_coalition / total_influence if total_influence > 0 else 1.0
                weighted_votes[vote] += weight
            
            # Find option with highest weighted vote
            chosen_option = max(weighted_votes, key=weighted_votes.get) if weighted_votes else None
        
        else:
            chosen_option = options[0] if options else None
        
        # Record decision
        decision_record = {
            'decision_id': f"{coalition_id}_{int(time.time() * 1000)}",
            'type': decision_type,
            'options': options,
            'chosen_option': chosen_option,
            'proposer_id': proposer_id,
            'timestamp': time.time(),
            'context': context or {},
            'method': coalition.decision_making_method,
            'success': chosen_option is not None
        }
        
        coalition.recent_decisions.append(decision_record)
        
        # Keep only recent decisions
        if len(coalition.recent_decisions) > 50:
            coalition.recent_decisions = coalition.recent_decisions[-50:]
        
        return chosen_option
    
    def get_coalition_stability(self, coalition_id: str) -> float:
        """Calculate stability score for a coalition"""
        if coalition_id not in self.coalitions:
            return 0.0
        
        # Check cache
        if self._is_cache_valid() and coalition_id in self._stability_cache:
            return self._stability_cache[coalition_id]
        
        coalition = self.coalitions[coalition_id]
        
        # Factors affecting stability
        factors = []
        
        # Member satisfaction
        if coalition.members:
            avg_satisfaction = mean(m.satisfaction for m in coalition.members.values())
            factors.append(avg_satisfaction)
        
        # Member commitment
        if coalition.members:
            avg_commitment = mean(m.commitment_level for m in coalition.members.values())
            factors.append(avg_commitment)
        
        # Low exit probability
        if coalition.members:
            avg_exit_prob = mean(m.exit_probability for m in coalition.members.values())
            factors.append(1.0 - avg_exit_prob)
        
        # Goal progress
        if coalition.goals:
            avg_progress = mean(g.progress for g in coalition.goals.values())
            factors.append(avg_progress)
        
        # Few internal conflicts
        recent_conflicts = [c for c in coalition.internal_conflicts 
                          if time.time() - c.get('timestamp', 0) < 86400]  # Last 24 hours
        conflict_factor = max(0.0, 1.0 - len(recent_conflicts) * 0.1)
        factors.append(conflict_factor)
        
        # Decision success rate
        recent_decisions = [d for d in coalition.recent_decisions 
                           if time.time() - d.get('timestamp', 0) < 604800]  # Last week
        if recent_decisions:
            success_rate = sum(1 for d in recent_decisions if d.get('success', False)) / len(recent_decisions)
            factors.append(success_rate)
        
        # Member retention
        if coalition.members:
            avg_tenure = mean(time.time() - m.join_time for m in coalition.members.values())
            tenure_factor = min(1.0, avg_tenure / 604800)  # Normalize to 1 week
            factors.append(tenure_factor)
        
        # Calculate overall stability
        stability = mean(factors) if factors else 0.0
        
        # Cache result
        self._stability_cache[coalition_id] = stability
        
        return stability
    
    def analyze_coalition_performance(self, coalition_id: str) -> Dict[str, Any]:
        """Analyze comprehensive performance metrics for a coalition"""
        if coalition_id not in self.coalitions:
            return {}
        
        coalition = self.coalitions[coalition_id]
        
        # Basic metrics
        analysis = {
            'coalition_id': coalition_id,
            'name': coalition.name,
            'type': coalition.coalition_type.value,
            'status': coalition.status.value,
            'age_days': (time.time() - coalition.created_time) / 86400,
            'member_count': len(coalition.members),
            'goal_count': len(coalition.goals),
            
            # Performance scores
            'cohesion_score': coalition.cohesion_score,
            'effectiveness_score': coalition.effectiveness_score,
            'stability_score': self.get_coalition_stability(coalition_id),
            'success_rate': coalition.success_rate,
            
            # Member metrics
            'average_satisfaction': mean(m.satisfaction for m in coalition.members.values()) if coalition.members else 0.0,
            'average_commitment': mean(m.commitment_level for m in coalition.members.values()) if coalition.members else 0.0,
            'average_contribution': mean(m.contribution_score for m in coalition.members.values()) if coalition.members else 0.0,
            'member_turnover': self._calculate_member_turnover(coalition_id),
            
            # Goal metrics
            'goals_completed': sum(1 for g in coalition.goals.values() if g.progress >= 1.0),
            'average_goal_progress': mean(g.progress for g in coalition.goals.values()) if coalition.goals else 0.0,
            'overdue_goals': sum(1 for g in coalition.goals.values() if g.target_completion < time.time() and g.progress < 1.0),
            
            # Decision making
            'recent_decisions': len(coalition.recent_decisions),
            'decision_success_rate': self._calculate_decision_success_rate(coalition_id),
            'decision_method': coalition.decision_making_method,
            
            # Resources and skills
            'total_skills': len(coalition.collective_skills),
            'total_resources': sum(coalition.shared_resources.values()),
            'skill_diversity': self._calculate_skill_diversity(coalition_id),
            
            # Challenges
            'internal_conflicts': len(coalition.internal_conflicts),
            'high_exit_risk_members': sum(1 for m in coalition.members.values() if m.exit_probability > 0.7),
            'low_satisfaction_members': sum(1 for m in coalition.members.values() if m.satisfaction < 0.3),
            
            # Recommendations
            'recommendations': self._generate_performance_recommendations(coalition_id)
        }
        
        return analysis
    
    def get_agent_coalition_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get summary of an agent's coalition memberships"""
        if agent_id not in self.agents:
            return {}
        
        memberships = []
        for coalition_id in self.agent_memberships[agent_id]:
            if coalition_id in self.coalitions:
                coalition = self.coalitions[coalition_id]
                member = coalition.members.get(agent_id)
                
                if member:
                    memberships.append({
                        'coalition_id': coalition_id,
                        'name': coalition.name,
                        'type': coalition.coalition_type.value,
                        'status': coalition.status.value,
                        'role': member.role.value,
                        'member_count': len(coalition.members),
                        'satisfaction': member.satisfaction,
                        'contribution_score': member.contribution_score,
                        'influence': member.influence_within_coalition,
                        'days_active': (time.time() - member.join_time) / 86400
                    })
        
        return {
            'agent_id': agent_id,
            'total_coalitions': len(memberships),
            'active_coalitions': len([m for m in memberships if m['status'] == 'active']),
            'leadership_roles': len([m for m in memberships if m['role'] in ['leader', 'co_leader']]),
            'average_satisfaction': mean(m['satisfaction'] for m in memberships) if memberships else 0.0,
            'average_contribution': mean(m['contribution_score'] for m in memberships) if memberships else 0.0,
            'total_influence': sum(m['influence'] for m in memberships),
            'memberships': memberships
        }
    
    def export_system_data(self) -> Dict[str, Any]:
        """Export all coalition system data"""
        export_data = {
            'agents': list(self.agents),
            'coalitions': {},
            'agent_profiles': {
                'skills': {k: dict(v) for k, v in self.agent_skills.items()},
                'interests': {k: list(v) for k, v in self.agent_interests.items()},
                'values': {k: dict(v) for k, v in self.agent_values.items()},
                'resources': {k: dict(v) for k, v in self.agent_resources.items()}
            },
            'system_statistics': self._get_system_statistics()
        }
        
        # Export coalition data
        for coalition_id, coalition in self.coalitions.items():
            export_data['coalitions'][coalition_id] = {
                'basic_info': {
                    'coalition_id': coalition.coalition_id,
                    'name': coalition.name,
                    'type': coalition.coalition_type.value,
                    'status': coalition.status.value,
                    'created_time': coalition.created_time,
                    'member_count': len(coalition.members)
                },
                'members': {
                    member_id: asdict(member) 
                    for member_id, member in coalition.members.items()
                },
                'goals': {
                    goal_id: asdict(goal) 
                    for goal_id, goal in coalition.goals.items()
                },
                'performance': self.analyze_coalition_performance(coalition_id)
            }
        
        return export_data
    
    # Private helper methods
    
    def _form_by_interest_similarity(self, agent_id: str, task_requirements: Optional[Dict[str, Any]] = None) -> List[List[str]]:
        """Form coalitions based on shared interests"""
        agent_interests = self.agent_interests[agent_id]
        potential_coalitions = []
        
        # Find agents with similar interests
        similar_agents = []
        for other_agent in self.agents:
            if other_agent != agent_id:
                similarity = self._calculate_interest_similarity(agent_id, other_agent)
                if similarity > 0.3:  # Threshold for similarity
                    similar_agents.append((other_agent, similarity))
        
        # Sort by similarity
        similar_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Form coalitions of different sizes
        for size in range(2, min(8, len(similar_agents) + 2)):
            coalition_members = [agent_id] + [a[0] for a in similar_agents[:size-1]]
            potential_coalitions.append(coalition_members)
        
        return potential_coalitions
    
    def _form_by_skill_complementarity(self, agent_id: str, task_requirements: Optional[Dict[str, Any]] = None) -> List[List[str]]:
        """Form coalitions based on complementary skills"""
        agent_skills = self.agent_skills[agent_id]
        potential_coalitions = []
        
        # Required skills from task requirements
        required_skills = set()
        if task_requirements and 'required_skills' in task_requirements:
            required_skills = set(task_requirements['required_skills'])
        
        # Find agents with complementary skills
        complementary_agents = []
        for other_agent in self.agents:
            if other_agent != agent_id:
                other_skills = self.agent_skills[other_agent]
                complementarity = self._calculate_skill_complementarity(agent_skills, other_skills, required_skills)
                if complementarity > 0.2:
                    complementary_agents.append((other_agent, complementarity))
        
        # Sort by complementarity
        complementary_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Form coalitions optimizing skill coverage
        for size in range(2, min(6, len(complementary_agents) + 2)):
            coalition_members = [agent_id]
            remaining_agents = [a[0] for a in complementary_agents]
            
            # Greedily add agents that maximize skill coverage
            covered_skills = set(agent_skills.keys())
            for _ in range(size - 1):
                best_agent = None
                best_coverage = 0
                
                for candidate in remaining_agents:
                    candidate_skills = set(self.agent_skills[candidate].keys())
                    new_coverage = len(candidate_skills - covered_skills)
                    if new_coverage > best_coverage:
                        best_coverage = new_coverage
                        best_agent = candidate
                
                if best_agent:
                    coalition_members.append(best_agent)
                    remaining_agents.remove(best_agent)
                    covered_skills.update(self.agent_skills[best_agent].keys())
            
            potential_coalitions.append(coalition_members)
        
        return potential_coalitions
    
    def _form_by_value_alignment(self, agent_id: str, task_requirements: Optional[Dict[str, Any]] = None) -> List[List[str]]:
        """Form coalitions based on shared values"""
        agent_values = self.agent_values[agent_id]
        potential_coalitions = []
        
        # Find agents with aligned values
        aligned_agents = []
        for other_agent in self.agents:
            if other_agent != agent_id:
                alignment = self._calculate_value_alignment(agent_id, other_agent)
                if alignment > 0.4:  # Threshold for alignment
                    aligned_agents.append((other_agent, alignment))
        
        # Sort by alignment
        aligned_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Form coalitions of different sizes
        for size in range(2, min(7, len(aligned_agents) + 2)):
            coalition_members = [agent_id] + [a[0] for a in aligned_agents[:size-1]]
            potential_coalitions.append(coalition_members)
        
        return potential_coalitions
    
    def _form_by_resource_synergy(self, agent_id: str, task_requirements: Optional[Dict[str, Any]] = None) -> List[List[str]]:
        """Form coalitions based on resource synergy"""
        agent_resources = self.agent_resources[agent_id]
        potential_coalitions = []
        
        # Required resources from task requirements
        required_resources = {}
        if task_requirements and 'required_resources' in task_requirements:
            required_resources = task_requirements['required_resources']
        
        # Find agents with synergistic resources
        synergistic_agents = []
        for other_agent in self.agents:
            if other_agent != agent_id:
                other_resources = self.agent_resources[other_agent]
                synergy = self._calculate_resource_synergy(agent_resources, other_resources, required_resources)
                if synergy > 0.3:
                    synergistic_agents.append((other_agent, synergy))
        
        # Sort by synergy
        synergistic_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Form coalitions optimizing resource coverage
        for size in range(2, min(5, len(synergistic_agents) + 2)):
            coalition_members = [agent_id] + [a[0] for a in synergistic_agents[:size-1]]
            potential_coalitions.append(coalition_members)
        
        return potential_coalitions
    
    def _form_by_social_network(self, agent_id: str, task_requirements: Optional[Dict[str, Any]] = None) -> List[List[str]]:
        """Form coalitions based on social network connections"""
        # This would integrate with the relationship network system
        # For now, return basic groupings
        potential_coalitions = []
        
        # Simple approach: form with other agents (would be enhanced with relationship data)
        other_agents = [a for a in self.agents if a != agent_id]
        
        for size in range(2, min(6, len(other_agents) + 2)):
            coalition_members = [agent_id] + other_agents[:size-1]
            potential_coalitions.append(coalition_members)
        
        return potential_coalitions
    
    def _calculate_coalition_potential(self, members: List[str], task_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the potential effectiveness of a coalition"""
        if len(members) < 2:
            return 0.0
        
        scores = []
        
        # Skill coverage
        all_skills = set()
        for member in members:
            all_skills.update(self.agent_skills[member].keys())
        
        required_skills = set()
        if task_requirements and 'required_skills' in task_requirements:
            required_skills = set(task_requirements['required_skills'])
        
        if required_skills:
            skill_coverage = len(all_skills & required_skills) / len(required_skills)
            scores.append(skill_coverage)
        
        # Interest alignment
        interest_alignment = self._calculate_group_interest_alignment(members)
        scores.append(interest_alignment)
        
        # Value alignment
        value_alignment = self._calculate_group_value_alignment(members)
        scores.append(value_alignment)
        
        # Resource availability
        total_resources = 0.0
        for member in members:
            total_resources += sum(self.agent_resources[member].values())
        
        resource_score = min(1.0, total_resources / 100.0)  # Normalize
        scores.append(resource_score)
        
        # Size penalty for very large groups
        size_penalty = 1.0 if len(members) <= 5 else max(0.5, 1.0 - (len(members) - 5) * 0.1)
        
        return mean(scores) * size_penalty if scores else 0.0
    
    def _calculate_interest_similarity(self, agent_a: str, agent_b: str) -> float:
        """Calculate interest similarity between two agents"""
        interests_a = self.agent_interests[agent_a]
        interests_b = self.agent_interests[agent_b]
        
        if not interests_a or not interests_b:
            return 0.0
        
        intersection = len(interests_a & interests_b)
        union = len(interests_a | interests_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_skill_complementarity(
        self, 
        skills_a: Dict[str, float], 
        skills_b: Dict[str, float], 
        required_skills: Set[str]
    ) -> float:
        """Calculate skill complementarity between two agents"""
        all_skills = set(skills_a.keys()) | set(skills_b.keys())
        
        if not all_skills:
            return 0.0
        
        # Coverage of required skills
        coverage_score = 0.0
        if required_skills:
            covered_required = (set(skills_a.keys()) | set(skills_b.keys())) & required_skills
            coverage_score = len(covered_required) / len(required_skills)
        
        # Complementarity (non-overlapping skills)
        overlap = set(skills_a.keys()) & set(skills_b.keys())
        complementarity_score = 1.0 - (len(overlap) / len(all_skills))
        
        # Skill strength
        avg_strength_a = mean(skills_a.values()) if skills_a else 0.0
        avg_strength_b = mean(skills_b.values()) if skills_b else 0.0
        strength_score = (avg_strength_a + avg_strength_b) / 2.0
        
        return (coverage_score * 0.4 + complementarity_score * 0.3 + strength_score * 0.3)
    
    def _calculate_value_alignment(self, agent_a: str, agent_b: str) -> float:
        """Calculate value alignment between two agents"""
        values_a = self.agent_values[agent_a]
        values_b = self.agent_values[agent_b]
        
        if not values_a or not values_b:
            return 0.5  # Neutral if no values
        
        # Calculate cosine similarity
        common_values = set(values_a.keys()) & set(values_b.keys())
        
        if not common_values:
            return 0.5  # Neutral if no common values
        
        dot_product = sum(values_a[v] * values_b[v] for v in common_values)
        norm_a = math.sqrt(sum(values_a[v] ** 2 for v in common_values))
        norm_b = math.sqrt(sum(values_b[v] ** 2 for v in common_values))
        
        if norm_a == 0 or norm_b == 0:
            return 0.5
        
        cosine_similarity = dot_product / (norm_a * norm_b)
        
        # Convert from [-1, 1] to [0, 1]
        return (cosine_similarity + 1.0) / 2.0
    
    def _calculate_resource_synergy(
        self, 
        resources_a: Dict[str, float], 
        resources_b: Dict[str, float], 
        required_resources: Dict[str, float]
    ) -> float:
        """Calculate resource synergy between two agents"""
        combined_resources = defaultdict(float)
        
        for resource, amount in resources_a.items():
            combined_resources[resource] += amount
        
        for resource, amount in resources_b.items():
            combined_resources[resource] += amount
        
        if not required_resources:
            # General resource synergy
            total_combined = sum(combined_resources.values())
            total_individual = sum(resources_a.values()) + sum(resources_b.values())
            return min(1.0, total_combined / max(total_individual, 1.0))
        
        # Coverage of required resources
        coverage_score = 0.0
        for resource, required_amount in required_resources.items():
            available = combined_resources.get(resource, 0.0)
            coverage_score += min(1.0, available / required_amount) if required_amount > 0 else 1.0
        
        return coverage_score / len(required_resources) if required_resources else 0.0
    
    def _calculate_group_interest_alignment(self, members: List[str]) -> float:
        """Calculate interest alignment across a group of agents"""
        if len(members) < 2:
            return 1.0
        
        all_interests = []
        for member in members:
            all_interests.append(self.agent_interests[member])
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                similarity = self._calculate_interest_similarity(members[i], members[j])
                similarities.append(similarity)
        
        return mean(similarities) if similarities else 0.0
    
    def _calculate_group_value_alignment(self, members: List[str]) -> float:
        """Calculate value alignment across a group of agents"""
        if len(members) < 2:
            return 1.0
        
        # Calculate pairwise alignments
        alignments = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                alignment = self._calculate_value_alignment(members[i], members[j])
                alignments.append(alignment)
        
        return mean(alignments) if alignments else 0.5
    
    def _calculate_agent_coalition_compatibility(self, agent_id: str, coalition_id: str) -> float:
        """Calculate how compatible an agent is with a coalition"""
        coalition = self.coalitions[coalition_id]
        
        if not coalition.members:
            return 0.5  # Neutral for empty coalition
        
        compatibility_scores = []
        
        # Interest compatibility
        agent_interests = self.agent_interests[agent_id]
        coalition_interests = coalition.common_interests
        
        if agent_interests and coalition_interests:
            interest_overlap = len(agent_interests & coalition_interests)
            interest_total = len(agent_interests | coalition_interests)
            interest_compatibility = interest_overlap / interest_total if interest_total > 0 else 0.0
            compatibility_scores.append(interest_compatibility)
        
        # Value compatibility
        agent_values = self.agent_values[agent_id]
        if agent_values and coalition.shared_values:
            value_alignment = 0.0
            common_values = set(agent_values.keys()) & set(coalition.shared_values.keys())
            
            if common_values:
                for value in common_values:
                    diff = abs(agent_values[value] - coalition.shared_values[value])
                    value_alignment += 1.0 - diff  # Assuming values are 0-1
                
                value_alignment /= len(common_values)
                compatibility_scores.append(value_alignment)
        
        # Skill complementarity
        agent_skills = set(self.agent_skills[agent_id].keys())
        coalition_skills = set(coalition.collective_skills.keys())
        
        skill_complement = len(agent_skills - coalition_skills) / max(len(agent_skills), 1)
        compatibility_scores.append(skill_complement)
        
        return mean(compatibility_scores) if compatibility_scores else 0.5
    
    def _simulate_agent_choice(self, agent_id: str, options: List[str], context: Optional[Dict[str, Any]] = None) -> str:
        """Simulate an agent's choice from options (simplified)"""
        # This would integrate with the agent's decision-making system
        # For now, return a simple choice
        if not options:
            return ""
        
        # Simple heuristic: choose based on agent's interests/values
        agent_interests = self.agent_interests[agent_id]
        agent_values = self.agent_values[agent_id]
        
        # Score each option
        option_scores = {}
        for option in options:
            score = 0.5  # Base score
            
            # Check if option aligns with interests
            if agent_interests:
                for interest in agent_interests:
                    if interest.lower() in option.lower():
                        score += 0.2
            
            # Check if option aligns with values
            if agent_values:
                for value, strength in agent_values.items():
                    if value.lower() in option.lower():
                        score += strength * 0.1
            
            option_scores[option] = score
        
        # Return highest scoring option
        return max(option_scores, key=option_scores.get)
    
    def _update_coalition_capabilities(self, coalition_id: str):
        """Update coalition's collective capabilities"""
        if coalition_id not in self.coalitions:
            return
        
        coalition = self.coalitions[coalition_id]
        
        # Reset capabilities
        coalition.collective_skills.clear()
        coalition.shared_resources.clear()
        coalition.common_interests.clear()
        coalition.shared_values.clear()
        
        if not coalition.members:
            return
        
        # Aggregate skills
        for member_id in coalition.members:
            member_skills = self.agent_skills[member_id]
            for skill, level in member_skills.items():
                if skill in coalition.collective_skills:
                    coalition.collective_skills[skill] = max(coalition.collective_skills[skill], level)
                else:
                    coalition.collective_skills[skill] = level
        
        # Aggregate resources
        for member_id in coalition.members:
            member_resources = self.agent_resources[member_id]
            for resource, amount in member_resources.items():
                coalition.shared_resources[resource] = coalition.shared_resources.get(resource, 0.0) + amount
        
        # Find common interests
        if coalition.members:
            member_list = list(coalition.members.keys())
            common_interests = self.agent_interests[member_list[0]].copy()
            
            for member_id in member_list[1:]:
                common_interests &= self.agent_interests[member_id]
            
            coalition.common_interests = common_interests
        
        # Calculate shared values (average of member values)
        value_sums = defaultdict(float)
        value_counts = defaultdict(int)
        
        for member_id in coalition.members:
            member_values = self.agent_values[member_id]
            for value, strength in member_values.items():
                value_sums[value] += strength
                value_counts[value] += 1
        
        for value in value_sums:
            coalition.shared_values[value] = value_sums[value] / value_counts[value]
    
    def _update_coalition_metrics(self, coalition_id: str):
        """Update coalition performance metrics"""
        if coalition_id not in self.coalitions:
            return
        
        coalition = self.coalitions[coalition_id]
        
        # Cohesion score
        if coalition.members:
            satisfaction_scores = [m.satisfaction for m in coalition.members.values()]
            commitment_scores = [m.commitment_level for m in coalition.members.values()]
            
            avg_satisfaction = mean(satisfaction_scores)
            avg_commitment = mean(commitment_scores)
            
            coalition.cohesion_score = (avg_satisfaction + avg_commitment) / 2.0
        
        # Effectiveness score
        if coalition.goals:
            progress_scores = [g.progress for g in coalition.goals.values()]
            coalition.effectiveness_score = mean(progress_scores)
        
        # Success rate
        recent_decisions = [d for d in coalition.recent_decisions 
                           if time.time() - d.get('timestamp', 0) < 604800]  # Last week
        if recent_decisions:
            successful_decisions = sum(1 for d in recent_decisions if d.get('success', False))
            coalition.success_rate = successful_decisions / len(recent_decisions)
        
        coalition.last_updated = time.time()
    
    def _add_goal_to_coalition(self, coalition_id: str, goal_data: Dict[str, Any]):
        """Add a goal to a coalition"""
        if coalition_id not in self.coalitions:
            return
        
        coalition = self.coalitions[coalition_id]
        
        goal_id = goal_data.get('goal_id', f"goal_{int(time.time() * 1000)}")
        
        goal = CoalitionGoal(
            goal_id=goal_id,
            description=goal_data.get('description', ''),
            priority=goal_data.get('priority', 0.5),
            target_completion=goal_data.get('target_completion', time.time() + 86400),
            required_skills=set(goal_data.get('required_skills', [])),
            required_resources=goal_data.get('required_resources', {}),
            progress=0.0,
            assigned_members=set(),
            dependencies=set(goal_data.get('dependencies', [])),
            success_criteria=goal_data.get('success_criteria', {})
        )
        
        coalition.goals[goal_id] = goal
    
    def _dissolve_coalition(self, coalition_id: str):
        """Dissolve a coalition"""
        if coalition_id not in self.coalitions:
            return
        
        coalition = self.coalitions[coalition_id]
        
        # Remove from agent memberships
        for member_id in coalition.members:
            self.agent_memberships[member_id].discard(coalition_id)
        
        # Update status and record dissolution
        coalition.status = CoalitionStatus.DISSOLVED
        coalition.last_updated = time.time()
        
        # Add to history
        dissolution_record = {
            'event': 'dissolution',
            'timestamp': time.time(),
            'reason': 'insufficient_members',
            'final_member_count': len(coalition.members),
            'lifespan_days': (time.time() - coalition.created_time) / 86400
        }
        
        self.coalition_history[coalition_id].append(dissolution_record)
        
        # Remove from active coalitions
        del self.coalitions[coalition_id]
    
    def _elect_new_leader(self, coalition_id: str):
        """Elect a new leader for a coalition"""
        if coalition_id not in self.coalitions:
            return
        
        coalition = self.coalitions[coalition_id]
        
        if not coalition.members:
            return
        
        # Find member with highest influence
        best_candidate = None
        highest_influence = 0.0
        
        for member in coalition.members.values():
            if member.influence_within_coalition > highest_influence:
                highest_influence = member.influence_within_coalition
                best_candidate = member
        
        if best_candidate:
            best_candidate.role = MemberRole.LEADER
            best_candidate.influence_within_coalition = min(1.0, best_candidate.influence_within_coalition + 0.1)
    
    def _calculate_member_turnover(self, coalition_id: str) -> float:
        """Calculate member turnover rate for a coalition"""
        # This would require tracking historical membership
        # For now, return a simple estimate based on current exit probabilities
        if coalition_id not in self.coalitions:
            return 0.0
        
        coalition = self.coalitions[coalition_id]
        
        if not coalition.members:
            return 0.0
        
        avg_exit_probability = mean(m.exit_probability for m in coalition.members.values())
        return avg_exit_probability
    
    def _calculate_decision_success_rate(self, coalition_id: str) -> float:
        """Calculate decision success rate for a coalition"""
        if coalition_id not in self.coalitions:
            return 0.0
        
        coalition = self.coalitions[coalition_id]
        
        if not coalition.recent_decisions:
            return 0.0
        
        successful_decisions = sum(1 for d in coalition.recent_decisions if d.get('success', False))
        return successful_decisions / len(coalition.recent_decisions)
    
    def _calculate_skill_diversity(self, coalition_id: str) -> float:
        """Calculate skill diversity within a coalition"""
        if coalition_id not in self.coalitions:
            return 0.0
        
        coalition = self.coalitions[coalition_id]
        
        if not coalition.collective_skills:
            return 0.0
        
        # Simple diversity measure: number of unique skills / potential maximum
        max_possible_skills = 50  # Assume max 50 different skills exist
        return len(coalition.collective_skills) / max_possible_skills
    
    def _generate_performance_recommendations(self, coalition_id: str) -> List[str]:
        """Generate recommendations for improving coalition performance"""
        if coalition_id not in self.coalitions:
            return []
        
        coalition = self.coalitions[coalition_id]
        recommendations = []
        
        # Low satisfaction
        if coalition.members:
            avg_satisfaction = mean(m.satisfaction for m in coalition.members.values())
            if avg_satisfaction < 0.4:
                recommendations.append("Address member satisfaction through better communication and goal alignment")
        
        # Low cohesion
        if coalition.cohesion_score < 0.5:
            recommendations.append("Improve team cohesion through team building activities and shared experiences")
        
        # Poor goal progress
        if coalition.goals:
            overdue_goals = sum(1 for g in coalition.goals.values() 
                              if g.target_completion < time.time() and g.progress < 1.0)
            if overdue_goals > 0:
                recommendations.append(f"Focus on completing {overdue_goals} overdue goals")
        
        # High exit risk
        high_risk_members = sum(1 for m in coalition.members.values() if m.exit_probability > 0.7)
        if high_risk_members > 0:
            recommendations.append(f"Address concerns of {high_risk_members} members at risk of leaving")
        
        # Decision making issues
        if coalition.recent_decisions:
            success_rate = self._calculate_decision_success_rate(coalition_id)
            if success_rate < 0.6:
                recommendations.append("Review and improve decision-making processes")
        
        # Size issues
        if len(coalition.members) < coalition.min_size:
            recommendations.append("Recruit new members to meet minimum size requirements")
        elif len(coalition.members) > coalition.max_size * 0.9:
            recommendations.append("Consider splitting into smaller, more manageable groups")
        
        return recommendations
    
    def _get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide coalition statistics"""
        active_coalitions = [c for c in self.coalitions.values() if c.status == CoalitionStatus.ACTIVE]
        
        return {
            'total_agents': len(self.agents),
            'total_coalitions': len(self.coalitions),
            'active_coalitions': len(active_coalitions),
            'average_coalition_size': mean(len(c.members) for c in active_coalitions) if active_coalitions else 0.0,
            'average_coalition_age_days': mean((time.time() - c.created_time) / 86400 for c in active_coalitions) if active_coalitions else 0.0,
            'coalition_types': {ct.value: len([c for c in active_coalitions if c.coalition_type == ct]) for ct in CoalitionType},
            'average_member_satisfaction': mean(mean(m.satisfaction for m in c.members.values()) for c in active_coalitions if c.members) if active_coalitions else 0.0,
            'formation_success_rate': self.formation_success_rate,
            'average_coalition_lifespan': self.average_coalition_lifespan
        }
    
    def _invalidate_cache(self):
        """Invalidate cached values"""
        self._similarity_cache.clear()
        self._stability_cache.clear()
        self._cache_timestamp = 0.0
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        return (time.time() - self._cache_timestamp) < self._cache_ttl