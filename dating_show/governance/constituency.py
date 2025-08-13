"""
File: constituency.py
Description: Constituency Management for democratic voting rights and representation.
Manages voting eligibility, representation algorithms, demographic-based grouping,
and fair allocation of voting power in the collective rules system.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging
import math

# Import governance components
from ..agents.memory_structures.store_integration import MemoryStoreIntegration, StoreNamespace
from ..agents.enhanced_agent_state import GovernanceData, SpecializationData


class VotingRights(Enum):
    """Types of voting rights."""
    FULL = "full"  # Full voting rights
    LIMITED = "limited"  # Limited to specific types of proposals
    OBSERVER = "observer"  # Can observe but not vote
    SUSPENDED = "suspended"  # Temporarily suspended rights
    PROXY = "proxy"  # Voting through proxy/delegate


class RepresentationModel(Enum):
    """Models for constituency representation."""
    DIRECT_DEMOCRACY = "direct_democracy"  # One agent, one vote
    WEIGHTED_EXPERIENCE = "weighted_experience"  # Vote weight based on experience
    SKILL_BASED = "skill_based"  # Representation based on relevant skills
    TENURE_BASED = "tenure_based"  # Based on time in community
    HYBRID = "hybrid"  # Combination of factors


@dataclass
class VotingProfile:
    """Voting profile for an agent."""
    agent_id: str
    voting_rights: VotingRights
    voting_weight: float  # Base voting weight (0.0 to 2.0)
    expertise_multipliers: Dict[str, float]  # Domain-specific multipliers
    participation_history: List[Dict[str, Any]]
    registration_date: datetime
    last_vote_date: Optional[datetime]
    suspension_until: Optional[datetime]
    delegation_target: Optional[str]  # Agent ID if delegating votes
    delegated_votes: Set[str]  # Agent IDs who delegate to this agent
    total_delegated_weight: float
    constituency_groups: Set[str]
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.delegated_votes:
            self.delegated_votes = set()
        if not self.constituency_groups:
            self.constituency_groups = set()
        if not self.metadata:
            self.metadata = {}


@dataclass
class ConstituencyGroup:
    """A group of agents with shared characteristics or interests."""
    group_id: str
    group_name: str
    group_type: str  # 'demographic', 'skill', 'geographic', 'interest', 'role'
    description: str
    members: Set[str]  # Agent IDs
    representation_weight: float  # How much weight this group gets
    interests: List[str]  # Topics this group cares about
    spokesperson_agent_id: Optional[str]  # Elected representative
    created_at: datetime
    is_active: bool
    min_members: int = 3
    max_members: int = 100


@dataclass
class RepresentationAllocation:
    """Allocation of representation for a specific voting scenario."""
    allocation_id: str
    proposal_type: str
    constituency_weights: Dict[str, float]  # group_id -> weight
    individual_weights: Dict[str, float]  # agent_id -> weight
    total_voting_power: float
    allocation_method: RepresentationModel
    created_for_proposal: Optional[str]
    is_active: bool
    created_at: datetime


class ConstituencyManager:
    """
    Constituency Management System for democratic representation.
    Manages voting rights, representation models, and demographic grouping.
    """

    def __init__(self, store_integration: MemoryStoreIntegration,
                 postgres_persistence=None, community_size: int = 50):
        """
        Initialize the Constituency Manager.
        
        Args:
            store_integration: Store API integration
            postgres_persistence: PostgreSQL persistence layer
            community_size: Expected community size
        """
        self.store_integration = store_integration
        self.postgres_persistence = postgres_persistence
        self.community_size = community_size
        self.logger = logging.getLogger(f"{__name__}.ConstituencyManager")
        
        # Voting profiles and constituencies
        self.voting_profiles = {}  # agent_id -> VotingProfile
        self.constituency_groups = {}  # group_id -> ConstituencyGroup
        self.active_allocations = {}  # allocation_id -> RepresentationAllocation
        
        # Configuration
        self.config = {
            "max_voting_weight": 2.0,
            "min_voting_weight": 0.1,
            "delegation_chain_limit": 3,
            "min_participation_for_weight": 0.1,  # 10% minimum participation
            "experience_weight_factor": 0.3,
            "skill_relevance_threshold": 0.5,
            "tenure_months_for_full_weight": 6,
            "suspension_period_days": 30,
            "delegation_decay_factor": 0.95  # Daily decay for unused delegated votes
        }
        
        # Metrics
        self.metrics = {
            "registered_voters": 0,
            "active_constituencies": 0,
            "delegation_chains": 0,
            "average_participation": 0.0,
            "representation_fairness": 0.0
        }

    # =====================================================
    # Voting Profile Management
    # =====================================================

    async def register_voter(self, agent_id: str, initial_weight: float = 1.0,
                           expertise_areas: List[str] = None) -> bool:
        """
        Register an agent as a voter with initial profile.
        
        Args:
            agent_id: Agent identifier
            initial_weight: Initial voting weight
            expertise_areas: Areas of expertise for weighted voting
        
        Returns:
            bool: True if successfully registered
        """
        try:
            if agent_id in self.voting_profiles:
                self.logger.warning(f"Agent {agent_id} already registered")
                return False
            
            # Create voting profile
            profile = VotingProfile(
                agent_id=agent_id,
                voting_rights=VotingRights.FULL,
                voting_weight=max(self.config["min_voting_weight"], 
                                min(self.config["max_voting_weight"], initial_weight)),
                expertise_multipliers={area: 1.2 for area in (expertise_areas or [])},
                participation_history=[],
                registration_date=datetime.now(),
                last_vote_date=None,
                suspension_until=None,
                delegation_target=None,
                delegated_votes=set(),
                total_delegated_weight=0.0,
                constituency_groups=set(),
                metadata={"registration_source": "system"}
            )
            
            # Store profile
            self.voting_profiles[agent_id] = profile
            
            # Automatically assign to relevant constituency groups
            await self._auto_assign_constituencies(agent_id)
            
            # Store in Store API
            await self._store_voting_profile(profile)
            
            # Store in PostgreSQL
            if self.postgres_persistence:
                await self._store_profile_in_db(profile)
            
            self.metrics["registered_voters"] += 1
            
            self.logger.info(f"Registered voter {agent_id} with weight {initial_weight}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register voter {agent_id}: {str(e)}")
            return False

    async def update_voting_weight(self, agent_id: str, new_weight: float, 
                                 reason: str = "manual_update") -> bool:
        """Update an agent's voting weight."""
        try:
            if agent_id not in self.voting_profiles:
                return False
            
            profile = self.voting_profiles[agent_id]
            old_weight = profile.voting_weight
            
            # Apply limits
            profile.voting_weight = max(self.config["min_voting_weight"],
                                      min(self.config["max_voting_weight"], new_weight))
            
            # Log the change
            profile.metadata["weight_changes"] = profile.metadata.get("weight_changes", [])
            profile.metadata["weight_changes"].append({
                "old_weight": old_weight,
                "new_weight": profile.voting_weight,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            })
            
            # Update storage
            await self._store_voting_profile(profile)
            
            self.logger.info(f"Updated voting weight for {agent_id}: {old_weight} -> {profile.voting_weight}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update voting weight for {agent_id}: {str(e)}")
            return False

    async def suspend_voting_rights(self, agent_id: str, suspension_days: int, 
                                  reason: str) -> bool:
        """Suspend voting rights for an agent."""
        try:
            if agent_id not in self.voting_profiles:
                return False
            
            profile = self.voting_profiles[agent_id]
            profile.voting_rights = VotingRights.SUSPENDED
            profile.suspension_until = datetime.now() + timedelta(days=suspension_days)
            
            # Log suspension
            profile.metadata["suspensions"] = profile.metadata.get("suspensions", [])
            profile.metadata["suspensions"].append({
                "reason": reason,
                "suspended_at": datetime.now().isoformat(),
                "suspension_until": profile.suspension_until.isoformat(),
                "duration_days": suspension_days
            })
            
            await self._store_voting_profile(profile)
            
            # Notify community
            await self.store_integration._broadcast_community_event("voter_suspended", {
                "agent_id": agent_id,
                "reason": reason,
                "duration_days": suspension_days
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to suspend voting rights for {agent_id}: {str(e)}")
            return False

    async def delegate_vote(self, delegator_id: str, delegate_to_id: str) -> bool:
        """Set up vote delegation between agents."""
        try:
            if delegator_id not in self.voting_profiles or delegate_to_id not in self.voting_profiles:
                return False
            
            delegator_profile = self.voting_profiles[delegator_id]
            delegate_profile = self.voting_profiles[delegate_to_id]
            
            # Validate delegation is allowed
            if not await self._validate_delegation(delegator_id, delegate_to_id):
                return False
            
            # Remove from previous delegate if any
            if delegator_profile.delegation_target:
                prev_delegate = self.voting_profiles[delegator_profile.delegation_target]
                prev_delegate.delegated_votes.discard(delegator_id)
                prev_delegate.total_delegated_weight -= delegator_profile.voting_weight
            
            # Set up new delegation
            delegator_profile.delegation_target = delegate_to_id
            delegator_profile.voting_rights = VotingRights.PROXY
            
            delegate_profile.delegated_votes.add(delegator_id)
            delegate_profile.total_delegated_weight += delegator_profile.voting_weight
            
            # Update storage
            await self._store_voting_profile(delegator_profile)
            await self._store_voting_profile(delegate_profile)
            
            self.metrics["delegation_chains"] += 1
            
            self.logger.info(f"Set up delegation: {delegator_id} -> {delegate_to_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set up delegation: {str(e)}")
            return False

    # =====================================================
    # Constituency Group Management
    # =====================================================

    async def create_constituency_group(self, group_name: str, group_type: str,
                                      description: str, interests: List[str],
                                      min_members: int = 3) -> str:
        """Create a new constituency group."""
        try:
            group_id = str(uuid.uuid4())
            
            group = ConstituencyGroup(
                group_id=group_id,
                group_name=group_name,
                group_type=group_type,
                description=description,
                members=set(),
                representation_weight=1.0,
                interests=interests,
                spokesperson_agent_id=None,
                created_at=datetime.now(),
                is_active=True,
                min_members=min_members
            )
            
            self.constituency_groups[group_id] = group
            
            # Store in Store API
            await self._store_constituency_group(group)
            
            # Store in PostgreSQL
            if self.postgres_persistence:
                await self._store_group_in_db(group)
            
            self.metrics["active_constituencies"] += 1
            
            self.logger.info(f"Created constituency group '{group_name}' ({group_id})")
            return group_id
            
        except Exception as e:
            self.logger.error(f"Failed to create constituency group: {str(e)}")
            raise

    async def join_constituency_group(self, agent_id: str, group_id: str) -> bool:
        """Add an agent to a constituency group."""
        try:
            if agent_id not in self.voting_profiles or group_id not in self.constituency_groups:
                return False
            
            group = self.constituency_groups[group_id]
            profile = self.voting_profiles[agent_id]
            
            # Check if group has space
            if len(group.members) >= group.max_members:
                return False
            
            # Add to group
            group.members.add(agent_id)
            profile.constituency_groups.add(group_id)
            
            # Update storage
            await self._store_constituency_group(group)
            await self._store_voting_profile(profile)
            
            self.logger.info(f"Agent {agent_id} joined constituency group {group.group_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to join constituency group: {str(e)}")
            return False

    async def elect_spokesperson(self, group_id: str, candidate_agent_id: str) -> bool:
        """Elect a spokesperson for a constituency group."""
        try:
            if group_id not in self.constituency_groups:
                return False
            
            group = self.constituency_groups[group_id]
            
            if candidate_agent_id not in group.members:
                return False
            
            # Simple election - in practice this would involve group voting
            group.spokesperson_agent_id = candidate_agent_id
            
            # Update representation weight based on spokesperson's expertise
            if candidate_agent_id in self.voting_profiles:
                profile = self.voting_profiles[candidate_agent_id]
                # Spokesperson gets enhanced weight for group representation
                group.representation_weight = 1.0 + (profile.voting_weight * 0.2)
            
            await self._store_constituency_group(group)
            
            self.logger.info(f"Elected {candidate_agent_id} as spokesperson for {group.group_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to elect spokesperson: {str(e)}")
            return False

    # =====================================================
    # Representation Calculation
    # =====================================================

    async def calculate_representation_allocation(self, proposal_type: str,
                                                proposal_id: str = None,
                                                representation_model: RepresentationModel = RepresentationModel.HYBRID) -> str:
        """
        Calculate representation allocation for a specific voting scenario.
        
        Args:
            proposal_type: Type of proposal being voted on
            proposal_id: Specific proposal ID (optional)
            representation_model: Model to use for representation
        
        Returns:
            allocation_id: Unique identifier for the allocation
        """
        try:
            allocation_id = str(uuid.uuid4())
            
            # Calculate individual weights based on model
            individual_weights = {}
            constituency_weights = {}
            
            if representation_model == RepresentationModel.DIRECT_DEMOCRACY:
                individual_weights = await self._calculate_direct_weights()
                
            elif representation_model == RepresentationModel.WEIGHTED_EXPERIENCE:
                individual_weights = await self._calculate_experience_weights()
                
            elif representation_model == RepresentationModel.SKILL_BASED:
                individual_weights = await self._calculate_skill_weights(proposal_type)
                
            elif representation_model == RepresentationModel.TENURE_BASED:
                individual_weights = await self._calculate_tenure_weights()
                
            elif representation_model == RepresentationModel.HYBRID:
                individual_weights = await self._calculate_hybrid_weights(proposal_type)
            
            # Calculate constituency group weights
            constituency_weights = await self._calculate_constituency_weights(proposal_type)
            
            # Create allocation
            allocation = RepresentationAllocation(
                allocation_id=allocation_id,
                proposal_type=proposal_type,
                constituency_weights=constituency_weights,
                individual_weights=individual_weights,
                total_voting_power=sum(individual_weights.values()),
                allocation_method=representation_model,
                created_for_proposal=proposal_id,
                is_active=True,
                created_at=datetime.now()
            )
            
            self.active_allocations[allocation_id] = allocation
            
            # Store in Store API
            await self._store_allocation(allocation)
            
            self.logger.info(f"Created representation allocation {allocation_id} for {proposal_type}")
            return allocation_id
            
        except Exception as e:
            self.logger.error(f"Failed to calculate representation allocation: {str(e)}")
            raise

    async def get_effective_voting_weight(self, agent_id: str, 
                                        allocation_id: str = None) -> float:
        """Get effective voting weight for an agent in a specific allocation."""
        try:
            if agent_id not in self.voting_profiles:
                return 0.0
            
            profile = self.voting_profiles[agent_id]
            
            # Check if suspended
            if profile.voting_rights == VotingRights.SUSPENDED:
                if profile.suspension_until and datetime.now() < profile.suspension_until:
                    return 0.0
                else:
                    # Restore rights after suspension
                    profile.voting_rights = VotingRights.FULL
                    profile.suspension_until = None
            
            # Base weight
            effective_weight = profile.voting_weight
            
            # Add delegated weight
            effective_weight += profile.total_delegated_weight
            
            # Apply allocation-specific weight if available
            if allocation_id and allocation_id in self.active_allocations:
                allocation = self.active_allocations[allocation_id]
                if agent_id in allocation.individual_weights:
                    effective_weight = allocation.individual_weights[agent_id]
            
            return effective_weight
            
        except Exception as e:
            self.logger.error(f"Failed to get effective voting weight: {str(e)}")
            return 0.0

    # =====================================================
    # Helper Methods - Weight Calculations
    # =====================================================

    async def _calculate_direct_weights(self) -> Dict[str, float]:
        """Calculate direct democracy weights (equal for all)."""
        weights = {}
        for agent_id, profile in self.voting_profiles.items():
            if profile.voting_rights in [VotingRights.FULL, VotingRights.LIMITED]:
                weights[agent_id] = 1.0
        return weights

    async def _calculate_experience_weights(self) -> Dict[str, float]:
        """Calculate weights based on participation experience."""
        weights = {}
        
        for agent_id, profile in self.voting_profiles.items():
            if profile.voting_rights in [VotingRights.FULL, VotingRights.LIMITED]:
                participation_count = len(profile.participation_history)
                base_weight = profile.voting_weight
                
                # Experience bonus
                experience_bonus = min(0.5, participation_count * 0.01)
                weights[agent_id] = base_weight + experience_bonus
        
        return weights

    async def _calculate_skill_weights(self, proposal_type: str) -> Dict[str, float]:
        """Calculate weights based on relevant skills for the proposal type."""
        weights = {}
        
        # Map proposal types to relevant skills (simplified)
        skill_mappings = {
            "economic": ["economics", "finance", "resource_management"],
            "social": ["social_dynamics", "communication", "leadership"],
            "technical": ["programming", "system_design", "analysis"],
            "governance": ["leadership", "law", "negotiation"]
        }
        
        relevant_skills = skill_mappings.get(proposal_type, [])
        
        for agent_id, profile in self.voting_profiles.items():
            if profile.voting_rights in [VotingRights.FULL, VotingRights.LIMITED]:
                base_weight = profile.voting_weight
                skill_bonus = 0.0
                
                for skill in relevant_skills:
                    if skill in profile.expertise_multipliers:
                        skill_bonus += profile.expertise_multipliers[skill] * 0.1
                
                weights[agent_id] = base_weight + skill_bonus
        
        return weights

    async def _calculate_tenure_weights(self) -> Dict[str, float]:
        """Calculate weights based on tenure in community."""
        weights = {}
        now = datetime.now()
        
        for agent_id, profile in self.voting_profiles.items():
            if profile.voting_rights in [VotingRights.FULL, VotingRights.LIMITED]:
                months_tenure = (now - profile.registration_date).days / 30
                base_weight = profile.voting_weight
                
                # Tenure bonus (max 0.5 additional weight after 6 months)
                tenure_bonus = min(0.5, months_tenure / self.config["tenure_months_for_full_weight"] * 0.5)
                weights[agent_id] = base_weight + tenure_bonus
        
        return weights

    async def _calculate_hybrid_weights(self, proposal_type: str) -> Dict[str, float]:
        """Calculate hybrid weights combining multiple factors."""
        # Get weights from different models
        direct_weights = await self._calculate_direct_weights()
        experience_weights = await self._calculate_experience_weights()
        skill_weights = await self._calculate_skill_weights(proposal_type)
        tenure_weights = await self._calculate_tenure_weights()
        
        hybrid_weights = {}
        
        for agent_id in direct_weights:
            if agent_id in self.voting_profiles:
                # Weighted average of different models
                hybrid_weight = (
                    direct_weights.get(agent_id, 0) * 0.3 +
                    experience_weights.get(agent_id, 0) * 0.3 +
                    skill_weights.get(agent_id, 0) * 0.2 +
                    tenure_weights.get(agent_id, 0) * 0.2
                )
                
                # Apply limits
                hybrid_weights[agent_id] = max(self.config["min_voting_weight"],
                                             min(self.config["max_voting_weight"], hybrid_weight))
        
        return hybrid_weights

    async def _calculate_constituency_weights(self, proposal_type: str) -> Dict[str, float]:
        """Calculate constituency group representation weights."""
        weights = {}
        
        for group_id, group in self.constituency_groups.items():
            if not group.is_active or len(group.members) < group.min_members:
                continue
            
            # Base weight from group size
            size_weight = math.log(len(group.members)) / math.log(self.community_size)
            
            # Interest relevance bonus
            interest_bonus = 0.0
            if proposal_type in group.interests:
                interest_bonus = 0.3
            
            weights[group_id] = (size_weight + interest_bonus) * group.representation_weight
        
        return weights

    # =====================================================
    # Validation and Maintenance
    # =====================================================

    async def _validate_delegation(self, delegator_id: str, delegate_to_id: str) -> bool:
        """Validate that delegation setup is valid."""
        # Check if both agents exist
        if delegator_id not in self.voting_profiles or delegate_to_id not in self.voting_profiles:
            return False
        
        # Check if delegate has voting rights
        delegate_profile = self.voting_profiles[delegate_to_id]
        if delegate_profile.voting_rights not in [VotingRights.FULL, VotingRights.LIMITED]:
            return False
        
        # Check delegation chain length
        chain_length = await self._calculate_delegation_chain_length(delegate_to_id)
        if chain_length >= self.config["delegation_chain_limit"]:
            return False
        
        # Prevent circular delegation
        if await self._would_create_circular_delegation(delegator_id, delegate_to_id):
            return False
        
        return True

    async def _calculate_delegation_chain_length(self, agent_id: str) -> int:
        """Calculate the length of delegation chain for an agent."""
        visited = set()
        current = agent_id
        length = 0
        
        while current and current not in visited:
            visited.add(current)
            profile = self.voting_profiles.get(current)
            if not profile or not profile.delegation_target:
                break
            current = profile.delegation_target
            length += 1
            
            if length >= self.config["delegation_chain_limit"]:
                break
        
        return length

    async def _would_create_circular_delegation(self, delegator_id: str, delegate_to_id: str) -> bool:
        """Check if delegation would create a circular reference."""
        visited = set()
        current = delegate_to_id
        
        while current and current not in visited:
            if current == delegator_id:
                return True
            
            visited.add(current)
            profile = self.voting_profiles.get(current)
            if not profile:
                break
            current = profile.delegation_target
        
        return False

    async def _auto_assign_constituencies(self, agent_id: str) -> None:
        """Automatically assign agent to relevant constituency groups."""
        # This would analyze agent characteristics and assign to appropriate groups
        # Simplified implementation
        pass

    # =====================================================
    # Storage Methods
    # =====================================================

    async def _store_voting_profile(self, profile: VotingProfile) -> None:
        """Store voting profile in Store API."""
        if self.store_integration.store:
            profile_data = asdict(profile)
            profile_data["registration_date"] = profile.registration_date.isoformat()
            profile_data["last_vote_date"] = profile.last_vote_date.isoformat() if profile.last_vote_date else None
            profile_data["suspension_until"] = profile.suspension_until.isoformat() if profile.suspension_until else None
            profile_data["delegated_votes"] = list(profile.delegated_votes)
            profile_data["constituency_groups"] = list(profile.constituency_groups)
            profile_data["voting_rights"] = profile.voting_rights.value
            
            await self.store_integration.store.aput(
                StoreNamespace.GOVERNANCE.value,
                f"voting_profile_{profile.agent_id}",
                profile_data
            )

    async def _store_constituency_group(self, group: ConstituencyGroup) -> None:
        """Store constituency group in Store API."""
        if self.store_integration.store:
            group_data = asdict(group)
            group_data["members"] = list(group.members)
            group_data["created_at"] = group.created_at.isoformat()
            
            await self.store_integration.store.aput(
                StoreNamespace.GOVERNANCE.value,
                f"constituency_{group.group_id}",
                group_data
            )

    async def _store_allocation(self, allocation: RepresentationAllocation) -> None:
        """Store allocation in Store API."""
        if self.store_integration.store:
            allocation_data = asdict(allocation)
            allocation_data["created_at"] = allocation.created_at.isoformat()
            allocation_data["allocation_method"] = allocation.allocation_method.value
            
            await self.store_integration.store.aput(
                StoreNamespace.GOVERNANCE.value,
                f"allocation_{allocation.allocation_id}",
                allocation_data
            )

    async def _store_profile_in_db(self, profile: VotingProfile) -> None:
        """Store voting profile in PostgreSQL."""
        if self.postgres_persistence:
            async with self.postgres_persistence.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO voting_profiles (agent_id, voting_rights, voting_weight, expertise_multipliers,
                                               participation_history, registration_date, last_vote_date,
                                               suspension_until, delegation_target, total_delegated_weight,
                                               constituency_groups, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (agent_id) DO UPDATE SET
                        voting_rights = EXCLUDED.voting_rights,
                        voting_weight = EXCLUDED.voting_weight,
                        expertise_multipliers = EXCLUDED.expertise_multipliers,
                        suspension_until = EXCLUDED.suspension_until,
                        delegation_target = EXCLUDED.delegation_target,
                        total_delegated_weight = EXCLUDED.total_delegated_weight,
                        constituency_groups = EXCLUDED.constituency_groups,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, profile.agent_id, profile.voting_rights.value, profile.voting_weight,
                json.dumps(profile.expertise_multipliers), json.dumps(profile.participation_history),
                profile.registration_date, profile.last_vote_date, profile.suspension_until,
                profile.delegation_target, profile.total_delegated_weight,
                json.dumps(list(profile.constituency_groups)), json.dumps(profile.metadata))

    async def _store_group_in_db(self, group: ConstituencyGroup) -> None:
        """Store constituency group in PostgreSQL."""
        if self.postgres_persistence:
            async with self.postgres_persistence.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO constituency_groups (group_id, group_name, group_type, description,
                                                   members, representation_weight, interests,
                                                   spokesperson_agent_id, created_at, is_active,
                                                   min_members, max_members)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, group.group_id, group.group_name, group.group_type, group.description,
                json.dumps(list(group.members)), group.representation_weight, json.dumps(group.interests),
                group.spokesperson_agent_id, group.created_at, group.is_active,
                group.min_members, group.max_members)

    # =====================================================
    # System Metrics and Health
    # =====================================================

    async def get_constituency_metrics(self) -> Dict[str, Any]:
        """Get comprehensive constituency system metrics."""
        try:
            active_voters = len([p for p in self.voting_profiles.values() 
                               if p.voting_rights in [VotingRights.FULL, VotingRights.LIMITED]])
            
            suspended_voters = len([p for p in self.voting_profiles.values()
                                  if p.voting_rights == VotingRights.SUSPENDED])
            
            delegation_count = len([p for p in self.voting_profiles.values()
                                  if p.delegation_target is not None])
            
            active_groups = len([g for g in self.constituency_groups.values()
                               if g.is_active and len(g.members) >= g.min_members])
            
            # Calculate representation fairness (simplified metric)
            total_weight = sum(p.voting_weight for p in self.voting_profiles.values())
            weight_variance = sum((p.voting_weight - (total_weight / len(self.voting_profiles)))**2 
                                for p in self.voting_profiles.values()) / len(self.voting_profiles)
            fairness_score = 1.0 / (1.0 + weight_variance)  # Higher is more fair
            
            return {
                "registered_voters": len(self.voting_profiles),
                "active_voters": active_voters,
                "suspended_voters": suspended_voters,
                "delegation_count": delegation_count,
                "active_constituency_groups": active_groups,
                "total_constituency_groups": len(self.constituency_groups),
                "representation_fairness": fairness_score,
                "average_voting_weight": total_weight / max(1, len(self.voting_profiles)),
                "delegation_rate": delegation_count / max(1, len(self.voting_profiles))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get constituency metrics: {str(e)}")
            return {}


# Helper functions

def create_constituency_manager(store_integration: MemoryStoreIntegration,
                              postgres_persistence=None, community_size: int = 50) -> ConstituencyManager:
    """Create a ConstituencyManager instance."""
    return ConstituencyManager(store_integration, postgres_persistence, community_size)


# Example usage
if __name__ == "__main__":
    async def test_constituency_manager():
        """Test the Constituency Manager."""
        print("Constituency Manager loaded successfully")
        
    asyncio.run(test_constituency_manager())