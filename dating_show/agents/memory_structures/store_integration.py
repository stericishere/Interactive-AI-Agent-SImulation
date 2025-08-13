"""
File: store_integration.py
Description: LangGraph Store API integration for cross-agent cultural and governance data sharing.
Enables cultural meme propagation, governance coordination, and social influence networks.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Import enhanced agent state components
from ..enhanced_agent_state import CulturalData, GovernanceData, PerformanceMetrics


class StoreNamespace(Enum):
    """Store API namespaces for organizing shared data."""
    CULTURAL_MEMES = "cultural_memes"
    GOVERNANCE = "governance"
    SOCIAL_ROLES = "social_roles"
    INFLUENCE_NETWORK = "influence_network"
    COMMUNITY_EVENTS = "community_events"


@dataclass
class CulturalMeme:
    """Cultural meme data structure for cross-agent propagation."""
    meme_id: str
    meme_name: str
    meme_type: str  # 'behavior', 'value', 'norm', 'tradition', 'slang'
    description: str
    origin_agent_id: str
    strength: float
    virality: float  # How easily it spreads
    stability: float  # How resistant to change
    created_at: datetime
    adopters: Set[str]
    transmission_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class GovernanceRule:
    """Governance rule for democratic community management."""
    rule_id: str
    rule_name: str
    rule_text: str
    rule_type: str  # 'constitutional', 'procedural', 'behavioral', 'resource'
    priority: int
    scope: str
    created_at: datetime
    enacted_at: Optional[datetime]
    created_by_agent_id: str
    version: int
    is_active: bool
    enforcement_level: float
    community_support: float
    metadata: Dict[str, Any]


@dataclass
class GovernanceProposal:
    """Governance proposal for democratic changes."""
    proposal_id: str
    proposal_title: str
    proposal_text: str
    proposal_type: str
    proposed_by_agent_id: str
    proposed_at: datetime
    voting_starts_at: Optional[datetime]
    voting_ends_at: Optional[datetime]
    status: str  # 'draft', 'voting', 'passed', 'rejected', 'withdrawn'
    required_majority: float
    votes: Dict[str, Dict[str, Any]]  # agent_id -> vote_data
    rationale: str
    metadata: Dict[str, Any]


@dataclass
class SocialRole:
    """Social role definition and tracking."""
    role_id: str
    role_name: str
    description: str
    expectations: Dict[str, Any]
    status: float  # Social status/prestige
    availability: int  # How many agents can have this role
    current_holders: Set[str]
    created_at: datetime
    is_active: bool


@dataclass
class InfluenceRelation:
    """Agent influence relationship."""
    influencer_agent_id: str
    influenced_agent_id: str
    influence_strength: float
    influence_type: str  # 'leadership', 'expertise', 'social', 'cultural'
    last_interaction: datetime
    interaction_count: int
    created_at: datetime
    is_active: bool


class MemoryStoreIntegration:
    """
    LangGraph Store API integration for Enhanced PIANO cross-agent sharing.
    Manages cultural memes, governance, and social dynamics across 50+ agents.
    """
    
    def __init__(self, store=None, postgres_persistence=None):
        """
        Initialize Store API integration.
        
        Args:
            store: LangGraph Store API instance (BaseStore)
            postgres_persistence: PostgreSQL persistence layer
        """
        self.store = store  # Will be injected by LangGraph framework
        self.postgres_persistence = postgres_persistence
        self.logger = logging.getLogger(f"{__name__}.MemoryStoreIntegration")
        
        # Performance tracking
        self.operation_times = {}
        self.operation_counts = {}
        
        # Cache for frequently accessed data
        self._meme_cache = {}
        self._rule_cache = {}
        self._cache_expiry = {}
        self._cache_ttl = 300  # 5 minutes
    
    def _track_operation_time(self, operation: str, duration: float) -> None:
        """Track operation performance metrics."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
        
        self.operation_times[operation].append(duration)
        self.operation_counts[operation] += 1
        
        # Keep only last 100 measurements
        if len(self.operation_times[operation]) > 100:
            self.operation_times[operation] = self.operation_times[operation][-100:]
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[key]
    
    def _cache_data(self, key: str, data: Any) -> None:
        """Cache data with expiry time."""
        self._cache_expiry[key] = datetime.now() + timedelta(seconds=self._cache_ttl)
        
        if key.startswith("meme_"):
            self._meme_cache[key] = data
        elif key.startswith("rule_"):
            self._rule_cache[key] = data
    
    # =====================================================
    # Cultural Meme System
    # =====================================================
    
    async def propagate_meme(self, meme: CulturalMeme, target_agents: Set[str] = None) -> Dict[str, Any]:
        """
        Propagate a cultural meme to target agents or community.
        
        Args:
            meme: Cultural meme to propagate
            target_agents: Specific agents to target (None = broadcast)
        
        Returns:
            Propagation results
        """
        start_time = datetime.now()
        
        try:
            # Store meme in Store API
            meme_key = f"meme_{meme.meme_id}"
            meme_data = {
                **asdict(meme),
                "created_at": meme.created_at.isoformat(),
                "adopters": list(meme.adopters)
            }
            
            if self.store:
                await self.store.aput(StoreNamespace.CULTURAL_MEMES.value, meme_key, meme_data)
            
            # Store in PostgreSQL for persistence
            if self.postgres_persistence:
                async with self.postgres_persistence.get_connection() as conn:
                    await conn.execute("""
                        INSERT INTO cultural_memes (meme_id, meme_name, meme_type, description, origin_agent_id,
                                                   strength, virality, stability, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (meme_id) DO UPDATE SET
                            strength = EXCLUDED.strength,
                            virality = EXCLUDED.virality,
                            metadata = EXCLUDED.metadata,
                            updated_at = CURRENT_TIMESTAMP
                    """, meme.meme_id, meme.meme_name, meme.meme_type, meme.description,
                    meme.origin_agent_id, meme.strength, meme.virality, meme.stability,
                    json.dumps(meme.metadata))
            
            # Record transmission events
            transmission_results = {}
            if target_agents:
                for agent_id in target_agents:
                    success_probability = meme.virality * 0.8  # Base success rate
                    success = hash(f"{meme.meme_id}_{agent_id}_{datetime.now().timestamp()}") % 100 < success_probability * 100
                    
                    transmission_results[agent_id] = {
                        "success": success,
                        "transmission_strength": meme.virality,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if success:
                        # Record adoption
                        if self.postgres_persistence:
                            async with self.postgres_persistence.get_connection() as conn:
                                await conn.execute("""
                                    INSERT INTO agent_meme_adoption (agent_id, meme_id, influence_strength, source_agent_id)
                                    VALUES ($1, $2, $3, $4)
                                    ON CONFLICT (agent_id, meme_id) DO UPDATE SET
                                        influence_strength = GREATEST(agent_meme_adoption.influence_strength, EXCLUDED.influence_strength),
                                        last_reinforcement = CURRENT_TIMESTAMP,
                                        exposure_count = agent_meme_adoption.exposure_count + 1
                                """, agent_id, meme.meme_id, meme.virality, meme.origin_agent_id)
            
            # Update meme statistics
            await self._update_meme_statistics(meme.meme_id)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("propagate_meme", duration)
            
            return {
                "meme_id": meme.meme_id,
                "transmission_results": transmission_results,
                "total_transmissions": len(target_agents) if target_agents else 0,
                "successful_transmissions": sum(1 for r in transmission_results.values() if r["success"]),
                "propagation_time_ms": duration
            }
            
        except Exception as e:
            self.logger.error(f"Failed to propagate meme {meme.meme_id}: {str(e)}")
            raise
    
    async def get_agent_memes(self, agent_id: str, include_inactive: bool = False) -> List[CulturalMeme]:
        """Get all memes adopted by an agent."""
        start_time = datetime.now()
        
        try:
            cache_key = f"meme_agent_{agent_id}"
            if self._is_cache_valid(cache_key):
                return self._meme_cache.get(cache_key, [])
            
            memes = []
            
            if self.postgres_persistence:
                async with self.postgres_persistence.get_connection() as conn:
                    rows = await conn.fetch("""
                        SELECT cm.meme_id, cm.meme_name, cm.meme_type, cm.description, cm.origin_agent_id,
                               cm.strength, cm.virality, cm.stability, cm.created_at, cm.metadata,
                               ama.influence_strength, ama.adoption_date
                        FROM cultural_memes cm
                        JOIN agent_meme_adoption ama ON cm.meme_id = ama.meme_id
                        WHERE ama.agent_id = $1 AND (ama.is_active = TRUE OR $2 = TRUE)
                        ORDER BY ama.influence_strength DESC, ama.adoption_date DESC
                    """, agent_id, include_inactive)
                    
                    for row in rows:
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}
                        
                        # Get adoption data from other agents
                        adopters_rows = await conn.fetch("""
                            SELECT agent_id FROM agent_meme_adoption 
                            WHERE meme_id = $1 AND is_active = TRUE
                        """, row['meme_id'])
                        
                        meme = CulturalMeme(
                            meme_id=row['meme_id'],
                            meme_name=row['meme_name'],
                            meme_type=row['meme_type'],
                            description=row['description'],
                            origin_agent_id=row['origin_agent_id'],
                            strength=row['strength'],
                            virality=row['virality'],
                            stability=row['stability'],
                            created_at=row['created_at'],
                            adopters=set(r['agent_id'] for r in adopters_rows),
                            transmission_history=[],  # Could be loaded separately if needed
                            metadata=metadata
                        )
                        memes.append(meme)
            
            self._cache_data(cache_key, memes)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("get_agent_memes", duration)
            
            return memes
            
        except Exception as e:
            self.logger.error(f"Failed to get agent memes for {agent_id}: {str(e)}")
            raise
    
    async def _update_meme_statistics(self, meme_id: str) -> None:
        """Update meme virality and adoption statistics."""
        if self.postgres_persistence:
            async with self.postgres_persistence.get_connection() as conn:
                await conn.execute("SELECT calculate_meme_virality($1)", meme_id)
    
    # =====================================================
    # Governance System
    # =====================================================
    
    async def submit_proposal(self, proposal: GovernanceProposal) -> str:
        """Submit a governance proposal for community voting."""
        start_time = datetime.now()
        
        try:
            # Store proposal in Store API
            proposal_key = f"proposal_{proposal.proposal_id}"
            proposal_data = {
                **asdict(proposal),
                "proposed_at": proposal.proposed_at.isoformat(),
                "voting_starts_at": proposal.voting_starts_at.isoformat() if proposal.voting_starts_at else None,
                "voting_ends_at": proposal.voting_ends_at.isoformat() if proposal.voting_ends_at else None,
            }
            
            if self.store:
                await self.store.aput(StoreNamespace.GOVERNANCE.value, proposal_key, proposal_data)
            
            # Store in PostgreSQL for persistence
            if self.postgres_persistence:
                async with self.postgres_persistence.get_connection() as conn:
                    await conn.execute("""
                        INSERT INTO governance_proposals (proposal_id, proposal_title, proposal_text, proposal_type,
                                                        proposed_by_agent_id, proposed_at, voting_starts_at, voting_ends_at,
                                                        status, required_majority, rationale, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """, proposal.proposal_id, proposal.proposal_title, proposal.proposal_text, proposal.proposal_type,
                    proposal.proposed_by_agent_id, proposal.proposed_at, proposal.voting_starts_at, proposal.voting_ends_at,
                    proposal.status, proposal.required_majority, proposal.rationale, json.dumps(proposal.metadata))
            
            # Notify community about new proposal
            await self._broadcast_community_event("governance_proposal", {
                "proposal_id": proposal.proposal_id,
                "title": proposal.proposal_title,
                "proposed_by": proposal.proposed_by_agent_id,
                "voting_starts": proposal.voting_starts_at.isoformat() if proposal.voting_starts_at else None
            })
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("submit_proposal", duration)
            
            return proposal.proposal_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit proposal {proposal.proposal_id}: {str(e)}")
            raise
    
    async def cast_vote(self, proposal_id: str, voter_agent_id: str, vote_choice: str, 
                       vote_strength: float = 1.0, reasoning: str = None) -> bool:
        """Cast a vote on a governance proposal."""
        start_time = datetime.now()
        
        try:
            vote_data = {
                "vote_choice": vote_choice,  # 'approve', 'reject', 'abstain'
                "vote_strength": vote_strength,
                "reasoning": reasoning,
                "cast_at": datetime.now().isoformat()
            }
            
            # Store vote in Store API
            vote_key = f"vote_{proposal_id}_{voter_agent_id}"
            if self.store:
                await self.store.aput(StoreNamespace.GOVERNANCE.value, vote_key, vote_data)
            
            # Store in PostgreSQL for persistence
            if self.postgres_persistence:
                async with self.postgres_persistence.get_connection() as conn:
                    await conn.execute("""
                        INSERT INTO votes (proposal_id, voter_agent_id, vote_choice, vote_strength, reasoning)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (proposal_id, voter_agent_id) DO UPDATE SET
                            vote_choice = EXCLUDED.vote_choice,
                            vote_strength = EXCLUDED.vote_strength,
                            reasoning = EXCLUDED.reasoning,
                            cast_at = CURRENT_TIMESTAMP
                    """, proposal_id, voter_agent_id, vote_choice, vote_strength, reasoning)
            
            # Check if voting is complete and tally results
            await self._check_voting_completion(proposal_id)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("cast_vote", duration)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cast vote for {voter_agent_id} on {proposal_id}: {str(e)}")
            return False
    
    async def get_active_proposals(self) -> List[GovernanceProposal]:
        """Get all active governance proposals."""
        start_time = datetime.now()
        
        try:
            proposals = []
            
            if self.postgres_persistence:
                async with self.postgres_persistence.get_connection() as conn:
                    rows = await conn.fetch("""
                        SELECT proposal_id, proposal_title, proposal_text, proposal_type,
                               proposed_by_agent_id, proposed_at, voting_starts_at, voting_ends_at,
                               status, required_majority, rationale, metadata
                        FROM governance_proposals
                        WHERE status IN ('draft', 'voting')
                        ORDER BY proposed_at DESC
                    """)
                    
                    for row in rows:
                        # Get votes for this proposal
                        vote_rows = await conn.fetch("""
                            SELECT voter_agent_id, vote_choice, vote_strength, reasoning, cast_at
                            FROM votes WHERE proposal_id = $1 AND is_valid = TRUE
                        """, row['proposal_id'])
                        
                        votes = {}
                        for vote_row in vote_rows:
                            votes[vote_row['voter_agent_id']] = {
                                "choice": vote_row['vote_choice'],
                                "strength": vote_row['vote_strength'],
                                "reasoning": vote_row['reasoning'],
                                "cast_at": vote_row['cast_at'].isoformat()
                            }
                        
                        metadata = json.loads(row['metadata']) if row['metadata'] else {}
                        
                        proposal = GovernanceProposal(
                            proposal_id=row['proposal_id'],
                            proposal_title=row['proposal_title'],
                            proposal_text=row['proposal_text'],
                            proposal_type=row['proposal_type'],
                            proposed_by_agent_id=row['proposed_by_agent_id'],
                            proposed_at=row['proposed_at'],
                            voting_starts_at=row['voting_starts_at'],
                            voting_ends_at=row['voting_ends_at'],
                            status=row['status'],
                            required_majority=row['required_majority'],
                            votes=votes,
                            rationale=row['rationale'],
                            metadata=metadata
                        )
                        proposals.append(proposal)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("get_active_proposals", duration)
            
            return proposals
            
        except Exception as e:
            self.logger.error(f"Failed to get active proposals: {str(e)}")
            raise
    
    async def _check_voting_completion(self, proposal_id: str) -> None:
        """Check if proposal voting is complete and process results."""
        if self.postgres_persistence:
            async with self.postgres_persistence.get_connection() as conn:
                # Get proposal details and voting results
                result = await conn.fetchrow("""
                    SELECT * FROM calculate_proposal_results($1)
                """, proposal_id)
                
                if result and result['total_votes'] > 0:
                    # Update proposal status if voting period ended or threshold reached
                    if result['passed']:
                        await conn.execute("""
                            UPDATE governance_proposals 
                            SET status = 'passed'
                            WHERE proposal_id = $1 AND status = 'voting'
                        """, proposal_id)
                        
                        # Create and enact rule if it's a rule creation proposal
                        await self._enact_proposal_if_passed(proposal_id)
    
    async def _enact_proposal_if_passed(self, proposal_id: str) -> None:
        """Enact a passed proposal by creating corresponding rules."""
        # Implementation would depend on proposal type
        # This is a placeholder for the full implementation
        pass
    
    # =====================================================
    # Social Influence Network
    # =====================================================
    
    async def update_influence_relationship(self, influencer_id: str, influenced_id: str, 
                                          influence_type: str, strength_change: float) -> bool:
        """Update influence relationship between agents."""
        start_time = datetime.now()
        
        try:
            influence_key = f"influence_{influencer_id}_{influenced_id}_{influence_type}"
            
            # Get current relationship
            current_influence = 0.0
            if self.store:
                current_data = await self.store.aget(StoreNamespace.INFLUENCE_NETWORK.value, influence_key)
                if current_data:
                    current_influence = current_data.get("influence_strength", 0.0)
            
            # Calculate new influence strength
            new_strength = max(0.0, min(1.0, current_influence + strength_change))
            
            if new_strength > 0.1:  # Only store significant influences
                influence_data = {
                    "influencer_agent_id": influencer_id,
                    "influenced_agent_id": influenced_id,
                    "influence_strength": new_strength,
                    "influence_type": influence_type,
                    "last_interaction": datetime.now().isoformat(),
                    "interaction_count": 1,
                    "created_at": datetime.now().isoformat(),
                    "is_active": True
                }
                
                if self.store:
                    await self.store.aput(StoreNamespace.INFLUENCE_NETWORK.value, influence_key, influence_data)
                
                # Store in PostgreSQL
                if self.postgres_persistence:
                    async with self.postgres_persistence.get_connection() as conn:
                        await conn.execute("""
                            INSERT INTO influence_network (influencer_agent_id, influenced_agent_id, 
                                                         influence_strength, influence_type, interaction_count)
                            VALUES ($1, $2, $3, $4, 1)
                            ON CONFLICT (influencer_agent_id, influenced_agent_id, influence_type) DO UPDATE SET
                                influence_strength = $3,
                                last_interaction = CURRENT_TIMESTAMP,
                                interaction_count = influence_network.interaction_count + 1,
                                is_active = TRUE
                        """, influencer_id, influenced_id, new_strength, influence_type)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("update_influence_relationship", duration)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update influence relationship: {str(e)}")
            return False
    
    async def get_agent_influence_network(self, agent_id: str) -> Dict[str, List[InfluenceRelation]]:
        """Get agent's influence network (who they influence and who influences them)."""
        start_time = datetime.now()
        
        try:
            network = {"influences": [], "influenced_by": []}
            
            if self.postgres_persistence:
                async with self.postgres_persistence.get_connection() as conn:
                    # Get who this agent influences
                    influences_rows = await conn.fetch("""
                        SELECT influenced_agent_id, influence_strength, influence_type, 
                               last_interaction, interaction_count, created_at, is_active
                        FROM influence_network
                        WHERE influencer_agent_id = $1 AND is_active = TRUE
                        ORDER BY influence_strength DESC
                    """, agent_id)
                    
                    for row in influences_rows:
                        network["influences"].append(InfluenceRelation(
                            influencer_agent_id=agent_id,
                            influenced_agent_id=row['influenced_agent_id'],
                            influence_strength=row['influence_strength'],
                            influence_type=row['influence_type'],
                            last_interaction=row['last_interaction'],
                            interaction_count=row['interaction_count'],
                            created_at=row['created_at'],
                            is_active=row['is_active']
                        ))
                    
                    # Get who influences this agent
                    influenced_by_rows = await conn.fetch("""
                        SELECT influencer_agent_id, influence_strength, influence_type,
                               last_interaction, interaction_count, created_at, is_active
                        FROM influence_network
                        WHERE influenced_agent_id = $1 AND is_active = TRUE
                        ORDER BY influence_strength DESC
                    """, agent_id)
                    
                    for row in influenced_by_rows:
                        network["influenced_by"].append(InfluenceRelation(
                            influencer_agent_id=row['influencer_agent_id'],
                            influenced_agent_id=agent_id,
                            influence_strength=row['influence_strength'],
                            influence_type=row['influence_type'],
                            last_interaction=row['last_interaction'],
                            interaction_count=row['interaction_count'],
                            created_at=row['created_at'],
                            is_active=row['is_active']
                        ))
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("get_agent_influence_network", duration)
            
            return network
            
        except Exception as e:
            self.logger.error(f"Failed to get influence network for {agent_id}: {str(e)}")
            raise
    
    # =====================================================
    # Community Events and Coordination
    # =====================================================
    
    async def _broadcast_community_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Broadcast a community event to all agents."""
        event_key = f"event_{datetime.now().timestamp()}_{event_type}"
        
        community_event = {
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": datetime.now().isoformat(),
            "participants": []
        }
        
        if self.store:
            await self.store.aput(StoreNamespace.COMMUNITY_EVENTS.value, event_key, community_event)
    
    async def get_recent_community_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent community events."""
        # This would typically fetch from Store API or PostgreSQL
        # Implementation depends on specific Store API capabilities
        pass
    
    # =====================================================
    # Performance and Maintenance
    # =====================================================
    
    async def get_cultural_metrics(self) -> Dict[str, Any]:
        """Get cultural system performance metrics."""
        start_time = datetime.now()
        
        try:
            metrics = {}
            
            if self.postgres_persistence:
                async with self.postgres_persistence.get_connection() as conn:
                    # Meme propagation effectiveness
                    meme_stats = await conn.fetchrow("""
                        SELECT COUNT(*) as total_memes,
                               COUNT(*) FILTER (WHERE virality > 0.5) as viral_memes,
                               AVG(virality) as avg_virality,
                               COUNT(DISTINCT origin_agent_id) as meme_creators
                        FROM cultural_memes WHERE is_active = TRUE
                    """)
                    
                    # Adoption rates
                    adoption_stats = await conn.fetchrow("""
                        SELECT COUNT(*) as total_adoptions,
                               COUNT(DISTINCT agent_id) as adopting_agents,
                               AVG(influence_strength) as avg_influence
                        FROM agent_meme_adoption WHERE is_active = TRUE
                    """)
                    
                    # Transmission success rates
                    transmission_stats = await conn.fetchrow("""
                        SELECT COUNT(*) as total_transmissions,
                               COUNT(*) FILTER (WHERE success = TRUE) as successful_transmissions,
                               AVG(transmission_strength) as avg_transmission_strength
                        FROM meme_transmissions
                        WHERE recorded_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                    """)
                    
                    metrics = {
                        "meme_statistics": dict(meme_stats) if meme_stats else {},
                        "adoption_statistics": dict(adoption_stats) if adoption_stats else {},
                        "transmission_statistics": dict(transmission_stats) if transmission_stats else {},
                        "operation_performance": {
                            "average_times_ms": {op: sum(times) / len(times) for op, times in self.operation_times.items() if times},
                            "operation_counts": self.operation_counts.copy()
                        }
                    }
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("get_cultural_metrics", duration)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get cultural metrics: {str(e)}")
            raise
    
    async def run_cultural_maintenance(self) -> Dict[str, Any]:
        """Run maintenance operations for cultural system."""
        try:
            results = {}
            
            if self.postgres_persistence:
                async with self.postgres_persistence.get_connection() as conn:
                    # Update influence network decay
                    decayed_influences = await conn.fetchval("""
                        SELECT update_influence_network()
                    """)
                    results["decayed_influences"] = decayed_influences
                    
                    # Update meme virality scores
                    virality_updates = 0
                    meme_ids = await conn.fetch("SELECT meme_id FROM cultural_memes WHERE is_active = TRUE")
                    for row in meme_ids:
                        await conn.execute("SELECT calculate_meme_virality($1)", row['meme_id'])
                        virality_updates += 1
                    
                    results["virality_updates"] = virality_updates
                    
                    # Clear expired cache entries
                    now = datetime.now()
                    expired_keys = [key for key, expiry in self._cache_expiry.items() if now >= expiry]
                    for key in expired_keys:
                        del self._cache_expiry[key]
                        self._meme_cache.pop(key, None)
                        self._rule_cache.pop(key, None)
                    
                    results["cache_cleanup"] = len(expired_keys)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to run cultural maintenance: {str(e)}")
            raise


# Helper functions for easy integration

def create_store_integration(store=None, postgres_persistence=None) -> MemoryStoreIntegration:
    """Create Store API integration instance."""
    return MemoryStoreIntegration(store, postgres_persistence)


# Example usage
if __name__ == "__main__":
    async def test_store_integration():
        """Test the Store API integration."""
        
        # Create integration (would normally get store from LangGraph framework)
        integration = create_store_integration()
        
        # Test meme creation and propagation
        meme = CulturalMeme(
            meme_id=str(uuid.uuid4()),
            meme_name="test_greeting",
            meme_type="behavior",
            description="Special greeting behavior",
            origin_agent_id="test_agent_001",
            strength=0.7,
            virality=0.6,
            stability=0.8,
            created_at=datetime.now(),
            adopters=set(),
            transmission_history=[],
            metadata={"test": True}
        )
        
        print(f"Created test meme: {meme.meme_name}")
        print("Store API integration loaded successfully")
    
    asyncio.run(test_store_integration())