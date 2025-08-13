"""
File: voting_system.py
Description: Democratic voting system with LangGraph Store API integration.
Handles multi-agent voting coordination, ballot management, and vote aggregation
for democratic decision-making in the collective rules system.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging

# Import Store integration and governance types
try:
    from ..agents.memory_structures.store_integration import (
        MemoryStoreIntegration, StoreNamespace, GovernanceProposal, GovernanceRule
    )
    from ..agents.enhanced_agent_state import GovernanceData
except ImportError:
    try:
        from dating_show.agents.memory_structures.store_integration import (
            MemoryStoreIntegration, StoreNamespace, GovernanceProposal, GovernanceRule
        )
        from dating_show.agents.enhanced_agent_state import GovernanceData
    except ImportError:
        # Mock for testing
        class MemoryStoreIntegration:
            pass
        class StoreNamespace:
            GOVERNANCE = "governance"
        class GovernanceProposal:
            pass
        class GovernanceRule:
            pass
        class GovernanceData:
            pass


class VoteChoice(Enum):
    """Standard vote choices for proposals."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DELEGATE = "delegate"  # Delegate vote to another agent


class VotingMechanism(Enum):
    """Different voting mechanisms supported."""
    SIMPLE_MAJORITY = "simple_majority"  # > 50%
    SUPERMAJORITY = "supermajority"  # >= 2/3
    UNANIMOUS = "unanimous"  # 100%
    QUALIFIED_MAJORITY = "qualified_majority"  # Custom threshold
    RANKED_CHOICE = "ranked_choice"  # Ranked preference voting


@dataclass
class Vote:
    """Individual vote cast by an agent."""
    vote_id: str
    voter_agent_id: str
    proposal_id: str
    vote_choice: VoteChoice
    vote_strength: float  # Weighted voting (0.0 to 1.0)
    reasoning: Optional[str]
    delegation_chain: List[str]  # For delegated voting
    cast_at: datetime
    is_final: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VotingSession:
    """A voting session for a specific proposal."""
    session_id: str
    proposal_id: str
    voting_mechanism: VotingMechanism
    required_threshold: float
    eligible_voters: Set[str]
    voting_start: datetime
    voting_end: datetime
    votes_cast: Dict[str, Vote]  # voter_id -> Vote
    is_active: bool
    results_calculated: bool = False
    winning_choice: Optional[VoteChoice] = None
    participation_rate: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VotingResults:
    """Results of a completed voting session."""
    session_id: str
    proposal_id: str
    total_eligible_voters: int
    votes_cast: Dict[VoteChoice, int]
    weighted_results: Dict[VoteChoice, float]
    winning_choice: VoteChoice
    margin_of_victory: float
    participation_rate: float
    threshold_met: bool
    abstention_rate: float
    calculated_at: datetime
    vote_breakdown: Dict[str, Dict[str, Any]]  # Detailed per-agent breakdown


class DemocraticVotingSystem:
    """
    Democratic Process Engine for multi-agent voting coordination.
    Integrates with LangGraph Store API for real-time voting coordination.
    """

    def __init__(self, store_integration: MemoryStoreIntegration, 
                 postgres_persistence=None, community_size: int = 50):
        """
        Initialize the Democratic Voting System.
        
        Args:
            store_integration: Store API integration instance
            postgres_persistence: PostgreSQL persistence layer
            community_size: Expected community size for optimization
        """
        self.store_integration = store_integration
        self.postgres_persistence = postgres_persistence
        self.community_size = community_size
        self.logger = logging.getLogger(f"{__name__}.DemocraticVotingSystem")
        
        # Active voting sessions
        self.active_sessions = {}  # session_id -> VotingSession
        
        # Vote validation rules
        self.validation_rules = {
            "min_voting_period_hours": 24,
            "max_voting_period_days": 30,
            "min_participation_rate": 0.25,  # 25% minimum participation
            "delegation_chain_limit": 3,  # Max delegation depth
        }
        
        # Performance metrics
        self.metrics = {
            "sessions_created": 0,
            "votes_processed": 0,
            "average_participation": 0.0,
            "voting_efficiency": 0.0
        }

    # =====================================================
    # Voting Session Management
    # =====================================================

    async def create_voting_session(self, proposal_id: str, voting_mechanism: VotingMechanism,
                                  eligible_voters: Set[str], voting_duration_hours: int = 48,
                                  required_threshold: float = None) -> str:
        """
        Create a new voting session for a proposal.
        
        Args:
            proposal_id: ID of the proposal being voted on
            voting_mechanism: Type of voting mechanism to use
            eligible_voters: Set of agent IDs eligible to vote
            voting_duration_hours: How long voting remains open
            required_threshold: Custom threshold (if applicable)
        
        Returns:
            session_id: Unique identifier for the voting session
        """
        start_time = datetime.now()
        
        try:
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Set default threshold based on mechanism
            if required_threshold is None:
                threshold_map = {
                    VotingMechanism.SIMPLE_MAJORITY: 0.5,
                    VotingMechanism.SUPERMAJORITY: 2/3,
                    VotingMechanism.UNANIMOUS: 1.0,
                    VotingMechanism.QUALIFIED_MAJORITY: 0.6,
                    VotingMechanism.RANKED_CHOICE: 0.5
                }
                required_threshold = threshold_map.get(voting_mechanism, 0.5)
            
            # Validate voting duration
            min_hours = self.validation_rules["min_voting_period_hours"]
            max_hours = self.validation_rules["max_voting_period_days"] * 24
            voting_duration_hours = max(min_hours, min(max_hours, voting_duration_hours))
            
            # Create voting session
            voting_session = VotingSession(
                session_id=session_id,
                proposal_id=proposal_id,
                voting_mechanism=voting_mechanism,
                required_threshold=required_threshold,
                eligible_voters=eligible_voters.copy(),
                voting_start=start_time,
                voting_end=start_time + timedelta(hours=voting_duration_hours),
                votes_cast={},
                is_active=True,
                metadata={
                    "created_by_system": True,
                    "voting_duration_hours": voting_duration_hours,
                    "estimated_participation": min(1.0, len(eligible_voters) / self.community_size)
                }
            )
            
            # Store session in memory
            self.active_sessions[session_id] = voting_session
            
            # Store in Store API for cross-agent access
            session_data = asdict(voting_session)
            session_data["eligible_voters"] = list(eligible_voters)
            session_data["voting_start"] = voting_session.voting_start.isoformat()
            session_data["voting_end"] = voting_session.voting_end.isoformat()
            
            if self.store_integration.store:
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"voting_session_{session_id}",
                    session_data
                )
            
            # Store in PostgreSQL for persistence
            if self.postgres_persistence:
                async with self.postgres_persistence.get_connection() as conn:
                    await conn.execute("""
                        INSERT INTO voting_sessions (session_id, proposal_id, voting_mechanism, required_threshold,
                                                   eligible_voters, voting_start, voting_end, is_active, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """, session_id, proposal_id, voting_mechanism.value, required_threshold,
                    json.dumps(list(eligible_voters)), voting_session.voting_start, voting_session.voting_end,
                    True, json.dumps(voting_session.metadata))
            
            # Notify eligible voters about the new voting session
            await self._notify_voters_of_new_session(session_id, eligible_voters)
            
            self.metrics["sessions_created"] += 1
            
            self.logger.info(f"Created voting session {session_id} for proposal {proposal_id} "
                           f"with {len(eligible_voters)} eligible voters")
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create voting session: {str(e)}")
            raise

    async def cast_vote(self, session_id: str, voter_agent_id: str, vote_choice: VoteChoice,
                       vote_strength: float = 1.0, reasoning: str = None,
                       delegate_to: str = None) -> bool:
        """
        Cast a vote in an active voting session.
        
        Args:
            session_id: ID of the voting session
            voter_agent_id: ID of the agent casting the vote
            vote_choice: The vote choice (approve/reject/abstain/delegate)
            vote_strength: Weight of the vote (0.0 to 1.0)
            reasoning: Optional reasoning for the vote
            delegate_to: Agent ID to delegate vote to (if delegating)
        
        Returns:
            bool: True if vote was successfully cast
        """
        start_time = datetime.now()
        
        try:
            # Validate session exists and is active
            if session_id not in self.active_sessions:
                # Try to load from Store API
                await self._load_session_from_store(session_id)
            
            session = self.active_sessions.get(session_id)
            if not session or not session.is_active:
                self.logger.warning(f"Voting session {session_id} is not active")
                return False
            
            # Check if voting period is still open
            if datetime.now() > session.voting_end:
                self.logger.warning(f"Voting period ended for session {session_id}")
                await self._close_voting_session(session_id)
                return False
            
            # Validate voter eligibility
            if voter_agent_id not in session.eligible_voters:
                self.logger.warning(f"Agent {voter_agent_id} not eligible to vote in session {session_id}")
                return False
            
            # Handle delegation
            delegation_chain = []
            if vote_choice == VoteChoice.DELEGATE and delegate_to:
                if not await self._validate_delegation(voter_agent_id, delegate_to, session):
                    return False
                delegation_chain = [voter_agent_id, delegate_to]
            
            # Create vote record
            vote = Vote(
                vote_id=str(uuid.uuid4()),
                voter_agent_id=voter_agent_id,
                proposal_id=session.proposal_id,
                vote_choice=vote_choice,
                vote_strength=max(0.0, min(1.0, vote_strength)),
                reasoning=reasoning,
                delegation_chain=delegation_chain,
                cast_at=start_time,
                is_final=True,
                metadata={
                    "session_id": session_id,
                    "voting_mechanism": session.voting_mechanism.value
                }
            )
            
            # Store vote in session
            session.votes_cast[voter_agent_id] = vote
            
            # Store vote in Store API
            vote_data = asdict(vote)
            vote_data["cast_at"] = vote.cast_at.isoformat()
            
            if self.store_integration.store:
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"vote_{session_id}_{voter_agent_id}",
                    vote_data
                )
            
            # Store in PostgreSQL
            if self.postgres_persistence:
                async with self.postgres_persistence.get_connection() as conn:
                    await conn.execute("""
                        INSERT INTO votes_cast (vote_id, session_id, voter_agent_id, proposal_id, 
                                              vote_choice, vote_strength, reasoning, delegation_chain,
                                              cast_at, is_final, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (session_id, voter_agent_id) DO UPDATE SET
                            vote_id = EXCLUDED.vote_id,
                            vote_choice = EXCLUDED.vote_choice,
                            vote_strength = EXCLUDED.vote_strength,
                            reasoning = EXCLUDED.reasoning,
                            cast_at = EXCLUDED.cast_at,
                            is_final = EXCLUDED.is_final,
                            metadata = EXCLUDED.metadata
                    """, vote.vote_id, session_id, voter_agent_id, session.proposal_id,
                    vote_choice.value, vote.vote_strength, reasoning, json.dumps(delegation_chain),
                    vote.cast_at, vote.is_final, json.dumps(vote.metadata))
            
            # Update session participation
            session.participation_rate = len(session.votes_cast) / len(session.eligible_voters)
            
            # Check if early termination conditions are met
            await self._check_early_termination(session_id)
            
            self.metrics["votes_processed"] += 1
            
            self.logger.info(f"Vote cast by {voter_agent_id} in session {session_id}: {vote_choice.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cast vote: {str(e)}")
            return False

    async def get_voting_results(self, session_id: str) -> Optional[VotingResults]:
        """
        Get the results of a voting session.
        
        Args:
            session_id: ID of the voting session
        
        Returns:
            VotingResults if session is complete, None otherwise
        """
        try:
            if session_id not in self.active_sessions:
                await self._load_session_from_store(session_id)
            
            session = self.active_sessions.get(session_id)
            if not session:
                return None
            
            # Calculate results if not already calculated
            if not session.results_calculated or session.is_active:
                await self._calculate_session_results(session_id)
                session = self.active_sessions[session_id]  # Refresh session
            
            # Create results object
            vote_counts = {}
            weighted_results = {}
            vote_breakdown = {}
            
            total_weight = 0.0
            for choice in VoteChoice:
                vote_counts[choice] = 0
                weighted_results[choice] = 0.0
            
            for voter_id, vote in session.votes_cast.items():
                vote_counts[vote.vote_choice] += 1
                weighted_results[vote.vote_choice] += vote.vote_strength
                total_weight += vote.vote_strength
                
                vote_breakdown[voter_id] = {
                    "choice": vote.vote_choice.value,
                    "strength": vote.vote_strength,
                    "reasoning": vote.reasoning,
                    "cast_at": vote.cast_at.isoformat()
                }
            
            # Normalize weighted results
            if total_weight > 0:
                for choice in weighted_results:
                    weighted_results[choice] = weighted_results[choice] / total_weight
            
            # Determine winning choice and margin
            winning_choice = max(weighted_results.keys(), key=lambda x: weighted_results[x])
            sorted_results = sorted(weighted_results.items(), key=lambda x: x[1], reverse=True)
            margin_of_victory = sorted_results[0][1] - (sorted_results[1][1] if len(sorted_results) > 1 else 0)
            
            # Check if threshold was met
            threshold_met = weighted_results[winning_choice] >= session.required_threshold
            
            results = VotingResults(
                session_id=session_id,
                proposal_id=session.proposal_id,
                total_eligible_voters=len(session.eligible_voters),
                votes_cast=vote_counts,
                weighted_results=weighted_results,
                winning_choice=winning_choice,
                margin_of_victory=margin_of_victory,
                participation_rate=session.participation_rate,
                threshold_met=threshold_met,
                abstention_rate=weighted_results.get(VoteChoice.ABSTAIN, 0.0),
                calculated_at=datetime.now(),
                vote_breakdown=vote_breakdown
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get voting results for session {session_id}: {str(e)}")
            return None

    # =====================================================
    # Session Management and Coordination
    # =====================================================

    async def _notify_voters_of_new_session(self, session_id: str, eligible_voters: Set[str]) -> None:
        """Notify eligible voters about a new voting session."""
        notification_data = {
            "type": "new_voting_session",
            "session_id": session_id,
            "eligible_voters": list(eligible_voters),
            "timestamp": datetime.now().isoformat()
        }
        
        await self.store_integration._broadcast_community_event("voting_notification", notification_data)

    async def _validate_delegation(self, voter_id: str, delegate_to: str, session: VotingSession) -> bool:
        """Validate that a vote delegation is allowed."""
        # Check if delegate is eligible voter
        if delegate_to not in session.eligible_voters:
            return False
        
        # Check delegation chain length
        chain_length = len(session.votes_cast.get(voter_id, Vote(
            "", "", "", VoteChoice.APPROVE, 0, None, [], datetime.now()
        )).delegation_chain)
        
        if chain_length >= self.validation_rules["delegation_chain_limit"]:
            return False
        
        # Prevent circular delegation (basic check)
        if delegate_to == voter_id:
            return False
        
        return True

    async def _check_early_termination(self, session_id: str) -> bool:
        """Check if voting session can be terminated early."""
        session = self.active_sessions[session_id]
        
        # Early termination conditions
        participation_rate = len(session.votes_cast) / len(session.eligible_voters)
        
        # If 100% participation reached
        if participation_rate >= 1.0:
            await self._close_voting_session(session_id)
            return True
        
        # If unanimous decision with sufficient participation
        if participation_rate >= self.validation_rules["min_participation_rate"]:
            choices = set(vote.vote_choice for vote in session.votes_cast.values()
                         if vote.vote_choice != VoteChoice.DELEGATE)
            if len(choices) == 1 and VoteChoice.ABSTAIN not in choices:
                await self._close_voting_session(session_id)
                return True
        
        return False

    async def _close_voting_session(self, session_id: str) -> None:
        """Close and finalize a voting session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.is_active = False
            
            # Calculate final results
            await self._calculate_session_results(session_id)
            
            # Update Store API
            if self.store_integration.store:
                session_data = asdict(session)
                session_data["eligible_voters"] = list(session.eligible_voters)
                session_data["voting_start"] = session.voting_start.isoformat()
                session_data["voting_end"] = session.voting_end.isoformat()
                
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"voting_session_{session_id}",
                    session_data
                )
            
            # Notify community of results
            results = await self.get_voting_results(session_id)
            if results:
                await self.store_integration._broadcast_community_event("voting_completed", {
                    "session_id": session_id,
                    "proposal_id": session.proposal_id,
                    "winning_choice": results.winning_choice.value,
                    "participation_rate": results.participation_rate,
                    "threshold_met": results.threshold_met
                })

    async def _calculate_session_results(self, session_id: str) -> None:
        """Calculate and store voting session results."""
        session = self.active_sessions[session_id]
        session.results_calculated = True
        
        # Update metrics
        self.metrics["average_participation"] = (
            (self.metrics["average_participation"] * (self.metrics["sessions_created"] - 1) + 
             session.participation_rate) / self.metrics["sessions_created"]
        )

    async def _load_session_from_store(self, session_id: str) -> bool:
        """Load a voting session from Store API."""
        try:
            if self.store_integration.store:
                session_data = await self.store_integration.store.aget(
                    StoreNamespace.GOVERNANCE.value,
                    f"voting_session_{session_id}"
                )
                
                if session_data:
                    # Reconstruct VotingSession object
                    session = VotingSession(
                        session_id=session_data["session_id"],
                        proposal_id=session_data["proposal_id"],
                        voting_mechanism=VotingMechanism(session_data["voting_mechanism"]),
                        required_threshold=session_data["required_threshold"],
                        eligible_voters=set(session_data["eligible_voters"]),
                        voting_start=datetime.fromisoformat(session_data["voting_start"]),
                        voting_end=datetime.fromisoformat(session_data["voting_end"]),
                        votes_cast={},  # Will be loaded separately if needed
                        is_active=session_data["is_active"],
                        results_calculated=session_data.get("results_calculated", False),
                        metadata=session_data.get("metadata", {})
                    )
                    
                    self.active_sessions[session_id] = session
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load session {session_id} from store: {str(e)}")
            return False

    # =====================================================
    # System Metrics and Health
    # =====================================================

    async def get_voting_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive voting system performance metrics."""
        try:
            active_sessions_count = len([s for s in self.active_sessions.values() if s.is_active])
            
            # Calculate recent participation trends
            recent_participation = []
            for session in self.active_sessions.values():
                if not session.is_active and session.results_calculated:
                    recent_participation.append(session.participation_rate)
            
            avg_recent_participation = (
                sum(recent_participation) / len(recent_participation)
                if recent_participation else 0.0
            )
            
            return {
                "active_sessions": active_sessions_count,
                "total_sessions_created": self.metrics["sessions_created"],
                "total_votes_processed": self.metrics["votes_processed"],
                "average_participation_rate": self.metrics["average_participation"],
                "recent_participation_rate": avg_recent_participation,
                "system_efficiency": min(1.0, self.metrics["votes_processed"] / max(1, self.metrics["sessions_created"] * 10)),
                "community_engagement": min(1.0, avg_recent_participation / self.validation_rules["min_participation_rate"])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get voting system metrics: {str(e)}")
            return {}

    async def cleanup_completed_sessions(self, days_old: int = 30) -> int:
        """Clean up old completed voting sessions."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cleaned_count = 0
            
            sessions_to_remove = []
            for session_id, session in self.active_sessions.items():
                if not session.is_active and session.voting_end < cutoff_date:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
                cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup completed sessions: {str(e)}")
            return 0


# Helper functions for integration

def create_democratic_voting_system(store_integration: MemoryStoreIntegration, 
                                  postgres_persistence=None, community_size: int = 50) -> DemocraticVotingSystem:
    """Create a DemocraticVotingSystem instance."""
    return DemocraticVotingSystem(store_integration, postgres_persistence, community_size)


# Example usage
if __name__ == "__main__":
    async def test_voting_system():
        """Test the Democratic Voting System."""
        from ..agents.memory_structures.store_integration import create_store_integration
        
        # Create integration (would normally get store from LangGraph framework)
        store_integration = create_store_integration()
        voting_system = create_democratic_voting_system(store_integration)
        
        # Test session creation
        eligible_voters = {f"agent_{i}" for i in range(10)}
        session_id = await voting_system.create_voting_session(
            proposal_id="test_proposal_001",
            voting_mechanism=VotingMechanism.SIMPLE_MAJORITY,
            eligible_voters=eligible_voters,
            voting_duration_hours=48
        )
        
        print(f"Created voting session: {session_id}")
        
        # Test vote casting
        success = await voting_system.cast_vote(
            session_id=session_id,
            voter_agent_id="agent_0",
            vote_choice=VoteChoice.APPROVE,
            reasoning="This proposal benefits the community"
        )
        
        print(f"Vote cast successfully: {success}")
        print("Democratic Voting System loaded successfully")
    
    asyncio.run(test_voting_system())