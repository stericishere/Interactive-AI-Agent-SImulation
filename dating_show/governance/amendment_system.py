"""
File: amendment_system.py
Description: Amendment Proposal System for democratic rule changes.
Handles rule change proposals, community discussion simulation, amendment workflows,
and integration with the voting system for democratic governance evolution.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging

# Import governance and voting components
from .voting_system import DemocraticVotingSystem, VotingMechanism, VoteChoice
from ..agents.memory_structures.store_integration import (
    MemoryStoreIntegration, StoreNamespace, GovernanceProposal, GovernanceRule
)


class AmendmentType(Enum):
    """Types of amendments that can be proposed."""
    CONSTITUTIONAL = "constitutional"  # Fundamental governance changes
    PROCEDURAL = "procedural"  # Process and procedure changes
    BEHAVIORAL = "behavioral"  # Community behavior rules
    RESOURCE = "resource"  # Resource allocation and management
    ENFORCEMENT = "enforcement"  # Rule enforcement mechanisms
    EMERGENCY = "emergency"  # Emergency provisions


class AmendmentStatus(Enum):
    """Status states for amendment proposals."""
    DRAFT = "draft"  # Being prepared
    DISCUSSION = "discussion"  # Community discussion period
    REVIEW = "review"  # Formal review process
    VOTING = "voting"  # Active voting
    PASSED = "passed"  # Approved and ready for implementation
    REJECTED = "rejected"  # Rejected by community
    WITHDRAWN = "withdrawn"  # Withdrawn by proposer
    IMPLEMENTED = "implemented"  # Successfully implemented
    EXPIRED = "expired"  # Proposal expired


@dataclass
class AmendmentProposal:
    """Structure for amendment proposals."""
    proposal_id: str
    title: str
    summary: str
    full_text: str
    amendment_type: AmendmentType
    proposed_by_agent_id: str
    co_sponsors: Set[str]
    target_rule_ids: List[str]  # Rules being amended/replaced
    proposed_at: datetime
    discussion_period_end: Optional[datetime]
    voting_period_start: Optional[datetime] 
    voting_period_end: Optional[datetime]
    status: AmendmentStatus
    priority: int  # 1-10 scale
    impact_assessment: Dict[str, Any]
    community_support: Dict[str, float]  # agent_id -> support_score
    discussion_threads: List[Dict[str, Any]]
    required_majority: float
    metadata: Dict[str, Any]

    def __post_init__(self):
        if not self.co_sponsors:
            self.co_sponsors = set()
        if not self.metadata:
            self.metadata = {}


@dataclass
class DiscussionPost:
    """Community discussion post about an amendment."""
    post_id: str
    amendment_id: str
    author_agent_id: str
    content: str
    post_type: str  # 'comment', 'concern', 'suggestion', 'support', 'opposition'
    parent_post_id: Optional[str]  # For threaded discussions
    created_at: datetime
    endorsements: Set[str]  # agent_ids who endorse this post
    relevance_score: float
    sentiment: float  # -1.0 (negative) to 1.0 (positive)


@dataclass
class ImpactAssessment:
    """Assessment of amendment impact on the community."""
    assessment_id: str
    amendment_id: str
    assessed_by: str  # agent_id or 'system'
    social_impact: float  # -1.0 to 1.0
    economic_impact: float
    governance_impact: float
    implementation_complexity: float  # 0.0 to 1.0
    risk_factors: List[str]
    benefits: List[str]
    estimated_adoption_time: timedelta
    confidence_level: float  # 0.0 to 1.0
    created_at: datetime


class AmendmentSystem:
    """
    Amendment Proposal System for democratic rule evolution.
    Manages proposal lifecycle, community discussion, and voting coordination.
    """

    def __init__(self, voting_system: DemocraticVotingSystem,
                 store_integration: MemoryStoreIntegration,
                 postgres_persistence=None):
        """
        Initialize the Amendment System.
        
        Args:
            voting_system: Democratic voting system instance
            store_integration: Store API integration
            postgres_persistence: PostgreSQL persistence layer
        """
        self.voting_system = voting_system
        self.store_integration = store_integration
        self.postgres_persistence = postgres_persistence
        self.logger = logging.getLogger(f"{__name__}.AmendmentSystem")
        
        # Active proposals
        self.active_proposals = {}  # proposal_id -> AmendmentProposal
        self.discussion_threads = {}  # amendment_id -> List[DiscussionPost]
        
        # System configuration
        self.config = {
            "min_discussion_period_hours": 72,  # 3 days minimum
            "max_discussion_period_days": 14,  # 2 weeks maximum
            "min_co_sponsors_required": 2,
            "max_concurrent_proposals": 5,
            "discussion_engagement_threshold": 0.3,  # 30% of community
            "emergency_fast_track_hours": 24,
            "impact_assessment_required": True
        }
        
        # Metrics
        self.metrics = {
            "proposals_created": 0,
            "proposals_passed": 0,
            "proposals_rejected": 0,
            "average_discussion_posts": 0.0,
            "community_engagement_rate": 0.0
        }

    # =====================================================
    # Proposal Creation and Management
    # =====================================================

    async def create_amendment_proposal(self, title: str, summary: str, full_text: str,
                                      amendment_type: AmendmentType, proposed_by_agent_id: str,
                                      target_rule_ids: List[str] = None, priority: int = 5,
                                      co_sponsors: Set[str] = None) -> str:
        """
        Create a new amendment proposal.
        
        Args:
            title: Proposal title
            summary: Brief summary
            full_text: Full proposal text
            amendment_type: Type of amendment
            proposed_by_agent_id: ID of proposing agent
            target_rule_ids: Rules being amended
            priority: Priority level (1-10)
            co_sponsors: Initial co-sponsors
        
        Returns:
            proposal_id: Unique identifier for the proposal
        """
        try:
            # Check if maximum concurrent proposals reached
            active_count = len([p for p in self.active_proposals.values() 
                              if p.status in [AmendmentStatus.DISCUSSION, AmendmentStatus.REVIEW, AmendmentStatus.VOTING]])
            
            if active_count >= self.config["max_concurrent_proposals"]:
                raise ValueError(f"Maximum concurrent proposals ({self.config['max_concurrent_proposals']}) reached")
            
            # Generate proposal ID
            proposal_id = str(uuid.uuid4())
            
            # Validate co-sponsors
            if co_sponsors is None:
                co_sponsors = set()
            
            if len(co_sponsors) < self.config["min_co_sponsors_required"]:
                self.logger.warning(f"Proposal {proposal_id} has insufficient co-sponsors. Required: {self.config['min_co_sponsors_required']}")
            
            # Determine discussion period
            discussion_end = datetime.now() + timedelta(hours=self.config["min_discussion_period_hours"])
            if amendment_type == AmendmentType.EMERGENCY:
                discussion_end = datetime.now() + timedelta(hours=self.config["emergency_fast_track_hours"])
            
            # Set required majority based on amendment type
            majority_requirements = {
                AmendmentType.CONSTITUTIONAL: 0.75,  # 75% supermajority
                AmendmentType.PROCEDURAL: 0.60,     # 60% qualified majority
                AmendmentType.BEHAVIORAL: 0.55,     # 55% majority
                AmendmentType.RESOURCE: 0.65,       # 65% majority
                AmendmentType.ENFORCEMENT: 0.70,    # 70% majority
                AmendmentType.EMERGENCY: 0.80       # 80% supermajority
            }
            required_majority = majority_requirements.get(amendment_type, 0.60)
            
            # Create amendment proposal
            proposal = AmendmentProposal(
                proposal_id=proposal_id,
                title=title,
                summary=summary,
                full_text=full_text,
                amendment_type=amendment_type,
                proposed_by_agent_id=proposed_by_agent_id,
                co_sponsors=co_sponsors.copy(),
                target_rule_ids=target_rule_ids or [],
                proposed_at=datetime.now(),
                discussion_period_end=discussion_end,
                voting_period_start=None,
                voting_period_end=None,
                status=AmendmentStatus.DRAFT,
                priority=max(1, min(10, priority)),
                impact_assessment={},
                community_support={},
                discussion_threads=[],
                required_majority=required_majority,
                metadata={
                    "created_via": "amendment_system",
                    "version": "1.0"
                }
            )
            
            # Store proposal
            self.active_proposals[proposal_id] = proposal
            
            # Store in Store API
            await self._store_proposal_in_api(proposal)
            
            # Store in PostgreSQL
            if self.postgres_persistence:
                await self._store_proposal_in_db(proposal)
            
            # Initialize impact assessment if required
            if self.config["impact_assessment_required"]:
                await self._initiate_impact_assessment(proposal_id)
            
            # Begin discussion period
            await self._begin_discussion_period(proposal_id)
            
            self.metrics["proposals_created"] += 1
            
            self.logger.info(f"Created amendment proposal {proposal_id}: {title}")
            
            return proposal_id
            
        except Exception as e:
            self.logger.error(f"Failed to create amendment proposal: {str(e)}")
            raise

    async def add_co_sponsor(self, proposal_id: str, agent_id: str) -> bool:
        """Add a co-sponsor to an existing proposal."""
        try:
            if proposal_id not in self.active_proposals:
                return False
            
            proposal = self.active_proposals[proposal_id]
            
            # Only allow co-sponsorship during draft/discussion phase
            if proposal.status not in [AmendmentStatus.DRAFT, AmendmentStatus.DISCUSSION]:
                return False
            
            proposal.co_sponsors.add(agent_id)
            
            # Update storage
            await self._store_proposal_in_api(proposal)
            
            self.logger.info(f"Added co-sponsor {agent_id} to proposal {proposal_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add co-sponsor: {str(e)}")
            return False

    async def withdraw_proposal(self, proposal_id: str, requesting_agent_id: str) -> bool:
        """Withdraw a proposal (only by original proposer)."""
        try:
            if proposal_id not in self.active_proposals:
                return False
            
            proposal = self.active_proposals[proposal_id]
            
            # Only original proposer can withdraw
            if proposal.proposed_by_agent_id != requesting_agent_id:
                return False
            
            # Can't withdraw during voting
            if proposal.status == AmendmentStatus.VOTING:
                return False
            
            proposal.status = AmendmentStatus.WITHDRAWN
            
            # Update storage
            await self._store_proposal_in_api(proposal)
            
            # Notify community
            await self.store_integration._broadcast_community_event("proposal_withdrawn", {
                "proposal_id": proposal_id,
                "title": proposal.title,
                "withdrawn_by": requesting_agent_id
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to withdraw proposal: {str(e)}")
            return False

    # =====================================================
    # Community Discussion System
    # =====================================================

    async def post_discussion_comment(self, amendment_id: str, author_agent_id: str,
                                    content: str, post_type: str = "comment",
                                    parent_post_id: str = None) -> str:
        """Post a comment in amendment discussion."""
        try:
            if amendment_id not in self.active_proposals:
                raise ValueError(f"Amendment {amendment_id} not found")
            
            proposal = self.active_proposals[amendment_id]
            if proposal.status not in [AmendmentStatus.DISCUSSION, AmendmentStatus.REVIEW]:
                raise ValueError(f"Discussion not open for amendment {amendment_id}")
            
            # Create discussion post
            post = DiscussionPost(
                post_id=str(uuid.uuid4()),
                amendment_id=amendment_id,
                author_agent_id=author_agent_id,
                content=content,
                post_type=post_type,
                parent_post_id=parent_post_id,
                created_at=datetime.now(),
                endorsements=set(),
                relevance_score=0.0,
                sentiment=await self._analyze_sentiment(content)
            )
            
            # Store discussion post
            if amendment_id not in self.discussion_threads:
                self.discussion_threads[amendment_id] = []
            
            self.discussion_threads[amendment_id].append(post)
            
            # Update community support tracking
            await self._update_community_support(amendment_id, author_agent_id, post.sentiment)
            
            # Store in Store API
            await self._store_discussion_post(post)
            
            # Update proposal discussion thread references
            proposal.discussion_threads.append({
                "post_id": post.post_id,
                "author": author_agent_id,
                "type": post_type,
                "timestamp": post.created_at.isoformat()
            })
            
            await self._store_proposal_in_api(proposal)
            
            self.logger.info(f"Discussion post added to {amendment_id} by {author_agent_id}")
            
            return post.post_id
            
        except Exception as e:
            self.logger.error(f"Failed to post discussion comment: {str(e)}")
            raise

    async def endorse_discussion_post(self, post_id: str, endorser_agent_id: str) -> bool:
        """Endorse a discussion post."""
        try:
            # Find the post across all discussion threads
            for amendment_id, posts in self.discussion_threads.items():
                for post in posts:
                    if post.post_id == post_id:
                        post.endorsements.add(endorser_agent_id)
                        
                        # Update relevance score based on endorsements
                        post.relevance_score = len(post.endorsements) * 0.1
                        
                        # Store updated post
                        await self._store_discussion_post(post)
                        
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to endorse discussion post: {str(e)}")
            return False

    async def get_discussion_summary(self, amendment_id: str) -> Dict[str, Any]:
        """Get a summary of discussion for an amendment."""
        try:
            if amendment_id not in self.discussion_threads:
                return {"total_posts": 0, "participants": 0, "sentiment_distribution": {}}
            
            posts = self.discussion_threads[amendment_id]
            
            # Analyze discussion
            participants = set(post.author_agent_id for post in posts)
            post_types = {}
            sentiment_sum = 0.0
            
            for post in posts:
                post_types[post.post_type] = post_types.get(post.post_type, 0) + 1
                sentiment_sum += post.sentiment
            
            avg_sentiment = sentiment_sum / len(posts) if posts else 0.0
            
            # Categorize sentiment
            sentiment_distribution = {
                "positive": len([p for p in posts if p.sentiment > 0.2]),
                "negative": len([p for p in posts if p.sentiment < -0.2]),
                "neutral": len([p for p in posts if -0.2 <= p.sentiment <= 0.2])
            }
            
            return {
                "total_posts": len(posts),
                "participants": len(participants),
                "post_types": post_types,
                "average_sentiment": avg_sentiment,
                "sentiment_distribution": sentiment_distribution,
                "engagement_rate": len(participants) / max(1, self.voting_system.community_size)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get discussion summary: {str(e)}")
            return {}

    # =====================================================
    # Amendment Workflow Management
    # =====================================================

    async def advance_to_voting(self, proposal_id: str) -> bool:
        """Advance a proposal from discussion to voting phase."""
        try:
            if proposal_id not in self.active_proposals:
                return False
            
            proposal = self.active_proposals[proposal_id]
            
            # Validate readiness for voting
            if not await self._validate_voting_readiness(proposal):
                return False
            
            # Set voting period
            voting_start = datetime.now()
            voting_duration = 48  # Default 48 hours
            
            # Emergency amendments get shorter voting period
            if proposal.amendment_type == AmendmentType.EMERGENCY:
                voting_duration = 24
            
            proposal.voting_period_start = voting_start
            proposal.voting_period_end = voting_start + timedelta(hours=voting_duration)
            proposal.status = AmendmentStatus.VOTING
            
            # Create voting session
            eligible_voters = await self._determine_eligible_voters(proposal)
            voting_mechanism = await self._determine_voting_mechanism(proposal)
            
            session_id = await self.voting_system.create_voting_session(
                proposal_id=proposal_id,
                voting_mechanism=voting_mechanism,
                eligible_voters=eligible_voters,
                voting_duration_hours=voting_duration,
                required_threshold=proposal.required_majority
            )
            
            proposal.metadata["voting_session_id"] = session_id
            
            # Update storage
            await self._store_proposal_in_api(proposal)
            
            # Notify community
            await self.store_integration._broadcast_community_event("voting_started", {
                "proposal_id": proposal_id,
                "title": proposal.title,
                "voting_session_id": session_id,
                "voting_ends": proposal.voting_period_end.isoformat()
            })
            
            self.logger.info(f"Amendment {proposal_id} advanced to voting phase")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to advance proposal to voting: {str(e)}")
            return False

    async def process_voting_results(self, proposal_id: str) -> bool:
        """Process voting results for an amendment."""
        try:
            if proposal_id not in self.active_proposals:
                return False
            
            proposal = self.active_proposals[proposal_id]
            session_id = proposal.metadata.get("voting_session_id")
            
            if not session_id:
                return False
            
            # Get voting results
            results = await self.voting_system.get_voting_results(session_id)
            if not results:
                return False
            
            # Update proposal status based on results
            if results.threshold_met and results.winning_choice == VoteChoice.APPROVE:
                proposal.status = AmendmentStatus.PASSED
                self.metrics["proposals_passed"] += 1
                
                # Schedule implementation
                await self._schedule_implementation(proposal_id)
                
            else:
                proposal.status = AmendmentStatus.REJECTED
                self.metrics["proposals_rejected"] += 1
            
            # Store final results
            proposal.metadata["voting_results"] = {
                "threshold_met": results.threshold_met,
                "winning_choice": results.winning_choice.value,
                "participation_rate": results.participation_rate,
                "margin_of_victory": results.margin_of_victory
            }
            
            await self._store_proposal_in_api(proposal)
            
            # Notify community of results
            await self.store_integration._broadcast_community_event("voting_completed", {
                "proposal_id": proposal_id,
                "title": proposal.title,
                "result": proposal.status.value,
                "participation_rate": results.participation_rate
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process voting results: {str(e)}")
            return False

    # =====================================================
    # Helper Methods
    # =====================================================

    async def _begin_discussion_period(self, proposal_id: str) -> None:
        """Begin the discussion period for a proposal."""
        proposal = self.active_proposals[proposal_id]
        proposal.status = AmendmentStatus.DISCUSSION
        
        await self._store_proposal_in_api(proposal)
        
        # Notify community
        await self.store_integration._broadcast_community_event("discussion_started", {
            "proposal_id": proposal_id,
            "title": proposal.title,
            "discussion_ends": proposal.discussion_period_end.isoformat()
        })

    async def _analyze_sentiment(self, content: str) -> float:
        """Analyze sentiment of discussion content."""
        # Simple keyword-based sentiment analysis
        # In a real implementation, this would use NLP models
        positive_words = ["support", "good", "excellent", "beneficial", "approve", "agree"]
        negative_words = ["oppose", "bad", "terrible", "harmful", "reject", "disagree"]
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count == negative_count:
            return 0.0
        
        total_words = len(content.split())
        sentiment_score = (positive_count - negative_count) / max(1, total_words)
        return max(-1.0, min(1.0, sentiment_score * 10))  # Scale to -1.0 to 1.0

    async def _update_community_support(self, amendment_id: str, agent_id: str, sentiment: float) -> None:
        """Update community support tracking."""
        proposal = self.active_proposals[amendment_id]
        
        # Update or add support score
        current_support = proposal.community_support.get(agent_id, 0.0)
        # Weight recent sentiment more heavily
        new_support = (current_support * 0.7) + (sentiment * 0.3)
        proposal.community_support[agent_id] = max(-1.0, min(1.0, new_support))

    async def _validate_voting_readiness(self, proposal: AmendmentProposal) -> bool:
        """Validate that a proposal is ready for voting."""
        # Check minimum discussion period
        if datetime.now() < proposal.discussion_period_end:
            return False
        
        # Check minimum engagement
        discussion_summary = await self.get_discussion_summary(proposal.proposal_id)
        engagement_rate = discussion_summary.get("engagement_rate", 0.0)
        
        if engagement_rate < self.config["discussion_engagement_threshold"]:
            self.logger.warning(f"Insufficient engagement for proposal {proposal.proposal_id}: {engagement_rate}")
            # Don't block emergency amendments
            if proposal.amendment_type != AmendmentType.EMERGENCY:
                return False
        
        # Check co-sponsor requirement
        if len(proposal.co_sponsors) < self.config["min_co_sponsors_required"]:
            return False
        
        return True

    async def _determine_eligible_voters(self, proposal: AmendmentProposal) -> Set[str]:
        """Determine eligible voters for a proposal."""
        # Default: all community members
        # Could be customized based on amendment type or other criteria
        return {f"agent_{i}" for i in range(self.voting_system.community_size)}

    async def _determine_voting_mechanism(self, proposal: AmendmentProposal) -> VotingMechanism:
        """Determine appropriate voting mechanism for proposal."""
        if proposal.amendment_type == AmendmentType.CONSTITUTIONAL:
            return VotingMechanism.SUPERMAJORITY
        elif proposal.amendment_type == AmendmentType.EMERGENCY:
            return VotingMechanism.QUALIFIED_MAJORITY
        else:
            return VotingMechanism.SIMPLE_MAJORITY

    async def _initiate_impact_assessment(self, proposal_id: str) -> None:
        """Initiate automated impact assessment for proposal."""
        # This would integrate with analysis systems
        # Placeholder implementation
        pass

    async def _schedule_implementation(self, proposal_id: str) -> None:
        """Schedule implementation of passed amendment."""
        proposal = self.active_proposals[proposal_id]
        
        # Implementation would depend on amendment type and target rules
        proposal.metadata["scheduled_implementation"] = (
            datetime.now() + timedelta(days=7)
        ).isoformat()
        
        await self._store_proposal_in_api(proposal)

    async def _store_proposal_in_api(self, proposal: AmendmentProposal) -> None:
        """Store proposal in Store API."""
        if self.store_integration.store:
            proposal_data = asdict(proposal)
            proposal_data["co_sponsors"] = list(proposal.co_sponsors)
            proposal_data["proposed_at"] = proposal.proposed_at.isoformat()
            proposal_data["discussion_period_end"] = proposal.discussion_period_end.isoformat() if proposal.discussion_period_end else None
            proposal_data["voting_period_start"] = proposal.voting_period_start.isoformat() if proposal.voting_period_start else None
            proposal_data["voting_period_end"] = proposal.voting_period_end.isoformat() if proposal.voting_period_end else None
            proposal_data["amendment_type"] = proposal.amendment_type.value
            proposal_data["status"] = proposal.status.value
            
            await self.store_integration.store.aput(
                StoreNamespace.GOVERNANCE.value,
                f"amendment_{proposal.proposal_id}",
                proposal_data
            )

    async def _store_proposal_in_db(self, proposal: AmendmentProposal) -> None:
        """Store proposal in PostgreSQL."""
        if self.postgres_persistence:
            async with self.postgres_persistence.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO amendment_proposals (proposal_id, title, summary, full_text, amendment_type,
                                                   proposed_by_agent_id, co_sponsors, target_rule_ids, proposed_at,
                                                   discussion_period_end, status, priority, required_majority, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """, proposal.proposal_id, proposal.title, proposal.summary, proposal.full_text,
                proposal.amendment_type.value, proposal.proposed_by_agent_id, json.dumps(list(proposal.co_sponsors)),
                json.dumps(proposal.target_rule_ids), proposal.proposed_at, proposal.discussion_period_end,
                proposal.status.value, proposal.priority, proposal.required_majority, json.dumps(proposal.metadata))

    async def _store_discussion_post(self, post: DiscussionPost) -> None:
        """Store discussion post in Store API."""
        if self.store_integration.store:
            post_data = asdict(post)
            post_data["created_at"] = post.created_at.isoformat()
            post_data["endorsements"] = list(post.endorsements)
            
            await self.store_integration.store.aput(
                StoreNamespace.GOVERNANCE.value,
                f"discussion_{post.post_id}",
                post_data
            )

    # =====================================================
    # System Metrics and Health
    # =====================================================

    async def get_amendment_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive amendment system metrics."""
        try:
            active_discussions = len([p for p in self.active_proposals.values() 
                                    if p.status == AmendmentStatus.DISCUSSION])
            active_voting = len([p for p in self.active_proposals.values() 
                               if p.status == AmendmentStatus.VOTING])
            
            # Calculate average discussion engagement
            total_posts = sum(len(posts) for posts in self.discussion_threads.values())
            total_discussions = len(self.discussion_threads)
            avg_posts = total_posts / max(1, total_discussions)
            
            success_rate = (self.metrics["proposals_passed"] / 
                          max(1, self.metrics["proposals_passed"] + self.metrics["proposals_rejected"]))
            
            return {
                "active_discussions": active_discussions,
                "active_voting": active_voting,
                "total_proposals_created": self.metrics["proposals_created"],
                "proposals_passed": self.metrics["proposals_passed"],
                "proposals_rejected": self.metrics["proposals_rejected"],
                "success_rate": success_rate,
                "average_discussion_posts": avg_posts,
                "community_engagement": self.metrics["community_engagement_rate"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get amendment system metrics: {str(e)}")
            return {}


# Helper functions

def create_amendment_system(voting_system: DemocraticVotingSystem,
                           store_integration: MemoryStoreIntegration,
                           postgres_persistence=None) -> AmendmentSystem:
    """Create an AmendmentSystem instance."""
    return AmendmentSystem(voting_system, store_integration, postgres_persistence)


# Example usage
if __name__ == "__main__":
    async def test_amendment_system():
        """Test the Amendment System."""
        print("Amendment System loaded successfully")
        
    asyncio.run(test_amendment_system())