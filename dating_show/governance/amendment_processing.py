"""
File: amendment_processing.py
Description: Constitutional amendment processing workflows with community notification.
Handles amendment proposals, validation, integration, and community notification for the constitutional system.
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import difflib

# Import constitutional storage and voting systems
from .constitution_storage import (
    ConstitutionalStorage, ConstitutionalRule, ConstitutionalAmendment,
    RuleType, RuleStatus, create_constitutional_rule
)
try:
    from .voting_system import DemocraticVotingSystem, VotingMechanism, VoteChoice
except ImportError:
    from dating_show.governance.voting_system import DemocraticVotingSystem, VotingMechanism, VoteChoice

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


class AmendmentType(Enum):
    """Types of constitutional amendments."""
    CREATE_NEW_RULE = "create_new_rule"
    MODIFY_EXISTING_RULE = "modify_existing_rule"
    REPEAL_RULE = "repeal_rule"
    MERGE_RULES = "merge_rules"
    REORGANIZE_RULES = "reorganize_rules"


class AmendmentStatus(Enum):
    """Status of amendment proposals."""
    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    VOTING_OPEN = "voting_open"
    VOTING_CLOSED = "voting_closed"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    FAILED_IMPLEMENTATION = "failed_implementation"


@dataclass
class AmendmentProposal:
    """A proposal to amend the constitutional system."""
    amendment_id: str
    title: str
    description: str
    amendment_type: AmendmentType
    target_rule_ids: List[str]  # Rules affected by this amendment
    proposed_changes: Dict[str, Any]  # Detailed changes
    justification: str
    impact_assessment: Dict[str, Any]
    proposed_by: str
    proposed_at: datetime
    status: AmendmentStatus
    voting_session_id: Optional[str] = None
    community_feedback: List[Dict[str, Any]] = field(default_factory=list)
    review_period_end: Optional[datetime] = None
    implementation_deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.community_feedback:
            self.community_feedback = []
        if not self.metadata:
            self.metadata = {}


@dataclass
class AmendmentImpactAnalysis:
    """Analysis of amendment impact on the constitutional system."""
    amendment_id: str
    affected_rules: List[str]
    conflicting_rules: List[str]
    dependent_rules: List[str]
    community_impact_score: float  # 0.0 to 1.0
    implementation_complexity: str  # "low", "medium", "high"
    resource_requirements: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    backwards_compatibility: bool
    estimated_adoption_time: int  # days
    stakeholder_analysis: Dict[str, List[str]]


class AmendmentProcessor:
    """
    Constitutional amendment processing engine.
    Handles the full lifecycle of constitutional amendments from proposal to implementation.
    """

    def __init__(self, constitutional_storage: ConstitutionalStorage,
                 voting_system: DemocraticVotingSystem,
                 store_integration: MemoryStoreIntegration):
        """
        Initialize the Amendment Processor.
        
        Args:
            constitutional_storage: Constitutional storage system
            voting_system: Democratic voting system
            store_integration: Store API integration
        """
        self.constitutional_storage = constitutional_storage
        self.voting_system = voting_system
        self.store_integration = store_integration
        self.logger = logging.getLogger(f"{__name__}.AmendmentProcessor")
        
        # Active amendment proposals
        self.active_proposals = {}  # amendment_id -> AmendmentProposal
        self.pending_implementations = {}  # amendment_id -> implementation_data
        
        # Configuration
        self.config = {
            "review_period_days": 7,
            "voting_period_hours": 48,
            "min_community_feedback": 3,
            "supermajority_threshold": 0.67,
            "implementation_timeout_days": 30,
            "max_concurrent_amendments": 5
        }
        
        # Metrics
        self.metrics = {
            "proposals_submitted": 0,
            "proposals_approved": 0,
            "proposals_rejected": 0,
            "implementations_completed": 0,
            "average_review_time": 0.0
        }

    # =====================================================
    # Amendment Proposal Management
    # =====================================================

    async def submit_amendment_proposal(self, title: str, description: str,
                                      amendment_type: AmendmentType,
                                      target_rule_ids: List[str],
                                      proposed_changes: Dict[str, Any],
                                      justification: str,
                                      proposed_by: str) -> Optional[str]:
        """
        Submit a new amendment proposal.
        
        Args:
            title: Amendment title
            description: Detailed description
            amendment_type: Type of amendment
            target_rule_ids: Rules affected
            proposed_changes: Specific changes requested
            justification: Reasoning for the amendment
            proposed_by: Agent ID of proposer
        
        Returns:
            amendment_id if successful, None otherwise
        """
        try:
            # Check if too many concurrent amendments
            active_count = len([p for p in self.active_proposals.values() 
                              if p.status in [AmendmentStatus.PROPOSED, AmendmentStatus.UNDER_REVIEW, AmendmentStatus.VOTING_OPEN]])
            
            if active_count >= self.config["max_concurrent_amendments"]:
                self.logger.warning(f"Too many concurrent amendments ({active_count}), rejecting new proposal")
                return None
            
            # Generate amendment ID
            amendment_id = f"amendment_{str(uuid.uuid4())[:8]}"
            
            # Perform initial impact assessment
            impact_assessment = await self._perform_impact_analysis(
                amendment_type, target_rule_ids, proposed_changes
            )
            
            # Create amendment proposal
            proposal = AmendmentProposal(
                amendment_id=amendment_id,
                title=title,
                description=description,
                amendment_type=amendment_type,
                target_rule_ids=target_rule_ids,
                proposed_changes=proposed_changes,
                justification=justification,
                impact_assessment=asdict(impact_assessment) if impact_assessment else {},
                proposed_by=proposed_by,
                proposed_at=datetime.now(),
                status=AmendmentStatus.PROPOSED,
                review_period_end=datetime.now() + timedelta(days=self.config["review_period_days"]),
                metadata={
                    "initial_impact_score": impact_assessment.community_impact_score if impact_assessment else 0.0,
                    "complexity": impact_assessment.implementation_complexity if impact_assessment else "unknown"
                }
            )
            
            # Store proposal
            success = await self._store_amendment_proposal(proposal)
            if not success:
                return None
            
            self.active_proposals[amendment_id] = proposal
            
            # Notify community
            await self._notify_community_of_proposal(proposal)
            
            # Schedule review period end
            asyncio.create_task(self._schedule_review_period_end(amendment_id))
            
            self.metrics["proposals_submitted"] += 1
            self.logger.info(f"Amendment proposal {amendment_id} submitted by {proposed_by}")
            
            return amendment_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit amendment proposal: {str(e)}")
            return None

    async def add_community_feedback(self, amendment_id: str, agent_id: str,
                                   feedback_type: str, content: str,
                                   supporting: bool) -> bool:
        """
        Add community feedback to an amendment proposal.
        
        Args:
            amendment_id: ID of the amendment
            agent_id: Agent providing feedback
            feedback_type: Type of feedback ("comment", "concern", "suggestion")
            content: Feedback content
            supporting: Whether this feedback supports the amendment
        
        Returns:
            bool: True if feedback added successfully
        """
        try:
            if amendment_id not in self.active_proposals:
                return False
            
            proposal = self.active_proposals[amendment_id]
            if proposal.status not in [AmendmentStatus.PROPOSED, AmendmentStatus.UNDER_REVIEW]:
                return False
            
            # Add feedback
            feedback = {
                "feedback_id": str(uuid.uuid4())[:8],
                "agent_id": agent_id,
                "feedback_type": feedback_type,
                "content": content,
                "supporting": supporting,
                "timestamp": datetime.now().isoformat()
            }
            
            proposal.community_feedback.append(feedback)
            
            # Update proposal status if enough feedback received
            if len(proposal.community_feedback) >= self.config["min_community_feedback"]:
                proposal.status = AmendmentStatus.UNDER_REVIEW
            
            # Store updated proposal
            await self._store_amendment_proposal(proposal)
            
            # Notify community of new feedback
            await self._notify_community_of_feedback(amendment_id, feedback)
            
            self.logger.info(f"Community feedback added to amendment {amendment_id} by {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add community feedback: {str(e)}")
            return False

    async def initiate_amendment_voting(self, amendment_id: str) -> Optional[str]:
        """
        Initiate voting on an amendment proposal.
        
        Args:
            amendment_id: ID of the amendment to vote on
        
        Returns:
            voting_session_id if successful, None otherwise
        """
        try:
            proposal = self.active_proposals.get(amendment_id)
            if not proposal or proposal.status != AmendmentStatus.UNDER_REVIEW:
                return None
            
            # Determine voting mechanism based on amendment type and affected rules
            voting_mechanism = await self._determine_voting_mechanism(proposal)
            
            # Get eligible voters (all active community members)
            eligible_voters = await self._get_eligible_voters()
            
            # Create voting session
            voting_session_id = await self.voting_system.create_voting_session(
                proposal_id=amendment_id,
                voting_mechanism=voting_mechanism,
                eligible_voters=eligible_voters,
                voting_duration_hours=self.config["voting_period_hours"]
            )
            
            if voting_session_id:
                proposal.voting_session_id = voting_session_id
                proposal.status = AmendmentStatus.VOTING_OPEN
                
                await self._store_amendment_proposal(proposal)
                
                # Notify community of voting
                await self._notify_community_of_voting(proposal)
                
                # Schedule voting completion check
                asyncio.create_task(self._monitor_amendment_voting(amendment_id))
                
                self.logger.info(f"Voting initiated for amendment {amendment_id}")
                return voting_session_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to initiate amendment voting: {str(e)}")
            return None

    # =====================================================
    # Amendment Implementation
    # =====================================================

    async def process_voting_results(self, amendment_id: str) -> bool:
        """
        Process the results of amendment voting and implement if approved.
        
        Args:
            amendment_id: ID of the amendment
        
        Returns:
            bool: True if processing completed successfully
        """
        try:
            proposal = self.active_proposals.get(amendment_id)
            if not proposal or not proposal.voting_session_id:
                return False
            
            # Get voting results
            results = await self.voting_system.get_voting_results(proposal.voting_session_id)
            if not results:
                return False
            
            proposal.status = AmendmentStatus.VOTING_CLOSED
            
            # Determine if amendment passed
            amendment_passed = (
                results.winning_choice == VoteChoice.APPROVE and
                results.threshold_met
            )
            
            if amendment_passed:
                proposal.status = AmendmentStatus.APPROVED
                proposal.implementation_deadline = datetime.now() + timedelta(
                    days=self.config["implementation_timeout_days"]
                )
                
                # Queue for implementation
                self.pending_implementations[amendment_id] = {
                    "proposal": proposal,
                    "voting_results": asdict(results),
                    "queued_at": datetime.now()
                }
                
                # Begin implementation process
                success = await self._implement_amendment(amendment_id)
                if success:
                    proposal.status = AmendmentStatus.IMPLEMENTED
                    self.metrics["implementations_completed"] += 1
                else:
                    proposal.status = AmendmentStatus.FAILED_IMPLEMENTATION
                
                self.metrics["proposals_approved"] += 1
                
            else:
                proposal.status = AmendmentStatus.REJECTED
                self.metrics["proposals_rejected"] += 1
            
            await self._store_amendment_proposal(proposal)
            
            # Notify community of results
            await self._notify_community_of_results(proposal, results, amendment_passed)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process voting results for amendment {amendment_id}: {str(e)}")
            return False

    async def _implement_amendment(self, amendment_id: str) -> bool:
        """
        Implement an approved amendment.
        
        Args:
            amendment_id: ID of the amendment to implement
        
        Returns:
            bool: True if implementation successful
        """
        try:
            implementation_data = self.pending_implementations.get(amendment_id)
            if not implementation_data:
                return False
            
            proposal = implementation_data["proposal"]
            
            # Implementation strategy based on amendment type
            if proposal.amendment_type == AmendmentType.CREATE_NEW_RULE:
                success = await self._implement_new_rule(proposal)
            elif proposal.amendment_type == AmendmentType.MODIFY_EXISTING_RULE:
                success = await self._implement_rule_modification(proposal)
            elif proposal.amendment_type == AmendmentType.REPEAL_RULE:
                success = await self._implement_rule_repeal(proposal)
            elif proposal.amendment_type == AmendmentType.MERGE_RULES:
                success = await self._implement_rule_merge(proposal)
            elif proposal.amendment_type == AmendmentType.REORGANIZE_RULES:
                success = await self._implement_rule_reorganization(proposal)
            else:
                self.logger.warning(f"Unknown amendment type: {proposal.amendment_type}")
                return False
            
            if success:
                # Remove from pending implementations
                del self.pending_implementations[amendment_id]
                
                # Store implementation record
                await self._store_implementation_record(amendment_id, proposal)
                
                self.logger.info(f"Successfully implemented amendment {amendment_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to implement amendment {amendment_id}: {str(e)}")
            return False

    async def _implement_new_rule(self, proposal: AmendmentProposal) -> bool:
        """Implement a new rule creation amendment."""
        try:
            rule_data = proposal.proposed_changes.get("new_rule", {})
            
            new_rule = create_constitutional_rule(
                title=rule_data.get("title", ""),
                content=rule_data.get("content", ""),
                rule_type=RuleType(rule_data.get("rule_type", "behavioral_norm")),
                created_by=f"amendment_{proposal.amendment_id}",
                status=RuleStatus.ACTIVE,
                effective_date=datetime.now(),
                precedence_level=rule_data.get("precedence_level", 100),
                requires_supermajority=rule_data.get("requires_supermajority", False),
                enforcement_mechanism=rule_data.get("enforcement_mechanism", "community_based"),
                metadata={
                    "created_by_amendment": proposal.amendment_id,
                    "implementation_date": datetime.now().isoformat()
                }
            )
            
            return await self.constitutional_storage.store_rule(new_rule)
            
        except Exception as e:
            self.logger.error(f"Failed to implement new rule: {str(e)}")
            return False

    async def _implement_rule_modification(self, proposal: AmendmentProposal) -> bool:
        """Implement a rule modification amendment."""
        try:
            if not proposal.target_rule_ids:
                return False
            
            target_rule_id = proposal.target_rule_ids[0]
            existing_rule = await self.constitutional_storage.retrieve_rule(target_rule_id)
            
            if not existing_rule:
                return False
            
            # Apply modifications
            modifications = proposal.proposed_changes.get("modifications", {})
            
            # Create updated rule
            updated_rule = ConstitutionalRule(
                rule_id=existing_rule.rule_id,
                title=modifications.get("title", existing_rule.title),
                content=modifications.get("content", existing_rule.content),
                rule_type=RuleType(modifications.get("rule_type", existing_rule.rule_type.value)),
                status=existing_rule.status,
                version=existing_rule.version,
                created_at=existing_rule.created_at,
                created_by=existing_rule.created_by,
                effective_date=datetime.now(),
                expiration_date=existing_rule.expiration_date,
                precedence_level=modifications.get("precedence_level", existing_rule.precedence_level),
                requires_supermajority=modifications.get("requires_supermajority", existing_rule.requires_supermajority),
                enforcement_mechanism=modifications.get("enforcement_mechanism", existing_rule.enforcement_mechanism),
                related_rules=existing_rule.related_rules,
                violation_penalties=modifications.get("violation_penalties", existing_rule.violation_penalties),
                amendment_history=existing_rule.amendment_history,
                metadata={
                    **existing_rule.metadata,
                    "modified_by_amendment": proposal.amendment_id,
                    "modification_date": datetime.now().isoformat(),
                    "previous_version": existing_rule.version
                }
            )
            
            # Create new version
            new_rule_id = await self.constitutional_storage.create_rule_version(
                target_rule_id, updated_rule
            )
            
            return bool(new_rule_id)
            
        except Exception as e:
            self.logger.error(f"Failed to implement rule modification: {str(e)}")
            return False

    async def _implement_rule_repeal(self, proposal: AmendmentProposal) -> bool:
        """Implement a rule repeal amendment."""
        try:
            if not proposal.target_rule_ids:
                return False
            
            success_count = 0
            for rule_id in proposal.target_rule_ids:
                success = await self.constitutional_storage.deprecate_rule(
                    rule_id, f"Repealed by amendment {proposal.amendment_id}"
                )
                if success:
                    success_count += 1
            
            return success_count == len(proposal.target_rule_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to implement rule repeal: {str(e)}")
            return False

    # =====================================================
    # Helper Methods
    # =====================================================

    async def _perform_impact_analysis(self, amendment_type: AmendmentType,
                                     target_rule_ids: List[str],
                                     proposed_changes: Dict[str, Any]) -> Optional[AmendmentImpactAnalysis]:
        """Perform impact analysis for an amendment proposal."""
        try:
            # Get affected rules
            affected_rules = []
            for rule_id in target_rule_ids:
                rule = await self.constitutional_storage.retrieve_rule(rule_id)
                if rule:
                    affected_rules.append(rule_id)
            
            # Calculate community impact score (simplified)
            impact_score = min(1.0, len(affected_rules) * 0.2)
            
            # Determine implementation complexity
            complexity_factors = {
                AmendmentType.CREATE_NEW_RULE: "low",
                AmendmentType.MODIFY_EXISTING_RULE: "medium",
                AmendmentType.REPEAL_RULE: "low",
                AmendmentType.MERGE_RULES: "high",
                AmendmentType.REORGANIZE_RULES: "high"
            }
            complexity = complexity_factors.get(amendment_type, "medium")
            
            return AmendmentImpactAnalysis(
                amendment_id="",  # Will be set later
                affected_rules=affected_rules,
                conflicting_rules=[],  # TODO: Implement conflict detection
                dependent_rules=[],    # TODO: Implement dependency analysis
                community_impact_score=impact_score,
                implementation_complexity=complexity,
                resource_requirements={"time": "medium", "consensus": "high"},
                risk_assessment={"legal_risk": "low", "stability_risk": "medium"},
                backwards_compatibility=True,  # TODO: Implement compatibility check
                estimated_adoption_time=7 if complexity == "low" else 14,
                stakeholder_analysis={}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to perform impact analysis: {str(e)}")
            return None

    async def _determine_voting_mechanism(self, proposal: AmendmentProposal) -> VotingMechanism:
        """Determine appropriate voting mechanism for an amendment."""
        # Check if any affected rules require supermajority
        for rule_id in proposal.target_rule_ids:
            rule = await self.constitutional_storage.retrieve_rule(rule_id)
            if rule and rule.requires_supermajority:
                return VotingMechanism.SUPERMAJORITY
        
        # Check amendment impact
        impact_score = proposal.impact_assessment.get("community_impact_score", 0.0)
        if impact_score > 0.7:
            return VotingMechanism.SUPERMAJORITY
        
        return VotingMechanism.SIMPLE_MAJORITY

    async def _get_eligible_voters(self) -> Set[str]:
        """Get set of eligible voters for amendments."""
        # TODO: Implement proper voter eligibility logic
        # For now, return a placeholder set
        return {f"agent_{i}" for i in range(50)}

    async def _store_amendment_proposal(self, proposal: AmendmentProposal) -> bool:
        """Store amendment proposal in Store API."""
        try:
            proposal_data = asdict(proposal)
            proposal_data["proposed_at"] = proposal.proposed_at.isoformat()
            proposal_data["review_period_end"] = proposal.review_period_end.isoformat() if proposal.review_period_end else None
            proposal_data["implementation_deadline"] = proposal.implementation_deadline.isoformat() if proposal.implementation_deadline else None
            proposal_data["amendment_type"] = proposal.amendment_type.value
            proposal_data["status"] = proposal.status.value
            
            if self.store_integration.store:
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"amendment_proposal_{proposal.amendment_id}",
                    proposal_data
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store amendment proposal: {str(e)}")
            return False

    async def _notify_community_of_proposal(self, proposal: AmendmentProposal) -> None:
        """Notify community of new amendment proposal."""
        notification_data = {
            "type": "new_amendment_proposal",
            "amendment_id": proposal.amendment_id,
            "title": proposal.title,
            "proposed_by": proposal.proposed_by,
            "amendment_type": proposal.amendment_type.value,
            "review_period_end": proposal.review_period_end.isoformat() if proposal.review_period_end else None,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.store_integration._broadcast_community_event("amendment_proposal", notification_data)

    async def _notify_community_of_voting(self, proposal: AmendmentProposal) -> None:
        """Notify community that voting has opened."""
        notification_data = {
            "type": "amendment_voting_open",
            "amendment_id": proposal.amendment_id,
            "title": proposal.title,
            "voting_session_id": proposal.voting_session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.store_integration._broadcast_community_event("amendment_voting", notification_data)

    async def _notify_community_of_results(self, proposal: AmendmentProposal, results, passed: bool) -> None:
        """Notify community of voting results."""
        notification_data = {
            "type": "amendment_results",
            "amendment_id": proposal.amendment_id,
            "title": proposal.title,
            "passed": passed,
            "participation_rate": results.participation_rate,
            "winning_choice": results.winning_choice.value,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.store_integration._broadcast_community_event("amendment_results", notification_data)

    async def _schedule_review_period_end(self, amendment_id: str) -> None:
        """Schedule the end of review period for an amendment."""
        proposal = self.active_proposals.get(amendment_id)
        if not proposal or not proposal.review_period_end:
            return
        
        # Wait until review period ends
        wait_seconds = (proposal.review_period_end - datetime.now()).total_seconds()
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)
        
        # Check if ready for voting
        if (proposal.status == AmendmentStatus.UNDER_REVIEW and 
            len(proposal.community_feedback) >= self.config["min_community_feedback"]):
            await self.initiate_amendment_voting(amendment_id)

    async def _monitor_amendment_voting(self, amendment_id: str) -> None:
        """Monitor amendment voting and process results when complete."""
        proposal = self.active_proposals.get(amendment_id)
        if not proposal or not proposal.voting_session_id:
            return
        
        # Wait for voting to complete (check periodically)
        while proposal.status == AmendmentStatus.VOTING_OPEN:
            await asyncio.sleep(3600)  # Check every hour
            
            # Check if voting session is complete
            results = await self.voting_system.get_voting_results(proposal.voting_session_id)
            if results and proposal.voting_session_id not in self.voting_system.active_sessions:
                await self.process_voting_results(amendment_id)
                break

    # Additional helper methods for rule merge and reorganization would go here
    async def _implement_rule_merge(self, proposal: AmendmentProposal) -> bool:
        """Implement rule merge amendment (placeholder)."""
        # TODO: Implement rule merging logic
        return False

    async def _implement_rule_reorganization(self, proposal: AmendmentProposal) -> bool:
        """Implement rule reorganization amendment (placeholder)."""
        # TODO: Implement rule reorganization logic
        return False

    async def _store_implementation_record(self, amendment_id: str, proposal: AmendmentProposal) -> None:
        """Store implementation record."""
        try:
            record = {
                "amendment_id": amendment_id,
                "implemented_at": datetime.now().isoformat(),
                "implementation_type": proposal.amendment_type.value,
                "affected_rules": proposal.target_rule_ids,
                "implemented_by": "system"
            }
            
            if self.store_integration.store:
                await self.store_integration.store.aput(
                    StoreNamespace.GOVERNANCE.value,
                    f"amendment_implementation_{amendment_id}",
                    record
                )
                
        except Exception as e:
            self.logger.error(f"Failed to store implementation record: {str(e)}")

    async def _notify_community_of_feedback(self, amendment_id: str, feedback: Dict[str, Any]) -> None:
        """Notify community of new feedback on amendment."""
        notification_data = {
            "type": "amendment_feedback",
            "amendment_id": amendment_id,
            "feedback_type": feedback["feedback_type"],
            "agent_id": feedback["agent_id"],
            "supporting": feedback["supporting"],
            "timestamp": feedback["timestamp"]
        }
        
        await self.store_integration._broadcast_community_event("amendment_feedback", notification_data)


# Helper functions
def create_amendment_processor(constitutional_storage: ConstitutionalStorage,
                             voting_system: DemocraticVotingSystem,
                             store_integration: MemoryStoreIntegration) -> AmendmentProcessor:
    """Create an AmendmentProcessor instance."""
    return AmendmentProcessor(constitutional_storage, voting_system, store_integration)