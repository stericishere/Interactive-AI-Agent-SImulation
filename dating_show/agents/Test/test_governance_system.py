#!/usr/bin/env python3
"""
Comprehensive Test Suite for Week 4 Governance System Components
Tests voting system, amendment system, constituency management, compliance monitoring, and behavioral adaptation.
"""

import asyncio
import sys
import os
import unittest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# Import governance components
from dating_show.governance.voting_system import (
    DemocraticVotingSystem, VotingMechanism, VoteChoice, VotingSession, create_democratic_voting_system
)
from dating_show.governance.amendment_system import (
    AmendmentSystem, AmendmentType, AmendmentStatus, AmendmentProposal, create_amendment_system
)
from dating_show.governance.constituency import (
    ConstituencyManager, VotingRights, RepresentationModel, VotingProfile, create_constituency_manager
)
from dating_show.governance.compliance_monitoring import (
    ComplianceMonitor, RuleCategory, ViolationType, Rule, create_compliance_monitor
)
from dating_show.governance.behavioral_adaptation import (
    BehavioralAdaptationSystem, AdaptationStrategy, AdaptationPhase, create_behavioral_adaptation_system
)
from dating_show.agents.memory_structures.store_integration import create_store_integration

# Configure logging
logging.basicConfig(level=logging.INFO)


class MockStore:
    """Mock Store API for testing."""
    
    def __init__(self):
        self.data = {}
    
    async def aput(self, namespace, key, value):
        """Store a value."""
        if namespace not in self.data:
            self.data[namespace] = {}
        self.data[namespace][key] = value
        return True
    
    async def aget(self, namespace, key):
        """Retrieve a value."""
        return self.data.get(namespace, {}).get(key)
    
    async def adelete(self, namespace, key):
        """Delete a value."""
        if namespace in self.data and key in self.data[namespace]:
            del self.data[namespace][key]
            return True
        return False


class TestVotingSystem(unittest.TestCase):
    """Test the Democratic Voting System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_store = MockStore()
        self.store_integration = create_store_integration()
        self.store_integration.store = self.mock_store
        self.voting_system = create_democratic_voting_system(self.store_integration)
    
    def test_voting_system_initialization(self):
        """Test voting system initializes correctly."""
        self.assertIsInstance(self.voting_system, DemocraticVotingSystem)
        self.assertEqual(self.voting_system.community_size, 50)
        self.assertEqual(len(self.voting_system.active_sessions), 0)
    
    async def test_create_voting_session(self):
        """Test creating a voting session."""
        eligible_voters = {f"agent_{i}" for i in range(10)}
        
        session_id = await self.voting_system.create_voting_session(
            proposal_id="test_proposal_001",
            voting_mechanism=VotingMechanism.SIMPLE_MAJORITY,
            eligible_voters=eligible_voters,
            voting_duration_hours=48
        )
        
        self.assertIsInstance(session_id, str)
        self.assertIn(session_id, self.voting_system.active_sessions)
        
        session = self.voting_system.active_sessions[session_id]
        self.assertEqual(session.proposal_id, "test_proposal_001")
        self.assertEqual(session.voting_mechanism, VotingMechanism.SIMPLE_MAJORITY)
        self.assertEqual(session.eligible_voters, eligible_voters)
        self.assertTrue(session.is_active)
    
    async def test_cast_vote(self):
        """Test casting votes."""
        eligible_voters = {f"agent_{i}" for i in range(5)}
        
        session_id = await self.voting_system.create_voting_session(
            proposal_id="test_proposal_002",
            voting_mechanism=VotingMechanism.SIMPLE_MAJORITY,
            eligible_voters=eligible_voters,
            voting_duration_hours=48
        )
        
        # Cast approve vote
        success = await self.voting_system.cast_vote(
            session_id=session_id,
            voter_agent_id="agent_0",
            vote_choice=VoteChoice.APPROVE,
            reasoning="This proposal is beneficial"
        )
        
        self.assertTrue(success)
        
        session = self.voting_system.active_sessions[session_id]
        self.assertIn("agent_0", session.votes_cast)
        self.assertEqual(session.votes_cast["agent_0"].vote_choice, VoteChoice.APPROVE)
    
    async def test_vote_validation(self):
        """Test vote validation rules."""
        eligible_voters = {"agent_0", "agent_1"}
        
        session_id = await self.voting_system.create_voting_session(
            proposal_id="test_proposal_003",
            voting_mechanism=VotingMechanism.SIMPLE_MAJORITY,
            eligible_voters=eligible_voters,
            voting_duration_hours=48
        )
        
        # Test voting by non-eligible agent
        success = await self.voting_system.cast_vote(
            session_id=session_id,
            voter_agent_id="agent_99",  # Not eligible
            vote_choice=VoteChoice.APPROVE
        )
        
        self.assertFalse(success)
    
    async def test_voting_results_calculation(self):
        """Test voting results calculation."""
        eligible_voters = {f"agent_{i}" for i in range(5)}
        
        session_id = await self.voting_system.create_voting_session(
            proposal_id="test_proposal_004",
            voting_mechanism=VotingMechanism.SIMPLE_MAJORITY,
            eligible_voters=eligible_voters,
            voting_duration_hours=48
        )
        
        # Cast votes
        await self.voting_system.cast_vote(session_id, "agent_0", VoteChoice.APPROVE)
        await self.voting_system.cast_vote(session_id, "agent_1", VoteChoice.APPROVE)
        await self.voting_system.cast_vote(session_id, "agent_2", VoteChoice.REJECT)
        await self.voting_system.cast_vote(session_id, "agent_3", VoteChoice.APPROVE)
        
        # Close session and get results
        await self.voting_system._close_voting_session(session_id)
        results = await self.voting_system.get_voting_results(session_id)
        
        self.assertIsNotNone(results)
        self.assertEqual(results.winning_choice, VoteChoice.APPROVE)
        self.assertEqual(results.votes_cast[VoteChoice.APPROVE], 3)
        self.assertEqual(results.votes_cast[VoteChoice.REJECT], 1)
        self.assertTrue(results.threshold_met)


class TestAmendmentSystem(unittest.TestCase):
    """Test the Amendment System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_store = MockStore()
        self.store_integration = create_store_integration()
        self.store_integration.store = self.mock_store
        self.voting_system = create_democratic_voting_system(self.store_integration)
        self.amendment_system = create_amendment_system(self.voting_system, self.store_integration)
    
    async def test_create_amendment_proposal(self):
        """Test creating amendment proposals."""
        proposal_id = await self.amendment_system.create_amendment_proposal(
            title="Test Amendment",
            summary="A test amendment proposal",
            full_text="This is a comprehensive test amendment to improve system behavior.",
            amendment_type=AmendmentType.BEHAVIORAL,
            proposed_by_agent_id="agent_proposer",
            co_sponsors={"agent_1", "agent_2", "agent_3"}
        )
        
        self.assertIsInstance(proposal_id, str)
        self.assertIn(proposal_id, self.amendment_system.active_proposals)
        
        proposal = self.amendment_system.active_proposals[proposal_id]
        self.assertEqual(proposal.title, "Test Amendment")
        self.assertEqual(proposal.amendment_type, AmendmentType.BEHAVIORAL)
        self.assertEqual(len(proposal.co_sponsors), 3)
        self.assertEqual(proposal.status, AmendmentStatus.DRAFT)
    
    async def test_discussion_posting(self):
        """Test discussion posting functionality."""
        proposal_id = await self.amendment_system.create_amendment_proposal(
            title="Discussion Test Amendment",
            summary="Testing discussion functionality",
            full_text="This amendment tests discussion features.",
            amendment_type=AmendmentType.PROCEDURAL,
            proposed_by_agent_id="agent_proposer"
        )
        
        # Begin discussion
        await self.amendment_system._begin_discussion_period(proposal_id)
        
        # Post discussion comment
        post_id = await self.amendment_system.post_discussion_comment(
            amendment_id=proposal_id,
            author_agent_id="agent_commenter",
            content="I support this amendment because it will improve our processes.",
            post_type="support"
        )
        
        self.assertIsInstance(post_id, str)
        self.assertIn(proposal_id, self.amendment_system.discussion_threads)
        self.assertTrue(len(self.amendment_system.discussion_threads[proposal_id]) > 0)
    
    async def test_discussion_summary(self):
        """Test discussion summary generation."""
        proposal_id = await self.amendment_system.create_amendment_proposal(
            title="Summary Test Amendment",
            summary="Testing summary generation",
            full_text="This amendment tests summary features.",
            amendment_type=AmendmentType.BEHAVIORAL,
            proposed_by_agent_id="agent_proposer"
        )
        
        await self.amendment_system._begin_discussion_period(proposal_id)
        
        # Add multiple discussion posts
        await self.amendment_system.post_discussion_comment(
            proposal_id, "agent_1", "I strongly support this proposal.", "support"
        )
        await self.amendment_system.post_discussion_comment(
            proposal_id, "agent_2", "I have concerns about implementation.", "concern"
        )
        await self.amendment_system.post_discussion_comment(
            proposal_id, "agent_3", "This is excellent work!", "support"
        )
        
        summary = await self.amendment_system.get_discussion_summary(proposal_id)
        
        self.assertEqual(summary["total_posts"], 3)
        self.assertEqual(summary["participants"], 3)
        self.assertIn("sentiment_distribution", summary)
        self.assertGreater(summary["sentiment_distribution"]["positive"], 0)


class TestConstituencyManager(unittest.TestCase):
    """Test the Constituency Management System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_store = MockStore()
        self.store_integration = create_store_integration()
        self.store_integration.store = self.mock_store
        self.constituency_manager = create_constituency_manager(self.store_integration)
    
    async def test_voter_registration(self):
        """Test voter registration."""
        success = await self.constituency_manager.register_voter(
            agent_id="test_agent_1",
            initial_weight=1.0,
            expertise_areas=["economics", "social_dynamics"]
        )
        
        self.assertTrue(success)
        self.assertIn("test_agent_1", self.constituency_manager.voting_profiles)
        
        profile = self.constituency_manager.voting_profiles["test_agent_1"]
        self.assertEqual(profile.voting_rights, VotingRights.FULL)
        self.assertEqual(profile.voting_weight, 1.0)
        self.assertEqual(len(profile.expertise_multipliers), 2)
    
    async def test_constituency_group_creation(self):
        """Test creating constituency groups."""
        group_id = await self.constituency_manager.create_constituency_group(
            group_name="Economics Experts",
            group_type="skill",
            description="Agents with economic expertise",
            interests=["economics", "resource_management"]
        )
        
        self.assertIsInstance(group_id, str)
        self.assertIn(group_id, self.constituency_manager.constituency_groups)
        
        group = self.constituency_manager.constituency_groups[group_id]
        self.assertEqual(group.group_name, "Economics Experts")
        self.assertEqual(group.group_type, "skill")
        self.assertTrue(group.is_active)
    
    async def test_vote_delegation(self):
        """Test vote delegation system."""
        # Register voters
        await self.constituency_manager.register_voter("agent_delegator")
        await self.constituency_manager.register_voter("agent_delegate")
        
        # Set up delegation
        success = await self.constituency_manager.delegate_vote(
            delegator_id="agent_delegator",
            delegate_to_id="agent_delegate"
        )
        
        self.assertTrue(success)
        
        delegator_profile = self.constituency_manager.voting_profiles["agent_delegator"]
        delegate_profile = self.constituency_manager.voting_profiles["agent_delegate"]
        
        self.assertEqual(delegator_profile.delegation_target, "agent_delegate")
        self.assertEqual(delegator_profile.voting_rights, VotingRights.PROXY)
        self.assertIn("agent_delegator", delegate_profile.delegated_votes)
    
    async def test_representation_calculation(self):
        """Test representation allocation calculation."""
        # Register multiple voters
        for i in range(5):
            await self.constituency_manager.register_voter(f"agent_{i}")
        
        allocation_id = await self.constituency_manager.calculate_representation_allocation(
            proposal_type="economic",
            representation_model=RepresentationModel.DIRECT_DEMOCRACY
        )
        
        self.assertIsInstance(allocation_id, str)
        self.assertIn(allocation_id, self.constituency_manager.active_allocations)
        
        allocation = self.constituency_manager.active_allocations[allocation_id]
        self.assertEqual(allocation.allocation_method, RepresentationModel.DIRECT_DEMOCRACY)
        self.assertEqual(len(allocation.individual_weights), 5)


class TestComplianceMonitor(unittest.TestCase):
    """Test the Compliance Monitoring System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_store = MockStore()
        self.store_integration = create_store_integration()
        self.store_integration.store = self.mock_store
        self.compliance_monitor = create_compliance_monitor(self.store_integration)
    
    async def test_rule_creation(self):
        """Test creating compliance rules."""
        rule_id = await self.compliance_monitor.add_rule(
            rule_name="No Harassment",
            rule_text="Agents must not engage in harassment or bullying behavior.",
            category=RuleCategory.BEHAVIORAL,
            priority=8,
            enforcement_level=0.9
        )
        
        self.assertIsInstance(rule_id, str)
        self.assertIn(rule_id, self.compliance_monitor.active_rules)
        
        rule = self.compliance_monitor.active_rules[rule_id]
        self.assertEqual(rule.rule_name, "No Harassment")
        self.assertEqual(rule.category, RuleCategory.BEHAVIORAL)
        self.assertTrue(rule.is_active)
        self.assertEqual(rule.priority, 8)
    
    async def test_compliance_record_initialization(self):
        """Test compliance record initialization."""
        await self.compliance_monitor._initialize_compliance_record("test_agent")
        
        self.assertIn("test_agent", self.compliance_monitor.compliance_records)
        
        record = self.compliance_monitor.compliance_records["test_agent"]
        self.assertEqual(record.overall_compliance_score, 1.0)  # Start with perfect score
        self.assertEqual(len(record.violation_count), 0)
    
    async def test_action_monitoring(self):
        """Test monitoring agent actions."""
        # Create a rule
        rule_id = await self.compliance_monitor.add_rule(
            rule_name="Respectful Communication",
            rule_text="All communication must be respectful and professional.",
            category=RuleCategory.SOCIAL
        )
        
        # Monitor a potentially violating action
        test_action = {
            "type": "communication",
            "content": "This is a respectful message",
            "target": "agent_2",
            "timestamp": datetime.now().isoformat()
        }
        
        violations = await self.compliance_monitor.monitor_action("test_agent", test_action)
        
        # Should be no violations for respectful communication
        self.assertEqual(len(violations), 0)
    
    async def test_compliance_score_retrieval(self):
        """Test getting compliance scores."""
        score = await self.compliance_monitor.get_compliance_score("new_agent")
        
        # New agents should have perfect compliance
        self.assertEqual(score, 1.0)


class TestBehavioralAdaptation(unittest.TestCase):
    """Test the Behavioral Adaptation System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_store = MockStore()
        self.store_integration = create_store_integration()
        self.store_integration.store = self.mock_store
        self.compliance_monitor = create_compliance_monitor(self.store_integration)
        self.adaptation_system = create_behavioral_adaptation_system(
            self.compliance_monitor, self.store_integration
        )
    
    async def test_agent_adaptation_initialization(self):
        """Test initializing agent adaptation."""
        await self.adaptation_system.initialize_agent_adaptation(
            agent_id="test_agent",
            initial_values={"cooperation": 0.8, "fairness": 0.7}
        )
        
        self.assertIn("test_agent", self.adaptation_system.agent_patterns)
        self.assertIn("test_agent", self.adaptation_system.adaptation_contexts)
        
        context = self.adaptation_system.adaptation_contexts["test_agent"]
        self.assertEqual(context.agent_id, "test_agent")
    
    async def test_behavior_adaptation_for_action(self):
        """Test adapting behavior for proposed actions."""
        await self.adaptation_system.initialize_agent_adaptation("test_agent")
        
        proposed_action = {
            "type": "social_interaction",
            "target": "agent_2",
            "message": "Hello there!",
            "intensity": 0.7
        }
        
        result = await self.adaptation_system.adapt_behavior_for_action(
            agent_id="test_agent",
            proposed_action=proposed_action
        )
        
        self.assertIn("adapted_action", result)
        self.assertIn("original_action", result)
        self.assertIn("adaptation_confidence", result)
        self.assertIsInstance(result["adaptation_confidence"], float)
    
    async def test_learning_from_outcome(self):
        """Test learning from action outcomes."""
        await self.adaptation_system.initialize_agent_adaptation("test_agent")
        
        action = {
            "type": "cooperative_behavior",
            "cooperation_level": 0.8
        }
        
        outcome = {
            "success": True,
            "social_approval": 0.6,
            "resource_gain": 0.1
        }
        
        await self.adaptation_system.learn_from_action_outcome(
            agent_id="test_agent",
            action=action,
            outcome=outcome,
            violations=[]
        )
        
        context = self.adaptation_system.adaptation_contexts["test_agent"]
        self.assertTrue(len(context.reward_history) > 0)
        self.assertGreater(context.reward_history[-1], 0)  # Should be positive reward
    
    async def test_compliance_motivation(self):
        """Test compliance motivation calculation."""
        await self.adaptation_system.initialize_agent_adaptation("test_agent")
        
        # Add a rule to compliance monitor
        rule_id = await self.compliance_monitor.add_rule(
            rule_name="Test Rule",
            rule_text="This is a test rule",
            category=RuleCategory.BEHAVIORAL
        )
        
        # Initialize rule internalization
        await self.adaptation_system.initialize_agent_adaptation("test_agent")
        
        motivation = await self.adaptation_system.get_compliance_motivation("test_agent", rule_id)
        
        self.assertIsInstance(motivation, float)
        self.assertGreaterEqual(motivation, 0.0)
        self.assertLessEqual(motivation, 1.0)


class TestGovernanceIntegration(unittest.TestCase):
    """Test integration between governance components."""
    
    def setUp(self):
        """Set up integrated test environment."""
        self.mock_store = MockStore()
        self.store_integration = create_store_integration()
        self.store_integration.store = self.mock_store
        
        # Create all governance components
        self.voting_system = create_democratic_voting_system(self.store_integration)
        self.amendment_system = create_amendment_system(self.voting_system, self.store_integration)
        self.constituency_manager = create_constituency_manager(self.store_integration)
        self.compliance_monitor = create_compliance_monitor(self.store_integration)
        self.adaptation_system = create_behavioral_adaptation_system(
            self.compliance_monitor, self.store_integration
        )
    
    async def test_end_to_end_amendment_process(self):
        """Test complete amendment process from proposal to voting."""
        # Register voters
        voters = {f"agent_{i}" for i in range(5)}
        for voter in voters:
            await self.constituency_manager.register_voter(voter)
        
        # Create amendment proposal
        proposal_id = await self.amendment_system.create_amendment_proposal(
            title="Community Respect Rule",
            summary="Add rule requiring respectful communication",
            full_text="All agents must communicate respectfully with others.",
            amendment_type=AmendmentType.BEHAVIORAL,
            proposed_by_agent_id="agent_0",
            co_sponsors={"agent_1", "agent_2"}
        )
        
        # Begin discussion
        await self.amendment_system._begin_discussion_period(proposal_id)
        
        # Add discussion comments
        await self.amendment_system.post_discussion_comment(
            proposal_id, "agent_3", "I support this amendment.", "support"
        )
        
        # Advance to voting (simulate meeting requirements)
        proposal = self.amendment_system.active_proposals[proposal_id]
        proposal.discussion_period_end = datetime.now() - timedelta(hours=1)  # Force past discussion period
        
        success = await self.amendment_system.advance_to_voting(proposal_id)
        self.assertTrue(success)
        
        # Verify voting session was created
        session_id = proposal.metadata.get("voting_session_id")
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.voting_system.active_sessions)
    
    async def test_compliance_and_adaptation_integration(self):
        """Test integration between compliance monitoring and behavioral adaptation."""
        # Initialize agent in both systems
        agent_id = "integration_test_agent"
        await self.adaptation_system.initialize_agent_adaptation(agent_id)
        
        # Create a rule
        rule_id = await self.compliance_monitor.add_rule(
            rule_name="No Spam",
            rule_text="Agents must not send repetitive or spam messages.",
            category=RuleCategory.BEHAVIORAL
        )
        
        # Test action that might violate rule
        action = {
            "type": "communication",
            "content": "spam spam spam spam",
            "repetition": 4
        }
        
        # Monitor action for violations
        violations = await self.compliance_monitor.monitor_action(agent_id, action)
        
        # Learn from outcome (with violations)
        outcome = {"success": False, "violations": violations}
        await self.adaptation_system.learn_from_action_outcome(
            agent_id, action, outcome, violations
        )
        
        # Check that learning occurred
        context = self.adaptation_system.adaptation_contexts[agent_id]
        self.assertTrue(len(context.reward_history) > 0)
        if violations:
            self.assertLess(context.reward_history[-1], 0)  # Should be negative reward


async def run_all_tests():
    """Run all governance system tests."""
    print("🧪 Starting Comprehensive Governance System Tests")
    print("=" * 60)
    
    test_classes = [
        TestVotingSystem,
        TestAmendmentSystem,
        TestConstituencyManager,
        TestComplianceMonitor,
        TestBehavioralAdaptation,
        TestGovernanceIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}")
        print("-" * 40)
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            test_instance = test_class()
            test_instance.setUp()
            
            try:
                test_method = getattr(test_instance, method_name)
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                
                print(f"  ✅ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ❌ {method_name}: {str(e)}")
                failed_tests.append(f"{test_class.__name__}.{method_name}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests:
        print("\n❌ FAILED TESTS:")
        for failure in failed_tests:
            print(f"  - {failure}")
    else:
        print("\n🎉 ALL TESTS PASSED!")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)