"""
Test Suite: Constitutional System Integration Tests
Description: Comprehensive test suite for the constitutional system including rule storage,
amendment processing, and rule interpretation following TDD principles.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import the modules under test
try:
    from dating_show.governance.constitution_storage import (
        ConstitutionalStorage, ConstitutionalRule, ConstitutionalAmendment,
        RuleType, RuleStatus, create_constitutional_storage, create_constitutional_rule
    )
    from dating_show.governance.amendment_processing import (
        AmendmentProcessor, AmendmentProposal, AmendmentType, AmendmentStatus,
        create_amendment_processor
    )
    from dating_show.governance.rule_interpretation import (
        RuleInterpreter, RuleInterpretation, InterpretationContext, InterpretationResult,
        create_rule_interpreter
    )
    from dating_show.governance.voting_system import (
        DemocraticVotingSystem, VotingMechanism, VoteChoice,
        create_democratic_voting_system
    )
    from dating_show.agents.memory_structures.store_integration import (
        MemoryStoreIntegration, create_store_integration
    )
except ImportError:
    # Mock classes for development
    class ConstitutionalStorage:
        pass
    class ConstitutionalRule:
        pass
    class ConstitutionalAmendment:
        pass
    class RuleType:
        pass
    class RuleStatus:
        pass
    class AmendmentProcessor:
        pass
    class AmendmentProposal:
        pass
    class AmendmentType:
        pass
    class AmendmentStatus:
        pass
    class RuleInterpreter:
        pass
    class RuleInterpretation:
        pass
    class InterpretationContext:
        pass
    class InterpretationResult:
        pass
    class DemocraticVotingSystem:
        pass
    class VotingMechanism:
        pass
    class VoteChoice:
        pass
    class MemoryStoreIntegration:
        pass
    
    def create_constitutional_storage(*args, **kwargs):
        return ConstitutionalStorage()
    def create_constitutional_rule(*args, **kwargs):
        return ConstitutionalRule()
    def create_amendment_processor(*args, **kwargs):
        return AmendmentProcessor()
    def create_rule_interpreter(*args, **kwargs):
        return RuleInterpreter()
    def create_democratic_voting_system(*args, **kwargs):
        return DemocraticVotingSystem()
    def create_store_integration(*args, **kwargs):
        return MemoryStoreIntegration()

# Mock classes for testing
class MockStore:
    """Mock Store API for testing."""
    
    def __init__(self):
        self.data = {}
    
    async def aput(self, namespace: str, key: str, value: Any):
        self.data[f"{namespace}:{key}"] = value
    
    async def aget(self, namespace: str, key: str):
        return self.data.get(f"{namespace}:{key}")
    
    async def asearch(self, namespace: str, query: str = None):
        keys = [k for k in self.data.keys() if k.startswith(f"{namespace}:")]
        return [k.split(":", 1)[1] for k in keys if query is None or query in k]

class MockPostgresPersistence:
    """Mock PostgreSQL persistence for testing."""
    
    def __init__(self):
        self.data = {}
        self.queries = []
    
    def get_connection(self):
        return self
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def execute(self, query: str, *args):
        self.queries.append((query, args))

# Test fixtures
@pytest.fixture
async def mock_store():
    """Create a mock Store API."""
    return MockStore()

@pytest.fixture
async def mock_postgres():
    """Create a mock PostgreSQL persistence layer."""
    return MockPostgresPersistence()

@pytest.fixture
async def store_integration(mock_store):
    """Create store integration with mock store."""
    integration = create_store_integration()
    integration.store = mock_store
    return integration

@pytest.fixture
async def constitutional_storage(store_integration, mock_postgres):
    """Create constitutional storage system."""
    return create_constitutional_storage(store_integration, mock_postgres)

@pytest.fixture
async def voting_system(store_integration, mock_postgres):
    """Create voting system."""
    return create_democratic_voting_system(store_integration, mock_postgres, 50)

@pytest.fixture
async def amendment_processor(constitutional_storage, voting_system, store_integration):
    """Create amendment processor."""
    return create_amendment_processor(constitutional_storage, voting_system, store_integration)

@pytest.fixture
async def rule_interpreter(constitutional_storage, store_integration):
    """Create rule interpreter."""
    return create_rule_interpreter(constitutional_storage, store_integration)

@pytest.fixture
def sample_rule():
    """Create a sample constitutional rule."""
    return create_constitutional_rule(
        title="Respectful Communication Rule",
        content="All agents shall communicate respectfully with other community members. "
                "Harassment, threats, or discriminatory language is prohibited.",
        rule_type=RuleType.BEHAVIORAL_NORM,
        created_by="test_system",
        status=RuleStatus.ACTIVE,
        precedence_level=100,
        requires_supermajority=False
    )

# =====================================================
# Constitutional Storage Tests
# =====================================================

class TestConstitutionalStorage:
    """Test cases for Constitutional Storage system."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_rule(self, constitutional_storage, sample_rule):
        """Test storing and retrieving a constitutional rule."""
        # Store the rule
        success = await constitutional_storage.store_rule(sample_rule)
        assert success, "Rule should be stored successfully"
        
        # Retrieve the rule
        retrieved_rule = await constitutional_storage.retrieve_rule(sample_rule.rule_id)
        assert retrieved_rule is not None, "Rule should be retrievable"
        assert retrieved_rule.rule_id == sample_rule.rule_id
        assert retrieved_rule.title == sample_rule.title
        assert retrieved_rule.content == sample_rule.content
        assert retrieved_rule.rule_type == sample_rule.rule_type
    
    @pytest.mark.asyncio
    async def test_get_rules_by_type(self, constitutional_storage):
        """Test retrieving rules by type."""
        # Create rules of different types
        behavioral_rule = create_constitutional_rule(
            title="Behavioral Rule",
            content="Test behavioral rule content",
            rule_type=RuleType.BEHAVIORAL_NORM,
            created_by="test"
        )
        
        governance_rule = create_constitutional_rule(
            title="Governance Rule",
            content="Test governance rule content", 
            rule_type=RuleType.GOVERNANCE_PROCEDURE,
            created_by="test"
        )
        
        # Store both rules
        await constitutional_storage.store_rule(behavioral_rule)
        await constitutional_storage.store_rule(governance_rule)
        
        # Retrieve rules by type
        behavioral_rules = await constitutional_storage.get_rules_by_type(RuleType.BEHAVIORAL_NORM)
        governance_rules = await constitutional_storage.get_rules_by_type(RuleType.GOVERNANCE_PROCEDURE)
        
        assert len(behavioral_rules) >= 1, "Should find at least one behavioral rule"
        assert len(governance_rules) >= 1, "Should find at least one governance rule"
        assert behavioral_rules[0].rule_type == RuleType.BEHAVIORAL_NORM
        assert governance_rules[0].rule_type == RuleType.GOVERNANCE_PROCEDURE
    
    @pytest.mark.asyncio
    async def test_rule_versioning(self, constitutional_storage, sample_rule):
        """Test rule versioning functionality."""
        # Store original rule
        await constitutional_storage.store_rule(sample_rule)
        
        # Create updated version
        updated_rule = create_constitutional_rule(
            title="Updated Respectful Communication Rule",
            content="All agents must communicate respectfully. Updated policy includes zero tolerance.",
            rule_type=RuleType.BEHAVIORAL_NORM,
            created_by="test_system_v2"
        )
        
        # Create new version
        new_rule_id = await constitutional_storage.create_rule_version(
            sample_rule.rule_id, updated_rule
        )
        
        assert new_rule_id, "New version should be created successfully"
        assert "_v2" in new_rule_id, "New rule ID should include version suffix"
        
        # Get rule history
        history = await constitutional_storage.get_rule_history(sample_rule.rule_id)
        assert len(history) >= 1, "Should have rule history"
    
    @pytest.mark.asyncio
    async def test_rule_deprecation(self, constitutional_storage, sample_rule):
        """Test rule deprecation."""
        # Store and then deprecate rule
        await constitutional_storage.store_rule(sample_rule)
        success = await constitutional_storage.deprecate_rule(
            sample_rule.rule_id, "Rule is outdated"
        )
        
        assert success, "Rule should be deprecated successfully"
        
        # Verify rule is marked as deprecated
        deprecated_rule = await constitutional_storage.retrieve_rule(sample_rule.rule_id)
        assert deprecated_rule.status == RuleStatus.DEPRECATED
        assert "deprecated_at" in deprecated_rule.metadata
        assert deprecated_rule.metadata["deprecation_reason"] == "Rule is outdated"
    
    @pytest.mark.asyncio
    async def test_get_all_active_rules(self, constitutional_storage):
        """Test getting all active rules."""
        # Create mix of active and deprecated rules
        active_rule = create_constitutional_rule(
            title="Active Rule", content="Active content", rule_type=RuleType.BEHAVIORAL_NORM,
            created_by="test", status=RuleStatus.ACTIVE
        )
        
        proposed_rule = create_constitutional_rule(
            title="Proposed Rule", content="Proposed content", rule_type=RuleType.BEHAVIORAL_NORM,
            created_by="test", status=RuleStatus.PROPOSED
        )
        
        await constitutional_storage.store_rule(active_rule)
        await constitutional_storage.store_rule(proposed_rule)
        
        # Get all active rules
        active_rules = await constitutional_storage.get_all_active_rules()
        
        # Should only include active rules
        active_rule_ids = list(active_rules.keys())
        assert active_rule.rule_id in active_rule_ids
        assert proposed_rule.rule_id not in active_rule_ids
    
    @pytest.mark.asyncio
    async def test_rule_integrity_validation(self, constitutional_storage, sample_rule):
        """Test rule integrity validation."""
        await constitutional_storage.store_rule(sample_rule)
        
        # Validate integrity
        is_valid = await constitutional_storage.validate_rule_integrity(sample_rule.rule_id)
        assert is_valid, "Rule integrity should be valid"
    
    @pytest.mark.asyncio
    async def test_storage_metrics(self, constitutional_storage, sample_rule):
        """Test storage metrics."""
        await constitutional_storage.store_rule(sample_rule)
        await constitutional_storage.retrieve_rule(sample_rule.rule_id)
        
        metrics = await constitutional_storage.get_storage_metrics()
        
        assert "total_rules_stored" in metrics
        assert "total_rules_retrieved" in metrics
        assert "cache_hit_rate" in metrics
        assert metrics["total_rules_stored"] >= 1
        assert metrics["total_rules_retrieved"] >= 1


# =====================================================
# Amendment Processing Tests  
# =====================================================

class TestAmendmentProcessor:
    """Test cases for Amendment Processing system."""
    
    @pytest.mark.asyncio
    async def test_submit_amendment_proposal(self, amendment_processor):
        """Test submitting a new amendment proposal."""
        amendment_id = await amendment_processor.submit_amendment_proposal(
            title="Test Amendment",
            description="This is a test amendment proposal",
            amendment_type=AmendmentType.CREATE_NEW_RULE,
            target_rule_ids=[],
            proposed_changes={
                "new_rule": {
                    "title": "Test New Rule",
                    "content": "This is a new test rule",
                    "rule_type": "behavioral_norm"
                }
            },
            justification="Testing the amendment system",
            proposed_by="test_agent"
        )
        
        assert amendment_id is not None, "Amendment should be submitted successfully"
        assert amendment_id.startswith("amendment_"), "Amendment ID should have correct prefix"
        
        # Verify proposal is stored
        proposal = amendment_processor.active_proposals.get(amendment_id)
        assert proposal is not None, "Proposal should be stored in active proposals"
        assert proposal.title == "Test Amendment"
        assert proposal.status == AmendmentStatus.PROPOSED
    
    @pytest.mark.asyncio
    async def test_add_community_feedback(self, amendment_processor):
        """Test adding community feedback to amendment."""
        # Submit a proposal first
        amendment_id = await amendment_processor.submit_amendment_proposal(
            title="Feedback Test Amendment",
            description="Testing feedback system",
            amendment_type=AmendmentType.CREATE_NEW_RULE,
            target_rule_ids=[],
            proposed_changes={"new_rule": {"title": "Test", "content": "Test"}},
            justification="Testing",
            proposed_by="test_agent"
        )
        
        # Add feedback
        success = await amendment_processor.add_community_feedback(
            amendment_id=amendment_id,
            agent_id="feedback_agent",
            feedback_type="comment",
            content="This is a great proposal!",
            supporting=True
        )
        
        assert success, "Feedback should be added successfully"
        
        # Verify feedback is stored
        proposal = amendment_processor.active_proposals[amendment_id]
        assert len(proposal.community_feedback) == 1
        assert proposal.community_feedback[0]["agent_id"] == "feedback_agent"
        assert proposal.community_feedback[0]["supporting"] is True
    
    @pytest.mark.asyncio
    async def test_amendment_voting_initiation(self, amendment_processor):
        """Test initiating voting on an amendment."""
        # Submit proposal and add sufficient feedback to move to review
        amendment_id = await amendment_processor.submit_amendment_proposal(
            title="Voting Test Amendment",
            description="Testing voting initiation",
            amendment_type=AmendmentType.CREATE_NEW_RULE,
            target_rule_ids=[],
            proposed_changes={"new_rule": {"title": "Test", "content": "Test"}},
            justification="Testing voting",
            proposed_by="test_agent"
        )
        
        # Add minimum required feedback
        for i in range(3):
            await amendment_processor.add_community_feedback(
                amendment_id, f"agent_{i}", "comment", f"Feedback {i}", True
            )
        
        # Set status to under review
        amendment_processor.active_proposals[amendment_id].status = AmendmentStatus.UNDER_REVIEW
        
        # Initiate voting
        voting_session_id = await amendment_processor.initiate_amendment_voting(amendment_id)
        
        assert voting_session_id is not None, "Voting should be initiated successfully"
        
        # Verify proposal status updated
        proposal = amendment_processor.active_proposals[amendment_id]
        assert proposal.status == AmendmentStatus.VOTING_OPEN
        assert proposal.voting_session_id == voting_session_id


# =====================================================
# Rule Interpretation Tests
# =====================================================

class TestRuleInterpreter:
    """Test cases for Rule Interpretation system."""
    
    @pytest.mark.asyncio
    async def test_interpret_rule_for_situation(self, rule_interpreter, constitutional_storage, sample_rule):
        """Test interpreting a rule for a specific situation."""
        # Store the rule first
        await constitutional_storage.store_rule(sample_rule)
        
        # Test interpretation
        interpretation = await rule_interpreter.interpret_rule_for_situation(
            rule_id=sample_rule.rule_id,
            context=InterpretationContext.AGENT_ACTION,
            situation_description="Agent sends a respectful message to another agent",
            contextual_factors={"agent_id": "test_agent", "message_tone": "respectful"},
            agent_id="test_agent"
        )
        
        assert interpretation is not None, "Rule should be interpreted successfully"
        assert interpretation.rule_id == sample_rule.rule_id
        assert interpretation.context == InterpretationContext.AGENT_ACTION
        assert interpretation.confidence_score > 0.0
        assert interpretation.interpretation_result in [
            InterpretationResult.COMPLIANT, 
            InterpretationResult.NOT_APPLICABLE,
            InterpretationResult.AMBIGUOUS
        ]
    
    @pytest.mark.asyncio
    async def test_interpret_multiple_rules(self, rule_interpreter, constitutional_storage):
        """Test interpreting multiple rules for conflict detection."""
        # Create two potentially conflicting rules
        rule1 = create_constitutional_rule(
            title="Allow Free Speech",
            content="Agents may express their opinions freely",
            rule_type=RuleType.FUNDAMENTAL_RIGHT,
            created_by="test"
        )
        
        rule2 = create_constitutional_rule(
            title="Prohibit Hate Speech", 
            content="Agents shall not use discriminatory language",
            rule_type=RuleType.BEHAVIORAL_NORM,
            created_by="test"
        )
        
        await constitutional_storage.store_rule(rule1)
        await constitutional_storage.store_rule(rule2)
        
        # Interpret both rules for a potentially conflicting situation
        interpretations = await rule_interpreter.interpret_multiple_rules(
            rule_ids=[rule1.rule_id, rule2.rule_id],
            context=InterpretationContext.AGENT_ACTION,
            situation_description="Agent makes a controversial political statement",
            agent_id="test_agent"
        )
        
        assert len(interpretations) == 2, "Should interpret both rules"
        assert all(i.rule_id in [rule1.rule_id, rule2.rule_id] for i in interpretations)
    
    @pytest.mark.asyncio
    async def test_evaluate_compliance(self, rule_interpreter, constitutional_storage, sample_rule):
        """Test evaluating action compliance with rules."""
        await constitutional_storage.store_rule(sample_rule)
        
        # Test compliant action
        compliance_result = await rule_interpreter.evaluate_compliance(
            agent_action={
                "type": "send_message",
                "target": "other_agent",
                "content": "Hello, how are you doing today?",
                "tone": "friendly"
            },
            context=InterpretationContext.AGENT_ACTION,
            agent_id="test_agent"
        )
        
        assert "compliant" in compliance_result
        assert "confidence" in compliance_result
        assert "details" in compliance_result
        assert isinstance(compliance_result["compliant"], bool)
    
    @pytest.mark.asyncio
    async def test_resolve_rule_conflict(self, rule_interpreter, constitutional_storage):
        """Test resolving conflicts between rules."""
        # Create conflicting rules with different precedence levels
        high_precedence_rule = create_constitutional_rule(
            title="High Precedence Rule",
            content="Rule with high precedence",
            rule_type=RuleType.FUNDAMENTAL_RIGHT,
            created_by="test",
            precedence_level=200
        )
        
        low_precedence_rule = create_constitutional_rule(
            title="Low Precedence Rule", 
            content="Rule with low precedence",
            rule_type=RuleType.BEHAVIORAL_NORM,
            created_by="test",
            precedence_level=50
        )
        
        await constitutional_storage.store_rule(high_precedence_rule)
        await constitutional_storage.store_rule(low_precedence_rule)
        
        # Resolve conflict
        resolution = await rule_interpreter.resolve_rule_conflict(
            conflicting_rules=[high_precedence_rule.rule_id, low_precedence_rule.rule_id],
            conflict_description="Test conflict between high and low precedence rules",
            context=InterpretationContext.DISPUTE_RESOLUTION
        )
        
        assert resolution is not None, "Conflict should be resolved"
        assert resolution.winning_rule == high_precedence_rule.rule_id
        assert resolution.resolution_strategy == "precedence"
        assert resolution.precedence_applied is True


# =====================================================
# Integration Tests
# =====================================================

class TestConstitutionalSystemIntegration:
    """Integration tests for the complete constitutional system."""
    
    @pytest.mark.asyncio
    async def test_full_amendment_lifecycle(self, amendment_processor, constitutional_storage, voting_system):
        """Test complete amendment lifecycle from proposal to implementation."""
        # Step 1: Submit amendment proposal
        amendment_id = await amendment_processor.submit_amendment_proposal(
            title="Integration Test Amendment",
            description="Testing full amendment lifecycle",
            amendment_type=AmendmentType.CREATE_NEW_RULE,
            target_rule_ids=[],
            proposed_changes={
                "new_rule": {
                    "title": "Integration Test Rule",
                    "content": "This rule was created through the amendment process",
                    "rule_type": "behavioral_norm",
                    "precedence_level": 100
                }
            },
            justification="Testing integration",
            proposed_by="integration_test_agent"
        )
        
        assert amendment_id is not None
        
        # Step 2: Add community feedback
        for i in range(3):
            success = await amendment_processor.add_community_feedback(
                amendment_id, f"test_agent_{i}", "comment", 
                f"Integration test feedback {i}", True
            )
            assert success
        
        # Step 3: Move to voting (simulate review completion)
        amendment_processor.active_proposals[amendment_id].status = AmendmentStatus.UNDER_REVIEW
        voting_session_id = await amendment_processor.initiate_amendment_voting(amendment_id)
        assert voting_session_id is not None
        
        # Step 4: Cast votes (simulate community voting)
        eligible_voters = {f"voter_agent_{i}" for i in range(10)}
        
        # Create voting session manually for testing
        from governance.voting_system import VotingSession, VotingMechanism, Vote
        voting_session = VotingSession(
            session_id=voting_session_id,
            proposal_id=amendment_id,
            voting_mechanism=VotingMechanism.SIMPLE_MAJORITY,
            required_threshold=0.5,
            eligible_voters=eligible_voters,
            voting_start=datetime.now(),
            voting_end=datetime.now() + timedelta(hours=48),
            votes_cast={},
            is_active=True
        )
        voting_system.active_sessions[voting_session_id] = voting_session
        
        # Cast votes in favor
        for i in range(6):  # 6 out of 10 votes in favor
            success = await voting_system.cast_vote(
                voting_session_id, f"voter_agent_{i}", VoteChoice.APPROVE
            )
            assert success
        
        # Step 5: Process voting results
        success = await amendment_processor.process_voting_results(amendment_id)
        assert success
        
        # Verify amendment was approved and implemented
        proposal = amendment_processor.active_proposals[amendment_id]
        assert proposal.status in [AmendmentStatus.IMPLEMENTED, AmendmentStatus.APPROVED]
    
    @pytest.mark.asyncio
    async def test_rule_storage_and_interpretation_integration(self, constitutional_storage, rule_interpreter):
        """Test integration between rule storage and interpretation."""
        # Create and store a rule
        test_rule = create_constitutional_rule(
            title="Integration Communication Rule",
            content="Agents must use polite language when communicating. "
                   "Rude or aggressive language is prohibited.",
            rule_type=RuleType.BEHAVIORAL_NORM,
            created_by="integration_test",
            status=RuleStatus.ACTIVE
        )
        
        success = await constitutional_storage.store_rule(test_rule)
        assert success
        
        # Interpret the rule for different situations
        polite_interpretation = await rule_interpreter.interpret_rule_for_situation(
            test_rule.rule_id,
            InterpretationContext.AGENT_ACTION,
            "Agent says 'Please help me with this task'",
            agent_id="polite_agent"
        )
        
        rude_interpretation = await rule_interpreter.interpret_rule_for_situation(
            test_rule.rule_id, 
            InterpretationContext.AGENT_ACTION,
            "Agent says 'You're stupid, do this now!'",
            agent_id="rude_agent"
        )
        
        assert polite_interpretation is not None
        assert rude_interpretation is not None
        
        # Polite communication should be compliant or not applicable
        assert polite_interpretation.interpretation_result in [
            InterpretationResult.COMPLIANT, InterpretationResult.NOT_APPLICABLE
        ]
        
        # The system should detect some level of concern with rude language
        # (exact result depends on implementation details)
        assert rude_interpretation.interpretation_result in [
            InterpretationResult.VIOLATION, InterpretationResult.AMBIGUOUS, 
            InterpretationResult.NOT_APPLICABLE
        ]
    
    @pytest.mark.asyncio
    async def test_system_performance_under_load(self, constitutional_storage, rule_interpreter):
        """Test system performance with multiple concurrent operations."""
        import time
        
        # Create multiple rules
        rules = []
        for i in range(10):
            rule = create_constitutional_rule(
                title=f"Performance Test Rule {i}",
                content=f"This is performance test rule number {i}",
                rule_type=RuleType.BEHAVIORAL_NORM,
                created_by="performance_test"
            )
            rules.append(rule)
        
        # Store all rules concurrently
        start_time = time.time()
        tasks = [constitutional_storage.store_rule(rule) for rule in rules]
        results = await asyncio.gather(*tasks)
        store_time = time.time() - start_time
        
        assert all(results), "All rules should be stored successfully"
        assert store_time < 5.0, "Storage should complete within 5 seconds"
        
        # Retrieve all rules concurrently  
        start_time = time.time()
        retrieve_tasks = [constitutional_storage.retrieve_rule(rule.rule_id) for rule in rules]
        retrieved_rules = await asyncio.gather(*retrieve_tasks)
        retrieve_time = time.time() - start_time
        
        assert all(r is not None for r in retrieved_rules), "All rules should be retrieved"
        assert retrieve_time < 3.0, "Retrieval should complete within 3 seconds"
        
        # Interpret rules concurrently
        start_time = time.time()
        interpret_tasks = [
            rule_interpreter.interpret_rule_for_situation(
                rule.rule_id,
                InterpretationContext.AGENT_ACTION,
                f"Test situation {i}",
                agent_id="performance_agent"
            ) for i, rule in enumerate(rules)
        ]
        interpretations = await asyncio.gather(*interpret_tasks)
        interpret_time = time.time() - start_time
        
        assert all(i is not None for i in interpretations), "All rules should be interpreted"
        assert interpret_time < 10.0, "Interpretation should complete within 10 seconds"


# =====================================================
# Test Execution
# =====================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])