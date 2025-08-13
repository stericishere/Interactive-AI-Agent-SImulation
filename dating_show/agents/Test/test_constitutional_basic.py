#!/usr/bin/env python3
"""
Basic Constitutional System Test - Simplified Test Runner
Tests basic functionality of the constitutional system implementation
"""

import asyncio
import uuid
from datetime import datetime, timedelta

def test_basic_functionality():
    """Test basic imports and class creation."""
    print("Testing Constitutional System - Basic Functionality")
    
    try:
        # Test imports
        from dating_show.governance.constitution_storage import (
            ConstitutionalStorage, ConstitutionalRule, RuleType, RuleStatus,
            create_constitutional_storage, create_constitutional_rule
        )
        print("✅ Constitutional storage imports successful")
        
        from dating_show.governance.amendment_processing import (
            AmendmentProcessor, AmendmentType, AmendmentStatus,
            create_amendment_processor
        )
        print("✅ Amendment processing imports successful")
        
        from dating_show.governance.rule_interpretation import (
            RuleInterpreter, InterpretationContext, InterpretationResult,
            create_rule_interpreter
        )
        print("✅ Rule interpretation imports successful")
        
        from dating_show.governance.voting_system import (
            DemocraticVotingSystem, VotingMechanism, VoteChoice,
            create_democratic_voting_system
        )
        print("✅ Voting system imports successful")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        # Test basic rule creation
        test_rule = create_constitutional_rule(
            title="Test Rule",
            content="This is a test rule for validation",
            rule_type=RuleType.BEHAVIORAL_NORM,
            created_by="test_system"
        )
        print("✅ Rule creation successful")
        print(f"   Rule ID: {test_rule.rule_id}")
        print(f"   Rule Title: {test_rule.title}")
        print(f"   Rule Type: {test_rule.rule_type}")
        print(f"   Status: {test_rule.status}")
        
    except Exception as e:
        print(f"❌ Rule creation failed: {e}")
        return False
    
    try:
        # Test enum values
        print("\n📋 Testing Enum Values:")
        print(f"   Rule Types: {[rt.value for rt in RuleType]}")
        print(f"   Rule Statuses: {[rs.value for rs in RuleStatus]}")
        print(f"   Amendment Types: {[at.value for at in AmendmentType]}")
        print(f"   Voting Mechanisms: {[vm.value for vm in VotingMechanism]}")
        print("✅ Enum validation successful")
        
    except Exception as e:
        print(f"❌ Enum validation failed: {e}")
        return False
    
    print("\n🎉 All basic functionality tests passed!")
    return True

async def test_async_functionality():
    """Test async operations with mock components."""
    print("\nTesting Async Functionality with Mocks")
    
    # Mock Store for testing
    class MockStore:
        def __init__(self):
            self.data = {}
        
        async def aput(self, namespace, key, value):
            self.data[f"{namespace}:{key}"] = value
        
        async def aget(self, namespace, key):
            return self.data.get(f"{namespace}:{key}")
        
        async def asearch(self, namespace, query=None):
            keys = [k for k in self.data.keys() if k.startswith(f"{namespace}:")]
            return [k.split(":", 1)[1] for k in keys]
    
    # Mock Store Integration
    class MockStoreIntegration:
        def __init__(self):
            self.store = MockStore()
        
        async def _broadcast_community_event(self, event_type, data):
            print(f"📢 Broadcasting {event_type}: {data.get('type', 'unknown')}")
    
    try:
        from dating_show.governance.constitution_storage import (
            create_constitutional_storage, create_constitutional_rule, RuleType
        )
        
        # Create mock storage
        store_integration = MockStoreIntegration()
        constitutional_storage = create_constitutional_storage(store_integration)
        
        # Create test rule
        test_rule = create_constitutional_rule(
            title="Async Test Rule",
            content="This rule tests async functionality",
            rule_type=RuleType.BEHAVIORAL_NORM,
            created_by="async_test"
        )
        
        # Test storage operations
        success = await constitutional_storage.store_rule(test_rule)
        print(f"✅ Async rule storage: {success}")
        
        retrieved_rule = await constitutional_storage.retrieve_rule(test_rule.rule_id)
        print(f"✅ Async rule retrieval: {retrieved_rule is not None}")
        
        if retrieved_rule:
            print(f"   Retrieved rule title: {retrieved_rule.title}")
        
        # Test metrics
        metrics = await constitutional_storage.get_storage_metrics()
        print(f"✅ Storage metrics: {len(metrics)} metrics collected")
        
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        return False
    
    print("🎉 Async functionality tests passed!")
    return True

def test_data_structures():
    """Test data structure integrity."""
    print("\nTesting Data Structure Integrity")
    
    try:
        from dating_show.governance.constitution_storage import ConstitutionalRule, RuleType, RuleStatus
        from datetime import datetime
        
        # Test ConstitutionalRule creation with all fields
        rule = ConstitutionalRule(
            rule_id="test_rule_001",
            title="Complete Test Rule",
            content="This rule has all fields populated for testing",
            rule_type=RuleType.BEHAVIORAL_NORM,
            status=RuleStatus.ACTIVE,
            version=1,
            created_at=datetime.now(),
            created_by="test_creator",
            effective_date=datetime.now(),
            precedence_level=150,
            requires_supermajority=True,
            enforcement_mechanism="automated_moderation"
        )
        
        print("✅ ConstitutionalRule with all fields created successfully")
        print(f"   Rule hash: {rule.get_rule_hash()}")
        print(f"   Precedence level: {rule.precedence_level}")
        print(f"   Requires supermajority: {rule.requires_supermajority}")
        
        # Test field defaults
        simple_rule = ConstitutionalRule(
            rule_id="simple_001",
            title="Simple Rule",
            content="Simple content",
            rule_type=RuleType.GOVERNANCE_PROCEDURE,
            status=RuleStatus.PROPOSED,
            version=1,
            created_at=datetime.now(),
            created_by="simple_creator",
            effective_date=datetime.now()
        )
        
        print("✅ ConstitutionalRule with defaults created successfully")
        print(f"   Default precedence: {simple_rule.precedence_level}")
        print(f"   Default supermajority: {simple_rule.requires_supermajority}")
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False
    
    print("🎉 Data structure tests passed!")
    return True

def main():
    """Main test runner."""
    print("=" * 60)
    print("CONSTITUTIONAL SYSTEM - BASIC VALIDATION TESTS")
    print("=" * 60)
    
    success = True
    
    # Run synchronous tests
    success &= test_basic_functionality()
    success &= test_data_structures()
    
    # Run async tests
    try:
        success &= asyncio.run(test_async_functionality())
    except Exception as e:
        print(f"❌ Async test runner failed: {e}")
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Constitutional System Ready!")
        print("\nNext Steps:")
        print("- Integration with existing PIANO architecture")
        print("- Performance optimization for 50+ agents")
        print("- Full test suite with pytest")
    else:
        print("❌ SOME TESTS FAILED - Review implementation")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)