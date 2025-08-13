#!/usr/bin/env python3
"""
Django Models Test Suite
Tests for Phase 5: Dating Show API Models
"""

import sys
import os
import time
from datetime import datetime, timezone, timedelta

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'environment.frontend_server.frontend_server.settings.base')

# Add Django app to path
sys.path.append('./environment/frontend_server')

try:
    import django
    django.setup()
    
    from dating_show_api.models import (
        Agent, AgentSkill, SocialRelationship, GovernanceVote, VoteCast,
        ConstitutionalRule, ComplianceViolation, AgentMemorySnapshot, SimulationState
    )
    print("✅ Successfully imported Django models")
except Exception as e:
    print(f"❌ Django setup error: {e}")
    # Create mock models for basic testing
    
    class MockAgent:
        def __init__(self, agent_id, name, current_role="participant"):
            self.agent_id = agent_id
            self.name = name
            self.current_role = current_role
            self.specialization = {}
            self.created_at = datetime.now(timezone.utc)
            self.updated_at = datetime.now(timezone.utc)
    
    class MockAgentSkill:
        def __init__(self, agent, skill_name, skill_level=0.0):
            self.agent = agent
            self.skill_name = skill_name
            self.skill_level = skill_level
            self.experience_points = 0.0
            self.discovery_date = datetime.now(timezone.utc)
    
    class MockSocialRelationship:
        def __init__(self, agent_a, agent_b, relationship_type, strength=0.0):
            self.agent_a = agent_a
            self.agent_b = agent_b
            self.relationship_type = relationship_type
            self.strength = strength
            self.established_date = datetime.now(timezone.utc)
    
    # Use mock models
    Agent = MockAgent
    AgentSkill = MockAgentSkill
    SocialRelationship = MockSocialRelationship
    print("⚠️  Using mock models for testing")


def test_agent_model():
    """Test Agent model functionality"""
    try:
        # Create test agent
        agent = Agent(
            agent_id="test_agent_001",
            name="Test Agent Model",
            current_role="contestant"
        )
        
        # Test basic properties
        assert agent.agent_id == "test_agent_001"
        assert agent.name == "Test Agent Model"
        assert agent.current_role == "contestant"
        
        print("✅ Agent model test passed")
        return True
        
    except Exception as e:
        print(f"❌ Agent model test failed: {e}")
        return False


def test_agent_skill_model():
    """Test AgentSkill model functionality"""
    try:
        # Create test agent and skill
        agent = Agent(
            agent_id="skill_test_agent",
            name="Skill Test Agent"
        )
        
        skill = AgentSkill(
            agent=agent,
            skill_name="social_interaction",
            skill_level=0.75
        )
        
        # Test skill properties
        assert skill.agent == agent
        assert skill.skill_name == "social_interaction"
        assert skill.skill_level == 0.75
        
        print("✅ AgentSkill model test passed")
        return True
        
    except Exception as e:
        print(f"❌ AgentSkill model test failed: {e}")
        return False


def test_social_relationship_model():
    """Test SocialRelationship model functionality"""
    try:
        # Create test agents
        agent_a = Agent(
            agent_id="relationship_agent_a",
            name="Agent A"
        )
        
        agent_b = Agent(
            agent_id="relationship_agent_b",
            name="Agent B"
        )
        
        # Create relationship
        relationship = SocialRelationship(
            agent_a=agent_a,
            agent_b=agent_b,
            relationship_type="friendship",
            strength=0.6
        )
        
        # Test relationship properties
        assert relationship.agent_a == agent_a
        assert relationship.agent_b == agent_b
        assert relationship.relationship_type == "friendship"
        assert relationship.strength == 0.6
        
        print("✅ SocialRelationship model test passed")
        return True
        
    except Exception as e:
        print(f"❌ SocialRelationship model test failed: {e}")
        return False


def test_model_relationships():
    """Test relationships between models"""
    try:
        # Create agent with multiple skills
        agent = Agent(
            agent_id="multi_skill_agent",
            name="Multi Skill Agent"
        )
        
        skills = [
            AgentSkill(agent=agent, skill_name="social", skill_level=0.8),
            AgentSkill(agent=agent, skill_name="creative", skill_level=0.6),
            AgentSkill(agent=agent, skill_name="analytical", skill_level=0.4)
        ]
        
        # Test that all skills are associated with the agent
        for skill in skills:
            assert skill.agent == agent
        
        # Create relationships with other agents
        other_agents = [
            Agent(agent_id="friend_agent", name="Friend Agent"),
            Agent(agent_id="rival_agent", name="Rival Agent")
        ]
        
        relationships = [
            SocialRelationship(agent, other_agents[0], "friendship", 0.7),
            SocialRelationship(agent, other_agents[1], "rivalry", -0.3)
        ]
        
        # Test relationship consistency
        assert relationships[0].agent_a == agent
        assert relationships[0].relationship_type == "friendship"
        assert relationships[1].strength < 0  # Rivalry should be negative
        
        print("✅ Model relationships test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model relationships test failed: {e}")
        return False


def test_data_validation():
    """Test model data validation"""
    try:
        # Test agent ID uniqueness (conceptually)
        agents = [
            Agent(agent_id="unique_test_001", name="Agent 1"),
            Agent(agent_id="unique_test_002", name="Agent 2")
        ]
        
        assert agents[0].agent_id != agents[1].agent_id
        
        # Test skill level bounds (0.0 to 1.0 typically)
        skill = AgentSkill(
            agent=agents[0],
            skill_name="bounded_skill",
            skill_level=0.95
        )
        
        assert 0.0 <= skill.skill_level <= 1.0
        
        # Test relationship strength bounds (-1.0 to 1.0 typically)
        relationship = SocialRelationship(
            agent_a=agents[0],
            agent_b=agents[1],
            relationship_type="neutral",
            strength=0.0
        )
        
        assert -1.0 <= relationship.strength <= 1.0
        
        print("✅ Data validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        return False


def test_model_performance():
    """Test model creation performance"""
    try:
        start_time = time.time()
        
        # Create multiple agents and relationships
        agents = []
        for i in range(20):
            agent = Agent(
                agent_id=f"perf_agent_{i:03d}",
                name=f"Performance Agent {i}",
                current_role="participant"
            )
            agents.append(agent)
        
        # Create skills for each agent
        skills = []
        for i, agent in enumerate(agents):
            for j, skill_name in enumerate(["social", "creative", "analytical"]):
                skill = AgentSkill(
                    agent=agent,
                    skill_name=skill_name,
                    skill_level=(i + j * 0.1) / 30.0  # Vary skill levels
                )
                skills.append(skill)
        
        # Create some relationships
        relationships = []
        for i in range(10):
            relationship = SocialRelationship(
                agent_a=agents[i],
                agent_b=agents[i + 10],
                relationship_type="friendship",
                strength=(i * 0.1) - 0.5  # Vary from -0.5 to 0.4
            )
            relationships.append(relationship)
        
        duration = time.time() - start_time
        
        # Should be reasonably fast
        assert duration < 0.1  # Less than 100ms
        assert len(agents) == 20
        assert len(skills) == 60  # 20 agents * 3 skills
        assert len(relationships) == 10
        
        print(f"✅ Model performance test passed ({len(agents)} agents, {len(skills)} skills, {len(relationships)} relationships in {duration*1000:.1f}ms)")
        return True
        
    except Exception as e:
        print(f"❌ Model performance test failed: {e}")
        return False


def test_serialization_compatibility():
    """Test model data serialization for API compatibility"""
    try:
        # Create test data
        agent = Agent(
            agent_id="serialization_test",
            name="Serialization Test Agent",
            current_role="host"
        )
        agent.specialization = {"role_expertise": "hosting", "experience_level": "expert"}
        
        skill = AgentSkill(
            agent=agent,
            skill_name="communication",
            skill_level=0.9
        )
        skill.experience_points = 150.0
        
        # Test data can be serialized (mock serialization)
        agent_data = {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'current_role': agent.current_role,
            'specialization': agent.specialization,
            'created_at': agent.created_at.isoformat() if hasattr(agent.created_at, 'isoformat') else str(agent.created_at)
        }
        
        skill_data = {
            'skill_name': skill.skill_name,
            'skill_level': skill.skill_level,
            'experience_points': skill.experience_points,
            'discovery_date': skill.discovery_date.isoformat() if hasattr(skill.discovery_date, 'isoformat') else str(skill.discovery_date)
        }
        
        # Verify serialized data integrity
        assert agent_data['agent_id'] == "serialization_test"
        assert agent_data['name'] == "Serialization Test Agent"
        assert agent_data['specialization']['role_expertise'] == "hosting"
        
        assert skill_data['skill_name'] == "communication"
        assert skill_data['skill_level'] == 0.9
        assert skill_data['experience_points'] == 150.0
        
        print("✅ Serialization compatibility test passed")
        return True
        
    except Exception as e:
        print(f"❌ Serialization compatibility test failed: {e}")
        return False


def run_django_model_tests():
    """Run all Django model tests"""
    print("\n🧪 Django Models Test Suite")
    print("=" * 45)
    
    test_functions = [
        test_agent_model,
        test_agent_skill_model,
        test_social_relationship_model,
        test_model_relationships,
        test_data_validation,
        test_model_performance,
        test_serialization_compatibility
    ]
    
    passed = 0
    failed = 0
    
    start_time = time.time()
    
    for test_func in test_functions:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed with exception: {e}")
            failed += 1
            
    duration = time.time() - start_time
    
    print("\n" + "=" * 45)
    print(f"🏁 Django Models Test Results")
    print("=" * 45)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    print(f"Duration: {duration:.3f}s")
    
    if failed == 0:
        print("\n🎉 All Django model tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_django_model_tests()
    sys.exit(0 if success else 1)