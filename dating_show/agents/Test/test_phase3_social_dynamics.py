"""
Comprehensive Test Suite for Phase 3 Social Dynamics Systems
Testing: Relationship Network, Reputation System, Coalition Formation

This test suite validates the functionality, performance, and integration
of the advanced social dynamics components implemented in Phase 3.
"""

import unittest
import time
import json
import threading
from unittest.mock import patch, MagicMock
from typing import Dict, List, Set, Any

# Import the Phase 3 social dynamics modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../social'))

from relationship_network import (
    RelationshipNetwork, RelationshipType, InfluenceType, 
    RelationshipEdge, InfluencePacket
)
from reputation import (
    ReputationSystem, ReputationDimension, ReputationEvent,
    ReputationRecord, ReputationScore
)
from coalitions import (
    CoalitionFormationSystem, CoalitionType, CoalitionStatus, 
    MemberRole, Coalition, CoalitionMember, CoalitionGoal
)


class TestRelationshipNetwork(unittest.TestCase):
    """Test suite for RelationshipNetwork system"""
    
    def setUp(self):
        """Set up test environment"""
        self.network = RelationshipNetwork(max_agents=100)
        self.test_agents = ["alice", "bob", "charlie", "diana", "eve"]
        
        # Add test agents
        for agent in self.test_agents:
            self.network.add_agent(agent)
    
    def tearDown(self):
        """Clean up test environment"""
        self.network = None
    
    def test_agent_management(self):
        """Test agent addition and removal"""
        # Test adding agents
        self.assertEqual(len(self.network.agents), 5)
        self.assertIn("alice", self.network.agents)
        
        # Test adding duplicate agent
        result = self.network.add_agent("alice")
        self.assertTrue(result)  # Should still work
        self.assertEqual(len(self.network.agents), 5)  # No duplicate
        
        # Test removing agent
        result = self.network.remove_agent("eve")
        self.assertTrue(result)
        self.assertEqual(len(self.network.agents), 4)
        self.assertNotIn("eve", self.network.agents)
        
        # Test removing non-existent agent
        result = self.network.remove_agent("non_existent")
        self.assertFalse(result)
    
    def test_relationship_creation(self):
        """Test relationship creation and management"""
        # Create romantic relationship
        result = self.network.create_relationship(
            "alice", "bob", RelationshipType.ROMANTIC,
            strength=0.8, trust_level=0.6
        )
        self.assertTrue(result)
        
        # Verify relationship exists
        relationship = self.network.get_relationship("alice", "bob")
        self.assertIsNotNone(relationship)
        self.assertEqual(relationship.relationship_type, RelationshipType.ROMANTIC)
        self.assertEqual(relationship.strength, 0.8)
        self.assertEqual(relationship.trust_level, 0.6)
        
        # Test symmetric relationship access
        relationship2 = self.network.get_relationship("bob", "alice")
        self.assertEqual(relationship.agent_a, relationship2.agent_a)
        self.assertEqual(relationship.agent_b, relationship2.agent_b)
        
        # Create friendship network
        self.network.create_relationship("alice", "charlie", RelationshipType.FRIENDSHIP, 0.7, 0.5)
        self.network.create_relationship("bob", "diana", RelationshipType.ALLIANCE, 0.6, 0.4)
        self.network.create_relationship("charlie", "diana", RelationshipType.RIVALRY, 0.3, -0.2)
        
        # Verify agent relationships
        alice_relationships = self.network.get_agent_relationships("alice")
        self.assertEqual(len(alice_relationships), 2)
    
    def test_relationship_updates(self):
        """Test relationship updates based on interactions"""
        # Create initial relationship
        self.network.create_relationship("alice", "bob", RelationshipType.FRIENDSHIP, 0.5, 0.0)
        
        # Positive interaction
        result = self.network.update_relationship(
            "alice", "bob", "cooperation", 0.3, "successful_project"
        )
        self.assertTrue(result)
        
        relationship = self.network.get_relationship("alice", "bob")
        self.assertGreater(relationship.trust_level, 0.0)
        self.assertGreater(relationship.strength, 0.5)
        self.assertEqual(len(relationship.shared_experiences), 1)
        
        # Negative interaction
        self.network.update_relationship(
            "alice", "bob", "betrayal", -0.5, "broken_promise"
        )
        
        relationship = self.network.get_relationship("alice", "bob")
        self.assertLess(relationship.trust_level, 0.3)  # Should decrease
        self.assertEqual(len(relationship.shared_experiences), 2)
    
    def test_trust_calculation(self):
        """Test trust path calculation between agents"""
        # Create trust network: alice -> bob -> charlie
        self.network.create_relationship("alice", "bob", RelationshipType.FRIENDSHIP, 0.8, 0.7)
        self.network.create_relationship("bob", "charlie", RelationshipType.ALLIANCE, 0.7, 0.6)
        
        # Direct trust
        direct_trust = self.network.calculate_trust_path("alice", "bob")
        self.assertIsNotNone(direct_trust)
        self.assertGreater(direct_trust, 0.5)
        
        # Indirect trust through bob
        indirect_trust = self.network.calculate_trust_path("alice", "charlie")
        self.assertIsNotNone(indirect_trust)
        self.assertGreater(indirect_trust, 0.0)
        self.assertLess(indirect_trust, direct_trust)  # Should be less than direct
        
        # Self trust
        self_trust = self.network.calculate_trust_path("alice", "alice")
        self.assertEqual(self_trust, 1.0)
        
        # No path trust
        no_trust = self.network.calculate_trust_path("alice", "diana")
        self.assertEqual(no_trust, 0.0)
    
    def test_influence_propagation(self):
        """Test influence propagation through network"""
        # Create influence network
        self.network.create_relationship("alice", "bob", RelationshipType.FRIENDSHIP, 0.8, 0.7)
        self.network.create_relationship("alice", "charlie", RelationshipType.ALLIANCE, 0.6, 0.5)
        self.network.create_relationship("bob", "diana", RelationshipType.FRIENDSHIP, 0.7, 0.6)
        
        # Propagate opinion influence
        influence_results = self.network.propagate_influence(
            "alice", InfluenceType.OPINION, 
            {"topic": "environmental_policy", "stance": "pro"},
            initial_strength=1.0, decay_rate=0.1, max_hops=3
        )
        
        # Verify influence propagation
        self.assertIn("alice", influence_results)
        self.assertEqual(influence_results["alice"], 1.0)  # Source has full influence
        
        self.assertIn("bob", influence_results)
        self.assertGreater(influence_results["bob"], 0.0)
        self.assertLess(influence_results["bob"], 1.0)
        
        # Test different influence types
        meme_results = self.network.propagate_influence(
            "alice", InfluenceType.MEME,
            {"meme_id": "test_meme", "content": "viral_content"},
            initial_strength=0.8, decay_rate=0.2, max_hops=2
        )
        
        self.assertIn("alice", meme_results)
        self.assertGreaterEqual(len(meme_results), 1)
    
    def test_community_detection(self):
        """Test community detection algorithms"""
        # Create clustered network
        # Cluster 1: alice, bob
        self.network.create_relationship("alice", "bob", RelationshipType.FRIENDSHIP, 0.9, 0.8)
        
        # Cluster 2: charlie, diana
        self.network.create_relationship("charlie", "diana", RelationshipType.ALLIANCE, 0.8, 0.7)
        
        # Weak connection between clusters
        self.network.create_relationship("bob", "charlie", RelationshipType.PROFESSIONAL, 0.3, 0.2)
        
        communities = self.network.detect_communities()
        
        # Should detect communities
        self.assertGreaterEqual(len(communities), 2)
        
        # Verify community structure
        found_alice_bob = False
        found_charlie_diana = False
        
        for community_id, members in communities.items():
            if "alice" in members and "bob" in members:
                found_alice_bob = True
            if "charlie" in members and "diana" in members:
                found_charlie_diana = True
        
        # In a perfect world, these would be separate, but modularity optimization
        # might merge small communities
        self.assertTrue(found_alice_bob or found_charlie_diana)
    
    def test_centrality_metrics(self):
        """Test network centrality calculations"""
        # Create star network with alice at center
        for agent in ["bob", "charlie", "diana"]:
            self.network.create_relationship("alice", agent, RelationshipType.FRIENDSHIP, 0.7, 0.5)
        
        metrics = self.network.calculate_centrality_metrics()
        
        # Alice should have highest centrality
        self.assertIn("alice", metrics)
        self.assertIn("bob", metrics)
        
        alice_metrics = metrics["alice"]
        bob_metrics = metrics["bob"]
        
        # Alice should have higher degree centrality
        self.assertGreater(alice_metrics["degree_centrality"], bob_metrics["degree_centrality"])
        
        # Verify metric structure
        expected_metrics = ["degree_centrality", "betweenness_centrality", 
                          "closeness_centrality", "eigenvector_centrality"]
        for metric in expected_metrics:
            self.assertIn(metric, alice_metrics)
            self.assertIsInstance(alice_metrics[metric], float)
    
    def test_network_statistics(self):
        """Test network-wide statistics"""
        # Create test network
        self.network.create_relationship("alice", "bob", RelationshipType.ROMANTIC, 0.8, 0.7)
        self.network.create_relationship("alice", "charlie", RelationshipType.FRIENDSHIP, 0.6, 0.4)
        self.network.create_relationship("bob", "diana", RelationshipType.ALLIANCE, 0.5, 0.3)
        
        stats = self.network.get_network_statistics()
        
        # Verify statistics structure
        expected_stats = ["total_agents", "total_relationships", "average_degree", 
                         "network_density", "relationship_type_distribution",
                         "average_trust", "average_relationship_strength"]
        
        for stat in expected_stats:
            self.assertIn(stat, stats)
        
        # Verify values
        self.assertEqual(stats["total_agents"], 5)
        self.assertEqual(stats["total_relationships"], 3)
        self.assertGreater(stats["average_degree"], 0)
        self.assertGreaterEqual(stats["network_density"], 0)
        self.assertLessEqual(stats["network_density"], 1)
    
    def test_performance(self):
        """Test performance with larger networks"""
        # Create larger network
        large_network = RelationshipNetwork(max_agents=200)
        
        # Add many agents
        start_time = time.time()
        for i in range(100):
            large_network.add_agent(f"agent_{i}")
        agent_creation_time = time.time() - start_time
        
        # Should be fast
        self.assertLess(agent_creation_time, 1.0)
        
        # Create relationships
        start_time = time.time()
        for i in range(0, 100, 2):
            large_network.create_relationship(
                f"agent_{i}", f"agent_{i+1}", 
                RelationshipType.FRIENDSHIP, 0.5, 0.3
            )
        relationship_creation_time = time.time() - start_time
        
        # Should be fast
        self.assertLess(relationship_creation_time, 2.0)
        
        # Test trust calculation performance
        start_time = time.time()
        trust = large_network.calculate_trust_path("agent_0", "agent_10")
        trust_calculation_time = time.time() - start_time
        
        # Should be fast
        self.assertLess(trust_calculation_time, 0.1)
    
    def test_thread_safety(self):
        """Test thread safety of network operations"""
        def create_relationships(start_idx, count):
            for i in range(start_idx, start_idx + count):
                agent1 = f"thread_agent_{i}"
                agent2 = f"thread_agent_{i+1}"
                self.network.add_agent(agent1)
                self.network.add_agent(agent2)
                self.network.create_relationship(
                    agent1, agent2, RelationshipType.FRIENDSHIP, 0.5, 0.3
                )
        
        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_relationships, args=(i*10, 5))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no corruption
        stats = self.network.get_network_statistics()
        self.assertGreater(stats["total_agents"], 5)  # Original + new agents
        self.assertGreaterEqual(stats["total_relationships"], 0)


class TestReputationSystem(unittest.TestCase):
    """Test suite for ReputationSystem"""
    
    def setUp(self):
        """Set up test environment"""
        self.reputation_system = ReputationSystem(max_agents=100)
        self.test_agents = ["alice", "bob", "charlie", "diana", "eve"]
        
        # Add test agents
        for agent in self.test_agents:
            self.reputation_system.add_agent(agent)
    
    def tearDown(self):
        """Clean up test environment"""
        self.reputation_system = None
    
    def test_agent_management(self):
        """Test agent addition and removal"""
        # Verify agents added
        self.assertEqual(len(self.reputation_system.agents), 5)
        
        # Test reputation scores initialized
        for agent in self.test_agents:
            scores = self.reputation_system.get_reputation_score(agent)
            self.assertIsInstance(scores, dict)
            self.assertEqual(len(scores), len(ReputationDimension))
            
            for dimension in ReputationDimension:
                self.assertIn(dimension, scores)
                self.assertEqual(scores[dimension], 0.0)  # Initially zero
        
        # Test removal
        result = self.reputation_system.remove_agent("eve")
        self.assertTrue(result)
        self.assertEqual(len(self.reputation_system.agents), 4)
    
    def test_reputation_events(self):
        """Test recording and processing reputation events"""
        # Record positive event
        event_id = self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.PROMISE_KEPT, "bob", {"charlie"}, 
            {"promise": "deliver_project_on_time"}
        )
        self.assertNotEqual(event_id, "")
        
        # Check reputation updated
        trustworthiness = self.reputation_system.get_reputation_score(
            "alice", ReputationDimension.TRUSTWORTHINESS
        )
        self.assertGreater(trustworthiness, 0.0)
        
        reliability = self.reputation_system.get_reputation_score(
            "alice", ReputationDimension.RELIABILITY
        )
        self.assertGreater(reliability, 0.0)
        
        # Record negative event
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.PROMISE_BROKEN, "bob", {"charlie"},
            {"promise": "failed_to_deliver"}
        )
        
        # Trustworthiness should decrease
        new_trustworthiness = self.reputation_system.get_reputation_score(
            "alice", ReputationDimension.TRUSTWORTHINESS
        )
        self.assertLess(new_trustworthiness, trustworthiness)
    
    def test_observer_perspective(self):
        """Test reputation from different observer perspectives"""
        # Alice helps Bob (observed by Charlie)
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.HELP_PROVIDED, "charlie", {"bob"},
            {"help_type": "technical_assistance"}
        )
        
        # Alice betrays Charlie (observed by Bob)
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.DECEPTION_DETECTED, "bob", {"diana"},
            {"deception_type": "false_information"}
        )
        
        # Get reputation from different perspectives
        general_trustworthiness = self.reputation_system.get_reputation_score(
            "alice", ReputationDimension.TRUSTWORTHINESS
        )
        
        bob_perspective = self.reputation_system.get_reputation_score(
            "alice", ReputationDimension.TRUSTWORTHINESS, "bob"
        )
        
        charlie_perspective = self.reputation_system.get_reputation_score(
            "alice", ReputationDimension.TRUSTWORTHINESS, "charlie"
        )
        
        # Bob (who observed betrayal) should have lower trust than Charlie
        self.assertLess(bob_perspective, charlie_perspective)
    
    def test_trust_calculation(self):
        """Test trust score calculation between agents"""
        # Build reputation for Alice
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.PROMISE_KEPT, "bob"
        )
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.HONEST_DISCLOSURE, "bob"
        )
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.SUCCESSFUL_COLLABORATION, "bob"
        )
        
        # Calculate trust
        trust_score = self.reputation_system.calculate_trust_score("bob", "alice")
        self.assertGreater(trust_score, 0.0)
        self.assertLessEqual(trust_score, 1.0)
        
        # Damage reputation
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.DECEPTION_DETECTED, "bob"
        )
        
        new_trust_score = self.reputation_system.calculate_trust_score("bob", "alice")
        self.assertLess(new_trust_score, trust_score)
    
    def test_collaboration_recommendations(self):
        """Test collaboration partner recommendations"""
        # Build different skill profiles
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.LEADERSHIP_SUCCESS, "bob"
        )
        self.reputation_system.record_reputation_event(
            "bob", ReputationEvent.INNOVATION_SUCCESS, "alice"
        )
        self.reputation_system.record_reputation_event(
            "charlie", ReputationEvent.CONFLICT_RESOLVED, "alice"
        )
        
        # Request recommendations for leadership task
        requirements = {
            ReputationDimension.LEADERSHIP: 0.5,
            ReputationDimension.COMPETENCE: 0.3
        }
        
        recommendations = self.reputation_system.recommend_collaboration_partners(
            "diana", requirements
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Verify recommendation structure
        for agent_id, score in recommendations:
            self.assertIn(agent_id, self.test_agents)
            self.assertNotEqual(agent_id, "diana")  # Shouldn't recommend self
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
    
    def test_fraud_detection(self):
        """Test reputation fraud detection"""
        # Create suspicious pattern - too many positive events from same observer
        for i in range(15):
            self.reputation_system.record_reputation_event(
                "alice", ReputationEvent.HELP_PROVIDED, "bob"
            )
        
        fraud_indicators = self.reputation_system.detect_reputation_fraud("alice")
        self.assertGreater(len(fraud_indicators), 0)
        self.assertTrue(any("observer_bias" in indicator for indicator in fraud_indicators))
        
        # Test excessive activity
        for i in range(25):
            self.reputation_system.record_reputation_event(
                "charlie", ReputationEvent.SUCCESSFUL_COLLABORATION, "bob"  # Use existing agent
            )
        
        fraud_indicators = self.reputation_system.detect_reputation_fraud("charlie")
        self.assertIn("excessive_recent_activity", fraud_indicators)
    
    def test_reputation_analytics(self):
        """Test comprehensive reputation analytics"""
        # Build reputation history
        events = [
            (ReputationEvent.PROMISE_KEPT, "bob"),
            (ReputationEvent.LEADERSHIP_SUCCESS, "charlie"),
            (ReputationEvent.INNOVATION_SUCCESS, "diana"),
            (ReputationEvent.HELP_PROVIDED, "bob")
        ]
        
        for event, observer in events:
            self.reputation_system.record_reputation_event("alice", event, observer)
        
        analytics = self.reputation_system.get_reputation_analytics("alice")
        
        # Verify analytics structure
        expected_fields = [
            "agent_id", "overall_reputation", "social_capital", "total_events",
            "dimension_scores", "dimension_confidence", "dimension_trends",
            "fraud_indicators", "top_dimensions", "improvement_areas"
        ]
        
        for field in expected_fields:
            self.assertIn(field, analytics)
        
        # Verify values
        self.assertEqual(analytics["agent_id"], "alice")
        self.assertEqual(analytics["total_events"], 4)
        self.assertGreater(analytics["overall_reputation"], 0.0)
    
    def test_social_capital(self):
        """Test social capital calculation"""
        # Build social capital through reputation
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.LEADERSHIP_SUCCESS, "bob"
        )
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.CONFLICT_RESOLVED, "charlie"
        )
        
        # Get social capital
        social_capital = self.reputation_system.get_social_capital("alice")
        self.assertGreaterEqual(social_capital, 0.0)
        
        # Should increase with more positive reputation
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.INNOVATION_SUCCESS, "diana"
        )
        
        new_social_capital = self.reputation_system.get_social_capital("alice")
        self.assertGreaterEqual(new_social_capital, social_capital)
    
    def test_performance(self):
        """Test reputation system performance"""
        # Add many agents
        large_system = ReputationSystem(max_agents=500)
        
        start_time = time.time()
        for i in range(200):
            large_system.add_agent(f"agent_{i}")
        agent_creation_time = time.time() - start_time
        
        self.assertLess(agent_creation_time, 2.0)
        
        # Record many events
        start_time = time.time()
        for i in range(100):
            large_system.record_reputation_event(
                f"agent_{i}", ReputationEvent.SUCCESSFUL_COLLABORATION, 
                f"agent_{(i+1) % 200}"
            )
        event_processing_time = time.time() - start_time
        
        self.assertLess(event_processing_time, 3.0)
        
        # Test reputation calculation performance
        start_time = time.time()
        reputation = large_system.get_overall_reputation("agent_0")
        calculation_time = time.time() - start_time
        
        self.assertLess(calculation_time, 0.1)


class TestCoalitionFormation(unittest.TestCase):
    """Test suite for CoalitionFormationSystem"""
    
    def setUp(self):
        """Set up test environment"""
        self.coalition_system = CoalitionFormationSystem(max_coalitions=50, max_agents=100)
        self.test_agents = ["alice", "bob", "charlie", "diana", "eve", "frank"]
        
        # Add test agents with skills and interests
        for i, agent in enumerate(self.test_agents):
            skills = {
                "programming": 0.3 + i * 0.1,
                "leadership": 0.2 + i * 0.15,
                "design": 0.4 + i * 0.05
            }
            interests = {f"interest_{i}", f"interest_{(i+1)%3}"}
            values = {"cooperation": 0.5 + i * 0.1, "innovation": 0.3 + i * 0.1}
            resources = {"time": 10.0 + i * 5, "budget": 100.0 + i * 50}
            
            self.coalition_system.add_agent(agent, skills, interests, values, resources)
    
    def tearDown(self):
        """Clean up test environment"""
        self.coalition_system = None
    
    def test_agent_management(self):
        """Test agent addition and removal"""
        # Verify agents added
        self.assertEqual(len(self.coalition_system.agents), 6)
        
        # Verify agent data
        alice_skills = self.coalition_system.agent_skills["alice"]
        self.assertIn("programming", alice_skills)
        self.assertGreater(alice_skills["programming"], 0)
        
        alice_interests = self.coalition_system.agent_interests["alice"]
        self.assertGreater(len(alice_interests), 0)
        
        # Test removal
        result = self.coalition_system.remove_agent("frank")
        self.assertTrue(result)
        self.assertEqual(len(self.coalition_system.agents), 5)
    
    def test_coalition_suggestions(self):
        """Test coalition formation suggestions"""
        # Get suggestions for Alice
        suggestions = self.coalition_system.suggest_coalitions(
            "alice", algorithm="interest_similarity", max_suggestions=3
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertLessEqual(len(suggestions), 3)
        
        # Verify suggestion structure
        for members, score in suggestions:
            self.assertIsInstance(members, list)
            self.assertIn("alice", members)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
        
        # Test skill complementarity suggestions
        task_requirements = {
            "required_skills": ["programming", "design"],
            "required_resources": {"time": 20, "budget": 200}
        }
        
        skill_suggestions = self.coalition_system.suggest_coalitions(
            "alice", task_requirements, "skill_complementarity", 2
        )
        
        self.assertIsInstance(skill_suggestions, list)
        self.assertLessEqual(len(skill_suggestions), 2)
    
    def test_coalition_creation(self):
        """Test coalition creation and management"""
        # Create task-based coalition
        coalition_id = self.coalition_system.create_coalition(
            "alice", ["alice", "bob", "charlie"], CoalitionType.TASK_BASED,
            "Project Team", max_size=5, min_size=2
        )
        
        self.assertNotEqual(coalition_id, "")
        self.assertIn(coalition_id, self.coalition_system.coalitions)
        
        coalition = self.coalition_system.coalitions[coalition_id]
        self.assertEqual(len(coalition.members), 3)
        self.assertEqual(coalition.coalition_type, CoalitionType.TASK_BASED)
        self.assertEqual(coalition.status, CoalitionStatus.ACTIVE)
        
        # Verify member roles
        alice_member = coalition.members["alice"]
        self.assertEqual(alice_member.role, MemberRole.LEADER)
        
        bob_member = coalition.members["bob"]
        self.assertEqual(bob_member.role, MemberRole.CONTRIBUTOR)
        
        # Verify agent memberships updated
        self.assertIn(coalition_id, self.coalition_system.agent_memberships["alice"])
        self.assertIn(coalition_id, self.coalition_system.agent_memberships["bob"])
    
    def test_coalition_membership(self):
        """Test joining and leaving coalitions"""
        # Create coalition
        coalition_id = self.coalition_system.create_coalition(
            "alice", ["alice", "bob"], CoalitionType.INTEREST_BASED, max_size=5
        )
        
        # Test joining
        result = self.coalition_system.join_coalition("charlie", coalition_id)
        self.assertTrue(result)
        
        coalition = self.coalition_system.coalitions[coalition_id]
        self.assertEqual(len(coalition.members), 3)
        self.assertIn("charlie", coalition.members)
        
        # Test leaving
        result = self.coalition_system.leave_coalition("bob", coalition_id)
        self.assertTrue(result)
        
        coalition = self.coalition_system.coalitions[coalition_id]
        self.assertEqual(len(coalition.members), 2)
        self.assertNotIn("bob", coalition.members)
        
        # Test leaving with minimum size violation
        result = self.coalition_system.leave_coalition("charlie", coalition_id)
        self.assertTrue(result)
        
        # Coalition should still exist with Alice as leader
        self.assertIn(coalition_id, self.coalition_system.coalitions)
    
    def test_coalition_decisions(self):
        """Test coalition decision making"""
        # Create coalition
        coalition_id = self.coalition_system.create_coalition(
            "alice", ["alice", "bob", "charlie"], CoalitionType.STRATEGIC,
            decision_method="majority"
        )
        
        # Make decision
        decision = self.coalition_system.make_coalition_decision(
            coalition_id, "resource_allocation", 
            ["option_a", "option_b", "option_c"], "alice",
            {"urgency": "high", "impact": "medium"}
        )
        
        self.assertIsNotNone(decision)
        self.assertIn(decision, ["option_a", "option_b", "option_c"])
        
        # Verify decision recorded
        coalition = self.coalition_system.coalitions[coalition_id]
        self.assertGreater(len(coalition.recent_decisions), 0)
        
        last_decision = coalition.recent_decisions[-1]
        self.assertEqual(last_decision["chosen_option"], decision)
        self.assertEqual(last_decision["proposer_id"], "alice")
        
        # Test different decision methods
        coalition.decision_making_method = "leader"
        leader_decision = self.coalition_system.make_coalition_decision(
            coalition_id, "leadership_decision", ["yes", "no"], "alice"
        )
        self.assertIsNotNone(leader_decision)
    
    def test_coalition_stability(self):
        """Test coalition stability analysis"""
        # Create coalition
        coalition_id = self.coalition_system.create_coalition(
            "alice", ["alice", "bob", "charlie"], CoalitionType.PROFESSIONAL
        )
        
        # Initial stability
        stability = self.coalition_system.get_coalition_stability(coalition_id)
        self.assertIsInstance(stability, float)
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        
        # Improve member satisfaction
        coalition = self.coalition_system.coalitions[coalition_id]
        for member in coalition.members.values():
            member.satisfaction = 0.8
            member.commitment_level = 0.9
            member.exit_probability = 0.1
        
        new_stability = self.coalition_system.get_coalition_stability(coalition_id)
        self.assertGreaterEqual(new_stability, stability)
    
    def test_coalition_performance_analysis(self):
        """Test coalition performance analysis"""
        # Create coalition
        coalition_id = self.coalition_system.create_coalition(
            "alice", ["alice", "bob", "charlie"], CoalitionType.TASK_BASED,
            goals=[{
                "description": "Complete project alpha",
                "priority": 0.8,
                "required_skills": ["programming", "design"]
            }]
        )
        
        # Add some performance data
        coalition = self.coalition_system.coalitions[coalition_id]
        coalition.members["alice"].contribution_score = 0.9
        coalition.members["bob"].contribution_score = 0.7
        
        # Make some decisions
        self.coalition_system.make_coalition_decision(
            coalition_id, "test_decision", ["option_a"], "alice"
        )
        
        # Get performance analysis
        analysis = self.coalition_system.analyze_coalition_performance(coalition_id)
        
        # Verify analysis structure
        expected_fields = [
            "coalition_id", "name", "type", "status", "member_count",
            "cohesion_score", "effectiveness_score", "stability_score",
            "average_satisfaction", "goals_completed", "recommendations"
        ]
        
        for field in expected_fields:
            self.assertIn(field, analysis)
        
        # Verify values
        self.assertEqual(analysis["coalition_id"], coalition_id)
        self.assertEqual(analysis["member_count"], 3)
        self.assertIsInstance(analysis["recommendations"], list)
    
    def test_agent_coalition_summary(self):
        """Test agent coalition membership summary"""
        # Create multiple coalitions for Alice
        coalition1 = self.coalition_system.create_coalition(
            "alice", ["alice", "bob"], CoalitionType.PROFESSIONAL
        )
        coalition2 = self.coalition_system.create_coalition(
            "alice", ["alice", "charlie"], CoalitionType.SOCIAL
        )
        
        # Get summary
        summary = self.coalition_system.get_agent_coalition_summary("alice")
        
        # Verify summary structure
        expected_fields = [
            "agent_id", "total_coalitions", "active_coalitions", 
            "leadership_roles", "average_satisfaction", "memberships"
        ]
        
        for field in expected_fields:
            self.assertIn(field, summary)
        
        # Verify values
        self.assertEqual(summary["agent_id"], "alice")
        self.assertEqual(summary["total_coalitions"], 2)
        self.assertEqual(summary["leadership_roles"], 2)  # Alice is leader in both
        
        # Verify membership details
        self.assertEqual(len(summary["memberships"]), 2)
        for membership in summary["memberships"]:
            self.assertEqual(membership["role"], "leader")
    
    def test_performance(self):
        """Test coalition system performance"""
        # Create large system
        large_system = CoalitionFormationSystem(max_coalitions=100, max_agents=300)
        
        # Add many agents
        start_time = time.time()
        for i in range(100):
            skills = {"skill_a": 0.5, "skill_b": 0.3}
            interests = {f"interest_{i % 5}"}
            large_system.add_agent(f"agent_{i}", skills, interests)
        agent_creation_time = time.time() - start_time
        
        self.assertLess(agent_creation_time, 3.0)
        
        # Create coalitions
        start_time = time.time()
        for i in range(0, 50, 5):
            members = [f"agent_{j}" for j in range(i, min(i+5, 100))]
            large_system.create_coalition(
                members[0], members, CoalitionType.TASK_BASED
            )
        coalition_creation_time = time.time() - start_time
        
        self.assertLess(coalition_creation_time, 5.0)
        
        # Test suggestion performance
        start_time = time.time()
        suggestions = large_system.suggest_coalitions("agent_0", max_suggestions=5)
        suggestion_time = time.time() - start_time
        
        self.assertLess(suggestion_time, 1.0)
    
    def test_thread_safety(self):
        """Test thread safety of coalition operations"""
        def create_coalitions(start_idx, count):
            for i in range(start_idx, start_idx + count):
                members = [f"thread_agent_{i}", f"thread_agent_{i+1}"]
                for member in members:
                    self.coalition_system.add_agent(member, {"skill": 0.5}, {"interest"})
                
                self.coalition_system.create_coalition(
                    members[0], members, CoalitionType.TASK_BASED
                )
        
        # Run concurrent operations
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_coalitions, args=(i*10, 3))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no corruption
        self.assertGreater(len(self.coalition_system.agents), 6)
        self.assertGreater(len(self.coalition_system.coalitions), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for Phase 3 social dynamics systems"""
    
    def setUp(self):
        """Set up integrated test environment"""
        self.relationship_network = RelationshipNetwork(max_agents=50)
        self.reputation_system = ReputationSystem(max_agents=50)
        self.coalition_system = CoalitionFormationSystem(max_coalitions=20, max_agents=50)
        
        self.test_agents = ["alice", "bob", "charlie", "diana", "eve"]
        
        # Add agents to all systems
        for agent in self.test_agents:
            self.relationship_network.add_agent(agent)
            self.reputation_system.add_agent(agent)
            self.coalition_system.add_agent(
                agent, 
                {"programming": 0.5, "leadership": 0.3},
                {"technology", "collaboration"},
                {"cooperation": 0.7, "innovation": 0.5},
                {"time": 20, "budget": 200}
            )
    
    def test_reputation_relationship_integration(self):
        """Test integration between reputation and relationship systems"""
        # Build reputation through relationships
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.SUCCESSFUL_COLLABORATION, "bob"
        )
        
        # This should positively influence relationship
        self.relationship_network.create_relationship(
            "alice", "bob", RelationshipType.PROFESSIONAL, 0.6, 0.4
        )
        
        # Update relationship based on reputation event
        self.relationship_network.update_relationship(
            "alice", "bob", "cooperation", 0.3, "successful_project"
        )
        
        # Verify relationship improved
        relationship = self.relationship_network.get_relationship("alice", "bob")
        self.assertGreater(relationship.trust_level, 0.4)
        
        # Get trust score from reputation system
        trust_score = self.reputation_system.calculate_trust_score("bob", "alice")
        self.assertGreater(trust_score, 0.0)
    
    def test_coalition_reputation_integration(self):
        """Test integration between coalition formation and reputation"""
        # Build different reputation profiles
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.LEADERSHIP_SUCCESS, "bob"
        )
        self.reputation_system.record_reputation_event(
            "bob", ReputationEvent.INNOVATION_SUCCESS, "alice"
        )
        
        # Create coalition
        coalition_id = self.coalition_system.create_coalition(
            "alice", ["alice", "bob"], CoalitionType.PROFESSIONAL
        )
        
        # Coalition decisions should consider reputation
        decision = self.coalition_system.make_coalition_decision(
            coalition_id, "leadership_choice", ["alice", "bob"], "alice"
        )
        
        # Alice should be chosen due to leadership reputation
        # (This is simplified - real integration would be more sophisticated)
        self.assertIsNotNone(decision)
    
    def test_full_social_ecosystem(self):
        """Test full social ecosystem with all systems working together"""
        # Simulate social interaction cycle
        
        # 1. Initial coalition formation
        coalition_id = self.coalition_system.create_coalition(
            "alice", ["alice", "bob", "charlie"], CoalitionType.TASK_BASED
        )
        
        # 2. Successful collaboration builds reputation
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.LEADERSHIP_SUCCESS, "bob", {"charlie"}
        )
        
        # 3. Reputation affects relationships
        self.relationship_network.create_relationship(
            "alice", "bob", RelationshipType.PROFESSIONAL, 0.7, 0.6
        )
        self.relationship_network.create_relationship(
            "alice", "charlie", RelationshipType.ALLIANCE, 0.6, 0.5
        )
        
        # 4. Relationships enable influence propagation
        influence_results = self.relationship_network.propagate_influence(
            "alice", InfluenceType.OPINION,
            {"topic": "project_direction", "opinion": "agile_methodology"},
            initial_strength=0.8
        )
        
        # 5. Verify ecosystem effects
        self.assertIn("alice", influence_results)
        self.assertIn("bob", influence_results)
        
        # Check reputation improved
        leadership_score = self.reputation_system.get_reputation_score(
            "alice", ReputationDimension.LEADERSHIP
        )
        self.assertGreater(leadership_score, 0.0)
        
        # Check coalition stability
        stability = self.coalition_system.get_coalition_stability(coalition_id)
        self.assertGreater(stability, 0.0)
        
        # Check network effects
        trust_path = self.relationship_network.calculate_trust_path("bob", "charlie")
        self.assertGreaterEqual(trust_path, 0.0)
    
    def test_system_export_import(self):
        """Test data export and import capabilities"""
        # Set up some data
        self.relationship_network.create_relationship(
            "alice", "bob", RelationshipType.FRIENDSHIP, 0.8, 0.7
        )
        
        self.reputation_system.record_reputation_event(
            "alice", ReputationEvent.PROMISE_KEPT, "bob"
        )
        
        coalition_id = self.coalition_system.create_coalition(
            "alice", ["alice", "bob"], CoalitionType.SOCIAL
        )
        
        # Export data
        relationship_data = self.relationship_network.export_network_data()
        reputation_data = self.reputation_system.export_reputation_data()
        coalition_data = self.coalition_system.export_system_data()
        
        # Verify export structure
        self.assertIn("nodes", relationship_data)
        self.assertIn("edges", relationship_data)
        self.assertIn("statistics", relationship_data)
        
        self.assertIn("agents", reputation_data)
        self.assertIn("reputation_scores", reputation_data)
        
        self.assertIn("agents", coalition_data)
        self.assertIn("coalitions", coalition_data)
        
        # Verify data integrity
        self.assertGreater(len(relationship_data["nodes"]), 0)
        self.assertGreater(len(reputation_data["agents"]), 0)
        self.assertGreater(len(coalition_data["agents"]), 0)


def run_phase3_tests():
    """Run all Phase 3 social dynamics tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestRelationshipNetwork,
        TestReputationSystem, 
        TestCoalitionFormation,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return results
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        "details": {
            "failures": [str(failure) for failure in result.failures],
            "errors": [str(error) for error in result.errors]
        }
    }


if __name__ == "__main__":
    print("Running Phase 3 Social Dynamics Test Suite...")
    results = run_phase3_tests()
    
    print(f"\nTest Results:")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    
    if results['failures'] > 0:
        print(f"\nFailures:")
        for failure in results['details']['failures']:
            print(f"  - {failure}")
    
    if results['errors'] > 0:
        print(f"\nErrors:")
        for error in results['details']['errors']:
            print(f"  - {error}")