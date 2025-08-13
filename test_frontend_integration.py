#!/usr/bin/env python3
"""
Frontend Integration Test Suite
Tests for Phase 5: Frontend Bridge Service and PIANO Integration
"""

import sys
import time
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Any
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add current directory to path
sys.path.append('.')
sys.path.append('./dating_show')

try:
    from dating_show.api.frontend_bridge import FrontendBridge, AgentUpdate, GovernanceUpdate, SocialUpdate
    from dating_show.api.piano_integration import PianoFrontendIntegration
    print("✅ Successfully imported frontend integration modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class TestFrontendBridge(unittest.TestCase):
    """Test the frontend bridge service functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.bridge = FrontendBridge(
            frontend_url="http://localhost:8000",
            update_interval=0.1  # Fast interval for testing
        )
        
    def test_bridge_initialization(self):
        """Test bridge service initialization"""
        self.assertEqual(self.bridge.frontend_url, "http://localhost:8000")
        self.assertEqual(self.bridge.update_interval, 0.1)
        self.assertFalse(self.bridge.running)
        self.assertEqual(len(self.bridge.agent_state_cache), 0)
        
    def test_agent_update_creation(self):
        """Test AgentUpdate dataclass creation"""
        update = AgentUpdate(
            agent_id="test_agent_001",
            name="Test Agent",
            current_role="contestant",
            specialization={"role": "contestant", "skills": ["social"]},
            skills={"social": {"level": 0.5, "experience": 10.0}},
            memory={"recent_thoughts": ["I am thinking"]},
            location={"sector": "villa", "arena": "living_room"},
            current_action="socializing",
            timestamp=datetime.now(timezone.utc)
        )
        
        self.assertEqual(update.agent_id, "test_agent_001")
        self.assertEqual(update.name, "Test Agent")
        self.assertEqual(update.current_role, "contestant")
        self.assertIn("social", update.skills)
        
    def test_queue_agent_update(self):
        """Test queuing agent updates"""
        agent_data = {
            'name': 'Test Agent',
            'current_role': 'contestant', 
            'specialization': {'role': 'contestant'},
            'skills': {'social': {'level': 0.5}},
            'memory': {'thoughts': ['test thought']},
            'location': {'sector': 'villa'},
            'current_action': 'thinking'
        }
        
        initial_queue_size = self.bridge.agent_updates.qsize()
        self.bridge.queue_agent_update("test_agent", agent_data)
        
        self.assertEqual(self.bridge.agent_updates.qsize(), initial_queue_size + 1)
        
    def test_queue_governance_update(self):
        """Test queuing governance updates"""
        data = {
            'vote_id': 'test_vote_001',
            'title': 'Test Vote',
            'description': 'A test governance vote'
        }
        
        initial_queue_size = self.bridge.governance_updates.qsize()
        self.bridge.queue_governance_update("new_vote", data)
        
        self.assertEqual(self.bridge.governance_updates.qsize(), initial_queue_size + 1)
        
    def test_queue_social_update(self):
        """Test queuing social network updates"""
        initial_queue_size = self.bridge.social_updates.qsize()
        self.bridge.queue_social_update(
            "agent_a", "agent_b", "friendship", 0.7, "conversation"
        )
        
        self.assertEqual(self.bridge.social_updates.qsize(), initial_queue_size + 1)
        
    def test_bridge_status(self):
        """Test bridge status reporting"""
        status = self.bridge.get_bridge_status()
        
        self.assertIn('running', status)
        self.assertIn('cached_agents', status)
        self.assertIn('queue_sizes', status)
        self.assertIn('frontend_url', status)
        self.assertIn('update_interval', status)
        
        self.assertEqual(status['frontend_url'], "http://localhost:8000")
        self.assertEqual(status['update_interval'], 0.1)
        
    def test_cache_management(self):
        """Test agent state caching"""
        # Add mock agent to cache
        update = AgentUpdate(
            agent_id="cached_agent",
            name="Cached Agent",
            current_role="host",
            specialization={},
            skills={},
            memory={},
            location={},
            current_action="hosting",
            timestamp=datetime.now(timezone.utc)
        )
        
        self.bridge.agent_state_cache["cached_agent"] = update
        
        # Test retrieval
        cached = self.bridge.get_cached_agent_state("cached_agent")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.name, "Cached Agent")
        self.assertEqual(cached.current_role, "host")
        
        # Test cache clearing
        self.bridge.clear_cache()
        self.assertEqual(len(self.bridge.agent_state_cache), 0)
        
    def tearDown(self):
        """Clean up after tests"""
        if self.bridge.running:
            self.bridge.stop_bridge()


class TestPianoIntegration(unittest.TestCase):
    """Test PIANO system integration functionality"""
    
    def setUp(self):
        """Set up test environment with mock bridge"""
        # Create a mock bridge
        self.mock_bridge = Mock(spec=FrontendBridge)
        self.integration = PianoFrontendIntegration(self.mock_bridge)
        
    def test_integration_initialization(self):
        """Test integration layer initialization"""
        self.assertIsNotNone(self.integration.bridge)
        self.assertEqual(len(self.integration.monitored_agents), 0)
        self.assertEqual(len(self.integration.last_sync_states), 0)
        
    def test_agent_registration(self):
        """Test agent registration for monitoring"""
        # Create a mock agent
        mock_agent = Mock()
        mock_agent.scratch = Mock()
        mock_agent.scratch.name = "test_agent_001"
        mock_agent.scratch.act_address = ["thinking", "talking"]
        mock_agent.scratch.curr_tile = ["villa", "living_room", "sofa"]
        
        # Register agent
        self.integration.register_agent(mock_agent)
        
        # Verify registration
        self.assertIn("test_agent_001", self.integration.monitored_agents)
        self.assertEqual(self.integration.monitored_agents["test_agent_001"], mock_agent)
        
        # Verify bridge was called to queue update
        self.mock_bridge.queue_agent_update.assert_called_once()
        
    def test_agent_unregistration(self):
        """Test agent unregistration"""
        # Register then unregister an agent
        mock_agent = Mock()
        mock_agent.scratch = Mock()
        mock_agent.scratch.name = "test_agent_002"
        mock_agent.scratch.act_address = []
        mock_agent.scratch.curr_tile = []
        
        self.integration.register_agent(mock_agent)
        self.assertIn("test_agent_002", self.integration.monitored_agents)
        
        self.integration.unregister_agent("test_agent_002")
        self.assertNotIn("test_agent_002", self.integration.monitored_agents)
        
    def test_agent_data_extraction(self):
        """Test extraction of agent data for frontend"""
        # Create a comprehensive mock agent
        mock_agent = Mock()
        mock_agent.scratch = Mock()
        mock_agent.scratch.name = "comprehensive_agent"
        mock_agent.scratch.act_address = ["socializing", "talking", "laughing"]
        mock_agent.scratch.curr_tile = ["villa", "kitchen", "counter"]
        mock_agent.scratch.innate = "friendly; outgoing; creative"
        
        # Extract data
        data = self.integration._extract_agent_data(mock_agent)
        
        # Verify extraction
        self.assertEqual(data['name'], "comprehensive_agent")
        self.assertEqual(data['current_action'], "laughing")
        self.assertEqual(data['location']['sector'], "villa")
        self.assertEqual(data['location']['arena'], "kitchen")
        self.assertEqual(data['location']['game_object'], "counter")
        
        # Verify data structure
        self.assertIn('specialization', data)
        self.assertIn('skills', data)
        self.assertIn('memory', data)
        self.assertIn('social_context', data)
        
    def test_role_detection(self):
        """Test automatic role detection"""
        mock_agent = Mock()
        mock_agent.scratch = Mock()
        mock_agent.scratch.act_address = ["hosting", "introducing", "moderating"]
        
        role = self.integration._detect_current_role(mock_agent)
        self.assertEqual(role, "host")
        
        # Test contestant detection
        mock_agent.scratch.act_address = ["contest", "competing", "participating"]
        role = self.integration._detect_current_role(mock_agent)
        self.assertEqual(role, "contestant")
        
    def test_governance_event_handling(self):
        """Test governance event forwarding"""
        event_data = {
            'vote_id': 'governance_test_001',
            'title': 'Test Governance Event',
            'type': 'new_vote'
        }
        
        self.integration.handle_governance_event("new_vote", event_data)
        
        # Verify event was forwarded to bridge
        self.mock_bridge.queue_governance_update.assert_called_once_with("new_vote", event_data)
        
    def test_social_event_handling(self):
        """Test social event forwarding"""
        self.integration.handle_social_event(
            "agent_a", "agent_b", "romantic", 0.8, "date"
        )
        
        # Verify event was forwarded to bridge
        self.mock_bridge.queue_social_update.assert_called_once_with(
            "agent_a", "agent_b", "romantic", 0.8, "date"
        )
        
    def test_integration_status(self):
        """Test integration status reporting"""
        # Register a mock agent
        mock_agent = Mock()
        mock_agent.scratch = Mock()
        mock_agent.scratch.name = "status_test_agent"
        mock_agent.scratch.act_address = []
        mock_agent.scratch.curr_tile = []
        
        self.integration.register_agent(mock_agent)
        
        # Get status
        status = self.integration.get_integration_status()
        
        self.assertEqual(status['monitored_agents'], 1)
        self.assertIn("status_test_agent", status['agent_ids'])
        self.assertIn('bridge_status', status)
        self.assertIn('last_sync_count', status)


class TestIntegrationPerformance(unittest.TestCase):
    """Test performance aspects of the integration"""
    
    def setUp(self):
        """Set up performance testing environment"""
        self.bridge = FrontendBridge(update_interval=0.01)  # Very fast for testing
        
    def test_agent_update_performance(self):
        """Test performance of agent update queuing"""
        start_time = time.time()
        
        # Queue many updates
        for i in range(100):
            agent_data = {
                'name': f'Performance Agent {i}',
                'current_role': 'contestant',
                'specialization': {'test': True},
                'skills': {'performance': {'level': i * 0.01}},
                'memory': {'count': i},
                'location': {'test': True},
                'current_action': f'action_{i}'
            }
            self.bridge.queue_agent_update(f"perf_agent_{i:03d}", agent_data)
            
        duration = time.time() - start_time
        
        # Should be very fast (less than 100ms for 100 updates)
        self.assertLess(duration, 0.1)
        self.assertEqual(self.bridge.agent_updates.qsize(), 100)
        
    def test_cache_performance(self):
        """Test cache retrieval performance"""
        # Populate cache
        for i in range(50):
            update = AgentUpdate(
                agent_id=f"cache_agent_{i:03d}",
                name=f"Cache Agent {i}",
                current_role="participant",
                specialization={},
                skills={},
                memory={},
                location={},
                current_action="cached",
                timestamp=datetime.now(timezone.utc)
            )
            self.bridge.agent_state_cache[f"cache_agent_{i:03d}"] = update
            
        # Test retrieval performance
        start_time = time.time()
        
        for i in range(50):
            cached = self.bridge.get_cached_agent_state(f"cache_agent_{i:03d}")
            self.assertIsNotNone(cached)
            
        duration = time.time() - start_time
        
        # Cache retrieval should be very fast
        self.assertLess(duration, 0.01)  # Less than 10ms for 50 retrievals


def run_integration_tests():
    """Run all frontend integration tests"""
    print("\n🧪 Frontend Integration Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestFrontendBridge,
        TestPianoIntegration,
        TestIntegrationPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    start_time = time.time()
    result = runner.run(test_suite)
    duration = time.time() - start_time
    
    # Generate summary
    print("\n" + "=" * 60)
    print(f"🏁 Frontend Integration Test Results")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Duration: {duration:.2f}s")
    
    if result.failures:
        print(f"\n❌ Failures ({len(result.failures)}):")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0] if 'AssertionError: ' in traceback else 'See details above'
            print(f"  • {test}: {error_msg}")
            
    if result.errors:
        print(f"\n🚨 Errors ({len(result.errors)}):")
        for test, traceback in result.errors:
            lines = traceback.split('\n')
            error_msg = lines[-2] if len(lines) > 1 else traceback
            print(f"  • {test}: {error_msg}")
    
    if len(result.failures) == 0 and len(result.errors) == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {len(result.failures) + len(result.errors)} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)