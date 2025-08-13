#!/usr/bin/env python3
"""
Simplified Frontend Bridge Test
Tests the core bridge functionality without complex dependencies
"""

import sys
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Any
from dataclasses import dataclass
import asyncio

@dataclass
class AgentUpdate:
    """Simple agent update for testing"""
    agent_id: str
    name: str
    current_role: str
    specialization: Dict[str, Any]
    skills: Dict[str, Dict[str, float]]
    memory: Dict[str, Any]
    location: Dict[str, Any]
    current_action: str
    timestamp: datetime


class SimpleFrontendBridge:
    """Simplified bridge for testing core functionality"""
    
    def __init__(self, frontend_url: str = "http://localhost:8000", 
                 update_interval: float = 1.0):
        self.frontend_url = frontend_url
        self.update_interval = update_interval
        self.agent_updates = asyncio.Queue(maxsize=1000)
        self.governance_updates = asyncio.Queue(maxsize=500)
        self.social_updates = asyncio.Queue(maxsize=500)
        self.agent_state_cache = {}
        self.running = False
        
    def queue_agent_update(self, agent_id: str, agent_data: Dict[str, Any]):
        """Queue an agent state update"""
        try:
            update = AgentUpdate(
                agent_id=agent_id,
                name=agent_data.get('name', ''),
                current_role=agent_data.get('current_role', ''),
                specialization=agent_data.get('specialization', {}),
                skills=agent_data.get('skills', {}),
                memory=agent_data.get('memory', {}),
                location=agent_data.get('location', {}),
                current_action=agent_data.get('current_action', ''),
                timestamp=datetime.now(timezone.utc)
            )
            
            if not self.agent_updates.full():
                self.agent_updates.put_nowait(update)
                return True
            return False
        except Exception as e:
            print(f"Error queuing agent update: {e}")
            return False
            
    def get_cached_agent_state(self, agent_id: str):
        """Get cached agent state"""
        return self.agent_state_cache.get(agent_id)
        
    def clear_cache(self):
        """Clear cache"""
        self.agent_state_cache.clear()
        
    def get_bridge_status(self):
        """Get bridge status"""
        return {
            'running': self.running,
            'cached_agents': len(self.agent_state_cache),
            'queue_sizes': {
                'agent_updates': self.agent_updates.qsize(),
                'governance_updates': self.governance_updates.qsize(),
                'social_updates': self.social_updates.qsize()
            },
            'frontend_url': self.frontend_url,
            'update_interval': self.update_interval
        }


def test_bridge_initialization():
    """Test bridge initialization"""
    bridge = SimpleFrontendBridge()
    assert bridge.frontend_url == "http://localhost:8000"
    assert bridge.update_interval == 1.0
    assert not bridge.running
    assert len(bridge.agent_state_cache) == 0
    print("✅ Bridge initialization test passed")


def test_agent_update_creation():
    """Test agent update creation"""
    update = AgentUpdate(
        agent_id="test_001",
        name="Test Agent",
        current_role="contestant",
        specialization={"role": "contestant"},
        skills={"social": {"level": 0.5, "experience": 10.0}},
        memory={"thoughts": ["test thought"]},
        location={"sector": "villa", "arena": "living_room"},
        current_action="socializing",
        timestamp=datetime.now(timezone.utc)
    )
    
    assert update.agent_id == "test_001"
    assert update.name == "Test Agent"
    assert update.current_role == "contestant"
    assert "social" in update.skills
    print("✅ Agent update creation test passed")


def test_queue_operations():
    """Test queue operations"""
    bridge = SimpleFrontendBridge()
    
    # Test agent update queuing
    agent_data = {
        'name': 'Queue Test Agent',
        'current_role': 'participant',
        'specialization': {'test': True},
        'skills': {'testing': {'level': 1.0}},
        'memory': {'queue_test': True},
        'location': {'test_area': True},
        'current_action': 'testing'
    }
    
    initial_size = bridge.agent_updates.qsize()
    success = bridge.queue_agent_update("queue_test_001", agent_data)
    
    assert success
    assert bridge.agent_updates.qsize() == initial_size + 1
    print("✅ Queue operations test passed")


def test_cache_operations():
    """Test cache operations"""
    bridge = SimpleFrontendBridge()
    
    # Create test update
    update = AgentUpdate(
        agent_id="cache_test_001",
        name="Cache Test Agent",
        current_role="host",
        specialization={},
        skills={},
        memory={},
        location={},
        current_action="hosting",
        timestamp=datetime.now(timezone.utc)
    )
    
    # Test cache storage
    bridge.agent_state_cache["cache_test_001"] = update
    
    # Test cache retrieval
    cached = bridge.get_cached_agent_state("cache_test_001")
    assert cached is not None
    assert cached.name == "Cache Test Agent"
    assert cached.current_role == "host"
    
    # Test cache clearing
    bridge.clear_cache()
    assert len(bridge.agent_state_cache) == 0
    print("✅ Cache operations test passed")


def test_status_reporting():
    """Test status reporting"""
    bridge = SimpleFrontendBridge()
    
    # Add some test data
    bridge.agent_state_cache["status_agent"] = AgentUpdate(
        agent_id="status_agent",
        name="Status Agent",
        current_role="participant",
        specialization={},
        skills={},
        memory={},
        location={},
        current_action="reporting",
        timestamp=datetime.now(timezone.utc)
    )
    
    status = bridge.get_bridge_status()
    
    assert 'running' in status
    assert 'cached_agents' in status
    assert 'queue_sizes' in status
    assert 'frontend_url' in status
    assert 'update_interval' in status
    
    assert status['cached_agents'] == 1
    assert status['frontend_url'] == "http://localhost:8000"
    print("✅ Status reporting test passed")


def test_performance_basic():
    """Test basic performance"""
    bridge = SimpleFrontendBridge()
    
    # Time agent update queuing
    start_time = time.time()
    
    for i in range(50):
        agent_data = {
            'name': f'Perf Agent {i}',
            'current_role': 'contestant',
            'specialization': {'perf_test': True},
            'skills': {'performance': {'level': i * 0.02}},
            'memory': {'iteration': i},
            'location': {'test_zone': True},
            'current_action': f'performing_{i}'
        }
        success = bridge.queue_agent_update(f"perf_agent_{i:03d}", agent_data)
        assert success
        
    duration = time.time() - start_time
    
    # Should be very fast (less than 50ms for 50 updates)
    assert duration < 0.05
    assert bridge.agent_updates.qsize() == 50
    print(f"✅ Performance test passed (50 updates in {duration*1000:.1f}ms)")


def run_all_tests():
    """Run all simplified bridge tests"""
    print("\n🧪 Simplified Frontend Bridge Test Suite")
    print("=" * 50)
    
    test_functions = [
        test_bridge_initialization,
        test_agent_update_creation,
        test_queue_operations,
        test_cache_operations,
        test_status_reporting,
        test_performance_basic
    ]
    
    passed = 0
    failed = 0
    
    start_time = time.time()
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
            
    duration = time.time() - start_time
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results")
    print("=" * 50)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    print(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
    print(f"Duration: {duration:.3f}s")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)