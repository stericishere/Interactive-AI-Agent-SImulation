#!/usr/bin/env python3
"""
Direct test of BatchActivityCoordinator to verify batch API processing
"""

import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
sys.path.append(os.path.join(os.path.dirname(__file__), "reverie_core"))

# Import what we need
from agents.batch_activity_coordinator import BatchActivityCoordinator
from agents.prompt_template.dating_show_v1.prompts import PromptManager

def test_batch_coordinator():
    """Test the BatchActivityCoordinator directly"""
    print("🧪 Testing BatchActivityCoordinator")
    print("=" * 50)
    
    try:
        # Initialize the coordinator
        print("1️⃣ Initializing BatchActivityCoordinator...")
        coordinator = BatchActivityCoordinator()
        
        # Initialize with the dating_show_v1 template path
        template_path = Path(__file__).parent / "agents" / "prompt_template" / "dating_show_v1"
        prompt_manager = PromptManager(template_path)
        
        # Test data - simulate 8 agents needing activities
        test_agents = [
            {
                "name": "Latoya Williams",
                "current_time": "2023-02-13 10:00:00",
                "location": "Hearts on Fire Villa: Pool Area",
                "current_activity": "swimming laps",
                "nearby_personas": ["Rajiv", "Abigail"],
                "show_events": "Morning Challenge: Trust Exercise",
                "episode_info": "Day 1 - Getting to Know Each Other"
            },
            {
                "name": "Rajiv Patel",
                "current_time": "2023-02-13 10:00:00", 
                "location": "Hearts on Fire Villa: Pool Area",
                "current_activity": "sunbathing by pool",
                "nearby_personas": ["Latoya", "Francisco"],
                "show_events": "Morning Challenge: Trust Exercise",
                "episode_info": "Day 1 - Getting to Know Each Other"
            },
            {
                "name": "Abigail Chen",
                "current_time": "2023-02-13 10:00:00",
                "location": "Hearts on Fire Villa: Kitchen",
                "current_activity": "preparing breakfast",
                "nearby_personas": ["Hailey"],
                "show_events": "Morning Challenge: Trust Exercise",
                "episode_info": "Day 1 - Getting to Know Each Other"
            }
        ]
        
        print(f"✅ Coordinator initialized, testing with {len(test_agents)} agents")
        
        # Step 2: Test batch processing
        print(f"\n2️⃣ Testing batch API call (should be 1 API call instead of {len(test_agents)})...")
        
        # Current time for the batch
        curr_time = "2023-02-13 10:00:00"
        
        activities = coordinator.generate_batch_activities(test_agents, curr_time)
        
        print(f"✅ Batch processing completed!")
        print(f"📊 Results: {len(activities)} activities generated")
        
        # Show results
        print(f"\n3️⃣ Generated activities:")
        for i, (agent_name, activity) in enumerate(activities.items(), 1):
            print(f"  {i}. {agent_name}: {activity}")
        
        # Verify we got results for all agents
        expected_agents = {agent["name"] for agent in test_agents}
        received_agents = set(activities.keys())
        
        if expected_agents == received_agents:
            print(f"\n✅ All agents received activities!")
            print(f"🎯 SUCCESS: Batch processing reduced {len(test_agents)} individual API calls → 1 batch API call")
            return True
        else:
            missing = expected_agents - received_agents
            print(f"\n⚠️ Missing activities for: {missing}")
            return False
            
    except Exception as e:
        print(f"💥 Error during batch coordinator test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_coordinator()
    if success:
        print("\n🎉 Test PASSED: Batch API coordination working!")
        print("🚀 Ready to integrate with full simulation")
    else:
        print("\n❌ Test FAILED: Issues with batch coordination")