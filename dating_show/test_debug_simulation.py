#!/usr/bin/env python3
"""
Debug Test for Simulation Step Advancement
Shows complete debug output for step file generation and simulation advancement
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dating_show.services.enhanced_step_manager import get_enhanced_step_manager
from dating_show.main import ReverieIntegrationManager

async def test_step_generation():
    """Test step file generation with debug output"""
    print("=" * 60)
    print("🧪 TESTING ENHANCED STEP FILE GENERATION")
    print("=" * 60)
    
    step_manager = get_enhanced_step_manager()
    
    # Test multiple steps to show different strategies
    for step in [1, 2, 3]:
        print(f"\n🎯 Testing step {step} generation:")
        print("-" * 40)
        
        result = await step_manager.ensure_step_files_exist('dating_show_25_agents', step)
        
        print(f"Result: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print(f"Strategy: {result.strategy_used}")
        print(f"Time: {result.execution_time:.3f}s")
        
        if result.error:
            print(f"Error: {result.error}")
    
    print("\n" + "=" * 60)
    print("🏁 STEP GENERATION TEST COMPLETE")
    print("=" * 60)

def test_simulation_manager():
    """Test simulation manager initialization"""
    print("\n" + "=" * 60)
    print("🎬 TESTING SIMULATION MANAGER")
    print("=" * 60)
    
    try:
        # Initialize simulation manager
        print("🔧 Initializing ReverieIntegrationManager...")
        sim_manager = ReverieIntegrationManager()
        print(f"✅ Simulation manager created: {sim_manager}")
        
        # Check if reverie server is available
        print(f"🔍 Reverie server available: {hasattr(sim_manager, 'reverie_server')}")
        
        if hasattr(sim_manager, 'reverie_server') and sim_manager.reverie_server:
            print(f"🎯 Current step: {sim_manager.reverie_server.step}")
        else:
            print("⚠️ Reverie server not initialized - this is expected in test mode")
        
        print("✅ Simulation manager test complete")
        
    except Exception as e:
        print(f"❌ Simulation manager test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("🚀 Starting comprehensive debug test...")
    
    # Test step generation
    await test_step_generation()
    
    # Test simulation manager
    test_simulation_manager()
    
    print(f"\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())