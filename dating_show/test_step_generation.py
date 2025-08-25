#!/usr/bin/env python3
"""
Test Script for Enhanced Step File Generation
Verifies that step files can be generated correctly before simulation starts
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# Import the step manager
from services.enhanced_step_manager import get_enhanced_step_manager

async def test_step_generation():
    """Test step file generation for dating_show_25_agents"""
    
    print("🧪 Testing Enhanced Step File Generation")
    print("=" * 50)
    
    # Get the step manager
    step_manager = get_enhanced_step_manager()
    print(f"📁 Storage path: {step_manager.storage_path}")
    
    # Test generating step 1 files
    sim_code = "dating_show_25_agents"
    test_step = 1
    
    print(f"\n🎯 Testing generation for {sim_code}, step {test_step}")
    
    result = await step_manager.ensure_step_files_exist(sim_code, test_step)
    
    print(f"\n📊 Generation Result:")
    print(f"   Success: {result.success}")
    print(f"   Step: {result.step}")
    print(f"   Sim Code: {result.sim_code}")
    print(f"   Strategy Used: {result.strategy_used}")
    print(f"   Files Created: {result.files_created}")
    print(f"   Execution Time: {result.execution_time:.2f}s")
    
    if result.error:
        print(f"   Error: {result.error}")
    
    # Verify files exist
    sim_path = step_manager.storage_path / sim_code
    env_file = sim_path / 'environment' / f'{test_step}.json'
    mov_file = sim_path / 'movement' / f'{test_step}.json'
    
    print(f"\n🔍 File Verification:")
    print(f"   Environment file exists: {env_file.exists()}")
    print(f"   Movement file exists: {mov_file.exists()}")
    
    if env_file.exists():
        print(f"   Environment file path: {env_file}")
    if mov_file.exists():
        print(f"   Movement file path: {mov_file}")
    
    return result.success

def test_sync_generation():
    """Test synchronous generation (mimics what main.py does)"""
    print("\n🔄 Testing Synchronous Generation (Main.py Style)")
    print("-" * 50)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        success = loop.run_until_complete(test_step_generation())
        print(f"\n✅ Synchronous test result: {'SUCCESS' if success else 'FAILED'}")
        return success
    except Exception as e:
        print(f"\n❌ Synchronous test error: {e}")
        return False
    finally:
        loop.close()

if __name__ == "__main__":
    print("🚀 Starting Step Generation Test")
    success = test_sync_generation()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("   Step 1 files should now exist for dating_show_25_agents")
        print("   The simulation should be able to advance past the blocking point")
    else:
        print("\n💥 Test failed!")
        print("   There are still issues with step file generation")
    
    sys.exit(0 if success else 1)