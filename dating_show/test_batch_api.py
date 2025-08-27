#!/usr/bin/env python3
"""
Test script for batch API processing - run a few simulation steps automatically
"""

import sys
import os
from pathlib import Path
import time

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
sys.path.append(os.path.join(os.path.dirname(__file__), "reverie_core"))

# Import what we need
from services.integration_example import ensure_clean_8_agent_simulation
from reverie_core.reverie import ReverieServer
from global_methods import *

def test_batch_api_processing():
    """Test batch API processing with actual simulation steps"""
    print("🧪 Testing Batch API Processing")
    print("=" * 50)
    
    try:
        # Step 1: Ensure clean simulation
        print("1️⃣ Setting up clean simulation...")
        setup_success = ensure_clean_8_agent_simulation()
        if not setup_success:
            print("❌ Failed to setup clean simulation")
            return False
        print("✅ Clean simulation ready")
        
        # Step 2: Initialize ReverieServer
        print("\n2️⃣ Initializing ReverieServer...")
        sim_code = "dating_show_8_agents"
        sim_folder = Path(fs_storage) / sim_code
        temp_folder = Path(fs_storage) / f"{sim_code}_clean"
        
        if sim_folder.exists():
            # Clean up any existing temp folder
            if temp_folder.exists():
                import shutil
                shutil.rmtree(temp_folder)
            
            # Rename to temp for ReverieServer fork
            sim_folder.rename(temp_folder)
            
            # Create ReverieServer
            reverie_server = ReverieServer(f"{sim_code}_clean", sim_code)
            
            # Clean up temp folder
            if temp_folder.exists():
                import shutil
                shutil.rmtree(temp_folder)
            
            print(f"✅ ReverieServer initialized with {len(reverie_server.personas)} agents")
        else:
            print("❌ Simulation folder not found")
            return False
        
        # Step 3: Run 3 simulation steps programmatically
        print(f"\n3️⃣ Running 3 simulation steps to test batch API...")
        
        start_time = time.time()
        
        # Run 3 steps using the start_server method
        print("🎬 Starting simulation...")
        reverie_server.start_server(3)  # Run 3 steps
        
        total_time = time.time() - start_time
        print(f"⏱️ All steps completed in {total_time:.2f}s")
        
        # Show final agent statuses
        print("\n👥 Final agent statuses:")
        for i, (name, persona) in enumerate(list(reverie_server.personas.items())[:3], 1):  # Show first 3
            current_action = persona.scratch.act_description if hasattr(persona.scratch, 'act_description') else "unknown"
            print(f"  {i}. {name}: {current_action}")
        
        print(f"\n✅ Batch API test completed successfully!")
        return True
        
    except Exception as e:
        print(f"💥 Error during batch API test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_api_processing()
    if success:
        print("\n🎉 Test PASSED: Batch API processing working!")
    else:
        print("\n❌ Test FAILED: Issues with batch API processing")