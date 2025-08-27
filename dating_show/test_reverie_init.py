#!/usr/bin/env python3
"""
Test script to verify the Reverie simulation initialization without Django setup
"""

import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "reverie", "backend_server"))
sys.path.append(os.path.join(os.path.dirname(__file__), "reverie_core"))

# Import what we need
from services.integration_example import ensure_clean_8_agent_simulation
from reverie_core.reverie import ReverieServer
from global_methods import *

def test_reverie_initialization():
    """Test just the Reverie simulation initialization"""
    print("🧪 Testing Reverie Simulation Initialization")
    print("=" * 50)
    
    try:
        # Step 1: Ensure clean 8-agent simulation
        print("1️⃣ Ensuring clean 8-agent simulation...")
        setup_success = ensure_clean_8_agent_simulation()
        if setup_success:
            print("✅ Clean simulation setup completed")
        else:
            print("❌ Failed to setup clean simulation")
            return False
        
        # Step 2: Try the ReverieServer initialization with temp rename approach
        print("\n2️⃣ Testing ReverieServer initialization...")
        
        sim_code = "dating_show_8_agents"
        sim_folder = Path(fs_storage) / sim_code
        temp_folder = Path(fs_storage) / f"{sim_code}_clean"
        
        print(f"📁 Simulation folder: {sim_folder}")
        print(f"📁 Exists: {sim_folder.exists()}")
        
        if sim_folder.exists():
            print("🔄 Temporarily renaming simulation for ReverieServer fork...")
            
            # Clean up any existing temp folder
            if temp_folder.exists():
                import shutil
                shutil.rmtree(temp_folder)
                
            # Rename to temp
            sim_folder.rename(temp_folder)
            print(f"✅ Renamed to: {temp_folder}")
            
            # Create ReverieServer
            print("🏗️ Creating ReverieServer...")
            reverie_server = ReverieServer(f"{sim_code}_clean", sim_code)
            
            # Clean up temp folder
            if temp_folder.exists():
                import shutil
                shutil.rmtree(temp_folder)
                print("🧹 Cleaned up temp folder")
            
            # Verify the result
            print(f"\n✅ ReverieServer created successfully!")
            print(f"👥 Agents loaded: {len(reverie_server.personas)}")
            
            agent_names = list(reverie_server.personas.keys())
            for i, name in enumerate(agent_names, 1):
                print(f"  {i:2d}. {name}")
            
            return True
        else:
            print("❌ Simulation folder does not exist")
            return False
            
    except Exception as e:
        print(f"💥 Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reverie_initialization()
    if success:
        print("\n🎉 Test PASSED: Reverie simulation initialized successfully!")
    else:
        print("\n❌ Test FAILED: Could not initialize Reverie simulation")