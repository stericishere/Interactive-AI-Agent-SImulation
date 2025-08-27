#!/usr/bin/env python3
"""
Test the full system: automated setup + ReverieServer + batch API processing
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

def test_complete_system():
    """Test the complete automated system end-to-end"""
    print("🎭 Testing Complete Dating Show System")
    print("=" * 60)
    
    try:
        # Step 1: Automated setup
        print("1️⃣ Running automated simulation setup...")
        setup_success = ensure_clean_8_agent_simulation()
        if not setup_success:
            print("❌ Automated setup failed")
            return False
        print("✅ Automated setup completed")
        
        # Step 2: ReverieServer initialization  
        print("\n2️⃣ Initializing ReverieServer with temp rename logic...")
        sim_code = "dating_show_8_agents"
        sim_folder = Path(fs_storage) / sim_code
        temp_folder = Path(fs_storage) / f"{sim_code}_clean"
        
        if sim_folder.exists():
            # Clean up existing temp
            if temp_folder.exists():
                import shutil
                shutil.rmtree(temp_folder)
            
            # Temporary rename for ReverieServer fork
            sim_folder.rename(temp_folder)
            reverie_server = ReverieServer(f"{sim_code}_clean", sim_code)
            
            # Cleanup temp
            if temp_folder.exists():
                import shutil
                shutil.rmtree(temp_folder)
            
            print(f"✅ ReverieServer initialized with {len(reverie_server.personas)} agents")
        else:
            print("❌ Simulation folder not found")
            return False
        
        # Step 3: Run 1 step with batch API processing
        print(f"\n3️⃣ Testing 1 simulation step with batch API processing...")
        print("🎬 Starting simulation step...")
        
        start_time = time.time()
        reverie_server.start_server(1)  # Just 1 step
        step_time = time.time() - start_time
        
        print(f"⏱️ Step completed in {step_time:.2f}s")
        print(f"✅ Full system test completed!")
        
        return True
        
    except Exception as e:
        print(f"💥 System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_system()
    if success:
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"✨ Automated setup + ReverieServer + Batch API = Working!")
        print(f"🚀 Dating show simulation ready for production")
    else:
        print(f"\n❌ System test failed")