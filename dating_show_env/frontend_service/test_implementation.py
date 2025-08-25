#!/usr/bin/env python3
"""
Test Implementation Script
Validates the frontend logic implementation and missing file fixes
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.startup_initializer import StartupInitializer
from core.movement_generator import MovementFileGenerator
from core.config import Settings

async def test_implementation():
    """Test the complete implementation"""
    print("🧪 Testing Frontend Logic Implementation")
    print("=" * 50)
    
    # Initialize components
    settings = Settings()
    initializer = StartupInitializer(settings)
    
    sim_code = "dating_show_25_agents"
    
    try:
        # Test 1: Storage initialization
        print("\n📁 Test 1: Storage Initialization")
        success = await initializer.initialize_all(sim_code)
        if success:
            print("✅ Storage initialization: PASSED")
        else:
            print("⚠️  Storage initialization: PARTIAL")
            
        # Test 2: Movement file generation
        print("\n🎯 Test 2: Movement File Generation")
        missing_files = initializer.movement_generator.get_missing_movement_files(sim_code)
        print(f"Missing files before generation: {len(missing_files)}")
        
        if missing_files:
            gen_success = await initializer.movement_generator.generate_missing_files_for_simulation(sim_code)
            if gen_success:
                print("✅ Movement file generation: PASSED")
            else:
                print("❌ Movement file generation: FAILED")
        else:
            print("✅ All movement files already exist: PASSED")
            
        # Test 3: File validation
        print("\n🔍 Test 3: File Validation")
        validation_passed = 0
        validation_total = 0
        
        # Check movement file 0
        validation_total += 1
        if await initializer.movement_generator.validate_movement_file(sim_code, 0):
            validation_passed += 1
            print("✅ Movement file 0: VALID")
        else:
            print("❌ Movement file 0: INVALID")
            
        # Check storage structure
        storage_path = Path(initializer.storage_path) / sim_code
        required_dirs = ["movement", "environment", "personas", "reverie"]
        
        for dir_name in required_dirs:
            validation_total += 1
            if (storage_path / dir_name).exists():
                validation_passed += 1
                print(f"✅ {dir_name} directory: EXISTS")
            else:
                print(f"❌ {dir_name} directory: MISSING")
        
        # Test 4: Demo data creation
        print("\n🎭 Test 4: Demo Data Creation")
        demo_success = await initializer.initialize_with_demo_data(f"{sim_code}_demo")
        if demo_success:
            print("✅ Demo data creation: PASSED")
        else:
            print("❌ Demo data creation: FAILED")
        
        # Summary
        print("\n📊 Test Summary")
        print("=" * 30)
        print(f"Validation checks: {validation_passed}/{validation_total} passed")
        print(f"Success rate: {(validation_passed/validation_total)*100:.1f}%")
        
        if validation_passed >= validation_total * 0.8:  # 80% success rate
            print("🎉 OVERALL RESULT: IMPLEMENTATION SUCCESSFUL")
            return True
        else:
            print("⚠️  OVERALL RESULT: IMPLEMENTATION NEEDS ATTENTION")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def check_specific_files():
    """Check specific files that caused the original error"""
    print("\n🔎 Checking Specific Problem Files")
    print("=" * 40)
    
    problem_file = "/Applications/Projects/Open source/generative_agents/environment/frontend_server/storage/dating_show_25_agents/movement/0.json"
    
    if Path(problem_file).exists():
        print(f"✅ Problem file now exists: {problem_file}")
        
        # Validate content
        try:
            with open(problem_file, 'r') as f:
                data = json.load(f)
                if "persona" in data and "meta" in data:
                    agent_count = len(data["persona"])
                    print(f"✅ File structure valid with {agent_count} agents")
                else:
                    print("❌ File structure invalid")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    else:
        print(f"❌ Problem file still missing: {problem_file}")
    
    # Check movement directory
    movement_dir = "/Applications/Projects/Open source/generative_agents/environment/frontend_server/storage/dating_show_25_agents/movement"
    if Path(movement_dir).exists():
        files = list(Path(movement_dir).glob("*.json"))
        print(f"✅ Movement directory exists with {len(files)} files")
    else:
        print(f"❌ Movement directory missing: {movement_dir}")

def print_usage_instructions():
    """Print usage instructions for the implementation"""
    print("\n📖 Usage Instructions")
    print("=" * 30)
    print("1. Start the frontend service:")
    print("   cd dating_show_env/frontend_service")
    print("   python main.py")
    print()
    print("2. Check movement files status:")
    print("   curl http://localhost:8001/health/files/movement/dating_show_25_agents")
    print()
    print("3. Fix missing movement files (if needed):")
    print("   curl -X POST http://localhost:8001/admin/fix/movement/dating_show_25_agents")
    print()
    print("4. View the dashboard:")
    print("   http://localhost:8001/dashboard")
    print()
    print("5. View simulation home:")
    print("   http://localhost:8001/simulator_home")

if __name__ == "__main__":
    print("🚀 Frontend Logic Implementation Test")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Run tests
    success = asyncio.run(test_implementation())
    asyncio.run(check_specific_files())
    
    # Print usage instructions
    print_usage_instructions()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)