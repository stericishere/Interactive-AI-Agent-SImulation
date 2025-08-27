#!/usr/bin/env python3
"""
Integration Example: Using SimulationSetupService with Main Simulation

This example demonstrates how to integrate the automated 8-agent simulation 
setup service with the main dating show simulation initialization.

This ensures that your simulation always starts with a clean, validated 
8-agent configuration before proceeding with the main simulation logic.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dating_show.services import (
    SimulationSetupService,
    create_dating_show_simulation,
    validate_dating_show_simulation,
    get_dating_show_status
)


def ensure_clean_8_agent_simulation(force_recreate=False, max_repair_attempts=3):
    """
    Ensure a clean 8-agent simulation is available before starting main simulation.
    
    This function:
    1. Checks if a valid 8-agent simulation exists
    2. Creates one if missing or invalid
    3. Attempts repairs if validation fails
    4. Provides detailed status feedback
    
    Args:
        force_recreate: If True, recreate simulation even if valid one exists
        max_repair_attempts: Maximum number of repair attempts before giving up
        
    Returns:
        bool: True if clean simulation is ready, False if setup failed
        
    Raises:
        Exception: If simulation cannot be created or repaired after max attempts
    """
    
    print("🔍 Checking 8-agent simulation status...")
    
    # Check current status
    status = get_dating_show_status()
    
    if status["exists"] and status["is_valid"] and not force_recreate:
        print("✅ Valid 8-agent simulation already exists")
        print(f"   📅 Last modified: {status['last_modified']}")
        print(f"   👥 Agents: {status['agent_count']}/8")
        return True
        
    if force_recreate:
        print("🔄 Force recreating simulation...")
    elif not status["exists"]:
        print("📁 No simulation found, creating new one...")
    elif not status["is_valid"]:
        print("⚠️  Invalid simulation found, attempting repair first...")
        
        # Try repairs before recreating
        service = SimulationSetupService()
        repair_attempts = 0
        
        while repair_attempts < max_repair_attempts:
            repair_attempts += 1
            print(f"🔧 Repair attempt {repair_attempts}/{max_repair_attempts}...")
            
            repair_result = service.repair_simulation()
            
            if repair_result["success"]:
                print("✅ Simulation successfully repaired!")
                
                # Validate after repair
                validation = validate_dating_show_simulation()
                if validation["is_valid"]:
                    print("✅ Post-repair validation passed")
                    return True
                else:
                    print("❌ Post-repair validation failed, trying again...")
                    continue
            else:
                print(f"❌ Repair attempt {repair_attempts} failed:")
                for error in repair_result["errors"]:
                    print(f"   • {error}")
                    
        print(f"⚠️  All {max_repair_attempts} repair attempts failed, recreating simulation...")
        
    # Create new simulation
    print("🆕 Creating fresh 8-agent simulation...")
    result = create_dating_show_simulation(force_recreate=True)
    
    if result["success"]:
        print("✅ Successfully created 8-agent simulation!")
        print(f"   👥 Agents: {len(result['agents_created'])}")
        print(f"   📄 Files processed: {len(result['files_processed'])}")
        
        # Final validation
        print("🔍 Running post-creation validation...")
        validation = validate_dating_show_simulation()
        
        if validation["is_valid"]:
            print("✅ Final validation passed - simulation ready!")
            return True
        else:
            print("❌ Final validation failed:")
            for error in validation["errors"]:
                print(f"   • {error}")
            raise Exception("Simulation creation succeeded but validation failed")
            
    else:
        print("❌ Failed to create simulation:")
        for error in result["errors"]:
            print(f"   • {error}")
        raise Exception("Could not create valid 8-agent simulation")


def initialize_dating_show_simulation(simulation_name="dating_show_8_agents", 
                                    ensure_clean=True,
                                    force_recreate=False):
    """
    Initialize the dating show simulation with proper 8-agent setup.
    
    Args:
        simulation_name: Name of the simulation to initialize
        ensure_clean: Whether to ensure a clean 8-agent setup first
        force_recreate: Whether to force recreation of the simulation setup
        
    Returns:
        dict: Initialization result with status and simulation object
    """
    
    result = {
        "success": False,
        "simulation": None,
        "message": "",
        "setup_performed": False
    }
    
    try:
        if ensure_clean:
            print("=== 8-Agent Simulation Setup Phase ===")
            setup_success = ensure_clean_8_agent_simulation(force_recreate=force_recreate)
            
            if not setup_success:
                result["message"] = "Failed to ensure clean 8-agent simulation setup"
                return result
                
            result["setup_performed"] = True
            print()
            
        print("=== Main Simulation Initialization Phase ===")
        print(f"🚀 Initializing {simulation_name} simulation...")
        
        # Here you would integrate with your main simulation class
        # For now, we'll just show the integration pattern
        
        # Example integration (uncomment and modify for your actual simulation):
        # from dating_show.simulation import DatingShowSimulation
        # simulation = DatingShowSimulation(simulation_name)
        # result["simulation"] = simulation
        
        # For this example, we'll just show successful setup
        result["simulation"] = f"Mock simulation for {simulation_name}"
        result["success"] = True
        result["message"] = "Simulation initialized successfully with clean 8-agent setup"
        
        print("✅ Dating show simulation ready to run!")
        
        # Show final status
        final_status = get_dating_show_status()
        print(f"\n📊 Final Status:")
        print(f"   👥 Agents: {final_status['agent_count']}/8")
        print(f"   🔍 Valid: {'✅' if final_status['is_valid'] else '❌'}")
        
    except Exception as e:
        result["message"] = f"Initialization failed: {str(e)}"
        print(f"\n🚨 Initialization error: {str(e)}")
        
    return result


def main():
    """
    Example usage of the integrated simulation setup.
    """
    
    print("=== Dating Show Simulation Integration Example ===")
    print()
    
    try:
        # Method 1: Quick status check
        print("Method 1: Quick Status Check")
        print("-" * 30)
        status = get_dating_show_status()
        print(f"Current simulation status: {'✅ Ready' if status['is_valid'] else '❌ Needs Setup'}")
        print()
        
        # Method 2: Ensure clean setup then initialize
        print("Method 2: Integrated Initialization")
        print("-" * 30)
        result = initialize_dating_show_simulation(
            ensure_clean=True,
            force_recreate=False  # Set to True to force recreation
        )
        
        if result["success"]:
            print(f"\n🎉 Success! {result['message']}")
            if result["setup_performed"]:
                print("   📋 Clean 8-agent setup was ensured")
            print(f"   🎮 Simulation object: {result['simulation']}")
        else:
            print(f"\n❌ Failed: {result['message']}")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n🚨 Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()