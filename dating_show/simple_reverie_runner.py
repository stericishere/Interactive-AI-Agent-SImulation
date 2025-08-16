#!/usr/bin/env python3
"""
Simplified Direct Reverie Runner
Runs 25-agent Smallville simulation with dating show frontend integration
Without complex database orchestration dependencies
"""

import sys
import os
import json
import time
import asyncio
from pathlib import Path
from typing import Optional

# Add paths for reverie imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "reverie" / "backend_server"))

# Set up minimal Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'environment.frontend_server.frontend_server.settings')

try:
    import django
    frontend_server_path = project_root / "environment" / "frontend_server"
    if frontend_server_path.exists():
        sys.path.insert(0, str(frontend_server_path))
    django.setup()
    DJANGO_AVAILABLE = True
except Exception as e:
    print(f"Django setup warning: {e}")
    DJANGO_AVAILABLE = False

# Import reverie components directly from backend_server directory
REVERIE_AVAILABLE = False
try:
    # Change to reverie backend directory for imports
    reverie_backend_path = str(project_root / "reverie" / "backend_server")
    if reverie_backend_path not in sys.path:
        sys.path.insert(0, reverie_backend_path)
    
    # Define required storage paths
    fs_storage = str(project_root / "environment" / "frontend_server" / "storage")
    fs_temp_storage = str(project_root / "environment" / "frontend_server" / "temp_storage")
    env_matrix = str(project_root / "environment" / "frontend_server" / "static_dirs" / "assets" / "the_ville" / "matrix")
    
    # Import required modules and inject paths
    import reverie
    import global_methods
    import maze
    
    # Inject paths into modules
    reverie.fs_storage = fs_storage
    reverie.fs_temp_storage = fs_temp_storage
    global_methods.fs_storage = fs_storage
    reverie.env_matrix = env_matrix
    global_methods.env_matrix = env_matrix  
    maze.env_matrix = env_matrix
    
    # Get the classes we need
    ReverieServer = reverie.ReverieServer
    
    REVERIE_AVAILABLE = True
    print("✅ Reverie backend successfully imported")
    print(f"📁 Storage path: {fs_storage}")
    
except Exception as e:
    print(f"❌ Reverie import failed: {e}")
    import traceback
    traceback.print_exc()
    REVERIE_AVAILABLE = False


class SimpleReverieRunner:
    """
    Simplified runner for 25-agent Smallville simulation
    Bypasses complex orchestration and focuses on core simulation + frontend sync
    """
    
    def __init__(self, fork_sim_code="base_the_ville_n25", sim_code="dating_show_25_agents"):
        self.fork_sim_code = fork_sim_code
        self.sim_code = sim_code
        self.reverie_server = None
        self.fs_temp_storage = None
        
        print(f"🎭 Simple Reverie Runner initialized")
        print(f"   Fork from: {fork_sim_code}")
        print(f"   New sim: {sim_code}")
    
    def create_temp_storage_files(self, step=1):
        """Create required JSON files for frontend communication"""
        if not DJANGO_AVAILABLE:
            print("⚠️  Django not available, skipping temp storage")
            return False
            
        try:
            # Create temp_storage directory
            self.fs_temp_storage = f"{fs_storage}/../temp_storage"
            os.makedirs(self.fs_temp_storage, exist_ok=True)
            
            # Create curr_sim_code.json (persistent)
            curr_sim_code = {"sim_code": self.sim_code}
            with open(f"{self.fs_temp_storage}/curr_sim_code.json", "w") as f:
                json.dump(curr_sim_code, f, indent=2)
            
            # Create curr_step.json (will be deleted by frontend)
            curr_step = {"step": step}
            with open(f"{self.fs_temp_storage}/curr_step.json", "w") as f:
                json.dump(curr_step, f, indent=2)
            
            print(f"📄 Created temp storage files for step {step}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating temp storage: {e}")
            return False
    
    def update_step_json(self, step):
        """Update curr_step.json for frontend synchronization"""
        if not self.fs_temp_storage:
            return False
            
        try:
            curr_step = {"step": step}
            with open(f"{self.fs_temp_storage}/curr_step.json", "w") as f:
                json.dump(curr_step, f, indent=2)
            print(f"🔄 Updated frontend sync to step {step}")
            return True
        except Exception as e:
            print(f"❌ Error updating step JSON: {e}")
            return False
    
    def initialize_simulation(self):
        """Initialize the 25-agent Smallville simulation"""
        if not REVERIE_AVAILABLE:
            print("❌ Reverie not available - cannot initialize simulation")
            return False
            
        try:
            print("🚀 Initializing 25-agent Smallville simulation...")
            
            # Create temp storage files
            if not self.create_temp_storage_files():
                print("⚠️  Temp storage creation failed, continuing anyway...")
            
            # Create reverie server
            print(f"🏗️  Creating ReverieServer...")
            self.reverie_server = ReverieServer(self.fork_sim_code, self.sim_code)
            
            print("✅ Reverie server created successfully!")
            print(f"👥 Loaded {len(self.reverie_server.personas)} agents:")
            
            # List all agents
            for i, persona_name in enumerate(self.reverie_server.personas.keys(), 1):
                print(f"   {i:2d}. {persona_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Simulation initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_simulation_step(self):
        """Run a single simulation step"""
        if not self.reverie_server:
            print("❌ No simulation server available")
            return False
            
        try:
            old_step = self.reverie_server.step
            
            # This is where we'd call the actual reverie step logic
            # For now, just increment step and update positions
            self.reverie_server.step += 1
            
            # Simulate agent movements (basic)
            for persona_name, persona in self.reverie_server.personas.items():
                # Call the persona's move method
                current_pos = self.reverie_server.personas_tile[persona_name]
                new_pos, emoji, desc = persona.move(
                    self.reverie_server.maze, 
                    self.reverie_server.personas,
                    current_pos,
                    self.reverie_server.curr_time
                )
                
                # Update position
                self.reverie_server.personas_tile[persona_name] = new_pos
            
            # Update frontend JSON
            self.update_step_json(self.reverie_server.step)
            
            print(f"⏭️  Simulation step: {old_step} → {self.reverie_server.step}")
            return True
            
        except Exception as e:
            print(f"❌ Simulation step failed: {e}")
            return False
    
    def run_auto_simulation(self, steps=10, delay=1.0):
        """Run simulation automatically for specified steps"""
        print(f"🎬 Starting auto-simulation for {steps} steps...")
        print(f"⏱️  Step delay: {delay}s")
        print("🌐 Frontend URL: http://localhost:8000/simulator_home")
        print("-" * 50)
        
        success_count = 0
        
        for i in range(steps):
            print(f"\n📍 Step {i+1}/{steps}")
            
            if self.run_simulation_step():
                success_count += 1
                print(f"✅ Step completed successfully")
            else:
                print(f"❌ Step failed")
                break
            
            # Delay between steps
            if i < steps - 1:  # Don't delay after last step
                time.sleep(delay)
        
        print(f"\n🏁 Auto-simulation completed!")
        print(f"📊 Success rate: {success_count}/{steps} steps")
        
        if self.reverie_server:
            current_step = self.reverie_server.step
            print(f"🎯 Final simulation step: {current_step}")
        
        return success_count == steps
    
    def run_interactive(self):
        """Run simulation in interactive mode"""
        print("🎮 Interactive Simulation Mode")
        print("Commands:")
        print("  step       - Run one simulation step")
        print("  auto N     - Auto-run N steps (default: 10)")
        print("  status     - Show current status")
        print("  save       - Save simulation state")
        print("  quit       - Exit simulation")
        print("-" * 50)
        
        while True:
            try:
                command = input("\n🎭 Dating Show> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    print("👋 Exiting simulation...")
                    break
                    
                elif command == 'step':
                    self.run_simulation_step()
                    
                elif command.startswith('auto'):
                    parts = command.split()
                    steps = int(parts[1]) if len(parts) > 1 else 10
                    self.run_auto_simulation(steps)
                    
                elif command == 'status':
                    if self.reverie_server:
                        print(f"📊 Simulation Status:")
                        print(f"   Current step: {self.reverie_server.step}")
                        print(f"   Agents: {len(self.reverie_server.personas)}")
                        print(f"   Simulation code: {self.sim_code}")
                    else:
                        print("❌ No simulation running")
                        
                elif command == 'save':
                    if self.reverie_server:
                        print("💾 Saving simulation state...")
                        self.reverie_server.save()
                        print("✅ Simulation saved!")
                    else:
                        print("❌ No simulation to save")
                        
                else:
                    print(f"❓ Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                break
            except ValueError:
                print("❌ Invalid number format")
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main entry point"""
    print("🎭 Simple Reverie Runner for Dating Show")
    print("=" * 50)
    
    # Check availability
    print(f"Django available: {'✅' if DJANGO_AVAILABLE else '❌'}")
    print(f"Reverie available: {'✅' if REVERIE_AVAILABLE else '❌'}")
    
    if not REVERIE_AVAILABLE:
        print("\n❌ Reverie backend not available!")
        print("Please ensure all dependencies are installed.")
        return 1
    
    # Create runner
    runner = SimpleReverieRunner()
    
    # Initialize simulation
    if not runner.initialize_simulation():
        print("❌ Failed to initialize simulation")
        return 1
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto':
            steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            runner.run_auto_simulation(steps)
        elif sys.argv[1] == '--step':
            runner.run_simulation_step()
        else:
            print(f"❓ Unknown argument: {sys.argv[1]}")
    else:
        # Interactive mode
        runner.run_interactive()
    
    return 0


if __name__ == "__main__":
    exit(main())