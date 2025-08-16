"""
Dating Show Main Entry Point
Complete integration between PIANO agents and Django frontend with enterprise-grade orchestration
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json
import django

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "reverie" / "backend_server"))

# Set up Django environment for the frontend server (only if needed)
def setup_django():
    """Setup Django environment if reverie components are needed"""
    try:
        # Add the frontend server path to Python path
        frontend_server_path = project_root / "environment" / "frontend_server"
        if frontend_server_path.exists():
            sys.path.insert(0, str(frontend_server_path))
            
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'environment.frontend_server.frontend_server.settings')
        django.setup()
        return True
    except Exception as e:
        logger.warning(f"Django setup failed: {e}")
        return False

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to setup Django, but don't fail if it doesn't work
DJANGO_AVAILABLE = setup_django()

from dating_show.services.orchestrator import create_orchestrator, OrchestrationConfig
from dating_show.services.database_service import DatabaseService
from dating_show.services.enhanced_bridge import EnhancedFrontendBridge

# Import reverie components for 25-agent simulation
try:
    # Create a minimal utils.py if it doesn't exist
    utils_path = project_root / "reverie" / "backend_server" / "utils.py"
    if not utils_path.exists():
        logger.info("Creating minimal utils.py for reverie backend")
        with open(utils_path, 'w') as f:
            f.write('"""Minimal utils module for reverie backend"""\n')
            f.write('# Placeholder utils file\n')
    
    from reverie.backend_server.reverie import ReverieServer
    from reverie.backend_server.global_methods import fs_storage
    REVERIE_AVAILABLE = True
    logger.info("Reverie backend successfully imported")
except ImportError as e:
    logger.warning(f"Reverie backend not available: {e}")
    REVERIE_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error setting up reverie: {e}")
    REVERIE_AVAILABLE = False


class ReverieIntegrationManager:
    """Manages integration between Reverie 25-agent simulation and dating show frontend"""
    
    def __init__(self):
        self.reverie_server = None
        self.fs_temp_storage = None
        
    def create_temp_storage_files(self, sim_code: str, step: int = 1):
        """Create required JSON files in temp_storage following reverie.py pattern"""
        if not (REVERIE_AVAILABLE and DJANGO_AVAILABLE):
            logger.warning("Reverie or Django not available, skipping temp storage file creation")
            return
            
        # Create temp_storage directory if it doesn't exist  
        self.fs_temp_storage = f"{fs_storage}/../temp_storage"
        os.makedirs(self.fs_temp_storage, exist_ok=True)
        
        # Create curr_sim_code.json (persistent - not deleted by frontend)
        curr_sim_code = {"sim_code": sim_code}
        with open(f"{self.fs_temp_storage}/curr_sim_code.json", "w") as outfile: 
            outfile.write(json.dumps(curr_sim_code, indent=2))
        
        # Create curr_step.json (will be deleted by frontend after reading)
        curr_step = {"step": step}
        with open(f"{self.fs_temp_storage}/curr_step.json", "w") as outfile: 
            outfile.write(json.dumps(curr_step, indent=2))
        
        logger.info(f"Created temp_storage files for simulation: {sim_code}, step: {step}")
        
    def update_curr_step_json(self, step: int):
        """Update curr_step.json after each simulation step"""
        if not self.fs_temp_storage:
            return
            
        curr_step = {"step": step}
        with open(f"{self.fs_temp_storage}/curr_step.json", "w") as outfile: 
            outfile.write(json.dumps(curr_step, indent=2))
        logger.info(f"Frontend sync: Updated curr_step.json to step {step}")
        
    def initialize_reverie_simulation(self, fork_sim_code: str = "base_the_ville_n25", 
                                     sim_code: str = "dating_show_25_agents"):
        """Initialize the 25-agent Smallville simulation"""
        if not (REVERIE_AVAILABLE and DJANGO_AVAILABLE):
            logger.error("Reverie or Django not available, cannot initialize 25-agent simulation")
            return None
            
        try:
            # Create the required JSON files for frontend communication
            self.create_temp_storage_files(sim_code, step=1)
            
            # Create reverie server instance
            logger.info(f"Creating ReverieServer: forking from '{fork_sim_code}' -> '{sim_code}'")
            self.reverie_server = ReverieServer(fork_sim_code, sim_code)
            
            logger.info("ReverieServer initialized successfully!")
            logger.info(f"Simulation contains {len(self.reverie_server.personas)} agents:")
            for i, persona_name in enumerate(self.reverie_server.personas.keys(), 1):
                logger.info(f"  {i:2d}. {persona_name}")
            
            return self.reverie_server
            
        except Exception as e:
            logger.error(f"Failed to initialize Reverie simulation: {e}")
            return None
            
    def get_agent_list(self):
        """Get list of agents from reverie simulation"""
        if not self.reverie_server:
            return []
            
        agents = []
        for i, (persona_name, persona) in enumerate(self.reverie_server.personas.items()):
            agent_data = {
                'agent_id': f'agent_{i:03d}',
                'name': persona_name,
                'current_role': 'contestant',  # All are contestants in dating show context
                'location': {
                    'area': 'the_ville', 
                    'room': 'unknown',
                    'x': self.reverie_server.personas_tile[persona_name][0],
                    'y': self.reverie_server.personas_tile[persona_name][1]
                },
                'current_action': getattr(persona.scratch, 'daily_plan_req', 'socializing'),
                'memory': {'recent_events': [], 'long_term': {}},
                'reverie_persona': persona  # Store reference to original persona
            }
            agents.append(agent_data)
            
        return agents


class DatingShowMain:
    """
    Main application entry point for the dating show simulation.
    
    Provides command-line interface and orchestrates the complete system:
    - Database service initialization
    - Enhanced frontend bridge
    - PIANO agent registration
    - Simulation execution with real-time frontend updates
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize main application"""
        self.config_path = config_path
        self.config = self.load_configuration(config_path)
        self.orchestrator = None
        self.shutdown_event = asyncio.Event()
        
        # Initialize reverie integration manager
        self.reverie_manager = ReverieIntegrationManager()
        self.reverie_server = None
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        logger.info("Dating Show application initialized")
        logger.info(f"Django available: {DJANGO_AVAILABLE}")
        logger.info(f"Reverie available: {REVERIE_AVAILABLE}")
        if REVERIE_AVAILABLE and DJANGO_AVAILABLE:
            logger.info("Will use 25-agent Smallville simulation")
        else:
            logger.info("Will use mock agents for testing")
    
    def load_configuration(self, config_path: Optional[str]) -> OrchestrationConfig:
        """Load configuration from file or use defaults"""
        try:
            if config_path and os.path.exists(config_path):
                logger.info(f"Loading configuration from {config_path}")
                return OrchestrationConfig.from_file(config_path)
            else:
                logger.info("Using default configuration")
                default_config = OrchestrationConfig()
                
                # Auto-detect paths relative to project structure
                # Handle both running from project root and dating_show directory
                current_file = Path(__file__).resolve()
                
                # If running from dating_show directory, parent.parent gets project root
                # If running from project root, we need to detect this
                if current_file.parent.name == "dating_show":
                    project_root = current_file.parent.parent
                else:
                    project_root = current_file.parent
                
                # Try to find the frontend server directory
                frontend_paths = [
                    project_root / "environment" / "frontend_server",
                    Path.cwd() / "environment" / "frontend_server",
                    Path.cwd().parent / "environment" / "frontend_server"
                ]
                
                frontend_server_path = None
                for path in frontend_paths:
                    if path.exists():
                        frontend_server_path = str(path)
                        break
                
                if frontend_server_path:
                    default_config.frontend_server_path = frontend_server_path
                    default_config.piano_config_path = str(project_root / "dating_show" / "config")
                    logger.info(f"Frontend server path verified: {frontend_server_path}")
                else:
                    logger.error("Frontend server path not found in any expected location")
                    logger.info("Please ensure the Django frontend server exists at environment/frontend_server/")
                    # Set a default path anyway
                    default_config.frontend_server_path = str(project_root / "environment" / "frontend_server")
                    default_config.piano_config_path = str(project_root / "dating_show" / "config")
                
                return default_config
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Falling back to default configuration")
            return OrchestrationConfig()
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self) -> None:
        """Execute the complete dating show with frontend integration"""
        try:
            logger.info("Starting Dating Show application...")
            
            # Create orchestrator
            self.orchestrator = await create_orchestrator(
                config_path=self.config_path,
                **self.config.__dict__
            )
            
            # Phase 1: Initialize database
            logger.info("Phase 1: Initializing database service...")
            await self.orchestrator.initialize_database()
            
            # Phase 2: Start frontend bridge
            logger.info("Phase 2: Starting frontend bridge...")
            await self.orchestrator.start_frontend_bridge()
            
            # Phase 3: Load and register PIANO agents
            logger.info("Phase 3: Loading PIANO agents...")
            agents = await self.load_piano_agents()
            await self.orchestrator.register_piano_agents(agents)
            
            # Phase 4: Start simulation loop
            logger.info("Phase 4: Starting simulation...")
            await self.run_simulation()
            
            logger.info("Dating Show application completed successfully")
            
        except Exception as e:
            logger.error(f"Application failed: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def run_with_auto_steps(self, steps: int) -> None:
        """Run application with automatic simulation steps"""
        try:
            logger.info("Starting Dating Show with auto-run simulation...")
            
            # Phase 1: Initialize database
            logger.info("Phase 1: Initializing database service...")
            self.orchestrator = await create_orchestrator(
                config_path=self.config_path,
                **self.config.__dict__
            )
            await self.orchestrator.initialize_database()
            
            # Phase 2: Start frontend bridge
            logger.info("Phase 2: Starting frontend bridge...")
            await self.orchestrator.start_frontend_bridge()
            
            # Phase 3: Load and register agents (25-agent Smallville)
            logger.info("Phase 3: Loading Smallville agents...")
            agents = await self.load_piano_agents()
            await self.orchestrator.register_piano_agents(agents)
            
            # Phase 4: Auto-run simulation steps
            logger.info(f"Phase 4: Auto-running {steps} simulation steps...")
            await self.auto_run_simulation(steps)
            
            # Phase 5: Keep running for frontend interaction
            logger.info("Phase 5: Keeping simulation alive for frontend...")
            logger.info(f"Frontend URL: {self.config.frontend_url}/simulator_home")
            logger.info("Press Ctrl+C to stop")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            logger.info("Dating Show auto-run completed successfully")
            
        except Exception as e:
            logger.error(f"Auto-run failed: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def load_piano_agents(self) -> list:
        """Load agents for the dating show simulation - use 25-agent Smallville if available"""
        try:
            if REVERIE_AVAILABLE and DJANGO_AVAILABLE:
                # Initialize the 25-agent Smallville simulation
                logger.info("Loading 25-agent Smallville simulation...")
                self.reverie_server = self.reverie_manager.initialize_reverie_simulation()
                
                if self.reverie_server:
                    # Get real agents from reverie simulation
                    agents = self.reverie_manager.get_agent_list()
                    logger.info(f"Loaded {len(agents)} real Smallville agents")
                    return agents
                else:
                    logger.warning("Failed to initialize Reverie, falling back to mock agents")
            
            # Fallback to mock agents if reverie not available
            logger.info("Creating mock agents for testing...")
            mock_agents = []
            
            # Create mock dating show agents with names from Smallville if possible
            agent_names = [
                "Abigail Chen", "Adam Smith", "Arthur Burton", "Ayesha Khan", "Carlos Gomez",
                "Carmen Ortiz", "Eddy Lin", "Francisco Lopez", "Giorgio Rossi", "Hailey Johnson",
                "Isabella Rodriguez", "Jane Moreno", "Jennifer Moore", "John Lin", "Klaus Mueller",
                "Latoya Williams", "Maria Lopez", "Mei Lin", "Rajiv Patel", "Ryan Park",
                "Sam Moore", "Tamara Taylor", "Tom Moreno", "Wolfgang Schulz", "Yuriko Yamamoto"
            ]
            
            for i, name in enumerate(agent_names[:self.config.max_agents]):
                mock_agent = type('MockAgent', (), {
                    'agent_id': f'agent_{i:03d}',
                    'name': name,
                    'current_role': 'contestant',
                    'location': {'area': 'the_ville', 'room': 'main_area', 'x': 50, 'y': 50},
                    'current_action': 'socializing',
                    'memory': {'recent_events': [], 'long_term': {}},
                    'specialization': {
                        'type': ['social', 'creative', 'analytical'][i % 3],
                        'level': 'intermediate'
                    }
                })()
                
                mock_agents.append(mock_agent)
            
            logger.info(f"Loaded {len(mock_agents)} mock agents")
            return mock_agents
            
        except Exception as e:
            logger.error(f"Failed to load agents: {e}")
            raise
    
    async def run_simulation_step(self):
        """Run a single simulation step and update JSON files"""
        if self.reverie_server and self.reverie_manager:
            # Get current step before increment
            old_step = self.reverie_server.step
            
            # This is where you'd run the actual reverie step logic
            # For now, just increment step as placeholder
            self.reverie_server.step += 1
            
            # Update the JSON file for frontend synchronization
            self.reverie_manager.update_curr_step_json(self.reverie_server.step)
            
            logger.info(f"Simulation step: {old_step} -> {self.reverie_server.step}")
            return True
        return False
        
    async def auto_run_simulation(self, steps: int = 10):
        """Auto-run simulation for specified number of steps"""
        logger.info(f"Auto-running simulation for {steps} steps...")
        
        for i in range(steps):
            success = await self.run_simulation_step()
            if not success:
                logger.error(f"Simulation step failed at step {i+1}")
                break
                
            # Small delay between steps
            await asyncio.sleep(0.5)
            
        logger.info(f"Auto-run completed. Current step: {self.reverie_server.step if self.reverie_server else 'unknown'}")
    
    async def run_simulation(self) -> None:
        """Run the main simulation with monitoring"""
        try:
            # Start simulation loop in background
            simulation_task = asyncio.create_task(
                self.orchestrator.start_simulation_loop()
            )
            
            # Monitor health and status
            monitor_task = asyncio.create_task(
                self.monitor_health()
            )
            
            # Create shutdown task
            shutdown_task = asyncio.create_task(self._wait_for_shutdown())
            
            # Wait for shutdown signal or simulation completion
            done, pending = await asyncio.wait(
                [simulation_task, monitor_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    
    async def monitor_health(self) -> None:
        """Monitor system health and log status"""
        try:
            while not self.shutdown_event.is_set():
                if self.orchestrator:
                    status = self.orchestrator.get_orchestrator_status()
                    
                    # Log periodic status updates
                    logger.info(f"System Status: {status['status']}, "
                               f"Uptime: {status['uptime_seconds']:.0f}s, "
                               f"Agents: {status['services']['piano_agents']}")
                    
                    # Check for errors
                    if status['service_health']['error_message']:
                        logger.warning(f"System warning: {status['service_health']['error_message']}")
                
                await asyncio.sleep(30)  # Log every 30 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
    
    async def _wait_for_shutdown(self) -> None:
        """Wait for shutdown signal"""
        await self.shutdown_event.wait()
        logger.info("Shutdown signal received")
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.orchestrator:
                await self.orchestrator.handle_shutdown()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command-line arguments"""
        parser = argparse.ArgumentParser(
            description="Dating Show - PIANO Agent Simulation with Frontend Integration",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python dating_show/main.py                    # Run with defaults
  python dating_show/main.py --config config.json   # Use custom config
  python dating_show/main.py --agents 25 --steps 500  # Override parameters
  python dating_show/main.py --debug             # Enable debug logging
            """
        )
        
        parser.add_argument(
            '--config', '-c',
            type=str,
            help='Path to configuration file (JSON)'
        )
        
        parser.add_argument(
            '--agents', '-a',
            type=int,
            default=25,
            help='Maximum number of agents (default: 25)'
        )
        
        parser.add_argument(
            '--steps', '-s',
            type=int,
            default=1000,
            help='Number of simulation steps (default: 1000)'
        )
        
        parser.add_argument(
            '--frontend-url',
            type=str,
            default='http://localhost:8001',
            help='Frontend server URL (default: http://localhost:8000)'
        )
        
        parser.add_argument(
            '--database-url',
            type=str,
            help='Database connection URL'
        )
        
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug logging'
        )
        
        parser.add_argument(
            '--save-config',
            type=str,
            help='Save current configuration to file'
        )
        
        parser.add_argument(
            '--status',
            action='store_true',
            help='Show system status and exit'
        )
        
        parser.add_argument(
            '--run-steps',
            type=int,
            help='Auto-run simulation for specified number of steps'
        )
        
        parser.add_argument(
            '--use-reverie',
            action='store_true',
            default=True,
            help='Use 25-agent Smallville simulation (default: True)'
        )
        
        return parser.parse_args()
    
    def apply_arguments(self, args: argparse.Namespace) -> None:
        """Apply command-line arguments to configuration"""
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            self.config.log_level = "DEBUG"
        
        if args.agents:
            self.config.max_agents = args.agents
        
        if args.steps:
            self.config.simulation_steps = args.steps
        
        if args.frontend_url:
            self.config.frontend_url = args.frontend_url
        
        if args.database_url:
            self.config.database_url = args.database_url
        
        if args.save_config:
            self.config.to_file(args.save_config)
            logger.info(f"Configuration saved to {args.save_config}")


async def main():
    """Main entry point"""
    try:
        app = DatingShowMain()
        args = app.parse_arguments()
        
        if args.config:
            app.config_path = args.config
            app.config = app.load_configuration(args.config)
        
        app.apply_arguments(args)
        
        if args.status:
            # Show status and exit
            print(json.dumps(app.config.__dict__, indent=2, default=str))
            return
        
        # Check if we should auto-run simulation steps
        if args.run_steps:
            await app.run_with_auto_steps(args.run_steps)
        else:
            # Run the application normally
            await app.run()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())