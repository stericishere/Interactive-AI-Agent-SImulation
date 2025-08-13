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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dating_show.services.orchestrator import create_orchestrator, OrchestrationConfig
from dating_show.services.database_service import DatabaseService
from dating_show.services.enhanced_bridge import EnhancedFrontendBridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        
        # Setup signal handlers
        self.setup_signal_handlers()
        
        logger.info("Dating Show application initialized")
    
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
                log_level="INFO"
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
    
    async def load_piano_agents(self) -> list:
        """Load PIANO agents for the dating show simulation"""
        try:
            # This would load actual PIANO agents
            # For now, we'll create mock agents for testing
            mock_agents = []
            
            # Create mock dating show agents
            agent_names = [
                "Alex", "Bailey", "Casey", "Drew", "Emery", "Finley", 
                "Gray", "Harper", "Indigo", "Jordan", "Kelly", "Lane",
                "Morgan", "Noel", "Oakley", "Parker", "Quinn", "River",
                "Sage", "Taylor", "Unity", "Vale", "Winter", "Xen", "Yuki"
            ]
            
            for i, name in enumerate(agent_names[:self.config.max_agents]):
                mock_agent = type('MockAgent', (), {
                    'agent_id': f'agent_{i:03d}',
                    'name': name,
                    'current_role': 'contestant' if i < 20 else 'host' if i < 22 else 'producer',
                    'specialization': {
                        'type': ['social', 'creative', 'analytical'][i % 3],
                        'level': 'intermediate'
                    },
                    'skills': {
                        'social': {'level': 50 + (i % 50), 'experience': 100},
                        'creative': {'level': 30 + (i % 40), 'experience': 80},
                        'analytical': {'level': 40 + (i % 35), 'experience': 60}
                    },
                    'memory': {'recent_events': [], 'long_term': {}},
                    'location': {'area': 'villa', 'room': 'main_area'},
                    'current_action': 'socializing'
                })()
                
                mock_agents.append(mock_agent)
            
            logger.info(f"Loaded {len(mock_agents)} PIANO agents")
            return mock_agents
            
        except Exception as e:
            logger.error(f"Failed to load PIANO agents: {e}")
            raise
    
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
            default='http://localhost:8000',
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
        
        # Run the application
        await app.run()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())