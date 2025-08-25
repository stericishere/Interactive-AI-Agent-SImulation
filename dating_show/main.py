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
import time
print(sys.path)
from pathlib import Path
from typing import Optional, Dict, Any
import json
import django
from datetime import datetime
import random

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
    print(f"🔑 [DEBUG] Loaded .env file, OPENROUTER_API_KEY present: {'OPENROUTER_API_KEY' in os.environ}")
except ImportError:
    print("⚠️ [DEBUG] python-dotenv not installed, .env file not loaded")
except Exception as e:
    print(f"⚠️ [DEBUG] Error loading .env file: {e}")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# IMPORTANT: Insert local reverie_core at the very beginning to override external reverie
reverie_core_path = str(Path(__file__).parent / "reverie_core")
if reverie_core_path in sys.path:
    sys.path.remove(reverie_core_path)
sys.path.insert(0, reverie_core_path)

# Add external reverie path last as fallback
external_reverie_path = str(project_root / "reverie" / "backend_server")
if external_reverie_path not in sys.path:
    sys.path.append(external_reverie_path)

# Set up Django environment for the frontend server (only if needed)
def setup_django():
    """Setup Django environment if reverie components are needed"""
    try:
        # Try dating_show_env first, then fallback to environment
        dating_show_frontend = project_root / "dating_show_env" / "frontend_service"
        original_frontend = project_root / "environment" / "frontend_server"
        
        if dating_show_frontend.exists():
            sys.path.insert(0, str(dating_show_frontend))
            # Note: dating_show_env may not need Django setup
            logger.info("Using dating_show_env frontend service")
            return True
        elif original_frontend.exists():
            sys.path.insert(0, str(original_frontend))
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'environment.frontend_server.frontend_server.settings')
            django.setup()
            return True
        
        return False
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

# Import dating show specific components
try:
    from dating_show.agents.prompt_template.dating_show_v1.prompts import PromptManager
    from dating_show.services.piano_integration import DatingShowReverieServer
    from dating_show.agents.enhanced_agent_state import EnhancedAgentStateManager, create_enhanced_agent_state
    DATING_SHOW_AGENTS_AVAILABLE = True
    logger.info("Dating show agent system imported successfully")
except ImportError as e:
    logger.warning(f"Dating show agents not available: {e}")
    DATING_SHOW_AGENTS_AVAILABLE = False

# Import reverie components for 25-agent simulation
# Try local reverie_core first, then fallback to external reverie
try:
    # Try local reverie_core first
    logger.info("Attempting to import local reverie_core...")
    logger.info(f"Current sys.path[0]: {sys.path[0]}")
    import global_methods
    logger.info(f"global_methods module location: {global_methods.__file__}")
    from global_methods import fs_storage, fs_temp_storage
    logger.info("✅ global_methods imported successfully")
    from reverie_core.reverie import ReverieServer
    logger.info("✅ ReverieServer imported successfully")
    REVERIE_AVAILABLE = True
    LOCAL_REVERIE = True
    logger.info("✅ Local reverie core successfully imported")
except ImportError as e:
    logger.warning(f"Local reverie import failed: {e}")
    try:
        # Fallback to external reverie
        logger.info("Attempting external reverie import...")
        utils_path = project_root / "reverie" / "backend_server" / "utils.py"
        if not utils_path.exists():
            logger.info("Creating minimal utils.py for reverie backend")
            with open(utils_path, 'w') as f:
                f.write('"""Minimal utils module for reverie backend"""\n')
                f.write('# Placeholder utils file\n')
        
        from reverie.backend_server.reverie import ReverieServer
        from reverie.backend_server.global_methods import fs_storage
        REVERIE_AVAILABLE = True
        LOCAL_REVERIE = False
        logger.info("External reverie backend successfully imported")
    except ImportError as e:
        logger.warning(f"External reverie backend not available: {e}")
        REVERIE_AVAILABLE = False
        LOCAL_REVERIE = False
except Exception as e:
    logger.warning(f"Error setting up reverie: {e}")
    import traceback
    logger.warning(f"Full traceback: {traceback.format_exc()}")
    REVERIE_AVAILABLE = False
    LOCAL_REVERIE = False


class ReverieIntegrationManager:
    """DEPRECATED: Legacy Reverie integration - use UnifiedAgentManager instead
    
    This class is kept for backward compatibility but should not be used in new code.
    The unified architecture provides superior data flow without lossy conversions.
    """
    
    def __init__(self):
        self.reverie_server = None
        self.fs_temp_storage = None
        
    def create_temp_storage_files(self, sim_code: str, step: int = 1):
        """Create required JSON files in temp_storage following reverie.py pattern"""
        if not REVERIE_AVAILABLE:
            logger.warning("Reverie not available, skipping temp storage file creation")
            return
            
        # Create temp_storage directory - prefer dating_show_env
        if LOCAL_REVERIE:
            # Use dating_show_env path for local reverie
            dating_show_storage = project_root / "dating_show_env" / "frontend_service" / "storage"
            self.fs_temp_storage = str(dating_show_storage.parent / "temp_storage")
        else:
            # Use original path for external reverie
            self.fs_temp_storage = f"{fs_storage}/../temp_storage"
            
        os.makedirs(self.fs_temp_storage, exist_ok=True)
        logger.info(f"📄 Created temp storage files for step {step}")
        
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
                                     sim_code: str = "dating_show_8_agents"):
        """Initialize the 8-agent dating show simulation"""
        if not REVERIE_AVAILABLE:
            logger.error("Reverie not available, cannot initialize 8-agent simulation")
            return None
            
        try:
            # Create the required JSON files for frontend communication
            self.create_temp_storage_files(sim_code, step=0)
            
            # Create reverie server instance
            logger.info(f"🏗️  Creating ReverieServer...")
            logger.info(f"Creating ReverieServer: forking from '{fork_sim_code}' -> '{sim_code}'")
            
            # Debug: Check if target simulation already exists
            sim_folder = Path(fs_storage) / sim_code
            print(f"🔍 [DEBUG] Checking if target simulation exists: {sim_folder}")
            print(f"🔍 [DEBUG] Exists: {sim_folder.exists()}")
            if sim_folder.exists():
                meta_path = sim_folder / "reverie" / "meta.json"
                print(f"🔍 [DEBUG] Meta file exists: {meta_path.exists()}")
                if meta_path.exists():
                    import json
                    with open(meta_path) as f:
                        meta_data = json.load(f)
                    print(f"🔍 [DEBUG] Persona count in meta.json: {len(meta_data.get('persona_names', []))}")
                    print(f"🔍 [DEBUG] Persona names: {meta_data.get('persona_names', [])}")
            
            # Use existing 8-agent simulation
            logger.info(f"🎯 [DEBUG] Using existing 8-agent simulation: {sim_code}")
            
            # Just load the existing simulation directly without forking
            sim_folder = Path(fs_storage) / sim_code
            if not sim_folder.exists():
                logger.error(f"8-agent simulation not found at: {sim_folder}")
                return None
                
            # Create ReverieServer by forking to same name (which will just load existing)
            # We need to temporarily rename to allow "forking" 
            temp_name = f"{sim_code}_temp"
            temp_folder = Path(fs_storage) / temp_name
            
            # Clean up any temp folder from previous runs
            if temp_folder.exists():
                import shutil
                shutil.rmtree(temp_folder)
            
            # Temporarily rename the target so ReverieServer can "fork" to it
            sim_folder.rename(temp_folder)
            
            # Now fork from temp back to original name
            self.reverie_server = ReverieServer(temp_name, sim_code)
            
            # Clean up temp folder
            if temp_folder.exists():
                import shutil
                shutil.rmtree(temp_folder)
            
            logger.info("ReverieServer initialized successfully!")
            logger.info(f"Simulation contains {len(self.reverie_server.personas)} agents:")
            for i, persona_name in enumerate(self.reverie_server.personas.keys(), 1):
                logger.info(f"  {i:2d}. {persona_name}")
            
            return self.reverie_server
            
        except Exception as e:
            logger.error(f"Failed to initialize Reverie simulation: {e}")
            return None
            
    def get_agent_list(self):
        """Get list of agents from reverie simulation with dating show roles"""
        if not self.reverie_server:
            return []
            
        agents = []
        total_personas = len(self.reverie_server.personas)
        
        # Dating show role assignment
        contestants_count = min(20, max(10, total_personas - 5))
        hosts_count = min(3, max(1, total_personas // 15))
        
        for i, (persona_name, persona) in enumerate(self.reverie_server.personas.items()):
            # Assign dating show roles
            if i < contestants_count:
                role = 'contestant'
                specialization = self._assign_contestant_specialization(i)
            elif i < contestants_count + hosts_count:
                role = 'host'
                specialization = {'type': 'social', 'level': 'expert'}
            else:
                role = 'producer'
                specialization = {'type': 'analytical', 'level': 'advanced'}
            
            agent_data = {
                'agent_id': f'agent_{i:03d}',
                'name': persona_name,
                'current_role': role,
                'specialization': specialization,
                'dating_show_traits': self._generate_dating_show_traits(persona_name, role),
                'location': {
                    'area': 'the_ville', 
                    'room': 'unknown',
                    'x': self.reverie_server.personas_tile[persona_name][0],
                    'y': self.reverie_server.personas_tile[persona_name][1]
                },
                'current_action': getattr(persona.scratch, 'daily_plan_req', 'socializing'),
                'memory': {'recent_events': [], 'long_term': {}},
                'reverie_persona': persona,  # Store reference to original persona
                'relationship_status': 'single' if role == 'contestant' else 'n/a',
                'roses_given': 0 if role == 'contestant' else None,
                'roses_received': 0 if role == 'contestant' else None
            }
            agents.append(agent_data)
            
        logger.info(f"Assigned {contestants_count} contestants, {hosts_count} hosts, "
                   f"{total_personas - contestants_count - hosts_count} producers")
        return agents
    
    def create_enhanced_agent_managers(self, main_app_instance):
        """Create enhanced agent state managers for all personas"""
        if not self.reverie_server:
            logger.warning("No reverie server available for creating enhanced managers")
            return
            
        if not DATING_SHOW_AGENTS_AVAILABLE:
            logger.warning("Enhanced agent state system not available")
            return
            
        logger.info("Creating enhanced agent state managers...")
        enhanced_managers = {}
        
        for i, (persona_name, persona) in enumerate(self.reverie_server.personas.items()):
            agent_id = f'agent_{i:03d}'
            
            # Extract personality traits from persona if available
            personality_traits = self._extract_personality_traits(persona)
            
            # Create enhanced agent state manager
            try:
                enhanced_manager = create_enhanced_agent_state(
                    agent_id=agent_id,
                    name=persona_name,
                    personality_traits=personality_traits
                )
                
                # Initialize with current persona state
                self._initialize_from_persona(enhanced_manager, persona, agent_id)
                
                enhanced_managers[agent_id] = enhanced_manager
                main_app_instance.agent_personalities[agent_id] = personality_traits
                
                logger.debug(f"Created enhanced state manager for {persona_name} ({agent_id})")
                
            except Exception as e:
                logger.error(f"Failed to create enhanced manager for {persona_name}: {e}")
                continue
        
        main_app_instance.enhanced_agent_managers = enhanced_managers
        logger.info(f"Created {len(enhanced_managers)} enhanced agent state managers")
        
    def _extract_personality_traits(self, persona) -> Dict[str, float]:
        """Extract personality traits from reverie persona"""
        # Default personality traits for dating show context
        default_traits = {
            "openness": 0.5,
            "conscientiousness": 0.5, 
            "extroversion": 0.6,
            "agreeableness": 0.6,
            "neuroticism": 0.3,
            "confidence": 0.5,
            "romantic_interest": 0.7,
            "competitiveness": 0.4
        }
        
        # Try to extract from persona if it has personality info
        if hasattr(persona, 'scratch') and hasattr(persona.scratch, 'personality'):
            persona_traits = getattr(persona.scratch, 'personality', {})
            if isinstance(persona_traits, dict):
                default_traits.update(persona_traits)
        
        # Normalize values to 0-1 range
        for trait, value in default_traits.items():
            if not isinstance(value, (int, float)):
                default_traits[trait] = 0.5
            else:
                default_traits[trait] = max(0.0, min(1.0, float(value)))
        
        return default_traits
    
    def _initialize_from_persona(self, enhanced_manager: EnhancedAgentStateManager, 
                                persona, agent_id: str):
        """Initialize enhanced manager with data from reverie persona"""
        try:
            # Set current location
            if persona_name := persona.name:
                if persona_name in self.reverie_server.personas_tile:
                    x, y = self.reverie_server.personas_tile[persona_name]
                    enhanced_manager.state["current_location"] = f"the_ville_{x}_{y}"
            
            # Set current activity from scratch
            if hasattr(persona, 'scratch'):
                activity = getattr(persona.scratch, 'daily_plan_req', 'socializing')
                enhanced_manager.state["current_activity"] = activity
                
                # Add any existing memories if available
                if hasattr(persona.scratch, 'memory'):
                    memories = getattr(persona.scratch, 'memory', [])
                    for memory in memories[:5]:  # Add recent memories
                        if isinstance(memory, str):
                            enhanced_manager.add_memory(memory, "imported", 0.5)
            
            # Set role-based specialization
            role_skills = {
                "contestant": {"social_skills": 0.7, "attractiveness": 0.6, "communication": 0.6},
                "host": {"leadership": 0.8, "communication": 0.9, "social_skills": 0.8},
                "producer": {"analytical": 0.8, "organization": 0.7, "observation": 0.8}
            }
            
            # Determine role from agent_id (contestants are first 20, etc.)
            agent_num = int(agent_id.split('_')[1])
            if agent_num < 20:
                role = "contestant"
            elif agent_num < 23:
                role = "host"
            else:
                role = "producer"
                
            skills = role_skills.get(role, {"general": 0.5})
            enhanced_manager.update_specialization("initialization", skills)
            
            logger.debug(f"Initialized enhanced manager for {persona.name} as {role}")
            
        except Exception as e:
            logger.warning(f"Error initializing enhanced manager from persona: {e}")
    
    def _assign_contestant_specialization(self, index: int) -> Dict[str, str]:
        """Assign specialization to contestants for variety"""
        specializations = [
            {'type': 'social', 'level': 'advanced'},
            {'type': 'creative', 'level': 'intermediate'},
            {'type': 'analytical', 'level': 'intermediate'},
            {'type': 'physical', 'level': 'advanced'},
            {'type': 'social', 'level': 'expert'}
        ]
        return specializations[index % len(specializations)]
    
    def _generate_dating_show_traits(self, persona_name: str, role: str) -> Dict[str, Any]:
        """Generate dating show specific traits for personas"""
        if role == 'contestant':
            return {
                'romantic_interest': 'open',
                'elimination_risk': 'low',
                'confessional_style': ['honest', 'dramatic', 'strategic'][hash(persona_name) % 3],
                'relationship_goals': ['love', 'adventure', 'experience'][hash(persona_name + 'goal') % 3],
                'social_strategy': ['authentic', 'competitive', 'diplomatic'][hash(persona_name + 'strategy') % 3]
            }
        elif role == 'host':
            return {
                'hosting_style': ['dramatic', 'supportive', 'neutral'][hash(persona_name) % 3],
                'signature_phrases': [f"{persona_name}'s catchphrase here"]
            }
        else:  # producer
            return {
                'production_focus': ['drama', 'romance', 'logistics'][hash(persona_name) % 3]
            }
    
    def _create_8_agent_meta(self, sim_folder):
        """Modify the simulation to only include 8 agents"""
        try:
            meta_path = sim_folder / "reverie" / "meta.json"
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
            
            # Select first 8 agents from the full list
            all_agents = meta_data.get('persona_names', [])
            selected_agents = all_agents[:8]  # Take first 8
            
            meta_data['persona_names'] = selected_agents
            
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            logger.info(f"Updated meta.json to include 8 agents: {selected_agents}")
            
        except Exception as e:
            logger.error(f"Failed to create 8-agent meta: {e}")
            # Continue anyway, it might still work


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
        
        # Enhanced agent state management
        self.enhanced_agent_managers: Dict[str, EnhancedAgentStateManager] = {}
        self.agent_personalities = {}
        
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
                    
                    # Create enhanced agent state managers
                    logger.info("Creating enhanced agent state managers...")
                    self.reverie_manager.create_enhanced_agent_managers(self)
                    
                    logger.info(f"Loaded {len(agents)} real Smallville agents with enhanced state management")
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
            
            # Create enhanced managers for mock agents if available
            if DATING_SHOW_AGENTS_AVAILABLE:
                logger.info("Creating enhanced agent managers for mock agents...")
                
            for i, name in enumerate(agent_names[:self.config.max_agents]):
                agent_id = f'agent_{i:03d}'
                
                mock_agent = type('MockAgent', (), {
                    'agent_id': agent_id,
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
                
                # Create enhanced state manager for mock agent
                if DATING_SHOW_AGENTS_AVAILABLE:
                    try:
                        personality_traits = {
                            "openness": 0.5 + (i % 3) * 0.2,
                            "conscientiousness": 0.4 + (i % 4) * 0.15,
                            "extroversion": 0.6 + (i % 5) * 0.1,
                            "agreeableness": 0.5 + (i % 2) * 0.3,
                            "neuroticism": 0.2 + (i % 3) * 0.2,
                            "confidence": 0.4 + (i % 4) * 0.2,
                            "romantic_interest": 0.6 + (i % 3) * 0.2,
                            "competitiveness": 0.3 + (i % 4) * 0.25
                        }
                        
                        enhanced_manager = create_enhanced_agent_state(
                            agent_id=agent_id,
                            name=name,
                            personality_traits=personality_traits
                        )
                        
                        # Initialize with mock data
                        enhanced_manager.state["current_location"] = "villa_main_area"
                        enhanced_manager.state["current_activity"] = "socializing"
                        enhanced_manager.add_memory(f"Arrived at the dating show villa", "event", 0.8)
                        enhanced_manager.add_memory(f"Meeting other contestants", "social", 0.6)
                        
                        # Set role and skills
                        role_skills = {"social_skills": 0.6, "communication": 0.5, "attractiveness": 0.5}
                        enhanced_manager.update_specialization("mock_initialization", role_skills)
                        
                        self.enhanced_agent_managers[agent_id] = enhanced_manager
                        self.agent_personalities[agent_id] = personality_traits
                        
                    except Exception as e:
                        logger.warning(f"Failed to create enhanced manager for mock agent {name}: {e}")
                
                mock_agents.append(mock_agent)
            
            logger.info(f"Loaded {len(mock_agents)} mock agents")
            return mock_agents
            
        except Exception as e:
            logger.error(f"Failed to load agents: {e}")
            raise
    
    async def run_simulation_step(self):
        """Run a single simulation step and update JSON files with comprehensive error recovery"""
        from services.error_recovery import get_error_recovery, ErrorSeverity, safe_execute
        
        error_recovery = get_error_recovery()
        
        if self.reverie_server and self.reverie_manager:
            # Get current step before increment
            old_step = self.reverie_server.step
            print(f"🎬 [DEBUG] Starting run_simulation_step for step {old_step}")
            print(f"🎬 [DEBUG] Reverie server step: {self.reverie_server.step}")
            print(f"🎬 [DEBUG] Reverie manager available: {self.reverie_manager is not None}")
            
            # Ensure environment files exist for current step
            def ensure_environment_files():
                from services.environment_generator import get_environment_generator
                env_generator = get_environment_generator()
                
                # Ensure files exist for current step
                sim_code = "dating_show_8_agents"  # Use consistent simulation code
                env_generator.ensure_step_files_exist(sim_code, old_step)
                
                # If this is a new simulation, initialize storage
                if old_step == 0:
                    agent_names = [persona.name for persona in self.reverie_server.personas.values()]
                    env_generator.initialize_simulation_storage(sim_code, agent_names)
                    logger.info(f"Initialized simulation storage for {sim_code} with {len(agent_names)} agents")
                
                return True
            
            # Safe execution of environment file setup
            env_success, _ = safe_execute(
                'environment_generator', 
                ensure_environment_files,
                ErrorSeverity.MEDIUM,
                {'step': old_step, 'sim_code': 'dating_show_8_agents'}
            )
            
            if not env_success:
                logger.warning("Environment file setup failed, continuing with degraded functionality")
            
            # CRITICAL FIX: Generate NEXT step files BEFORE running reverie server
            # The reverie server will block waiting for environment/{next_step}.json
            # So we must ensure it exists before calling start_server()
            next_step = old_step + 1
            def pre_generate_next_step_files():
                print(f"🎬 [DEBUG] PRE-GENERATING step {next_step} files BEFORE reverie server call")
                from services.enhanced_step_manager import get_enhanced_step_manager
                import asyncio
                
                step_manager = get_enhanced_step_manager()
                
                # Run sync generation using direct file creation to avoid async issues
                def run_sync_generation():
                    try:
                        import json
                        import os
                        from pathlib import Path
                        
                        # Auto-detect correct storage path
                        legacy_path = "/Applications/Projects/Open source/generative_agents/environment/frontend_server/storage"
                        modern_path = "/Applications/Projects/Open source/generative_agents/dating_show_env/frontend_service/storage"
                        
                        storage_path = legacy_path
                        if os.path.exists(os.path.join(legacy_path, "dating_show_8_agents")):
                            storage_path = legacy_path
                            print(f"🎯 [DEBUG] Using legacy storage path: {storage_path}")
                        elif os.path.exists(os.path.join(modern_path, "dating_show_8_agents")):
                            storage_path = modern_path
                            print(f"🎯 [DEBUG] Using modern storage path: {storage_path}")
                        
                        # Create step files directly
                        sim_path = Path(storage_path) / "dating_show_8_agents"
                        env_file = sim_path / "environment" / f"{next_step}.json"
                        mov_file = sim_path / "movement" / f"{next_step}.json"
                        
                        # Ensure directories exist
                        env_file.parent.mkdir(parents=True, exist_ok=True)
                        mov_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Read previous step as template if available
                        prev_env_file = sim_path / "environment" / f"{next_step-1}.json"
                        prev_mov_file = sim_path / "movement" / f"{next_step-1}.json"
                        
                        if prev_env_file.exists() and prev_mov_file.exists():
                            # Copy and modify previous step
                            with open(prev_env_file, 'r') as f:
                                env_data = json.load(f)
                            with open(prev_mov_file, 'r') as f:
                                mov_data = json.load(f)
                            
                            # Update step info
                            if "meta" in env_data:
                                env_data["meta"]["step"] = next_step
                            if "meta" in mov_data:
                                mov_data["meta"]["step"] = next_step
                            
                            # Write files
                            with open(env_file, 'w') as f:
                                json.dump(env_data, f, indent=2)
                            with open(mov_file, 'w') as f:
                                json.dump(mov_data, f, indent=2)
                                
                            print(f"✅ [DEBUG] Created step {next_step} files from previous step")
                            return True
                        else:
                            print(f"⚠️ [DEBUG] Previous step {next_step-1} files not found, will skip pre-generation")
                            return False
                            
                    except Exception as e:
                        print(f"🚨 [DEBUG] Pre-generation error: {e}")
                        return False
                
                return run_sync_generation()
            
            # Safe execution of pre-generation
            pre_gen_success, _ = safe_execute(
                'step_file_pre_generation',
                pre_generate_next_step_files,
                ErrorSeverity.HIGH,
                {'next_step': next_step}
            )
            
            if not pre_gen_success:
                logger.error(f"Failed to pre-generate step {next_step} files - reverie server will block!")
            else:
                logger.info(f"✅ Pre-generated step {next_step} files successfully")
            
            # Run actual reverie simulation step
            def run_reverie_step():
                print(f"🎬 [DEBUG] About to call start_server(1) - current step: {self.reverie_server.step}")
                # Use the actual reverie simulation logic to run one step
                self.reverie_server.start_server(1)
                print(f"🎬 [DEBUG] start_server(1) completed - new step: {self.reverie_server.step}")
                logger.info(f"Executed reverie simulation step for step {self.reverie_server.step}")
                return True
            
            # Safe execution of reverie step
            reverie_success, _ = safe_execute(
                'reverie_simulation',
                run_reverie_step,
                ErrorSeverity.HIGH,
                {'old_step': old_step}
            )
            
            if not reverie_success:
                logger.error("Reverie simulation step failed, using fallback increment")
                self.reverie_server.step += 1
            
            # Generate environment files for the new step using enhanced step manager
            def generate_new_step_files():
                from services.enhanced_step_manager import get_enhanced_step_manager
                import asyncio
                
                new_step = self.reverie_server.step
                print(f"🎬 [DEBUG] Starting generate_new_step_files for step {new_step}")
                logger.info(f"🔧 Generating enhanced step files for step {new_step}")
                
                # Get enhanced step manager
                step_manager = get_enhanced_step_manager()
                print(f"🎬 [DEBUG] Got enhanced step manager: {step_manager}")
                
                # Use direct file creation to avoid async issues
                def create_step_files_directly():
                    try:
                        import json
                        import os
                        from pathlib import Path
                        
                        # Auto-detect correct storage path
                        legacy_path = "/Applications/Projects/Open source/generative_agents/environment/frontend_server/storage"
                        modern_path = "/Applications/Projects/Open source/generative_agents/dating_show_env/frontend_service/storage"
                        
                        storage_path = legacy_path
                        if os.path.exists(os.path.join(legacy_path, "dating_show_8_agents")):
                            storage_path = legacy_path
                            print(f"🎯 [DEBUG] Using legacy storage path: {storage_path}")
                        elif os.path.exists(os.path.join(modern_path, "dating_show_8_agents")):
                            storage_path = modern_path
                            print(f"🎯 [DEBUG] Using modern storage path: {storage_path}")
                        
                        # Create step files directly
                        sim_path = Path(storage_path) / "dating_show_8_agents"
                        env_file = sim_path / "environment" / f"{new_step}.json"
                        mov_file = sim_path / "movement" / f"{new_step}.json"
                        
                        # Skip if files already exist
                        if env_file.exists() and mov_file.exists():
                            print(f"✅ [DEBUG] Step {new_step} files already exist")
                            return True
                        
                        # Ensure directories exist
                        env_file.parent.mkdir(parents=True, exist_ok=True)
                        mov_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Read previous step as template if available
                        prev_env_file = sim_path / "environment" / f"{new_step-1}.json"
                        prev_mov_file = sim_path / "movement" / f"{new_step-1}.json"
                        
                        if prev_env_file.exists() and prev_mov_file.exists():
                            # Copy and modify previous step
                            with open(prev_env_file, 'r') as f:
                                env_data = json.load(f)
                            with open(prev_mov_file, 'r') as f:
                                mov_data = json.load(f)
                            
                            # Update step info
                            if "meta" in env_data:
                                env_data["meta"]["step"] = new_step
                            if "meta" in mov_data:
                                mov_data["meta"]["step"] = new_step
                            
                            # Write files
                            with open(env_file, 'w') as f:
                                json.dump(env_data, f, indent=2)
                            with open(mov_file, 'w') as f:
                                json.dump(mov_data, f, indent=2)
                                
                            print(f"✅ [DEBUG] Created step {new_step} files from previous step")
                            return True
                        else:
                            print(f"⚠️ [DEBUG] Previous step {new_step-1} files not found, cannot create step {new_step}")
                            return False
                            
                    except Exception as e:
                        print(f"🚨 [DEBUG] Direct file creation error: {e}")
                        return False
                
                print(f"🎬 [DEBUG] Creating step files directly")
                success = create_step_files_directly()
                print(f"🎬 [DEBUG] Direct file creation returned: {success}")
                
                if success:
                    print(f"✅ [DEBUG] Enhanced step files generated successfully for step {new_step}")
                    logger.info(f"✅ Enhanced step files generated for step {new_step}")
                else:
                    print(f"⚠️ [DEBUG] Enhanced step file generation had issues for step {new_step}")
                    logger.warning(f"⚠️ Enhanced step file generation had issues for step {new_step}")
                
                print(f"🎬 [DEBUG] generate_new_step_files returning: {success}")
                return success
            
            # Safe execution of new step file generation
            file_gen_success, _ = safe_execute(
                'environment_generator',
                generate_new_step_files,
                ErrorSeverity.MEDIUM,
                {'new_step': self.reverie_server.step}
            )
            
            if not file_gen_success:
                logger.warning("Environment file generation failed for new step")
            
            # Update the JSON file for frontend synchronization
            def update_step_json():
                self.reverie_manager.update_curr_step_json(self.reverie_server.step)
                return True
            
            # Safe execution of step JSON update
            json_success, _ = safe_execute(
                'file_io',
                update_step_json,
                ErrorSeverity.LOW,
                {'step': self.reverie_server.step}
            )
            
            if not json_success:
                logger.warning("Failed to update step JSON file")
            
            logger.info(f"Simulation step: {old_step} -> {self.reverie_server.step}")
            
            # Return success if core simulation step worked
            return reverie_success
        
        logger.error("No reverie server or manager available")
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
    
    def run_interactive_mode(self):
        """Run in interactive mode with step-by-step control"""
        
        print("=" * 50)
        print(f"Local reverie core available: {'✅' if LOCAL_REVERIE else '❌'}")
        print("🎭 Standalone Dating Show Simulation")
        print("   Fork from: base_the_ville_n25")
        print("   New sim: dating_show_8_agents (8 contestants)")
        
        # Initialize simulation
        if not self._initialize_interactive_simulation():
            print("❌ Failed to initialize simulation")
            return
        
        print("\n🎮 Interactive Simulation Mode")
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
                    self._run_interactive_step()
                    
                elif command.startswith('auto'):
                    parts = command.split()
                    steps = int(parts[1]) if len(parts) > 1 else 10
                    self._run_interactive_auto(steps)
                    
                elif command == 'status':
                    self._show_interactive_status()
                        
                elif command == 'save':
                    self._save_interactive_simulation()
                        
                else:
                    print(f"❓ Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                break
            except ValueError:
                print("❌ Invalid number format")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _initialize_interactive_simulation(self) -> bool:
        """Initialize simulation for interactive mode"""
        try:
            print("🚀 Initializing 25-agent Smallville simulation...")
            
            # Initialize reverie manager
            self.reverie_server = self.reverie_manager.initialize_reverie_simulation()
            
            if self.reverie_server:
                print("✅ Reverie server created successfully!")
                print(f"👥 Loaded {len(self.reverie_server.personas)} agents:")
                
                # List agents with roles
                for i, persona_name in enumerate(self.reverie_server.personas.keys(), 1):
                    role = "contestant" if i <= 20 else "host" if i <= 22 else "producer"
                    print(f"    {i:2d}. {persona_name} ({role})")
                    
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Simulation initialization failed: {e}")
            return False
    
    def _run_interactive_step(self):
        """Run a single interactive simulation step"""
        if not self.reverie_server:
            print("❌ No simulation server available")
            return
            
        try:
            old_step = self.reverie_server.step
            
            # Enhanced step with actual persona movement
            success = self._run_enhanced_simulation_step()
            
            if success:
                print(f"⏭️  Simulation step: {old_step} → {self.reverie_server.step}")
                print("✅ Step completed successfully")
            else:
                print("❌ Step failed")
                
        except Exception as e:
            print(f"❌ Simulation step failed: {e}")
    
    def _run_interactive_auto(self, steps: int):
        """Run auto simulation in interactive mode"""
        
        print(f"🎬 Starting auto-simulation for {steps} steps...")
        print(f"⏱️  Step delay: 1.0s")
        
        success_count = 0
        
        for i in range(steps):
            print(f"\n📍 Step {i+1}/{steps}")
            
            if self.reverie_server:
                old_step = self.reverie_server.step
                success = self._run_enhanced_simulation_step()
                
                if success:
                    success_count += 1
                    print(f"⏭️  Simulation step: {old_step} → {self.reverie_server.step}")
                    print("✅ Step completed successfully")
                else:
                    print("❌ Step failed")
                    break
            else:
                print("❌ No simulation server")
                break
            
            # Delay between steps
            if i < steps - 1:
                time.sleep(1.0)
        
        print(f"\n🏁 Auto-simulation completed!")
        print(f"📊 Success rate: {success_count}/{steps} steps")
        
        if self.reverie_server:
            print(f"🎯 Final simulation step: {self.reverie_server.step}")
    
    def _show_interactive_status(self):
        """Show current simulation status"""
        if self.reverie_server:
            print(f"📊 Simulation Status:")
            print(f"   Current step: {self.reverie_server.step}")
            print(f"   Agents: {len(self.reverie_server.personas)}")
            print(f"   Simulation code: dating_show_8_agents (8 contestants)")
            print(f"   Local reverie: {'✅' if LOCAL_REVERIE else '❌'}")
        else:
            print("❌ No simulation running")
    
    def _save_interactive_simulation(self):
        """Save simulation state"""
        if self.reverie_server:
            print("💾 Saving simulation state...")
            try:
                self.reverie_server.save()
                print("✅ Simulation saved!")
            except Exception as e:
                print(f"❌ Save failed: {e}")
        else:
            print("❌ No simulation to save")
    
    def _run_enhanced_simulation_step(self) -> bool:
        """DEPRECATED: Use UnifiedAgentManager with UpdatePipeline instead
        
        This method contains duplicate logic that's better handled by
        the unified architecture's real-time update system.
        """
        if not self.reverie_server:
            return False
            
        try:
            # Get current step before running reverie simulation
            old_step = self.reverie_server.step
            
            # CRITICAL FIX: Pre-generate next step files to prevent blocking
            next_step = old_step + 1
            print(f"🎬 [DEBUG] Pre-generating step {next_step} files before reverie server call")
            
            from services.enhanced_step_manager import get_enhanced_step_manager
            import asyncio
            
            step_manager = get_enhanced_step_manager()
            
            # Pre-generate next step files synchronously using direct file creation
            def run_sync_generation():
                try:
                    # Use direct file creation to avoid async issues
                    import json
                    import os
                    from pathlib import Path
                    
                    # Auto-detect correct storage path
                    legacy_path = "/Applications/Projects/Open source/generative_agents/environment/frontend_server/storage"
                    modern_path = "/Applications/Projects/Open source/generative_agents/dating_show_env/frontend_service/storage"
                    
                    storage_path = legacy_path
                    if os.path.exists(os.path.join(legacy_path, "dating_show_8_agents")):
                        storage_path = legacy_path
                        print(f"🎯 [DEBUG] Using legacy storage path: {storage_path}")
                    elif os.path.exists(os.path.join(modern_path, "dating_show_8_agents")):
                        storage_path = modern_path
                        print(f"🎯 [DEBUG] Using modern storage path: {storage_path}")
                    
                    # Create step files directly
                    sim_path = Path(storage_path) / "dating_show_8_agents"
                    env_file = sim_path / "environment" / f"{next_step}.json"
                    mov_file = sim_path / "movement" / f"{next_step}.json"
                    
                    # Ensure directories exist
                    env_file.parent.mkdir(parents=True, exist_ok=True)
                    mov_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Read previous step as template if available
                    prev_env_file = sim_path / "environment" / f"{next_step-1}.json"
                    prev_mov_file = sim_path / "movement" / f"{next_step-1}.json"
                    
                    if prev_env_file.exists() and prev_mov_file.exists():
                        # Copy and modify previous step
                        with open(prev_env_file, 'r') as f:
                            env_data = json.load(f)
                        with open(prev_mov_file, 'r') as f:
                            mov_data = json.load(f)
                        
                        # Update step info
                        if "meta" in env_data:
                            env_data["meta"]["step"] = next_step
                        if "meta" in mov_data:
                            mov_data["meta"]["step"] = next_step
                        
                        # Write files
                        with open(env_file, 'w') as f:
                            json.dump(env_data, f, indent=2)
                        with open(mov_file, 'w') as f:
                            json.dump(mov_data, f, indent=2)
                            
                        print(f"✅ [DEBUG] Created step {next_step} files from previous step")
                        return True
                    else:
                        print(f"⚠️ [DEBUG] Previous step {next_step-1} files not found, skipping pre-generation")
                        return False
                        
                except Exception as e:
                    print(f"🚨 [DEBUG] Pre-generation error: {e}")
                    return False
            
            pre_gen_success = run_sync_generation()
            if not pre_gen_success:
                logger.warning(f"Failed to pre-generate step {next_step} files")
            else:
                print(f"✅ [DEBUG] Step {next_step} files ready for reverie server")
            
            # Run actual reverie simulation step
            try:
                print(f"🎬 [DEBUG] Calling start_server(1) - current step: {old_step}")
                # Use the actual reverie simulation logic to run one step
                self.reverie_server.start_server(1)
                print(f"🎬 [DEBUG] start_server(1) completed - new step: {self.reverie_server.step}")
                logger.info(f"Executed reverie simulation step for step {self.reverie_server.step}")
            except Exception as e:
                logger.error(f"Error running simulation step: {e}")
                # Fallback: just increment step if there's an error
                self.reverie_server.step += 1
                
                # Update simulation time manually if reverie failed
                import datetime
                self.reverie_server.curr_time += datetime.timedelta(seconds=self.reverie_server.sec_per_step)
            
            # Update enhanced agent states with dating show behaviors
            for i, (persona_name, persona) in enumerate(self.reverie_server.personas.items()):
                agent_id = f'agent_{i:03d}'
                current_pos = self.reverie_server.personas_tile[persona_name]
                
                # Get enhanced manager for this agent
                enhanced_manager = self.enhanced_agent_managers.get(agent_id)
                
                # If reverie simulation worked, use its results
                if enhanced_manager:
                    # Use current persona state from reverie simulation
                    current_action = getattr(persona.scratch, 'daily_plan_req', 'socializing')
                    self._update_enhanced_agent_from_simulation(enhanced_manager, agent_id, current_pos, current_action, old_step)
                    
                    # Add dating show specific actions based on step
                    self._process_dating_show_actions(persona_name, current_action, self.reverie_server.step)
            
            # Process social interactions between agents
            self._process_social_interactions(self.reverie_server.step)
            
            # Special dating show events based on step
            self._process_dating_show_events(self.reverie_server.step)
            
            # Queue agent state updates for frontend synchronization
            self._queue_enhanced_agent_updates(self.reverie_server.step)
            
            # Update frontend sync
            if hasattr(self.reverie_manager, 'update_curr_step_json'):
                self.reverie_manager.update_curr_step_json(self.reverie_server.step)
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced simulation step failed: {e}")
            return False
    
    def _update_enhanced_agent_from_move(self, enhanced_manager: EnhancedAgentStateManager, 
                                        agent_id: str, new_pos: tuple, desc: str, 
                                        emoji: str, step: int):
        """Update enhanced agent state from real persona movement"""
        try:
            # Update location
            x, y = new_pos
            new_location = f"the_ville_{x}_{y}"
            enhanced_manager.state["current_location"] = new_location
            enhanced_manager.state["current_activity"] = desc
            
            # Add memory of the movement/action
            memory_content = f"Step {step}: {desc} at location ({x}, {y})"
            if emoji:
                memory_content = f"{emoji} {memory_content}"
            
            enhanced_manager.add_memory(memory_content, "movement", 0.4, {
                "location": {"x": x, "y": y},
                "step": step,
                "emoji": emoji
            })
            
            # Update emotional state based on action
            emotion_changes = self._get_emotion_changes_for_action(desc, enhanced_manager)
            if emotion_changes:
                enhanced_manager.update_emotional_state(emotion_changes)
            
            # Update skills based on action type
            skill_updates = self._get_skill_updates_for_action(desc)
            if skill_updates:
                enhanced_manager.update_specialization(desc, skill_updates)
            
            logger.debug(f"Updated enhanced state for {agent_id}: {desc} at ({x}, {y})")
            
        except Exception as e:
            logger.warning(f"Error updating enhanced agent {agent_id} from move: {e}")
    
    def _update_enhanced_agent_from_simulation(self, enhanced_manager: EnhancedAgentStateManager,
                                             agent_id: str, new_pos: tuple, action: str, step: int):
        """Update enhanced agent state from simulated movement"""
        try:
            # Update location
            x, y = new_pos
            new_location = f"the_ville_{x}_{y}"
            enhanced_manager.state["current_location"] = new_location
            enhanced_manager.state["current_activity"] = action
            
            # Add memory of the simulated action
            actions = ["chatting with other contestants", "exploring the villa", "reflecting on relationships", 
                      "participating in group activities", "having private conversations"]
            detailed_action = actions[step % len(actions)]
            
            memory_content = f"Step {step}: {detailed_action} at location ({x}, {y})"
            enhanced_manager.add_memory(memory_content, "activity", 0.3, {
                "location": {"x": x, "y": y},
                "step": step,
                "simulated": True
            })
            
            # Simulate emotional changes
            emotion_changes = {
                "happiness": (hash(agent_id + str(step)) % 21 - 10) / 100,  # -0.1 to 0.1
                "excitement": (hash(agent_id + str(step + 1)) % 21 - 10) / 200,  # -0.05 to 0.05
                "anxiety": (hash(agent_id + str(step + 2)) % 11 - 5) / 200,  # -0.025 to 0.025
            }
            enhanced_manager.update_emotional_state(emotion_changes)
            
            # Simulate skill development
            if step % 10 == 0:  # Every 10 steps, small skill improvement
                skill_updates = {"social_skills": 0.01, "communication": 0.005}
                enhanced_manager.update_specialization(f"practice_{step}", skill_updates)
                
            logger.debug(f"Updated simulated state for {agent_id}: {detailed_action} at ({x}, {y})")
            
        except Exception as e:
            logger.warning(f"Error updating enhanced agent {agent_id} from simulation: {e}")
    
    def _get_emotion_changes_for_action(self, action_desc: str, 
                                       enhanced_manager: EnhancedAgentStateManager = None) -> Dict[str, float]:
        """Get emotional changes based on action description"""
        action_lower = action_desc.lower()
        changes = {}
        
        # Social interactions increase happiness
        if any(word in action_lower for word in ["chat", "talk", "conversation", "socialize"]):
            changes["happiness"] = 0.05
            changes["confidence"] = 0.02
            
        # Movement/exploration can increase excitement
        if any(word in action_lower for word in ["move", "explore", "walk", "go"]):
            changes["excitement"] = 0.03
            
        # Romantic actions
        if any(word in action_lower for word in ["flirt", "date", "romantic", "intimate"]):
            changes["happiness"] = 0.08
            changes["excitement"] = 0.1
            changes["anxiety"] = 0.02  # slight nervousness
            
        # Competition or conflict
        if any(word in action_lower for word in ["compete", "argue", "tension", "conflict"]):
            changes["anxiety"] = 0.05
            changes["excitement"] = 0.03
            changes["happiness"] = -0.02
            
        return changes
    
    def _get_skill_updates_for_action(self, action_desc: str) -> Dict[str, float]:
        """Get skill updates based on action description"""
        action_lower = action_desc.lower()
        skills = {}
        
        # Communication skills from talking
        if any(word in action_lower for word in ["chat", "talk", "conversation"]):
            skills["communication"] = 0.01
            skills["social_skills"] = 0.01
            
        # Leadership from organizing or leading
        if any(word in action_lower for word in ["lead", "organize", "guide"]):
            skills["leadership"] = 0.02
            skills["confidence"] = 0.01
            
        # Creativity from creative activities
        if any(word in action_lower for word in ["create", "art", "music", "creative"]):
            skills["creativity"] = 0.02
            
        # Physical activities
        if any(word in action_lower for word in ["exercise", "sport", "physical", "dance"]):
            skills["physical_fitness"] = 0.02
            skills["confidence"] = 0.01
            
        return skills
    
    def _process_social_interactions(self, step: int):
        """Process social interactions between agents during this step"""
        try:
            # Only process social interactions every few steps to avoid overwhelming
            if step % 5 != 0:
                return
                
            # Get list of agent IDs that have enhanced managers
            active_agents = list(self.enhanced_agent_managers.keys())
            if len(active_agents) < 2:
                return
                
            # Simulate some social interactions based on proximity and personality
            import random
            random.seed(step)  # Deterministic but varied interactions
            
            # Number of interactions this step (1-3)
            num_interactions = min(random.randint(1, 3), len(active_agents) // 2)
            
            for _ in range(num_interactions):
                # Choose two agents to interact
                agent_a_id = random.choice(active_agents)
                agent_b_id = random.choice([aid for aid in active_agents if aid != agent_a_id])
                
                agent_a_manager = self.enhanced_agent_managers[agent_a_id]
                agent_b_manager = self.enhanced_agent_managers[agent_b_id]
                
                # Determine interaction type based on roles and step
                interaction_type = self._determine_interaction_type(agent_a_id, agent_b_id, step)
                
                # Generate interaction content
                interaction_content = self._generate_interaction_content(
                    agent_a_manager.name, agent_b_manager.name, interaction_type, step
                )
                
                # Calculate emotional impact based on personalities and interaction
                emotional_impact_a = self._calculate_emotional_impact(
                    agent_a_manager, agent_b_manager, interaction_type
                )
                emotional_impact_b = self._calculate_emotional_impact(
                    agent_b_manager, agent_a_manager, interaction_type
                )
                
                # Process interaction for both agents
                agent_a_manager.process_social_interaction(
                    agent_b_manager.name, interaction_type, interaction_content, emotional_impact_a
                )
                agent_b_manager.process_social_interaction(
                    agent_a_manager.name, interaction_type, interaction_content, emotional_impact_b
                )
                
                logger.debug(f"Social interaction: {agent_a_manager.name} <-> {agent_b_manager.name}: {interaction_type}")
                
        except Exception as e:
            logger.warning(f"Error processing social interactions: {e}")
    
    def _determine_interaction_type(self, agent_a_id: str, agent_b_id: str, step: int) -> str:
        """Determine the type of social interaction between two agents"""
        # Extract agent numbers to determine roles
        agent_a_num = int(agent_a_id.split('_')[1])
        agent_b_num = int(agent_b_id.split('_')[1])
        
        # Contestants (0-19), Hosts (20-22), Producers (23+)
        a_is_contestant = agent_a_num < 20
        b_is_contestant = agent_b_num < 20
        a_is_host = 20 <= agent_a_num < 23
        b_is_host = 20 <= agent_b_num < 23
        
        # Interaction types based on roles and step
        if a_is_contestant and b_is_contestant:
            if step % 100 < 20:  # Early in cycle, more getting-to-know
                return random.choice(["conversation", "flirtation", "friendship"])
            else:  # Later in cycle, more competition
                return random.choice(["conversation", "flirtation", "competition", "alliance"])
        elif (a_is_host or b_is_host):
            return random.choice(["interview", "guidance", "ceremony_interaction"])
        else:  # Producer interactions
            return random.choice(["direction", "observation", "coordination"])
    
    def _generate_interaction_content(self, name_a: str, name_b: str, 
                                    interaction_type: str, step: int) -> str:
        """Generate content description for the interaction"""
        content_templates = {
            "conversation": [
                f"had a deep conversation about their backgrounds",
                f"shared stories about their dating experiences",
                f"discussed their goals for the show",
                f"talked about their families and interests"
            ],
            "flirtation": [
                f"shared some playful banter and flirtatious looks",
                f"had a romantic moment by the pool",
                f"exchanged compliments and showed mutual interest",
                f"enjoyed some intimate conversation away from others"
            ],
            "competition": [
                f"felt some tension over romantic interests", 
                f"competed for attention during group activities",
                f"had a subtle disagreement about strategy",
                f"sensed rivalry in the air"
            ],
            "friendship": [
                f"bonded over shared experiences",
                f"supported each other through challenges",
                f"developed a genuine friendship",
                f"shared laughs and good moments"
            ],
            "alliance": [
                f"formed a strategic alliance",
                f"agreed to support each other",
                f"discussed vote strategies privately",
                f"planned their approach together"
            ],
            "interview": [
                f"had a one-on-one interview session",
                f"discussed feelings and progress",
                f"received advice and guidance",
                f"reflected on the journey so far"
            ],
            "guidance": [
                f"received emotional support and advice",
                f"talked through difficult decisions",
                f"got perspective on relationships",
                f"discussed future steps"
            ],
            "ceremony_interaction": [
                f"participated in the rose ceremony",
                f"shared a meaningful moment during elimination",
                f"exchanged words during the group gathering",
                f"had an emotional ceremony moment"
            ]
        }
        
        templates = content_templates.get(interaction_type, ["had an interaction"])
        content = random.choice(templates)
        return content
    
    def _calculate_emotional_impact(self, agent_manager: EnhancedAgentStateManager,
                                  other_manager: EnhancedAgentStateManager, 
                                  interaction_type: str) -> float:
        """Calculate emotional impact of interaction based on personalities"""
        agent_traits = agent_manager.state["personality_traits"]
        
        # Base impact based on interaction type
        base_impacts = {
            "conversation": 0.1,
            "flirtation": 0.3,
            "competition": -0.1,
            "friendship": 0.2,
            "alliance": 0.1,
            "interview": 0.0,
            "guidance": 0.1,
            "ceremony_interaction": 0.2
        }
        
        base_impact = base_impacts.get(interaction_type, 0.0)
        
        # Modify based on personality traits
        if interaction_type in ["flirtation", "conversation"]:
            # Extroverts enjoy social interactions more
            base_impact += agent_traits.get("extroversion", 0.5) * 0.1
            # Confident agents have more positive interactions
            base_impact += agent_traits.get("confidence", 0.5) * 0.05
            
        elif interaction_type == "competition":
            # Competitive agents might actually enjoy competition
            if agent_traits.get("competitiveness", 0.5) > 0.7:
                base_impact += 0.1
            # But less agreeable agents have more negative reactions
            base_impact -= (1.0 - agent_traits.get("agreeableness", 0.5)) * 0.1
            
        # Add some randomness
        import random
        base_impact += random.uniform(-0.05, 0.05)
        
        # Clamp to reasonable range
        return max(-0.5, min(0.5, base_impact))
    
    def _process_dating_show_actions(self, persona_name: str, action_desc: str, step: int):
        """Process dating show specific actions for personas"""
        try:
            # Add dating show context to actions
            if step % 50 == 0:  # Every 50 steps, potential confessional
                if hash(persona_name + str(step)) % 5 == 0:  # 20% chance
                    logger.info(f"🎥 {persona_name} is giving a confessional interview")
            
            if step % 100 == 0:  # Every 100 steps, potential rose ceremony
                if hash(persona_name + str(step)) % 10 == 0:  # 10% chance
                    logger.info(f"🌹 {persona_name} is participating in rose ceremony")
            
            # Track social interactions for relationship building
            if "chat" in action_desc.lower() or "talk" in action_desc.lower():
                logger.debug(f"💬 {persona_name} is building relationships through conversation")
                
        except Exception as e:
            logger.debug(f"Error processing dating show actions for {persona_name}: {e}")
    
    def _process_dating_show_events(self, step: int):
        """Process dating show wide events and update enhanced agent states"""
        try:
            # Major dating show events
            if step % 200 == 0:  # Every 200 steps, major event
                events = ["group date", "elimination ceremony", "new contestant arrival", "special challenge"]
                event = events[step // 200 % len(events)]
                logger.info(f"🎭 Major dating show event: {event}")
                
                # Update all enhanced agents with event memory
                for agent_id, enhanced_manager in self.enhanced_agent_managers.items():
                    memory_content = f"Participated in major event: {event}"
                    enhanced_manager.add_memory(memory_content, "major_event", 0.8, {
                        "event_type": event,
                        "step": step,
                        "participants": list(self.enhanced_agent_managers.keys())
                    })
                    
                    # Event-specific emotional responses
                    emotion_changes = self._get_event_emotion_changes(event, enhanced_manager)
                    if emotion_changes:
                        enhanced_manager.update_emotional_state(emotion_changes)
            
            if step % 300 == 0:  # Every 300 steps, drama event
                logger.info(f"💥 Drama alert! Tensions rising in the villa...")
                
                # Update all contestants with drama tension
                for agent_id, enhanced_manager in self.enhanced_agent_managers.items():
                    agent_num = int(agent_id.split('_')[1])
                    if agent_num < 20:  # Only contestants affected by drama
                        memory_content = "Tensions are rising in the villa, drama is building"
                        enhanced_manager.add_memory(memory_content, "drama", 0.6, {
                            "step": step,
                            "drama_level": "high"
                        })
                        
                        # Drama increases anxiety and excitement
                        drama_emotions = {
                            "anxiety": 0.1,
                            "excitement": 0.05,
                            "happiness": -0.02
                        }
                        enhanced_manager.update_emotional_state(drama_emotions)
            
            # Rose ceremony every 500 steps
            if step % 500 == 0 and step > 0:
                logger.info(f"🌹 Rose ceremony time - elimination round!")
                
                # All contestants experience ceremony stress
                for agent_id, enhanced_manager in self.enhanced_agent_managers.items():
                    agent_num = int(agent_id.split('_')[1])
                    if agent_num < 20:  # Contestants
                        memory_content = "Rose ceremony - will I receive a rose tonight?"
                        enhanced_manager.add_memory(memory_content, "ceremony", 0.9, {
                            "step": step,
                            "ceremony_type": "rose_ceremony"
                        })
                        
                        # High stress emotions
                        ceremony_emotions = {
                            "anxiety": 0.2,
                            "excitement": 0.1,
                            "nervousness": 0.15
                        }
                        enhanced_manager.update_emotional_state(ceremony_emotions)
                
        except Exception as e:
            logger.debug(f"Error processing dating show events: {e}")
    
    def _get_event_emotion_changes(self, event: str, 
                                  enhanced_manager: EnhancedAgentStateManager) -> Dict[str, float]:
        """Get emotional changes based on dating show event type"""
        agent_traits = enhanced_manager.state["personality_traits"]
        
        event_emotions = {
            "group date": {
                "happiness": 0.1,
                "excitement": 0.15,
                "anxiety": 0.05  # Some nervousness
            },
            "elimination ceremony": {
                "anxiety": 0.2,
                "nervousness": 0.15,
                "excitement": 0.05
            },
            "new contestant arrival": {
                "excitement": 0.1,
                "anxiety": 0.08,  # Competition anxiety
                "curiosity": 0.12
            },
            "special challenge": {
                "excitement": 0.12,
                "confidence": 0.05,
                "competitive_spirit": 0.1
            }
        }
        
        base_emotions = event_emotions.get(event, {})
        
        # Modify based on personality
        modified_emotions = {}
        for emotion, value in base_emotions.items():
            modified_value = value
            
            # Extroverts enjoy events more
            if emotion in ["happiness", "excitement"] and agent_traits.get("extroversion", 0.5) > 0.6:
                modified_value *= 1.2
                
            # Neurotic agents have stronger anxiety responses
            if emotion in ["anxiety", "nervousness"] and agent_traits.get("neuroticism", 0.5) > 0.6:
                modified_value *= 1.3
                
            # Confident agents have less anxiety
            if emotion in ["anxiety", "nervousness"] and agent_traits.get("confidence", 0.5) > 0.7:
                modified_value *= 0.7
                
            modified_emotions[emotion] = modified_value
            
        return modified_emotions
    
    def _queue_enhanced_agent_updates(self, step: int):
        """Queue agent state updates from enhanced managers to frontend bridge"""
        try:
            if not self.orchestrator or not self.orchestrator.frontend_bridge:
                logger.debug("No orchestrator or frontend bridge available for queueing updates")
                return
                
            # Queue updates for all enhanced agents (but only send some each step to avoid overwhelming)
            update_count = 0
            max_updates_per_step = 5  # Limit to 5 updates per step
            
            for agent_id, enhanced_manager in self.enhanced_agent_managers.items():
                # Only update some agents each step (rotate through them)
                agent_num = int(agent_id.split('_')[1])
                if (step + agent_num) % 3 == 0 and update_count < max_updates_per_step:
                    
                    # Extract comprehensive agent data from enhanced manager
                    agent_data = self._extract_enhanced_agent_data(enhanced_manager, agent_id)
                    
                    # Queue the update
                    self.orchestrator.frontend_bridge.queue_agent_update(agent_id, agent_data)
                    update_count += 1
                    
                    logger.debug(f"Queued enhanced update for {agent_id}: {agent_data['current_action']}")
            
            logger.debug(f"Queued {update_count} enhanced agent updates for step {step}")
            
        except Exception as e:
            logger.warning(f"Error queueing enhanced agent updates: {e}")
    
    def _extract_enhanced_agent_data(self, enhanced_manager: EnhancedAgentStateManager, 
                                   agent_id: str) -> Dict[str, Any]:
        """DEPRECATED: Use FrontendStateAdapter.convert() instead
        
        This method is redundant with the new unified architecture.
        FrontendStateAdapter provides zero-loss conversion with better performance.
        """
        try:
            state = enhanced_manager.state
            
            # Get recent memories summary
            recent_memories = []
            for memory in state.get("working_memory", [])[:3]:  # Last 3 memories
                if isinstance(memory, dict):
                    recent_memories.append({
                        "content": memory.get("content", ""),
                        "type": memory.get("type", "unknown"),
                        "importance": memory.get("importance", 0.5)
                    })
            
            # Get current emotional state
            emotional_state = state.get("emotional_state", {})
            
            # Get specialization info
            specialization_data = state.get("specialization")
            if hasattr(specialization_data, '__dict__'):
                # Convert dataclass to dict
                specialization = {
                    "current_role": specialization_data.current_role,
                    "skills": specialization_data.skills,
                    "expertise_level": specialization_data.expertise_level,
                    "role_consistency": specialization_data.role_consistency_score
                }
            else:
                specialization = {"current_role": "contestant", "skills": {}, "expertise_level": 0.5}
            
            # Get performance metrics
            performance_data = state.get("performance")
            if hasattr(performance_data, '__dict__'):
                performance = {
                    "coherence_score": performance_data.coherence_score,
                    "social_integration": performance_data.social_integration,
                    "memory_efficiency": performance_data.memory_efficiency,
                    "adaptation_rate": performance_data.adaptation_rate
                }
            else:
                performance = {"coherence_score": 0.5, "social_integration": 0.5}
            
            # Extract location information
            location_str = state.get("current_location", "villa_main_area")
            if "the_ville_" in location_str:
                coords = location_str.replace("the_ville_", "").split("_")
                if len(coords) >= 2:
                    try:
                        x, y = int(coords[0]), int(coords[1])
                        location = {"area": "the_ville", "room": "unknown", "x": x, "y": y}
                    except ValueError:
                        location = {"area": "villa", "room": "main_area", "x": 50, "y": 50}
                else:
                    location = {"area": "villa", "room": "main_area", "x": 50, "y": 50}
            else:
                location = {"area": "villa", "room": location_str, "x": 50, "y": 50}
            
            # Get conversation partners for social context
            conversation_partners = list(state.get("conversation_partners", set()))
            
            return {
                'agent_id': agent_id,
                'name': state.get("name", f"Agent {agent_id}"),
                'current_role': specialization.get("current_role", "contestant"),
                'specialization': specialization,
                'skills': specialization.get("skills", {}),
                'memory': {
                    'recent_events': recent_memories,
                    'total_memories': len(state.get("working_memory", [])),
                    'emotional_state': emotional_state
                },
                'location': location,
                'current_action': state.get("current_activity", "socializing"),
                'emotional_state': emotional_state,
                'performance_metrics': performance,
                'social_context': {
                    'conversation_partners': conversation_partners,
                    'recent_interactions': len(state.get("recent_interactions", []))
                },
                'personality_traits': state.get("personality_traits", {}),
                'goals': state.get("goals", []),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Error extracting enhanced agent data for {agent_id}: {e}")
            # Return basic fallback data
            return {
                'agent_id': agent_id,
                'name': enhanced_manager.name if hasattr(enhanced_manager, 'name') else f"Agent {agent_id}",
                'current_role': 'contestant',
                'specialization': {},
                'skills': {},
                'memory': {'recent_events': []},
                'location': {'area': 'villa', 'room': 'main_area', 'x': 50, 'y': 50},
                'current_action': 'unknown',
                'emotional_state': {},
                'timestamp': datetime.now().isoformat()
            }
    
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
            help='Frontend server URL (default: http://localhost:8001)'
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
        
        parser.add_argument(
            '--interactive',
            action='store_true',
            default=True,
            help='Run in interactive mode with step-by-step control (default: True)'
        )
        
        parser.add_argument(
            '--standalone',
            action='store_true',
            help='Use standalone mode with local reverie_core (no orchestration)'
        )
        
        parser.add_argument(
            '--no-interactive',
            action='store_true',
            help='Disable interactive mode, run full orchestration instead'
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
        
        # Handle different execution modes
        if args.run_steps:
            # Auto-run simulation steps (async) - explicit override
            await app.run_with_auto_steps(args.run_steps)
        elif args.no_interactive:
            # Explicitly disabled interactive mode - run full orchestration
            await app.run()
        else:
            # Default: Run in interactive/standalone mode (synchronous)
            app.run_interactive_mode()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())