"""
Bridge to connect with the existing Django simulation backend
"""

import json
import os
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

from .models import SimulationState, AgentState, EnvironmentState, Position, AgentRole, SimulationMode
from .config import Settings
from .movement_generator import MovementFileGenerator

logger = logging.getLogger(__name__)


class SimulationBridge:
    """Bridge to connect with Django backend simulation data"""
    
    def __init__(self):
        self.settings = Settings()
        # Force use current directory for now
        self.simulation_data_path = Path(".")
        
        # Always use local paths when running from frontend_service directory
        self.temp_storage_path = self.simulation_data_path / "temp_storage"
        self.storage_path = self.simulation_data_path / "storage"
            
        self._cached_state: Optional[SimulationState] = None
        self._last_update: Optional[datetime] = None
        
        # Initialize movement file generator
        self.movement_generator = MovementFileGenerator(str(self.storage_path))
    
    async def initialize(self):
        """Initialize the simulation bridge"""
        logger.info("Initializing simulation bridge")
        await self._validate_paths()
        await self._ensure_movement_files()
        await self._load_initial_state()
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up simulation bridge")
    
    async def _validate_paths(self):
        """Validate that required paths exist"""
        if not self.simulation_data_path.exists():
            logger.warning(f"Simulation data path does not exist: {self.simulation_data_path}")
            logger.info("Running in standalone mode without simulation data")
            # Create mock directories for Docker environment
            self.temp_storage_path.mkdir(parents=True, exist_ok=True)
            self.storage_path.mkdir(parents=True, exist_ok=True)
            return
        
        if not self.temp_storage_path.exists():
            logger.warning(f"Temp storage path does not exist: {self.temp_storage_path}")
            self.temp_storage_path.mkdir(parents=True, exist_ok=True)
        
        if not self.storage_path.exists():
            logger.warning(f"Storage path does not exist: {self.storage_path}")
            self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def _ensure_movement_files(self):
        """Ensure movement files exist for current simulation"""
        try:
            sim_code = await self._get_current_sim_code()
            if sim_code:
                logger.info(f"Checking movement files for simulation: {sim_code}")
                missing_files = self.movement_generator.get_missing_movement_files(sim_code)
                if missing_files:
                    logger.info(f"Creating {len(missing_files)} missing movement files")
                    await self.movement_generator.generate_missing_files_for_simulation(sim_code)
                else:
                    logger.info("All required movement files exist")
        except Exception as e:
            logger.warning(f"Error ensuring movement files: {e}")
            # Continue without failing - frontend can work without movement files
    
    async def _load_initial_state(self):
        """Load initial simulation state"""
        try:
            sim_code = await self._get_current_sim_code()
            if sim_code:
                self._cached_state = await self._build_simulation_state(sim_code)
                self._last_update = datetime.now()
                logger.info(f"Loaded initial state for simulation: {sim_code}")
            else:
                logger.info("No current simulation code found, creating demo state")
                self._cached_state = await self._create_demo_state()
                self._last_update = datetime.now()
        except Exception as e:
            logger.warning(f"Error loading initial state: {e}")
            logger.info("Creating demo state for standalone mode")
            self._cached_state = await self._create_demo_state()
            self._last_update = datetime.now()
    
    async def _get_current_sim_code(self) -> Optional[str]:
        """Get current simulation code from temp storage"""
        try:
            curr_sim_file = self.temp_storage_path / "curr_sim_code.json"
            if curr_sim_file.exists():
                async with aiofiles.open(curr_sim_file, 'r') as f:
                    data = json.loads(await f.read())
                    return data.get("sim_code")
        except Exception as e:
            logger.error(f"Error reading current sim code: {e}")
        return None
    
    async def _get_current_step(self) -> Optional[int]:
        """Get current simulation step"""
        try:
            curr_step_file = self.temp_storage_path / "curr_step.json"
            if curr_step_file.exists():
                async with aiofiles.open(curr_step_file, 'r') as f:
                    data = json.loads(await f.read())
                    step = data.get("step", 0)
                
                # Delete the file after reading (Django behavior)
                try:
                    curr_step_file.unlink()
                    logger.info(f"Deleted curr_step.json after reading step {step}")
                except Exception as e:
                    logger.warning(f"Could not delete curr_step.json: {e}")
                
                return step
        except Exception as e:
            logger.error(f"Error reading current step: {e}")
        return 0
    
    async def _load_simulation_metadata(self, sim_code: str) -> Dict[str, Any]:
        """Load simulation metadata from meta.json"""
        try:
            meta_file = self.storage_path / sim_code / "reverie" / "meta.json"
            if meta_file.exists():
                async with aiofiles.open(meta_file, 'r') as f:
                    return json.loads(await f.read())
        except Exception as e:
            logger.error(f"Error loading metadata for {sim_code}: {e}")
        return {}
    
    async def _load_agent_states(self, sim_code: str, step: int) -> List[AgentState]:
        """Load agent states from storage - matches Django find_filenames logic"""
        agents = []
        try:
            personas_path = self.storage_path / sim_code / "personas"
            if personas_path.exists():
                # Use Django-compatible directory listing (matches find_filenames)
                for persona_dir in personas_path.iterdir():
                    if (persona_dir.is_dir() and 
                        not persona_dir.name.startswith('.') and
                        persona_dir.name.strip() != ""):
                        agent = await self._load_single_agent(persona_dir, step, sim_code)
                        if agent:
                            agents.append(agent)
                logger.info(f"Loaded {len(agents)} agents from {personas_path}")
        except Exception as e:
            logger.error(f"Error loading agent states: {e}")
        
        return agents
    
    async def _load_single_agent(self, persona_dir: Path, step: int, sim_code: str) -> Optional[AgentState]:
        """Load single agent state"""
        try:
            agent_name = persona_dir.name
            
            # Load agent position from environment data
            position = await self._load_agent_position(agent_name, step, sim_code)
            
            # Load agent scratch data if available
            scratch_file = persona_dir / "bootstrap_memory" / "scratch.json"
            scratch_data = {}
            if scratch_file.exists():
                async with aiofiles.open(scratch_file, 'r') as f:
                    scratch_data = json.loads(await f.read())
            
            # Determine agent role (default to contestant for dating show)
            role = AgentRole.CONTESTANT
            if "host" in agent_name.lower():
                role = AgentRole.HOST
            
            return AgentState(
                name=agent_name,
                role=role,
                position=position or Position(x=0, y=0),
                current_action=scratch_data.get("action", {}).get("description"),
                current_location=scratch_data.get("curr_tile", ["Unknown"])[0] if scratch_data.get("curr_tile") else "Unknown",
                emotional_state={},
                relationship_scores={},
                dialogue_history=[],
                last_updated=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error loading agent {persona_dir.name}: {e}")
            return None
    
    async def _load_agent_position(self, agent_name: str, step: int, sim_code: str) -> Optional[Position]:
        """Load agent position from environment data"""
        try:
            
            env_dir = self.storage_path / sim_code / "environment"
            if not env_dir.exists():
                return None
            
            # Find the latest environment file
            env_files = [f for f in env_dir.glob("*.json") if f.stem.isdigit()]
            if not env_files:
                return None
            
            latest_file = max(env_files, key=lambda f: int(f.stem))
            
            async with aiofiles.open(latest_file, 'r') as f:
                env_data = json.loads(await f.read())
                
            agent_data = env_data.get(agent_name)
            if agent_data:
                return Position(
                    x=agent_data.get("x", 0),
                    y=agent_data.get("y", 0),
                    sector=agent_data.get("sector")
                )
        except Exception as e:
            logger.error(f"Error loading position for {agent_name}: {e}")
        
        return Position(x=0, y=0)
    
    async def _build_simulation_state(self, sim_code: str) -> SimulationState:
        """Build complete simulation state"""
        current_step = await self._get_current_step() or 0
        metadata = await self._load_simulation_metadata(sim_code)
        agents = await self._load_agent_states(sim_code, current_step)
        
        # Determine simulation mode
        mode = SimulationMode.STOPPED
        curr_step_file = self.temp_storage_path / "curr_step.json"
        if curr_step_file.exists():
            mode = SimulationMode.RUNNING
        
        return SimulationState(
            sim_code=sim_code,
            current_step=current_step,
            mode=mode,
            agents=agents,
            environment=EnvironmentState(),
            start_time=datetime.now(),  # This should be parsed from metadata
            current_time=datetime.now(),
            metadata=metadata
        )
    
    async def get_current_simulation_state(self) -> Optional[SimulationState]:
        """Get current simulation state"""
        try:
            # Check if we need to refresh the cached state
            if (not self._cached_state or 
                not self._last_update or 
                (datetime.now() - self._last_update).seconds > self.settings.auto_refresh_interval):
                
                sim_code = await self._get_current_sim_code()
                if sim_code:
                    self._cached_state = await self._build_simulation_state(sim_code)
                    self._last_update = datetime.now()
            
            return self._cached_state
        except Exception as e:
            logger.error(f"Error getting simulation state: {e}")
            return None
    
    async def get_all_agents(self) -> List[AgentState]:
        """Get all agent states"""
        state = await self.get_current_simulation_state()
        return state.agents if state else []
    
    async def get_agent_state(self, agent_name: str) -> Optional[AgentState]:
        """Get specific agent state"""
        agents = await self.get_all_agents()
        for agent in agents:
            if agent.name == agent_name:
                return agent
        return None
    
    async def advance_simulation(self) -> Dict[str, Any]:
        """Advance simulation by one step"""
        # This would interface with the Django backend
        # For now, we'll simulate the advancement
        try:
            current_step = await self._get_current_step() or 0
            new_step = current_step + 1
            
            # In a real implementation, this would trigger the Django backend
            # to advance the simulation and then reload the state
            
            # Update cached state
            sim_code = await self._get_current_sim_code()
            if sim_code:
                self._cached_state = await self._build_simulation_state(sim_code)
                self._last_update = datetime.now()
            
            return {
                "success": True,
                "step": new_step,
                "message": f"Advanced to step {new_step}"
            }
        except Exception as e:
            logger.error(f"Error advancing simulation: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _create_demo_state(self) -> SimulationState:
        """Create demo simulation state for standalone mode"""
        demo_agents = []
        
        # Create demo dating show contestants
        contestant_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        
        for i, name in enumerate(contestant_names):
            # Use more realistic Phaser.js coordinates (tile-based system)
            agent = AgentState(
                name=name,
                role=AgentRole.CONTESTANT,
                position=Position(x=30 + (i * 15) % 100, y=20 + (i // 3) * 20),
                current_action=["chatting", "cooking", "exercising", "relaxing"][i % 4],
                current_location=["Hobbs Cafe", "The Rose and Crown Pub", "Harvey Oak Supply Store", "The Willows Market and Pharmacy"][i % 4],
                emotional_state={"happiness": 0.7 + (i * 0.05), "excitement": 0.6},
                relationship_scores={other: 0.5 + (i * 0.1) % 0.5 for other in contestant_names if other != name},
                dialogue_history=[f"{name} introduced themselves"],
                last_updated=datetime.now()
            )
            demo_agents.append(agent)
        
        # Add a host
        host = AgentState(
            name="Host Alex",
            role=AgentRole.HOST,
            position=Position(x=50, y=50),
            current_action="hosting",
            current_location="The Rose and Crown Pub",
            emotional_state={"confidence": 0.9, "enthusiasm": 0.8},
            relationship_scores={},
            dialogue_history=["Welcome to the dating show!"],
            last_updated=datetime.now()
        )
        demo_agents.append(host)
        
        return SimulationState(
            sim_code="demo_simulation",
            current_step=1,
            mode=SimulationMode.RUNNING,
            agents=demo_agents,
            environment=EnvironmentState(
                time_of_day="evening",
                weather="sunny",
                active_events=["Welcome party", "Speed dating round"]
            ),
            start_time=datetime.now(),
            current_time=datetime.now(),
            metadata={"demo": True, "version": "1.0"}
        )