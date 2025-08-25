"""
Movement File Generator Utility
Generates missing movement.json files for frontend visualization
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MovementFileGenerator:
    """Utility to generate movement files for frontend visualization"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        
        # Dating show activities for realistic agent behavior
        self.dating_activities = [
            "socializing", "mingling", "chatting", "connecting", "bonding",
            "exploring", "networking", "observing", "conversing", "discussing",
            "interacting", "engaging", "participating", "analyzing", "strategizing"
        ]
        
        # Location options for dating show
        self.dating_locations = [
            "dating show location", "villa entrance", "pool area", "garden",
            "cocktail area", "fire pit", "terrace", "living room"
        ]
    
    async def ensure_movement_files_exist(self, sim_code: str, max_step: int = 10) -> bool:
        """Ensure movement files exist for all steps up to max_step"""
        try:
            sim_path = self.storage_path / sim_code
            movement_dir = sim_path / "movement"
            
            # Create movement directory if it doesn't exist
            movement_dir.mkdir(parents=True, exist_ok=True)
            
            # Get environment data for agent positions
            env_data = await self._load_environment_data(sim_code)
            if not env_data:
                logger.warning(f"No environment data found for {sim_code}")
                return False
            
            # Generate missing movement files
            files_created = 0
            for step in range(max_step + 1):
                movement_file = movement_dir / f"{step}.json"
                if not movement_file.exists():
                    success = await self._generate_movement_file(sim_code, step, env_data)
                    if success:
                        files_created += 1
                        logger.info(f"Generated movement file for step {step}")
            
            logger.info(f"Created {files_created} movement files for {sim_code}")
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring movement files exist: {e}")
            return False
    
    async def _load_environment_data(self, sim_code: str) -> Optional[Dict[str, Any]]:
        """Load environment data for agent positions"""
        try:
            sim_path = self.storage_path / sim_code
            env_dir = sim_path / "environment"
            
            if not env_dir.exists():
                return None
            
            # Find the latest environment file
            env_files = [f for f in env_dir.glob("*.json") if f.stem.isdigit()]
            if not env_files:
                return None
            
            latest_file = max(env_files, key=lambda f: int(f.stem))
            
            with open(latest_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading environment data: {e}")
            return None
    
    async def _generate_movement_file(self, sim_code: str, step: int, env_data: Dict[str, Any]) -> bool:
        """Generate a single movement file for a specific step"""
        try:
            sim_path = self.storage_path / sim_code
            movement_file = sim_path / "movement" / f"{step}.json"
            
            # Build movement data structure
            movement_data = {
                "persona": {},
                "meta": {
                    "curr_time": datetime.now().strftime("%B %d, %Y, %H:%M:%S")
                }
            }
            
            # Generate movement data for each agent
            for agent_name, agent_env_data in env_data.items():
                if isinstance(agent_env_data, dict) and 'x' in agent_env_data and 'y' in agent_env_data:
                    # Add some random variation for different steps
                    base_x = agent_env_data['x']
                    base_y = agent_env_data['y']
                    
                    # Small random movement for animation (max 5 tiles)
                    # Use proper map boundaries: 140x100 (0-139, 0-99)
                    max_x = 139  # maze_width - 1
                    max_y = 99   # maze_height - 1
                    
                    if step > 0:
                        x_offset = random.randint(-2, 2)
                        y_offset = random.randint(-2, 2)
                        pos_x = max(0, min(max_x, base_x + x_offset))
                        pos_y = max(0, min(max_y, base_y + y_offset))
                    else:
                        # Ensure initial positions are also within bounds
                        pos_x = max(0, min(max_x, base_x))
                        pos_y = max(0, min(max_y, base_y))
                    
                    # Random activity and location
                    activity = random.choice(self.dating_activities)
                    location = random.choice(self.dating_locations)
                    
                    movement_data["persona"][agent_name] = {
                        "movement": [pos_x, pos_y],
                        "pronunciatio": activity,
                        "description": f"{agent_name} is {activity} at {location}",
                        "chat": "",
                        "scratch": {
                            "curr_tile": [location],
                            "daily_plan_req": activity
                        }
                    }
            
            # Write movement file
            with open(movement_file, 'w') as f:
                json.dump(movement_data, f, indent=2)
            
            logger.debug(f"Generated movement file: {movement_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating movement file for step {step}: {e}")
            return False
    
    async def generate_missing_files_for_simulation(self, sim_code: str) -> bool:
        """Generate all missing files for a simulation"""
        try:
            # Check what step files already exist
            sim_path = self.storage_path / sim_code
            movement_dir = sim_path / "movement"
            
            if not movement_dir.exists():
                movement_dir.mkdir(parents=True, exist_ok=True)
            
            # Find highest step number in environment files
            env_dir = sim_path / "environment"
            max_step = 0
            
            if env_dir.exists():
                env_files = [f for f in env_dir.glob("*.json") if f.stem.isdigit()]
                if env_files:
                    max_step = max(int(f.stem) for f in env_files)
            
            # Generate movement files up to the highest step
            return await self.ensure_movement_files_exist(sim_code, max_step)
            
        except Exception as e:
            logger.error(f"Error generating missing files for {sim_code}: {e}")
            return False
    
    def get_missing_movement_files(self, sim_code: str, max_step: int = 10) -> List[int]:
        """Get list of missing movement file step numbers"""
        missing_steps = []
        try:
            sim_path = self.storage_path / sim_code
            movement_dir = sim_path / "movement"
            
            if not movement_dir.exists():
                return list(range(max_step + 1))
            
            for step in range(max_step + 1):
                movement_file = movement_dir / f"{step}.json"
                if not movement_file.exists():
                    missing_steps.append(step)
            
        except Exception as e:
            logger.error(f"Error checking missing movement files: {e}")
            
        return missing_steps
    
    async def validate_movement_file(self, sim_code: str, step: int) -> bool:
        """Validate that a movement file exists and has correct structure"""
        try:
            sim_path = self.storage_path / sim_code
            movement_file = sim_path / "movement" / f"{step}.json"
            
            if not movement_file.exists():
                return False
            
            with open(movement_file, 'r') as f:
                data = json.load(f)
            
            # Check required structure
            if "persona" not in data or "meta" not in data:
                return False
            
            # Check that personas have movement data
            for persona_name, persona_data in data["persona"].items():
                if "movement" not in persona_data:
                    return False
                if not isinstance(persona_data["movement"], list) or len(persona_data["movement"]) != 2:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating movement file {step}: {e}")
            return False


# Utility functions for direct usage
async def create_initial_movement_files(storage_path: str, sim_code: str = "dating_show_25_agents"):
    """Create initial movement files for a simulation"""
    generator = MovementFileGenerator(storage_path)
    return await generator.generate_missing_files_for_simulation(sim_code)


async def ensure_movement_directory_exists(storage_path: str, sim_code: str = "dating_show_25_agents"):
    """Ensure movement directory exists for a simulation"""
    sim_path = Path(storage_path) / sim_code / "movement"
    sim_path.mkdir(parents=True, exist_ok=True)
    return sim_path.exists()