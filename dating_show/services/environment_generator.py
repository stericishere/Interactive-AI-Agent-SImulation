"""
Environment File Generation Service
Handles creation and management of environment and movement files for the simulation
Ensures proper directory structure and file format compatibility
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentGenerator:
    """
    Generates and manages environment and movement files for the simulation
    Handles both initial bootstrap and ongoing step-by-step generation
    """
    
    def __init__(self, storage_base_path: str = None):
        # Default to the expected structure if not provided
        if storage_base_path is None:
            storage_base_path = "/Applications/Projects/Open source/generative_agents/dating_show_env/frontend_service/storage"
        
        self.storage_base_path = Path(storage_base_path)
        self.simulation_configs = {}
        
        # Dating show emoji pool for agent expressions
        self.dating_show_emojis = [
            "💕", "🌹", "💬", "😊", "🥰", "💭", "✨", "🌟", "💃", "🎭", 
            "🔥", "🌺", "📚", "🎵", "🍃", "⚡", "🎮", "🌸", "🛠️", "🎬",
            "🍳", "💼", "🌙", "🏈", "🎓", "😎", "💪", "🌊", "🎨"
        ]
        
        # Dating show activity descriptions
        self.dating_activities = [
            "socializing with other contestants",
            "exploring the villa",
            "having conversations by the pool", 
            "preparing for the next challenge",
            "reflecting on relationships",
            "enjoying villa amenities",
            "getting ready for dates",
            "sharing stories with housemates",
            "relaxing in the lounge",
            "participating in group activities",
            "building romantic connections",
            "engaging in deep conversations",
            "working out in the gym",
            "cooking together",
            "planning group events"
        ]
    
    def initialize_simulation_storage(self, sim_code: str, agent_names: List[str]) -> bool:
        """
        Initialize complete storage structure for a new simulation
        
        Args:
            sim_code: Simulation identifier
            agent_names: List of agent names to initialize
            
        Returns:
            bool: Success status
        """
        try:
            sim_path = self.storage_base_path / sim_code
            
            # Create directory structure
            self._create_directory_structure(sim_path)
            
            # Store simulation config
            self.simulation_configs[sim_code] = {
                'agent_names': agent_names,
                'current_step': 0,
                'initialized': True,
                'path': str(sim_path)
            }
            
            # Generate initial environment file (step 0)
            self._generate_environment_file(sim_code, 0, agent_names)
            
            # Generate initial movement file (step 0)
            self._generate_movement_file(sim_code, 0, agent_names)
            
            # Initialize persona directories and files
            self._initialize_persona_storage(sim_path, agent_names)
            
            # Create meta.json for reverie compatibility
            self._create_reverie_meta(sim_path, sim_code)
            
            logger.info(f"Successfully initialized simulation storage for {sim_code} with {len(agent_names)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing simulation storage for {sim_code}: {e}")
            return False
    
    def _create_directory_structure(self, sim_path: Path):
        """Create the required directory structure"""
        directories = ['environment', 'movement', 'personas', 'reverie']
        
        for directory in directories:
            dir_path = sim_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Created directory structure at {sim_path}")
    
    def _initialize_persona_storage(self, sim_path: Path, agent_names: List[str]):
        """Initialize persona storage directories and basic files"""
        personas_path = sim_path / 'personas'
        
        for agent_name in agent_names:
            agent_path = personas_path / agent_name / 'bootstrap_memory'
            agent_path.mkdir(parents=True, exist_ok=True)
            
            # Create basic scratch.json for persona
            scratch_data = {
                "daily_plan_req": random.choice(self.dating_activities),
                "curr_tile": ["the_ville", "villa_area"],
                "chat": "",
                "curr_time": datetime.now().isoformat()
            }
            
            scratch_file = agent_path / 'scratch.json'
            with open(scratch_file, 'w') as f:
                json.dump(scratch_data, f, indent=2)
            
            # Create basic spatial_memory.json
            spatial_data = {
                "world": "the_ville",
                "sectors": {},
                "arenas": {},
                "game_objects": {}
            }
            
            spatial_file = agent_path / 'spatial_memory.json'
            with open(spatial_file, 'w') as f:
                json.dump(spatial_data, f, indent=2)
    
    def _create_reverie_meta(self, sim_path: Path, sim_code: str):
        """Create meta.json for reverie compatibility"""
        meta_data = {
            "sim_code": sim_code,
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "maze": "the_ville",
            "step": 0
        }
        
        meta_file = sim_path / 'reverie' / 'meta.json'
        with open(meta_file, 'w') as f:
            json.dump(meta_data, f, indent=2)
    
    def generate_next_step_files(self, sim_code: str, step: int, 
                                agent_states: Dict[str, Any]) -> bool:
        """
        Generate environment and movement files for the next simulation step
        
        Args:
            sim_code: Simulation identifier
            step: Step number to generate
            agent_states: Current agent state data
            
        Returns:
            bool: Success status
        """
        try:
            if sim_code not in self.simulation_configs:
                logger.error(f"Simulation {sim_code} not initialized")
                return False
            
            # Generate environment file
            success_env = self._generate_environment_file(sim_code, step, None, agent_states)
            
            # Generate movement file
            success_mov = self._generate_movement_file(sim_code, step, None, agent_states)
            
            if success_env and success_mov:
                self.simulation_configs[sim_code]['current_step'] = step
                logger.info(f"Generated step {step} files for simulation {sim_code}")
                return True
            else:
                logger.error(f"Failed to generate step {step} files for simulation {sim_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error generating step {step} files for {sim_code}: {e}")
            return False
    
    def _generate_environment_file(self, sim_code: str, step: int, 
                                 agent_names: Optional[List[str]] = None,
                                 agent_states: Optional[Dict[str, Any]] = None) -> bool:
        """Generate environment.json file for a specific step"""
        try:
            sim_path = self.storage_base_path / sim_code
            env_file = sim_path / 'environment' / f'{step}.json'
            
            environment_data = {}
            
            if agent_states:
                # Use provided agent states
                for agent_id, state in agent_states.items():
                    if hasattr(state, 'name'):
                        name = state.name
                        x = getattr(state, 'last_position', (50, 50))[0] if hasattr(state, 'last_position') else 50
                        y = getattr(state, 'last_position', (50, 50))[1] if hasattr(state, 'last_position') else 50
                    elif isinstance(state, dict):
                        name = state.get('name', agent_id)
                        position = state.get('position', {'x': 50, 'y': 50})
                        x = position.get('x', 50) if isinstance(position, dict) else 50
                        y = position.get('y', 50) if isinstance(position, dict) else 50
                    else:
                        continue
                        
                    environment_data[name] = {
                        "maze": "the_ville",
                        "x": int(x),
                        "y": int(y)
                    }
            elif agent_names:
                # Generate initial positions for new simulation
                positions = self._generate_initial_positions(len(agent_names))
                for i, name in enumerate(agent_names):
                    x, y = positions[i]
                    environment_data[name] = {
                        "maze": "the_ville",
                        "x": x,
                        "y": y
                    }
            else:
                logger.error("No agent data provided for environment file generation")
                return False
            
            # Write environment file
            with open(env_file, 'w') as f:
                json.dump(environment_data, f, indent=2)
            
            logger.debug(f"Generated environment file: {env_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating environment file for step {step}: {e}")
            return False
    
    def _generate_movement_file(self, sim_code: str, step: int,
                              agent_names: Optional[List[str]] = None,
                              agent_states: Optional[Dict[str, Any]] = None) -> bool:
        """Generate movement.json file for a specific step"""
        try:
            sim_path = self.storage_base_path / sim_code
            mov_file = sim_path / 'movement' / f'{step}.json'
            
            movement_data = {
                "persona": {},
                "meta": {
                    "curr_time": datetime.now().strftime("%B %d, %Y, %H:%M:%S")
                }
            }
            
            if agent_states:
                # Use provided agent states
                for agent_id, state in agent_states.items():
                    if hasattr(state, 'name'):
                        name = state.name
                        position = getattr(state, 'last_position', (50, 50))
                        if hasattr(state, 'scratch'):
                            action = getattr(state.scratch, 'daily_plan_req', 'socializing')
                        else:
                            action = random.choice(self.dating_activities)
                    elif isinstance(state, dict):
                        name = state.get('name', agent_id)
                        position_data = state.get('position', {'x': 50, 'y': 50})
                        if isinstance(position_data, dict):
                            position = (position_data.get('x', 50), position_data.get('y', 50))
                        else:
                            position = (50, 50)
                        action = state.get('current_action', random.choice(self.dating_activities))
                    else:
                        continue
                    
                    movement_data["persona"][name] = {
                        "movement": [int(position[0]), int(position[1])],
                        "pronunciatio": random.choice(self.dating_show_emojis),
                        "description": action,
                        "chat": None
                    }
            elif agent_names:
                # Generate initial movement data for new simulation
                positions = self._generate_initial_positions(len(agent_names))
                for i, name in enumerate(agent_names):
                    x, y = positions[i]
                    movement_data["persona"][name] = {
                        "movement": [x, y],
                        "pronunciatio": random.choice(self.dating_show_emojis),
                        "description": random.choice(self.dating_activities),
                        "chat": None
                    }
            else:
                logger.error("No agent data provided for movement file generation")
                return False
            
            # Ensure the movement directory exists before writing
            mov_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write movement file
            with open(mov_file, 'w') as f:
                json.dump(movement_data, f, indent=2)
            
            logger.debug(f"Generated movement file: {mov_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating movement file for step {step}: {e}")
            return False
    
    def _generate_initial_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        """Generate well-distributed initial positions for agents"""
        positions = []
        
        # Define villa areas with good spread
        areas = [
            # Pool area
            [(16, 18), (26, 18), (36, 18)],
            # Lounge area
            [(16, 32), (26, 32), (36, 32)],
            # Garden area
            [(53, 14), (65, 19), (72, 14), (86, 18), (94, 18)],
            # Kitchen/dining area
            [(126, 46), (123, 57), (118, 61), (107, 62)],
            # Bedroom area
            [(90, 74), (91, 74), (93, 74), (72, 74), (73, 74)],
            # Common area
            [(54, 74), (57, 74), (20, 65), (28, 65), (36, 65), (37, 65)]
        ]
        
        # Flatten areas into a single list
        all_positions = []
        for area in areas:
            all_positions.extend(area)
        
        # Use positions in order, with some randomization
        random.shuffle(all_positions)
        
        # If we need more positions than available, generate additional ones
        if num_agents > len(all_positions):
            logger.warning(f"Need {num_agents} positions but only have {len(all_positions)} predefined")
            # Generate additional random positions within villa bounds
            for i in range(num_agents - len(all_positions)):
                x = random.randint(15, 130)
                y = random.randint(15, 80)
                all_positions.append((x, y))
        
        return all_positions[:num_agents]
    
    def ensure_step_files_exist(self, sim_code: str, step: int) -> bool:
        """
        Ensure that environment and movement files exist for a given step
        Generate them if missing
        
        Args:
            sim_code: Simulation identifier
            step: Step number to check/generate
            
        Returns:
            bool: Success status
        """
        try:
            sim_path = self.storage_base_path / sim_code
            env_file = sim_path / 'environment' / f'{step}.json'
            mov_file = sim_path / 'movement' / f'{step}.json'
            
            files_exist = env_file.exists() and mov_file.exists()
            
            if files_exist:
                logger.debug(f"Step {step} files already exist for {sim_code}")
                return True
            
            # Files missing, try to generate them
            logger.info(f"Step {step} files missing for {sim_code}, attempting to generate...")
            
            if sim_code not in self.simulation_configs:
                logger.error(f"Cannot generate files: simulation {sim_code} not initialized")
                return False
            
            # Try to get agent data from previous step or simulation config
            agent_names = self.simulation_configs[sim_code]['agent_names']
            
            # If this is step > 0, try to copy and modify from previous step
            if step > 0:
                prev_env_file = sim_path / 'environment' / f'{step-1}.json'
                prev_mov_file = sim_path / 'movement' / f'{step-1}.json'
                
                if prev_env_file.exists() and prev_mov_file.exists():
                    # Load previous step data and generate variations
                    return self._generate_step_from_previous(sim_code, step, step-1)
            
            # Fallback: generate with basic agent names
            return self._generate_environment_file(sim_code, step, agent_names) and \
                   self._generate_movement_file(sim_code, step, agent_names)
                   
        except Exception as e:
            logger.error(f"Error ensuring step {step} files exist for {sim_code}: {e}")
            return False
    
    def _generate_step_from_previous(self, sim_code: str, current_step: int, 
                                   previous_step: int) -> bool:
        """Generate current step files based on previous step with small modifications"""
        try:
            sim_path = self.storage_base_path / sim_code
            
            # Load previous environment data
            prev_env_file = sim_path / 'environment' / f'{previous_step}.json'
            with open(prev_env_file, 'r') as f:
                prev_env_data = json.load(f)
            
            # Load previous movement data
            prev_mov_file = sim_path / 'movement' / f'{previous_step}.json'
            with open(prev_mov_file, 'r') as f:
                prev_mov_data = json.load(f)
            
            # Generate new environment data with small position changes
            new_env_data = {}
            for name, data in prev_env_data.items():
                # Small random movement
                new_x = max(10, min(130, data['x'] + random.randint(-3, 3)))
                new_y = max(10, min(80, data['y'] + random.randint(-3, 3)))
                
                new_env_data[name] = {
                    "maze": "the_ville",
                    "x": new_x,
                    "y": new_y
                }
            
            # Generate new movement data
            new_mov_data = {
                "persona": {},
                "meta": {
                    "curr_time": datetime.now().strftime("%B %d, %Y, %H:%M:%S")
                }
            }
            
            for name, env_data in new_env_data.items():
                new_mov_data["persona"][name] = {
                    "movement": [env_data['x'], env_data['y']],
                    "pronunciatio": random.choice(self.dating_show_emojis),
                    "description": random.choice(self.dating_activities),
                    "chat": None
                }
            
            # Write new files
            new_env_file = sim_path / 'environment' / f'{current_step}.json'
            with open(new_env_file, 'w') as f:
                json.dump(new_env_data, f, indent=2)
            
            new_mov_file = sim_path / 'movement' / f'{current_step}.json'
            with open(new_mov_file, 'w') as f:
                json.dump(new_mov_data, f, indent=2)
            
            logger.info(f"Generated step {current_step} files from step {previous_step} for {sim_code}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating step {current_step} from previous step: {e}")
            return False
    
    def get_simulation_info(self, sim_code: str) -> Optional[Dict[str, Any]]:
        """Get information about a simulation"""
        return self.simulation_configs.get(sim_code)
    
    def cleanup_old_files(self, sim_code: str, keep_last_n: int = 10):
        """Clean up old environment and movement files, keeping only the last N"""
        try:
            sim_path = self.storage_base_path / sim_code
            
            for directory in ['environment', 'movement']:
                dir_path = sim_path / directory
                if not dir_path.exists():
                    continue
                
                # Get all json files and sort by step number
                files = []
                for file_path in dir_path.glob('*.json'):
                    try:
                        step_num = int(file_path.stem)
                        files.append((step_num, file_path))
                    except ValueError:
                        continue
                
                files.sort(key=lambda x: x[0])
                
                # Remove old files
                if len(files) > keep_last_n:
                    files_to_remove = files[:-keep_last_n]
                    for step_num, file_path in files_to_remove:
                        file_path.unlink()
                        logger.debug(f"Removed old file: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error cleaning up old files for {sim_code}: {e}")


# Global generator instance
_env_generator: Optional[EnvironmentGenerator] = None


def get_environment_generator() -> EnvironmentGenerator:
    """Get or create global environment generator instance"""
    global _env_generator
    if _env_generator is None:
        _env_generator = EnvironmentGenerator()
    return _env_generator


def ensure_simulation_files(sim_code: str, step: int) -> bool:
    """Quick helper to ensure simulation files exist for a step"""
    generator = get_environment_generator()
    return generator.ensure_step_files_exist(sim_code, step)