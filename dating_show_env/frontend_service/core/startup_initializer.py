"""
Startup Initialization Script
Ensures all required files and directories exist for frontend operation
"""

import json
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .config import Settings
from .movement_generator import MovementFileGenerator

logger = logging.getLogger(__name__)


class StartupInitializer:
    """Initialize all required files and directories for frontend operation"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.simulation_data_path = Path(self.settings.simulation_data_path)
        
        # Determine storage paths
        if not self.simulation_data_path.exists():
            self.temp_storage_path = Path("/tmp/temp_storage")
            self.storage_path = Path("/tmp/storage")
        else:
            self.temp_storage_path = self.simulation_data_path / "temp_storage"
            self.storage_path = self.simulation_data_path / "storage"
        
        self.movement_generator = MovementFileGenerator(str(self.storage_path))
    
    async def initialize_all(self, sim_code: str = "dating_show_25_agents") -> bool:
        """Initialize all required files and directories"""
        try:
            logger.info("Starting frontend initialization...")
            
            # Step 1: Create basic directory structure
            await self._create_directories(sim_code)
            
            # Step 2: Initialize current simulation files
            await self._initialize_current_sim_files(sim_code)
            
            # Step 3: Ensure movement files exist
            await self._ensure_movement_files(sim_code)
            
            # Step 4: Validate critical files exist
            validation_result = await self._validate_critical_files(sim_code)
            
            if validation_result:
                logger.info("Frontend initialization completed successfully")
            else:
                logger.warning("Frontend initialization completed with warnings")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error during frontend initialization: {e}")
            return False
    
    async def _create_directories(self, sim_code: str):
        """Create all required directories"""
        directories = [
            self.temp_storage_path,
            self.storage_path,
            self.storage_path / sim_code,
            self.storage_path / sim_code / "movement",
            self.storage_path / sim_code / "environment",
            self.storage_path / sim_code / "personas",
            self.storage_path / sim_code / "reverie"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    async def _initialize_current_sim_files(self, sim_code: str):
        """Initialize current simulation tracking files"""
        try:
            # Create curr_sim_code.json
            curr_sim_file = self.temp_storage_path / "curr_sim_code.json"
            if not curr_sim_file.exists():
                sim_data = {
                    "sim_code": sim_code,
                    "created_at": datetime.now().isoformat(),
                    "status": "initialized"
                }
                
                with open(curr_sim_file, 'w') as f:
                    json.dump(sim_data, f, indent=2)
                
                logger.info(f"Created curr_sim_code.json for {sim_code}")
            
            # Create initial curr_step.json (will be consumed by simulation bridge)
            curr_step_file = self.temp_storage_path / "curr_step.json"
            if not curr_step_file.exists():
                step_data = {
                    "step": 0,
                    "created_at": datetime.now().isoformat()
                }
                
                with open(curr_step_file, 'w') as f:
                    json.dump(step_data, f, indent=2)
                
                logger.info("Created initial curr_step.json")
            
        except Exception as e:
            logger.error(f"Error initializing current sim files: {e}")
    
    async def _ensure_movement_files(self, sim_code: str):
        """Ensure movement files exist"""
        try:
            logger.info("Ensuring movement files exist...")
            success = await self.movement_generator.generate_missing_files_for_simulation(sim_code)
            if success:
                logger.info("Movement files initialization complete")
            else:
                logger.warning("Movement files initialization had issues")
        except Exception as e:
            logger.error(f"Error ensuring movement files: {e}")
    
    async def _validate_critical_files(self, sim_code: str) -> bool:
        """Validate that critical files exist and are properly formatted"""
        validation_results = []
        
        try:
            # Check curr_sim_code.json
            curr_sim_file = self.temp_storage_path / "curr_sim_code.json"
            if curr_sim_file.exists():
                with open(curr_sim_file, 'r') as f:
                    data = json.load(f)
                    if "sim_code" in data:
                        validation_results.append(True)
                        logger.debug("✓ curr_sim_code.json is valid")
                    else:
                        validation_results.append(False)
                        logger.warning("✗ curr_sim_code.json missing sim_code field")
            else:
                validation_results.append(False)
                logger.warning("✗ curr_sim_code.json does not exist")
            
            # Check movement directory and initial file
            movement_dir = self.storage_path / sim_code / "movement"
            movement_file_0 = movement_dir / "0.json"
            
            if movement_dir.exists() and movement_file_0.exists():
                # Validate movement file structure
                with open(movement_file_0, 'r') as f:
                    data = json.load(f)
                    if "persona" in data and "meta" in data:
                        validation_results.append(True)
                        logger.debug("✓ Movement files are valid")
                    else:
                        validation_results.append(False)
                        logger.warning("✗ Movement file structure invalid")
            else:
                validation_results.append(False)
                logger.warning("✗ Movement files missing")
            
            # Check simulation directory structure
            sim_dir = self.storage_path / sim_code
            required_subdirs = ["movement", "environment", "personas", "reverie"]
            
            for subdir in required_subdirs:
                if (sim_dir / subdir).exists():
                    validation_results.append(True)
                    logger.debug(f"✓ {subdir} directory exists")
                else:
                    validation_results.append(False)
                    logger.warning(f"✗ {subdir} directory missing")
            
            success_rate = sum(validation_results) / len(validation_results)
            logger.info(f"Validation complete: {sum(validation_results)}/{len(validation_results)} checks passed ({success_rate:.1%})")
            
            return success_rate >= 0.8  # 80% success rate required
            
        except Exception as e:
            logger.error(f"Error validating critical files: {e}")
            return False
    
    async def create_demo_environment_file(self, sim_code: str) -> bool:
        """Create demo environment file if none exists"""
        try:
            env_dir = self.storage_path / sim_code / "environment"
            env_file_0 = env_dir / "0.json"
            
            if not env_file_0.exists():
                # Create demo environment data with 25 agents
                demo_agents = [
                    "Latoya Williams", "Rajiv Patel", "Abigail Chen", "Francisco Lopez",
                    "Hailey Johnson", "Arthur Burton", "Ryan Park", "Isabella Rodriguez",
                    "Giorgio Rossi", "Carlos Gomez", "Klaus Mueller", "Maria Lopez",
                    "Ayesha Khan", "Wolfgang Schulz", "Mei Lin", "John Lin", "Eddy Lin",
                    "Jane Moreno", "Tom Moreno", "Tamara Taylor", "Carmen Ortiz",
                    "Adam Smith", "Yuriko Yamamoto", "Sam Moore", "Jennifer Moore"
                ]
                
                demo_env_data = {}
                
                # Generate positions in a grid-like pattern
                for i, agent_name in enumerate(demo_agents):
                    row = i // 5
                    col = i % 5
                    x = 20 + (col * 20)
                    y = 20 + (row * 15)
                    
                    demo_env_data[agent_name] = {
                        "maze": "the_ville",
                        "x": x,
                        "y": y
                    }
                
                with open(env_file_0, 'w') as f:
                    json.dump(demo_env_data, f, indent=2)
                
                logger.info(f"Created demo environment file: {env_file_0}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating demo environment file: {e}")
            
        return False
    
    async def create_demo_metadata(self, sim_code: str) -> bool:
        """Create demo metadata file if none exists"""
        try:
            meta_dir = self.storage_path / sim_code / "reverie"
            meta_file = meta_dir / "meta.json"
            
            if not meta_file.exists():
                demo_meta = {
                    "fork_sim_code": "base_the_ville_n25",
                    "sim_code": sim_code,
                    "created_at": datetime.now().isoformat(),
                    "description": "Dating show simulation with 25 agents",
                    "agents_count": 25,
                    "current_step": 0,
                    "status": "initialized"
                }
                
                with open(meta_file, 'w') as f:
                    json.dump(demo_meta, f, indent=2)
                
                logger.info(f"Created demo metadata file: {meta_file}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating demo metadata: {e}")
            
        return False
    
    async def initialize_with_demo_data(self, sim_code: str = "dating_show_25_agents") -> bool:
        """Initialize with demo data for testing"""
        try:
            logger.info("Initializing with demo data...")
            
            # Create basic structure
            await self._create_directories(sim_code)
            await self._initialize_current_sim_files(sim_code)
            
            # Create demo environment file
            await self.create_demo_environment_file(sim_code)
            
            # Create demo metadata
            await self.create_demo_metadata(sim_code)
            
            # Generate movement files
            await self._ensure_movement_files(sim_code)
            
            # Validate
            return await self._validate_critical_files(sim_code)
            
        except Exception as e:
            logger.error(f"Error initializing with demo data: {e}")
            return False


# Utility functions
async def initialize_frontend_storage(sim_code: str = "dating_show_25_agents", 
                                    create_demo: bool = False) -> bool:
    """Initialize frontend storage with optional demo data"""
    initializer = StartupInitializer()
    
    if create_demo:
        return await initializer.initialize_with_demo_data(sim_code)
    else:
        return await initializer.initialize_all(sim_code)


async def quick_fix_missing_files(sim_code: str = "dating_show_25_agents") -> bool:
    """Quick fix for missing movement files"""
    initializer = StartupInitializer()
    
    # Just ensure movement files exist
    await initializer._create_directories(sim_code)
    await initializer._ensure_movement_files(sim_code)
    
    return True