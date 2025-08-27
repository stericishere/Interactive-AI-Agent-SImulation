"""
Automated 8-Agent Simulation Setup Service

This service handles the complete automation of creating clean 8-agent simulations
from the 25-agent base configuration. It ensures consistency across all files
and provides error recovery capabilities.

Key Features:
- Automatic agent selection and cleanup
- Validation of configuration consistency  
- Error recovery and repair functionality
- Comprehensive logging and monitoring
"""

import json
import os
import shutil
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class SimulationConfig:
    """Configuration for simulation setup"""
    fork_sim_code: str
    start_date: str = "February 13, 2023"
    curr_time: str = "February 13, 2023, 00:00:00"
    sec_per_step: int = 10
    maze_name: str = "the_ville"
    step: int = 0
    

@dataclass
class AgentPosition:
    """Agent position in simulation"""
    maze: str
    x: int
    y: int


class SimulationSetupService:
    """
    Automated service for creating and managing 8-agent dating show simulations
    """
    
    # First 8 agents from the base_the_ville_n25 configuration
    TARGET_AGENTS = [
        "Latoya Williams",
        "Rajiv Patel", 
        "Abigail Chen",
        "Francisco Lopez",
        "Hailey Johnson", 
        "Arthur Burton",
        "Ryan Park",
        "Isabella Rodriguez"
    ]
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the simulation setup service
        
        Args:
            base_path: Base path to the storage directory. If None, uses default location.
        """
        if base_path is None:
            # Default path based on project structure
            self.base_path = Path("/Applications/Projects/Open source/generative_agents/trash/reverie/environment/frontend_server/storage")
        else:
            self.base_path = Path(base_path)
            
        self.source_sim = "base_the_ville_n25"
        self.target_sim = "dating_show_8_agents"
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging for the service"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('simulation_setup.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_8_agent_simulation(self, force_recreate: bool = False) -> Dict[str, any]:
        """
        Create a clean 8-agent simulation from the 25-agent base
        
        Args:
            force_recreate: If True, recreate even if target exists
            
        Returns:
            Dictionary with creation results and status
        """
        try:
            result = {
                "success": False,
                "message": "",
                "agents_created": [],
                "files_processed": [],
                "errors": []
            }
            
            source_path = self.base_path / self.source_sim
            target_path = self.base_path / self.target_sim
            
            # Validate source exists
            if not source_path.exists():
                raise FileNotFoundError(f"Source simulation not found: {source_path}")
                
            self.logger.info(f"Starting 8-agent simulation creation from {source_path}")
            
            # Handle existing target
            if target_path.exists():
                if not force_recreate:
                    self.logger.warning(f"Target simulation already exists: {target_path}")
                    result["message"] = "Target simulation already exists. Use force_recreate=True to overwrite."
                    return result
                else:
                    self.logger.info(f"Removing existing target: {target_path}")
                    shutil.rmtree(target_path)
                    
            # Create target directory structure
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Copy and process configuration files
            self._copy_configuration_files(source_path, target_path, result)
            
            # Copy and filter agent data
            self._copy_agent_data(source_path, target_path, result)
            
            # Copy environment data (filtered)
            self._copy_environment_data(source_path, target_path, result)
            
            # Copy other necessary files
            self._copy_additional_files(source_path, target_path, result)
            
            # Validate the created simulation
            validation_result = self.validate_simulation_consistency()
            if not validation_result["is_valid"]:
                result["errors"].extend(validation_result["errors"])
                raise Exception("Simulation validation failed after creation")
                
            result["success"] = True
            result["message"] = "8-agent simulation created successfully"
            result["agents_created"] = self.TARGET_AGENTS.copy()
            
            self.logger.info("8-agent simulation creation completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create 8-agent simulation: {str(e)}")
            result["errors"].append(str(e))
            result["message"] = f"Creation failed: {str(e)}"
            return result
            
    def _copy_configuration_files(self, source_path: Path, target_path: Path, result: Dict) -> None:
        """Copy and modify configuration files"""
        
        # Copy and modify meta.json
        meta_source = source_path / "reverie" / "meta.json"
        meta_target = target_path / "reverie"
        meta_target.mkdir(parents=True, exist_ok=True)
        
        if meta_source.exists():
            with open(meta_source, 'r') as f:
                meta_data = json.load(f)
                
            # Update configuration for 8 agents
            meta_data["fork_sim_code"] = "dating_show_8_agents"
            meta_data["persona_names"] = self.TARGET_AGENTS.copy()
            
            with open(meta_target / "meta.json", 'w') as f:
                json.dump(meta_data, f, indent=2)
                
            result["files_processed"].append("reverie/meta.json")
            self.logger.info("Updated meta.json with 8-agent configuration")
        else:
            raise FileNotFoundError("Source meta.json not found")
            
    def _copy_agent_data(self, source_path: Path, target_path: Path, result: Dict) -> None:
        """Copy persona data for selected agents only"""
        
        source_personas = source_path / "personas"
        target_personas = target_path / "personas"
        target_personas.mkdir(parents=True, exist_ok=True)
        
        for agent_name in self.TARGET_AGENTS:
            agent_source = source_personas / agent_name
            agent_target = target_personas / agent_name
            
            if agent_source.exists():
                shutil.copytree(agent_source, agent_target, dirs_exist_ok=True)
                result["files_processed"].append(f"personas/{agent_name}")
                self.logger.info(f"Copied persona data for {agent_name}")
            else:
                self.logger.warning(f"Source persona not found: {agent_name}")
                result["errors"].append(f"Missing source persona: {agent_name}")
                
    def _copy_environment_data(self, source_path: Path, target_path: Path, result: Dict) -> None:
        """Copy environment data, filtering for selected agents only"""
        
        source_env = source_path / "environment"
        target_env = target_path / "environment"
        target_env.mkdir(parents=True, exist_ok=True)
        
        # Process each environment file
        for env_file in source_env.glob("*.json"):
            with open(env_file, 'r') as f:
                env_data = json.load(f)
                
            # Filter for target agents only
            filtered_data = {
                agent: data for agent, data in env_data.items() 
                if agent in self.TARGET_AGENTS
            }
            
            target_file = target_env / env_file.name
            with open(target_file, 'w') as f:
                json.dump(filtered_data, f, indent=2)
                
            result["files_processed"].append(f"environment/{env_file.name}")
            self.logger.info(f"Filtered environment file: {env_file.name}")
            
    def _copy_additional_files(self, source_path: Path, target_path: Path, result: Dict) -> None:
        """Copy any additional files that might be needed"""
        
        # Copy movement files if they exist
        source_movement = source_path / "movement"
        if source_movement.exists():
            target_movement = target_path / "movement"
            target_movement.mkdir(parents=True, exist_ok=True)
            
            for movement_file in source_movement.glob("*.json"):
                with open(movement_file, 'r') as f:
                    movement_data = json.load(f)
                    
                # Filter movement data for target agents
                if isinstance(movement_data, dict):
                    filtered_data = {
                        agent: data for agent, data in movement_data.items()
                        if agent in self.TARGET_AGENTS
                    }
                    
                    target_file = target_movement / movement_file.name
                    with open(target_file, 'w') as f:
                        json.dump(filtered_data, f, indent=2)
                        
                    result["files_processed"].append(f"movement/{movement_file.name}")
                    
    def validate_simulation_consistency(self) -> Dict[str, any]:
        """
        Validate that the simulation is properly configured and consistent
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "checks_performed": []
        }
        
        try:
            target_path = self.base_path / self.target_sim
            
            if not target_path.exists():
                validation_result["is_valid"] = False
                validation_result["errors"].append("Target simulation directory does not exist")
                return validation_result
                
            # Check meta.json
            self._validate_meta_json(target_path, validation_result)
            
            # Check persona directories
            self._validate_persona_directories(target_path, validation_result)
            
            # Check environment files
            self._validate_environment_files(target_path, validation_result)
            
            # Check for extra personas (should not exist)
            self._validate_no_extra_personas(target_path, validation_result)
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Validation failed with exception: {str(e)}")
            
        return validation_result
        
    def _validate_meta_json(self, target_path: Path, result: Dict) -> None:
        """Validate meta.json configuration"""
        
        meta_file = target_path / "reverie" / "meta.json"
        result["checks_performed"].append("meta.json validation")
        
        if not meta_file.exists():
            result["is_valid"] = False
            result["errors"].append("meta.json not found")
            return
            
        try:
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
                
            # Check required fields
            required_fields = ["fork_sim_code", "persona_names", "start_date", "curr_time"]
            for field in required_fields:
                if field not in meta_data:
                    result["is_valid"] = False
                    result["errors"].append(f"Missing required field in meta.json: {field}")
                    
            # Check persona names
            if "persona_names" in meta_data:
                if len(meta_data["persona_names"]) != 8:
                    result["is_valid"] = False
                    result["errors"].append(f"Expected 8 persona names, found {len(meta_data['persona_names'])}")
                    
                missing_agents = set(self.TARGET_AGENTS) - set(meta_data["persona_names"])
                if missing_agents:
                    result["is_valid"] = False
                    result["errors"].append(f"Missing expected agents in meta.json: {missing_agents}")
                    
                extra_agents = set(meta_data["persona_names"]) - set(self.TARGET_AGENTS)
                if extra_agents:
                    result["is_valid"] = False
                    result["errors"].append(f"Unexpected agents in meta.json: {extra_agents}")
                    
        except json.JSONDecodeError as e:
            result["is_valid"] = False
            result["errors"].append(f"Invalid JSON in meta.json: {str(e)}")
            
    def _validate_persona_directories(self, target_path: Path, result: Dict) -> None:
        """Validate persona directories exist and are complete"""
        
        personas_path = target_path / "personas"
        result["checks_performed"].append("persona directories validation")
        
        if not personas_path.exists():
            result["is_valid"] = False
            result["errors"].append("Personas directory does not exist")
            return
            
        # Check each expected agent
        for agent in self.TARGET_AGENTS:
            agent_path = personas_path / agent
            if not agent_path.exists():
                result["is_valid"] = False
                result["errors"].append(f"Missing persona directory: {agent}")
                continue
                
            # Check for required subdirectories/files
            bootstrap_path = agent_path / "bootstrap_memory"
            if not bootstrap_path.exists():
                result["warnings"].append(f"Missing bootstrap_memory for {agent}")
            else:
                # Check for key memory files
                required_files = ["scratch.json", "spatial_memory.json"]
                for req_file in required_files:
                    if not (bootstrap_path / req_file).exists():
                        result["warnings"].append(f"Missing {req_file} for {agent}")
                        
    def _validate_environment_files(self, target_path: Path, result: Dict) -> None:
        """Validate environment files contain only target agents"""
        
        env_path = target_path / "environment"
        result["checks_performed"].append("environment files validation")
        
        if not env_path.exists():
            result["is_valid"] = False
            result["errors"].append("Environment directory does not exist")
            return
            
        # Check environment files
        for env_file in env_path.glob("*.json"):
            try:
                with open(env_file, 'r') as f:
                    env_data = json.load(f)
                    
                if isinstance(env_data, dict):
                    env_agents = set(env_data.keys())
                    expected_agents = set(self.TARGET_AGENTS)
                    
                    missing_agents = expected_agents - env_agents
                    if missing_agents:
                        result["warnings"].append(f"Missing agents in {env_file.name}: {missing_agents}")
                        
                    extra_agents = env_agents - expected_agents
                    if extra_agents:
                        result["is_valid"] = False
                        result["errors"].append(f"Unexpected agents in {env_file.name}: {extra_agents}")
                        
            except (json.JSONDecodeError, IOError) as e:
                result["is_valid"] = False
                result["errors"].append(f"Error reading {env_file.name}: {str(e)}")
                
    def _validate_no_extra_personas(self, target_path: Path, result: Dict) -> None:
        """Ensure no extra persona directories exist"""
        
        personas_path = target_path / "personas"
        result["checks_performed"].append("extra personas validation")
        
        if not personas_path.exists():
            return
            
        existing_personas = set(d.name for d in personas_path.iterdir() if d.is_dir())
        expected_personas = set(self.TARGET_AGENTS)
        
        extra_personas = existing_personas - expected_personas
        if extra_personas:
            result["is_valid"] = False
            result["errors"].append(f"Found unexpected persona directories: {extra_personas}")
            
    def cleanup_extra_personas(self) -> Dict[str, any]:
        """
        Remove any persona directories that are not in the target 8 agents
        
        Returns:
            Dictionary with cleanup results
        """
        cleanup_result = {
            "success": False,
            "removed_personas": [],
            "errors": []
        }
        
        try:
            target_path = self.base_path / self.target_sim
            personas_path = target_path / "personas"
            
            if not personas_path.exists():
                cleanup_result["success"] = True
                cleanup_result["message"] = "Personas directory does not exist"
                return cleanup_result
                
            existing_personas = [d for d in personas_path.iterdir() if d.is_dir()]
            expected_personas = set(self.TARGET_AGENTS)
            
            for persona_dir in existing_personas:
                if persona_dir.name not in expected_personas:
                    self.logger.info(f"Removing extra persona: {persona_dir.name}")
                    shutil.rmtree(persona_dir)
                    cleanup_result["removed_personas"].append(persona_dir.name)
                    
            cleanup_result["success"] = True
            cleanup_result["message"] = f"Cleaned up {len(cleanup_result['removed_personas'])} extra personas"
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup extra personas: {str(e)}")
            cleanup_result["errors"].append(str(e))
            cleanup_result["message"] = f"Cleanup failed: {str(e)}"
            
        return cleanup_result
        
    def repair_simulation(self) -> Dict[str, any]:
        """
        Attempt to repair common issues in the simulation
        
        Returns:
            Dictionary with repair results
        """
        repair_result = {
            "success": False,
            "repairs_attempted": [],
            "repairs_successful": [],
            "errors": []
        }
        
        try:
            # First, validate to identify issues
            validation = self.validate_simulation_consistency()
            
            if validation["is_valid"]:
                repair_result["success"] = True
                repair_result["message"] = "Simulation is already valid, no repairs needed"
                return repair_result
                
            self.logger.info("Starting simulation repair process")
            
            # Repair 1: Fix meta.json if corrupted
            if any("meta.json" in error for error in validation["errors"]):
                repair_result["repairs_attempted"].append("meta.json repair")
                if self._repair_meta_json():
                    repair_result["repairs_successful"].append("meta.json repair")
                    
            # Repair 2: Remove extra personas
            if any("unexpected persona" in error.lower() for error in validation["errors"]):
                repair_result["repairs_attempted"].append("extra persona removal")
                cleanup_result = self.cleanup_extra_personas()
                if cleanup_result["success"]:
                    repair_result["repairs_successful"].append("extra persona removal")
                else:
                    repair_result["errors"].extend(cleanup_result["errors"])
                    
            # Repair 3: Recreate missing environment files
            if any("environment" in error.lower() for error in validation["errors"]):
                repair_result["repairs_attempted"].append("environment files repair")
                if self._repair_environment_files():
                    repair_result["repairs_successful"].append("environment files repair")
                    
            # Final validation
            final_validation = self.validate_simulation_consistency()
            repair_result["success"] = final_validation["is_valid"]
            
            if repair_result["success"]:
                repair_result["message"] = "Simulation successfully repaired"
            else:
                repair_result["message"] = "Some repairs failed, simulation may still have issues"
                repair_result["errors"].extend(final_validation["errors"])
                
        except Exception as e:
            self.logger.error(f"Repair process failed: {str(e)}")
            repair_result["errors"].append(str(e))
            repair_result["message"] = f"Repair failed: {str(e)}"
            
        return repair_result
        
    def _repair_meta_json(self) -> bool:
        """Repair meta.json file"""
        try:
            target_path = self.base_path / self.target_sim
            meta_file = target_path / "reverie" / "meta.json"
            
            # Create the directory if it doesn't exist
            meta_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a new meta.json with correct configuration
            meta_data = {
                "fork_sim_code": "dating_show_8_agents",
                "start_date": "February 13, 2023",
                "curr_time": "February 13, 2023, 00:00:00",
                "sec_per_step": 10,
                "maze_name": "the_ville",
                "persona_names": self.TARGET_AGENTS.copy(),
                "step": 0
            }
            
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f, indent=2)
                
            self.logger.info("Successfully repaired meta.json")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to repair meta.json: {str(e)}")
            return False
            
    def _repair_environment_files(self) -> bool:
        """Repair environment files by recreating from source"""
        try:
            source_path = self.base_path / self.source_sim
            target_path = self.base_path / self.target_sim
            
            source_env = source_path / "environment"
            target_env = target_path / "environment"
            
            if not source_env.exists():
                self.logger.error("Source environment directory not found")
                return False
                
            target_env.mkdir(parents=True, exist_ok=True)
            
            # Recreate environment files
            for env_file in source_env.glob("*.json"):
                with open(env_file, 'r') as f:
                    env_data = json.load(f)
                    
                # Filter for target agents
                filtered_data = {
                    agent: data for agent, data in env_data.items()
                    if agent in self.TARGET_AGENTS
                }
                
                target_file = target_env / env_file.name
                with open(target_file, 'w') as f:
                    json.dump(filtered_data, f, indent=2)
                    
            self.logger.info("Successfully repaired environment files")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to repair environment files: {str(e)}")
            return False
            
    def get_simulation_status(self) -> Dict[str, any]:
        """
        Get comprehensive status of the 8-agent simulation
        
        Returns:
            Dictionary with simulation status information
        """
        status = {
            "exists": False,
            "is_valid": False,
            "agent_count": 0,
            "expected_agents": self.TARGET_AGENTS.copy(),
            "actual_agents": [],
            "missing_agents": [],
            "extra_agents": [],
            "files_status": {},
            "last_modified": None
        }
        
        try:
            target_path = self.base_path / self.target_sim
            
            if not target_path.exists():
                return status
                
            status["exists"] = True
            status["last_modified"] = datetime.fromtimestamp(target_path.stat().st_mtime).isoformat()
            
            # Check personas
            personas_path = target_path / "personas"
            if personas_path.exists():
                actual_agents = [d.name for d in personas_path.iterdir() if d.is_dir()]
                status["actual_agents"] = actual_agents
                status["agent_count"] = len(actual_agents)
                
                expected_set = set(self.TARGET_AGENTS)
                actual_set = set(actual_agents)
                
                status["missing_agents"] = list(expected_set - actual_set)
                status["extra_agents"] = list(actual_set - expected_set)
                
            # Check file status
            key_files = [
                "reverie/meta.json",
                "environment/0.json",
                "personas"
            ]
            
            for file_path in key_files:
                full_path = target_path / file_path
                status["files_status"][file_path] = {
                    "exists": full_path.exists(),
                    "is_file": full_path.is_file() if full_path.exists() else False,
                    "is_dir": full_path.is_dir() if full_path.exists() else False
                }
                
            # Overall validity check
            validation = self.validate_simulation_consistency()
            status["is_valid"] = validation["is_valid"]
            
        except Exception as e:
            status["error"] = str(e)
            
        return status


# Convenience functions for easy usage
def create_dating_show_simulation(force_recreate: bool = False) -> Dict[str, any]:
    """
    Convenience function to create an 8-agent dating show simulation
    
    Args:
        force_recreate: Whether to recreate if already exists
        
    Returns:
        Creation result dictionary
    """
    service = SimulationSetupService()
    return service.create_8_agent_simulation(force_recreate=force_recreate)


def validate_dating_show_simulation() -> Dict[str, any]:
    """
    Convenience function to validate the 8-agent dating show simulation
    
    Returns:
        Validation result dictionary
    """
    service = SimulationSetupService()
    return service.validate_simulation_consistency()


def repair_dating_show_simulation() -> Dict[str, any]:
    """
    Convenience function to repair the 8-agent dating show simulation
    
    Returns:
        Repair result dictionary
    """
    service = SimulationSetupService()
    return service.repair_simulation()


def get_dating_show_status() -> Dict[str, any]:
    """
    Convenience function to get status of the 8-agent dating show simulation
    
    Returns:
        Status information dictionary
    """
    service = SimulationSetupService()
    return service.get_simulation_status()


if __name__ == "__main__":
    # Example usage
    service = SimulationSetupService()
    
    print("Creating 8-agent dating show simulation...")
    result = service.create_8_agent_simulation(force_recreate=True)
    
    if result["success"]:
        print("✅ Simulation created successfully!")
        print(f"Agents created: {len(result['agents_created'])}")
        print(f"Files processed: {len(result['files_processed'])}")
        
        print("\nValidating simulation...")
        validation = service.validate_simulation_consistency()
        
        if validation["is_valid"]:
            print("✅ Simulation validation passed!")
        else:
            print("❌ Simulation validation failed:")
            for error in validation["errors"]:
                print(f"  - {error}")
    else:
        print("❌ Simulation creation failed:")
        for error in result["errors"]:
            print(f"  - {error}")