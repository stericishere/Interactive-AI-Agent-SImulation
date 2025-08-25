"""
Enhanced Step File Management System
Implements hybrid generation strategy combining proactive file creation with robust fallbacks
Prevents simulation blocking on missing environment/movement files
"""

import json
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import random
import shutil
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class StepGenerationConfig:
    """Configuration for step file generation strategies"""
    strategy: str = "hybrid"  # reactive, proactive, hybrid
    fallback_enabled: bool = True
    cache_templates: bool = True
    max_retries: int = 3
    generation_timeout: float = 30.0
    
    generation_order: List[str] = field(default_factory=lambda: [
        "previous_step",
        "base_simulation", 
        "minimal_fallback"
    ])

@dataclass 
class StepGenerationResult:
    """Result of step file generation attempt"""
    success: bool
    step: int
    sim_code: str
    strategy_used: Optional[str] = None
    files_created: List[str] = field(default_factory=list)
    error: Optional[str] = None
    execution_time: float = 0.0

class EnhancedStepFileManager:
    """
    Advanced step file generation system that combines best practices:
    - Proactive generation (Dating Show approach)
    - Robust fallback mechanisms (Legacy patterns)
    - Template caching for performance
    - Async operations for non-blocking execution
    """
    
    def __init__(self, storage_path: str, config: Optional[StepGenerationConfig] = None):
        self.storage_path = Path(storage_path)
        self.config = config or StepGenerationConfig()
        self.template_cache: Dict[str, Dict] = {}
        self.generation_lock = asyncio.Lock()
        
        # Dating show specific enhancements
        self.dating_show_emojis = [
            "💕", "🌹", "💬", "😊", "🥰", "💭", "✨", "🌟", "💃", "🎭", 
            "🔥", "🌺", "📚", "🎵", "🍃", "⚡", "🎮", "🌸", "🛠️", "🎬"
        ]
        
        self.villa_activities = [
            "socializing with other contestants",
            "exploring the villa grounds", 
            "having conversations by the pool",
            "preparing for the next challenge",
            "reflecting on romantic connections",
            "enjoying villa amenities",
            "building relationships with housemates"
        ]
    
    async def ensure_step_files_exist(self, sim_code: str, step: int) -> StepGenerationResult:
        """
        Ensure environment and movement files exist for a given step
        Uses hybrid generation strategy with multiple fallbacks
        
        Args:
            sim_code: Simulation identifier
            step: Step number to check/generate
            
        Returns:
            StepGenerationResult: Generation outcome with details
        """
        start_time = datetime.now()
        
        print(f"🔍 [DEBUG] Starting ensure_step_files_exist for {sim_code} step {step}")
        
        async with self.generation_lock:
            try:
                # Check if files already exist
                print(f"🔍 [DEBUG] Checking if files exist for {sim_code} step {step}")
                files_exist = await self._files_exist(sim_code, step)
                print(f"🔍 [DEBUG] Files exist check result: {files_exist}")
                
                if files_exist:
                    print(f"✅ [DEBUG] Files already exist for {sim_code} step {step}")
                    return StepGenerationResult(
                        success=True,
                        step=step,
                        sim_code=sim_code,
                        strategy_used="existing_files",
                        execution_time=self._get_execution_time(start_time)
                    )
                
                print(f"🔧 [DEBUG] Files missing, generating step {step} files for {sim_code}")
                logger.info(f"🔧 Generating step {step} files for {sim_code}")
                
                # Try generation strategies in order
                for i, strategy in enumerate(self.config.generation_order):
                    print(f"🎯 [DEBUG] Trying strategy {i+1}/{len(self.config.generation_order)}: {strategy}")
                    result = await self._try_generation_strategy(sim_code, step, strategy)
                    
                    print(f"🎯 [DEBUG] Strategy {strategy} result: success={result.success}")
                    if result.error:
                        print(f"🎯 [DEBUG] Strategy {strategy} error: {result.error}")
                    
                    if result.success:
                        result.execution_time = self._get_execution_time(start_time)
                        print(f"✅ [DEBUG] Successfully generated step {step} files using {strategy} strategy")
                        print(f"✅ [DEBUG] Files created: {result.files_created}")
                        logger.info(f"✅ Generated step {step} files using {strategy} strategy")
                        return result
                    
                    print(f"❌ [DEBUG] Strategy {strategy} failed, trying next...")
                    logger.warning(f"❌ Strategy {strategy} failed: {result.error}")
                
                # All strategies failed
                print(f"💥 [DEBUG] All generation strategies failed for {sim_code} step {step}")
                return StepGenerationResult(
                    success=False,
                    step=step,
                    sim_code=sim_code,
                    error="All generation strategies failed",
                    execution_time=self._get_execution_time(start_time)
                )
                
            except Exception as e:
                print(f"🚨 [DEBUG] Exception in ensure_step_files_exist: {e}")
                logger.error(f"🚨 Step file generation error: {e}")
                return StepGenerationResult(
                    success=False,
                    step=step,
                    sim_code=sim_code,
                    error=str(e),
                    execution_time=self._get_execution_time(start_time)
                )
    
    async def _try_generation_strategy(self, sim_code: str, step: int, strategy: str) -> StepGenerationResult:
        """Try a specific generation strategy"""
        try:
            if strategy == "previous_step":
                return await self._generate_from_previous_step(sim_code, step)
            elif strategy == "base_simulation":
                return await self._generate_from_base_simulation(sim_code, step)
            elif strategy == "minimal_fallback":
                return await self._create_minimal_step_files(sim_code, step)
            else:
                return StepGenerationResult(
                    success=False,
                    step=step,
                    sim_code=sim_code,
                    error=f"Unknown strategy: {strategy}"
                )
        except Exception as e:
            return StepGenerationResult(
                success=False,
                step=step,
                sim_code=sim_code,
                strategy_used=strategy,
                error=str(e)
            )
    
    async def _generate_from_previous_step(self, sim_code: str, step: int) -> StepGenerationResult:
        """
        Generate step files using previous step as template
        Enhanced with dating show context and agent evolution
        """
        if step <= 0:
            return StepGenerationResult(
                success=False,
                step=step,
                sim_code=sim_code,
                error="Cannot use previous step strategy for step 0"
            )
        
        sim_path = self.storage_path / sim_code
        prev_step = step - 1
        
        # Load previous step files
        prev_env_file = sim_path / 'environment' / f'{prev_step}.json'
        prev_mov_file = sim_path / 'movement' / f'{prev_step}.json'
        
        if not (prev_env_file.exists() and prev_mov_file.exists()):
            return StepGenerationResult(
                success=False,
                step=step,
                sim_code=sim_code,
                error=f"Previous step {prev_step} files not found"
            )
        
        # Load and enhance previous step data
        prev_env = await self._load_json_async(prev_env_file)
        prev_mov = await self._load_json_async(prev_mov_file)
        
        # Apply dating show enhancements
        enhanced_env = await self._apply_dating_context(prev_env, step)
        enhanced_mov = await self._evolve_agent_positions(prev_mov, step)
        
        # Save new step files
        new_env_file = sim_path / 'environment' / f'{step}.json'
        new_mov_file = sim_path / 'movement' / f'{step}.json'
        
        await self._save_json_async(new_env_file, enhanced_env)
        await self._save_json_async(new_mov_file, enhanced_mov)
        
        return StepGenerationResult(
            success=True,
            step=step,
            sim_code=sim_code,
            strategy_used="previous_step",
            files_created=[str(new_env_file), str(new_mov_file)]
        )
    
    async def _generate_from_base_simulation(self, sim_code: str, step: int) -> StepGenerationResult:
        """
        Generate step files using base simulation template
        Fallback when previous step is unavailable
        """
        base_sim_candidates = [
            "base_the_ville_n25",
            "base_the_ville_isabella_maria_klaus",
            f"base_{sim_code}"
        ]
        
        for base_sim in base_sim_candidates:
            base_path = self.storage_path / base_sim
            base_env_file = base_path / 'environment' / '0.json'
            
            if base_env_file.exists():
                # Load base template
                base_env = await self._load_json_async(base_env_file)
                
                # Adapt for current step and simulation
                adapted_env = await self._adapt_base_template(base_env, sim_code, step)
                adapted_mov = await self._create_movement_from_environment(adapted_env, step)
                
                # Save adapted files
                sim_path = self.storage_path / sim_code
                new_env_file = sim_path / 'environment' / f'{step}.json'
                new_mov_file = sim_path / 'movement' / f'{step}.json'
                
                # Ensure directories exist
                new_env_file.parent.mkdir(parents=True, exist_ok=True)
                new_mov_file.parent.mkdir(parents=True, exist_ok=True)
                
                await self._save_json_async(new_env_file, adapted_env)
                await self._save_json_async(new_mov_file, adapted_mov)
                
                return StepGenerationResult(
                    success=True,
                    step=step,
                    sim_code=sim_code,
                    strategy_used="base_simulation",
                    files_created=[str(new_env_file), str(new_mov_file)]
                )
        
        return StepGenerationResult(
            success=False,
            step=step,
            sim_code=sim_code,
            error="No suitable base simulation template found"
        )
    
    async def _create_minimal_step_files(self, sim_code: str, step: int) -> StepGenerationResult:
        """
        Create minimal viable step files as last resort
        Ensures simulation can progress even with limited data
        """
        sim_path = self.storage_path / sim_code
        
        # Create minimal environment data
        minimal_env = {
            "personas": {},
            "meta": {
                "curr_time": datetime.now(timezone.utc).strftime("%B %d, %Y, %H:%M:%S"),
                "step": step,
                "generation_strategy": "minimal_fallback"
            }
        }
        
        # Create minimal movement data  
        minimal_mov = {
            "persona": {},
            "meta": {
                "curr_time": datetime.now(timezone.utc).strftime("%B %d, %Y, %H:%M:%S"),
                "step": step,
                "generation_strategy": "minimal_fallback"
            }
        }
        
        # If we can find agent names anywhere, add them
        agent_names = await self._discover_agent_names(sim_code)
        if agent_names:
            # Add basic agent data
            for i, agent_name in enumerate(agent_names):
                # Distribute agents across villa positions
                x = 20 + (i % 10) * 10
                y = 20 + (i // 10) * 10
                
                minimal_env["personas"][agent_name] = {
                    "x": x,
                    "y": y,
                    "action": "exploring the villa"
                }
                
                minimal_mov["persona"][agent_name] = {
                    "movement": [x, y],
                    "pronunciatio": random.choice(self.dating_show_emojis),
                    "description": f"{agent_name} is {random.choice(self.villa_activities)} @ villa",
                    "chat": ""
                }
        
        # Save minimal files
        new_env_file = sim_path / 'environment' / f'{step}.json'
        new_mov_file = sim_path / 'movement' / f'{step}.json'
        
        # Ensure directories exist
        new_env_file.parent.mkdir(parents=True, exist_ok=True)
        new_mov_file.parent.mkdir(parents=True, exist_ok=True)
        
        await self._save_json_async(new_env_file, minimal_env)
        await self._save_json_async(new_mov_file, minimal_mov)
        
        return StepGenerationResult(
            success=True,
            step=step,
            sim_code=sim_code,
            strategy_used="minimal_fallback",
            files_created=[str(new_env_file), str(new_mov_file)]
        )
    
    async def _apply_dating_context(self, env_data: Dict, step: int) -> Dict:
        """Apply dating show specific enhancements to environment data"""
        enhanced_env = env_data.copy()
        
        # Update timestamp
        if "meta" not in enhanced_env:
            enhanced_env["meta"] = {}
        
        enhanced_env["meta"].update({
            "curr_time": datetime.now(timezone.utc).strftime("%B %d, %Y, %H:%M:%S"),
            "step": step,
            "context": "dating_show_villa",
            "enhancement_applied": True
        })
        
        # Apply dating show context to agent actions
        if "personas" in enhanced_env:
            for persona_name, persona_data in enhanced_env["personas"].items():
                if isinstance(persona_data, dict):
                    # Update action with dating context
                    if "action" not in persona_data or persona_data["action"] in ["idle", ""]:
                        persona_data["action"] = random.choice(self.villa_activities)
        
        return enhanced_env
    
    async def _evolve_agent_positions(self, mov_data: Dict, step: int) -> Dict:
        """Evolve agent positions with small natural movements"""
        enhanced_mov = mov_data.copy()
        
        # Update meta information
        if "meta" not in enhanced_mov:
            enhanced_mov["meta"] = {}
            
        enhanced_mov["meta"].update({
            "curr_time": datetime.now(timezone.utc).strftime("%B %d, %Y, %H:%M:%S"),
            "step": step,
            "position_evolution_applied": True
        })
        
        # Apply small position changes to simulate natural movement
        if "persona" in enhanced_mov:
            for persona_name, persona_data in enhanced_mov["persona"].items():
                if isinstance(persona_data, dict) and "movement" in persona_data:
                    current_pos = persona_data["movement"]
                    if isinstance(current_pos, list) and len(current_pos) == 2:
                        # Small random movement (±5 tiles)
                        # Use proper map boundaries: 140x100 (0-139, 0-99)
                        dx = random.randint(-5, 5)
                        dy = random.randint(-5, 5)
                        
                        new_x = max(0, min(139, current_pos[0] + dx))
                        new_y = max(0, min(99, current_pos[1] + dy))
                        
                        persona_data["movement"] = [new_x, new_y]
                        
                        # Update activity description occasionally
                        if random.random() < 0.3:  # 30% chance
                            persona_data["description"] = f"{persona_name} is {random.choice(self.villa_activities)} @ villa"
                            persona_data["pronunciatio"] = random.choice(self.dating_show_emojis)
        
        return enhanced_mov
    
    # Utility methods
    
    async def _files_exist(self, sim_code: str, step: int) -> bool:
        """Check if both environment and movement files exist for given step"""
        sim_path = self.storage_path / sim_code
        env_file = sim_path / 'environment' / f'{step}.json'
        mov_file = sim_path / 'movement' / f'{step}.json'
        
        print(f"🔍 [DEBUG] Checking file paths:")
        print(f"🔍 [DEBUG]   Storage path: {self.storage_path}")
        print(f"🔍 [DEBUG]   Sim path: {sim_path}")
        print(f"🔍 [DEBUG]   Env file: {env_file}")
        print(f"🔍 [DEBUG]   Mov file: {mov_file}")
        print(f"🔍 [DEBUG]   Env exists: {env_file.exists()}")
        print(f"🔍 [DEBUG]   Mov exists: {mov_file.exists()}")
        
        result = env_file.exists() and mov_file.exists()
        print(f"🔍 [DEBUG] Files exist result: {result}")
        return result
    
    async def _load_json_async(self, file_path: Path) -> Dict:
        """Load JSON file asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._load_json_sync, file_path)
    
    def _load_json_sync(self, file_path: Path) -> Dict:
        """Synchronous JSON loading"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    async def _save_json_async(self, file_path: Path, data: Dict) -> None:
        """Save JSON file asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_json_sync, file_path, data)
    
    def _save_json_sync(self, file_path: Path, data: Dict) -> None:
        """Synchronous JSON saving"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    async def _discover_agent_names(self, sim_code: str) -> List[str]:
        """Discover agent names from existing files or personas directory"""
        sim_path = self.storage_path / sim_code
        agent_names = []
        
        # Try to find agent names from personas directory
        personas_dir = sim_path / 'personas'
        if personas_dir.exists():
            agent_names = [d.name for d in personas_dir.iterdir() if d.is_dir()]
        
        # Fallback: try to find from existing step files
        if not agent_names:
            for step in range(10):  # Check first 10 steps
                env_file = sim_path / 'environment' / f'{step}.json'
                if env_file.exists():
                    try:
                        env_data = await self._load_json_async(env_file)
                        if "personas" in env_data:
                            agent_names = list(env_data["personas"].keys())
                            break
                    except:
                        continue
        
        return agent_names[:25]  # Limit to reasonable number
    
    async def _adapt_base_template(self, base_env: Dict, sim_code: str, step: int) -> Dict:
        """Adapt base simulation template for current simulation and step"""
        adapted = base_env.copy()
        
        # Update meta information
        if "meta" not in adapted:
            adapted["meta"] = {}
            
        adapted["meta"].update({
            "curr_time": datetime.now(timezone.utc).strftime("%B %d, %Y, %H:%M:%S"),
            "step": step,
            "sim_code": sim_code,
            "adapted_from_base": True
        })
        
        return adapted
    
    async def _create_movement_from_environment(self, env_data: Dict, step: int) -> Dict:
        """Create movement data from environment data"""
        movement_data = {
            "persona": {},
            "meta": {
                "curr_time": datetime.now(timezone.utc).strftime("%B %d, %Y, %H:%M:%S"),
                "step": step,
                "generated_from_environment": True
            }
        }
        
        if "personas" in env_data:
            for persona_name, persona_data in env_data["personas"].items():
                if isinstance(persona_data, dict):
                    x = persona_data.get("x", 50)
                    y = persona_data.get("y", 50)
                    
                    movement_data["persona"][persona_name] = {
                        "movement": [x, y],
                        "pronunciatio": random.choice(self.dating_show_emojis),
                        "description": f"{persona_name} is {random.choice(self.villa_activities)} @ villa",
                        "chat": ""
                    }
        
        return movement_data
    
    def _get_execution_time(self, start_time: datetime) -> float:
        """Calculate execution time in seconds"""
        return (datetime.now() - start_time).total_seconds()


# Global instance for easy access
_enhanced_step_manager = None

def get_enhanced_step_manager(storage_path: str = None) -> EnhancedStepFileManager:
    """Get singleton instance of EnhancedStepFileManager"""
    global _enhanced_step_manager
    
    if _enhanced_step_manager is None:
        if storage_path is None:
            # Auto-detect the correct storage path based on where simulation exists
            dating_show_path = "/Applications/Projects/Open source/generative_agents/dating_show_env/frontend_service/storage"
            legacy_path = "/Applications/Projects/Open source/generative_agents/environment/frontend_server/storage"
            
            import os
            if os.path.exists(os.path.join(legacy_path, "dating_show_8_agents")):
                storage_path = legacy_path
                print(f"🎯 [DEBUG] Auto-detected legacy environment storage: {storage_path}")
            elif os.path.exists(os.path.join(dating_show_path, "dating_show_8_agents")):
                storage_path = dating_show_path  
                print(f"🎯 [DEBUG] Auto-detected dating_show_env storage: {storage_path}")
            else:
                # Default to dating_show_env for new simulations
                storage_path = dating_show_path
                print(f"🎯 [DEBUG] No existing simulation found, defaulting to: {storage_path}")
        
        _enhanced_step_manager = EnhancedStepFileManager(storage_path)
    
    return _enhanced_step_manager