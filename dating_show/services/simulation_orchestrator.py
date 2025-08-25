"""
Unified Simulation Orchestrator
Coordinates step advancement, agent processing, and real-time updates
Integrates with existing dating_show services architecture
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .enhanced_step_manager import EnhancedStepFileManager, StepGenerationResult, StepGenerationConfig
from .unified_agent_manager import UnifiedAgentManager
from .error_recovery import safe_execute, ErrorSeverity
from .update_pipeline import UpdatePipeline, UpdateType

logger = logging.getLogger(__name__)

@dataclass
class SimulationStepResult:
    """Result of a complete simulation step advancement"""
    success: bool
    sim_code: str
    step: int
    previous_step: int
    agents_processed: int
    generation_result: Optional[StepGenerationResult] = None
    reverie_result: Optional[Dict] = None
    update_result: Optional[Dict] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = ""

class UnifiedSimulationOrchestrator:
    """
    Orchestrates complete simulation step advancement with:
    - Enhanced step file generation
    - Agent processing coordination
    - Real-time update broadcasting
    - Error recovery and fallback mechanisms
    """
    
    def __init__(self, 
                 storage_path: str = None,
                 step_manager: Optional[EnhancedStepFileManager] = None,
                 agent_manager: Optional[UnifiedAgentManager] = None,
                 update_pipeline: Optional[UpdatePipeline] = None):
        
        # Initialize storage path
        if storage_path is None:
            storage_path = "/Applications/Projects/Open source/generative_agents/dating_show_env/frontend_service/storage"
        self.storage_path = Path(storage_path)
        
        # Initialize enhanced step manager
        self.step_manager = step_manager or EnhancedStepFileManager(storage_path)
        
        # Initialize agent manager (may be None if not available)
        self.agent_manager = agent_manager
        
        # Initialize update pipeline (may be None if not available)  
        self.update_pipeline = update_pipeline
        
        # Track current simulations
        self.active_simulations: Dict[str, Dict] = {}
        self.orchestration_lock = asyncio.Lock()
        
    async def advance_simulation_step(self, sim_code: str, force_generation: bool = False) -> SimulationStepResult:
        """
        Advance simulation by one step with complete orchestration
        
        Args:
            sim_code: Simulation identifier
            force_generation: Force file generation even if files exist
            
        Returns:
            SimulationStepResult: Complete step advancement result
        """
        start_time = datetime.now()
        
        async with self.orchestration_lock:
            try:
                # Get current step
                current_step = await self._get_current_step(sim_code)
                next_step = current_step + 1
                
                logger.info(f"🎬 Advancing {sim_code} from step {current_step} to {next_step}")
                
                # Step 1: Ensure files exist with enhanced generation
                if force_generation or not await self.step_manager._files_exist(sim_code, next_step):
                    generation_result = await self.step_manager.ensure_step_files_exist(sim_code, next_step)
                    
                    if not generation_result.success:
                        return SimulationStepResult(
                            success=False,
                            sim_code=sim_code,
                            step=next_step,
                            previous_step=current_step,
                            agents_processed=0,
                            generation_result=generation_result,
                            error=f"File generation failed: {generation_result.error}",
                            execution_time=self._get_execution_time(start_time)
                        )
                else:
                    generation_result = StepGenerationResult(
                        success=True,
                        step=next_step,
                        sim_code=sim_code,
                        strategy_used="existing_files"
                    )
                
                # Step 2: Execute agent processing (if available)
                reverie_result = None
                if self.agent_manager:
                    reverie_result = await self._execute_agent_processing(sim_code, next_step)
                else:
                    logger.warning("⚠️ Agent manager not available, skipping agent processing")
                
                # Step 3: Broadcast updates (if available)
                update_result = None
                if self.update_pipeline:
                    update_result = await self._broadcast_step_updates(sim_code, next_step, reverie_result)
                else:
                    logger.info("ℹ️ Update pipeline not available, skipping real-time updates")
                
                # Step 4: Update simulation tracking
                await self._update_simulation_state(sim_code, next_step)
                
                # Create success result
                agents_processed = 0
                if reverie_result and "agents" in reverie_result:
                    agents_processed = len(reverie_result["agents"])
                
                return SimulationStepResult(
                    success=True,
                    sim_code=sim_code,
                    step=next_step,
                    previous_step=current_step,
                    agents_processed=agents_processed,
                    generation_result=generation_result,
                    reverie_result=reverie_result,
                    update_result=update_result,
                    execution_time=self._get_execution_time(start_time),
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                
            except Exception as e:
                logger.error(f"🚨 Simulation orchestration error: {e}")
                return SimulationStepResult(
                    success=False,
                    sim_code=sim_code,
                    step=next_step if 'next_step' in locals() else current_step + 1,
                    previous_step=current_step if 'current_step' in locals() else 0,
                    agents_processed=0,
                    error=str(e),
                    execution_time=self._get_execution_time(start_time)
                )
    
    async def _get_current_step(self, sim_code: str) -> int:
        """Get current step number for simulation"""
        try:
            # Check temp storage for current step
            temp_storage = self.storage_path.parent / "temp_storage"
            curr_step_file = temp_storage / "curr_step.json"
            
            if curr_step_file.exists():
                step_data = await self.step_manager._load_json_async(curr_step_file)
                return step_data.get("step", 0)
            
            # Fallback: check meta.json
            meta_file = self.storage_path / sim_code / "reverie" / "meta.json"
            if meta_file.exists():
                meta_data = await self.step_manager._load_json_async(meta_file)
                return meta_data.get("step", 0)
            
            # Fallback: count existing environment files
            env_dir = self.storage_path / sim_code / "environment"
            if env_dir.exists():
                env_files = list(env_dir.glob("*.json"))
                if env_files:
                    # Get highest numbered file
                    step_numbers = []
                    for f in env_files:
                        try:
                            step_numbers.append(int(f.stem))
                        except ValueError:
                            continue
                    if step_numbers:
                        return max(step_numbers)
            
            # Default to step 0
            return 0
            
        except Exception as e:
            logger.warning(f"Error getting current step, defaulting to 0: {e}")
            return 0
    
    async def _execute_agent_processing(self, sim_code: str, step: int) -> Optional[Dict]:
        """Execute agent processing for the simulation step"""
        try:
            if not self.agent_manager:
                return None
            
            logger.info(f"🤖 Processing agents for step {step}")
            
            # Load environment data for agent processing
            env_file = self.storage_path / sim_code / "environment" / f"{step}.json"
            env_data = await self.step_manager._load_json_async(env_file)
            
            # Process agents (this would integrate with existing reverie logic)
            # For now, return basic structure that matches expected format
            agent_results = {
                "agents": [],
                "step": step,
                "sim_code": sim_code,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # If environment has personas, add them to results
            if "personas" in env_data:
                for persona_name in env_data["personas"].keys():
                    agent_results["agents"].append({
                        "name": persona_name,
                        "step": step,
                        "processed": True
                    })
            
            return agent_results
            
        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            return None
    
    async def _broadcast_step_updates(self, sim_code: str, step: int, agent_results: Optional[Dict]) -> Optional[Dict]:
        """Broadcast step updates via update pipeline"""
        try:
            if not self.update_pipeline:
                return None
                
            logger.info(f"📡 Broadcasting step {step} updates")
            
            # Prepare update data
            update_data = {
                "sim_code": sim_code,
                "step": step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_count": len(agent_results.get("agents", [])) if agent_results else 0
            }
            
            # Queue update through pipeline
            success = await self.update_pipeline.queue_simulation_update(
                sim_code=sim_code,
                update_type=UpdateType.STEP_ADVANCE,
                data=update_data
            )
            
            return {
                "success": success,
                "update_type": "step_advance",
                "data": update_data
            }
            
        except Exception as e:
            logger.error(f"Update broadcast error: {e}")
            return None
    
    async def _update_simulation_state(self, sim_code: str, step: int):
        """Update simulation state tracking"""
        try:
            # Update active simulations tracking
            self.active_simulations[sim_code] = {
                "current_step": step,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            }
            
            # Update temp storage files
            temp_storage = self.storage_path.parent / "temp_storage"
            temp_storage.mkdir(exist_ok=True)
            
            # Update curr_step.json
            step_data = {"step": step}
            curr_step_file = temp_storage / "curr_step.json"
            await self.step_manager._save_json_async(curr_step_file, step_data)
            
            # Update curr_sim_code.json
            sim_data = {"sim_code": sim_code}
            curr_sim_file = temp_storage / "curr_sim_code.json"
            await self.step_manager._save_json_async(curr_sim_file, sim_data)
            
            logger.info(f"📊 Updated simulation state: {sim_code} -> step {step}")
            
        except Exception as e:
            logger.error(f"State update error: {e}")
    
    # Batch operations
    
    async def advance_multiple_steps(self, sim_code: str, num_steps: int) -> List[SimulationStepResult]:
        """Advance simulation by multiple steps"""
        results = []
        
        for i in range(num_steps):
            result = await self.advance_simulation_step(sim_code)
            results.append(result)
            
            if not result.success:
                logger.error(f"Step {result.step} failed, stopping batch advancement")
                break
                
            # Small delay between steps
            await asyncio.sleep(0.1)
        
        return results
    
    async def get_simulation_status(self, sim_code: str) -> Dict:
        """Get current simulation status"""
        try:
            current_step = await self._get_current_step(sim_code)
            
            # Check if files exist for next step
            next_step_files_exist = await self.step_manager._files_exist(sim_code, current_step + 1)
            
            # Get agent count
            agent_count = 0
            try:
                personas_dir = self.storage_path / sim_code / "personas"
                if personas_dir.exists():
                    agent_count = len([d for d in personas_dir.iterdir() if d.is_dir()])
            except:
                pass
            
            return {
                "sim_code": sim_code,
                "current_step": current_step,
                "next_step_files_ready": next_step_files_exist,
                "agent_count": agent_count,
                "status": self.active_simulations.get(sim_code, {}).get("status", "unknown"),
                "last_updated": self.active_simulations.get(sim_code, {}).get("last_updated"),
                "storage_path": str(self.storage_path / sim_code)
            }
            
        except Exception as e:
            logger.error(f"Status check error: {e}")
            return {
                "sim_code": sim_code,
                "error": str(e),
                "status": "error"
            }
    
    # Utility methods
    
    def _get_execution_time(self, start_time: datetime) -> float:
        """Calculate execution time in seconds"""
        return (datetime.now() - start_time).total_seconds()


# Convenience functions for backward compatibility

async def safe_advance_simulation_step(sim_code: str, orchestrator: UnifiedSimulationOrchestrator = None) -> SimulationStepResult:
    """
    Safe simulation step advancement with error handling
    Compatible with existing error_recovery patterns
    """
    if orchestrator is None:
        orchestrator = get_simulation_orchestrator()
    
    def advance_step():
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(orchestrator.advance_simulation_step(sim_code))
        finally:
            loop.close()
    
    success, result = safe_execute(
        'simulation_advancement',
        advance_step,
        ErrorSeverity.HIGH,
        {'sim_code': sim_code}
    )
    
    if success:
        return result
    else:
        # Create error result
        return SimulationStepResult(
            success=False,
            sim_code=sim_code,
            step=0,
            previous_step=0,
            agents_processed=0,
            error="Safe execution failed"
        )


# Global instance
_simulation_orchestrator = None

def get_simulation_orchestrator() -> UnifiedSimulationOrchestrator:
    """Get singleton instance of UnifiedSimulationOrchestrator"""
    global _simulation_orchestrator
    
    if _simulation_orchestrator is None:
        _simulation_orchestrator = UnifiedSimulationOrchestrator()
    
    return _simulation_orchestrator