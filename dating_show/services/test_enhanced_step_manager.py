"""
Test Suite for Enhanced Step File Manager
Tests proactive generation, fallback mechanisms, and integration
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path
import pytest
import logging

from .enhanced_step_manager import (
    EnhancedStepFileManager, 
    StepGenerationResult,
    StepGenerationConfig,
    get_enhanced_step_manager
)
from .config_manager import ConfigManager, SimulationConfig

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEnhancedStepManager:
    """Test suite for EnhancedStepFileManager"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary storage directory
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_sim_code = "test_dating_show"
        
        # Initialize step manager with test directory
        self.step_manager = EnhancedStepFileManager(str(self.temp_dir))
        
        logger.info(f"🧪 Test setup: {self.temp_dir}")
    
    def teardown_method(self):
        """Cleanup test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        logger.info("🧹 Test cleanup completed")
    
    def create_test_simulation_structure(self):
        """Create basic simulation structure for testing"""
        sim_path = self.temp_dir / self.test_sim_code
        
        # Create directories
        (sim_path / "environment").mkdir(parents=True, exist_ok=True)
        (sim_path / "movement").mkdir(parents=True, exist_ok=True)
        (sim_path / "personas").mkdir(parents=True, exist_ok=True)
        (sim_path / "reverie").mkdir(parents=True, exist_ok=True)
        
        # Create test agents
        test_agents = ["Alice", "Bob", "Charlie"]
        for agent in test_agents:
            (sim_path / "personas" / agent).mkdir(exist_ok=True)
        
        # Create initial step 0 files
        step_0_env = {
            "personas": {
                "Alice": {"x": 20, "y": 20, "action": "exploring"},
                "Bob": {"x": 30, "y": 30, "action": "socializing"},
                "Charlie": {"x": 40, "y": 40, "action": "relaxing"}
            },
            "meta": {
                "curr_time": "February 13, 2023, 00:00:00",
                "step": 0
            }
        }
        
        step_0_mov = {
            "persona": {
                "Alice": {"movement": [20, 20], "pronunciatio": "😊", "description": "Alice @ villa", "chat": ""},
                "Bob": {"movement": [30, 30], "pronunciatio": "💬", "description": "Bob @ villa", "chat": ""},
                "Charlie": {"movement": [40, 40], "pronunciatio": "✨", "description": "Charlie @ villa", "chat": ""}
            },
            "meta": {
                "curr_time": "February 13, 2023, 00:00:00",
                "step": 0
            }
        }
        
        # Save initial files
        with open(sim_path / "environment" / "0.json", 'w') as f:
            json.dump(step_0_env, f, indent=2)
        
        with open(sim_path / "movement" / "0.json", 'w') as f:
            json.dump(step_0_mov, f, indent=2)
        
        return sim_path
    
    async def test_files_exist_check(self):
        """Test file existence checking"""
        # Setup test simulation
        self.create_test_simulation_structure()
        
        # Test existing files
        assert await self.step_manager._files_exist(self.test_sim_code, 0)
        
        # Test non-existing files
        assert not await self.step_manager._files_exist(self.test_sim_code, 1)
        
        logger.info("✅ File existence check test passed")
    
    async def test_previous_step_generation(self):
        """Test generation from previous step"""
        # Setup test simulation
        self.create_test_simulation_structure()
        
        # Generate step 1 from step 0
        result = await self.step_manager.ensure_step_files_exist(self.test_sim_code, 1)
        
        assert result.success
        assert result.strategy_used == "previous_step"
        assert len(result.files_created) == 2
        
        # Verify files were created
        sim_path = self.temp_dir / self.test_sim_code
        assert (sim_path / "environment" / "1.json").exists()
        assert (sim_path / "movement" / "1.json").exists()
        
        # Verify content evolution
        with open(sim_path / "environment" / "1.json") as f:
            env_data = json.load(f)
        
        assert env_data["meta"]["step"] == 1
        assert "enhancement_applied" in env_data["meta"]
        
        logger.info("✅ Previous step generation test passed")
    
    async def test_minimal_fallback_generation(self):
        """Test minimal fallback generation"""
        # Create simulation with personas but no step files
        sim_path = self.temp_dir / self.test_sim_code
        (sim_path / "personas" / "TestAgent1").mkdir(parents=True, exist_ok=True)
        (sim_path / "personas" / "TestAgent2").mkdir(parents=True, exist_ok=True)
        
        # Test minimal generation for step 1 (no previous step available)
        result = await self.step_manager._create_minimal_step_files(self.test_sim_code, 1)
        
        assert result.success
        assert result.strategy_used == "minimal_fallback"
        
        # Verify minimal files were created
        assert (sim_path / "environment" / "1.json").exists()
        assert (sim_path / "movement" / "1.json").exists()
        
        # Verify content
        with open(sim_path / "environment" / "1.json") as f:
            env_data = json.load(f)
        
        assert env_data["meta"]["generation_strategy"] == "minimal_fallback"
        
        logger.info("✅ Minimal fallback generation test passed")
    
    async def test_dating_context_enhancement(self):
        """Test dating show context enhancements"""
        # Create test environment data
        test_env = {
            "personas": {
                "TestAgent": {"x": 50, "y": 50, "action": "idle"}
            },
            "meta": {"step": 0}
        }
        
        # Apply dating context
        enhanced = await self.step_manager._apply_dating_context(test_env, 1)
        
        assert enhanced["meta"]["step"] == 1
        assert enhanced["meta"]["context"] == "dating_show_villa"
        assert "enhancement_applied" in enhanced["meta"]
        
        # Check that idle action was replaced
        agent_action = enhanced["personas"]["TestAgent"]["action"]
        assert agent_action != "idle"
        assert agent_action in self.step_manager.villa_activities
        
        logger.info("✅ Dating context enhancement test passed")
    
    async def test_position_evolution(self):
        """Test agent position evolution"""
        test_mov = {
            "persona": {
                "TestAgent": {"movement": [50, 50], "description": "old description"}
            },
            "meta": {"step": 0}
        }
        
        # Apply position evolution
        evolved = await self.step_manager._evolve_agent_positions(test_mov, 1)
        
        assert evolved["meta"]["step"] == 1
        assert "position_evolution_applied" in evolved["meta"]
        
        # Check position changed (within bounds)
        new_pos = evolved["persona"]["TestAgent"]["movement"]
        assert len(new_pos) == 2
        assert 15 <= new_pos[0] <= 90
        assert 15 <= new_pos[1] <= 75
        
        # Position should be different (with very high probability)
        assert new_pos != [50, 50]
        
        logger.info("✅ Position evolution test passed")
    
    async def test_full_generation_workflow(self):
        """Test complete generation workflow with multiple strategies"""
        # Setup simulation with only step 0
        self.create_test_simulation_structure()
        
        # Test step 1 generation (should use previous_step strategy)
        result1 = await self.step_manager.ensure_step_files_exist(self.test_sim_code, 1)
        assert result1.success
        assert result1.strategy_used == "previous_step"
        
        # Test step 2 generation (should also use previous_step strategy)
        result2 = await self.step_manager.ensure_step_files_exist(self.test_sim_code, 2)
        assert result2.success
        assert result2.strategy_used == "previous_step"
        
        # Test existing file check
        result_existing = await self.step_manager.ensure_step_files_exist(self.test_sim_code, 1)
        assert result_existing.success
        assert result_existing.strategy_used == "existing_files"
        
        logger.info("✅ Full generation workflow test passed")
    
    async def test_error_handling(self):
        """Test error handling and recovery"""
        # Test generation for non-existent simulation
        result = await self.step_manager.ensure_step_files_exist("nonexistent_sim", 1)
        assert not result.success
        assert "error" in result.__dict__
        
        logger.info("✅ Error handling test passed")
    
    def test_config_integration(self):
        """Test configuration system integration"""
        # Create test config
        config = StepGenerationConfig(
            strategy="proactive",
            fallback_enabled=True,
            max_retries=5
        )
        
        step_manager = EnhancedStepFileManager(str(self.temp_dir), config)
        
        assert step_manager.config.strategy == "proactive"
        assert step_manager.config.max_retries == 5
        
        logger.info("✅ Config integration test passed")

async def run_test_suite():
    """Run complete test suite"""
    test_instance = TestEnhancedStepManager()
    
    tests = [
        test_instance.test_files_exist_check,
        test_instance.test_previous_step_generation, 
        test_instance.test_minimal_fallback_generation,
        test_instance.test_dating_context_enhancement,
        test_instance.test_position_evolution,
        test_instance.test_full_generation_workflow,
        test_instance.test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test_instance.setup_method()
            await test()
            test_instance.teardown_method()
            passed += 1
            logger.info(f"✅ {test.__name__} PASSED")
        except Exception as e:
            test_instance.teardown_method()
            failed += 1
            logger.error(f"❌ {test.__name__} FAILED: {e}")
    
    # Run sync test
    try:
        test_instance.setup_method()
        test_instance.test_config_integration()
        test_instance.teardown_method()
        passed += 1
        logger.info("✅ test_config_integration PASSED")
    except Exception as e:
        test_instance.teardown_method()
        failed += 1
        logger.error(f"❌ test_config_integration FAILED: {e}")
    
    logger.info(f"🏁 Test Results: {passed} passed, {failed} failed")
    return passed, failed

if __name__ == "__main__":
    asyncio.run(run_test_suite())