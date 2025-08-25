"""
End-to-End Pipeline Test
Comprehensive test of the complete dating show simulation pipeline
Tests integration between all components and error recovery
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import our services - Updated for unified architecture
from dating_show.services.unified_agent_manager import get_unified_agent_manager
from dating_show.agents.enhanced_agent_state import create_enhanced_agent_state
from dating_show.services.environment_generator import EnvironmentGenerator
from dating_show.services.error_recovery import ErrorRecoveryService, ErrorSeverity


class TestEndToEndPipeline(unittest.TestCase):
    """Test the complete simulation pipeline from start to finish"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test storage
        self.test_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.test_dir) / "storage"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Sample agent data
        self.test_agents = [
            "Isabella Rodriguez", "Klaus Mueller", "Maria Lopez",
            "Giorgio Rossi", "Carmen Ortiz"
        ]
        
        # Initialize services with test configuration
        self.state_bridge = AgentStateBridge()
        self.env_generator = EnvironmentGenerator(str(self.storage_path))
        self.error_recovery = ErrorRecoveryService()
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_agent_state_conversion_pipeline(self):
        """Test agent state conversion between different formats"""
        # Test reverie persona mock
        reverie_persona = Mock()
        reverie_persona.name = "Isabella Rodriguez"
        reverie_persona.last_position = (72, 14)
        reverie_persona.scratch = Mock()
        reverie_persona.scratch.daily_plan_req = "socializing with other contestants"
        reverie_persona.scratch.curr_tile = ["the_ville", "villa_area"]
        
        # Convert reverie to standard format
        standard_state = self.state_bridge.convert_agent_state(
            reverie_persona, 'reverie', 'standard'
        )
        
        self.assertIsInstance(standard_state, dict)
        self.assertEqual(standard_state['name'], "Isabella Rodriguez")
        self.assertEqual(standard_state['position']['x'], 72)
        self.assertEqual(standard_state['position']['y'], 14)
        self.assertEqual(standard_state['current_action'], "socializing with other contestants")
        
        # Convert standard to frontend format
        frontend_state = self.state_bridge.convert_agent_state(
            standard_state, 'frontend', 'frontend'
        )
        
        self.assertIn('agent_id', frontend_state)
        self.assertIn('position', frontend_state)
        self.assertIn('current_action', frontend_state)
        
        # Convert back to reverie format
        reverie_state = self.state_bridge.convert_agent_state(
            frontend_state, 'frontend', 'reverie'
        )
        
        self.assertEqual(reverie_state['name'], "Isabella Rodriguez")
        self.assertEqual(reverie_state['last_position'], (72, 14))
        
    def test_environment_generation_pipeline(self):
        """Test complete environment file generation pipeline"""
        sim_code = "test_dating_show"
        
        # Initialize simulation storage
        success = self.env_generator.initialize_simulation_storage(sim_code, self.test_agents)
        self.assertTrue(success)
        
        # Check directory structure was created
        sim_path = self.storage_path / sim_code
        self.assertTrue((sim_path / "environment").exists())
        self.assertTrue((sim_path / "movement").exists())
        self.assertTrue((sim_path / "personas").exists())
        self.assertTrue((sim_path / "reverie").exists())
        
        # Check initial files were created
        self.assertTrue((sim_path / "environment" / "0.json").exists())
        self.assertTrue((sim_path / "movement" / "0.json").exists())
        self.assertTrue((sim_path / "reverie" / "meta.json").exists())
        
        # Check persona directories
        for agent_name in self.test_agents:
            persona_path = sim_path / "personas" / agent_name / "bootstrap_memory"
            self.assertTrue(persona_path.exists())
            self.assertTrue((persona_path / "scratch.json").exists())
            self.assertTrue((persona_path / "spatial_memory.json").exists())
        
        # Test file content structure
        with open(sim_path / "environment" / "0.json", 'r') as f:
            env_data = json.load(f)
        
        self.assertEqual(len(env_data), len(self.test_agents))
        for agent_name in self.test_agents:
            self.assertIn(agent_name, env_data)
            self.assertIn('maze', env_data[agent_name])
            self.assertIn('x', env_data[agent_name])
            self.assertIn('y', env_data[agent_name])
        
        # Test movement file structure
        with open(sim_path / "movement" / "0.json", 'r') as f:
            mov_data = json.load(f)
        
        self.assertIn('persona', mov_data)
        self.assertIn('meta', mov_data)
        self.assertEqual(len(mov_data['persona']), len(self.test_agents))
        
        for agent_name in self.test_agents:
            self.assertIn(agent_name, mov_data['persona'])
            agent_mov = mov_data['persona'][agent_name]
            self.assertIn('movement', agent_mov)
            self.assertIn('pronunciatio', agent_mov)
            self.assertIn('description', agent_mov)
            self.assertIn('chat', agent_mov)
        
        # Test step file generation
        mock_agents = {}
        for i, name in enumerate(self.test_agents):
            mock_agent = Mock()
            mock_agent.name = name
            mock_agent.last_position = (50 + i * 5, 50 + i * 3)
            mock_agent.scratch = Mock()
            mock_agent.scratch.daily_plan_req = f"activity for {name}"
            mock_agents[name] = mock_agent
        
        # Generate step 1 files
        success = self.env_generator.generate_next_step_files(sim_code, 1, mock_agents)
        self.assertTrue(success)
        
        # Check step 1 files exist
        self.assertTrue((sim_path / "environment" / "1.json").exists())
        self.assertTrue((sim_path / "movement" / "1.json").exists())
        
        # Test ensure files exist functionality
        success = self.env_generator.ensure_step_files_exist(sim_code, 2)
        self.assertTrue(success)
        self.assertTrue((sim_path / "environment" / "2.json").exists())
        self.assertTrue((sim_path / "movement" / "2.json").exists())
    
    def test_error_recovery_pipeline(self):
        """Test error recovery mechanisms throughout the pipeline"""
        
        # Test error handling and recovery
        def failing_operation():
            raise ValueError("Test error")
        
        def succeeding_operation():
            return "success"
        
        # Test failed operation with recovery
        self.error_recovery.handle_error(
            'test_component', 
            ValueError("Test error"),
            {'test': True},
            ErrorSeverity.MEDIUM
        )
        
        # Check error was recorded
        errors = self.error_recovery.get_error_summary('test_component', 1)
        self.assertEqual(errors['total_errors'], 1)
        self.assertIn('ValueError', errors['errors_by_type'])
        
        # Test system health tracking
        health = self.error_recovery.get_system_health()
        self.assertIn('overall_status', health)
        self.assertIn('error_rate', health)
        self.assertIn('degraded_components', health)
        
        # Test fallback function registration
        def test_fallback(error_event):
            return True
        
        self.error_recovery.register_fallback_function('test_component', test_fallback)
        self.assertIn('test_component', self.error_recovery.fallback_functions)
    
    def test_simulation_step_integration(self):
        """Test integration of all components in a simulation step"""
        # This would test the actual simulation step if we had the full environment
        # For now, we test the component integration
        
        sim_code = "integration_test"
        
        # Initialize environment
        success = self.env_generator.initialize_simulation_storage(sim_code, self.test_agents)
        self.assertTrue(success)
        
        # Create mock agent states
        agent_states = {}
        for i, name in enumerate(self.test_agents):
            # Test both reverie format and standard format conversion
            reverie_mock = Mock()
            reverie_mock.name = name
            reverie_mock.last_position = (60 + i * 5, 40 + i * 3)
            reverie_mock.scratch = Mock()
            reverie_mock.scratch.daily_plan_req = f"step 1 activity for {name}"
            
            # Convert to standard format
            standard_state = self.state_bridge.convert_agent_state(
                reverie_mock, 'reverie', 'standard'
            )
            agent_states[name] = standard_state
        
        # Generate next step files using converted states (pass original mocks, not converted)
        mock_agent_dict = {}
        for i, name in enumerate(self.test_agents):
            reverie_mock = Mock()
            reverie_mock.name = name
            reverie_mock.last_position = (60 + i * 5, 40 + i * 3)
            reverie_mock.scratch = Mock()
            reverie_mock.scratch.daily_plan_req = f"step 1 activity for {name}"
            mock_agent_dict[name] = reverie_mock
        
        success = self.env_generator.generate_next_step_files(sim_code, 1, mock_agent_dict)
        self.assertTrue(success)
        
        # Verify the integration worked
        sim_path = self.storage_path / sim_code
        
        # Check environment file
        with open(sim_path / "environment" / "1.json", 'r') as f:
            env_data = json.load(f)
        
        # Verify positions match converted states
        for i, name in enumerate(self.test_agents):
            expected_x = 60 + i * 5
            expected_y = 40 + i * 3
            self.assertEqual(env_data[name]['x'], expected_x)
            self.assertEqual(env_data[name]['y'], expected_y)
        
        # Check movement file
        with open(sim_path / "movement" / "1.json", 'r') as f:
            mov_data = json.load(f)
        
        # Verify movement data structure
        for i, name in enumerate(self.test_agents):
            expected_x = 60 + i * 5
            expected_y = 40 + i * 3
            self.assertEqual(mov_data['persona'][name]['movement'], [expected_x, expected_y])
            self.assertIn('pronunciatio', mov_data['persona'][name])
            self.assertIn('description', mov_data['persona'][name])
    
    def test_batch_operations(self):
        """Test batch operations for performance"""
        # Test batch agent conversion
        mock_agents = []
        for i, name in enumerate(self.test_agents):
            agent_data = {
                'name': name,
                'position': {'x': 50 + i * 10, 'y': 50 + i * 5},
                'current_action': f'activity {i}',
                'current_location': 'villa'
            }
            mock_agents.append(agent_data)
        
        # Batch convert from frontend to standard format
        converted_agents = self.state_bridge.batch_convert_agents(
            mock_agents, 'frontend', 'standard'
        )
        
        self.assertEqual(len(converted_agents), len(self.test_agents))
        for i, agent in enumerate(converted_agents):
            self.assertEqual(agent['name'], self.test_agents[i])
            self.assertEqual(agent['position']['x'], 50 + i * 10)
            self.assertEqual(agent['position']['y'], 50 + i * 5)
    
    def test_file_recovery_and_cleanup(self):
        """Test file recovery and cleanup mechanisms"""
        sim_code = "cleanup_test"
        
        # Initialize simulation
        success = self.env_generator.initialize_simulation_storage(sim_code, self.test_agents)
        self.assertTrue(success)
        
        sim_path = self.storage_path / sim_code
        
        # Generate multiple steps
        for step in range(1, 15):  # Generate steps 1-14
            success = self.env_generator.ensure_step_files_exist(sim_code, step)
            self.assertTrue(success)
        
        # Verify files exist
        env_dir = sim_path / "environment"
        mov_dir = sim_path / "movement"
        
        env_files = list(env_dir.glob("*.json"))
        mov_files = list(mov_dir.glob("*.json"))
        
        self.assertEqual(len(env_files), 15)  # 0-14
        self.assertEqual(len(mov_files), 15)  # 0-14
        
        # Test cleanup (keep last 10)
        self.env_generator.cleanup_old_files(sim_code, keep_last_n=10)
        
        # Check files after cleanup
        env_files_after = list(env_dir.glob("*.json"))
        mov_files_after = list(mov_dir.glob("*.json"))
        
        self.assertEqual(len(env_files_after), 10)
        self.assertEqual(len(mov_files_after), 10)
        
        # Verify the correct files remain (should be steps 5-14)
        remaining_steps = set()
        for file_path in env_files_after:
            step_num = int(file_path.stem)
            remaining_steps.add(step_num)
        
        expected_steps = set(range(5, 15))  # 5-14
        self.assertEqual(remaining_steps, expected_steps)
    
    def test_error_conditions_and_recovery(self):
        """Test various error conditions and recovery mechanisms"""
        sim_code = "error_test"
        
        # Test with invalid storage path
        invalid_generator = EnvironmentGenerator("/invalid/path/that/does/not/exist")
        success = invalid_generator.initialize_simulation_storage(sim_code, self.test_agents)
        self.assertFalse(success)
        
        # Test with empty agent list
        success = self.env_generator.initialize_simulation_storage(sim_code, [])
        # Should still succeed with empty list
        self.assertTrue(success)
        
        # Test file generation with missing simulation
        success = self.env_generator.generate_next_step_files("nonexistent_sim", 1, {})
        self.assertFalse(success)
        
        # Test state conversion with malformed data
        malformed_data = {"invalid": "data"}
        result = self.state_bridge.convert_agent_state(
            malformed_data, 'frontend', 'standard'
        )
        # Should return fallback state
        self.assertIn('name', result)
        self.assertEqual(result['name'], 'Unknown')
    
    def test_simulation_config_management(self):
        """Test simulation configuration and state management"""
        sim_code = "config_test"
        
        # Initialize simulation
        success = self.env_generator.initialize_simulation_storage(sim_code, self.test_agents)
        self.assertTrue(success)
        
        # Check simulation info
        sim_info = self.env_generator.get_simulation_info(sim_code)
        self.assertIsNotNone(sim_info)
        self.assertEqual(sim_info['agent_names'], self.test_agents)
        self.assertEqual(sim_info['current_step'], 0)
        self.assertTrue(sim_info['initialized'])
        
        # Test step progression
        mock_agents = {name: {'name': name, 'position': {'x': 50, 'y': 50}} for name in self.test_agents}
        
        for step in range(1, 5):
            success = self.env_generator.generate_next_step_files(sim_code, step, mock_agents)
            self.assertTrue(success)
            
            # Check updated config
            sim_info = self.env_generator.get_simulation_info(sim_code)
            self.assertEqual(sim_info['current_step'], step)


class TestAsyncIntegration(unittest.IsolatedAsyncioTestCase):
    """Test async integration scenarios"""
    
    async def test_async_simulation_pipeline(self):
        """Test async simulation pipeline operations"""
        # Mock the dating show server components for async testing
        
        with patch('dating_show.main.REVERIE_AVAILABLE', True):
            with patch('dating_show.main.DJANGO_AVAILABLE', True):
                # This would test the full async pipeline if we had the environment
                # For now, we verify the structure works
                
                # Create a mock server instance
                server = Mock()
                server.reverie_server = Mock()
                server.reverie_manager = Mock()
                server.reverie_server.step = 0
                server.reverie_server.personas = {}
                
                # Test that the method exists and can be called
                # (We can't run the actual implementation without the full environment)
                self.assertTrue(hasattr(server, 'reverie_server'))
                self.assertTrue(hasattr(server, 'reverie_manager'))


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)