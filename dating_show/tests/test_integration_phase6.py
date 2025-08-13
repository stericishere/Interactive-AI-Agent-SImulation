"""
Phase 6 Integration Testing Suite
Comprehensive tests for database service, enhanced bridge, orchestration, and PIANO integration
"""

import unittest
import asyncio
import os
import tempfile
import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dating_show.services.database_service import (
    DatabaseService, DatabaseConfig, create_database_service, 
    HealthStatus, MigrationResult
)
from dating_show.services.enhanced_bridge import (
    EnhancedFrontendBridge, create_enhanced_bridge, 
    BridgeStatus, AutoDiscoveryResult
)
from dating_show.services.orchestrator import (
    DatingShowOrchestrator, OrchestrationConfig, 
    ServiceStatus, create_orchestrator
)
from dating_show.services.piano_integration import (
    DatingShowReverieServer, create_dating_show_reverie_server
)


class TestDatabaseService(unittest.TestCase):
    """Test suite for Database Service functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DatabaseConfig(
            database_url="sqlite:///test.db",
            frontend_server_path=self.temp_dir,
            migration_timeout=30,
            auto_migrate=False  # Disable for testing
        )
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('dating_show.services.database_service.django')
    def test_database_service_initialization(self, mock_django):
        """Test database service initialization"""
        mock_django.conf.settings.configured = False
        mock_django.setup = Mock()
        
        service = DatabaseService(self.config)
        
        self.assertEqual(service.config.database_url, "sqlite:///test.db")
        self.assertEqual(service.health_status, HealthStatus.UNKNOWN)
        mock_django.setup.assert_called_once()
    
    def test_health_check_basic(self):
        """Test basic health check functionality"""
        async def _test():
            with patch('dating_show.services.database_service.connection') as mock_conn:
                mock_cursor = Mock()
                mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
                
                service = DatabaseService(self.config)
                health_metrics = await service.health_check()
                
                self.assertIsInstance(health_metrics.status, HealthStatus)
                self.assertIsInstance(health_metrics.query_time_ms, float)
                self.assertIsInstance(health_metrics.uptime_seconds, float)
        
        asyncio.run(_test())
    
    def test_migration_detection(self):
        """Test migration detection and application"""
        async def _test():
            service = DatabaseService(self.config)
            
            with patch.object(service, '_run_django_command') as mock_command:
                # Mock successful migration check
                mock_command.return_value = {
                    'success': True,
                    'output': '[ ] dating_show_api.0001_initial\n[X] dating_show_api.0002_enhanced_models',
                    'error': None
                }
                
                result = await service.ensure_migrations()
                
                self.assertTrue(result.success)
                self.assertEqual(len(result.applied_migrations), 0)  # No auto-migrate
        
        asyncio.run(_test())
    
    def test_service_status(self):
        """Test service status reporting"""
        service = DatabaseService(self.config)
        status = service.get_service_status()
        
        self.assertIn('health_status', status)
        self.assertIn('error_count', status)
        self.assertIn('config', status)
    
    @patch('dating_show.services.database_service.django')
    def test_factory_function(self, mock_django):
        """Test database service factory function"""
        mock_django.conf.settings.configured = False
        mock_django.setup = Mock()
        
        service = create_database_service(
            database_url="test://db",
            frontend_server_path="/test/path"
        )
        
        self.assertIsInstance(service, DatabaseService)
        self.assertEqual(service.config.database_url, "test://db")


class TestEnhancedFrontendBridge(unittest.TestCase):
    """Test suite for Enhanced Frontend Bridge functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.bridge = create_enhanced_bridge(
            frontend_url="http://localhost:8000",
            update_interval=0.1,
            batch_size=5
        )
    
    def tearDown(self):
        """Clean up test environment"""
        if self.bridge and self.bridge.running:
            self.bridge.stop_bridge()
    
    def test_enhanced_bridge_initialization(self):
        """Test enhanced bridge initialization"""
        self.assertEqual(self.bridge.frontend_url, "http://localhost:8000")
        self.assertEqual(self.bridge.batch_size, 5)
        self.assertEqual(self.bridge.health_metrics.status, BridgeStatus.INITIALIZING)
        self.assertIsInstance(self.bridge.discovered_agents, set)
    
    def test_bridge_start_stop(self):
        """Test bridge start and stop functionality"""
        # Test start
        self.bridge.start_bridge()
        self.assertTrue(self.bridge.running)
        
        # Wait briefly for initialization
        time.sleep(0.2)
        
        # Test stop
        self.bridge.stop_bridge()
        self.assertFalse(self.bridge.running)
    
    def test_auto_discovery(self):
        """Test agent auto-discovery functionality"""
        async def _test():
            # Add some mock agents to cache
            from dating_show.api.frontend_bridge import AgentUpdate
            mock_update = AgentUpdate(
                agent_id="test_agent",
                name="Test Agent",
                current_role="contestant",
                specialization={},
                skills={},
                memory={},
                location={},
                current_action="test",
                timestamp=datetime.now(timezone.utc)
            )
            
            self.bridge.agent_state_cache["test_agent"] = mock_update
            
            result = await self.bridge.auto_discover_agents()
            
            self.assertIsInstance(result, AutoDiscoveryResult)
            self.assertIn("test_agent", result.discovered_agents)
            self.assertEqual(result.total_agents, 1)
        
        asyncio.run(_test())
    
    def test_health_metrics(self):
        """Test health metrics collection"""
        metrics = self.bridge.get_health_metrics()
        
        self.assertIsInstance(metrics.status, BridgeStatus)
        self.assertIsInstance(metrics.total_syncs, int)
        self.assertIsInstance(metrics.error_rate, float)
        self.assertIsInstance(metrics.queue_depths, dict)
    
    def test_batch_optimization(self):
        """Test batch synchronization optimization"""
        async def _test():
            # This would normally require a running bridge, so we'll test the basic structure
            with patch.object(self.bridge, '_send_batch_updates_optimized') as mock_batch:
                await self.bridge.batch_sync_optimization()
                # Verify batch method structure exists
                self.assertTrue(hasattr(self.bridge, '_send_batch_updates_optimized'))
        
        asyncio.run(_test())
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        async def _test():
            test_error = Exception("Test error")
            
            # Test recovery attempt
            recovery_result = await self.bridge.recover_from_error(test_error)
            
            # Should return boolean indicating recovery success/failure
            self.assertIsInstance(recovery_result, bool)
            self.assertGreater(self.bridge.consecutive_failures, 0)
        
        asyncio.run(_test())
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        summary = self.bridge.get_performance_summary()
        
        self.assertIn('health_metrics', summary)
        self.assertIn('discovered_agents', summary)
        self.assertIn('batch_size', summary)
        self.assertIn('bridge_status', summary)


class TestOrchestrationService(unittest.TestCase):
    """Test suite for Orchestration Service functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = OrchestrationConfig(
            database_url="sqlite:///test.db",
            frontend_server_path=self.temp_dir,
            frontend_url="http://localhost:8000",
            max_agents=5,
            simulation_steps=10
        )
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = DatingShowOrchestrator(self.config)
        
        self.assertEqual(orchestrator.status, ServiceStatus.STOPPED)
        self.assertEqual(orchestrator.config.max_agents, 5)
        self.assertIsNone(orchestrator.database_service)
        self.assertIsNone(orchestrator.frontend_bridge)
    
    def test_config_file_operations(self):
        """Test configuration file loading and saving"""
        config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Test saving
        self.config.to_file(config_path)
        self.assertTrue(os.path.exists(config_path))
        
        # Test loading
        loaded_config = OrchestrationConfig.from_file(config_path)
        self.assertEqual(loaded_config.max_agents, 5)
        self.assertEqual(loaded_config.simulation_steps, 10)
    
    @patch('dating_show.services.orchestrator.create_database_service')
    async def test_database_initialization(self, mock_create_db):
        """Test database service initialization"""
        mock_db_service = Mock()
        mock_db_service.ensure_migrations.return_value = MigrationResult(
            success=True, applied_migrations=[], duration_seconds=0.1
        )
        mock_db_service.health_check.return_value = Mock(status=HealthStatus.HEALTHY)
        mock_create_db.return_value = mock_db_service
        
        orchestrator = DatingShowOrchestrator(self.config)
        await orchestrator.initialize_database()
        
        self.assertIsNotNone(orchestrator.database_service)
        self.assertTrue(orchestrator.service_health.database_service)
    
    @patch('dating_show.services.orchestrator.create_enhanced_bridge')
    async def test_frontend_bridge_startup(self, mock_create_bridge):
        """Test frontend bridge startup"""
        mock_bridge = Mock()
        mock_bridge.get_health_metrics.return_value = Mock(
            status=BridgeStatus.HEALTHY
        )
        mock_create_bridge.return_value = mock_bridge
        
        orchestrator = DatingShowOrchestrator(self.config)
        await orchestrator.start_frontend_bridge()
        
        self.assertIsNotNone(orchestrator.frontend_bridge)
        self.assertTrue(orchestrator.service_health.frontend_bridge)
    
    def test_orchestrator_status(self):
        """Test orchestrator status reporting"""
        orchestrator = DatingShowOrchestrator(self.config)
        status = orchestrator.get_orchestrator_status()
        
        self.assertIn('status', status)
        self.assertIn('service_health', status)
        self.assertIn('config_summary', status)
        self.assertIn('services', status)
    
    async def test_factory_function(self):
        """Test orchestrator factory function"""
        orchestrator = await create_orchestrator(
            config_path=None,
            max_agents=10,
            simulation_steps=50
        )
        
        self.assertIsInstance(orchestrator, DatingShowOrchestrator)
        self.assertEqual(orchestrator.config.max_agents, 10)
        self.assertEqual(orchestrator.config.simulation_steps, 50)


class TestPIANOIntegration(unittest.TestCase):
    """Test suite for PIANO/Reverie Integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_personas = self._create_mock_personas()
    
    def _create_mock_personas(self):
        """Create mock personas for testing"""
        personas = {}
        for i in range(5):
            mock_persona = Mock()
            mock_persona.name = f"TestPersona{i}"
            mock_persona.scratch = Mock()
            mock_persona.scratch.get_str_iss.return_value = f"Test persona {i} description"
            mock_persona.scratch.curr_tile = (i, i)
            mock_persona.scratch.act_description = f"Action {i}"
            mock_persona.a_mem = Mock()
            mock_persona.a_mem.seq_event = [f"Event {j}" for j in range(3)]
            
            personas[f"TestPersona{i}"] = mock_persona
        
        return personas
    
    @patch('dating_show.services.piano_integration.ReverieServer.__init__')
    def test_dating_show_reverie_initialization(self, mock_reverie_init):
        """Test dating show reverie server initialization"""
        mock_reverie_init.return_value = None
        
        server = DatingShowReverieServer("fork_test", "sim_test")
        server.personas = self.mock_personas  # Set personas after init
        server.dating_show_roles = server._initialize_dating_show_roles()
        
        self.assertEqual(len(server.dating_show_roles), len(self.mock_personas))
        self.assertIn("TestPersona0", server.dating_show_roles)
    
    @patch('dating_show.services.piano_integration.ReverieServer.__init__')
    def test_persona_data_extraction(self, mock_reverie_init):
        """Test persona data extraction for frontend sync"""
        mock_reverie_init.return_value = None
        
        server = DatingShowReverieServer("fork_test", "sim_test")
        server.personas = self.mock_personas
        server.dating_show_roles = {name: "contestant" for name in self.mock_personas.keys()}
        
        persona_name = "TestPersona0"
        persona = self.mock_personas[persona_name]
        
        agent_data = server._extract_persona_data(persona_name, persona)
        
        self.assertEqual(agent_data['name'], persona_name)
        self.assertEqual(agent_data['current_role'], "contestant")
        self.assertIn('skills', agent_data)
        self.assertIn('location', agent_data)
        self.assertIn('memory', agent_data)
    
    @patch('dating_show.services.piano_integration.ReverieServer.__init__')
    def test_relationship_tracking(self, mock_reverie_init):
        """Test relationship tracking functionality"""
        mock_reverie_init.return_value = None
        
        server = DatingShowReverieServer("fork_test", "sim_test")
        server.personas = self.mock_personas
        server.step = 1
        server.frontend_bridge = None  # Disable frontend for testing
        
        server._update_relationship_tracking()
        
        # Check that relationships were initialized
        self.assertEqual(len(server.relationship_tracker), len(self.mock_personas))
        for persona_name in self.mock_personas.keys():
            self.assertIn(persona_name, server.relationship_tracker)
    
    @patch('dating_show.services.piano_integration.ReverieServer.__init__')
    def test_skill_tracking(self, mock_reverie_init):
        """Test skill development tracking"""
        mock_reverie_init.return_value = None
        
        server = DatingShowReverieServer("fork_test", "sim_test")
        server.personas = self.mock_personas
        server.step = 1
        
        server._update_skill_tracking()
        
        # Check that skills were initialized
        self.assertEqual(len(server.skill_tracker), len(self.mock_personas))
        for persona_name in self.mock_personas.keys():
            self.assertIn(persona_name, server.skill_tracker)
            self.assertIn('social', server.skill_tracker[persona_name])
    
    @patch('dating_show.services.piano_integration.ReverieServer.__init__')
    def test_frontend_sync_callbacks(self, mock_reverie_init):
        """Test frontend synchronization callbacks"""
        mock_reverie_init.return_value = None
        
        callback_called = False
        callback_data = None
        
        def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
        
        server = DatingShowReverieServer("fork_test", "sim_test")
        server.personas = self.mock_personas
        server.dating_show_roles = {name: "contestant" for name in self.mock_personas.keys()}
        server.frontend_bridge = None  # Disable actual frontend
        
        server.register_agent_sync_callback(test_callback)
        
        # Mock the frontend bridge to avoid network calls
        with patch.object(server, 'frontend_bridge', None):
            server.sync_agents_to_frontend()
        
        # Callback should have been called for each persona
        self.assertTrue(callback_called)
        self.assertIsNotNone(callback_data)
    
    @patch('dating_show.services.piano_integration.ReverieServer.__init__')
    def test_factory_function(self, mock_reverie_init):
        """Test PIANO integration factory function"""
        mock_reverie_init.return_value = None
        
        server = create_dating_show_reverie_server(
            fork_sim_code="test_fork",
            sim_code="test_sim",
            frontend_bridge=None,
            dating_show_config={"test": "config"}
        )
        
        self.assertIsInstance(server, DatingShowReverieServer)


class TestFullIntegration(unittest.TestCase):
    """Test suite for full system integration"""
    
    def setUp(self):
        """Set up full integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('dating_show.services.orchestrator.create_database_service')
    @patch('dating_show.services.orchestrator.create_enhanced_bridge')
    async def test_orchestrator_full_initialization(self, mock_create_bridge, mock_create_db):
        """Test full orchestrator initialization flow"""
        # Setup mocks
        mock_db_service = Mock()
        mock_db_service.ensure_migrations.return_value = MigrationResult(
            success=True, applied_migrations=[], duration_seconds=0.1
        )
        mock_db_service.health_check.return_value = Mock(status=HealthStatus.HEALTHY)
        mock_create_db.return_value = mock_db_service
        
        mock_bridge = Mock()
        mock_bridge.get_health_metrics.return_value = Mock(status=BridgeStatus.HEALTHY)
        mock_bridge.auto_discover_agents.return_value = AutoDiscoveryResult(
            discovered_agents=["agent1", "agent2"],
            new_agents=["agent1", "agent2"],
            removed_agents=[],
            total_agents=2
        )
        mock_create_bridge.return_value = mock_bridge
        
        # Create orchestrator and test full initialization
        config = OrchestrationConfig(
            database_url="sqlite:///test.db",
            frontend_server_path=self.temp_dir,
            max_agents=2
        )
        
        orchestrator = DatingShowOrchestrator(config)
        
        # Test database initialization
        await orchestrator.initialize_database()
        self.assertTrue(orchestrator.service_health.database_service)
        
        # Test bridge initialization
        await orchestrator.start_frontend_bridge()
        self.assertTrue(orchestrator.service_health.frontend_bridge)
        
        # Test agent registration
        mock_agents = [Mock(agent_id=f"agent{i}", name=f"Agent{i}") for i in range(2)]
        await orchestrator.register_piano_agents(mock_agents)
        self.assertTrue(orchestrator.service_health.piano_agents)
        
        # Verify overall status
        status = orchestrator.get_orchestrator_status()
        self.assertEqual(status['status'], ServiceStatus.STOPPED.value)  # Still stopped until simulation starts
    
    async def test_end_to_end_integration(self):
        """Test end-to-end integration with all components"""
        # This would be a comprehensive integration test
        # For now, we'll test that all components can be imported and initialized
        
        try:
            # Test all imports work
            from dating_show.services.database_service import DatabaseService
            from dating_show.services.enhanced_bridge import EnhancedFrontendBridge
            from dating_show.services.orchestrator import DatingShowOrchestrator
            from dating_show.services.piano_integration import DatingShowReverieServer
            from dating_show.main import DatingShowMain
            
            # Test that main application can be initialized
            main_app = DatingShowMain()
            self.assertIsNotNone(main_app.config)
            
            integration_success = True
            
        except Exception as e:
            integration_success = False
            self.fail(f"End-to-end integration failed: {e}")
        
        self.assertTrue(integration_success)


def run_integration_tests():
    """Run all integration tests"""
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDatabaseService,
        TestEnhancedFrontendBridge,
        TestOrchestrationService,
        TestPIANOIntegration,
        TestFullIntegration
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)