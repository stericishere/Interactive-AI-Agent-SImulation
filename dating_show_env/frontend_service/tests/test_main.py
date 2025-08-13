"""
Test suite for the Dating Show Frontend Service
Following TDD principles as specified in CLAUDE.md
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import json

from main import app
from core.models import SimulationState, AgentState, Position, AgentRole, SimulationMode


class TestDatingShowFrontendService:
    """Test cases for the main FastAPI application"""
    
    def setup_method(self):
        """Set up test client and mock data"""
        self.client = TestClient(app)
        
        # Mock simulation data
        self.mock_agent = AgentState(
            name="Test Agent",
            role=AgentRole.CONTESTANT,
            position=Position(x=100, y=200),
            current_action="talking",
            current_location="villa"
        )
        
        self.mock_simulation = SimulationState(
            sim_code="test_sim",
            current_step=5,
            mode=SimulationMode.RUNNING,
            agents=[self.mock_agent]
        )
    
    def test_home_endpoint_returns_dashboard(self):
        """Test that the home endpoint returns the dashboard HTML"""
        with patch('core.simulation_bridge.SimulationBridge.get_current_simulation_state') as mock_get_state:
            mock_get_state.return_value = asyncio.Future()
            mock_get_state.return_value.set_result(self.mock_simulation)
            
            response = self.client.get("/")
            
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            assert "Dating Show - Live Dashboard" in response.text
    
    def test_simulation_state_endpoint(self):
        """Test the simulation state API endpoint"""
        with patch('core.simulation_bridge.SimulationBridge.get_current_simulation_state') as mock_get_state:
            mock_get_state.return_value = asyncio.Future()
            mock_get_state.return_value.set_result(self.mock_simulation)
            
            response = self.client.get("/api/simulation/state")
            
            assert response.status_code == 200
            data = response.json()
            assert data["sim_code"] == "test_sim"
            assert data["current_step"] == 5
            assert data["mode"] == "running"
    
    def test_agents_endpoint(self):
        """Test the agents API endpoint"""
        with patch('core.simulation_bridge.SimulationBridge.get_all_agents') as mock_get_agents:
            mock_get_agents.return_value = asyncio.Future()
            mock_get_agents.return_value.set_result([self.mock_agent])
            
            response = self.client.get("/api/agents")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["name"] == "Test Agent"
            assert data[0]["role"] == "contestant"
    
    def test_specific_agent_endpoint(self):
        """Test the specific agent API endpoint"""
        with patch('core.simulation_bridge.SimulationBridge.get_agent_state') as mock_get_agent:
            mock_get_agent.return_value = asyncio.Future()
            mock_get_agent.return_value.set_result(self.mock_agent)
            
            response = self.client.get("/api/agents/Test Agent")
            
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Agent"
            assert data["current_location"] == "villa"
    
    def test_agent_not_found(self):
        """Test that 404 is returned for non-existent agent"""
        with patch('core.simulation_bridge.SimulationBridge.get_agent_state') as mock_get_agent:
            mock_get_agent.return_value = asyncio.Future()
            mock_get_agent.return_value.set_result(None)
            
            response = self.client.get("/api/agents/Nonexistent Agent")
            
            assert response.status_code == 404
    
    def test_advance_simulation_endpoint(self):
        """Test the simulation advancement endpoint"""
        mock_result = {
            "success": True,
            "step": 6,
            "message": "Advanced to step 6"
        }
        
        with patch('core.simulation_bridge.SimulationBridge.advance_simulation') as mock_advance:
            mock_advance.return_value = asyncio.Future()
            mock_advance.return_value.set_result(mock_result)
            
            response = self.client.post("/api/simulation/step")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["step"] == 6


class TestSimulationBridge:
    """Test cases for the simulation bridge"""
    
    @pytest.fixture
    def bridge(self):
        """Create a simulation bridge instance for testing"""
        from core.simulation_bridge import SimulationBridge
        return SimulationBridge()
    
    def test_bridge_initialization(self, bridge):
        """Test that the bridge initializes correctly"""
        assert bridge is not None
        assert hasattr(bridge, 'settings')
        assert hasattr(bridge, 'simulation_data_path')
    
    @pytest.mark.asyncio
    async def test_get_current_sim_code(self, bridge):
        """Test reading current simulation code"""
        # Mock file reading
        mock_data = {"sim_code": "test_simulation"}
        
        with patch('aiofiles.open') as mock_open:
            mock_file = AsyncMock()
            mock_file.read.return_value = json.dumps(mock_data)
            mock_open.return_value.__aenter__.return_value = mock_file
            
            with patch.object(bridge.temp_storage_path, 'exists', return_value=True):
                sim_code = await bridge._get_current_sim_code()
                
            assert sim_code == "test_simulation"
    
    @pytest.mark.asyncio
    async def test_get_current_step(self, bridge):
        """Test reading current simulation step"""
        mock_data = {"step": 10}
        
        with patch('aiofiles.open') as mock_open:
            mock_file = AsyncMock()
            mock_file.read.return_value = json.dumps(mock_data)
            mock_open.return_value.__aenter__.return_value = mock_file
            
            with patch.object(bridge.temp_storage_path, 'exists', return_value=True):
                step = await bridge._get_current_step()
                
            assert step == 10


class TestWebSocketManager:
    """Test cases for WebSocket management"""
    
    @pytest.fixture
    def ws_manager(self):
        """Create a WebSocket manager instance for testing"""
        from core.websocket_manager import WebSocketManager
        return WebSocketManager()
    
    def test_websocket_manager_initialization(self, ws_manager):
        """Test that WebSocket manager initializes correctly"""
        assert ws_manager is not None
        assert len(ws_manager.active_connections) == 0
        assert len(ws_manager.connection_metadata) == 0
    
    def test_connection_count(self, ws_manager):
        """Test connection counting"""
        assert ws_manager.get_connection_count() == 0
    
    def test_subscriber_count(self, ws_manager):
        """Test subscriber counting for topics"""
        assert ws_manager.get_subscriber_count("test_topic") == 0


class TestModels:
    """Test cases for Pydantic models"""
    
    def test_agent_state_creation(self):
        """Test creating an AgentState instance"""
        position = Position(x=100, y=200, sector="villa")
        agent = AgentState(
            name="Test Agent",
            role=AgentRole.CONTESTANT,
            position=position,
            current_action="talking"
        )
        
        assert agent.name == "Test Agent"
        assert agent.role == AgentRole.CONTESTANT
        assert agent.position.x == 100
        assert agent.position.y == 200
        assert agent.current_action == "talking"
    
    def test_simulation_state_creation(self):
        """Test creating a SimulationState instance"""
        simulation = SimulationState(
            sim_code="test_sim",
            current_step=5,
            mode=SimulationMode.RUNNING
        )
        
        assert simulation.sim_code == "test_sim"
        assert simulation.current_step == 5
        assert simulation.mode == SimulationMode.RUNNING
        assert len(simulation.agents) == 0
    
    def test_position_validation(self):
        """Test Position model validation"""
        position = Position(x=10.5, y=20.3)
        assert position.x == 10.5
        assert position.y == 20.3
        assert position.sector is None
        
        position_with_sector = Position(x=0, y=0, sector="kitchen")
        assert position_with_sector.sector == "kitchen"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])