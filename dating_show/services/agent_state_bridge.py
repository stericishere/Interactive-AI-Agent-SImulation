"""
Agent State Bridge Service
Handles conversion between reverie persona data and dating show frontend formats
Ensures compatibility and seamless data flow between different agent representations
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StandardAgentState:
    """Standardized agent state format for cross-system compatibility"""
    agent_id: str
    name: str
    role: str = "contestant"
    position: Dict[str, float] = None
    current_action: str = "idle"
    current_location: str = "villa"
    emotional_state: Dict[str, float] = None
    relationship_scores: Dict[str, float] = None
    dialogue_history: List[str] = None
    memory: Dict[str, Any] = None
    specialization: Dict[str, Any] = None
    skills: Dict[str, Any] = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.position is None:
            self.position = {"x": 50.0, "y": 50.0}
        if self.emotional_state is None:
            self.emotional_state = {}
        if self.relationship_scores is None:
            self.relationship_scores = {}
        if self.dialogue_history is None:
            self.dialogue_history = []
        if self.memory is None:
            self.memory = {}
        if self.specialization is None:
            self.specialization = {}
        if self.skills is None:
            self.skills = {}
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc)


class AgentStateBridge:
    """
    Handles conversion between different agent state formats:
    - Reverie Persona format
    - Dating Show Frontend format  
    - Enhanced Agent Manager format
    - PIANO Agent format
    """
    
    def __init__(self):
        self.conversion_cache: Dict[str, StandardAgentState] = {}
        self.format_converters = {
            'reverie': self._convert_from_reverie,
            'frontend': self._convert_from_frontend,
            'enhanced': self._convert_from_enhanced,
            'piano': self._convert_from_piano
        }
        
    def convert_agent_state(self, agent_data: Any, source_format: str, 
                          target_format: str) -> Dict[str, Any]:
        """
        Convert agent state between different formats
        
        Args:
            agent_data: Source agent data in any supported format
            source_format: Format of source data ('reverie', 'frontend', 'enhanced', 'piano')
            target_format: Desired output format ('standard', 'frontend', 'reverie')
            
        Returns:
            Dict containing converted agent state
        """
        try:
            # First convert to standard format
            if source_format not in self.format_converters:
                raise ValueError(f"Unsupported source format: {source_format}")
                
            converter = self.format_converters[source_format]
            standard_state = converter(agent_data)
            
            # Cache the standard state
            self.conversion_cache[standard_state.agent_id] = standard_state
            
            # Then convert to target format
            if target_format == 'standard':
                return asdict(standard_state)
            elif target_format == 'frontend':
                return self._convert_to_frontend(standard_state)
            elif target_format == 'reverie':
                return self._convert_to_reverie(standard_state)
            elif target_format == 'enhanced':
                return self._convert_to_enhanced(standard_state)
            else:
                raise ValueError(f"Unsupported target format: {target_format}")
                
        except Exception as e:
            logger.error(f"Error converting agent state from {source_format} to {target_format}: {e}")
            # Return safe fallback
            return self._create_fallback_state(agent_data, target_format)
    
    def _convert_from_reverie(self, persona) -> StandardAgentState:
        """Convert from Reverie Persona object to standard format"""
        try:
            # Extract basic info
            agent_id = getattr(persona, 'agent_id', persona.name.replace(' ', '_').lower())
            name = persona.name
            
            # Extract position from scratch or last_position
            position = {"x": 50.0, "y": 50.0}
            if hasattr(persona, 'last_position') and persona.last_position:
                if isinstance(persona.last_position, (tuple, list)) and len(persona.last_position) >= 2:
                    position = {"x": float(persona.last_position[0]), "y": float(persona.last_position[1])}
            
            # Extract current action and location
            current_action = "socializing"
            current_location = "villa"
            if hasattr(persona, 'scratch'):
                current_action = getattr(persona.scratch, 'daily_plan_req', 'socializing')
                if hasattr(persona.scratch, 'curr_tile') and persona.scratch.curr_tile:
                    current_location = persona.scratch.curr_tile[-1] if persona.scratch.curr_tile else "villa"
            
            # Create memory structure
            memory = {
                "working_memory": [],
                "long_term": {},
                "recent_events": []
            }
            
            return StandardAgentState(
                agent_id=agent_id,
                name=name,
                role="contestant",
                position=position,
                current_action=current_action,
                current_location=current_location,
                memory=memory,
                specialization={"type": "social", "level": "intermediate"},
                last_updated=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Error converting reverie persona: {e}")
            return self._create_default_state(getattr(persona, 'name', 'Unknown'))
    
    def _convert_from_frontend(self, frontend_data: Dict[str, Any]) -> StandardAgentState:
        """Convert from frontend format to standard format"""
        try:
            return StandardAgentState(
                agent_id=frontend_data.get('agent_id', frontend_data.get('name', 'unknown').replace(' ', '_').lower()),
                name=frontend_data.get('name', 'Unknown'),
                role=frontend_data.get('role', 'contestant'),
                position=frontend_data.get('position', {"x": 50.0, "y": 50.0}),
                current_action=frontend_data.get('current_action', 'idle'),
                current_location=frontend_data.get('current_location', 'villa'),
                emotional_state=frontend_data.get('emotional_state', {}),
                relationship_scores=frontend_data.get('relationship_scores', {}),
                dialogue_history=frontend_data.get('dialogue_history', []),
                memory=frontend_data.get('memory', {}),
                specialization=frontend_data.get('specialization', {}),
                skills=frontend_data.get('skills', {}),
                last_updated=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Error converting frontend data: {e}")
            return self._create_default_state(frontend_data.get('name', 'Unknown'))
    
    def _convert_from_enhanced(self, enhanced_data: Dict[str, Any]) -> StandardAgentState:
        """Convert from enhanced agent manager format to standard format"""
        try:
            # Enhanced format has nested state structure
            state = enhanced_data.get('state', enhanced_data)
            
            # Extract position
            position = {"x": 50.0, "y": 50.0}
            location_data = state.get('location', {})
            if isinstance(location_data, dict):
                position = {"x": location_data.get('x', 50.0), "y": location_data.get('y', 50.0)}
            
            return StandardAgentState(
                agent_id=enhanced_data.get('agent_id', state.get('name', 'unknown').replace(' ', '_').lower()),
                name=state.get('name', 'Unknown'),
                role=state.get('current_role', 'contestant'),
                position=position,
                current_action=state.get('current_action', 'idle'),
                current_location=location_data.get('area', 'villa') if isinstance(location_data, dict) else 'villa',
                emotional_state=state.get('emotional_state', {}),
                relationship_scores=state.get('relationship_scores', {}),
                memory=state.get('memory', {}),
                specialization=state.get('specialization', {}),
                skills=state.get('skills', {}),
                last_updated=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Error converting enhanced data: {e}")
            return self._create_default_state(enhanced_data.get('name', 'Unknown'))
    
    def _convert_from_piano(self, piano_data: Dict[str, Any]) -> StandardAgentState:
        """Convert from PIANO agent format to standard format"""
        try:
            return StandardAgentState(
                agent_id=piano_data.get('agent_id', piano_data.get('name', 'unknown').replace(' ', '_').lower()),
                name=piano_data.get('name', 'Unknown'),
                role=piano_data.get('current_role', 'contestant'),
                position=piano_data.get('location', {"x": 50.0, "y": 50.0}),
                current_action=piano_data.get('current_action', 'idle'),
                current_location=piano_data.get('current_location', 'villa'),
                memory=piano_data.get('memory', {}),
                specialization=piano_data.get('specialization', {}),
                skills=piano_data.get('skills', {}),
                last_updated=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Error converting PIANO data: {e}")
            return self._create_default_state(piano_data.get('name', 'Unknown'))
    
    def _convert_to_frontend(self, state: StandardAgentState) -> Dict[str, Any]:
        """Convert from standard format to frontend format"""
        return {
            "agent_id": state.agent_id,
            "name": state.name,
            "role": state.role,
            "position": state.position,
            "current_action": state.current_action,
            "current_location": state.current_location,
            "emotional_state": state.emotional_state,
            "relationship_scores": state.relationship_scores,
            "dialogue_history": state.dialogue_history[-10:],  # Last 10 messages
            "memory": state.memory,
            "specialization": state.specialization,
            "skills": state.skills,
            "last_updated": state.last_updated.isoformat()
        }
    
    def _convert_to_reverie(self, state: StandardAgentState) -> Dict[str, Any]:
        """Convert from standard format to reverie-compatible format"""
        return {
            "name": state.name,
            "last_position": (state.position["x"], state.position["y"]),
            "scratch": {
                "daily_plan_req": state.current_action,
                "curr_tile": ["the_ville", state.current_location],
                "chat": state.dialogue_history[-1] if state.dialogue_history else "",
                "curr_time": state.last_updated.isoformat()
            },
            "memory": state.memory
        }
    
    def _convert_to_enhanced(self, state: StandardAgentState) -> Dict[str, Any]:
        """Convert from standard format to enhanced manager format"""
        return {
            "agent_id": state.agent_id,
            "state": {
                "name": state.name,
                "current_role": state.role,
                "location": {
                    "area": state.current_location,
                    "x": state.position["x"],
                    "y": state.position["y"]
                },
                "current_action": state.current_action,
                "emotional_state": state.emotional_state,
                "relationship_scores": state.relationship_scores,
                "memory": state.memory,
                "specialization": state.specialization,
                "skills": state.skills
            },
            "last_updated": state.last_updated.isoformat()
        }
    
    def _create_default_state(self, name: str) -> StandardAgentState:
        """Create a safe default state for error cases"""
        agent_id = name.replace(' ', '_').lower()
        return StandardAgentState(
            agent_id=agent_id,
            name=name,
            role="contestant",
            position={"x": 50.0, "y": 50.0},
            current_action="socializing",
            current_location="villa",
            memory={"working_memory": [], "long_term": {}},
            specialization={"type": "social", "level": "beginner"}
        )
    
    def _create_fallback_state(self, agent_data: Any, target_format: str) -> Dict[str, Any]:
        """Create fallback state when conversion fails"""
        name = "Unknown"
        if hasattr(agent_data, 'name'):
            name = agent_data.name
        elif isinstance(agent_data, dict):
            name = agent_data.get('name', 'Unknown')
        
        default_state = self._create_default_state(name)
        
        if target_format == 'frontend':
            return self._convert_to_frontend(default_state)
        elif target_format == 'reverie':
            return self._convert_to_reverie(default_state)
        elif target_format == 'enhanced':
            return self._convert_to_enhanced(default_state)
        else:
            return asdict(default_state)
    
    def batch_convert_agents(self, agents_data: List[Any], source_format: str, 
                           target_format: str) -> List[Dict[str, Any]]:
        """Convert multiple agents in batch for performance"""
        converted_agents = []
        
        for agent_data in agents_data:
            try:
                converted = self.convert_agent_state(agent_data, source_format, target_format)
                converted_agents.append(converted)
            except Exception as e:
                logger.error(f"Error in batch conversion: {e}")
                # Add fallback state
                fallback = self._create_fallback_state(agent_data, target_format)
                converted_agents.append(fallback)
        
        return converted_agents
    
    def get_cached_state(self, agent_id: str) -> Optional[StandardAgentState]:
        """Get cached standard state for an agent"""
        return self.conversion_cache.get(agent_id)
    
    def clear_cache(self):
        """Clear the conversion cache"""
        self.conversion_cache.clear()
        logger.info("Agent state conversion cache cleared")


# Global bridge instance
_state_bridge: Optional[AgentStateBridge] = None


def get_state_bridge() -> AgentStateBridge:
    """Get or create global state bridge instance"""
    global _state_bridge
    if _state_bridge is None:
        _state_bridge = AgentStateBridge()
    return _state_bridge


def convert_reverie_to_frontend(persona) -> Dict[str, Any]:
    """Quick helper to convert reverie persona to frontend format"""
    bridge = get_state_bridge()
    return bridge.convert_agent_state(persona, 'reverie', 'frontend')


def convert_frontend_to_reverie(frontend_data: Dict[str, Any]) -> Dict[str, Any]:
    """Quick helper to convert frontend data to reverie format"""
    bridge = get_state_bridge()
    return bridge.convert_agent_state(frontend_data, 'frontend', 'reverie')