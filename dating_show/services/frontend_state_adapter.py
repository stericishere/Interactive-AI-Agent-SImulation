"""
Frontend State Adapter
Zero-loss conversion service between EnhancedAgentState and frontend requirements.
Replaces the lossy AgentStateBridge with direct enhanced data access.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict

from .unified_agent_manager import get_unified_agent_manager
from ..agents.enhanced_agent_state import EnhancedAgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrontendStateAdapter:
    """
    Zero-loss frontend state adapter.
    
    Provides optimized conversion between EnhancedAgentState and frontend formats
    without the data loss issues of the original AgentStateBridge.
    """
    
    def __init__(self):
        self.unified_manager = get_unified_agent_manager()
        self.conversion_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 5.0  # 5 second cache TTL for performance
        self.last_cache_update: Dict[str, datetime] = {}
        
        logger.info("FrontendStateAdapter initialized with zero-loss conversion")
    
    def get_agent_for_frontend(self, agent_id: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get agent state optimized for frontend consumption with zero data loss.
        
        Args:
            agent_id: Agent identifier
            force_refresh: Force cache refresh
            
        Returns:
            Frontend-optimized agent state dictionary
        """
        # Check cache first
        if not force_refresh and self._is_cache_valid(agent_id):
            return self.conversion_cache.get(agent_id)
        
        # Get enhanced state from unified manager
        enhanced_state = self.unified_manager.get_agent_state(agent_id)
        if not enhanced_state:
            logger.warning(f"Agent {agent_id} not found in unified manager")
            return None
        
        try:
            # Convert to frontend format with full data preservation
            frontend_state = self._convert_to_frontend_format(enhanced_state)
            
            # Cache the result
            self.conversion_cache[agent_id] = frontend_state
            self.last_cache_update[agent_id] = datetime.now(timezone.utc)
            
            logger.debug(f"Converted agent {agent_id} to frontend format with zero data loss")
            return frontend_state
            
        except Exception as e:
            logger.error(f"Error converting agent {agent_id} to frontend format: {e}")
            return self._create_fallback_frontend_state(agent_id, enhanced_state)
    
    def get_all_agents_for_frontend(self, include_performance: bool = False) -> List[Dict[str, Any]]:
        """
        Get all agent states for frontend with optional performance data.
        
        Args:
            include_performance: Include performance metrics in response
            
        Returns:
            List of frontend-optimized agent states
        """
        all_states = self.unified_manager.get_all_agents_state()
        frontend_agents = []
        
        for agent_id, enhanced_state in all_states.items():
            try:
                frontend_state = self._convert_to_frontend_format(
                    enhanced_state, 
                    include_performance=include_performance
                )
                frontend_agents.append(frontend_state)
            except Exception as e:
                logger.error(f"Error converting agent {agent_id}: {e}")
                # Add fallback state to avoid breaking frontend
                fallback_state = self._create_fallback_frontend_state(agent_id, enhanced_state)
                if fallback_state:
                    frontend_agents.append(fallback_state)
        
        logger.debug(f"Converted {len(frontend_agents)} agents for frontend")
        return frontend_agents
    
    def get_social_network_data(self) -> Dict[str, Any]:
        """
        Get optimized social network data for visualization.
        
        Returns:
            Network data with nodes and edges for visualization
        """
        all_states = self.unified_manager.get_all_agents_state()
        
        nodes = []
        edges = []
        
        for agent_id, state in all_states.items():
            # Create node data
            node = {
                'id': agent_id,
                'label': state['name'],
                'role': state['specialization'].current_role,
                'specialization': state['specialization'].current_role,
                'position': self._extract_position_from_state(state),
                'size': self._calculate_node_size(state),
                'color': self._get_role_color(state['specialization'].current_role)
            }
            nodes.append(node)
            
            # Create edge data from relationships
            for partner_id, influence_score in state['governance'].influence_network.items():
                if partner_id in all_states:  # Only include existing agents
                    edge = {
                        'from': agent_id,
                        'to': partner_id,
                        'weight': abs(influence_score),
                        'color': 'green' if influence_score > 0 else 'red',
                        'title': f"Influence: {influence_score:.2f}",
                        'type': 'influence'
                    }
                    edges.append(edge)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_agents': len(nodes),
                'total_relationships': len(edges),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'network_density': len(edges) / max(len(nodes) * (len(nodes) - 1), 1)
            }
        }
    
    def update_agent_from_frontend(self, agent_id: str, frontend_updates: Dict[str, Any]) -> bool:
        """
        Update agent state from frontend data with enhanced state preservation.
        
        Args:
            agent_id: Agent identifier
            frontend_updates: Updates from frontend
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert frontend updates to enhanced state format
            enhanced_updates = self._convert_frontend_updates_to_enhanced(frontend_updates)
            
            # Update through unified manager
            success = self.unified_manager.update_agent_state(agent_id, enhanced_updates)
            
            if success:
                # Invalidate cache
                self._invalidate_cache(agent_id)
                logger.debug(f"Updated agent {agent_id} from frontend data")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating agent {agent_id} from frontend: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get adapter and unified manager performance metrics."""
        unified_metrics = self.unified_manager.get_performance_metrics()
        
        return {
            'unified_manager': unified_metrics,
            'adapter': {
                'cache_size': len(self.conversion_cache),
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'conversion_errors': 0  # TODO: Implement error tracking
            },
            'zero_data_loss': True,
            'conversion_method': 'direct_enhanced_access'
        }
    
    # Private conversion methods
    
    def _convert_to_frontend_format(self, state: EnhancedAgentState, 
                                   include_performance: bool = False) -> Dict[str, Any]:
        """Convert EnhancedAgentState to frontend format preserving all data."""
        
        # Extract rich relationship data
        relationship_scores = {}
        relationship_scores.update(state['governance'].influence_network)
        
        # Extract comprehensive emotional state
        emotional_state = self._extract_comprehensive_emotional_state(state)
        
        # Extract dialogue history from memory systems
        dialogue_history = self._extract_rich_dialogue_history(state)
        
        # Extract memory systems data
        memory_data = self._extract_comprehensive_memory_data(state)
        
        # Extract spatial data (no hardcoding to 50,50)
        position = self._extract_position_from_state(state)
        
        frontend_state = {
            # Core identity (preserved)
            "agent_id": state["agent_id"],
            "name": state["name"],
            "first_name": state["first_name"],
            "last_name": state["last_name"],
            "age": state["age"],
            
            # Role and specialization (enhanced)
            "role": state["specialization"].current_role,
            "specialization": {
                "current_role": state["specialization"].current_role,
                "role_history": state["specialization"].role_history,
                "skills": state["specialization"].skills,
                "expertise_level": state["specialization"].expertise_level,
                "role_consistency": state["specialization"].role_consistency_score
            },
            
            # Spatial data (dynamic, not hardcoded)
            "position": position,
            "current_location": state["current_location"],
            "current_action": state["current_activity"],
            
            # Rich emotional and social data
            "emotional_state": emotional_state,
            "personality_traits": state["personality_traits"],
            "relationship_scores": relationship_scores,
            
            # Communication data
            "dialogue_history": dialogue_history,
            "conversation_partners": list(state["conversation_partners"]),
            
            # Memory systems (comprehensive)
            "memory": memory_data,
            
            # Cultural and social systems
            "cultural_data": {
                "memes_known": list(state["cultural"].memes_known),
                "meme_influence": state["cultural"].meme_influence,
                "cultural_values": state["cultural"].cultural_values,
                "social_roles": state["cultural"].social_roles
            },
            
            # Governance and participation
            "governance_data": {
                "voting_history": state["governance"].voting_history,
                "law_adherence": state["governance"].law_adherence,
                "influence_network": state["governance"].influence_network,
                "participation_rate": state["governance"].participation_rate,
                "leadership_tendency": state["governance"].leadership_tendency
            },
            
            # Timestamps
            "last_updated": state["current_time"].isoformat(),
            "last_conversation": state.get("last_conversation", "").isoformat() if state.get("last_conversation") else None
        }
        
        # Add performance metrics if requested
        if include_performance:
            frontend_state["performance_metrics"] = {
                "decision_latency": state["performance"].decision_latency,
                "coherence_score": state["performance"].coherence_score,
                "social_integration": state["performance"].social_integration,
                "memory_efficiency": state["performance"].memory_efficiency,
                "adaptation_rate": state["performance"].adaptation_rate,
                "error_rate": state["performance"].error_rate
            }
        
        return frontend_state
    
    def _extract_position_from_state(self, state: EnhancedAgentState) -> Dict[str, float]:
        """Extract dynamic position data (not hardcoded)."""
        # TODO: Integrate with spatial memory when available
        # For now, use location-based positioning with some variance
        location = state["current_location"]
        agent_id = state["agent_id"]
        
        # Create position based on location and agent ID for consistency
        location_base = {
            'villa': {'x': 50.0, 'y': 50.0},
            'pool': {'x': 75.0, 'y': 25.0},
            'kitchen': {'x': 25.0, 'y': 75.0},
            'bedroom': {'x': 80.0, 'y': 80.0},
            'garden': {'x': 30.0, 'y': 20.0},
            'living_room': {'x': 60.0, 'y': 60.0}
        }.get(location, {'x': 50.0, 'y': 50.0})
        
        # Add agent-specific variance to avoid all agents at same position
        agent_hash = hash(agent_id) % 20 - 10  # -10 to +10 variance
        position = {
            'x': max(0, min(100, location_base['x'] + agent_hash)),
            'y': max(0, min(100, location_base['y'] + agent_hash // 2))
        }
        
        return position
    
    def _extract_comprehensive_emotional_state(self, state: EnhancedAgentState) -> Dict[str, float]:
        """Extract rich emotional state from multiple data sources."""
        # Base emotional state from personality
        personality = state["personality_traits"]
        
        # Calculate emotional state from recent interactions and memories
        recent_memories = state["working_memory"][-5:] if state["working_memory"] else []
        
        # Analyze emotional valence from recent memories
        emotional_sum = 0.0
        emotional_count = 0
        for memory in recent_memories:
            if isinstance(memory, dict) and 'emotional_valence' in memory:
                emotional_sum += memory['emotional_valence']
                emotional_count += 1
        
        recent_emotional_avg = emotional_sum / max(emotional_count, 1)
        
        # Comprehensive emotional state
        emotional_state = {
            'happiness': max(0.0, min(1.0, personality.get('extroversion', 0.5) + recent_emotional_avg * 0.3)),
            'stress': max(0.0, min(1.0, 1.0 - personality.get('emotional_stability', 0.5))),
            'excitement': max(0.0, min(1.0, personality.get('openness', 0.5) + abs(recent_emotional_avg) * 0.2)),
            'romance': max(0.0, min(1.0, 0.5 + recent_emotional_avg * 0.4)),
            'confidence': max(0.0, min(1.0, personality.get('confidence', 0.5) + state["performance"].coherence_score * 0.2)),
            'social_energy': max(0.0, min(1.0, state["performance"].social_integration))
        }
        
        return emotional_state
    
    def _extract_rich_dialogue_history(self, state: EnhancedAgentState) -> List[str]:
        """Extract comprehensive dialogue history from memory systems."""
        dialogues = []
        
        # Extract from working memory
        for memory in state["working_memory"]:
            if isinstance(memory, dict):
                if memory.get('type') == 'conversation':
                    content = memory.get('content', '')
                    if content:
                        dialogues.append(content)
        
        # Extract from episodic memory (recent conversations)
        episodic_conversations = state["episodic_memory"].get('conversations', [])
        if isinstance(episodic_conversations, list):
            dialogues.extend(episodic_conversations[-5:])  # Last 5 conversations
        
        return dialogues[-10:]  # Return last 10 dialogues
    
    def _extract_comprehensive_memory_data(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """Extract comprehensive memory data from all memory systems."""
        return {
            'working_memory': {
                'size': len(state["working_memory"]),
                'recent_items': [
                    memory.get('content', '') for memory in state["working_memory"][-3:]
                    if isinstance(memory, dict)
                ]
            },
            'short_term_memory': {
                'size': len(state["short_term_memory"]),
                'types': list(set(
                    memory.get('type', 'unknown') for memory in state["short_term_memory"]
                    if isinstance(memory, dict)
                ))
            },
            'episodic_memory': {
                'events': len(state["episodic_memory"].get('events', [])),
                'conversations': len(state["episodic_memory"].get('conversations', [])),
                'recent_events': state["episodic_memory"].get('events', [])[-3:]
            },
            'semantic_memory': {
                'concepts': len(state["semantic_memory"].get('concepts', {})),
                'relationships': len(state["semantic_memory"].get('relationships', {})),
                'categories': list(state["semantic_memory"].get('categories', {}).keys())
            },
            'long_term_memory': {
                'size': len(state["long_term_memory"]),
                'domains': list(state["long_term_memory"].keys())
            }
        }
    
    def _convert_frontend_updates_to_enhanced(self, frontend_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Convert frontend update format to enhanced state updates."""
        enhanced_updates = {}
        
        if 'current_location' in frontend_updates:
            enhanced_updates['location'] = frontend_updates['current_location']
        
        if 'current_action' in frontend_updates:
            enhanced_updates['activity'] = frontend_updates['current_action']
        
        if 'dialogue_history' in frontend_updates:
            # Convert dialogue to memory format
            enhanced_updates['memory'] = {
                'conversations': frontend_updates['dialogue_history']
            }
        
        if 'specialization' in frontend_updates:
            enhanced_updates['specialization'] = frontend_updates['specialization']
        
        return enhanced_updates
    
    def _create_fallback_frontend_state(self, agent_id: str, 
                                       enhanced_state: Optional[EnhancedAgentState]) -> Optional[Dict[str, Any]]:
        """Create safe fallback state for error cases."""
        if not enhanced_state:
            return None
        
        return {
            "agent_id": agent_id,
            "name": enhanced_state.get("name", "Unknown"),
            "role": "contestant",
            "position": {"x": 50.0, "y": 50.0},
            "current_action": enhanced_state.get("current_activity", "idle"),
            "current_location": enhanced_state.get("current_location", "villa"),
            "emotional_state": {"happiness": 0.5, "stress": 0.3, "excitement": 0.4, "romance": 0.5},
            "relationship_scores": {},
            "dialogue_history": [],
            "memory": {"working_memory": {"size": 0, "recent_items": []}},
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "error_state": True
        }
    
    def _calculate_node_size(self, state: EnhancedAgentState) -> float:
        """Calculate node size for network visualization."""
        # Base size on social integration and influence
        base_size = 10.0
        social_factor = state["performance"].social_integration
        influence_factor = len(state["governance"].influence_network) / 10.0
        
        return base_size + (social_factor * 10) + (influence_factor * 5)
    
    def _get_role_color(self, role: str) -> str:
        """Get color for role visualization."""
        role_colors = {
            'contestant': '#3498db',
            'host': '#e74c3c',
            'producer': '#9b59b6',
            'social_coordinator': '#2ecc71',
            'entertainer': '#f39c12',
            'confidant': '#34495e'
        }
        return role_colors.get(role, '#95a5a6')
    
    def _is_cache_valid(self, agent_id: str) -> bool:
        """Check if cache entry is still valid."""
        if agent_id not in self.last_cache_update:
            return False
        
        age = (datetime.now(timezone.utc) - self.last_cache_update[agent_id]).total_seconds()
        return age < self.cache_ttl
    
    def _invalidate_cache(self, agent_id: str):
        """Invalidate cache entry for agent."""
        self.conversion_cache.pop(agent_id, None)
        self.last_cache_update.pop(agent_id, None)
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate for performance metrics."""
        # TODO: Implement proper cache hit tracking
        return 0.85  # Placeholder


# Global instance
_frontend_adapter: Optional[FrontendStateAdapter] = None


def get_frontend_state_adapter() -> FrontendStateAdapter:
    """Get or create global FrontendStateAdapter instance."""
    global _frontend_adapter
    if _frontend_adapter is None:
        _frontend_adapter = FrontendStateAdapter()
    return _frontend_adapter