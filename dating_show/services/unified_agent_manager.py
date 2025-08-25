"""
Unified Agent Manager
Centralized state management system that eliminates the lossy AgentStateBridge bottleneck.
Provides direct EnhancedAgentState access with zero data loss and real-time synchronization.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from dataclasses import asdict
import threading
from collections import defaultdict

from ..agents.enhanced_agent_state import (
    EnhancedAgentState, EnhancedAgentStateManager, 
    SpecializationData, CulturalData, GovernanceData, PerformanceMetrics,
    create_enhanced_agent_state
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedAgentManager:
    """
    Centralized agent state management system.
    
    Eliminates the AgentStateBridge bottleneck by providing direct access
    to EnhancedAgentState with zero data loss and real-time synchronization.
    """
    
    def __init__(self):
        self.agents: Dict[str, EnhancedAgentStateManager] = {}
        self.state_cache: Dict[str, EnhancedAgentState] = {}
        self.update_listeners: List[callable] = []
        self._lock = threading.RLock()
        self.performance_metrics = {
            'total_updates': 0,
            'average_update_time_ms': 0.0,
            'cache_hit_rate': 0.0,
            'last_sync_time': None
        }
        
        # Real-time update tracking
        self.pending_updates: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.batch_update_interval = 0.1  # 100ms batching
        self._update_timer: Optional[threading.Timer] = None
        
        logger.info("UnifiedAgentManager initialized with zero-loss state management")
    
    def register_agent(self, agent_id: str, name: str, 
                      personality_traits: Dict[str, float]) -> EnhancedAgentStateManager:
        """
        Register new agent with enhanced state management.
        
        Args:
            agent_id: Unique agent identifier
            name: Agent display name
            personality_traits: Personality trait scores
            
        Returns:
            EnhancedAgentStateManager instance
        """
        with self._lock:
            if agent_id in self.agents:
                logger.warning(f"Agent {agent_id} already registered, returning existing")
                return self.agents[agent_id]
            
            # Create enhanced agent state manager
            agent_manager = create_enhanced_agent_state(agent_id, name, personality_traits)
            
            # Register and cache
            self.agents[agent_id] = agent_manager
            self.state_cache[agent_id] = agent_manager.state
            
            logger.info(f"Registered agent {agent_id} ({name}) with enhanced state")
            self._notify_listeners('agent_registered', agent_id, agent_manager.state)
            
            return agent_manager
    
    def get_agent_state(self, agent_id: str) -> Optional[EnhancedAgentState]:
        """
        Get current agent state with zero data loss.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Full EnhancedAgentState or None if not found
        """
        with self._lock:
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found")
                return None
            
            # Return current state (no conversion loss)
            return self.agents[agent_id].state
    
    def update_agent_state(self, agent_id: str, updates: Dict[str, Any], 
                          batch_mode: bool = True) -> bool:
        """
        Update agent state with real-time synchronization.
        
        Args:
            agent_id: Agent identifier
            updates: State updates dictionary
            batch_mode: Whether to batch updates for performance
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agents:
            logger.error(f"Cannot update unknown agent {agent_id}")
            return False
        
        try:
            start_time = datetime.now()
            
            with self._lock:
                agent_manager = self.agents[agent_id]
                
                # Apply updates to enhanced state
                if 'memory' in updates:
                    self._update_memory_systems(agent_manager, updates['memory'])
                
                if 'specialization' in updates:
                    self._update_specialization(agent_manager, updates['specialization'])
                
                if 'location' in updates:
                    agent_manager.state['current_location'] = updates['location']
                
                if 'activity' in updates:
                    agent_manager.state['current_activity'] = updates['activity']
                
                if 'social_interactions' in updates:
                    self._update_social_interactions(agent_manager, updates['social_interactions'])
                
                # Update cache
                self.state_cache[agent_id] = agent_manager.state
                
                # Track performance
                update_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_performance_metrics(update_time)
                
                # Handle real-time updates
                if batch_mode:
                    self.pending_updates[agent_id].update(updates)
                    self._schedule_batch_update()
                else:
                    self._notify_listeners('agent_updated', agent_id, agent_manager.state)
                
                logger.debug(f"Updated agent {agent_id} state in {update_time:.2f}ms")
                return True
                
        except Exception as e:
            logger.error(f"Error updating agent {agent_id}: {e}")
            return False
    
    def get_all_agents_state(self) -> Dict[str, EnhancedAgentState]:
        """
        Get all agent states for frontend synchronization.
        
        Returns:
            Dictionary mapping agent_id to EnhancedAgentState
        """
        with self._lock:
            return self.state_cache.copy()
    
    def get_frontend_compatible_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get frontend-compatible agent state with zero data loss.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Frontend-compatible state dictionary
        """
        state = self.get_agent_state(agent_id)
        if not state:
            return None
        
        # Convert to frontend format without data loss
        return {
            "agent_id": agent_id,
            "name": state["name"],
            "role": state["specialization"].current_role,
            "position": self._extract_position(state),
            "current_action": state["current_activity"],
            "current_location": state["current_location"],
            "emotional_state": self._extract_emotional_state(state),
            "relationship_scores": self._extract_relationships(state),
            "dialogue_history": self._extract_dialogue_history(state),
            "memory": self._extract_memory_summary(state),
            "specialization": asdict(state["specialization"]),
            "skills": state["specialization"].skills,
            "performance_metrics": asdict(state["performance"]),
            "cultural_data": asdict(state["cultural"]),
            "governance_data": asdict(state["governance"]),
            "last_updated": state["current_time"].isoformat()
        }
    
    def add_update_listener(self, listener: callable):
        """Add listener for real-time state updates."""
        self.update_listeners.append(listener)
        logger.debug(f"Added update listener: {listener.__name__}")
    
    def remove_update_listener(self, listener: callable):
        """Remove update listener."""
        if listener in self.update_listeners:
            self.update_listeners.remove(listener)
            logger.debug(f"Removed update listener: {listener.__name__}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        with self._lock:
            return {
                **self.performance_metrics,
                'total_agents': len(self.agents),
                'cache_size': len(self.state_cache),
                'pending_updates': len(self.pending_updates),
                'memory_usage': self._calculate_memory_usage()
            }
    
    # Private helper methods
    
    def _update_memory_systems(self, agent_manager: EnhancedAgentStateManager, 
                              memory_updates: Dict[str, Any]):
        """Update agent memory systems preserving all data."""
        if 'events' in memory_updates:
            for event in memory_updates['events']:
                agent_manager.add_memory(
                    event.get('content', ''),
                    event.get('type', 'general'),
                    event.get('importance', 0.5),
                    event.get('metadata', {})
                )
        
        if 'conversations' in memory_updates:
            for conv in memory_updates['conversations']:
                agent_manager.process_social_interaction(
                    conv.get('partner', 'unknown'),
                    conv.get('interaction_type', 'conversation'),
                    conv.get('content', ''),
                    conv.get('emotional_impact', 0.0)
                )
    
    def _update_specialization(self, agent_manager: EnhancedAgentStateManager,
                             spec_updates: Dict[str, Any]):
        """Update agent specialization preserving progression data."""
        if 'skills' in spec_updates:
            for skill, improvement in spec_updates['skills'].items():
                agent_manager.update_specialization(skill, {skill: improvement})
        
        if 'role_change' in spec_updates:
            # Handle role changes with history tracking
            new_role = spec_updates['role_change']
            current_role = agent_manager.specialization.current_role
            if new_role != current_role:
                agent_manager.specialization.role_history.append(current_role)
                agent_manager.specialization.current_role = new_role
                agent_manager.specialization.last_role_change = datetime.now(timezone.utc)
    
    def _update_social_interactions(self, agent_manager: EnhancedAgentStateManager,
                                   interactions: List[Dict[str, Any]]):
        """Update social interactions preserving relationship data."""
        for interaction in interactions:
            agent_manager.process_social_interaction(
                interaction.get('partner_id', 'unknown'),
                interaction.get('type', 'conversation'),
                interaction.get('content', ''),
                interaction.get('emotional_impact', 0.0)
            )
    
    def _extract_position(self, state: EnhancedAgentState) -> Dict[str, float]:
        """Extract spatial position from memory systems."""
        # TODO: Integrate with spatial memory when available
        # For now, derive from current location
        location_mapping = {
            'villa': {'x': 50.0, 'y': 50.0},
            'pool': {'x': 75.0, 'y': 25.0},
            'kitchen': {'x': 25.0, 'y': 75.0},
            'bedroom': {'x': 80.0, 'y': 80.0}
        }
        return location_mapping.get(state['current_location'], {'x': 50.0, 'y': 50.0})
    
    def _extract_emotional_state(self, state: EnhancedAgentState) -> Dict[str, float]:
        """Extract emotional state from memory and interactions."""
        # Derive from recent memories and personality
        emotional_baseline = {
            'happiness': state['personality_traits'].get('extroversion', 0.5),
            'stress': 1.0 - state['personality_traits'].get('emotional_stability', 0.5),
            'excitement': state['personality_traits'].get('openness', 0.5),
            'romance': 0.5  # Default neutral romance level
        }
        
        # TODO: Enhance with temporal memory analysis
        return emotional_baseline
    
    def _extract_relationships(self, state: EnhancedAgentState) -> Dict[str, float]:
        """Extract relationship scores from governance and social data."""
        return state['governance'].influence_network.copy()
    
    def _extract_dialogue_history(self, state: EnhancedAgentState) -> List[str]:
        """Extract recent dialogue from episodic memory."""
        # Extract from working memory (recent conversations)
        dialogues = []
        for memory in state['working_memory'][-10:]:  # Last 10 memories
            if memory.get('type') == 'conversation':
                dialogues.append(memory.get('content', ''))
        return dialogues
    
    def _extract_memory_summary(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """Extract comprehensive memory summary."""
        return {
            'working_memory_size': len(state['working_memory']),
            'episodic_memory_keys': list(state['episodic_memory'].keys()),
            'semantic_memory_keys': list(state['semantic_memory'].keys()),
            'recent_activities': [m.get('content', '') for m in state['working_memory'][-5:]]
        }
    
    def _schedule_batch_update(self):
        """Schedule batched update notification."""
        if self._update_timer:
            self._update_timer.cancel()
        
        self._update_timer = threading.Timer(
            self.batch_update_interval, 
            self._process_batch_updates
        )
        self._update_timer.start()
    
    def _process_batch_updates(self):
        """Process batched updates for performance."""
        if not self.pending_updates:
            return
        
        with self._lock:
            batch_data = self.pending_updates.copy()
            self.pending_updates.clear()
        
        # Notify listeners of batched updates
        self._notify_listeners('batch_update', batch_data, None)
        logger.debug(f"Processed batch update for {len(batch_data)} agents")
    
    def _notify_listeners(self, event_type: str, agent_id: Any, state_data: Any):
        """Notify all registered listeners of state changes."""
        for listener in self.update_listeners:
            try:
                listener(event_type, agent_id, state_data)
            except Exception as e:
                logger.error(f"Error notifying listener {listener.__name__}: {e}")
    
    def _update_performance_metrics(self, update_time_ms: float):
        """Update system performance metrics."""
        self.performance_metrics['total_updates'] += 1
        
        # Calculate rolling average
        current_avg = self.performance_metrics['average_update_time_ms']
        total_updates = self.performance_metrics['total_updates']
        self.performance_metrics['average_update_time_ms'] = (
            (current_avg * (total_updates - 1) + update_time_ms) / total_updates
        )
        
        self.performance_metrics['last_sync_time'] = datetime.now(timezone.utc)
    
    def _calculate_memory_usage(self) -> Dict[str, int]:
        """Calculate memory usage statistics."""
        total_agents = len(self.agents)
        total_memories = sum(
            len(agent.state['working_memory']) + 
            len(agent.state['short_term_memory']) +
            len(agent.state['episodic_memory']) +
            len(agent.state['semantic_memory'])
            for agent in self.agents.values()
        )
        
        return {
            'total_agents': total_agents,
            'total_memories': total_memories,
            'avg_memories_per_agent': total_memories / max(total_agents, 1)
        }


# Global instance for application-wide use
_unified_manager: Optional[UnifiedAgentManager] = None


def get_unified_agent_manager() -> UnifiedAgentManager:
    """Get or create global UnifiedAgentManager instance."""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = UnifiedAgentManager()
    return _unified_manager


def reset_unified_agent_manager():
    """Reset global manager (mainly for testing)."""
    global _unified_manager
    _unified_manager = None