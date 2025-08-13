"""
PIANO Integration Module
Integrates the Frontend Bridge with existing PIANO agent architecture
Provides seamless integration with LangGraph agents and memory systems
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

from ..agents.agent import AgentPersona
from ..agents.enhanced_agent_state import EnhancedAgentState
from ..governance.voting_system import VotingSystem
from ..governance.compliance_monitoring import ComplianceMonitor
from ..social.relationship_network import RelationshipNetwork
from .frontend_bridge import get_bridge, FrontendBridge

logger = logging.getLogger(__name__)


class PianoFrontendIntegration:
    """
    Integration layer between PIANO agents and the Django frontend
    Handles automatic state synchronization and event propagation
    """
    
    def __init__(self, bridge: Optional[FrontendBridge] = None):
        self.bridge = bridge or get_bridge()
        self.monitored_agents: Dict[str, AgentPersona] = {}
        self.last_sync_states: Dict[str, Dict] = {}
        
        # Hook into PIANO systems
        self._setup_event_listeners()
        
    def _setup_event_listeners(self):
        """Setup event listeners for PIANO systems"""
        logger.info("Setting up PIANO-Frontend integration event listeners")
        
    def register_agent(self, agent: AgentPersona):
        """Register an agent for frontend synchronization"""
        agent_id = agent.scratch.name  # Using name as ID for now
        self.monitored_agents[agent_id] = agent
        
        # Send initial state
        self._sync_agent_state(agent)
        logger.info(f"Registered agent {agent_id} for frontend sync")
        
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from frontend synchronization"""
        if agent_id in self.monitored_agents:
            del self.monitored_agents[agent_id]
            if agent_id in self.last_sync_states:
                del self.last_sync_states[agent_id]
            logger.info(f"Unregistered agent {agent_id} from frontend sync")
            
    def _sync_agent_state(self, agent: AgentPersona):
        """Synchronize a single agent's state to the frontend"""
        try:
            agent_id = agent.scratch.name
            
            # Extract agent data
            agent_data = self._extract_agent_data(agent)
            
            # Check if state has changed significantly
            if self._should_sync_agent(agent_id, agent_data):
                self.bridge.queue_agent_update(agent_id, agent_data)
                self.last_sync_states[agent_id] = agent_data.copy()
                
        except Exception as e:
            logger.error(f"Error syncing agent state: {e}", exc_info=True)
            
    def _extract_agent_data(self, agent: AgentPersona) -> Dict[str, Any]:
        """Extract relevant data from a PIANO agent for frontend display"""
        try:
            # Basic agent information
            agent_data = {
                'name': agent.scratch.name,
                'current_role': self._detect_current_role(agent),
                'current_action': agent.scratch.act_address[-1] if agent.scratch.act_address else "idle",
                'location': {
                    'sector': agent.scratch.curr_tile[0] if agent.scratch.curr_tile else "",
                    'arena': agent.scratch.curr_tile[1] if agent.scratch.curr_tile else "",
                    'game_object': agent.scratch.curr_tile[2] if agent.scratch.curr_tile else ""
                },
                'specialization': {
                    'current_role': self._detect_current_role(agent),
                    'role_history': getattr(agent.scratch, 'role_history', []),
                    'expertise_areas': self._extract_expertise_areas(agent)
                }
            }
            
            # Skills information
            agent_data['skills'] = self._extract_skills(agent)
            
            # Memory information for frontend display
            agent_data['memory'] = self._extract_memory_summary(agent)
            
            # Social context
            agent_data['social_context'] = self._extract_social_context(agent)
            
            return agent_data
            
        except Exception as e:
            logger.error(f"Error extracting agent data: {e}", exc_info=True)
            return {
                'name': agent.scratch.name if hasattr(agent, 'scratch') else 'Unknown',
                'current_role': 'unknown',
                'current_action': 'error',
                'location': {},
                'specialization': {},
                'skills': {},
                'memory': {},
                'social_context': {}
            }
            
    def _detect_current_role(self, agent: AgentPersona) -> str:
        """Detect the agent's current professional role"""
        try:
            # Check if agent has role detection module
            if hasattr(agent, 'role_detector'):
                return agent.role_detector.get_current_role()
                
            # Fallback: analyze recent actions
            recent_actions = agent.scratch.act_address[-10:] if agent.scratch.act_address else []
            
            # Simple heuristic role detection
            if any('host' in action.lower() for action in recent_actions):
                return 'host'
            elif any('produce' in action.lower() for action in recent_actions):
                return 'producer'
            elif any('contest' in action.lower() for action in recent_actions):
                return 'contestant'
            else:
                return 'participant'
                
        except Exception as e:
            logger.error(f"Error detecting role: {e}")
            return 'unknown'
            
    def _extract_expertise_areas(self, agent: AgentPersona) -> List[str]:
        """Extract agent's areas of expertise"""
        try:
            expertise = []
            
            # Check for skill-based expertise
            if hasattr(agent, 'skills'):
                high_skills = [skill for skill, level in agent.skills.items() if level > 0.7]
                expertise.extend(high_skills)
                
            # Check personality-based expertise
            if hasattr(agent.scratch, 'innate'):
                innate_traits = agent.scratch.innate.split(';')
                expertise.extend([trait.strip() for trait in innate_traits])
                
            return list(set(expertise))
            
        except Exception as e:
            logger.error(f"Error extracting expertise: {e}")
            return []
            
    def _extract_skills(self, agent: AgentPersona) -> Dict[str, Dict[str, float]]:
        """Extract agent's skill information"""
        try:
            skills = {}
            
            # Check for skill development module
            if hasattr(agent, 'skill_system'):
                for skill_name, skill_data in agent.skill_system.skills.items():
                    skills[skill_name] = {
                        'level': skill_data.get('level', 0.0),
                        'experience': skill_data.get('experience', 0.0),
                        'last_practiced': skill_data.get('last_practiced', 0)
                    }
                    
            return skills
            
        except Exception as e:
            logger.error(f"Error extracting skills: {e}")
            return {}
            
    def _extract_memory_summary(self, agent: AgentPersona) -> Dict[str, Any]:
        """Extract memory summary for frontend display"""
        try:
            memory_summary = {
                'working_memory': [],
                'recent_events': [],
                'important_thoughts': [],
                'social_memories': []
            }
            
            # Working memory (recent observations and plans)
            if hasattr(agent.scratch, 'act_address'):
                memory_summary['working_memory'] = agent.scratch.act_address[-5:]
                
            # Recent associative memory nodes
            if hasattr(agent, 'associative_memory'):
                recent_nodes = agent.associative_memory.retrieve_nodes(
                    agent.scratch.name, retrieved=5
                )
                
                for node in recent_nodes:
                    if node.type == 'event':
                        memory_summary['recent_events'].append({
                            'description': node.description,
                            'created': node.created.isoformat(),
                            'importance': node.poignancy
                        })
                    elif node.type == 'thought':
                        memory_summary['important_thoughts'].append({
                            'description': node.description,
                            'created': node.created.isoformat(),
                            'importance': node.poignancy
                        })
                        
            return memory_summary
            
        except Exception as e:
            logger.error(f"Error extracting memory: {e}")
            return {}
            
    def _extract_social_context(self, agent: AgentPersona) -> Dict[str, Any]:
        """Extract social context information"""
        try:
            social_context = {
                'current_conversation': None,
                'recent_interactions': [],
                'relationship_status': {}
            }
            
            # Current conversation
            if hasattr(agent.scratch, 'chatting_with') and agent.scratch.chatting_with:
                social_context['current_conversation'] = agent.scratch.chatting_with
                
            # Recent interactions (from memory)
            if hasattr(agent, 'associative_memory'):
                chat_nodes = agent.associative_memory.retrieve_nodes(
                    agent.scratch.name, node_type='chat', retrieved=5
                )
                
                for node in chat_nodes:
                    social_context['recent_interactions'].append({
                        'partner': node.subject if hasattr(node, 'subject') else 'unknown',
                        'summary': node.description[:100] + '...' if len(node.description) > 100 else node.description,
                        'timestamp': node.created.isoformat()
                    })
                    
            return social_context
            
        except Exception as e:
            logger.error(f"Error extracting social context: {e}")
            return {}
            
    def _should_sync_agent(self, agent_id: str, current_data: Dict[str, Any]) -> bool:
        """Determine if agent state should be synced based on changes"""
        if agent_id not in self.last_sync_states:
            return True
            
        last_state = self.last_sync_states[agent_id]
        
        # Check for significant changes
        significant_changes = [
            current_data.get('current_action') != last_state.get('current_action'),
            current_data.get('location') != last_state.get('location'),
            current_data.get('current_role') != last_state.get('current_role'),
            len(current_data.get('skills', {})) != len(last_state.get('skills', {}))
        ]
        
        return any(significant_changes)
        
    def sync_all_agents(self):
        """Manually sync all registered agents"""
        for agent in self.monitored_agents.values():
            self._sync_agent_state(agent)
            
    def handle_governance_event(self, event_type: str, data: Dict[str, Any]):
        """Handle governance system events"""
        try:
            self.bridge.queue_governance_update(event_type, data)
            logger.debug(f"Governance event queued: {event_type}")
        except Exception as e:
            logger.error(f"Error handling governance event: {e}")
            
    def handle_social_event(self, agent_a: str, agent_b: str, 
                           relationship_type: str, strength: float, 
                           interaction_type: str = "interaction"):
        """Handle social interaction events"""
        try:
            self.bridge.queue_social_update(
                agent_a, agent_b, relationship_type, strength, interaction_type
            )
            logger.debug(f"Social event queued: {agent_a} -> {agent_b}")
        except Exception as e:
            logger.error(f"Error handling social event: {e}")
            
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status information"""
        return {
            'monitored_agents': len(self.monitored_agents),
            'agent_ids': list(self.monitored_agents.keys()),
            'bridge_status': self.bridge.get_bridge_status(),
            'last_sync_count': len(self.last_sync_states)
        }


# Global integration instance
_integration_instance: Optional[PianoFrontendIntegration] = None


def get_integration() -> PianoFrontendIntegration:
    """Get or create the global integration instance"""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = PianoFrontendIntegration()
    return _integration_instance


def initialize_integration(bridge: Optional[FrontendBridge] = None) -> PianoFrontendIntegration:
    """Initialize the integration with custom configuration"""
    global _integration_instance
    _integration_instance = PianoFrontendIntegration(bridge)
    return _integration_instance