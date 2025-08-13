"""
PIANO Integration Service
Integrates the dating show system with the existing PIANO/Reverie simulation framework
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

# Add reverie to path for imports
project_root = Path(__file__).parent.parent.parent
reverie_path = project_root / "reverie" / "backend_server"
sys.path.insert(0, str(reverie_path))

try:
    from reverie import ReverieServer
    from persona.persona import Persona
except ImportError as e:
    logging.warning(f"Could not import PIANO/Reverie modules: {e}")
    # Create mock classes for development
    class ReverieServer:
        def __init__(self, *args, **kwargs):
            pass
    class Persona:
        def __init__(self, *args, **kwargs):
            pass

from .enhanced_bridge import EnhancedFrontendBridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatingShowReverieServer(ReverieServer):
    """
    Enhanced Reverie server for dating show simulation.
    
    Extends the base ReverieServer to add:
    - Frontend bridge integration
    - Real-time agent state synchronization
    - Dating show specific behaviors
    - Enhanced monitoring and logging
    """
    
    def __init__(self, fork_sim_code: str, sim_code: str, 
                 frontend_bridge: Optional[EnhancedFrontendBridge] = None,
                 dating_show_config: Optional[Dict[str, Any]] = None):
        """Initialize dating show reverie server"""
        
        # Initialize parent class
        super().__init__(fork_sim_code, sim_code)
        
        # Dating show enhancements
        self.frontend_bridge = frontend_bridge
        self.dating_show_config = dating_show_config or {}
        self.agent_sync_callbacks: List[Callable] = []
        self.governance_callbacks: List[Callable] = []
        self.social_callbacks: List[Callable] = []
        
        # Dating show specific state
        self.dating_show_roles = {}
        self.relationship_tracker = {}
        self.skill_tracker = {}
        
        # Initialize dating show roles if personas are available
        if hasattr(self, 'personas') and self.personas:
            self.dating_show_roles = self._initialize_dating_show_roles()
        
        personas_count = len(self.personas) if hasattr(self, 'personas') and self.personas else 0
        logger.info(f"Dating Show Reverie Server initialized with {personas_count} personas")
    
    def _initialize_dating_show_roles(self) -> Dict[str, str]:
        """Initialize dating show specific roles for personas"""
        roles = {}
        persona_names = list(self.personas.keys())
        
        # Assign roles based on persona count
        total_personas = len(persona_names)
        
        # Most personas are contestants
        contestants_count = min(20, max(10, total_personas - 5))
        # 2-3 hosts
        hosts_count = min(3, max(1, total_personas // 15))
        # Remaining are producers/staff
        
        for i, persona_name in enumerate(persona_names):
            if i < contestants_count:
                roles[persona_name] = "contestant"
            elif i < contestants_count + hosts_count:
                roles[persona_name] = "host"
            else:
                roles[persona_name] = "producer"
        
        logger.info(f"Assigned roles: {contestants_count} contestants, {hosts_count} hosts, "
                   f"{total_personas - contestants_count - hosts_count} producers")
        
        return roles
    
    def register_agent_sync_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for agent state synchronization"""
        self.agent_sync_callbacks.append(callback)
    
    def register_governance_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register callback for governance events"""
        self.governance_callbacks.append(callback)
    
    def register_social_callback(self, callback: Callable[[str, str, Dict[str, Any]], None]):
        """Register callback for social interactions"""
        self.social_callbacks.append(callback)
    
    def sync_agents_to_frontend(self):
        """Synchronize all agent states to frontend bridge"""
        if not self.frontend_bridge:
            return
        
        try:
            for persona_name, persona in self.personas.items():
                agent_data = self._extract_persona_data(persona_name, persona)
                
                # Queue agent update with bridge
                self.frontend_bridge.queue_agent_update(
                    agent_data['agent_id'], 
                    agent_data
                )
                
                # Call registered callbacks
                for callback in self.agent_sync_callbacks:
                    try:
                        callback(agent_data)
                    except Exception as e:
                        logger.error(f"Agent sync callback failed: {e}")
                        
        except Exception as e:
            logger.error(f"Failed to sync agents to frontend: {e}")
    
    def enhanced_simulation_step(self, int_counter: int):
        """
        Enhanced simulation step with dating show features.
        
        Extends the base simulation step to include:
        - Real-time frontend synchronization
        - Dating show specific behaviors
        - Relationship tracking
        - Skill development monitoring
        """
        try:
            # Store original step for monitoring
            original_step = self.step
            
            # Run the original simulation step
            self.simulation_server(int_counter)
            
            # Dating show enhancements after each step
            if self.step > original_step:  # Step was actually executed
                self._post_step_enhancements()
                
        except Exception as e:
            logger.error(f"Enhanced simulation step failed: {e}")
            # Fall back to original simulation
            super().simulation_server(int_counter)
    
    def _post_step_enhancements(self):
        """Perform dating show enhancements after simulation step"""
        try:
            # 1. Sync agents to frontend
            self.sync_agents_to_frontend()
            
            # 2. Track relationships
            self._update_relationship_tracking()
            
            # 3. Monitor skill development
            self._update_skill_tracking()
            
            # 4. Generate governance events
            self._check_governance_events()
            
            # 5. Log dating show specific metrics
            self._log_dating_show_metrics()
            
        except Exception as e:
            logger.error(f"Post-step enhancements failed: {e}")
    
    def _extract_persona_data(self, persona_name: str, persona: Persona) -> Dict[str, Any]:
        """Extract persona data for frontend synchronization"""
        try:
            # Get basic persona information
            agent_data = {
                'agent_id': persona_name.replace(' ', '_').lower(),
                'name': persona_name,
                'current_role': self.dating_show_roles.get(persona_name, 'participant'),
                'specialization': self._extract_specialization(persona),
                'skills': self._extract_skills(persona),
                'memory': self._extract_memory_summary(persona),
                'location': self._extract_location(persona),
                'current_action': self._extract_current_action(persona),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return agent_data
            
        except Exception as e:
            logger.warning(f"Error extracting persona data for {persona_name}: {e}")
            # Return minimal data
            return {
                'agent_id': persona_name.replace(' ', '_').lower(),
                'name': persona_name,
                'current_role': 'participant',
                'specialization': {},
                'skills': {},
                'memory': {},
                'location': {},
                'current_action': 'unknown',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _extract_specialization(self, persona: Persona) -> Dict[str, Any]:
        """Extract persona specialization information"""
        try:
            # Try to get specialization from persona attributes
            if hasattr(persona, 'specialization'):
                return persona.specialization
            
            # Infer from persona description or traits
            if hasattr(persona, 'scratch') and hasattr(persona.scratch, 'get_str_iss'):
                description = persona.scratch.get_str_iss()
                return {
                    'inferred_type': self._infer_specialization_type(description),
                    'description': description[:100] + "..." if len(description) > 100 else description
                }
            
            return {'type': 'general', 'level': 'intermediate'}
            
        except Exception as e:
            logger.debug(f"Error extracting specialization: {e}")
            return {'type': 'general', 'level': 'intermediate'}
    
    def _extract_skills(self, persona: Persona) -> Dict[str, Any]:
        """Extract persona skills information"""
        try:
            # Mock skill system based on persona behaviors
            skills = {
                'social': {'level': 50, 'experience': 100},
                'creative': {'level': 40, 'experience': 80},
                'analytical': {'level': 45, 'experience': 90}
            }
            
            # If persona has actual skills, use those
            if hasattr(persona, 'skills'):
                skills.update(persona.skills)
            
            return skills
            
        except Exception as e:
            logger.debug(f"Error extracting skills: {e}")
            return {'social': {'level': 50, 'experience': 100}}
    
    def _extract_memory_summary(self, persona: Persona) -> Dict[str, Any]:
        """Extract persona memory summary"""
        try:
            memory_summary = {
                'recent_events': [],
                'important_memories': [],
                'total_memories': 0
            }
            
            if hasattr(persona, 'a_mem') and hasattr(persona.a_mem, 'seq_event'):
                # Get recent events
                recent_events = persona.a_mem.seq_event[-5:] if persona.a_mem.seq_event else []
                memory_summary['recent_events'] = [str(event) for event in recent_events]
                memory_summary['total_memories'] = len(persona.a_mem.seq_event)
            
            return memory_summary
            
        except Exception as e:
            logger.debug(f"Error extracting memory: {e}")
            return {'recent_events': [], 'important_memories': [], 'total_memories': 0}
    
    def _extract_location(self, persona: Persona) -> Dict[str, Any]:
        """Extract persona location information"""
        try:
            if hasattr(persona, 'scratch') and hasattr(persona.scratch, 'curr_tile'):
                tile = persona.scratch.curr_tile
                return {
                    'x': tile[0] if tile else 0,
                    'y': tile[1] if tile else 0,
                    'area': getattr(persona.scratch, 'curr_area', 'unknown'),
                    'sector': getattr(persona.scratch, 'curr_sector', 'unknown')
                }
            
            return {'x': 0, 'y': 0, 'area': 'unknown', 'sector': 'unknown'}
            
        except Exception as e:
            logger.debug(f"Error extracting location: {e}")
            return {'x': 0, 'y': 0, 'area': 'unknown', 'sector': 'unknown'}
    
    def _extract_current_action(self, persona: Persona) -> str:
        """Extract persona's current action"""
        try:
            if hasattr(persona, 'scratch') and hasattr(persona.scratch, 'act_description'):
                return persona.scratch.act_description or 'idle'
            
            return 'socializing'
            
        except Exception as e:
            logger.debug(f"Error extracting current action: {e}")
            return 'unknown'
    
    def _infer_specialization_type(self, description: str) -> str:
        """Infer specialization type from description"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['social', 'people', 'relationship', 'friend']):
            return 'social'
        elif any(word in description_lower for word in ['creative', 'art', 'music', 'design']):
            return 'creative'
        elif any(word in description_lower for word in ['think', 'analyze', 'research', 'study']):
            return 'analytical'
        elif any(word in description_lower for word in ['physical', 'sport', 'exercise', 'athletic']):
            return 'physical'
        else:
            return 'general'
    
    def _update_relationship_tracking(self):
        """Update relationship tracking for social dynamics"""
        try:
            # This would analyze persona interactions and update relationship states
            # For now, we'll simulate basic relationship tracking
            
            for persona_name in self.personas.keys():
                if persona_name not in self.relationship_tracker:
                    self.relationship_tracker[persona_name] = {}
                
                # Track relationships with other personas
                for other_name in self.personas.keys():
                    if other_name != persona_name:
                        # Simulate relationship strength changes
                        current_strength = self.relationship_tracker[persona_name].get(other_name, 0.0)
                        # Small random changes for simulation
                        change = (hash(f"{persona_name}{other_name}{self.step}") % 21 - 10) / 1000  # -0.01 to 0.01
                        new_strength = max(-1.0, min(1.0, current_strength + change))
                        self.relationship_tracker[persona_name][other_name] = new_strength
            
            # Queue social updates to frontend
            if self.frontend_bridge:
                self._queue_social_updates()
                
        except Exception as e:
            logger.error(f"Error updating relationship tracking: {e}")
    
    def _update_skill_tracking(self):
        """Update skill development tracking"""
        try:
            for persona_name, persona in self.personas.items():
                if persona_name not in self.skill_tracker:
                    self.skill_tracker[persona_name] = {
                        'social': 50,
                        'creative': 40,
                        'analytical': 45
                    }
                
                # Simulate skill growth based on actions
                # This would be based on actual persona actions in a real implementation
                for skill in self.skill_tracker[persona_name]:
                    growth = (hash(f"{persona_name}{skill}{self.step}") % 3) / 100  # 0-0.02 growth
                    self.skill_tracker[persona_name][skill] = min(100, 
                        self.skill_tracker[persona_name][skill] + growth)
                        
        except Exception as e:
            logger.error(f"Error updating skill tracking: {e}")
    
    def _check_governance_events(self):
        """Check for governance events and queue updates"""
        try:
            # Simulate governance events occasionally
            if self.step % 100 == 0:  # Every 100 steps
                governance_data = {
                    'event_type': 'rule_proposal',
                    'step': self.step,
                    'description': f'New rule proposed at step {self.step}',
                    'participants': list(self.personas.keys())[:5]  # First 5 personas
                }
                
                if self.frontend_bridge:
                    self.frontend_bridge.queue_governance_update(
                        'rule_proposal', 
                        governance_data
                    )
                
                # Call governance callbacks
                for callback in self.governance_callbacks:
                    try:
                        callback('rule_proposal', governance_data)
                    except Exception as e:
                        logger.error(f"Governance callback failed: {e}")
                        
        except Exception as e:
            logger.error(f"Error checking governance events: {e}")
    
    def _queue_social_updates(self):
        """Queue social relationship updates to frontend"""
        try:
            if not self.frontend_bridge:
                return
            
            # Queue significant relationship changes
            for persona_a, relationships in self.relationship_tracker.items():
                for persona_b, strength in relationships.items():
                    # Queue updates for significant relationships
                    if abs(strength) > 0.3:  # Only significant relationships
                        self.frontend_bridge.queue_social_update(
                            persona_a.replace(' ', '_').lower(),
                            persona_b.replace(' ', '_').lower(),
                            'friendship' if strength > 0 else 'rivalry',
                            strength,
                            'ongoing'
                        )
                        
        except Exception as e:
            logger.error(f"Error queuing social updates: {e}")
    
    def _log_dating_show_metrics(self):
        """Log dating show specific metrics"""
        try:
            if self.step % 50 == 0:  # Log every 50 steps
                total_relationships = sum(len(rels) for rels in self.relationship_tracker.values())
                avg_skills = {}
                
                for persona_skills in self.skill_tracker.values():
                    for skill, level in persona_skills.items():
                        if skill not in avg_skills:
                            avg_skills[skill] = []
                        avg_skills[skill].append(level)
                
                for skill in avg_skills:
                    avg_skills[skill] = sum(avg_skills[skill]) / len(avg_skills[skill])
                
                logger.info(f"Dating Show Metrics - Step {self.step}: "
                           f"{len(self.personas)} personas, "
                           f"{total_relationships} relationships, "
                           f"Avg skills: {avg_skills}")
                           
        except Exception as e:
            logger.debug(f"Error logging metrics: {e}")


def create_dating_show_reverie_server(
    fork_sim_code: str,
    sim_code: str,
    frontend_bridge: Optional[EnhancedFrontendBridge] = None,
    dating_show_config: Optional[Dict[str, Any]] = None
) -> DatingShowReverieServer:
    """
    Factory function to create dating show reverie server.
    
    Args:
        fork_sim_code: Source simulation code to fork from
        sim_code: New simulation code
        frontend_bridge: Optional frontend bridge for real-time sync
        dating_show_config: Optional dating show configuration
        
    Returns:
        Configured DatingShowReverieServer instance
    """
    return DatingShowReverieServer(
        fork_sim_code=fork_sim_code,
        sim_code=sim_code,
        frontend_bridge=frontend_bridge,
        dating_show_config=dating_show_config
    )