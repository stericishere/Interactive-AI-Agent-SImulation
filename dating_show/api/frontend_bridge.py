"""
Frontend Bridge Service
Integration layer connecting PIANO agents with Django frontend server
Handles state synchronization, real-time updates, and API communication
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import requests
from urllib.parse import urljoin
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentUpdate:
    """Data class for agent state updates"""
    agent_id: str
    name: str
    current_role: str
    specialization: Dict[str, Any]
    skills: Dict[str, Dict[str, float]]
    memory: Dict[str, Any]
    location: Dict[str, Any]
    current_action: str
    timestamp: datetime


@dataclass
class GovernanceUpdate:
    """Data class for governance system updates"""
    update_type: str  # 'new_vote', 'vote_cast', 'rule_created', 'violation_detected'
    data: Dict[str, Any]
    timestamp: datetime


@dataclass
class SocialUpdate:
    """Data class for social network updates"""
    agent_a_id: str
    agent_b_id: str
    relationship_type: str
    strength: float
    interaction_type: str
    timestamp: datetime


class FrontendBridge:
    """
    Service layer that bridges PIANO agents with Django frontend
    Handles bidirectional communication and real-time updates
    """
    
    def __init__(self, frontend_url: str = "http://localhost:8001", 
                 update_interval: float = 1.0):
        self.frontend_url = frontend_url
        self.update_interval = update_interval
        self.session = requests.Session()
        
        # Update queues
        self.agent_updates: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.governance_updates: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.social_updates: asyncio.Queue = asyncio.Queue(maxsize=500)
        
        # State caches for performance optimization
        self.agent_state_cache: Dict[str, AgentUpdate] = {}
        self.last_sync_time: float = 0.0
        
        # Control flags
        self.running = False
        self.sync_thread: Optional[threading.Thread] = None
        
    def start_bridge(self):
        """Start the bridge service"""
        if self.running:
            logger.warning("Bridge service is already running")
            return
            
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        logger.info("Frontend bridge service started")
        
    def stop_bridge(self):
        """Stop the bridge service"""
        self.running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5.0)
        logger.info("Frontend bridge service stopped")
        
    def _sync_loop(self):
        """Main synchronization loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Process queued updates
                self._process_agent_updates()
                self._process_governance_updates()
                self._process_social_updates()
                
                # Send batch updates to frontend
                self._send_batch_updates()
                
                # Calculate sleep time to maintain update interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in sync loop: {e}", exc_info=True)
                time.sleep(1.0)  # Brief pause before retrying
                
    def _process_agent_updates(self):
        """Process pending agent updates"""
        batch_updates = {}
        
        try:
            # Process all available agent updates
            while True:
                try:
                    update = self.agent_updates.get_nowait()
                    batch_updates[update.agent_id] = update
                except asyncio.QueueEmpty:
                    break
                    
            # Send batch updates
            for agent_id, update in batch_updates.items():
                self._sync_agent_to_frontend(update)
                self.agent_state_cache[agent_id] = update
                
        except Exception as e:
            logger.error(f"Error processing agent updates: {e}")
            
    def _process_governance_updates(self):
        """Process pending governance updates"""
        try:
            while True:
                try:
                    update = self.governance_updates.get_nowait()
                    self._sync_governance_to_frontend(update)
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            logger.error(f"Error processing governance updates: {e}")
            
    def _process_social_updates(self):
        """Process pending social updates"""
        try:
            while True:
                try:
                    update = self.social_updates.get_nowait()
                    self._sync_social_to_frontend(update)
                except asyncio.QueueEmpty:
                    break
        except Exception as e:
            logger.error(f"Error processing social updates: {e}")
            
    def _send_batch_updates(self):
        """Send batch updates for performance optimization"""
        current_time = time.time()
        
        # Send performance metrics every 5 seconds
        if current_time - self.last_sync_time > 5.0:
            self._send_performance_metrics()
            self.last_sync_time = current_time
            
    def _sync_agent_to_frontend(self, update: AgentUpdate):
        """Synchronize agent state to Django frontend"""
        try:
            url = urljoin(self.frontend_url, f"/dating_show/api/agents/{update.agent_id}/state/")
            
            payload = {
                'name': update.name,
                'current_role': update.current_role,
                'specialization': update.specialization,
                'skills': update.skills,
                'memory': update.memory,
                'location': update.location,
                'current_action': update.current_action,
                'timestamp': update.timestamp.isoformat()
            }
            
            response = self.session.post(url, json=payload, timeout=5.0)
            
            if response.status_code == 200:
                logger.debug(f"Agent {update.agent_id} synced successfully")
            else:
                logger.warning(f"Failed to sync agent {update.agent_id}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error syncing agent {update.agent_id}: {e}")
            
    def _sync_governance_to_frontend(self, update: GovernanceUpdate):
        """Synchronize governance updates to Django frontend"""
        try:
            url = urljoin(self.frontend_url, "/dating_show/api/broadcast/update/")
            
            payload = {
                'type': 'governance',
                'payload': {
                    'update_type': update.update_type,
                    'data': update.data,
                    'timestamp': update.timestamp.isoformat()
                }
            }
            
            response = self.session.post(url, json=payload, timeout=5.0)
            
            if response.status_code == 200:
                logger.debug(f"Governance update {update.update_type} synced successfully")
            else:
                logger.warning(f"Failed to sync governance update: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error syncing governance update: {e}")
            
    def _sync_social_to_frontend(self, update: SocialUpdate):
        """Synchronize social network updates to Django frontend"""
        try:
            url = urljoin(self.frontend_url, "/dating_show/api/broadcast/update/")
            
            payload = {
                'type': 'social',
                'payload': {
                    'agent_a_id': update.agent_a_id,
                    'agent_b_id': update.agent_b_id,
                    'relationship_type': update.relationship_type,
                    'strength': update.strength,
                    'interaction_type': update.interaction_type,
                    'timestamp': update.timestamp.isoformat()
                }
            }
            
            response = self.session.post(url, json=payload, timeout=5.0)
            
            if response.status_code == 200:
                logger.debug(f"Social update synced successfully")
            else:
                logger.warning(f"Failed to sync social update: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error syncing social update: {e}")
            
    def _send_performance_metrics(self):
        """Send system performance metrics to frontend"""
        try:
            metrics = {
                'active_agents': len(self.agent_state_cache),
                'queue_sizes': {
                    'agent_updates': self.agent_updates.qsize(),
                    'governance_updates': self.governance_updates.qsize(),
                    'social_updates': self.social_updates.qsize()
                },
                'last_sync': datetime.now(timezone.utc).isoformat(),
                'uptime_seconds': time.time() - self.last_sync_time
            }
            
            url = urljoin(self.frontend_url, "/dating_show/api/broadcast/update/")
            payload = {
                'type': 'performance',
                'payload': metrics
            }
            
            response = self.session.post(url, json=payload, timeout=5.0)
            logger.debug(f"Performance metrics sent: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Error sending performance metrics: {e}")
            
    # ==================== PUBLIC API METHODS ====================
    
    def queue_agent_update(self, agent_id: str, agent_data: Dict[str, Any]):
        """Queue an agent state update for synchronization"""
        try:
            update = AgentUpdate(
                agent_id=agent_id,
                name=agent_data.get('name', ''),
                current_role=agent_data.get('current_role', ''),
                specialization=agent_data.get('specialization', {}),
                skills=agent_data.get('skills', {}),
                memory=agent_data.get('memory', {}),
                location=agent_data.get('location', {}),
                current_action=agent_data.get('current_action', ''),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Use put_nowait to avoid blocking
            if not self.agent_updates.full():
                self.agent_updates.put_nowait(update)
            else:
                logger.warning(f"Agent update queue is full, dropping update for {agent_id}")
                
        except Exception as e:
            logger.error(f"Error queuing agent update for {agent_id}: {e}")
            
    def queue_governance_update(self, update_type: str, data: Dict[str, Any]):
        """Queue a governance system update"""
        try:
            update = GovernanceUpdate(
                update_type=update_type,
                data=data,
                timestamp=datetime.now(timezone.utc)
            )
            
            if not self.governance_updates.full():
                self.governance_updates.put_nowait(update)
            else:
                logger.warning("Governance update queue is full, dropping update")
                
        except Exception as e:
            logger.error(f"Error queuing governance update: {e}")
            
    def queue_social_update(self, agent_a_id: str, agent_b_id: str, 
                          relationship_type: str, strength: float, 
                          interaction_type: str = "interaction"):
        """Queue a social network update"""
        try:
            update = SocialUpdate(
                agent_a_id=agent_a_id,
                agent_b_id=agent_b_id,
                relationship_type=relationship_type,
                strength=strength,
                interaction_type=interaction_type,
                timestamp=datetime.now(timezone.utc)
            )
            
            if not self.social_updates.full():
                self.social_updates.put_nowait(update)
            else:
                logger.warning("Social update queue is full, dropping update")
                
        except Exception as e:
            logger.error(f"Error queuing social update: {e}")
            
    def get_cached_agent_state(self, agent_id: str) -> Optional[AgentUpdate]:
        """Get cached agent state for fast access"""
        return self.agent_state_cache.get(agent_id)
        
    def clear_cache(self):
        """Clear all cached states"""
        self.agent_state_cache.clear()
        logger.info("Agent state cache cleared")
        
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge service status"""
        return {
            'running': self.running,
            'cached_agents': len(self.agent_state_cache),
            'queue_sizes': {
                'agent_updates': self.agent_updates.qsize(),
                'governance_updates': self.governance_updates.qsize(),
                'social_updates': self.social_updates.qsize()
            },
            'frontend_url': self.frontend_url,
            'update_interval': self.update_interval
        }


# Global bridge instance
_bridge_instance: Optional[FrontendBridge] = None


def get_bridge() -> FrontendBridge:
    """Get or create the global bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = FrontendBridge()
        _bridge_instance.start_bridge()
    return _bridge_instance


def initialize_bridge(frontend_url: str = "http://localhost:8001", 
                     update_interval: float = 1.0) -> FrontendBridge:
    """Initialize the bridge with custom configuration"""
    global _bridge_instance
    if _bridge_instance is not None:
        _bridge_instance.stop_bridge()
    
    _bridge_instance = FrontendBridge(frontend_url, update_interval)
    _bridge_instance.start_bridge()
    return _bridge_instance


def shutdown_bridge():
    """Shutdown the global bridge instance"""
    global _bridge_instance
    if _bridge_instance is not None:
        _bridge_instance.stop_bridge()
        _bridge_instance = None