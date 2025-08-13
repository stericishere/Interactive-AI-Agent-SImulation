"""
Relationship Network Engine for Enhanced PIANO Architecture
Phase 3: Advanced Features - Week 9: Complex Social Dynamics

This module implements a sophisticated social graph management system that tracks
and analyzes relationships between agents, enabling complex social dynamics
including influence propagation and trust network dynamics.
"""

import json
import math
import time
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import weakref


class RelationshipType(Enum):
    """Types of relationships between agents"""
    ROMANTIC = "romantic"
    FRIENDSHIP = "friendship"
    ALLIANCE = "alliance"
    RIVALRY = "rivalry"
    MENTORSHIP = "mentorship"
    PROFESSIONAL = "professional"
    FAMILY = "family"
    NEUTRAL = "neutral"
    CONFLICT = "conflict"


class InfluenceType(Enum):
    """Types of influence that can propagate through the network"""
    OPINION = "opinion"
    EMOTION = "emotion"
    MEME = "meme"
    BEHAVIOR = "behavior"
    INFORMATION = "information"
    TRUST = "trust"


@dataclass
class RelationshipEdge:
    """Represents a relationship between two agents"""
    agent_a: str
    agent_b: str
    relationship_type: RelationshipType
    strength: float  # 0.0 to 1.0
    trust_level: float  # -1.0 to 1.0 (negative for distrust)
    reciprocal: bool  # Whether the relationship is mutual
    created_time: float
    last_interaction: float
    interaction_count: int
    shared_experiences: List[str]
    emotional_valence: float  # -1.0 to 1.0 (negative to positive)
    stability: float  # How stable this relationship is (0.0 to 1.0)
    
    def __post_init__(self):
        """Validate relationship edge data"""
        self.strength = max(0.0, min(1.0, self.strength))
        self.trust_level = max(-1.0, min(1.0, self.trust_level))
        self.emotional_valence = max(-1.0, min(1.0, self.emotional_valence))
        self.stability = max(0.0, min(1.0, self.stability))


@dataclass
class InfluencePacket:
    """Represents influence propagating through the network"""
    source_agent: str
    influence_type: InfluenceType
    content: Dict[str, Any]
    strength: float
    decay_rate: float
    max_hops: int
    current_hops: int
    visited_agents: Set[str]
    timestamp: float
    
    def decay(self) -> float:
        """Calculate current influence strength after decay"""
        hop_decay = (1.0 - self.decay_rate) ** self.current_hops
        time_decay = max(0.1, 1.0 - (time.time() - self.timestamp) / 3600.0)  # Decay over 1 hour
        return self.strength * hop_decay * time_decay


class RelationshipNetwork:
    """
    Social graph management system for tracking and analyzing agent relationships.
    
    Features:
    - Dynamic relationship tracking with multiple relationship types
    - Trust network analysis and propagation
    - Influence propagation algorithms
    - Social clustering and community detection
    - Relationship prediction and recommendation
    - Network metrics and analytics
    """
    
    def __init__(self, max_agents: int = 1000):
        self.max_agents = max_agents
        self.agents: Set[str] = set()
        self.relationships: Dict[Tuple[str, str], RelationshipEdge] = {}
        self.adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        self.influence_queue: deque = deque()
        self.communities: Dict[str, Set[str]] = {}
        self.network_metrics: Dict[str, float] = {}
        self.lock = threading.RLock()
        
        # Performance optimization with LRU-like behavior
        self._trust_cache: Dict[Tuple[str, str], float] = {}
        self._centrality_cache: Dict[str, float] = {}
        self._cache_timestamp = 0.0
        self._cache_ttl = 300.0  # 5 minutes
        self._max_cache_size = 1000  # Limit cache size for memory efficiency
        
    def add_agent(self, agent_id: str) -> bool:
        """Add an agent to the network"""
        with self.lock:
            if len(self.agents) >= self.max_agents:
                return False
            
            self.agents.add(agent_id)
            if agent_id not in self.adjacency_list:
                self.adjacency_list[agent_id] = set()
            
            self._invalidate_cache()
            return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent and all their relationships from the network"""
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            # Remove all relationships involving this agent
            to_remove = []
            for (a, b), edge in self.relationships.items():
                if a == agent_id or b == agent_id:
                    to_remove.append((a, b))
            
            for edge_key in to_remove:
                del self.relationships[edge_key]
                a, b = edge_key
                self.adjacency_list[a].discard(b)
                self.adjacency_list[b].discard(a)
            
            self.agents.remove(agent_id)
            del self.adjacency_list[agent_id]
            
            self._invalidate_cache()
            return True
    
    def create_relationship(
        self, 
        agent_a: str, 
        agent_b: str, 
        relationship_type: RelationshipType,
        strength: float = 0.5,
        trust_level: float = 0.0,
        reciprocal: bool = True
    ) -> bool:
        """Create or update a relationship between two agents"""
        with self.lock:
            if agent_a not in self.agents or agent_b not in self.agents:
                return False
            
            if agent_a == agent_b:
                return False
            
            # Ensure consistent ordering
            if agent_a > agent_b:
                agent_a, agent_b = agent_b, agent_a
            
            edge_key = (agent_a, agent_b)
            current_time = time.time()
            
            if edge_key in self.relationships:
                # Update existing relationship
                edge = self.relationships[edge_key]
                edge.relationship_type = relationship_type
                edge.strength = max(0.0, min(1.0, strength))
                edge.trust_level = max(-1.0, min(1.0, trust_level))
                edge.last_interaction = current_time
                edge.interaction_count += 1
                edge.reciprocal = reciprocal
            else:
                # Create new relationship
                edge = RelationshipEdge(
                    agent_a=agent_a,
                    agent_b=agent_b,
                    relationship_type=relationship_type,
                    strength=strength,
                    trust_level=trust_level,
                    reciprocal=reciprocal,
                    created_time=current_time,
                    last_interaction=current_time,
                    interaction_count=1,
                    shared_experiences=[],
                    emotional_valence=0.0,
                    stability=0.5
                )
                self.relationships[edge_key] = edge
                
                # Update adjacency list
                self.adjacency_list[agent_a].add(agent_b)
                self.adjacency_list[agent_b].add(agent_a)
            
            self._invalidate_cache()
            return True
    
    def update_relationship(
        self, 
        agent_a: str, 
        agent_b: str, 
        interaction_type: str,
        emotional_impact: float = 0.0,
        shared_experience: Optional[str] = None
    ) -> bool:
        """Update relationship based on an interaction"""
        with self.lock:
            if agent_a > agent_b:
                agent_a, agent_b = agent_b, agent_a
            
            edge_key = (agent_a, agent_b)
            if edge_key not in self.relationships:
                return False
            
            edge = self.relationships[edge_key]
            
            # Update interaction metrics
            edge.last_interaction = time.time()
            edge.interaction_count += 1
            
            # Update emotional valence
            edge.emotional_valence = (edge.emotional_valence * 0.9) + (emotional_impact * 0.1)
            edge.emotional_valence = max(-1.0, min(1.0, edge.emotional_valence))
            
            # Update trust based on interaction type and emotional impact
            trust_change = self._calculate_trust_change(interaction_type, emotional_impact)
            edge.trust_level += trust_change
            edge.trust_level = max(-1.0, min(1.0, edge.trust_level))
            
            # Update relationship strength
            strength_change = self._calculate_strength_change(interaction_type, emotional_impact)
            edge.strength += strength_change
            edge.strength = max(0.0, min(1.0, edge.strength))
            
            # Add shared experience
            if shared_experience:
                edge.shared_experiences.append(shared_experience)
                # Keep only recent experiences
                if len(edge.shared_experiences) > 10:
                    edge.shared_experiences = edge.shared_experiences[-10:]
            
            # Update stability based on interaction frequency
            time_since_creation = time.time() - edge.created_time
            interaction_rate = edge.interaction_count / max(time_since_creation / 86400, 1.0)  # Per day
            edge.stability = min(1.0, interaction_rate * 0.1)
            
            self._invalidate_cache()
            return True
    
    def get_relationship(self, agent_a: str, agent_b: str) -> Optional[RelationshipEdge]:
        """Get relationship between two agents"""
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a
        
        return self.relationships.get((agent_a, agent_b))
    
    def get_agent_relationships(self, agent_id: str) -> List[RelationshipEdge]:
        """Get all relationships for a specific agent"""
        relationships = []
        for (a, b), edge in self.relationships.items():
            if a == agent_id or b == agent_id:
                relationships.append(edge)
        return relationships
    
    def calculate_trust_path(self, source: str, target: str, max_hops: int = 5) -> Optional[float]:
        """Calculate trust level along the shortest path between two agents"""
        if source == target:
            return 1.0
        
        # Check cache
        cache_key = (source, target)
        if self._is_cache_valid() and cache_key in self._trust_cache:
            return self._trust_cache[cache_key]
        
        # BFS to find shortest trust path
        queue = deque([(source, 1.0, 0)])
        visited = {source}
        
        while queue:
            current_agent, current_trust, hops = queue.popleft()
            
            if hops >= max_hops:
                continue
            
            for neighbor in self.adjacency_list[current_agent]:
                if neighbor in visited:
                    continue
                
                edge = self.get_relationship(current_agent, neighbor)
                if not edge:
                    continue
                
                # Calculate trust propagation
                edge_trust = edge.trust_level * edge.strength
                propagated_trust = current_trust * edge_trust
                
                if neighbor == target:
                    self._trust_cache[cache_key] = propagated_trust
                    return propagated_trust
                
                if propagated_trust > 0.1:  # Only continue if trust is significant
                    queue.append((neighbor, propagated_trust, hops + 1))
                    visited.add(neighbor)
        
        self._trust_cache[cache_key] = 0.0
        return 0.0
    
    def propagate_influence(
        self, 
        source_agent: str, 
        influence_type: InfluenceType,
        content: Dict[str, Any],
        initial_strength: float = 1.0,
        decay_rate: float = 0.2,
        max_hops: int = 5
    ) -> Dict[str, float]:
        """Propagate influence through the network"""
        if source_agent not in self.agents:
            return {}
        
        influence_packet = InfluencePacket(
            source_agent=source_agent,
            influence_type=influence_type,
            content=content,
            strength=initial_strength,
            decay_rate=decay_rate,
            max_hops=max_hops,
            current_hops=0,
            visited_agents={source_agent},
            timestamp=time.time()
        )
        
        influence_results = {source_agent: initial_strength}
        queue = deque([(source_agent, influence_packet)])
        
        while queue:
            current_agent, packet = queue.popleft()
            
            if packet.current_hops >= packet.max_hops:
                continue
            
            current_strength = packet.decay()
            if current_strength < 0.01:  # Stop if influence too weak
                continue
            
            for neighbor in self.adjacency_list[current_agent]:
                if neighbor in packet.visited_agents:
                    continue
                
                edge = self.get_relationship(current_agent, neighbor)
                if not edge:
                    continue
                
                # Calculate influence transmission
                transmission_strength = self._calculate_influence_transmission(
                    edge, influence_type, current_strength
                )
                
                if transmission_strength > 0.01:
                    new_packet = InfluencePacket(
                        source_agent=packet.source_agent,
                        influence_type=packet.influence_type,
                        content=packet.content,
                        strength=transmission_strength,
                        decay_rate=packet.decay_rate,
                        max_hops=packet.max_hops,
                        current_hops=packet.current_hops + 1,
                        visited_agents=packet.visited_agents | {neighbor},
                        timestamp=packet.timestamp
                    )
                    
                    queue.append((neighbor, new_packet))
                    influence_results[neighbor] = transmission_strength
        
        return influence_results
    
    def detect_communities(self) -> Dict[str, Set[str]]:
        """Detect communities in the network using modularity optimization"""
        if not self.agents:
            return {}
        
        # Initialize each agent in its own community
        communities = {agent: {agent} for agent in self.agents}
        
        # Calculate modularity for different community assignments
        best_modularity = self._calculate_modularity(communities)
        improved = True
        
        while improved:
            improved = False
            
            for agent in self.agents:
                best_community = None
                best_gain = 0.0
                
                # Try moving agent to each neighbor's community
                for neighbor in self.adjacency_list[agent]:
                    neighbor_community = None
                    for comm_id, members in communities.items():
                        if neighbor in members:
                            neighbor_community = comm_id
                            break
                    
                    if neighbor_community and neighbor_community != agent:
                        # Calculate modularity gain
                        gain = self._calculate_modularity_gain(
                            agent, neighbor_community, communities
                        )
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_community = neighbor_community
                
                # Move agent to best community if improvement found
                if best_community and best_gain > 0.001:
                    # Remove from current community
                    current_community = None
                    for comm_id, members in communities.items():
                        if agent in members:
                            current_community = comm_id
                            break
                    
                    if current_community:
                        communities[current_community].remove(agent)
                        if not communities[current_community]:
                            del communities[current_community]
                    
                    # Add to new community
                    communities[best_community].add(agent)
                    improved = True
        
        # Clean up empty communities and renumber
        final_communities = {}
        community_id = 0
        for members in communities.values():
            if members:
                final_communities[f"community_{community_id}"] = members
                community_id += 1
        
        self.communities = final_communities
        return final_communities
    
    def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality metrics for all agents"""
        if self._is_cache_valid() and self._centrality_cache:
            return self._centrality_cache
        
        metrics = {}
        
        for agent in self.agents:
            metrics[agent] = {
                'degree_centrality': self._calculate_degree_centrality(agent),
                'betweenness_centrality': self._calculate_betweenness_centrality(agent),
                'closeness_centrality': self._calculate_closeness_centrality(agent),
                'eigenvector_centrality': self._calculate_eigenvector_centrality(agent)
            }
        
        self._centrality_cache = metrics
        self._cache_timestamp = time.time()
        return metrics
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        if not self.agents:
            return {}
        
        stats = {
            'total_agents': len(self.agents),
            'total_relationships': len(self.relationships),
            'average_degree': sum(len(neighbors) for neighbors in self.adjacency_list.values()) / len(self.agents),
            'network_density': len(self.relationships) / (len(self.agents) * (len(self.agents) - 1) / 2),
            'relationship_type_distribution': self._get_relationship_type_distribution(),
            'average_trust': self._get_average_trust(),
            'average_relationship_strength': self._get_average_relationship_strength(),
            'communities': len(self.communities),
            'largest_community_size': max(len(members) for members in self.communities.values()) if self.communities else 0
        }
        
        return stats
    
    def export_network_data(self) -> Dict[str, Any]:
        """Export network data for visualization or analysis"""
        nodes = []
        edges = []
        
        # Export nodes
        centrality_metrics = self.calculate_centrality_metrics()
        for agent in self.agents:
            node_data = {
                'id': agent,
                'degree': len(self.adjacency_list[agent]),
                'metrics': centrality_metrics.get(agent, {})
            }
            nodes.append(node_data)
        
        # Export edges
        for (a, b), edge in self.relationships.items():
            edge_data = {
                'source': a,
                'target': b,
                'type': edge.relationship_type.value,
                'strength': edge.strength,
                'trust': edge.trust_level,
                'reciprocal': edge.reciprocal,
                'interactions': edge.interaction_count,
                'emotional_valence': edge.emotional_valence
            }
            edges.append(edge_data)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'statistics': self.get_network_statistics(),
            'communities': {k: list(v) for k, v in self.communities.items()}
        }
    
    # Private helper methods
    
    def _calculate_trust_change(self, interaction_type: str, emotional_impact: float) -> float:
        """Calculate how trust changes based on interaction"""
        base_change = 0.0
        
        if interaction_type in ['cooperation', 'help', 'support']:
            base_change = 0.05
        elif interaction_type in ['betrayal', 'harm', 'deception']:
            base_change = -0.1
        elif interaction_type in ['neutral', 'casual']:
            base_change = 0.01
        
        # Emotional impact amplifies trust change
        multiplier = 1.0 + abs(emotional_impact) * 0.5
        if emotional_impact < 0:
            multiplier *= -1
        
        return base_change * multiplier
    
    def _calculate_strength_change(self, interaction_type: str, emotional_impact: float) -> float:
        """Calculate how relationship strength changes based on interaction"""
        base_change = 0.0
        
        if interaction_type in ['meaningful_conversation', 'shared_activity', 'cooperation']:
            base_change = 0.02
        elif interaction_type in ['conflict', 'argument', 'betrayal']:
            base_change = -0.03
        elif interaction_type in ['casual', 'neutral']:
            base_change = 0.005
        
        # Positive emotional impact strengthens relationships
        if emotional_impact > 0:
            base_change += emotional_impact * 0.01
        else:
            base_change += emotional_impact * 0.02  # Negative emotions have stronger impact
        
        return base_change
    
    def _calculate_influence_transmission(
        self, 
        edge: RelationshipEdge, 
        influence_type: InfluenceType,
        current_strength: float
    ) -> float:
        """Calculate how much influence transmits across an edge"""
        # Base transmission based on relationship strength and trust
        base_transmission = edge.strength * (1.0 + edge.trust_level) / 2.0
        
        # Influence type affects transmission
        type_multiplier = 1.0
        if influence_type == InfluenceType.TRUST:
            type_multiplier = 1.5 if edge.relationship_type in [RelationshipType.FRIENDSHIP, RelationshipType.ALLIANCE] else 0.7
        elif influence_type == InfluenceType.EMOTION:
            type_multiplier = 1.3 if edge.emotional_valence > 0 else 0.8
        elif influence_type == InfluenceType.MEME:
            type_multiplier = 1.2 if edge.relationship_type == RelationshipType.FRIENDSHIP else 1.0
        
        return current_strength * base_transmission * type_multiplier
    
    def _calculate_modularity(self, communities: Dict[str, Set[str]]) -> float:
        """Calculate modularity of current community assignment"""
        if not communities:
            return 0.0
        
        total_edges = len(self.relationships)
        if total_edges == 0:
            return 0.0
        
        modularity = 0.0
        
        for members in communities.values():
            for agent_a in members:
                for agent_b in members:
                    if agent_a != agent_b:
                        # Check if edge exists
                        edge_exists = self.get_relationship(agent_a, agent_b) is not None
                        
                        # Expected probability of edge
                        degree_a = len(self.adjacency_list[agent_a])
                        degree_b = len(self.adjacency_list[agent_b])
                        expected = (degree_a * degree_b) / (2 * total_edges)
                        
                        if edge_exists:
                            modularity += 1 - expected
                        else:
                            modularity -= expected
        
        return modularity / (2 * total_edges)
    
    def _calculate_modularity_gain(
        self, 
        agent: str, 
        target_community: str, 
        communities: Dict[str, Set[str]]
    ) -> float:
        """Calculate modularity gain from moving agent to target community"""
        # Simplified modularity gain calculation
        current_connections = 0
        target_connections = 0
        
        for neighbor in self.adjacency_list[agent]:
            # Check current community connections
            for comm_id, members in communities.items():
                if agent in members and neighbor in members:
                    current_connections += 1
                    break
            
            # Check target community connections
            if neighbor in communities.get(target_community, set()):
                target_connections += 1
        
        return target_connections - current_connections
    
    def _calculate_degree_centrality(self, agent: str) -> float:
        """Calculate degree centrality for an agent"""
        if len(self.agents) <= 1:
            return 0.0
        return len(self.adjacency_list[agent]) / (len(self.agents) - 1)
    
    def _calculate_betweenness_centrality(self, agent: str) -> float:
        """Calculate betweenness centrality for an agent"""
        if len(self.agents) <= 2:
            return 0.0
        
        betweenness = 0.0
        total_pairs = 0
        
        # For efficiency, sample pairs instead of computing all pairs
        agent_list = list(self.agents)
        for i, source in enumerate(agent_list):
            for target in agent_list[i+1:]:
                if source != agent and target != agent:
                    total_pairs += 1
                    # Simple shortest path check (could be optimized)
                    paths_through_agent = self._count_shortest_paths_through(source, target, agent)
                    total_paths = max(1, self._count_shortest_paths(source, target))
                    betweenness += paths_through_agent / total_paths
        
        if total_pairs > 0:
            betweenness /= total_pairs
        
        return betweenness
    
    def _calculate_closeness_centrality(self, agent: str) -> float:
        """Calculate closeness centrality for an agent"""
        if len(self.agents) <= 1:
            return 0.0
        
        total_distance = 0
        reachable_agents = 0
        
        for other_agent in self.agents:
            if other_agent != agent:
                distance = self._shortest_path_length(agent, other_agent)
                if distance > 0:
                    total_distance += distance
                    reachable_agents += 1
        
        if reachable_agents == 0:
            return 0.0
        
        return reachable_agents / total_distance
    
    def _calculate_eigenvector_centrality(self, agent: str) -> float:
        """Calculate eigenvector centrality for an agent (simplified)"""
        # Simplified version - sum of neighbor centralities
        if not self.adjacency_list[agent]:
            return 0.0
        
        neighbor_centralities = 0.0
        for neighbor in self.adjacency_list[agent]:
            neighbor_centralities += len(self.adjacency_list[neighbor])
        
        max_possible = len(self.agents) * (len(self.agents) - 1)
        return neighbor_centralities / max(max_possible, 1)
    
    def _count_shortest_paths_through(self, source: str, target: str, through: str) -> int:
        """Count shortest paths from source to target that go through a specific agent"""
        # Simplified implementation
        if source == target or source == through or target == through:
            return 0
        
        # Check if there's a path source -> through -> target
        path1_length = self._shortest_path_length(source, through)
        path2_length = self._shortest_path_length(through, target)
        direct_length = self._shortest_path_length(source, target)
        
        if path1_length > 0 and path2_length > 0 and (path1_length + path2_length) == direct_length:
            return 1
        
        return 0
    
    def _count_shortest_paths(self, source: str, target: str) -> int:
        """Count total shortest paths between two agents"""
        if source == target:
            return 1
        
        # Simplified - assume there's at most one shortest path
        return 1 if self._shortest_path_length(source, target) > 0 else 0
    
    def _shortest_path_length(self, source: str, target: str) -> int:
        """Calculate shortest path length between two agents"""
        if source == target:
            return 0
        
        if source not in self.agents or target not in self.agents:
            return -1
        
        # BFS for shortest path
        queue = deque([(source, 0)])
        visited = {source}
        
        while queue:
            current, distance = queue.popleft()
            
            for neighbor in self.adjacency_list[current]:
                if neighbor == target:
                    return distance + 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        return -1  # No path found
    
    def _get_relationship_type_distribution(self) -> Dict[str, int]:
        """Get distribution of relationship types"""
        distribution = defaultdict(int)
        for edge in self.relationships.values():
            distribution[edge.relationship_type.value] += 1
        return dict(distribution)
    
    def _get_average_trust(self) -> float:
        """Calculate average trust level across all relationships"""
        if not self.relationships:
            return 0.0
        
        total_trust = sum(edge.trust_level for edge in self.relationships.values())
        return total_trust / len(self.relationships)
    
    def _get_average_relationship_strength(self) -> float:
        """Calculate average relationship strength"""
        if not self.relationships:
            return 0.0
        
        total_strength = sum(edge.strength for edge in self.relationships.values())
        return total_strength / len(self.relationships)
    
    def _invalidate_cache(self):
        """Invalidate cached values with size management"""
        # Selective cache invalidation to preserve frequently used entries
        if len(self._trust_cache) > self._max_cache_size:
            # Keep only the most recent 50% of entries
            items = list(self._trust_cache.items())
            keep_count = self._max_cache_size // 2
            self._trust_cache = dict(items[-keep_count:])
        
        if len(self._centrality_cache) > self._max_cache_size:
            self._centrality_cache.clear()
        
        self._cache_timestamp = time.time()
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        return (time.time() - self._cache_timestamp) < self._cache_ttl