"""
File: memory_association.py
Description: Cross-Memory Association module for enhanced PIANO architecture.
Handles episodic-semantic memory linking, cultural memory influence on personal memory,
and memory-based learning patterns.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
import time
import math
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum

from .langgraph_base_module import LangGraphBaseModule, ModuleExecutionConfig, ExecutionTimeScale, ModulePriority
from ..enhanced_agent_state import EnhancedAgentState, EnhancedAgentStateManager
from ..memory_structures.episodic_memory import Episode, EpisodeType, CausalRelationType
from ..memory_structures.semantic_memory import SemanticConcept, ConceptType, SemanticRelationType


class AssociationType(Enum):
    """Types of memory associations."""
    EPISODIC_SEMANTIC = "episodic_semantic"          # Episode → Concept
    SEMANTIC_EPISODIC = "semantic_episodic"          # Concept → Episode
    CULTURAL_PERSONAL = "cultural_personal"          # Cultural meme → Personal memory
    TEMPORAL_EPISODIC = "temporal_episodic"          # Temporal memory → Episode
    CROSS_EPISODIC = "cross_episodic"                # Episode → Episode
    CONCEPT_CHAIN = "concept_chain"                  # Concept → Concept chain
    LEARNING_PATTERN = "learning_pattern"            # Pattern-based association


class AssociationStrength(Enum):
    """Strength levels for associations."""
    WEAK = 0.2
    MODERATE = 0.5
    STRONG = 0.8
    VERY_STRONG = 0.95


@dataclass
class MemoryAssociation:
    """Represents an association between memories."""
    association_id: str
    association_type: AssociationType
    source_memory_id: str
    source_memory_system: str
    target_memory_id: str
    target_memory_system: str
    strength: float
    confidence: float
    created_at: datetime
    last_reinforced: datetime
    reinforcement_count: int
    evidence: List[str]
    metadata: Dict[str, Any]


@dataclass
class LearningPattern:
    """Represents a learned pattern from memory associations."""
    pattern_id: str
    pattern_type: str
    pattern_description: str
    associated_memories: List[str]
    confidence: float
    usage_count: int
    last_used: datetime
    created_at: datetime


class MemoryAssociationModule(LangGraphBaseModule):
    """
    Cross-memory association module that creates and manages links between
    different memory systems and identifies learning patterns.
    """
    
    def __init__(self, state_manager: Optional[EnhancedAgentStateManager] = None):
        """
        Initialize Memory Association Module.
        
        Args:
            state_manager: Enhanced agent state manager
        """
        config = ModuleExecutionConfig(
            time_scale=ExecutionTimeScale.MEDIUM,
            priority=ModulePriority.MEDIUM,
            can_run_parallel=True,
            requires_completion=False,
            max_execution_time=2.0
        )
        
        super().__init__("memory_association", config, state_manager)
        
        # Association settings
        self.min_association_strength = 0.1
        self.max_associations_per_memory = 10
        self.association_decay_rate = 0.05
        self.learning_confidence_threshold = 0.7
        
        # Storage for associations and patterns
        self.associations: Dict[str, MemoryAssociation] = {}
        self.learning_patterns: Dict[str, LearningPattern] = {}
        
        # Association indices for fast lookup
        self.source_index: Dict[str, List[str]] = defaultdict(list)  # source_id -> association_ids
        self.target_index: Dict[str, List[str]] = defaultdict(list)  # target_id -> association_ids
        self.type_index: Dict[AssociationType, List[str]] = defaultdict(list)  # type -> association_ids
        
        # Performance tracking
        self.association_stats = {
            "total_associations": 0,
            "associations_by_type": {},
            "total_patterns": 0,
            "avg_association_strength": 0.0,
            "reinforcements_today": 0,
            "last_pattern_discovery": None
        }
        
        self._next_association_id = 1
        self._next_pattern_id = 1
        
        self.logger = logging.getLogger("MemoryAssociation")
    
    def process_state(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Process agent state for memory association opportunities.
        
        Args:
            state: Current enhanced agent state
        
        Returns:
            Dictionary with association results and state updates
        """
        start_time = time.time()
        
        try:
            # Discover new associations based on recent memory activity
            new_associations = self._discover_associations(state)
            
            # Reinforce existing associations based on current context
            reinforced_associations = self._reinforce_associations(state)
            
            # Identify learning patterns from association networks
            new_patterns = self._identify_learning_patterns()
            
            # Apply cultural influence on personal memories
            cultural_influences = self._apply_cultural_influences(state)
            
            # Update association strengths with decay
            self._update_association_decay()
            
            # Clean up weak associations
            pruned_associations = self._prune_weak_associations()
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "state_changes": {
                    "performance": {
                        **state.get("performance", {}),
                        "coherence_score": self._calculate_coherence_enhancement()
                    }
                },
                "output_data": {
                    "new_associations": len(new_associations),
                    "reinforced_associations": len(reinforced_associations),
                    "new_patterns": len(new_patterns),
                    "cultural_influences": len(cultural_influences),
                    "pruned_associations": pruned_associations
                },
                "performance_metrics": {
                    "processing_time_ms": processing_time,
                    "total_associations": len(self.associations),
                    "association_density": self._calculate_association_density()
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error in memory association processing: {str(e)}")
            return {
                "output_data": {"error": str(e)},
                "performance_metrics": {"processing_time_ms": (time.time() - start_time) * 1000}
            }
    
    def _discover_associations(self, state: EnhancedAgentState) -> List[MemoryAssociation]:
        """
        Discover new associations between memories.
        
        Args:
            state: Current agent state
        
        Returns:
            List of newly discovered associations
        """
        new_associations = []
        
        if not self.state_manager:
            return new_associations
        
        try:
            # Get recent memories from different systems
            recent_working = self.state_manager.circular_buffer.get_recent_memories(10)
            recent_temporal = self.state_manager.temporal_memory.retrieve_recent_memories(hours_back=2, limit=20)
            recent_episodes = list(self.state_manager.episodic_memory.episodes.values())[-10:]
            activated_concepts = self.state_manager.semantic_memory.retrieve_by_activation(threshold=0.4, limit=15)
            
            # Discover episodic-semantic associations
            for episode in recent_episodes:
                for concept in activated_concepts:
                    if self._should_associate_episode_concept(episode, concept):
                        association = self._create_association(
                            AssociationType.EPISODIC_SEMANTIC,
                            episode.episode_id, "episodic",
                            concept.concept_id, "semantic",
                            self._calculate_episode_concept_strength(episode, concept)
                        )
                        new_associations.append(association)
            
            # Discover temporal-episodic associations
            for memory in recent_temporal:
                for episode in recent_episodes:
                    if self._should_associate_temporal_episode(memory, episode):
                        association = self._create_association(
                            AssociationType.TEMPORAL_EPISODIC,
                            memory.get("id", "unknown"), "temporal",
                            episode.episode_id, "episodic",
                            self._calculate_temporal_episode_strength(memory, episode)
                        )
                        new_associations.append(association)
            
            # Discover cross-episodic associations
            for i, episode1 in enumerate(recent_episodes):
                for episode2 in recent_episodes[i+1:]:
                    if self._should_associate_episodes(episode1, episode2):
                        association = self._create_association(
                            AssociationType.CROSS_EPISODIC,
                            episode1.episode_id, "episodic",
                            episode2.episode_id, "episodic",
                            self._calculate_episode_episode_strength(episode1, episode2)
                        )
                        new_associations.append(association)
            
            # Discover concept chains
            for concept1 in activated_concepts:
                for concept2 in activated_concepts:
                    if (concept1.concept_id != concept2.concept_id and
                        self._should_chain_concepts(concept1, concept2)):
                        association = self._create_association(
                            AssociationType.CONCEPT_CHAIN,
                            concept1.concept_id, "semantic",
                            concept2.concept_id, "semantic",
                            self._calculate_concept_chain_strength(concept1, concept2)
                        )
                        new_associations.append(association)
        
        except Exception as e:
            self.logger.error(f"Error discovering associations: {str(e)}")
        
        return new_associations
    
    def _should_associate_episode_concept(self, episode: Episode, concept: SemanticConcept) -> bool:
        """Determine if episode and concept should be associated."""
        # Check if concept name appears in episode content
        episode_text = f"{episode.title} {episode.summary}".lower()
        concept_text = f"{concept.name} {concept.description}".lower()
        
        # Simple keyword matching
        concept_words = set(concept_text.split())
        episode_words = set(episode_text.split())
        
        overlap = len(concept_words & episode_words)
        return overlap > 0 and concept.activation_level > 0.3
    
    def _should_associate_temporal_episode(self, memory: Dict[str, Any], episode: Episode) -> bool:
        """Determine if temporal memory and episode should be associated."""
        # Check temporal proximity
        memory_time = memory.get("timestamp", datetime.now())
        time_diff = abs((episode.end_time - memory_time).total_seconds())
        
        # Within 1 hour and content similarity
        if time_diff < 3600:
            memory_content = memory.get("content", "").lower()
            episode_content = f"{episode.title} {episode.summary}".lower()
            
            memory_words = set(memory_content.split())
            episode_words = set(episode_content.split())
            
            if len(memory_words) > 0 and len(episode_words) > 0:
                similarity = len(memory_words & episode_words) / len(memory_words | episode_words)
                return similarity > 0.2
        
        return False
    
    def _should_associate_episodes(self, episode1: Episode, episode2: Episode) -> bool:
        """Determine if two episodes should be associated."""
        # Check participant overlap
        participant_overlap = len(episode1.participants & episode2.participants)
        
        # Check temporal proximity (within 24 hours)
        time_diff = abs((episode1.end_time - episode2.end_time).total_seconds())
        
        # Check location similarity
        location_match = episode1.location == episode2.location
        
        return (participant_overlap > 0 and time_diff < 86400) or location_match
    
    def _should_chain_concepts(self, concept1: SemanticConcept, concept2: SemanticConcept) -> bool:
        """Determine if concepts should be chained."""
        # Check if they're both highly activated
        both_activated = concept1.activation_level > 0.5 and concept2.activation_level > 0.5
        
        # Check semantic relatedness (simplified)
        related_types = {
            ConceptType.PERSON: [ConceptType.RELATIONSHIP, ConceptType.EMOTION],
            ConceptType.PLACE: [ConceptType.ACTIVITY, ConceptType.OBJECT],
            ConceptType.ACTION: [ConceptType.GOAL, ConceptType.EMOTION],
            ConceptType.EMOTION: [ConceptType.PERSON, ConceptType.TRAIT]
        }
        
        type_related = (concept2.concept_type in related_types.get(concept1.concept_type, []) or
                       concept1.concept_type in related_types.get(concept2.concept_type, []))
        
        return both_activated and type_related
    
    def _create_association(self, association_type: AssociationType, 
                           source_id: str, source_system: str,
                           target_id: str, target_system: str,
                           strength: float) -> MemoryAssociation:
        """Create a new memory association."""
        association_id = f"assoc_{self._next_association_id}"
        self._next_association_id += 1
        
        association = MemoryAssociation(
            association_id=association_id,
            association_type=association_type,
            source_memory_id=source_id,
            source_memory_system=source_system,
            target_memory_id=target_id,
            target_memory_system=target_system,
            strength=max(self.min_association_strength, min(strength, 1.0)),
            confidence=0.8,  # Initial confidence
            created_at=datetime.now(),
            last_reinforced=datetime.now(),
            reinforcement_count=1,
            evidence=[],
            metadata={}
        )
        
        # Store association and update indices
        self.associations[association_id] = association
        self.source_index[source_id].append(association_id)
        self.target_index[target_id].append(association_id)
        self.type_index[association_type].append(association_id)
        
        self.association_stats["total_associations"] += 1
        type_name = association_type.value
        self.association_stats["associations_by_type"][type_name] = (
            self.association_stats["associations_by_type"].get(type_name, 0) + 1
        )
        
        return association
    
    def _calculate_episode_concept_strength(self, episode: Episode, concept: SemanticConcept) -> float:
        """Calculate association strength between episode and concept."""
        strength_factors = []
        
        # Content similarity
        episode_text = f"{episode.title} {episode.summary}".lower()
        concept_text = f"{concept.name} {concept.description}".lower()
        
        episode_words = set(episode_text.split())
        concept_words = set(concept_text.split())
        
        if episode_words and concept_words:
            similarity = len(episode_words & concept_words) / len(episode_words | concept_words)
            strength_factors.append(similarity)
        
        # Concept activation level
        strength_factors.append(concept.activation_level)
        
        # Episode importance
        strength_factors.append(episode.importance)
        
        return sum(strength_factors) / len(strength_factors) if strength_factors else 0.5
    
    def _calculate_temporal_episode_strength(self, memory: Dict[str, Any], episode: Episode) -> float:
        """Calculate association strength between temporal memory and episode."""
        strength_factors = []
        
        # Temporal proximity (closer = stronger)
        memory_time = memory.get("timestamp", datetime.now())
        time_diff_hours = abs((episode.end_time - memory_time).total_seconds()) / 3600
        temporal_strength = max(0, 1.0 - (time_diff_hours / 24))  # Decay over 24 hours
        strength_factors.append(temporal_strength)
        
        # Content relevance
        memory_content = memory.get("content", "").lower()
        episode_content = f"{episode.title} {episode.summary}".lower()
        
        memory_words = set(memory_content.split())
        episode_words = set(episode_content.split())
        
        if memory_words and episode_words:
            similarity = len(memory_words & episode_words) / len(memory_words | episode_words)
            strength_factors.append(similarity)
        
        # Memory importance
        strength_factors.append(memory.get("importance", 0.5))
        
        return sum(strength_factors) / len(strength_factors) if strength_factors else 0.3
    
    def _calculate_episode_episode_strength(self, episode1: Episode, episode2: Episode) -> float:
        """Calculate association strength between two episodes."""
        strength_factors = []
        
        # Participant overlap
        if episode1.participants or episode2.participants:
            overlap = len(episode1.participants & episode2.participants)
            total_unique = len(episode1.participants | episode2.participants)
            if total_unique > 0:
                participant_strength = overlap / total_unique
                strength_factors.append(participant_strength)
        
        # Location similarity
        if episode1.location and episode2.location:
            location_strength = 1.0 if episode1.location == episode2.location else 0.2
            strength_factors.append(location_strength)
        
        # Temporal proximity
        time_diff_hours = abs((episode1.end_time - episode2.end_time).total_seconds()) / 3600
        temporal_strength = max(0, 1.0 - (time_diff_hours / 168))  # Decay over 1 week
        strength_factors.append(temporal_strength)
        
        # Episode type similarity
        type_strength = 1.0 if episode1.episode_type == episode2.episode_type else 0.5
        strength_factors.append(type_strength)
        
        return sum(strength_factors) / len(strength_factors) if strength_factors else 0.4
    
    def _calculate_concept_chain_strength(self, concept1: SemanticConcept, concept2: SemanticConcept) -> float:
        """Calculate association strength for concept chaining."""
        strength_factors = []
        
        # Activation levels
        activation_strength = (concept1.activation_level + concept2.activation_level) / 2
        strength_factors.append(activation_strength)
        
        # Importance combination
        importance_strength = (concept1.importance + concept2.importance) / 2
        strength_factors.append(importance_strength)
        
        # Type compatibility
        related_types = {
            ConceptType.PERSON: [ConceptType.RELATIONSHIP, ConceptType.EMOTION],
            ConceptType.PLACE: [ConceptType.ACTIVITY, ConceptType.OBJECT],
            ConceptType.ACTION: [ConceptType.GOAL, ConceptType.EMOTION],
            ConceptType.EMOTION: [ConceptType.PERSON, ConceptType.TRAIT]
        }
        
        type_strength = 0.8 if (concept2.concept_type in related_types.get(concept1.concept_type, []) or
                               concept1.concept_type in related_types.get(concept2.concept_type, [])) else 0.3
        strength_factors.append(type_strength)
        
        return sum(strength_factors) / len(strength_factors) if strength_factors else 0.4
    
    def _reinforce_associations(self, state: EnhancedAgentState) -> List[str]:
        """
        Reinforce existing associations based on current context.
        
        Args:
            state: Current agent state
        
        Returns:
            List of reinforced association IDs
        """
        reinforced = []
        
        if not self.state_manager:
            return reinforced
        
        try:
            # Get current context
            current_activity = state.get("current_activity", "")
            conversation_partners = state.get("conversation_partners", set())
            recent_interactions = state.get("recent_interactions", [])
            
            # Find associations that match current context
            for association in self.associations.values():
                should_reinforce = False
                
                # Check if association is relevant to current activity
                if association.association_type == AssociationType.EPISODIC_SEMANTIC:
                    # Get the concept and check if it's related to current activity
                    if association.target_memory_system == "semantic":
                        concept_id = association.target_memory_id
                        if concept_id in self.state_manager.semantic_memory.concepts:
                            concept = self.state_manager.semantic_memory.concepts[concept_id]
                            if concept.name.lower() in current_activity.lower():
                                should_reinforce = True
                
                # Check if association involves current conversation partners
                if conversation_partners and association.association_type == AssociationType.CROSS_EPISODIC:
                    # Check if episodes involve current partners
                    if association.source_memory_system == "episodic":
                        episode_id = association.source_memory_id
                        if episode_id in self.state_manager.episodic_memory.episodes:
                            episode = self.state_manager.episodic_memory.episodes[episode_id]
                            if episode.participants & conversation_partners:
                                should_reinforce = True
                
                if should_reinforce and association.strength < 0.9:  # Don't over-reinforce
                    # Increase strength slightly
                    association.strength = min(association.strength + 0.1, 1.0)
                    association.last_reinforced = datetime.now()
                    association.reinforcement_count += 1
                    reinforced.append(association.association_id)
                    
                    self.association_stats["reinforcements_today"] += 1
        
        except Exception as e:
            self.logger.error(f"Error reinforcing associations: {str(e)}")
        
        return reinforced
    
    def _identify_learning_patterns(self) -> List[LearningPattern]:
        """
        Identify learning patterns from association networks.
        
        Returns:
            List of newly identified learning patterns
        """
        new_patterns = []
        
        try:
            # Pattern 1: Frequently reinforced association chains
            strong_associations = [
                assoc for assoc in self.associations.values()
                if assoc.strength > 0.7 and assoc.reinforcement_count > 3
            ]
            
            if len(strong_associations) > 5:
                pattern = LearningPattern(
                    pattern_id=f"pattern_{self._next_pattern_id}",
                    pattern_type="reinforcement_chain",
                    pattern_description=f"Strong association network with {len(strong_associations)} connections",
                    associated_memories=[assoc.source_memory_id for assoc in strong_associations],
                    confidence=0.8,
                    usage_count=0,
                    last_used=datetime.now(),
                    created_at=datetime.now()
                )
                
                self.learning_patterns[pattern.pattern_id] = pattern
                new_patterns.append(pattern)
                self._next_pattern_id += 1
            
            # Pattern 2: Episodic-semantic clustering
            episodic_semantic_associations = [
                assoc for assoc in self.associations.values()
                if assoc.association_type == AssociationType.EPISODIC_SEMANTIC
            ]
            
            # Group by semantic concept
            concept_clusters = defaultdict(list)
            for assoc in episodic_semantic_associations:
                concept_clusters[assoc.target_memory_id].append(assoc)
            
            for concept_id, assocs in concept_clusters.items():
                if len(assocs) > 3:  # Concept associated with multiple episodes
                    pattern = LearningPattern(
                        pattern_id=f"pattern_{self._next_pattern_id}",
                        pattern_type="concept_clustering",
                        pattern_description=f"Concept {concept_id} associated with {len(assocs)} episodes",
                        associated_memories=[assoc.source_memory_id for assoc in assocs],
                        confidence=0.7,
                        usage_count=0,
                        last_used=datetime.now(),
                        created_at=datetime.now()
                    )
                    
                    self.learning_patterns[pattern.pattern_id] = pattern
                    new_patterns.append(pattern)
                    self._next_pattern_id += 1
            
            # Update statistics
            if new_patterns:
                self.association_stats["total_patterns"] = len(self.learning_patterns)
                self.association_stats["last_pattern_discovery"] = datetime.now().isoformat()
        
        except Exception as e:
            self.logger.error(f"Error identifying learning patterns: {str(e)}")
        
        return new_patterns
    
    def _apply_cultural_influences(self, state: EnhancedAgentState) -> List[str]:
        """
        Apply cultural meme influence on personal memories.
        
        Args:
            state: Current agent state
        
        Returns:
            List of influenced memory IDs
        """
        influenced_memories = []
        
        if not self.state_manager:
            return influenced_memories
        
        try:
            # Get cultural data from state
            cultural_data = state.get("cultural")
            if not cultural_data:
                return influenced_memories
            
            strong_memes = {
                meme_id: strength for meme_id, strength in cultural_data.meme_influence.items()
                if strength > 0.6
            }
            
            # Influence episodic memories
            for episode in self.state_manager.episodic_memory.episodes.values():
                for meme_id, strength in strong_memes.items():
                    # Check if meme is relevant to episode
                    if self._meme_relevant_to_episode(meme_id, episode):
                        # Create cultural influence association
                        association = self._create_association(
                            AssociationType.CULTURAL_PERSONAL,
                            meme_id, "cultural",
                            episode.episode_id, "episodic",
                            strength * 0.7  # Cultural influence is moderate
                        )
                        influenced_memories.append(episode.episode_id)
            
            # Influence semantic concepts
            for concept in self.state_manager.semantic_memory.concepts.values():
                for meme_id, strength in strong_memes.items():
                    if self._meme_relevant_to_concept(meme_id, concept):
                        # Boost concept activation based on cultural influence
                        boost = strength * 0.3
                        concept.activation_level = min(concept.activation_level + boost, 1.0)
                        influenced_memories.append(concept.concept_id)
        
        except Exception as e:
            self.logger.error(f"Error applying cultural influences: {str(e)}")
        
        return influenced_memories
    
    def _meme_relevant_to_episode(self, meme_id: str, episode: Episode) -> bool:
        """Check if cultural meme is relevant to an episode."""
        # Simplified relevance check - in practice would be more sophisticated
        meme_keywords = meme_id.lower().split('_')
        episode_text = f"{episode.title} {episode.summary}".lower()
        
        return any(keyword in episode_text for keyword in meme_keywords)
    
    def _meme_relevant_to_concept(self, meme_id: str, concept: SemanticConcept) -> bool:
        """Check if cultural meme is relevant to a concept."""
        meme_keywords = meme_id.lower().split('_')
        concept_text = f"{concept.name} {concept.description}".lower()
        
        return any(keyword in concept_text for keyword in meme_keywords)
    
    def _update_association_decay(self) -> None:
        """Apply decay to association strengths over time."""
        current_time = datetime.now()
        
        for association in self.associations.values():
            # Calculate days since last reinforcement
            days_since_reinforcement = (current_time - association.last_reinforced).days
            
            if days_since_reinforcement > 0:
                # Apply exponential decay
                decay_factor = math.exp(-self.association_decay_rate * days_since_reinforcement)
                association.strength *= decay_factor
                
                # Ensure minimum strength
                association.strength = max(association.strength, self.min_association_strength)
    
    def _prune_weak_associations(self) -> int:
        """Remove associations that have become too weak."""
        weak_associations = []
        
        for assoc_id, association in self.associations.items():
            if association.strength < self.min_association_strength:
                weak_associations.append(assoc_id)
        
        # Remove weak associations
        for assoc_id in weak_associations:
            association = self.associations[assoc_id]
            
            # Remove from indices
            self.source_index[association.source_memory_id].remove(assoc_id)
            self.target_index[association.target_memory_id].remove(assoc_id)
            self.type_index[association.association_type].remove(assoc_id)
            
            # Remove association
            del self.associations[assoc_id]
            
            # Update statistics
            self.association_stats["total_associations"] -= 1
            type_name = association.association_type.value
            if type_name in self.association_stats["associations_by_type"]:
                self.association_stats["associations_by_type"][type_name] -= 1
        
        return len(weak_associations)
    
    def _calculate_coherence_enhancement(self) -> float:
        """Calculate how much associations enhance memory coherence."""
        if not self.associations:
            return 0.5  # Base coherence
        
        # Calculate average association strength
        avg_strength = sum(assoc.strength for assoc in self.associations.values()) / len(self.associations)
        
        # Factor in number of associations (more = better coherence up to a point)
        association_factor = min(len(self.associations) / 50, 1.0)  # Normalize to 50 associations
        
        # Factor in learning patterns (patterns indicate coherent structure)
        pattern_factor = min(len(self.learning_patterns) / 10, 0.2)  # Up to 20% boost
        
        coherence_score = 0.5 + (avg_strength * 0.3) + (association_factor * 0.2) + pattern_factor
        
        return min(coherence_score, 1.0)
    
    def _calculate_association_density(self) -> float:
        """Calculate the density of the association network."""
        if not self.state_manager:
            return 0.0
        
        # Count total memories across all systems
        total_memories = (
            len(self.state_manager.circular_buffer.buffer) +
            sum(len(memories) for memories in self.state_manager.temporal_memory.memories.values()) +
            len(self.state_manager.episodic_memory.episodes) +
            len(self.state_manager.semantic_memory.concepts)
        )
        
        if total_memories == 0:
            return 0.0
        
        # Density = associations / possible associations
        max_possible_associations = total_memories * (total_memories - 1) / 2
        current_associations = len(self.associations)
        
        return current_associations / max_possible_associations if max_possible_associations > 0 else 0.0
    
    def get_associations_for_memory(self, memory_id: str) -> List[MemoryAssociation]:
        """
        Get all associations for a specific memory.
        
        Args:
            memory_id: ID of the memory
        
        Returns:
            List of associations involving the memory
        """
        associations = []
        
        # Get associations where this memory is the source
        for assoc_id in self.source_index.get(memory_id, []):
            if assoc_id in self.associations:
                associations.append(self.associations[assoc_id])
        
        # Get associations where this memory is the target
        for assoc_id in self.target_index.get(memory_id, []):
            if assoc_id in self.associations:
                associations.append(self.associations[assoc_id])
        
        return associations
    
    def get_association_summary(self) -> Dict[str, Any]:
        """
        Get summary of association activity and patterns.
        
        Returns:
            Association summary with statistics
        """
        # Update average association strength
        if self.associations:
            avg_strength = sum(assoc.strength for assoc in self.associations.values()) / len(self.associations)
            self.association_stats["avg_association_strength"] = avg_strength
        
        return {
            "module_name": self.module_name,
            "total_associations": len(self.associations),
            "total_patterns": len(self.learning_patterns),
            "association_stats": self.association_stats.copy(),
            "association_density": self._calculate_association_density(),
            "coherence_enhancement": self._calculate_coherence_enhancement(),
            "associations_by_type": {
                assoc_type.value: len(assoc_ids)
                for assoc_type, assoc_ids in self.type_index.items()
            },
            "recent_patterns": [
                {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "created_at": pattern.created_at.isoformat()
                }
                for pattern in list(self.learning_patterns.values())[-5:]  # Last 5 patterns
            ]
        }


# Example usage and testing
if __name__ == "__main__":
    # Example of memory association module usage
    from ..enhanced_agent_state import create_enhanced_agent_state
    
    # Create state manager with test data
    state_manager = create_enhanced_agent_state(
        "test_agent", "Test Agent", {"confidence": 0.8}
    )
    
    # Add test memories and episodes
    state_manager.add_memory("Had coffee with Maria", "conversation", 0.8)
    state_manager.add_memory("Maria likes hiking", "observation", 0.9)
    state_manager.add_memory("Planning outdoor date", "plan", 0.7)
    
    # Create some semantic concepts
    maria_concept = state_manager.semantic_memory.add_concept(
        "Maria", ConceptType.PERSON, "Fellow contestant", 0.8
    )
    hiking_concept = state_manager.semantic_memory.add_concept(
        "Hiking", ConceptType.ACTIVITY, "Outdoor activity", 0.7
    )
    
    # Activate concepts
    state_manager.semantic_memory.activate_concept("Maria", 0.8)
    state_manager.semantic_memory.activate_concept("Hiking", 0.6)
    
    # Create association module
    association_module = MemoryAssociationModule(state_manager)
    
    print("Testing memory association...")
    
    # Process state to discover associations
    result = association_module(state_manager.state)
    
    print(f"Association result: {result}")
    
    # Get association summary
    summary = association_module.get_association_summary()
    print(f"\nAssociation summary:")
    print(f"- Total associations: {summary['total_associations']}")
    print(f"- Total patterns: {summary['total_patterns']}")
    print(f"- Association density: {summary['association_density']:.4f}")
    print(f"- Coherence enhancement: {summary['coherence_enhancement']:.3f}")
    
    # Show associations by type
    print(f"\nAssociations by type:")
    for assoc_type, count in summary['associations_by_type'].items():
        print(f"- {assoc_type}: {count}")
    
    print("Memory association module example completed!")