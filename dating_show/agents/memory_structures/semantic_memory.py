"""
File: semantic_memory.py
Description: SemanticMemory with associative retrieval and concept-based knowledge storage.
Enhanced PIANO architecture with vector embeddings and semantic relationships.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from .security_utils import SecurityValidator, SecurityError


class ConceptType(Enum):
    """Types of semantic concepts."""
    PERSON = "person"
    PLACE = "place"
    ACTION = "action"
    EMOTION = "emotion"
    TRAIT = "trait"
    GOAL = "goal"
    BELIEF = "belief"
    RELATIONSHIP = "relationship"
    ACTIVITY = "activity"
    OBJECT = "object"


class SemanticRelationType(Enum):
    """Types of semantic relationships between concepts."""
    IS_A = "is_a"  # Isabella is a contestant
    HAS_A = "has_a"  # Isabella has trait confident
    RELATED_TO = "related_to"  # Coffee related to morning routine
    CAUSES = "causes"  # Stress causes anxiety
    ENABLES = "enables"  # Trust enables intimacy
    SIMILAR_TO = "similar_to"  # Maria similar to Isabella
    OPPOSITE_OF = "opposite_of"  # Extrovert opposite of introvert
    LOCATED_AT = "located_at"  # Kitchen located at villa
    USED_FOR = "used_for"  # Gym used for exercise
    KNOWS = "knows"  # Person knows another person


@dataclass
class SemanticConcept:
    """Represents a concept in semantic memory."""
    concept_id: str
    name: str
    concept_type: ConceptType
    description: str
    importance: float
    activation_level: float  # Current activation strength
    base_activation: float  # Base activation level
    created_at: datetime
    last_accessed: datetime
    access_count: int
    attributes: Dict[str, Any]
    embedding: Optional[List[float]] = None  # Vector embedding
    
    def __post_init__(self):
        if self.embedding is None:
            self.embedding = []


@dataclass
class SemanticRelation:
    """Represents a relationship between concepts."""
    relation_id: str
    source_concept_id: str
    target_concept_id: str
    relation_type: SemanticRelationType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    created_at: datetime
    evidence: List[str]
    context: Dict[str, Any]


class SemanticMemory:
    """
    Enhanced SemanticMemory system for concept-based knowledge storage and retrieval.
    Supports associative retrieval, spreading activation, and vector similarity.
    """
    
    def __init__(self, max_concepts: int = 1000, activation_decay: float = 0.1,
                 spreading_factor: float = 0.8, similarity_threshold: float = 0.7):
        """
        Initialize SemanticMemory.
        
        Args:
            max_concepts: Maximum number of concepts to maintain
            activation_decay: Rate of activation decay over time
            spreading_factor: Factor for spreading activation through relationships
            similarity_threshold: Threshold for concept similarity matching
        """
        self.max_concepts = max_concepts
        self.activation_decay = activation_decay
        self.spreading_factor = spreading_factor
        self.similarity_threshold = similarity_threshold
        
        # Core storage
        self.concepts: Dict[str, SemanticConcept] = {}  # concept_id -> concept
        self.relations: Dict[str, SemanticRelation] = {}  # relation_id -> relation
        
        # Indexing structures
        self.name_index: Dict[str, str] = {}  # name -> concept_id
        self.type_index: Dict[ConceptType, Set[str]] = defaultdict(set)  # type -> concept_ids
        self.relation_index: Dict[str, List[str]] = defaultdict(list)  # concept_id -> relation_ids
        
        # Activation network
        self.activation_network: Dict[str, Set[str]] = defaultdict(set)  # concept_id -> connected_concept_ids
        
        # Text processing
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        self._next_concept_id = 1
        self._next_relation_id = 1
    
    def add_concept(self, name: str, concept_type: ConceptType, description: str = "",
                   importance: float = 0.5, attributes: Optional[Dict[str, Any]] = None,
                   embedding: Optional[List[float]] = None) -> str:
        """
        Add a new concept to semantic memory.
        
        Args:
            name: Name of the concept
            concept_type: Type of the concept
            description: Description of the concept
            importance: Importance score
            attributes: Additional attributes
            embedding: Vector embedding (if available)
        
        Returns:
            Concept ID
        """
        # Check if concept already exists
        existing_id = self.name_index.get(name.lower())
        if existing_id and existing_id in self.concepts:
            # Update existing concept
            concept = self.concepts[existing_id]
            concept.description = description or concept.description
            concept.importance = max(concept.importance, importance)
            if attributes:
                concept.attributes.update(attributes)
            if embedding:
                concept.embedding = embedding
            concept.last_accessed = datetime.now()
            concept.access_count += 1
            return existing_id
        
        concept_id = f"concept_{self._next_concept_id}"
        self._next_concept_id += 1
        
        concept = SemanticConcept(
            concept_id=concept_id,
            name=name,
            concept_type=concept_type,
            description=description,
            importance=importance,
            activation_level=importance,
            base_activation=importance,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            attributes=attributes or {},
            embedding=embedding
        )
        
        self.concepts[concept_id] = concept
        self.name_index[name.lower()] = concept_id
        self.type_index[concept_type].add(concept_id)
        
        # Cleanup if necessary
        self._cleanup_old_concepts()
        
        return concept_id
    
    def add_relation(self, source_concept: Union[str, str], target_concept: Union[str, str],
                    relation_type: SemanticRelationType, strength: float = 0.8,
                    confidence: float = 0.8, evidence: Optional[List[str]] = None,
                    context: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a semantic relationship between concepts.
        
        Args:
            source_concept: Source concept name or ID
            target_concept: Target concept name or ID
            relation_type: Type of relationship
            strength: Relationship strength
            confidence: Confidence in the relationship
            evidence: Supporting evidence
            context: Additional context
        
        Returns:
            Relation ID
        """
        # Resolve concept IDs
        source_id = self._resolve_concept_id(source_concept)
        target_id = self._resolve_concept_id(target_concept)
        
        if not source_id or not target_id:
            raise ValueError(f"Could not resolve concept IDs: {source_concept}, {target_concept}")
        
        relation_id = f"relation_{self._next_relation_id}"
        self._next_relation_id += 1
        
        relation = SemanticRelation(
            relation_id=relation_id,
            source_concept_id=source_id,
            target_concept_id=target_id,
            relation_type=relation_type,
            strength=strength,
            confidence=confidence,
            created_at=datetime.now(),
            evidence=evidence or [],
            context=context or {}
        )
        
        self.relations[relation_id] = relation
        self.relation_index[source_id].append(relation_id)
        self.relation_index[target_id].append(relation_id)
        
        # Update activation network
        self.activation_network[source_id].add(target_id)
        self.activation_network[target_id].add(source_id)
        
        return relation_id
    
    def _resolve_concept_id(self, concept: Union[str, str]) -> Optional[str]:
        """
        Resolve a concept name or ID to a concept ID.
        
        Args:
            concept: Concept name or ID
        
        Returns:
            Concept ID or None if not found
        """
        if concept in self.concepts:
            return concept
        
        # Try as name
        return self.name_index.get(concept.lower())
    
    def activate_concept(self, concept: Union[str, str], activation_boost: float = 1.0) -> None:
        """
        Activate a concept and spread activation through the network.
        
        Args:
            concept: Concept name or ID to activate
            activation_boost: Amount of activation to add
        """
        concept_id = self._resolve_concept_id(concept)
        if not concept_id or concept_id not in self.concepts:
            return
        
        concept_obj = self.concepts[concept_id]
        concept_obj.activation_level = min(concept_obj.activation_level + activation_boost, 1.0)
        concept_obj.last_accessed = datetime.now()
        concept_obj.access_count += 1
        
        # Spread activation to connected concepts
        self._spread_activation(concept_id, activation_boost * self.spreading_factor)
    
    def _spread_activation(self, source_concept_id: str, activation_amount: float, 
                          visited: Optional[Set[str]] = None, depth: int = 0, max_depth: int = 2) -> None:
        """
        Spread activation through the semantic network.
        
        Args:
            source_concept_id: Source concept ID
            activation_amount: Amount of activation to spread
            visited: Set of already visited concepts
            depth: Current depth in the network
            max_depth: Maximum depth to spread activation
        """
        if depth >= max_depth or activation_amount < 0.1:
            return
        
        if visited is None:
            visited = set()
        
        if source_concept_id in visited:
            return
        
        visited.add(source_concept_id)
        
        # Find connected concepts through relations
        connected_concepts = []
        for relation_id in self.relation_index.get(source_concept_id, []):
            relation = self.relations[relation_id]
            
            if relation.source_concept_id == source_concept_id:
                target_id = relation.target_concept_id
            else:
                target_id = relation.source_concept_id
            
            if target_id not in visited and target_id in self.concepts:
                # Weight activation by relation strength
                weighted_activation = activation_amount * relation.strength
                connected_concepts.append((target_id, weighted_activation))
        
        # Apply activation to connected concepts
        for concept_id, activation in connected_concepts:
            concept = self.concepts[concept_id]
            concept.activation_level = min(concept.activation_level + activation, 1.0)
            
            # Continue spreading with reduced activation
            self._spread_activation(concept_id, activation * self.spreading_factor, 
                                  visited.copy(), depth + 1, max_depth)
    
    def retrieve_by_activation(self, threshold: float = 0.3, limit: int = 10) -> List[SemanticConcept]:
        """
        Retrieve concepts by activation level.
        
        Args:
            threshold: Minimum activation threshold
            limit: Maximum number of concepts to return
        
        Returns:
            List of activated concepts
        """
        activated_concepts = [
            concept for concept in self.concepts.values()
            if concept.activation_level >= threshold
        ]
        
        # Sort by activation level
        activated_concepts.sort(key=lambda x: x.activation_level, reverse=True)
        
        return activated_concepts[:limit]
    
    def retrieve_by_association(self, query_concepts: List[Union[str, str]], 
                              max_hops: int = 2, limit: int = 10) -> List[Tuple[SemanticConcept, float]]:
        """
        Retrieve concepts associated with query concepts.
        
        Args:
            query_concepts: List of concept names or IDs to query
            max_hops: Maximum number of relationship hops
            limit: Maximum number of results
        
        Returns:
            List of (concept, association_score) tuples
        """
        # Resolve query concept IDs
        query_ids = []
        for query_concept in query_concepts:
            concept_id = self._resolve_concept_id(query_concept)
            if concept_id:
                query_ids.append(concept_id)
        
        if not query_ids:
            return []
        
        # Find associated concepts using breadth-first search
        association_scores = defaultdict(float)
        visited = set(query_ids)
        current_level = set(query_ids)
        
        for hop in range(max_hops):
            next_level = set()
            decay_factor = 0.8 ** hop  # Decay association strength by hop distance
            
            for concept_id in current_level:
                # Find connected concepts
                for relation_id in self.relation_index.get(concept_id, []):
                    relation = self.relations[relation_id]
                    
                    if relation.source_concept_id == concept_id:
                        target_id = relation.target_concept_id
                    else:
                        target_id = relation.source_concept_id
                    
                    if target_id not in visited and target_id in self.concepts:
                        # Calculate association score
                        score = relation.strength * decay_factor
                        association_scores[target_id] += score
                        next_level.add(target_id)
                        visited.add(target_id)
            
            current_level = next_level
            if not current_level:
                break
        
        # Convert to results
        results = []
        for concept_id, score in association_scores.items():
            if concept_id in self.concepts:
                results.append((self.concepts[concept_id], score))
        
        # Sort by association score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:limit]
    
    def retrieve_by_similarity(self, query_text: str, limit: int = 10) -> List[Tuple[SemanticConcept, float]]:
        """
        Retrieve concepts similar to query text using simple text similarity.
        
        Args:
            query_text: Query text
            limit: Maximum number of results
        
        Returns:
            List of (concept, similarity_score) tuples
        """
        query_words = self._extract_keywords(query_text.lower())
        if not query_words:
            return []
        
        concept_scores = []
        
        for concept in self.concepts.values():
            # Calculate text similarity
            concept_text = f"{concept.name} {concept.description}".lower()
            concept_words = self._extract_keywords(concept_text)
            
            similarity = self._calculate_text_similarity(query_words, concept_words)
            
            if similarity >= self.similarity_threshold:
                concept_scores.append((concept, similarity))
        
        # Sort by similarity
        concept_scores.sort(key=lambda x: x[1], reverse=True)
        
        return concept_scores[:limit]
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
        
        Returns:
            Set of keywords
        """
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = {word for word in words if word not in self.stop_words and len(word) > 2}
        return keywords
    
    def _calculate_text_similarity(self, words1: Set[str], words2: Set[str]) -> float:
        """
        Calculate similarity between two sets of words.
        
        Args:
            words1: First set of words
            words2: Second set of words
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def retrieve_by_type(self, concept_type: ConceptType, limit: int = 10) -> List[SemanticConcept]:
        """
        Retrieve concepts by type.
        
        Args:
            concept_type: Type of concepts to retrieve
            limit: Maximum number of results
        
        Returns:
            List of concepts of the specified type
        """
        concept_ids = self.type_index.get(concept_type, set())
        concepts = [self.concepts[cid] for cid in concept_ids if cid in self.concepts]
        
        # Sort by importance and activation
        concepts.sort(key=lambda x: (x.importance, x.activation_level), reverse=True)
        
        return concepts[:limit]
    
    def get_concept_relationships(self, concept: Union[str, str]) -> List[Tuple[SemanticRelation, SemanticConcept]]:
        """
        Get all relationships for a concept.
        
        Args:
            concept: Concept name or ID
        
        Returns:
            List of (relation, related_concept) tuples
        """
        concept_id = self._resolve_concept_id(concept)
        if not concept_id:
            return []
        
        relationships = []
        
        for relation_id in self.relation_index.get(concept_id, []):
            relation = self.relations[relation_id]
            
            # Determine the related concept
            if relation.source_concept_id == concept_id:
                related_concept_id = relation.target_concept_id
            else:
                related_concept_id = relation.source_concept_id
            
            if related_concept_id in self.concepts:
                related_concept = self.concepts[related_concept_id]
                relationships.append((relation, related_concept))
        
        return relationships
    
    def update_activation_decay(self) -> None:
        """Apply activation decay to all concepts."""
        current_time = datetime.now()
        
        for concept in self.concepts.values():
            # Calculate time-based decay
            time_since_access = (current_time - concept.last_accessed).total_seconds() / 3600  # hours
            decay_factor = math.exp(-self.activation_decay * time_since_access)
            
            # Apply decay but maintain base activation
            decayed_activation = concept.base_activation + (concept.activation_level - concept.base_activation) * decay_factor
            concept.activation_level = max(decayed_activation, concept.base_activation * 0.1)  # Minimum 10% of base
    
    def consolidate_concepts(self, similarity_threshold: float = 0.9) -> List[Tuple[str, str]]:
        """
        Consolidate similar concepts to reduce redundancy.
        
        Args:
            similarity_threshold: Threshold for concept consolidation
        
        Returns:
            List of (kept_concept_id, merged_concept_id) tuples
        """
        consolidated_pairs = []
        concepts_to_remove = set()
        
        concept_list = list(self.concepts.values())
        
        for i, concept1 in enumerate(concept_list):
            if concept1.concept_id in concepts_to_remove:
                continue
            
            for concept2 in concept_list[i+1:]:
                if concept2.concept_id in concepts_to_remove:
                    continue
                
                # Check if concepts are similar enough to consolidate
                if (concept1.concept_type == concept2.concept_type and
                    concept1.name.lower() != concept2.name.lower()):
                    
                    # Calculate similarity
                    words1 = self._extract_keywords(f"{concept1.name} {concept1.description}")
                    words2 = self._extract_keywords(f"{concept2.name} {concept2.description}")
                    similarity = self._calculate_text_similarity(words1, words2)
                    
                    if similarity >= similarity_threshold:
                        # Merge concepts - keep the more important one
                        if concept1.importance >= concept2.importance:
                            self._merge_concepts(concept1.concept_id, concept2.concept_id)
                            consolidated_pairs.append((concept1.concept_id, concept2.concept_id))
                            concepts_to_remove.add(concept2.concept_id)
                        else:
                            self._merge_concepts(concept2.concept_id, concept1.concept_id)
                            consolidated_pairs.append((concept2.concept_id, concept1.concept_id))
                            concepts_to_remove.add(concept1.concept_id)
                            break
        
        return consolidated_pairs
    
    def _merge_concepts(self, keep_concept_id: str, merge_concept_id: str) -> None:
        """
        Merge one concept into another.
        
        Args:
            keep_concept_id: ID of concept to keep
            merge_concept_id: ID of concept to merge and remove
        """
        if keep_concept_id not in self.concepts or merge_concept_id not in self.concepts:
            return
        
        keep_concept = self.concepts[keep_concept_id]
        merge_concept = self.concepts[merge_concept_id]
        
        # Merge attributes
        keep_concept.importance = max(keep_concept.importance, merge_concept.importance)
        keep_concept.activation_level = max(keep_concept.activation_level, merge_concept.activation_level)
        keep_concept.access_count += merge_concept.access_count
        keep_concept.attributes.update(merge_concept.attributes)
        
        # Update description
        if merge_concept.description and merge_concept.description not in keep_concept.description:
            keep_concept.description += f" | {merge_concept.description}"
        
        # Transfer relationships
        for relation_id in self.relation_index.get(merge_concept_id, []):
            relation = self.relations[relation_id]
            
            # Update relation to point to kept concept
            if relation.source_concept_id == merge_concept_id:
                relation.source_concept_id = keep_concept_id
            elif relation.target_concept_id == merge_concept_id:
                relation.target_concept_id = keep_concept_id
            
            # Add to kept concept's relations
            if relation_id not in self.relation_index[keep_concept_id]:
                self.relation_index[keep_concept_id].append(relation_id)
        
        # Remove merged concept
        self._remove_concept(merge_concept_id)
    
    def _remove_concept(self, concept_id: str) -> None:
        """
        Remove a concept and clean up references.
        
        Args:
            concept_id: ID of concept to remove
        """
        if concept_id not in self.concepts:
            return
        
        concept = self.concepts[concept_id]
        
        # Remove from indices
        self.name_index.pop(concept.name.lower(), None)
        self.type_index[concept.concept_type].discard(concept_id)
        
        # Remove relations
        relation_ids = self.relation_index.get(concept_id, [])
        for relation_id in relation_ids:
            if relation_id in self.relations:
                del self.relations[relation_id]
        
        del self.relation_index[concept_id]
        
        # Remove from activation network
        for connected_id in self.activation_network.get(concept_id, set()):
            self.activation_network[connected_id].discard(concept_id)
        del self.activation_network[concept_id]
        
        # Remove concept
        del self.concepts[concept_id]
    
    def _cleanup_old_concepts(self) -> None:
        """Remove old concepts if we exceed the maximum limit."""
        if len(self.concepts) <= self.max_concepts:
            return
        
        # Sort concepts by importance and recent activation
        sorted_concepts = sorted(
            self.concepts.values(),
            key=lambda x: (x.importance, x.activation_level, x.access_count),
            reverse=True
        )
        
        # Keep only the most important concepts
        concepts_to_keep = sorted_concepts[:self.max_concepts]
        concepts_to_remove = [c for c in self.concepts.values() if c not in concepts_to_keep]
        
        for concept in concepts_to_remove:
            self._remove_concept(concept.concept_id)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of semantic memory.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.concepts:
            return {
                "total_concepts": 0,
                "total_relations": 0,
                "concept_types": {},
                "relation_types": {},
                "avg_activation": 0.0,
                "highly_activated": 0
            }
        
        concept_type_counts = Counter(c.concept_type for c in self.concepts.values())
        relation_type_counts = Counter(r.relation_type for r in self.relations.values())
        
        total_activation = sum(c.activation_level for c in self.concepts.values())
        avg_activation = total_activation / len(self.concepts)
        highly_activated = sum(1 for c in self.concepts.values() if c.activation_level > 0.7)
        
        return {
            "total_concepts": len(self.concepts),
            "total_relations": len(self.relations),
            "concept_types": {k.value: v for k, v in concept_type_counts.items()},
            "relation_types": {k.value: v for k, v in relation_type_counts.items()},
            "avg_activation": avg_activation,
            "highly_activated": highly_activated,
            "network_density": len(self.relations) / max(len(self.concepts), 1)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        serialized_concepts = {}
        for concept_id, concept in self.concepts.items():
            serialized_concepts[concept_id] = {
                "concept_id": concept.concept_id,
                "name": concept.name,
                "concept_type": concept.concept_type.value,
                "description": concept.description,
                "importance": concept.importance,
                "activation_level": concept.activation_level,
                "base_activation": concept.base_activation,
                "created_at": concept.created_at.isoformat(),
                "last_accessed": concept.last_accessed.isoformat(),
                "access_count": concept.access_count,
                "attributes": concept.attributes,
                "embedding": concept.embedding
            }
        
        serialized_relations = {}
        for relation_id, relation in self.relations.items():
            serialized_relations[relation_id] = {
                "relation_id": relation.relation_id,
                "source_concept_id": relation.source_concept_id,
                "target_concept_id": relation.target_concept_id,
                "relation_type": relation.relation_type.value,
                "strength": relation.strength,
                "confidence": relation.confidence,
                "created_at": relation.created_at.isoformat(),
                "evidence": relation.evidence,
                "context": relation.context
            }
        
        return {
            "max_concepts": self.max_concepts,
            "activation_decay": self.activation_decay,
            "spreading_factor": self.spreading_factor,
            "similarity_threshold": self.similarity_threshold,
            "concepts": serialized_concepts,
            "relations": serialized_relations,
            "name_index": self.name_index,
            "type_index": {k.value: list(v) for k, v in self.type_index.items()},
            "relation_index": {k: v for k, v in self.relation_index.items()},
            "activation_network": {k: list(v) for k, v in self.activation_network.items()},
            "next_concept_id": self._next_concept_id,
            "next_relation_id": self._next_relation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticMemory':
        """Create from dictionary representation."""
        # Handle backward compatibility for activation_decay vs activation_decay_rate
        activation_decay = data.get("activation_decay") or data.get("activation_decay_rate", 0.1)
        
        # Handle backward compatibility for missing fields
        spreading_factor = data.get("spreading_factor", 0.3)
        similarity_threshold = data.get("similarity_threshold", 0.7)
        
        semantic_mem = cls(
            max_concepts=data["max_concepts"],
            activation_decay=activation_decay,
            spreading_factor=spreading_factor,
            similarity_threshold=similarity_threshold
        )
        
        # Handle backward compatibility for ID counters
        semantic_mem._next_concept_id = data.get("next_concept_id", 1)
        semantic_mem._next_relation_id = data.get("next_relation_id", 1)
        
        # Deserialize concepts
        for concept_id, concept_data in data["concepts"].items():
            concept = SemanticConcept(
                concept_id=concept_data["concept_id"],
                name=concept_data["name"],
                concept_type=ConceptType(concept_data["concept_type"]),
                description=concept_data["description"],
                importance=concept_data["importance"],
                activation_level=concept_data["activation_level"],
                base_activation=concept_data["base_activation"],
                created_at=datetime.fromisoformat(concept_data["created_at"]),
                last_accessed=datetime.fromisoformat(concept_data["last_accessed"]),
                access_count=concept_data["access_count"],
                attributes=concept_data["attributes"],
                embedding=concept_data["embedding"]
            )
            semantic_mem.concepts[concept_id] = concept
        
        # Deserialize relations
        for relation_id, relation_data in data["relations"].items():
            relation = SemanticRelation(
                relation_id=relation_data["relation_id"],
                source_concept_id=relation_data["source_concept_id"],
                target_concept_id=relation_data["target_concept_id"],
                relation_type=SemanticRelationType(relation_data["relation_type"]),
                strength=relation_data["strength"],
                confidence=relation_data["confidence"],
                created_at=datetime.fromisoformat(relation_data["created_at"]),
                evidence=relation_data["evidence"],
                context=relation_data["context"]
            )
            semantic_mem.relations[relation_id] = relation
        
        # Deserialize indices
        semantic_mem.name_index = data["name_index"]
        semantic_mem.type_index = {
            ConceptType(k): set(v) for k, v in data["type_index"].items()
        }
        semantic_mem.relation_index = defaultdict(list, data["relation_index"])
        semantic_mem.activation_network = defaultdict(set)
        for k, v in data["activation_network"].items():
            semantic_mem.activation_network[k] = set(v)
        
        return semantic_mem


# Example usage
if __name__ == "__main__":
    # Example of SemanticMemory usage
    semantic_mem = SemanticMemory(max_concepts=500)
    
    # Add some test concepts
    isabella_id = semantic_mem.add_concept("Isabella", ConceptType.PERSON, "Confident contestant", 0.8)
    maria_id = semantic_mem.add_concept("Maria", ConceptType.PERSON, "Artistic contestant", 0.7)
    coffee_id = semantic_mem.add_concept("Coffee", ConceptType.OBJECT, "Morning beverage", 0.5)
    morning_id = semantic_mem.add_concept("Morning routine", ConceptType.ACTIVITY, "Daily morning activities", 0.6)
    
    # Add relationships
    semantic_mem.add_relation(isabella_id, "confident", SemanticRelationType.HAS_A, 0.9)
    semantic_mem.add_relation(coffee_id, morning_id, SemanticRelationType.RELATED_TO, 0.7)
    
    # Activate concepts
    semantic_mem.activate_concept("Isabella", 0.8)
    
    print("Memory summary:", semantic_mem.get_memory_summary())
    print("Activated concepts:", len(semantic_mem.retrieve_by_activation()))
    
    # Test association retrieval
    associated = semantic_mem.retrieve_by_association(["Isabella"], max_hops=2)
    print("Associated with Isabella:", [(c.name, score) for c, score in associated])