"""
File: enhanced_agent_state.py
Description: Enhanced AgentState with LangGraph integration for PIANO architecture Phase 1.
Integrates all enhanced memory structures with StateGraph and Store API.
"""

from typing import Dict, List, Any, Optional, Set, Annotated, TypedDict
from datetime import datetime
import json
from dataclasses import dataclass, asdict

# Import enhanced memory structures
from .memory_structures.circular_buffer import CircularBuffer, CircularBufferReducer
from .memory_structures.temporal_memory import TemporalMemory
from .memory_structures.episodic_memory import EpisodicMemory
from .memory_structures.semantic_memory import SemanticMemory, ConceptType, SemanticRelationType
from .memory_structures.associative_memory import AssociativeMemory
from .memory_structures.scratch import Scratch
from .memory_structures.spatial_memory import MemoryTree


@dataclass
class SpecializationData:
    """Agent specialization tracking data."""
    current_role: str
    role_history: List[str]
    skills: Dict[str, float]  # skill_name -> proficiency (0.0 to 1.0)
    expertise_level: float
    role_consistency_score: float
    last_role_change: Optional[datetime] = None


@dataclass
class CulturalData:
    """Agent cultural system data."""
    memes_known: Set[str]
    meme_influence: Dict[str, float]  # meme_id -> influence_strength
    cultural_values: Dict[str, float]  # value_name -> strength
    social_roles: List[str]
    cultural_adaptation_rate: float = 0.1


@dataclass
class GovernanceData:
    """Agent governance and social data."""
    voting_history: List[Dict[str, Any]]
    law_adherence: Dict[str, float]  # law_id -> adherence_score
    influence_network: Dict[str, float]  # agent_id -> influence_score
    participation_rate: float
    leadership_tendency: float = 0.5


@dataclass
class PerformanceMetrics:
    """Agent performance tracking metrics."""
    decision_latency: float  # Average decision time in ms
    coherence_score: float  # Behavioral coherence
    social_integration: float  # Social network integration
    memory_efficiency: float  # Memory usage efficiency
    adaptation_rate: float  # Rate of learning/adaptation
    error_rate: float = 0.0


class EnhancedAgentState(TypedDict):
    """
    Enhanced AgentState schema compatible with LangGraph StateGraph.
    Includes all memory layers, specialization, cultural, and governance systems.
    """
    
    # Core Identity
    agent_id: str
    name: str
    first_name: str
    last_name: str
    age: int
    personality_traits: Dict[str, float]
    
    # Specialization System
    specialization: SpecializationData
    
    # Memory Architecture (with LangGraph reducers)
    working_memory: Annotated[List[Dict[str, Any]], CircularBufferReducer(max_size=20)]
    short_term_memory: List[Dict[str, Any]]  # Handled by TemporalMemory
    long_term_memory: Dict[str, Any]  # Handled by AssociativeMemory
    episodic_memory: Dict[str, Any]  # Handled by EpisodicMemory
    semantic_memory: Dict[str, Any]  # Handled by SemanticMemory
    
    # Cultural System (shared via Store API)
    cultural: CulturalData
    
    # Governance & Social (shared via Store API)
    governance: GovernanceData
    
    # Performance Metrics
    performance: PerformanceMetrics
    
    # Current State
    current_time: datetime
    current_location: str
    current_activity: str
    emotional_state: Dict[str, float]
    goals: List[str]
    
    # Interaction Context
    conversation_partners: Set[str]
    recent_interactions: List[Dict[str, Any]]
    social_context: Dict[str, Any]


class EnhancedAgentStateManager:
    """
    Manager class for EnhancedAgentState with integrated memory systems.
    Provides high-level interface for memory operations and state management.
    """
    
    def __init__(self, agent_id: str, name: str, personality_traits: Dict[str, float],
                 working_memory_size: int = 20, temporal_retention_hours: int = 1,
                 max_episodes: int = 100, max_concepts: int = 1000):
        """
        Initialize EnhancedAgentStateManager.
        
        Args:
            agent_id: Unique agent identifier
            name: Agent name
            personality_traits: Personality trait scores
            working_memory_size: Size of circular buffer for working memory
            temporal_retention_hours: Retention period for temporal memory
            max_episodes: Maximum number of episodes in episodic memory
            max_concepts: Maximum number of concepts in semantic memory
        """
        self.agent_id = agent_id
        self.name = name
        
        # Initialize memory systems
        self.circular_buffer = CircularBuffer(max_size=working_memory_size)
        self.temporal_memory = TemporalMemory(retention_hours=temporal_retention_hours)
        self.episodic_memory = EpisodicMemory(max_episodes=max_episodes)
        self.semantic_memory = SemanticMemory(max_concepts=max_concepts)
        
        # Legacy memory systems (for compatibility)
        self.associative_memory = None  # Will be set if loading from existing data
        self.scratch = None
        self.spatial_memory = None
        
        # Initialize state components
        self.specialization = SpecializationData(
            current_role="contestant",
            role_history=["contestant"],
            skills={},
            expertise_level=0.1,
            role_consistency_score=1.0
        )
        
        self.cultural = CulturalData(
            memes_known=set(),
            meme_influence={},
            cultural_values={},
            social_roles=["contestant"]
        )
        
        self.governance = GovernanceData(
            voting_history=[],
            law_adherence={},
            influence_network={},
            participation_rate=0.5
        )
        
        self.performance = PerformanceMetrics(
            decision_latency=0.0,
            coherence_score=0.5,
            social_integration=0.0,
            memory_efficiency=1.0,
            adaptation_rate=0.1
        )
        
        self.state = self._create_initial_state(personality_traits)
    
    def _create_initial_state(self, personality_traits: Dict[str, float]) -> EnhancedAgentState:
        """
        Create initial agent state.
        
        Args:
            personality_traits: Personality trait scores
        
        Returns:
            Initial EnhancedAgentState
        """
        names = self.name.split(' ', 1)
        first_name = names[0]
        last_name = names[1] if len(names) > 1 else ""
        
        return EnhancedAgentState(
            agent_id=self.agent_id,
            name=self.name,
            first_name=first_name,
            last_name=last_name,
            age=25,  # Default age
            personality_traits=personality_traits,
            specialization=self.specialization,
            working_memory=[],
            short_term_memory=[],
            long_term_memory={},
            episodic_memory={},
            semantic_memory={},
            cultural=self.cultural,
            governance=self.governance,
            performance=self.performance,
            current_time=datetime.now(),
            current_location="villa",
            current_activity="idle",
            emotional_state={"happiness": 0.5, "anxiety": 0.1, "excitement": 0.3},
            goals=[],
            conversation_partners=set(),
            recent_interactions=[],
            social_context={}
        )
    
    def add_memory(self, content: str, memory_type: str = "event", 
                   importance: float = 0.5, context: Optional[Dict] = None) -> str:
        """
        Add a memory to all relevant memory systems.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score
            context: Additional context
        
        Returns:
            Memory ID from episodic memory
        """
        timestamp = datetime.now()
        
        # Add to circular buffer (working memory)
        working_memory_entry = {
            "content": content,
            "type": memory_type,
            "timestamp": timestamp.isoformat(),
            "importance": importance,
            "context": context or {}
        }
        self.circular_buffer.add_memory(content, memory_type, importance, context)
        
        # Add to temporal memory
        temp_mem_id = self.temporal_memory.add_memory(content, memory_type, importance, context, timestamp)
        
        # Add to episodic memory
        participants = set()
        location = self.state["current_location"]
        if context and "participants" in context:
            participants = set(context["participants"])
        
        episodic_mem_id = self.episodic_memory.add_event(
            content, memory_type, importance, participants, location,
            emotional_valence=context.get("emotional_valence", 0.0) if context else 0.0,
            metadata=context, timestamp=timestamp
        )
        
        # Extract concepts for semantic memory
        self._extract_and_add_concepts(content, memory_type, importance, context)
        
        # Update state
        self._update_state_from_memories()
        
        return episodic_mem_id
    
    def _extract_and_add_concepts(self, content: str, memory_type: str, 
                                 importance: float, context: Optional[Dict]) -> None:
        """
        Extract concepts from memory content and add to semantic memory.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score
            context: Additional context
        """
        # Simple concept extraction (in practice, would use NLP)
        words = content.lower().split()
        
        # Extract potential person names (capitalized words)
        for word in content.split():
            if word and word[0].isupper() and len(word) > 2:
                if word not in ["I", "The", "A", "An"]:
                    concept_id = self.semantic_memory.add_concept(
                        word, ConceptType.PERSON, f"Person mentioned: {word}",
                        importance * 0.5
                    )
        
        # Extract actions (verbs - simplified)
        action_words = ["talk", "walk", "eat", "sleep", "think", "feel", "go", "come", "see", "hear"]
        for word in words:
            if any(action in word for action in action_words):
                self.semantic_memory.add_concept(
                    word, ConceptType.ACTION, f"Action: {word}",
                    importance * 0.3
                )
        
        # Extract emotions
        emotion_words = ["happy", "sad", "angry", "excited", "nervous", "calm", "anxious"]
        for word in words:
            if any(emotion in word for emotion in emotion_words):
                self.semantic_memory.add_concept(
                    word, ConceptType.EMOTION, f"Emotion: {word}",
                    importance * 0.4
                )
    
    def _update_state_from_memories(self) -> None:
        """Update state based on current memory contents."""
        # Update working memory in state
        recent_memories = self.circular_buffer.get_recent_memories(5)
        self.state["working_memory"] = [
            {
                "content": mem["content"],
                "type": mem["type"],
                "timestamp": mem["timestamp"].isoformat(),
                "importance": mem["importance"]
            }
            for mem in recent_memories
        ]
        
        # Update performance metrics
        buffer_utilization = len(self.circular_buffer) / self.circular_buffer.max_size
        self.performance.memory_efficiency = 1.0 - buffer_utilization * 0.5
        
        # Update coherence score based on memory consistency
        self.performance.coherence_score = self._calculate_coherence_score()
        
        self.state["performance"] = self.performance
    
    def _calculate_coherence_score(self) -> float:
        """
        Calculate behavioral coherence score based on memory patterns.
        
        Returns:
            Coherence score (0.0 to 1.0)
        """
        # Simple coherence calculation based on role consistency
        if not self.specialization.skills:
            return 0.5
        
        # Check consistency between recent actions and current role
        recent_memories = self.circular_buffer.get_recent_memories(10)
        role_consistent_actions = 0
        
        for memory in recent_memories:
            if memory["type"] == "action":
                # Simple heuristic: actions related to current role
                if self.specialization.current_role.lower() in memory["content"].lower():
                    role_consistent_actions += 1
        
        if recent_memories:
            consistency = role_consistent_actions / len(recent_memories)
            return min(consistency + 0.3, 1.0)  # Boost baseline
        
        return 0.5
    
    def update_specialization(self, action: str, skill_gained: Optional[Dict[str, float]] = None) -> None:
        """
        Update agent specialization based on actions.
        
        Args:
            action: Action performed
            skill_gained: Skills gained from action
        """
        if skill_gained:
            for skill, gain in skill_gained.items():
                current_level = self.specialization.skills.get(skill, 0.0)
                self.specialization.skills[skill] = min(current_level + gain, 1.0)
        
        # Update expertise level
        if self.specialization.skills:
            self.specialization.expertise_level = sum(self.specialization.skills.values()) / len(self.specialization.skills)
        
        # Update role consistency
        self.specialization.role_consistency_score = self._calculate_coherence_score()
        
        self.state["specialization"] = self.specialization
    
    def update_cultural_influence(self, meme_id: str, influence_change: float) -> None:
        """
        Update cultural meme influence.
        
        Args:
            meme_id: ID of the meme
            influence_change: Change in influence strength
        """
        current_influence = self.cultural.meme_influence.get(meme_id, 0.0)
        new_influence = max(0.0, min(1.0, current_influence + influence_change))
        
        if new_influence > 0.1:
            self.cultural.meme_influence[meme_id] = new_influence
            self.cultural.memes_known.add(meme_id)
        else:
            # Remove weak influences
            self.cultural.meme_influence.pop(meme_id, None)
            self.cultural.memes_known.discard(meme_id)
        
        self.state["cultural"] = self.cultural
    
    def add_governance_vote(self, vote_data: Dict[str, Any]) -> None:
        """
        Add a governance vote to history.
        
        Args:
            vote_data: Vote information
        """
        vote_record = {
            **vote_data,
            "timestamp": datetime.now().isoformat(),
            "agent_id": self.agent_id
        }
        
        self.governance.voting_history.append(vote_record)
        
        # Update participation rate
        total_opportunities = vote_data.get("total_opportunities", len(self.governance.voting_history))
        if total_opportunities > 0:
            self.governance.participation_rate = len(self.governance.voting_history) / total_opportunities
        
        self.state["governance"] = self.governance
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive memory system summary.
        
        Returns:
            Dictionary with memory statistics
        """
        return {
            "working_memory": self.circular_buffer.get_memory_summary(),
            "temporal_memory": self.temporal_memory.get_temporal_summary(),
            "episodic_memory": {
                "total_episodes": len(self.episodic_memory.episodes),
                "total_events": len(self.episodic_memory.events)
            },
            "semantic_memory": self.semantic_memory.get_memory_summary(),
            "performance": asdict(self.performance)
        }
    
    def process_social_interaction(self, other_agent: str, interaction_type: str, 
                                 content: str, emotional_impact: float = 0.0) -> None:
        """
        Process a social interaction with another agent.
        
        Args:
            other_agent: Name/ID of other agent
            interaction_type: Type of interaction
            content: Interaction content
            emotional_impact: Emotional impact of interaction
        """
        # Add to conversation partners
        self.state["conversation_partners"].add(other_agent)
        
        # Create interaction record
        interaction = {
            "partner": other_agent,
            "type": interaction_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "emotional_impact": emotional_impact
        }
        
        self.state["recent_interactions"].append(interaction)
        
        # Keep only recent interactions
        if len(self.state["recent_interactions"]) > 10:
            self.state["recent_interactions"] = self.state["recent_interactions"][-10:]
        
        # Add to memory systems
        memory_content = f"{interaction_type} with {other_agent}: {content}"
        self.add_memory(
            memory_content, "conversation", 
            abs(emotional_impact) + 0.3,  # Social interactions are generally important
            {
                "participants": [other_agent],
                "emotional_valence": emotional_impact,
                "interaction_type": interaction_type
            }
        )
        
        # Update social integration score
        unique_partners = len(self.state["conversation_partners"])
        self.performance.social_integration = min(unique_partners * 0.1, 1.0)
        self.state["performance"] = self.performance
    
    def update_emotional_state(self, emotion_changes: Dict[str, float]) -> None:
        """
        Update agent's emotional state.
        
        Args:
            emotion_changes: Changes to emotional values
        """
        for emotion, change in emotion_changes.items():
            current_value = self.state["emotional_state"].get(emotion, 0.0)
            new_value = max(0.0, min(1.0, current_value + change))
            self.state["emotional_state"][emotion] = new_value
        
        # Add emotional state change to memory
        emotion_description = ", ".join([f"{k}: {v:.2f}" for k, v in emotion_changes.items()])
        self.add_memory(
            f"Emotional state changed: {emotion_description}",
            "emotion", 0.4,
            {"emotion_changes": emotion_changes}
        )
    
    def cleanup_memories(self) -> Dict[str, int]:
        """
        Perform cleanup on all memory systems.
        
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {}
        
        # Cleanup circular buffer (expired memories)
        expired_working = self.circular_buffer.cleanup_expired_memories()
        stats["expired_working_memories"] = len(expired_working)
        
        # Cleanup temporal memory
        removed_temporal = self.temporal_memory.cleanup_expired_memories()
        stats["removed_temporal_memories"] = len(removed_temporal)
        
        # Update activation decay in semantic memory
        self.semantic_memory.update_activation_decay()
        
        # Consolidate similar concepts
        consolidated = self.semantic_memory.consolidate_concepts()
        stats["consolidated_concepts"] = len(consolidated)
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entire state to dictionary for persistence.
        
        Returns:
            Dictionary representation
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": {
                **self.state,
                "current_time": self.state["current_time"].isoformat(),
                "conversation_partners": list(self.state["conversation_partners"]),
                "specialization": asdict(self.specialization),
                "cultural": {
                    **asdict(self.cultural),
                    "memes_known": list(self.cultural.memes_known)
                },
                "governance": asdict(self.governance),
                "performance": asdict(self.performance)
            },
            "memory_systems": {
                "circular_buffer": self.circular_buffer.to_dict(),
                "temporal_memory": self.temporal_memory.to_dict(),
                "episodic_memory": self.episodic_memory.to_dict(),
                "semantic_memory": self.semantic_memory.to_dict()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedAgentStateManager':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary containing agent state data
        
        Returns:
            EnhancedAgentStateManager instance
        """
        state_data = data["state"]
        
        # Create manager
        manager = cls(
            agent_id=data["agent_id"],
            name=data["name"],
            personality_traits=state_data["personality_traits"]
        )
        
        # Restore memory systems
        memory_systems = data["memory_systems"]
        manager.circular_buffer = CircularBuffer.from_dict(memory_systems["circular_buffer"])
        manager.temporal_memory = TemporalMemory.from_dict(memory_systems["temporal_memory"])
        manager.episodic_memory = EpisodicMemory.from_dict(memory_systems["episodic_memory"])
        manager.semantic_memory = SemanticMemory.from_dict(memory_systems["semantic_memory"])
        
        # Restore state components
        manager.specialization = SpecializationData(**state_data["specialization"])
        
        cultural_data = state_data["cultural"]
        manager.cultural = CulturalData(
            **{**cultural_data, "memes_known": set(cultural_data["memes_known"])}
        )
        
        manager.governance = GovernanceData(**state_data["governance"])
        manager.performance = PerformanceMetrics(**state_data["performance"])
        
        # Restore state
        manager.state = EnhancedAgentState(
            **{
                **state_data,
                "current_time": datetime.fromisoformat(state_data["current_time"]),
                "conversation_partners": set(state_data["conversation_partners"]),
                "specialization": manager.specialization,
                "cultural": manager.cultural,
                "governance": manager.governance,
                "performance": manager.performance
            }
        )
        
        return manager


# Helper functions for LangGraph integration
def create_enhanced_agent_state(agent_id: str, name: str, 
                               personality_traits: Dict[str, float]) -> EnhancedAgentStateManager:
    """
    Create an EnhancedAgentStateManager for use in LangGraph.
    
    Args:
        agent_id: Unique agent identifier
        name: Agent name
        personality_traits: Personality trait scores
    
    Returns:
        EnhancedAgentStateManager instance
    """
    return EnhancedAgentStateManager(agent_id, name, personality_traits)


# Example usage
if __name__ == "__main__":
    # Example of EnhancedAgentState usage
    manager = create_enhanced_agent_state(
        "agent_001", 
        "Isabella Rodriguez",
        {"confidence": 0.8, "openness": 0.7, "extroversion": 0.9}
    )
    
    # Add some memories
    manager.add_memory("Started morning routine with coffee", "activity", 0.6)
    manager.add_memory("Had conversation with Maria about dating", "conversation", 0.8, 
                      {"participants": ["Maria"], "emotional_valence": 0.3})
    
    # Update specialization
    manager.update_specialization("social_interaction", {"communication": 0.1, "empathy": 0.05})
    
    # Process social interaction
    manager.process_social_interaction("Klaus", "flirtation", "Shared a laugh about dating experiences", 0.4)
    
    print("Memory Summary:", manager.get_memory_summary())
    print("Agent State Keys:", list(manager.state.keys()))
    print("Specialization:", manager.specialization)
    print("Performance:", manager.performance)