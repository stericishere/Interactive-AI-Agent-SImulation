"""
File: episodic_memory.py
Description: EpisodicMemory for event sequences and narrative coherence.
Enhanced PIANO architecture implementation with causal relationship tracking.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid


class CausalRelationType(Enum):
    """Types of causal relationships between events."""
    CAUSES = "causes"
    ENABLES = "enables"
    PREVENTS = "prevents"
    FOLLOWS = "follows"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"


@dataclass
class CausalRelation:
    """Represents a causal relationship between two events."""
    source_event_id: str
    target_event_id: str
    relation_type: CausalRelationType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    created_at: datetime
    evidence: List[str] = None  # Supporting evidence
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


class EpisodeType(Enum):
    """Types of episodic sequences."""
    CONVERSATION = "conversation"
    ACTIVITY = "activity"
    SOCIAL_EVENT = "social_event"
    DECISION_MAKING = "decision_making"
    EMOTIONAL_SEQUENCE = "emotional_sequence"
    ROUTINE = "routine"


@dataclass
class Episode:
    """Represents a sequence of related events forming a coherent episode."""
    episode_id: str
    episode_type: EpisodeType
    start_time: datetime
    end_time: datetime
    event_ids: List[str]
    title: str
    summary: str
    importance: float
    coherence_score: float
    participants: Set[str] = None  # Other agents involved
    location: str = None
    outcome: str = None
    emotional_impact: float = 0.0
    
    def __post_init__(self):
        if self.participants is None:
            self.participants = set()


class EpisodicMemory:
    """
    Enhanced EpisodicMemory system for managing event sequences and narrative coherence.
    Tracks causal relationships and maintains coherent episode structures.
    """
    
    def __init__(self, max_episodes: int = 100, coherence_threshold: float = 0.6):
        """
        Initialize EpisodicMemory.
        
        Args:
            max_episodes: Maximum number of episodes to maintain
            coherence_threshold: Minimum coherence score for episode creation
        """
        self.max_episodes = max_episodes
        self.coherence_threshold = coherence_threshold
        
        # Core storage
        self.events: Dict[str, Dict[str, Any]] = {}  # event_id -> event_data
        self.episodes: Dict[str, Episode] = {}  # episode_id -> episode
        self.causal_relations: List[CausalRelation] = []
        
        # Indexing
        self.temporal_index: List[Tuple[datetime, str]] = []  # (timestamp, event_id)
        self.participant_index: Dict[str, List[str]] = {}  # participant -> episode_ids
        self.type_index: Dict[EpisodeType, List[str]] = {}  # episode_type -> episode_ids
        
        self._next_event_id = 1
        self._next_episode_id = 1
    
    def add_event(self, content: str, event_type: str, importance: float = 0.5,
                  participants: Optional[Set[str]] = None, location: Optional[str] = None,
                  emotional_valence: float = 0.0, metadata: Optional[Dict] = None,
                  timestamp: Optional[datetime] = None) -> str:
        """
        Add a new event to episodic memory.
        
        Args:
            content: Event description
            event_type: Type of event (action, thought, conversation, etc.)
            importance: Event importance score
            participants: Other agents involved
            location: Where the event occurred
            emotional_valence: Emotional impact (-1.0 to 1.0)
            metadata: Additional event metadata
            timestamp: Event timestamp
        
        Returns:
            Event ID
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        event_id = f"event_{self._next_event_id}"
        self._next_event_id += 1
        
        event_data = {
            "id": event_id,
            "content": content,
            "type": event_type,
            "timestamp": timestamp,
            "importance": importance,
            "participants": participants or set(),
            "location": location,
            "emotional_valence": emotional_valence,
            "metadata": metadata or {},
            "episode_ids": set(),  # Episodes this event belongs to
            "causal_predecessors": set(),  # Events that caused this
            "causal_successors": set()  # Events this caused
        }
        
        self.events[event_id] = event_data
        
        # Update temporal index
        self.temporal_index.append((timestamp, event_id))
        self.temporal_index.sort(key=lambda x: x[0])
        
        # Try to integrate into existing episodes or create new ones
        self._integrate_event_into_episodes(event_id)
        
        return event_id
    
    def _integrate_event_into_episodes(self, event_id: str) -> None:
        """
        Integrate a new event into existing episodes or create new episodes.
        
        Args:
            event_id: ID of the event to integrate
        """
        event_data = self.events[event_id]
        timestamp = event_data["timestamp"]
        
        # Look for recent episodes that could incorporate this event
        recent_episodes = self._get_recent_episodes(hours_back=2)
        
        integrated = False
        
        for episode in recent_episodes:
            # Check if event fits within episode timeframe and context
            time_gap = (timestamp - episode.end_time).total_seconds() / 60  # minutes
            
            if time_gap <= 30:  # Within 30 minutes
                coherence = self._calculate_event_episode_coherence(event_id, episode.episode_id)
                
                if coherence >= self.coherence_threshold:
                    self._add_event_to_episode(event_id, episode.episode_id)
                    integrated = True
                    break
        
        # If not integrated into existing episode, try to create a new one
        if not integrated:
            self._try_create_new_episode(event_id)
    
    def _add_event_to_episode(self, event_id: str, episode_id: str) -> None:
        """
        Add an event to an existing episode.
        
        Args:
            event_id: ID of the event to add
            episode_id: ID of the episode to add to
        """
        if episode_id not in self.episodes or event_id not in self.events:
            return
        
        episode = self.episodes[episode_id]
        event_data = self.events[event_id]
        
        # Add event to episode
        episode.event_ids.append(event_id)
        
        # Update episode end time if this event is later
        if event_data["timestamp"] > episode.end_time:
            episode.end_time = event_data["timestamp"]
        
        # Update episode participants
        episode.participants.update(event_data["participants"])
        
        # Update event's episode references
        event_data["episode_ids"].add(episode_id)
        
        # Recalculate episode metrics
        events = [self.events[eid] for eid in episode.event_ids]
        episode.importance = sum(e["importance"] for e in events) / len(events)
        episode.coherence_score = self._calculate_episode_coherence(episode.event_ids)
        episode.emotional_impact = sum(e["emotional_valence"] for e in events) / len(events)
    
    def _get_recent_episodes(self, hours_back: int = 2) -> List[Episode]:
        """
        Get episodes that ended within the specified time window.
        
        Args:
            hours_back: Hours to look back
        
        Returns:
            List of recent episodes
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_episodes = [
            episode for episode in self.episodes.values()
            if episode.end_time >= cutoff_time
        ]
        
        return sorted(recent_episodes, key=lambda x: x.end_time, reverse=True)
    
    def _calculate_event_episode_coherence(self, event_id: str, episode_id: str) -> float:
        """
        Calculate how well an event fits into an existing episode.
        
        Args:
            event_id: ID of the event
            episode_id: ID of the episode
        
        Returns:
            Coherence score (0.0 to 1.0)
        """
        event_data = self.events[event_id]
        episode = self.episodes[episode_id]
        
        coherence_factors = []
        
        # 1. Participant overlap
        if episode.participants:
            participant_overlap = len(event_data["participants"] & episode.participants)
            max_participants = max(len(event_data["participants"]), len(episode.participants))
            if max_participants > 0:
                coherence_factors.append(participant_overlap / max_participants)
        
        # 2. Location consistency
        if event_data["location"] and episode.location:
            location_match = 1.0 if event_data["location"] == episode.location else 0.3
            coherence_factors.append(location_match)
        
        # 3. Episode type compatibility
        type_compatibility = self._get_type_compatibility(event_data["type"], episode.episode_type)
        coherence_factors.append(type_compatibility)
        
        # 4. Causal relationship with episode events
        causal_score = self._calculate_causal_coherence(event_id, episode.event_ids)
        coherence_factors.append(causal_score)
        
        # 5. Temporal continuity
        last_event_id = episode.event_ids[-1] if episode.event_ids else None
        if last_event_id:
            time_gap = (event_data["timestamp"] - self.events[last_event_id]["timestamp"]).total_seconds() / 60
            temporal_score = max(0, 1.0 - (time_gap / 60))  # Decay over 1 hour
            coherence_factors.append(temporal_score)
        
        return sum(coherence_factors) / len(coherence_factors) if coherence_factors else 0.0
    
    def _get_type_compatibility(self, event_type: str, episode_type: EpisodeType) -> float:
        """
        Calculate compatibility between event type and episode type.
        
        Args:
            event_type: Type of the event
            episode_type: Type of the episode
        
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        compatibility_matrix = {
            EpisodeType.CONVERSATION: {
                "conversation": 1.0, "thought": 0.8, "action": 0.3, "observation": 0.6
            },
            EpisodeType.ACTIVITY: {
                "action": 1.0, "observation": 0.8, "thought": 0.6, "conversation": 0.4
            },
            EpisodeType.SOCIAL_EVENT: {
                "conversation": 0.9, "action": 0.8, "observation": 0.7, "thought": 0.5
            },
            EpisodeType.DECISION_MAKING: {
                "thought": 1.0, "action": 0.8, "conversation": 0.6, "observation": 0.7
            },
            EpisodeType.EMOTIONAL_SEQUENCE: {
                "thought": 0.9, "conversation": 0.8, "action": 0.6, "observation": 0.5
            },
            EpisodeType.ROUTINE: {
                "action": 1.0, "observation": 0.6, "thought": 0.4, "conversation": 0.3
            }
        }
        
        return compatibility_matrix.get(episode_type, {}).get(event_type, 0.5)
    
    def _calculate_causal_coherence(self, event_id: str, episode_event_ids: List[str]) -> float:
        """
        Calculate causal coherence between an event and episode events.
        
        Args:
            event_id: ID of the event
            episode_event_ids: List of event IDs in the episode
        
        Returns:
            Causal coherence score (0.0 to 1.0)
        """
        if not episode_event_ids:
            return 0.5
        
        causal_connections = 0
        total_possible = len(episode_event_ids)
        
        for relation in self.causal_relations:
            if (relation.source_event_id == event_id and 
                relation.target_event_id in episode_event_ids) or \
               (relation.target_event_id == event_id and 
                relation.source_event_id in episode_event_ids):
                causal_connections += relation.strength
        
        return min(causal_connections / total_possible, 1.0) if total_possible > 0 else 0.0
    
    def _try_create_new_episode(self, event_id: str) -> Optional[str]:
        """
        Try to create a new episode starting with the given event.
        
        Args:
            event_id: ID of the initial event
        
        Returns:
            Episode ID if created, None otherwise
        """
        event_data = self.events[event_id]
        
        # Look for recent related events that could form an episode
        recent_events = self._get_recent_related_events(event_id, hours_back=1)
        
        if len(recent_events) >= 2:  # Need at least 2 events for an episode
            episode_id = self._create_episode(recent_events)
            return episode_id
        
        return None
    
    def _get_recent_related_events(self, event_id: str, hours_back: int = 1) -> List[str]:
        """
        Get recent events related to the given event.
        
        Args:
            event_id: Reference event ID
            hours_back: Hours to look back
        
        Returns:
            List of related event IDs
        """
        event_data = self.events[event_id]
        cutoff_time = event_data["timestamp"] - timedelta(hours=hours_back)
        
        related_events = [event_id]
        
        # Find events with similar participants, location, or causal relationships
        for timestamp, other_event_id in self.temporal_index:
            if timestamp < cutoff_time:
                continue
            if other_event_id == event_id:
                continue
            
            other_event = self.events[other_event_id]
            
            # Check relatedness criteria
            participant_overlap = len(event_data["participants"] & other_event["participants"])
            location_match = event_data["location"] == other_event["location"]
            
            if participant_overlap > 0 or location_match:
                related_events.append(other_event_id)
        
        return sorted(related_events, key=lambda x: self.events[x]["timestamp"])
    
    def _create_episode(self, event_ids: List[str]) -> str:
        """
        Create a new episode from a list of event IDs.
        
        Args:
            event_ids: List of event IDs to include in episode
        
        Returns:
            Episode ID
        """
        episode_id = f"episode_{self._next_episode_id}"
        self._next_episode_id += 1
        
        # Gather episode metadata
        events = [self.events[eid] for eid in event_ids]
        start_time = min(e["timestamp"] for e in events)
        end_time = max(e["timestamp"] for e in events)
        
        participants = set()
        locations = set()
        for event in events:
            participants.update(event["participants"])
            if event["location"]:
                locations.add(event["location"])
        
        # Determine episode type
        episode_type = self._infer_episode_type(events)
        
        # Generate title and summary
        title = self._generate_episode_title(events, episode_type)
        summary = self._generate_episode_summary(events)
        
        # Calculate importance and coherence
        importance = sum(e["importance"] for e in events) / len(events)
        coherence_score = self._calculate_episode_coherence(event_ids)
        
        # Calculate emotional impact
        emotional_impact = sum(e["emotional_valence"] for e in events) / len(events)
        
        episode = Episode(
            episode_id=episode_id,
            episode_type=episode_type,
            start_time=start_time,
            end_time=end_time,
            event_ids=event_ids,
            title=title,
            summary=summary,
            importance=importance,
            coherence_score=coherence_score,
            participants=participants,
            location=list(locations)[0] if len(locations) == 1 else None,
            emotional_impact=emotional_impact
        )
        
        self.episodes[episode_id] = episode
        
        # Update event references
        for event_id in event_ids:
            self.events[event_id]["episode_ids"].add(episode_id)
        
        # Update indices
        for participant in participants:
            if participant not in self.participant_index:
                self.participant_index[participant] = []
            self.participant_index[participant].append(episode_id)
        
        if episode_type not in self.type_index:
            self.type_index[episode_type] = []
        self.type_index[episode_type].append(episode_id)
        
        # Cleanup old episodes if necessary
        self._cleanup_old_episodes()
        
        return episode_id
    
    def _infer_episode_type(self, events: List[Dict[str, Any]]) -> EpisodeType:
        """
        Infer the most appropriate episode type from events.
        
        Args:
            events: List of event data
        
        Returns:
            Inferred episode type
        """
        type_counts = {}
        
        for event in events:
            event_type = event["type"]
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        # Simple heuristics for episode type inference
        if type_counts.get("conversation", 0) > len(events) * 0.6:
            return EpisodeType.CONVERSATION
        elif type_counts.get("thought", 0) > len(events) * 0.6:
            return EpisodeType.DECISION_MAKING
        elif type_counts.get("action", 0) > len(events) * 0.6:
            return EpisodeType.ACTIVITY
        elif sum(abs(e["emotional_valence"]) for e in events) / len(events) > 0.5:
            return EpisodeType.EMOTIONAL_SEQUENCE
        elif len({participant for e in events if e["participants"] for participant in e["participants"]}) > 1:
            return EpisodeType.SOCIAL_EVENT
        else:
            return EpisodeType.ROUTINE
    
    def _generate_episode_title(self, events: List[Dict[str, Any]], episode_type: EpisodeType) -> str:
        """
        Generate a descriptive title for the episode.
        
        Args:
            events: List of event data
            episode_type: Type of episode
        
        Returns:
            Episode title
        """
        if episode_type == EpisodeType.CONVERSATION:
            participants = set()
            for event in events:
                participants.update(event["participants"])
            if participants:
                return f"Conversation with {', '.join(participants)}"
            return "Conversation"
        
        elif episode_type == EpisodeType.ACTIVITY:
            # Extract key actions
            actions = [e["content"].split()[0] for e in events if e["type"] == "action"]
            if actions:
                return f"Activity: {actions[0]}"
            return "Activity sequence"
        
        elif episode_type == EpisodeType.DECISION_MAKING:
            return "Decision making process"
        
        elif episode_type == EpisodeType.EMOTIONAL_SEQUENCE:
            avg_valence = sum(e["emotional_valence"] for e in events) / len(events)
            emotion = "positive" if avg_valence > 0 else "negative"
            return f"Emotional experience ({emotion})"
        
        elif episode_type == EpisodeType.SOCIAL_EVENT:
            return "Social interaction"
        
        else:
            return "Daily routine"
    
    def _generate_episode_summary(self, events: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the episode.
        
        Args:
            events: List of event data
        
        Returns:
            Episode summary
        """
        # Simple summary - first and last events
        if len(events) == 1:
            return events[0]["content"]
        
        first_event = events[0]["content"]
        last_event = events[-1]["content"]
        
        return f"Started with: {first_event}. Ended with: {last_event}"
    
    def _calculate_episode_coherence(self, event_ids: List[str]) -> float:
        """
        Calculate overall coherence score for an episode.
        
        Args:
            event_ids: List of event IDs in the episode
        
        Returns:
            Coherence score (0.0 to 1.0)
        """
        if len(event_ids) < 2:
            return 1.0
        
        coherence_scores = []
        
        # Temporal coherence
        events = [self.events[eid] for eid in event_ids]
        time_gaps = []
        for i in range(1, len(events)):
            gap = (events[i]["timestamp"] - events[i-1]["timestamp"]).total_seconds() / 60
            time_gaps.append(gap)
        
        avg_gap = sum(time_gaps) / len(time_gaps)
        temporal_coherence = max(0, 1.0 - (avg_gap / 60))  # Decay over 1 hour
        coherence_scores.append(temporal_coherence)
        
        # Participant coherence
        all_participants = set()
        for event in events:
            all_participants.update(event["participants"])
        
        if all_participants:
            participant_consistency = []
            for event in events:
                if event["participants"]:
                    overlap = len(event["participants"] & all_participants)
                    consistency = overlap / len(all_participants)
                    participant_consistency.append(consistency)
            
            if participant_consistency:
                coherence_scores.append(sum(participant_consistency) / len(participant_consistency))
        
        # Causal coherence
        causal_connections = 0
        total_pairs = len(event_ids) * (len(event_ids) - 1) / 2
        
        for relation in self.causal_relations:
            if (relation.source_event_id in event_ids and 
                relation.target_event_id in event_ids):
                causal_connections += relation.strength
        
        if total_pairs > 0:
            causal_coherence = min(causal_connections / total_pairs, 1.0)
            coherence_scores.append(causal_coherence)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
    
    def _cleanup_old_episodes(self) -> None:
        """Remove old episodes if we exceed the maximum limit."""
        if len(self.episodes) <= self.max_episodes:
            return
        
        # Sort episodes by importance and recency
        sorted_episodes = sorted(
            self.episodes.values(),
            key=lambda x: (x.importance, x.end_time),
            reverse=True
        )
        
        # Keep only the most important/recent episodes
        episodes_to_keep = sorted_episodes[:self.max_episodes]
        episodes_to_remove = [ep for ep in self.episodes.values() 
                            if ep not in episodes_to_keep]
        
        for episode in episodes_to_remove:
            self._remove_episode(episode.episode_id)
    
    def _remove_episode(self, episode_id: str) -> None:
        """
        Remove an episode and clean up references.
        
        Args:
            episode_id: ID of episode to remove
        """
        if episode_id not in self.episodes:
            return
        
        episode = self.episodes[episode_id]
        
        # Remove episode references from events
        for event_id in episode.event_ids:
            if event_id in self.events:
                self.events[event_id]["episode_ids"].discard(episode_id)
        
        # Remove from indices
        for participant in episode.participants:
            if participant in self.participant_index:
                self.participant_index[participant] = [
                    ep_id for ep_id in self.participant_index[participant]
                    if ep_id != episode_id
                ]
        
        if episode.episode_type in self.type_index:
            self.type_index[episode.episode_type] = [
                ep_id for ep_id in self.type_index[episode.episode_type]
                if ep_id != episode_id
            ]
        
        # Remove episode
        del self.episodes[episode_id]
    
    def add_causal_relation(self, source_event_id: str, target_event_id: str,
                          relation_type: CausalRelationType, strength: float = 0.8,
                          confidence: float = 0.8, evidence: Optional[List[str]] = None) -> None:
        """
        Add a causal relationship between two events.
        
        Args:
            source_event_id: ID of the source event
            target_event_id: ID of the target event
            relation_type: Type of causal relationship
            strength: Strength of the relationship
            confidence: Confidence in the relationship
            evidence: Supporting evidence
        """
        relation = CausalRelation(
            source_event_id=source_event_id,
            target_event_id=target_event_id,
            relation_type=relation_type,
            strength=strength,
            confidence=confidence,
            created_at=datetime.now(),
            evidence=evidence or []
        )
        
        self.causal_relations.append(relation)
        
        # Update event references
        if source_event_id in self.events:
            self.events[source_event_id]["causal_successors"].add(target_event_id)
        
        if target_event_id in self.events:
            self.events[target_event_id]["causal_predecessors"].add(source_event_id)
    
    def get_episodes_by_participant(self, participant: str) -> List[Episode]:
        """
        Get all episodes involving a specific participant.
        
        Args:
            participant: Name of the participant
        
        Returns:
            List of episodes involving the participant
        """
        episode_ids = self.participant_index.get(participant, [])
        return [self.episodes[ep_id] for ep_id in episode_ids if ep_id in self.episodes]
    
    def get_episodes_by_type(self, episode_type: EpisodeType) -> List[Episode]:
        """
        Get all episodes of a specific type.
        
        Args:
            episode_type: Type of episodes to retrieve
        
        Returns:
            List of episodes of the specified type
        """
        episode_ids = self.type_index.get(episode_type, [])
        return [self.episodes[ep_id] for ep_id in episode_ids if ep_id in self.episodes]
    
    def get_episode_narrative(self, episode_id: str) -> str:
        """
        Generate a narrative description of an episode.
        
        Args:
            episode_id: ID of the episode
        
        Returns:
            Narrative description
        """
        if episode_id not in self.episodes:
            return ""
        
        episode = self.episodes[episode_id]
        events = [self.events[eid] for eid in episode.event_ids]
        events.sort(key=lambda x: x["timestamp"])
        
        narrative = f"**{episode.title}**\n"
        narrative += f"*{episode.start_time.strftime('%Y-%m-%d %H:%M')} - {episode.end_time.strftime('%H:%M')}*\n\n"
        
        for i, event in enumerate(events, 1):
            timestamp = event["timestamp"].strftime("%H:%M")
            narrative += f"{i}. [{timestamp}] {event['content']}\n"
        
        if episode.outcome:
            narrative += f"\n**Outcome:** {episode.outcome}\n"
        
        return narrative
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Convert datetime objects and sets to serializable formats
        serialized_events = {}
        for event_id, event_data in self.events.items():
            serialized_events[event_id] = {
                **event_data,
                "timestamp": event_data["timestamp"].isoformat(),
                "participants": list(event_data["participants"]),
                "episode_ids": list(event_data["episode_ids"]),
                "causal_predecessors": list(event_data["causal_predecessors"]),
                "causal_successors": list(event_data["causal_successors"])
            }
        
        serialized_episodes = {}
        for episode_id, episode in self.episodes.items():
            serialized_episodes[episode_id] = {
                "episode_id": episode.episode_id,
                "episode_type": episode.episode_type.value,
                "start_time": episode.start_time.isoformat(),
                "end_time": episode.end_time.isoformat(),
                "event_ids": episode.event_ids,
                "title": episode.title,
                "summary": episode.summary,
                "importance": episode.importance,
                "coherence_score": episode.coherence_score,
                "participants": list(episode.participants),
                "location": episode.location,
                "outcome": episode.outcome,
                "emotional_impact": episode.emotional_impact
            }
        
        serialized_relations = []
        for relation in self.causal_relations:
            serialized_relations.append({
                "source_event_id": relation.source_event_id,
                "target_event_id": relation.target_event_id,
                "relation_type": relation.relation_type.value,
                "strength": relation.strength,
                "confidence": relation.confidence,
                "created_at": relation.created_at.isoformat(),
                "evidence": relation.evidence
            })
        
        return {
            "max_episodes": self.max_episodes,
            "coherence_threshold": self.coherence_threshold,
            "events": serialized_events,
            "episodes": serialized_episodes,
            "causal_relations": serialized_relations,
            "temporal_index": [(ts.isoformat(), event_id) for ts, event_id in self.temporal_index],
            "participant_index": {k: v for k, v in self.participant_index.items()},
            "type_index": {k.value: v for k, v in self.type_index.items()},
            "next_event_id": self._next_event_id,
            "next_episode_id": self._next_episode_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """Create from dictionary representation."""
        episodic_mem = cls(
            max_episodes=data["max_episodes"],
            coherence_threshold=data["coherence_threshold"]
        )
        
        episodic_mem._next_event_id = data["next_event_id"]
        episodic_mem._next_episode_id = data["next_episode_id"]
        
        # Deserialize events
        for event_id, event_data in data["events"].items():
            episodic_mem.events[event_id] = {
                **event_data,
                "timestamp": datetime.fromisoformat(event_data["timestamp"]),
                "participants": set(event_data["participants"]),
                "episode_ids": set(event_data["episode_ids"]),
                "causal_predecessors": set(event_data["causal_predecessors"]),
                "causal_successors": set(event_data["causal_successors"])
            }
        
        # Deserialize episodes
        for episode_id, episode_data in data["episodes"].items():
            episode = Episode(
                episode_id=episode_data["episode_id"],
                episode_type=EpisodeType(episode_data["episode_type"]),
                start_time=datetime.fromisoformat(episode_data["start_time"]),
                end_time=datetime.fromisoformat(episode_data["end_time"]),
                event_ids=episode_data["event_ids"],
                title=episode_data["title"],
                summary=episode_data["summary"],
                importance=episode_data["importance"],
                coherence_score=episode_data["coherence_score"],
                participants=set(episode_data["participants"]),
                location=episode_data["location"],
                outcome=episode_data["outcome"],
                emotional_impact=episode_data["emotional_impact"]
            )
            episodic_mem.episodes[episode_id] = episode
        
        # Deserialize causal relations
        for relation_data in data["causal_relations"]:
            relation = CausalRelation(
                source_event_id=relation_data["source_event_id"],
                target_event_id=relation_data["target_event_id"],
                relation_type=CausalRelationType(relation_data["relation_type"]),
                strength=relation_data["strength"],
                confidence=relation_data["confidence"],
                created_at=datetime.fromisoformat(relation_data["created_at"]),
                evidence=relation_data["evidence"]
            )
            episodic_mem.causal_relations.append(relation)
        
        # Deserialize indices
        episodic_mem.temporal_index = [
            (datetime.fromisoformat(ts), event_id) 
            for ts, event_id in data["temporal_index"]
        ]
        
        episodic_mem.participant_index = data["participant_index"]
        episodic_mem.type_index = {
            EpisodeType(k): v for k, v in data["type_index"].items()
        }
        
        return episodic_mem


# Example usage
if __name__ == "__main__":
    # Example of EpisodicMemory usage
    episodic_mem = EpisodicMemory(max_episodes=50)
    
    # Add some test events
    event1 = episodic_mem.add_event(
        "Started conversation with Maria about dating preferences",
        "conversation", 0.8, participants={"Maria"}, location="Living Room"
    )
    
    event2 = episodic_mem.add_event(
        "Maria expressed interest in outdoor activities",
        "conversation", 0.7, participants={"Maria"}, location="Living Room"
    )
    
    event3 = episodic_mem.add_event(
        "Decided to suggest hiking for first date",
        "thought", 0.6, emotional_valence=0.3
    )
    
    # Add causal relation
    episodic_mem.add_causal_relation(
        event2, event3, CausalRelationType.CAUSES, 0.9, 0.8,
        evidence=["Maria's preference directly influenced decision"]
    )
    
    print("Episodes created:", len(episodic_mem.episodes))
    print("Events stored:", len(episodic_mem.events))
    
    # Get episodes by participant
    maria_episodes = episodic_mem.get_episodes_by_participant("Maria")
    print(f"Episodes with Maria: {len(maria_episodes)}")
    
    if maria_episodes:
        print("Episode narrative:")
        print(episodic_mem.get_episode_narrative(maria_episodes[0].episode_id))