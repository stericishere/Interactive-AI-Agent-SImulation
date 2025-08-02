"""
File: temporal_memory.py
Description: TemporalMemory with time-based decay for enhanced PIANO architecture.
Integrates with LangGraph Store API for cross-agent temporal context sharing.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import math
from collections import defaultdict


class TemporalMemory:
    """
    Time-indexed memory storage with decay functions and temporal querying.
    Supports integration with LangGraph Store API for cross-thread access.
    """
    
    def __init__(self, retention_hours: int = 1, decay_rate: float = 0.1, 
                 temporal_resolution: int = 60):
        """
        Initialize TemporalMemory.
        
        Args:
            retention_hours: Base retention period in hours
            decay_rate: Rate of memory strength decay (0.0 to 1.0)
            temporal_resolution: Time resolution in seconds for indexing
        """
        self.retention_hours = retention_hours
        self.decay_rate = decay_rate
        self.temporal_resolution = temporal_resolution
        
        # Time-indexed memory storage
        self.memories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)  # time_key -> memory_ids
        
        # Memory metadata
        self.memory_metadata: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1
    
    def _get_time_key(self, timestamp: datetime) -> str:
        """
        Generate time key for temporal indexing.
        
        Args:
            timestamp: Datetime to generate key for
        
        Returns:
            Time key string for indexing
        """
        # Round to temporal resolution
        total_seconds = int(timestamp.timestamp())
        rounded_seconds = (total_seconds // self.temporal_resolution) * self.temporal_resolution
        rounded_time = datetime.fromtimestamp(rounded_seconds)
        
        return rounded_time.strftime("%Y-%m-%d_%H:%M:%S")
    
    def add_memory(self, content: str, memory_type: str = "event", 
                   importance: float = 0.5, context: Optional[Dict] = None,
                   timestamp: Optional[datetime] = None) -> str:
        """
        Add a memory with temporal indexing.
        
        Args:
            content: Memory content
            memory_type: Type of memory (event, thought, observation, etc.)
            importance: Initial importance score
            context: Additional context information
            timestamp: Memory timestamp (defaults to now)
        
        Returns:
            Memory ID
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        memory_id = f"temp_mem_{self._next_id}"
        self._next_id += 1
        
        time_key = self._get_time_key(timestamp)
        
        memory_entry = {
            "id": memory_id,
            "content": content,
            "type": memory_type,
            "timestamp": timestamp,
            "importance": importance,
            "initial_strength": importance,
            "context": context or {},
            "access_count": 0,
            "last_accessed": timestamp
        }
        
        # Store in time-indexed structure
        self.memories[time_key].append(memory_entry)
        self.temporal_index[time_key].append(memory_id)
        
        # Store metadata for quick access
        self.memory_metadata[memory_id] = {
            "time_key": time_key,
            "created": timestamp,
            "type": memory_type
        }
        
        return memory_id
    
    def get_memory_strength(self, memory_id: str, current_time: Optional[datetime] = None) -> float:
        """
        Calculate current memory strength with decay applied.
        
        Args:
            memory_id: ID of the memory
            current_time: Current time for decay calculation
        
        Returns:
            Current memory strength (0.0 to 1.0)
        """
        if current_time is None:
            current_time = datetime.now()
        
        metadata = self.memory_metadata.get(memory_id)
        if not metadata:
            return 0.0
        
        time_key = metadata["time_key"]
        memory_entry = None
        
        # Find the memory entry
        for memory in self.memories[time_key]:
            if memory["id"] == memory_id:
                memory_entry = memory
                break
        
        if not memory_entry:
            return 0.0
        
        # Calculate time-based decay
        age_hours = (current_time - memory_entry["timestamp"]).total_seconds() / 3600
        
        # Exponential decay formula
        decay_factor = math.exp(-self.decay_rate * age_hours)
        current_strength = memory_entry["initial_strength"] * decay_factor
        
        # Access-based reinforcement
        access_boost = min(0.1 * memory_entry["access_count"], 0.3)
        
        return min(current_strength + access_boost, 1.0)
    
    def retrieve_memories_by_timerange(self, start_time: datetime, end_time: datetime,
                                     memory_type: Optional[str] = None,
                                     min_strength: float = 0.1) -> List[Dict[str, Any]]:
        """
        Retrieve memories within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            memory_type: Filter by memory type (optional)
            min_strength: Minimum memory strength threshold
        
        Returns:
            List of memory entries with current strength
        """
        current_time = datetime.now()
        results = []
        
        # Generate all time keys in range
        current = start_time
        while current <= end_time:
            time_key = self._get_time_key(current)
            
            for memory in self.memories.get(time_key, []):
                # Check time range (precise)
                if start_time <= memory["timestamp"] <= end_time:
                    # Check memory type filter
                    if memory_type and memory["type"] != memory_type:
                        continue
                    
                    # Calculate current strength
                    strength = self.get_memory_strength(memory["id"], current_time)
                    
                    if strength >= min_strength:
                        # Update access tracking
                        memory["access_count"] += 1
                        memory["last_accessed"] = current_time
                        
                        result_memory = memory.copy()
                        result_memory["current_strength"] = strength
                        results.append(result_memory)
            
            current += timedelta(seconds=self.temporal_resolution)
        
        # Sort by strength and recency
        results.sort(key=lambda x: (x["current_strength"], x["timestamp"]), reverse=True)
        return results
    
    def retrieve_recent_memories(self, hours_back: int = 1, memory_type: Optional[str] = None,
                               limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent memories within specified hours.
        
        Args:
            hours_back: Number of hours to look back
            memory_type: Filter by memory type
            limit: Maximum number of memories to return
        
        Returns:
            List of recent memory entries
        """
        current_time = datetime.now()
        start_time = current_time - timedelta(hours=hours_back)
        
        memories = self.retrieve_memories_by_timerange(
            start_time, current_time, memory_type, min_strength=0.0
        )
        
        return memories[:limit]
    
    def retrieve_memories_by_pattern(self, temporal_pattern: str, 
                                   reference_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Retrieve memories matching temporal patterns (daily, weekly, etc.).
        
        Args:
            temporal_pattern: Pattern type ('same_hour', 'same_day', 'same_weekday')
            reference_time: Reference time for pattern matching
        
        Returns:
            List of matching memory entries
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        results = []
        current_time = datetime.now()
        
        for time_key, memories in self.memories.items():
            for memory in memories:
                match = False
                mem_time = memory["timestamp"]
                
                if temporal_pattern == "same_hour":
                    match = mem_time.hour == reference_time.hour
                elif temporal_pattern == "same_day":
                    match = (mem_time.month == reference_time.month and 
                            mem_time.day == reference_time.day)
                elif temporal_pattern == "same_weekday":
                    match = mem_time.weekday() == reference_time.weekday()
                
                if match:
                    strength = self.get_memory_strength(memory["id"], current_time)
                    if strength > 0.1:  # Only include non-trivial memories
                        result_memory = memory.copy()
                        result_memory["current_strength"] = strength
                        results.append(result_memory)
        
        results.sort(key=lambda x: x["current_strength"], reverse=True)
        return results
    
    def cleanup_expired_memories(self, strength_threshold: float = 0.05) -> List[str]:
        """
        Remove memories that have decayed below threshold.
        
        Args:
            strength_threshold: Minimum strength to retain memory
        
        Returns:
            List of removed memory IDs
        """
        current_time = datetime.now()
        removed_ids = []
        
        for time_key in list(self.memories.keys()):
            active_memories = []
            
            for memory in self.memories[time_key]:
                strength = self.get_memory_strength(memory["id"], current_time)
                
                if strength >= strength_threshold:
                    active_memories.append(memory)
                else:
                    removed_ids.append(memory["id"])
                    # Clean up metadata
                    if memory["id"] in self.memory_metadata:
                        del self.memory_metadata[memory["id"]]
            
            if active_memories:
                self.memories[time_key] = active_memories
                # Update temporal index
                self.temporal_index[time_key] = [m["id"] for m in active_memories]
            else:
                # Remove empty time slots
                del self.memories[time_key]
                if time_key in self.temporal_index:
                    del self.temporal_index[time_key]
        
        return removed_ids
    
    def get_temporal_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get summary of temporal memory patterns.
        
        Args:
            hours_back: Hours to analyze
        
        Returns:
            Dictionary with temporal statistics
        """
        current_time = datetime.now()
        start_time = current_time - timedelta(hours=hours_back)
        
        recent_memories = self.retrieve_memories_by_timerange(start_time, current_time)
        
        # Analyze temporal patterns
        hourly_distribution = defaultdict(int)
        type_distribution = defaultdict(int)
        strength_distribution = {"strong": 0, "medium": 0, "weak": 0}
        
        for memory in recent_memories:
            hour = memory["timestamp"].hour
            hourly_distribution[hour] += 1
            type_distribution[memory["type"]] += 1
            
            strength = memory["current_strength"]
            if strength > 0.7:
                strength_distribution["strong"] += 1
            elif strength > 0.3:
                strength_distribution["medium"] += 1
            else:
                strength_distribution["weak"] += 1
        
        return {
            "total_memories": len(recent_memories),
            "time_span_hours": hours_back,
            "hourly_distribution": dict(hourly_distribution),
            "type_distribution": dict(type_distribution),
            "strength_distribution": dict(strength_distribution),
            "avg_strength": sum(m["current_strength"] for m in recent_memories) / len(recent_memories) if recent_memories else 0
        }
    
    def consolidate_memories(self, similarity_threshold: float = 0.8, 
                           min_strength: float = 0.0, max_memories: int = None) -> int:
        """
        Consolidate similar memories to reduce redundancy.
        
        Args:
            similarity_threshold: Threshold for considering memories similar
            min_strength: Minimum strength threshold for memories to keep
            max_memories: Maximum number of memories to retain after consolidation
        
        Returns:
            Number of memories consolidated
        """
        # This is a simplified consolidation - in practice would use embeddings
        consolidated_count = 0
        consolidated_ids = []
        
        # First, filter by minimum strength
        for time_key in list(self.memories.keys()):
            memories = self.memories[time_key]
            filtered_memories = []
            
            for memory in memories:
                current_strength = self._calculate_current_strength(memory["initial_strength"], memory["timestamp"])
                if current_strength >= min_strength:
                    filtered_memories.append(memory)
                else:
                    consolidated_ids.append(memory["id"])
                    consolidated_count += 1
            
            if filtered_memories:
                self.memories[time_key] = filtered_memories
            else:
                del self.memories[time_key]
        
        # Then consolidate similar memories
        for time_key in self.memories:
            memories = self.memories[time_key]
            
            # Group by type and look for similar content
            type_groups = defaultdict(list)
            for memory in memories:
                type_groups[memory["type"]].append(memory)
            
            for mem_list in type_groups.values():
                if len(mem_list) > 1:
                    # Simple similarity based on content overlap
                    for i, mem1 in enumerate(mem_list):
                        for mem2 in mem_list[i+1:]:
                            content1_words = set(mem1["content"].lower().split())
                            content2_words = set(mem2["content"].lower().split())
                            
                            if content1_words and content2_words:
                                overlap = len(content1_words & content2_words)
                                union = len(content1_words | content2_words)
                                similarity = overlap / union if union > 0 else 0
                                
                                if similarity > similarity_threshold:
                                    # Merge memories - keep the more important one
                                    if mem1["importance"] >= mem2["importance"]:
                                        mem1["content"] += f" [Consolidated: {mem2['content']}]"
                                        mem1["importance"] = max(mem1["importance"], mem2["importance"])
                                        consolidated_ids.append(mem2["id"])
                                        consolidated_count += 1
                                    else:
                                        mem2["content"] += f" [Consolidated: {mem1['content']}]"
                                        mem2["importance"] = max(mem1["importance"], mem2["importance"])
                                        consolidated_ids.append(mem1["id"])
                                        consolidated_count += 1
        
        # Remove consolidated memories
        for memory_id in consolidated_ids:
            self._remove_memory_by_id(memory_id)
        
        # Apply max_memories limit if specified
        if max_memories is not None:
            total_memories = sum(len(memories) for memories in self.memories.values())
            if total_memories > max_memories:
                # Remove oldest, least important memories
                all_memories = []
                for time_key, memories in self.memories.items():
                    for memory in memories:
                        all_memories.append((time_key, memory))
                
                # Sort by importance and recency
                all_memories.sort(key=lambda x: (x[1]["importance"], x[1]["timestamp"]), reverse=True)
                
                # Keep only the top max_memories
                memories_to_keep = all_memories[:max_memories]
                memories_to_remove = all_memories[max_memories:]
                
                # Rebuild memories structure
                new_memories = defaultdict(list)
                for time_key, memory in memories_to_keep:
                    new_memories[time_key].append(memory)
                
                consolidated_count += len(memories_to_remove)
                self.memories = new_memories
        
        return consolidated_count
    
    def _calculate_current_strength(self, initial_strength: float, timestamp: datetime) -> float:
        """
        Calculate current strength of a memory with decay applied.
        
        Args:
            initial_strength: Original strength of the memory
            timestamp: When the memory was created
        
        Returns:
            Current strength after decay
        """
        current_time = datetime.now()
        time_elapsed = (current_time - timestamp).total_seconds() / 3600  # hours
        
        # Exponential decay: strength = initial * e^(-decay_rate * time)
        current_strength = initial_strength * math.exp(-self.decay_rate * time_elapsed)
        return max(current_strength, 0.0)
    
    def _remove_memory_by_id(self, memory_id: str) -> bool:
        """
        Remove a memory by its ID.
        
        Args:
            memory_id: ID of memory to remove
        
        Returns:
            True if memory was removed, False if not found
        """
        metadata = self.memory_metadata.get(memory_id)
        if not metadata:
            return False
        
        time_key = metadata["time_key"]
        
        # Remove from memories
        self.memories[time_key] = [
            m for m in self.memories[time_key] if m["id"] != memory_id
        ]
        
        # Remove from temporal index
        if time_key in self.temporal_index:
            self.temporal_index[time_key] = [
                mid for mid in self.temporal_index[time_key] if mid != memory_id
            ]
        
        # Remove metadata
        del self.memory_metadata[memory_id]
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        # Convert datetime objects to ISO strings
        serialized_memories = {}
        for time_key, memories in self.memories.items():
            serialized_memories[time_key] = [
                {
                    **memory,
                    "timestamp": memory["timestamp"].isoformat(),
                    "last_accessed": memory["last_accessed"].isoformat()
                }
                for memory in memories
            ]
        
        serialized_metadata = {}
        for mem_id, metadata in self.memory_metadata.items():
            serialized_metadata[mem_id] = {
                **metadata,
                "created": metadata["created"].isoformat()
            }
        
        return {
            "retention_hours": self.retention_hours,
            "decay_rate": self.decay_rate,
            "temporal_resolution": self.temporal_resolution,
            "memories": serialized_memories,
            "temporal_index": dict(self.temporal_index),
            "memory_metadata": serialized_metadata,
            "next_id": self._next_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TemporalMemory':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary containing temporal memory data
        
        Returns:
            TemporalMemory instance
        """
        temporal_mem = cls(
            retention_hours=data["retention_hours"],
            decay_rate=data["decay_rate"],
            temporal_resolution=data["temporal_resolution"]
        )
        
        temporal_mem._next_id = data["next_id"]
        
        # Deserialize memories
        for time_key, memories in data["memories"].items():
            temporal_mem.memories[time_key] = [
                {
                    **memory,
                    "timestamp": datetime.fromisoformat(memory["timestamp"]),
                    "last_accessed": datetime.fromisoformat(memory["last_accessed"])
                }
                for memory in memories
            ]
        
        # Deserialize temporal index
        for time_key, mem_ids in data["temporal_index"].items():
            temporal_mem.temporal_index[time_key] = mem_ids
        
        # Deserialize metadata
        for mem_id, metadata in data["memory_metadata"].items():
            temporal_mem.memory_metadata[mem_id] = {
                **metadata,
                "created": datetime.fromisoformat(metadata["created"])
            }
        
        return temporal_mem


# Example usage
if __name__ == "__main__":
    # Example of TemporalMemory usage
    temporal_mem = TemporalMemory(retention_hours=2, decay_rate=0.1)
    
    # Add some test memories
    mem1 = temporal_mem.add_memory("Agent started morning routine", "event", 0.8)
    mem2 = temporal_mem.add_memory("Thinking about breakfast", "thought", 0.5)
    mem3 = temporal_mem.add_memory("Noticed Maria looking upset", "observation", 0.7)
    
    print("Recent memories:", len(temporal_mem.retrieve_recent_memories()))
    print("Temporal summary:", temporal_mem.get_temporal_summary())