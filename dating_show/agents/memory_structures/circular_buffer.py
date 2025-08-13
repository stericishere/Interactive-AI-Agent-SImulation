"""
File: circular_buffer.py
Description: CircularBuffer memory structure with LangGraph reducer integration for working memory.
Enhanced PIANO architecture implementation for Phase 1.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from .security_utils import SecurityValidator, SecurityError


class CircularBufferReducer:
    """
    LangGraph reducer for circular buffer behavior in working memory.
    Maintains a fixed-size buffer with automatic pruning of oldest entries.
    """
    
    def __init__(self, max_size: int = 20):
        """
        Initialize CircularBufferReducer.
        
        Args:
            max_size: Maximum number of entries to maintain in buffer
        """
        self.max_size = max_size
    
    def __call__(self, current: List[Dict], updates: List[Dict]) -> List[Dict]:
        """
        LangGraph reducer function for circular buffer behavior.
        
        Args:
            current: Current state of the memory buffer
            updates: New memory entries to add
        
        Returns:
            Updated memory buffer with size constraint enforced
        """
        # Combine current memories with new updates
        combined = (current or []) + (updates or [])
        
        # If we exceed max size, keep only the most recent entries
        if len(combined) > self.max_size:
            return combined[-self.max_size:]
        
        return combined


class CircularBuffer:
    """
    Enhanced CircularBuffer implementation for working memory management.
    Integrates with LangGraph StateGraph for concurrent agent processing.
    """
    
    def __init__(self, max_size: int = 20, retention_minutes: int = 60):
        """
        Initialize CircularBuffer.
        
        Args:
            max_size: Maximum number of entries to store
            retention_minutes: Time in minutes to retain entries
        """
        self.max_size = max_size
        self.retention_minutes = retention_minutes
        self.buffer: List[Dict[str, Any]] = []
        self._next_id = 1
    
    def add_memory(self, content: str, memory_type: str = "event", 
                   importance: float = 0.5, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Add a new memory entry to the circular buffer.
        
        Args:
            content: The memory content/description
            memory_type: Type of memory (event, thought, conversation, etc.)
            importance: Importance score (0.0 to 1.0)
            metadata: Additional metadata dictionary
        
        Returns:
            The created memory entry
        """
        # Security validation and sanitization
        safe_content, safe_type, safe_importance, safe_metadata = SecurityValidator.sanitize_memory_data(
            content, memory_type, importance, metadata
        )
        
        timestamp = datetime.now()
        
        memory_entry = {
            "id": f"mem_{self._next_id}",
            "content": safe_content,
            "type": safe_type,
            "timestamp": timestamp,
            "importance": safe_importance,
            "metadata": safe_metadata or {},
            "accessed_count": 0,
            "last_accessed": timestamp
        }
        
        self._next_id += 1
        
        # Add to buffer
        self.buffer.append(memory_entry)
        
        # Maintain size constraint
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # Remove oldest entry
        
        return memory_entry
    
    def get_recent_memories(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent memories from the buffer.
        
        Args:
            count: Number of recent memories to retrieve
        
        Returns:
            List of recent memory entries
        """
        return self.buffer[-count:] if self.buffer else []
    
    def get_memories_by_type(self, memory_type: str) -> List[Dict[str, Any]]:
        """
        Get all memories of a specific type.
        
        Args:
            memory_type: The type of memories to retrieve
        
        Returns:
            List of memory entries of the specified type
        """
        return [mem for mem in self.buffer if mem["type"] == memory_type]
    
    def get_important_memories(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get memories above an importance threshold.
        
        Args:
            threshold: Minimum importance score
        
        Returns:
            List of important memory entries
        """
        return [mem for mem in self.buffer if mem["importance"] >= threshold]
    
    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        """
        Simple text search in memory contents.
        
        Args:
            query: Search query string
        
        Returns:
            List of matching memory entries
        """
        query_lower = query.lower()
        matches = []
        
        for memory in self.buffer:
            if query_lower in memory["content"].lower():
                # Update access tracking
                memory["accessed_count"] += 1
                memory["last_accessed"] = datetime.now()
                matches.append(memory)
        
        return matches
    
    def cleanup_expired_memories(self) -> List[Dict[str, Any]]:
        """
        Remove memories older than retention period.
        
        Returns:
            List of removed memory entries
        """
        current_time = datetime.now()
        expired_memories = []
        active_memories = []
        
        for memory in self.buffer:
            age_minutes = (current_time - memory["timestamp"]).total_seconds() / 60
            
            if age_minutes > self.retention_minutes:
                expired_memories.append(memory)
            else:
                active_memories.append(memory)
        
        self.buffer = active_memories
        return expired_memories
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the memory buffer.
        
        Returns:
            Dictionary containing buffer statistics
        """
        if not self.buffer:
            return {
                "total_memories": 0,
                "memory_types": {},
                "avg_importance": 0.0,
                "oldest_memory": None,
                "newest_memory": None
            }
        
        memory_types = {}
        total_importance = 0
        
        for memory in self.buffer:
            mem_type = memory["type"]
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
            total_importance += memory["importance"]
        
        return {
            "total_memories": len(self.buffer),
            "memory_types": memory_types,
            "avg_importance": total_importance / len(self.buffer),
            "oldest_memory": self.buffer[0]["timestamp"] if self.buffer else None,
            "newest_memory": self.buffer[-1]["timestamp"] if self.buffer else None,
            "buffer_utilization": len(self.buffer) / self.max_size
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert buffer to dictionary for serialization.
        
        Returns:
            Dictionary representation of the buffer
        """
        return {
            "max_size": self.max_size,
            "retention_minutes": self.retention_minutes,
            "buffer": [
                {
                    **memory,
                    "timestamp": memory["timestamp"].isoformat(),
                    "last_accessed": memory["last_accessed"].isoformat()
                }
                for memory in self.buffer
            ],
            "next_id": self._next_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CircularBuffer':
        """
        Create CircularBuffer from dictionary representation.
        
        Args:
            data: Dictionary containing buffer data
        
        Returns:
            CircularBuffer instance
        """
        buffer = cls(
            max_size=data["max_size"],
            retention_minutes=data["retention_minutes"]
        )
        
        # Handle backward compatibility for next_id
        buffer._next_id = data.get("next_id", 1)
        
        # Handle backward compatibility for buffer vs memories field
        memories_data = data.get("buffer") or data.get("memories", [])
        
        for mem_data in memories_data:
            memory = mem_data.copy()
            # Handle datetime conversion if needed
            if isinstance(memory.get("timestamp"), str):
                memory["timestamp"] = datetime.fromisoformat(memory["timestamp"])
            if isinstance(memory.get("last_accessed"), str):
                memory["last_accessed"] = datetime.fromisoformat(memory["last_accessed"])
            elif "last_accessed" not in memory:
                # Add missing last_accessed field for legacy data
                memory["last_accessed"] = memory.get("timestamp", datetime.now())
            
            buffer.buffer.append(memory)
        
        return buffer
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save buffer to JSON file with security validation.
        
        Args:
            filepath: Path to save the buffer data
            
        Raises:
            SecurityError: If filepath is dangerous
        """
        # Validate file path for security
        safe_filepath = SecurityValidator.validate_filepath(filepath)
        
        with open(safe_filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'CircularBuffer':
        """
        Load buffer from JSON file with security validation.
        
        Args:
            filepath: Path to load the buffer data from
        
        Returns:
            CircularBuffer instance
            
        Raises:
            SecurityError: If filepath is dangerous
        """
        # Validate file path for security
        safe_filepath = SecurityValidator.validate_filepath(filepath)
        
        with open(safe_filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get memory entry by index."""
        return self.buffer[index]
    
    def __iter__(self):
        """Make buffer iterable."""
        return iter(self.buffer)


# Helper function for LangGraph integration
def create_circular_buffer_reducer(max_size: int = 20) -> CircularBufferReducer:
    """
    Create a CircularBufferReducer for use in LangGraph StateGraph.
    
    Args:
        max_size: Maximum size of the circular buffer
    
    Returns:
        CircularBufferReducer instance
    """
    return CircularBufferReducer(max_size=max_size)


# Example usage for LangGraph StateGraph
if __name__ == "__main__":
    # Example of how to use CircularBuffer
    buffer = CircularBuffer(max_size=5, retention_minutes=30)
    
    # Add some test memories
    buffer.add_memory("Agent woke up and started daily routine", "event", 0.6)
    buffer.add_memory("Thinking about breakfast options", "thought", 0.4)
    buffer.add_memory("Talked to Maria about dating preferences", "conversation", 0.8)
    
    print("Buffer Summary:", buffer.get_memory_summary())
    print("Recent Memories:", buffer.get_recent_memories(3))
    print("Important Memories:", buffer.get_important_memories(0.7))