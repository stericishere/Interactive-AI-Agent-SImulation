"""
File: memory_retrieval.py
Description: Memory Retrieval Optimization module for enhanced PIANO architecture.
Provides fast memory lookup algorithms, context-based memory activation,
and relevance scoring for memory selection.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
import time
import math
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import re

from .langgraph_base_module import LangGraphBaseModule, ModuleExecutionConfig, ExecutionTimeScale, ModulePriority
from ..enhanced_agent_state import EnhancedAgentState, EnhancedAgentStateManager
from ..memory_structures.circular_buffer import CircularBuffer
from ..memory_structures.temporal_memory import TemporalMemory
from ..memory_structures.episodic_memory import EpisodicMemory, Episode
from ..memory_structures.semantic_memory import SemanticMemory, SemanticConcept


class RetrievalStrategy(Enum):
    """Memory retrieval strategies."""
    RECENCY_BASED = "recency_based"          # Most recent memories first
    IMPORTANCE_BASED = "importance_based"    # Most important memories first  
    RELEVANCE_BASED = "relevance_based"      # Most relevant to context
    ACTIVATION_BASED = "activation_based"    # Most activated memories
    MIXED = "mixed"                         # Combined scoring approach


class RetrievalContext(Enum):
    """Context types for memory retrieval."""
    CONVERSATION = "conversation"            # Retrieving for conversation
    DECISION_MAKING = "decision_making"      # Retrieving for decisions
    REFLECTION = "reflection"               # Retrieving for reflection
    PLANNING = "planning"                   # Retrieving for planning
    GENERAL = "general"                     # General purpose retrieval


@dataclass
class RetrievalQuery:
    """Represents a memory retrieval query."""
    query_id: str
    query_text: str
    query_context: RetrievalContext
    strategy: RetrievalStrategy
    max_results: int = 10
    importance_threshold: float = 0.1
    recency_weight: float = 0.3
    relevance_weight: float = 0.4
    importance_weight: float = 0.3
    time_window_hours: Optional[int] = None
    memory_types: Optional[List[str]] = None
    participants: Optional[Set[str]] = None


@dataclass
class RetrievalResult:
    """Result of memory retrieval with scoring."""
    memory_id: str
    memory_system: str  # working, temporal, episodic, semantic
    content: str
    memory_type: str
    importance: float
    relevance_score: float
    recency_score: float
    activation_score: float
    combined_score: float
    metadata: Dict[str, Any]
    timestamp: datetime


class MemoryRetrievalModule(LangGraphBaseModule):
    """
    Memory retrieval optimization module providing fast lookup algorithms
    and context-based memory activation.
    """
    
    def __init__(self, state_manager: Optional[EnhancedAgentStateManager] = None):
        """
        Initialize Memory Retrieval Module.
        
        Args:
            state_manager: Enhanced agent state manager
        """
        config = ModuleExecutionConfig(
            time_scale=ExecutionTimeScale.FAST,
            priority=ModulePriority.HIGH,
            can_run_parallel=True,
            requires_completion=False,
            max_execution_time=0.5
        )
        
        super().__init__("memory_retrieval", config, state_manager)
        
        # Retrieval settings
        self.default_strategy = RetrievalStrategy.MIXED
        self.cache_enabled = True
        self.cache_ttl_seconds = 60
        self.max_cache_size = 100
        
        # Query cache for performance optimization
        self.query_cache: Dict[str, Tuple[List[RetrievalResult], datetime]] = {}
        
        # Performance tracking
        self.retrieval_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_retrieval_time_ms": 0.0,
            "avg_results_per_query": 0.0,
            "last_query": None
        }
        
        # Context-specific weights
        self.context_weights = {
            RetrievalContext.CONVERSATION: {
                "recency": 0.4, "relevance": 0.4, "importance": 0.2
            },
            RetrievalContext.DECISION_MAKING: {
                "recency": 0.2, "relevance": 0.3, "importance": 0.5
            },
            RetrievalContext.REFLECTION: {
                "recency": 0.1, "relevance": 0.4, "importance": 0.5
            },
            RetrievalContext.PLANNING: {
                "recency": 0.3, "relevance": 0.4, "importance": 0.3
            },
            RetrievalContext.GENERAL: {
                "recency": 0.3, "relevance": 0.4, "importance": 0.3
            }
        }
        
        self.logger = logging.getLogger("MemoryRetrieval")
    
    def process_state(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Process agent state for memory retrieval optimization.
        
        Args:
            state: Current enhanced agent state
        
        Returns:
            Dictionary with retrieval optimization results
        """
        start_time = time.time()
        
        try:
            # Analyze current context for proactive memory activation
            context_analysis = self._analyze_current_context(state)
            
            # Perform context-based memory activation
            activation_results = self._activate_contextual_memories(context_analysis)
            
            # Update memory access patterns
            self._update_access_patterns(state)
            
            # Clean up query cache if needed
            self._cleanup_cache()
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "output_data": {
                    "context_analysis": context_analysis,
                    "activated_memories": len(activation_results),
                    "cache_size": len(self.query_cache)
                },
                "performance_metrics": {
                    "processing_time_ms": processing_time,
                    "cache_hit_rate": self._calculate_cache_hit_rate(),
                    "avg_retrieval_time": self.retrieval_stats["avg_retrieval_time_ms"]
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error in memory retrieval processing: {str(e)}")
            return {
                "output_data": {"error": str(e)},
                "performance_metrics": {"processing_time_ms": (time.time() - start_time) * 1000}
            }
    
    def retrieve_memories(self, query: Union[str, RetrievalQuery]) -> List[RetrievalResult]:
        """
        Retrieve memories based on query.
        
        Args:
            query: Query string or RetrievalQuery object
        
        Returns:
            List of retrieved memories with scores
        """
        start_time = time.time()
        
        # Convert string query to RetrievalQuery
        if isinstance(query, str):
            query_obj = RetrievalQuery(
                query_id=f"query_{int(time.time())}",
                query_text=query,
                query_context=RetrievalContext.GENERAL,
                strategy=self.default_strategy
            )
        else:
            query_obj = query
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query_obj)
            if self.cache_enabled and cache_key in self.query_cache:
                cached_results, cache_time = self.query_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < self.cache_ttl_seconds:
                    self.retrieval_stats["cache_hits"] += 1
                    return cached_results
            
            # Perform retrieval
            results = self._execute_retrieval(query_obj)
            
            # Cache results
            if self.cache_enabled:
                self.query_cache[cache_key] = (results, datetime.now())
            
            # Update statistics
            retrieval_time = (time.time() - start_time) * 1000
            self._update_retrieval_stats(retrieval_time, len(results))
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error retrieving memories: {str(e)}")
            return []
    
    def _analyze_current_context(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Analyze current agent context to determine relevant memory activation.
        
        Args:
            state: Current agent state
        
        Returns:
            Context analysis results
        """
        context = {
            "current_activity": state.get("current_activity", "unknown"),
            "current_location": state.get("current_location", "unknown"),
            "conversation_partners": list(state.get("conversation_partners", set())),
            "recent_interactions": state.get("recent_interactions", []),
            "emotional_state": state.get("emotional_state", {}),
            "goals": state.get("goals", []),
            "context_keywords": [],
            "suggested_retrieval_context": RetrievalContext.GENERAL
        }
        
        # Extract context keywords
        context_text = f"{context['current_activity']} {' '.join(context['goals'])}"
        context["context_keywords"] = self._extract_keywords(context_text)
        
        # Determine suggested retrieval context
        if context["conversation_partners"]:
            context["suggested_retrieval_context"] = RetrievalContext.CONVERSATION
        elif "decision" in context["current_activity"].lower() or "choose" in context["current_activity"].lower():
            context["suggested_retrieval_context"] = RetrievalContext.DECISION_MAKING
        elif "plan" in context["current_activity"].lower():
            context["suggested_retrieval_context"] = RetrievalContext.PLANNING
        elif "reflect" in context["current_activity"].lower() or "think" in context["current_activity"].lower():
            context["suggested_retrieval_context"] = RetrievalContext.REFLECTION
        
        return context
    
    def _activate_contextual_memories(self, context_analysis: Dict[str, Any]) -> List[str]:
        """
        Proactively activate memories relevant to current context.
        
        Args:
            context_analysis: Results from context analysis
        
        Returns:
            List of activated memory IDs
        """
        if not self.state_manager:
            return []
        
        activated_memories = []
        
        try:
            # Activate semantic concepts related to current context
            for keyword in context_analysis["context_keywords"]:
                self.state_manager.semantic_memory.activate_concept(keyword, 0.5)
            
            # Activate memories related to conversation partners
            for partner in context_analysis["conversation_partners"]:
                self.state_manager.semantic_memory.activate_concept(partner, 0.7)
                activated_memories.append(f"concept_{partner}")
            
            # Activate location-related memories
            location = context_analysis["current_location"]
            if location != "unknown":
                self.state_manager.semantic_memory.activate_concept(location, 0.4)
                activated_memories.append(f"concept_{location}")
        
        except Exception as e:
            self.logger.error(f"Error activating contextual memories: {str(e)}")
        
        return activated_memories
    
    def _execute_retrieval(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Execute memory retrieval based on query parameters.
        
        Args:
            query: Retrieval query
        
        Returns:
            List of retrieval results
        """
        if not self.state_manager:
            return []
        
        all_results = []
        
        # Retrieve from each memory system
        working_results = self._retrieve_from_working_memory(query)
        temporal_results = self._retrieve_from_temporal_memory(query)
        episodic_results = self._retrieve_from_episodic_memory(query)
        semantic_results = self._retrieve_from_semantic_memory(query)
        
        all_results.extend(working_results)
        all_results.extend(temporal_results)
        all_results.extend(episodic_results)
        all_results.extend(semantic_results)
        
        # Apply strategy-based scoring and filtering
        scored_results = self._apply_retrieval_strategy(all_results, query)
        
        # Sort by combined score and limit results
        scored_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return scored_results[:query.max_results]
    
    def _retrieve_from_working_memory(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve memories from working memory (circular buffer)."""
        results = []
        
        try:
            # Search working memory
            if hasattr(self.state_manager, 'circular_buffer'):
                matching_memories = self.state_manager.circular_buffer.search_memories(query.query_text)
                
                for memory in matching_memories:
                    if self._passes_filters(memory, query):
                        result = RetrievalResult(
                            memory_id=memory.get("id", "unknown"),
                            memory_system="working",
                            content=memory.get("content", ""),
                            memory_type=memory.get("type", "unknown"),
                            importance=memory.get("importance", 0.5),
                            relevance_score=self._calculate_relevance_score(memory.get("content", ""), query.query_text),
                            recency_score=self._calculate_recency_score(memory.get("timestamp")),
                            activation_score=0.8,  # Working memory is always highly activated
                            combined_score=0.0,  # Will be calculated later
                            metadata=memory.get("metadata", {}),
                            timestamp=memory.get("timestamp", datetime.now())
                        )
                        results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error retrieving from working memory: {str(e)}")
        
        return results
    
    def _retrieve_from_temporal_memory(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve memories from temporal memory."""
        results = []
        
        try:
            if hasattr(self.state_manager, 'temporal_memory'):
                # Use time window if specified
                if query.time_window_hours:
                    end_time = datetime.now()
                    start_time = end_time - timedelta(hours=query.time_window_hours)
                    memories = self.state_manager.temporal_memory.retrieve_memories_by_timerange(
                        start_time, end_time, min_strength=query.importance_threshold
                    )
                else:
                    memories = self.state_manager.temporal_memory.retrieve_recent_memories(
                        hours_back=24, limit=50
                    )
                
                for memory in memories:
                    if self._passes_filters(memory, query):
                        # Filter by content relevance
                        relevance = self._calculate_relevance_score(memory.get("content", ""), query.query_text)
                        if relevance > 0.1:  # Minimum relevance threshold
                            result = RetrievalResult(
                                memory_id=memory.get("id", "unknown"),
                                memory_system="temporal",
                                content=memory.get("content", ""),
                                memory_type=memory.get("type", "unknown"),
                                importance=memory.get("importance", 0.5),
                                relevance_score=relevance,
                                recency_score=self._calculate_recency_score(memory.get("timestamp")),
                                activation_score=memory.get("current_strength", 0.5),
                                combined_score=0.0,
                                metadata=memory.get("context", {}),
                                timestamp=memory.get("timestamp", datetime.now())
                            )
                            results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error retrieving from temporal memory: {str(e)}")
        
        return results
    
    def _retrieve_from_episodic_memory(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve episodes from episodic memory."""
        results = []
        
        try:
            if hasattr(self.state_manager, 'episodic_memory'):
                # Search through episodes
                for episode in self.state_manager.episodic_memory.episodes.values():
                    if self._episode_passes_filters(episode, query):
                        relevance = self._calculate_relevance_score(
                            f"{episode.title} {episode.summary}", query.query_text
                        )
                        
                        if relevance > 0.1:
                            result = RetrievalResult(
                                memory_id=episode.episode_id,
                                memory_system="episodic",
                                content=f"{episode.title}: {episode.summary}",
                                memory_type="episode",
                                importance=episode.importance,
                                relevance_score=relevance,
                                recency_score=self._calculate_recency_score(episode.end_time),
                                activation_score=episode.coherence_score,
                                combined_score=0.0,
                                metadata={
                                    "participants": list(episode.participants),
                                    "location": episode.location,
                                    "episode_type": episode.episode_type.value,
                                    "event_count": len(episode.event_ids)
                                },
                                timestamp=episode.end_time
                            )
                            results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error retrieving from episodic memory: {str(e)}")
        
        return results
    
    def _retrieve_from_semantic_memory(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve concepts from semantic memory."""
        results = []
        
        try:
            if hasattr(self.state_manager, 'semantic_memory'):
                # Get activated concepts
                activated_concepts = self.state_manager.semantic_memory.retrieve_by_activation(
                    threshold=0.3, limit=20
                )
                
                # Also search by similarity
                similar_concepts = self.state_manager.semantic_memory.retrieve_by_similarity(
                    query.query_text, limit=20
                )
                
                # Combine and deduplicate
                all_concepts = {}
                for concept, similarity in similar_concepts:
                    all_concepts[concept.concept_id] = (concept, similarity)
                
                for concept in activated_concepts:
                    if concept.concept_id not in all_concepts:
                        all_concepts[concept.concept_id] = (concept, 0.5)  # Default similarity
                
                for concept, similarity in all_concepts.values():
                    relevance = max(similarity, self._calculate_relevance_score(
                        f"{concept.name} {concept.description}", query.query_text
                    ))
                    
                    if relevance > 0.1:
                        result = RetrievalResult(
                            memory_id=concept.concept_id,
                            memory_system="semantic",
                            content=f"{concept.name}: {concept.description}",
                            memory_type=concept.concept_type.value,
                            importance=concept.importance,
                            relevance_score=relevance,
                            recency_score=self._calculate_recency_score(concept.last_accessed),
                            activation_score=concept.activation_level,
                            combined_score=0.0,
                            metadata={
                                "concept_type": concept.concept_type.value,
                                "access_count": concept.access_count,
                                "attributes": concept.attributes
                            },
                            timestamp=concept.last_accessed
                        )
                        results.append(result)
        
        except Exception as e:
            self.logger.error(f"Error retrieving from semantic memory: {str(e)}")
        
        return results
    
    def _apply_retrieval_strategy(self, results: List[RetrievalResult], query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Apply retrieval strategy to score and rank results.
        
        Args:
            results: Raw retrieval results
            query: Original query
        
        Returns:
            Scored and ranked results
        """
        # Get weights based on context
        weights = self.context_weights.get(query.query_context, {
            "recency": query.recency_weight,
            "relevance": query.relevance_weight,
            "importance": query.importance_weight
        })
        
        for result in results:
            if query.strategy == RetrievalStrategy.RECENCY_BASED:
                result.combined_score = result.recency_score
            elif query.strategy == RetrievalStrategy.IMPORTANCE_BASED:
                result.combined_score = result.importance
            elif query.strategy == RetrievalStrategy.RELEVANCE_BASED:
                result.combined_score = result.relevance_score
            elif query.strategy == RetrievalStrategy.ACTIVATION_BASED:
                result.combined_score = result.activation_score
            else:  # MIXED strategy
                result.combined_score = (
                    weights["recency"] * result.recency_score +
                    weights["relevance"] * result.relevance_score +
                    weights["importance"] * result.importance
                )
        
        return results
    
    def _calculate_relevance_score(self, content: str, query_text: str) -> float:
        """Calculate relevance score between content and query."""
        if not content or not query_text:
            return 0.0
        
        # Extract keywords from both content and query
        content_keywords = set(self._extract_keywords(content.lower()))
        query_keywords = set(self._extract_keywords(query_text.lower()))
        
        if not content_keywords or not query_keywords:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(content_keywords & query_keywords)
        union = len(content_keywords | query_keywords)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_recency_score(self, timestamp: Optional[datetime]) -> float:
        """Calculate recency score based on timestamp."""
        if not timestamp:
            return 0.0
        
        # Calculate hours since timestamp
        hours_ago = (datetime.now() - timestamp).total_seconds() / 3600
        
        # Exponential decay with half-life of 24 hours
        return math.exp(-0.693 * hours_ago / 24)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _passes_filters(self, memory: Dict[str, Any], query: RetrievalQuery) -> bool:
        """Check if memory passes query filters."""
        # Check importance threshold
        if memory.get("importance", 0.0) < query.importance_threshold:
            return False
        
        # Check memory types filter
        if query.memory_types and memory.get("type") not in query.memory_types:
            return False
        
        # Check participants filter
        if query.participants:
            memory_participants = set(memory.get("participants", []))
            if not (query.participants & memory_participants):
                return False
        
        return True
    
    def _episode_passes_filters(self, episode: Episode, query: RetrievalQuery) -> bool:
        """Check if episode passes query filters."""
        # Check importance threshold
        if episode.importance < query.importance_threshold:
            return False
        
        # Check participants filter
        if query.participants and not (query.participants & episode.participants):
            return False
        
        return True
    
    def _generate_cache_key(self, query: RetrievalQuery) -> str:
        """Generate cache key for query."""
        key_parts = [
            query.query_text,
            query.query_context.value,
            query.strategy.value,
            str(query.max_results),
            str(query.importance_threshold)
        ]
        
        if query.memory_types:
            key_parts.append(",".join(sorted(query.memory_types)))
        
        if query.participants:
            key_parts.append(",".join(sorted(query.participants)))
        
        return "|".join(key_parts)
    
    def _update_access_patterns(self, state: EnhancedAgentState) -> None:
        """Update memory access patterns based on current state."""
        # This could be used to learn user patterns and optimize retrieval
        # For now, just track basic statistics
        pass
    
    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        if not self.cache_enabled:
            return
        
        current_time = datetime.now()
        expired_keys = []
        
        for key, (results, cache_time) in self.query_cache.items():
            if (current_time - cache_time).total_seconds() > self.cache_ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.query_cache[key]
        
        # Also limit cache size
        if len(self.query_cache) > self.max_cache_size:
            # Remove oldest entries
            items = list(self.query_cache.items())
            items.sort(key=lambda x: x[1][1])  # Sort by cache time
            
            for key, _ in items[:-self.max_cache_size]:
                del self.query_cache[key]
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_queries = self.retrieval_stats["total_queries"]
        if total_queries == 0:
            return 0.0
        
        return self.retrieval_stats["cache_hits"] / total_queries
    
    def _update_retrieval_stats(self, retrieval_time_ms: float, result_count: int) -> None:
        """Update retrieval statistics."""
        self.retrieval_stats["total_queries"] += 1
        
        # Update average retrieval time
        total_queries = self.retrieval_stats["total_queries"]
        current_avg = self.retrieval_stats["avg_retrieval_time_ms"]
        self.retrieval_stats["avg_retrieval_time_ms"] = (
            (current_avg * (total_queries - 1) + retrieval_time_ms) / total_queries
        )
        
        # Update average results per query
        current_avg_results = self.retrieval_stats["avg_results_per_query"]
        self.retrieval_stats["avg_results_per_query"] = (
            (current_avg_results * (total_queries - 1) + result_count) / total_queries
        )
        
        self.retrieval_stats["last_query"] = datetime.now().isoformat()
    
    def get_retrieval_summary(self) -> Dict[str, Any]:
        """
        Get summary of retrieval performance and statistics.
        
        Returns:
            Retrieval summary
        """
        return {
            "module_name": self.module_name,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.query_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "retrieval_stats": self.retrieval_stats.copy(),
            "context_weights": {
                ctx.value: weights for ctx, weights in self.context_weights.items()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Example of memory retrieval module usage
    from ..enhanced_agent_state import create_enhanced_agent_state
    
    # Create state manager with some test data
    state_manager = create_enhanced_agent_state(
        "test_agent", "Test Agent", {"confidence": 0.8}
    )
    
    # Add some test memories
    state_manager.add_memory("Had coffee with Maria this morning", "conversation", 0.8)
    state_manager.add_memory("Planning to go hiking tomorrow", "plan", 0.7)
    state_manager.add_memory("Feeling excited about the date", "emotion", 0.6)
    state_manager.add_memory("Maria likes outdoor activities", "observation", 0.9)
    
    # Create retrieval module
    retrieval_module = MemoryRetrievalModule(state_manager)
    
    # Test retrieval
    query = RetrievalQuery(
        query_id="test_query_1",
        query_text="Maria outdoor activities",
        query_context=RetrievalContext.CONVERSATION,
        strategy=RetrievalStrategy.MIXED,
        max_results=5
    )
    
    print("Testing memory retrieval...")
    results = retrieval_module.retrieve_memories(query)
    
    print(f"\nFound {len(results)} memories:")
    for result in results:
        print(f"- [{result.memory_system}] {result.content}")
        print(f"  Score: {result.combined_score:.3f} (R:{result.relevance_score:.2f}, "
              f"Rec:{result.recency_score:.2f}, I:{result.importance:.2f})")
    
    # Test simple string query
    print("\nTesting simple string query...")
    simple_results = retrieval_module.retrieve_memories("coffee")
    print(f"Found {len(simple_results)} memories for 'coffee'")
    
    # Get performance summary
    summary = retrieval_module.get_retrieval_summary()
    print(f"\nRetrieval summary: {summary}")
    
    print("Memory retrieval module example completed!")