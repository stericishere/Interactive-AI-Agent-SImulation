"""
File: memory_consolidation_module.py
Description: Background Memory Consolidation Module for enhanced PIANO architecture.
Handles asynchronous memory processing, working memory → long-term memory transfer,
and memory importance scoring and pruning.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import asyncio
import threading
import time
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum

from .langgraph_base_module import LangGraphBaseModule, ModuleExecutionConfig, ExecutionTimeScale, ModulePriority
from ..enhanced_agent_state import EnhancedAgentState, EnhancedAgentStateManager
from ..memory_structures.circular_buffer import CircularBuffer
from ..memory_structures.temporal_memory import TemporalMemory
from ..memory_structures.episodic_memory import EpisodicMemory, CausalRelationType
from ..memory_structures.semantic_memory import SemanticMemory, ConceptType, SemanticRelationType


class ConsolidationStrategy(Enum):
    """Memory consolidation strategies."""
    IMPORTANCE_BASED = "importance_based"    # Consolidate by importance scores
    FREQUENCY_BASED = "frequency_based"      # Consolidate frequently accessed memories
    RECENCY_BASED = "recency_based"         # Consolidate recent memories first
    SEMANTIC_BASED = "semantic_based"        # Consolidate by semantic similarity
    MIXED = "mixed"                         # Combined strategy


@dataclass
class ConsolidationTask:
    """Represents a memory consolidation task."""
    task_id: str
    task_type: str
    source_memory_ids: List[str]
    target_memory_system: str
    importance_threshold: float
    created_at: datetime
    estimated_duration: float
    priority: int = 1  # 1 = highest priority


@dataclass
class ConsolidationResult:
    """Result of memory consolidation process."""
    task_id: str
    success: bool
    memories_processed: int
    memories_consolidated: int
    memories_pruned: int
    processing_time_ms: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class MemoryConsolidationModule(LangGraphBaseModule):
    """
    Background memory consolidation module that transfers memories between systems
    and performs cleanup operations without blocking agent decisions.
    """
    
    def __init__(self, state_manager: Optional[EnhancedAgentStateManager] = None):
        """
        Initialize Memory Consolidation Module.
        
        Args:
            state_manager: Enhanced agent state manager
        """
        config = ModuleExecutionConfig(
            time_scale=ExecutionTimeScale.SLOW,
            priority=ModulePriority.LOW,
            can_run_parallel=True,
            requires_completion=False,
            max_execution_time=10.0
        )
        
        super().__init__("memory_consolidation", config, state_manager)
        
        # Consolidation settings
        self.consolidation_strategy = ConsolidationStrategy.MIXED
        self.importance_threshold = 0.3
        self.batch_size = 50
        self.max_working_memory_size = 20
        
        # Task queue for background processing
        self.consolidation_queue: List[ConsolidationTask] = []
        self.active_tasks: Dict[str, ConsolidationTask] = {}
        self.completed_tasks: List[ConsolidationResult] = []
        
        # Performance tracking
        self.consolidation_stats = {
            "total_consolidations": 0,
            "total_memories_processed": 0,
            "total_memories_pruned": 0,
            "avg_processing_time": 0.0,
            "last_consolidation": None
        }
        
        # Background processing
        self._background_thread = None
        self._stop_background = threading.Event()
        self._background_active = False
        
        self.logger = logging.getLogger("MemoryConsolidation")
    
    def process_state(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Process agent state for memory consolidation opportunities.
        
        Args:
            state: Current enhanced agent state
        
        Returns:
            Dictionary with consolidation results and state updates
        """
        start_time = time.time()
        
        try:
            # Check if we need to start background processing
            if not self._background_active:
                self._start_background_processing()
            
            # Analyze current memory state
            memory_analysis = self._analyze_memory_state(state)
            
            # Determine consolidation needs
            consolidation_needs = self._assess_consolidation_needs(memory_analysis)
            
            # Queue consolidation tasks if needed
            tasks_queued = 0
            if consolidation_needs["needs_consolidation"]:
                tasks_queued = self._queue_consolidation_tasks(consolidation_needs)
            
            # Perform immediate critical consolidations
            immediate_results = self._perform_immediate_consolidations(state)
            
            # Update consolidation statistics
            self._update_consolidation_stats(immediate_results)
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "state_changes": {
                    "performance": {
                        **state.get("performance", {}),
                        "memory_efficiency": memory_analysis["efficiency_score"]
                    }
                },
                "output_data": {
                    "memory_analysis": memory_analysis,
                    "consolidation_needs": consolidation_needs,
                    "tasks_queued": tasks_queued,
                    "immediate_consolidations": len(immediate_results)
                },
                "performance_metrics": {
                    "processing_time_ms": processing_time,
                    "memory_pressure": memory_analysis["pressure_score"],
                    "consolidation_queue_size": len(self.consolidation_queue)
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error in memory consolidation: {str(e)}")
            return {
                "output_data": {"error": str(e)},
                "performance_metrics": {"processing_time_ms": (time.time() - start_time) * 1000}
            }
    
    def _analyze_memory_state(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """
        Analyze current memory state to determine consolidation needs.
        
        Args:
            state: Current agent state
        
        Returns:
            Memory analysis results
        """
        if not self.state_manager:
            return {"efficiency_score": 1.0, "pressure_score": 0.0, "needs_attention": False}
        
        analysis = {
            "working_memory_utilization": 0.0,
            "temporal_memory_size": 0,
            "episodic_memory_size": 0,
            "semantic_memory_size": 0,
            "pressure_score": 0.0,
            "efficiency_score": 1.0,
            "needs_attention": False
        }
        
        try:
            # Analyze working memory (circular buffer)
            working_mem_size = len(self.state_manager.circular_buffer)
            working_mem_utilization = working_mem_size / self.state_manager.circular_buffer.max_size
            analysis["working_memory_utilization"] = working_mem_utilization
            
            # Analyze other memory systems
            analysis["temporal_memory_size"] = len(self.state_manager.temporal_memory.memories)
            analysis["episodic_memory_size"] = len(self.state_manager.episodic_memory.episodes)
            analysis["semantic_memory_size"] = len(self.state_manager.semantic_memory.concepts)
            
            # Calculate pressure score (0.0 to 1.0)
            pressure_factors = [
                working_mem_utilization,
                min(analysis["temporal_memory_size"] / 1000, 1.0),  # Normalize to 1000 memories
                min(analysis["episodic_memory_size"] / 200, 1.0),   # Normalize to 200 episodes
                min(analysis["semantic_memory_size"] / 500, 1.0)    # Normalize to 500 concepts
            ]
            
            analysis["pressure_score"] = sum(pressure_factors) / len(pressure_factors)
            
            # Calculate efficiency score (inverse of fragmentation and redundancy)
            efficiency_factors = []
            
            # Working memory efficiency
            if working_mem_size > 0:
                important_memories = len(self.state_manager.circular_buffer.get_important_memories(0.6))
                efficiency_factors.append(important_memories / working_mem_size)
            
            # Temporal memory efficiency (based on decay)
            temporal_memories = self.state_manager.temporal_memory.retrieve_recent_memories(hours_back=24)
            if temporal_memories:
                avg_strength = sum(m.get("current_strength", 0.5) for m in temporal_memories) / len(temporal_memories)
                efficiency_factors.append(avg_strength)
            
            if efficiency_factors:
                analysis["efficiency_score"] = sum(efficiency_factors) / len(efficiency_factors)
            
            # Determine if memory system needs attention
            analysis["needs_attention"] = (
                analysis["pressure_score"] > 0.7 or 
                analysis["efficiency_score"] < 0.4 or
                working_mem_utilization > 0.9
            )
        
        except Exception as e:
            self.logger.error(f"Error analyzing memory state: {str(e)}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _assess_consolidation_needs(self, memory_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess what types of consolidation are needed.
        
        Args:
            memory_analysis: Results from memory state analysis
        
        Returns:
            Consolidation needs assessment
        """
        needs = {
            "needs_consolidation": False,
            "urgency_level": "low",  # low, medium, high, critical
            "consolidation_types": [],
            "estimated_tasks": 0
        }
        
        # Check working memory pressure
        if memory_analysis.get("working_memory_utilization", 0) > 0.8:
            needs["consolidation_types"].append("working_to_temporal")
            needs["needs_consolidation"] = True
            if memory_analysis["working_memory_utilization"] > 0.95:
                needs["urgency_level"] = "critical"
            elif memory_analysis["working_memory_utilization"] > 0.9:
                needs["urgency_level"] = "high"
        
        # Check temporal memory consolidation needs
        if memory_analysis.get("temporal_memory_size", 0) > 500:
            needs["consolidation_types"].append("temporal_cleanup")
            needs["needs_consolidation"] = True
            if needs["urgency_level"] == "low":
                needs["urgency_level"] = "medium"
        
        # Check episodic memory fragmentation
        if memory_analysis.get("episodic_memory_size", 0) > 150:
            needs["consolidation_types"].append("episodic_consolidation")
            needs["needs_consolidation"] = True
        
        # Check semantic memory optimization
        if memory_analysis.get("semantic_memory_size", 0) > 400:
            needs["consolidation_types"].append("semantic_optimization")
            needs["needs_consolidation"] = True
        
        # Check overall efficiency
        if memory_analysis.get("efficiency_score", 1.0) < 0.5:
            needs["consolidation_types"].append("efficiency_optimization")
            needs["needs_consolidation"] = True
            if memory_analysis["efficiency_score"] < 0.3:
                needs["urgency_level"] = "high"
        
        # Estimate number of tasks needed
        needs["estimated_tasks"] = len(needs["consolidation_types"])
        
        return needs
    
    def _queue_consolidation_tasks(self, consolidation_needs: Dict[str, Any]) -> int:
        """
        Queue appropriate consolidation tasks based on needs assessment.
        
        Args:
            consolidation_needs: Results from consolidation needs assessment
        
        Returns:
            Number of tasks queued
        """
        tasks_queued = 0
        task_id_counter = len(self.consolidation_queue) + len(self.active_tasks)
        
        for consolidation_type in consolidation_needs["consolidation_types"]:
            task_id = f"consolidation_{task_id_counter}_{consolidation_type}"
            
            # Set priority based on urgency
            priority = {
                "critical": 1,
                "high": 2,
                "medium": 3,
                "low": 4
            }.get(consolidation_needs["urgency_level"], 4)
            
            task = ConsolidationTask(
                task_id=task_id,
                task_type=consolidation_type,
                source_memory_ids=[],  # Will be populated during execution
                target_memory_system=self._get_target_system(consolidation_type),
                importance_threshold=self.importance_threshold,
                created_at=datetime.now(),
                estimated_duration=self._estimate_task_duration(consolidation_type),
                priority=priority
            )
            
            self.consolidation_queue.append(task)
            tasks_queued += 1
            task_id_counter += 1
        
        # Sort queue by priority
        self.consolidation_queue.sort(key=lambda t: t.priority)
        
        return tasks_queued
    
    def _get_target_system(self, consolidation_type: str) -> str:
        """Get target memory system for consolidation type."""
        target_mapping = {
            "working_to_temporal": "temporal_memory",
            "temporal_cleanup": "temporal_memory",
            "episodic_consolidation": "episodic_memory",
            "semantic_optimization": "semantic_memory",
            "efficiency_optimization": "all_systems"
        }
        return target_mapping.get(consolidation_type, "temporal_memory")
    
    def _estimate_task_duration(self, consolidation_type: str) -> float:
        """Estimate task duration in seconds."""
        duration_mapping = {
            "working_to_temporal": 1.0,
            "temporal_cleanup": 3.0,
            "episodic_consolidation": 2.0,
            "semantic_optimization": 4.0,
            "efficiency_optimization": 5.0
        }
        return duration_mapping.get(consolidation_type, 2.0)
    
    def _perform_immediate_consolidations(self, state: EnhancedAgentState) -> List[ConsolidationResult]:
        """
        Perform critical consolidations that can't wait for background processing.
        
        Args:
            state: Current agent state
        
        Returns:
            List of consolidation results
        """
        results = []
        
        if not self.state_manager:
            return results
        
        try:
            # Critical: Working memory overflow
            if len(self.state_manager.circular_buffer) >= self.state_manager.circular_buffer.max_size:
                result = self._consolidate_working_memory_immediate()
                if result:
                    results.append(result)
            
            # Critical: Temporal memory cleanup for expired memories
            expired_count = len(self.state_manager.temporal_memory.cleanup_expired_memories())
            if expired_count > 0:
                results.append(ConsolidationResult(
                    task_id="immediate_temporal_cleanup",
                    success=True,
                    memories_processed=expired_count,
                    memories_consolidated=0,
                    memories_pruned=expired_count,
                    processing_time_ms=50.0  # Estimated
                ))
        
        except Exception as e:
            self.logger.error(f"Error in immediate consolidations: {str(e)}")
            results.append(ConsolidationResult(
                task_id="immediate_error",
                success=False,
                memories_processed=0,
                memories_consolidated=0,
                memories_pruned=0,
                processing_time_ms=0.0,
                error_message=str(e)
            ))
        
        return results
    
    def _consolidate_working_memory_immediate(self) -> Optional[ConsolidationResult]:
        """
        Immediately consolidate working memory to prevent overflow.
        
        Returns:
            Consolidation result if performed
        """
        if not self.state_manager:
            return None
        
        start_time = time.time()
        
        try:
            # Get memories from working memory buffer
            memories = list(self.state_manager.circular_buffer.buffer)
            
            memories_processed = 0
            memories_consolidated = 0
            
            # Transfer important memories to temporal memory
            for memory in memories:
                if memory["importance"] >= self.importance_threshold:
                    # Add to temporal memory
                    self.state_manager.temporal_memory.add_memory(
                        content=memory["content"],
                        memory_type=memory["type"],
                        importance=memory["importance"],
                        context=memory.get("metadata", {}),
                        timestamp=memory["timestamp"]
                    )
                    memories_consolidated += 1
                
                memories_processed += 1
            
            # Clear some working memory entries (keep most recent important ones)
            important_recent = self.state_manager.circular_buffer.get_important_memories(0.6)
            most_recent = self.state_manager.circular_buffer.get_recent_memories(10)
            
            # Keep union of important and recent memories
            keep_ids = set()
            for memory in important_recent + most_recent:
                keep_ids.add(memory["id"])
            
            # Filter buffer to keep only selected memories
            self.state_manager.circular_buffer.buffer = [
                m for m in self.state_manager.circular_buffer.buffer
                if m["id"] in keep_ids
            ]
            
            processing_time = (time.time() - start_time) * 1000
            
            return ConsolidationResult(
                task_id="immediate_working_memory",
                success=True,
                memories_processed=memories_processed,
                memories_consolidated=memories_consolidated,
                memories_pruned=len(memories) - len(self.state_manager.circular_buffer.buffer),
                processing_time_ms=processing_time
            )
        
        except Exception as e:
            self.logger.error(f"Error in immediate working memory consolidation: {str(e)}")
            return ConsolidationResult(
                task_id="immediate_working_memory_error",
                success=False,
                memories_processed=0,
                memories_consolidated=0,
                memories_pruned=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def _start_background_processing(self) -> None:
        """Start background consolidation processing thread."""
        if self._background_active:
            return
        
        self._background_active = True
        self._stop_background.clear()
        self._background_thread = threading.Thread(
            target=self._background_consolidation_loop,
            name="MemoryConsolidationBackground"
        )
        self._background_thread.start()
        self.logger.info("Background memory consolidation started")
    
    def _background_consolidation_loop(self) -> None:
        """Main background processing loop for memory consolidation."""
        while not self._stop_background.is_set():
            try:
                # Process consolidation queue
                if self.consolidation_queue:
                    task = self.consolidation_queue.pop(0)
                    self.active_tasks[task.task_id] = task
                    
                    try:
                        result = self._execute_consolidation_task(task)
                        self.completed_tasks.append(result)
                        
                        # Keep only recent completed tasks
                        if len(self.completed_tasks) > 100:
                            self.completed_tasks = self.completed_tasks[-50:]
                    
                    except Exception as e:
                        self.logger.error(f"Error executing consolidation task {task.task_id}: {str(e)}")
                        error_result = ConsolidationResult(
                            task_id=task.task_id,
                            success=False,
                            memories_processed=0,
                            memories_consolidated=0,
                            memories_pruned=0,
                            processing_time_ms=0.0,
                            error_message=str(e)
                        )
                        self.completed_tasks.append(error_result)
                    
                    finally:
                        self.active_tasks.pop(task.task_id, None)
                
                # Sleep briefly before next iteration
                time.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Error in background consolidation loop: {str(e)}")
                time.sleep(1.0)  # Longer sleep on error
        
        self._background_active = False
        self.logger.info("Background memory consolidation stopped")
    
    def _execute_consolidation_task(self, task: ConsolidationTask) -> ConsolidationResult:
        """
        Execute a specific consolidation task.
        
        Args:
            task: Consolidation task to execute
        
        Returns:
            Consolidation result
        """
        start_time = time.time()
        
        try:
            if task.task_type == "working_to_temporal":
                return self._consolidate_working_to_temporal(task, start_time)
            elif task.task_type == "temporal_cleanup":
                return self._cleanup_temporal_memory(task, start_time)
            elif task.task_type == "episodic_consolidation":
                return self._consolidate_episodic_memory(task, start_time)
            elif task.task_type == "semantic_optimization":
                return self._optimize_semantic_memory(task, start_time)
            elif task.task_type == "efficiency_optimization":
                return self._optimize_memory_efficiency(task, start_time)
            else:
                return ConsolidationResult(
                    task_id=task.task_id,
                    success=False,
                    memories_processed=0,
                    memories_consolidated=0,
                    memories_pruned=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    error_message=f"Unknown task type: {task.task_type}"
                )
        
        except Exception as e:
            return ConsolidationResult(
                task_id=task.task_id,
                success=False,
                memories_processed=0,
                memories_consolidated=0,
                memories_pruned=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def _consolidate_working_to_temporal(self, task: ConsolidationTask, start_time: float) -> ConsolidationResult:
        """Consolidate working memory to temporal memory."""
        if not self.state_manager:
            raise ValueError("State manager not available")
        
        memories = list(self.state_manager.circular_buffer.buffer)
        memories_processed = len(memories)
        memories_consolidated = 0
        
        # Transfer memories above importance threshold
        for memory in memories:
            if memory["importance"] >= task.importance_threshold:
                self.state_manager.temporal_memory.add_memory(
                    content=memory["content"],
                    memory_type=memory["type"],
                    importance=memory["importance"],
                    context=memory.get("metadata", {}),
                    timestamp=memory["timestamp"]
                )
                memories_consolidated += 1
        
        # Keep only most important/recent memories in working memory
        important_memories = self.state_manager.circular_buffer.get_important_memories(0.7)
        recent_memories = self.state_manager.circular_buffer.get_recent_memories(5)
        
        keep_ids = set()
        for memory in important_memories + recent_memories:
            keep_ids.add(memory["id"])
        
        original_size = len(self.state_manager.circular_buffer.buffer)
        self.state_manager.circular_buffer.buffer = [
            m for m in self.state_manager.circular_buffer.buffer if m["id"] in keep_ids
        ]
        memories_pruned = original_size - len(self.state_manager.circular_buffer.buffer)
        
        return ConsolidationResult(
            task_id=task.task_id,
            success=True,
            memories_processed=memories_processed,
            memories_consolidated=memories_consolidated,
            memories_pruned=memories_pruned,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _cleanup_temporal_memory(self, task: ConsolidationTask, start_time: float) -> ConsolidationResult:
        """Clean up temporal memory by removing weak memories."""
        if not self.state_manager:
            raise ValueError("State manager not available")
        
        # Get all temporal memories
        all_memories = []
        for time_key, memories in self.state_manager.temporal_memory.memories.items():
            all_memories.extend(memories)
        
        memories_processed = len(all_memories)
        
        # Remove expired memories
        expired_ids = self.state_manager.temporal_memory.cleanup_expired_memories()
        memories_pruned = len(expired_ids)
        
        # Consolidate similar memories
        consolidated_ids = self.state_manager.temporal_memory.consolidate_memories()
        memories_consolidated = len(consolidated_ids)
        
        return ConsolidationResult(
            task_id=task.task_id,
            success=True,
            memories_processed=memories_processed,
            memories_consolidated=memories_consolidated,
            memories_pruned=memories_pruned,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _consolidate_episodic_memory(self, task: ConsolidationTask, start_time: float) -> ConsolidationResult:
        """Consolidate episodic memory by merging similar episodes."""
        if not self.state_manager:
            raise ValueError("State manager not available")
        
        episodes = list(self.state_manager.episodic_memory.episodes.values())
        memories_processed = len(episodes)
        
        # Find episodes that can be consolidated (similar participants, location, time)
        consolidated_count = 0
        
        # Simple consolidation based on time proximity and participant overlap
        for i, episode1 in enumerate(episodes):
            for j, episode2 in enumerate(episodes[i+1:], i+1):
                if (episode1.episode_id in self.state_manager.episodic_memory.episodes and
                    episode2.episode_id in self.state_manager.episodic_memory.episodes):
                    
                    # Check if episodes can be merged
                    time_gap = abs((episode1.end_time - episode2.start_time).total_seconds())
                    participant_overlap = len(episode1.participants & episode2.participants)
                    
                    if (time_gap < 1800 and  # Within 30 minutes
                        participant_overlap > 0 and
                        episode1.location == episode2.location):
                        
                        # Merge episodes (keep the more important one)
                        if episode1.importance >= episode2.importance:
                            # Merge episode2 into episode1
                            episode1.event_ids.extend(episode2.event_ids)
                            episode1.end_time = max(episode1.end_time, episode2.end_time)
                            episode1.participants.update(episode2.participants)
                            
                            # Remove episode2
                            self.state_manager.episodic_memory._remove_episode(episode2.episode_id)
                            consolidated_count += 1
        
        return ConsolidationResult(
            task_id=task.task_id,
            success=True,
            memories_processed=memories_processed,
            memories_consolidated=consolidated_count,
            memories_pruned=0,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _optimize_semantic_memory(self, task: ConsolidationTask, start_time: float) -> ConsolidationResult:
        """Optimize semantic memory by consolidating similar concepts."""
        if not self.state_manager:
            raise ValueError("State manager not available")
        
        concepts = list(self.state_manager.semantic_memory.concepts.values())
        memories_processed = len(concepts)
        
        # Update activation decay
        self.state_manager.semantic_memory.update_activation_decay()
        
        # Consolidate similar concepts
        consolidated_pairs = self.state_manager.semantic_memory.consolidate_concepts()
        memories_consolidated = len(consolidated_pairs)
        
        return ConsolidationResult(
            task_id=task.task_id,
            success=True,
            memories_processed=memories_processed,
            memories_consolidated=memories_consolidated,
            memories_pruned=0,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _optimize_memory_efficiency(self, task: ConsolidationTask, start_time: float) -> ConsolidationResult:
        """Perform comprehensive memory efficiency optimization."""
        if not self.state_manager:
            raise ValueError("State manager not available")
        
        total_processed = 0
        total_consolidated = 0
        total_pruned = 0
        
        # Clean up all memory systems
        cleanup_stats = self.state_manager.cleanup_memories()
        
        total_pruned += cleanup_stats.get("expired_working_memories", 0)
        total_pruned += cleanup_stats.get("removed_temporal_memories", 0)
        total_consolidated += cleanup_stats.get("consolidated_concepts", 0)
        
        # Count total memories processed
        total_processed += len(self.state_manager.circular_buffer.buffer)
        total_processed += sum(len(memories) for memories in self.state_manager.temporal_memory.memories.values())
        total_processed += len(self.state_manager.semantic_memory.concepts)
        
        return ConsolidationResult(
            task_id=task.task_id,
            success=True,
            memories_processed=total_processed,
            memories_consolidated=total_consolidated,
            memories_pruned=total_pruned,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _update_consolidation_stats(self, results: List[ConsolidationResult]) -> None:
        """Update consolidation statistics."""
        for result in results:
            if result.success:
                self.consolidation_stats["total_consolidations"] += 1
                self.consolidation_stats["total_memories_processed"] += result.memories_processed
                self.consolidation_stats["total_memories_pruned"] += result.memories_pruned
                
                # Update average processing time
                current_avg = self.consolidation_stats["avg_processing_time"]
                total_consolidations = self.consolidation_stats["total_consolidations"]
                
                self.consolidation_stats["avg_processing_time"] = (
                    (current_avg * (total_consolidations - 1) + result.processing_time_ms) / total_consolidations
                )
                
                self.consolidation_stats["last_consolidation"] = datetime.now().isoformat()
    
    def get_consolidation_summary(self) -> Dict[str, Any]:
        """
        Get summary of consolidation activity.
        
        Returns:
            Consolidation summary with statistics
        """
        return {
            "module_name": self.module_name,
            "background_active": self._background_active,
            "queue_size": len(self.consolidation_queue),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "consolidation_stats": self.consolidation_stats.copy(),
            "recent_results": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "memories_processed": r.memories_processed,
                    "processing_time_ms": r.processing_time_ms
                }
                for r in self.completed_tasks[-5:]  # Last 5 results
            ]
        }
    
    def shutdown(self) -> None:
        """Shutdown memory consolidation module."""
        self.logger.info("Shutting down memory consolidation module")
        
        # Stop background processing
        if self._background_active:
            self._stop_background.set()
            if self._background_thread and self._background_thread.is_alive():
                self._background_thread.join(timeout=5.0)
        
        # Call parent shutdown
        super().shutdown()
        
        self.logger.info("Memory consolidation module shutdown complete")


# Example usage and testing
if __name__ == "__main__":
    # Example of memory consolidation module usage
    from ..enhanced_agent_state import create_enhanced_agent_state
    
    # Create state manager
    state_manager = create_enhanced_agent_state(
        "test_agent", "Test Agent", {"confidence": 0.8}
    )
    
    # Add some test memories to trigger consolidation
    for i in range(25):  # Exceed working memory limit
        state_manager.add_memory(
            f"Test memory {i}",
            "event",
            importance=0.3 + (i % 10) * 0.07  # Varying importance
        )
    
    # Create consolidation module
    consolidation_module = MemoryConsolidationModule(state_manager)
    
    print("Initial memory state:")
    print(f"Working memory size: {len(state_manager.circular_buffer)}")
    print(f"Temporal memory size: {len(state_manager.temporal_memory.memories)}")
    
    # Execute consolidation
    result = consolidation_module(state_manager.state)
    
    print("\nAfter consolidation:")
    print(f"Working memory size: {len(state_manager.circular_buffer)}")
    print(f"Temporal memory size: {len(state_manager.temporal_memory.memories)}")
    print(f"Consolidation result: {result}")
    
    # Get consolidation summary
    summary = consolidation_module.get_consolidation_summary()
    print(f"\nConsolidation summary: {summary}")
    
    # Wait a bit for background processing
    time.sleep(2)
    
    # Shutdown
    consolidation_module.shutdown()
    print("Memory consolidation module example completed!")