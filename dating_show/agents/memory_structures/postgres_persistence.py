"""
File: postgres_persistence.py
Description: PostgreSQL persistence integration for Enhanced PIANO memory systems.
Provides high-performance database operations for 50+ concurrent agents with <100ms response times.
"""

import asyncio
import asyncpg
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import asdict
from contextlib import asynccontextmanager
import os
from pathlib import Path

# Import memory structures
from .circular_buffer import CircularBuffer, MemoryEntry
from .temporal_memory import TemporalMemory, TemporalEntry
from .episodic_memory import EpisodicMemory, Episode, Event, CausalRelation, EpisodeType
from .semantic_memory import SemanticMemory, Concept, SemanticRelation, ConceptType, SemanticRelationType


class PostgresConfig:
    """Configuration for PostgreSQL connection and performance settings."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "piano_agents",
                 username: str = "piano_user",
                 password: str = "piano_password",
                 min_connections: int = 5,
                 max_connections: int = 50,
                 command_timeout: float = 10.0,
                 connection_timeout: float = 5.0):
        """
        Initialize PostgreSQL configuration.
        
        Args:
            host: Database host
            port: Database port
            database: Database name
            username: Database username
            password: Database password
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
            command_timeout: Command timeout in seconds
            connection_timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.command_timeout = command_timeout
        self.connection_timeout = connection_timeout
        
        # Override with environment variables if available
        self.host = os.getenv("POSTGRES_HOST", self.host)
        self.port = int(os.getenv("POSTGRES_PORT", str(self.port)))
        self.database = os.getenv("POSTGRES_DB", self.database)
        self.username = os.getenv("POSTGRES_USER", self.username)
        self.password = os.getenv("POSTGRES_PASSWORD", self.password)
    
    @property
    def dsn(self) -> str:
        """Get PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class PostgresMemoryPersistence:
    """
    High-performance PostgreSQL persistence layer for Enhanced PIANO memory systems.
    Optimized for 50+ concurrent agents with <100ms response times.
    """
    
    def __init__(self, config: PostgresConfig):
        """
        Initialize PostgreSQL persistence layer.
        
        Args:
            config: PostgreSQL configuration
        """
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(f"{__name__}.PostgresMemoryPersistence")
        
        # Performance tracking
        self.operation_times = {}
        self.operation_counts = {}
    
    async def initialize(self) -> None:
        """Initialize connection pool and ensure schema is ready."""
        try:
            self.logger.info(f"Initializing PostgreSQL connection pool to {self.config.host}:{self.config.port}")
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.config.dsn,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout,
                server_settings={
                    'application_name': 'piano_agents',
                    'tcp_keepalives_idle': '600',
                    'tcp_keepalives_interval': '30',
                    'tcp_keepalives_count': '3',
                }
            )
            
            # Verify connection and schema
            async with self.pool.acquire() as connection:
                # Check if tables exist
                result = await connection.fetchval(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'agents'"
                )
                
                if result == 0:
                    raise Exception("Database schema not initialized. Please run the schema creation scripts first.")
                
                self.logger.info("PostgreSQL persistence layer initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL persistence: {str(e)}")
            raise
    
    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("PostgreSQL connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        if not self.pool:
            raise Exception("PostgreSQL persistence not initialized")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    def _track_operation_time(self, operation: str, duration: float) -> None:
        """Track operation performance metrics."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
        
        self.operation_times[operation].append(duration)
        self.operation_counts[operation] += 1
        
        # Keep only last 100 measurements for memory efficiency
        if len(self.operation_times[operation]) > 100:
            self.operation_times[operation] = self.operation_times[operation][-100:]
    
    # =====================================================
    # Agent Management
    # =====================================================
    
    async def ensure_agent_exists(self, agent_id: str, name: str, personality_traits: Dict[str, float]) -> None:
        """Ensure agent exists in database."""
        start_time = datetime.now()
        
        try:
            # Parse name
            names = name.split(' ', 1)
            first_name = names[0]
            last_name = names[1] if len(names) > 1 else ""
            
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO agents (agent_id, name, first_name, last_name, personality_traits, current_role)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (agent_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        first_name = EXCLUDED.first_name,
                        last_name = EXCLUDED.last_name,
                        personality_traits = EXCLUDED.personality_traits,
                        updated_at = CURRENT_TIMESTAMP,
                        active = TRUE
                """, agent_id, name, first_name, last_name, json.dumps(personality_traits), "contestant")
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("ensure_agent_exists", duration)
            
        except Exception as e:
            self.logger.error(f"Failed to ensure agent exists: {str(e)}")
            raise
    
    # =====================================================
    # Working Memory (Circular Buffer) Operations
    # =====================================================
    
    async def store_working_memory(self, agent_id: str, memory_entry: MemoryEntry) -> str:
        """Store working memory entry."""
        start_time = datetime.now()
        
        try:
            memory_id = str(uuid.uuid4())
            
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO working_memory (memory_id, agent_id, content, memory_type, importance, context, 
                                              created_at, expires_at, sequence_number)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 
                           (SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM working_memory WHERE agent_id = $2))
                """, memory_id, agent_id, memory_entry.content, memory_entry.memory_type,
                memory_entry.importance, json.dumps(memory_entry.context), 
                memory_entry.timestamp, memory_entry.expires_at)
                
                # Cleanup old entries if exceeding buffer size
                await conn.execute("""
                    DELETE FROM working_memory 
                    WHERE agent_id = $1 AND sequence_number <= (
                        SELECT sequence_number FROM working_memory 
                        WHERE agent_id = $1 
                        ORDER BY sequence_number DESC 
                        OFFSET $2 LIMIT 1
                    )
                """, agent_id, 20)  # Default buffer size
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("store_working_memory", duration)
            
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store working memory: {str(e)}")
            raise
    
    async def retrieve_working_memory(self, agent_id: str, limit: int = 20) -> List[MemoryEntry]:
        """Retrieve recent working memory entries."""
        start_time = datetime.now()
        
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT content, memory_type, importance, context, created_at, expires_at
                    FROM working_memory
                    WHERE agent_id = $1 AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    ORDER BY sequence_number DESC
                    LIMIT $2
                """, agent_id, limit)
            
            memories = []
            for row in rows:
                context = json.loads(row['context']) if row['context'] else {}
                memories.append(MemoryEntry(
                    content=row['content'],
                    memory_type=row['memory_type'],
                    importance=row['importance'],
                    timestamp=row['created_at'],
                    context=context,
                    expires_at=row['expires_at']
                ))
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("retrieve_working_memory", duration)
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve working memory: {str(e)}")
            raise
    
    # =====================================================
    # Temporal Memory Operations
    # =====================================================
    
    async def store_temporal_memory(self, agent_id: str, temporal_entry: TemporalEntry) -> str:
        """Store temporal memory entry."""
        start_time = datetime.now()
        
        try:
            memory_id = str(uuid.uuid4())
            
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO temporal_memory (memory_id, agent_id, content, memory_type, importance, context,
                                               temporal_key, decay_factor, access_count, last_accessed, 
                                               created_at, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, memory_id, agent_id, temporal_entry.content, temporal_entry.memory_type,
                temporal_entry.importance, json.dumps(temporal_entry.context),
                temporal_entry.temporal_key, temporal_entry.decay_factor, 
                temporal_entry.access_count, temporal_entry.last_accessed,
                temporal_entry.created_at, temporal_entry.expires_at)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("store_temporal_memory", duration)
            
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store temporal memory: {str(e)}")
            raise
    
    async def retrieve_temporal_memory(self, agent_id: str, temporal_pattern: str = None, 
                                     limit: int = 100) -> List[TemporalEntry]:
        """Retrieve temporal memory entries."""
        start_time = datetime.now()
        
        try:
            async with self.get_connection() as conn:
                if temporal_pattern:
                    rows = await conn.fetch("""
                        SELECT content, memory_type, importance, context, temporal_key, decay_factor,
                               access_count, last_accessed, created_at, expires_at
                        FROM temporal_memory
                        WHERE agent_id = $1 AND temporal_key LIKE $2 
                        AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                        ORDER BY importance DESC, created_at DESC
                        LIMIT $3
                    """, agent_id, f"{temporal_pattern}%", limit)
                else:
                    rows = await conn.fetch("""
                        SELECT content, memory_type, importance, context, temporal_key, decay_factor,
                               access_count, last_accessed, created_at, expires_at
                        FROM temporal_memory
                        WHERE agent_id = $1 AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                        ORDER BY importance DESC, created_at DESC
                        LIMIT $2
                    """, agent_id, limit)
            
            memories = []
            for row in rows:
                context = json.loads(row['context']) if row['context'] else {}
                memories.append(TemporalEntry(
                    content=row['content'],
                    memory_type=row['memory_type'],
                    importance=row['importance'],
                    context=context,
                    temporal_key=row['temporal_key'],
                    decay_factor=row['decay_factor'],
                    access_count=row['access_count'],
                    last_accessed=row['last_accessed'],
                    created_at=row['created_at'],
                    expires_at=row['expires_at']
                ))
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("retrieve_temporal_memory", duration)
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve temporal memory: {str(e)}")
            raise
    
    # =====================================================
    # Episodic Memory Operations
    # =====================================================
    
    async def store_episode(self, agent_id: str, episode: Episode) -> str:
        """Store episodic memory episode."""
        start_time = datetime.now()
        
        try:
            episode_id = episode.episode_id
            
            async with self.get_connection() as conn:
                async with conn.transaction():
                    # Store episode
                    await conn.execute("""
                        INSERT INTO episodes (episode_id, agent_id, episode_type, title, summary,
                                            participants, location, start_time, end_time, coherence_score,
                                            importance, emotional_valence, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        ON CONFLICT (episode_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            summary = EXCLUDED.summary,
                            participants = EXCLUDED.participants,
                            end_time = EXCLUDED.end_time,
                            coherence_score = EXCLUDED.coherence_score,
                            importance = EXCLUDED.importance,
                            emotional_valence = EXCLUDED.emotional_valence,
                            metadata = EXCLUDED.metadata
                    """, episode_id, agent_id, episode.episode_type.value, episode.title, episode.summary,
                    episode.participants, episode.location, episode.start_time, episode.end_time,
                    episode.coherence_score, episode.importance, episode.emotional_valence,
                    json.dumps(episode.metadata))
                    
                    # Store events
                    for i, event in enumerate(episode.events):
                        await conn.execute("""
                            INSERT INTO episodic_events (event_id, episode_id, agent_id, content, event_type,
                                                       importance, participants, location, emotional_valence,
                                                       sequence_number, metadata, created_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                            ON CONFLICT (event_id) DO UPDATE SET
                                content = EXCLUDED.content,
                                importance = EXCLUDED.importance,
                                participants = EXCLUDED.participants,
                                emotional_valence = EXCLUDED.emotional_valence,
                                metadata = EXCLUDED.metadata
                        """, event.event_id, episode_id, agent_id, event.content, event.event_type,
                        event.importance, event.participants, event.location, event.emotional_valence,
                        i, json.dumps(event.metadata), event.timestamp)
                    
                    # Store causal relations
                    for relation in episode.causal_relations:
                        await conn.execute("""
                            INSERT INTO causal_relations (relation_id, cause_event_id, effect_event_id,
                                                         relation_type, strength, confidence)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (relation_id) DO UPDATE SET
                                strength = EXCLUDED.strength,
                                confidence = EXCLUDED.confidence
                        """, str(uuid.uuid4()), relation.cause_event_id, relation.effect_event_id,
                        relation.relation_type, relation.strength, relation.confidence)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("store_episode", duration)
            
            return episode_id
            
        except Exception as e:
            self.logger.error(f"Failed to store episode: {str(e)}")
            raise
    
    async def retrieve_episodes(self, agent_id: str, limit: int = 50) -> List[Episode]:
        """Retrieve episodic memory episodes."""
        start_time = datetime.now()
        
        try:
            async with self.get_connection() as conn:
                # Get episodes
                episode_rows = await conn.fetch("""
                    SELECT episode_id, episode_type, title, summary, participants, location,
                           start_time, end_time, coherence_score, importance, emotional_valence, metadata
                    FROM episodes
                    WHERE agent_id = $1
                    ORDER BY importance DESC, start_time DESC
                    LIMIT $2
                """, agent_id, limit)
                
                episodes = []
                for ep_row in episode_rows:
                    episode_id = ep_row['episode_id']
                    
                    # Get events for this episode
                    event_rows = await conn.fetch("""
                        SELECT event_id, content, event_type, importance, participants, location,
                               emotional_valence, metadata, created_at
                        FROM episodic_events
                        WHERE episode_id = $1
                        ORDER BY sequence_number
                    """, episode_id)
                    
                    events = []
                    for ev_row in event_rows:
                        metadata = json.loads(ev_row['metadata']) if ev_row['metadata'] else {}
                        events.append(Event(
                            event_id=ev_row['event_id'],
                            content=ev_row['content'],
                            event_type=ev_row['event_type'],
                            importance=ev_row['importance'],
                            participants=ev_row['participants'],
                            location=ev_row['location'],
                            emotional_valence=ev_row['emotional_valence'],
                            metadata=metadata,
                            timestamp=ev_row['created_at']
                        ))
                    
                    # Get causal relations
                    relation_rows = await conn.fetch("""
                        SELECT cr.relation_type, cr.strength, cr.confidence, cr.cause_event_id, cr.effect_event_id
                        FROM causal_relations cr
                        JOIN episodic_events ce ON cr.cause_event_id = ce.event_id
                        JOIN episodic_events ee ON cr.effect_event_id = ee.event_id
                        WHERE ce.episode_id = $1 OR ee.episode_id = $1
                    """, episode_id)
                    
                    causal_relations = []
                    for rel_row in relation_rows:
                        causal_relations.append(CausalRelation(
                            cause_event_id=rel_row['cause_event_id'],
                            effect_event_id=rel_row['effect_event_id'],
                            relation_type=rel_row['relation_type'],
                            strength=rel_row['strength'],
                            confidence=rel_row['confidence']
                        ))
                    
                    episode_metadata = json.loads(ep_row['metadata']) if ep_row['metadata'] else {}
                    episodes.append(Episode(
                        episode_id=episode_id,
                        episode_type=EpisodeType(ep_row['episode_type']),
                        title=ep_row['title'],
                        summary=ep_row['summary'],
                        events=events,
                        participants=set(ep_row['participants']),
                        location=ep_row['location'],
                        start_time=ep_row['start_time'],
                        end_time=ep_row['end_time'],
                        coherence_score=ep_row['coherence_score'],
                        importance=ep_row['importance'],
                        emotional_valence=ep_row['emotional_valence'],
                        causal_relations=causal_relations,
                        metadata=episode_metadata
                    ))
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("retrieve_episodes", duration)
            
            return episodes
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve episodes: {str(e)}")
            raise
    
    # =====================================================
    # Semantic Memory Operations
    # =====================================================
    
    async def store_semantic_concept(self, agent_id: str, concept: Concept) -> str:
        """Store semantic memory concept."""
        start_time = datetime.now()
        
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO semantic_concepts (concept_id, agent_id, name, concept_type, description,
                                                 importance, activation_level, access_frequency, 
                                                 last_accessed, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (agent_id, name, concept_type) DO UPDATE SET
                        description = EXCLUDED.description,
                        importance = EXCLUDED.importance,
                        activation_level = EXCLUDED.activation_level,
                        access_frequency = EXCLUDED.access_frequency,
                        last_accessed = EXCLUDED.last_accessed
                """, concept.concept_id, agent_id, concept.name, concept.concept_type.value,
                concept.description, concept.importance, concept.activation_level,
                concept.access_frequency, concept.last_accessed, concept.created_at)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("store_semantic_concept", duration)
            
            return concept.concept_id
            
        except Exception as e:
            self.logger.error(f"Failed to store semantic concept: {str(e)}")
            raise
    
    async def store_semantic_relation(self, relation: SemanticRelation) -> str:
        """Store semantic memory relation."""
        start_time = datetime.now()
        
        try:
            async with self.get_connection() as conn:
                relation_id = str(uuid.uuid4())
                await conn.execute("""
                    INSERT INTO semantic_relations (relation_id, source_concept_id, target_concept_id,
                                                   relation_type, strength)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (source_concept_id, target_concept_id, relation_type) DO UPDATE SET
                        strength = EXCLUDED.strength
                """, relation_id, relation.source_concept_id, relation.target_concept_id,
                relation.relation_type.value, relation.strength)
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("store_semantic_relation", duration)
            
            return relation_id
            
        except Exception as e:
            self.logger.error(f"Failed to store semantic relation: {str(e)}")
            raise
    
    async def retrieve_semantic_concepts(self, agent_id: str, concept_type: ConceptType = None,
                                       min_activation: float = 0.0, limit: int = 100) -> List[Concept]:
        """Retrieve semantic memory concepts."""
        start_time = datetime.now()
        
        try:
            async with self.get_connection() as conn:
                if concept_type:
                    rows = await conn.fetch("""
                        SELECT concept_id, name, concept_type, description, importance, activation_level,
                               access_frequency, last_accessed, created_at
                        FROM semantic_concepts
                        WHERE agent_id = $1 AND concept_type = $2 AND activation_level >= $3
                        ORDER BY activation_level DESC, access_frequency DESC
                        LIMIT $4
                    """, agent_id, concept_type.value, min_activation, limit)
                else:
                    rows = await conn.fetch("""
                        SELECT concept_id, name, concept_type, description, importance, activation_level,
                               access_frequency, last_accessed, created_at
                        FROM semantic_concepts
                        WHERE agent_id = $1 AND activation_level >= $2
                        ORDER BY activation_level DESC, access_frequency DESC
                        LIMIT $3
                    """, agent_id, min_activation, limit)
            
            concepts = []
            for row in rows:
                concepts.append(Concept(
                    concept_id=row['concept_id'],
                    name=row['name'],
                    concept_type=ConceptType(row['concept_type']),
                    description=row['description'],
                    importance=row['importance'],
                    activation_level=row['activation_level'],
                    access_frequency=row['access_frequency'],
                    last_accessed=row['last_accessed'],
                    created_at=row['created_at']
                ))
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("retrieve_semantic_concepts", duration)
            
            return concepts
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve semantic concepts: {str(e)}")
            raise
    
    # =====================================================
    # Performance and Maintenance Operations
    # =====================================================
    
    async def cleanup_expired_memories(self, agent_id: str = None) -> Dict[str, int]:
        """Clean up expired memories from all systems."""
        start_time = datetime.now()
        
        try:
            cleanup_stats = {}
            
            async with self.get_connection() as conn:
                if agent_id:
                    # Cleanup for specific agent
                    working_deleted = await conn.fetchval("""
                        DELETE FROM working_memory 
                        WHERE agent_id = $1 AND expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                        RETURNING COUNT(*)
                    """, agent_id) or 0
                    
                    temporal_deleted = await conn.fetchval("""
                        DELETE FROM temporal_memory 
                        WHERE agent_id = $1 AND expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                        RETURNING COUNT(*)
                    """, agent_id) or 0
                else:
                    # Global cleanup
                    working_deleted = await conn.fetchval("""
                        DELETE FROM working_memory 
                        WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                        RETURNING COUNT(*)
                    """) or 0
                    
                    temporal_deleted = await conn.fetchval("""
                        DELETE FROM temporal_memory 
                        WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP
                        RETURNING COUNT(*)
                    """) or 0
                
                cleanup_stats = {
                    "working_memory_deleted": working_deleted,
                    "temporal_memory_deleted": temporal_deleted
                }
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("cleanup_expired_memories", duration)
            
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired memories: {str(e)}")
            raise
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this persistence layer."""
        metrics = {
            "operation_counts": self.operation_counts.copy(),
            "average_times_ms": {}
        }
        
        for operation, times in self.operation_times.items():
            if times:
                metrics["average_times_ms"][operation] = sum(times) / len(times)
        
        # Add database connection pool stats
        if self.pool:
            metrics["pool_stats"] = {
                "size": self.pool.get_size(),
                "min_size": self.pool.get_min_size(),
                "max_size": self.pool.get_max_size(),
                "idle_connections": self.pool.get_idle_size()
            }
        
        return metrics
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance operations for optimal performance."""
        start_time = datetime.now()
        
        try:
            results = {}
            
            async with self.get_connection() as conn:
                # Update semantic concept activation decay
                decayed_concepts = await conn.fetchval("""
                    SELECT decay_semantic_activations()
                """)
                results["decayed_concepts"] = decayed_concepts
                
                # Cleanup expired memories
                cleanup_stats = await self.cleanup_expired_memories()
                results.update(cleanup_stats)
                
                # Update table statistics
                await conn.execute("SELECT refresh_table_statistics()")
                results["statistics_refreshed"] = True
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._track_operation_time("run_maintenance", duration)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to run maintenance: {str(e)}")
            raise


# Helper functions for easy integration

def create_postgres_config() -> PostgresConfig:
    """Create PostgreSQL configuration from environment variables or defaults."""
    return PostgresConfig()


async def create_postgres_persistence(config: PostgresConfig = None) -> PostgresMemoryPersistence:
    """Create and initialize PostgreSQL persistence layer."""
    if config is None:
        config = create_postgres_config()
    
    persistence = PostgresMemoryPersistence(config)
    await persistence.initialize()
    return persistence


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    async def test_postgres_persistence():
        """Test the PostgreSQL persistence layer."""
        
        # Initialize persistence
        config = create_postgres_config()
        persistence = await create_postgres_persistence(config)
        
        try:
            agent_id = "test_agent_001"
            
            # Test agent creation
            await persistence.ensure_agent_exists(
                agent_id, "Test Agent", {"confidence": 0.8, "openness": 0.7}
            )
            print("✅ Agent creation successful")
            
            # Test working memory
            from .circular_buffer import MemoryEntry
            memory = MemoryEntry(
                content="Test working memory entry",
                memory_type="test",
                importance=0.7,
                timestamp=datetime.now(),
                context={"test": True}
            )
            
            memory_id = await persistence.store_working_memory(agent_id, memory)
            retrieved_memories = await persistence.retrieve_working_memory(agent_id, 10)
            print(f"✅ Working memory test successful: stored {memory_id}, retrieved {len(retrieved_memories)} entries")
            
            # Test performance metrics
            metrics = await persistence.get_performance_metrics()
            print(f"✅ Performance metrics: {metrics['average_times_ms']}")
            
            # Test cleanup
            cleanup_stats = await persistence.cleanup_expired_memories(agent_id)
            print(f"✅ Cleanup successful: {cleanup_stats}")
            
        finally:
            await persistence.close()
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_postgres_persistence())
    else:
        print("PostgreSQL Persistence module loaded successfully")
        print("Run with 'test' argument to execute tests")