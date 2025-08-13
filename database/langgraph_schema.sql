-- PostgreSQL schema for LangGraph Checkpointer and Enhanced PIANO Memory Systems
-- Supports 50+ concurrent agents with optimized performance for <100ms operations

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =====================================================
-- LangGraph Checkpointer Tables
-- =====================================================

-- Main checkpoints table for LangGraph StateGraph persistence
CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id UUID NOT NULL,
    parent_checkpoint_id UUID,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Checkpoint writes table for atomic operations
CREATE TABLE IF NOT EXISTS langgraph_checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id UUID NOT NULL,
    task_id UUID NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- =====================================================
-- Enhanced Agent State Tables
-- =====================================================

-- Core agent metadata table
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    first_name TEXT NOT NULL,
    last_name TEXT,
    age INTEGER,
    personality_traits JSONB NOT NULL DEFAULT '{}',
    current_role TEXT DEFAULT 'contestant',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    active BOOLEAN DEFAULT TRUE
);

-- Agent state snapshots for performance
CREATE TABLE IF NOT EXISTS agent_states (
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    state_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    state_data JSONB NOT NULL,
    state_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_current BOOLEAN DEFAULT TRUE
);

-- =====================================================
-- Memory System Tables
-- =====================================================

-- Working memory (circular buffer) table
CREATE TABLE IF NOT EXISTS working_memory (
    memory_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    context JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    sequence_number INTEGER NOT NULL,
    CONSTRAINT valid_importance CHECK (importance >= 0.0 AND importance <= 1.0),
    CONSTRAINT valid_memory_type CHECK (memory_type ~ '^[a-zA-Z_][a-zA-Z0-9_]*$')
);

-- Temporal memory with time-based indexing
CREATE TABLE IF NOT EXISTS temporal_memory (
    memory_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    context JSONB DEFAULT '{}',
    temporal_key TEXT NOT NULL, -- Format: 'YYYY-MM-DD-HH-MM'
    decay_factor REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT valid_importance CHECK (importance >= 0.0 AND importance <= 1.0),
    CONSTRAINT valid_decay CHECK (decay_factor >= 0.0 AND decay_factor <= 1.0)
);

-- Episodic memory for event sequences
CREATE TABLE IF NOT EXISTS episodes (
    episode_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    episode_type TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    participants TEXT[] DEFAULT '{}',
    location TEXT,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    coherence_score REAL DEFAULT 0.5,
    importance REAL DEFAULT 0.5,
    emotional_valence REAL DEFAULT 0.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_coherence CHECK (coherence_score >= 0.0 AND coherence_score <= 1.0),
    CONSTRAINT valid_importance CHECK (importance >= 0.0 AND importance <= 1.0),
    CONSTRAINT valid_valence CHECK (emotional_valence >= -1.0 AND emotional_valence <= 1.0)
);

-- Individual events within episodes
CREATE TABLE IF NOT EXISTS episodic_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    event_type TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    participants TEXT[] DEFAULT '{}',
    location TEXT,
    emotional_valence REAL DEFAULT 0.0,
    sequence_number INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_importance CHECK (importance >= 0.0 AND importance <= 1.0),
    CONSTRAINT valid_valence CHECK (emotional_valence >= -1.0 AND emotional_valence <= 1.0)
);

-- Causal relationships between events
CREATE TABLE IF NOT EXISTS causal_relations (
    relation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cause_event_id UUID NOT NULL REFERENCES episodic_events(event_id) ON DELETE CASCADE,
    effect_event_id UUID NOT NULL REFERENCES episodic_events(event_id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_strength CHECK (strength >= 0.0 AND strength <= 1.0),
    CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT no_self_reference CHECK (cause_event_id != effect_event_id)
);

-- Semantic memory concepts
CREATE TABLE IF NOT EXISTS semantic_concepts (
    concept_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    concept_type TEXT NOT NULL,
    description TEXT,
    importance REAL NOT NULL DEFAULT 0.5,
    activation_level REAL NOT NULL DEFAULT 1.0,
    access_frequency INTEGER DEFAULT 1,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding VECTOR(384), -- For semantic similarity (optional)
    CONSTRAINT valid_importance CHECK (importance >= 0.0 AND importance <= 1.0),
    CONSTRAINT valid_activation CHECK (activation_level >= 0.0 AND activation_level <= 1.0),
    UNIQUE(agent_id, name, concept_type)
);

-- Semantic relationships between concepts
CREATE TABLE IF NOT EXISTS semantic_relations (
    relation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_concept_id UUID NOT NULL REFERENCES semantic_concepts(concept_id) ON DELETE CASCADE,
    target_concept_id UUID NOT NULL REFERENCES semantic_concepts(concept_id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    strength REAL NOT NULL DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_strength CHECK (strength >= 0.0 AND strength <= 1.0),
    CONSTRAINT no_self_reference CHECK (source_concept_id != target_concept_id)
);

-- =====================================================
-- Specialization System Tables
-- =====================================================

-- Agent specialization tracking
CREATE TABLE IF NOT EXISTS agent_specializations (
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    current_role TEXT NOT NULL,
    role_history TEXT[] DEFAULT '{}',
    skills JSONB NOT NULL DEFAULT '{}',
    expertise_level REAL NOT NULL DEFAULT 0.1,
    role_consistency_score REAL NOT NULL DEFAULT 1.0,
    last_role_change TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_id),
    CONSTRAINT valid_expertise CHECK (expertise_level >= 0.0 AND expertise_level <= 1.0),
    CONSTRAINT valid_consistency CHECK (role_consistency_score >= 0.0 AND role_consistency_score <= 1.0)
);

-- Skill development history
CREATE TABLE IF NOT EXISTS skill_history (
    skill_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    skill_name TEXT NOT NULL,
    skill_level REAL NOT NULL,
    gained_from TEXT, -- Action or experience that led to skill gain
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_skill_level CHECK (skill_level >= 0.0 AND skill_level <= 1.0)
);

-- =====================================================
-- Performance Monitoring Tables
-- =====================================================

-- Performance metrics tracking
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    decision_latency REAL,
    coherence_score REAL,
    social_integration REAL,
    memory_efficiency REAL,
    adaptation_rate REAL,
    error_rate REAL DEFAULT 0.0,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_latency CHECK (decision_latency >= 0.0),
    CONSTRAINT valid_coherence CHECK (coherence_score >= 0.0 AND coherence_score <= 1.0),
    CONSTRAINT valid_integration CHECK (social_integration >= 0.0 AND social_integration <= 1.0),
    CONSTRAINT valid_efficiency CHECK (memory_efficiency >= 0.0 AND memory_efficiency <= 1.0)
);

-- System-wide performance tracking
CREATE TABLE IF NOT EXISTS system_performance (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit TEXT,
    context JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- Indexes for Performance Optimization
-- =====================================================

-- LangGraph checkpointer indexes
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_time ON langgraph_checkpoints(thread_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_checkpoints_parent ON langgraph_checkpoints(parent_checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_checkpoint ON langgraph_checkpoint_writes(thread_id, checkpoint_id);

-- Agent state indexes
CREATE INDEX IF NOT EXISTS idx_agents_active ON agents(active) WHERE active = true;
CREATE INDEX IF NOT EXISTS idx_agent_states_current ON agent_states(agent_id, is_current) WHERE is_current = true;
CREATE INDEX IF NOT EXISTS idx_agent_states_time ON agent_states(created_at DESC);

-- Memory system indexes
CREATE INDEX IF NOT EXISTS idx_working_memory_agent_sequence ON working_memory(agent_id, sequence_number DESC);
CREATE INDEX IF NOT EXISTS idx_working_memory_expires ON working_memory(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_working_memory_importance ON working_memory(agent_id, importance DESC);

CREATE INDEX IF NOT EXISTS idx_temporal_memory_agent_time ON temporal_memory(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_temporal_memory_temporal_key ON temporal_memory(agent_id, temporal_key);
CREATE INDEX IF NOT EXISTS idx_temporal_memory_expires ON temporal_memory(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_temporal_memory_last_accessed ON temporal_memory(last_accessed DESC);

CREATE INDEX IF NOT EXISTS idx_episodes_agent_time ON episodes(agent_id, start_time DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_participants ON episodes USING GIN(participants);
CREATE INDEX IF NOT EXISTS idx_episodes_location ON episodes(location) WHERE location IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(agent_id, importance DESC);

CREATE INDEX IF NOT EXISTS idx_episodic_events_episode ON episodic_events(episode_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_episodic_events_agent ON episodic_events(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_events_participants ON episodic_events USING GIN(participants);

CREATE INDEX IF NOT EXISTS idx_causal_relations_cause ON causal_relations(cause_event_id);
CREATE INDEX IF NOT EXISTS idx_causal_relations_effect ON causal_relations(effect_event_id);

CREATE INDEX IF NOT EXISTS idx_semantic_concepts_agent_type ON semantic_concepts(agent_id, concept_type);
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_name_trgm ON semantic_concepts USING GIN(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_activation ON semantic_concepts(agent_id, activation_level DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_access ON semantic_concepts(last_accessed DESC);

CREATE INDEX IF NOT EXISTS idx_semantic_relations_source ON semantic_relations(source_concept_id);
CREATE INDEX IF NOT EXISTS idx_semantic_relations_target ON semantic_relations(target_concept_id);
CREATE INDEX IF NOT EXISTS idx_semantic_relations_type ON semantic_relations(relation_type);

-- Specialization indexes
CREATE INDEX IF NOT EXISTS idx_specializations_role ON agent_specializations(current_role);
CREATE INDEX IF NOT EXISTS idx_specializations_expertise ON agent_specializations(expertise_level DESC);

CREATE INDEX IF NOT EXISTS idx_skill_history_agent_skill ON skill_history(agent_id, skill_name);
CREATE INDEX IF NOT EXISTS idx_skill_history_time ON skill_history(created_at DESC);

-- Performance monitoring indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_agent_time ON performance_metrics(agent_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_system_performance_name_time ON system_performance(metric_name, recorded_at DESC);

-- =====================================================
-- Partitioning for Scalability (Optional)
-- =====================================================

-- Partition working memory by agent_id range for better performance
-- Note: This would be implemented based on agent ID distribution patterns

-- =====================================================
-- Functions and Triggers for Data Maintenance
-- =====================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_specializations_updated_at BEFORE UPDATE ON agent_specializations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to cleanup expired memories
CREATE OR REPLACE FUNCTION cleanup_expired_memories()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Cleanup working memory
    DELETE FROM working_memory WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Cleanup temporal memory
    DELETE FROM temporal_memory WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update semantic concept activation decay
CREATE OR REPLACE FUNCTION decay_semantic_activations()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
    decay_rate REAL := 0.01;
BEGIN
    UPDATE semantic_concepts 
    SET activation_level = GREATEST(activation_level - decay_rate, 0.0),
        last_accessed = CASE 
            WHEN activation_level - decay_rate <= 0.0 THEN CURRENT_TIMESTAMP - INTERVAL '1 hour'
            ELSE last_accessed
        END
    WHERE activation_level > 0.0;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Views for Common Queries
-- =====================================================

-- Current agent state view
CREATE OR REPLACE VIEW current_agent_states AS
SELECT 
    a.agent_id,
    a.name,
    a.current_role,
    a.personality_traits,
    s.state_data,
    s.created_at as state_created_at,
    spec.expertise_level,
    spec.role_consistency_score
FROM agents a
LEFT JOIN agent_states s ON a.agent_id = s.agent_id AND s.is_current = true
LEFT JOIN agent_specializations spec ON a.agent_id = spec.agent_id
WHERE a.active = true;

-- Recent memory summary view
CREATE OR REPLACE VIEW recent_memory_summary AS
SELECT 
    agent_id,
    'working' as memory_system,
    COUNT(*) as memory_count,
    AVG(importance) as avg_importance,
    MAX(created_at) as latest_memory
FROM working_memory 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY agent_id

UNION ALL

SELECT 
    agent_id,
    'temporal' as memory_system,
    COUNT(*) as memory_count,
    AVG(importance) as avg_importance,
    MAX(created_at) as latest_memory
FROM temporal_memory 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY agent_id

UNION ALL

SELECT 
    agent_id,
    'episodic' as memory_system,
    COUNT(*) as memory_count,
    AVG(importance) as avg_importance,
    MAX(created_at) as latest_memory
FROM episodes 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY agent_id;

-- Performance dashboard view
CREATE OR REPLACE VIEW performance_dashboard AS
SELECT 
    pm.agent_id,
    a.name,
    pm.decision_latency,
    pm.coherence_score,
    pm.social_integration,
    pm.memory_efficiency,
    pm.recorded_at,
    ROW_NUMBER() OVER (PARTITION BY pm.agent_id ORDER BY pm.recorded_at DESC) as rn
FROM performance_metrics pm
JOIN agents a ON pm.agent_id = a.agent_id
WHERE a.active = true;

-- Grant permissions for application user
-- Note: Adjust username as needed
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dating_show_app;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dating_show_app;

-- =====================================================
-- Comments for Documentation
-- =====================================================

COMMENT ON TABLE langgraph_checkpoints IS 'LangGraph StateGraph checkpoints for agent state persistence';
COMMENT ON TABLE agents IS 'Core agent metadata and configuration';
COMMENT ON TABLE working_memory IS 'Circular buffer working memory with size limits';
COMMENT ON TABLE temporal_memory IS 'Time-indexed memory with decay functions';
COMMENT ON TABLE episodes IS 'Episodic memory episodes containing event sequences';
COMMENT ON TABLE episodic_events IS 'Individual events within episodic memory episodes';
COMMENT ON TABLE semantic_concepts IS 'Concept-based semantic memory with spreading activation';
COMMENT ON TABLE semantic_relations IS 'Relationships between semantic concepts';
COMMENT ON TABLE performance_metrics IS 'Agent performance tracking and monitoring';

-- Schema creation complete
-- This schema supports:
-- - LangGraph StateGraph checkpointer integration
-- - Enhanced memory systems with PostgreSQL persistence
-- - Performance optimization for 50+ concurrent agents
-- - Target performance: <50ms working memory, <100ms long-term memory operations