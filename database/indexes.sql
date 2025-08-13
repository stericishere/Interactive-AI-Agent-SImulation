-- Performance Optimization Indexes for Enhanced PIANO Architecture
-- Designed for 50+ concurrent agents with <100ms response times

-- =====================================================
-- Composite Indexes for Complex Queries
-- =====================================================

-- Agent-centric memory retrieval (most common query pattern)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_working_memory_agent_time_importance 
ON working_memory(agent_id, created_at DESC, importance DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_temporal_memory_agent_key_importance 
ON temporal_memory(agent_id, temporal_key, importance DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_episodes_agent_time_importance 
ON episodes(agent_id, start_time DESC, importance DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_semantic_concepts_agent_activation_access 
ON semantic_concepts(agent_id, activation_level DESC, last_accessed DESC);

-- Cross-memory association queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_episodic_events_episode_sequence 
ON episodic_events(episode_id, sequence_number, importance DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_causal_relations_strength_confidence 
ON causal_relations(cause_event_id, strength DESC, confidence DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_semantic_relations_source_type_strength 
ON semantic_relations(source_concept_id, relation_type, strength DESC);

-- =====================================================
-- Agent State and Performance Indexes
-- =====================================================

-- Current agent state retrieval
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_states_agent_current_version 
ON agent_states(agent_id, is_current, state_version DESC) 
WHERE is_current = true;

-- Performance monitoring queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_agent_recorded 
ON performance_metrics(agent_id, recorded_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_latency_threshold 
ON performance_metrics(recorded_at DESC, decision_latency) 
WHERE decision_latency > 100.0; -- Monitor slow decisions

-- Specialization tracking
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_specializations_role_expertise 
ON agent_specializations(current_role, expertise_level DESC, updated_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_skill_history_agent_skill_time 
ON skill_history(agent_id, skill_name, created_at DESC);

-- =====================================================
-- Cultural System Performance Indexes
-- =====================================================

-- Meme propagation tracking
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meme_transmissions_time_success_strength 
ON meme_transmissions(recorded_at DESC, success, transmission_strength DESC) 
WHERE success = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_meme_adoption_active_influence 
ON agent_meme_adoption(is_active, influence_strength DESC, last_reinforcement DESC) 
WHERE is_active = true;

-- Cultural value alignment
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_value_alignment_value_strength 
ON agent_value_alignment(value_id, alignment_strength DESC, personal_importance DESC);

-- Social role performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_role_assignments_active_strength 
ON agent_role_assignments(is_active, role_strength DESC, community_recognition DESC) 
WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_social_roles_status_availability 
ON social_roles(is_active, status DESC, availability) 
WHERE is_active = true;

-- =====================================================
-- Governance System Performance Indexes
-- =====================================================

-- Active governance rules
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_governance_rules_active_priority_enforcement 
ON governance_rules(is_active, priority DESC, enforcement_level DESC) 
WHERE is_active = true;

-- Voting system performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_governance_proposals_voting_status 
ON governance_proposals(status, voting_ends_at DESC) 
WHERE status IN ('voting', 'draft');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_votes_proposal_choice_strength 
ON votes(proposal_id, vote_choice, vote_strength DESC, cast_at DESC) 
WHERE is_valid = true;

-- Rule compliance monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rule_compliance_agent_score_assessment 
ON rule_compliance(agent_id, compliance_score DESC, last_assessment DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_rule_violations_unresolved_severity 
ON rule_violations(resolved, severity DESC, detected_at DESC) 
WHERE resolved = false;

-- =====================================================
-- Social Network and Influence Indexes
-- =====================================================

-- Influence network queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_influence_network_active_strength_interaction 
ON influence_network(is_active, influence_strength DESC, last_interaction DESC) 
WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_influence_network_type_strength 
ON influence_network(influence_type, influence_strength DESC, interaction_count DESC) 
WHERE is_active = true;

-- Leadership metrics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_leadership_metrics_score_trust_reach 
ON leadership_metrics(leadership_score DESC, community_trust DESC, influence_reach DESC);

-- Community events
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_community_events_active_impact_time 
ON community_events(ended_at IS NULL, impact_level DESC, started_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_event_participation_active_level 
ON event_participation(participation_level DESC, impact_on_agent, joined_at DESC);

-- =====================================================
-- Store API Specific Indexes
-- =====================================================

-- Store items performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_store_items_namespace_key_version 
ON store_items(namespace, key, version DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_store_items_active_updated 
ON store_items(expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP, updated_at DESC);

-- =====================================================
-- Text Search Indexes for Content Queries
-- =====================================================

-- Memory content search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_working_memory_content_trgm 
ON working_memory USING GIN(content gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_temporal_memory_content_trgm 
ON temporal_memory USING GIN(content gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_episodic_events_content_trgm 
ON episodic_events USING GIN(content gin_trgm_ops);

-- Governance text search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_governance_rules_text_trgm 
ON governance_rules USING GIN(rule_text gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_governance_proposals_text_trgm 
ON governance_proposals USING GIN(proposal_text gin_trgm_ops);

-- =====================================================
-- JSONB Indexes for Metadata Queries
-- =====================================================

-- Context and metadata searches
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_working_memory_context_gin 
ON working_memory USING GIN(context);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_temporal_memory_context_gin 
ON temporal_memory USING GIN(context);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_episodes_metadata_gin 
ON episodes USING GIN(metadata);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_episodic_events_metadata_gin 
ON episodic_events USING GIN(metadata);

-- Agent state and personality
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_personality_gin 
ON agents USING GIN(personality_traits);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_states_data_gin 
ON agent_states USING GIN(state_data);

-- Governance metadata
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_governance_rules_metadata_gin 
ON governance_rules USING GIN(metadata);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_governance_proposals_metadata_gin 
ON governance_proposals USING GIN(metadata);

-- Cultural system metadata
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cultural_memes_metadata_gin 
ON cultural_memes USING GIN(metadata);

-- =====================================================
-- Partial Indexes for Common Filtered Queries
-- =====================================================

-- Active agents only
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_active_name 
ON agents(name, current_role) 
WHERE active = true;

-- Recent memories (last hour)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_working_memory_recent_agent_importance 
ON working_memory(agent_id, importance DESC, sequence_number DESC) 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_temporal_memory_recent_agent_importance 
ON temporal_memory(agent_id, importance DESC, temporal_key) 
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '1 hour';

-- High importance memories
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_episodes_high_importance_agent_time 
ON episodes(agent_id, start_time DESC, coherence_score DESC) 
WHERE importance > 0.7;

-- Active semantic concepts with high activation
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_semantic_concepts_high_activation_agent 
ON semantic_concepts(agent_id, activation_level DESC, access_frequency DESC) 
WHERE activation_level > 0.3;

-- Recent governance activity
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_votes_recent_proposal_choice 
ON votes(proposal_id, vote_choice, vote_strength DESC) 
WHERE cast_at > CURRENT_TIMESTAMP - INTERVAL '7 days';

-- Active cultural transmission
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_meme_transmissions_recent_successful 
ON meme_transmissions(meme_id, transmission_strength DESC, recorded_at DESC) 
WHERE success = true AND recorded_at > CURRENT_TIMESTAMP - INTERVAL '24 hours';

-- =====================================================
-- Multi-Column Statistics for Query Planner
-- =====================================================

-- Create extended statistics for correlated columns
CREATE STATISTICS IF NOT EXISTS stats_working_memory_agent_time_importance 
ON agent_id, created_at, importance 
FROM working_memory;

CREATE STATISTICS IF NOT EXISTS stats_episodes_agent_time_importance_coherence 
ON agent_id, start_time, importance, coherence_score 
FROM episodes;

CREATE STATISTICS IF NOT EXISTS stats_agent_meme_adoption_agent_influence_reinforcement 
ON agent_id, influence_strength, last_reinforcement 
FROM agent_meme_adoption;

CREATE STATISTICS IF NOT EXISTS stats_performance_metrics_agent_latency_coherence 
ON agent_id, decision_latency, coherence_score 
FROM performance_metrics;

-- =====================================================
-- Index Usage Monitoring Views
-- =====================================================

-- View to monitor index usage and effectiveness
CREATE OR REPLACE VIEW index_usage_stats AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    CASE 
        WHEN idx_scan > 0 THEN round(idx_tup_read::numeric / idx_scan, 2)
        ELSE 0
    END as avg_tuples_per_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- View to identify unused indexes
CREATE OR REPLACE VIEW unused_indexes AS
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) as size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND schemaname = 'public'
ORDER BY pg_relation_size(indexrelid) DESC;

-- View to monitor table and index sizes
CREATE OR REPLACE VIEW table_index_sizes AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) as index_size,
    round(100.0 * (pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) / 
          NULLIF(pg_total_relation_size(schemaname||'.'||tablename), 0), 2) as index_ratio_percent
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- =====================================================
-- Performance Monitoring Functions
-- =====================================================

-- Function to analyze slow queries
CREATE OR REPLACE FUNCTION analyze_slow_queries(min_duration_ms INTEGER DEFAULT 100)
RETURNS TABLE(
    query_text TEXT,
    mean_time_ms REAL,
    calls BIGINT,
    total_time_ms REAL,
    stddev_time_ms REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pg_stat_statements.query,
        round(pg_stat_statements.mean_time::numeric, 2)::REAL as mean_time_ms,
        pg_stat_statements.calls,
        round(pg_stat_statements.total_time::numeric, 2)::REAL as total_time_ms,
        round(pg_stat_statements.stddev_time::numeric, 2)::REAL as stddev_time_ms
    FROM pg_stat_statements
    WHERE pg_stat_statements.mean_time > min_duration_ms
    ORDER BY pg_stat_statements.mean_time DESC;
EXCEPTION
    WHEN undefined_table THEN
        RAISE NOTICE 'pg_stat_statements extension not installed. Install it for query performance monitoring.';
        RETURN;
END;
$$ LANGUAGE plpgsql;

-- Function to get table statistics
CREATE OR REPLACE FUNCTION get_table_stats()
RETURNS TABLE(
    table_name TEXT,
    n_tup_ins BIGINT,
    n_tup_upd BIGINT,
    n_tup_del BIGINT,
    n_tup_hot_upd BIGINT,
    n_live_tup BIGINT,
    n_dead_tup BIGINT,
    last_vacuum TIMESTAMP WITH TIME ZONE,
    last_autovacuum TIMESTAMP WITH TIME ZONE,
    last_analyze TIMESTAMP WITH TIME ZONE,
    last_autoanalyze TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pg_stat_user_tables.relname::TEXT,
        pg_stat_user_tables.n_tup_ins,
        pg_stat_user_tables.n_tup_upd,
        pg_stat_user_tables.n_tup_del,
        pg_stat_user_tables.n_tup_hot_upd,
        pg_stat_user_tables.n_live_tup,
        pg_stat_user_tables.n_dead_tup,
        pg_stat_user_tables.last_vacuum,
        pg_stat_user_tables.last_autovacuum,
        pg_stat_user_tables.last_analyze,
        pg_stat_user_tables.last_autoanalyze
    FROM pg_stat_user_tables
    ORDER BY pg_stat_user_tables.n_live_tup DESC;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Maintenance Functions
-- =====================================================

-- Function to refresh statistics for better query planning
CREATE OR REPLACE FUNCTION refresh_table_statistics()
RETURNS TEXT AS $$
DECLARE
    table_record RECORD;
    result_text TEXT := '';
BEGIN
    FOR table_record IN 
        SELECT tablename FROM pg_tables WHERE schemaname = 'public'
    LOOP
        EXECUTE 'ANALYZE ' || quote_ident(table_record.tablename);
        result_text := result_text || 'Analyzed ' || table_record.tablename || E'\n';
    END LOOP;
    
    RETURN result_text;
END;
$$ LANGUAGE plpgsql;

-- Function to reindex tables if needed
CREATE OR REPLACE FUNCTION reindex_if_needed(bloat_threshold REAL DEFAULT 20.0)
RETURNS TEXT AS $$
DECLARE
    index_record RECORD;
    result_text TEXT := '';
BEGIN
    -- This is a simplified version - in production, you'd want more sophisticated bloat detection
    FOR index_record IN 
        SELECT indexname, tablename FROM pg_indexes WHERE schemaname = 'public'
    LOOP
        -- Reindex if statistics show high activity
        IF EXISTS (
            SELECT 1 FROM pg_stat_user_tables 
            WHERE relname = index_record.tablename 
            AND n_tup_upd + n_tup_del > 1000
        ) THEN
            EXECUTE 'REINDEX INDEX CONCURRENTLY ' || quote_ident(index_record.indexname);
            result_text := result_text || 'Reindexed ' || index_record.indexname || E'\n';
        END IF;
    END LOOP;
    
    RETURN result_text;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Comments and Documentation
-- =====================================================

COMMENT ON FUNCTION analyze_slow_queries IS 'Analyze queries slower than specified threshold (requires pg_stat_statements)';
COMMENT ON FUNCTION get_table_stats IS 'Get comprehensive table statistics for monitoring';
COMMENT ON FUNCTION refresh_table_statistics IS 'Refresh table statistics for optimal query planning';
COMMENT ON FUNCTION reindex_if_needed IS 'Reindex tables with high update/delete activity';

COMMENT ON VIEW index_usage_stats IS 'Monitor index usage effectiveness';
COMMENT ON VIEW unused_indexes IS 'Identify potentially unused indexes for cleanup';
COMMENT ON VIEW table_index_sizes IS 'Monitor table and index storage usage';

-- =====================================================
-- Performance Optimization Complete
-- =====================================================

-- This index strategy is designed for:
-- 1. Agent-centric memory queries (most common pattern)
-- 2. Cross-memory association traversals
-- 3. Cultural meme propagation tracking
-- 4. Democratic governance operations
-- 5. Social influence network analysis
-- 6. Performance monitoring and optimization
--
-- Expected performance improvements:
-- - Memory retrieval: <50ms for working memory, <100ms for long-term
-- - Cultural propagation: <200ms for meme spreading
-- - Governance operations: <300ms for voting and rule compliance
-- - Agent state updates: <10ms for local operations, <50ms for shared state
--
-- All indexes are created CONCURRENTLY to avoid blocking operations
-- Statistics and monitoring functions help maintain optimal performance