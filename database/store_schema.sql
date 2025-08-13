-- PostgreSQL schema for LangGraph Store API - Cultural & Governance Shared State
-- Enables cross-agent sharing of cultural memes, governance rules, and social dynamics

-- Enable required extensions (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- =====================================================
-- LangGraph Store API Core Tables
-- =====================================================

-- Store namespaces for organizing shared data
CREATE TABLE IF NOT EXISTS store_namespaces (
    namespace_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Core store items table for LangGraph Store API
CREATE TABLE IF NOT EXISTS store_items (
    key TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    version INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (key, namespace)
);

-- =====================================================
-- Cultural Transmission System
-- =====================================================

-- Cultural memes registry
CREATE TABLE IF NOT EXISTS cultural_memes (
    meme_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meme_name TEXT NOT NULL UNIQUE,
    meme_type TEXT NOT NULL, -- 'behavior', 'value', 'norm', 'tradition', 'slang'
    description TEXT,
    origin_agent_id TEXT,
    strength REAL NOT NULL DEFAULT 0.5,
    virality REAL NOT NULL DEFAULT 0.1, -- How easily it spreads
    stability REAL NOT NULL DEFAULT 0.5, -- How resistant to change
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}',
    CONSTRAINT valid_strength CHECK (strength >= 0.0 AND strength <= 1.0),
    CONSTRAINT valid_virality CHECK (virality >= 0.0 AND virality <= 1.0),
    CONSTRAINT valid_stability CHECK (stability >= 0.0 AND stability <= 1.0)
);

-- Agent meme adoption tracking
CREATE TABLE IF NOT EXISTS agent_meme_adoption (
    adoption_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    meme_id UUID NOT NULL REFERENCES cultural_memes(meme_id) ON DELETE CASCADE,
    influence_strength REAL NOT NULL DEFAULT 0.5,
    adoption_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source_agent_id TEXT, -- Agent who transmitted this meme
    exposure_count INTEGER DEFAULT 1,
    resistance_level REAL DEFAULT 0.0, -- Agent's resistance to this meme
    last_reinforcement TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT valid_influence CHECK (influence_strength >= 0.0 AND influence_strength <= 1.0),
    CONSTRAINT valid_resistance CHECK (resistance_level >= 0.0 AND resistance_level <= 1.0),
    UNIQUE(agent_id, meme_id)
);

-- Meme transmission events
CREATE TABLE IF NOT EXISTS meme_transmissions (
    transmission_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    meme_id UUID NOT NULL REFERENCES cultural_memes(meme_id) ON DELETE CASCADE,
    source_agent_id TEXT NOT NULL,
    target_agent_id TEXT NOT NULL,
    transmission_strength REAL NOT NULL DEFAULT 0.5,
    context TEXT, -- Conversation, observation, etc.
    success BOOLEAN, -- Whether transmission was successful
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    interaction_type TEXT DEFAULT 'conversation',
    CONSTRAINT valid_transmission_strength CHECK (transmission_strength >= 0.0 AND transmission_strength <= 1.0)
);

-- Cultural values registry
CREATE TABLE IF NOT EXISTS cultural_values (
    value_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    value_name TEXT NOT NULL UNIQUE,
    description TEXT,
    category TEXT, -- 'social', 'moral', 'aesthetic', 'practical'
    base_strength REAL NOT NULL DEFAULT 0.5,
    community_consensus REAL NOT NULL DEFAULT 0.5,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT valid_base_strength CHECK (base_strength >= 0.0 AND base_strength <= 1.0),
    CONSTRAINT valid_consensus CHECK (community_consensus >= 0.0 AND community_consensus <= 1.0)
);

-- Agent value alignment tracking
CREATE TABLE IF NOT EXISTS agent_value_alignment (
    alignment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    value_id UUID NOT NULL REFERENCES cultural_values(value_id) ON DELETE CASCADE,
    alignment_strength REAL NOT NULL DEFAULT 0.5,
    personal_importance REAL NOT NULL DEFAULT 0.5,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_alignment CHECK (alignment_strength >= -1.0 AND alignment_strength <= 1.0),
    CONSTRAINT valid_importance CHECK (personal_importance >= 0.0 AND personal_importance <= 1.0),
    UNIQUE(agent_id, value_id)
);

-- Social roles and expectations
CREATE TABLE IF NOT EXISTS social_roles (
    role_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    role_name TEXT NOT NULL UNIQUE,
    description TEXT,
    expectations JSONB NOT NULL DEFAULT '{}', -- Expected behaviors, responsibilities
    status REAL NOT NULL DEFAULT 0.5, -- Social status/prestige of role
    availability INTEGER DEFAULT 1, -- How many agents can have this role (0 = unlimited)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT valid_status CHECK (status >= 0.0 AND status <= 1.0)
);

-- Agent role assignments
CREATE TABLE IF NOT EXISTS agent_role_assignments (
    assignment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    role_id UUID NOT NULL REFERENCES social_roles(role_id) ON DELETE CASCADE,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    assigned_by TEXT, -- 'community', 'self', 'emergence', specific agent_id
    role_strength REAL NOT NULL DEFAULT 0.5, -- How strongly agent embodies this role
    community_recognition REAL NOT NULL DEFAULT 0.5, -- How much community recognizes this role
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT valid_role_strength CHECK (role_strength >= 0.0 AND role_strength <= 1.0),
    CONSTRAINT valid_recognition CHECK (community_recognition >= 0.0 AND community_recognition <= 1.0)
);

-- =====================================================
-- Collective Governance System
-- =====================================================

-- Constitutional rules registry
CREATE TABLE IF NOT EXISTS governance_rules (
    rule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_name TEXT NOT NULL UNIQUE,
    rule_text TEXT NOT NULL,
    rule_type TEXT NOT NULL, -- 'constitutional', 'procedural', 'behavioral', 'resource'
    priority INTEGER NOT NULL DEFAULT 100,
    scope TEXT NOT NULL DEFAULT 'all', -- 'all', 'role:specific', 'context:specific'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    enacted_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_by_proposal_id UUID, -- References proposals table
    version INTEGER NOT NULL DEFAULT 1,
    parent_rule_id UUID REFERENCES governance_rules(rule_id),
    is_active BOOLEAN DEFAULT TRUE,
    enforcement_level REAL NOT NULL DEFAULT 0.8, -- How strictly enforced
    community_support REAL NOT NULL DEFAULT 0.5, -- Community support level
    metadata JSONB DEFAULT '{}',
    CONSTRAINT valid_enforcement CHECK (enforcement_level >= 0.0 AND enforcement_level <= 1.0),
    CONSTRAINT valid_support CHECK (community_support >= 0.0 AND community_support <= 1.0)
);

-- Governance proposals for democratic changes
CREATE TABLE IF NOT EXISTS governance_proposals (
    proposal_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposal_title TEXT NOT NULL,
    proposal_text TEXT NOT NULL,
    proposal_type TEXT NOT NULL, -- 'rule_creation', 'rule_amendment', 'rule_repeal', 'resource_allocation'
    proposed_by_agent_id TEXT NOT NULL,
    proposed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    voting_starts_at TIMESTAMP WITH TIME ZONE,
    voting_ends_at TIMESTAMP WITH TIME ZONE,
    status TEXT NOT NULL DEFAULT 'draft', -- 'draft', 'voting', 'passed', 'rejected', 'withdrawn'
    required_majority REAL NOT NULL DEFAULT 0.5, -- Required vote percentage to pass
    affected_rule_id UUID REFERENCES governance_rules(rule_id),
    rationale TEXT,
    metadata JSONB DEFAULT '{}',
    CONSTRAINT valid_majority CHECK (required_majority >= 0.0 AND required_majority <= 1.0)
);

-- Democratic voting system
CREATE TABLE IF NOT EXISTS votes (
    vote_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    proposal_id UUID NOT NULL REFERENCES governance_proposals(proposal_id) ON DELETE CASCADE,
    voter_agent_id TEXT NOT NULL,
    vote_choice TEXT NOT NULL, -- 'approve', 'reject', 'abstain'
    vote_strength REAL NOT NULL DEFAULT 1.0, -- Voting power (could be weighted)
    reasoning TEXT, -- Optional reasoning for the vote
    cast_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_valid BOOLEAN DEFAULT TRUE,
    CONSTRAINT valid_vote_choice CHECK (vote_choice IN ('approve', 'reject', 'abstain')),
    CONSTRAINT valid_vote_strength CHECK (vote_strength >= 0.0 AND vote_strength <= 1.0),
    UNIQUE(proposal_id, voter_agent_id)
);

-- Rule compliance tracking
CREATE TABLE IF NOT EXISTS rule_compliance (
    compliance_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    rule_id UUID NOT NULL REFERENCES governance_rules(rule_id) ON DELETE CASCADE,
    compliance_score REAL NOT NULL DEFAULT 1.0,
    last_violation TIMESTAMP WITH TIME ZONE,
    violation_count INTEGER DEFAULT 0,
    last_assessment TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    assessment_method TEXT DEFAULT 'automatic', -- 'automatic', 'peer_review', 'self_report'
    notes TEXT,
    CONSTRAINT valid_compliance CHECK (compliance_score >= 0.0 AND compliance_score <= 1.0),
    UNIQUE(agent_id, rule_id)
);

-- Rule violations and enforcement actions
CREATE TABLE IF NOT EXISTS rule_violations (
    violation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    rule_id UUID NOT NULL REFERENCES governance_rules(rule_id) ON DELETE CASCADE,
    violation_description TEXT NOT NULL,
    severity REAL NOT NULL DEFAULT 0.5, -- 0.0 = minor, 1.0 = severe
    detected_by TEXT, -- 'system', 'peer_report', 'self_report', agent_id
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    action_taken TEXT, -- 'warning', 'penalty', 'education', 'none'
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    CONSTRAINT valid_severity CHECK (severity >= 0.0 AND severity <= 1.0)
);

-- =====================================================
-- Social Network and Influence Tracking
-- =====================================================

-- Agent influence network
CREATE TABLE IF NOT EXISTS influence_network (
    influence_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    influencer_agent_id TEXT NOT NULL,
    influenced_agent_id TEXT NOT NULL,
    influence_strength REAL NOT NULL DEFAULT 0.5,
    influence_type TEXT NOT NULL, -- 'leadership', 'expertise', 'social', 'cultural'
    last_interaction TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    interaction_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    CONSTRAINT valid_influence_strength CHECK (influence_strength >= 0.0 AND influence_strength <= 1.0),
    CONSTRAINT no_self_influence CHECK (influencer_agent_id != influenced_agent_id),
    UNIQUE(influencer_agent_id, influenced_agent_id, influence_type)
);

-- Community leadership tracking
CREATE TABLE IF NOT EXISTS leadership_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id TEXT NOT NULL,
    leadership_score REAL NOT NULL DEFAULT 0.0,
    influence_reach INTEGER DEFAULT 0, -- Number of agents influenced
    proposal_success_rate REAL DEFAULT 0.0,
    community_trust REAL DEFAULT 0.5,
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    CONSTRAINT valid_leadership CHECK (leadership_score >= 0.0 AND leadership_score <= 1.0),
    CONSTRAINT valid_success_rate CHECK (proposal_success_rate >= 0.0 AND proposal_success_rate <= 1.0),
    CONSTRAINT valid_trust CHECK (community_trust >= 0.0 AND community_trust <= 1.0)
);

-- =====================================================
-- Event Broadcasting and Coordination
-- =====================================================

-- Community events that affect multiple agents
CREATE TABLE IF NOT EXISTS community_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_name TEXT NOT NULL,
    event_type TEXT NOT NULL, -- 'vote', 'celebration', 'crisis', 'rule_change', 'meme_emergence'
    description TEXT,
    initiated_by_agent_id TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    participants TEXT[] DEFAULT '{}',
    impact_level REAL DEFAULT 0.5, -- How much this event affects the community
    cultural_significance REAL DEFAULT 0.0, -- Long-term cultural impact
    metadata JSONB DEFAULT '{}',
    CONSTRAINT valid_impact CHECK (impact_level >= 0.0 AND impact_level <= 1.0),
    CONSTRAINT valid_significance CHECK (cultural_significance >= 0.0 AND cultural_significance <= 1.0)
);

-- Agent participation in community events
CREATE TABLE IF NOT EXISTS event_participation (
    participation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id UUID NOT NULL REFERENCES community_events(event_id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL,
    participation_level REAL NOT NULL DEFAULT 0.5, -- How actively they participated
    role_in_event TEXT, -- 'organizer', 'participant', 'observer', 'opponent'
    impact_on_agent REAL DEFAULT 0.0, -- How much the event affected this agent
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    left_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT valid_participation CHECK (participation_level >= 0.0 AND participation_level <= 1.0),
    CONSTRAINT valid_agent_impact CHECK (impact_on_agent >= -1.0 AND impact_on_agent <= 1.0),
    UNIQUE(event_id, agent_id)
);

-- =====================================================
-- Performance and Analytics Tracking
-- =====================================================

-- Cultural system metrics
CREATE TABLE IF NOT EXISTS cultural_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    context JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Governance effectiveness metrics  
CREATE TABLE IF NOT EXISTS governance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    context JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- Indexes for Optimal Performance
-- =====================================================

-- Store API indexes
CREATE INDEX IF NOT EXISTS idx_store_items_key ON store_items(key);
CREATE INDEX IF NOT EXISTS idx_store_items_namespace ON store_items(namespace);
CREATE INDEX IF NOT EXISTS idx_store_items_expires ON store_items(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_store_items_updated ON store_items(updated_at DESC);

-- Cultural system indexes
CREATE INDEX IF NOT EXISTS idx_cultural_memes_strength ON cultural_memes(strength DESC);
CREATE INDEX IF NOT EXISTS idx_cultural_memes_type ON cultural_memes(meme_type);
CREATE INDEX IF NOT EXISTS idx_cultural_memes_active ON cultural_memes(is_active) WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_agent_meme_adoption_agent ON agent_meme_adoption(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_meme_adoption_meme ON agent_meme_adoption(meme_id);
CREATE INDEX IF NOT EXISTS idx_agent_meme_adoption_strength ON agent_meme_adoption(influence_strength DESC);
CREATE INDEX IF NOT EXISTS idx_agent_meme_adoption_active ON agent_meme_adoption(is_active) WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_meme_transmissions_source ON meme_transmissions(source_agent_id);
CREATE INDEX IF NOT EXISTS idx_meme_transmissions_target ON meme_transmissions(target_agent_id);
CREATE INDEX IF NOT EXISTS idx_meme_transmissions_time ON meme_transmissions(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_meme_transmissions_success ON meme_transmissions(success) WHERE success = true;

CREATE INDEX IF NOT EXISTS idx_agent_value_alignment_agent ON agent_value_alignment(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_value_alignment_value ON agent_value_alignment(value_id);
CREATE INDEX IF NOT EXISTS idx_agent_value_alignment_strength ON agent_value_alignment(alignment_strength DESC);

-- Governance system indexes
CREATE INDEX IF NOT EXISTS idx_governance_rules_type ON governance_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_governance_rules_priority ON governance_rules(priority DESC);
CREATE INDEX IF NOT EXISTS idx_governance_rules_active ON governance_rules(is_active) WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_governance_proposals_status ON governance_proposals(status);
CREATE INDEX IF NOT EXISTS idx_governance_proposals_proposer ON governance_proposals(proposed_by_agent_id);
CREATE INDEX IF NOT EXISTS idx_governance_proposals_voting_ends ON governance_proposals(voting_ends_at DESC);

CREATE INDEX IF NOT EXISTS idx_votes_proposal ON votes(proposal_id);
CREATE INDEX IF NOT EXISTS idx_votes_voter ON votes(voter_agent_id);
CREATE INDEX IF NOT EXISTS idx_votes_cast_time ON votes(cast_at DESC);

CREATE INDEX IF NOT EXISTS idx_rule_compliance_agent ON rule_compliance(agent_id);
CREATE INDEX IF NOT EXISTS idx_rule_compliance_rule ON rule_compliance(rule_id);
CREATE INDEX IF NOT EXISTS idx_rule_compliance_score ON rule_compliance(compliance_score DESC);

CREATE INDEX IF NOT EXISTS idx_rule_violations_agent ON rule_violations(agent_id);
CREATE INDEX IF NOT EXISTS idx_rule_violations_rule ON rule_violations(rule_id);
CREATE INDEX IF NOT EXISTS idx_rule_violations_time ON rule_violations(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_rule_violations_unresolved ON rule_violations(resolved) WHERE resolved = false;

-- Social network indexes
CREATE INDEX IF NOT EXISTS idx_influence_network_influencer ON influence_network(influencer_agent_id);
CREATE INDEX IF NOT EXISTS idx_influence_network_influenced ON influence_network(influenced_agent_id);
CREATE INDEX IF NOT EXISTS idx_influence_network_strength ON influence_network(influence_strength DESC);
CREATE INDEX IF NOT EXISTS idx_influence_network_type ON influence_network(influence_type);
CREATE INDEX IF NOT EXISTS idx_influence_network_active ON influence_network(is_active) WHERE is_active = true;

CREATE INDEX IF NOT EXISTS idx_leadership_metrics_agent ON leadership_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_leadership_metrics_score ON leadership_metrics(leadership_score DESC);
CREATE INDEX IF NOT EXISTS idx_leadership_metrics_time ON leadership_metrics(calculated_at DESC);

-- Community event indexes
CREATE INDEX IF NOT EXISTS idx_community_events_type ON community_events(event_type);
CREATE INDEX IF NOT EXISTS idx_community_events_time ON community_events(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_community_events_initiator ON community_events(initiated_by_agent_id);
CREATE INDEX IF NOT EXISTS idx_community_events_participants ON community_events USING GIN(participants);

CREATE INDEX IF NOT EXISTS idx_event_participation_event ON event_participation(event_id);
CREATE INDEX IF NOT EXISTS idx_event_participation_agent ON event_participation(agent_id);
CREATE INDEX IF NOT EXISTS idx_event_participation_level ON event_participation(participation_level DESC);

-- =====================================================
-- Functions for Cultural and Governance Operations
-- =====================================================

-- Function to calculate meme virality
CREATE OR REPLACE FUNCTION calculate_meme_virality(target_meme_id UUID)
RETURNS REAL AS $$
DECLARE
    transmission_success_rate REAL;
    adoption_rate REAL;
    virality_score REAL;
BEGIN
    -- Calculate transmission success rate
    SELECT COALESCE(
        AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END), 0.0
    ) INTO transmission_success_rate
    FROM meme_transmissions 
    WHERE meme_id = target_meme_id 
    AND recorded_at > CURRENT_TIMESTAMP - INTERVAL '24 hours';
    
    -- Calculate adoption rate
    SELECT COALESCE(
        COUNT(DISTINCT agent_id)::REAL / NULLIF(
            (SELECT COUNT(DISTINCT agent_id) FROM agent_meme_adoption), 0
        ), 0.0
    ) INTO adoption_rate
    FROM agent_meme_adoption
    WHERE meme_id = target_meme_id AND is_active = true;
    
    virality_score := (transmission_success_rate * 0.6) + (adoption_rate * 0.4);
    
    -- Update meme virality
    UPDATE cultural_memes 
    SET virality = virality_score, updated_at = CURRENT_TIMESTAMP
    WHERE meme_id = target_meme_id;
    
    RETURN virality_score;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate proposal voting results
CREATE OR REPLACE FUNCTION calculate_proposal_results(target_proposal_id UUID)
RETURNS TABLE(
    total_votes INTEGER,
    approve_votes REAL,
    reject_votes REAL,
    abstain_votes REAL,
    approval_percentage REAL,
    passed BOOLEAN
) AS $$
DECLARE
    required_majority REAL;
BEGIN
    -- Get required majority for this proposal
    SELECT p.required_majority INTO required_majority
    FROM governance_proposals p
    WHERE p.proposal_id = target_proposal_id;
    
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as total_votes,
        COALESCE(SUM(CASE WHEN vote_choice = 'approve' THEN vote_strength ELSE 0 END), 0) as approve_votes,
        COALESCE(SUM(CASE WHEN vote_choice = 'reject' THEN vote_strength ELSE 0 END), 0) as reject_votes,
        COALESCE(SUM(CASE WHEN vote_choice = 'abstain' THEN vote_strength ELSE 0 END), 0) as abstain_votes,
        CASE 
            WHEN COUNT(*) > 0 THEN 
                COALESCE(SUM(CASE WHEN vote_choice = 'approve' THEN vote_strength ELSE 0 END), 0) / 
                NULLIF(SUM(vote_strength), 0)
            ELSE 0.0 
        END as approval_percentage,
        CASE 
            WHEN COUNT(*) > 0 AND 
                 COALESCE(SUM(CASE WHEN vote_choice = 'approve' THEN vote_strength ELSE 0 END), 0) / 
                 NULLIF(SUM(vote_strength), 0) >= required_majority
            THEN TRUE 
            ELSE FALSE 
        END as passed
    FROM votes 
    WHERE proposal_id = target_proposal_id AND is_valid = true;
END;
$$ LANGUAGE plpgsql;

-- Function to update agent influence network
CREATE OR REPLACE FUNCTION update_influence_network()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER;
BEGIN
    -- Decay influence strengths over time
    UPDATE influence_network 
    SET influence_strength = GREATEST(influence_strength * 0.95, 0.01)
    WHERE last_interaction < CURRENT_TIMESTAMP - INTERVAL '7 days'
    AND is_active = true;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    
    -- Remove very weak influences
    UPDATE influence_network 
    SET is_active = false
    WHERE influence_strength < 0.05 AND is_active = true;
    
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Triggers for Automated Updates
-- =====================================================

-- Update timestamp trigger for store items
CREATE TRIGGER update_store_items_updated_at 
    BEFORE UPDATE ON store_items
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Update cultural memes timestamp
CREATE TRIGGER update_cultural_memes_updated_at 
    BEFORE UPDATE ON cultural_memes
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- Views for Common Store API Queries
-- =====================================================

-- Active cultural memes view
CREATE OR REPLACE VIEW active_cultural_memes AS
SELECT 
    cm.meme_id,
    cm.meme_name,
    cm.meme_type,
    cm.strength,
    cm.virality,
    COUNT(ama.agent_id) as adopted_by_count,
    AVG(ama.influence_strength) as avg_influence_strength
FROM cultural_memes cm
LEFT JOIN agent_meme_adoption ama ON cm.meme_id = ama.meme_id AND ama.is_active = true
WHERE cm.is_active = true
GROUP BY cm.meme_id, cm.meme_name, cm.meme_type, cm.strength, cm.virality;

-- Governance status view
CREATE OR REPLACE VIEW governance_status AS
SELECT 
    'active_rules' as metric,
    COUNT(*)::TEXT as value
FROM governance_rules 
WHERE is_active = true

UNION ALL

SELECT 
    'pending_proposals' as metric,
    COUNT(*)::TEXT as value
FROM governance_proposals 
WHERE status = 'voting'

UNION ALL

SELECT 
    'average_compliance' as metric,
    ROUND(AVG(compliance_score)::NUMERIC, 3)::TEXT as value
FROM rule_compliance;

-- Community influence leaders view
CREATE OR REPLACE VIEW influence_leaders AS
SELECT 
    lm.agent_id,
    lm.leadership_score,
    lm.influence_reach,
    lm.community_trust,
    COUNT(DISTINCT ara.role_id) as active_roles_count,
    COUNT(DISTINCT iv.influenced_agent_id) as direct_influence_count
FROM leadership_metrics lm
LEFT JOIN agent_role_assignments ara ON lm.agent_id = ara.agent_id AND ara.is_active = true
LEFT JOIN influence_network iv ON lm.agent_id = iv.influencer_agent_id AND iv.is_active = true
WHERE lm.calculated_at = (
    SELECT MAX(calculated_at) 
    FROM leadership_metrics lm2 
    WHERE lm2.agent_id = lm.agent_id
)
GROUP BY lm.agent_id, lm.leadership_score, lm.influence_reach, lm.community_trust
ORDER BY lm.leadership_score DESC;

-- Insert initial namespaces for organization
INSERT INTO store_namespaces (namespace_name, description) VALUES 
    ('cultural_memes', 'Shared cultural memes and their propagation'),
    ('governance', 'Democratic governance rules and proposals'),
    ('social_roles', 'Community social roles and assignments'),
    ('influence_network', 'Agent influence and leadership metrics'),
    ('community_events', 'Shared community events and coordination')
ON CONFLICT (namespace_name) DO NOTHING;

-- Grant permissions (adjust username as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dating_show_app;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dating_show_app;

-- =====================================================
-- Comments for Documentation
-- =====================================================

COMMENT ON TABLE store_items IS 'Core LangGraph Store API items table for cross-agent shared state';
COMMENT ON TABLE cultural_memes IS 'Registry of cultural memes that spread between agents';
COMMENT ON TABLE agent_meme_adoption IS 'Tracking which agents have adopted which memes';
COMMENT ON TABLE governance_rules IS 'Constitutional and governance rules created democratically';
COMMENT ON TABLE governance_proposals IS 'Democratic proposals for rule changes';
COMMENT ON TABLE votes IS 'Democratic voting records for governance decisions';
COMMENT ON TABLE influence_network IS 'Social influence relationships between agents';
COMMENT ON TABLE community_events IS 'Community-wide events affecting multiple agents';

-- Store API schema creation complete
-- This schema supports:
-- - LangGraph Store API for cross-agent shared state
-- - Cultural meme propagation and evolution
-- - Democratic governance with voting and rule creation
-- - Social influence networks and community leadership
-- - Community event coordination and participation tracking
-- - Performance optimized for 50+ concurrent agents with <200ms cultural propagation