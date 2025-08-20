from django.contrib import admin
from .models import (
    Agent, AgentSkill, SocialRelationship, GovernanceVote, VoteCast,
    ConstitutionalRule, ComplianceViolation, AgentMemorySnapshot, SimulationState
)


@admin.register(Agent)
class AgentAdmin(admin.ModelAdmin):
    list_display = ('agent_id', 'name', 'current_role', 'created_at', 'updated_at')
    list_filter = ('current_role', 'created_at')
    search_fields = ('agent_id', 'name', 'current_role')
    readonly_fields = ('created_at', 'updated_at')


@admin.register(AgentSkill)
class AgentSkillAdmin(admin.ModelAdmin):
    list_display = ('agent', 'skill_name', 'skill_level', 'experience_points', 'last_practiced')
    list_filter = ('skill_name', 'discovery_date')
    search_fields = ('agent__name', 'skill_name')


@admin.register(SocialRelationship)
class SocialRelationshipAdmin(admin.ModelAdmin):
    list_display = ('agent_a', 'agent_b', 'relationship_type', 'strength', 'established_date')
    list_filter = ('relationship_type', 'established_date')
    search_fields = ('agent_a__name', 'agent_b__name')


@admin.register(GovernanceVote)
class GovernanceVoteAdmin(admin.ModelAdmin):
    list_display = ('vote_id', 'title', 'vote_type', 'proposed_by', 'is_active', 'created_at')
    list_filter = ('vote_type', 'is_active', 'created_at')
    search_fields = ('title', 'description', 'vote_id')


@admin.register(VoteCast)
class VoteCastAdmin(admin.ModelAdmin):
    list_display = ('vote', 'agent', 'choice', 'weight', 'cast_at')
    list_filter = ('choice', 'cast_at')
    search_fields = ('vote__title', 'agent__name')


@admin.register(ConstitutionalRule)
class ConstitutionalRuleAdmin(admin.ModelAdmin):
    list_display = ('rule_id', 'title', 'category', 'priority', 'is_active', 'created_at')
    list_filter = ('category', 'is_active', 'created_at')
    search_fields = ('rule_id', 'title', 'content')


@admin.register(ComplianceViolation)
class ComplianceViolationAdmin(admin.ModelAdmin):
    list_display = ('agent', 'rule', 'severity', 'resolved', 'detected_at')
    list_filter = ('severity', 'resolved', 'detected_at')
    search_fields = ('agent__name', 'rule__title', 'description')


@admin.register(AgentMemorySnapshot)
class AgentMemorySnapshotAdmin(admin.ModelAdmin):
    list_display = ('agent', 'memory_type', 'importance_score', 'created_at')
    list_filter = ('memory_type', 'created_at')
    search_fields = ('agent__name', 'memory_type')


@admin.register(SimulationState)
class SimulationStateAdmin(admin.ModelAdmin):
    list_display = ('simulation_id', 'status', 'current_step', 'total_agents', 'active_agents', 'updated_at')
    list_filter = ('status', 'created_at')
    search_fields = ('simulation_id',)
    readonly_fields = ('created_at', 'updated_at')