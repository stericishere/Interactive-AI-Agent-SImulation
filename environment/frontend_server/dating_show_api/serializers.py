"""
Dating Show API Serializers
Django REST framework serializers for the dating show API
"""

from django.utils import timezone
from .models import (
    Agent, AgentSkill, SocialRelationship, GovernanceVote, VoteCast,
    ConstitutionalRule, ComplianceViolation, AgentMemorySnapshot, SimulationState
)


class AgentSerializer:
    """Serializer for Agent model"""
    
    def __init__(self, instance, many=False):
        self.instance = instance
        self.many = many
    
    @property
    def data(self):
        if self.many:
            return [self._serialize_agent(agent) for agent in self.instance]
        return self._serialize_agent(self.instance)
    
    def _serialize_agent(self, agent):
        return {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'current_role': agent.current_role,
            'specialization': agent.specialization,
            'created_at': agent.created_at.isoformat(),
            'updated_at': agent.updated_at.isoformat()
        }


class AgentSkillSerializer:
    """Serializer for AgentSkill model"""
    
    def __init__(self, instance, many=False):
        self.instance = instance
        self.many = many
    
    @property
    def data(self):
        if self.many:
            return [self._serialize_skill(skill) for skill in self.instance]
        return self._serialize_skill(self.instance)
    
    def _serialize_skill(self, skill):
        return {
            'skill_name': skill.skill_name,
            'skill_level': skill.skill_level,
            'experience_points': skill.experience_points,
            'last_practiced': skill.last_practiced.isoformat() if skill.last_practiced else None,
            'discovery_date': skill.discovery_date.isoformat()
        }


class SocialRelationshipSerializer:
    """Serializer for SocialRelationship model"""
    
    def __init__(self, instance, many=False):
        self.instance = instance
        self.many = many
    
    @property
    def data(self):
        if self.many:
            return [self._serialize_relationship(rel) for rel in self.instance]
        return self._serialize_relationship(self.instance)
    
    def _serialize_relationship(self, relationship):
        return {
            'agent_a': {
                'agent_id': relationship.agent_a.agent_id,
                'name': relationship.agent_a.name
            },
            'agent_b': {
                'agent_id': relationship.agent_b.agent_id,
                'name': relationship.agent_b.name
            },
            'relationship_type': relationship.relationship_type,
            'strength': relationship.strength,
            'established_date': relationship.established_date.isoformat(),
            'last_interaction': relationship.last_interaction.isoformat() if relationship.last_interaction else None
        }


class GovernanceVoteSerializer:
    """Serializer for GovernanceVote model"""
    
    def __init__(self, instance, many=False):
        self.instance = instance
        self.many = many
    
    @property
    def data(self):
        if self.many:
            return [self._serialize_vote(vote) for vote in self.instance]
        return self._serialize_vote(self.instance)
    
    def _serialize_vote(self, vote):
        # Count votes
        votes_cast = vote.cast_votes.all()
        vote_counts = {'yes': 0, 'no': 0, 'abstain': 0}
        for cast_vote in votes_cast:
            vote_counts[cast_vote.choice] += cast_vote.weight
        
        return {
            'vote_id': vote.vote_id,
            'vote_type': vote.vote_type,
            'title': vote.title,
            'description': vote.description,
            'proposed_by': {
                'agent_id': vote.proposed_by.agent_id,
                'name': vote.proposed_by.name
            },
            'created_at': vote.created_at.isoformat(),
            'voting_deadline': vote.voting_deadline.isoformat(),
            'is_active': vote.is_active,
            'result': vote.result,
            'vote_counts': vote_counts,
            'total_votes_cast': sum(vote_counts.values())
        }


class ConstitutionalRuleSerializer:
    """Serializer for ConstitutionalRule model"""
    
    def __init__(self, instance, many=False):
        self.instance = instance
        self.many = many
    
    @property
    def data(self):
        if self.many:
            return [self._serialize_rule(rule) for rule in self.instance]
        return self._serialize_rule(self.instance)
    
    def _serialize_rule(self, rule):
        return {
            'rule_id': rule.rule_id,
            'category': rule.category,
            'title': rule.title,
            'content': rule.content,
            'priority': rule.priority,
            'created_at': rule.created_at.isoformat(),
            'amended_at': rule.amended_at.isoformat() if rule.amended_at else None,
            'is_active': rule.is_active,
            'created_by_vote': rule.created_by_vote.vote_id if rule.created_by_vote else None
        }


class AgentMemorySnapshotSerializer:
    """Serializer for AgentMemorySnapshot model"""
    
    def __init__(self, instance, many=False):
        self.instance = instance
        self.many = many
    
    @property
    def data(self):
        if self.many:
            return [self._serialize_memory(memory) for memory in self.instance]
        return self._serialize_memory(self.instance)
    
    def _serialize_memory(self, memory):
        return {
            'memory_type': memory.memory_type,
            'content': memory.content,
            'importance_score': memory.importance_score,
            'created_at': memory.created_at.isoformat()
        }


class ComplianceViolationSerializer:
    """Serializer for ComplianceViolation model"""
    
    def __init__(self, instance, many=False):
        self.instance = instance
        self.many = many
    
    @property
    def data(self):
        if self.many:
            return [self._serialize_violation(violation) for violation in self.instance]
        return self._serialize_violation(self.instance)
    
    def _serialize_violation(self, violation):
        return {
            'id': violation.id,
            'agent': {
                'agent_id': violation.agent.agent_id,
                'name': violation.agent.name
            },
            'rule': {
                'rule_id': violation.rule.rule_id,
                'title': violation.rule.title
            },
            'severity': violation.severity,
            'description': violation.description,
            'detected_at': violation.detected_at.isoformat(),
            'resolved': violation.resolved,
            'punishment_applied': violation.punishment_applied
        }


class SimulationStateSerializer:
    """Serializer for SimulationState model"""
    
    def __init__(self, instance, many=False):
        self.instance = instance
        self.many = many
    
    @property
    def data(self):
        if self.many:
            return [self._serialize_simulation(sim) for sim in self.instance]
        return self._serialize_simulation(self.instance)
    
    def _serialize_simulation(self, simulation):
        return {
            'simulation_id': simulation.simulation_id,
            'status': simulation.status,
            'current_step': simulation.current_step,
            'total_agents': simulation.total_agents,
            'active_agents': simulation.active_agents,
            'current_time': simulation.current_time.isoformat(),
            'performance_metrics': simulation.performance_metrics,
            'created_at': simulation.created_at.isoformat(),
            'updated_at': simulation.updated_at.isoformat()
        }