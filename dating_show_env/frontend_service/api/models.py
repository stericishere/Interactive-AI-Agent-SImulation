from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator, RegexValidator
from django.core.exceptions import ValidationError
from django.utils import timezone
# from django.contrib.postgres.fields import JSONField  # Deprecated in Django 3.1+
from django.db.models import JSONField
from typing import Dict, List, Optional, Union, Any
import json
import re
from datetime import timedelta


# Type aliases for better code documentation
AgentId = str
SkillName = str
RuleId = str
VoteId = str
SimulationId = str
MemoryContent = Dict[str, Any]
MetricsData = Dict[str, Union[int, float, str]]


class Agent(models.Model):
    """Enhanced model for agent basic information with validation.
    
    Represents an individual agent in the simulation with comprehensive
    tracking of roles, performance, and activity metrics.
    
    Attributes:
        agent_id (str): Unique identifier for the agent
        name (str): Human-readable name for the agent
        current_role (str): Current role in the simulation (contestant, host, etc.)
        specialization (dict): JSON field containing specialized capabilities
        is_active (bool): Whether the agent is currently active
        last_activity (datetime): Timestamp of last recorded activity
        performance_rating (float): Performance score from 0.0 to 10.0
        created_at (datetime): Agent creation timestamp
        updated_at (datetime): Last update timestamp
    """
    
    # Role choices for better data consistency
    ROLE_CHOICES = [
        ('contestant', 'Contestant'),
        ('host', 'Host'),
        ('producer', 'Producer'),
        ('participant', 'Participant'),
        ('observer', 'Observer'),
    ]
    
    agent_id = models.CharField(
        max_length=100, 
        unique=True, 
        primary_key=True,
        validators=[RegexValidator(r'^[a-zA-Z0-9_-]+$', 'Agent ID can only contain letters, numbers, underscores, and hyphens')]
    )
    name = models.CharField(
        max_length=200,
        validators=[RegexValidator(r'^[a-zA-Z0-9 .-]+$', 'Name can only contain letters, numbers, spaces, periods, and hyphens')]
    )
    current_role = models.CharField(max_length=100, choices=ROLE_CHOICES, blank=True, default='participant')
    specialization = JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    last_activity = models.DateTimeField(null=True, blank=True)
    performance_rating = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(10.0)],
        help_text="Performance rating from 0.0 to 10.0"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        app_label = 'dating_show_api'
        ordering = ['name']
        indexes = [
            models.Index(fields=['current_role']),
            models.Index(fields=['is_active']),
            models.Index(fields=['created_at']),
        ]
    
    def clean(self):
        """Custom validation for Agent model"""
        super().clean()
        
        # Validate agent_id format
        if not self.agent_id or len(self.agent_id.strip()) == 0:
            raise ValidationError('Agent ID cannot be empty')
        
        # Validate specialization JSON structure
        if self.specialization and not isinstance(self.specialization, dict):
            raise ValidationError('Specialization must be a valid JSON object')
    
    def save(self, *args, **kwargs):
        """Override save to add custom logic"""
        self.full_clean()  # Run validation
        if not self.last_activity:
            self.last_activity = timezone.now()
        super().save(*args, **kwargs)
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = timezone.now()
        self.save(update_fields=['last_activity'])
    
    def get_skill_count(self) -> int:
        """Get total number of skills for this agent."""
        return self.skills.count()
    
    def get_average_skill_level(self) -> float:
        """Get average skill level across all skills."""
        skills = self.skills.all()
        if not skills:
            return 0.0
        return sum(skill.skill_level for skill in skills) / len(skills)
    
    def get_relationship_count(self) -> int:
        """Get total number of relationships (both directions)."""
        return (self.relationships_as_a.count() + self.relationships_as_b.count())
    
    def is_recently_active(self, hours: int = 24) -> bool:
        """Check if agent has been active within specified hours."""
        if not self.last_activity:
            return False
        cutoff = timezone.now() - timedelta(hours=hours)
        return self.last_activity > cutoff
    
    def __str__(self):
        return f"{self.name} ({self.agent_id})"


class AgentSkill(models.Model):
    """Enhanced model for tracking agent skill progression with validation and methods.
    
    Manages individual skills for agents including experience tracking,
    decay mechanics, and proficiency ranking.
    
    Attributes:
        agent (Agent): Foreign key to the owning agent
        skill_name (str): Name of the skill
        skill_level (float): Current skill level from 0.0 to 100.0
        experience_points (float): Total accumulated experience points
        category (str): Skill category (physical, mental, social, etc.)
        proficiency_rank (str): Auto-calculated proficiency level
        decay_rate (float): Daily skill decay rate
        last_practiced (datetime): Last time skill was practiced
        discovery_date (datetime): When skill was first discovered
        practice_count (int): Total number of practice sessions
        mastery_bonus (float): Bonus points for skill mastery
    """
    
    # Skill categories for better organization
    SKILL_CATEGORIES = [
        ('physical', 'Physical'),
        ('mental', 'Mental'),
        ('social', 'Social'),
        ('creative', 'Creative'),
        ('technical', 'Technical'),
        ('survival', 'Survival'),
    ]
    
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='skills')
    skill_name = models.CharField(
        max_length=100,
        validators=[RegexValidator(r'^[a-zA-Z0-9_\s-]+$', 'Skill name can only contain letters, numbers, spaces, underscores, and hyphens')]
    )
    skill_level = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        help_text="Skill level from 0.0 to 100.0"
    )
    experience_points = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Total experience points accumulated"
    )
    category = models.CharField(max_length=20, choices=SKILL_CATEGORIES, blank=True)
    proficiency_rank = models.CharField(max_length=20, blank=True, editable=False)
    decay_rate = models.FloatField(
        default=0.01,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Daily skill decay rate (0.0-1.0)"
    )
    last_practiced = models.DateTimeField(null=True, blank=True)
    discovery_date = models.DateTimeField(auto_now_add=True)
    practice_count = models.PositiveIntegerField(default=0)
    mastery_bonus = models.FloatField(default=0.0, validators=[MinValueValidator(0.0)])
    
    class Meta:
        app_label = 'dating_show_api'
        unique_together = ['agent', 'skill_name']
        ordering = ['-skill_level', 'skill_name']
        indexes = [
            models.Index(fields=['skill_name']),
            models.Index(fields=['skill_level']),
            models.Index(fields=['category']),
            models.Index(fields=['last_practiced']),
        ]
    
    def clean(self):
        """Custom validation for AgentSkill model"""
        super().clean()
        
        if self.skill_level < 0 or self.skill_level > 100:
            raise ValidationError('Skill level must be between 0.0 and 100.0')
        
        if self.experience_points < 0:
            raise ValidationError('Experience points cannot be negative')
        
        # Auto-assign category based on skill name if not provided
        if not self.category:
            self.category = self._guess_category()
    
    def save(self, *args, **kwargs):
        """Override save to add custom logic"""
        self.full_clean()
        self._update_proficiency_rank()
        super().save(*args, **kwargs)
        
        # Update agent's last activity when skill is practiced
        if self.last_practiced:
            self.agent.update_activity()
    
    def _guess_category(self):
        """Guess skill category based on skill name"""
        name_lower = self.skill_name.lower()
        
        # Physical skills
        if any(word in name_lower for word in ['combat', 'athletics', 'stealth', 'acrobatics']):
            return 'physical'
        
        # Mental skills
        if any(word in name_lower for word in ['reasoning', 'memory', 'analysis', 'focus', 'learning']):
            return 'mental'
        
        # Social skills
        if any(word in name_lower for word in ['persuasion', 'empathy', 'leadership', 'negotiation']):
            return 'social'
        
        # Creative skills
        if any(word in name_lower for word in ['creativity', 'art', 'music', 'writing']):
            return 'creative'
        
        # Technical skills
        if any(word in name_lower for word in ['programming', 'engineering', 'research']):
            return 'technical'
        
        # Survival skills
        if any(word in name_lower for word in ['foraging', 'hunting', 'navigation', 'medicine']):
            return 'survival'
        
        return 'mental'  # Default category
    
    def _update_proficiency_rank(self):
        """Update proficiency rank based on skill level"""
        if self.skill_level < 10:
            self.proficiency_rank = 'Novice'
        elif self.skill_level < 25:
            self.proficiency_rank = 'Beginner'
        elif self.skill_level < 50:
            self.proficiency_rank = 'Competent'
        elif self.skill_level < 75:
            self.proficiency_rank = 'Proficient'
        elif self.skill_level < 90:
            self.proficiency_rank = 'Expert'
        else:
            self.proficiency_rank = 'Master'
    
    def add_experience(self, points):
        """Add experience points and update skill level"""
        if points < 0:
            raise ValueError("Experience points must be positive")
        
        self.experience_points += points
        self.practice_count += 1
        self.last_practiced = timezone.now()
        
        # Simple experience to level conversion (can be made more sophisticated)
        self.skill_level = min(100.0, self.experience_points ** 0.5 * 2.0)
        
        self.save()
    
    def days_since_practice(self):
        """Calculate days since last practice"""
        if not self.last_practiced:
            return float('inf')
        delta = timezone.now() - self.last_practiced
        return delta.days
    
    def calculate_decay(self):
        """Calculate current skill decay amount"""
        days = self.days_since_practice()
        if days == 0 or days == float('inf'):
            return 0.0
        
        # Exponential decay with decay_rate
        decay_amount = self.skill_level * (1 - (1 - self.decay_rate) ** days)
        return min(decay_amount, self.skill_level)  # Can't decay below 0
    
    def apply_decay(self):
        """Apply skill decay based on time since last practice"""
        decay = self.calculate_decay()
        if decay > 0:
            self.skill_level = max(0.0, self.skill_level - decay)
            self.save()
            return decay
        return 0.0
    
    def is_recently_practiced(self, days=7):
        """Check if skill was practiced recently"""
        return self.days_since_practice() <= days
    
    def get_progress_to_next_rank(self):
        """Get progress percentage to next proficiency rank"""
        thresholds = [10, 25, 50, 75, 90, 100]
        current_threshold_index = 0
        
        for i, threshold in enumerate(thresholds):
            if self.skill_level < threshold:
                current_threshold_index = i
                break
        else:
            return 100.0  # Already at max
        
        if current_threshold_index == 0:
            progress = (self.skill_level / thresholds[0]) * 100
        else:
            prev_threshold = thresholds[current_threshold_index - 1]
            next_threshold = thresholds[current_threshold_index]
            progress = ((self.skill_level - prev_threshold) / (next_threshold - prev_threshold)) * 100
        
        return min(100.0, max(0.0, progress))
    
    def __str__(self):
        return f"{self.agent.name}: {self.skill_name} ({self.proficiency_rank} - {self.skill_level:.1f})"


class SocialRelationship(models.Model):
    """Enhanced model for tracking relationships between agents with validation and methods.
    
    Manages bidirectional social relationships between agents with strength tracking,
    interaction history, and relationship evolution over time.
    
    Attributes:
        agent_a (Agent): First agent in the relationship
        agent_b (Agent): Second agent in the relationship
        relationship_type (str): Type of relationship (friendship, romantic, etc.)
        strength (float): Relationship strength from -1.0 to 1.0
        strength_level (str): Categorical strength level
        established_date (datetime): When relationship was established
        last_interaction (datetime): Last interaction timestamp
        interaction_count (int): Total number of interactions
        mutual (bool): Whether relationship is mutual
        notes (str): Additional relationship context
    """
    RELATIONSHIP_TYPES = [
        ('friendship', 'Friendship'),
        ('romantic', 'Romantic Interest'),
        ('alliance', 'Alliance'),
        ('rivalry', 'Rivalry'),
        ('neutral', 'Neutral'),
        ('mentorship', 'Mentorship'),
        ('conflict', 'Conflict'),
    ]
    
    STRENGTH_LEVELS = [
        ('very_weak', 'Very Weak'),
        ('weak', 'Weak'),
        ('moderate', 'Moderate'),
        ('strong', 'Strong'),
        ('very_strong', 'Very Strong'),
    ]
    
    agent_a = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='relationships_as_a')
    agent_b = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='relationships_as_b')
    relationship_type = models.CharField(max_length=20, choices=RELATIONSHIP_TYPES)
    strength = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(-1.0), MaxValueValidator(1.0)],
        help_text="Relationship strength from -1.0 (very negative) to 1.0 (very positive)"
    )
    strength_level = models.CharField(max_length=12, choices=STRENGTH_LEVELS, blank=True, editable=False)
    established_date = models.DateTimeField(auto_now_add=True)
    last_interaction = models.DateTimeField(null=True, blank=True)
    interaction_count = models.PositiveIntegerField(default=0)
    mutual = models.BooleanField(default=False, help_text="Whether this relationship is mutual")
    notes = models.TextField(blank=True, help_text="Additional relationship context")
    
    class Meta:
        app_label = 'dating_show_api'
        unique_together = ['agent_a', 'agent_b']
        ordering = ['-strength', '-last_interaction']
        indexes = [
            models.Index(fields=['relationship_type']),
            models.Index(fields=['strength']),
            models.Index(fields=['last_interaction']),
            models.Index(fields=['mutual']),
        ]
    
    def clean(self):
        """Custom validation for SocialRelationship model"""
        super().clean()
        
        # Prevent self-relationships
        if self.agent_a == self.agent_b:
            raise ValidationError('Agents cannot have relationships with themselves')
        
        # Validate strength range
        if self.strength < -1.0 or self.strength > 1.0:
            raise ValidationError('Relationship strength must be between -1.0 and 1.0')
    
    def save(self, *args, **kwargs):
        """Override save to add custom logic"""
        self.full_clean()
        self._update_strength_level()
        super().save(*args, **kwargs)
        
        # Update agents' last activity
        self.agent_a.update_activity()
        self.agent_b.update_activity()
    
    def _update_strength_level(self):
        """Update strength level based on numeric strength"""
        abs_strength = abs(self.strength)
        if abs_strength < 0.2:
            self.strength_level = 'very_weak'
        elif abs_strength < 0.4:
            self.strength_level = 'weak'
        elif abs_strength < 0.6:
            self.strength_level = 'moderate'
        elif abs_strength < 0.8:
            self.strength_level = 'strong'
        else:
            self.strength_level = 'very_strong'
    
    def update_interaction(self, strength_change=0.0):
        """Update interaction timestamp and optionally adjust strength"""
        self.last_interaction = timezone.now()
        self.interaction_count += 1
        
        if strength_change != 0.0:
            new_strength = self.strength + strength_change
            self.strength = max(-1.0, min(1.0, new_strength))  # Clamp to valid range
        
        self.save()
    
    def get_reciprocal_relationship(self):
        """Get the reciprocal relationship if it exists"""
        try:
            return SocialRelationship.objects.get(agent_a=self.agent_b, agent_b=self.agent_a)
        except SocialRelationship.DoesNotExist:
            return None
    
    def is_positive(self):
        """Check if this is a positive relationship"""
        return self.strength > 0
    
    def is_negative(self):
        """Check if this is a negative relationship"""
        return self.strength < 0
    
    def days_since_interaction(self):
        """Calculate days since last interaction"""
        if not self.last_interaction:
            return float('inf')
        delta = timezone.now() - self.last_interaction
        return delta.days
    
    def is_recently_active(self, days=7):
        """Check if relationship has been active recently"""
        return self.days_since_interaction() <= days
    
    def __str__(self):
        return f"{self.agent_a.name} -> {self.agent_b.name}: {self.relationship_type} ({self.strength_level}, {self.strength:.2f})"


class GovernanceVote(models.Model):
    """Enhanced model for tracking governance voting history with validation and methods.
    
    Manages democratic voting processes within the agent society including
    proposal creation, voting deadlines, and result calculation.
    
    Attributes:
        vote_id (str): Unique identifier for the vote
        vote_type (str): Type of vote (rule_proposal, amendment, etc.)
        title (str): Vote title/summary
        description (str): Detailed vote description
        proposed_by (Agent): Agent who proposed the vote
        created_at (datetime): Vote creation timestamp
        voting_deadline (datetime): Deadline for casting votes
        is_active (bool): Whether voting is currently active
        status (str): Current vote status (draft, active, completed, etc.)
        result (str): Final vote result (yes, no, abstain)
        required_majority (float): Required majority threshold
        minimum_participation (float): Minimum participation rate
        total_eligible_voters (int): Total eligible voters
    """
    VOTE_TYPES = [
        ('rule_proposal', 'Rule Proposal'),
        ('amendment', 'Constitutional Amendment'),
        ('punishment', 'Punishment Decision'),
        ('resource_allocation', 'Resource Allocation'),
        ('leadership_election', 'Leadership Election'),
        ('policy_change', 'Policy Change'),
    ]
    
    VOTE_CHOICES = [
        ('yes', 'Yes'),
        ('no', 'No'),
        ('abstain', 'Abstain'),
    ]
    
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('active', 'Active Voting'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]
    
    vote_id = models.CharField(
        max_length=100, 
        unique=True,
        validators=[RegexValidator(r'^[a-zA-Z0-9_-]+$', 'Vote ID can only contain letters, numbers, underscores, and hyphens')]
    )
    vote_type = models.CharField(max_length=30, choices=VOTE_TYPES)
    title = models.CharField(
        max_length=200,
        validators=[RegexValidator(r'^[\w\s.,!?()-]+$', 'Title contains invalid characters')]
    )
    description = models.TextField(validators=[RegexValidator(r'^[\w\s.,!?()\n\r-]+$', 'Description contains invalid characters')])
    proposed_by = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='proposed_votes')
    created_at = models.DateTimeField(auto_now_add=True)
    voting_deadline = models.DateTimeField()
    is_active = models.BooleanField(default=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    result = models.CharField(max_length=10, choices=VOTE_CHOICES, null=True, blank=True)
    required_majority = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Required majority as decimal (0.5 = 50%)"
    )
    minimum_participation = models.FloatField(
        default=0.25,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Minimum participation rate as decimal (0.25 = 25%)"
    )
    total_eligible_voters = models.PositiveIntegerField(default=0)
    
    class Meta:
        app_label = 'dating_show_api'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['vote_type']),
            models.Index(fields=['status']),
            models.Index(fields=['is_active']),
            models.Index(fields=['voting_deadline']),
        ]
    
    def clean(self):
        """Custom validation for GovernanceVote model"""
        super().clean()
        
        # Validate voting deadline is in the future
        if self.voting_deadline and self.voting_deadline <= timezone.now():
            raise ValidationError('Voting deadline must be in the future')
        
        # Validate majority requirements
        if self.required_majority <= 0 or self.required_majority > 1:
            raise ValidationError('Required majority must be between 0 and 1')
    
    def save(self, *args, **kwargs):
        """Override save to add custom logic"""
        self.full_clean()
        
        # Auto-update status based on deadline and activity
        if self.is_active and self.voting_deadline <= timezone.now():
            self.status = 'completed'
            self.is_active = False
            self._calculate_result()
        
        super().save(*args, **kwargs)
    
    def get_vote_counts(self):
        """Get vote counts for this vote"""
        votes = self.cast_votes.all()
        return {
            'yes': votes.filter(choice='yes').count(),
            'no': votes.filter(choice='no').count(),
            'abstain': votes.filter(choice='abstain').count(),
            'total': votes.count()
        }
    
    def get_participation_rate(self):
        """Calculate participation rate"""
        if self.total_eligible_voters == 0:
            return 0.0
        return self.cast_votes.count() / self.total_eligible_voters
    
    def _calculate_result(self):
        """Calculate voting result based on cast votes"""
        counts = self.get_vote_counts()
        participation_rate = self.get_participation_rate()
        
        # Check minimum participation
        if participation_rate < self.minimum_participation:
            self.result = 'abstain'  # Failed due to low participation
            return
        
        # Calculate majority
        total_decisive_votes = counts['yes'] + counts['no']
        if total_decisive_votes == 0:
            self.result = 'abstain'
            return
        
        yes_ratio = counts['yes'] / total_decisive_votes
        if yes_ratio >= self.required_majority:
            self.result = 'yes'
        else:
            self.result = 'no'
    
    def is_deadline_passed(self):
        """Check if voting deadline has passed"""
        return self.voting_deadline <= timezone.now()
    
    def days_until_deadline(self):
        """Calculate days until voting deadline"""
        if self.is_deadline_passed():
            return 0
        delta = self.voting_deadline - timezone.now()
        return delta.days
    
    def can_vote(self, agent):
        """Check if an agent can vote on this proposal"""
        if not self.is_active or self.is_deadline_passed():
            return False
        
        # Check if agent has already voted
        return not self.cast_votes.filter(agent=agent).exists()
    
    def __str__(self):
        return f"Vote: {self.title} ({self.vote_type}, {self.status})"


class VoteCast(models.Model):
    """Enhanced model for individual vote casts with validation and methods.
    
    Represents individual votes cast by agents on governance proposals
    with support for weighted voting and reasoning tracking.
    
    Attributes:
        vote (GovernanceVote): The vote this cast belongs to
        agent (Agent): Agent who cast the vote
        choice (str): Vote choice (yes, no, abstain)
        weight (float): Vote weight for weighted voting
        cast_at (datetime): When vote was cast
        reasoning (str): Agent's reasoning for their choice
        confidence (float): Agent's confidence in their choice
    """
    vote = models.ForeignKey(GovernanceVote, on_delete=models.CASCADE, related_name='cast_votes')
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='cast_votes')
    choice = models.CharField(max_length=10, choices=GovernanceVote.VOTE_CHOICES)
    weight = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(10.0)],
        help_text="Vote weight for weighted voting systems (0.0-10.0)"
    )
    cast_at = models.DateTimeField(auto_now_add=True)
    reasoning = models.TextField(
        blank=True,
        help_text="Agent's reasoning for their vote choice",
        validators=[RegexValidator(r'^[\w\s.,!?()\n\r-]*$', 'Reasoning contains invalid characters')]
    )
    confidence = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Agent's confidence in their vote (0.0-1.0)"
    )
    
    class Meta:
        app_label = 'dating_show_api'
        unique_together = ['vote', 'agent']
        ordering = ['-cast_at']
        indexes = [
            models.Index(fields=['choice']),
            models.Index(fields=['cast_at']),
            models.Index(fields=['weight']),
        ]
    
    def clean(self):
        """Custom validation for VoteCast model"""
        super().clean()
        
        # Validate vote is still active
        if self.vote and not self.vote.is_active:
            raise ValidationError('Cannot vote on inactive proposals')
        
        # Validate deadline hasn't passed
        if self.vote and self.vote.is_deadline_passed():
            raise ValidationError('Cannot vote after deadline has passed')
        
        # Validate confidence range
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValidationError('Confidence must be between 0.0 and 1.0')
    
    def save(self, *args, **kwargs):
        """Override save to add custom logic"""
        self.full_clean()
        super().save(*args, **kwargs)
        
        # Update agent's last activity
        self.agent.update_activity()
    
    def is_decisive_vote(self):
        """Check if this is a decisive vote (not abstain)"""
        return self.choice in ['yes', 'no']
    
    def get_weighted_value(self):
        """Get the weighted value of this vote"""
        if self.choice == 'yes':
            return self.weight
        elif self.choice == 'no':
            return -self.weight
        else:  # abstain
            return 0.0
    
    def days_since_cast(self):
        """Calculate days since vote was cast"""
        delta = timezone.now() - self.cast_at
        return delta.days
    
    def __str__(self):
        return f"{self.agent.name} voted {self.choice} on {self.vote.title} (weight: {self.weight})"


class ConstitutionalRule(models.Model):
    """Enhanced model for storing constitutional rules with validation and methods.
    
    Manages the legal framework of the agent society including rule creation,
    amendments, enforcement levels, and compliance tracking.
    
    Attributes:
        rule_id (str): Unique identifier for the rule
        category (str): Rule category (behavior, resource, social, etc.)
        title (str): Rule title/summary
        content (str): Full rule text
        priority (int): Rule priority from -100 to 100
        enforcement_level (str): Enforcement level (advisory, mandatory, etc.)
        created_at (datetime): Rule creation timestamp
        amended_at (datetime): Last amendment timestamp
        is_active (bool): Whether rule is currently active
        created_by_vote (GovernanceVote): Vote that created this rule
        amended_by_vote (GovernanceVote): Vote that amended this rule
        violation_count (int): Total number of violations
        compliance_rate (float): Compliance rate from 0.0 to 1.0
    """
    RULE_CATEGORIES = [
        ('behavior', 'Behavioral Rules'),
        ('resource', 'Resource Management'),
        ('social', 'Social Interaction'),
        ('governance', 'Governance Process'),
        ('punishment', 'Punishment Guidelines'),
        ('ethics', 'Ethical Guidelines'),
        ('safety', 'Safety Protocols'),
    ]
    
    ENFORCEMENT_LEVELS = [
        ('advisory', 'Advisory'),
        ('warning', 'Warning Level'),
        ('mandatory', 'Mandatory'),
        ('critical', 'Critical'),
    ]
    
    rule_id = models.CharField(
        max_length=100, 
        unique=True, 
        primary_key=True,
        validators=[RegexValidator(r'^[a-zA-Z0-9_-]+$', 'Rule ID can only contain letters, numbers, underscores, and hyphens')]
    )
    category = models.CharField(max_length=20, choices=RULE_CATEGORIES)
    title = models.CharField(
        max_length=200,
        validators=[RegexValidator(r'^[\w\s.,!?()-]+$', 'Title contains invalid characters')]
    )
    content = models.TextField(
        validators=[RegexValidator(r'^[\w\s.,!?()\n\r-]+$', 'Content contains invalid characters')]
    )
    priority = models.IntegerField(
        default=0,
        validators=[MinValueValidator(-100), MaxValueValidator(100)],
        help_text="Rule priority (-100 to 100, higher is more important)"
    )
    enforcement_level = models.CharField(max_length=12, choices=ENFORCEMENT_LEVELS, default='mandatory')
    created_at = models.DateTimeField(auto_now_add=True)
    amended_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_by_vote = models.ForeignKey(GovernanceVote, on_delete=models.SET_NULL, null=True, blank=True)
    amended_by_vote = models.ForeignKey(
        GovernanceVote, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='amended_rules'
    )
    violation_count = models.PositiveIntegerField(default=0, editable=False)
    compliance_rate = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        editable=False,
        help_text="Compliance rate (0.0-1.0)"
    )
    
    class Meta:
        app_label = 'dating_show_api'
        ordering = ['-priority', 'category', 'title']
        indexes = [
            models.Index(fields=['category']),
            models.Index(fields=['priority']),
            models.Index(fields=['is_active']),
            models.Index(fields=['enforcement_level']),
        ]
    
    def clean(self):
        """Custom validation for ConstitutionalRule model"""
        super().clean()
        
        # Validate priority range
        if self.priority < -100 or self.priority > 100:
            raise ValidationError('Priority must be between -100 and 100')
        
        # Validate rule_id format
        if not self.rule_id or len(self.rule_id.strip()) == 0:
            raise ValidationError('Rule ID cannot be empty')
    
    def save(self, *args, **kwargs):
        """Override save to add custom logic"""
        self.full_clean()
        
        # Update amended_at if content has changed
        if self.pk:
            try:
                old_rule = ConstitutionalRule.objects.get(pk=self.pk)
                if old_rule.content != self.content:
                    self.amended_at = timezone.now()
            except ConstitutionalRule.DoesNotExist:
                pass
        
        super().save(*args, **kwargs)
    
    def update_compliance_metrics(self):
        """Update violation count and compliance rate"""
        violations = self.complianceviolation_set.all()
        self.violation_count = violations.count()
        
        # Calculate compliance rate based on total agent interactions
        total_agents = Agent.objects.filter(is_active=True).count()
        if total_agents > 0:
            compliant_agents = total_agents - violations.values('agent').distinct().count()
            self.compliance_rate = compliant_agents / total_agents
        else:
            self.compliance_rate = 1.0
        
        self.save(update_fields=['violation_count', 'compliance_rate'])
    
    def get_violation_severity_breakdown(self):
        """Get breakdown of violations by severity"""
        violations = self.complianceviolation_set.all()
        return {
            'minor': violations.filter(severity='minor').count(),
            'moderate': violations.filter(severity='moderate').count(),
            'major': violations.filter(severity='major').count(),
            'severe': violations.filter(severity='severe').count(),
        }
    
    def is_frequently_violated(self, threshold=0.1):
        """Check if this rule is frequently violated"""
        return self.compliance_rate < (1.0 - threshold)
    
    def days_since_creation(self):
        """Calculate days since rule was created"""
        delta = timezone.now() - self.created_at
        return delta.days
    
    def days_since_amendment(self):
        """Calculate days since last amendment"""
        if not self.amended_at:
            return self.days_since_creation()
        delta = timezone.now() - self.amended_at
        return delta.days
    
    def __str__(self):
        return f"Rule: {self.title} ({self.category}, {self.enforcement_level})"


class ComplianceViolation(models.Model):
    """Enhanced model for tracking rule violations with validation and methods.
    
    Tracks violations of constitutional rules including severity assessment,
    resolution tracking, and appeal processes.
    
    Attributes:
        agent (Agent): Agent who violated the rule
        rule (ConstitutionalRule): Rule that was violated
        severity (str): Violation severity level
        description (str): Description of the violation
        evidence (str): Supporting evidence for the violation
        detected_at (datetime): When violation was detected
        resolved (bool): Whether violation has been resolved
        resolution_status (str): Current resolution status
        punishment_applied (str): Punishment that was applied
        resolved_at (datetime): When violation was resolved
        resolved_by (Agent): Agent who resolved the violation
        appeal_deadline (datetime): Deadline for appeals
        severity_score (int): Numeric severity score 1-10
    """
    VIOLATION_SEVERITY = [
        ('minor', 'Minor'),
        ('moderate', 'Moderate'),
        ('major', 'Major'),
        ('severe', 'Severe'),
        ('critical', 'Critical'),
    ]
    
    RESOLUTION_STATUS = [
        ('pending', 'Pending Review'),
        ('investigating', 'Under Investigation'),
        ('resolved', 'Resolved'),
        ('dismissed', 'Dismissed'),
        ('appealed', 'Under Appeal'),
    ]
    
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='violations')
    rule = models.ForeignKey(ConstitutionalRule, on_delete=models.CASCADE)
    severity = models.CharField(max_length=10, choices=VIOLATION_SEVERITY)
    description = models.TextField(
        validators=[RegexValidator(r'^[\w\s.,!?()\n\r-]+$', 'Description contains invalid characters')]
    )
    evidence = models.TextField(
        blank=True,
        help_text="Evidence supporting the violation claim",
        validators=[RegexValidator(r'^[\w\s.,!?()\n\r-]*$', 'Evidence contains invalid characters')]
    )
    detected_at = models.DateTimeField(auto_now_add=True)
    resolved = models.BooleanField(default=False)
    resolution_status = models.CharField(max_length=15, choices=RESOLUTION_STATUS, default='pending')
    punishment_applied = models.TextField(
        blank=True,
        validators=[RegexValidator(r'^[\w\s.,!?()\n\r-]*$', 'Punishment description contains invalid characters')]
    )
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolved_by = models.ForeignKey(
        Agent, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='resolved_violations'
    )
    appeal_deadline = models.DateTimeField(null=True, blank=True)
    severity_score = models.IntegerField(
        default=1,
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        help_text="Numeric severity score (1-10)"
    )
    
    class Meta:
        app_label = 'dating_show_api'
        ordering = ['-detected_at', '-severity_score']
        indexes = [
            models.Index(fields=['severity']),
            models.Index(fields=['resolved']),
            models.Index(fields=['resolution_status']),
            models.Index(fields=['detected_at']),
        ]
    
    def clean(self):
        """Custom validation for ComplianceViolation model"""
        super().clean()
        
        # Validate severity score matches severity level
        severity_scores = {
            'minor': (1, 2),
            'moderate': (3, 4),
            'major': (5, 7),
            'severe': (8, 9),
            'critical': (10, 10)
        }
        
        if self.severity in severity_scores:
            min_score, max_score = severity_scores[self.severity]
            if not (min_score <= self.severity_score <= max_score):
                self.severity_score = min_score  # Auto-correct to minimum for severity
    
    def save(self, *args, **kwargs):
        """Override save to add custom logic"""
        self.full_clean()
        
        # Auto-update resolution status when resolved flag changes
        if self.resolved and self.resolution_status == 'pending':
            self.resolution_status = 'resolved'
            self.resolved_at = timezone.now()
        
        # Set appeal deadline for major+ violations
        if self.severity in ['major', 'severe', 'critical'] and not self.appeal_deadline:
            self.appeal_deadline = timezone.now() + timedelta(days=7)
        
        super().save(*args, **kwargs)
        
        # Update rule compliance metrics
        if hasattr(self, 'rule'):
            self.rule.update_compliance_metrics()
    
    def resolve(self, resolved_by_agent, punishment="", status="resolved"):
        """Mark violation as resolved"""
        self.resolved = True
        self.resolution_status = status
        self.resolved_at = timezone.now()
        self.resolved_by = resolved_by_agent
        if punishment:
            self.punishment_applied = punishment
        self.save()
    
    def can_appeal(self):
        """Check if violation can still be appealed"""
        if not self.appeal_deadline:
            return False
        return timezone.now() <= self.appeal_deadline and self.resolution_status != 'appealed'
    
    def days_since_detection(self):
        """Calculate days since violation was detected"""
        delta = timezone.now() - self.detected_at
        return delta.days
    
    def days_until_appeal_deadline(self):
        """Calculate days until appeal deadline"""
        if not self.appeal_deadline:
            return 0
        if timezone.now() > self.appeal_deadline:
            return 0
        delta = self.appeal_deadline - timezone.now()
        return delta.days
    
    def get_severity_color(self):
        """Get color code for severity level"""
        colors = {
            'minor': 'yellow',
            'moderate': 'orange',
            'major': 'red',
            'severe': 'darkred',
            'critical': 'purple'
        }
        return colors.get(self.severity, 'gray')
    
    def is_recent(self, days=7):
        """Check if violation is recent"""
        return self.days_since_detection() <= days
    
    def __str__(self):
        return f"{self.agent.name} violated {self.rule.title} ({self.severity}, {self.resolution_status})"


class AgentMemorySnapshot(models.Model):
    """Enhanced model for storing agent memory snapshots with validation and methods.
    
    Manages agent memory storage with different memory types, importance scoring,
    access tracking, and natural decay mechanisms.
    
    Attributes:
        agent (Agent): Agent who owns this memory
        memory_type (str): Type of memory (episodic, semantic, etc.)
        content (dict): Memory content as JSON structure
        importance_score (float): Memory importance from 0.0 to 10.0
        emotional_valence (float): Emotional significance from -1.0 to 1.0
        created_at (datetime): Memory creation timestamp
        last_accessed (datetime): Last access timestamp
        access_count (int): Number of times accessed
        status (str): Memory status (active, archived, etc.)
        decay_rate (float): Daily decay rate
        tags (str): Comma-separated tags for categorization
    """
    MEMORY_TYPES = [
        ('episodic', 'Episodic Memory'),
        ('semantic', 'Semantic Memory'),
        ('temporal', 'Temporal Memory'),
        ('working', 'Working Memory'),
        ('skill', 'Skill Memory'),
        ('social', 'Social Memory'),
    ]
    
    MEMORY_STATUS = [
        ('active', 'Active'),
        ('archived', 'Archived'),
        ('deleted', 'Deleted'),
        ('decayed', 'Naturally Decayed'),
    ]
    
    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name='memory_snapshots')
    memory_type = models.CharField(max_length=50, choices=MEMORY_TYPES)
    content = JSONField(
        default=dict,
        help_text="Memory content as JSON structure"
    )
    importance_score = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(10.0)],
        help_text="Importance score from 0.0 to 10.0"
    )
    emotional_valence = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(-1.0), MaxValueValidator(1.0)],
        help_text="Emotional valence from -1.0 (negative) to 1.0 (positive)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    last_accessed = models.DateTimeField(null=True, blank=True)
    access_count = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=10, choices=MEMORY_STATUS, default='active')
    decay_rate = models.FloatField(
        default=0.01,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Daily memory decay rate (0.0-1.0)"
    )
    tags = models.CharField(
        max_length=500,
        blank=True,
        help_text="Comma-separated tags for categorization"
    )
    
    class Meta:
        app_label = 'dating_show_api'
        ordering = ['-created_at', '-importance_score']
        indexes = [
            models.Index(fields=['memory_type']),
            models.Index(fields=['importance_score']),
            models.Index(fields=['created_at']),
            models.Index(fields=['last_accessed']),
            models.Index(fields=['status']),
        ]
    
    def clean(self):
        """Custom validation for AgentMemorySnapshot model"""
        super().clean()
        
        # Validate importance score range
        if self.importance_score < 0.0 or self.importance_score > 10.0:
            raise ValidationError('Importance score must be between 0.0 and 10.0')
        
        # Validate emotional valence range
        if self.emotional_valence < -1.0 or self.emotional_valence > 1.0:
            raise ValidationError('Emotional valence must be between -1.0 and 1.0')
        
        # Validate content is a dictionary
        if self.content and not isinstance(self.content, dict):
            raise ValidationError('Content must be a valid JSON object')
    
    def save(self, *args, **kwargs):
        """Override save to add custom logic"""
        self.full_clean()
        super().save(*args, **kwargs)
    
    def access_memory(self):
        """Record memory access and update counters"""
        self.last_accessed = timezone.now()
        self.access_count += 1
        self.save(update_fields=['last_accessed', 'access_count'])
    
    def calculate_decay(self):
        """Calculate current memory decay amount"""
        days_since_creation = (timezone.now() - self.created_at).days
        if days_since_creation == 0:
            return 0.0
        
        # Factor in last access to reduce decay for frequently accessed memories
        days_since_access = (timezone.now() - (self.last_accessed or self.created_at)).days
        access_factor = max(0.1, 1.0 / (1.0 + self.access_count * 0.1))  # Reduce decay for frequently accessed
        
        decay_amount = self.importance_score * (1 - (1 - self.decay_rate * access_factor) ** days_since_creation)
        return min(decay_amount, self.importance_score)
    
    def apply_decay(self):
        """Apply natural memory decay"""
        decay = self.calculate_decay()
        if decay > 0:
            new_importance = max(0.0, self.importance_score - decay)
            self.importance_score = new_importance
            
            # Archive or delete very low importance memories
            if new_importance < 0.1:
                self.status = 'decayed'
            
            self.save(update_fields=['importance_score', 'status'])
            return decay
        return 0.0
    
    def get_tag_list(self):
        """Get tags as a list"""
        if not self.tags:
            return []
        return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
    
    def add_tag(self, tag):
        """Add a tag to the memory"""
        current_tags = self.get_tag_list()
        if tag not in current_tags:
            current_tags.append(tag)
            self.tags = ', '.join(current_tags)
            self.save(update_fields=['tags'])
    
    def days_since_creation(self):
        """Calculate days since memory was created"""
        delta = timezone.now() - self.created_at
        return delta.days
    
    def days_since_access(self):
        """Calculate days since last access"""
        if not self.last_accessed:
            return self.days_since_creation()
        delta = timezone.now() - self.last_accessed
        return delta.days
    
    def is_frequently_accessed(self, threshold=5):
        """Check if memory is frequently accessed"""
        return self.access_count >= threshold
    
    def is_emotionally_significant(self, threshold=0.5):
        """Check if memory has high emotional significance"""
        return abs(self.emotional_valence) >= threshold
    
    def __str__(self):
        return f"{self.agent.name} - {self.memory_type} memory (importance: {self.importance_score:.1f})"


class SimulationState(models.Model):
    """Enhanced model for tracking overall simulation state with validation and methods.
    
    Manages the overall simulation execution including step tracking,
    performance monitoring, and state transitions.
    
    Attributes:
        simulation_id (str): Unique identifier for the simulation
        status (str): Current simulation status
        current_step (int): Current simulation step number
        max_steps (int): Maximum number of simulation steps
        total_agents (int): Total number of agents in simulation
        active_agents (int): Number of currently active agents
        current_time (datetime): Current simulation time
        simulation_start_time (datetime): When simulation started
        simulation_end_time (datetime): When simulation ended
        performance_metrics (dict): Performance and statistical metrics
        configuration (dict): Simulation configuration parameters
        error_message (str): Error details if simulation failed
        step_duration_ms (float): Average step duration in milliseconds
        created_at (datetime): Model creation timestamp
        updated_at (datetime): Last update timestamp
    """
    SIMULATION_STATUS = [
        ('initializing', 'Initializing'),
        ('running', 'Running'),
        ('paused', 'Paused'),
        ('stopped', 'Stopped'),
        ('error', 'Error State'),
        ('completed', 'Completed'),
    ]
    
    simulation_id = models.CharField(
        max_length=100, 
        unique=True, 
        primary_key=True,
        validators=[RegexValidator(r'^[a-zA-Z0-9_-]+$', 'Simulation ID can only contain letters, numbers, underscores, and hyphens')]
    )
    status = models.CharField(max_length=15, choices=SIMULATION_STATUS, default='stopped')
    current_step = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Current simulation step number"
    )
    max_steps = models.IntegerField(
        default=1000,
        validators=[MinValueValidator(1)],
        help_text="Maximum number of simulation steps"
    )
    total_agents = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)]
    )
    active_agents = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)]
    )
    current_time = models.DateTimeField()
    simulation_start_time = models.DateTimeField(null=True, blank=True)
    simulation_end_time = models.DateTimeField(null=True, blank=True)
    performance_metrics = JSONField(
        default=dict,
        help_text="Performance and statistical metrics as JSON"
    )
    configuration = JSONField(
        default=dict,
        help_text="Simulation configuration parameters as JSON"
    )
    error_message = models.TextField(blank=True, help_text="Error details if simulation failed")
    step_duration_ms = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Average step duration in milliseconds"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        app_label = 'dating_show_api'
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['current_step']),
            models.Index(fields=['updated_at']),
        ]
    
    def clean(self):
        """Custom validation for SimulationState model"""
        super().clean()
        
        # Validate current_step doesn't exceed max_steps
        if self.current_step > self.max_steps:
            raise ValidationError('Current step cannot exceed maximum steps')
        
        # Validate active_agents doesn't exceed total_agents
        if self.active_agents > self.total_agents:
            raise ValidationError('Active agents cannot exceed total agents')
        
        # Validate configuration is a dictionary
        if self.configuration and not isinstance(self.configuration, dict):
            raise ValidationError('Configuration must be a valid JSON object')
        
        # Validate performance_metrics is a dictionary
        if self.performance_metrics and not isinstance(self.performance_metrics, dict):
            raise ValidationError('Performance metrics must be a valid JSON object')
    
    def save(self, *args, **kwargs):
        """Override save to add custom logic"""
        self.full_clean()
        
        # Set simulation_start_time when starting
        if self.status == 'running' and not self.simulation_start_time:
            self.simulation_start_time = timezone.now()
        
        # Set simulation_end_time when stopping or completing
        if self.status in ['stopped', 'completed', 'error'] and not self.simulation_end_time:
            self.simulation_end_time = timezone.now()
        
        super().save(*args, **kwargs)
    
    def start_simulation(self):
        """Start the simulation"""
        if self.status in ['stopped', 'paused']:
            self.status = 'running'
            if not self.simulation_start_time:
                self.simulation_start_time = timezone.now()
            self.save()
            return True
        return False
    
    def pause_simulation(self):
        """Pause the simulation"""
        if self.status == 'running':
            self.status = 'paused'
            self.save()
            return True
        return False
    
    def stop_simulation(self):
        """Stop the simulation"""
        if self.status in ['running', 'paused']:
            self.status = 'stopped'
            self.simulation_end_time = timezone.now()
            self.save()
            return True
        return False
    
    def set_error(self, error_message):
        """Set simulation to error state with message"""
        self.status = 'error'
        self.error_message = error_message
        self.simulation_end_time = timezone.now()
        self.save()
    
    def advance_step(self, step_duration_ms=None):
        """Advance simulation by one step"""
        if self.status == 'running':
            self.current_step += 1
            
            if step_duration_ms is not None:
                # Update average step duration with exponential moving average
                if self.step_duration_ms == 0.0:
                    self.step_duration_ms = step_duration_ms
                else:
                    self.step_duration_ms = 0.9 * self.step_duration_ms + 0.1 * step_duration_ms
            
            # Check if simulation should complete
            if self.current_step >= self.max_steps:
                self.status = 'completed'
                self.simulation_end_time = timezone.now()
            
            self.save()
            return True
        return False
    
    def get_progress_percentage(self):
        """Get simulation progress as percentage"""
        if self.max_steps == 0:
            return 0.0
        return min(100.0, (self.current_step / self.max_steps) * 100.0)
    
    def get_runtime_duration(self):
        """Get total runtime duration in seconds"""
        if not self.simulation_start_time:
            return 0.0
        
        end_time = self.simulation_end_time or timezone.now()
        duration = end_time - self.simulation_start_time
        return duration.total_seconds()
    
    def get_estimated_completion_time(self):
        """Estimate completion time based on current progress"""
        if self.status != 'running' or self.current_step == 0:
            return None
        
        progress = self.get_progress_percentage() / 100.0
        if progress == 0:
            return None
        
        runtime_seconds = self.get_runtime_duration()
        estimated_total_seconds = runtime_seconds / progress
        remaining_seconds = estimated_total_seconds - runtime_seconds
        
        return timezone.now() + timedelta(seconds=remaining_seconds)
    
    def get_performance_summary(self):
        """Get a summary of simulation performance"""
        return {
            'total_steps': self.current_step,
            'max_steps': self.max_steps,
            'progress_percentage': self.get_progress_percentage(),
            'runtime_seconds': self.get_runtime_duration(),
            'avg_step_duration_ms': self.step_duration_ms,
            'steps_per_second': (1000.0 / self.step_duration_ms) if self.step_duration_ms > 0 else 0.0,
            'estimated_completion': self.get_estimated_completion_time(),
            'total_agents': self.total_agents,
            'active_agents': self.active_agents,
            'agent_utilization': (self.active_agents / self.total_agents) if self.total_agents > 0 else 0.0
        }
    
    def is_running(self):
        """Check if simulation is currently running"""
        return self.status == 'running'
    
    def is_completed(self):
        """Check if simulation has completed"""
        return self.status == 'completed'
    
    def has_error(self):
        """Check if simulation is in error state"""
        return self.status == 'error'
    
    def __str__(self):
        return f"Simulation {self.simulation_id} - {self.status} (Step {self.current_step}/{self.max_steps})"