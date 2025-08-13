"""
Skill Development System for Enhanced PIANO Architecture
Task 3.2: Experience-based skill growth algorithms

This module implements a comprehensive skill development system featuring:
- Experience-based learning algorithms
- Skill specialization and mastery pathways
- Skill synergies and prerequisites
- Decay and maintenance mechanisms
- Teaching and knowledge transfer
- Performance calculation based on skills
- Integration with agent cognitive systems

Features:
- Dynamic skill progression based on practice and experience
- Realistic learning curves with diminishing returns
- Specialization paths that unlock at higher skill levels
- Skill interactions (synergies, prerequisites, competition)
- Time-based skill decay and maintenance requirements
- Agent-to-agent skill teaching and knowledge transfer
- Performance modifiers based on skill levels and context
- Integration with economic resource system
"""

import json
import math
import time
import random
import logging
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import threading
from statistics import mean, median, stdev

# Set up logging
logger = logging.getLogger(__name__)


class SkillType(Enum):
    """Types of skills agents can develop"""
    # Physical Skills
    COMBAT = "combat"
    ATHLETICS = "athletics" 
    CRAFTING = "crafting"
    STEALTH = "stealth"
    ACROBATICS = "acrobatics"
    
    # Mental Skills
    REASONING = "reasoning"
    MEMORY = "memory"
    ANALYSIS = "analysis"
    CREATIVITY = "creativity"
    FOCUS = "focus"
    LEARNING = "learning"
    
    # Social Skills
    PERSUASION = "persuasion"
    EMPATHY = "empathy"
    LEADERSHIP = "leadership"
    DECEPTION = "deception"
    NETWORKING = "networking"
    NEGOTIATION = "negotiation"
    
    # Survival Skills
    FORAGING = "foraging"
    HUNTING = "hunting"
    SHELTER_BUILDING = "shelter_building"
    NAVIGATION = "navigation"
    MEDICINE = "medicine"
    
    # Technical Skills
    ENGINEERING = "engineering"
    PROGRAMMING = "programming"
    RESEARCH = "research"
    PLANNING = "planning"


class SkillCategory(Enum):
    """Categories of skills for organization and synergy calculation"""
    PHYSICAL = "physical"
    MENTAL = "mental"
    SOCIAL = "social"
    SURVIVAL = "survival"
    CREATIVE = "creative"
    TECHNICAL = "technical"


class SkillLevel(Enum):
    """Skill progression levels with experience thresholds"""
    NOVICE = 0      # 0-100 XP
    BEGINNER = 1    # 100-300 XP
    COMPETENT = 2   # 300-600 XP
    PROFICIENT = 3  # 600-1000 XP
    EXPERT = 4      # 1000-1500 XP
    MASTER = 5      # 1500+ XP


class SkillInteractionType(Enum):
    """Types of interactions between skills"""
    SYNERGY = "synergy"        # Skills enhance each other
    PREREQUISITE = "prerequisite"  # One skill required for another
    COMPETITION = "competition"    # Skills compete for practice time
    SPECIALIZATION = "specialization"  # Exclusive specialization paths


class LearningSourceType(Enum):
    """Sources of experience and learning"""
    PRACTICE = "practice"      # Deliberate practice
    SUCCESS = "success"        # Successful task completion
    FAILURE = "failure"        # Learning from mistakes
    TEACHING = "teaching"      # Being taught by another agent
    OBSERVATION = "observation" # Learning by watching others
    RESEARCH = "research"      # Study and theoretical learning
    EXPERIMENTATION = "experimentation"  # Trial and error learning


@dataclass
class SkillInstance:
    """Represents an agent's instance of a specific skill"""
    skill_type: SkillType
    level: SkillLevel
    experience_points: float
    practice_hours: float
    success_rate: float  # Historical success rate (0.0-1.0)
    last_practiced: float  # Unix timestamp
    decay_rate: float  # Rate at which skill decays when unused
    mastery_bonus: float  # Bonus from specialization/mastery
    specialization_path: Optional[str]
    
    # Learning metrics
    total_attempts: int = 0
    successful_attempts: int = 0
    learning_velocity: float = 1.0  # How fast this agent learns this skill
    plateau_resistance: float = 0.5  # Resistance to learning plateaus
    
    # Context tracking
    preferred_contexts: Dict[str, float] = field(default_factory=dict)
    context_performance: Dict[str, float] = field(default_factory=dict)
    
    # Social learning
    teachers: Set[str] = field(default_factory=set)  # Agent IDs who taught this skill
    students: Set[str] = field(default_factory=set)  # Agent IDs taught this skill
    teaching_quality: float = 0.0  # Quality as a teacher for this skill
    
    def __post_init__(self):
        """Validate skill data"""
        self.experience_points = max(0.0, self.experience_points)
        self.practice_hours = max(0.0, self.practice_hours)
        self.success_rate = max(0.0, min(1.0, self.success_rate))
        self.mastery_bonus = max(0.0, min(1.0, self.mastery_bonus))
        self.learning_velocity = max(0.1, min(5.0, self.learning_velocity))


@dataclass
class ExperiencePoint:
    """Represents a single experience gain event"""
    skill_type: SkillType
    amount: float
    source: LearningSourceType
    multiplier: float  # Context and condition multipliers
    timestamp: float
    context: Dict[str, Any]
    
    # Performance metrics
    performance_score: float = 0.0  # How well the task was performed (0.0-1.0)
    difficulty_level: float = 0.5  # Task difficulty (0.0-1.0)
    effort_level: float = 0.5  # Effort put into the task (0.0-1.0)
    
    def get_effective_experience(self) -> float:
        """Calculate the effective experience considering all factors"""
        return self.amount * self.multiplier * (1.0 + self.performance_score)


@dataclass
class LearningEvent:
    """Represents a learning event that affects skill development"""
    agent_id: str
    skill_type: SkillType
    event_type: str  # Type of learning event
    experience_gained: float
    performance_score: float
    difficulty_level: float
    context: Dict[str, Any]
    timestamp: float
    
    # Event metadata
    duration: float = 0.0  # Duration in hours
    teacher_id: Optional[str] = None  # If learned from another agent
    tools_used: List[str] = field(default_factory=list)
    environmental_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class SkillSynergy:
    """Represents synergy between two skills"""
    primary_skill: SkillType
    secondary_skill: SkillType
    synergy_strength: float  # How much secondary skill boosts primary (0.0-1.0)
    synergy_type: str  # Type of synergy (e.g., "technique", "knowledge", "physical")
    minimum_level: SkillLevel  # Minimum level of secondary skill for synergy
    context_specific: bool = False  # Whether synergy only applies in specific contexts
    contexts: List[str] = field(default_factory=list)


@dataclass
class SkillPrerequisite:
    """Represents a prerequisite relationship between skills"""
    target_skill: SkillType
    prerequisite_skill: SkillType
    minimum_level: SkillLevel
    soft_requirement: bool = False  # Whether requirement can be bypassed with penalty


@dataclass
class SpecializationPath:
    """Represents a skill specialization path"""
    name: str
    primary_skill: SkillType
    description: str
    requirements: Dict[SkillType, SkillLevel]
    bonuses: Dict[str, float]  # Performance bonuses in specific areas
    penalties: Dict[str, float]  # Trade-offs or penalties
    exclusive_with: List[str] = field(default_factory=list)  # Mutually exclusive paths
    unlock_threshold: SkillLevel = SkillLevel.PROFICIENT


class SkillDevelopmentSystem:
    """
    Comprehensive skill development system for agents.
    
    Features:
    - Dynamic experience-based skill progression
    - Realistic learning curves with diminishing returns
    - Skill specialization and mastery pathways
    - Inter-skill synergies and prerequisites
    - Skill decay and maintenance mechanisms
    - Agent-to-agent teaching and knowledge transfer
    - Performance calculation with contextual modifiers
    """
    
    def __init__(self, max_agents: int = 1000):
        self.max_agents = max_agents
        
        # Core data structures
        self.agents: Set[str] = set()
        self.agent_skills: Dict[str, Dict[SkillType, SkillInstance]] = defaultdict(dict)
        self.learning_history: Dict[str, List[LearningEvent]] = defaultdict(list)
        self.experience_log: List[ExperiencePoint] = []
        
        # Skill system configuration
        self.skill_config = self._initialize_skill_config()
        self.specializations = self._initialize_specializations()
        self.synergies = self._initialize_synergies()
        self.prerequisites = self._initialize_prerequisites()
        
        # Learning parameters
        self.base_learning_rates = self._initialize_learning_rates()
        self.experience_thresholds = {
            SkillLevel.NOVICE: 0,
            SkillLevel.BEGINNER: 100,
            SkillLevel.COMPETENT: 300,
            SkillLevel.PROFICIENT: 600,
            SkillLevel.EXPERT: 1000,
            SkillLevel.MASTER: 1500
        }
        
        # Performance tracking
        self.performance_cache: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.teaching_records: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        
        # System parameters
        self.decay_base_rate = 0.01  # Base daily decay rate
        self.synergy_max_bonus = 0.5  # Maximum synergy bonus
        self.teaching_effectiveness = 0.3  # Base teaching effectiveness
        self.practice_efficiency_base = 1.0  # Base practice efficiency
        
        # Concurrency control
        self.lock = threading.RLock()
        
        # Caching
        self._cache_timestamp = 0.0
        self._cache_ttl = 300.0  # 5 minutes
        
        # Performance monitoring
        self.last_update_time = time.time()
        self.update_frequency = 3600.0  # Update skills every hour
        
        # Performance Optimization System - Task 3.2.6.1
        self.optimization_enabled = True
        self.batch_processing = True
        self.concurrent_processing = True
        self.calculation_cache = {}  # Cache for expensive calculations
        self.synergy_cache = {}  # Cache synergy calculations
        self.performance_metrics = {
            "calculation_time": deque(maxlen=100),
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_operations": 0,
            "concurrent_operations": 0
        }
        
        # Pre-computed lookup tables for faster calculations
        self._level_multipliers_cache = self._precompute_level_multipliers()
        self._synergy_lookup_cache = self._precompute_synergy_lookups()
        self._difficulty_cache = {}
        
        # Batch operation queues
        self._batch_queue = []
        self._batch_size = 50
        self._batch_timeout = 0.1  # Process batch after 100ms
    
    def add_agent(self, agent_id: str) -> bool:
        """Add an agent to the skill system"""
        with self.lock:
            if len(self.agents) >= self.max_agents or agent_id in self.agents:
                return False
            
            self.agents.add(agent_id)
            self._initialize_agent_skills(agent_id)
            return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the skill system"""
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            self.agents.remove(agent_id)
            self.agent_skills.pop(agent_id, None)
            self.learning_history.pop(agent_id, None)
            self.performance_cache.pop(agent_id, None)
            self.teaching_records.pop(agent_id, None)
            return True
    
    def get_skill_category(self, skill_type: SkillType) -> SkillCategory:
        """Get the category for a skill type"""
        return self.skill_config[skill_type]["category"]
    
    def calculate_skill_level(self, experience_points: float) -> SkillLevel:
        """Calculate skill level based on experience points"""
        for level in reversed(list(SkillLevel)):
            if experience_points >= self.experience_thresholds[level]:
                return level
        return SkillLevel.NOVICE
    
    def get_experience_for_level(self, level: SkillLevel) -> float:
        """Get minimum experience points required for a skill level"""
        return self.experience_thresholds[level]
    
    def get_experience_to_next_level(self, current_exp: float) -> Tuple[float, SkillLevel]:
        """Get experience needed to reach next level"""
        current_level = self.calculate_skill_level(current_exp)
        level_values = list(SkillLevel)
        
        if current_level == SkillLevel.MASTER:
            return 0.0, SkillLevel.MASTER
        
        current_index = level_values.index(current_level)
        next_level = level_values[current_index + 1]
        next_threshold = self.experience_thresholds[next_level]
        
        return next_threshold - current_exp, next_level
    
    def calculate_experience_gain(
        self,
        skill_type: SkillType,
        source: LearningSourceType,
        performance: float,
        difficulty: float,
        current_level: SkillLevel,
        duration: float = 1.0,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate experience gain from a learning event"""
        if context is None:
            context = {}
        
        # Base experience based on source type
        base_exp = {
            LearningSourceType.PRACTICE: 8.0,
            LearningSourceType.SUCCESS: 15.0,
            LearningSourceType.FAILURE: 5.0,
            LearningSourceType.TEACHING: 20.0,
            LearningSourceType.OBSERVATION: 3.0,
            LearningSourceType.RESEARCH: 6.0,
            LearningSourceType.EXPERIMENTATION: 10.0
        }[source]
        
        # Duration multiplier
        duration_mult = math.sqrt(duration)  # Diminishing returns on duration
        
        # Performance multiplier
        if source == LearningSourceType.SUCCESS:
            perf_mult = 0.5 + 1.5 * performance  # 0.5 to 2.0 multiplier
        elif source == LearningSourceType.FAILURE:
            perf_mult = 1.5 - 0.5 * performance  # Learn more from bigger failures
        else:
            perf_mult = 0.8 + 0.4 * performance  # 0.8 to 1.2 multiplier
        
        # Difficulty multiplier
        difficulty_mult = 0.7 + 0.6 * difficulty  # 0.7 to 1.3 multiplier
        
        # Level-based learning rate (diminishing returns at higher levels)
        level_mult = self.get_learning_rate(current_level, skill_type)
        
        # Calculate final experience
        experience = base_exp * duration_mult * perf_mult * difficulty_mult * level_mult
        
        # Apply contextual modifiers
        context_mult = self._calculate_context_multiplier(skill_type, context)
        experience *= context_mult
        
        return max(0.1, experience)  # Minimum experience gain
    
    def get_learning_rate(self, level: SkillLevel, skill_type: SkillType) -> float:
        """Get learning rate based on current skill level (diminishing returns)"""
        base_rate = self.base_learning_rates.get(skill_type, 1.0)
        
        # Learning rate decreases with level
        level_multipliers = {
            SkillLevel.NOVICE: 1.2,
            SkillLevel.BEGINNER: 1.0,
            SkillLevel.COMPETENT: 0.8,
            SkillLevel.PROFICIENT: 0.6,
            SkillLevel.EXPERT: 0.4,
            SkillLevel.MASTER: 0.2
        }
        
        return base_rate * level_multipliers[level]
    
    def calculate_practice_effectiveness(
        self,
        focus_level: float,
        fatigue_level: float,
        difficulty_level: float,
        tool_quality: float = 0.5,
        environmental_quality: float = 0.5
    ) -> float:
        """Calculate how effective a practice session will be"""
        # Focus is most important factor
        focus_contribution = focus_level * 0.4
        
        # Fatigue reduces effectiveness
        fatigue_penalty = fatigue_level * 0.3
        
        # Optimal difficulty is around 0.6-0.8
        optimal_difficulty = 0.7
        difficulty_effectiveness = 1.0 - abs(difficulty_level - optimal_difficulty)
        difficulty_contribution = difficulty_effectiveness * 0.2
        
        # Tool and environmental quality
        tool_contribution = tool_quality * 0.05
        env_contribution = environmental_quality * 0.05
        
        effectiveness = (focus_contribution + difficulty_contribution + 
                        tool_contribution + env_contribution - fatigue_penalty)
        
        return max(0.1, min(2.0, effectiveness))  # Clamp between 0.1 and 2.0
    
    def process_learning_event(self, event: LearningEvent) -> bool:
        """Process a learning event and update agent skills"""
        with self.lock:
            if event.agent_id not in self.agents:
                return False
            
            skill_type = event.skill_type
            
            # Get or create skill instance
            if skill_type not in self.agent_skills[event.agent_id]:
                self._initialize_skill(event.agent_id, skill_type)
            
            skill = self.agent_skills[event.agent_id][skill_type]
            
            # Update skill metrics
            skill.total_attempts += 1
            if event.performance_score > 0.5:  # Consider >0.5 performance a success
                skill.successful_attempts += 1
            
            # Update success rate (moving average)
            current_success_rate = skill.successful_attempts / skill.total_attempts
            skill.success_rate = 0.9 * skill.success_rate + 0.1 * current_success_rate
            
            # Add experience points
            old_level = skill.level
            skill.experience_points += event.experience_gained
            skill.practice_hours += event.duration
            skill.last_practiced = event.timestamp
            
            # Update level if threshold crossed
            new_level = self.calculate_skill_level(skill.experience_points)
            if new_level != old_level:
                skill.level = new_level
                logger.info(f"Agent {event.agent_id} advanced {skill_type.value} to {new_level.name}")
            
            # Update context performance tracking
            for context_key, context_value in event.context.items():
                if isinstance(context_value, (int, float)):
                    if context_key not in skill.context_performance:
                        skill.context_performance[context_key] = event.performance_score
                    else:
                        # Moving average
                        skill.context_performance[context_key] = (
                            0.8 * skill.context_performance[context_key] + 
                            0.2 * event.performance_score
                        )
            
            # Track learning history
            self.learning_history[event.agent_id].append(event)
            if len(self.learning_history[event.agent_id]) > 1000:
                self.learning_history[event.agent_id] = self.learning_history[event.agent_id][-500:]
            
            # Invalidate caches
            self._invalidate_cache()
            
            return True
    
    def get_specialization_paths(self, skill_type: SkillType) -> List[str]:
        """Get available specialization paths for a skill"""
        return [spec.name for spec in self.specializations 
                if spec.primary_skill == skill_type]
    
    def can_specialize(
        self, 
        skill_type: SkillType, 
        current_level: SkillLevel, 
        specialization: str
    ) -> bool:
        """Check if an agent can choose a specialization"""
        # Find specialization
        spec = next((s for s in self.specializations 
                    if s.name == specialization and s.primary_skill == skill_type), None)
        
        if not spec:
            return False
        
        # Check level requirement
        if current_level.value < spec.unlock_threshold.value:
            return False
        
        return True
    
    def apply_specialization(
        self, 
        agent_id: str, 
        skill_type: SkillType, 
        specialization_name: str
    ) -> bool:
        """Apply a specialization to an agent's skill"""
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            if skill_type not in self.agent_skills[agent_id]:
                return False
            
            skill = self.agent_skills[agent_id][skill_type]
            
            # Find specialization
            spec = next((s for s in self.specializations 
                        if s.name == specialization_name and s.primary_skill == skill_type), None)
            
            if not spec:
                return False
            
            # Check if can specialize
            if not self.can_specialize(skill_type, skill.level, specialization_name):
                return False
            
            # Check requirements
            for req_skill, req_level in spec.requirements.items():
                if (req_skill not in self.agent_skills[agent_id] or 
                    self.agent_skills[agent_id][req_skill].level.value < req_level.value):
                    return False
            
            # Apply specialization
            skill.specialization_path = specialization_name
            skill.mastery_bonus = spec.bonuses.get("mastery", 0.0)
            skill.decay_rate *= spec.bonuses.get("decay_resistance", 1.0)
            
            logger.info(f"Agent {agent_id} specialized in {specialization_name} for {skill_type.value}")
            
            return True
    
    def calculate_synergy_bonus(
        self, 
        primary_skill: SkillType, 
        secondary_skills: Dict[SkillType, SkillLevel]
    ) -> float:
        """Calculate synergy bonus from secondary skills"""
        total_bonus = 0.0
        
        for synergy in self.synergies:
            if synergy.primary_skill != primary_skill:
                continue
            
            if synergy.secondary_skill not in secondary_skills:
                continue
            
            secondary_level = secondary_skills[synergy.secondary_skill]
            
            if secondary_level.value >= synergy.minimum_level.value:
                # Calculate bonus based on secondary skill level
                level_factor = (secondary_level.value + 1) / (synergy.minimum_level.value + 1)
                bonus = synergy.synergy_strength * level_factor
                total_bonus += min(bonus, self.synergy_max_bonus)
        
        return min(total_bonus, self.synergy_max_bonus)
    
    def get_prerequisites(self, skill_type: SkillType) -> List[SkillType]:
        """Get prerequisite skills for a given skill"""
        return [prereq.prerequisite_skill for prereq in self.prerequisites 
                if prereq.target_skill == skill_type]
    
    def can_develop_skill(
        self, 
        target_skill: SkillType, 
        current_skills: Dict[SkillType, SkillLevel]
    ) -> bool:
        """Check if an agent can develop a skill based on prerequisites"""
        for prereq in self.prerequisites:
            if prereq.target_skill != target_skill:
                continue
            
            if prereq.soft_requirement:
                continue  # Soft requirements don't block development
            
            if (prereq.prerequisite_skill not in current_skills or
                current_skills[prereq.prerequisite_skill].value < prereq.minimum_level.value):
                return False
        
        return True
    
    def apply_skill_decay(self, skill: SkillInstance) -> SkillInstance:
        """Apply time-based skill decay"""
        time_since_practice = time.time() - skill.last_practiced
        days_since_practice = time_since_practice / 86400.0  # Convert to days
        
        if days_since_practice <= 1.0:  # No decay for first day
            return skill
        
        # Calculate decay amount
        decay_amount = self.calculate_decay_amount(
            skill.level,
            days_since_practice,
            skill.decay_rate,
            skill.specialization_path
        )
        
        # Apply decay to experience points and success rate
        skill.experience_points = max(0.0, skill.experience_points - decay_amount)
        skill.success_rate *= (1.0 - decay_amount / 1000.0)  # Gradual success rate decay
        skill.success_rate = max(0.1, skill.success_rate)  # Minimum success rate
        
        # Update skill level if necessary
        new_level = self.calculate_skill_level(skill.experience_points)
        if new_level != skill.level:
            skill.level = new_level
        
        return skill
    
    def calculate_decay_amount(
        self,
        skill_level: SkillLevel,
        days_unused: float,
        base_decay_rate: float,
        specialization_path: Optional[str]
    ) -> float:
        """Calculate how much experience decays over time"""
        # Base decay increases with time (logarithmic)
        time_factor = math.log(1 + days_unused)
        
        # Higher level skills resist decay better
        level_resistance = {
            SkillLevel.NOVICE: 0.8,
            SkillLevel.BEGINNER: 0.9,
            SkillLevel.COMPETENT: 1.0,
            SkillLevel.PROFICIENT: 1.1,
            SkillLevel.EXPERT: 1.2,
            SkillLevel.MASTER: 1.5
        }[skill_level]
        
        # Specialization provides decay resistance
        specialization_resistance = 1.5 if specialization_path else 1.0
        
        decay = base_decay_rate * time_factor / (level_resistance * specialization_resistance)
        
        return max(0.0, decay)
    
    def calculate_maintenance_experience(
        self,
        skill_type: SkillType,
        current_level: SkillLevel,
        practice_intensity: float,
        practice_duration: float
    ) -> float:
        """Calculate experience gained from light maintenance practice"""
        base_maintenance = self.base_learning_rates.get(skill_type, 1.0)
        intensity_factor = practice_intensity
        duration_factor = math.sqrt(practice_duration)  # Diminishing returns
        level_factor = 0.5  # Maintenance is less effective than focused practice
        
        return base_maintenance * intensity_factor * duration_factor * level_factor
    
    def calculate_teaching_bonus(
        self,
        teacher_skill: SkillInstance,
        student_skill: SkillInstance,
        teaching_duration: float
    ) -> float:
        """Calculate teaching effectiveness bonus"""
        # Teacher skill level affects teaching quality
        teacher_effectiveness = (teacher_skill.level.value + 1) * 0.15
        
        # Skill gap affects teaching efficiency
        skill_gap = teacher_skill.level.value - student_skill.level.value
        gap_effectiveness = min(1.0, max(0.2, skill_gap * 0.2))
        
        # Teacher's own teaching skill
        teaching_skill_bonus = teacher_skill.teaching_quality
        
        # Duration factor (diminishing returns)
        duration_factor = math.sqrt(teaching_duration)
        
        bonus = (teacher_effectiveness + gap_effectiveness + teaching_skill_bonus) * duration_factor
        
        return min(2.0, bonus)  # Cap at 2.0x multiplier
    
    def calculate_taught_experience(self, base_experience: float, teaching_bonus: float) -> float:
        """Calculate experience gained when being taught"""
        return base_experience * (1.0 + teaching_bonus)
    
    def calculate_teaching_efficiency(
        self, 
        teacher_level: SkillLevel, 
        student_level: SkillLevel
    ) -> float:
        """Calculate how efficient teaching is based on skill levels"""
        level_difference = teacher_level.value - student_level.value
        
        # Optimal teaching gap is 2-3 levels
        optimal_gap = 2.5
        efficiency = 1.0 - abs(level_difference - optimal_gap) * 0.2
        
        return max(0.1, min(1.0, efficiency))
    
    def calculate_teacher_experience(
        self,
        skill_being_taught: SkillType,
        teacher_level: SkillLevel,
        teaching_duration: float,
        student_improvement: float
    ) -> float:
        """Calculate experience teacher gains from teaching"""
        base_exp = 2.0  # Base teaching experience
        duration_factor = math.sqrt(teaching_duration)
        improvement_bonus = student_improvement * 2.0
        
        # Teaching high-level skills gives more experience
        level_bonus = teacher_level.value * 0.5
        
        return base_exp * duration_factor * (1.0 + improvement_bonus + level_bonus)
    
    def calculate_secondary_teaching_experience(
        self,
        secondary_skill: SkillType,
        teaching_session_quality: float
    ) -> float:
        """Calculate experience in secondary skills from teaching"""
        if secondary_skill in [SkillType.LEADERSHIP, SkillType.EMPATHY, SkillType.PERSUASION]:
            return 1.0 + teaching_session_quality
        return 0.0
    
    def calculate_task_performance(
        self, 
        skill: SkillInstance, 
        task_difficulty: float,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate performance on a task based on skill level"""
        if context is None:
            context = {}
        
        # Base performance from skill level
        base_performance = self.calculate_base_performance(skill.level)
        
        # Experience bonus within level
        level_exp_range = self._get_level_experience_range(skill.level)
        if level_exp_range > 0:
            exp_in_level = skill.experience_points - self.get_experience_for_level(skill.level)
            exp_bonus = min(0.2, (exp_in_level / level_exp_range) * 0.2)
            base_performance += exp_bonus
        
        # Success rate modifier
        base_performance *= (0.8 + 0.2 * skill.success_rate)
        
        # Mastery bonus
        base_performance *= (1.0 + skill.mastery_bonus)
        
        # Task difficulty modifier
        difficulty_penalty = max(0.0, (task_difficulty - 0.5) * 0.5)
        base_performance *= (1.0 - difficulty_penalty)
        
        # Apply contextual modifiers
        contextual_performance = self.apply_contextual_modifiers(base_performance, context)
        
        return max(0.0, min(1.0, contextual_performance))
    
    def calculate_base_performance(self, level: SkillLevel) -> float:
        """Calculate base performance for a skill level"""
        performance_levels = {
            SkillLevel.NOVICE: 0.3,
            SkillLevel.BEGINNER: 0.5,
            SkillLevel.COMPETENT: 0.65,
            SkillLevel.PROFICIENT: 0.75,
            SkillLevel.EXPERT: 0.85,
            SkillLevel.MASTER: 0.95
        }
        return performance_levels[level]
    
    def calculate_success_probability(
        self, 
        skill: SkillInstance, 
        difficulty: float,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate probability of success for a task"""
        performance = self.calculate_task_performance(skill, difficulty, context)
        
        # Convert performance to probability using sigmoid-like function
        # Higher performance increases probability, but never guarantees success
        probability = 1.0 / (1.0 + math.exp(-10 * (performance - 0.5)))
        
        # Difficulty affects probability
        difficulty_modifier = 1.0 - (difficulty - 0.5) * 0.3
        probability *= max(0.1, difficulty_modifier)
        
        return max(0.05, min(0.95, probability))  # Clamp between 5% and 95%
    
    def apply_contextual_modifiers(
        self, 
        base_performance: float, 
        context: Dict[str, Any]
    ) -> float:
        """Apply contextual modifiers to performance"""
        modified_performance = base_performance
        
        # Equipment quality
        equipment_quality = context.get("equipment_quality", 0.5)
        equipment_bonus = (equipment_quality - 0.5) * 0.2
        modified_performance += equipment_bonus
        
        # Environmental conditions
        env_conditions = context.get("environmental_conditions", 0.5)
        env_bonus = (env_conditions - 0.5) * 0.15
        modified_performance += env_bonus
        
        # Fatigue penalty
        fatigue_level = context.get("fatigue_level", 0.0)
        fatigue_penalty = fatigue_level * 0.25
        modified_performance -= fatigue_penalty
        
        # Stress penalty
        stress_level = context.get("stress_level", 0.0)
        stress_penalty = stress_level * 0.2
        modified_performance -= stress_penalty
        
        # Time pressure penalty
        time_pressure = context.get("time_pressure", 0.0)
        time_penalty = time_pressure * 0.15
        modified_performance -= time_penalty
        
        return modified_performance
    
    def initialize_agent_skills(
        self,
        agent_id: str,
        personality_traits: Optional[Dict[str, float]] = None,
        starting_background: Optional[str] = None
    ) -> List[SkillInstance]:
        """Initialize skills for a new agent"""
        if personality_traits is None:
            personality_traits = {}
        
        skills = []
        
        # Give all agents basic skills
        basic_skills = [
            SkillType.REASONING,
            SkillType.MEMORY,
            SkillType.EMPATHY,
            SkillType.ATHLETICS
        ]
        
        for skill_type in basic_skills:
            initial_exp = random.uniform(10, 50)  # Random starting experience
            skills.append(self._create_initial_skill(skill_type, initial_exp))
        
        # Add background-specific skills
        if starting_background:
            background_skills = self._get_background_skills(starting_background)
            for skill_type, exp_bonus in background_skills.items():
                initial_exp = 50 + exp_bonus
                skills.append(self._create_initial_skill(skill_type, initial_exp))
        
        # Personality-based skill affinities
        if personality_traits:
            personality_skills = self._get_personality_skills(personality_traits)
            for skill_type, exp_bonus in personality_skills.items():
                if skill_type not in [s.skill_type for s in skills]:
                    initial_exp = 25 + exp_bonus
                    skills.append(self._create_initial_skill(skill_type, initial_exp))
        
        # Store skills
        with self.lock:
            for skill in skills:
                self.agent_skills[agent_id][skill.skill_type] = skill
        
        return skills
    
    def execute_skill_action(
        self,
        agent_id: str,
        skill_type: SkillType,
        action: str,
        target: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a skill-based action"""
        if context is None:
            context = {}
        
        if agent_id not in self.agents:
            return {"success": False, "error": "Agent not found"}
        
        if skill_type not in self.agent_skills[agent_id]:
            self._initialize_skill(agent_id, skill_type)
        
        skill = self.agent_skills[agent_id][skill_type]
        
        # Determine task difficulty based on action and target
        task_difficulty = self._determine_task_difficulty(action, target, context)
        
        # Calculate success probability
        success_prob = self.calculate_success_probability(skill, task_difficulty, context)
        
        # Determine if action succeeds
        success = random.random() < success_prob
        
        # Calculate performance score
        base_performance = self.calculate_task_performance(skill, task_difficulty, context)
        performance_variance = 0.2  # ±20% variance
        performance_score = base_performance + random.uniform(-performance_variance, performance_variance)
        performance_score = max(0.0, min(1.0, performance_score))
        
        # Create learning event
        source = LearningSourceType.SUCCESS if success else LearningSourceType.FAILURE
        experience_gained = self.calculate_experience_gain(
            skill_type=skill_type,
            source=source,
            performance=performance_score,
            difficulty=task_difficulty,
            current_level=skill.level,
            duration=0.5,  # Assume 30 minutes for most actions
            context=context
        )
        
        learning_event = LearningEvent(
            agent_id=agent_id,
            skill_type=skill_type,
            event_type="action_execution",
            experience_gained=experience_gained,
            performance_score=performance_score,
            difficulty_level=task_difficulty,
            context=context,
            timestamp=time.time(),
            duration=0.5
        )
        
        # Process the learning event
        self.process_learning_event(learning_event)
        
        return {
            "success": success,
            "performance_score": performance_score,
            "experience_gained": experience_gained,
            "skill_level": skill.level,
            "task_difficulty": task_difficulty
        }
    
    def calculate_production_bonus(
        self,
        skill_type: SkillType,
        skill_level: SkillLevel,
        resource_type: str
    ) -> float:
        """Calculate production bonus from skills for economic system integration"""
        # Map skills to resources
        skill_resource_mapping = {
            SkillType.CRAFTING: ["tools", "fabric", "shelter"],
            SkillType.FORAGING: ["food", "medicine"],
            SkillType.HUNTING: ["food", "fabric"],
            SkillType.ENGINEERING: ["technology", "tools"],
            SkillType.RESEARCH: ["knowledge", "information"]
        }
        
        if skill_type not in skill_resource_mapping:
            return 0.0
        
        if resource_type not in skill_resource_mapping[skill_type]:
            return 0.0
        
        # Calculate bonus based on skill level
        level_bonuses = {
            SkillLevel.NOVICE: 0.0,
            SkillLevel.BEGINNER: 0.05,
            SkillLevel.COMPETENT: 0.15,
            SkillLevel.PROFICIENT: 0.25,
            SkillLevel.EXPERT: 0.35,
            SkillLevel.MASTER: 0.5
        }
        
        return level_bonuses[skill_level]
    
    def calculate_efficiency_bonus(
        self,
        skill_type: SkillType,
        skill_level: SkillLevel,
        resource_type: str
    ) -> float:
        """Calculate efficiency bonus for resource consumption"""
        # Similar to production bonus but for consumption efficiency
        efficiency_skills = {
            SkillType.FORAGING: ["food"],
            SkillType.MEDICINE: ["medicine"],
            SkillType.ENGINEERING: ["energy", "tools"],
            SkillType.ATHLETICS: ["energy"]  # Better fitness = less energy consumption
        }
        
        if skill_type not in efficiency_skills:
            return 0.0
        
        if resource_type not in efficiency_skills[skill_type]:
            return 0.0
        
        # Efficiency bonuses (reduce consumption)
        level_bonuses = {
            SkillLevel.NOVICE: 0.0,
            SkillLevel.BEGINNER: 0.02,
            SkillLevel.COMPETENT: 0.08,
            SkillLevel.PROFICIENT: 0.15,
            SkillLevel.EXPERT: 0.22,
            SkillLevel.MASTER: 0.3
        }
        
        return level_bonuses[skill_level]
    
    # Agent management methods
    def get_agent_skills(self, agent_id: str) -> Dict[SkillType, SkillInstance]:
        """Get all skills for an agent"""
        return self.agent_skills.get(agent_id, {})
    
    def get_skill(self, agent_id: str, skill_type: SkillType) -> Optional[SkillInstance]:
        """Get a specific skill for an agent"""
        return self.agent_skills.get(agent_id, {}).get(skill_type)
    
    def add_experience(self, agent_id: str, skill_type: SkillType, experience_points: float) -> Dict[str, Any]:
        """Add experience points to a specific skill for an agent"""
        # Input validation
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("Agent ID must be a non-empty string")
        
        if not isinstance(skill_type, SkillType):
            raise TypeError("skill_type must be a SkillType enum")
        
        if not isinstance(experience_points, (int, float)) or experience_points < 0:
            raise ValueError("Experience points must be a non-negative number")
        
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found in skill system")
        
        with self.lock:  # Thread safety
            try:
                # Get or create the skill
                if skill_type not in self.agent_skills[agent_id]:
                    self._initialize_skill(agent_id, skill_type)
                
                skill = self.agent_skills[agent_id][skill_type]
                old_level = skill.level
                old_experience = skill.experience_points
                
                # Add the experience
                skill.experience_points += experience_points
                skill.last_practiced = time.time()
                
                # Update level
                new_level = self.calculate_skill_level(skill.experience_points)
                skill.level = new_level
                
                # Log the learning event
                learning_event = LearningEvent(
                    agent_id=agent_id,
                    skill_type=skill_type,
                    experience_gained=experience_points,
                    timestamp=time.time(),
                    source=LearningSourceType.PRACTICE,
                    context="manual_experience_addition"
                )
                self.learning_history[agent_id].append(learning_event)
                
                # Update performance metrics
                if 'experience_additions' not in self.performance_metrics:
                    self.performance_metrics['experience_additions'] = 0
                self.performance_metrics['experience_additions'] += 1
                
                logger.info(f"Added {experience_points} experience to {skill_type.value} for agent {agent_id}. Level: {old_level} -> {new_level}")
                
                return {
                    "success": True,
                    "old_level": old_level,
                    "new_level": new_level,
                    "old_experience": old_experience,
                    "new_experience": skill.experience_points,
                    "experience_added": experience_points,
                    "level_up": new_level.value > old_level.value
                }
                
            except Exception as e:
                logger.error(f"Error adding experience to {skill_type.value} for agent {agent_id}: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "agent_id": agent_id,
                    "skill_type": skill_type.value
                }
    
    def add_skill(self, agent_id: str, skill_type: SkillType, level: SkillLevel) -> bool:
        """Add a skill to an agent at a specific level"""
        if agent_id not in self.agents:
            return False
        
        exp_for_level = self.get_experience_for_level(level)
        skill = self._create_initial_skill(skill_type, exp_for_level)
        skill.level = level
        
        self.agent_skills[agent_id][skill_type] = skill
        return True
    
    def get_all_agents(self) -> List[str]:
        """Get list of all agent IDs"""
        return list(self.agents)
    
    def practice_skill(
        self,
        agent_id: str,
        skill_type: SkillType,
        hours: float,
        focus_level: float = 0.7,
        difficulty: float = 0.6
    ) -> bool:
        """Have an agent practice a skill"""
        if agent_id not in self.agents:
            return False
        
        if skill_type not in self.agent_skills[agent_id]:
            self._initialize_skill(agent_id, skill_type)
        
        skill = self.agent_skills[agent_id][skill_type]
        
        # Calculate practice effectiveness
        effectiveness = self.calculate_practice_effectiveness(
            focus_level=focus_level,
            fatigue_level=0.2,  # Assume moderate fatigue
            difficulty_level=difficulty,
            tool_quality=0.6,   # Assume decent tools
            environmental_quality=0.7  # Assume good environment
        )
        
        # Calculate experience gained
        experience_gained = self.calculate_experience_gain(
            skill_type=skill_type,
            source=LearningSourceType.PRACTICE,
            performance=focus_level * effectiveness,
            difficulty=difficulty,
            current_level=skill.level,
            duration=hours
        )
        
        experience_gained *= effectiveness
        
        # Create learning event
        learning_event = LearningEvent(
            agent_id=agent_id,
            skill_type=skill_type,
            event_type="practice",
            experience_gained=experience_gained,
            performance_score=focus_level * effectiveness,
            difficulty_level=difficulty,
            context={"practice_hours": hours, "focus": focus_level},
            timestamp=time.time(),
            duration=hours
        )
        
        return self.process_learning_event(learning_event)
    
    def update_all_skills(self) -> None:
        """Update all agent skills (apply decay, etc.)"""
        current_time = time.time()
        
        if current_time - self.last_update_time < self.update_frequency:
            return  # Too soon for update
        
        with self.lock:
            for agent_id in self.agents:
                for skill_type, skill in self.agent_skills[agent_id].items():
                    # Apply skill decay
                    self.agent_skills[agent_id][skill_type] = self.apply_skill_decay(skill)
        
        self.last_update_time = current_time
        self._invalidate_cache()
    
    def get_skill_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get a summary of an agent's skills"""
        if agent_id not in self.agents:
            return {}
        
        skills = self.agent_skills[agent_id]
        
        # Calculate total skill points
        total_exp = sum(skill.experience_points for skill in skills.values())
        total_hours = sum(skill.practice_hours for skill in skills.values())
        
        # Find highest skills
        highest_skills = sorted(skills.items(), 
                               key=lambda x: x[1].experience_points, reverse=True)[:5]
        
        # Count skills by level
        level_counts = defaultdict(int)
        for skill in skills.values():
            level_counts[skill.level] += 1
        
        # Calculate specializations
        specializations = [skill.specialization_path for skill in skills.values() 
                          if skill.specialization_path]
        
        return {
            "total_skills": len(skills),
            "total_experience": total_exp,
            "total_practice_hours": total_hours,
            "highest_skills": [(skill_type.value, skill.level.name, skill.experience_points) 
                              for skill_type, skill in highest_skills],
            "level_distribution": dict(level_counts),
            "specializations": specializations,
            "average_success_rate": mean([skill.success_rate for skill in skills.values()]) 
                                   if skills else 0.0
        }
    
    # Private helper methods
    
    def _initialize_skill_config(self) -> Dict[SkillType, Dict[str, Any]]:
        """Initialize skill type configurations with comprehensive parameters"""
        return {
            # Physical Skills
            SkillType.COMBAT: {
                "category": SkillCategory.PHYSICAL, 
                "base_decay": 0.015,
                "learning_rate": 1.0,
                "max_level": 100.0,
                "difficulty": 0.8
            },
            SkillType.ATHLETICS: {
                "category": SkillCategory.PHYSICAL, 
                "base_decay": 0.02,
                "learning_rate": 1.2,
                "max_level": 100.0,
                "difficulty": 0.6
            },
            SkillType.CRAFTING: {
                "category": SkillCategory.PHYSICAL, 
                "base_decay": 0.008,
                "learning_rate": 0.8,
                "max_level": 100.0,
                "difficulty": 0.9
            },
            SkillType.STEALTH: {
                "category": SkillCategory.PHYSICAL, 
                "base_decay": 0.012,
                "learning_rate": 0.9,
                "max_level": 100.0,
                "difficulty": 0.7
            },
            SkillType.ACROBATICS: {
                "category": SkillCategory.PHYSICAL, 
                "base_decay": 0.018,
                "learning_rate": 1.1,
                "max_level": 100.0,
                "difficulty": 0.8
            },
            
            # Mental Skills  
            SkillType.REASONING: {
                "category": SkillCategory.MENTAL, 
                "base_decay": 0.005,
                "learning_rate": 0.7,
                "max_level": 100.0,
                "difficulty": 1.0
            },
            SkillType.MEMORY: {
                "category": SkillCategory.MENTAL, 
                "base_decay": 0.003,
                "learning_rate": 0.8,
                "max_level": 100.0,
                "difficulty": 0.6
            },
            SkillType.ANALYSIS: {
                "category": SkillCategory.MENTAL, 
                "base_decay": 0.004,
                "learning_rate": 0.9,
                "max_level": 100.0,
                "difficulty": 0.9
            },
            SkillType.CREATIVITY: {
                "category": SkillCategory.CREATIVE, 
                "base_decay": 0.006,
                "learning_rate": 0.6,
                "max_level": 100.0,
                "difficulty": 1.2
            },
            SkillType.FOCUS: {
                "category": SkillCategory.MENTAL, 
                "base_decay": 0.01,
                "learning_rate": 1.0,
                "max_level": 100.0,
                "difficulty": 0.7
            },
            SkillType.LEARNING: {
                "category": SkillCategory.MENTAL, 
                "base_decay": 0.002,
                "learning_rate": 1.5,
                "max_level": 100.0,
                "difficulty": 0.8
            },
            
            # Social Skills
            SkillType.PERSUASION: {
                "category": SkillCategory.SOCIAL, 
                "base_decay": 0.007,
                "learning_rate": 1.0,
                "max_level": 100.0,
                "difficulty": 0.8
            },
            SkillType.EMPATHY: {
                "category": SkillCategory.SOCIAL, 
                "base_decay": 0.005,
                "learning_rate": 0.9,
                "max_level": 100.0,
                "difficulty": 0.7
            },
            SkillType.LEADERSHIP: {
                "category": SkillCategory.SOCIAL, 
                "base_decay": 0.006,
                "learning_rate": 0.7,
                "max_level": 100.0,
                "difficulty": 1.1
            },
            SkillType.DECEPTION: {
                "category": SkillCategory.SOCIAL, 
                "base_decay": 0.01,
                "learning_rate": 0.8,
                "max_level": 100.0,
                "difficulty": 0.9
            },
            SkillType.NETWORKING: {
                "category": SkillCategory.SOCIAL, 
                "base_decay": 0.008,
                "learning_rate": 1.1,
                "max_level": 100.0,
                "difficulty": 0.6
            },
            SkillType.NEGOTIATION: {
                "category": SkillCategory.SOCIAL, 
                "base_decay": 0.007,
                "learning_rate": 0.9,
                "max_level": 100.0,
                "difficulty": 0.8
            },
            
            # Survival Skills
            SkillType.FORAGING: {
                "category": SkillCategory.SURVIVAL, 
                "base_decay": 0.01,
                "learning_rate": 1.0,
                "max_level": 100.0,
                "difficulty": 0.7
            },
            SkillType.HUNTING: {
                "category": SkillCategory.SURVIVAL, 
                "base_decay": 0.012,
                "learning_rate": 0.9,
                "max_level": 100.0,
                "difficulty": 0.9
            },
            SkillType.SHELTER_BUILDING: {
                "category": SkillCategory.SURVIVAL, 
                "base_decay": 0.009,
                "learning_rate": 0.8,
                "max_level": 100.0,
                "difficulty": 0.8
            },
            SkillType.NAVIGATION: {
                "category": SkillCategory.SURVIVAL, 
                "base_decay": 0.008,
                "learning_rate": 1.1,
                "max_level": 100.0,
                "difficulty": 0.6
            },
            SkillType.MEDICINE: {
                "category": SkillCategory.SURVIVAL, 
                "base_decay": 0.006,
                "learning_rate": 0.7,
                "max_level": 100.0,
                "difficulty": 1.0
            },
            
            # Technical Skills
            SkillType.ENGINEERING: {
                "category": SkillCategory.TECHNICAL, 
                "base_decay": 0.005,
                "learning_rate": 0.6,
                "max_level": 100.0,
                "difficulty": 1.2
            },
            SkillType.PROGRAMMING: {
                "category": SkillCategory.TECHNICAL, 
                "base_decay": 0.008,
                "learning_rate": 0.8,
                "max_level": 100.0,
                "difficulty": 1.0
            },
            SkillType.RESEARCH: {
                "category": SkillCategory.TECHNICAL, 
                "base_decay": 0.004,
                "learning_rate": 0.9,
                "max_level": 100.0,
                "difficulty": 0.8
            },
            SkillType.PLANNING: {
                "category": SkillCategory.TECHNICAL, 
                "base_decay": 0.006,
                "learning_rate": 1.0,
                "max_level": 100.0,
                "difficulty": 0.7
            }
        }
    
    def _initialize_learning_rates(self) -> Dict[SkillType, float]:
        """Initialize base learning rates for each skill type"""
        return {
            # Physical skills - moderate learning rates
            SkillType.COMBAT: 1.0,
            SkillType.ATHLETICS: 1.2,
            SkillType.CRAFTING: 0.8,
            SkillType.STEALTH: 0.9,
            SkillType.ACROBATICS: 1.1,
            
            # Mental skills - varied learning rates
            SkillType.REASONING: 0.7,  # Slower to develop
            SkillType.MEMORY: 1.0,
            SkillType.ANALYSIS: 0.8,
            SkillType.CREATIVITY: 0.6,  # Very slow to develop
            SkillType.FOCUS: 1.3,  # Can improve quickly with practice
            SkillType.LEARNING: 0.5,  # Meta-skill, very slow
            
            # Social skills - moderate learning rates
            SkillType.PERSUASION: 0.9,
            SkillType.EMPATHY: 0.7,
            SkillType.LEADERSHIP: 0.6,
            SkillType.DECEPTION: 1.1,
            SkillType.NETWORKING: 1.0,
            SkillType.NEGOTIATION: 0.8,
            
            # Survival skills - quick to pick up basics
            SkillType.FORAGING: 1.3,
            SkillType.HUNTING: 1.0,
            SkillType.SHELTER_BUILDING: 1.1,
            SkillType.NAVIGATION: 1.2,
            SkillType.MEDICINE: 0.7,
            
            # Technical skills - slower learning
            SkillType.ENGINEERING: 0.6,
            SkillType.PROGRAMMING: 0.8,
            SkillType.RESEARCH: 0.9,
            SkillType.PLANNING: 1.0
        }
    
    def _initialize_specializations(self) -> List[SpecializationPath]:
        """Initialize skill specialization paths"""
        specializations = []
        
        # Combat specializations
        specializations.extend([
            SpecializationPath(
                name="weapon_master",
                primary_skill=SkillType.COMBAT,
                description="Master of weapons and martial techniques",
                requirements={SkillType.COMBAT: SkillLevel.PROFICIENT, SkillType.FOCUS: SkillLevel.COMPETENT},
                bonuses={"mastery": 0.2, "decay_resistance": 0.7, "critical_chance": 0.15},
                penalties={"social_interaction": -0.1},
                unlock_threshold=SkillLevel.PROFICIENT
            ),
            SpecializationPath(
                name="tactical_fighter",
                primary_skill=SkillType.COMBAT,
                description="Strategic combat specialist",
                requirements={SkillType.COMBAT: SkillLevel.PROFICIENT, SkillType.REASONING: SkillLevel.PROFICIENT},
                bonuses={"mastery": 0.15, "group_combat": 0.25, "leadership_bonus": 0.1},
                penalties={},
                unlock_threshold=SkillLevel.PROFICIENT
            ),
            SpecializationPath(
                name="berserker",
                primary_skill=SkillType.COMBAT,
                description="Fierce warrior who fights with primal fury",
                requirements={SkillType.COMBAT: SkillLevel.PROFICIENT, SkillType.ATHLETICS: SkillLevel.COMPETENT},
                bonuses={"mastery": 0.25, "damage_bonus": 0.3, "fear_resistance": 0.4},
                penalties={"precision": -0.15, "social_interaction": -0.2},
                unlock_threshold=SkillLevel.PROFICIENT
            )
        ])
        
        # Social specializations
        specializations.extend([
            SpecializationPath(
                name="diplomat",
                primary_skill=SkillType.PERSUASION,
                description="Master of negotiation and peaceful resolution",
                requirements={SkillType.PERSUASION: SkillLevel.PROFICIENT, SkillType.EMPATHY: SkillLevel.COMPETENT},
                bonuses={"mastery": 0.2, "negotiation_bonus": 0.3, "reputation_gain": 0.15},
                penalties={},
                unlock_threshold=SkillLevel.PROFICIENT
            ),
            SpecializationPath(
                name="manipulator",
                primary_skill=SkillType.PERSUASION,
                description="Expert at influencing others through cunning",
                requirements={SkillType.PERSUASION: SkillLevel.PROFICIENT, SkillType.DECEPTION: SkillLevel.COMPETENT},
                bonuses={"mastery": 0.18, "deception_synergy": 0.25, "information_gathering": 0.2},
                penalties={"reputation_risk": 0.3, "empathy_penalty": -0.1},
                unlock_threshold=SkillLevel.PROFICIENT
            ),
            SpecializationPath(
                name="inspirational_leader",
                primary_skill=SkillType.LEADERSHIP,
                description="Leader who motivates others through inspiration",
                requirements={SkillType.LEADERSHIP: SkillLevel.PROFICIENT, SkillType.EMPATHY: SkillLevel.PROFICIENT},
                bonuses={"mastery": 0.2, "group_bonus": 0.25, "morale_boost": 0.3},
                penalties={},
                unlock_threshold=SkillLevel.PROFICIENT
            )
        ])
        
        # Crafting specializations
        specializations.extend([
            SpecializationPath(
                name="master_craftsman",
                primary_skill=SkillType.CRAFTING,
                description="Artisan capable of creating masterwork items",
                requirements={SkillType.CRAFTING: SkillLevel.PROFICIENT, SkillType.FOCUS: SkillLevel.COMPETENT},
                bonuses={"mastery": 0.25, "quality_bonus": 0.4, "efficiency": 0.2},
                penalties={"mass_production": -0.2},
                unlock_threshold=SkillLevel.PROFICIENT
            ),
            SpecializationPath(
                name="innovative_engineer",
                primary_skill=SkillType.ENGINEERING,
                description="Engineer focused on creating new technologies",
                requirements={SkillType.ENGINEERING: SkillLevel.PROFICIENT, SkillType.CREATIVITY: SkillLevel.COMPETENT},
                bonuses={"mastery": 0.2, "innovation_bonus": 0.35, "research_synergy": 0.15},
                penalties={"traditional_methods": -0.15},
                unlock_threshold=SkillLevel.PROFICIENT
            )
        ])
        
        return specializations
    
    def _initialize_synergies(self) -> List[SkillSynergy]:
        """Initialize skill synergy relationships"""
        synergies = []
        
        # Physical synergies
        synergies.extend([
            SkillSynergy(SkillType.COMBAT, SkillType.ATHLETICS, 0.25, "physical_conditioning", SkillLevel.BEGINNER),
            SkillSynergy(SkillType.COMBAT, SkillType.ACROBATICS, 0.2, "mobility", SkillLevel.COMPETENT),
            SkillSynergy(SkillType.ATHLETICS, SkillType.ACROBATICS, 0.3, "body_control", SkillLevel.BEGINNER),
            SkillSynergy(SkillType.STEALTH, SkillType.ACROBATICS, 0.2, "movement", SkillLevel.BEGINNER)
        ])
        
        # Mental synergies  
        synergies.extend([
            SkillSynergy(SkillType.REASONING, SkillType.ANALYSIS, 0.3, "logical_thinking", SkillLevel.BEGINNER),
            SkillSynergy(SkillType.ANALYSIS, SkillType.RESEARCH, 0.25, "investigation", SkillLevel.COMPETENT),
            SkillSynergy(SkillType.MEMORY, SkillType.LEARNING, 0.4, "knowledge_retention", SkillLevel.BEGINNER),
            SkillSynergy(SkillType.FOCUS, SkillType.LEARNING, 0.35, "concentration", SkillLevel.BEGINNER),
            SkillSynergy(SkillType.CREATIVITY, SkillType.REASONING, 0.2, "innovative_thinking", SkillLevel.PROFICIENT)
        ])
        
        # Social synergies
        synergies.extend([
            SkillSynergy(SkillType.PERSUASION, SkillType.EMPATHY, 0.3, "emotional_intelligence", SkillLevel.BEGINNER),
            SkillSynergy(SkillType.LEADERSHIP, SkillType.PERSUASION, 0.25, "influence", SkillLevel.COMPETENT),
            SkillSynergy(SkillType.LEADERSHIP, SkillType.EMPATHY, 0.35, "understanding", SkillLevel.COMPETENT),
            SkillSynergy(SkillType.NEGOTIATION, SkillType.ANALYSIS, 0.2, "strategic_thinking", SkillLevel.COMPETENT),
            SkillSynergy(SkillType.NETWORKING, SkillType.EMPATHY, 0.25, "social_connection", SkillLevel.BEGINNER)
        ])
        
        # Cross-category synergies
        synergies.extend([
            SkillSynergy(SkillType.COMBAT, SkillType.REASONING, 0.2, "tactical_thinking", SkillLevel.COMPETENT),
            SkillSynergy(SkillType.CRAFTING, SkillType.CREATIVITY, 0.3, "artistic_expression", SkillLevel.COMPETENT),
            SkillSynergy(SkillType.ENGINEERING, SkillType.REASONING, 0.25, "technical_analysis", SkillLevel.BEGINNER),
            SkillSynergy(SkillType.MEDICINE, SkillType.EMPATHY, 0.2, "bedside_manner", SkillLevel.COMPETENT),
            SkillSynergy(SkillType.HUNTING, SkillType.STEALTH, 0.3, "stalking", SkillLevel.BEGINNER)
        ])
        
        return synergies
    
    def _initialize_prerequisites(self) -> List[SkillPrerequisite]:
        """Initialize skill prerequisite relationships"""
        prerequisites = []
        
        # Leadership requires social skills
        prerequisites.extend([
            SkillPrerequisite(SkillType.LEADERSHIP, SkillType.PERSUASION, SkillLevel.COMPETENT),
            SkillPrerequisite(SkillType.LEADERSHIP, SkillType.EMPATHY, SkillLevel.BEGINNER)
        ])
        
        # Advanced combat requires athletics
        prerequisites.append(
            SkillPrerequisite(SkillType.COMBAT, SkillType.ATHLETICS, SkillLevel.BEGINNER, soft_requirement=True)
        )
        
        # Medicine requires some analytical thinking
        prerequisites.append(
            SkillPrerequisite(SkillType.MEDICINE, SkillType.ANALYSIS, SkillLevel.BEGINNER)
        )
        
        # Engineering requires reasoning and creativity
        prerequisites.extend([
            SkillPrerequisite(SkillType.ENGINEERING, SkillType.REASONING, SkillLevel.COMPETENT),
            SkillPrerequisite(SkillType.ENGINEERING, SkillType.CREATIVITY, SkillLevel.BEGINNER, soft_requirement=True)
        ])
        
        # Research requires analysis and memory
        prerequisites.extend([
            SkillPrerequisite(SkillType.RESEARCH, SkillType.ANALYSIS, SkillLevel.COMPETENT),
            SkillPrerequisite(SkillType.RESEARCH, SkillType.MEMORY, SkillLevel.BEGINNER)
        ])
        
        return prerequisites
    
    def _initialize_agent_skills(self, agent_id: str):
        """Initialize basic skills for a new agent"""
        # Everyone starts with basic reasoning and empathy
        basic_skills = [
            (SkillType.REASONING, random.uniform(20, 40)),
            (SkillType.EMPATHY, random.uniform(15, 35)),
            (SkillType.MEMORY, random.uniform(25, 45))
        ]
        
        for skill_type, initial_exp in basic_skills:
            skill = self._create_initial_skill(skill_type, initial_exp)
            self.agent_skills[agent_id][skill_type] = skill
    
    def _initialize_skill(self, agent_id: str, skill_type: SkillType):
        """Initialize a single skill for an agent"""
        initial_exp = random.uniform(1, 10)  # Very basic starting experience
        skill = self._create_initial_skill(skill_type, initial_exp)
        self.agent_skills[agent_id][skill_type] = skill
    
    def _create_initial_skill(self, skill_type: SkillType, experience_points: float) -> SkillInstance:
        """Create a new skill instance with initial values"""
        level = self.calculate_skill_level(experience_points)
        config = self.skill_config[skill_type]
        
        return SkillInstance(
            skill_type=skill_type,
            level=level,
            experience_points=experience_points,
            practice_hours=experience_points / 10.0,  # Rough conversion
            success_rate=0.3 + (level.value * 0.1),  # Base success rate by level
            last_practiced=time.time(),
            decay_rate=config["base_decay"],
            mastery_bonus=0.0,
            specialization_path=None,
            total_attempts=int(experience_points / 2),  # Rough estimate
            successful_attempts=int(experience_points / 3),  # Rough estimate
            learning_velocity=random.uniform(0.8, 1.2),
            plateau_resistance=random.uniform(0.3, 0.7),
            preferred_contexts={},
            context_performance={},
            teachers=set(),
            students=set(),
            teaching_quality=0.0
        )
    
    def _get_background_skills(self, background: str) -> Dict[SkillType, float]:
        """Get skill bonuses based on character background"""
        background_skills = {
            "craftsman": {
                SkillType.CRAFTING: 80,
                SkillType.FOCUS: 40,
                SkillType.ENGINEERING: 30
            },
            "scholar": {
                SkillType.RESEARCH: 80,
                SkillType.ANALYSIS: 60,
                SkillType.MEMORY: 50,
                SkillType.LEARNING: 40
            },
            "warrior": {
                SkillType.COMBAT: 90,
                SkillType.ATHLETICS: 60,
                SkillType.LEADERSHIP: 30
            },
            "diplomat": {
                SkillType.PERSUASION: 80,
                SkillType.NEGOTIATION: 70,
                SkillType.EMPATHY: 50,
                SkillType.NETWORKING: 40
            },
            "survivalist": {
                SkillType.FORAGING: 70,
                SkillType.HUNTING: 60,
                SkillType.SHELTER_BUILDING: 50,
                SkillType.NAVIGATION: 40
            }
        }
        
        return background_skills.get(background, {})
    
    def _get_personality_skills(self, personality_traits: Dict[str, float]) -> Dict[SkillType, float]:
        """Get skill affinities based on personality traits"""
        skill_bonuses = {}
        
        # Openness to experience
        openness = personality_traits.get("openness", 0.5)
        if openness > 0.6:
            skill_bonuses[SkillType.CREATIVITY] = (openness - 0.5) * 100
            skill_bonuses[SkillType.LEARNING] = (openness - 0.5) * 80
        
        # Conscientiousness
        conscientiousness = personality_traits.get("conscientiousness", 0.5)
        if conscientiousness > 0.6:
            skill_bonuses[SkillType.FOCUS] = (conscientiousness - 0.5) * 120
            skill_bonuses[SkillType.PLANNING] = (conscientiousness - 0.5) * 100
        
        # Extraversion
        extraversion = personality_traits.get("extraversion", 0.5)
        if extraversion > 0.6:
            skill_bonuses[SkillType.PERSUASION] = (extraversion - 0.5) * 100
            skill_bonuses[SkillType.NETWORKING] = (extraversion - 0.5) * 120
            skill_bonuses[SkillType.LEADERSHIP] = (extraversion - 0.5) * 80
        
        # Agreeableness
        agreeableness = personality_traits.get("agreeableness", 0.5)
        if agreeableness > 0.6:
            skill_bonuses[SkillType.EMPATHY] = (agreeableness - 0.5) * 120
            skill_bonuses[SkillType.MEDICINE] = (agreeableness - 0.5) * 60
        
        return skill_bonuses
    
    def _calculate_context_multiplier(self, skill_type: SkillType, context: Dict[str, Any]) -> float:
        """Calculate contextual multiplier for experience gain"""
        multiplier = 1.0
        
        # Tool quality affects experience gain
        tool_quality = context.get("tool_quality", 0.5)
        multiplier *= (0.8 + 0.4 * tool_quality)
        
        # Environmental factors
        environment = context.get("environment", "neutral")
        env_multipliers = {
            "optimal": 1.3,
            "good": 1.1,
            "neutral": 1.0,
            "poor": 0.8,
            "hostile": 0.6
        }
        multiplier *= env_multipliers.get(environment, 1.0)
        
        # Social context (for social skills)
        if self.get_skill_category(skill_type) == SkillCategory.SOCIAL:
            social_pressure = context.get("social_pressure", 0.5)
            multiplier *= (0.9 + 0.2 * social_pressure)
        
        # Risk level (affects learning for some skills)
        risk_level = context.get("risk_level", 0.0)
        if skill_type in [SkillType.COMBAT, SkillType.STEALTH, SkillType.HUNTING]:
            multiplier *= (1.0 + 0.3 * risk_level)  # Risk increases learning for these skills
        else:
            multiplier *= (1.0 - 0.1 * risk_level)  # Risk decreases learning for others
        
        return max(0.3, min(2.0, multiplier))  # Clamp between 0.3 and 2.0
    
    def _get_level_experience_range(self, level: SkillLevel) -> float:
        """Get the experience range for a skill level"""
        if level == SkillLevel.MASTER:
            return 500.0  # Arbitrary large range for master level
        
        level_values = list(SkillLevel)
        current_index = level_values.index(level)
        
        if current_index == len(level_values) - 1:
            return 500.0
        
        current_threshold = self.experience_thresholds[level]
        next_threshold = self.experience_thresholds[level_values[current_index + 1]]
        
        return next_threshold - current_threshold
    
    def _determine_task_difficulty(
        self, 
        action: str, 
        target: Optional[str], 
        context: Dict[str, Any]
    ) -> float:
        """Determine task difficulty based on action parameters"""
        # Base difficulty by action type
        action_difficulties = {
            "basic": 0.3,
            "simple": 0.4,
            "moderate": 0.5,
            "complex": 0.6,
            "difficult": 0.7,
            "expert": 0.8,
            "master": 0.9
        }
        
        base_difficulty = action_difficulties.get(action, 0.5)
        
        # Modify based on context
        complexity = context.get("complexity", 0.5)
        time_pressure = context.get("time_pressure", 0.0)
        resources_available = context.get("resources", 0.5)
        
        # Convert string complexity to float if needed
        if isinstance(complexity, str):
            complexity_map = {
                "trivial": 0.1, "easy": 0.2, "simple": 0.3, "medium": 0.5,
                "moderate": 0.5, "hard": 0.7, "difficult": 0.8, "extreme": 0.9
            }
            complexity = complexity_map.get(complexity.lower(), 0.5)
        
        # Calculate final difficulty
        difficulty = base_difficulty
        difficulty += complexity * 0.2
        difficulty += time_pressure * 0.15
        difficulty -= (resources_available - 0.5) * 0.1
        
        return max(0.1, min(0.95, difficulty))
    
    def _invalidate_cache(self):
        """Invalidate cached values"""
        self.performance_cache.clear()
        self._cache_timestamp = 0.0
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        return (time.time() - self._cache_timestamp) < self._cache_ttl
    
    # Dynamic Skill Discovery System - Task 3.2.1.1
    def discover_skills_from_actions(
        self,
        agent_id: str,
        action: str,
        performance: float,
        context: Dict[str, Any]
    ) -> List[SkillType]:
        """
        Discover new skills based on agent actions and performance.
        Returns list of newly discovered skills.
        """
        if agent_id not in self.agents:
            return []
        
        discovered_skills = []
        
        # Analyze action patterns to identify potential skills
        potential_skills = self._analyze_action_for_skills(action, context)
        
        for skill_type in potential_skills:
            # Check if agent already has this skill
            if skill_type in self.agent_skills.get(agent_id, {}):
                continue
            
            # Calculate discovery probability based on performance and context
            discovery_prob = self._calculate_discovery_probability(
                skill_type, performance, context, agent_id
            )
            
            if random.random() < discovery_prob:
                # Initialize skill at novice level with minimal experience
                initial_exp = random.uniform(1.0, 10.0)
                self._initialize_discovered_skill(agent_id, skill_type, initial_exp)
                discovered_skills.append(skill_type)
                
                logger.info(f"Agent {agent_id} discovered new skill: {skill_type.value}")
        
        return discovered_skills
    
    def _analyze_action_for_skills(self, action: str, context: Dict[str, Any]) -> List[SkillType]:
        """Analyze action and context to identify potential discoverable skills"""
        potential_skills = []
        action_lower = action.lower()
        
        # Action-to-skill mapping with discovery patterns
        skill_patterns = {
            SkillType.COMBAT: ["fight", "attack", "defend", "battle", "combat", "weapon"],
            SkillType.ATHLETICS: ["run", "jump", "climb", "swim", "exercise", "physical"],
            SkillType.CRAFTING: ["craft", "build", "create", "make", "construct", "forge"],
            SkillType.STEALTH: ["sneak", "hide", "stealth", "quiet", "unnoticed"],
            SkillType.ACROBATICS: ["acrobatic", "flip", "dodge", "agile", "balance"],
            SkillType.REASONING: ["think", "analyze", "logic", "reason", "deduce", "solve"],
            SkillType.MEMORY: ["remember", "recall", "memorize", "learn", "study"],
            SkillType.ANALYSIS: ["examine", "investigate", "analyze", "research", "inspect"],
            SkillType.CREATIVITY: ["create", "invent", "design", "artistic", "creative"],
            SkillType.FOCUS: ["concentrate", "focus", "meditate", "attention"],
            SkillType.LEARNING: ["learn", "study", "practice", "train", "education"],
            SkillType.PERSUASION: ["convince", "persuade", "argue", "influence"],
            SkillType.EMPATHY: ["understand", "empathize", "comfort", "relate"],
            SkillType.LEADERSHIP: ["lead", "command", "guide", "manage", "direct"],
            SkillType.DECEPTION: ["lie", "deceive", "trick", "mislead", "bluff"],
            SkillType.NETWORKING: ["socialize", "network", "connect", "relationship"],
            SkillType.NEGOTIATION: ["negotiate", "bargain", "deal", "compromise"],
            SkillType.FORAGING: ["forage", "gather", "collect", "find food"],
            SkillType.HUNTING: ["hunt", "track", "prey", "catch", "trap"],
            SkillType.SHELTER_BUILDING: ["shelter", "build", "construct", "home"],
            SkillType.NAVIGATION: ["navigate", "direction", "map", "find way"],
            SkillType.MEDICINE: ["heal", "treat", "medicine", "doctor", "cure"],
            SkillType.ENGINEERING: ["engineer", "design", "technical", "machine"],
            SkillType.PROGRAMMING: ["program", "code", "software", "computer"],
            SkillType.RESEARCH: ["research", "study", "investigate", "academic"],
            SkillType.PLANNING: ["plan", "strategy", "organize", "schedule"]
        }
        
        # Check for direct pattern matches
        for skill_type, patterns in skill_patterns.items():
            if any(pattern in action_lower for pattern in patterns):
                potential_skills.append(skill_type)
        
        # Context-based skill discovery
        context_skills = self._analyze_context_for_skills(context)
        potential_skills.extend(context_skills)
        
        # Environmental/situational skill discovery
        environment = context.get("environment", "")
        if environment:
            env_skills = self._analyze_environment_for_skills(environment)
            potential_skills.extend(env_skills)
        
        return list(set(potential_skills))  # Remove duplicates
    
    def _analyze_context_for_skills(self, context: Dict[str, Any]) -> List[SkillType]:
        """Analyze context for skill discovery opportunities"""
        skills = []
        
        # Social context
        if context.get("social_interaction"):
            skills.extend([SkillType.PERSUASION, SkillType.EMPATHY, SkillType.NETWORKING])
        
        # Dangerous context
        if context.get("danger_level", 0) > 0.5:
            skills.extend([SkillType.COMBAT, SkillType.STEALTH, SkillType.ATHLETICS])
        
        # Creative context
        if context.get("creativity_required", False):
            skills.extend([SkillType.CREATIVITY, SkillType.CRAFTING])
        
        # Learning context
        if context.get("learning_opportunity", False):
            skills.extend([SkillType.LEARNING, SkillType.MEMORY, SkillType.FOCUS])
        
        # Technical context
        if context.get("technical_challenge", False):
            skills.extend([SkillType.ENGINEERING, SkillType.PROGRAMMING, SkillType.RESEARCH])
        
        return skills
    
    def _analyze_environment_for_skills(self, environment: str) -> List[SkillType]:
        """Analyze environment for skill discovery opportunities"""
        skills = []
        env_lower = environment.lower()
        
        environment_mappings = {
            "wilderness": [SkillType.FORAGING, SkillType.HUNTING, SkillType.NAVIGATION, SkillType.SHELTER_BUILDING],
            "forest": [SkillType.FORAGING, SkillType.STEALTH, SkillType.NAVIGATION, SkillType.HUNTING],
            "urban": [SkillType.NETWORKING, SkillType.NEGOTIATION, SkillType.STEALTH],
            "laboratory": [SkillType.RESEARCH, SkillType.ANALYSIS, SkillType.FOCUS],
            "workshop": [SkillType.CRAFTING, SkillType.ENGINEERING, SkillType.CREATIVITY],
            "hospital": [SkillType.MEDICINE, SkillType.EMPATHY, SkillType.FOCUS],
            "battlefield": [SkillType.COMBAT, SkillType.LEADERSHIP, SkillType.ATHLETICS],
            "library": [SkillType.RESEARCH, SkillType.MEMORY, SkillType.LEARNING]
        }
        
        for env_type, env_skills in environment_mappings.items():
            if env_type in env_lower:
                skills.extend(env_skills)
        
        return skills
    
    def _calculate_discovery_probability(
        self,
        skill_type: SkillType,
        performance: float,
        context: Dict[str, Any],
        agent_id: str
    ) -> float:
        """Calculate probability of discovering a skill"""
        base_probability = 0.05  # 5% base chance
        
        # Performance modifier (better performance = higher discovery chance)
        perf_modifier = performance * 0.15  # Up to 15% bonus for perfect performance
        
        # Context modifiers
        difficulty = context.get("complexity", 0.5)
        difficulty_modifier = difficulty * 0.1  # Harder tasks = higher discovery chance
        
        # Agent trait modifiers (from context)
        curiosity = context.get("agent_curiosity", 0.5)
        curiosity_modifier = curiosity * 0.1
        
        # Learning trait bonus
        learning_bonus = 0.0
        if agent_id in self.agent_skills:
            learning_skill = self.agent_skills[agent_id].get(SkillType.LEARNING)
            if learning_skill:
                learning_bonus = float(learning_skill.level.value) * 0.02  # Up to 10% for master learning
        
        # Skill category experience bonus
        related_skills = self._get_related_skills(skill_type)
        category_bonus = 0.0
        if agent_id in self.agent_skills:
            for related_skill in related_skills:
                if related_skill in self.agent_skills[agent_id]:
                    skill_level = self.agent_skills[agent_id][related_skill].level
                    category_bonus += float(skill_level.value) * 0.01  # Small bonus per related skill
        
        # Environmental discovery bonus
        env_bonus = 0.0
        if context.get("environment"):
            # Some environments are more conducive to skill discovery
            discovery_environments = ["laboratory", "workshop", "training", "academy", "wilderness"]
            if any(env in context.get("environment", "").lower() for env in discovery_environments):
                env_bonus = 0.05
        
        # Calculate final probability
        probability = (base_probability + perf_modifier + difficulty_modifier + 
                      curiosity_modifier + learning_bonus + category_bonus + env_bonus)
        
        # Cap at reasonable maximum
        return min(0.3, max(0.01, probability))
    
    def _get_related_skills(self, skill_type: SkillType) -> List[SkillType]:
        """Get skills related to the given skill type"""
        skill_relationships = {
            SkillType.COMBAT: [SkillType.ATHLETICS, SkillType.STEALTH, SkillType.ACROBATICS],
            SkillType.ATHLETICS: [SkillType.COMBAT, SkillType.ACROBATICS],
            SkillType.CRAFTING: [SkillType.CREATIVITY, SkillType.ENGINEERING, SkillType.FOCUS],
            SkillType.STEALTH: [SkillType.ACROBATICS, SkillType.COMBAT],
            SkillType.REASONING: [SkillType.ANALYSIS, SkillType.MEMORY, SkillType.PLANNING],
            SkillType.MEMORY: [SkillType.LEARNING, SkillType.REASONING],
            SkillType.ANALYSIS: [SkillType.REASONING, SkillType.RESEARCH],
            SkillType.CREATIVITY: [SkillType.CRAFTING, SkillType.ENGINEERING],
            SkillType.PERSUASION: [SkillType.EMPATHY, SkillType.NETWORKING, SkillType.NEGOTIATION],
            SkillType.EMPATHY: [SkillType.PERSUASION, SkillType.LEADERSHIP],
            SkillType.LEADERSHIP: [SkillType.PERSUASION, SkillType.EMPATHY, SkillType.PLANNING],
            SkillType.DECEPTION: [SkillType.PERSUASION, SkillType.STEALTH],
            SkillType.NETWORKING: [SkillType.PERSUASION, SkillType.EMPATHY],
            SkillType.NEGOTIATION: [SkillType.PERSUASION, SkillType.REASONING],
            SkillType.FORAGING: [SkillType.NAVIGATION, SkillType.MEDICINE],
            SkillType.HUNTING: [SkillType.STEALTH, SkillType.NAVIGATION, SkillType.COMBAT],
            SkillType.NAVIGATION: [SkillType.FORAGING, SkillType.HUNTING],
            SkillType.MEDICINE: [SkillType.EMPATHY, SkillType.ANALYSIS, SkillType.MEMORY],
            SkillType.ENGINEERING: [SkillType.CRAFTING, SkillType.CREATIVITY, SkillType.REASONING],
            SkillType.PROGRAMMING: [SkillType.REASONING, SkillType.ANALYSIS, SkillType.CREATIVITY],
            SkillType.RESEARCH: [SkillType.ANALYSIS, SkillType.REASONING, SkillType.MEMORY],
            SkillType.PLANNING: [SkillType.REASONING, SkillType.LEADERSHIP, SkillType.ANALYSIS]
        }
        
        return skill_relationships.get(skill_type, [])
    
    def _initialize_discovered_skill(
        self,
        agent_id: str,
        skill_type: SkillType,
        initial_experience: float
    ):
        """Initialize a newly discovered skill for an agent"""
        # Create skill instance
        skill = SkillInstance(
            skill_type=skill_type,
            experience_points=initial_experience,
            level=self.calculate_skill_level(initial_experience),
            last_used=time.time(),
            specializations=[],
            decay_rate=self.skill_config[skill_type]["base_decay"]
        )
        
        # Add to agent's skills
        if agent_id not in self.agent_skills:
            self.agent_skills[agent_id] = {}
        
        self.agent_skills[agent_id][skill_type] = skill
        
        # Record discovery event
        if agent_id not in self.learning_history:
            self.learning_history[agent_id] = deque(maxlen=self.max_history_length)
        
        self.learning_history[agent_id].append({
            "timestamp": time.time(),
            "event_type": "skill_discovery",
            "skill": skill_type.value,
            "initial_experience": initial_experience,
            "discovery_method": "action_based"
        })
        
        # Invalidate cache
        self._invalidate_cache()
    
    def get_discoverable_skills(
        self,
        agent_id: str,
        action: str,
        context: Dict[str, Any]
    ) -> Dict[SkillType, float]:
        """
        Get skills that could potentially be discovered from an action,
        with their discovery probabilities.
        """
        if agent_id not in self.agents:
            return {}
        
        potential_skills = self._analyze_action_for_skills(action, context)
        probabilities = {}
        
        for skill_type in potential_skills:
            # Skip if agent already has this skill
            if skill_type in self.agent_skills.get(agent_id, {}):
                continue
            
            prob = self._calculate_discovery_probability(
                skill_type, 0.5, context, agent_id  # Use average performance for estimation
            )
            probabilities[skill_type] = prob
        
        return probabilities
    
    # Performance Optimization Methods - Task 3.2.6.1
    def _precompute_level_multipliers(self) -> Dict[SkillLevel, float]:
        """Pre-compute level-based multipliers for faster lookups"""
        return {
            SkillLevel.NOVICE: 1.2,
            SkillLevel.BEGINNER: 1.1,
            SkillLevel.COMPETENT: 1.0,
            SkillLevel.PROFICIENT: 0.9,
            SkillLevel.EXPERT: 0.8,
            SkillLevel.MASTER: 0.7
        }
    
    def _precompute_synergy_lookups(self) -> Dict[Tuple[SkillType, SkillType], float]:
        """Pre-compute synergy relationships for faster lookups"""
        synergy_lookup = {}
        for synergy in self.synergies:
            key = (synergy.primary_skill, synergy.secondary_skill)
            synergy_lookup[key] = synergy.synergy_strength
        return synergy_lookup
    
    def get_learning_rate_optimized(self, level: SkillLevel, skill_type: SkillType) -> float:
        """Optimized version of get_learning_rate with caching"""
        cache_key = f"{level.value}_{skill_type.value}"
        
        if cache_key in self.calculation_cache:
            self.performance_metrics["cache_hits"] += 1
            return self.calculation_cache[cache_key]
        
        self.performance_metrics["cache_misses"] += 1
        base_rate = self.base_learning_rates.get(skill_type, 1.0)
        level_mult = self._level_multipliers_cache[level]
        result = base_rate * level_mult
        
        # Cache the result
        self.calculation_cache[cache_key] = result
        return result
    
    def calculate_experience_gain_optimized(
        self,
        skill_type: SkillType,
        source: LearningSourceType,
        performance: float,
        difficulty: float,
        current_level: SkillLevel,
        duration: float = 1.0,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Optimized experience gain calculation with caching and batch processing"""
        start_time = time.time()
        
        # Create cache key for this calculation
        cache_key = f"exp_{skill_type.value}_{source.value}_{performance:.2f}_{difficulty:.2f}_{current_level.value}_{duration:.1f}"
        
        if cache_key in self.calculation_cache:
            self.performance_metrics["cache_hits"] += 1
            return self.calculation_cache[cache_key]
        
        self.performance_metrics["cache_misses"] += 1
        
        if context is None:
            context = {}
        
        # Use pre-computed base experience values
        base_exp_map = {
            LearningSourceType.PRACTICE: 8.0,
            LearningSourceType.SUCCESS: 15.0,
            LearningSourceType.FAILURE: 5.0,
            LearningSourceType.TEACHING: 20.0,
            LearningSourceType.OBSERVATION: 3.0,
            LearningSourceType.RESEARCH: 6.0,
            LearningSourceType.EXPERIMENTATION: 10.0
        }
        
        base_exp = base_exp_map[source]
        
        # Optimized multiplier calculations
        duration_mult = duration ** 0.5  # Faster than math.sqrt
        
        # Simplified performance multiplier calculation
        if source == LearningSourceType.SUCCESS:
            perf_mult = 0.5 + 1.5 * performance
        elif source == LearningSourceType.FAILURE:
            perf_mult = 1.5 - 0.5 * performance
        else:
            perf_mult = 0.8 + 0.4 * performance
        
        # Use pre-cached difficulty and level multipliers
        difficulty_mult = 0.7 + 0.6 * difficulty
        level_mult = self._level_multipliers_cache[current_level]
        
        # Calculate final experience
        experience = base_exp * duration_mult * perf_mult * difficulty_mult * level_mult
        
        # Apply contextual modifiers (simplified version)
        context_mult = self._calculate_context_multiplier_optimized(skill_type, context)
        experience *= context_mult
        
        result = max(0.1, experience)
        
        # Cache the result
        self.calculation_cache[cache_key] = result
        
        # Record performance metrics
        calc_time = time.time() - start_time
        self.performance_metrics["calculation_time"].append(calc_time)
        
        return result
    
    def _calculate_context_multiplier_optimized(self, skill_type: SkillType, context: Dict[str, Any]) -> float:
        """Optimized context multiplier calculation"""
        # Create simplified cache key
        cache_key = f"context_{skill_type.value}_{hash(str(sorted(context.items())))}"
        
        if cache_key in self.calculation_cache:
            return self.calculation_cache[cache_key]
        
        multiplier = 1.0
        
        # Simplified context processing for speed
        if context.get("equipment_quality", 0) > 0.5:
            multiplier *= 1.1
        
        if context.get("environment_suitability", 0.5) > 0.7:
            multiplier *= 1.15
        
        if context.get("fatigue_level", 0) > 0.3:
            multiplier *= 0.9
        
        if context.get("stress_level", 0) > 0.5:
            multiplier *= 0.85
        
        result = max(0.3, min(2.0, multiplier))
        self.calculation_cache[cache_key] = result
        return result
    
    def calculate_synergy_bonus_optimized(self, agent_id: str, skill_type: SkillType) -> float:
        """Optimized synergy bonus calculation using pre-computed lookups"""
        if agent_id not in self.agent_skills:
            return 0.0
        
        cache_key = f"synergy_{agent_id}_{skill_type.value}"
        
        if cache_key in self.synergy_cache:
            self.performance_metrics["cache_hits"] += 1
            return self.synergy_cache[cache_key]
        
        self.performance_metrics["cache_misses"] += 1
        
        agent_skills = self.agent_skills[agent_id]
        total_bonus = 0.0
        
        # Use pre-computed synergy lookup table
        for (primary, secondary), strength in self._synergy_lookup_cache.items():
            if primary == skill_type and secondary in agent_skills:
                secondary_skill = agent_skills[secondary]
                # Simplified level bonus calculation
                level_bonus = float(secondary_skill.level.value) * 0.1
                total_bonus += strength * level_bonus
        
        result = min(total_bonus, self.synergy_max_bonus)
        self.synergy_cache[cache_key] = result
        return result
    
    def batch_process_experience_updates(self, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple experience updates in a batch for better performance"""
        if not self.batch_processing:
            # Process individually if batch processing is disabled
            return [self._process_single_experience_update(update) for update in updates]
        
        self.performance_metrics["batch_operations"] += 1
        start_time = time.time()
        
        results = []
        
        # Sort updates by agent_id for better cache locality
        sorted_updates = sorted(updates, key=lambda x: x.get("agent_id", ""))
        
        # Process in batches
        for i in range(0, len(sorted_updates), self._batch_size):
            batch = sorted_updates[i:i + self._batch_size]
            batch_results = self._process_experience_batch(batch)
            results.extend(batch_results)
        
        # Clear caches periodically to prevent memory bloat
        if len(self.calculation_cache) > 1000:
            self._cleanup_caches()
        
        # Record performance metrics
        batch_time = time.time() - start_time
        self.performance_metrics["calculation_time"].append(batch_time)
        
        return results
    
    def _process_experience_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of experience updates efficiently"""
        results = []
        
        for update in batch:
            try:
                result = self._process_single_experience_update(update)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing experience update: {e}")
                results.append({"success": False, "error": str(e)})
        
        return results
    
    def _process_single_experience_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single experience update with comprehensive validation"""
        # Validate required fields
        required_fields = ["agent_id", "skill_type", "experience_gain"]
        for field in required_fields:
            if field not in update:
                return {
                    "success": False, 
                    "error": f"Missing required field: {field}",
                    "update": update
                }
        
        agent_id = update.get("agent_id")
        skill_type = update.get("skill_type")
        experience_gain = update.get("experience_gain", 0.0)
        
        # Enhanced validation
        if not agent_id or not str(agent_id).strip():
            return {"success": False, "error": "Agent ID cannot be empty"}
        
        # Handle skill_type conversion if it's a string
        if isinstance(skill_type, str):
            try:
                skill_type = SkillType(skill_type)
            except ValueError:
                return {
                    "success": False, 
                    "error": f"Invalid skill type: {skill_type}",
                    "valid_skills": [skill.value for skill in SkillType]
                }
        
        if not isinstance(skill_type, SkillType):
            return {"success": False, "error": "skill_type must be a SkillType enum or valid string"}
        
        try:
            experience_gain = float(experience_gain)
            if experience_gain < 0:
                return {"success": False, "error": "Experience gain cannot be negative"}
        except (ValueError, TypeError):
            return {"success": False, "error": "experience_gain must be a number"}
        
        try:
            result = self.add_experience(agent_id, skill_type, experience_gain)
            return {
                "success": True, 
                "result": result,
                "agent_id": agent_id,
                "skill_type": skill_type.value,
                "experience_gain": experience_gain
            }
        except Exception as e:
            logger.error(f"Error processing experience update: {e}", exc_info=True)
            return {
                "success": False, 
                "error": str(e),
                "agent_id": agent_id,
                "skill_type": skill_type.value if isinstance(skill_type, SkillType) else skill_type
            }
    
    def _cleanup_caches(self):
        """Clean up old cache entries to prevent memory bloat"""
        # Keep only the most recent 500 entries in each cache
        if len(self.calculation_cache) > 500:
            items = list(self.calculation_cache.items())
            self.calculation_cache = dict(items[-500:])
        
        if len(self.synergy_cache) > 100:
            items = list(self.synergy_cache.items())
            self.synergy_cache = dict(items[-100:])
        
        if len(self._difficulty_cache) > 200:
            items = list(self._difficulty_cache.items())
            self._difficulty_cache = dict(items[-200:])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance optimization metrics"""
        avg_calc_time = 0.0
        if self.performance_metrics["calculation_time"]:
            avg_calc_time = sum(self.performance_metrics["calculation_time"]) / len(self.performance_metrics["calculation_time"])
        
        total_requests = self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]
        cache_hit_rate = 0.0
        if total_requests > 0:
            cache_hit_rate = self.performance_metrics["cache_hits"] / total_requests
        
        return {
            "optimization_enabled": self.optimization_enabled,
            "average_calculation_time_ms": avg_calc_time * 1000,
            "cache_hit_rate": cache_hit_rate,
            "total_cache_hits": self.performance_metrics["cache_hits"],
            "total_cache_misses": self.performance_metrics["cache_misses"],
            "batch_operations": self.performance_metrics["batch_operations"],
            "concurrent_operations": self.performance_metrics["concurrent_operations"],
            "cache_sizes": {
                "calculation_cache": len(self.calculation_cache),
                "synergy_cache": len(self.synergy_cache),
                "difficulty_cache": len(self._difficulty_cache)
            }
        }
    
    def optimize_for_agent_count(self, agent_count: int):
        """Adjust optimization parameters based on agent count"""
        if agent_count < 10:
            # Small scale - disable some optimizations for simplicity
            self.batch_processing = False
            self._batch_size = 10
        elif agent_count < 100:
            # Medium scale - moderate optimizations
            self.batch_processing = True
            self._batch_size = 25
        else:
            # Large scale - full optimizations
            self.batch_processing = True
            self._batch_size = 50
            
        logger.info(f"Optimized skill system for {agent_count} agents")
    
    def clear_optimization_caches(self):
        """Clear all optimization caches (useful for testing or memory management)"""
        self.calculation_cache.clear()
        self.synergy_cache.clear()
        self._difficulty_cache.clear()
        
        # Reset metrics
        self.performance_metrics = {
            "calculation_time": deque(maxlen=100),
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_operations": 0,
            "concurrent_operations": 0
        }
        
        logger.info("Cleared all optimization caches")