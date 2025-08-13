"""
Skill-Economics Integration Module
Task 3.2: Experience-based skill growth algorithms - Economic System Integration

This module bridges the skill development system with the economic resource management
system, enabling skills to affect resource production, consumption, trade efficiency,
and economic outcomes.

Features:
- Skill-based resource production bonuses
- Skill-driven efficiency improvements
- Trade skill applications
- Skill-resource synergies
- Economic incentives for skill development
- Resource costs for skill training
- Skill-based specialization economics
"""

import time
import math
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from .skill_development import (
    SkillType, SkillLevel, SkillInstance, SkillDevelopmentSystem
)

# Import economic system components (assuming they exist)
try:
    from ...economics.resources import (
        ResourceType, ResourceManagementSystem, 
        ResourceInstance, TradeOffer
    )
    ECONOMIC_SYSTEM_AVAILABLE = True
except ImportError:
    # Create mock classes for development
    from enum import Enum
    
    class ResourceType(Enum):
        FOOD = "food"
        WATER = "water"
        TOOLS = "tools"
        SHELTER = "shelter"
        KNOWLEDGE = "knowledge"
    
    class ResourceManagementSystem:
        def __init__(self):
            pass
    
    class ResourceInstance:
        def __init__(self):
            pass
    
    class TradeOffer:
        def __init__(self):
            pass
    
    ECONOMIC_SYSTEM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SkillResourceMapping:
    """Mapping between skills and resources they affect"""
    skill_type: SkillType
    affected_resources: Set[str]  # Resource type names
    production_bonus: float  # Multiplier for production (1.0 = no bonus)
    efficiency_bonus: float  # Reduction in resource consumption (0.0-1.0)
    quality_bonus: float  # Improvement in resource quality (0.0-1.0)
    trade_bonus: float  # Improvement in trade outcomes (0.0-1.0)
    unlock_threshold: SkillLevel  # Minimum skill level for bonuses
    
    def get_effective_production_bonus(self, skill_level: SkillLevel) -> float:
        """Calculate effective production bonus based on skill level"""
        if skill_level.value < self.unlock_threshold.value:
            return 0.0
        
        level_multiplier = (skill_level.value + 1) / (self.unlock_threshold.value + 1)
        return self.production_bonus * level_multiplier
    
    def get_effective_efficiency_bonus(self, skill_level: SkillLevel) -> float:
        """Calculate effective efficiency bonus based on skill level"""
        if skill_level.value < self.unlock_threshold.value:
            return 0.0
        
        level_multiplier = (skill_level.value + 1) / (self.unlock_threshold.value + 1)
        return min(0.5, self.efficiency_bonus * level_multiplier)  # Cap at 50% efficiency


@dataclass
class SkillTrainingCost:
    """Resource costs for skill training and development"""
    skill_type: SkillType
    training_method: str  # "practice", "tutoring", "equipment", "books"
    resource_costs: Dict[str, float]  # Resource type -> amount required
    time_cost: float  # Hours required
    effectiveness_bonus: float  # Bonus to experience gain (0.0-2.0)
    level_scaling: float  # How costs scale with skill level (1.0 = linear)
    
    def calculate_total_cost(self, current_level: SkillLevel, hours: float) -> Dict[str, float]:
        """Calculate total resource cost for training session"""
        level_factor = (current_level.value + 1) ** self.level_scaling
        time_factor = hours / self.time_cost
        
        total_costs = {}
        for resource, base_cost in self.resource_costs.items():
            total_costs[resource] = base_cost * level_factor * time_factor
        
        return total_costs


class SkillEconomicsIntegration:
    """
    Integration system between skill development and economic resource management.
    
    This system handles:
    - Skill-based production and efficiency bonuses
    - Resource costs for skill development
    - Economic incentives for specialization
    - Trade skill applications
    - Skill-driven market dynamics
    """
    
    def __init__(
        self, 
        skill_system: SkillDevelopmentSystem,
        resource_system: Optional[ResourceManagementSystem] = None
    ):
        self.skill_system = skill_system
        self.resource_system = resource_system
        
        # Initialize skill-resource mappings
        self.skill_resource_mappings = self._initialize_skill_resource_mappings()
        self.skill_training_costs = self._initialize_training_costs()
        
        # Economic parameters
        self.skill_market_demand = defaultdict(float)  # Demand for each skill in economy
        self.skill_wage_multipliers = defaultdict(lambda: 1.0)  # Wage multipliers by skill
        self.resource_skill_requirements = defaultdict(set)  # Resources requiring specific skills
        
        # Performance tracking
        self.production_bonuses_applied = defaultdict(float)
        self.efficiency_savings = defaultdict(float)
        self.skill_training_investments = defaultdict(float)
        
        # Update intervals
        self.last_market_update = time.time()
        self.market_update_interval = 3600.0  # Update market dynamics hourly
    
    def calculate_production_bonus(
        self, 
        agent_id: str, 
        resource_type: str, 
        base_production: float
    ) -> float:
        """
        Calculate production bonus from agent skills for a specific resource.
        
        Args:
            agent_id: ID of the producing agent
            resource_type: Type of resource being produced
            base_production: Base production amount
            
        Returns:
            Enhanced production amount with skill bonuses
        """
        if not self.skill_system:
            return base_production
        
        agent_skills = self.skill_system.get_agent_skills(agent_id)
        total_bonus = 0.0
        
        # Check all skill-resource mappings
        for mapping in self.skill_resource_mappings:
            if resource_type not in mapping.affected_resources:
                continue
            
            skill_type = mapping.skill_type
            if skill_type not in agent_skills:
                continue
            
            skill = agent_skills[skill_type]
            production_bonus = mapping.get_effective_production_bonus(skill.level)
            
            # Apply mastery and specialization bonuses
            if skill.specialization_path:
                production_bonus *= (1.0 + skill.mastery_bonus)
            
            total_bonus += production_bonus
        
        # Apply synergy bonuses
        synergy_bonus = self._calculate_production_synergy_bonus(agent_skills, resource_type)
        total_bonus *= (1.0 + synergy_bonus)
        
        enhanced_production = base_production * (1.0 + total_bonus)
        
        # Track bonus application
        self.production_bonuses_applied[agent_id] += enhanced_production - base_production
        
        return enhanced_production
    
    def calculate_efficiency_bonus(
        self,
        agent_id: str,
        resource_type: str,
        base_consumption: float
    ) -> float:
        """
        Calculate resource consumption reduction from agent skills.
        
        Args:
            agent_id: ID of the consuming agent
            resource_type: Type of resource being consumed
            base_consumption: Base consumption amount
            
        Returns:
            Reduced consumption amount with skill efficiency bonuses
        """
        if not self.skill_system:
            return base_consumption
        
        agent_skills = self.skill_system.get_agent_skills(agent_id)
        total_efficiency = 0.0
        
        # Check all skill-resource mappings
        for mapping in self.skill_resource_mappings:
            if resource_type not in mapping.affected_resources:
                continue
            
            skill_type = mapping.skill_type
            if skill_type not in agent_skills:
                continue
            
            skill = agent_skills[skill_type]
            efficiency_bonus = mapping.get_effective_efficiency_bonus(skill.level)
            
            # Apply mastery and specialization bonuses
            if skill.specialization_path:
                efficiency_bonus *= (1.0 + skill.mastery_bonus * 0.5)  # Half effect for efficiency
            
            total_efficiency += efficiency_bonus
        
        # Cap maximum efficiency at 70%
        total_efficiency = min(0.7, total_efficiency)
        
        efficient_consumption = base_consumption * (1.0 - total_efficiency)
        
        # Track efficiency savings
        self.efficiency_savings[agent_id] += base_consumption - efficient_consumption
        
        return efficient_consumption
    
    def calculate_trade_skill_bonus(
        self,
        agent_id: str,
        trade_type: str,
        base_outcome: float
    ) -> float:
        """
        Calculate trade outcome improvements from social and economic skills.
        
        Args:
            agent_id: ID of the trading agent
            trade_type: Type of trade ("negotiate", "evaluate", "network")
            base_outcome: Base trade outcome (price, success rate, etc.)
            
        Returns:
            Enhanced trade outcome with skill bonuses
        """
        if not self.skill_system:
            return base_outcome
        
        agent_skills = self.skill_system.get_agent_skills(agent_id)
        trade_bonus = 0.0
        
        # Trade-relevant skills
        trade_skills = {
            "negotiate": [SkillType.NEGOTIATION, SkillType.PERSUASION, SkillType.EMPATHY],
            "evaluate": [SkillType.ANALYSIS, SkillType.REASONING, SkillType.MEMORY],
            "network": [SkillType.NETWORKING, SkillType.EMPATHY, SkillType.LEADERSHIP]
        }
        
        relevant_skills = trade_skills.get(trade_type, [])
        
        for skill_type in relevant_skills:
            if skill_type not in agent_skills:
                continue
            
            skill = agent_skills[skill_type]
            
            # Calculate skill contribution
            level_bonus = skill.level.value * 0.05  # 5% per level
            success_rate_bonus = (skill.success_rate - 0.5) * 0.2  # Up to 10% from success rate
            mastery_bonus = skill.mastery_bonus * 0.3  # 30% of mastery bonus
            
            skill_contribution = level_bonus + success_rate_bonus + mastery_bonus
            trade_bonus += skill_contribution
        
        # Apply synergy bonus for multiple relevant skills
        if len([s for s in relevant_skills if s in agent_skills]) > 1:
            trade_bonus *= 1.2  # 20% synergy bonus
        
        enhanced_outcome = base_outcome * (1.0 + trade_bonus)
        return enhanced_outcome
    
    def get_skill_training_cost(
        self,
        skill_type: SkillType,
        training_method: str,
        current_level: SkillLevel,
        training_hours: float
    ) -> Dict[str, float]:
        """
        Calculate resource costs for skill training.
        
        Args:
            skill_type: Skill being trained
            training_method: Method of training
            current_level: Current skill level
            training_hours: Hours of training
            
        Returns:
            Dictionary of resource costs
        """
        # Find matching training cost structure
        training_cost = None
        for cost in self.skill_training_costs:
            if cost.skill_type == skill_type and cost.training_method == training_method:
                training_cost = cost
                break
        
        if not training_cost:
            # Default basic training cost
            return {"time": training_hours}
        
        return training_cost.calculate_total_cost(current_level, training_hours)
    
    def can_afford_training(
        self,
        agent_id: str,
        skill_type: SkillType,
        training_method: str,
        current_level: SkillLevel,
        training_hours: float
    ) -> bool:
        """
        Check if agent can afford skill training costs.
        
        Args:
            agent_id: ID of the agent
            skill_type: Skill to train
            training_method: Training method
            current_level: Current skill level
            training_hours: Hours of training
            
        Returns:
            True if agent can afford training
        """
        if not self.resource_system:
            return True  # No resource constraints without economic system
        
        required_costs = self.get_skill_training_cost(
            skill_type, training_method, current_level, training_hours
        )
        
        # Check if agent has required resources
        for resource_type, required_amount in required_costs.items():
            if resource_type == "time":
                continue  # Time is handled separately
            
            agent_inventory = self.resource_system.get_agent_inventory(agent_id)
            available_amount = agent_inventory.get(resource_type, 0.0)
            
            if available_amount < required_amount:
                return False
        
        return True
    
    def consume_training_resources(
        self,
        agent_id: str,
        skill_type: SkillType,
        training_method: str,
        current_level: SkillLevel,
        training_hours: float
    ) -> bool:
        """
        Consume resources required for skill training.
        
        Args:
            agent_id: ID of the agent
            skill_type: Skill to train
            training_method: Training method
            current_level: Current skill level
            training_hours: Hours of training
            
        Returns:
            True if resources were successfully consumed
        """
        if not self.resource_system:
            return True
        
        required_costs = self.get_skill_training_cost(
            skill_type, training_method, current_level, training_hours
        )
        
        # Consume required resources
        for resource_type, required_amount in required_costs.items():
            if resource_type == "time":
                continue  # Time is handled by the training system
            
            # Find resource type enum
            resource_enum = None
            try:
                for rt in ResourceType:
                    if rt.value == resource_type:
                        resource_enum = rt
                        break
            except:
                continue  # Skip unknown resource types
            
            if resource_enum:
                consumed = self.resource_system.consume_resource(
                    agent_id, resource_enum, required_amount
                )
                
                if consumed < required_amount * 0.9:  # Allow 10% shortage
                    return False
        
        # Track investment
        total_value = sum(required_costs.values())
        self.skill_training_investments[agent_id] += total_value
        
        return True
    
    def calculate_skill_wages(
        self,
        agent_id: str,
        task_type: str,
        base_wage: float
    ) -> float:
        """
        Calculate wage premium based on relevant skills.
        
        Args:
            agent_id: ID of the agent
            task_type: Type of task/job
            base_wage: Base wage for the task
            
        Returns:
            Enhanced wage with skill bonuses
        """
        if not self.skill_system:
            return base_wage
        
        agent_skills = self.skill_system.get_agent_skills(agent_id)
        wage_multiplier = 1.0
        
        # Task-skill mapping
        task_skills = {
            "crafting": [SkillType.CRAFTING, SkillType.CREATIVITY, SkillType.FOCUS],
            "leadership": [SkillType.LEADERSHIP, SkillType.PERSUASION, SkillType.EMPATHY],
            "research": [SkillType.RESEARCH, SkillType.ANALYSIS, SkillType.MEMORY],
            "survival": [SkillType.FORAGING, SkillType.HUNTING, SkillType.SHELTER_BUILDING],
            "combat": [SkillType.COMBAT, SkillType.ATHLETICS, SkillType.STEALTH],
            "social": [SkillType.PERSUASION, SkillType.EMPATHY, SkillType.NETWORKING]
        }
        
        relevant_skills = task_skills.get(task_type, [])
        
        for skill_type in relevant_skills:
            if skill_type not in agent_skills:
                continue
            
            skill = agent_skills[skill_type]
            
            # Higher skill levels command higher wages
            level_multiplier = 1.0 + (skill.level.value * 0.15)  # 15% per level
            
            # Specialization adds premium
            if skill.specialization_path:
                level_multiplier *= (1.0 + skill.mastery_bonus)
            
            wage_multiplier *= level_multiplier
        
        # Market demand affects wages
        market_demand = self.skill_market_demand.get(task_type, 1.0)
        wage_multiplier *= market_demand
        
        return base_wage * wage_multiplier
    
    def update_skill_market_dynamics(self):
        """
        Update market dynamics affecting skill values and demands.
        """
        current_time = time.time()
        if current_time - self.last_market_update < self.market_update_interval:
            return
        
        if not self.resource_system:
            return
        
        # Analyze resource scarcity to determine skill demand
        for resource_type in ResourceType:
            availability = self.resource_system.get_resource_availability(resource_type)
            scarcity_level = availability.get('scarcity_level', 0.0)
            
            # High scarcity increases demand for production skills
            if scarcity_level > 0.5:
                for mapping in self.skill_resource_mappings:
                    if resource_type.value in mapping.affected_resources:
                        skill_name = mapping.skill_type.value
                        current_demand = self.skill_market_demand[skill_name]
                        demand_increase = scarcity_level * 0.5
                        self.skill_market_demand[skill_name] = min(2.0, current_demand + demand_increase)
        
        # Update wage multipliers based on supply and demand
        all_agents = self.skill_system.get_all_agents()
        
        for skill_type in SkillType:
            skill_name = skill_type.value
            
            # Count agents with this skill
            skilled_agents = 0
            total_skill_level = 0
            
            for agent_id in all_agents:
                agent_skills = self.skill_system.get_agent_skills(agent_id)
                if skill_type in agent_skills:
                    skilled_agents += 1
                    total_skill_level += agent_skills[skill_type].level.value
            
            if skilled_agents > 0:
                # High demand, low supply = higher wages
                supply_factor = skilled_agents / max(len(all_agents), 1)  # Proportion of skilled agents
                demand_factor = self.skill_market_demand[skill_name]
                
                wage_multiplier = demand_factor / max(supply_factor, 0.1)
                self.skill_wage_multipliers[skill_name] = min(3.0, max(0.5, wage_multiplier))
        
        self.last_market_update = current_time
    
    def get_economic_skill_report(self, agent_id: str) -> Dict[str, Any]:
        """
        Generate economic impact report for an agent's skills.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with economic skill analysis
        """
        if not self.skill_system:
            return {"error": "Skill system not available"}
        
        agent_skills = self.skill_system.get_agent_skills(agent_id)
        
        # Calculate economic impact
        production_impact = self.production_bonuses_applied.get(agent_id, 0.0)
        efficiency_impact = self.efficiency_savings.get(agent_id, 0.0)
        training_investment = self.skill_training_investments.get(agent_id, 0.0)
        
        # Analyze skill portfolio value
        skill_values = {}
        total_wage_premium = 0.0
        
        for skill_type, skill in agent_skills.items():
            skill_name = skill_type.value
            
            # Calculate potential wage premium
            base_wage = 10.0  # Baseline wage
            premium_wage = self.calculate_skill_wages(agent_id, skill_name, base_wage)
            wage_premium = premium_wage - base_wage
            total_wage_premium += wage_premium
            
            # Market demand for this skill
            market_demand = self.skill_market_demand.get(skill_name, 1.0)
            
            skill_values[skill_name] = {
                "level": skill.level.name,
                "experience": skill.experience_points,
                "wage_premium": wage_premium,
                "market_demand": market_demand,
                "specialization": skill.specialization_path,
                "economic_value": wage_premium * market_demand
            }
        
        # Overall economic analysis
        total_economic_value = sum(sv["economic_value"] for sv in skill_values.values())
        roi = (production_impact + efficiency_impact - training_investment) / max(training_investment, 1.0)
        
        return {
            "agent_id": agent_id,
            "skill_count": len(agent_skills),
            "total_economic_value": total_economic_value,
            "production_bonuses": production_impact,
            "efficiency_savings": efficiency_impact,
            "training_investment": training_investment,
            "roi": roi,
            "total_wage_premium": total_wage_premium,
            "skill_portfolio": skill_values,
            "market_positioning": "high_value" if total_economic_value > 50 else "developing"
        }
    
    def _initialize_skill_resource_mappings(self) -> List[SkillResourceMapping]:
        """Initialize mappings between skills and resources they affect."""
        return [
            # Physical production skills
            SkillResourceMapping(
                skill_type=SkillType.CRAFTING,
                affected_resources={"tools", "shelter", "clothing"},
                production_bonus=0.3,
                efficiency_bonus=0.2,
                quality_bonus=0.25,
                trade_bonus=0.15,
                unlock_threshold=SkillLevel.BEGINNER
            ),
            SkillResourceMapping(
                skill_type=SkillType.FORAGING,
                affected_resources={"food", "medicine", "materials"},
                production_bonus=0.4,
                efficiency_bonus=0.15,
                quality_bonus=0.2,
                trade_bonus=0.1,
                unlock_threshold=SkillLevel.NOVICE
            ),
            SkillResourceMapping(
                skill_type=SkillType.HUNTING,
                affected_resources={"food", "materials", "tools"},
                production_bonus=0.35,
                efficiency_bonus=0.1,
                quality_bonus=0.3,
                trade_bonus=0.05,
                unlock_threshold=SkillLevel.BEGINNER
            ),
            SkillResourceMapping(
                skill_type=SkillType.ENGINEERING,
                affected_resources={"tools", "technology", "infrastructure"},
                production_bonus=0.5,
                efficiency_bonus=0.3,
                quality_bonus=0.4,
                trade_bonus=0.2,
                unlock_threshold=SkillLevel.COMPETENT
            ),
            
            # Mental efficiency skills
            SkillResourceMapping(
                skill_type=SkillType.ANALYSIS,
                affected_resources={"information", "knowledge", "research"},
                production_bonus=0.25,
                efficiency_bonus=0.2,
                quality_bonus=0.35,
                trade_bonus=0.25,
                unlock_threshold=SkillLevel.BEGINNER
            ),
            SkillResourceMapping(
                skill_type=SkillType.MEMORY,
                affected_resources={"knowledge", "information"},
                production_bonus=0.15,
                efficiency_bonus=0.25,
                quality_bonus=0.2,
                trade_bonus=0.1,
                unlock_threshold=SkillLevel.NOVICE
            ),
            
            # Social economic skills
            SkillResourceMapping(
                skill_type=SkillType.NEGOTIATION,
                affected_resources={"currency", "trade_goods"},
                production_bonus=0.1,
                efficiency_bonus=0.05,
                quality_bonus=0.1,
                trade_bonus=0.4,
                unlock_threshold=SkillLevel.COMPETENT
            ),
            SkillResourceMapping(
                skill_type=SkillType.LEADERSHIP,
                affected_resources={"organization", "productivity"},
                production_bonus=0.2,
                efficiency_bonus=0.15,
                quality_bonus=0.15,
                trade_bonus=0.3,
                unlock_threshold=SkillLevel.PROFICIENT
            )
        ]
    
    def _initialize_training_costs(self) -> List[SkillTrainingCost]:
        """Initialize resource costs for skill training."""
        return [
            # Basic practice (low cost, moderate effectiveness)
            SkillTrainingCost(
                skill_type=SkillType.ATHLETICS,
                training_method="practice",
                resource_costs={"food": 2.0, "water": 3.0},
                time_cost=1.0,
                effectiveness_bonus=0.0,
                level_scaling=1.2
            ),
            
            # Equipment-based training (higher cost, better effectiveness)
            SkillTrainingCost(
                skill_type=SkillType.CRAFTING,
                training_method="equipment",
                resource_costs={"tools": 1.0, "materials": 5.0},
                time_cost=2.0,
                effectiveness_bonus=0.5,
                level_scaling=1.5
            ),
            
            # Tutoring (high cost, high effectiveness)
            SkillTrainingCost(
                skill_type=SkillType.ANALYSIS,
                training_method="tutoring",
                resource_costs={"currency": 20.0, "knowledge": 2.0},
                time_cost=0.5,
                effectiveness_bonus=1.0,
                level_scaling=2.0
            ),
            
            # Book learning (moderate cost, good for mental skills)
            SkillTrainingCost(
                skill_type=SkillType.RESEARCH,
                training_method="books",
                resource_costs={"knowledge": 3.0, "time": 4.0},
                time_cost=3.0,
                effectiveness_bonus=0.3,
                level_scaling=1.3
            )
        ]
    
    def _calculate_production_synergy_bonus(
        self, 
        agent_skills: Dict[SkillType, SkillInstance], 
        resource_type: str
    ) -> float:
        """Calculate synergy bonus for production from multiple related skills."""
        relevant_skills = []
        
        # Find skills that affect this resource
        for mapping in self.skill_resource_mappings:
            if resource_type in mapping.affected_resources:
                if mapping.skill_type in agent_skills:
                    relevant_skills.append(agent_skills[mapping.skill_type])
        
        if len(relevant_skills) < 2:
            return 0.0  # No synergy with less than 2 skills
        
        # Calculate synergy based on skill levels and complementarity
        avg_level = sum(skill.level.value for skill in relevant_skills) / len(relevant_skills)
        synergy_bonus = (len(relevant_skills) - 1) * 0.05 * (avg_level / 5.0)  # Max 5% per additional skill
        
        return min(0.25, synergy_bonus)  # Cap synergy bonus at 25%