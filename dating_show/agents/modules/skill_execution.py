"""
Enhanced Skill Execution Module with Comprehensive Skill Development Integration
Task 3.2: Experience-based skill growth algorithms

This module executes actions based on agent decisions and integrates with the
comprehensive skill development system to track skill usage, calculate performance,
and provide experience-based learning.

Features:
- Skill-based action execution with performance calculation
- Experience gain from action outcomes
- Integration with skill development system
- Context-aware skill application
- Performance feedback and learning
"""

import time
import random
from typing import Dict, Any, Optional, Tuple
from .base_module import BaseModule
from .skill_development import (
    SkillDevelopmentSystem, SkillType, LearningSourceType,
    LearningEvent
)

class SkillExecutionModule(BaseModule):
    """
    Enhanced skill execution module that integrates with the comprehensive
    skill development system for experience-based learning.
    """
    
    def __init__(self, agent_state, skill_system: Optional[SkillDevelopmentSystem] = None):
        super().__init__(agent_state)
        
        # Integration with skill development system
        self.skill_system = skill_system
        if self.skill_system and hasattr(agent_state, 'name'):
            # Ensure agent is registered with skill system
            self.skill_system.add_agent(agent_state.name)
        
        # Action-to-skill mapping
        self.action_skill_mapping = self._initialize_action_skill_mapping()
        
        # Performance tracking
        self.recent_performance = {}
        self.action_history = []
        
    def run(self):
        """
        Enhanced skill execution that considers agent skills and provides learning opportunities.
        """
        decision = self.agent_state.proprioception.get("current_decision")
        if not decision:
            return

        # Parse decision and determine required skills
        action_info = self._parse_decision(decision)
        if not action_info:
            return
            
        # Execute the action using the skill system
        execution_result = self._execute_skill_action(action_info)
        
        # Store results in agent state
        self.agent_state.proprioception["executed_action"] = execution_result
        self.agent_state.proprioception["last_action_performance"] = execution_result.get("performance_score", 0.5)
        
        # Update action history
        self.action_history.append({
            "timestamp": time.time(),
            "action": action_info,
            "result": execution_result
        })
        
        # Keep only recent history
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-50:]
    
    def _parse_decision(self, decision: str) -> Optional[Dict[str, Any]]:
        """
        Parse agent decision into actionable information with skill requirements.
        """
        action_info = None
        
        if "observe the environment" in decision:
            action_info = {
                "skill": SkillType.ANALYSIS,
                "action": "observe_environment",
                "target": "environment",
                "difficulty": 0.3,  # Basic observation
                "context": {
                    "observation_type": "general",
                    "detail_level": "medium"
                }
            }
            
        elif "form a connection with" in decision:
            target_agent = decision.split("form a connection with ")[-1].strip()
            action_info = {
                "skill": SkillType.EMPATHY,
                "action": "social_connection", 
                "target": target_agent,
                "difficulty": 0.5,  # Social connections are moderate difficulty
                "context": {
                    "interaction_type": "connection_building",
                    "target": target_agent,
                    "social_pressure": 0.4
                }
            }
            
        elif "persuade" in decision.lower():
            target = self._extract_target_from_decision(decision)
            action_info = {
                "skill": SkillType.PERSUASION,
                "action": "persuade",
                "target": target,
                "difficulty": 0.6,
                "context": {
                    "persuasion_type": "general",
                    "social_pressure": 0.6
                }
            }
            
        elif "craft" in decision.lower() or "make" in decision.lower():
            action_info = {
                "skill": SkillType.CRAFTING,
                "action": "craft_item",
                "target": "item",
                "difficulty": 0.5,
                "context": {
                    "crafting_type": "basic",
                    "tools_available": True
                }
            }
            
        elif "hunt" in decision.lower():
            action_info = {
                "skill": SkillType.HUNTING,
                "action": "hunt",
                "target": "prey",
                "difficulty": 0.7,
                "context": {
                    "environment": "wilderness",
                    "weather_conditions": "neutral"
                }
            }
            
        elif "gather" in decision.lower() or "forage" in decision.lower():
            action_info = {
                "skill": SkillType.FORAGING,
                "action": "gather_resources",
                "target": "resources",
                "difficulty": 0.4,
                "context": {
                    "resource_type": "food",
                    "area_familiarity": 0.5
                }
            }
            
        elif "build" in decision.lower():
            action_info = {
                "skill": SkillType.SHELTER_BUILDING,
                "action": "build_structure",
                "target": "structure",
                "difficulty": 0.6,
                "context": {
                    "structure_type": "basic",
                    "materials_available": True
                }
            }
            
        elif "fight" in decision.lower() or "attack" in decision.lower():
            target = self._extract_target_from_decision(decision)
            action_info = {
                "skill": SkillType.COMBAT,
                "action": "combat",
                "target": target,
                "difficulty": 0.7,
                "context": {
                    "combat_type": "direct",
                    "weapons_available": True
                }
            }
            
        elif "analyze" in decision.lower() or "study" in decision.lower():
            action_info = {
                "skill": SkillType.ANALYSIS,
                "action": "analyze",
                "target": "subject",
                "difficulty": 0.5,
                "context": {
                    "analysis_type": "general",
                    "complexity": "medium"
                }
            }
            
        elif "lead" in decision.lower() or "organize" in decision.lower():
            action_info = {
                "skill": SkillType.LEADERSHIP,
                "action": "leadership",
                "target": "group",
                "difficulty": 0.6,
                "context": {
                    "group_size": "small",
                    "leadership_style": "collaborative"
                }
            }
        
        # Add environmental context from agent state
        if action_info:
            action_info["context"].update(self._get_environmental_context())
            
        return action_info
    
    def _execute_skill_action(self, action_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action using the skill development system.
        """
        if not self.skill_system or not hasattr(self.agent_state, 'name'):
            # Fallback to basic execution without skill system
            return self._basic_action_execution(action_info)
        
        skill_type = action_info["skill"]
        action = action_info["action"]
        target = action_info["target"]
        context = action_info.get("context", {})
        
        # Add agent-specific context
        context.update({
            "agent_fatigue": self._get_agent_fatigue(),
            "agent_stress": self._get_agent_stress(),
            "recent_performance": self.recent_performance.get(skill_type, 0.5)
        })
        
        # Execute action through skill system
        result = self.skill_system.execute_skill_action(
            agent_id=self.agent_state.name,
            skill_type=skill_type,
            action=action,
            target=target,
            context=context
        )
        
        # Update recent performance tracking
        if result.get("success", False):
            performance = result.get("performance_score", 0.5)
            self.recent_performance[skill_type] = performance
        
        # Dynamic Skill Discovery Integration - Task 3.2.1.1
        discovered_skills = []
        if hasattr(self.skill_system, 'discover_skills_from_actions'):
            performance_score = result.get("performance_score", 0.0)
            discovered_skills = self.skill_system.discover_skills_from_actions(
                agent_id=self.agent_state.name,
                action=action,
                performance=performance_score,
                context=context
            )
        
        # Create enhanced result with additional context
        enhanced_result = {
            "skill_used": skill_type.value,
            "action": action,
            "target": target,
            "success": result.get("success", False),
            "performance_score": result.get("performance_score", 0.0),
            "experience_gained": result.get("experience_gained", 0.0),
            "skill_level": result.get("skill_level", "novice"),
            "task_difficulty": result.get("task_difficulty", 0.5),
            "timestamp": time.time(),
            "context": context,
            "discovered_skills": [skill.value for skill in discovered_skills]  # Add discovered skills
        }
        
        # Log skill discoveries
        if discovered_skills:
            discovery_msg = f"Agent {self.agent_state.name} discovered {len(discovered_skills)} new skills: {[s.value for s in discovered_skills]}"
            if hasattr(self.agent_state, 'add_to_memory'):
                self.agent_state.add_to_memory(discovery_msg)
        
        return enhanced_result
    
    def _basic_action_execution(self, action_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Basic action execution without skill system integration (fallback).
        """
        # Simple success probability based on difficulty
        difficulty = action_info.get("difficulty", 0.5)
        success_probability = 0.8 - (difficulty * 0.4)  # 40% to 80% success rate
        
        success = random.random() < success_probability
        performance = random.uniform(0.3, 0.9) if success else random.uniform(0.1, 0.4)
        
        return {
            "skill_used": action_info["skill"].value if hasattr(action_info["skill"], 'value') else str(action_info["skill"]),
            "action": action_info["action"],
            "target": action_info["target"], 
            "success": success,
            "performance_score": performance,
            "experience_gained": 0.0,  # No experience without skill system
            "skill_level": "unknown",
            "task_difficulty": difficulty,
            "timestamp": time.time(),
            "context": action_info.get("context", {})
        }
    
    def _extract_target_from_decision(self, decision: str) -> str:
        """Extract target entity from decision text."""
        words = decision.split()
        
        # Simple target extraction
        for i, word in enumerate(words):
            if word.lower() in ["with", "at", "against", "on"]:
                if i + 1 < len(words):
                    return words[i + 1]
        
        return "unknown"
    
    def _get_environmental_context(self) -> Dict[str, Any]:
        """Get environmental context from agent state."""
        context = {
            "time_of_day": "day",  # Could be extracted from game state
            "weather": "clear",    # Could be extracted from game state
            "location": "unknown", # Could be extracted from spatial memory
            "resources_available": 0.5,  # Could be calculated from inventory
            "social_context": "neutral"   # Could be derived from social state
        }
        
        # Try to get more specific context from agent state if available
        if hasattr(self.agent_state, 'spatial_memory'):
            context["location"] = getattr(self.agent_state.spatial_memory, 'current_location', 'unknown')
        
        if hasattr(self.agent_state, 'scratch'):
            # Try to infer time and environmental conditions
            if hasattr(self.agent_state.scratch, 'currently'):
                current_activity = getattr(self.agent_state.scratch, 'currently', '')
                if 'tired' in current_activity.lower():
                    context["agent_state"] = "tired"
                elif 'energetic' in current_activity.lower():
                    context["agent_state"] = "energetic"
        
        return context
    
    def _get_agent_fatigue(self) -> float:
        """Estimate agent fatigue level (0.0 = no fatigue, 1.0 = exhausted)."""
        # Simple fatigue model based on recent action count
        recent_actions = len([a for a in self.action_history 
                            if time.time() - a["timestamp"] < 3600])  # Last hour
        
        # More actions = more fatigue
        fatigue = min(1.0, recent_actions / 20.0)
        
        return fatigue
    
    def _get_agent_stress(self) -> float:
        """Estimate agent stress level (0.0 = calm, 1.0 = highly stressed)."""
        # Simple stress model based on recent failures
        recent_failures = len([a for a in self.action_history[-10:] 
                             if not a["result"].get("success", True)])
        
        # More recent failures = more stress
        stress = min(1.0, recent_failures / 5.0)
        
        return stress
    
    def _initialize_action_skill_mapping(self) -> Dict[str, SkillType]:
        """Initialize mapping of actions to required skills."""
        return {
            "observe": SkillType.ANALYSIS,
            "move_to": SkillType.ATHLETICS,
            "social_connection": SkillType.EMPATHY,
            "persuade": SkillType.PERSUASION,
            "craft_item": SkillType.CRAFTING,
            "hunt": SkillType.HUNTING,
            "gather_resources": SkillType.FORAGING,
            "build_structure": SkillType.SHELTER_BUILDING,
            "combat": SkillType.COMBAT,
            "analyze": SkillType.ANALYSIS,
            "leadership": SkillType.LEADERSHIP,
            "negotiate": SkillType.NEGOTIATION,
            "stealth": SkillType.STEALTH,
            "athletics": SkillType.ATHLETICS,
            "engineering": SkillType.ENGINEERING,
            "research": SkillType.RESEARCH,
            "planning": SkillType.PLANNING
        }
    
    def get_skill_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of recent skill performance."""
        if not self.action_history:
            return {"message": "No recent actions recorded"}
        
        # Analyze recent performance
        recent_actions = self.action_history[-20:]  # Last 20 actions
        
        skill_stats = {}
        for action_record in recent_actions:
            result = action_record["result"]
            skill_used = result.get("skill_used", "unknown")
            
            if skill_used not in skill_stats:
                skill_stats[skill_used] = {
                    "attempts": 0,
                    "successes": 0,
                    "total_performance": 0.0,
                    "total_experience": 0.0
                }
            
            stats = skill_stats[skill_used]
            stats["attempts"] += 1
            if result.get("success", False):
                stats["successes"] += 1
            stats["total_performance"] += result.get("performance_score", 0.0)
            stats["total_experience"] += result.get("experience_gained", 0.0)
        
        # Calculate averages
        for skill, stats in skill_stats.items():
            if stats["attempts"] > 0:
                stats["success_rate"] = stats["successes"] / stats["attempts"]
                stats["average_performance"] = stats["total_performance"] / stats["attempts"]
            else:
                stats["success_rate"] = 0.0
                stats["average_performance"] = 0.0
        
        return {
            "total_recent_actions": len(recent_actions),
            "skill_statistics": skill_stats,
            "overall_success_rate": sum(1 for a in recent_actions if a["result"].get("success", False)) / len(recent_actions),
            "total_experience_gained": sum(a["result"].get("experience_gained", 0.0) for a in recent_actions)
        }
