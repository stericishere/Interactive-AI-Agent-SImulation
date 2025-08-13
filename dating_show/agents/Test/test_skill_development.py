"""
Test Suite for Skill Development System
Task 3.2: Experience-based skill growth algorithms

This test suite validates the comprehensive skill development system including:
- Skill framework and categories
- Experience tracking and progression
- Learning algorithms
- Skill specialization and mastery
- Skill interactions and synergies
- Decay and maintenance mechanisms
- Teaching and knowledge transfer
- Performance metrics calculation
"""

import pytest
import time
import math
from unittest.mock import Mock, patch
from typing import Dict, List, Optional, Set

# Import system modules
from ..modules.skill_development import (
    SkillType, SkillCategory, SkillLevel, 
    SkillInstance, ExperiencePoint, LearningEvent,
    SkillDevelopmentSystem, SkillInteractionType, LearningSourceType
)


class TestSkillFramework:
    """Test the basic skill framework and data structures"""
    
    def test_skill_types_exist(self):
        """Test that all required skill types are defined"""
        # Physical skills
        assert SkillType.COMBAT in SkillType
        assert SkillType.ATHLETICS in SkillType
        assert SkillType.CRAFTING in SkillType
        
        # Mental skills
        assert SkillType.REASONING in SkillType
        assert SkillType.MEMORY in SkillType
        assert SkillType.ANALYSIS in SkillType
        
        # Social skills
        assert SkillType.PERSUASION in SkillType
        assert SkillType.EMPATHY in SkillType
        assert SkillType.LEADERSHIP in SkillType
        
        # Survival skills
        assert SkillType.FORAGING in SkillType
        assert SkillType.HUNTING in SkillType
        assert SkillType.SHELTER_BUILDING in SkillType
    
    def test_skill_categories_mapping(self):
        """Test that skills are properly categorized"""
        system = SkillDevelopmentSystem()
        
        # This will test the mapping when implemented
        assert hasattr(system, 'get_skill_category')
        
        # Physical category
        assert system.get_skill_category(SkillType.COMBAT) == SkillCategory.PHYSICAL
        assert system.get_skill_category(SkillType.ATHLETICS) == SkillCategory.PHYSICAL
        
        # Mental category 
        assert system.get_skill_category(SkillType.REASONING) == SkillCategory.MENTAL
        assert system.get_skill_category(SkillType.ANALYSIS) == SkillCategory.MENTAL
        
        # Social category
        assert system.get_skill_category(SkillType.PERSUASION) == SkillCategory.SOCIAL
        assert system.get_skill_category(SkillType.EMPATHY) == SkillCategory.SOCIAL
    
    def test_skill_levels_progression(self):
        """Test skill level progression thresholds"""
        system = SkillDevelopmentSystem()
        
        # Test XP to level conversion
        assert system.calculate_skill_level(50) == SkillLevel.NOVICE
        assert system.calculate_skill_level(150) == SkillLevel.BEGINNER
        assert system.calculate_skill_level(400) == SkillLevel.COMPETENT
        assert system.calculate_skill_level(800) == SkillLevel.PROFICIENT
        assert system.calculate_skill_level(1200) == SkillLevel.EXPERT
        assert system.calculate_skill_level(1600) == SkillLevel.MASTER
        
    def test_skill_instance_creation(self):
        """Test skill instance data structure"""
        skill = SkillInstance(
            skill_type=SkillType.COMBAT,
            level=SkillLevel.NOVICE,
            experience_points=50.0,
            practice_hours=10.0,
            success_rate=0.6,
            last_practiced=time.time(),
            decay_rate=0.01,
            mastery_bonus=0.0,
            specialization_path=None
        )
        
        assert skill.skill_type == SkillType.COMBAT
        assert skill.level == SkillLevel.NOVICE
        assert skill.experience_points == 50.0
        assert 0.0 <= skill.success_rate <= 1.0


class TestExperienceSystem:
    """Test experience tracking and calculation"""
    
    def test_experience_point_creation(self):
        """Test experience point data structure"""
        exp = ExperiencePoint(
            skill_type=SkillType.REASONING,
            amount=10.0,
            source="practice",
            multiplier=1.0,
            timestamp=time.time(),
            context={"difficulty": "medium", "assistance": False}
        )
        
        assert exp.skill_type == SkillType.REASONING
        assert exp.amount == 10.0
        assert exp.source == "practice"
        assert exp.multiplier == 1.0
    
    def test_experience_gain_calculation(self):
        """Test experience gain from different sources"""
        system = SkillDevelopmentSystem()
        
        # Practice experience (base rate)
        practice_exp = system.calculate_experience_gain(
            skill_type=SkillType.COMBAT,
            source="practice",
            performance=0.7,
            difficulty=0.5,
            current_level=SkillLevel.NOVICE
        )
        assert practice_exp > 0
        
        # Success experience (bonus)
        success_exp = system.calculate_experience_gain(
            skill_type=SkillType.COMBAT,
            source="success", 
            performance=0.9,
            difficulty=0.8,
            current_level=SkillLevel.NOVICE
        )
        assert success_exp > practice_exp
        
        # Failure experience (reduced but still positive)
        failure_exp = system.calculate_experience_gain(
            skill_type=SkillType.COMBAT,
            source="failure",
            performance=0.2,
            difficulty=0.8,
            current_level=SkillLevel.NOVICE
        )
        assert 0 < failure_exp < practice_exp
    
    def test_learning_curve_effects(self):
        """Test learning rate changes with skill level"""
        system = SkillDevelopmentSystem()
        
        # Learning should be faster at lower levels
        novice_rate = system.get_learning_rate(SkillLevel.NOVICE, SkillType.COMBAT)
        expert_rate = system.get_learning_rate(SkillLevel.EXPERT, SkillType.COMBAT) 
        
        assert novice_rate > expert_rate
        
        # Diminishing returns
        assert novice_rate > 0.5
        assert expert_rate < 0.3


class TestLearningAlgorithms:
    """Test learning algorithms and progression mechanics"""
    
    def test_practice_effectiveness(self):
        """Test that practice effectiveness varies with conditions"""
        system = SkillDevelopmentSystem()
        
        # Focused practice should be more effective
        focused_exp = system.calculate_practice_effectiveness(
            focus_level=0.9,
            fatigue_level=0.1,
            difficulty_level=0.6,
            tool_quality=0.8
        )
        
        unfocused_exp = system.calculate_practice_effectiveness(
            focus_level=0.3,
            fatigue_level=0.7, 
            difficulty_level=0.6,
            tool_quality=0.4
        )
        
        assert focused_exp > unfocused_exp
        
    def test_success_failure_learning(self):
        """Test learning from success vs failure"""
        system = SkillDevelopmentSystem()
        
        # Success provides confidence bonus
        success_event = LearningEvent(
            agent_id="test_agent",
            skill_type=SkillType.PERSUASION,
            event_type="success", 
            experience_gained=15.0,
            performance_score=0.85,
            context={"target_difficulty": 0.6},
            timestamp=time.time()
        )
        
        failure_event = LearningEvent(
            agent_id="test_agent",
            skill_type=SkillType.PERSUASION,
            event_type="failure",
            experience_gained=8.0,
            performance_score=0.25,
            context={"target_difficulty": 0.8},
            timestamp=time.time()
        )
        
        # Process learning events
        system.process_learning_event(success_event)
        system.process_learning_event(failure_event)
        
        # Success should provide more experience
        assert success_event.experience_gained > failure_event.experience_gained
        
    def test_difficulty_scaling(self):
        """Test that difficulty affects learning appropriately"""
        system = SkillDevelopmentSystem()
        
        # Higher difficulty should provide more experience when successful
        easy_exp = system.calculate_experience_gain(
            skill_type=SkillType.ANALYSIS,
            source="success",
            performance=0.8,
            difficulty=0.3,
            current_level=SkillLevel.COMPETENT
        )
        
        hard_exp = system.calculate_experience_gain(
            skill_type=SkillType.ANALYSIS,
            source="success",
            performance=0.8,
            difficulty=0.8,
            current_level=SkillLevel.COMPETENT
        )
        
        assert hard_exp > easy_exp


class TestSkillSpecialization:
    """Test skill specialization and mastery pathways"""
    
    def test_specialization_paths(self):
        """Test that specialization paths are available"""
        system = SkillDevelopmentSystem()
        
        # Combat specializations
        combat_specs = system.get_specialization_paths(SkillType.COMBAT)
        assert "weapon_master" in combat_specs
        assert "tactical_fighter" in combat_specs
        assert "berserker" in combat_specs
        
        # Social specializations
        social_specs = system.get_specialization_paths(SkillType.PERSUASION)
        assert "diplomat" in social_specs
        assert "manipulator" in social_specs
        assert "inspirational_leader" in social_specs
        
    def test_specialization_requirements(self):
        """Test specialization unlock requirements"""
        system = SkillDevelopmentSystem()
        
        # Should require minimum level
        can_specialize = system.can_specialize(
            skill_type=SkillType.COMBAT,
            current_level=SkillLevel.NOVICE,
            specialization="weapon_master"
        )
        assert not can_specialize
        
        # Should be available at higher levels
        can_specialize = system.can_specialize(
            skill_type=SkillType.COMBAT, 
            current_level=SkillLevel.PROFICIENT,
            specialization="weapon_master"
        )
        assert can_specialize
        
    def test_mastery_bonuses(self):
        """Test mastery bonuses for specialized skills"""
        system = SkillDevelopmentSystem()
        
        # Create specialized skill instance
        specialized_skill = SkillInstance(
            skill_type=SkillType.COMBAT,
            level=SkillLevel.EXPERT,
            experience_points=1200.0,
            practice_hours=100.0,
            success_rate=0.85,
            last_practiced=time.time(),
            decay_rate=0.005,  # Slower decay for specialized skills
            mastery_bonus=0.15,  # 15% bonus
            specialization_path="weapon_master"
        )
        
        # Calculate effective skill level with mastery bonus
        effective_performance = system.calculate_performance(specialized_skill)
        base_performance = system.calculate_base_performance(specialized_skill.level)
        
        assert effective_performance > base_performance


class TestSkillInteractions:
    """Test skill synergies and prerequisites"""
    
    def test_skill_synergies(self):
        """Test that related skills provide synergy bonuses"""
        system = SkillDevelopmentSystem()
        
        # Athletic skills should synergize with combat
        synergy_bonus = system.calculate_synergy_bonus(
            primary_skill=SkillType.COMBAT,
            secondary_skills={
                SkillType.ATHLETICS: SkillLevel.COMPETENT,
                SkillType.REASONING: SkillLevel.PROFICIENT
            }
        )
        
        assert synergy_bonus > 0
        
        # Higher secondary skill levels should provide bigger bonuses
        higher_synergy = system.calculate_synergy_bonus(
            primary_skill=SkillType.COMBAT,
            secondary_skills={
                SkillType.ATHLETICS: SkillLevel.EXPERT,
                SkillType.REASONING: SkillLevel.EXPERT  
            }
        )
        
        assert higher_synergy > synergy_bonus
        
    def test_skill_prerequisites(self):
        """Test skill prerequisite system"""
        system = SkillDevelopmentSystem()
        
        # Some skills should require prerequisites
        prerequisites = system.get_prerequisites(SkillType.LEADERSHIP)
        assert SkillType.PERSUASION in prerequisites
        assert SkillType.EMPATHY in prerequisites
        
        # Check if prerequisites are met
        can_develop = system.can_develop_skill(
            target_skill=SkillType.LEADERSHIP,
            current_skills={
                SkillType.PERSUASION: SkillLevel.COMPETENT,
                SkillType.EMPATHY: SkillLevel.BEGINNER
            }
        )
        assert can_develop
        
    def test_skill_competition(self):
        """Test that some skills compete for development time"""
        system = SkillDevelopmentSystem()
        
        # Developing one skill should slightly reduce others in same time period
        initial_combat = SkillInstance(
            skill_type=SkillType.COMBAT,
            level=SkillLevel.COMPETENT,
            experience_points=400.0,
            practice_hours=50.0,
            success_rate=0.7,
            last_practiced=time.time() - 3600,  # 1 hour ago
            decay_rate=0.01,
            mastery_bonus=0.0,
            specialization_path=None
        )
        
        # Heavy practice in athletics should affect combat practice time
        system.practice_skill("test_agent", SkillType.ATHLETICS, hours=4.0)
        
        # Combat skill should show some decay from lack of recent practice
        updated_combat = system.get_skill("test_agent", SkillType.COMBAT)
        assert updated_combat.experience_points <= initial_combat.experience_points


class TestSkillDecayAndMaintenance:
    """Test skill decay and maintenance mechanisms"""
    
    def test_skill_decay_over_time(self):
        """Test that unused skills decay over time"""
        system = SkillDevelopmentSystem()
        
        # Create skill with old last_practiced time
        old_skill = SkillInstance(
            skill_type=SkillType.CRAFTING,
            level=SkillLevel.PROFICIENT,
            experience_points=800.0,
            practice_hours=80.0,
            success_rate=0.8,
            last_practiced=time.time() - 86400*30,  # 30 days ago
            decay_rate=0.02,
            mastery_bonus=0.0,
            specialization_path=None
        )
        
        # Apply time-based decay
        decayed_skill = system.apply_skill_decay(old_skill)
        
        assert decayed_skill.experience_points < old_skill.experience_points
        assert decayed_skill.success_rate < old_skill.success_rate
        
    def test_decay_resistance(self):
        """Test that specialized skills resist decay better"""
        system = SkillDevelopmentSystem()
        
        # Regular skill
        regular_decay = system.calculate_decay_amount(
            skill_level=SkillLevel.EXPERT,
            days_unused=30,
            base_decay_rate=0.02,
            specialization_path=None
        )
        
        # Specialized skill  
        specialized_decay = system.calculate_decay_amount(
            skill_level=SkillLevel.EXPERT,
            days_unused=30,
            base_decay_rate=0.02,
            specialization_path="weapon_master"
        )
        
        assert specialized_decay < regular_decay
        
    def test_skill_maintenance(self):
        """Test skill maintenance through minimal practice"""
        system = SkillDevelopmentSystem()
        
        # Light maintenance practice should prevent decay
        maintenance_exp = system.calculate_maintenance_experience(
            skill_type=SkillType.MEMORY,
            current_level=SkillLevel.PROFICIENT,
            practice_intensity=0.2,  # Light practice
            practice_duration=0.5   # 30 minutes
        )
        
        # Should be enough to offset decay
        decay_amount = system.calculate_decay_amount(
            skill_level=SkillLevel.PROFICIENT,
            days_unused=1,
            base_decay_rate=0.01,
            specialization_path=None
        )
        
        assert maintenance_exp >= decay_amount


class TestSkillTeaching:
    """Test skill teaching and knowledge transfer"""
    
    def test_teaching_mechanics(self):
        """Test that agents can teach skills to each other"""
        system = SkillDevelopmentSystem()
        
        # Teacher with high skill level
        teacher_skill = SkillInstance(
            skill_type=SkillType.CRAFTING,
            level=SkillLevel.EXPERT,
            experience_points=1200.0,
            practice_hours=120.0,
            success_rate=0.9,
            last_practiced=time.time(),
            decay_rate=0.01,
            mastery_bonus=0.1,
            specialization_path="master_craftsman"
        )
        
        # Student with lower skill level
        student_skill = SkillInstance(
            skill_type=SkillType.CRAFTING,
            level=SkillLevel.BEGINNER,
            experience_points=150.0,
            practice_hours=15.0,
            success_rate=0.5,
            last_practiced=time.time(),
            decay_rate=0.02,
            mastery_bonus=0.0,
            specialization_path=None
        )
        
        # Calculate teaching effectiveness
        teaching_bonus = system.calculate_teaching_bonus(
            teacher_skill=teacher_skill,
            student_skill=student_skill,
            teaching_duration=2.0  # 2 hours
        )
        
        assert teaching_bonus > 0
        
        # Teaching should accelerate student learning
        accelerated_exp = system.calculate_taught_experience(
            base_experience=10.0,
            teaching_bonus=teaching_bonus
        )
        
        assert accelerated_exp > 10.0
        
    def test_knowledge_transfer_efficiency(self):
        """Test knowledge transfer efficiency factors"""
        system = SkillDevelopmentSystem()
        
        # Similar skill levels should have poor teaching efficiency
        similar_efficiency = system.calculate_teaching_efficiency(
            teacher_level=SkillLevel.COMPETENT,
            student_level=SkillLevel.PROFICIENT  
        )
        
        # Large skill gap should have better efficiency
        gap_efficiency = system.calculate_teaching_efficiency(
            teacher_level=SkillLevel.MASTER,
            student_level=SkillLevel.NOVICE
        )
        
        assert gap_efficiency > similar_efficiency
        
    def test_teaching_skill_development(self):
        """Test that teaching also develops the teacher's skills"""
        system = SkillDevelopmentSystem()
        
        # Teaching should provide some experience to teacher
        teacher_exp = system.calculate_teacher_experience(
            skill_being_taught=SkillType.ANALYSIS,
            teacher_level=SkillLevel.EXPERT,
            teaching_duration=3.0,
            student_improvement=0.4  # How much student improved
        )
        
        assert teacher_exp > 0
        
        # Teaching should also develop leadership/empathy skills
        leadership_exp = system.calculate_secondary_teaching_experience(
            secondary_skill=SkillType.LEADERSHIP,
            teaching_session_quality=0.8
        )
        
        assert leadership_exp > 0


class TestSkillPerformance:
    """Test skill-based performance calculations"""
    
    def test_performance_calculation(self):
        """Test that skill level affects performance"""
        system = SkillDevelopmentSystem()
        
        # Create skills at different levels
        novice_skill = SkillInstance(
            skill_type=SkillType.PERSUASION,
            level=SkillLevel.NOVICE,
            experience_points=50.0,
            practice_hours=5.0,
            success_rate=0.4,
            last_practiced=time.time(),
            decay_rate=0.02,
            mastery_bonus=0.0,
            specialization_path=None
        )
        
        expert_skill = SkillInstance(
            skill_type=SkillType.PERSUASION,
            level=SkillLevel.EXPERT,
            experience_points=1200.0,
            practice_hours=120.0,
            success_rate=0.9,
            last_practiced=time.time(),
            decay_rate=0.01,
            mastery_bonus=0.15,
            specialization_path="diplomat"
        )
        
        # Calculate performance for same task
        task_difficulty = 0.6
        novice_performance = system.calculate_task_performance(novice_skill, task_difficulty)
        expert_performance = system.calculate_task_performance(expert_skill, task_difficulty)
        
        assert expert_performance > novice_performance
        
    def test_success_probability(self):
        """Test success probability calculation"""
        system = SkillDevelopmentSystem()
        
        skill = SkillInstance(
            skill_type=SkillType.ATHLETICS,
            level=SkillLevel.PROFICIENT,
            experience_points=800.0,
            practice_hours=80.0,
            success_rate=0.75,
            last_practiced=time.time(),
            decay_rate=0.01,
            mastery_bonus=0.05,
            specialization_path=None
        )
        
        # Easy task should have high success probability
        easy_success = system.calculate_success_probability(skill, difficulty=0.3)
        assert easy_success > 0.8
        
        # Hard task should have lower success probability  
        hard_success = system.calculate_success_probability(skill, difficulty=0.9)
        assert hard_success < easy_success
        assert 0.0 <= hard_success <= 1.0
        
    def test_contextual_modifiers(self):
        """Test contextual performance modifiers"""
        system = SkillDevelopmentSystem()
        
        base_performance = 0.7
        
        # Favorable conditions should boost performance
        boosted_performance = system.apply_contextual_modifiers(
            base_performance=base_performance,
            context={
                "equipment_quality": 0.9,
                "environmental_conditions": 0.8,
                "fatigue_level": 0.2,
                "stress_level": 0.3
            }
        )
        
        # Poor conditions should reduce performance
        reduced_performance = system.apply_contextual_modifiers(
            base_performance=base_performance,
            context={
                "equipment_quality": 0.3,
                "environmental_conditions": 0.4, 
                "fatigue_level": 0.8,
                "stress_level": 0.9
            }
        )
        
        assert boosted_performance > base_performance > reduced_performance


class TestSkillSystemIntegration:
    """Test integration with existing agent systems"""
    
    def test_agent_state_integration(self):
        """Test integration with agent state"""
        system = SkillDevelopmentSystem()
        
        # Should be able to initialize agent skills
        agent_id = "test_agent"
        initial_skills = system.initialize_agent_skills(
            agent_id=agent_id,
            personality_traits={"openness": 0.7, "conscientiousness": 0.8},
            starting_background="craftsman"
        )
        
        assert len(initial_skills) > 0
        assert SkillType.CRAFTING in [s.skill_type for s in initial_skills]
        
    def test_skill_execution_integration(self):
        """Test integration with skill execution module"""
        system = SkillDevelopmentSystem()
        
        # Should be able to execute skill-based actions
        skill_result = system.execute_skill_action(
            agent_id="test_agent",
            skill_type=SkillType.COMBAT,
            action="attack",
            target="enemy",
            context={"weapon": "sword", "range": "close"}
        )
        
        assert "success" in skill_result
        assert "experience_gained" in skill_result
        assert "performance_score" in skill_result
        
    def test_economic_system_integration(self):
        """Test integration with economic resource system"""
        system = SkillDevelopmentSystem()
        
        # Skills should affect resource production
        production_bonus = system.calculate_production_bonus(
            skill_type=SkillType.CRAFTING,
            skill_level=SkillLevel.EXPERT,
            resource_type="tools"  # From ResourceType enum
        )
        
        assert production_bonus > 0
        
        # Skills should affect resource consumption efficiency
        efficiency_bonus = system.calculate_efficiency_bonus(
            skill_type=SkillType.FORAGING,
            skill_level=SkillLevel.PROFICIENT,
            resource_type="food"
        )
        
        assert efficiency_bonus > 0


# Integration tests for system performance
class TestSkillSystemPerformance:
    """Test system performance and scalability"""
    
    def test_skill_calculation_performance(self):
        """Test that skill calculations are performant"""
        system = SkillDevelopmentSystem()
        
        start_time = time.time()
        
        # Perform many skill calculations
        for i in range(1000):
            skill = SkillInstance(
                skill_type=SkillType.COMBAT,
                level=SkillLevel.COMPETENT,
                experience_points=400 + i,
                practice_hours=40 + i/10,
                success_rate=0.6 + i/10000,
                last_practiced=time.time(),
                decay_rate=0.01,
                mastery_bonus=0.0,
                specialization_path=None
            )
            
            performance = system.calculate_task_performance(skill, 0.5)
            assert 0.0 <= performance <= 1.0
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0  # Should complete in under 1 second
        
    def test_multi_agent_skill_tracking(self):
        """Test tracking skills for multiple agents"""
        system = SkillDevelopmentSystem()
        
        # Add skills for multiple agents
        for agent_id in [f"agent_{i}" for i in range(100)]:
            system.add_agent(agent_id)
            
            for skill_type in list(SkillType)[:5]:  # First 5 skills
                system.add_skill(agent_id, skill_type, SkillLevel.NOVICE)
        
        # Verify all agents and skills are tracked
        assert len(system.get_all_agents()) == 100
        
        total_skills = sum(len(system.get_agent_skills(agent_id)) 
                          for agent_id in system.get_all_agents())
        assert total_skills == 500  # 100 agents * 5 skills each


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])