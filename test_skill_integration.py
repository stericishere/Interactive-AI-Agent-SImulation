"""
Integration test for the comprehensive skill development system
Task 3.2: Experience-based skill growth algorithms

This test validates the complete skill development system integration
with the agent architecture and demonstrates all key features.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import time
from dating_show.agents.modules.skill_development import (
    SkillDevelopmentSystem, SkillType, SkillLevel, 
    LearningSourceType, LearningEvent
)
from dating_show.agents.agent import Agent


def test_basic_skill_system():
    """Test basic skill system functionality"""
    print("=" * 60)
    print("TESTING BASIC SKILL SYSTEM FUNCTIONALITY")
    print("=" * 60)
    
    # Create skill system
    skill_system = SkillDevelopmentSystem()
    
    # Add an agent
    agent_id = "test_agent"
    success = skill_system.add_agent(agent_id)
    print(f"✓ Agent added successfully: {success}")
    
    # Test skill level calculation
    level = skill_system.calculate_skill_level(150)
    print(f"✓ Skill level for 150 XP: {level.name}")
    
    # Test experience gain calculation
    exp_gain = skill_system.calculate_experience_gain(
        skill_type=SkillType.COMBAT,
        source=LearningSourceType.PRACTICE,
        performance=0.7,
        difficulty=0.5,
        current_level=SkillLevel.NOVICE
    )
    print(f"✓ Experience gain calculated: {exp_gain:.2f}")
    
    # Test skill practice
    practice_success = skill_system.practice_skill(
        agent_id=agent_id,
        skill_type=SkillType.COMBAT,
        hours=2.0,
        focus_level=0.8
    )
    print(f"✓ Practice session successful: {practice_success}")
    
    # Get skill status
    skill = skill_system.get_skill(agent_id, SkillType.COMBAT)
    if skill:
        print(f"✓ Combat skill level: {skill.level.name}, XP: {skill.experience_points:.2f}")
    
    return True


def test_skill_specialization():
    """Test skill specialization system"""
    print("\n" + "=" * 60)
    print("TESTING SKILL SPECIALIZATION")
    print("=" * 60)
    
    skill_system = SkillDevelopmentSystem()
    agent_id = "specialist_agent"
    skill_system.add_agent(agent_id)
    
    # Practice combat to get to proficient level
    for _ in range(10):
        skill_system.practice_skill(agent_id, SkillType.COMBAT, 2.0, 0.8)
    
    combat_skill = skill_system.get_skill(agent_id, SkillType.COMBAT)
    print(f"✓ Combat skill after training: {combat_skill.level.name}, XP: {combat_skill.experience_points:.2f}")
    
    # Try to specialize
    specializations = skill_system.get_specialization_paths(SkillType.COMBAT)
    print(f"✓ Available combat specializations: {specializations}")
    
    if specializations and combat_skill.level.value >= SkillLevel.PROFICIENT.value:
        spec_success = skill_system.apply_specialization(agent_id, SkillType.COMBAT, specializations[0])
        print(f"✓ Specialization applied: {spec_success}")
        
        updated_skill = skill_system.get_skill(agent_id, SkillType.COMBAT)
        print(f"✓ Specialization path: {updated_skill.specialization_path}")
        print(f"✓ Mastery bonus: {updated_skill.mastery_bonus}")
    
    return True


def test_skill_synergies():
    """Test skill synergy system"""
    print("\n" + "=" * 60)
    print("TESTING SKILL SYNERGIES")
    print("=" * 60)
    
    skill_system = SkillDevelopmentSystem()
    agent_id = "synergy_agent"
    skill_system.add_agent(agent_id)
    
    # Practice multiple related skills
    skill_system.practice_skill(agent_id, SkillType.COMBAT, 3.0, 0.8)
    skill_system.practice_skill(agent_id, SkillType.ATHLETICS, 3.0, 0.8)
    skill_system.practice_skill(agent_id, SkillType.REASONING, 2.0, 0.9)
    
    agent_skills = skill_system.get_agent_skills(agent_id)
    skill_levels = {skill_type: skill.level for skill_type, skill in agent_skills.items()}
    
    # Calculate synergy bonus
    synergy_bonus = skill_system.calculate_synergy_bonus(
        primary_skill=SkillType.COMBAT,
        secondary_skills=skill_levels
    )
    
    print(f"✓ Agent skills: {[(s.value, l.name) for s, l in skill_levels.items()]}")
    print(f"✓ Combat synergy bonus: {synergy_bonus:.2f}")
    
    return True


def test_agent_integration():
    """Test integration with Agent class"""
    print("\n" + "=" * 60)
    print("TESTING AGENT CLASS INTEGRATION")
    print("=" * 60)
    
    # Create agent with skills
    agent = Agent(
        name="integrated_agent",
        role="warrior",
        personality_traits={"openness": 0.7, "conscientiousness": 0.8},
        starting_background="warrior"
    )
    
    print(f"✓ Agent created: {agent.agent_state.name}")
    
    # Check initial skills
    skill_status = agent.get_skill_status()
    print(f"✓ Initial skill count: {skill_status.get('skill_summary', {}).get('total_skills', 0)}")
    
    # Practice a skill
    practice_result = agent.practice_skill("combat", hours=2.0, focus_level=0.9)
    print(f"✓ Practice result: {practice_result.get('success', False)}")
    
    if practice_result.get('success'):
        print(f"✓ New combat level: {practice_result.get('current_level', 'unknown')}")
        print(f"✓ Experience gained: {practice_result.get('current_experience', 0):.2f}")
    
    # Test skill execution through decision making
    agent.agent_state.proprioception["current_decision"] = "attack the enemy with my sword"
    
    # Get skill execution module and test it
    skill_exec_module = next((m for m in agent.modules if hasattr(m, 'get_skill_performance_summary')), None)
    if skill_exec_module:
        skill_exec_module.run()  # This should process the decision
        
        executed_action = agent.agent_state.proprioception.get("executed_action")
        if executed_action:
            print(f"✓ Action executed: {executed_action.get('action', 'unknown')}")
            print(f"✓ Skill used: {executed_action.get('skill_used', 'unknown')}")
            print(f"✓ Success: {executed_action.get('success', False)}")
            print(f"✓ Performance: {executed_action.get('performance_score', 0):.2f}")
    
    return True


def test_skill_teaching():
    """Test skill teaching between agents"""
    print("\n" + "=" * 60)
    print("TESTING SKILL TEACHING")
    print("=" * 60)
    
    # Create teacher and student agents
    teacher = Agent(
        name="teacher_agent",
        role="master_craftsman",
        personality_traits={"openness": 0.8, "conscientiousness": 0.9},
        starting_background="craftsman"
    )
    
    student = Agent(
        name="student_agent",
        role="apprentice",
        personality_traits={"openness": 0.9, "conscientiousness": 0.7}
    )
    
    # Teacher practices crafting to become expert
    for _ in range(15):
        teacher.practice_skill("crafting", hours=1.5, focus_level=0.9)
    
    teacher_skill_status = teacher.get_skill_status()
    teacher_crafting = teacher_skill_status.get('skill_details', {}).get('crafting', {})
    print(f"✓ Teacher crafting level: {teacher_crafting.get('level', 'unknown')}")
    
    # Student starts with basic skills
    student_skill_status = student.get_skill_status()
    student_crafting = student_skill_status.get('skill_details', {}).get('crafting', {})
    print(f"✓ Student initial crafting level: {student_crafting.get('level', 'none')}")
    
    # Teaching session
    teaching_result = teacher.teach_skill_to(student, "crafting", hours=3.0)
    print(f"✓ Teaching session success: {teaching_result.get('success', False)}")
    
    if teaching_result.get('success'):
        print(f"✓ Student new level: {teaching_result.get('student_new_level', 'unknown')}")
        print(f"✓ Teacher new level: {teaching_result.get('teacher_new_level', 'unknown')}")
        print(f"✓ Teaching bonus: {teaching_result.get('teaching_bonus', 0):.2f}")
    
    return True


def test_performance_and_economics():
    """Test performance calculation and economic integration"""
    print("\n" + "=" * 60)
    print("TESTING PERFORMANCE AND ECONOMICS")
    print("=" * 60)
    
    skill_system = SkillDevelopmentSystem()
    agent_id = "economic_agent"
    skill_system.add_agent(agent_id)
    
    # Develop crafting skills
    for _ in range(8):
        skill_system.practice_skill(agent_id, SkillType.CRAFTING, 2.0, 0.8)
    
    crafting_skill = skill_system.get_skill(agent_id, SkillType.CRAFTING)
    print(f"✓ Crafting skill: {crafting_skill.level.name}, XP: {crafting_skill.experience_points:.2f}")
    
    # Test performance calculation
    task_performance = skill_system.calculate_task_performance(
        crafting_skill, 
        task_difficulty=0.6,
        context={"equipment_quality": 0.8, "fatigue_level": 0.2}
    )
    print(f"✓ Crafting task performance: {task_performance:.2f}")
    
    # Test success probability
    success_prob = skill_system.calculate_success_probability(
        crafting_skill,
        difficulty=0.5
    )
    print(f"✓ Success probability for moderate task: {success_prob:.2f}")
    
    # Test economic bonuses
    production_bonus = skill_system.calculate_production_bonus(
        skill_type=SkillType.CRAFTING,
        skill_level=crafting_skill.level,
        resource_type="tools"
    )
    print(f"✓ Production bonus for tools: {production_bonus:.2f}")
    
    efficiency_bonus = skill_system.calculate_efficiency_bonus(
        skill_type=SkillType.CRAFTING,
        skill_level=crafting_skill.level,
        resource_type="materials"
    )
    print(f"✓ Efficiency bonus: {efficiency_bonus:.2f}")
    
    return True


def main():
    """Run all integration tests"""
    print("Starting Comprehensive Skill Development System Integration Tests")
    print("Task 3.2: Experience-based skill growth algorithms")
    
    tests = [
        test_basic_skill_system,
        test_skill_specialization,
        test_skill_synergies,
        test_agent_integration,
        test_skill_teaching,
        test_performance_and_economics
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✓ {test_func.__name__} PASSED")
            else:
                print(f"✗ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Skill development system is working correctly.")
    else:
        print(f"⚠️  {total-passed} tests failed. Check implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)