"""
Enhanced Agent class with integrated Skill Development System
Task 3.2: Experience-based skill growth algorithms

The main Agent class that integrates all PIANO components including
the comprehensive skill development system for experience-based learning.

Features:
- Integrated skill development system
- Experience-based learning from all agent actions
- Skill-aware decision making and execution
- Performance tracking and skill progression
- Agent lifecycle management with skill persistence
"""

import threading
import time
from typing import Optional, Dict, Any

from .agent_state import AgentState
from .cognitive_controller import CognitiveController
from .modules.social_awareness import SocialAwarenessModule
from .modules.goal_generation import GoalGenerationModule
from .modules.action_awareness import ActionAwarenessModule
from .modules.talking import TalkingModule
from .modules.skill_execution import SkillExecutionModule
from .modules.skill_development import SkillDevelopmentSystem

class Agent:
    """
    Enhanced Agent class with integrated skill development system.
    
    The agent now features comprehensive skill tracking, experience-based learning,
    and skill-aware decision making that evolves over time based on actions and outcomes.
    """
    
    def __init__(
        self, 
        name: str, 
        role: str, 
        personality_traits: Dict[str, float],
        skill_system: Optional[SkillDevelopmentSystem] = None,
        starting_background: Optional[str] = None
    ):
        self.agent_state = AgentState(name, role, personality_traits)
        self.cognitive_controller = CognitiveController(self.agent_state)
        
        # Initialize or use provided skill system
        if skill_system is None:
            self.skill_system = SkillDevelopmentSystem()
        else:
            self.skill_system = skill_system
        
        # Register agent with skill system and initialize skills
        self.skill_system.add_agent(name)
        initial_skills = self.skill_system.initialize_agent_skills(
            agent_id=name,
            personality_traits=personality_traits,
            starting_background=starting_background
        )
        
        # Store skill system reference in agent state for module access
        if not hasattr(self.agent_state, 'skill_system'):
            self.agent_state.skill_system = self.skill_system

        # Initialize all concurrent modules with skill system integration
        self.modules = [
            SocialAwarenessModule(self.agent_state),
            GoalGenerationModule(self.agent_state),
            ActionAwarenessModule(self.agent_state),
            TalkingModule(self.agent_state),
            SkillExecutionModule(self.agent_state, skill_system=self.skill_system)
        ]

        self.is_running = False
        self.threads = []
        
        # Skill development tracking
        self.skill_update_interval = 600.0  # Update skills every 10 minutes
        self.last_skill_update = time.time()
        
        # Performance metrics
        self.performance_history = []
        self.learning_sessions = 0

    def start(self):
        """
        Enhanced start method with skill development system integration.
        """
        self.is_running = True

        # Start the main cognitive controller loop
        controller_thread = threading.Thread(target=self._run_controller)
        self.threads.append(controller_thread)
        controller_thread.start()

        # Start all concurrent modules
        for module in self.modules:
            module_thread = threading.Thread(target=self._run_module, args=(module,))
            self.threads.append(module_thread)
            module_thread.start()
            
        # Start skill development update loop
        skill_thread = threading.Thread(target=self._run_skill_updates)
        self.threads.append(skill_thread)
        skill_thread.start()

    def stop(self):
        """
        Enhanced stop method with skill system cleanup.
        """
        self.is_running = False
        for thread in self.threads:
            thread.join()
        
        # Final skill update before shutdown
        self._update_skills()

    def _run_controller(self):
        """
        Enhanced cognitive controller loop with skill-aware decision making.
        """
        while self.is_running:
            # Update agent skills periodically
            if time.time() - self.last_skill_update > self.skill_update_interval:
                self._update_skills()
            
            # Make decision with skill context
            self.cognitive_controller.make_decision()
            
            # Track performance
            self._track_performance()
            
            time.sleep(1) # The controller runs at a slower, more deliberate pace

    def _run_module(self, module):
        """
        The main loop for a concurrent module.
        """
        while self.is_running:
            module.run()
            time.sleep(0.5) # Modules run more frequently than the controller
    
    def _run_skill_updates(self):
        """
        Dedicated loop for skill development updates.
        """
        while self.is_running:
            # Update all agent skills (decay, etc.)
            self.skill_system.update_all_skills()
            
            # Update every 5 minutes
            time.sleep(300)
    
    def _update_skills(self):
        """
        Update agent skills and track development.
        """
        try:
            # Apply skill decay and maintenance
            self.skill_system.update_all_skills()
            
            # Track skill development metrics
            skill_summary = self.skill_system.get_skill_summary(self.agent_state.name)
            
            # Store skill information in agent state for other modules
            if skill_summary:
                self.agent_state.proprioception["skill_summary"] = skill_summary
                self.agent_state.proprioception["total_skills"] = skill_summary.get("total_skills", 0)
                self.agent_state.proprioception["highest_skills"] = skill_summary.get("highest_skills", [])
            
            self.last_skill_update = time.time()
            
        except Exception as e:
            print(f"Error updating skills for {self.agent_state.name}: {e}")
    
    def _track_performance(self):
        """
        Track agent performance and learning progress.
        """
        try:
            # Get recent action performance
            last_performance = self.agent_state.proprioception.get("last_action_performance", 0.5)
            
            # Add to performance history
            self.performance_history.append({
                "timestamp": time.time(),
                "performance": last_performance
            })
            
            # Keep only recent history (last 100 actions)
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-50:]
            
            # Calculate running average performance
            if self.performance_history:
                avg_performance = sum(p["performance"] for p in self.performance_history[-20:]) / min(20, len(self.performance_history))
                self.agent_state.proprioception["average_performance"] = avg_performance
                
        except Exception as e:
            print(f"Error tracking performance for {self.agent_state.name}: {e}")
    
    def practice_skill(self, skill_name: str, hours: float = 1.0, focus_level: float = 0.7) -> Dict[str, Any]:
        """
        Have the agent practice a specific skill.
        
        Args:
            skill_name: Name of the skill to practice
            hours: How many hours to practice
            focus_level: Focus level during practice (0.0-1.0)
            
        Returns:
            Dictionary with practice results
        """
        try:
            from .modules.skill_development import SkillType
            
            # Convert skill name to SkillType enum
            skill_type = None
            for st in SkillType:
                if st.value.lower() == skill_name.lower():
                    skill_type = st
                    break
            
            if not skill_type:
                return {"error": f"Unknown skill: {skill_name}"}
            
            # Practice the skill
            success = self.skill_system.practice_skill(
                agent_id=self.agent_state.name,
                skill_type=skill_type,
                hours=hours,
                focus_level=focus_level
            )
            
            if success:
                self.learning_sessions += 1
                skill_instance = self.skill_system.get_skill(self.agent_state.name, skill_type)
                
                return {
                    "success": True,
                    "skill_practiced": skill_name,
                    "hours": hours,
                    "focus_level": focus_level,
                    "current_level": skill_instance.level.name if skill_instance else "unknown",
                    "current_experience": skill_instance.experience_points if skill_instance else 0,
                    "learning_sessions": self.learning_sessions
                }
            else:
                return {"error": "Failed to practice skill"}
                
        except Exception as e:
            return {"error": f"Error practicing skill: {e}"}
    
    def get_skill_status(self) -> Dict[str, Any]:
        """
        Get current skill status for the agent.
        
        Returns:
            Dictionary with comprehensive skill information
        """
        try:
            # Get skill summary from skill system
            skill_summary = self.skill_system.get_skill_summary(self.agent_state.name)
            
            # Get individual skills
            agent_skills = self.skill_system.get_agent_skills(self.agent_state.name)
            skill_details = {}
            
            for skill_type, skill_instance in agent_skills.items():
                skill_details[skill_type.value] = {
                    "level": skill_instance.level.name,
                    "experience": skill_instance.experience_points,
                    "practice_hours": skill_instance.practice_hours,
                    "success_rate": skill_instance.success_rate,
                    "specialization": skill_instance.specialization_path,
                    "last_practiced": skill_instance.last_practiced
                }
            
            # Get performance summary from skill execution module
            skill_execution_module = next((m for m in self.modules if isinstance(m, SkillExecutionModule)), None)
            performance_summary = {}
            if skill_execution_module:
                performance_summary = skill_execution_module.get_skill_performance_summary()
            
            return {
                "agent_name": self.agent_state.name,
                "skill_summary": skill_summary,
                "skill_details": skill_details,
                "performance_summary": performance_summary,
                "learning_sessions": self.learning_sessions,
                "average_recent_performance": self.agent_state.proprioception.get("average_performance", 0.5)
            }
            
        except Exception as e:
            return {"error": f"Error getting skill status: {e}"}
    
    def teach_skill_to(self, other_agent: 'Agent', skill_name: str, hours: float = 2.0) -> Dict[str, Any]:
        """
        Teach a skill to another agent.
        
        Args:
            other_agent: The agent to teach
            skill_name: Name of the skill to teach
            hours: Duration of teaching session
            
        Returns:
            Dictionary with teaching results
        """
        try:
            from .modules.skill_development import SkillType, LearningEvent, LearningSourceType
            
            # Convert skill name to SkillType enum
            skill_type = None
            for st in SkillType:
                if st.value.lower() == skill_name.lower():
                    skill_type = st
                    break
            
            if not skill_type:
                return {"error": f"Unknown skill: {skill_name}"}
            
            # Get teacher and student skills
            teacher_skill = self.skill_system.get_skill(self.agent_state.name, skill_type)
            student_skill = other_agent.skill_system.get_skill(other_agent.agent_state.name, skill_type)
            
            if not teacher_skill:
                return {"error": f"Teacher doesn't have skill: {skill_name}"}
            
            # Initialize student skill if they don't have it
            if not student_skill:
                other_agent.skill_system.add_skill(other_agent.agent_state.name, skill_type, skill_type.level.NOVICE)
                student_skill = other_agent.skill_system.get_skill(other_agent.agent_state.name, skill_type)
            
            # Calculate teaching effectiveness
            teaching_bonus = self.skill_system.calculate_teaching_bonus(
                teacher_skill=teacher_skill,
                student_skill=student_skill,
                teaching_duration=hours
            )
            
            # Calculate base experience for student
            base_experience = self.skill_system.calculate_experience_gain(
                skill_type=skill_type,
                source=LearningSourceType.TEACHING,
                performance=0.7,  # Assume good teaching performance
                difficulty=0.5,   # Moderate difficulty
                current_level=student_skill.level,
                duration=hours
            )
            
            # Apply teaching bonus
            student_experience = self.skill_system.calculate_taught_experience(base_experience, teaching_bonus)
            
            # Calculate teacher experience
            teacher_experience = self.skill_system.calculate_teacher_experience(
                skill_being_taught=skill_type,
                teacher_level=teacher_skill.level,
                teaching_duration=hours,
                student_improvement=teaching_bonus
            )
            
            # Create learning events
            student_event = LearningEvent(
                agent_id=other_agent.agent_state.name,
                skill_type=skill_type,
                event_type="learning_from_teacher",
                experience_gained=student_experience,
                performance_score=0.7 + teaching_bonus * 0.2,
                difficulty_level=0.5,
                context={"teacher": self.agent_state.name, "teaching_duration": hours},
                timestamp=time.time(),
                duration=hours,
                teacher_id=self.agent_state.name
            )
            
            teacher_event = LearningEvent(
                agent_id=self.agent_state.name,
                skill_type=skill_type,
                event_type="teaching",
                experience_gained=teacher_experience,
                performance_score=0.6 + teaching_bonus * 0.3,
                difficulty_level=0.4,
                context={"student": other_agent.agent_state.name, "teaching_duration": hours},
                timestamp=time.time(),
                duration=hours
            )
            
            # Process learning events
            student_success = other_agent.skill_system.process_learning_event(student_event)
            teacher_success = self.skill_system.process_learning_event(teacher_event)
            
            # Update teaching records
            teacher_skill.students.add(other_agent.agent_state.name)
            student_skill.teachers.add(self.agent_state.name)
            
            return {
                "success": student_success and teacher_success,
                "teacher": self.agent_state.name,
                "student": other_agent.agent_state.name,
                "skill": skill_name,
                "teaching_hours": hours,
                "teaching_bonus": teaching_bonus,
                "student_experience_gained": student_experience,
                "teacher_experience_gained": teacher_experience,
                "student_new_level": other_agent.skill_system.get_skill(other_agent.agent_state.name, skill_type).level.name,
                "teacher_new_level": self.skill_system.get_skill(self.agent_state.name, skill_type).level.name
            }
            
        except Exception as e:
            return {"error": f"Error during teaching: {e}"}
    
    def get_performance_history(self, limit: int = 20) -> list:
        """
        Get recent performance history.
        
        Args:
            limit: Maximum number of performance records to return
            
        Returns:
            List of recent performance records
        """
        return self.performance_history[-limit:] if self.performance_history else []
    
    def __del__(self):
        """
        Cleanup method to ensure proper shutdown.
        """
        if hasattr(self, 'is_running') and self.is_running:
            self.stop()
