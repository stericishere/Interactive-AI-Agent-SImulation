"""
File: agent.py
Description: The main Agent class that integrates all PIANO components.
"""

import threading
import time

from .agent_state import AgentState
from .cognitive_controller import CognitiveController
from .modules.social_awareness import SocialAwarenessModule
from .modules.goal_generation import GoalGenerationModule
from .modules.action_awareness import ActionAwarenessModule
from .modules.talking import TalkingModule
from .modules.skill_execution import SkillExecutionModule

class Agent:
    def __init__(self, name, role, personality_traits):
        self.agent_state = AgentState(name, role, personality_traits)
        self.cognitive_controller = CognitiveController(self.agent_state)

        # Initialize all concurrent modules
        self.modules = [
            SocialAwarenessModule(self.agent_state),
            GoalGenerationModule(self.agent_state),
            ActionAwarenessModule(self.agent_state),
            TalkingModule(self.agent_state),
            SkillExecutionModule(self.agent_state)
        ]

        self.is_running = False
        self.threads = []

    def start(self):
        """
        Starts the agent's cognitive loop and all concurrent modules.
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

    def stop(self):
        """
        Stops the agent and all its threads.
        """
        self.is_running = False
        for thread in self.threads:
            thread.join()

    def _run_controller(self):
        """
        The main loop for the Cognitive Controller.
        """
        while self.is_running:
            self.cognitive_controller.make_decision()
            time.sleep(1) # The controller runs at a slower, more deliberate pace

    def _run_module(self, module):
        """
        The main loop for a concurrent module.
        """
        while self.is_running:
            module.run()
            time.sleep(0.5) # Modules run more frequently than the controller
