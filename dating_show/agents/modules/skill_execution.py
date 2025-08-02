"""
File: skill_execution.py
Description: Defines the SkillExecutionModule for performing low-level actions.
"""

from .base_module import BaseModule

class SkillExecutionModule(BaseModule):
    def __init__(self, agent_state):
        super().__init__(agent_state)

    def run(self):
        """
        Translates the Cognitive Controller's decision into a specific action.
        """
        decision = self.agent_state.proprioception.get("current_decision")
        if not decision:
            return

        # This module executes non-verbal actions.
        # In a real game, this would involve interacting with the game engine.

        action = None
        if "observe the environment" in decision:
            action = {"skill": "observe", "target": "environment"}
        elif "form a connection with" in decision:
            target_agent = decision.split("form a connection with ")[-1]
            action = {"skill": "move_to", "target": target_agent}
        
        if action:
            # The action would be sent to the environment/event bus to be executed.
            self.agent_state.proprioception["executed_action"] = action
            # print(f"{self.agent_state.name} executes skill: {action}")
