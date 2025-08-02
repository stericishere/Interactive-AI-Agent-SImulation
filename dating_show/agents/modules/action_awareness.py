"""
File: action_awareness.py
Description: Defines the ActionAwarenessModule for grounding the agent's actions.
"""

from .base_module import BaseModule

class ActionAwarenessModule(BaseModule):
    def __init__(self, agent_state):
        super().__init__(agent_state)

    def run(self):
        """
        Compares the expected outcome of an action with the observed outcome.
        """
        # Placeholder logic:
        # This module would compare the agent's intended action with the actual result from the environment.
        # For example, if the agent intended to talk to someone, but they walked away, this module would notice.

        # This is a simplified example.
        intended_action = self.agent_state.proprioception.get("intended_action")
        observed_result = self.agent_state.environment_detail.get("last_action_result")

        if intended_action and observed_result and intended_action != observed_result:
            # The outcome was not what was expected. This could trigger a re-evaluation of the plan.
            self.agent_state.memory["working_memory"].append(f"Action failed: intended {intended_action}, but got {observed_result}")
            # print(f"{self.agent_state.name}: Action awareness triggered.")
