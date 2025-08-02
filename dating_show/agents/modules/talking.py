"""
File: talking.py
Description: Defines the TalkingModule for handling agent communication.
"""

from .base_module import BaseModule

class TalkingModule(BaseModule):
    def __init__(self, agent_state):
        super().__init__(agent_state)

    def run(self):
        """
        Manages the interpretation and generation of speech, guided by the Cognitive Controller.
        """
        # Placeholder logic:
        # This module is responsible for both understanding incoming chat and generating outgoing chat.

        # 1. Interpret incoming messages
        incoming_messages = self.agent_state.environment_detail.get("incoming_chat", [])
        for sender, message in incoming_messages:
            self._interpret_message(sender, message)

        # 2. Generate outgoing messages based on the controller's decision
        decision = self.agent_state.proprioception.get("current_decision")
        if decision and "talk to" in decision:
            target_agent = decision.split("talk to ")[-1]
            self._generate_dialogue(target_agent)

    def _interpret_message(self, sender, message):
        """
        Processes a single incoming message.
        """
        # In a full implementation, this would involve sentiment analysis, topic extraction, etc.
        # For now, just log the conversation.
        if sender not in self.agent_state.social["conversation_history"]:
            self.agent_state.social["conversation_history"][sender] = []
        self.agent_state.social["conversation_history"][sender].append(message)
        self.agent_state.memory["working_memory"].append(f"Heard from {sender}: {message}")

    def _generate_dialogue(self, target_agent):
        """
        Generates dialogue to send to another agent.
        """
        # This would use an LLM, conditioned by the agent's state and goals.
        # Simplified example:
        greeting = f"Hello, {target_agent}. It's a beautiful day, isn't it?"
        
        # The generated message would be sent to the environment/event bus.
        self.agent_state.proprioception["outgoing_chat"] = (target_agent, greeting)
        # print(f"{self.agent_state.name} says to {target_agent}: {greeting}")
