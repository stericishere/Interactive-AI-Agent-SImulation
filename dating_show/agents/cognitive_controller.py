"""
File: cognitive_controller.py
Description: Defines the CognitiveController, the central decision-making module for a PIANO-based agent.
"""

class CognitiveController:
    def __init__(self, agent_state):
        self.agent_state = agent_state

    def make_decision(self):
        """
        Makes a high-level decision based on a synthesized summary of the agent's state.
        This is the core of the agent's deliberate, coherent behavior.
        """
        # 1. Get the summarized state from the bottleneck.
        state_summary = self.agent_state.get_summary_for_controller()

        # 2. Process the summary to determine a high-level goal.
        #    This is where a large language model (LLM) would be used to reason about the state.
        #    For now, we'll use a simple rule-based approach for demonstration.
        decision = self._reason(state_summary)

        # 3. Broadcast the decision to the agent's state or a shared space
        #    so that concurrent modules can act on it.
        self.agent_state.proprioception['current_decision'] = decision
        return decision

    def _reason(self, state_summary):
        """
        A simplified reasoning process. In a full implementation, this would involve an LLM call.
        """
        # Example logic: If feeling low on happiness, try to socialize.
        if state_summary['emotional_state'].get('happiness', 0.5) < 0.4:
            return "find someone to talk to"

        # Example logic: If there are recent events, reflect on them.
        if state_summary['recent_events']:
            return "reflect on recent events"

        # Default action
        return "observe the environment"
