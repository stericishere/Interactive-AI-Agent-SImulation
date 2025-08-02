"""
File: base_module.py
Description: Defines the BaseModule class, the parent class for all concurrent modules in the PIANO architecture.
"""

class BaseModule:
    def __init__(self, agent_state):
        self.agent_state = agent_state

    def run(self):
        """
        The main execution method for the module. This will be called in a separate thread or process.
        This method should read from and write to the agent_state.
        """
        raise NotImplementedError("Each module must implement its own run method.")
