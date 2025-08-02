"""
File: agent_state.py
Description: Defines the AgentState class, which holds all data for a PIANO-based agent.
"""

from .memory_structures.associative_memory import AssociativeMemory
from .memory_structures.scratch import Scratch
from .memory_structures.spatial_memory import MemoryTree

class AgentState:
    def __init__(self, name, role, personality_traits, initial_emotional_state=None):
        # ==== CORE IDENTITY ====
        self.name = name
        self.role = role  # "contestant", "host", etc.
        self.traits = personality_traits  # Persistent personality characteristics

        # ==== MEMORY ====
        # Raw, unprocessed memories of events, conversations, etc.
        self.memory = {
            "working_memory": [],
            "short_term_memory": [],
            "long_term_memory": []
        }
        # The original memory structures can be adapted or used as part of the new memory system.
        self.s_mem = MemoryTree(None)
        self.a_mem = AssociativeMemory(None)
        self.scratch = Scratch(None)

        # ==== GOALS ====
        # The agent's current objectives.
        self.goals = []

        # ==== ENVIRONMENT ====
        # Information about the agent's surroundings.
        self.environment_detail = {}

        # ==== SOCIAL ====
        # Information about other agents and social relationships.
        self.social = {
            "relationships": {},  # e.g., {"agent_name": {"attraction": 0.8, "trust": 0.5}}
            "conversation_history": {}
        }

        # ==== PROPRIOCEPTION ====
        # The agent's own state.
        self.proprioception = {
            "emotional_state": initial_emotional_state if initial_emotional_state else {"happiness": 0.5, "jealousy": 0.1},
            "physical_state": {} # e.g., location, fatigue
        }

    def get_summary_for_controller(self):
        """
        Generates a synthesized summary of the agent's state for the Cognitive Controller.
        This is the "bottleneck" in the PIANO architecture.
        """
        # This method will be crucial for controlling the agent's behavior.
        # For now, it returns a basic summary.
        return {
            "name": self.name,
            "role": self.role,
            "goals": self.goals,
            "emotional_state": self.proprioception["emotional_state"],
            "recent_events": self.memory["working_memory"][-5:], # Last 5 events
            "relationships": self.social["relationships"]
        }

    




































