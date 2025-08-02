"""
File: game_master.py
Description: Defines the GameMaster class, which manages the state and flow of the dating show.
"""

from enum import Enum

class GameState(Enum):
    SHOW_START = 1
    ROUND_START = 2
    DATE = 3
    ELIMINATION = 4
    SHOW_END = 5

class GameMaster:
    def __init__(self, agents):
        self.agents = agents
        self.current_state = GameState.SHOW_START
        self.current_day = 1
        self.current_round = 1
        self.contestants = [agent for agent in agents if agent.role == "contestant"]
        self.host = next((agent for agent in agents if agent.role == "host"), None)
        self.eliminated_contestants = []

    def update_state(self):
        """
        Advances the game state and triggers corresponding events.
        This will be the main loop of the Game Master.
        """
        if self.current_state == GameState.SHOW_START:
            # Initial setup, introductions, etc.
            self.current_state = GameState.ROUND_START
        elif self.current_state == GameState.ROUND_START:
            # Start a new round, announce challenges, etc.
            self.current_state = GameState.DATE
        elif self.current_state == GameState.DATE:
            # Facilitate dates between contestants.
            self.current_state = GameState.ELIMINATION
        elif self.current_state == GameState.ELIMINATION:
            # Host the elimination ceremony.
            self.current_round += 1
            self.current_state = GameState.ROUND_START
        
        if not self.contestants:
            self.current_state = GameState.SHOW_END

    def get_game_state(self):
        """
        Returns the current state of the game.
        """
        return {
            "state": self.current_state.name,
            "day": self.current_day,
            "round": self.current_round,
            "contestants": [agent.name for agent in self.contestants],
            "host": self.host.name if self.host else None,
            "eliminated": [agent.name for agent in self.eliminated_contestants]
        }
