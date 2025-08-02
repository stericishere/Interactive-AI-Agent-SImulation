
"""
File: simulation.py
Description: The main entry point for running the dating show simulation.
"""

import time

from .agents.agent import Agent
from .game_master.game_master import GameMaster

def run_simulation():
    """
    Initializes and runs the dating show simulation.
    """
    # 1. Create Agents
    agents = [
        Agent(name="Alice", role="contestant", personality_traits={"openness": 0.9, "conscientiousness": 0.4}),
        Agent(name="Bob", role="contestant", personality_traits={"openness": 0.6, "conscientiousness": 0.8}),
        Agent(name="Charlie", role="host", personality_traits={"openness": 0.8, "conscientiousness": 0.9})
    ]

    # 2. Create Game Master
    game_master = GameMaster(agents)

    # 3. Start all agents
    for agent in agents:
        agent.start()

    # 4. Main simulation loop
    try:
        for i in range(10): # Run for 10 steps for demonstration
            print(f"\n--- Simulation Step {i+1} ---")
            game_master.update_state()
            print(f"Game State: {game_master.get_game_state()}")

            # In a real simulation, the environment would be updated here.
            # For now, we'll just print the agents' decisions.
            for agent in agents:
                print(f"  {agent.agent_state.name}'s decision: {agent.agent_state.proprioception.get('current_decision')}")
            
            time.sleep(1)

    finally:
        # 5. Stop all agents
        for agent in agents:
            agent.stop()

if __name__ == "__main__":
    run_simulation()

