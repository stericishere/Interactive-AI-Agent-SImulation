#!/usr/bin/env python3
"""
Simple PIANO Dating Show Example
A standalone demonstration of the dating show with basic agent functionality
"""

import time
import random
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class SimpleAgentState:
    """Simplified agent state for dating show demo"""
    name: str
    role: str  # "contestant", "host"
    personality: Dict[str, float]
    emotional_state: Dict[str, float] = field(default_factory=lambda: {"happiness": 0.5, "attraction": 0.0, "confidence": 0.5})
    relationships: Dict[str, float] = field(default_factory=dict)
    goals: List[str] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)
    current_action: str = "waiting"
    location: str = "villa"

class SimplePIANOAgent:
    """Simplified PIANO agent for dating show"""
    
    def __init__(self, name: str, role: str, personality: Dict[str, float]):
        self.state = SimpleAgentState(name, role, personality)
        self.decision_count = 0
        
        # Initialize based on role
        if role == "contestant":
            self.state.goals = ["find_love", "win_competition", "make_connections"]
        elif role == "host":
            self.state.goals = ["manage_show", "create_drama", "facilitate_connections"]
    
    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Perception module - understand current environment"""
        perceptions = {
            "current_time": environment.get("time", "morning"),
            "other_agents": environment.get("agents", []),
            "current_event": environment.get("event", "free_time"),
            "location": environment.get("location", "villa")
        }
        
        # Update agent's understanding
        self.state.location = perceptions["location"]
        
        return perceptions
    
    def plan(self, perceptions: Dict[str, Any]) -> str:
        """Planning module - decide what to do next"""
        current_event = perceptions["current_event"]
        
        if self.state.role == "contestant":
            if current_event == "rose_ceremony":
                return "express_feelings"
            elif current_event == "group_date":
                return "compete_for_attention"
            elif current_event == "one_on_one":
                return "build_connection"
            else:
                return random.choice(["socialize", "reflect", "strategize"])
        
        elif self.state.role == "host":
            if current_event == "rose_ceremony":
                return "facilitate_ceremony"
            elif current_event == "date_card":
                return "deliver_date_card"
            else:
                return "observe_dynamics"
        
        return "wait"
    
    def act(self, plan: str, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Action execution module"""
        self.state.current_action = plan
        self.decision_count += 1
        
        # Generate action results based on personality and plan
        action_result = {
            "agent": self.state.name,
            "action": plan,
            "success": random.random() < (self.state.personality.get("confidence", 0.5) + 0.3),
            "message": self._generate_message(plan, environment)
        }
        
        # Update emotional state based on action
        self._update_emotions(plan, action_result["success"])
        
        # Add to memory
        memory_entry = f"Step {self.decision_count}: {plan} - {'Success' if action_result['success'] else 'Failed'}"
        self.state.memory.append(memory_entry)
        if len(self.state.memory) > 10:  # Keep recent memories
            self.state.memory = self.state.memory[-10:]
        
        return action_result
    
    def reflect(self, recent_events: List[Dict[str, Any]]) -> None:
        """Reflection module - learn from experiences"""
        # Simple reflection: adjust confidence based on recent success
        recent_successes = [event.get("success", False) for event in recent_events[-3:]]
        if recent_successes:
            success_rate = sum(recent_successes) / len(recent_successes)
            # Gradually adjust confidence
            current_confidence = self.state.emotional_state["confidence"]
            self.state.emotional_state["confidence"] = min(1.0, max(0.0, 
                current_confidence + (success_rate - 0.5) * 0.1))
    
    def _generate_message(self, plan: str, environment: Dict[str, Any]) -> str:
        """Generate contextual message based on action"""
        messages = {
            "express_feelings": [
                f"{self.state.name}: I'm really hoping for a rose tonight...",
                f"{self.state.name}: This journey means everything to me.",
                f"{self.state.name}: I can see a future with you."
            ],
            "compete_for_attention": [
                f"{self.state.name}: I'm going to make the most of this time!",
                f"{self.state.name}: I deserve to be here.",
                f"{self.state.name}: Watch me shine!"
            ],
            "build_connection": [
                f"{self.state.name}: Tell me about your dreams...",
                f"{self.state.name}: I feel like we have a real connection.",
                f"{self.state.name}: This feels so natural with you."
            ],
            "facilitate_ceremony": [
                f"{self.state.name}: This is the final rose tonight.",
                f"{self.state.name}: Take a moment to think about your feelings.",
                f"{self.state.name}: The journey continues for some of you..."
            ],
            "socialize": [
                f"{self.state.name}: How is everyone feeling about tonight?",
                f"{self.state.name}: This house is getting intense!",
                f"{self.state.name}: Anyone want to chat by the pool?"
            ]
        }
        
        return random.choice(messages.get(plan, [f"{self.state.name}: {plan}"]))
    
    def _update_emotions(self, action: str, success: bool) -> None:
        """Update emotional state based on actions and outcomes"""
        if success:
            self.state.emotional_state["happiness"] = min(1.0, self.state.emotional_state["happiness"] + 0.1)
            self.state.emotional_state["confidence"] = min(1.0, self.state.emotional_state["confidence"] + 0.05)
        else:
            self.state.emotional_state["happiness"] = max(0.0, self.state.emotional_state["happiness"] - 0.05)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status for display"""
        return {
            "name": self.state.name,
            "role": self.state.role,
            "current_action": self.state.current_action,
            "location": self.state.location,
            "emotional_state": self.state.emotional_state,
            "decision_count": self.decision_count,
            "recent_memory": self.state.memory[-3:] if self.state.memory else []
        }

class DatingShowEnvironment:
    """Dating show environment and game master"""
    
    def __init__(self):
        self.current_step = 0
        self.current_event = "arrival"
        self.events = ["arrival", "group_date", "one_on_one", "cocktail_party", "rose_ceremony"]
        self.event_index = 0
        self.eliminated_contestants = []
        
    def get_environment_state(self, agents: List[SimplePIANOAgent]) -> Dict[str, Any]:
        """Get current environment state for agents"""
        return {
            "time": "evening" if self.current_step % 4 == 0 else "day",
            "event": self.current_event,
            "step": self.current_step,
            "location": "villa",
            "agents": [agent.state.name for agent in agents if agent.state.name not in self.eliminated_contestants]
        }
    
    def advance_step(self, agents: List[SimplePIANOAgent]) -> None:
        """Advance to next step and possibly next event"""
        self.current_step += 1
        
        # Change event every 3 steps
        if self.current_step % 3 == 0:
            self.event_index = (self.event_index + 1) % len(self.events)
            self.current_event = self.events[self.event_index]
            
            # Handle elimination during rose ceremony
            if self.current_event == "rose_ceremony" and len([a for a in agents if a.state.role == "contestant"]) > 2:
                contestants = [a for a in agents if a.state.role == "contestant" and a.state.name not in self.eliminated_contestants]
                if contestants:
                    # Eliminate contestant with lowest happiness
                    eliminated = min(contestants, key=lambda x: x.state.emotional_state["happiness"])
                    self.eliminated_contestants.append(eliminated.state.name)
                    print(f"💔 {eliminated.state.name} was eliminated!")

def run_piano_dating_show():
    """Run the PIANO dating show simulation"""
    print("🌹 PIANO Dating Show Simulation 🌹")
    print("=" * 50)
    
    # Create agents
    agents = [
        SimplePIANOAgent("Alice", "contestant", {"confidence": 0.8, "openness": 0.9, "competitiveness": 0.7}),
        SimplePIANOAgent("Bella", "contestant", {"confidence": 0.6, "openness": 0.8, "competitiveness": 0.9}),
        SimplePIANOAgent("Clara", "contestant", {"confidence": 0.7, "openness": 0.6, "competitiveness": 0.5}),
        SimplePIANOAgent("David", "host", {"confidence": 0.9, "openness": 0.7, "competitiveness": 0.3})
    ]
    
    # Create environment
    environment = DatingShowEnvironment()
    
    print(f"🏠 Contestants: {[a.state.name for a in agents if a.state.role == 'contestant']}")
    print(f"🎙️  Host: {[a.state.name for a in agents if a.state.role == 'host'][0]}")
    print()
    
    # Main simulation loop
    all_events = []
    
    for step in range(15):  # Run for 15 steps
        print(f"\n--- Step {step + 1}: {environment.current_event.upper()} ---")
        
        # Get environment state
        env_state = environment.get_environment_state(agents)
        step_events = []
        
        # Each agent goes through PIANO cycle
        active_agents = [a for a in agents if a.state.name not in environment.eliminated_contestants]
        
        for agent in active_agents:
            # PIANO Cycle: Perceive -> Plan -> Act
            perceptions = agent.perceive(env_state)
            plan = agent.plan(perceptions)
            action_result = agent.act(plan, env_state)
            
            step_events.append(action_result)
            print(f"  {action_result['message']}")
        
        # Agents reflect on recent events
        for agent in active_agents:
            agent.reflect(step_events)
        
        all_events.extend(step_events)
        
        # Show agent statuses
        print(f"\n📊 Agent Status:")
        for agent in active_agents:
            status = agent.get_status()
            emotions = ", ".join([f"{k}:{v:.1f}" for k, v in status["emotional_state"].items()])
            print(f"  {status['name']}: {status['current_action']} | {emotions}")
        
        # Advance environment
        environment.advance_step(agents)
        
        time.sleep(1.5)  # Pause for readability
    
    # Final results
    print(f"\n🎉 SIMULATION COMPLETE 🎉")
    remaining_contestants = [a.state.name for a in agents 
                           if a.state.role == "contestant" 
                           and a.state.name not in environment.eliminated_contestants]
    
    if remaining_contestants:
        print(f"🏆 Final contestants: {remaining_contestants}")
        # Winner is the one with highest happiness
        winner = max([a for a in agents if a.state.name in remaining_contestants], 
                    key=lambda x: x.state.emotional_state["happiness"])
        print(f"👑 Winner: {winner.state.name}!")
    
    print(f"📈 Total interactions: {len(all_events)}")
    print(f"💔 Eliminated: {environment.eliminated_contestants}")

if __name__ == "__main__":
    run_piano_dating_show()