"""
Minimal Persona implementation for 25-agent Smallville simulation
Designed to provide basic functionality for dating show frontend integration
"""

import json
import os
import random
from datetime import datetime


class PersonaScratch:
    """Manages persona's current state and temporary information"""
    
    def __init__(self):
        self.daily_plan_req = "socializing in the dating show"
        self.curr_tile = ["the_ville", "main_area"]
        self.chat = ""
        self.curr_time = datetime.now()
        self.planned_path = []
        
    def get_curr_event_and_desc(self):
        """Get current event description for the persona"""
        return (self.daily_plan_req, "general", "socializing", "dating show area")
        
    def get_curr_obj_event_and_desc(self):
        """Get current object event description"""
        return (self.daily_plan_req, "object", "interacting", "social interaction")


class Persona:
    """
    Minimal Persona class for generative agents simulation
    Represents a single agent in the 25-agent Smallville dating show
    """
    
    def __init__(self, name, persona_folder):
        """Initialize persona with basic information"""
        self.name = name
        self.persona_folder = persona_folder
        
        # Initialize scratch (current state)
        self.scratch = PersonaScratch()
        
        # Load or create basic persona data
        self._load_persona_data()
        
        # Initialize movement and behavior
        self.last_position = (50, 50)  # Default starting position
        self.target_position = None
        
    def _load_persona_data(self):
        """Load persona data from folder or create defaults"""
        try:
            # Try to load existing data
            scratch_file = os.path.join(self.persona_folder, "bootstrap_memory", "scratch.json")
            if os.path.exists(scratch_file):
                with open(scratch_file, 'r') as f:
                    data = json.load(f)
                    # Load relevant data
                    if 'daily_plan_req' in data:
                        self.scratch.daily_plan_req = data['daily_plan_req']
            else:
                # Create default persona behavior for dating show
                self._create_default_persona()
                
        except Exception as e:
            print(f"Warning: Could not load persona data for {self.name}: {e}")
            self._create_default_persona()
    
    def _create_default_persona(self):
        """Create default dating show persona behavior"""
        activities = [
            "socializing with other contestants",
            "exploring the villa",
            "having conversations by the pool",
            "preparing for the next challenge",
            "reflecting on relationships",
            "enjoying villa amenities"
        ]
        
        self.scratch.daily_plan_req = random.choice(activities)
        self.scratch.curr_tile = ["the_ville", "villa_area"]
        
    def move(self, maze, personas, current_tile, curr_time):
        """
        Core movement function for the persona
        Returns: (next_tile, pronunciatio_emoji, description)
        """
        try:
            # Simple movement logic for dating show
            x, y = current_tile
            
            # Random movement within bounds
            new_x = max(10, min(90, x + random.randint(-2, 2)))
            new_y = max(10, min(90, y + random.randint(-2, 2)))
            
            next_tile = (new_x, new_y)
            
            # Dating show appropriate emojis
            emojis = ["💕", "🌹", "💬", "😊", "🥰", "💭", "✨", "🌟"]
            pronunciatio = random.choice(emojis)
            
            # Generate description based on current activity
            description = f"{self.scratch.daily_plan_req} @ villa"
            
            # Update internal state
            self.last_position = next_tile
            self.scratch.curr_time = curr_time
            
            return next_tile, pronunciatio, description
            
        except Exception as e:
            print(f"Error in persona movement for {self.name}: {e}")
            # Return safe default
            return current_tile, "🤔", "thinking"
    
    def save(self, save_folder):
        """Save persona state to folder"""
        try:
            os.makedirs(save_folder, exist_ok=True)
            
            # Save basic scratch data
            scratch_data = {
                "daily_plan_req": self.scratch.daily_plan_req,
                "curr_tile": self.scratch.curr_tile,
                "chat": self.scratch.chat,
                "curr_time": self.scratch.curr_time.isoformat() if hasattr(self.scratch.curr_time, 'isoformat') else str(self.scratch.curr_time)
            }
            
            scratch_file = os.path.join(save_folder, "scratch.json")
            with open(scratch_file, 'w') as f:
                json.dump(scratch_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save persona data for {self.name}: {e}")


# For backwards compatibility and easier imports
__all__ = ['Persona', 'PersonaScratch']