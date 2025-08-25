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
        Core movement function for the persona with AI-generated behaviors
        Returns: (next_tile, pronunciatio_emoji, description)
        """
        print(f"🤖 [PERSONA DEBUG] {self.name}.move() called at {curr_time} from position {current_tile}")
        try:
            # Get current position
            x, y = current_tile
            
            # Use actual map dimensions from maze configuration
            max_x = maze.maze_width - 1 if hasattr(maze, 'maze_width') else 139  # Default to 139 (140-1)
            max_y = maze.maze_height - 1 if hasattr(maze, 'maze_height') else 99   # Default to 99 (100-1)
            
            # Generate movement attempt with realistic boundaries
            attempts = 0
            max_attempts = 5
            
            while attempts < max_attempts:
                # Random movement within map bounds
                delta_x = random.randint(-2, 2)
                delta_y = random.randint(-2, 2)
                
                new_x = max(0, min(max_x, x + delta_x))
                new_y = max(0, min(max_y, y + delta_y))
                
                # Check collision detection if maze has collision data
                if hasattr(maze, 'collision_maze') and maze.collision_maze:
                    # Ensure coordinates are within collision maze bounds
                    if (0 <= new_y < len(maze.collision_maze) and 
                        0 <= new_x < len(maze.collision_maze[0])):
                        # Check if position is not blocked (collision_maze uses "0" for open space)
                        if maze.collision_maze[new_y][new_x] == "0":
                            break  # Valid position found
                    attempts += 1
                else:
                    # No collision detection available, use the coordinates
                    break
            
            # If all attempts failed, stay at current position
            if attempts >= max_attempts:
                new_x, new_y = x, y
                print(f"🚧 [COLLISION] {self.name} couldn't find valid move from {current_tile}, staying put")
            
            next_tile = (new_x, new_y)
            
            # Generate AI-powered description
            description = self._generate_ai_description(curr_time, current_tile, personas)
            
            # Dating show appropriate emojis
            emojis = ["💕", "🌹", "💬", "😊", "🥰", "💭", "✨", "🌟", "🔥", "🌺", "📚", "🎵", "🍃", "⚡", "🎮", "🌸", "🛠️", "🎬"]
            pronunciatio = random.choice(emojis)
            
            # Update internal state
            self.last_position = next_tile
            self.scratch.curr_time = curr_time
            
            return next_tile, pronunciatio, description
            
        except Exception as e:
            print(f"Error in persona movement for {self.name}: {e}")
            # Return safe default
            return current_tile, "🤔", "thinking"
    
    def _generate_ai_description(self, curr_time, current_tile, personas):
        """Generate AI-powered description for the persona's current activity"""
        print(f"🧠 [AI DEBUG] Generating AI description for {self.name}")
        
        # First, check if there's a cached batch result
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "dating_show"))
            from agents.batch_activity_coordinator import batch_coordinator
            
            cached_activity = batch_coordinator.get_cached_activity(self.name)
            if cached_activity and batch_coordinator.last_batch_time:
                # Check if cache is recent (within the same simulation step)
                time_diff = abs((curr_time - batch_coordinator.last_batch_time).total_seconds()) if hasattr(curr_time, 'total_seconds') else 0
                if time_diff < 60:  # Cache valid for 1 minute
                    print(f"📋 [CACHE] Using cached batch result for {self.name}: {cached_activity}")
                    return cached_activity
        except Exception as e:
            print(f"⚠️ [CACHE] Could not access batch cache: {e}")
        
        # Fall back to individual API call with template
        try:
            import sys
            import os
            import json
            import requests
            import time
            import random
            
            # Load OpenRouter API key directly
            env_path = "/Applications/Projects/Open source/generative_agents/.env"
            openrouter_api_key = None
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith('OPENROUTER_API_KEY='):
                            openrouter_api_key = line.split('=', 1)[1].strip()
                            break
            
            if not openrouter_api_key:
                print(f"⚠️ [DEBUG] No API key found for {self.name}")
                return f"{self.name} is {self.scratch.daily_plan_req} @ villa"
            
            print(f"🔑 [DEBUG] API key loaded for {self.name}: {openrouter_api_key[:20]}...")
            
            # Add random delay to avoid rate limits (0.5-2.0 seconds)
            delay = random.uniform(0.5, 2.0)
            print(f"⏰ [DEBUG] Adding {delay:.1f}s delay for rate limiting for {self.name}")
            time.sleep(delay)
            
            # Get nearby personas for context
            nearby_personas = self._get_nearby_personas(current_tile, personas)
            
            # Use PromptManager to create template-based prompt
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "dating_show"))
                from agents.prompt_template.dating_show_v1.prompts import PromptManager
                
                template_path = os.path.join(os.path.dirname(__file__), "..", "..", "dating_show", "agents", "prompt_template", "dating_show_v1")
                prompt_manager = PromptManager(template_path)
                
                # Get current show events (could be enhanced with real events later)
                show_events = self._get_current_show_events(curr_time)
                episode_info = self._get_episode_info(curr_time)
                
                # Generate prompt using template
                prompt = prompt_manager.format_activity_prompt(
                    agent_name=self.name,
                    curr_time=curr_time,
                    current_tile=current_tile,
                    activity=self.scratch.daily_plan_req,
                    nearby_personas=nearby_personas,
                    show_events=show_events,
                    episode_info=episode_info
                )
                
                print(f"📝 [TEMPLATE DEBUG] Using template-based prompt for {self.name}")
                
            except Exception as template_error:
                print(f"⚠️ [TEMPLATE] Template loading failed, using fallback: {template_error}")
                # Fallback prompt
                prompt = f"""As {self.name} on dating show "Hearts on Fire," describe what I'm doing right now in 10-15 words.

Current context:
- Time: {curr_time.strftime('%I:%M %p')} at the villa
- Activity: {self.scratch.daily_plan_req}
- Others nearby: {', '.join(nearby_personas) if nearby_personas else 'alone'}

Examples of good descriptions:
- chatting by the pool about relationships
- cooking dinner and flirting with someone
- strategizing about tonight's elimination
- having a heart-to-heart conversation
- getting ready for the challenge
- relaxing and bonding with others

Return only the activity description - no quotes, no roleplay, no meta-commentary."""

            # Make direct API request with retry logic
            headers = {
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/generative-agents/dating-show",
                "X-Title": "Dating Show Simulation"
            }
            
            data = {
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 50
            }
            
            # Retry logic for rate limiting
            for attempt in range(3):
                try:
                    print(f"🌐 [DEBUG] Making API call for {self.name} (attempt {attempt+1}/3)...")
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=15
                    )
                    
                    if response.status_code == 429:
                        retry_delay = (attempt + 1) * 2  # 2, 4, 6 seconds
                        print(f"🚦 [DEBUG] Rate limited, retrying in {retry_delay}s for {self.name}")
                        time.sleep(retry_delay)
                        continue
                    elif response.status_code == 401:
                        print(f"🚨 [DEBUG] 401 Unauthorized for {self.name}. Response: {response.text}")
                        print(f"🚨 [DEBUG] API Key being used: {openrouter_api_key[:20]}...{openrouter_api_key[-5:]}")
                        raise requests.exceptions.RequestException(f"401 Unauthorized - API key invalid")
                    
                    response.raise_for_status()
                    break
                    
                except requests.exceptions.RequestException as e:
                    if attempt < 2:  # Don't sleep on last attempt
                        print(f"⚠️ [DEBUG] API error attempt {attempt+1}, retrying: {e}")
                        time.sleep(1)
                    else:
                        raise e
            
            result = response.json()
            if result and 'choices' in result and result['choices']:
                ai_response = result['choices'][0]['message']['content'].strip()
                print(f"🔍 [DEBUG] Raw AI response for {self.name}: {ai_response}")
                
                # Clean up the response - remove thinking tags and extract the description
                ai_description = ai_response
                if '<think>' in ai_description:
                    # Remove thinking process, keep only the description part
                    thinking_end = ai_description.rfind('>')
                    if thinking_end != -1:
                        ai_description = ai_description[thinking_end + 1:].strip()
                
                # Remove any remaining markdown or formatting
                ai_description = ai_description.replace('**', '').replace('*', '').strip()
                
                # If description is too short or empty, use a fallback
                if len(ai_description) < 5 or ai_description.startswith('@') or 'villa' not in ai_description:
                    ai_description = f"{self.name} is {self.scratch.daily_plan_req}"
                
                # Ensure it ends with @ villa
                if not ai_description.endswith('@ villa'):
                    ai_description = f"{ai_description} @ villa"
                
                print(f"🤖 [DEBUG] Cleaned AI description for {self.name}: {ai_description}")
                return ai_description
            else:
                # Fallback if AI call fails
                print(f"⚠️ [DEBUG] AI response empty for {self.name}")
                return f"{self.name} is {self.scratch.daily_plan_req} @ villa"
                
        except Exception as e:
            # Fallback if AI generation fails
            print(f"🚨 [DEBUG] AI description generation failed for {self.name}: {e}")
            return f"{self.name} is {self.scratch.daily_plan_req} @ villa"
    
    def _get_nearby_personas(self, current_tile, personas, distance_threshold=10):
        """Get list of personas within distance threshold"""
        nearby = []
        x, y = current_tile
        
        for name, persona in personas.items():
            if name != self.name and hasattr(persona, 'last_position'):
                other_x, other_y = persona.last_position
                distance = ((x - other_x) ** 2 + (y - other_y) ** 2) ** 0.5
                if distance <= distance_threshold:
                    nearby.append(name)
        
        return nearby[:3]  # Limit to 3 nearby personas
    
    def _get_current_show_events(self, curr_time):
        """Get current show events based on time - can be enhanced later"""
        # Simple time-based events for now
        hour = curr_time.hour if hasattr(curr_time, 'hour') else 12
        
        if 8 <= hour < 12:
            return "Morning activities and breakfast conversations"
        elif 12 <= hour < 16:
            return "Afternoon dates and pool time"
        elif 16 <= hour < 20:
            return "Evening preparation and villa activities"
        elif 20 <= hour < 24:
            return "Dinner time and potential elimination ceremony"
        else:
            return "Late night conversations and strategic planning"
    
    def _get_episode_info(self, curr_time):
        """Get current episode info - can be enhanced with real episode tracking"""
        # Simple episode calculation based on simulation time
        # Could be enhanced to track actual episodes
        return "Hearts on Fire - Episode 1"
    
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