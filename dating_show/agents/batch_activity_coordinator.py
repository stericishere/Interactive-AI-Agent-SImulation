"""
Batch Activity Coordinator for Dating Show Simulation
Handles batch API calls for all agents to reduce API usage from 24 calls to 1 call per iteration
"""

import os
import sys
import json
import requests
import time
from typing import List, Dict, Any

# Add path for prompt template access
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from agents.prompt_template.dating_show_v1.prompts import PromptManager


class BatchActivityCoordinator:
    """Coordinates batch activity generation for all personas"""
    
    def __init__(self):
        # Initialize PromptManager
        template_path = os.path.join(os.path.dirname(__file__), "prompt_template", "dating_show_v1")
        self.prompt_manager = PromptManager(template_path)
        
        # Cache for batch results
        self.activity_cache = {}
        self.last_batch_time = None
        
        # Load API key
        self.openrouter_api_key = self._load_api_key()
    
    def _load_api_key(self):
        """Load OpenRouter API key from environment"""
        env_path = "/Applications/Projects/Open source/generative_agents/.env"
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('OPENROUTER_API_KEY='):
                        return line.split('=', 1)[1].strip()
        return None
    
    def generate_batch_activities(self, personas_data: List[Dict[str, Any]], curr_time, shared_context: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate activities for all personas in a single API call
        
        Args:
            personas_data: List of persona dictionaries with name, position, activity, nearby_personas
            curr_time: Current simulation time
            shared_context: Shared context like show events, episode info
            
        Returns:
            Dictionary mapping persona names to activity descriptions
        """
        try:
            print(f"🔄 [BATCH] Starting batch activity generation for {len(personas_data)} agents")
            
            if not self.openrouter_api_key:
                print("⚠️ [BATCH] No API key found, falling back to individual generation")
                return self._fallback_activities(personas_data)
            
            # Prepare shared context
            if not shared_context:
                shared_context = {}
            
            show_events = shared_context.get('show_events', self._get_current_show_events(curr_time))
            episode_info = shared_context.get('episode_info', 'Hearts on Fire - Episode 1')
            
            # Format contestants data for batch prompt
            contestants_data = []
            for persona_data in personas_data:
                contestants_data.append({
                    'name': persona_data['name'],
                    'position': persona_data['position'],
                    'activity': persona_data['activity'],
                    'nearby_personas': persona_data.get('nearby_personas', [])
                })
            
            # Generate batch prompt using template
            prompt = self.prompt_manager.format_batch_activities_prompt(
                curr_time=curr_time,
                episode_info=episode_info,
                show_events=show_events,
                contestants_data=contestants_data
            )
            
            print(f"📝 [BATCH] Generated batch prompt for {len(contestants_data)} contestants")
            
            # Make single API call
            response = self._make_batch_api_call(prompt)
            
            if response:
                # Parse batch results
                activities = self._parse_batch_response(response, personas_data)
                
                # Cache results
                self.activity_cache = activities
                self.last_batch_time = curr_time
                
                print(f"✅ [BATCH] Successfully generated {len(activities)} activities")
                return activities
            else:
                print("❌ [BATCH] API call failed, falling back to individual generation")
                return self._fallback_activities(personas_data)
                
        except Exception as e:
            print(f"🚨 [BATCH] Error in batch generation: {e}")
            return self._fallback_activities(personas_data)
    
    def _make_batch_api_call(self, prompt: str) -> str:
        """Make the actual API call for batch processing"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/generative-agents/dating-show",
                "X-Title": "Dating Show Simulation - Batch Processing"
            }
            
            data = {
                "model": "deepseek/deepseek-chat-v3-0324:free",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 400  # More tokens for batch response
            }
            
            print(f"🌐 [BATCH] Making batch API call...")
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30  # Longer timeout for batch processing
            )
            
            if response.status_code == 429:
                print(f"🚦 [BATCH] Rate limited, waiting 5s...")
                time.sleep(5)
                return None
            elif response.status_code == 401:
                print(f"🚨 [BATCH] 401 Unauthorized - API key invalid")
                return None
            
            response.raise_for_status()
            result = response.json()
            
            if result and 'choices' in result and result['choices']:
                return result['choices'][0]['message']['content'].strip()
            else:
                print("⚠️ [BATCH] Empty API response")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"🚨 [BATCH] API request failed: {e}")
            return None
        except Exception as e:
            print(f"🚨 [BATCH] Unexpected error in API call: {e}")
            return None
    
    def _parse_batch_response(self, response: str, personas_data: List[Dict]) -> Dict[str, str]:
        """Parse the batch API response into individual activities"""
        activities = {}
        
        try:
            # Split response into lines and find numbered activities
            lines = response.split('\n')
            activity_lines = []
            
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.'))):
                    # Remove the number prefix
                    if '.' in line:
                        activity = line.split('.', 1)[1].strip()
                    else:
                        activity = line[1:].strip() if line[0].isdigit() else line
                    
                    if activity:
                        activity_lines.append(activity)
            
            # Map activities to personas
            for i, persona_data in enumerate(personas_data):
                name = persona_data['name']
                if i < len(activity_lines):
                    activity = activity_lines[i]
                    # Ensure it ends with @ villa
                    if not activity.endswith('@ villa'):
                        activity = f"{activity} @ villa"
                    activities[name] = activity
                    print(f"🎯 [BATCH] {name}: {activity}")
                else:
                    # Fallback for missing activities
                    fallback = f"{name} is {persona_data.get('activity', 'socializing')} @ villa"
                    activities[name] = fallback
                    print(f"⚠️ [BATCH] Using fallback for {name}: {fallback}")
            
        except Exception as e:
            print(f"🚨 [BATCH] Error parsing batch response: {e}")
            print(f"🔍 [BATCH] Raw response: {response}")
            
            # Return fallback activities if parsing fails
            return self._fallback_activities(personas_data)
        
        return activities
    
    def _fallback_activities(self, personas_data: List[Dict]) -> Dict[str, str]:
        """Generate fallback activities when batch processing fails"""
        activities = {}
        for persona_data in personas_data:
            name = persona_data['name']
            activity = persona_data.get('activity', 'socializing in the dating show')
            activities[name] = f"{name} is {activity} @ villa"
        return activities
    
    def _get_current_show_events(self, curr_time):
        """Get current show events based on time"""
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
    
    def get_cached_activity(self, persona_name: str) -> str:
        """Get cached activity for a persona"""
        return self.activity_cache.get(persona_name, f"{persona_name} is socializing @ villa")
    
    def clear_cache(self):
        """Clear the activity cache"""
        self.activity_cache = {}
        self.last_batch_time = None


# Global instance for easy access
batch_coordinator = BatchActivityCoordinator()