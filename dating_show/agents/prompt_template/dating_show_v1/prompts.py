
import os
import json

class PromptManager:
    def __init__(self, template_path):
        self.template_path = template_path
        self.templates = self._load_templates()
        self.world_context = self._load_world_context()
        self.villa_coordinates = self._load_villa_coordinates()

    def _load_templates(self):
        """
        Loads all prompt templates from the specified directory.
        """
        templates = {}
        for root, _, files in os.walk(self.template_path):
            for file in files:
                if file.endswith(".txt"):
                    template_name = os.path.splitext(file)[0]
                    with open(os.path.join(root, file), 'r') as f:
                        templates[template_name] = f.read()
        return templates

    def format_prompt(self, template_name, inputs):
        """
        Formats a prompt with the given inputs.

        Args:
            template_name: The name of the template to use (e.g., "generate_goal").
            inputs: A dictionary where the keys are the placeholder names (e.g., "!<INPUT 0>!")
                    and the values are the strings to insert.

        Returns:
            The formatted prompt string.
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found.")

        prompt = self.templates[template_name]
        for key, value in inputs.items():
            prompt = prompt.replace(key, str(value))
        
        return prompt

    def _load_world_context(self):
        """Load world setting and agent roles for context"""
        context = {}
        try:
            # Load world setting
            world_setting_path = os.path.join(self.template_path, "global", "world_setting.txt")
            if os.path.exists(world_setting_path):
                with open(world_setting_path, 'r') as f:
                    context['world_setting'] = f.read()
            
            # Load agent roles
            agent_roles_path = os.path.join(self.template_path, "global", "agent_roles.txt")
            if os.path.exists(agent_roles_path):
                with open(agent_roles_path, 'r') as f:
                    context['agent_roles'] = f.read()
                    
        except Exception as e:
            print(f"Warning: Could not load world context: {e}")
            
        return context

    def _load_villa_coordinates(self):
        """Load villa coordinate mapping for spatial intelligence"""
        try:
            coord_path = os.path.join(self.template_path, "villa_coordinate_map.json")
            if os.path.exists(coord_path):
                with open(coord_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load villa coordinates: {e}")
        return {"villa_areas": {}, "default_areas": {}}

    def _get_location_context(self, current_tile):
        """Convert coordinates to rich location context"""
        # Convert tuple or list to string format
        if isinstance(current_tile, (tuple, list)):
            coord_key = f"({current_tile[0]}, {current_tile[1]})"
        else:
            coord_key = str(current_tile)
        
        # Look up in villa coordinates
        villa_areas = self.villa_coordinates.get("villa_areas", {})
        if coord_key in villa_areas:
            area_info = villa_areas[coord_key]
            return {
                "area_name": area_info.get("area_name", "Villa Area"),
                "description": area_info.get("description", ""),
                "atmosphere": area_info.get("atmosphere", ""),
                "features": area_info.get("features", []),
                "strategic_value": area_info.get("strategic_value", ""),
                "nearby_areas": area_info.get("nearby_areas", []),
                "activity_suggestions": area_info.get("activity_suggestions", [])
            }
        else:
            # Return default context
            default_area = self.villa_coordinates.get("default_areas", {}).get("villa_general", {})
            return {
                "area_name": "Villa Area",
                "description": default_area.get("description", "General villa space"),
                "atmosphere": default_area.get("atmosphere", "social, comfortable"),
                "features": default_area.get("features", ["villa amenities"]),
                "strategic_value": default_area.get("strategic_value", "suitable for various interactions"),
                "nearby_areas": default_area.get("nearby_areas", ["other villa areas"]),
                "activity_suggestions": default_area.get("activity_suggestions", ["socializing", "relaxing"])
            }

    def format_activity_prompt(self, agent_name, curr_time, current_tile, activity, nearby_personas, show_events="", episode_info="Episode 1"):
        """
        Format the generate_activity template with agent context and spatial intelligence.
        
        Args:
            agent_name: Name of the agent
            curr_time: Current simulation time
            current_tile: Agent's current position (x, y)
            activity: Agent's current planned activity
            nearby_personas: List of nearby agent names
            show_events: Current show events or challenges
            episode_info: Current episode information
            
        Returns:
            Formatted prompt string for activity generation with spatial context
        """
        # Get rich location context
        location_context = self._get_location_context(current_tile)
        
        # Create enhanced location description
        location_description = f"{location_context['area_name']} - {location_context['description']}"
        if location_context['atmosphere']:
            location_description += f" (Atmosphere: {location_context['atmosphere']})"
        
        inputs = {
            "!<INPUT 0>!": agent_name,
            "!<INPUT 1>!": curr_time.strftime('%I:%M %p') if hasattr(curr_time, 'strftime') else str(curr_time),
            "!<INPUT 2>!": location_description,
            "!<INPUT 3>!": activity,
            "!<INPUT 4>!": ', '.join(nearby_personas) if nearby_personas else 'alone',
            "!<INPUT 5>!": show_events,
            "!<INPUT 6>!": episode_info,
            "!<WORLD_SETTING>!": self.world_context.get('world_setting', ''),
            "!<LOCATION_FEATURES>!": ', '.join(location_context['features']) if location_context['features'] else 'standard villa amenities',
            "!<STRATEGIC_VALUE>!": location_context['strategic_value'],
            "!<ACTIVITY_SUGGESTIONS>!": ', '.join(location_context['activity_suggestions']) if location_context['activity_suggestions'] else 'general socializing'
        }
        
        return self.format_prompt("generate_activity", inputs)

    def get_world_context(self):
        """Get the loaded world setting and agent roles"""
        return self.world_context

    def format_batch_activities_prompt(self, curr_time, episode_info, show_events, contestants_data):
        """
        Format the batch_activities template for all contestants with spatial intelligence.
        
        Args:
            curr_time: Current simulation time
            episode_info: Current episode information
            show_events: Current show events or challenges
            contestants_data: List of dictionaries with contestant info
                             Each dict should have: name, position, activity, nearby_personas
            
        Returns:
            Formatted prompt string for batch activity generation with spatial context
        """
        # Format contestants data into readable text with location context
        formatted_contestants = []
        for i, contestant in enumerate(contestants_data, 1):
            name = contestant.get('name', f'Contestant {i}')
            position = contestant.get('position', '(0,0)')
            activity = contestant.get('activity', 'socializing')
            nearby = contestant.get('nearby_personas', [])
            nearby_str = ', '.join(nearby) if nearby else 'alone'
            
            # Get location context for this contestant
            location_context = self._get_location_context(position)
            location_description = f"{location_context['area_name']} - {location_context['description']}"
            
            formatted_contestants.append(
                f"{i}. {name} in {location_description}\n"
                f"   - Current activity: {activity}\n"
                f"   - Others nearby: {nearby_str}\n"
                f"   - Location features: {', '.join(location_context['features'][:3])}{'...' if len(location_context['features']) > 3 else ''}\n"
                f"   - Strategic value: {location_context['strategic_value']}"
            )
        
        contestants_text = '\n\n'.join(formatted_contestants)
        
        inputs = {
            "!<INPUT 0>!": curr_time.strftime('%I:%M %p') if hasattr(curr_time, 'strftime') else str(curr_time),
            "!<INPUT 1>!": episode_info,
            "!<INPUT 2>!": show_events,
            "!<INPUT 3>!": contestants_text,
            "!<WORLD_SETTING>!": self.world_context.get('world_setting', '')
        }
        
        return self.format_prompt("batch_activities", inputs)

# Example usage:
if __name__ == '__main__':
    # This assumes the script is run from the root of the project.
    prompt_manager = PromptManager("dating_show/agents/prompt_template/dating_show_v1")

    # Example for generate_goal
    goal_inputs = {
        "!<INPUT 0>!": '{"happiness": 0.2, "jealousy": 0.8}',
        "!<INPUT 1>!": 'Episode 3: Heartbreak Hotel'
    }
    formatted_goal_prompt = prompt_manager.format_prompt("generate_goal", goal_inputs)
    print("--- Formatted Generate Goal Prompt ---")
    print(formatted_goal_prompt)

    # Example for decide_on_rose
    rose_inputs = {
        "!<INPUT 0>!": '{"Alice": {"attraction": 0.9}, "Bob": {"attraction": 0.4}}'
    }
    formatted_rose_prompt = prompt_manager.format_prompt("decide_on_rose", rose_inputs)
    print("\n--- Formatted Decide on Rose Prompt ---")
    print(formatted_rose_prompt)

