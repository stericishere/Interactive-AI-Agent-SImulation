
import os

class PromptManager:
    def __init__(self, template_path):
        self.template_path = template_path
        self.templates = self._load_templates()
        self.world_context = self._load_world_context()

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

    def format_activity_prompt(self, agent_name, curr_time, current_tile, activity, nearby_personas, show_events="", episode_info="Episode 1"):
        """
        Format the generate_activity template with agent context.
        
        Args:
            agent_name: Name of the agent
            curr_time: Current simulation time
            current_tile: Agent's current position (x, y)
            activity: Agent's current planned activity
            nearby_personas: List of nearby agent names
            show_events: Current show events or challenges
            episode_info: Current episode information
            
        Returns:
            Formatted prompt string for activity generation
        """
        inputs = {
            "!<INPUT 0>!": agent_name,
            "!<INPUT 1>!": curr_time.strftime('%I:%M %p') if hasattr(curr_time, 'strftime') else str(curr_time),
            "!<INPUT 2>!": str(current_tile),
            "!<INPUT 3>!": activity,
            "!<INPUT 4>!": ', '.join(nearby_personas) if nearby_personas else 'alone',
            "!<INPUT 5>!": show_events,
            "!<INPUT 6>!": episode_info
        }
        
        return self.format_prompt("generate_activity", inputs)

    def get_world_context(self):
        """Get the loaded world setting and agent roles"""
        return self.world_context

    def format_batch_activities_prompt(self, curr_time, episode_info, show_events, contestants_data):
        """
        Format the batch_activities template for all contestants.
        
        Args:
            curr_time: Current simulation time
            episode_info: Current episode information
            show_events: Current show events or challenges
            contestants_data: List of dictionaries with contestant info
                             Each dict should have: name, position, activity, nearby_personas
            
        Returns:
            Formatted prompt string for batch activity generation
        """
        # Format contestants data into readable text
        formatted_contestants = []
        for i, contestant in enumerate(contestants_data, 1):
            name = contestant.get('name', f'Contestant {i}')
            position = contestant.get('position', '(0,0)')
            activity = contestant.get('activity', 'socializing')
            nearby = contestant.get('nearby_personas', [])
            nearby_str = ', '.join(nearby) if nearby else 'alone'
            
            formatted_contestants.append(
                f"{i}. {name} at position {position}\n"
                f"   - Current activity: {activity}\n"
                f"   - Others nearby: {nearby_str}"
            )
        
        contestants_text = '\n\n'.join(formatted_contestants)
        
        inputs = {
            "!<INPUT 0>!": curr_time.strftime('%I:%M %p') if hasattr(curr_time, 'strftime') else str(curr_time),
            "!<INPUT 1>!": episode_info,
            "!<INPUT 2>!": show_events,
            "!<INPUT 3>!": contestants_text
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

