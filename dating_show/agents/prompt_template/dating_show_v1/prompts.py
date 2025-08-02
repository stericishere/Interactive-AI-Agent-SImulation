
import os

class PromptManager:
    def __init__(self, template_path):
        self.template_path = template_path
        self.templates = self._load_templates()

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

