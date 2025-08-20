"""
Configuration file for the dating show simulation backend.
Contains API keys and path configurations for OpenRouter integration.
"""
import os

# OpenRouter API configuration (using environment variable for security)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key-here")

# For backwards compatibility with existing code that expects openai_api_key
openai_api_key = openrouter_api_key

# Put your name
key_owner = os.getenv("KEY_OWNER", "Dating Show User")

# Path configurations for the dating show environment  
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
maze_assets_loc = os.path.join(_project_root, "dating_show_env", "frontend_service", "static_dirs", "assets")
env_matrix = os.path.join(maze_assets_loc, "the_ville", "matrix")
env_visuals = os.path.join(maze_assets_loc, "the_ville", "visuals")

fs_storage = os.path.join(_project_root, "dating_show_env", "frontend_service", "storage")
fs_temp_storage = os.path.join(_project_root, "dating_show_env", "frontend_service", "temp_storage")

collision_block_id = "32125"

# Verbose debugging
debug = True
