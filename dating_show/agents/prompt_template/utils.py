"""
Utility configuration for API keys and model settings.
"""
import os

# OpenRouter API configuration
openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key-here")

# For backwards compatibility with existing code
openai_api_key = ''