"""
Utility configuration for API keys and model settings.
"""
import os

# OpenRouter API configuration
openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key-here")

# For backwards compatibility with existing code
openai_api_key = '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'