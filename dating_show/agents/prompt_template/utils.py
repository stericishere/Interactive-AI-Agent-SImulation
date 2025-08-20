"""
Utility configuration for API keys and model settings.
"""
import os

# OpenRouter API configuration
openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key-here")

# For backwards compatibility with existing code
openai_api_key = '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'