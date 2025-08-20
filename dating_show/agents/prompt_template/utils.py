"""
Utility configuration for API keys and model settings.
"""
import os

# OpenRouter API configuration
openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "NO-key found")

