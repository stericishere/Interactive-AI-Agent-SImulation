# OpenRouter Migration Guide

This document describes the migration from OpenAI to OpenRouter using the llama-4-maverick model.

## Changes Made

### 1. Updated `gpt_structure.py`
- Replaced OpenAI API calls with OpenRouter API calls
- Added `openrouter_request()` function to handle API requests
- Updated all model functions to use llama-4-maverick model
- Maintained backward compatibility with existing function signatures

### 2. Created `utils.py`
- Added configuration for OpenRouter API key
- Maintains backward compatibility with existing imports

### 3. Function Mappings
- `ChatGPT_single_request()` → Now uses OpenRouter with llama-4-maverick
- `GPT4_request()` → Now uses OpenRouter with llama-4-maverick  
- `ChatGPT_request()` → Now uses OpenRouter with llama-4-maverick
- `GPT_request()` → Now uses OpenRouter with llama-4-maverick
- `get_embedding()` → Returns dummy embeddings (needs proper implementation)

## Setup Instructions

### 1. Set Environment Variable
```bash
export OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 2. Alternative: Update utils.py
Edit `agents/prompt_template/utils.py` and replace the placeholder with your actual API key:
```python
openrouter_api_key = "your_actual_openrouter_api_key_here"
```

### 3. Test the Integration
Run the test script to verify everything works:
```bash
cd dating_show
python test_openrouter_integration.py
```

## API Differences

### OpenAI → OpenRouter Changes
- **Endpoint**: `https://api.openai.com/v1` → `https://openrouter.ai/api/v1`
- **Model**: `gpt-3.5-turbo`/`gpt-4` → `llama-4-maverick`
- **Authentication**: OpenAI API key → OpenRouter API key
- **Response Format**: Compatible (uses same JSON structure)

### Important Notes
1. **Embeddings**: The `get_embedding()` function currently returns dummy embeddings. If embeddings are needed, consider using a dedicated embedding service.

2. **Rate Limits**: OpenRouter may have different rate limits than OpenAI. Monitor usage accordingly.

3. **Error Handling**: Error messages now reference "OpenRouter" instead of "ChatGPT" or "OpenAI".

4. **Model Parameters**: Some legacy GPT parameters are mapped to OpenRouter equivalents:
   - `temperature` → temperature
   - `max_tokens` → max_tokens
   - Other parameters are ignored for compatibility

## Troubleshooting

### Common Issues
1. **"OpenRouter API Error"**: Check your API key and network connection
2. **Import errors**: Ensure `utils.py` exists and contains the API key
3. **Module not found**: Verify the file paths are correct

### Testing Individual Components
```python
from agents.prompt_template.gpt_structure import openrouter_request

# Test direct API call
messages = [{"role": "user", "content": "Hello!"}]
response = openrouter_request(messages)
print(response)
```

## Rollback Instructions

If you need to rollback to OpenAI:
1. Restore the original `gpt_structure.py` from git
2. Remove or rename `utils.py` 
3. Update imports to use OpenAI API key configuration
4. Install OpenAI package: `pip install openai`

## Next Steps

1. **Test thoroughly** with your specific use cases
2. **Monitor performance** compared to previous OpenAI implementation
3. **Implement proper embeddings** if needed
4. **Update documentation** for end users
5. **Consider caching** for frequently used prompts to reduce API costs