#!/usr/bin/env python3
"""
Test script for OpenRouter integration with llama-4-maverick model.
"""

import os
import sys

# Add the agents directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents', 'prompt_template'))

def test_openrouter_integration():
    """Test the OpenRouter integration"""
    
    print("Testing OpenRouter integration with llama-4-maverick...")
    
    try:
        # Import the modified gpt_structure
        from gpt_structure import ChatGPT_single_request, GPT4_request, openrouter_request
        
        # Test basic chat completion
        test_prompt = "Hello! Please respond with 'OpenRouter integration successful' if you can see this message."
        
        print(f"Sending test prompt: {test_prompt}")
        
        # Test the OpenRouter request function directly
        messages = [{"role": "user", "content": test_prompt}]
        response = openrouter_request(messages)
        
        print(f"OpenRouter direct response: {response}")
        
        # Test the ChatGPT compatibility function
        response2 = ChatGPT_single_request(test_prompt)
        print(f"ChatGPT compatibility response: {response2}")
        
        # Test the GPT4 compatibility function  
        response3 = GPT4_request(test_prompt)
        print(f"GPT4 compatibility response: {response3}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the utils.py file exists and contains the OpenRouter API key.")
        return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Make sure to set your OPENROUTER_API_KEY environment variable.")
        return False

def check_configuration():
    """Check the configuration"""
    print("Checking configuration...")
    
    # Check if the OpenRouter API key is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        print(f"✅ OPENROUTER_API_KEY is set (length: {len(api_key)})")
    else:
        print("⚠️  OPENROUTER_API_KEY environment variable not set")
        print("   Please set it with: export OPENROUTER_API_KEY=your_key_here")
    
    # Check if utils.py exists
    utils_path = os.path.join(os.path.dirname(__file__), 'agents', 'prompt_template', 'utils.py')
    if os.path.exists(utils_path):
        print(f"✅ utils.py exists at {utils_path}")
    else:
        print(f"❌ utils.py not found at {utils_path}")
    
    # Check if gpt_structure.py exists
    gpt_path = os.path.join(os.path.dirname(__file__), 'agents', 'prompt_template', 'gpt_structure.py')
    if os.path.exists(gpt_path):
        print(f"✅ gpt_structure.py exists at {gpt_path}")
    else:
        print(f"❌ gpt_structure.py not found at {gpt_path}")

if __name__ == "__main__":
    print("OpenRouter Integration Test")
    print("=" * 50)
    
    check_configuration()
    print()
    
    if test_openrouter_integration():
        print("\n🎉 Integration test passed!")
    else:
        print("\n💥 Integration test failed!")
        sys.exit(1)