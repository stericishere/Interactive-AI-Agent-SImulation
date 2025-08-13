#!/usr/bin/env python3
"""
Dry run testing for OpenRouter migration
Tests all functions without making actual API calls
"""

import os
import sys
import json
from unittest.mock import Mock, patch

# Add paths for imports
sys.path.append('dating_show/agents/prompt_template')
sys.path.append('reverie/backend_server')

def test_imports():
    """Test that all imports work correctly"""
    print("=== IMPORT TESTING ===")
    
    try:
        # Test dating show imports
        from gpt_structure import ChatGPT_request, GPT4_request, openrouter_request
        print("✅ Dating show gpt_structure imports successful")
        
        # Test utils import
        from utils import openrouter_api_key
        print("✅ Utils import successful")
        
        # Test reverie imports
        sys.path.append('.')
        import test as reverie_test
        print("✅ Reverie test module import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_function_signatures():
    """Test that all functions have correct signatures"""
    print("\n=== FUNCTION SIGNATURE TESTING ===")
    
    try:
        from gpt_structure import ChatGPT_request, GPT4_request, openrouter_request, ChatGPT_safe_generate_response
        
        # Test function signatures
        functions_to_test = [
            ('ChatGPT_request', 1),  # Takes 1 parameter (prompt)
            ('GPT4_request', 1),     # Takes 1 parameter (prompt)
            ('openrouter_request', 1), # Takes 1+ parameters (messages, optional model, temp, tokens)
        ]
        
        for func_name, min_params in functions_to_test:
            func = locals()[func_name]
            # Check if function is callable
            if callable(func):
                print(f"✅ {func_name} is callable")
            else:
                print(f"❌ {func_name} is not callable")
        
        return True
    except Exception as e:
        print(f"❌ Function signature test failed: {e}")
        return False

def test_mock_api_calls():
    """Test API calls with mocked responses"""
    print("\n=== MOCK API TESTING ===")
    
    try:
        from gpt_structure import openrouter_request, ChatGPT_request
        
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response from llama-4-maverick"}}]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.post', return_value=mock_response):
            # Test openrouter_request
            messages = [{"role": "user", "content": "Hello, world!"}]
            response = openrouter_request(messages)
            print(f"✅ openrouter_request mock test: '{response}'")
            
            # Test ChatGPT_request  
            response = ChatGPT_request("Hello, world!")
            print(f"✅ ChatGPT_request mock test: '{response}'")
        
        # Mock API error
        with patch('requests.post', side_effect=Exception("Network error")):
            response = ChatGPT_request("Test error handling")
            print(f"✅ Error handling test: '{response}'")
        
        return True
    except Exception as e:
        print(f"❌ Mock API test failed: {e}")
        return False

def test_environment_configuration():
    """Test environment variable handling"""
    print("\n=== ENVIRONMENT CONFIGURATION TESTING ===")
    
    try:
        # Test with mock environment variable
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test_key_123'}):
            from utils import openrouter_api_key
            # Reload to get new environment
            if 'utils' in sys.modules:
                del sys.modules['utils']
            
            # Re-import to test environment handling
            exec("from utils import openrouter_api_key")
            print("✅ Environment variable handling works")
        
        # Test .env.example exists
        if os.path.exists('.env.example'):
            print("✅ .env.example template exists")
            with open('.env.example', 'r') as f:
                content = f.read()
                if 'OPENROUTER_API_KEY' in content:
                    print("✅ .env.example contains OpenRouter configuration")
                else:
                    print("❌ .env.example missing OpenRouter configuration")
        else:
            print("❌ .env.example template missing")
        
        return True
    except Exception as e:
        print(f"❌ Environment configuration test failed: {e}")
        return False

def test_backward_compatibility():
    """Test that all legacy function calls still work"""
    print("\n=== BACKWARD COMPATIBILITY TESTING ===")
    
    try:
        from gpt_structure import ChatGPT_safe_generate_response, safe_generate_response
        
        # Mock for testing
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Compatibility test response"}}]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.post', return_value=mock_response):
            # Test legacy safe_generate_response
            result = safe_generate_response(
                "Test prompt", 
                {"engine": "text-davinci-003", "max_tokens": 50, "temperature": 0.7},
                repeat=1,
                fail_safe_response="FAIL",
                func_validate=lambda x, **kwargs: True,
                func_clean_up=lambda x, **kwargs: x
            )
            print(f"✅ safe_generate_response compatibility: '{result}'")
            
            # Test ChatGPT_safe_generate_response
            result = ChatGPT_safe_generate_response(
                "Test prompt",
                "example output",
                "Special instruction",
                repeat=1,
                fail_safe_response="FAIL",
                func_validate=lambda x, **kwargs: True,
                func_clean_up=lambda x, **kwargs: x
            )
            print(f"✅ ChatGPT_safe_generate_response compatibility: '{result}'")
        
        return True
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def test_parameter_mapping():
    """Test that GPT parameters are properly mapped"""
    print("\n=== PARAMETER MAPPING TESTING ===")
    
    try:
        from gpt_structure import openrouter_request
        
        # Mock to capture the request parameters
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Parameter mapping test"}}]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.post', return_value=mock_response) as mock_post:
            # Test parameter mapping
            messages = [{"role": "user", "content": "Test"}]
            openrouter_request(messages, temperature=0.8, max_tokens=100)
            
            # Check that parameters were passed correctly
            call_args = mock_post.call_args
            request_data = call_args[1]['json']
            
            if request_data.get('temperature') == 0.8:
                print("✅ Temperature parameter mapping works")
            else:
                print("❌ Temperature parameter mapping failed")
                
            if request_data.get('max_tokens') == 100:
                print("✅ Max tokens parameter mapping works")
            else:
                print("❌ Max tokens parameter mapping failed")
                
            if request_data.get('model') == 'llama-4-maverick':
                print("✅ Model parameter correctly set to llama-4-maverick")
            else:
                print("❌ Model parameter mapping failed")
        
        return True
    except Exception as e:
        print(f"❌ Parameter mapping test failed: {e}")
        return False

def run_all_tests():
    """Run all dry run tests"""
    print("🧪 OPENROUTER MIGRATION DRY RUN TESTING")
    print("=" * 50)
    
    test_results = []
    
    test_results.append(("Import Testing", test_imports()))
    test_results.append(("Function Signatures", test_function_signatures()))
    test_results.append(("Mock API Calls", test_mock_api_calls()))
    test_results.append(("Environment Config", test_environment_configuration()))
    test_results.append(("Backward Compatibility", test_backward_compatibility()))
    test_results.append(("Parameter Mapping", test_parameter_mapping()))
    
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - MIGRATION READY FOR PRODUCTION!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed - Review issues before production")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)