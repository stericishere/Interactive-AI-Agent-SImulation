#!/usr/bin/env python3
"""
Rate Limit Debug Tool for OpenRouter API
Analyzes rate limiting errors and provides detailed debugging information
"""

import os
import sys
import json
import time
import requests
from datetime import datetime
from typing import Dict, Any, Optional

def load_api_key() -> Optional[str]:
    """Load API key from environment or .env file"""
    # Try environment first
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    
    # Try .env file
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('OPENROUTER_API_KEY='):
                    return line.split('=', 1)[1].strip()
    return None

def debug_rate_limits():
    """Main debugging function to analyze rate limits"""
    print("🔍 OpenRouter Rate Limit Debug Tool")
    print("=" * 50)
    
    # Check API key
    api_key = load_api_key()
    if not api_key:
        print("❌ No API key found!")
        print("   Please set OPENROUTER_API_KEY environment variable or add it to .env file")
        return
    
    print(f"✅ API key found (length: {len(api_key)})")
    print()
    
    # Test basic connectivity
    print("🌐 Testing basic connectivity...")
    test_url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(test_url, headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response time: {response.elapsed.total_seconds():.2f}s")
        
        # Check rate limit headers
        print("\n📊 Rate Limit Headers:")
        rate_limit_headers = [
            'X-RateLimit-Limit',
            'X-RateLimit-Remaining', 
            'X-RateLimit-Reset',
            'X-RateLimit-Reset-Requests',
            'X-RateLimit-Reset-Tokens',
            'Retry-After',
            'X-OpenRouter-Rate-Limit',
            'X-OpenRouter-Rate-Remaining'
        ]
        
        found_headers = False
        for header in rate_limit_headers:
            if header in response.headers:
                print(f"   {header}: {response.headers[header]}")
                found_headers = True
        
        if not found_headers:
            print("   ⚠️  No standard rate limit headers found")
        
        print("\n🔍 All Response Headers:")
        for key, value in response.headers.items():
            if 'rate' in key.lower() or 'limit' in key.lower():
                print(f"   {key}: {value}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Connectivity test failed: {e}")
        return
    
    print("\n" + "=" * 50)
    
    # Test actual API calls with different patterns
    test_patterns = [
        {"name": "Single request", "requests": 1, "delay": 0},
        {"name": "Rapid requests (5)", "requests": 5, "delay": 0.1},
        {"name": "Burst requests (10)", "requests": 10, "delay": 0.05},
    ]
    
    for pattern in test_patterns:
        print(f"\n🧪 Testing: {pattern['name']}")
        print(f"   Requests: {pattern['requests']}, Delay: {pattern['delay']}s")
        
        success_count = 0
        error_count = 0
        rate_limit_count = 0
        
        for i in range(pattern['requests']):
            try:
                messages = [{"role": "user", "content": f"Test message {i+1}. Respond with just 'OK'"}]
                
                start_time = time.time()
                result = test_api_call_with_headers(api_key, messages)
                end_time = time.time()
                
                if result['success']:
                    success_count += 1
                    print(f"   ✅ Request {i+1}: Success ({end_time - start_time:.2f}s)")
                elif result['rate_limited']:
                    rate_limit_count += 1
                    print(f"   🚦 Request {i+1}: Rate limited - {result['error']}")
                    if result['retry_after']:
                        print(f"      Retry-After: {result['retry_after']}s")
                else:
                    error_count += 1
                    print(f"   ❌ Request {i+1}: Error - {result['error']}")
                
                # Show rate limit info if available
                if result['rate_info']:
                    print(f"      Rate info: {result['rate_info']}")
                
                if pattern['delay'] > 0:
                    time.sleep(pattern['delay'])
                    
            except KeyboardInterrupt:
                print("\n   ⏹️  Test interrupted by user")
                break
        
        print(f"   📊 Results: {success_count} success, {error_count} errors, {rate_limit_count} rate limited")
    
    print("\n" + "=" * 50)
    print("🎯 Recommendations:")
    print("1. If you see 429 errors, implement exponential backoff")
    print("2. Check for Retry-After header and respect it")  
    print("3. Consider request queuing to avoid burst limits")
    print("4. Monitor token usage as well as request count")
    print("5. Use batch processing when possible to reduce API calls")

def test_api_call_with_headers(api_key: str, messages: list) -> Dict[str, Any]:
    """Test API call and extract detailed error/rate limit information"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 10
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        # Extract rate limit info
        rate_info = {}
        for header_name in response.headers:
            if 'rate' in header_name.lower() or 'limit' in header_name.lower():
                rate_info[header_name] = response.headers[header_name]
        
        retry_after = response.headers.get('Retry-After')
        
        if response.status_code == 429:
            return {
                'success': False,
                'rate_limited': True,
                'error': f"429 - {response.text[:100]}",
                'retry_after': retry_after,
                'rate_info': rate_info,
                'status_code': 429
            }
        elif response.status_code != 200:
            return {
                'success': False,
                'rate_limited': False,
                'error': f"{response.status_code} - {response.text[:100]}",
                'retry_after': retry_after,
                'rate_info': rate_info,
                'status_code': response.status_code
            }
        else:
            result = response.json()
            return {
                'success': True,
                'rate_limited': False,
                'error': None,
                'retry_after': retry_after,
                'rate_info': rate_info,
                'response': result,
                'status_code': 200
            }
            
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'rate_limited': False,
            'error': str(e),
            'retry_after': None,
            'rate_info': {},
            'status_code': None
        }

def analyze_current_usage():
    """Analyze current API usage patterns from logs if available"""
    print("\n📈 Current Usage Analysis:")
    print("   Looking for recent API calls and patterns...")
    
    # Look for recent simulation runs or logs
    log_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if 'log' in file.lower() or file.endswith('.log'):
                log_files.append(os.path.join(root, file))
    
    if log_files:
        print(f"   Found {len(log_files)} potential log files")
        # Could analyze these for rate limit patterns
    else:
        print("   No log files found for analysis")

if __name__ == "__main__":
    debug_rate_limits()
    analyze_current_usage()