#!/usr/bin/env python3
"""
Dating Show Setup Script
Automated setup for the generative agents dating show simulation
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_basic_requirements():
    """Install minimal requirements for dating show"""
    requirements = [
        "requests>=2.25.0",
        "aiohttp>=3.8.0", 
        "openai>=0.27.0"
    ]
    
    for req in requirements:
        if not run_command(f"pip install {req}", f"Installing {req}"):
            return False
    return True

def install_django_optional():
    """Install Django for frontend (optional)"""
    django_requirements = [
        "Django>=3.2,<4.0",
        "django-cors-headers>=3.0"
    ]
    
    print("\n🤔 Would you like to install Django for the web frontend? (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice in ['y', 'yes']:
        for req in django_requirements:
            if not run_command(f"pip install {req}", f"Installing {req}"):
                return False
        return True
    else:
        print("📝 Django installation skipped - you can install it later if needed")
        return True

def test_dating_show():
    """Test if the dating show simulation works"""
    print("\n🧪 Testing dating show simulation...")
    
    dating_show_path = Path("dating_show")
    if not dating_show_path.exists():
        print("❌ dating_show directory not found")
        return False
    
    # Test simple simulation
    test_command = "cd dating_show && python simple_piano_example.py"
    
    try:
        # Run a quick test (just first few steps)
        result = subprocess.run(test_command, shell=True, capture_output=True, text=True, timeout=10)
        if "PIANO Dating Show Simulation" in result.stdout:
            print("✅ Dating show simulation is working!")
            return True
        else:
            print("❌ Dating show simulation test failed")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✅ Dating show simulation started successfully (test timed out as expected)")
        return True
    except Exception as e:
        print(f"❌ Dating show test failed: {e}")
        return False

def setup_environment_variables():
    """Guide user to set up environment variables"""
    print("\n🔑 Environment Setup:")
    print("For full functionality, you may need:")
    print("1. OpenRouter API Key: export OPENROUTER_API_KEY='your_key_here'")
    print("2. OpenAI API Key (alternative): export OPENAI_API_KEY='your_key_here'")
    print("\nYou can set these in your ~/.bashrc or ~/.zshrc file")
    
    # Check if any API keys are already set
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if openrouter_key or openai_key:
        print("✅ API key detected in environment")
    else:
        print("⚠️  No API keys detected - some features may be limited")

def main():
    """Main setup function"""
    print("🌹 Dating Show Simulation Setup 🌹")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install basic requirements
    print("\n📦 Installing basic requirements...")
    if not install_basic_requirements():
        print("❌ Failed to install basic requirements")
        sys.exit(1)
    
    # Install Django (optional)
    if not install_django_optional():
        print("❌ Failed to install Django components")
        sys.exit(1)
    
    # Test the simulation
    if not test_dating_show():
        print("❌ Dating show simulation test failed")
        sys.exit(1)
    
    # Setup environment variables
    setup_environment_variables()
    
    # Final instructions
    print("\n🎉 Setup Complete! 🎉")
    print("\nHow to run the dating show:")
    print("1. Simple demo:     cd dating_show && python simulation.py")
    print("2. PIANO simulation: cd dating_show && python simple_piano_example.py")
    print("3. Django frontend:  cd environment/frontend_server && python manage.py runserver")
    print("\n📚 For more advanced features, see the task.md file")
    print("💡 Tip: Set your OpenRouter API key for AI-powered agents")

if __name__ == "__main__":
    main()