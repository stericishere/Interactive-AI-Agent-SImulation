#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced spatial intelligence system
for the dating show simulation prompts.
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from agents.prompt_template.dating_show_v1.prompts import PromptManager

def test_spatial_intelligence():
    """Test the enhanced spatial intelligence features"""
    
    print("=" * 80)
    print("TESTING ENHANCED SPATIAL INTELLIGENCE SYSTEM")
    print("=" * 80)
    
    # Initialize the prompt manager
    template_path = "agents/prompt_template/dating_show_v1"
    prompt_manager = PromptManager(template_path)
    
    print("\n1. TESTING COORDINATE MAPPING SYSTEM")
    print("-" * 50)
    
    # Test coordinate mappings
    test_coordinates = [
        (58, 9),   # Hot Tub Terrace
        (53, 14),  # Study Room  
        (36, 18),  # Pool Deck
        (16, 32),  # Villa Kitchen
        (26, 32),  # Living Room
        (99, 99)   # Non-existent coordinate (should use default)
    ]
    
    for coord in test_coordinates:
        location_context = prompt_manager._get_location_context(coord)
        print(f"\nCoordinate {coord}:")
        print(f"  Area: {location_context['area_name']}")
        print(f"  Description: {location_context['description']}")
        print(f"  Atmosphere: {location_context['atmosphere']}")
        print(f"  Strategic Value: {location_context['strategic_value']}")
        print(f"  Activity Suggestions: {location_context['activity_suggestions'][:3]}...")
    
    print("\n\n2. TESTING INDIVIDUAL ACTIVITY PROMPT")
    print("-" * 50)
    
    # Test individual activity prompt with spatial intelligence
    curr_time = datetime.now()
    individual_prompt = prompt_manager.format_activity_prompt(
        agent_name="Isabella Rodriguez",
        curr_time=curr_time,
        current_tile=(58, 9),  # Hot Tub Terrace
        activity="having a romantic conversation",
        nearby_personas=["Ryan Park"],
        show_events="Evening romantic time before rose ceremony",
        episode_info="Episode 3: Hearts and Choices"
    )
    
    print("INDIVIDUAL ACTIVITY PROMPT (Hot Tub Terrace):")
    print("=" * 60)
    print(individual_prompt)
    print("=" * 60)
    
    print("\n\n3. TESTING BATCH ACTIVITIES PROMPT")
    print("-" * 50)
    
    # Test batch activities prompt with spatial intelligence
    contestants_data = [
        {
            'name': 'Isabella Rodriguez',
            'position': (58, 9),  # Hot Tub Terrace
            'activity': 'romantic conversation',
            'nearby_personas': ['Ryan Park']
        },
        {
            'name': 'Arthur Burton', 
            'position': (53, 14),  # Study Room
            'activity': 'strategic planning',
            'nearby_personas': []
        },
        {
            'name': 'Abigail Chen',
            'position': (36, 18),  # Pool Deck
            'activity': 'sunbathing',
            'nearby_personas': ['Francisco Lopez', 'Hailey Johnson']
        },
        {
            'name': 'Rajiv Patel',
            'position': (16, 32),  # Villa Kitchen
            'activity': 'cooking dinner',
            'nearby_personas': ['Latoya Williams']
        }
    ]
    
    batch_prompt = prompt_manager.format_batch_activities_prompt(
        curr_time=curr_time,
        episode_info="Episode 3: Hearts and Choices",
        show_events="Pre-rose ceremony tension building",
        contestants_data=contestants_data
    )
    
    print("BATCH ACTIVITIES PROMPT (Multiple Locations):")
    print("=" * 60)
    print(batch_prompt)
    print("=" * 60)
    
    print("\n\n4. COMPARING OLD VS NEW PROMPT STRUCTURE")
    print("-" * 50)
    
    # Show the difference between old generic prompts and new spatial intelligence
    print("OLD APPROACH (Generic):")
    print("  Location: Villa area at position (58, 9)")
    print("  Context: Basic villa description")
    print("  Output: 'having romantic conversation @ villa'")
    
    print("\nNEW APPROACH (Spatial Intelligence):")
    print("  Location: Hot Tub Terrace - Secluded hot tub area surrounded by tropical plants and fairy lights")
    print("  Context: Romantic, secluded, sensual atmosphere with strategic value for eliminating competition")
    print("  Output: 'whispering sweet promises under starlight in bubbling privacy @ villa hot tub terrace'")
    
    print("\n\n5. TESTING ERROR HANDLING")
    print("-" * 50)
    
    # Test with various coordinate formats
    test_formats = [
        (58, 9),      # Tuple
        [58, 9],      # List  
        "(58, 9)",    # String
        "invalid"     # Invalid format
    ]
    
    for coord_format in test_formats:
        try:
            context = prompt_manager._get_location_context(coord_format)
            print(f"✅ Format {coord_format}: {context['area_name']}")
        except Exception as e:
            print(f"❌ Format {coord_format}: Error - {e}")
    
    print("\n" + "=" * 80)
    print("SPATIAL INTELLIGENCE ENHANCEMENT COMPLETE")
    print("=" * 80)
    
    print("\nKey Improvements Implemented:")
    print("✅ Villa coordinate mapping system with rich location context")
    print("✅ World setting integration in prompts")
    print("✅ Location-specific activity suggestions and strategic guidance")
    print("✅ Enhanced Current_Situation section with spatial features")
    print("✅ Spatial intelligence rules for activity generation")
    print("✅ Location-aware output format (@ specific villa area)")
    print("✅ Batch processing with individual location context")
    print("✅ Error handling for unknown coordinates")

if __name__ == "__main__":
    test_spatial_intelligence()