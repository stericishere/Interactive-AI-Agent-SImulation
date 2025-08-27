#!/usr/bin/env python3
"""
Example API Prompts for OpenRouter - Dating Show Simulation

This shows the exact prompts and API calls being made when the 429 error occurs.
"""

import json
from datetime import datetime

def show_individual_activity_prompt():
    """Example of individual agent activity generation prompt"""
    print("🎬 INDIVIDUAL ACTIVITY PROMPT EXAMPLE")
    print("=" * 60)
    
    # This is what gets sent when Isabella Rodriguez generates her activity
    example_prompt = """generate_activity.txt

Variables:
!<INPUT 0>! -- Agent name
!<INPUT 1>! -- Current time
!<INPUT 2>! -- Current location coordinates
!<INPUT 3>! -- Agent's planned activity
!<INPUT 4>! -- Nearby contestants (comma-separated)
!<INPUT 5>! -- Current show events/challenges
!<INPUT 6>! -- Episode information

<commentblockmarker>###</commentblockmarker>

<World_Context>
You are Isabella Rodriguez, a contestant on the dating show "Hearts on Fire."

Villa Setting: You're in a luxurious, isolated villa with a pool, hot tub, campfire area, and several private date spots. The villa is designed for romance, connection, and strategic gameplay.

Show Rules:
1. Form genuine connections with other contestants
2. Participate in all mandatory events and challenges  
3. Give honest confessional interviews
4. Survive weekly rose ceremonies to stay in the game
5. The final couple remaining wins "Hearts on Fire"
</World_Context>

<Current_Situation>
Episode: Hearts on Fire - Episode 1
Time: 02:30 PM at the villa
Your Location: Villa area at position (58, 9)
Your Current Activity: reading a book in the study room
Other Contestants Nearby: Ryan Park, Francisco Lopez
Current Show Events: First rose ceremony approaching tonight
</Current_Situation>

<Instructions>
Based on the villa setting and your current situation, generate a realistic 10-15 word description of what you're doing right now as a dating show contestant.

Focus on authentic dating show activities:
- Romantic conversations and flirting
- Strategic discussions about eliminations
- Building connections and alliances
- Participating in challenges or dates
- Confessional-worthy moments and drama
- Villa life activities (poolside, cooking, relaxing)

The description should:
- Be specific to the Hearts on Fire villa environment
- Reflect dating show dynamics and relationship building
- Consider nearby contestants for potential interactions
- Match the time of day and current show events
- Sound natural and compelling for viewers

Return only the activity description, ending with "@ villa"
</Instructions>

<Activity>
</Activity>"""

    print("PROMPT CONTENT:")
    print(example_prompt)
    print("\n" + "=" * 60)
    
    # Show the API call structure
    api_call_example = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {
                "role": "user", 
                "content": example_prompt
            }
        ],
        "temperature": 0.8,
        "max_tokens": 50
    }
    
    print("API CALL STRUCTURE:")
    print(json.dumps(api_call_example, indent=2))
    print(f"\nESTIMATED TOKENS: {len(example_prompt.split()) * 1.3:.0f} tokens (prompt only)")

def show_batch_activity_prompt():
    """Example of batch processing prompt for all 8 agents"""
    print("\n🎬 BATCH ACTIVITY PROMPT EXAMPLE")
    print("=" * 60)
    
    batch_prompt = """batch_activities.txt

Variables:
!<INPUT 0>! -- Current time
!<INPUT 1>! -- Episode information  
!<INPUT 2>! -- Current show events/challenges
!<INPUT 3>! -- Contestants data (formatted list)

<commentblockmarker>###</commentblockmarker>

<World_Context>
Dating show "Hearts on Fire" - Luxury villa with pool, hot tub, campfire area, and private date spots.

Show Rules:
1. Form genuine connections with other contestants
2. Participate in all mandatory events and challenges
3. Give honest confessional interviews  
4. Survive weekly rose ceremonies to stay in the game
5. The final couple remaining wins "Hearts on Fire"
</World_Context>

<Current_Simulation>
Episode: Hearts on Fire - Episode 1
Time: 02:30 PM at Hearts on Fire villa
Current Show Events: First rose ceremony approaching tonight

Contestants and their current situations:
1. Isabella Rodriguez at position (58, 9)
   - Current activity: reading a book in the study room
   - Others nearby: Ryan Park, Francisco Lopez

2. Ryan Park at position (60, 12)
   - Current activity: working out in the gym
   - Others nearby: Isabella Rodriguez, Arthur Burton

3. Francisco Lopez at position (45, 15)
   - Current activity: cooking lunch in the kitchen
   - Others nearby: Hailey Johnson, Latoya Williams

4. Hailey Johnson at position (43, 20)
   - Current activity: sunbathing by the pool
   - Others nearby: Francisco Lopez, Abigail Chen

5. Latoya Williams at position (50, 25)
   - Current activity: having coffee on the terrace
   - Others nearby: Francisco Lopez, Rajiv Patel

6. Abigail Chen at position (65, 18)
   - Current activity: practicing yoga in the garden
   - Others nearby: Hailey Johnson

7. Arthur Burton at position (55, 8)
   - Current activity: strategizing for tonight's ceremony
   - Others nearby: Ryan Park

8. Rajiv Patel at position (48, 28)
   - Current activity: writing in his journal
   - Others nearby: Latoya Williams
</Current_Simulation>

<Instructions>
Generate realistic 10-15 word activity descriptions for each contestant based on their current situation in the Hearts on Fire villa.

Consider the villa atmosphere and dating show dynamics:
- Romantic conversations and flirting between contestants
- Strategic discussions about rose ceremonies and eliminations  
- Building connections, alliances, and romantic relationships
- Participating in challenges, dates, and villa activities
- Confessional-worthy moments, drama, and emotional conversations
- Time-appropriate activities (poolside relaxation, dinner prep, evening conversations)

Each description should:
- Be specific to the Hearts on Fire show and villa setting
- Reflect authentic dating show contestant behavior
- Consider nearby contestants for potential interactions
- Match the current time and ongoing show events
- Sound natural and engaging for viewers
- End with "@ villa"

Format: Return exactly one numbered description per contestant, in the order provided.
</Instructions>

<Activities>
1.
2.
3.
4.
5.
6.
7.
8.
</Activities>"""

    print("BATCH PROMPT CONTENT:")
    print(batch_prompt)
    print("\n" + "=" * 60)
    
    # Show the API call structure for batch
    batch_api_call = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {
                "role": "user", 
                "content": batch_prompt
            }
        ],
        "temperature": 0.8,
        "max_tokens": 400,
        "extra_headers": {
            "HTTP-Referer": "https://github.com/generative-agents/dating-show",
            "X-Title": "Dating Show Simulation - Batch Processing"
        }
    }
    
    print("BATCH API CALL STRUCTURE:")
    print(json.dumps(batch_api_call, indent=2))
    print(f"\nESTIMATED TOKENS: {len(batch_prompt.split()) * 1.3:.0f} tokens (prompt only)")

def show_token_analysis():
    """Analyze token usage patterns causing the rate limit"""
    print("\n📊 TOKEN USAGE ANALYSIS")
    print("=" * 60)
    
    analysis = {
        "individual_call": {
            "prompt_tokens": 450,
            "max_response_tokens": 50,
            "total_per_call": 500
        },
        "batch_call": {
            "prompt_tokens": 800,
            "max_response_tokens": 400,
            "total_per_call": 1200
        },
        "simulation_patterns": {
            "agents_count": 8,
            "individual_calls_per_step": 8,
            "simulation_steps": 5,
            "total_individual_calls": 40,
            "batch_calls_per_step": 1,
            "total_batch_calls": 5
        }
    }
    
    print("TOKEN USAGE BREAKDOWN:")
    print(f"Individual call: ~{analysis['individual_call']['total_per_call']} tokens")
    print(f"Batch call: ~{analysis['batch_call']['total_per_call']} tokens")
    print(f"Agents per simulation: {analysis['simulation_patterns']['agents_count']}")
    print(f"Simulation steps: {analysis['simulation_patterns']['simulation_steps']}")
    
    print(f"\nCOMPARISON (5 steps):")
    individual_total = (analysis['individual_call']['total_per_call'] * 
                       analysis['simulation_patterns']['total_individual_calls'])
    batch_total = (analysis['batch_call']['total_per_call'] * 
                  analysis['simulation_patterns']['total_batch_calls'])
    
    print(f"Individual approach: {analysis['simulation_patterns']['total_individual_calls']} calls × {analysis['individual_call']['total_per_call']} tokens = ~{individual_total:,} tokens")
    print(f"Batch approach: {analysis['simulation_patterns']['total_batch_calls']} calls × {analysis['batch_call']['total_per_call']} tokens = ~{batch_total:,} tokens")
    print(f"Token savings: {individual_total - batch_total:,} tokens ({((individual_total - batch_total) / individual_total * 100):.1f}% reduction)")

def show_rate_limit_context():
    """Show the rate limiting context that's causing issues"""
    print("\n🚦 RATE LIMIT CONTEXT")
    print("=" * 60)
    
    rate_info = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "daily_limit": 50,
        "current_remaining": 0,
        "reset_time": "Wed Aug 27 20:00:00 EDT 2025",
        "error_message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free models per day"
    }
    
    print(f"Model: {rate_info['model']}")
    print(f"Daily limit: {rate_info['daily_limit']} requests")
    print(f"Remaining today: {rate_info['current_remaining']} requests")
    print(f"Reset time: {rate_info['reset_time']}")
    print(f"Error: {rate_info['error_message']}")
    
    print("\nSOLUTIONS:")
    print("1. Wait until 8 PM EDT for quota reset")
    print("2. Add $10 credit for 1000 requests/day")
    print("3. Use batch processing to reduce API calls 8x")
    print("4. Implement exponential backoff with rate limit headers")

if __name__ == "__main__":
    print("🔍 OpenRouter API Prompt Examples - Dating Show Simulation")
    print("=" * 80)
    
    show_individual_activity_prompt()
    show_batch_activity_prompt()
    show_token_analysis()
    show_rate_limit_context()