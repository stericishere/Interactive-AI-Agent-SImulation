
"""
File: simulation.py
Description: Simple dating show simulation demo.
"""

import time

def run_simple_demo():
    """
    Run a simple demo showing the dating show concept.
    """
    print("🌹 Dating Show Simulation Demo 🌹")
    print("=" * 40)
    
    # Simple agent representations
    agents = {
        "Alice": {"role": "contestant", "status": "active", "roses": 2},
        "Bob": {"role": "contestant", "status": "active", "roses": 1}, 
        "Charlie": {"role": "host", "status": "active", "roses": 0}
    }
    
    game_states = ["SHOW_START", "ROUND_START", "DATE", "ELIMINATION"]
    current_state_idx = 0
    
    print("\n🎬 Starting Dating Show Simulation...")
    print(f"Contestants: {[name for name, info in agents.items() if info['role'] == 'contestant']}")
    print(f"Host: {[name for name, info in agents.items() if info['role'] == 'host'][0]}")
    
    # Simple simulation loop
    for step in range(6):
        print(f"\n--- Step {step + 1}: {game_states[current_state_idx]} ---")
        
        if game_states[current_state_idx] == "SHOW_START":
            print("🎙️  Charlie: Welcome to the Dating Show!")
            print("💕 Alice: I'm so excited to find love!")
            print("💪 Bob: Ready to compete for the final rose!")
            
        elif game_states[current_state_idx] == "ROUND_START":
            print("🎙️  Charlie: Time for today's dating challenge!")
            print("💕 Alice: I hope I get chosen for a date...")
            print("💪 Bob: I'm going to win this challenge!")
            
        elif game_states[current_state_idx] == "DATE":
            print("💕 Alice and Bob go on a romantic date...")
            print("🍷 They share stories over dinner")
            print("✨ Connection level increasing...")
            agents["Alice"]["roses"] += 1
            
        elif game_states[current_state_idx] == "ELIMINATION":
            print("🌹 Charlie: It's time for the elimination ceremony...")
            print("💔 One contestant will be sent home tonight...")
            if step == 5:  # Final elimination
                print("🏆 Alice wins the final rose!")
                agents["Bob"]["status"] = "eliminated"
            
        # Show current status
        print("\n📊 Current Status:")
        for name, info in agents.items():
            status_emoji = "✅" if info["status"] == "active" else "❌"
            print(f"  {status_emoji} {name}: {info['role']}, roses: {info['roses']}")
            
        # Advance state
        current_state_idx = (current_state_idx + 1) % len(game_states)
        time.sleep(2)
    
    print("\n🎉 Simulation Complete!")
    winner = [name for name, info in agents.items() if info["status"] == "active" and info["role"] == "contestant"]
    if winner:
        print(f"🏆 Winner: {winner[0]}")
    
    print("\n💡 To run the full PIANO agent simulation:")
    print("   cd ../reverie/backend_server")
    print("   python reverie.py")

if __name__ == "__main__":
    run_simple_demo()

