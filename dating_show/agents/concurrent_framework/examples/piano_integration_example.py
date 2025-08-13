"""
File: piano_integration_example.py
Description: Integration example showing how to use concurrent framework with existing PIANO architecture
Enhanced PIANO architecture demonstration
"""

import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Any

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '../../..'))

# Import existing PIANO components
from modules.base_module import BaseModule
from memory_structures.episodic_memory import EpisodicMemory
from memory_structures.semantic_memory import SemanticMemory
from memory_structures.temporal_memory import TemporalMemory
from memory_structures.circular_buffer import CircularBuffer

# Import concurrent framework
from concurrent_framework.concurrent_module_manager import ConcurrentModuleManager, Priority
from concurrent_framework.resource_coordinator import ResourceType, AccessMode
from concurrent_framework.state_coordinator import StateChangeType, SyncPolicy
from concurrent_framework.enhanced_modules.parallel_perception_module import ParallelPerceptionModule


class MockAgentState:
    """Mock agent state that simulates the PIANO agent state structure."""
    
    def __init__(self):
        # Basic agent info
        self.name = "Isabella Rodriguez"
        self.curr_time = datetime.now()
        self.curr_tile = (15, 20)
        self.vision_r = 3
        self.att_bandwidth = 5
        
        # Current activity
        self.act_description = "thinking about breakfast options"
        self.act_event = ("Isabella Rodriguez", "thinks", "breakfast", "considering healthy breakfast choices")
        
        # Memory systems
        self.episodic_memory = EpisodicMemory(max_episodes=100)
        self.semantic_memory = SemanticMemory(max_concepts=500)
        self.temporal_memory = TemporalMemory(retention_hours=2)
        self.circular_buffer = CircularBuffer(max_size=20)
        
        # Spatial memory simulation
        self.maze = MockMaze()
        
        # Daily planning
        self.daily_plan = [
            "wake up and stretch",
            "make healthy breakfast",
            "review dating preferences",
            "engage in conversations with other contestants"
        ]
        
        # Emotional state
        self.emotional_state = {
            "happiness": 0.7,
            "confidence": 0.8,
            "romantic_interest": 0.6,
            "social_energy": 0.7
        }


class MockMaze:
    """Mock maze environment for spatial navigation."""
    
    def __init__(self):
        self.tiles = self._generate_mock_tiles()
    
    def _generate_mock_tiles(self) -> Dict:
        """Generate mock tile data for testing."""
        tiles = {}
        
        # Create a simple villa layout
        for x in range(10, 25):
            for y in range(15, 30):
                tiles[(x, y)] = {
                    "world": "villa",
                    "sector": self._get_sector(x, y),
                    "arena": self._get_arena(x, y),
                    "game_object": self._get_game_object(x, y),
                    "events": self._get_events(x, y),
                    "collision": False
                }
        
        return tiles
    
    def _get_sector(self, x: int, y: int) -> str:
        """Determine sector based on coordinates."""
        if 10 <= x < 17:
            return "living_area"
        elif 17 <= x < 22:
            return "kitchen_dining"
        else:
            return "bedroom_area"
    
    def _get_arena(self, x: int, y: int) -> str:
        """Determine arena based on coordinates."""
        sector = self._get_sector(x, y)
        if sector == "living_area":
            return "main_lounge" if y < 22 else "conversation_nook"
        elif sector == "kitchen_dining":
            return "kitchen" if y < 20 else "dining_room"
        else:
            return "master_bedroom" if x > 23 else "guest_bedroom"
    
    def _get_game_object(self, x: int, y: int) -> str:
        """Get game object at coordinates."""
        arena = self._get_arena(x, y)
        objects = {
            "main_lounge": "sofa",
            "conversation_nook": "armchair", 
            "kitchen": "counter",
            "dining_room": "dining_table",
            "master_bedroom": "bed",
            "guest_bedroom": "desk"
        }
        return objects.get(arena, "floor")
    
    def _get_events(self, x: int, y: int) -> List:
        """Get events happening at coordinates."""
        events = []
        
        # Add some random events for demonstration
        import random
        if random.random() < 0.1:  # 10% chance of events
            other_agents = ["Maria Lopez", "Klaus Mueller", "James Wilson"]
            agent = random.choice(other_agents)
            actions = ["is walking", "is sitting", "is talking", "is cooking"]
            action = random.choice(actions)
            
            events.append((
                f"{agent}:person",
                "is",
                action.split()[1],
                f"{agent} {action} in {self._get_arena(x, y)}"
            ))
        
        return events
    
    def access_tile(self, coordinates):
        """Access tile information."""
        return self.tiles.get(coordinates, {})
    
    def get_nearby_tiles(self, center, radius):
        """Get tiles within radius of center."""
        x, y = center
        nearby = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                
                tile_coords = (x + dx, y + dy)
                if tile_coords in self.tiles:
                    nearby.append(tile_coords)
        
        return nearby


class EnhancedPerceptionModule(BaseModule):
    """Enhanced perception module that integrates with concurrent framework."""
    
    def __init__(self, agent_state, concurrent_manager):
        super().__init__(agent_state)
        self.concurrent_manager = concurrent_manager
        self.perception_count = 0
    
    def run(self):
        """Enhanced perception with concurrent processing."""
        self.perception_count += 1
        
        # Use resource coordinator to safely access memory
        try:
            with self.concurrent_manager.resource_coordinator.acquire_context(
                "episodic_memory", AccessMode.WRITE, "perception_module"
            ) as episodic_mem:
                # Perceive environment using parallel perception
                perception_results = self._parallel_perceive()
                
                # Store significant perceptions in episodic memory
                for result in perception_results:
                    if result['importance'] > 0.5:
                        episodic_mem.add_event(
                            content=result['description'],
                            event_type="perception",
                            importance=result['importance'],
                            metadata={'sensor_type': result['type']}
                        )
                
                return perception_results
                
        except Exception as e:
            print(f"Error in enhanced perception: {e}")
            return []
    
    def _parallel_perceive(self):
        """Simulate parallel perception processing."""
        # Get nearby tiles
        nearby_tiles = self.agent_state.maze.get_nearby_tiles(
            self.agent_state.curr_tile, 
            self.agent_state.vision_r
        )
        
        perceptions = []
        
        # Process spatial perceptions
        for tile_coords in nearby_tiles[:3]:  # Limit for demo
            tile_info = self.agent_state.maze.access_tile(tile_coords)
            if tile_info and tile_info.get('game_object'):
                perceptions.append({
                    'type': 'visual',
                    'description': f"Observed {tile_info['game_object']} at {tile_coords}",
                    'importance': 0.4,
                    'location': tile_coords
                })
        
        # Process social perceptions
        for tile_coords in nearby_tiles:
            tile_info = self.agent_state.maze.access_tile(tile_coords)
            if tile_info and tile_info.get('events'):
                for event in tile_info['events']:
                    if len(event) >= 4:
                        agent_name = event[0].split(':')[0]
                        description = event[3]
                        perceptions.append({
                            'type': 'social',
                            'description': f"Social perception: {description}",
                            'importance': 0.7,
                            'location': tile_coords,
                            'other_agent': agent_name
                        })
        
        return perceptions


class EnhancedPlanningModule(BaseModule):
    """Enhanced planning module with concurrent goal processing."""
    
    def __init__(self, agent_state, concurrent_manager):
        super().__init__(agent_state)
        self.concurrent_manager = concurrent_manager
        self.planning_count = 0
    
    def run(self):
        """Enhanced planning with state coordination."""
        self.planning_count += 1
        
        # Propose state changes for current goals
        current_goal = self._determine_current_goal()
        
        # Use state coordinator to update agent's current goal
        change_id = self.concurrent_manager.state_coordinator.propose_change(
            module_id="planning_module",
            path="act_description", 
            new_value=current_goal,
            change_type=StateChangeType.UPDATE,
            priority=1
        )
        
        return {
            'current_goal': current_goal,
            'planning_iteration': self.planning_count,
            'change_id': change_id
        }
    
    def _determine_current_goal(self):
        """Determine current goal based on agent state and time."""
        time_of_day = self.agent_state.curr_time.hour
        
        if 6 <= time_of_day < 10:
            return "preparing and eating breakfast"
        elif 10 <= time_of_day < 12:
            return "engaging in morning activities and socializing"
        elif 12 <= time_of_day < 14:
            return "having lunch and conversation"
        elif 14 <= time_of_day < 18:
            return "participating in dating show activities"
        elif 18 <= time_of_day < 20:
            return "preparing and sharing dinner"
        elif 20 <= time_of_day < 23:
            return "evening socializing and reflection"
        else:
            return "preparing for rest and sleep"


class EnhancedRetrievalModule(BaseModule):
    """Enhanced memory retrieval with concurrent access."""
    
    def __init__(self, agent_state, concurrent_manager):
        super().__init__(agent_state)
        self.concurrent_manager = concurrent_manager
        self.retrieval_count = 0
    
    def run(self, query_context="current_situation"):
        """Enhanced memory retrieval with resource coordination."""
        self.retrieval_count += 1
        retrieved_memories = {}
        
        try:
            # Concurrent access to multiple memory systems
            tasks = []
            
            # Submit concurrent retrieval tasks
            episodic_task_id = self.concurrent_manager.submit_task(
                module_name="retrieval_module",
                method_name="_retrieve_episodic",
                args=(query_context,),
                priority=Priority.HIGH
            )
            tasks.append(('episodic', episodic_task_id))
            
            semantic_task_id = self.concurrent_manager.submit_task(
                module_name="retrieval_module", 
                method_name="_retrieve_semantic",
                args=(query_context,),
                priority=Priority.NORMAL
            )
            tasks.append(('semantic', semantic_task_id))
            
            temporal_task_id = self.concurrent_manager.submit_task(
                module_name="retrieval_module",
                method_name="_retrieve_temporal", 
                args=(query_context,),
                priority=Priority.NORMAL
            )
            tasks.append(('temporal', temporal_task_id))
            
            return {
                'query_context': query_context,
                'retrieval_iteration': self.retrieval_count,
                'submitted_tasks': [task_id for _, task_id in tasks],
                'task_count': len(tasks)
            }
            
        except Exception as e:
            print(f"Error in enhanced retrieval: {e}")
            return {'error': str(e)}
    
    def _retrieve_episodic(self, query_context):
        """Retrieve from episodic memory."""
        with self.concurrent_manager.resource_coordinator.acquire_context(
            "episodic_memory", AccessMode.READ, "retrieval_module"
        ) as episodic_mem:
            # Get recent episodes
            recent_episodes = episodic_mem.get_recent_episodes(hours_back=1)
            return {
                'type': 'episodic',
                'results': [ep.title for ep in recent_episodes[:3]],
                'count': len(recent_episodes)
            }
    
    def _retrieve_semantic(self, query_context):
        """Retrieve from semantic memory."""
        with self.concurrent_manager.resource_coordinator.acquire_context(
            "semantic_memory", AccessMode.READ, "retrieval_module"  
        ) as semantic_mem:
            # Get activated concepts
            activated = semantic_mem.retrieve_by_activation(threshold=0.3, limit=5)
            return {
                'type': 'semantic',
                'results': [concept.name for concept in activated],
                'count': len(activated)
            }
    
    def _retrieve_temporal(self, query_context):
        """Retrieve from temporal memory."""
        with self.concurrent_manager.resource_coordinator.acquire_context(
            "temporal_memory", AccessMode.READ, "retrieval_module"
        ) as temporal_mem:
            # Get recent memories
            recent_memories = temporal_mem.retrieve_recent_memories(hours_back=1, limit=5)
            return {
                'type': 'temporal',
                'results': [mem['content'] for mem in recent_memories],
                'count': len(recent_memories)
            }


def demonstrate_piano_integration():
    """Demonstrate integration of concurrent framework with PIANO architecture."""
    
    print("🎹 PIANO Concurrent Framework Integration Demo")
    print("=" * 60)
    
    # Initialize agent state
    print("\n1. Initializing Enhanced Agent State...")
    agent_state = MockAgentState()
    print(f"   Agent: {agent_state.name}")
    print(f"   Location: {agent_state.curr_tile}")
    print(f"   Current Activity: {agent_state.act_description}")
    
    # Initialize concurrent framework
    print("\n2. Initializing Concurrent Module Manager...")
    concurrent_manager = ConcurrentModuleManager(
        agent_state=agent_state,
        max_workers=4,
        max_queue_size=20,
        enable_monitoring=True
    )
    
    try:
        # Register memory systems as resources
        print("\n3. Registering Memory Systems as Managed Resources...")
        concurrent_manager.resource_coordinator.register_resource(
            "episodic_memory", agent_state.episodic_memory, ResourceType.EPISODIC_MEMORY
        )
        concurrent_manager.resource_coordinator.register_resource(
            "semantic_memory", agent_state.semantic_memory, ResourceType.SEMANTIC_MEMORY
        )
        concurrent_manager.resource_coordinator.register_resource(
            "temporal_memory", agent_state.temporal_memory, ResourceType.TEMPORAL_MEMORY
        )
        concurrent_manager.resource_coordinator.register_resource(
            "circular_buffer", agent_state.circular_buffer, ResourceType.CIRCULAR_BUFFER
        )
        print("   ✓ All memory systems registered for coordinated access")
        
        # Register enhanced cognitive modules
        print("\n4. Registering Enhanced Cognitive Modules...")
        
        perception_module = EnhancedPerceptionModule(agent_state, concurrent_manager)
        planning_module = EnhancedPlanningModule(agent_state, concurrent_manager) 
        retrieval_module = EnhancedRetrievalModule(agent_state, concurrent_manager)
        
        concurrent_manager.register_module("perception_module", perception_module)
        concurrent_manager.register_module("planning_module", planning_module)
        concurrent_manager.register_module("retrieval_module", retrieval_module)
        
        print("   ✓ Perception Module - parallel sensory processing")
        print("   ✓ Planning Module - state-coordinated goal management")  
        print("   ✓ Retrieval Module - concurrent memory access")
        
        # Register modules with state coordinator
        concurrent_manager.state_coordinator.register_module("perception_module", perception_module)
        concurrent_manager.state_coordinator.register_module("planning_module", planning_module)
        concurrent_manager.state_coordinator.register_module("retrieval_module", retrieval_module)
        
        # Set up state change subscriptions
        concurrent_manager.state_coordinator.subscribe_to_path("retrieval_module", "act_description")
        
        print("\n5. Demonstrating Concurrent Cognitive Processing...")
        
        # Simulate a full cognitive cycle
        print("\n   🧠 Cognitive Cycle 1: Morning Routine")
        print("   " + "-" * 40)
        
        # Submit concurrent tasks for different cognitive processes
        task_results = {}
        
        # Perception task
        perception_task_id = concurrent_manager.submit_task(
            module_name="perception_module",
            method_name="run", 
            priority=Priority.HIGH
        )
        task_results['perception'] = perception_task_id
        print(f"   → Submitted perception task: {perception_task_id}")
        
        # Planning task  
        planning_task_id = concurrent_manager.submit_task(
            module_name="planning_module",
            method_name="run",
            priority=Priority.NORMAL
        )
        task_results['planning'] = planning_task_id
        print(f"   → Submitted planning task: {planning_task_id}")
        
        # Retrieval task
        retrieval_task_id = concurrent_manager.submit_task(
            module_name="retrieval_module", 
            method_name="run",
            args=("breakfast_context",),
            priority=Priority.NORMAL
        )
        task_results['retrieval'] = retrieval_task_id
        print(f"   → Submitted retrieval task: {retrieval_task_id}")
        
        # Wait for tasks to complete
        print(f"\n   ⏳ Processing {len(task_results)} concurrent cognitive tasks...")
        time.sleep(3.0)
        
        # Check task completion
        completed_tasks = 0
        for process_name, task_id in task_results.items():
            status = concurrent_manager.get_task_status(task_id)
            if status:
                print(f"   ✓ {process_name.capitalize()} task completed: {status.value}")
                completed_tasks += 1
            else:
                print(f"   ⚠ {process_name.capitalize()} task status unknown")
        
        print(f"\n   📊 Results: {completed_tasks}/{len(task_results)} tasks completed")
        
        # Demonstrate state coordination
        print("\n6. Demonstrating State Coordination...")
        print("   " + "-" * 40)
        
        # Show current agent state
        print(f"   Current agent description: {agent_state.act_description}")
        
        # Check state coordination status
        coord_status = concurrent_manager.state_coordinator.get_coordination_status()
        print(f"   State version: {coord_status['state_version']}")
        print(f"   Registered modules: {coord_status['registered_modules']}")
        print(f"   Pending changes: {coord_status['pending_changes']}")
        
        # Demonstrate resource coordination metrics
        print("\n7. Resource Coordination Metrics...")
        print("   " + "-" * 40)
        
        resource_metrics = concurrent_manager.resource_coordinator.get_coordinator_metrics()
        print(f"   Total resource requests: {resource_metrics.total_requests}")
        print(f"   Successful acquisitions: {resource_metrics.successful_acquisitions}")
        print(f"   Average wait time: {resource_metrics.average_wait_time:.3f}s")
        
        # Show system status
        print("\n8. System Performance Status...")
        print("   " + "-" * 40)
        
        system_status = concurrent_manager.get_system_status()
        print(f"   Registered modules: {system_status['registered_modules']}")
        print(f"   Completed tasks: {system_status['completed_tasks']}")
        print(f"   System uptime: {system_status['uptime_seconds']:.1f}s")
        
        executor_metrics = concurrent_manager.module_executor.get_metrics()
        print(f"   Tasks executed: {executor_metrics.tasks_executed}")
        print(f"   Tasks failed: {executor_metrics.tasks_failed}")
        print(f"   Peak concurrent tasks: {executor_metrics.peak_thread_count}")
        
        # Demonstrate parallel perception module
        print("\n9. Parallel Perception Module Demo...")
        print("   " + "-" * 40)
        
        parallel_perception = ParallelPerceptionModule(
            agent_state=agent_state,
            max_workers=2,
            attention_bandwidth=4,
            vision_radius=3
        )
        
        try:
            concurrent_manager.register_module("parallel_perception", parallel_perception)
            
            # Submit parallel perception task
            parallel_task_id = concurrent_manager.submit_task(
                module_name="parallel_perception",
                method_name="run",
                priority=Priority.CRITICAL
            )
            
            print(f"   → Submitted parallel perception task: {parallel_task_id}")
            
            # Wait for processing
            time.sleep(2.0)
            
            # Check perception metrics
            perception_metrics = parallel_perception.get_perception_metrics()
            print(f"   ✓ Tasks processed: {perception_metrics.total_tasks_processed}")
            print(f"   ✓ Average processing time: {perception_metrics.average_processing_time:.3f}s")
            print(f"   ✓ Peak concurrent processing: {perception_metrics.concurrent_processing_peak}")
            
            # Check processing status
            perception_status = parallel_perception.get_processing_status()
            print(f"   ✓ Attention bandwidth: {perception_status['attention_bandwidth']}")
            print(f"   ✓ Vision radius: {perception_status['vision_radius']}")
            print(f"   ✓ Caching enabled: {perception_status['caching_enabled']}")
            
        finally:
            parallel_perception.shutdown()
        
        # Final demonstration: Concurrent cognitive cycle
        print("\n10. Full Concurrent Cognitive Cycle...")
        print("    " + "-" * 39)
        
        # Submit a full set of cognitive tasks concurrently
        cycle_tasks = []
        
        for i in range(2):  # Two cognitive cycles
            print(f"\n    Cycle {i+1}:")
            
            # Perception
            p_task = concurrent_manager.submit_task(
                "perception_module", "run", priority=Priority.HIGH
            )
            cycle_tasks.append(('perception', p_task))
            print(f"    → Perception: {p_task}")
            
            # Planning  
            plan_task = concurrent_manager.submit_task(
                "planning_module", "run", priority=Priority.NORMAL
            )
            cycle_tasks.append(('planning', plan_task))
            print(f"    → Planning: {plan_task}")
            
            # Retrieval
            ret_task = concurrent_manager.submit_task(
                "retrieval_module", "run", 
                args=(f"cycle_{i+1}_context",), priority=Priority.NORMAL
            )
            cycle_tasks.append(('retrieval', ret_task))
            print(f"    → Retrieval: {ret_task}")
        
        print(f"\n    ⏳ Processing {len(cycle_tasks)} tasks across 2 cognitive cycles...")
        
        # Wait for all cycles to complete
        time.sleep(4.0)
        
        # Check final results
        final_status = concurrent_manager.get_system_status()
        print(f"\n    📈 Final Results:")
        print(f"    • Total tasks completed: {final_status['completed_tasks']}")
        print(f"    • System ran for: {final_status['uptime_seconds']:.1f}s")
        
        final_metrics = concurrent_manager.module_executor.get_metrics()
        throughput = final_metrics.tasks_executed / final_status['uptime_seconds']
        print(f"    • Average throughput: {throughput:.2f} tasks/second")
        print(f"    • Total execution time: {final_metrics.total_execution_time:.2f}s")
        
        efficiency = (final_metrics.total_execution_time / final_status['uptime_seconds']) * 100
        print(f"    • CPU efficiency: {efficiency:.1f}% (concurrent processing benefit)")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean shutdown
        print(f"\n11. Shutting down concurrent framework...")
        concurrent_manager.shutdown(timeout=10.0)
        print("    ✓ Shutdown complete")
    
    print(f"\n🎉 PIANO Concurrent Framework Integration Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_piano_integration()