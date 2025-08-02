"""
File: langgraph_integration_example.py
Description: Example integration of Enhanced PIANO architecture with LangGraph StateGraph.
Demonstrates how to use the enhanced memory systems in a StateGraph workflow.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

# LangGraph imports (would be actual imports in real implementation)
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.store.sqlite import SqliteStore
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Mock classes for demonstration
    LANGGRAPH_AVAILABLE = False
    
    class StateGraph:
        def __init__(self, state_schema): self.state_schema = state_schema
        def add_node(self, name, func): pass
        def add_edge(self, from_node, to_node): pass
        def set_entry_point(self, node): pass
        def set_finish_point(self, node): pass
        def compile(self, checkpointer=None, store=None): return MockCompiledGraph()
    
    class MockCompiledGraph:
        async def ainvoke(self, state, config=None): return state
    
    class SqliteSaver: pass
    class SqliteStore: pass
    START, END = "START", "END"

# Import our enhanced agent state
from .enhanced_agent_state import EnhancedAgentState, EnhancedAgentStateManager


class EnhancedPIANOAgent:
    """
    Enhanced PIANO agent with LangGraph StateGraph integration.
    Demonstrates Phase 1 implementation with enhanced memory architecture.
    """
    
    def __init__(self, agent_id: str, name: str, personality_traits: Dict[str, float]):
        """
        Initialize Enhanced PIANO Agent.
        
        Args:
            agent_id: Unique agent identifier
            name: Agent name
            personality_traits: Personality trait scores
        """
        self.agent_id = agent_id
        self.name = name
        
        # Initialize enhanced state manager
        self.state_manager = EnhancedAgentStateManager(
            agent_id, name, personality_traits,
            working_memory_size=20,
            temporal_retention_hours=2,
            max_episodes=100,
            max_concepts=500
        )
        
        # Create StateGraph
        self.graph = self._create_state_graph()
        
        # Performance tracking
        self.decision_count = 0
        self.total_decision_time = 0.0
    
    def _create_state_graph(self) -> StateGraph:
        """
        Create LangGraph StateGraph for enhanced PIANO architecture.
        
        Returns:
            Compiled StateGraph
        """
        # Create StateGraph with EnhancedAgentState schema
        graph = StateGraph(EnhancedAgentState)
        
        # Add cognitive module nodes
        graph.add_node("perceive", self._perception_module)
        graph.add_node("plan", self._planning_module)
        graph.add_node("execute", self._execution_module)
        graph.add_node("reflect", self._reflection_module)
        graph.add_node("socialize", self._social_module)
        graph.add_node("specialize", self._specialization_module)
        graph.add_node("consolidate_memory", self._memory_consolidation_module)
        graph.add_node("update_cultural", self._cultural_update_module)
        
        # Define execution flow with parallel branches
        graph.add_edge(START, "perceive")
        
        # Parallel execution of core modules
        graph.add_edge("perceive", "plan")
        graph.add_edge("perceive", "socialize")  # Parallel branch
        graph.add_edge("perceive", "specialize")  # Parallel branch
        
        # Sequential execution after parallel modules
        graph.add_edge("plan", "execute")
        graph.add_edge("socialize", "execute")
        graph.add_edge("specialize", "execute")
        
        graph.add_edge("execute", "reflect")
        graph.add_edge("reflect", "consolidate_memory")
        graph.add_edge("consolidate_memory", "update_cultural")
        graph.add_edge("update_cultural", END)
        
        # Compile with checkpointer for persistence
        if LANGGRAPH_AVAILABLE:
            checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{self.agent_id}_checkpoints.db")
            store = SqliteStore.from_conn_string(f"sqlite:///{self.agent_id}_store.db")
            return graph.compile(checkpointer=checkpointer, store=store)
        else:
            return graph.compile()
    
    async def _perception_module(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """
        Perception module: Process environmental inputs and update working memory.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        start_time = datetime.now()
        
        # Simulate environmental perception
        environment_data = self._get_environment_data(state)
        
        # Add perception to memory
        perception_content = f"Perceived environment at {state['current_location']}: {environment_data}"
        self.state_manager.add_memory(perception_content, "perception", 0.5, 
                                    {"location": state["current_location"]})
        
        # Update state with new perceptions
        updated_state = self.state_manager.state.copy()
        
        # Track performance
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self._update_performance_metrics(decision_time)
        
        return updated_state
    
    async def _planning_module(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """
        Planning module: Generate goals and action plans based on current state.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        start_time = datetime.now()
        
        # Retrieve relevant memories for planning
        recent_memories = self.state_manager.circular_buffer.get_recent_memories(5)
        
        # Generate goals based on personality and recent experiences
        new_goals = self._generate_goals(state, recent_memories)
        
        # Add planning thoughts to memory
        planning_content = f"Planning session generated goals: {', '.join(new_goals)}"
        self.state_manager.add_memory(planning_content, "thought", 0.7,
                                    {"planning_session": True, "goals": new_goals})
        
        # Update state with new goals
        updated_state = self.state_manager.state.copy()
        updated_state["goals"] = new_goals
        
        # Track performance
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self._update_performance_metrics(decision_time)
        
        return updated_state
    
    async def _execution_module(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """
        Execution module: Execute planned actions.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        start_time = datetime.now()
        
        # Select action based on current goals and context
        action = self._select_action(state)
        
        # Execute action
        action_result = self._execute_action(action, state)
        
        # Add action to memory
        action_content = f"Executed action: {action} with result: {action_result}"
        self.state_manager.add_memory(action_content, "action", 0.6,
                                    {"action": action, "result": action_result})
        
        # Update specialization based on action
        skills_gained = self._calculate_skill_gain(action)
        if skills_gained:
            self.state_manager.update_specialization(action, skills_gained)
        
        # Update state
        updated_state = self.state_manager.state.copy()
        updated_state["current_activity"] = action
        
        # Track performance
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self._update_performance_metrics(decision_time)
        
        return updated_state
    
    async def _reflection_module(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """
        Reflection module: Analyze recent experiences and generate insights.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        start_time = datetime.now()
        
        # Retrieve recent episodes for reflection
        recent_episodes = list(self.state_manager.episodic_memory.episodes.values())[-3:]
        
        # Generate reflection insights
        insights = self._generate_reflection_insights(recent_episodes, state)
        
        # Add reflection to memory
        if insights:
            reflection_content = f"Reflection insights: {'; '.join(insights)}"
            self.state_manager.add_memory(reflection_content, "thought", 0.8,
                                        {"reflection": True, "insights": insights})
        
        # Update emotional state based on reflection
        emotional_changes = self._calculate_emotional_impact(insights)
        if emotional_changes:
            self.state_manager.update_emotional_state(emotional_changes)
        
        # Track performance
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self._update_performance_metrics(decision_time)
        
        return self.state_manager.state.copy()
    
    async def _social_module(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """
        Social module: Process social interactions and relationships.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        start_time = datetime.now()
        
        # Check for social interaction opportunities
        social_context = self._analyze_social_context(state)
        
        if social_context.get("interaction_opportunity"):
            partner = social_context["potential_partner"]
            interaction_type = social_context["interaction_type"]
            
            # Process social interaction
            self.state_manager.process_social_interaction(
                partner, interaction_type, 
                f"Social interaction with {partner}",
                social_context.get("emotional_impact", 0.0)
            )
        
        # Track performance
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self._update_performance_metrics(decision_time)
        
        return self.state_manager.state.copy()
    
    async def _specialization_module(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """
        Specialization module: Update role and skill development.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        start_time = datetime.now()
        
        # Analyze recent actions for role consistency
        recent_actions = self.state_manager.circular_buffer.get_memories_by_type("action")
        
        # Update role if pattern emerges
        potential_role = self._analyze_role_emergence(recent_actions, state)
        if potential_role and potential_role != state["specialization"].current_role:
            self.state_manager.specialization.role_history.append(
                self.state_manager.specialization.current_role
            )
            self.state_manager.specialization.current_role = potential_role
            self.state_manager.specialization.last_role_change = datetime.now()
            
            # Add role change to memory
            role_change_content = f"Role evolved from {state['specialization'].current_role} to {potential_role}"
            self.state_manager.add_memory(role_change_content, "thought", 0.9,
                                        {"role_change": True, "new_role": potential_role})
        
        # Track performance
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self._update_performance_metrics(decision_time)
        
        return self.state_manager.state.copy()
    
    async def _memory_consolidation_module(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """
        Memory consolidation module: Background memory processing.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        start_time = datetime.now()
        
        # Perform memory consolidation
        cleanup_stats = self.state_manager.cleanup_memories()
        
        # Update activation decay in semantic memory
        self.state_manager.semantic_memory.update_activation_decay()
        
        # Consolidate episodic memories if needed
        consolidated_episodes = self.state_manager.episodic_memory.consolidate_memories()
        
        if cleanup_stats or consolidated_episodes:
            consolidation_content = f"Memory consolidation: {cleanup_stats}"
            self.state_manager.add_memory(consolidation_content, "system", 0.3,
                                        {"consolidation_stats": cleanup_stats})
        
        # Track performance
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self._update_performance_metrics(decision_time)
        
        return self.state_manager.state.copy()
    
    async def _cultural_update_module(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """
        Cultural update module: Process cultural meme propagation.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated agent state
        """
        start_time = datetime.now()
        
        # Check for cultural meme exposure
        cultural_exposure = self._check_cultural_exposure(state)
        
        if cultural_exposure:
            for meme_id, influence in cultural_exposure.items():
                self.state_manager.update_cultural_influence(meme_id, influence)
        
        # Track performance
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self._update_performance_metrics(decision_time)
        
        return self.state_manager.state.copy()
    
    def _get_environment_data(self, state: EnhancedAgentState) -> str:
        """Simulate environmental perception."""
        location = state["current_location"]
        time_of_day = datetime.now().hour
        
        if 6 <= time_of_day < 12:
            return f"Morning atmosphere at {location}, other agents are active"
        elif 12 <= time_of_day < 18:
            return f"Afternoon activities at {location}, social opportunities available"
        else:
            return f"Evening ambiance at {location}, relaxed social setting"
    
    def _generate_goals(self, state: EnhancedAgentState, recent_memories: List[Dict]) -> List[str]:
        """Generate goals based on current state and recent experiences."""
        goals = []
        
        # Basic survival goals
        if not any("eat" in mem["content"].lower() for mem in recent_memories[-3:]):
            goals.append("find_food")
        
        # Social goals based on personality
        if state["personality_traits"].get("extroversion", 0.5) > 0.6:
            goals.append("social_interaction")
        
        # Specialization goals
        current_role = state["specialization"].current_role
        if current_role == "contestant":
            goals.append("romantic_connection")
        
        return goals
    
    def _select_action(self, state: EnhancedAgentState) -> str:
        """Select action based on current goals and context."""
        goals = state["goals"]
        
        if "find_food" in goals:
            return "go_to_kitchen"
        elif "social_interaction" in goals:
            return "approach_other_agent"
        elif "romantic_connection" in goals:
            return "flirtation_attempt"
        else:
            return "idle_observation"
    
    def _execute_action(self, action: str, state: EnhancedAgentState) -> str:
        """Execute action and return result."""
        action_results = {
            "go_to_kitchen": "Found snacks and coffee",
            "approach_other_agent": "Started conversation with nearby agent",
            "flirtation_attempt": "Exchanged meaningful glances",
            "idle_observation": "Observed social dynamics"
        }
        
        return action_results.get(action, "Action completed")
    
    def _calculate_skill_gain(self, action: str) -> Optional[Dict[str, float]]:
        """Calculate skill gains from action."""
        skill_gains = {
            "approach_other_agent": {"communication": 0.02, "confidence": 0.01},
            "flirtation_attempt": {"charisma": 0.03, "emotional_intelligence": 0.01},
            "go_to_kitchen": {"self_care": 0.01}
        }
        
        return skill_gains.get(action)
    
    def _generate_reflection_insights(self, recent_episodes: List, state: EnhancedAgentState) -> List[str]:
        """Generate insights from recent experiences."""
        insights = []
        
        # Analyze social interactions
        social_episodes = [ep for ep in recent_episodes if "conversation" in ep.title.lower()]
        if len(social_episodes) > 2:
            insights.append("I've been very social recently, building good connections")
        
        # Analyze role consistency
        role_consistency = state["specialization"].role_consistency_score
        if role_consistency < 0.6:
            insights.append("My actions don't align well with my role, should be more consistent")
        
        return insights
    
    def _calculate_emotional_impact(self, insights: List[str]) -> Optional[Dict[str, float]]:
        """Calculate emotional changes from insights."""
        emotional_changes = {}
        
        for insight in insights:
            if "social" in insight.lower() and "good" in insight.lower():
                emotional_changes["happiness"] = 0.1
                emotional_changes["confidence"] = 0.05
            elif "consistent" in insight.lower():
                emotional_changes["anxiety"] = 0.05
        
        return emotional_changes if emotional_changes else None
    
    def _analyze_social_context(self, state: EnhancedAgentState) -> Dict[str, Any]:
        """Analyze current social context for interaction opportunities."""
        # Simulate social context analysis
        if state["current_location"] == "villa" and datetime.now().hour < 22:
            return {
                "interaction_opportunity": True,
                "potential_partner": "Maria",  # Simulated
                "interaction_type": "casual_conversation",
                "emotional_impact": 0.2
            }
        
        return {"interaction_opportunity": False}
    
    def _analyze_role_emergence(self, recent_actions: List[Dict], state: EnhancedAgentState) -> Optional[str]:
        """Analyze if a new role is emerging from recent actions."""
        if len(recent_actions) < 5:
            return None
        
        # Simple role analysis based on action patterns
        social_actions = sum(1 for action in recent_actions 
                           if "social" in action["content"].lower() or "conversation" in action["content"].lower())
        
        if social_actions > len(recent_actions) * 0.7:
            return "social_connector"
        
        return None
    
    def _check_cultural_exposure(self, state: EnhancedAgentState) -> Optional[Dict[str, float]]:
        """Check for cultural meme exposure."""
        # Simulate cultural exposure
        if len(state["recent_interactions"]) > 0:
            return {"confidence_meme": 0.1, "romance_meme": 0.05}
        
        return None
    
    def _update_performance_metrics(self, decision_time_ms: float) -> None:
        """Update performance tracking metrics."""
        self.decision_count += 1
        self.total_decision_time += decision_time_ms
        
        avg_decision_time = self.total_decision_time / self.decision_count
        self.state_manager.performance.decision_latency = avg_decision_time
        
        # Update performance in state
        self.state_manager.state["performance"] = self.state_manager.performance
    
    async def run_cognitive_cycle(self, input_data: Optional[Dict[str, Any]] = None) -> EnhancedAgentState:
        """
        Run a complete cognitive cycle through the StateGraph.
        
        Args:
            input_data: Optional input data for the cycle
        
        Returns:
            Updated agent state
        """
        # Prepare initial state
        initial_state = self.state_manager.state.copy()
        if input_data:
            initial_state.update(input_data)
        
        # Run through StateGraph
        config = {"configurable": {"thread_id": self.agent_id}}
        result_state = await self.graph.ainvoke(initial_state, config=config)
        
        # Update state manager
        self.state_manager.state = result_state
        
        return result_state
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get comprehensive agent summary."""
        return {
            "agent_info": {
                "id": self.agent_id,
                "name": self.name,
                "current_role": self.state_manager.specialization.current_role,
                "expertise_level": self.state_manager.specialization.expertise_level
            },
            "memory_summary": self.state_manager.get_memory_summary(),
            "performance": {
                "decision_count": self.decision_count,
                "avg_decision_time_ms": self.state_manager.performance.decision_latency,
                "coherence_score": self.state_manager.performance.coherence_score,
                "social_integration": self.state_manager.performance.social_integration
            },
            "current_state": {
                "location": self.state_manager.state["current_location"],
                "activity": self.state_manager.state["current_activity"],
                "goals": self.state_manager.state["goals"],
                "emotional_state": self.state_manager.state["emotional_state"]
            }
        }


async def main():
    """Example usage of Enhanced PIANO Agent with LangGraph."""
    print("Enhanced PIANO Agent with LangGraph Integration Example")
    print("=" * 60)
    
    # Create enhanced agent
    agent = EnhancedPIANOAgent(
        "agent_001",
        "Isabella Rodriguez",
        {
            "confidence": 0.8,
            "extroversion": 0.9,
            "openness": 0.7,
            "emotional_intelligence": 0.6
        }
    )
    
    print(f"Created agent: {agent.name}")
    print(f"Initial state: {agent.state_manager.specialization.current_role}")
    
    # Run several cognitive cycles
    for cycle in range(3):
        print(f"\n--- Cognitive Cycle {cycle + 1} ---")
        
        # Add some environmental input
        input_data = {
            "current_time": datetime.now(),
            "current_location": "villa_living_room" if cycle % 2 == 0 else "villa_kitchen"
        }
        
        # Run cognitive cycle
        result_state = await agent.run_cognitive_cycle(input_data)
        
        # Print results
        print(f"Location: {result_state['current_location']}")
        print(f"Activity: {result_state['current_activity']}")
        print(f"Goals: {result_state['goals']}")
        print(f"Working Memory: {len(result_state['working_memory'])} items")
        print(f"Decision Time: {agent.state_manager.performance.decision_latency:.2f}ms")
        print(f"Coherence Score: {agent.state_manager.performance.coherence_score:.2f}")
    
    # Print final summary
    print("\n--- Final Agent Summary ---")
    summary = agent.get_agent_summary()
    
    print(f"Agent: {summary['agent_info']['name']}")
    print(f"Role: {summary['agent_info']['current_role']}")
    print(f"Expertise: {summary['agent_info']['expertise_level']:.2f}")
    print(f"Total Decisions: {summary['performance']['decision_count']}")
    print(f"Average Decision Time: {summary['performance']['avg_decision_time_ms']:.2f}ms")
    print(f"Social Integration: {summary['performance']['social_integration']:.2f}")
    
    print("\nMemory Systems:")
    memory_summary = summary['memory_summary']
    print(f"  Working Memory: {memory_summary['working_memory']['total_memories']} items")
    print(f"  Temporal Memory: {memory_summary['temporal_memory']['total_memories']} items")
    print(f"  Episodes: {memory_summary['episodic_memory']['total_episodes']}")
    print(f"  Concepts: {memory_summary['semantic_memory']['total_concepts']}")
    
    print(f"\nCurrent Emotional State: {summary['current_state']['emotional_state']}")
    
    print("\n✅ Enhanced PIANO Agent demonstration completed!")
    print("Key Features Demonstrated:")
    print("  • Multi-layered memory architecture")
    print("  • LangGraph StateGraph integration")
    print("  • Concurrent cognitive module execution")
    print("  • Specialization and skill development")
    print("  • Performance monitoring and optimization")
    print("  • Cultural and governance system integration")


if __name__ == "__main__":
    asyncio.run(main())