"""
File: enhanced_langgraph_integration.py
Description: Complete LangGraph StateGraph integration for Enhanced PIANO architecture.
Creates the actual StateGraph using all enhanced memory systems with PostgreSQL and Store API.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Annotated, TypedDict, Union
from dataclasses import asdict
import logging
import os

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresCheckpointer
from langgraph.store.postgres import PostgresStore
from langgraph.pregel import Pregel

# Import enhanced memory and state components
from .enhanced_agent_state import EnhancedAgentState, EnhancedAgentStateManager, create_enhanced_agent_state
from .memory_structures.postgres_persistence import PostgresMemoryPersistence, PostgresConfig
from .memory_structures.store_integration import MemoryStoreIntegration
from .memory_structures.performance_monitor import MemoryPerformanceMonitor, track_performance
from .memory_structures.circular_buffer import CircularBufferReducer

# Import cognitive modules
from .modules.langgraph_base_module import LangGraphBaseModule, ModuleExecutionConfig, ExecutionTimeScale, ModulePriority
from .modules.memory_consolidation_module import MemoryConsolidationModule
from .modules.memory_retrieval import MemoryRetrievalModule
from .modules.memory_association import MemoryAssociationModule
from .modules.specialization_detection import SpecializationDetectionModule


class EnhancedPIANOStateGraph:
    """
    Complete LangGraph StateGraph implementation for Enhanced PIANO architecture.
    Integrates memory systems, cognitive modules, and cross-agent coordination.
    """
    
    def __init__(self, 
                 postgres_config: PostgresConfig = None,
                 enable_performance_monitoring: bool = True,
                 enable_store_integration: bool = True):
        """
        Initialize Enhanced PIANO StateGraph.
        
        Args:
            postgres_config: PostgreSQL configuration
            enable_performance_monitoring: Enable performance tracking
            enable_store_integration: Enable Store API integration
        """
        self.postgres_config = postgres_config or PostgresConfig()
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_store_integration = enable_store_integration
        
        # Core components
        self.postgres_persistence: Optional[PostgresMemoryPersistence] = None
        self.store_integration: Optional[MemoryStoreIntegration] = None
        self.performance_monitor: Optional[MemoryPerformanceMonitor] = None
        
        # LangGraph components
        self.checkpointer: Optional[PostgresCheckpointer] = None
        self.store: Optional[PostgresStore] = None
        self.graph: Optional[Pregel] = None
        
        # Agent state managers
        self.agent_managers: Dict[str, EnhancedAgentStateManager] = {}
        
        self.logger = logging.getLogger(f"{__name__}.EnhancedPIANOStateGraph")
        self.logger.info("Enhanced PIANO StateGraph initialized")
    
    async def initialize(self) -> None:
        """Initialize all components and create StateGraph."""
        try:
            self.logger.info("Initializing Enhanced PIANO StateGraph components...")
            
            # Initialize PostgreSQL persistence
            self.postgres_persistence = PostgresMemoryPersistence(self.postgres_config)
            await self.postgres_persistence.initialize()
            
            # Initialize performance monitoring
            if self.enable_performance_monitoring:
                from .memory_structures.performance_monitor import create_performance_monitor
                self.performance_monitor = create_performance_monitor()
            
            # Initialize LangGraph components
            await self._initialize_langgraph_components()
            
            # Initialize Store API integration
            if self.enable_store_integration:
                self.store_integration = MemoryStoreIntegration(
                    store=self.store,
                    postgres_persistence=self.postgres_persistence
                )
            
            # Create and compile StateGraph
            self.graph = self._create_state_graph()
            
            self.logger.info("Enhanced PIANO StateGraph initialization complete")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced PIANO StateGraph: {str(e)}")
            raise
    
    async def _initialize_langgraph_components(self) -> None:
        """Initialize LangGraph checkpointer and store."""
        # Create PostgreSQL checkpointer
        self.checkpointer = PostgresCheckpointer(
            connection_string=self.postgres_config.dsn,
            schema_name="public"
        )
        
        # Create PostgreSQL store
        self.store = PostgresStore(
            connection_string=self.postgres_config.dsn,
            schema_name="public"
        )
        
        # Setup schemas if needed
        await self.checkpointer.setup()
        await self.store.setup()
    
    def _create_state_graph(self) -> Pregel:
        """Create and compile the Enhanced PIANO StateGraph."""
        # Create the graph builder
        graph_builder = StateGraph(EnhancedAgentState)
        
        # Add all cognitive processing nodes
        graph_builder.add_node("perception", self._perception_node)
        graph_builder.add_node("memory_retrieval", self._memory_retrieval_node)
        graph_builder.add_node("planning", self._planning_node)
        graph_builder.add_node("decision_making", self._decision_making_node)
        graph_builder.add_node("action_execution", self._action_execution_node)
        graph_builder.add_node("social_processing", self._social_processing_node)
        graph_builder.add_node("reflection", self._reflection_node)
        
        # Add memory management nodes
        graph_builder.add_node("memory_consolidation", self._memory_consolidation_node)
        graph_builder.add_node("memory_association", self._memory_association_node)
        
        # Add specialization node
        graph_builder.add_node("specialization_update", self._specialization_update_node)
        
        # Add cultural/governance nodes
        graph_builder.add_node("cultural_processing", self._cultural_processing_node)
        graph_builder.add_node("governance_processing", self._governance_processing_node)
        
        # Define execution flow with parallel branches
        graph_builder.add_edge(START, "perception")
        graph_builder.add_edge("perception", "memory_retrieval")
        
        # Parallel processing branches
        graph_builder.add_edge("memory_retrieval", "planning")
        graph_builder.add_edge("memory_retrieval", "social_processing")
        graph_builder.add_edge("memory_retrieval", "cultural_processing")
        
        # Convergence to decision making
        graph_builder.add_edge("planning", "decision_making")
        graph_builder.add_edge("social_processing", "decision_making")
        graph_builder.add_edge("cultural_processing", "decision_making")
        
        # Action and reflection
        graph_builder.add_edge("decision_making", "action_execution")
        graph_builder.add_edge("action_execution", "reflection")
        
        # Memory consolidation (parallel to reflection)
        graph_builder.add_edge("action_execution", "memory_consolidation")
        graph_builder.add_edge("memory_consolidation", "memory_association")
        
        # Specialization and governance updates
        graph_builder.add_edge("reflection", "specialization_update")
        graph_builder.add_edge("reflection", "governance_processing")
        
        # End conditions
        graph_builder.add_edge("specialization_update", END)
        graph_builder.add_edge("governance_processing", END)
        graph_builder.add_edge("memory_association", END)
        
        # Compile with checkpointer and store
        return graph_builder.compile(
            checkpointer=self.checkpointer,
            store=self.store
        )
    
    async def get_agent_manager(self, agent_id: str, name: str, personality_traits: Dict[str, float]) -> EnhancedAgentStateManager:
        """Get or create agent state manager."""
        if agent_id not in self.agent_managers:
            # Ensure agent exists in database
            await self.postgres_persistence.ensure_agent_exists(agent_id, name, personality_traits)
            
            # Create agent manager
            manager = create_enhanced_agent_state(agent_id, name, personality_traits)
            self.agent_managers[agent_id] = manager
            
            self.logger.info(f"Created agent manager for {name} ({agent_id})")
        
        return self.agent_managers[agent_id]
    
    # =====================================================
    # StateGraph Node Implementations
    # =====================================================
    
    @track_performance("perception_node")
    async def _perception_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Process environmental perceptions."""
        agent_id = state["agent_id"]
        current_location = state["current_location"]
        current_time = datetime.now()
        
        # Simulate perception processing
        perceptions = [
            f"Observing environment at {current_location}",
            "Assessing social dynamics",
            "Detecting emotional atmosphere"
        ]
        
        # Add perception to working memory via agent manager
        if agent_id in self.agent_managers:
            manager = self.agent_managers[agent_id]
            for perception in perceptions:
                await self._add_memory_via_persistence(
                    agent_id, perception, "perception", 0.6,
                    {"location": current_location, "perception_type": "environmental"}
                )
        
        # Update state
        state["current_time"] = current_time
        state["current_activity"] = "perceiving_environment"
        
        return state
    
    @track_performance("memory_retrieval_node")
    async def _memory_retrieval_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Retrieve relevant memories for current context."""
        agent_id = state["agent_id"]
        current_activity = state.get("current_activity", "general")
        
        # Retrieve recent working memory
        working_memories = await self.postgres_persistence.retrieve_working_memory(agent_id, 10)
        
        # Retrieve relevant temporal memories
        temporal_pattern = datetime.now().strftime("%Y-%m-%d-%H")
        temporal_memories = await self.postgres_persistence.retrieve_temporal_memory(
            agent_id, temporal_pattern, 5
        )
        
        # Update state with retrieved memories
        state["working_memory"] = [
            {
                "content": mem.content,
                "type": mem.memory_type,
                "importance": mem.importance,
                "timestamp": mem.timestamp.isoformat()
            }
            for mem in working_memories
        ]
        
        # Add context for decision making
        if len(working_memories) > 0 or len(temporal_memories) > 0:
            state["memory_context"] = {
                "working_memory_count": len(working_memories),
                "temporal_memory_count": len(temporal_memories),
                "last_memory_time": working_memories[0].timestamp.isoformat() if working_memories else None
            }
        
        return state
    
    @track_performance("planning_node")
    async def _planning_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Generate plans based on goals and memories."""
        goals = state.get("goals", [])
        working_memory = state.get("working_memory", [])
        
        # Simple planning logic
        if not goals:
            goals = ["maintain social connections", "be authentic", "explore relationships"]
            state["goals"] = goals
        
        # Generate action plan based on current context
        plan = {
            "primary_goal": goals[0] if goals else "socialize",
            "actions": ["engage_in_conversation", "observe_social_dynamics", "express_personality"],
            "priority": 0.7,
            "confidence": 0.8
        }
        
        state["current_plan"] = plan
        
        return state
    
    @track_performance("decision_making_node")
    async def _decision_making_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Make decisions based on plans and social context."""
        current_plan = state.get("current_plan", {})
        emotional_state = state.get("emotional_state", {})
        social_context = state.get("social_context", {})
        
        # Decision making logic
        decision = {
            "action": current_plan.get("actions", ["socialize"])[0],
            "target": "group",  # Could be specific agent
            "reasoning": "Following current plan and social context",
            "confidence": current_plan.get("confidence", 0.5)
        }
        
        state["current_decision"] = decision
        state["decision_timestamp"] = datetime.now()
        
        return state
    
    @track_performance("action_execution_node")
    async def _action_execution_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Execute the decided action."""
        decision = state.get("current_decision", {})
        action = decision.get("action", "idle")
        
        # Execute action (simplified)
        execution_result = {
            "action_taken": action,
            "success": True,
            "outcome": f"Successfully executed {action}",
            "timestamp": datetime.now(),
            "participants": [decision.get("target", "none")]
        }
        
        state["last_action_result"] = execution_result
        state["current_activity"] = action
        
        # Add action to memory
        if state["agent_id"] in self.agent_managers:
            await self._add_memory_via_persistence(
                state["agent_id"],
                f"Executed action: {action}",
                "action",
                0.7,
                execution_result
            )
        
        return state
    
    @track_performance("social_processing_node")
    async def _social_processing_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Process social interactions and relationships."""
        conversation_partners = state.get("conversation_partners", set())
        recent_interactions = state.get("recent_interactions", [])
        
        # Social analysis
        social_analysis = {
            "active_relationships": len(conversation_partners),
            "recent_interaction_count": len(recent_interactions),
            "social_mood": "friendly",
            "interaction_quality": 0.7
        }
        
        # Update social integration score
        if state["agent_id"] in self.agent_managers:
            manager = self.agent_managers[state["agent_id"]]
            manager.performance.social_integration = min(len(conversation_partners) * 0.2, 1.0)
            state["performance"] = manager.performance
        
        state["social_analysis"] = social_analysis
        
        return state
    
    @track_performance("reflection_node")
    async def _reflection_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Reflect on recent actions and experiences."""
        last_action = state.get("last_action_result", {})
        emotional_state = state.get("emotional_state", {})
        
        # Reflection logic
        reflection = {
            "action_evaluation": "positive" if last_action.get("success", False) else "negative",
            "emotional_impact": 0.1,  # Small positive impact
            "lessons_learned": ["social interaction successful", "maintain current approach"],
            "future_adjustments": []
        }
        
        # Update emotional state slightly
        if "happiness" in emotional_state:
            emotional_state["happiness"] = min(emotional_state["happiness"] + reflection["emotional_impact"], 1.0)
        
        state["last_reflection"] = reflection
        state["emotional_state"] = emotional_state
        
        return state
    
    @track_performance("memory_consolidation_node")
    async def _memory_consolidation_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Consolidate memories from working to long-term storage."""
        agent_id = state["agent_id"]
        
        if agent_id in self.agent_managers:
            manager = self.agent_managers[agent_id]
            
            # Consolidate memories (transfer important working memories to long-term)
            working_memories = await self.postgres_persistence.retrieve_working_memory(agent_id, 20)
            
            # Process high-importance memories for long-term storage
            for memory in working_memories:
                if memory.importance > 0.7:
                    # Store in temporal memory for time-based access
                    await self.postgres_persistence.store_temporal_memory(agent_id, memory)
        
        state["memory_consolidation_complete"] = True
        
        return state
    
    @track_performance("memory_association_node")
    async def _memory_association_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Create associations between memories."""
        agent_id = state["agent_id"]
        
        # Simple association logic - in practice would be more sophisticated
        recent_memories = await self.postgres_persistence.retrieve_working_memory(agent_id, 5)
        
        if len(recent_memories) >= 2:
            # Create associations between related memories
            association_count = 0
            for i in range(len(recent_memories) - 1):
                if recent_memories[i].memory_type == recent_memories[i + 1].memory_type:
                    association_count += 1
            
            state["new_associations"] = association_count
        
        return state
    
    @track_performance("specialization_update_node")
    async def _specialization_update_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Update agent specialization based on recent actions."""
        agent_id = state["agent_id"]
        last_action = state.get("last_action_result", {})
        
        if agent_id in self.agent_managers:
            manager = self.agent_managers[agent_id]
            
            # Update skills based on action
            action = last_action.get("action_taken", "")
            if "conversation" in action.lower():
                await self._update_skill(agent_id, "communication", 0.01)
            elif "social" in action.lower():
                await self._update_skill(agent_id, "social_awareness", 0.01)
            
            # Update specialization data
            state["specialization"] = manager.specialization
        
        return state
    
    @track_performance("cultural_processing_node")
    async def _cultural_processing_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Process cultural memes and values."""
        agent_id = state["agent_id"]
        
        if self.store_integration:
            # Get agent's cultural memes
            agent_memes = await self.store_integration.get_agent_memes(agent_id)
            
            # Update cultural influence
            cultural_influence_score = len(agent_memes) * 0.1
            state["cultural_influence"] = min(cultural_influence_score, 1.0)
        
        return state
    
    @track_performance("governance_processing_node") 
    async def _governance_processing_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
        """Process governance and rule compliance."""
        agent_id = state["agent_id"]
        
        if self.store_integration:
            # Check for active governance proposals
            active_proposals = await self.store_integration.get_active_proposals()
            
            # Update governance participation
            state["governance_proposals"] = len(active_proposals)
            state["governance_participation"] = 0.8  # Placeholder
        
        return state
    
    # =====================================================
    # Helper Methods
    # =====================================================
    
    async def _add_memory_via_persistence(self, agent_id: str, content: str, 
                                         memory_type: str, importance: float, context: Dict[str, Any]) -> None:
        """Add memory using persistence layer."""
        if agent_id in self.agent_managers:
            manager = self.agent_managers[agent_id]
            manager.add_memory(content, memory_type, importance, context)
            
            # Also store in PostgreSQL
            from .memory_structures.circular_buffer import MemoryEntry
            memory_entry = MemoryEntry(
                content=content,
                memory_type=memory_type,
                importance=importance,
                timestamp=datetime.now(),
                context=context
            )
            await self.postgres_persistence.store_working_memory(agent_id, memory_entry)
    
    async def _update_skill(self, agent_id: str, skill_name: str, skill_gain: float) -> None:
        """Update agent skill level."""
        if agent_id in self.agent_managers:
            manager = self.agent_managers[agent_id]
            manager.update_specialization("skill_practice", {skill_name: skill_gain})
    
    # =====================================================
    # Graph Execution Methods
    # =====================================================
    
    async def execute_agent_cycle(self, agent_id: str, name: str, 
                                personality_traits: Dict[str, float],
                                initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute one cognitive cycle for an agent.
        
        Args:
            agent_id: Agent identifier
            name: Agent name
            personality_traits: Personality traits
            initial_state: Initial state data
        
        Returns:
            Execution results
        """
        start_time = datetime.now()
        
        try:
            # Get agent manager
            manager = await self.get_agent_manager(agent_id, name, personality_traits)
            
            # Create thread configuration
            config = {
                "configurable": {
                    "thread_id": f"agent_{agent_id}_{int(start_time.timestamp())}",
                    "store_namespace": f"agent_{agent_id}"
                }
            }
            
            # Get current state or use initial state
            if initial_state:
                current_state = {**manager.state, **initial_state}
            else:
                current_state = manager.state
            
            # Execute the graph
            result = await self.graph.ainvoke(current_state, config)
            
            # Update agent manager state
            manager.state = result
            
            # Track performance
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if self.performance_monitor:
                with self.performance_monitor.track_operation("agent_cognitive_cycle", agent_id) as tracker:
                    tracker.add_context(
                        execution_time_ms=execution_time,
                        state_keys=list(result.keys())
                    )
            
            return {
                "success": True,
                "execution_time_ms": execution_time,
                "final_state": result,
                "agent_id": agent_id,
                "cycle_timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute agent cycle for {agent_id}: {str(e)}")
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time,
                "agent_id": agent_id,
                "cycle_timestamp": start_time.isoformat()
            }
    
    async def execute_multi_agent_cycle(self, agents: List[Dict[str, Any]], 
                                      max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """
        Execute cognitive cycles for multiple agents concurrently.
        
        Args:
            agents: List of agent configurations
            max_concurrent: Maximum concurrent executions
        
        Returns:
            List of execution results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(agent_config):
            async with semaphore:
                return await self.execute_agent_cycle(
                    agent_config["agent_id"],
                    agent_config["name"],
                    agent_config["personality_traits"],
                    agent_config.get("initial_state")
                )
        
        # Execute all agents concurrently with limit
        tasks = [execute_with_semaphore(agent) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "agent_id": agents[i]["agent_id"],
                    "execution_time_ms": 0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    # =====================================================
    # Management and Monitoring
    # =====================================================
    
    async def get_system_performance(self) -> Dict[str, Any]:
        """Get system-wide performance metrics."""
        if not self.performance_monitor:
            return {"error": "Performance monitoring not enabled"}
        
        summary = self.performance_monitor.get_performance_summary(60)
        recommendations = self.performance_monitor.get_optimization_recommendations()
        
        # Add agent-specific metrics
        agent_metrics = {}
        for agent_id, manager in self.agent_managers.items():
            agent_metrics[agent_id] = {
                "specialization": asdict(manager.specialization),
                "performance": asdict(manager.performance),
                "memory_summary": manager.get_memory_summary()
            }
        
        return {
            "system_performance": summary,
            "optimization_recommendations": recommendations,
            "agent_metrics": agent_metrics,
            "total_agents": len(self.agent_managers)
        }
    
    async def cleanup_and_shutdown(self) -> None:
        """Cleanup resources and shutdown gracefully."""
        self.logger.info("Shutting down Enhanced PIANO StateGraph...")
        
        # Cleanup agent managers
        for agent_id, manager in self.agent_managers.items():
            try:
                manager.cleanup_memories()
            except Exception as e:
                self.logger.error(f"Error cleaning up agent {agent_id}: {str(e)}")
        
        # Close persistence layer
        if self.postgres_persistence:
            await self.postgres_persistence.close()
        
        # Close store
        if self.store:
            await self.store.aclose()
        
        self.logger.info("Enhanced PIANO StateGraph shutdown complete")


# Factory function for easy creation
async def create_enhanced_piano_graph(postgres_config: PostgresConfig = None,
                                    enable_performance_monitoring: bool = True,
                                    enable_store_integration: bool = True) -> EnhancedPIANOStateGraph:
    """
    Create and initialize Enhanced PIANO StateGraph.
    
    Args:
        postgres_config: PostgreSQL configuration
        enable_performance_monitoring: Enable performance tracking
        enable_store_integration: Enable Store API integration
    
    Returns:
        Initialized EnhancedPIANOStateGraph
    """
    graph = EnhancedPIANOStateGraph(
        postgres_config=postgres_config,
        enable_performance_monitoring=enable_performance_monitoring,
        enable_store_integration=enable_store_integration
    )
    
    await graph.initialize()
    return graph


# Example usage
if __name__ == "__main__":
    async def test_enhanced_piano_graph():
        """Test the Enhanced PIANO StateGraph."""
        
        # Create configuration
        config = PostgresConfig(
            host="localhost",
            database="piano_test",
            username="test_user",
            password="test_password"
        )
        
        try:
            # Create and initialize graph
            graph = await create_enhanced_piano_graph(config)
            
            # Test agent execution
            test_agents = [
                {
                    "agent_id": "test_001",
                    "name": "Isabella Rodriguez",
                    "personality_traits": {"confidence": 0.8, "openness": 0.7}
                },
                {
                    "agent_id": "test_002",
                    "name": "Klaus Mueller",
                    "personality_traits": {"confidence": 0.6, "openness": 0.9}
                }
            ]
            
            # Execute multi-agent cycle
            results = await graph.execute_multi_agent_cycle(test_agents)
            
            print("Execution Results:")
            for result in results:
                print(f"  Agent {result['agent_id']}: "
                      f"{'Success' if result['success'] else 'Failed'} "
                      f"({result['execution_time_ms']:.1f}ms)")
            
            # Get performance metrics
            performance = await graph.get_system_performance()
            print(f"\nSystem Performance:")
            print(f"  Total Agents: {performance['total_agents']}")
            print(f"  Recommendations: {len(performance['optimization_recommendations'])}")
            
            print("\nEnhanced PIANO StateGraph test completed successfully!")
            
        except Exception as e:
            print(f"Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            if 'graph' in locals():
                await graph.cleanup_and_shutdown()
    
    # Run test
    asyncio.run(test_enhanced_piano_graph())