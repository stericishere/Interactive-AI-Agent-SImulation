"""
UpdatePipeline Integration Example and Usage Guide

This module demonstrates how to use the UpdatePipeline service with the existing
UnifiedAgentManager and FrontendStateAdapter for real-time synchronization.

Example scenarios:
1. Basic integration setup
2. Real-time agent state broadcasting
3. Frontend WebSocket connection handling
4. Performance monitoring and health checks
5. Circuit breaker behavior during failures
6. Batch processing optimization
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_integration():
    """
    Example 1: Basic integration with existing services
    
    Shows how to integrate UpdatePipeline with UnifiedAgentManager
    and FrontendStateAdapter for automatic real-time updates.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Integration Setup")
    print("="*60)
    
    try:
        # Import the services
        from .update_pipeline import get_update_pipeline, setup_update_pipeline_integration
        from .unified_agent_manager import get_unified_agent_manager
        from .frontend_state_adapter import get_frontend_state_adapter
        
        # Setup integration
        print("Setting up UpdatePipeline integration...")
        pipeline = setup_update_pipeline_integration()
        
        if pipeline:
            # Start the pipeline
            await pipeline.start()
            print("✅ UpdatePipeline started successfully")
            
            # Get the integrated services
            unified_manager = get_unified_agent_manager()
            frontend_adapter = get_frontend_state_adapter()
            
            # Register a test agent
            print("Registering test agent...")
            test_personality = {
                "openness": 0.7,
                "conscientiousness": 0.6,
                "extroversion": 0.8,
                "agreeableness": 0.5,
                "neuroticism": 0.3
            }
            
            agent_manager = unified_manager.register_agent(
                agent_id="test_001",
                name="Test Agent",
                personality_traits=test_personality
            )
            
            print("✅ Test agent registered")
            
            # Update agent state - this should trigger real-time broadcast
            print("Updating agent state...")
            success = unified_manager.update_agent_state(
                "test_001",
                {
                    "location": "villa_pool",
                    "activity": "swimming",
                    "memory": {
                        "events": [{
                            "content": "Enjoying a swim in the pool",
                            "type": "activity",
                            "importance": 0.6
                        }]
                    }
                }
            )
            
            if success:
                print("✅ Agent state updated - real-time broadcast should be triggered")
            
            # Wait a moment for processing
            await asyncio.sleep(1)
            
            # Check performance metrics
            metrics = pipeline.get_performance_metrics()
            print(f"📊 Performance Metrics:")
            print(f"   Total Updates: {metrics['total_updates']}")
            print(f"   Success Rate: {metrics['success_rate']:.2%}")
            print(f"   Avg Latency: {metrics['average_latency_ms']:.1f}ms")
            print(f"   WebSocket Connections: {metrics['websocket_connections']}")
            
            # Stop the pipeline
            await pipeline.stop()
            print("✅ UpdatePipeline stopped cleanly")
            
        else:
            print("❌ Failed to setup UpdatePipeline integration")
            
    except Exception as e:
        print(f"❌ Error in basic integration example: {e}")
        logger.error(f"Basic integration failed: {e}", exc_info=True)


async def example_websocket_simulation():
    """
    Example 2: WebSocket connection simulation
    
    Shows how WebSocket connections work with the UpdatePipeline,
    including connection management and message broadcasting.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: WebSocket Connection Simulation")
    print("="*60)
    
    try:
        from .update_pipeline import get_update_pipeline, UpdateMessage, UpdatePriority
        
        # Mock WebSocket class for simulation
        class MockWebSocket:
            def __init__(self, ws_id):
                self.id = ws_id
                self.messages = []
                self.closed = False
            
            async def send(self, data):
                if not self.closed:
                    self.messages.append(data)
                    print(f"📱 WebSocket {self.id} received: {len(data)} bytes")
            
            async def close(self):
                self.closed = True
                print(f"🔌 WebSocket {self.id} closed")
        
        # Get pipeline and start it
        pipeline = get_update_pipeline()
        await pipeline.start()
        print("✅ UpdatePipeline started")
        
        # Create mock WebSocket connections
        websockets = []
        for i in range(3):
            ws = MockWebSocket(f"client_{i}")
            websockets.append(ws)
            
            success = await pipeline.register_websocket(f"conn_{i}", ws)
            if success:
                print(f"✅ WebSocket client_{i} connected")
            else:
                print(f"❌ WebSocket client_{i} failed to connect")
        
        # Simulate agent updates
        print("\nSimulating agent state updates...")
        for i in range(5):
            pipeline.queue_agent_update(
                agent_id=f"agent_{i:03d}",
                update_type="state_change",
                data={
                    "name": f"Agent {i}",
                    "location": f"room_{i}",
                    "activity": f"activity_{i}",
                    "emotional_state": {
                        "happiness": 0.5 + i * 0.1,
                        "excitement": 0.6
                    }
                },
                priority=UpdatePriority.NORMAL
            )
            print(f"📤 Queued update for agent_{i:03d}")
        
        # Wait for batch processing
        await asyncio.sleep(0.2)
        
        # Check that all WebSockets received messages
        for ws in websockets:
            print(f"📊 WebSocket {ws.id} received {len(ws.messages)} messages")
            for msg in ws.messages[:2]:  # Show first 2 messages
                try:
                    data = json.loads(msg)
                    print(f"   Message type: {data.get('type', 'unknown')}")
                except:
                    print(f"   Raw message: {msg[:50]}...")
        
        # Disconnect WebSockets
        for i, ws in enumerate(websockets):
            await pipeline.unregister_websocket(f"conn_{i}")
            await ws.close()
        
        # Stop pipeline
        await pipeline.stop()
        print("✅ Example completed")
        
    except Exception as e:
        print(f"❌ Error in WebSocket simulation: {e}")
        logger.error(f"WebSocket simulation failed: {e}", exc_info=True)


async def example_performance_monitoring():
    """
    Example 3: Performance monitoring and health checks
    
    Demonstrates the monitoring capabilities of UpdatePipeline
    including metrics collection and health status reporting.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Performance Monitoring")
    print("="*60)
    
    try:
        from .update_pipeline import get_update_pipeline, UpdatePriority
        
        pipeline = get_update_pipeline()
        await pipeline.start()
        print("✅ UpdatePipeline started for monitoring")
        
        # Generate test load
        print("Generating test load...")
        start_time = time.time()
        
        for i in range(100):
            priority = UpdatePriority.CRITICAL if i % 10 == 0 else \
                      UpdatePriority.HIGH if i % 5 == 0 else \
                      UpdatePriority.NORMAL
            
            pipeline.queue_agent_update(
                agent_id=f"load_test_{i % 10}",
                update_type="load_test",
                data={"iteration": i, "timestamp": time.time()},
                priority=priority
            )
            
            # Small delay to spread load
            if i % 20 == 0:
                await asyncio.sleep(0.01)
        
        # Wait for processing
        await asyncio.sleep(1)
        
        load_duration = time.time() - start_time
        print(f"⏱️  Generated 100 updates in {load_duration:.2f}s")
        
        # Get detailed metrics
        metrics = pipeline.get_performance_metrics()
        print(f"\n📊 Performance Metrics:")
        print(f"   Total Updates: {metrics['total_updates']}")
        print(f"   Successful: {metrics['successful_updates']}")
        print(f"   Failed: {metrics['failed_updates']}")
        print(f"   Success Rate: {metrics['success_rate']:.2%}")
        print(f"   Average Latency: {metrics['average_latency_ms']:.1f}ms")
        print(f"   Last Batch Size: {metrics['last_batch_size']}")
        print(f"   Last Batch Duration: {metrics['last_batch_duration_ms']:.1f}ms")
        print(f"   Updates/Second: {metrics['updates_per_second']:.1f}")
        print(f"   Circuit Breaker: {metrics['circuit_breaker_state']}")
        
        # Get health status
        health = pipeline.get_health_status()
        print(f"\n🏥 Health Status:")
        print(f"   Health Score: {health['health_score']}/100")
        print(f"   Status: {health['health_status']}")
        if health['issues']:
            print(f"   Issues: {', '.join(health['issues'])}")
        else:
            print(f"   Issues: None")
        
        # Test performance under different loads
        print("\nTesting different priority levels...")
        priority_tests = [
            (UpdatePriority.CRITICAL, 10, "Critical"),
            (UpdatePriority.HIGH, 20, "High"),
            (UpdatePriority.NORMAL, 50, "Normal"),
            (UpdatePriority.LOW, 30, "Low")
        ]
        
        for priority, count, name in priority_tests:
            test_start = time.time()
            
            for i in range(count):
                pipeline.queue_agent_update(
                    agent_id=f"priority_test_{i}",
                    update_type=f"{name.lower()}_priority_test",
                    data={"priority": name, "index": i},
                    priority=priority
                )
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            test_duration = (time.time() - test_start) * 1000  # ms
            print(f"   {name}: {count} updates in {test_duration:.1f}ms")
        
        await pipeline.stop()
        print("✅ Performance monitoring example completed")
        
    except Exception as e:
        print(f"❌ Error in performance monitoring: {e}")
        logger.error(f"Performance monitoring failed: {e}", exc_info=True)


async def example_circuit_breaker():
    """
    Example 4: Circuit breaker behavior simulation
    
    Shows how the circuit breaker protects against failures
    and recovers automatically when service is restored.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Circuit Breaker Simulation")
    print("="*60)
    
    try:
        from .update_pipeline import get_update_pipeline, CircuitState
        
        pipeline = get_update_pipeline()
        await pipeline.start()
        print("✅ UpdatePipeline started for circuit breaker test")
        
        # Check initial circuit breaker state
        cb_state = pipeline.circuit_breaker.get_state()
        print(f"🔌 Initial circuit breaker state: {cb_state.value}")
        
        # Force some failures to trip the circuit breaker
        print("Simulating failures to trip circuit breaker...")
        for i in range(6):  # More than failure threshold (5)
            pipeline.circuit_breaker.record_failure()
            state = pipeline.circuit_breaker.get_state()
            print(f"   Failure {i+1}/6 - Circuit state: {state.value}")
        
        # Try to queue updates while circuit is open
        print("\nTrying to queue updates while circuit is open...")
        can_execute = pipeline.circuit_breaker.can_execute()
        print(f"Can execute operations: {can_execute}")
        
        if not can_execute:
            print("✅ Circuit breaker correctly blocking operations")
        
        # Simulate waiting for recovery timeout (shortened for demo)
        print("Simulating recovery timeout...")
        
        # Manually set recovery time for demo (normally 30 seconds)
        pipeline.circuit_breaker.metrics.recovery_timeout = 1.0  # 1 second for demo
        await asyncio.sleep(1.2)  # Wait for recovery timeout
        
        # Check if circuit moves to half-open
        can_execute = pipeline.circuit_breaker.can_execute()
        state = pipeline.circuit_breaker.get_state()
        print(f"After recovery timeout - State: {state.value}, Can execute: {can_execute}")
        
        # Simulate successful operations to close circuit
        print("Simulating successful operations...")
        for i in range(3):  # Half-open max calls
            if pipeline.circuit_breaker.can_execute():
                pipeline.circuit_breaker.record_success()
                state = pipeline.circuit_breaker.get_state()
                print(f"   Success {i+1}/3 - Circuit state: {state.value}")
        
        # Check final state
        final_state = pipeline.circuit_breaker.get_state()
        print(f"🔌 Final circuit breaker state: {final_state.value}")
        
        if final_state == CircuitState.CLOSED:
            print("✅ Circuit breaker successfully recovered")
        
        await pipeline.stop()
        print("✅ Circuit breaker example completed")
        
    except Exception as e:
        print(f"❌ Error in circuit breaker example: {e}")
        logger.error(f"Circuit breaker example failed: {e}", exc_info=True)


async def example_batch_processing():
    """
    Example 5: Batch processing optimization
    
    Demonstrates how the UpdatePipeline optimizes performance
    through intelligent batch processing of updates.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Processing Optimization")
    print("="*60)
    
    try:
        from .update_pipeline import get_update_pipeline, UpdatePriority
        
        pipeline = get_update_pipeline()
        await pipeline.start()
        print("✅ UpdatePipeline started for batch processing test")
        
        # Test different batch scenarios
        scenarios = [
            ("Small batch", 5, 0.01),
            ("Medium batch", 25, 0.005),
            ("Large batch", 75, 0.001),
            ("Mixed priorities", 50, 0.002)
        ]
        
        for scenario_name, update_count, delay in scenarios:
            print(f"\n🧪 Testing {scenario_name} ({update_count} updates)...")
            
            start_time = time.time()
            
            for i in range(update_count):
                # Mix priorities for the last scenario
                if scenario_name == "Mixed priorities":
                    if i < 10:
                        priority = UpdatePriority.CRITICAL
                    elif i < 25:
                        priority = UpdatePriority.HIGH
                    elif i < 40:
                        priority = UpdatePriority.NORMAL
                    else:
                        priority = UpdatePriority.LOW
                else:
                    priority = UpdatePriority.NORMAL
                
                pipeline.queue_agent_update(
                    agent_id=f"batch_test_{i % 10}",
                    update_type=f"batch_{scenario_name.lower().replace(' ', '_')}",
                    data={
                        "scenario": scenario_name,
                        "index": i,
                        "timestamp": time.time()
                    },
                    priority=priority
                )
                
                # Add delay between updates
                if delay > 0:
                    await asyncio.sleep(delay)
            
            # Wait for batch processing
            await asyncio.sleep(0.2)
            
            duration = (time.time() - start_time) * 1000  # ms
            
            # Get metrics
            metrics = pipeline.get_performance_metrics()
            
            print(f"   ⏱️  Total time: {duration:.1f}ms")
            print(f"   📦 Last batch size: {metrics['last_batch_size']}")
            print(f"   ⚡ Last batch duration: {metrics['last_batch_duration_ms']:.1f}ms")
            print(f"   📈 Updates/second: {metrics['updates_per_second']:.1f}")
            
            # Check if performance targets are met
            if metrics['last_batch_duration_ms'] < 100:
                print(f"   ✅ Performance target met (<100ms)")
            else:
                print(f"   ⚠️  Performance target missed (>100ms)")
        
        # Test pending queue status
        pending = pipeline.batch_processor.get_pending_count()
        print(f"\n📋 Pending updates by priority:")
        for priority, count in pending.items():
            if count > 0:
                print(f"   {priority}: {count}")
        
        await pipeline.stop()
        print("✅ Batch processing example completed")
        
    except Exception as e:
        print(f"❌ Error in batch processing example: {e}")
        logger.error(f"Batch processing example failed: {e}", exc_info=True)


async def run_all_examples():
    """Run all UpdatePipeline examples"""
    print("🚀 Starting UpdatePipeline Integration Examples")
    print("="*80)
    
    examples = [
        ("Basic Integration", example_basic_integration),
        ("WebSocket Simulation", example_websocket_simulation),
        ("Performance Monitoring", example_performance_monitoring),
        ("Circuit Breaker", example_circuit_breaker),
        ("Batch Processing", example_batch_processing)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n🔬 Running {name} example...")
            await example_func()
            print(f"✅ {name} example completed successfully")
        except Exception as e:
            print(f"❌ {name} example failed: {e}")
            logger.error(f"{name} example error: {e}", exc_info=True)
    
    print("\n" + "="*80)
    print("🏁 All UpdatePipeline examples completed")


def example_integration_with_main():
    """
    Example 6: Integration with main dating show application
    
    Shows how to integrate UpdatePipeline with the main application.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Integration with Main Application")
    print("="*60)
    
    integration_code = '''
# In dating_show/main.py - add UpdatePipeline integration

from dating_show.services.update_pipeline import setup_update_pipeline_integration, start_update_pipeline_service

class DatingShowMain:
    def __init__(self, config_path: Optional[str] = None):
        # ... existing initialization ...
        
        # Add UpdatePipeline initialization
        self.update_pipeline = None
    
    async def run(self) -> None:
        try:
            # ... existing initialization ...
            
            # Phase 2.5: Start UpdatePipeline service
            logger.info("Phase 2.5: Starting UpdatePipeline service...")
            self.update_pipeline = await start_update_pipeline_service()
            
            if self.update_pipeline:
                logger.info("✅ UpdatePipeline service started successfully")
            else:
                logger.warning("⚠️  UpdatePipeline service failed to start")
            
            # ... continue with existing phases ...
            
        except Exception as e:
            logger.error(f"Application failed: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def cleanup(self) -> None:
        try:
            # Stop UpdatePipeline first
            if self.update_pipeline:
                await self.update_pipeline.stop()
                logger.info("UpdatePipeline stopped")
            
            # ... existing cleanup ...
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# WebSocket routing configuration (if using Django Channels)
# In dating_show/routing.py or asgi.py

from dating_show.services.websocket_consumer import get_websocket_routing

websocket_urlpatterns = get_websocket_routing()

# In your Django ASGI application
application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})
'''
    
    print("Integration code example:")
    print(integration_code)
    
    frontend_code = '''
// Frontend JavaScript integration example

class DatingShowWebSocket {
    constructor(url = 'ws://localhost:8001/ws/dating_show/') {
        this.url = url;
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        
        this.connect();
    }
    
    connect() {
        try {
            this.socket = new WebSocket(this.url);
            
            this.socket.onopen = (event) => {
                console.log('✅ Connected to dating show updates');
                this.reconnectAttempts = 0;
                this.onConnected(event);
            };
            
            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleMessage(data);
            };
            
            this.socket.onclose = (event) => {
                console.log('🔌 Disconnected from dating show updates');
                this.attemptReconnect();
            };
            
            this.socket.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
            };
            
        } catch (error) {
            console.error('Failed to connect:', error);
            this.attemptReconnect();
        }
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'agent_update':
                this.onAgentUpdate(data);
                break;
            case 'system_notification':
                this.onSystemNotification(data);
                break;
            case 'connection_established':
                console.log('Connection established:', data.connection_id);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    onAgentUpdate(data) {
        // Update agent state in UI
        const agentElement = document.getElementById(`agent-${data.agent_id}`);
        if (agentElement) {
            // Update agent display
            this.updateAgentDisplay(agentElement, data.data);
        }
    }
    
    onSystemNotification(data) {
        // Show system notification in UI
        this.showNotification(data.level, data.message);
    }
    
    sendAgentAction(agentId, action, params = {}) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({
                type: 'agent_action',
                agent_id: agentId,
                action: action,
                params: params
            }));
        }
    }
    
    // ... additional methods ...
}

// Initialize WebSocket connection
const datingShowWS = new DatingShowWebSocket();
'''
    
    print("\nFrontend integration example:")
    print(frontend_code)
    
    print("\n✅ Integration examples provided")
    print("See the code comments for step-by-step integration instructions")


if __name__ == "__main__":
    # Run examples if called directly
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1].lower()
        
        examples_map = {
            "basic": example_basic_integration,
            "websocket": example_websocket_simulation,
            "performance": example_performance_monitoring,
            "circuit": example_circuit_breaker,
            "batch": example_batch_processing,
            "integration": example_integration_with_main
        }
        
        if example_name in examples_map:
            if example_name == "integration":
                example_integration_with_main()
            else:
                asyncio.run(examples_map[example_name]())
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available examples: {', '.join(examples_map.keys())}")
    else:
        # Run all examples
        asyncio.run(run_all_examples())