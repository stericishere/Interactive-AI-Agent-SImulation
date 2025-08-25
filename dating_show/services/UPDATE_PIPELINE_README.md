# UpdatePipeline Service - Real-time Synchronization Pipeline

## Overview

The UpdatePipeline service provides real-time state change broadcasting for Epic 2: Real-time Synchronization Pipeline. It bridges the gap between the existing UnifiedAgentManager/FrontendStateAdapter architecture and real-time frontend updates through WebSocket broadcasting, batch processing, and reliability patterns.

## Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ UnifiedAgent    │    │ UpdatePipeline   │    │ WebSocket       │
│ Manager         │───▶│ Service          │───▶│ Clients         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                       │
        ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ FrontendState   │    │ Batch Processor  │    │ Django Channels │
│ Adapter         │    │ Circuit Breaker  │    │ (Optional)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Features

1. **WebSocket Broadcasting**: Real-time updates to frontend clients
2. **Batch Processing**: Optimized update delivery with <100ms targets
3. **Circuit Breaker**: Reliability protection against cascading failures  
4. **Performance Monitoring**: Comprehensive metrics and health checks
5. **Event-Driven Architecture**: Seamless integration with existing services
6. **Fallback Support**: Works with or without Django Channels

## Integration Points

### 1. UnifiedAgentManager Integration

The UpdatePipeline automatically registers as an update listener with the UnifiedAgentManager:

```python
# UpdatePipeline receives state changes automatically
def _handle_agent_state_change(self, event_type: str, agent_id: Any, state_data: Any):
    if event_type == 'agent_updated':
        # Convert and broadcast update
        frontend_data = self.frontend_adapter.get_agent_for_frontend(agent_id)
        self.queue_agent_update(agent_id, 'state_change', frontend_data)
```

### 2. FrontendStateAdapter Integration

The UpdatePipeline uses the FrontendStateAdapter for zero-loss data conversion:

```python
# Automatic data conversion for frontend compatibility
if self.frontend_adapter:
    frontend_data = self.frontend_adapter.get_agent_for_frontend(agent_id)
    if frontend_data:
        self.queue_agent_update(agent_id, update_type, frontend_data)
```

### 3. Django Channels Integration (Optional)

When Django Channels is available, the UpdatePipeline provides enhanced WebSocket capabilities:

```python
# WebSocket consumer integration
from dating_show.services.websocket_consumer import DatingShowConsumer

# Django Channels routing
websocket_urlpatterns = [
    re_path(r'ws/dating_show/$', DatingShowConsumer.as_asgi()),
]
```

## Performance Architecture

### Batch Processing Engine

The UpdatePipeline uses intelligent batch processing to meet performance targets:

- **Target Latency**: <100ms for normal priority updates
- **Critical Updates**: <50ms immediate processing
- **Batch Optimization**: Adaptive batch sizes (1-50 updates)
- **Priority Queuing**: Critical → High → Normal → Low priority processing

```python
# Performance targets by priority
UpdatePriority.CRITICAL:  < 50ms  (immediate)
UpdatePriority.HIGH:      < 50ms  (fast batch)  
UpdatePriority.NORMAL:    < 100ms (normal batch)
UpdatePriority.LOW:       < 500ms (background)
```

### Circuit Breaker Pattern

Protects against cascading failures with automatic recovery:

```python
# Circuit breaker states
CLOSED:    Normal operation (0-4 failures)
OPEN:      Blocking requests (5+ failures, 30s timeout)
HALF_OPEN: Testing recovery (limited requests)
```

### Memory and Resource Management

- **Connection Pooling**: Efficient WebSocket connection management
- **Memory Monitoring**: Automatic cleanup of stale connections
- **Resource Limits**: Configurable batch sizes and timeouts
- **Graceful Degradation**: Fallback modes during high load

## API Reference

### UpdatePipeline Class

#### Core Methods

```python
# Service lifecycle
await pipeline.start()                    # Start the service
await pipeline.stop()                     # Stop the service gracefully

# WebSocket management
await pipeline.register_websocket(id, ws)   # Register WebSocket connection
await pipeline.unregister_websocket(id)     # Remove WebSocket connection

# Update queuing
pipeline.queue_agent_update(
    agent_id="agent_001",
    update_type="state_change",
    data=frontend_data,
    priority=UpdatePriority.NORMAL
)

# Monitoring
metrics = pipeline.get_performance_metrics()
health = pipeline.get_health_status()
```

#### Integration Setup

```python
from dating_show.services.update_pipeline import setup_update_pipeline_integration

# Automatic integration with existing services
pipeline = setup_update_pipeline_integration()
await pipeline.start()
```

### WebSocket Consumer API

#### Message Types (Client → Server)

```javascript
// Subscribe to updates
{
  "type": "subscribe",
  "channels": ["dating_show_updates", "agent_notifications"]
}

// Send agent action
{
  "type": "agent_action", 
  "agent_id": "agent_001",
  "action": "move",
  "params": {"location": "pool"}
}

// Ping for connection health
{
  "type": "ping",
  "timestamp": Date.now()
}
```

#### Message Types (Server → Client)

```javascript
// Agent state update
{
  "type": "agent_update",
  "agent_id": "agent_001", 
  "update_type": "state_change",
  "data": { /* full agent state */ },
  "timestamp": 1623456789.123
}

// System notification
{
  "type": "system_notification",
  "level": "info",
  "message": "Rose ceremony starting",
  "timestamp": 1623456789.123
}

// Connection confirmation
{
  "type": "connection_established",
  "connection_id": "ws_12345",
  "timestamp": 1623456789.123
}
```

## Installation and Setup

### 1. Basic Setup (No Django Channels)

The UpdatePipeline works out-of-the-box with the existing Django setup:

```python
# In your main application
from dating_show.services.update_pipeline import start_update_pipeline_service

async def main():
    # Start UpdatePipeline service
    pipeline = await start_update_pipeline_service()
    
    # Your application logic here
    
    # Cleanup
    await pipeline.stop()
```

### 2. Enhanced Setup (With Django Channels)

For full WebSocket support, install Django Channels:

```bash
pip install channels
pip install channels-redis  # For production Redis backend
```

Update your Django settings:

```python
# settings.py
INSTALLED_APPS = [
    # ... existing apps ...
    'channels',
]

ASGI_APPLICATION = 'dating_show.asgi.application'

# Channel layer configuration
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [("127.0.0.1", 6379)],
        },
    },
}
```

Create ASGI configuration:

```python
# dating_show/asgi.py
import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

from dating_show.services.websocket_consumer import get_websocket_routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dating_show.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(get_websocket_routing())
    ),
})
```

### 3. Frontend Integration

#### JavaScript WebSocket Client

```javascript
class DatingShowClient {
    constructor() {
        this.socket = new WebSocket('ws://localhost:8001/ws/dating_show/');
        this.setupEventHandlers();
    }
    
    setupEventHandlers() {
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleUpdate(data);
        };
    }
    
    handleUpdate(data) {
        if (data.type === 'agent_update') {
            this.updateAgentUI(data.agent_id, data.data);
        }
    }
    
    sendAgentAction(agentId, action, params = {}) {
        this.socket.send(JSON.stringify({
            type: 'agent_action',
            agent_id: agentId,
            action: action,
            params: params
        }));
    }
}

// Initialize client
const client = new DatingShowClient();
```

#### React Integration Example

```jsx
import { useState, useEffect, useRef } from 'react';

function useDatingShowWebSocket() {
    const [agents, setAgents] = useState({});
    const [connectionStatus, setConnectionStatus] = useState('connecting');
    const socketRef = useRef(null);
    
    useEffect(() => {
        const socket = new WebSocket('ws://localhost:8001/ws/dating_show/');
        socketRef.current = socket;
        
        socket.onopen = () => setConnectionStatus('connected');
        socket.onclose = () => setConnectionStatus('disconnected');
        
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'agent_update') {
                setAgents(prev => ({
                    ...prev,
                    [data.agent_id]: data.data
                }));
            }
        };
        
        return () => socket.close();
    }, []);
    
    const sendAgentAction = (agentId, action, params = {}) => {
        if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify({
                type: 'agent_action',
                agent_id: agentId,
                action: action,
                params: params
            }));
        }
    };
    
    return { agents, connectionStatus, sendAgentAction };
}

// Usage in component
function DatingShowDashboard() {
    const { agents, connectionStatus, sendAgentAction } = useDatingShowWebSocket();
    
    return (
        <div>
            <div>Status: {connectionStatus}</div>
            {Object.entries(agents).map(([id, agent]) => (
                <AgentCard 
                    key={id} 
                    agent={agent} 
                    onAction={(action, params) => sendAgentAction(id, action, params)}
                />
            ))}
        </div>
    );
}
```

## Monitoring and Debugging

### Performance Metrics

```python
# Get comprehensive metrics
metrics = pipeline.get_performance_metrics()

print(f"Updates/second: {metrics['updates_per_second']}")
print(f"Average latency: {metrics['average_latency_ms']}ms")
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"WebSocket connections: {metrics['websocket_connections']}")
print(f"Circuit breaker state: {metrics['circuit_breaker_state']}")
```

### Health Monitoring

```python
# Get health status
health = pipeline.get_health_status()

print(f"Health score: {health['health_score']}/100")
print(f"Status: {health['health_status']}")
if health['issues']:
    print(f"Issues: {', '.join(health['issues'])}")
```

### Debug Logging

```python
import logging

# Enable debug logging for UpdatePipeline
logging.getLogger('dating_show.services.update_pipeline').setLevel(logging.DEBUG)
logging.getLogger('dating_show.services.websocket_consumer').setLevel(logging.DEBUG)
```

### Admin WebSocket Interface

Connect to the admin WebSocket for enhanced monitoring:

```javascript
// Admin monitoring client
const adminSocket = new WebSocket('ws://localhost:8001/ws/dating_show/admin/');

adminSocket.onopen = () => {
    // Get current health status
    adminSocket.send(JSON.stringify({
        command: 'get_health'
    }));
    
    // Get performance metrics
    adminSocket.send(JSON.stringify({
        command: 'get_metrics'
    }));
};

adminSocket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Admin data:', data);
};
```

## Error Handling and Reliability

### Circuit Breaker Configuration

```python
# Configure circuit breaker thresholds
pipeline = UpdatePipeline()
pipeline.circuit_breaker.metrics.failure_threshold = 10  # Trip after 10 failures
pipeline.circuit_breaker.metrics.recovery_timeout = 60.0  # 60 second recovery
```

### Retry Logic

The UpdatePipeline includes automatic retry logic for failed operations:

```python
# Update messages include retry tracking
class UpdateMessage:
    retry_count: int = 0  # Automatic retry tracking
    max_retries: int = 3  # Maximum retry attempts
```

### Graceful Degradation

When WebSocket connections fail, the system gracefully degrades:

1. **Connection Loss**: Automatic reconnection attempts
2. **Circuit Breaker Trip**: Updates queued until recovery
3. **Resource Limits**: Batch size reduction under load
4. **Memory Pressure**: Automatic cleanup of stale data

## Production Considerations

### Scaling Configuration

```python
# Production configuration example
pipeline = UpdatePipeline()

# Batch processing optimization
pipeline.batch_processor.max_batch_size = 100  # Larger batches for efficiency
pipeline.batch_processor.batch_timeout_ms = 25.0  # Faster processing

# Circuit breaker tuning
pipeline.circuit_breaker.metrics.failure_threshold = 15
pipeline.circuit_breaker.metrics.recovery_timeout = 120.0

# Performance targets
# Target: <50ms for 95% of updates
# Target: <100ms for 99% of updates
```

### Redis Backend (Channels)

For production deployments with multiple servers:

```python
# settings.py
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {
            "hosts": [("redis-server", 6379)],
            "capacity": 1500,      # Buffer size per channel
            "expiry": 60,          # Message expiry (seconds)
        },
    },
}
```

### Load Balancing

The UpdatePipeline supports horizontal scaling:

```python
# Multiple UpdatePipeline instances can run simultaneously
# Use Redis for coordination when using Django Channels
# WebSocket connections are automatically distributed
```

### Monitoring and Alerting

Set up monitoring for production:

```python
# Health check endpoint
async def health_check():
    pipeline = get_update_pipeline()
    health = pipeline.get_health_status()
    
    if health['health_score'] < 70:
        # Alert: Poor performance
        send_alert('UpdatePipeline performance degraded')
    
    if health['health_status'] == 'critical':
        # Alert: Service critical
        send_alert('UpdatePipeline service critical')
    
    return health
```

## Troubleshooting

### Common Issues

1. **High Latency**
   - Check batch processor configuration
   - Monitor system resource usage
   - Verify network connectivity to clients

2. **Circuit Breaker Trips**
   - Check underlying service health
   - Review error logs for root cause
   - Consider adjusting failure thresholds

3. **WebSocket Connection Issues**
   - Verify Django Channels configuration
   - Check WebSocket URL accessibility
   - Monitor connection cleanup

4. **Memory Usage**
   - Monitor connection count growth
   - Check for stale WebSocket connections
   - Review batch processing queue sizes

### Debug Commands

```python
# Reset circuit breaker
pipeline.circuit_breaker.metrics.state = CircuitState.CLOSED
pipeline.circuit_breaker.metrics.failure_count = 0

# Clear pending updates
for queue in pipeline.batch_processor.pending_updates.values():
    queue.clear()

# Force garbage collection
import gc
gc.collect()
```

## Testing

### Unit Tests

```python
# Test UpdatePipeline functionality
import pytest
from dating_show.services.update_pipeline import UpdatePipeline, UpdatePriority

@pytest.mark.asyncio
async def test_update_pipeline_basic():
    pipeline = UpdatePipeline()
    await pipeline.start()
    
    # Test update queuing
    pipeline.queue_agent_update(
        agent_id="test_001",
        update_type="test",
        data={"test": True},
        priority=UpdatePriority.NORMAL
    )
    
    # Test metrics
    metrics = pipeline.get_performance_metrics()
    assert metrics['total_updates'] > 0
    
    await pipeline.stop()
```

### Integration Tests

```python
# Test with UnifiedAgentManager integration
@pytest.mark.asyncio
async def test_unified_manager_integration():
    from dating_show.services.update_pipeline import setup_update_pipeline_integration
    
    pipeline = setup_update_pipeline_integration()
    await pipeline.start()
    
    # Trigger agent update through UnifiedAgentManager
    # Verify UpdatePipeline receives and processes update
    
    await pipeline.stop()
```

### Load Testing

```python
# Performance load test
import asyncio
import time

async def load_test():
    pipeline = get_update_pipeline()
    await pipeline.start()
    
    start_time = time.time()
    
    # Generate load
    for i in range(1000):
        pipeline.queue_agent_update(
            agent_id=f"load_test_{i % 10}",
            update_type="load_test",
            data={"iteration": i},
            priority=UpdatePriority.NORMAL
        )
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Check performance
    metrics = pipeline.get_performance_metrics()
    duration = time.time() - start_time
    
    print(f"Processed 1000 updates in {duration:.2f}s")
    print(f"Average latency: {metrics['average_latency_ms']:.1f}ms")
    
    await pipeline.stop()
```

## Conclusion

The UpdatePipeline service provides a robust, scalable solution for real-time synchronization in the dating show application. Its integration with the existing UnifiedAgentManager and FrontendStateAdapter ensures zero data loss while adding real-time capabilities. The service is designed for production use with comprehensive monitoring, reliability patterns, and performance optimization.

Key benefits:
- **Real-time Updates**: Immediate frontend synchronization
- **High Performance**: <100ms update delivery targets
- **Reliability**: Circuit breaker protection and automatic recovery
- **Scalability**: Horizontal scaling support with Redis backend
- **Monitoring**: Comprehensive metrics and health checks
- **Flexibility**: Works with or without Django Channels

The service successfully bridges the gap between the enhanced agent state management and real-time frontend requirements, providing the foundation for Epic 2's real-time synchronization pipeline.