/**
 * WebSocket Client Library for Dating Show Real-time Updates
 * Provides reactive state management and real-time synchronization
 */

class DatingShowWebSocketClient {
    constructor(options = {}) {
        this.baseUrl = options.baseUrl || this.getWebSocketBaseUrl();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
        this.reconnectDelay = options.reconnectDelay || 1000;
        this.pingInterval = options.pingInterval || 30000;
        
        // Connection management
        this.connections = new Map();
        this.subscriptions = new Map();
        this.eventListeners = new Map();
        
        // State management
        this.agentStates = new Map();
        this.socialNetwork = null;
        this.systemMetrics = null;
        
        // UI update callbacks
        this.updateCallbacks = new Set();
        
        this.log('WebSocket client initialized');
    }
    
    /**
     * Connect to agent state updates
     */
    async connectToAgents(roomName = 'general') {
        const connectionKey = `agents_${roomName}`;
        
        if (this.connections.has(connectionKey)) {
            this.log(`Already connected to agents room: ${roomName}`);
            return this.connections.get(connectionKey);
        }
        
        try {
            const wsUrl = `${this.baseUrl}/ws/agents/${roomName}/`;
            const ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                this.log(`Connected to agents room: ${roomName}`);
                this.reconnectAttempts = 0;
                this.startPingInterval(ws);
            };
            
            ws.onmessage = (event) => {
                this.handleAgentMessage(JSON.parse(event.data));
            };
            
            ws.onclose = (event) => {
                this.log(`Agent connection closed: ${event.code} ${event.reason}`);
                this.connections.delete(connectionKey);
                this.attemptReconnect(connectionKey, () => this.connectToAgents(roomName));
            };
            
            ws.onerror = (error) => {
                this.error('Agent WebSocket error:', error);
            };
            
            this.connections.set(connectionKey, ws);
            return ws;
            
        } catch (error) {
            this.error('Failed to connect to agents:', error);
            throw error;
        }
    }
    
    /**
     * Connect to system monitoring
     */
    async connectToSystem() {
        const connectionKey = 'system';
        
        if (this.connections.has(connectionKey)) {
            this.log('Already connected to system monitoring');
            return this.connections.get(connectionKey);
        }
        
        try {
            const wsUrl = `${this.baseUrl}/ws/system/`;
            const ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                this.log('Connected to system monitoring');
                this.reconnectAttempts = 0;
                this.startPingInterval(ws);
                
                // Request initial system status
                this.requestSystemStatus();
            };
            
            ws.onmessage = (event) => {
                this.handleSystemMessage(JSON.parse(event.data));
            };
            
            ws.onclose = (event) => {
                this.log(`System connection closed: ${event.code} ${event.reason}`);
                this.connections.delete(connectionKey);
                this.attemptReconnect(connectionKey, () => this.connectToSystem());
            };
            
            ws.onerror = (error) => {
                this.error('System WebSocket error:', error);
            };
            
            this.connections.set(connectionKey, ws);
            return ws;
            
        } catch (error) {
            this.error('Failed to connect to system:', error);
            throw error;
        }
    }
    
    /**
     * Subscribe to specific agent updates
     */
    subscribeToAgent(agentId) {
        const agentConnection = this.getAgentConnection();
        if (agentConnection && agentConnection.readyState === WebSocket.OPEN) {
            const message = {
                type: 'subscribe_agent',
                agent_id: agentId
            };
            
            agentConnection.send(JSON.stringify(message));
            this.subscriptions.set(agentId, true);
            this.log(`Subscribed to agent: ${agentId}`);
        } else {
            this.error('No agent connection available for subscription');
        }
    }
    
    /**
     * Unsubscribe from agent updates
     */
    unsubscribeFromAgent(agentId) {
        const agentConnection = this.getAgentConnection();
        if (agentConnection && agentConnection.readyState === WebSocket.OPEN) {
            const message = {
                type: 'unsubscribe_agent',
                agent_id: agentId
            };
            
            agentConnection.send(JSON.stringify(message));
            this.subscriptions.delete(agentId);
            this.log(`Unsubscribed from agent: ${agentId}`);
        }
    }
    
    /**
     * Request current agent state
     */
    requestAgentState(agentId) {
        const agentConnection = this.getAgentConnection();
        if (agentConnection && agentConnection.readyState === WebSocket.OPEN) {
            const message = {
                type: 'request_agent_state',
                agent_id: agentId
            };
            
            agentConnection.send(JSON.stringify(message));
        }
    }
    
    /**
     * Request social network data
     */
    requestSocialNetwork() {
        const agentConnection = this.getAgentConnection();
        if (agentConnection && agentConnection.readyState === WebSocket.OPEN) {
            const message = {
                type: 'request_social_network'
            };
            
            agentConnection.send(JSON.stringify(message));
        }
    }
    
    /**
     * Request system status
     */
    requestSystemStatus() {
        const systemConnection = this.getSystemConnection();
        if (systemConnection && systemConnection.readyState === WebSocket.OPEN) {
            const message = {
                type: 'request_system_status'
            };
            
            systemConnection.send(JSON.stringify(message));
        }
    }
    
    /**
     * Request performance metrics
     */
    requestPerformanceMetrics() {
        const systemConnection = this.getSystemConnection();
        if (systemConnection && systemConnection.readyState === WebSocket.OPEN) {
            const message = {
                type: 'request_performance_metrics'
            };
            
            systemConnection.send(JSON.stringify(message));
        }
    }
    
    /**
     * Handle agent messages
     */
    handleAgentMessage(data) {
        switch (data.type) {
            case 'connection_established':
                this.log('Agent connection established:', data.connection_id);
                this.emit('agent_connected', data);
                break;
                
            case 'agent_update':
                this.handleAgentUpdate(data);
                break;
                
            case 'batch_update':
                this.handleBatchUpdate(data);
                break;
                
            case 'agent_state_response':
                this.handleAgentStateResponse(data);
                break;
                
            case 'social_network_response':
                this.handleSocialNetworkResponse(data);
                break;
                
            case 'error':
                this.error('Agent WebSocket error:', data.message);
                this.emit('agent_error', data);
                break;
                
            case 'pong':
                // Ping response received
                break;
                
            default:
                this.log('Unknown agent message type:', data.type);
        }
    }
    
    /**
     * Handle system messages
     */
    handleSystemMessage(data) {
        switch (data.type) {
            case 'system_status':
                this.systemMetrics = data.data;
                this.emit('system_status', data.data);
                break;
                
            case 'performance_metrics':
                this.handlePerformanceMetrics(data);
                break;
                
            case 'system_alert':
                this.emit('system_alert', data);
                break;
                
            case 'error':
                this.error('System WebSocket error:', data.message);
                this.emit('system_error', data);
                break;
                
            default:
                this.log('Unknown system message type:', data.type);
        }
    }
    
    /**
     * Handle individual agent updates
     */
    handleAgentUpdate(data) {
        const { agent_id, data: agentData } = data;
        
        // Update local state
        this.agentStates.set(agent_id, agentData);
        
        // Emit update event
        this.emit('agent_updated', { agentId: agent_id, state: agentData });
        
        // Trigger UI updates
        this.triggerUIUpdate('agent_update', { agentId: agent_id, state: agentData });
        
        this.log(`Agent updated: ${agent_id}`);
    }
    
    /**
     * Handle batch updates
     */
    handleBatchUpdate(data) {
        this.emit('batch_update', data);
        this.triggerUIUpdate('batch_update', data);
        this.log(`Batch update completed: ${data.successful_updates} agents`);
    }
    
    /**
     * Handle agent state response
     */
    handleAgentStateResponse(data) {
        const { agent_id, data: agentData } = data;
        this.agentStates.set(agent_id, agentData);
        this.emit('agent_state', { agentId: agent_id, state: agentData });
        this.triggerUIUpdate('agent_state', { agentId: agent_id, state: agentData });
    }
    
    /**
     * Handle social network response
     */
    handleSocialNetworkResponse(data) {
        this.socialNetwork = data.data;
        this.emit('social_network', data.data);
        this.triggerUIUpdate('social_network', data.data);
    }
    
    /**
     * Handle performance metrics
     */
    handlePerformanceMetrics(data) {
        this.emit('performance_metrics', data.data);
        this.triggerUIUpdate('performance_metrics', data.data);
    }
    
    /**
     * Event listener management
     */
    on(event, callback) {
        if (!this.eventListeners.has(event)) {
            this.eventListeners.set(event, new Set());
        }
        this.eventListeners.get(event).add(callback);
    }
    
    off(event, callback) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).delete(callback);
        }
    }
    
    emit(event, data) {
        if (this.eventListeners.has(event)) {
            this.eventListeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    this.error('Event callback error:', error);
                }
            });
        }
    }
    
    /**
     * UI update management
     */
    onUIUpdate(callback) {
        this.updateCallbacks.add(callback);
    }
    
    offUIUpdate(callback) {
        this.updateCallbacks.delete(callback);
    }
    
    triggerUIUpdate(type, data) {
        this.updateCallbacks.forEach(callback => {
            try {
                callback(type, data);
            } catch (error) {
                this.error('UI update callback error:', error);
            }
        });
    }
    
    /**
     * Connection utilities
     */
    getAgentConnection() {
        // Return first agent connection found
        for (const [key, connection] of this.connections) {
            if (key.startsWith('agents_')) {
                return connection;
            }
        }
        return null;
    }
    
    getSystemConnection() {
        return this.connections.get('system');
    }
    
    /**
     * Ping interval for connection keepalive
     */
    startPingInterval(ws) {
        const pingTimer = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            } else {
                clearInterval(pingTimer);
            }
        }, this.pingInterval);
    }
    
    /**
     * Reconnection logic
     */
    attemptReconnect(connectionKey, reconnectFunction) {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            this.error(`Max reconnection attempts reached for ${connectionKey}`);
            this.emit('max_reconnect_attempts', { connectionKey });
            return;
        }
        
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        
        this.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts} for ${connectionKey} in ${delay}ms`);
        
        setTimeout(() => {
            reconnectFunction();
        }, delay);
    }
    
    /**
     * Utility methods
     */
    getWebSocketBaseUrl() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${protocol}//${window.location.host}`;
    }
    
    /**
     * Get current agent state
     */
    getAgentState(agentId) {
        return this.agentStates.get(agentId);
    }
    
    /**
     * Get all agent states
     */
    getAllAgentStates() {
        return Object.fromEntries(this.agentStates);
    }
    
    /**
     * Get social network data
     */
    getSocialNetwork() {
        return this.socialNetwork;
    }
    
    /**
     * Get system metrics
     */
    getSystemMetrics() {
        return this.systemMetrics;
    }
    
    /**
     * Check connection status
     */
    isConnected(connectionType = 'agents') {
        const connection = connectionType === 'system' 
            ? this.getSystemConnection()
            : this.getAgentConnection();
        
        return connection && connection.readyState === WebSocket.OPEN;
    }
    
    /**
     * Close all connections
     */
    disconnect() {
        this.connections.forEach((connection, key) => {
            if (connection.readyState === WebSocket.OPEN) {
                connection.close();
                this.log(`Disconnected from ${key}`);
            }
        });
        
        this.connections.clear();
        this.subscriptions.clear();
        this.agentStates.clear();
        this.socialNetwork = null;
        this.systemMetrics = null;
    }
    
    /**
     * Logging utilities
     */
    log(...args) {
        console.log('[DatingShow WebSocket]', ...args);
    }
    
    error(...args) {
        console.error('[DatingShow WebSocket]', ...args);
    }
}

/**
 * Real-time Agent State Manager
 * Provides reactive state management for UI components
 */
class AgentStateManager {
    constructor(webSocketClient) {
        this.wsClient = webSocketClient;
        this.observers = new Map();
        this.agentFilters = new Map();
        
        // Set up WebSocket event listeners
        this.wsClient.on('agent_updated', this.handleAgentUpdate.bind(this));
        this.wsClient.on('batch_update', this.handleBatchUpdate.bind(this));
        this.wsClient.on('social_network', this.handleSocialNetworkUpdate.bind(this));
    }
    
    /**
     * Subscribe to agent state changes
     */
    observe(agentId, callback, filter = null) {
        if (!this.observers.has(agentId)) {
            this.observers.set(agentId, new Set());
            this.wsClient.subscribeToAgent(agentId);
        }
        
        this.observers.get(agentId).add(callback);
        
        if (filter) {
            this.agentFilters.set(callback, filter);
        }
        
        // Send current state if available
        const currentState = this.wsClient.getAgentState(agentId);
        if (currentState) {
            this.notifyObserver(callback, agentId, currentState, filter);
        }
    }
    
    /**
     * Unsubscribe from agent state changes
     */
    unobserve(agentId, callback) {
        if (this.observers.has(agentId)) {
            this.observers.get(agentId).delete(callback);
            this.agentFilters.delete(callback);
            
            if (this.observers.get(agentId).size === 0) {
                this.observers.delete(agentId);
                this.wsClient.unsubscribeFromAgent(agentId);
            }
        }
    }
    
    /**
     * Handle agent updates from WebSocket
     */
    handleAgentUpdate(data) {
        const { agentId, state } = data;
        this.notifyObservers(agentId, state);
    }
    
    /**
     * Handle batch updates
     */
    handleBatchUpdate(data) {
        // Notify all observers of batch completion
        this.observers.forEach((callbacks, agentId) => {
            const currentState = this.wsClient.getAgentState(agentId);
            if (currentState) {
                this.notifyObservers(agentId, currentState);
            }
        });
    }
    
    /**
     * Handle social network updates
     */
    handleSocialNetworkUpdate(data) {
        // Notify social network observers
        if (this.observers.has('social_network')) {
            this.observers.get('social_network').forEach(callback => {
                callback(data);
            });
        }
    }
    
    /**
     * Notify observers
     */
    notifyObservers(agentId, state) {
        if (this.observers.has(agentId)) {
            this.observers.get(agentId).forEach(callback => {
                const filter = this.agentFilters.get(callback);
                this.notifyObserver(callback, agentId, state, filter);
            });
        }
    }
    
    /**
     * Notify individual observer with filtering
     */
    notifyObserver(callback, agentId, state, filter) {
        try {
            let filteredState = state;
            
            if (filter && typeof filter === 'function') {
                filteredState = filter(state);
            } else if (filter && Array.isArray(filter)) {
                // Filter by specified fields
                filteredState = {};
                filter.forEach(field => {
                    if (state[field] !== undefined) {
                        filteredState[field] = state[field];
                    }
                });
            }
            
            callback(filteredState, agentId);
        } catch (error) {
            console.error('Observer callback error:', error);
        }
    }
    
    /**
     * Get filtered agent state
     */
    getFilteredState(agentId, fields) {
        const state = this.wsClient.getAgentState(agentId);
        if (!state || !fields) return state;
        
        const filtered = {};
        fields.forEach(field => {
            if (state[field] !== undefined) {
                filtered[field] = state[field];
            }
        });
        
        return filtered;
    }
}

// Export for use
window.DatingShowWebSocketClient = DatingShowWebSocketClient;
window.AgentStateManager = AgentStateManager;