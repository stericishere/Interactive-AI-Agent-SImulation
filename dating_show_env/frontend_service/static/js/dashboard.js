// Dating Show Dashboard JavaScript

class DatingShowDashboard {
    constructor() {
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.autoAdvance = false;
        this.autoAdvanceInterval = null;
        this.currentSimulation = null;
    }

    init() {
        console.log('Initializing Dating Show Dashboard');
        this.setupEventListeners();
        this.connectWebSocket();
        this.updateUI();
        
        // Load initial simulation data if available
        if (window.initialSimulation) {
            this.currentSimulation = window.initialSimulation;
            this.updateSimulationUI();
        }
    }

    setupEventListeners() {
        // Control buttons
        document.getElementById('step-btn').addEventListener('click', () => {
            this.advanceSimulation();
        });

        document.getElementById('pause-btn').addEventListener('click', () => {
            this.pauseSimulation();
        });

        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetSimulation();
        });

        // Auto-advance controls
        const autoAdvanceCheckbox = document.getElementById('auto-advance');
        autoAdvanceCheckbox.addEventListener('change', (e) => {
            this.autoAdvance = e.target.checked;
            this.toggleAutoAdvance();
        });

        const speedSelect = document.getElementById('speed-select');
        speedSelect.addEventListener('change', (e) => {
            if (this.autoAdvance) {
                this.stopAutoAdvance();
                this.startAutoAdvance(parseInt(e.target.value));
            }
        });

        // Agent card interactions
        document.addEventListener('click', (e) => {
            if (e.target.closest('.agent-card')) {
                const agentCard = e.target.closest('.agent-card');
                const agentName = agentCard.dataset.agent;
                this.showAgentDetails(agentName);
            }
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                
                // Subscribe to simulation updates
                this.sendWebSocketMessage({
                    type: 'subscribe',
                    topic: 'simulation_updates'
                });
            };

            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWebSocketMessage(message);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                this.attemptReconnect();
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };

        } catch (error) {
            console.error('Error creating WebSocket connection:', error);
            this.updateConnectionStatus(false);
        }
    }

    handleWebSocketMessage(message) {
        console.log('Received WebSocket message:', message);
        
        switch (message.type) {
            case 'simulation_update':
                this.handleSimulationUpdate(message.data);
                break;
            case 'agent_update':
                this.handleAgentUpdate(message.data);
                break;
            case 'subscribed':
                console.log(`Subscribed to topic: ${message.topic}`);
                break;
            case 'pong':
                console.log('Received pong');
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }

    handleSimulationUpdate(updateData) {
        console.log('Simulation update:', updateData);
        
        // Update step counter
        if (updateData.step !== undefined) {
            document.getElementById('current-step').textContent = updateData.step;
        }

        // Add update to feed
        this.addUpdateToFeed(`Simulation advanced to step ${updateData.step}`);
        
        // Refresh simulation state
        this.refreshSimulationState();
    }

    handleAgentUpdate(agentData) {
        console.log('Agent update:', agentData);
        this.addUpdateToFeed(`${agentData.agent_name} state updated`);
        
        // Update specific agent card if visible
        this.updateAgentCard(agentData.agent_name, agentData.agent_data);
    }

    sendWebSocketMessage(message) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected, cannot send message');
        }
    }

    updateConnectionStatus(connected) {
        const indicator = document.getElementById('ws-indicator');
        const status = document.getElementById('ws-status');
        
        if (connected) {
            indicator.classList.add('connected');
            status.textContent = 'WebSocket Connected';
        } else {
            indicator.classList.remove('connected');
            status.textContent = 'WebSocket Disconnected';
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connectWebSocket();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.error('Max reconnection attempts reached');
            this.addUpdateToFeed('Connection lost. Please refresh the page.');
        }
    }

    async advanceSimulation() {
        try {
            const response = await fetch('/api/simulation/step', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addUpdateToFeed(`Advanced to step ${result.step}`);
            } else {
                this.addUpdateToFeed(`Error: ${result.error}`, 'error');
            }
        } catch (error) {
            console.error('Error advancing simulation:', error);
            this.addUpdateToFeed('Error advancing simulation', 'error');
        }
    }

    pauseSimulation() {
        this.stopAutoAdvance();
        document.getElementById('auto-advance').checked = false;
        this.addUpdateToFeed('Simulation paused');
    }

    resetSimulation() {
        if (confirm('Are you sure you want to reset the simulation?')) {
            this.stopAutoAdvance();
            document.getElementById('auto-advance').checked = false;
            this.addUpdateToFeed('Simulation reset');
            // Here you would call the reset API
        }
    }

    toggleAutoAdvance() {
        if (this.autoAdvance) {
            const speed = parseInt(document.getElementById('speed-select').value);
            this.startAutoAdvance(speed);
        } else {
            this.stopAutoAdvance();
        }
    }

    startAutoAdvance(interval) {
        this.stopAutoAdvance(); // Clear any existing interval
        this.autoAdvanceInterval = setInterval(() => {
            this.advanceSimulation();
        }, interval);
        this.addUpdateToFeed(`Auto-advance started (${interval/1000}s intervals)`);
    }

    stopAutoAdvance() {
        if (this.autoAdvanceInterval) {
            clearInterval(this.autoAdvanceInterval);
            this.autoAdvanceInterval = null;
            this.addUpdateToFeed('Auto-advance stopped');
        }
    }

    async refreshSimulationState() {
        try {
            const response = await fetch('/api/simulation/state');
            const simulation = await response.json();
            
            this.currentSimulation = simulation;
            this.updateSimulationUI();
        } catch (error) {
            console.error('Error refreshing simulation state:', error);
        }
    }

    updateSimulationUI() {
        if (!this.currentSimulation) return;

        // Update status indicator
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');
        
        statusIndicator.className = `status-indicator ${this.currentSimulation.mode}`;
        statusText.textContent = this.currentSimulation.mode;

        // Update step counter
        document.getElementById('current-step').textContent = this.currentSimulation.current_step;

        // Update agents grid
        this.updateAgentsGrid();

        // Update environment info
        this.updateEnvironmentInfo();
    }

    updateAgentsGrid() {
        const agentsGrid = document.getElementById('agents-grid');
        
        if (!this.currentSimulation.agents || this.currentSimulation.agents.length === 0) {
            agentsGrid.innerHTML = '<div class="no-agents">No active agents</div>';
            return;
        }

        agentsGrid.innerHTML = this.currentSimulation.agents.map(agent => `
            <div class="agent-card" data-agent="${agent.name}">
                <div class="agent-header">
                    <h3>${agent.name}</h3>
                    <span class="agent-role">${agent.role}</span>
                </div>
                <div class="agent-status">
                    <div class="location">📍 ${agent.current_location || 'Unknown'}</div>
                    <div class="action">🎬 ${agent.current_action || 'Idle'}</div>
                </div>
                <div class="agent-position">
                    Position: (${agent.position.x}, ${agent.position.y})
                </div>
            </div>
        `).join('');
    }

    updateAgentCard(agentName, agentData) {
        const agentCard = document.querySelector(`[data-agent="${agentName}"]`);
        if (agentCard && agentData) {
            // Update specific fields in the agent card
            const locationDiv = agentCard.querySelector('.location');
            const actionDiv = agentCard.querySelector('.action');
            const positionDiv = agentCard.querySelector('.agent-position');
            
            if (locationDiv && agentData.current_location) {
                locationDiv.textContent = `📍 ${agentData.current_location}`;
            }
            
            if (actionDiv && agentData.current_action) {
                actionDiv.textContent = `🎬 ${agentData.current_action}`;
            }
            
            if (positionDiv && agentData.position) {
                positionDiv.textContent = `Position: (${agentData.position.x}, ${agentData.position.y})`;
            }
        }
    }

    updateEnvironmentInfo() {
        if (!this.currentSimulation.environment) return;

        const timeElement = document.getElementById('env-time');
        const weatherElement = document.getElementById('env-weather');
        const eventsElement = document.getElementById('env-events');

        if (timeElement) {
            timeElement.textContent = this.currentSimulation.environment.time_of_day || 'Unknown';
        }
        
        if (weatherElement) {
            weatherElement.textContent = this.currentSimulation.environment.weather || 'Unknown';
        }
        
        if (eventsElement) {
            const eventCount = this.currentSimulation.environment.active_events ? 
                              this.currentSimulation.environment.active_events.length : 0;
            eventsElement.textContent = `${eventCount} events`;
        }
    }

    showAgentDetails(agentName) {
        // For now, just log the agent name
        // This could open a modal or navigate to a detailed view
        console.log(`Showing details for agent: ${agentName}`);
        this.addUpdateToFeed(`Viewing details for ${agentName}`);
    }

    addUpdateToFeed(message, type = 'info') {
        const updatesFeed = document.getElementById('updates-feed');
        const timestamp = new Date().toLocaleTimeString();
        
        const updateItem = document.createElement('div');
        updateItem.className = `update-item new ${type}`;
        updateItem.innerHTML = `<strong>${timestamp}</strong> - ${message}`;
        
        updatesFeed.insertBefore(updateItem, updatesFeed.firstChild);
        
        // Remove the 'new' class after animation
        setTimeout(() => {
            updateItem.classList.remove('new');
        }, 2000);
        
        // Keep only last 50 updates
        const updates = updatesFeed.querySelectorAll('.update-item');
        if (updates.length > 50) {
            updates[updates.length - 1].remove();
        }
    }

    updateUI() {
        // Any additional UI updates can go here
        this.addUpdateToFeed('Dashboard initialized');
    }
}

// Initialize dashboard when DOM is loaded
window.dashboard = new DatingShowDashboard();

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.dashboard.init();
    });
} else {
    window.dashboard.init();
}