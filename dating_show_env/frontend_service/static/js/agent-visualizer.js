/**
 * Agent State Visualization Components
 * Real-time UI components for dating show agents with unified architecture integration
 */

class AgentCard {
    constructor(containerId, agentId, stateManager) {
        this.containerId = containerId;
        this.agentId = agentId;
        this.stateManager = stateManager;
        this.container = document.getElementById(containerId);
        this.currentState = null;
        
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        
        this.init();
    }
    
    init() {
        this.createCardStructure();
        this.bindEvents();
        
        // Subscribe to agent state changes
        this.stateManager.observe(this.agentId, this.updateCard.bind(this));
        
        // Request initial state
        this.stateManager.wsClient.requestAgentState(this.agentId);
    }
    
    createCardStructure() {
        this.container.innerHTML = `
            <div class="agent-card" data-agent-id="${this.agentId}">
                <div class="agent-header">
                    <div class="agent-avatar">
                        <div class="agent-status-indicator"></div>
                    </div>
                    <div class="agent-info">
                        <h3 class="agent-name">Loading...</h3>
                        <p class="agent-role">...</p>
                    </div>
                    <div class="agent-actions">
                        <button class="btn-details" title="View Details">👁️</button>
                        <button class="btn-subscribe" title="Subscribe">🔔</button>
                    </div>
                </div>
                
                <div class="agent-body">
                    <div class="agent-location">
                        <span class="location-icon">📍</span>
                        <span class="location-text">Unknown</span>
                    </div>
                    
                    <div class="agent-activity">
                        <span class="activity-icon">⚡</span>
                        <span class="activity-text">No activity</span>
                    </div>
                    
                    <div class="agent-emotional-state">
                        <div class="emotion-bar">
                            <div class="emotion happiness">
                                <span>😊</span>
                                <div class="bar"><div class="fill"></div></div>
                                <span class="value">0%</span>
                            </div>
                            <div class="emotion stress">
                                <span>😰</span>
                                <div class="bar"><div class="fill"></div></div>
                                <span class="value">0%</span>
                            </div>
                            <div class="emotion romance">
                                <span>💕</span>
                                <div class="bar"><div class="fill"></div></div>
                                <span class="value">0%</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="agent-relationships">
                        <h4>Top Relationships</h4>
                        <div class="relationship-list"></div>
                    </div>
                    
                    <div class="agent-memory">
                        <h4>Recent Memories</h4>
                        <div class="memory-list"></div>
                    </div>
                    
                    <div class="agent-performance">
                        <div class="performance-metric">
                            <span>Skills</span>
                            <div class="skills-list"></div>
                        </div>
                    </div>
                </div>
                
                <div class="agent-footer">
                    <div class="last-updated">
                        <span class="update-time">Never updated</span>
                        <div class="connection-status"></div>
                    </div>
                </div>
            </div>
        `;
        
        this.addCardStyles();
    }
    
    addCardStyles() {
        if (document.getElementById('agent-card-styles')) return;
        
        const styles = `
            <style id="agent-card-styles">
                .agent-card {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 16px;
                    padding: 20px;
                    margin: 16px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    color: white;
                    min-width: 320px;
                    transition: all 0.3s ease;
                }
                
                .agent-card:hover {
                    transform: translateY(-4px);
                    box-shadow: 0 12px 40px rgba(0,0,0,0.15);
                }
                
                .agent-header {
                    display: flex;
                    align-items: center;
                    margin-bottom: 16px;
                }
                
                .agent-avatar {
                    position: relative;
                    width: 48px;
                    height: 48px;
                    background: rgba(255,255,255,0.2);
                    border-radius: 50%;
                    margin-right: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 20px;
                }
                
                .agent-status-indicator {
                    position: absolute;
                    top: -2px;
                    right: -2px;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background: #4CAF50;
                    border: 2px solid white;
                    animation: pulse 2s infinite;
                }
                
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                
                .agent-info {
                    flex: 1;
                }
                
                .agent-name {
                    margin: 0;
                    font-size: 18px;
                    font-weight: 600;
                }
                
                .agent-role {
                    margin: 4px 0 0 0;
                    opacity: 0.8;
                    font-size: 14px;
                }
                
                .agent-actions {
                    display: flex;
                    gap: 8px;
                }
                
                .agent-actions button {
                    background: rgba(255,255,255,0.2);
                    border: none;
                    border-radius: 8px;
                    padding: 8px;
                    color: white;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                
                .agent-actions button:hover {
                    background: rgba(255,255,255,0.3);
                }
                
                .agent-body > div {
                    margin-bottom: 12px;
                }
                
                .agent-location, .agent-activity {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    background: rgba(255,255,255,0.1);
                    padding: 8px 12px;
                    border-radius: 8px;
                    font-size: 14px;
                }
                
                .emotion-bar {
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }
                
                .emotion {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 12px;
                }
                
                .emotion .bar {
                    flex: 1;
                    height: 6px;
                    background: rgba(255,255,255,0.2);
                    border-radius: 3px;
                    overflow: hidden;
                }
                
                .emotion .fill {
                    height: 100%;
                    background: linear-gradient(90deg, #ff6b6b, #feca57);
                    width: 0%;
                    transition: width 0.5s ease;
                }
                
                .emotion .value {
                    width: 30px;
                    text-align: right;
                    font-size: 10px;
                }
                
                .agent-relationships h4, .agent-memory h4 {
                    margin: 0 0 8px 0;
                    font-size: 14px;
                    opacity: 0.9;
                }
                
                .relationship-list, .memory-list {
                    max-height: 80px;
                    overflow-y: auto;
                    font-size: 12px;
                }
                
                .relationship-item {
                    display: flex;
                    justify-content: space-between;
                    padding: 4px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                
                .memory-item {
                    padding: 4px 0;
                    opacity: 0.8;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                }
                
                .skills-list {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 4px;
                    margin-top: 4px;
                }
                
                .skill-tag {
                    background: rgba(255,255,255,0.2);
                    padding: 2px 6px;
                    border-radius: 12px;
                    font-size: 10px;
                }
                
                .agent-footer {
                    margin-top: 16px;
                    padding-top: 12px;
                    border-top: 1px solid rgba(255,255,255,0.2);
                }
                
                .last-updated {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 11px;
                    opacity: 0.7;
                }
                
                .connection-status {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #4CAF50;
                }
                
                .connection-status.disconnected {
                    background: #f44336;
                }
                
                .loading {
                    opacity: 0.6;
                    animation: pulse 1.5s infinite;
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    bindEvents() {
        const detailsBtn = this.container.querySelector('.btn-details');
        const subscribeBtn = this.container.querySelector('.btn-subscribe');
        
        detailsBtn?.addEventListener('click', () => {
            this.showAgentDetails();
        });
        
        subscribeBtn?.addEventListener('click', () => {
            this.toggleSubscription();
        });
    }
    
    updateCard(state, agentId) {
        if (!state) return;
        
        this.currentState = state;
        
        // Update basic info
        this.updateBasicInfo(state);
        this.updateLocation(state);
        this.updateActivity(state);
        this.updateEmotionalState(state);
        this.updateRelationships(state);
        this.updateMemory(state);
        this.updateSkills(state);
        this.updateTimestamp();
        
        // Remove loading state
        this.container.querySelector('.agent-card').classList.remove('loading');
    }
    
    updateBasicInfo(state) {
        const nameEl = this.container.querySelector('.agent-name');
        const roleEl = this.container.querySelector('.agent-role');
        const avatarEl = this.container.querySelector('.agent-avatar');
        
        if (nameEl) nameEl.textContent = state.name || 'Unknown Agent';
        if (roleEl) roleEl.textContent = state.role || 'No Role';
        
        // Set avatar emoji based on agent type or role
        if (avatarEl) {
            const emoji = this.getAgentEmoji(state.role);
            avatarEl.innerHTML = `${emoji}<div class="agent-status-indicator"></div>`;
        }
    }
    
    updateLocation(state) {
        const locationEl = this.container.querySelector('.location-text');
        if (locationEl) {
            locationEl.textContent = state.current_location || 'Unknown';
        }
    }
    
    updateActivity(state) {
        const activityEl = this.container.querySelector('.activity-text');
        if (activityEl) {
            activityEl.textContent = state.current_action || 'No activity';
        }
    }
    
    updateEmotionalState(state) {
        const emotions = state.emotional_state || {};
        
        const updateEmotion = (name, value) => {
            const emotionEl = this.container.querySelector(`.emotion.${name}`);
            if (emotionEl) {
                const fill = emotionEl.querySelector('.fill');
                const valueEl = emotionEl.querySelector('.value');
                
                const percentage = Math.round((value || 0) * 100);
                fill.style.width = `${percentage}%`;
                valueEl.textContent = `${percentage}%`;
            }
        };
        
        updateEmotion('happiness', emotions.happiness);
        updateEmotion('stress', emotions.stress);
        updateEmotion('romance', emotions.romance);
    }
    
    updateRelationships(state) {
        const relationshipsList = this.container.querySelector('.relationship-list');
        if (!relationshipsList) return;
        
        const relationships = state.relationship_scores || {};
        const sortedRelationships = Object.entries(relationships)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 3);
        
        if (sortedRelationships.length === 0) {
            relationshipsList.innerHTML = '<div class="no-data">No relationships yet</div>';
            return;
        }
        
        relationshipsList.innerHTML = sortedRelationships.map(([agentName, score]) => `
            <div class="relationship-item">
                <span>${agentName}</span>
                <span>${(score * 100).toFixed(0)}%</span>
            </div>
        `).join('');
    }
    
    updateMemory(state) {
        const memoryList = this.container.querySelector('.memory-list');
        if (!memoryList) return;
        
        const recentActivities = state.memory?.recent_activities || [];
        
        if (recentActivities.length === 0) {
            memoryList.innerHTML = '<div class="no-data">No recent memories</div>';
            return;
        }
        
        memoryList.innerHTML = recentActivities.slice(0, 3).map(activity => `
            <div class="memory-item">${activity}</div>
        `).join('');
    }
    
    updateSkills(state) {
        const skillsList = this.container.querySelector('.skills-list');
        if (!skillsList) return;
        
        const skills = state.skills || {};
        const skillEntries = Object.entries(skills);
        
        if (skillEntries.length === 0) {
            skillsList.innerHTML = '<span class="no-data">No skills tracked</span>';
            return;
        }
        
        skillsList.innerHTML = skillEntries.map(([skill, level]) => `
            <span class="skill-tag">${skill}: ${level}</span>
        `).join('');
    }
    
    updateTimestamp() {
        const timestampEl = this.container.querySelector('.update-time');
        if (timestampEl) {
            timestampEl.textContent = `Updated ${new Date().toLocaleTimeString()}`;
        }
        
        // Update connection status
        const statusEl = this.container.querySelector('.connection-status');
        if (statusEl) {
            const isConnected = this.stateManager.wsClient.isConnected();
            statusEl.classList.toggle('disconnected', !isConnected);
        }
    }
    
    getAgentEmoji(role) {
        const emojiMap = {
            'host': '🎤',
            'contestant': '💃',
            'judge': '👨‍⚖️',
            'producer': '🎬',
            'camera': '📹',
            'audience': '👥',
            'chef': '👨‍🍳',
            'stylist': '💄',
            'doctor': '👨‍⚕️',
            'therapist': '🧠',
            'default': '👤'
        };
        
        return emojiMap[role?.toLowerCase()] || emojiMap.default;
    }
    
    showAgentDetails() {
        if (!this.currentState) return;
        
        // Create modal or navigate to detail page
        console.log('Show details for agent:', this.agentId, this.currentState);
        
        // For now, log the full state
        this.showDetailModal();
    }
    
    showDetailModal() {
        // Create and show modal with detailed agent information
        const modal = document.createElement('div');
        modal.className = 'agent-detail-modal';
        modal.innerHTML = `
            <div class="modal-backdrop"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h2>${this.currentState.name}</h2>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <pre>${JSON.stringify(this.currentState, null, 2)}</pre>
                </div>
            </div>
        `;
        
        // Add modal styles
        this.addModalStyles();
        
        document.body.appendChild(modal);
        
        // Bind close events
        modal.querySelector('.modal-close').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        modal.querySelector('.modal-backdrop').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
    }
    
    addModalStyles() {
        if (document.getElementById('agent-modal-styles')) return;
        
        const styles = `
            <style id="agent-modal-styles">
                .agent-detail-modal {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    z-index: 1000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .modal-backdrop {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.5);
                }
                
                .modal-content {
                    position: relative;
                    background: white;
                    border-radius: 8px;
                    max-width: 80%;
                    max-height: 80%;
                    overflow: hidden;
                }
                
                .modal-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 16px 20px;
                    border-bottom: 1px solid #eee;
                }
                
                .modal-close {
                    background: none;
                    border: none;
                    font-size: 24px;
                    cursor: pointer;
                }
                
                .modal-body {
                    padding: 20px;
                    overflow: auto;
                    max-height: 60vh;
                }
                
                .modal-body pre {
                    white-space: pre-wrap;
                    font-family: monospace;
                    font-size: 12px;
                    background: #f5f5f5;
                    padding: 16px;
                    border-radius: 4px;
                    overflow: auto;
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    toggleSubscription() {
        // Toggle subscription state
        console.log('Toggle subscription for agent:', this.agentId);
    }
    
    destroy() {
        this.stateManager.unobserve(this.agentId, this.updateCard.bind(this));
        this.container.innerHTML = '';
    }
}

/**
 * Agent Grid Layout Manager
 */
class AgentGrid {
    constructor(containerId, stateManager) {
        this.containerId = containerId;
        this.stateManager = stateManager;
        this.container = document.getElementById(containerId);
        this.agentCards = new Map();
        
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        
        this.init();
    }
    
    init() {
        this.createGridStructure();
        this.loadAgents();
        
        // Listen for new agents
        this.stateManager.wsClient.on('agent_updated', (data) => {
            this.ensureAgentCard(data.agentId);
        });
    }
    
    createGridStructure() {
        this.container.innerHTML = `
            <div class="agent-grid-header">
                <h2>Dating Show Agents</h2>
                <div class="grid-controls">
                    <button id="refresh-agents">🔄 Refresh</button>
                    <button id="toggle-layout">📋 Layout</button>
                </div>
            </div>
            <div class="agent-grid-container">
                <div class="loading-spinner">Loading agents...</div>
            </div>
        `;
        
        this.addGridStyles();
        this.bindGridEvents();
    }
    
    addGridStyles() {
        if (document.getElementById('agent-grid-styles')) return;
        
        const styles = `
            <style id="agent-grid-styles">
                .agent-grid-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding: 0 16px;
                }
                
                .grid-controls {
                    display: flex;
                    gap: 8px;
                }
                
                .grid-controls button {
                    padding: 8px 16px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    background: white;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                
                .grid-controls button:hover {
                    background: #f5f5f5;
                }
                
                .agent-grid-container {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
                    gap: 16px;
                    padding: 0 16px;
                }
                
                .agent-grid-container.list-layout {
                    grid-template-columns: 1fr;
                }
                
                .loading-spinner {
                    grid-column: 1 / -1;
                    text-align: center;
                    padding: 40px;
                    color: #666;
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    bindGridEvents() {
        const refreshBtn = this.container.querySelector('#refresh-agents');
        const layoutBtn = this.container.querySelector('#toggle-layout');
        
        refreshBtn?.addEventListener('click', () => {
            this.refreshAgents();
        });
        
        layoutBtn?.addEventListener('click', () => {
            this.toggleLayout();
        });
    }
    
    async loadAgents() {
        try {
            // Fetch agent list from API
            const response = await fetch('/api/agents/');
            const data = await response.json();
            
            const gridContainer = this.container.querySelector('.agent-grid-container');
            gridContainer.innerHTML = '';
            
            if (data.agents && data.agents.length > 0) {
                data.agents.forEach(agent => {
                    this.createAgentCard(agent.agent_id);
                });
            } else {
                gridContainer.innerHTML = '<div class="no-agents">No agents found</div>';
            }
        } catch (error) {
            console.error('Failed to load agents:', error);
            const gridContainer = this.container.querySelector('.agent-grid-container');
            gridContainer.innerHTML = '<div class="error">Failed to load agents</div>';
        }
    }
    
    createAgentCard(agentId) {
        if (this.agentCards.has(agentId)) return;
        
        const cardContainer = document.createElement('div');
        cardContainer.id = `agent-card-${agentId}`;
        
        const gridContainer = this.container.querySelector('.agent-grid-container');
        gridContainer.appendChild(cardContainer);
        
        const agentCard = new AgentCard(cardContainer.id, agentId, this.stateManager);
        this.agentCards.set(agentId, agentCard);
    }
    
    ensureAgentCard(agentId) {
        if (!this.agentCards.has(agentId)) {
            this.createAgentCard(agentId);
        }
    }
    
    refreshAgents() {
        this.loadAgents();
    }
    
    toggleLayout() {
        const gridContainer = this.container.querySelector('.agent-grid-container');
        gridContainer.classList.toggle('list-layout');
    }
    
    destroy() {
        this.agentCards.forEach(card => card.destroy());
        this.agentCards.clear();
    }
}

// Export for use
window.AgentCard = AgentCard;
window.AgentGrid = AgentGrid;