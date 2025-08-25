/**
 * Dating Show Application
 * Main application integration with unified architecture
 */

class DatingShowApp {
    constructor(options = {}) {
        this.config = {
            autoConnect: options.autoConnect !== false,
            enablePerformanceMonitoring: options.enablePerformanceMonitoring !== false,
            enableSocialNetwork: options.enableSocialNetwork !== false,
            reconnectAttempts: options.reconnectAttempts || 5,
            ...options
        };
        
        // Core components
        this.wsClient = null;
        this.stateManager = null;
        this.agentGrid = null;
        this.socialNetworkViz = null;
        this.performanceDashboard = null;
        
        // Application state
        this.isInitialized = false;
        this.connectionStatus = 'disconnected';
        this.activeView = 'agents';
        
        this.init();
    }
    
    async init() {
        try {
            await this.initializeCore();
            await this.initializeComponents();
            this.setupNavigation();
            this.setupEventListeners();
            
            if (this.config.autoConnect) {
                await this.connect();
            }
            
            this.isInitialized = true;
            this.updateConnectionStatus('initialized');
            
            console.log('Dating Show App initialized successfully');
        } catch (error) {
            console.error('Failed to initialize Dating Show App:', error);
            this.showError('Application initialization failed');
        }
    }
    
    async initializeCore() {
        // Initialize WebSocket client
        this.wsClient = new DatingShowWebSocketClient({
            maxReconnectAttempts: this.config.reconnectAttempts,
            reconnectDelay: 1000
        });
        
        // Initialize state manager
        this.stateManager = new AgentStateManager(this.wsClient);
        
        // Set up core event listeners
        this.wsClient.on('agent_connected', () => {
            this.updateConnectionStatus('connected');
        });
        
        this.wsClient.on('max_reconnect_attempts', () => {
            this.updateConnectionStatus('failed');
            this.showError('Connection failed after maximum retry attempts');
        });
        
        this.wsClient.on('agent_error', (data) => {
            this.showError(`WebSocket error: ${data.message}`);
        });
    }
    
    async initializeComponents() {
        // Create main application layout
        this.createMainLayout();
        
        // Initialize agent grid
        const agentGridContainer = document.getElementById('agent-grid-view');
        if (agentGridContainer) {
            this.agentGrid = new AgentGrid('agent-grid-view', this.stateManager);
        }
        
        // Initialize social network visualization
        if (this.config.enableSocialNetwork) {
            const socialNetworkContainer = document.getElementById('social-network-view');
            if (socialNetworkContainer) {
                this.socialNetworkViz = new SocialNetworkVisualization('social-network-view', this.stateManager, {
                    width: Math.min(window.innerWidth * 0.9, 1200),
                    height: Math.min(window.innerHeight * 0.7, 800)
                });
            }
        }
        
        // Initialize performance dashboard
        if (this.config.enablePerformanceMonitoring) {
            const performanceContainer = document.getElementById('performance-view');
            if (performanceContainer) {
                this.performanceDashboard = new PerformanceDashboard('performance-view', this.stateManager);
            }
        }
    }
    
    createMainLayout() {
        // Check if layout already exists
        if (document.getElementById('dating-show-app')) {
            return;
        }
        
        // Create main app container
        const appContainer = document.createElement('div');
        appContainer.id = 'dating-show-app';
        appContainer.innerHTML = `
            <div class="app-header">
                <div class="app-title">
                    <h1>Dating Show Live Dashboard</h1>
                    <div class="app-subtitle">Real-time Agent Monitoring & Social Network</div>
                </div>
                <div class="app-status">
                    <div class="connection-status" id="connection-status">
                        <div class="status-indicator"></div>
                        <span class="status-text">Initializing...</span>
                    </div>
                    <div class="unified-architecture-status" id="unified-status">
                        <span class="unified-label">Unified Architecture:</span>
                        <span class="unified-text">Checking...</span>
                    </div>
                </div>
            </div>
            
            <nav class="app-navigation">
                <button class="nav-btn active" data-view="agents">
                    👥 Agents
                </button>
                <button class="nav-btn" data-view="social-network" ${!this.config.enableSocialNetwork ? 'disabled' : ''}>
                    🌐 Social Network
                </button>
                <button class="nav-btn" data-view="performance" ${!this.config.enablePerformanceMonitoring ? 'disabled' : ''}>
                    📊 Performance
                </button>
                <button class="nav-btn" data-view="settings">
                    ⚙️ Settings
                </button>
            </nav>
            
            <main class="app-main">
                <div class="app-view active" id="agent-grid-view"></div>
                <div class="app-view" id="social-network-view"></div>
                <div class="app-view" id="performance-view"></div>
                <div class="app-view" id="settings-view">
                    <div class="settings-panel">
                        <h2>Application Settings</h2>
                        <div class="settings-group">
                            <h3>Connection Settings</h3>
                            <label>
                                <input type="checkbox" id="auto-reconnect" checked>
                                Auto-reconnect on disconnect
                            </label>
                            <label>
                                Max reconnection attempts:
                                <input type="number" id="max-reconnects" value="${this.config.reconnectAttempts}" min="1" max="20">
                            </label>
                        </div>
                        <div class="settings-group">
                            <h3>Feature Toggles</h3>
                            <label>
                                <input type="checkbox" id="enable-social-network" ${this.config.enableSocialNetwork ? 'checked' : ''}>
                                Enable Social Network Visualization
                            </label>
                            <label>
                                <input type="checkbox" id="enable-performance" ${this.config.enablePerformanceMonitoring ? 'checked' : ''}>
                                Enable Performance Monitoring
                            </label>
                        </div>
                        <div class="settings-group">
                            <h3>System Information</h3>
                            <div class="system-info" id="system-info">
                                Loading system information...
                            </div>
                        </div>
                        <div class="settings-actions">
                            <button id="validate-system">🔍 Validate System</button>
                            <button id="reset-connection">🔄 Reset Connection</button>
                            <button id="export-settings">💾 Export Settings</button>
                        </div>
                    </div>
                </div>
            </main>
            
            <div class="app-footer">
                <div class="footer-info">
                    <span>Unified Architecture v2.0</span>
                    <span>•</span>
                    <span>Real-time Synchronization Active</span>
                    <span>•</span>
                    <span id="last-update">Never updated</span>
                </div>
                <div class="footer-actions">
                    <button id="emergency-disconnect">🚨 Emergency Disconnect</button>
                </div>
            </div>
        `;
        
        // Find a suitable container or append to body
        const container = document.getElementById('main-content') || document.body;
        container.appendChild(appContainer);
        
        this.addMainStyles();
    }
    
    addMainStyles() {
        if (document.getElementById('dating-show-app-styles')) return;
        
        const styles = `
            <style id="dating-show-app-styles">
                #dating-show-app {
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    background: #f8f9fa;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                }
                
                .app-header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px 24px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                
                .app-title h1 {
                    margin: 0;
                    font-size: 28px;
                    font-weight: 700;
                }
                
                .app-subtitle {
                    margin-top: 4px;
                    opacity: 0.9;
                    font-size: 14px;
                }
                
                .app-status {
                    display: flex;
                    flex-direction: column;
                    align-items: flex-end;
                    gap: 8px;
                }
                
                .connection-status {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-size: 14px;
                }
                
                .status-indicator {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background: #ccc;
                    transition: background 0.3s;
                }
                
                .status-indicator.connected {
                    background: #4CAF50;
                    animation: pulse 2s infinite;
                }
                
                .status-indicator.connecting {
                    background: #FF9800;
                    animation: pulse 1s infinite;
                }
                
                .status-indicator.disconnected {
                    background: #f44336;
                }
                
                .unified-architecture-status {
                    font-size: 12px;
                    opacity: 0.9;
                }
                
                .unified-label {
                    margin-right: 6px;
                }
                
                .app-navigation {
                    background: white;
                    border-bottom: 1px solid #ddd;
                    padding: 0 24px;
                    display: flex;
                    gap: 0;
                    overflow-x: auto;
                }
                
                .nav-btn {
                    background: none;
                    border: none;
                    padding: 16px 20px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                    color: #666;
                    border-bottom: 3px solid transparent;
                    transition: all 0.2s;
                    white-space: nowrap;
                }
                
                .nav-btn:hover:not(:disabled) {
                    color: #333;
                    background: #f8f9fa;
                }
                
                .nav-btn.active {
                    color: #2196F3;
                    border-bottom-color: #2196F3;
                }
                
                .nav-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
                
                .app-main {
                    flex: 1;
                    position: relative;
                    overflow: hidden;
                }
                
                .app-view {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    overflow: auto;
                    opacity: 0;
                    pointer-events: none;
                    transition: opacity 0.3s ease;
                }
                
                .app-view.active {
                    opacity: 1;
                    pointer-events: auto;
                }
                
                .settings-panel {
                    max-width: 800px;
                    margin: 40px auto;
                    padding: 24px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                
                .settings-panel h2 {
                    margin: 0 0 24px 0;
                    color: #333;
                }
                
                .settings-group {
                    margin-bottom: 32px;
                    padding-bottom: 24px;
                    border-bottom: 1px solid #eee;
                }
                
                .settings-group:last-child {
                    border-bottom: none;
                }
                
                .settings-group h3 {
                    margin: 0 0 16px 0;
                    color: #555;
                    font-size: 16px;
                }
                
                .settings-group label {
                    display: block;
                    margin-bottom: 12px;
                    font-size: 14px;
                    color: #666;
                }
                
                .settings-group input {
                    margin-right: 8px;
                }
                
                .settings-group input[type="number"] {
                    margin-left: 8px;
                    padding: 4px 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    width: 80px;
                }
                
                .system-info {
                    background: #f8f9fa;
                    padding: 16px;
                    border-radius: 6px;
                    font-family: monospace;
                    font-size: 12px;
                    line-height: 1.5;
                }
                
                .settings-actions {
                    display: flex;
                    gap: 12px;
                    flex-wrap: wrap;
                }
                
                .settings-actions button {
                    padding: 10px 16px;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    background: white;
                    cursor: pointer;
                    font-size: 14px;
                    transition: all 0.2s;
                }
                
                .settings-actions button:hover {
                    background: #f5f5f5;
                    border-color: #bbb;
                }
                
                .app-footer {
                    background: white;
                    border-top: 1px solid #ddd;
                    padding: 12px 24px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 12px;
                    color: #666;
                }
                
                .footer-info {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }
                
                .footer-actions button {
                    padding: 6px 12px;
                    border: 1px solid #f44336;
                    border-radius: 4px;
                    background: #fff5f5;
                    color: #f44336;
                    cursor: pointer;
                    font-size: 11px;
                    transition: all 0.2s;
                }
                
                .footer-actions button:hover {
                    background: #f44336;
                    color: white;
                }
                
                .error-message {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #f44336;
                    color: white;
                    padding: 12px 16px;
                    border-radius: 6px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    z-index: 10000;
                    animation: slideInRight 0.3s ease;
                }
                
                .success-message {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #4CAF50;
                    color: white;
                    padding: 12px 16px;
                    border-radius: 6px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    z-index: 10000;
                    animation: slideInRight 0.3s ease;
                }
                
                @keyframes slideInRight {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.6; }
                }
                
                @media (max-width: 768px) {
                    .app-header {
                        flex-direction: column;
                        text-align: center;
                        gap: 16px;
                    }
                    
                    .app-status {
                        align-items: center;
                    }
                    
                    .app-navigation {
                        justify-content: center;
                    }
                    
                    .settings-actions {
                        justify-content: center;
                    }
                    
                    .app-footer {
                        flex-direction: column;
                        gap: 8px;
                        text-align: center;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    setupNavigation() {
        const navButtons = document.querySelectorAll('.nav-btn');
        
        navButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                if (button.disabled) return;
                
                const view = button.getAttribute('data-view');
                this.switchView(view);
            });
        });
    }
    
    switchView(viewName) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        document.querySelector(`[data-view="${viewName}"]`)?.classList.add('active');
        
        // Update views
        document.querySelectorAll('.app-view').forEach(view => {
            view.classList.remove('active');
        });
        
        const targetView = document.getElementById(`${viewName}-view`) || document.getElementById(`${viewName.replace('-', '-')}-view`);
        if (targetView) {
            targetView.classList.add('active');
        }
        
        this.activeView = viewName;
        
        // Trigger view-specific initialization if needed
        this.onViewSwitch(viewName);
    }
    
    onViewSwitch(viewName) {
        switch (viewName) {
            case 'social-network':
                if (this.socialNetworkViz) {
                    this.socialNetworkViz.refreshNetwork();
                }
                break;
            case 'performance':
                if (this.performanceDashboard) {
                    this.performanceDashboard.refreshMetrics();
                }
                break;
            case 'settings':
                this.loadSystemInfo();
                break;
        }
    }
    
    setupEventListeners() {
        // Settings event listeners
        document.getElementById('validate-system')?.addEventListener('click', () => {
            this.validateSystem();
        });
        
        document.getElementById('reset-connection')?.addEventListener('click', () => {
            this.resetConnection();
        });
        
        document.getElementById('export-settings')?.addEventListener('click', () => {
            this.exportSettings();
        });
        
        document.getElementById('emergency-disconnect')?.addEventListener('click', () => {
            this.emergencyDisconnect();
        });
        
        // Settings change listeners
        document.getElementById('auto-reconnect')?.addEventListener('change', (e) => {
            // Update reconnection behavior
        });
        
        document.getElementById('max-reconnects')?.addEventListener('change', (e) => {
            this.config.reconnectAttempts = parseInt(e.target.value);
        });
        
        // Update last update timestamp
        setInterval(() => {
            this.updateLastUpdateTime();
        }, 1000);
    }
    
    async connect() {
        try {
            this.updateConnectionStatus('connecting');
            
            // Connect to agents
            await this.wsClient.connectToAgents();
            
            // Connect to system monitoring if enabled
            if (this.config.enablePerformanceMonitoring) {
                await this.wsClient.connectToSystem();
            }
            
            this.updateConnectionStatus('connected');
            this.showSuccess('Connected to Dating Show backend');
            
            // Load unified architecture status
            await this.loadUnifiedArchitectureStatus();
            
        } catch (error) {
            this.updateConnectionStatus('disconnected');
            this.showError(`Connection failed: ${error.message}`);
            throw error;
        }
    }
    
    async disconnect() {
        this.wsClient.disconnect();
        this.updateConnectionStatus('disconnected');
    }
    
    async resetConnection() {
        try {
            await this.disconnect();
            await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
            await this.connect();
            this.showSuccess('Connection reset successfully');
        } catch (error) {
            this.showError(`Connection reset failed: ${error.message}`);
        }
    }
    
    emergencyDisconnect() {
        this.disconnect();
        this.showSuccess('Emergency disconnect completed');
    }
    
    updateConnectionStatus(status) {
        this.connectionStatus = status;
        
        const statusIndicator = document.querySelector('.status-indicator');
        const statusText = document.querySelector('.status-text');
        
        if (statusIndicator && statusText) {
            statusIndicator.className = `status-indicator ${status}`;
            
            switch (status) {
                case 'connected':
                    statusText.textContent = 'Connected';
                    break;
                case 'connecting':
                    statusText.textContent = 'Connecting...';
                    break;
                case 'disconnected':
                    statusText.textContent = 'Disconnected';
                    break;
                case 'failed':
                    statusText.textContent = 'Connection Failed';
                    break;
                default:
                    statusText.textContent = status;
            }
        }
    }
    
    async loadUnifiedArchitectureStatus() {
        try {
            const response = await fetch('/api/unified/status/');
            const data = await response.json();
            
            const unifiedText = document.querySelector('.unified-text');
            if (unifiedText) {
                if (data.unified_architecture && data.update_pipeline) {
                    unifiedText.textContent = '✅ Active (Pipeline Ready)';
                } else if (data.unified_architecture) {
                    unifiedText.textContent = '⚠️ Active (No Pipeline)';
                } else {
                    unifiedText.textContent = '❌ Unavailable';
                }
            }
        } catch (error) {
            console.error('Failed to load unified architecture status:', error);
            const unifiedText = document.querySelector('.unified-text');
            if (unifiedText) {
                unifiedText.textContent = '❓ Unknown';
            }
        }
    }
    
    async loadSystemInfo() {
        try {
            const response = await fetch('/api/unified/status/');
            const data = await response.json();
            
            const systemInfo = document.getElementById('system-info');
            if (systemInfo) {
                systemInfo.textContent = JSON.stringify(data, null, 2);
            }
        } catch (error) {
            const systemInfo = document.getElementById('system-info');
            if (systemInfo) {
                systemInfo.textContent = `Error loading system info: ${error.message}`;
            }
        }
    }
    
    async validateSystem() {
        try {
            const response = await fetch('/api/pipeline/validate/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ type: 'comprehensive' })
            });
            
            const data = await response.json();
            
            if (data.results && data.results.validation_summary) {
                const summary = data.results.validation_summary;
                const successRate = summary.success_rate;
                
                if (successRate >= 90) {
                    this.showSuccess(`System validation passed: ${successRate.toFixed(1)}% success rate`);
                } else {
                    this.showError(`System validation warnings: ${successRate.toFixed(1)}% success rate`);
                }
            } else {
                this.showError('System validation failed');
            }
        } catch (error) {
            this.showError(`Validation error: ${error.message}`);
        }
    }
    
    exportSettings() {
        const settings = {
            config: this.config,
            connectionStatus: this.connectionStatus,
            activeView: this.activeView,
            timestamp: new Date().toISOString()
        };
        
        const dataStr = JSON.stringify(settings, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `dating-show-settings-${Date.now()}.json`;
        link.click();
    }
    
    updateLastUpdateTime() {
        const lastUpdateEl = document.getElementById('last-update');
        if (lastUpdateEl && this.connectionStatus === 'connected') {
            lastUpdateEl.textContent = `Updated ${new Date().toLocaleTimeString()}`;
        }
    }
    
    showError(message) {
        this.showNotification(message, 'error');
    }
    
    showSuccess(message) {
        this.showNotification(message, 'success');
    }
    
    showNotification(message, type = 'error') {
        const notification = document.createElement('div');
        notification.className = `${type}-message`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            if (notification.parentNode) {
                document.body.removeChild(notification);
            }
        }, 5000);
    }
    
    destroy() {
        if (this.wsClient) {
            this.wsClient.disconnect();
        }
        
        if (this.agentGrid) {
            this.agentGrid.destroy();
        }
        
        if (this.socialNetworkViz) {
            this.socialNetworkViz.destroy();
        }
        
        if (this.performanceDashboard) {
            this.performanceDashboard.destroy();
        }
        
        const appElement = document.getElementById('dating-show-app');
        if (appElement) {
            appElement.remove();
        }
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Check if we should auto-initialize
    if (typeof window.datingShowAutoInit === 'undefined' || window.datingShowAutoInit) {
        window.datingShowApp = new DatingShowApp();
    }
});

// Export for manual initialization
window.DatingShowApp = DatingShowApp;