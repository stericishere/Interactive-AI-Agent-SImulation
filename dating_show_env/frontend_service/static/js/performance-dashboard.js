/**
 * Performance Dashboard for UpdatePipeline Metrics
 * Real-time monitoring and visualization of system performance
 */

class PerformanceDashboard {
    constructor(containerId, stateManager, options = {}) {
        this.containerId = containerId;
        this.stateManager = stateManager;
        this.container = document.getElementById(containerId);
        
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        
        // Configuration
        this.config = {
            refreshInterval: options.refreshInterval || 5000, // 5 seconds
            historyLength: options.historyLength || 50,
            chartWidth: options.chartWidth || 400,
            chartHeight: options.chartHeight || 200,
            showAlerts: options.showAlerts !== false,
            thresholds: {
                processing_time_warning: 80, // ms
                processing_time_critical: 100, // ms
                success_rate_warning: 95, // %
                success_rate_critical: 90, // %
                queue_size_warning: 50,
                queue_size_critical: 100
            },
            ...options
        };
        
        // State
        this.currentMetrics = null;
        this.metricsHistory = [];
        this.refreshTimer = null;
        this.charts = {};
        this.alerts = [];
        
        this.init();
    }
    
    init() {
        this.createDashboard();
        this.setupEventListeners();
        this.startRefreshTimer();
        this.loadInitialMetrics();
    }
    
    createDashboard() {
        this.container.innerHTML = `
            <div class="performance-dashboard">
                <div class="dashboard-header">
                    <h2>UpdatePipeline Performance</h2>
                    <div class="dashboard-controls">
                        <button id="refresh-metrics">🔄 Refresh</button>
                        <button id="reset-metrics">🗑️ Reset</button>
                        <button id="export-metrics">📊 Export</button>
                        <div class="auto-refresh">
                            <label>
                                <input type="checkbox" id="auto-refresh" checked>
                                Auto-refresh (${this.config.refreshInterval / 1000}s)
                            </label>
                        </div>
                    </div>
                </div>
                
                <div class="alerts-panel" id="alerts-panel"></div>
                
                <div class="metrics-grid">
                    <div class="metric-card primary">
                        <div class="metric-header">
                            <h3>Processing Time</h3>
                            <div class="metric-status" id="processing-status"></div>
                        </div>
                        <div class="metric-value">
                            <span class="value" id="processing-time">--</span>
                            <span class="unit">ms</span>
                        </div>
                        <div class="metric-chart" id="processing-chart"></div>
                        <div class="metric-details">
                            <div class="detail">
                                <span class="label">Target:</span>
                                <span class="value">&lt;100ms</span>
                            </div>
                            <div class="detail">
                                <span class="label">Batch:</span>
                                <span class="value" id="batch-time">--ms</span>
                            </div>
                            <div class="detail">
                                <span class="label">WebSocket:</span>
                                <span class="value" id="websocket-time">--ms</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <h3>Success Rate</h3>
                            <div class="metric-status" id="success-status"></div>
                        </div>
                        <div class="metric-value">
                            <span class="value" id="success-rate">--</span>
                            <span class="unit">%</span>
                        </div>
                        <div class="metric-chart" id="success-chart"></div>
                        <div class="metric-details">
                            <div class="detail">
                                <span class="label">Successful:</span>
                                <span class="value" id="successful-updates">--</span>
                            </div>
                            <div class="detail">
                                <span class="label">Failed:</span>
                                <span class="value" id="failed-updates">--</span>
                            </div>
                            <div class="detail">
                                <span class="label">Total:</span>
                                <span class="value" id="total-updates">--</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <h3>Queue Status</h3>
                            <div class="metric-status" id="queue-status"></div>
                        </div>
                        <div class="metric-value">
                            <span class="value" id="queue-size">--</span>
                            <span class="unit">items</span>
                        </div>
                        <div class="metric-chart" id="queue-chart"></div>
                        <div class="metric-details">
                            <div class="detail">
                                <span class="label">Update Queue:</span>
                                <span class="value" id="update-queue">--</span>
                            </div>
                            <div class="detail">
                                <span class="label">Batch Queue:</span>
                                <span class="value" id="batch-queue">--</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-header">
                            <h3>Connections</h3>
                            <div class="metric-status" id="connection-status"></div>
                        </div>
                        <div class="metric-value">
                            <span class="value" id="websocket-connections">--</span>
                            <span class="unit">active</span>
                        </div>
                        <div class="metric-chart" id="connection-chart"></div>
                        <div class="metric-details">
                            <div class="detail">
                                <span class="label">WebSockets:</span>
                                <span class="value" id="active-websockets">--</span>
                            </div>
                            <div class="detail">
                                <span class="label">Groups:</span>
                                <span class="value" id="websocket-groups">--</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="metric-card wide">
                        <div class="metric-header">
                            <h3>Circuit Breaker Status</h3>
                            <div class="metric-status" id="circuit-status"></div>
                        </div>
                        <div class="circuit-breaker-display">
                            <div class="circuit-state">
                                <div class="state-indicator" id="circuit-indicator"></div>
                                <div class="state-info">
                                    <div class="state-label" id="circuit-state">UNKNOWN</div>
                                    <div class="state-detail" id="circuit-detail">Checking status...</div>
                                </div>
                            </div>
                            <div class="circuit-metrics">
                                <div class="metric">
                                    <span class="label">Trips:</span>
                                    <span class="value" id="circuit-trips">--</span>
                                </div>
                                <div class="metric">
                                    <span class="label">Uptime:</span>
                                    <span class="value" id="system-uptime">--</span>
                                </div>
                                <div class="metric">
                                    <span class="label">Target Met:</span>
                                    <span class="value" id="target-met">--</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="metric-card wide">
                        <div class="metric-header">
                            <h3>Performance Timeline</h3>
                            <div class="timeline-controls">
                                <select id="timeline-metric">
                                    <option value="processing_time">Processing Time</option>
                                    <option value="success_rate">Success Rate</option>
                                    <option value="queue_size">Queue Size</option>
                                    <option value="connections">Connections</option>
                                </select>
                            </div>
                        </div>
                        <div class="timeline-chart" id="timeline-chart"></div>
                    </div>
                </div>
            </div>
        `;
        
        this.addStyles();
        this.bindEvents();
        this.initializeCharts();
    }
    
    addStyles() {
        if (document.getElementById('performance-dashboard-styles')) return;
        
        const styles = `
            <style id="performance-dashboard-styles">
                .performance-dashboard {
                    padding: 20px;
                    background: #f8f9fa;
                    min-height: 100vh;
                }
                
                .dashboard-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                    padding: 16px 20px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .dashboard-controls {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }
                
                .dashboard-controls button {
                    padding: 8px 16px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    background: white;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                
                .dashboard-controls button:hover {
                    background: #f5f5f5;
                    border-color: #bbb;
                }
                
                .auto-refresh label {
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    font-size: 14px;
                    color: #666;
                }
                
                .alerts-panel {
                    margin-bottom: 20px;
                }
                
                .alert {
                    padding: 12px 16px;
                    margin-bottom: 8px;
                    border-radius: 6px;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    animation: slideIn 0.3s ease;
                }
                
                .alert.warning {
                    background: #fff3cd;
                    border: 1px solid #ffeaa7;
                    color: #856404;
                }
                
                .alert.critical {
                    background: #f8d7da;
                    border: 1px solid #f5c6cb;
                    color: #721c24;
                }
                
                .alert.success {
                    background: #d4edda;
                    border: 1px solid #c3e6cb;
                    color: #155724;
                }
                
                .alert-close {
                    background: none;
                    border: none;
                    font-size: 18px;
                    cursor: pointer;
                    opacity: 0.6;
                }
                
                .alert-close:hover {
                    opacity: 1;
                }
                
                @keyframes slideIn {
                    from { transform: translateY(-20px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }
                
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                }
                
                .metric-card {
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    transition: transform 0.2s, box-shadow 0.2s;
                }
                
                .metric-card:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                }
                
                .metric-card.wide {
                    grid-column: span 2;
                }
                
                .metric-card.primary {
                    border-left: 4px solid #2196F3;
                }
                
                .metric-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 16px;
                }
                
                .metric-header h3 {
                    margin: 0;
                    font-size: 16px;
                    color: #333;
                }
                
                .metric-status {
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background: #ccc;
                    transition: background 0.3s;
                }
                
                .metric-status.good {
                    background: #4CAF50;
                }
                
                .metric-status.warning {
                    background: #FF9800;
                }
                
                .metric-status.critical {
                    background: #f44336;
                }
                
                .metric-value {
                    display: flex;
                    align-items: baseline;
                    margin-bottom: 16px;
                }
                
                .metric-value .value {
                    font-size: 32px;
                    font-weight: 600;
                    color: #333;
                    margin-right: 8px;
                }
                
                .metric-value .unit {
                    font-size: 14px;
                    color: #666;
                }
                
                .metric-chart {
                    height: 80px;
                    margin-bottom: 16px;
                    background: #f8f9fa;
                    border-radius: 4px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 12px;
                    color: #666;
                }
                
                .metric-details {
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }
                
                .detail {
                    display: flex;
                    justify-content: space-between;
                    font-size: 13px;
                }
                
                .detail .label {
                    color: #666;
                }
                
                .detail .value {
                    font-weight: 500;
                    color: #333;
                }
                
                .circuit-breaker-display {
                    display: flex;
                    align-items: center;
                    gap: 20px;
                }
                
                .circuit-state {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }
                
                .state-indicator {
                    width: 24px;
                    height: 24px;
                    border-radius: 50%;
                    background: #ccc;
                    transition: all 0.3s;
                    position: relative;
                }
                
                .state-indicator.closed {
                    background: #4CAF50;
                }
                
                .state-indicator.open {
                    background: #f44336;
                    animation: pulse 1s infinite;
                }
                
                .state-indicator.half-open {
                    background: #FF9800;
                }
                
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                
                .state-info {
                    display: flex;
                    flex-direction: column;
                }
                
                .state-label {
                    font-weight: 600;
                    font-size: 14px;
                    color: #333;
                }
                
                .state-detail {
                    font-size: 12px;
                    color: #666;
                }
                
                .circuit-metrics {
                    display: flex;
                    gap: 20px;
                    margin-left: auto;
                }
                
                .circuit-metrics .metric {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    font-size: 12px;
                }
                
                .circuit-metrics .label {
                    color: #666;
                    margin-bottom: 4px;
                }
                
                .circuit-metrics .value {
                    font-weight: 600;
                    color: #333;
                }
                
                .timeline-controls {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                
                .timeline-controls select {
                    padding: 4px 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    font-size: 12px;
                }
                
                .timeline-chart {
                    height: 200px;
                    background: #f8f9fa;
                    border-radius: 4px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 12px;
                    color: #666;
                }
                
                .loading {
                    opacity: 0.6;
                    pointer-events: none;
                }
                
                .no-data {
                    text-align: center;
                    color: #999;
                    font-style: italic;
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    bindEvents() {
        const refreshBtn = document.getElementById('refresh-metrics');
        const resetBtn = document.getElementById('reset-metrics');
        const exportBtn = document.getElementById('export-metrics');
        const autoRefreshCheckbox = document.getElementById('auto-refresh');
        const timelineMetricSelect = document.getElementById('timeline-metric');
        
        refreshBtn?.addEventListener('click', () => {
            this.refreshMetrics();
        });
        
        resetBtn?.addEventListener('click', () => {
            this.resetMetrics();
        });
        
        exportBtn?.addEventListener('click', () => {
            this.exportMetrics();
        });
        
        autoRefreshCheckbox?.addEventListener('change', (e) => {
            if (e.target.checked) {
                this.startRefreshTimer();
            } else {
                this.stopRefreshTimer();
            }
        });
        
        timelineMetricSelect?.addEventListener('change', (e) => {
            this.updateTimelineChart(e.target.value);
        });
    }
    
    setupEventListeners() {
        // Listen for system performance metrics
        this.stateManager.wsClient.on('performance_metrics', (data) => {
            this.updateMetrics(data);
        });
        
        this.stateManager.wsClient.on('system_status', (data) => {
            this.updateSystemStatus(data);
        });
    }
    
    initializeCharts() {
        // Initialize mini charts for each metric
        this.createMiniChart('processing-chart', 'processing_time');
        this.createMiniChart('success-chart', 'success_rate');
        this.createMiniChart('queue-chart', 'queue_size');
        this.createMiniChart('connection-chart', 'connections');
        this.createTimelineChart();
    }
    
    createMiniChart(containerId, metricType) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '<div class="mini-chart-placeholder">Chart will appear here</div>';
        
        // Store chart reference for updates
        this.charts[containerId] = {
            container,
            metricType,
            data: []
        };
    }
    
    createTimelineChart() {
        const container = document.getElementById('timeline-chart');
        if (!container) return;
        
        container.innerHTML = '<div class="timeline-placeholder">Timeline chart will appear here</div>';
        
        this.charts.timeline = {
            container,
            data: []
        };
    }
    
    async loadInitialMetrics() {
        try {
            // Request system status and performance metrics
            this.stateManager.wsClient.requestSystemStatus();
            this.stateManager.wsClient.requestPerformanceMetrics();
            
            // Also fetch via API as fallback
            const response = await fetch('/api/unified/status/');
            const data = await response.json();
            
            if (data.pipeline_metrics) {
                this.updateMetrics(data.pipeline_metrics);
            }
            
            if (data.performance) {
                this.updateSystemStatus(data);
            }
        } catch (error) {
            console.error('Failed to load initial metrics:', error);
            this.showAlert('Failed to load performance metrics', 'warning');
        }
    }
    
    updateMetrics(metrics) {
        if (!metrics) return;
        
        this.currentMetrics = metrics;
        this.addToHistory(metrics);
        
        // Update primary metrics
        this.updateProcessingTime(metrics);
        this.updateSuccessRate(metrics);
        this.updateQueueStatus(metrics);
        this.updateConnections(metrics);
        this.updateCircuitBreaker(metrics);
        
        // Update charts
        this.updateMiniCharts();
        this.updateTimelineChart();
        
        // Check for alerts
        this.checkAlerts(metrics);
        
        // Update timestamp
        this.updateLastRefresh();
    }
    
    updateProcessingTime(metrics) {
        const processingTime = metrics.average_processing_time_ms || 0;
        const batchTime = metrics.batch_processing_time_ms || 0;
        const websocketTime = metrics.websocket_broadcast_time_ms || 0;
        
        document.getElementById('processing-time').textContent = processingTime.toFixed(1);
        document.getElementById('batch-time').textContent = batchTime.toFixed(1) + 'ms';
        document.getElementById('websocket-time').textContent = websocketTime.toFixed(1) + 'ms';
        
        // Update status indicator
        const status = document.getElementById('processing-status');
        if (processingTime < this.config.thresholds.processing_time_warning) {
            status.className = 'metric-status good';
        } else if (processingTime < this.config.thresholds.processing_time_critical) {
            status.className = 'metric-status warning';
        } else {
            status.className = 'metric-status critical';
        }
    }
    
    updateSuccessRate(metrics) {
        const successRate = metrics.success_rate || 0;
        const successful = metrics.successful_updates || 0;
        const failed = metrics.failed_updates || 0;
        const total = metrics.total_updates || 0;
        
        document.getElementById('success-rate').textContent = successRate.toFixed(1);
        document.getElementById('successful-updates').textContent = successful;
        document.getElementById('failed-updates').textContent = failed;
        document.getElementById('total-updates').textContent = total;
        
        // Update status indicator
        const status = document.getElementById('success-status');
        if (successRate >= this.config.thresholds.success_rate_warning) {
            status.className = 'metric-status good';
        } else if (successRate >= this.config.thresholds.success_rate_critical) {
            status.className = 'metric-status warning';
        } else {
            status.className = 'metric-status critical';
        }
    }
    
    updateQueueStatus(metrics) {
        const queueSize = (metrics.queue_size || 0) + (metrics.batch_queue_size || 0);
        const updateQueue = metrics.queue_size || 0;
        const batchQueue = metrics.batch_queue_size || 0;
        
        document.getElementById('queue-size').textContent = queueSize;
        document.getElementById('update-queue').textContent = updateQueue;
        document.getElementById('batch-queue').textContent = batchQueue;
        
        // Update status indicator
        const status = document.getElementById('queue-status');
        if (queueSize < this.config.thresholds.queue_size_warning) {
            status.className = 'metric-status good';
        } else if (queueSize < this.config.thresholds.queue_size_critical) {
            status.className = 'metric-status warning';
        } else {
            status.className = 'metric-status critical';
        }
    }
    
    updateConnections(metrics) {
        const activeWebsockets = metrics.active_websockets || 0;
        const websocketGroups = metrics.websocket_groups || 0;
        
        document.getElementById('websocket-connections').textContent = activeWebsockets;
        document.getElementById('active-websockets').textContent = activeWebsockets;
        document.getElementById('websocket-groups').textContent = websocketGroups;
        
        // Update status indicator
        const status = document.getElementById('connection-status');
        status.className = activeWebsockets > 0 ? 'metric-status good' : 'metric-status warning';
    }
    
    updateCircuitBreaker(metrics) {
        const circuitState = metrics.circuit_breaker_state || 'unknown';
        const circuitTrips = metrics.circuit_breaker_trips || 0;
        const uptime = metrics.uptime_seconds || 0;
        const targetMet = metrics.performance_target_met || false;
        
        // Update circuit breaker display
        const indicator = document.getElementById('circuit-indicator');
        const stateLabel = document.getElementById('circuit-state');
        const stateDetail = document.getElementById('circuit-detail');
        const status = document.getElementById('circuit-status');
        
        indicator.className = `state-indicator ${circuitState.toLowerCase()}`;
        stateLabel.textContent = circuitState.toUpperCase();
        
        switch (circuitState.toLowerCase()) {
            case 'closed':
                stateDetail.textContent = 'Normal operation';
                status.className = 'metric-status good';
                break;
            case 'open':
                stateDetail.textContent = 'Circuit breaker tripped';
                status.className = 'metric-status critical';
                break;
            case 'half_open':
                stateDetail.textContent = 'Testing recovery';
                status.className = 'metric-status warning';
                break;
            default:
                stateDetail.textContent = 'Status unknown';
                status.className = 'metric-status';
        }
        
        // Update circuit metrics
        document.getElementById('circuit-trips').textContent = circuitTrips;
        document.getElementById('system-uptime').textContent = this.formatUptime(uptime);
        document.getElementById('target-met').textContent = targetMet ? '✅ Yes' : '❌ No';
    }
    
    addToHistory(metrics) {
        const timestamp = Date.now();
        const historyEntry = {
            timestamp,
            ...metrics
        };
        
        this.metricsHistory.push(historyEntry);
        
        // Limit history length
        if (this.metricsHistory.length > this.config.historyLength) {
            this.metricsHistory.shift();
        }
    }
    
    updateMiniCharts() {
        // Update mini sparkline charts
        Object.values(this.charts).forEach(chart => {
            if (chart.metricType && this.metricsHistory.length > 0) {
                this.renderMiniChart(chart);
            }
        });
    }
    
    renderMiniChart(chart) {
        const data = this.metricsHistory.map(entry => {
            switch (chart.metricType) {
                case 'processing_time':
                    return entry.average_processing_time_ms || 0;
                case 'success_rate':
                    return entry.success_rate || 0;
                case 'queue_size':
                    return (entry.queue_size || 0) + (entry.batch_queue_size || 0);
                case 'connections':
                    return entry.active_websockets || 0;
                default:
                    return 0;
            }
        });
        
        if (data.length === 0) return;
        
        // Simple ASCII sparkline
        const max = Math.max(...data);
        const min = Math.min(...data);
        const range = max - min || 1;
        
        const sparkline = data.map(value => {
            const normalized = (value - min) / range;
            const height = Math.floor(normalized * 8);
            return '▁▂▃▄▅▆▇█'[height] || '▁';
        }).join('');
        
        chart.container.innerHTML = `
            <div style="font-family: monospace; font-size: 16px; text-align: center; color: #2196F3;">
                ${sparkline}
            </div>
            <div style="font-size: 10px; text-align: center; color: #666; margin-top: 4px;">
                ${data.length} data points
            </div>
        `;
    }
    
    updateTimelineChart(selectedMetric = 'processing_time') {
        const timelineChart = this.charts.timeline;
        if (!timelineChart || this.metricsHistory.length === 0) return;
        
        // Create simple timeline visualization
        const data = this.metricsHistory.slice(-20); // Last 20 data points
        const values = data.map(entry => {
            switch (selectedMetric) {
                case 'processing_time':
                    return entry.average_processing_time_ms || 0;
                case 'success_rate':
                    return entry.success_rate || 0;
                case 'queue_size':
                    return (entry.queue_size || 0) + (entry.batch_queue_size || 0);
                case 'connections':
                    return entry.active_websockets || 0;
                default:
                    return 0;
            }
        });
        
        const max = Math.max(...values);
        const min = Math.min(...values);
        
        timelineChart.container.innerHTML = `
            <div style="display: flex; align-items: end; height: 160px; gap: 2px; padding: 20px;">
                ${values.map((value, index) => {
                    const height = max > 0 ? ((value - min) / (max - min)) * 140 : 0;
                    const isRecent = index >= values.length - 3;
                    return `
                        <div style="
                            width: 15px;
                            height: ${Math.max(height, 2)}px;
                            background: ${isRecent ? '#2196F3' : '#e0e0e0'};
                            border-radius: 2px;
                            transition: all 0.3s;
                            position: relative;
                        " title="${value.toFixed(1)}">
                        </div>
                    `;
                }).join('')}
            </div>
            <div style="text-align: center; font-size: 12px; color: #666; margin-top: 8px;">
                Range: ${min.toFixed(1)} - ${max.toFixed(1)} | Latest: ${values[values.length - 1]?.toFixed(1) || 'N/A'}
            </div>
        `;
    }
    
    checkAlerts(metrics) {
        if (!this.config.showAlerts) return;
        
        const newAlerts = [];
        
        // Processing time alerts
        const processingTime = metrics.average_processing_time_ms || 0;
        if (processingTime >= this.config.thresholds.processing_time_critical) {
            newAlerts.push({
                type: 'critical',
                message: `Processing time critical: ${processingTime.toFixed(1)}ms (target: <100ms)`,
                id: 'processing_time_critical'
            });
        } else if (processingTime >= this.config.thresholds.processing_time_warning) {
            newAlerts.push({
                type: 'warning',
                message: `Processing time warning: ${processingTime.toFixed(1)}ms (target: <100ms)`,
                id: 'processing_time_warning'
            });
        }
        
        // Success rate alerts
        const successRate = metrics.success_rate || 0;
        if (successRate < this.config.thresholds.success_rate_critical) {
            newAlerts.push({
                type: 'critical',
                message: `Success rate critical: ${successRate.toFixed(1)}% (target: >90%)`,
                id: 'success_rate_critical'
            });
        } else if (successRate < this.config.thresholds.success_rate_warning) {
            newAlerts.push({
                type: 'warning',
                message: `Success rate warning: ${successRate.toFixed(1)}% (target: >95%)`,
                id: 'success_rate_warning'
            });
        }
        
        // Queue size alerts
        const queueSize = (metrics.queue_size || 0) + (metrics.batch_queue_size || 0);
        if (queueSize >= this.config.thresholds.queue_size_critical) {
            newAlerts.push({
                type: 'critical',
                message: `Queue size critical: ${queueSize} items (threshold: ${this.config.thresholds.queue_size_critical})`,
                id: 'queue_size_critical'
            });
        } else if (queueSize >= this.config.thresholds.queue_size_warning) {
            newAlerts.push({
                type: 'warning',
                message: `Queue size warning: ${queueSize} items (threshold: ${this.config.thresholds.queue_size_warning})`,
                id: 'queue_size_warning'
            });
        }
        
        // Circuit breaker alerts
        const circuitState = metrics.circuit_breaker_state || 'unknown';
        if (circuitState === 'open') {
            newAlerts.push({
                type: 'critical',
                message: 'Circuit breaker is OPEN - system protection activated',
                id: 'circuit_breaker_open'
            });
        }
        
        // Show new alerts
        newAlerts.forEach(alert => {
            this.showAlert(alert.message, alert.type, alert.id);
        });
    }
    
    showAlert(message, type = 'warning', id = null) {
        const alertsPanel = document.getElementById('alerts-panel');
        if (!alertsPanel) return;
        
        // Check if alert already exists
        if (id && alertsPanel.querySelector(`[data-alert-id="${id}"]`)) {
            return;
        }
        
        const alert = document.createElement('div');
        alert.className = `alert ${type}`;
        if (id) alert.setAttribute('data-alert-id', id);
        
        alert.innerHTML = `
            <span>${message}</span>
            <button class="alert-close">&times;</button>
        `;
        
        alert.querySelector('.alert-close').addEventListener('click', () => {
            alertsPanel.removeChild(alert);
        });
        
        alertsPanel.appendChild(alert);
        
        // Auto-remove after 30 seconds for warnings, keep critical alerts
        if (type === 'warning') {
            setTimeout(() => {
                if (alert.parentNode) {
                    alertsPanel.removeChild(alert);
                }
            }, 30000);
        }
    }
    
    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }
    
    updateLastRefresh() {
        // Could add a last refresh timestamp display
    }
    
    refreshMetrics() {
        this.stateManager.wsClient.requestPerformanceMetrics();
        this.stateManager.wsClient.requestSystemStatus();
    }
    
    async resetMetrics() {
        try {
            const response = await fetch('/api/pipeline/validate/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ type: 'reset_metrics' })
            });
            
            if (response.ok) {
                this.showAlert('Metrics reset successfully', 'success');
                this.refreshMetrics();
            } else {
                this.showAlert('Failed to reset metrics', 'warning');
            }
        } catch (error) {
            console.error('Reset metrics error:', error);
            this.showAlert('Error resetting metrics', 'warning');
        }
    }
    
    exportMetrics() {
        const exportData = {
            current_metrics: this.currentMetrics,
            metrics_history: this.metricsHistory,
            config: this.config,
            timestamp: new Date().toISOString()
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `performance-metrics-${Date.now()}.json`;
        link.click();
    }
    
    startRefreshTimer() {
        this.stopRefreshTimer();
        this.refreshTimer = setInterval(() => {
            this.refreshMetrics();
        }, this.config.refreshInterval);
    }
    
    stopRefreshTimer() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }
    
    updateSystemStatus(data) {
        // Handle system status updates
        console.log('System status update:', data);
    }
    
    destroy() {
        this.stopRefreshTimer();
        this.container.innerHTML = '';
    }
}

// Export for use
window.PerformanceDashboard = PerformanceDashboard;