/**
 * Social Network Real-time Visualization
 * D3.js-based interactive network graph with real-time updates
 */

class SocialNetworkVisualization {
    constructor(containerId, stateManager, options = {}) {
        this.containerId = containerId;
        this.stateManager = stateManager;
        this.container = document.getElementById(containerId);
        
        if (!this.container) {
            throw new Error(`Container ${containerId} not found`);
        }
        
        // Configuration
        this.config = {
            width: options.width || 800,
            height: options.height || 600,
            nodeRadius: options.nodeRadius || 20,
            linkDistance: options.linkDistance || 100,
            linkStrength: options.linkStrength || 0.1,
            chargeStrength: options.chargeStrength || -300,
            animationDuration: options.animationDuration || 1000,
            showLabels: options.showLabels !== false,
            showMetrics: options.showMetrics !== false,
            colorScheme: options.colorScheme || 'relationship'
        };
        
        // State
        this.nodes = [];
        this.links = [];
        this.simulation = null;
        this.svg = null;
        this.nodeGroup = null;
        this.linkGroup = null;
        this.labelGroup = null;
        
        // Scales and colors
        this.colorScale = null;
        this.sizeScale = null;
        
        this.init();
    }
    
    init() {
        this.createVisualization();
        this.setupEventListeners();
        this.loadInitialData();
    }
    
    createVisualization() {
        // Clear container
        this.container.innerHTML = '';
        
        // Create control panel
        this.createControlPanel();
        
        // Create SVG
        this.svg = d3.select(this.container)
            .append('div')
            .attr('class', 'network-viz-container')
            .append('svg')
            .attr('width', this.config.width)
            .attr('height', this.config.height)
            .attr('class', 'social-network-svg');
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                this.svg.select('.network-content').attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        // Create main content group
        const content = this.svg.append('g').attr('class', 'network-content');
        
        // Create groups for different elements
        this.linkGroup = content.append('g').attr('class', 'links');
        this.nodeGroup = content.append('g').attr('class', 'nodes');
        this.labelGroup = content.append('g').attr('class', 'labels');
        
        // Create force simulation
        this.simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(this.config.linkDistance).strength(this.config.linkStrength))
            .force('charge', d3.forceManyBody().strength(this.config.chargeStrength))
            .force('center', d3.forceCenter(this.config.width / 2, this.config.height / 2))
            .force('collision', d3.forceCollide().radius(this.config.nodeRadius + 5));
        
        this.addStyles();
    }
    
    createControlPanel() {
        const controlPanel = document.createElement('div');
        controlPanel.className = 'network-controls';
        controlPanel.innerHTML = `
            <div class="control-group">
                <label>
                    <input type="checkbox" id="show-labels" ${this.config.showLabels ? 'checked' : ''}>
                    Show Labels
                </label>
                <label>
                    <input type="checkbox" id="show-metrics" ${this.config.showMetrics ? 'checked' : ''}>
                    Show Metrics
                </label>
            </div>
            <div class="control-group">
                <label>
                    Color by:
                    <select id="color-scheme">
                        <option value="relationship" ${this.config.colorScheme === 'relationship' ? 'selected' : ''}>Relationship Strength</option>
                        <option value="role" ${this.config.colorScheme === 'role' ? 'selected' : ''}>Agent Role</option>
                        <option value="emotion" ${this.config.colorScheme === 'emotion' ? 'selected' : ''}>Emotional State</option>
                        <option value="activity" ${this.config.colorScheme === 'activity' ? 'selected' : ''}>Activity Level</option>
                    </select>
                </label>
            </div>
            <div class="control-group">
                <button id="refresh-network">🔄 Refresh</button>
                <button id="center-network">🎯 Center</button>
                <button id="export-network">💾 Export</button>
            </div>
            <div class="network-stats">
                <div class="stat">
                    <span class="stat-label">Nodes:</span>
                    <span class="stat-value" id="node-count">0</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Connections:</span>
                    <span class="stat-value" id="link-count">0</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Avg Relationship:</span>
                    <span class="stat-value" id="avg-relationship">0%</span>
                </div>
            </div>
        `;
        
        this.container.appendChild(controlPanel);
        this.bindControlEvents(controlPanel);
    }
    
    bindControlEvents(controlPanel) {
        const showLabelsCheckbox = controlPanel.querySelector('#show-labels');
        const showMetricsCheckbox = controlPanel.querySelector('#show-metrics');
        const colorSchemeSelect = controlPanel.querySelector('#color-scheme');
        const refreshBtn = controlPanel.querySelector('#refresh-network');
        const centerBtn = controlPanel.querySelector('#center-network');
        const exportBtn = controlPanel.querySelector('#export-network');
        
        showLabelsCheckbox?.addEventListener('change', (e) => {
            this.config.showLabels = e.target.checked;
            this.updateLabels();
        });
        
        showMetricsCheckbox?.addEventListener('change', (e) => {
            this.config.showMetrics = e.target.checked;
            this.updateMetrics();
        });
        
        colorSchemeSelect?.addEventListener('change', (e) => {
            this.config.colorScheme = e.target.value;
            this.updateColors();
        });
        
        refreshBtn?.addEventListener('click', () => {
            this.refreshNetwork();
        });
        
        centerBtn?.addEventListener('click', () => {
            this.centerNetwork();
        });
        
        exportBtn?.addEventListener('click', () => {
            this.exportNetwork();
        });
    }
    
    setupEventListeners() {
        // Listen for social network updates
        this.stateManager.observe('social_network', (data) => {
            this.updateNetwork(data);
        });
        
        // Listen for individual agent updates
        this.stateManager.wsClient.on('agent_updated', (data) => {
            this.updateAgentNode(data.agentId, data.state);
        });
        
        // Request initial social network data
        this.stateManager.wsClient.requestSocialNetwork();
    }
    
    async loadInitialData() {
        try {
            // Fetch initial social network data from API
            const response = await fetch('/api/social/network/');
            const data = await response.json();
            
            if (data.social_network) {
                this.updateNetwork(data.social_network);
            }
        } catch (error) {
            console.error('Failed to load initial social network data:', error);
        }
    }
    
    updateNetwork(networkData) {
        if (!networkData) return;
        
        // Process network data
        this.processNetworkData(networkData);
        
        // Update visualization
        this.updateVisualization();
        
        // Update statistics
        this.updateStatistics();
    }
    
    processNetworkData(networkData) {
        // Extract nodes and links from network data
        const nodes = new Map();
        const links = [];
        
        // Create nodes from agents
        if (networkData.agents) {
            Object.entries(networkData.agents).forEach(([agentId, agentData]) => {
                nodes.set(agentId, {
                    id: agentId,
                    name: agentData.name || agentId,
                    role: agentData.role || 'unknown',
                    emotional_state: agentData.emotional_state || {},
                    current_location: agentData.current_location || 'unknown',
                    current_activity: agentData.current_activity || 'idle',
                    relationship_scores: agentData.relationship_scores || {},
                    // Visual properties
                    size: this.calculateNodeSize(agentData),
                    color: this.calculateNodeColor(agentData),
                    x: agentData.position?.x || Math.random() * this.config.width,
                    y: agentData.position?.y || Math.random() * this.config.height
                });
            });
        }
        
        // Create links from relationships
        nodes.forEach((sourceNode, sourceId) => {
            Object.entries(sourceNode.relationship_scores).forEach(([targetId, strength]) => {
                if (nodes.has(targetId) && strength > 0.1) { // Only show meaningful relationships
                    links.push({
                        source: sourceId,
                        target: targetId,
                        strength: strength,
                        width: Math.max(1, strength * 5),
                        color: this.getRelationshipColor(strength)
                    });
                }
            });
        });
        
        this.nodes = Array.from(nodes.values());
        this.links = links;
    }
    
    updateVisualization() {
        // Update nodes
        const nodeSelection = this.nodeGroup
            .selectAll('.node')
            .data(this.nodes, d => d.id);
        
        // Remove old nodes
        nodeSelection.exit().remove();
        
        // Add new nodes
        const nodeEnter = nodeSelection.enter()
            .append('g')
            .attr('class', 'node')
            .call(this.createDragBehavior());
        
        nodeEnter.append('circle')
            .attr('r', 0)
            .transition()
            .duration(this.config.animationDuration)
            .attr('r', d => d.size);
        
        nodeEnter.append('text')
            .attr('class', 'node-emoji')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .text(d => this.getAgentEmoji(d.role))
            .style('font-size', '16px')
            .style('pointer-events', 'none');
        
        // Update existing nodes
        const nodeUpdate = nodeEnter.merge(nodeSelection);
        
        nodeUpdate.select('circle')
            .transition()
            .duration(this.config.animationDuration)
            .attr('r', d => d.size)
            .attr('fill', d => d.color)
            .attr('stroke', '#fff')
            .attr('stroke-width', 2);
        
        nodeUpdate.select('.node-emoji')
            .text(d => this.getAgentEmoji(d.role));
        
        // Add tooltips
        nodeUpdate
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip())
            .on('click', (event, d) => this.selectNode(d));
        
        // Update links
        const linkSelection = this.linkGroup
            .selectAll('.link')
            .data(this.links);
        
        linkSelection.exit().remove();
        
        const linkEnter = linkSelection.enter()
            .append('line')
            .attr('class', 'link')
            .attr('stroke-width', 0)
            .transition()
            .duration(this.config.animationDuration)
            .attr('stroke-width', d => d.width);
        
        const linkUpdate = linkEnter.merge(linkSelection);
        
        linkUpdate
            .transition()
            .duration(this.config.animationDuration)
            .attr('stroke', d => d.color)
            .attr('stroke-width', d => d.width)
            .attr('opacity', 0.6);
        
        // Update labels
        this.updateLabels();
        
        // Update simulation
        this.simulation.nodes(this.nodes);
        this.simulation.force('link').links(this.links);
        this.simulation.alpha(1).restart();
        
        // Set up simulation tick
        this.simulation.on('tick', () => {
            linkUpdate
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            nodeUpdate
                .attr('transform', d => `translate(${d.x},${d.y})`);
            
            this.labelGroup.selectAll('.node-label')
                .attr('transform', d => `translate(${d.x},${d.y + d.size + 15})`);
        });
    }
    
    updateLabels() {
        if (!this.config.showLabels) {
            this.labelGroup.selectAll('.node-label').remove();
            return;
        }
        
        const labelSelection = this.labelGroup
            .selectAll('.node-label')
            .data(this.nodes, d => d.id);
        
        labelSelection.exit().remove();
        
        const labelEnter = labelSelection.enter()
            .append('text')
            .attr('class', 'node-label')
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('fill', '#333')
            .style('pointer-events', 'none');
        
        labelEnter.merge(labelSelection)
            .text(d => d.name);
    }
    
    calculateNodeSize(agentData) {
        // Size based on relationship count and activity
        const relationshipCount = Object.keys(agentData.relationship_scores || {}).length;
        const baseSize = this.config.nodeRadius;
        const sizeMultiplier = 1 + (relationshipCount * 0.1);
        return Math.min(baseSize * sizeMultiplier, baseSize * 2);
    }
    
    calculateNodeColor(agentData) {
        switch (this.config.colorScheme) {
            case 'relationship':
                const avgRelationship = this.getAverageRelationship(agentData.relationship_scores);
                return d3.interpolateRdYlBu(1 - avgRelationship);
            
            case 'role':
                return this.getRoleColor(agentData.role);
            
            case 'emotion':
                const happiness = agentData.emotional_state?.happiness || 0;
                return d3.interpolateRdYlGn(happiness);
            
            case 'activity':
                const isActive = agentData.current_activity !== 'idle';
                return isActive ? '#4CAF50' : '#9E9E9E';
            
            default:
                return '#2196F3';
        }
    }
    
    getAverageRelationship(relationships) {
        if (!relationships || Object.keys(relationships).length === 0) return 0;
        const values = Object.values(relationships);
        return values.reduce((sum, val) => sum + val, 0) / values.length;
    }
    
    getRoleColor(role) {
        const roleColors = {
            'host': '#FF5722',
            'contestant': '#E91E63',
            'judge': '#9C27B0',
            'producer': '#673AB7',
            'camera': '#3F51B5',
            'audience': '#2196F3',
            'chef': '#FF9800',
            'stylist': '#CDDC39',
            'doctor': '#4CAF50',
            'therapist': '#009688'
        };
        
        return roleColors[role?.toLowerCase()] || '#607D8B';
    }
    
    getRelationshipColor(strength) {
        return d3.interpolateRdYlGn(strength);
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
            'therapist': '🧠'
        };
        
        return emojiMap[role?.toLowerCase()] || '👤';
    }
    
    createDragBehavior() {
        return d3.drag()
            .on('start', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            })
            .on('drag', (event, d) => {
                d.fx = event.x;
                d.fy = event.y;
            })
            .on('end', (event, d) => {
                if (!event.active) this.simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            });
    }
    
    showTooltip(event, d) {
        const tooltip = d3.select('body').append('div')
            .attr('class', 'network-tooltip')
            .style('opacity', 0);
        
        const relationships = Object.entries(d.relationship_scores)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 3);
        
        tooltip.html(`
            <div class="tooltip-header">
                <strong>${d.name}</strong>
                <span class="tooltip-role">${d.role}</span>
            </div>
            <div class="tooltip-body">
                <div>Location: ${d.current_location}</div>
                <div>Activity: ${d.current_activity}</div>
                <div class="relationships">
                    <strong>Top Relationships:</strong>
                    ${relationships.map(([name, score]) => 
                        `<div>${name}: ${(score * 100).toFixed(0)}%</div>`
                    ).join('')}
                </div>
            </div>
        `)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px')
        .transition()
        .duration(200)
        .style('opacity', 1);
    }
    
    hideTooltip() {
        d3.selectAll('.network-tooltip').remove();
    }
    
    selectNode(node) {
        console.log('Selected node:', node);
        // Highlight connected nodes and links
        this.highlightConnections(node);
    }
    
    highlightConnections(selectedNode) {
        // Reset all nodes and links
        this.nodeGroup.selectAll('.node circle')
            .style('opacity', 0.3);
        
        this.linkGroup.selectAll('.link')
            .style('opacity', 0.1);
        
        // Highlight selected node
        this.nodeGroup.selectAll('.node')
            .filter(d => d.id === selectedNode.id)
            .select('circle')
            .style('opacity', 1)
            .style('stroke', '#FF5722')
            .style('stroke-width', 4);
        
        // Highlight connected nodes and links
        const connectedNodes = new Set([selectedNode.id]);
        
        this.links.forEach(link => {
            if (link.source.id === selectedNode.id || link.target.id === selectedNode.id) {
                connectedNodes.add(link.source.id);
                connectedNodes.add(link.target.id);
                
                // Highlight link
                this.linkGroup.selectAll('.link')
                    .filter(d => d === link)
                    .style('opacity', 0.8)
                    .style('stroke-width', d => d.width * 2);
            }
        });
        
        // Highlight connected nodes
        this.nodeGroup.selectAll('.node')
            .filter(d => connectedNodes.has(d.id))
            .select('circle')
            .style('opacity', 1);
        
        // Clear highlights after 3 seconds
        setTimeout(() => {
            this.clearHighlights();
        }, 3000);
    }
    
    clearHighlights() {
        this.nodeGroup.selectAll('.node circle')
            .style('opacity', 1)
            .style('stroke', '#fff')
            .style('stroke-width', 2);
        
        this.linkGroup.selectAll('.link')
            .style('opacity', 0.6)
            .style('stroke-width', d => d.width);
    }
    
    updateAgentNode(agentId, agentState) {
        // Update specific node data
        const nodeIndex = this.nodes.findIndex(n => n.id === agentId);
        if (nodeIndex !== -1) {
            this.nodes[nodeIndex] = {
                ...this.nodes[nodeIndex],
                name: agentState.name || this.nodes[nodeIndex].name,
                role: agentState.role || this.nodes[nodeIndex].role,
                emotional_state: agentState.emotional_state || this.nodes[nodeIndex].emotional_state,
                current_location: agentState.current_location || this.nodes[nodeIndex].current_location,
                current_activity: agentState.current_activity || this.nodes[nodeIndex].current_activity,
                relationship_scores: agentState.relationship_scores || this.nodes[nodeIndex].relationship_scores,
                size: this.calculateNodeSize(agentState),
                color: this.calculateNodeColor(agentState)
            };
            
            // Update visualization
            this.updateVisualization();
        }
    }
    
    updateColors() {
        this.nodes.forEach(node => {
            node.color = this.calculateNodeColor(node);
        });
        
        this.nodeGroup.selectAll('.node circle')
            .transition()
            .duration(500)
            .attr('fill', d => d.color);
    }
    
    updateMetrics() {
        if (!this.config.showMetrics) return;
        
        // Update network statistics display
        // This would be implemented based on your specific metrics needs
    }
    
    updateStatistics() {
        const nodeCount = this.nodes.length;
        const linkCount = this.links.length;
        const avgRelationship = this.links.reduce((sum, link) => sum + link.strength, 0) / linkCount || 0;
        
        const nodeCountEl = document.getElementById('node-count');
        const linkCountEl = document.getElementById('link-count');
        const avgRelationshipEl = document.getElementById('avg-relationship');
        
        if (nodeCountEl) nodeCountEl.textContent = nodeCount;
        if (linkCountEl) linkCountEl.textContent = linkCount;
        if (avgRelationshipEl) avgRelationshipEl.textContent = `${(avgRelationship * 100).toFixed(0)}%`;
    }
    
    refreshNetwork() {
        this.stateManager.wsClient.requestSocialNetwork();
    }
    
    centerNetwork() {
        const transform = d3.zoomIdentity.translate(0, 0).scale(1);
        this.svg.transition()
            .duration(750)
            .call(this.svg.__zoom.transform, transform);
    }
    
    exportNetwork() {
        // Export network data as JSON
        const networkData = {
            nodes: this.nodes,
            links: this.links,
            config: this.config,
            timestamp: new Date().toISOString()
        };
        
        const dataStr = JSON.stringify(networkData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `social-network-${Date.now()}.json`;
        link.click();
    }
    
    addStyles() {
        if (document.getElementById('social-network-styles')) return;
        
        const styles = `
            <style id="social-network-styles">
                .network-viz-container {
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    overflow: hidden;
                    background: #fafafa;
                }
                
                .network-controls {
                    background: white;
                    padding: 16px;
                    border-bottom: 1px solid #ddd;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    flex-wrap: wrap;
                    gap: 16px;
                }
                
                .control-group {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                }
                
                .control-group label {
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    font-size: 14px;
                }
                
                .control-group select {
                    padding: 4px 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                
                .control-group button {
                    padding: 6px 12px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    background: white;
                    cursor: pointer;
                    font-size: 12px;
                    transition: background 0.2s;
                }
                
                .control-group button:hover {
                    background: #f5f5f5;
                }
                
                .network-stats {
                    display: flex;
                    gap: 16px;
                    font-size: 12px;
                }
                
                .stat {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }
                
                .stat-label {
                    color: #666;
                    font-size: 10px;
                }
                
                .stat-value {
                    font-weight: bold;
                    color: #333;
                }
                
                .social-network-svg {
                    display: block;
                    background: white;
                }
                
                .node {
                    cursor: pointer;
                }
                
                .node circle {
                    transition: all 0.3s ease;
                }
                
                .node:hover circle {
                    stroke-width: 3px !important;
                }
                
                .link {
                    transition: all 0.3s ease;
                }
                
                .node-label {
                    font-family: Arial, sans-serif;
                    font-size: 12px;
                    text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
                }
                
                .network-tooltip {
                    position: absolute;
                    padding: 10px;
                    background: rgba(0,0,0,0.9);
                    color: white;
                    border-radius: 6px;
                    font-size: 12px;
                    pointer-events: none;
                    z-index: 1000;
                    max-width: 200px;
                }
                
                .tooltip-header {
                    border-bottom: 1px solid rgba(255,255,255,0.3);
                    padding-bottom: 6px;
                    margin-bottom: 6px;
                }
                
                .tooltip-role {
                    color: #ccc;
                    font-size: 10px;
                    margin-left: 8px;
                }
                
                .tooltip-body div {
                    margin-bottom: 4px;
                }
                
                .relationships {
                    margin-top: 6px;
                    padding-top: 6px;
                    border-top: 1px solid rgba(255,255,255,0.3);
                }
                
                .relationships div {
                    margin-left: 8px;
                    font-size: 11px;
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    destroy() {
        if (this.simulation) {
            this.simulation.stop();
        }
        this.container.innerHTML = '';
    }
}

// Export for use
window.SocialNetworkVisualization = SocialNetworkVisualization;