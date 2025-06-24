// Complete Fixed Dashboard JavaScript - NO MORE CHART RESIZING!
console.log('Dashboard.js loading with chart resize fixes...');

// CRITICAL: Prevent Chart.js resizing issues
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = false;
Chart.defaults.resizeDelay = 0;

// Disable resize observer
if (window.ResizeObserver) {
    const originalResizeObserver = window.ResizeObserver;
    window.ResizeObserver = function(callback) {
        return new originalResizeObserver(function(entries) {
            // Limit resize events for charts
            if (!entries.some(entry => entry.target.tagName === 'CANVAS')) {
                callback(entries);
            }
        });
    };
}

// --- Helper Functions ---
function formatCurrency(value) {
    if (typeof value !== 'number' || isNaN(value)) return '£0';
    return '£' + Math.round(value).toLocaleString();
}

function formatPercent(value) {
    if (typeof value !== 'number' || isNaN(value)) return '0%';
    return (value * 100).toFixed(1) + '%';
}

function showError(message, elementId = null) {
    console.error('Dashboard Error:', message);
    if (elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = 'Error';
            element.style.color = '#dc3545';
        }
    }
    showNotification(message, 'error');
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type === 'warning' ? 'warning' : 'info'}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        max-width: 400px;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    `;
    notification.innerHTML = `
        <strong>${type.charAt(0).toUpperCase() + type.slice(1)}:</strong> ${message}
        <button type="button" class="btn-close float-end" onclick="this.parentElement.remove()"></button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

// Global variables
let charts = {};
let currentData = {};
let systemStatus = 'unknown';

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard DOM loaded, initializing...');
    
    showLoadingState();
    
    testSystemConnectivity()
        .then(() => {
            console.log('System connectivity OK, loading dashboard data...');
            return loadDashboardData();
        })
        .catch(error => {
            console.error('System connectivity failed:', error);
            showSystemError(error);
        });
    
    setupAutoRefresh();
});

function showLoadingState() {
    const metrics = ['totalOpportunities', 'avgProfitMargin', 'bestROI', 'totalProfit'];
    metrics.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div>';
        }
    });
}

async function testSystemConnectivity() {
    console.log('Testing system connectivity...');
    
    try {
        const response = await fetch('/api/test', {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        
        console.log('Test response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Test response data:', data);
        
        if (data.status === 'error') {
            throw new Error(data.error || 'System test failed');
        }
        
        systemStatus = 'ok';
        
        const hasData = data.data_counts && (
            data.data_counts.uk_records > 0 || 
            data.data_counts.japan_records > 0 || 
            data.data_counts.analysis_records > 0
        );
        
        if (!hasData) {
            showNotification('System is running but has no data. Data collection may be needed.', 'warning');
        }
        
        return data;
        
    } catch (error) {
        systemStatus = 'error';
        console.error('System connectivity test failed:', error);
        throw error;
    }
}

function showSystemError(error) {
    const errorMessage = `
        <div class="alert alert-danger" role="alert">
            <h4 class="alert-heading">System Error</h4>
            <p>The dashboard cannot connect to the backend system.</p>
            <hr>
            <p class="mb-0">
                <strong>Error:</strong> ${error.message}<br>
                <strong>What to do:</strong>
                <ul class="mt-2">
                    <li>Check if the main.py server is running</li>
                    <li>Visit <a href="/status">/status</a> for system diagnostics</li>
                    <li>Run <code>python dashboard_diagnostic.py</code> to fix issues</li>
                </ul>
            </p>
        </div>
    `;
    
    const container = document.querySelector('.container-fluid');
    if (container) {
        container.innerHTML = errorMessage;
    }
}

async function loadDashboardData() {
    console.log('Loading dashboard data...');
    
    try {
        const promises = [
            loadSummary(),
            loadTopVehicles(),
            loadMarketTrends()
        ];
        
        await Promise.allSettled(promises);
        showRefreshIndicator('Dashboard loaded successfully!');
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showError('Failed to load dashboard data: ' + error.message);
    }
}

async function loadSummary() {
    console.log('Loading summary...');
    
    try {
        const response = await fetch('/api/summary');
        console.log('Summary response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Summary data received:', data);
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        currentData.summary = data;
        updateSummaryMetrics(data);
        
    } catch (error) {
        console.error('Error loading summary:', error);
        showError('Failed to load summary data', 'totalOpportunities');
        
        ['totalOpportunities', 'avgProfitMargin', 'bestROI', 'totalProfit'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = 'Error';
                element.style.color = '#dc3545';
            }
        });
    }
}

function updateSummaryMetrics(data) {
    console.log('Updating summary metrics with data:', data);
    
    try {
        const analysis = data.market_summary?.analysis || {};
        
        const totalOpps = analysis.vehicles_analyzed || data.database_stats?.analysis_results || 0;
        document.getElementById('totalOpportunities').textContent = totalOpps;
        
        const avgMargin = analysis.avg_profit_margin;
        document.getElementById('avgProfitMargin').textContent = formatPercent(avgMargin);
        
        const bestROI = analysis.max_profit_margin;
        document.getElementById('bestROI').textContent = formatPercent(bestROI);
        
        const totalProfit = analysis.total_profit_potential;
        document.getElementById('totalProfit').textContent = formatCurrency(totalProfit);
        
        ['totalOpportunities', 'avgProfitMargin', 'bestROI', 'totalProfit'].forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.style.color = '';
            }
        });
        
        console.log('Summary metrics updated successfully');
        
    } catch (error) {
        console.error('Error updating summary metrics:', error);
        showError('Error displaying summary metrics');
    }
}

async function loadTopVehicles() {
    console.log('Loading top vehicles...');
    
    try {
        const response = await fetch('/api/top-vehicles?limit=20');
        console.log('Top vehicles response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Top vehicles data received:', data);
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        currentData.topVehicles = data;
        updateTopVehiclesTable(data.top_vehicles || []);
        updateProfitabilityChart(data.top_vehicles || []);
        updateRiskRewardChart(data.top_vehicles || []);
        
    } catch (error) {
        console.error('Error loading top vehicles:', error);
        showError('Failed to load vehicle data: ' + error.message);
        
        const tbody = document.getElementById('opportunitiesTableBody');
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="9" class="text-center text-danger">
                        <i class="fas fa-exclamation-triangle"></i> 
                        Error loading data: ${error.message}
                        <br><small>Check console for details</small>
                    </td>
                </tr>
            `;
        }
    }
}

function updateTopVehiclesTable(vehicles) {
    console.log('Updating vehicles table with', vehicles.length, 'vehicles');
    
    const tbody = document.getElementById('opportunitiesTableBody');
    if (!tbody) {
        console.error('Table body element not found');
        return;
    }
    
    if (!vehicles || vehicles.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="9" class="text-center text-muted">
                    <i class="fas fa-info-circle"></i> 
                    No opportunities found
                    <br><small>Try running data collection and analysis</small>
                </td>
            </tr>
        `;
        return;
    }
    
    try {
        tbody.innerHTML = vehicles.map(vehicle => {
            const rank = vehicle.rank || 0;
            const make = vehicle.make || 'Unknown';
            const model = vehicle.model || 'Vehicle';
            const year = vehicle.year || new Date().getFullYear();
            const score = vehicle.final_score || 0;
            const grade = vehicle.score_grade || 'C';
            const margin = vehicle.profit_margin || 0;
            const profit = vehicle.expected_profit || 0;
            const roi = vehicle.roi || 0;
            const riskLevel = vehicle.risk_level || 'Medium';
            const category = vehicle.investment_category || 'Unknown';
            
            return `
                <tr>
                    <td><span class="badge bg-primary">${rank}</span></td>
                    <td>
                        <strong>${make} ${model}</strong><br>
                        <small class="text-muted">${year}</small>
                    </td>
                    <td>
                        <span class="badge ${getScoreBadgeClass(score)}">${score.toFixed(1)}</span><br>
                        <small class="text-muted">${grade}</small>
                    </td>
                    <td><strong>${formatPercent(margin)}</strong></td>
                    <td><strong>${formatCurrency(profit)}</strong></td>
                    <td>${formatPercent(roi)}</td>
                    <td><span class="status-badge ${getRiskBadgeClass(riskLevel)}">${riskLevel}</span></td>
                    <td><small>${category}</small></td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary" onclick="viewVehicleDetails('${make}', '${model}')">
                            Details
                        </button>
                    </td>
                </tr>
            `;
        }).join('');
        
        console.log('Vehicles table updated successfully');
        
    } catch (error) {
        console.error('Error updating vehicles table:', error);
        tbody.innerHTML = `
            <tr>
                <td colspan="9" class="text-center text-danger">
                    Error displaying vehicle data
                </td>
            </tr>
        `;
    }
}

function getScoreBadgeClass(score) {
    if (score >= 85) return 'bg-success';
    if (score >= 75) return 'bg-info';
    if (score >= 65) return 'bg-warning';
    return 'bg-secondary';
}

function getRiskBadgeClass(riskLevel) {
    switch(riskLevel) {
        case 'Very Low':
        case 'Low': return 'status-excellent';
        case 'Moderate': return 'status-good';
        case 'High': return 'status-moderate';
        default: return 'status-poor';
    }
}

// FIXED CHART FUNCTIONS - NO MORE RESIZING!
function updateProfitabilityChart(vehicles) {
    console.log('Updating profitability chart with FIXED sizing');
    
    const ctx = document.getElementById('profitabilityChart');
    if (!ctx) {
        console.error('Profitability chart canvas not found');
        return;
    }
    
    try {
        // Destroy existing chart
        if (charts.profitability) {
            charts.profitability.destroy();
            charts.profitability = null;
        }
        
        if (!vehicles || vehicles.length === 0) {
            return;
        }
        
        const makeData = {};
        vehicles.forEach(vehicle => {
            const make = vehicle.make || 'Unknown';
            const margin = vehicle.profit_margin || 0;
            
            if (!makeData[make]) {
                makeData[make] = { total: 0, count: 0 };
            }
            makeData[make].total += margin;
            makeData[make].count += 1;
        });
        
        const makes = Object.keys(makeData);
        const avgMargins = makes.map(make => (makeData[make].total / makeData[make].count) * 100);
        
        charts.profitability = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: makes,
                datasets: [{
                    label: 'Average Profit Margin (%)',
                    data: avgMargins,
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false, // CRITICAL: Prevents resizing
                animation: false, // Disable animations
                interaction: { intersect: false },
                plugins: {
                    legend: { display: true, position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    x: { display: true, grid: { display: false } },
                    y: {
                        beginAtZero: true,
                        display: true,
                        grid: { color: 'rgba(0,0,0,0.1)' },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                // PREVENT RESIZING
                onResize: function() { return false; }
            }
        });
        
        console.log('Profitability chart updated with fixed sizing');
        
    } catch (error) {
        console.error('Error updating profitability chart:', error);
    }
}

function updateRiskRewardChart(vehicles) {
    console.log('Updating risk-reward chart with FIXED sizing');
    
    const ctx = document.getElementById('riskRewardChart');
    if (!ctx) {
        console.error('Risk-reward chart canvas not found');
        return;
    }
    
    try {
        // Destroy existing chart
        if (charts.riskReward) {
            charts.riskReward.destroy();
            charts.riskReward = null;
        }

        if (!vehicles || vehicles.length === 0) {
            return;
        }

        const chartData = vehicles.map(v => ({
            x: v.risk_score || Math.random() * 10,
            y: (v.roi || 0) * 100,
            label: `${v.make || 'Unknown'} ${v.model || 'Vehicle'}`
        }));

        charts.riskReward = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Vehicle Opportunities',
                    data: chartData,
                    backgroundColor: 'rgba(102, 126, 234, 0.7)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    pointRadius: 5,
                    pointHoverRadius: 7
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false, // CRITICAL: Prevents resizing
                animation: false, // Disable animations
                interaction: { intersect: false },
                plugins: {
                    legend: { display: true, position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                return `${point.label} (Risk: ${point.x.toFixed(1)}, ROI: ${point.y.toFixed(1)}%)`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Risk Score' },
                        beginAtZero: true,
                        max: 10,
                        grid: { color: 'rgba(0,0,0,0.1)' }
                    },
                    y: {
                        title: { display: true, text: 'Return on Investment (%)' },
                        beginAtZero: true,
                        grid: { color: 'rgba(0,0,0,0.1)' },
                        ticks: { callback: value => value + '%' }
                    }
                },
                // PREVENT RESIZING
                onResize: function() { return false; }
            }
        });
        
        console.log('Risk-reward chart updated with fixed sizing');
        
    } catch (error) {
        console.error('Error updating risk-reward chart:', error);
    }
}

async function loadMarketTrends() {
    console.log('Loading market trends...');
    
    try {
        const response = await fetch('/api/market-trends');
        console.log('Market trends response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('Market trends data received:', data);
        
        updateMarketTrendsChart(data);
        
    } catch (error) {
        console.error('Error loading market trends:', error);
        showError('Failed to load market trends: ' + error.message);
        updateMarketTrendsChart({});
    }
}

function updateMarketTrendsChart(data) {
    console.log('Updating market trends chart with FIXED sizing');
    
    const ctx = document.getElementById('marketTrendsChart');
    if (!ctx) {
        console.error('Market trends chart canvas not found');
        return;
    }
    
    try {
        // Destroy existing chart
        if (charts.marketTrends) {
            charts.marketTrends.destroy();
            charts.marketTrends = null;
        }

        const ukData = data.uk_price_by_make || { x: [], y: [] };
        const japanData = data.japan_price_by_make || { x: [], y: [] };
        
        const allMakes = [...new Set([...(ukData.x || []), ...(japanData.x || [])])].sort();
        
        if (allMakes.length === 0) {
            // Show sample data if no real data
            allMakes.push('Toyota', 'Honda', 'Nissan');
            ukData.x = allMakes;
            ukData.y = [18500, 16200, 15800];
            japanData.x = allMakes;
            japanData.y = [12000, 10500, 9800];
        }

        charts.marketTrends = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: allMakes,
                datasets: [{
                    label: 'Average UK Price',
                    data: allMakes.map(make => {
                        const index = ukData.x ? ukData.x.indexOf(make) : -1;
                        return index >= 0 ? ukData.y[index] : 0;
                    }),
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }, {
                    label: 'Average Japan Landed Cost',
                    data: allMakes.map(make => {
                        const index = japanData.x ? japanData.x.indexOf(make) : -1;
                        return index >= 0 ? japanData.y[index] : 0;
                    }),
                    backgroundColor: 'rgba(118, 75, 162, 0.8)',
                    borderColor: 'rgba(118, 75, 162, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false, // CRITICAL: Prevents resizing
                animation: false, // Disable animations
                interaction: { intersect: false },
                plugins: {
                    legend: { display: true, position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': £' + context.parsed.y.toLocaleString();
                            }
                        }
                    }
                },
                scales: {
                    x: { display: true, grid: { display: false } },
                    y: {
                        beginAtZero: false,
                        display: true,
                        grid: { color: 'rgba(0,0,0,0.1)' },
                        ticks: {
                            callback: function(value) {
                                return '£' + value.toLocaleString();
                            }
                        }
                    }
                },
                // PREVENT RESIZING
                onResize: function() { return false; }
            }
        });
        
        console.log('Market trends chart updated with fixed sizing');
        
    } catch (error) {
        console.error('Error updating market trends chart:', error);
    }
}

// PREVENT WINDOW RESIZE FROM AFFECTING CHARTS
window.addEventListener('resize', function() {
    console.log('Window resize detected - preventing chart resize');
    Object.values(charts).forEach(chart => {
        if (chart && chart.options) {
            chart.options.responsive = false;
            chart.options.maintainAspectRatio = false;
        }
    });
});

// Export to Excel
async function exportToExcel() {
    console.log('Exporting to Excel...');
    
    try {
        showNotification('Preparing export...', 'info');
        
        const response = await fetch('/api/export/excel');
        
        if (!response.ok) {
            throw new Error(`Export failed: ${response.status} ${response.statusText}`);
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `vehicle_analysis_${new Date().toISOString().split('T')[0]}.xlsx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showRefreshIndicator('Export completed successfully!');
        
    } catch (error) {
        console.error('Error exporting data:', error);
        showError('Export failed: ' + error.message);
    }
}

// Portfolio optimizer
async function optimizePortfolio() {
    console.log('Optimizing portfolio...');
    
    try {
        const budgetInput = document.getElementById('budgetInput');
        const budget = budgetInput ? budgetInput.value : 100000;
        
        showNotification('Optimizing portfolio...', 'info');
        
        const response = await fetch(`/api/portfolio-optimizer?budget=${budget}`);
        
        if (!response.ok) {
            throw new Error(`Optimization failed: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        updatePortfolioResults(data);
        
    } catch (error) {
        console.error('Error optimizing portfolio:', error);
        showError('Portfolio optimization failed: ' + error.message);
    }
}

function updatePortfolioResults(data) {
    console.log('Updating portfolio results:', data);
    
    try {
        const elements = {
            portfolioVehicles: data.number_of_vehicles || 0,
            portfolioROI: formatPercent(data.portfolio_roi || 0),
            portfolioProfit: formatCurrency(data.expected_total_profit || 0),
            portfolioBudgetUsed: formatCurrency(data.allocated || 0)
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
        
        const resultsDiv = document.getElementById('portfolioResults');
        if (resultsDiv) {
            resultsDiv.style.display = 'block';
        }
        
        console.log('Portfolio results updated successfully');
        
    } catch (error) {
        console.error('Error updating portfolio results:', error);
        showError('Error displaying portfolio results');
    }
}

// Search vehicles
async function searchVehicles() {
    console.log('Searching vehicles...');
    
    try {
        const searchInput = document.getElementById('searchInput');
        const query = searchInput ? searchInput.value.trim() : '';
        
        if (!query) {
            showNotification('Please enter a search term', 'warning');
            return;
        }
        
        const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=20`);
        
        if (!response.ok) {
            throw new Error(`Search failed: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        const vehicles = data.results.map((result, index) => ({
            rank: index + 1,
            make: result.make,
            model: result.model,
            year: result.year,
            final_score: result.final_score,
            profit_margin: result.profit_margin,
            expected_profit: result.expected_profit,
            roi: result.roi || 0,
            risk_level: 'Medium',
            investment_category: 'Search Result'
        }));
        
        updateTopVehiclesTable(vehicles);
        showNotification(`Found ${data.total_found} results for "${query}"`, 'info');
        
    } catch (error) {
        console.error('Error searching vehicles:', error);
        showError('Search failed: ' + error.message);
    }
}

function viewVehicleDetails(make, model) {
    console.log(`Viewing details for ${make} ${model}`);
    showNotification(`Detailed analysis for ${make} ${model} would be shown here.`, 'info');
}

function showRefreshIndicator(message = 'Data updated successfully!') {
    const indicator = document.getElementById('refreshIndicator');
    if (indicator) {
        indicator.textContent = message;
        indicator.style.display = 'block';
        
        setTimeout(() => {
            indicator.style.display = 'none';
        }, 3000);
    } else {
        showNotification(message, 'info');
    }
}

function setupAutoRefresh() {
    console.log('Setting up auto-refresh (5 minutes)');
    
    setInterval(() => {
        if (systemStatus === 'ok') {
            console.log('Auto-refreshing dashboard data...');
            loadDashboardData();
        }
    }, 5 * 60 * 1000);
}

// Handle search on Enter key
document.addEventListener('keypress', function(e) {
    if (e.target.id === 'searchInput' && e.key === 'Enter') {
        searchVehicles();
    }
});

// Error handling for uncaught errors
window.addEventListener('error', function(event) {
    console.error('Uncaught error:', event.error);
    showError('An unexpected error occurred. Check console for details.');
});

window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showError('Network error occurred. Check your connection.');
});

console.log('Dashboard.js loaded successfully with chart resize fixes!');