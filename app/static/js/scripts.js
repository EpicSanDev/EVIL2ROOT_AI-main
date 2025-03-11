// Exemple simple pour ajouter des interactions
document.addEventListener("DOMContentLoaded", function() {
    console.log("Dashboard loaded");
    // Ajouter des interactions ici
});

// Dashboard functions
document.addEventListener('DOMContentLoaded', function() {
    // Initial data fetch
    fetchBotStatus();
    
    // Set up refresh intervals
    setInterval(fetchBotStatus, 5000);  // Every 5 seconds
    setInterval(updateSystemStats, 10000);  // Every 10 seconds
});

// Fetch bot status from API
async function fetchBotStatus() {
    try {
        const response = await fetch('/bot_status');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update status indicator
        const botState = document.getElementById('bot-state');
        if (botState) {
            botState.textContent = data.state || 'Unknown';
            botState.className = 'status-indicator ' + (data.state || 'unknown');
        }
        
        // Update signals table
        updateSignalsTable(data.signals || []);
        
    } catch (error) {
        console.error('Error fetching bot status:', error);
        // Show error message in the UI
        const botState = document.getElementById('bot-state');
        if (botState) {
            botState.textContent = 'Connection Error';
            botState.className = 'status-indicator error';
        }
    }
}

// Update system statistics
async function updateSystemStats() {
    try {
        // In a real implementation, this would fetch from a system stats endpoint
        // For now, we'll just update the CPU chart with random data
        updateCpuChart();
    } catch (error) {
        console.error('Error updating system stats:', error);
    }
}

// Update CPU chart with fresh data
function updateCpuChart() {
    const cpuChart = document.getElementById('cpu-chart');
    if (!cpuChart) return;
    
    // Récupérer les vraies données CPU depuis l'API
    fetch('/api/system/cpu')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Mettre à jour le graphique avec les vraies données
            Plotly.update('cpu-chart', {
                y: [[data.cpu_percent]]
            });
        })
        .catch(error => {
            console.error('Error fetching CPU data:', error);
        });
}

// Update signals table with new data
function updateSignalsTable(signals) {
    const signalsTable = document.getElementById('signals-data');
    if (!signalsTable) return;
    
    // Clear current table content
    signalsTable.innerHTML = '';
    
    if (signals && signals.length > 0) {
        signals.forEach(signal => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${signal.symbol || 'Unknown'}</td>
                <td class="signal-${signal.decision || 'UNKNOWN'}">${signal.decision || 'Unknown'}</td>
                <td>${signal.timestamp || new Date().toLocaleString()}</td>
                <td>${typeof signal.confidence !== 'undefined' ? signal.confidence : 'N/A'}</td>
            `;
            signalsTable.appendChild(row);
        });
    } else {
        signalsTable.innerHTML = '<tr><td colspan="4">No signals available</td></tr>';
    }
}