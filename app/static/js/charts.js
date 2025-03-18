/**
 * Advanced charts module for the trading bot UI
 * Uses Chart.js and additional plugins for enhanced visualizations
 */

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    
    // Setup refresh intervals for live data
    setInterval(updatePriceCharts, 60000); // Update price charts every minute
    setInterval(updatePerformanceMetrics, 300000); // Update performance every 5 minutes
    setupDarkModeObserver();
});

/**
 * Initialize all charts on the page
 */
function initCharts() {
    // Check which charts are required on the current page
    if (document.getElementById('equity-chart')) {
        initEquityChart();
    }
    
    if (document.getElementById('trade-distribution-chart')) {
        initTradeDistributionChart();
    }
    
    if (document.getElementById('symbol-performance-chart')) {
        initSymbolPerformanceChart();
    }
    
    if (document.getElementById('price-prediction-chart')) {
        initPricePredictionChart();
    }
    
    if (document.getElementById('risk-heatmap')) {
        initRiskHeatmap();
    }
    
    if (document.getElementById('sentiment-timeline')) {
        initSentimentTimeline();
    }
}

/**
 * Initialize equity/balance chart with multi-axis
 */
function initEquityChart() {
    const ctx = document.getElementById('equity-chart').getContext('2d');
    
    // Fetch initial data
    fetch('/api/performance/equity')
        .then(response => response.json())
        .then(data => {
            const equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.dates,
                    datasets: [
                        {
                            label: 'Equity',
                            data: data.equity,
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            fill: true,
                            tension: 0.4,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Balance',
                            data: data.balance,
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                            fill: true,
                            tension: 0.4,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Drawdown',
                            data: data.drawdown,
                            borderColor: 'rgb(255, 99, 132)',
                            backgroundColor: 'rgba(255, 99, 132, 0.1)',
                            fill: true,
                            tension: 0.1,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Portfolio Performance'
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        label += new Intl.NumberFormat('en-US', {
                                            style: 'currency', 
                                            currency: 'USD'
                                        }).format(context.parsed.y);
                                    }
                                    return label;
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        },
                        y: {
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Value ($)'
                            }
                        },
                        y1: {
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Drawdown (%)'
                            },
                            // Grid lines are configured to look like they're coming from the y-axis
                            grid: {
                                drawOnChartArea: false
                            },
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            });
            
            // Store chart instance for later updates
            window.equityChart = equityChart;
        })
        .catch(error => console.error('Error loading equity chart data:', error));
}

/**
 * Initialize trade distribution chart (pie/doughnut)
 */
function initTradeDistributionChart() {
    const ctx = document.getElementById('trade-distribution-chart').getContext('2d');
    
    fetch('/api/performance/trade_distribution')
        .then(response => response.json())
        .then(data => {
            const tradeDistChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Winning Trades', 'Losing Trades', 'Break-even Trades'],
                    datasets: [{
                        data: [data.winning_trades, data.losing_trades, data.breakeven_trades],
                        backgroundColor: [
                            'rgb(75, 192, 192)',  // Green for winning
                            'rgb(255, 99, 132)',  // Red for losing
                            'rgb(201, 203, 207)'  // Grey for break-even
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom',
                        },
                        title: {
                            display: true,
                            text: 'Trade Distribution'
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const label = context.label || '';
                                    const value = context.raw;
                                    const percentage = Math.round(value / data.total_trades * 100);
                                    return `${label}: ${value} (${percentage}%)`;
                                }
                            }
                        }
                    }
                }
            });
            
            window.tradeDistChart = tradeDistChart;
        })
        .catch(error => console.error('Error loading trade distribution data:', error));
}

/**
 * Initialize symbol performance comparison chart
 */
function initSymbolPerformanceChart() {
    const ctx = document.getElementById('symbol-performance-chart').getContext('2d');
    
    fetch('/api/performance/symbol_performance')
        .then(response => response.json())
        .then(data => {
            // Sort symbols by total PnL
            const sortedSymbols = Object.keys(data.symbols).sort((a, b) => 
                data.symbols[b].total_pnl - data.symbols[a].total_pnl
            );
            
            const symbolLabels = sortedSymbols;
            const totalPnl = sortedSymbols.map(symbol => data.symbols[symbol].total_pnl);
            const winRates = sortedSymbols.map(symbol => data.symbols[symbol].win_rate * 100);
            
            const symbolChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: symbolLabels,
                    datasets: [
                        {
                            label: 'Total PnL ($)',
                            data: totalPnl,
                            backgroundColor: totalPnl.map(pnl => 
                                pnl >= 0 ? 'rgba(75, 192, 192, 0.7)' : 'rgba(255, 99, 132, 0.7)'
                            ),
                            borderColor: totalPnl.map(pnl => 
                                pnl >= 0 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'
                            ),
                            borderWidth: 1,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Win Rate (%)',
                            data: winRates,
                            type: 'line',
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            borderWidth: 2,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Symbol Performance Comparison'
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Trading Symbol'
                            }
                        },
                        y: {
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Total PnL ($)'
                            },
                            beginAtZero: false
                        },
                        y1: {
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Win Rate (%)'
                            },
                            min: 0,
                            max: 100,
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    }
                }
            });
            
            window.symbolChart = symbolChart;
        })
        .catch(error => console.error('Error loading symbol performance data:', error));
}

/**
 * Initialize price prediction chart with actual vs predicted prices
 */
function initPricePredictionChart() {
    const ctx = document.getElementById('price-prediction-chart').getContext('2d');
    
    // Get selected symbol from dropdown or use default
    const symbolSelector = document.getElementById('symbol-selector');
    const selectedSymbol = symbolSelector ? symbolSelector.value : 'AAPL';
    
    fetchPricePredictionData(selectedSymbol)
        .then(data => {
            const predictionChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.dates,
                    datasets: [
                        {
                            label: 'Actual Price',
                            data: data.actual_prices,
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                            fill: false,
                            tension: 0.1
                        },
                        {
                            label: 'Predicted Price',
                            data: data.predicted_prices,
                            borderColor: 'rgb(255, 159, 64)',
                            backgroundColor: 'rgba(255, 159, 64, 0.1)',
                            fill: false,
                            tension: 0.1,
                            borderDash: [5, 5]
                        },
                        {
                            label: 'Transformer Prediction',
                            data: data.transformer_predictions,
                            borderColor: 'rgb(153, 102, 255)',
                            backgroundColor: 'rgba(153, 102, 255, 0.1)',
                            fill: false,
                            tension: 0.1,
                            borderDash: [10, 5]
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: `Price Predictions for ${selectedSymbol}`
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Price ($)'
                            }
                        }
                    }
                }
            });
            
            window.predictionChart = predictionChart;
            
            // Add event listener for symbol selection
            if (symbolSelector) {
                symbolSelector.addEventListener('change', function() {
                    const newSymbol = this.value;
                    updatePricePredictionChart(newSymbol);
                });
            }
        })
        .catch(error => console.error('Error loading price prediction data:', error));
}

/**
 * Fetch price prediction data for a given symbol
 */
function fetchPricePredictionData(symbol) {
    return fetch(`/api/predictions/${symbol}`)
        .then(response => response.json());
}

/**
 * Update price prediction chart with new symbol data
 */
function updatePricePredictionChart(symbol) {
    fetchPricePredictionData(symbol)
        .then(data => {
            if (window.predictionChart) {
                window.predictionChart.data.labels = data.dates;
                window.predictionChart.data.datasets[0].data = data.actual_prices;
                window.predictionChart.data.datasets[1].data = data.predicted_prices;
                window.predictionChart.data.datasets[2].data = data.transformer_predictions;
                window.predictionChart.options.plugins.title.text = `Price Predictions for ${symbol}`;
                window.predictionChart.update();
            }
        })
        .catch(error => console.error(`Error updating price prediction chart for ${symbol}:`, error));
}

/**
 * Initialize risk heatmap visualization
 */
function initRiskHeatmap() {
    const ctx = document.getElementById('risk-heatmap').getContext('2d');
    
    fetch('/api/risk/heatmap')
        .then(response => response.json())
        .then(data => {
            // Create a heatmap using chart.js
            const chartData = {
                labels: data.symbols,
                datasets: [{
                    label: 'Risk Score',
                    data: data.risk_scores.map((score, index) => ({
                        x: index,
                        y: 0,
                        v: score
                    })),
                    backgroundColor: function(context) {
                        const value = context.dataset.data[context.dataIndex].v;
                        // Color scale from green (low risk) to red (high risk)
                        const red = Math.floor(255 * value);
                        const green = Math.floor(255 * (1 - value));
                        return `rgba(${red}, ${green}, 0, 0.8)`;
                    },
                    borderColor: 'rgba(0, 0, 0, 0.2)',
                    borderWidth: 1,
                    barPercentage: 0.95,
                    categoryPercentage: 0.95
                }]
            };
            
            const riskChart = new Chart(ctx, {
                type: 'bar',
                data: chartData,
                options: {
                    responsive: true,
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Symbol'
                            }
                        },
                        y: {
                            display: false
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: 'Risk Heatmap by Symbol'
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const value = context.dataset.data[context.dataIndex].v;
                                    return `Risk Score: ${(value * 100).toFixed(2)}%`;
                                }
                            }
                        }
                    }
                }
            });
            
            window.riskChart = riskChart;
        })
        .catch(error => console.error('Error loading risk heatmap data:', error));
}

/**
 * Initialize sentiment timeline chart
 */
function initSentimentTimeline() {
    const ctx = document.getElementById('sentiment-timeline').getContext('2d');
    
    fetch('/api/sentiment/timeline')
        .then(response => response.json())
        .then(data => {
            const sentimentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.dates,
                    datasets: data.symbols.map((symbol, index) => ({
                        label: symbol,
                        data: data.sentiment_scores[index],
                        borderColor: getColorByIndex(index),
                        backgroundColor: getColorByIndex(index, 0.1),
                        fill: false,
                        tension: 0.4
                    }))
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Market Sentiment Timeline'
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        },
                        y: {
                            display: true,
                            title: {
                                display: true,
                                text: 'Sentiment Score'
                            },
                            min: -1,
                            max: 1,
                            ticks: {
                                callback: function(value) {
                                    if (value === 1) return 'Very Bullish';
                                    if (value === 0.5) return 'Bullish';
                                    if (value === 0) return 'Neutral';
                                    if (value === -0.5) return 'Bearish';
                                    if (value === -1) return 'Very Bearish';
                                    return '';
                                }
                            }
                        }
                    }
                }
            });
            
            window.sentimentChart = sentimentChart;
        })
        .catch(error => console.error('Error loading sentiment timeline data:', error));
}

/**
 * Update all price charts with latest data
 */
function updatePriceCharts() {
    if (window.predictionChart) {
        const symbolSelector = document.getElementById('symbol-selector');
        const selectedSymbol = symbolSelector ? symbolSelector.value : 'AAPL';
        updatePricePredictionChart(selectedSymbol);
    }
}

/**
 * Update all performance metrics
 */
function updatePerformanceMetrics() {
    if (window.equityChart) {
        fetch('/api/performance/equity')
            .then(response => response.json())
            .then(data => {
                window.equityChart.data.labels = data.dates;
                window.equityChart.data.datasets[0].data = data.equity;
                window.equityChart.data.datasets[1].data = data.balance;
                window.equityChart.data.datasets[2].data = data.drawdown;
                window.equityChart.update();
            })
            .catch(error => console.error('Error updating equity chart:', error));
    }
    
    if (window.tradeDistChart) {
        fetch('/api/performance/trade_distribution')
            .then(response => response.json())
            .then(data => {
                window.tradeDistChart.data.datasets[0].data = [
                    data.winning_trades, 
                    data.losing_trades, 
                    data.breakeven_trades
                ];
                window.tradeDistChart.update();
            })
            .catch(error => console.error('Error updating trade distribution chart:', error));
    }
    
    if (window.symbolChart) {
        fetch('/api/performance/symbol_performance')
            .then(response => response.json())
            .then(data => {
                const sortedSymbols = Object.keys(data.symbols).sort((a, b) => 
                    data.symbols[b].total_pnl - data.symbols[a].total_pnl
                );
                
                window.symbolChart.data.labels = sortedSymbols;
                window.symbolChart.data.datasets[0].data = sortedSymbols.map(
                    symbol => data.symbols[symbol].total_pnl
                );
                window.symbolChart.data.datasets[1].data = sortedSymbols.map(
                    symbol => data.symbols[symbol].win_rate * 100
                );
                
                // Update colors based on PnL values
                window.symbolChart.data.datasets[0].backgroundColor = window.symbolChart.data.datasets[0].data.map(
                    pnl => pnl >= 0 ? 'rgba(75, 192, 192, 0.7)' : 'rgba(255, 99, 132, 0.7)'
                );
                window.symbolChart.data.datasets[0].borderColor = window.symbolChart.data.datasets[0].data.map(
                    pnl => pnl >= 0 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'
                );
                
                window.symbolChart.update();
            })
            .catch(error => console.error('Error updating symbol performance chart:', error));
    }
}

/**
 * Get a color from the chart color palette by index
 */
function getColorByIndex(index, alpha = 1) {
    const colorPalette = [
        `rgba(75, 192, 192, ${alpha})`,   // Teal
        `rgba(255, 99, 132, ${alpha})`,   // Red
        `rgba(54, 162, 235, ${alpha})`,   // Blue
        `rgba(255, 159, 64, ${alpha})`,   // Orange
        `rgba(153, 102, 255, ${alpha})`,  // Purple
        `rgba(255, 206, 86, ${alpha})`,   // Yellow
        `rgba(231, 233, 237, ${alpha})`,  // Grey
        `rgba(97, 205, 187, ${alpha})`,   // Turquoise
        `rgba(232, 126, 4, ${alpha})`,    // Dark Orange
        `rgba(56, 128, 255, ${alpha})`    // Royal Blue
    ];
    
    return colorPalette[index % colorPalette.length];
}

/**
 * Configuration optimisée pour Plotly
 * Utilise des templates minimalistes pour réduire le temps de génération
 */
const defaultLayout = {
    margin: { l: 40, r: 20, t: 40, b: 30 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: false,
    hovermode: 'closest',
    autosize: true,
    font: {
        family: 'Inter, system-ui, sans-serif',
        size: 12
    },
    modebar: {
        orientation: 'v',
        bgcolor: 'rgba(0,0,0,0)',
        color: '#64748b',
        activecolor: '#3b82f6'
    }
};

/**
 * Options optimisées pour Plotly avec désactivation de la validation et compression
 */
const optimizedConfig = {
    displayModeBar: true,
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
        'autoScale2d', 'resetScale2d', 'toggleSpikelines'
    ],
    toImageButtonOptions: {
        format: 'svg',
        filename: 'chart',
        width: 1200,
        height: 800,
        scale: 1
    }
};

/**
 * Création optimisée d'un graphique d'aperçu du marché
 * @param {string} containerId - L'ID du conteneur HTML
 * @param {Array} data - Les données pour le graphique
 * @param {Object} options - Options de configuration supplémentaires
 */
window.createMarketChart = function(containerId, data, options = {}) {
    if (!data || !Array.isArray(data) || data.length === 0) {
        console.warn('Données invalides pour le graphique de marché');
        return;
    }

    // Limiter les points de données pour améliorer les performances
    const maxPoints = options.maxPoints || 100;
    const step = data.length > maxPoints ? Math.floor(data.length / maxPoints) : 1;
    const limitedData = data.filter((_, i) => i % step === 0);

    // Extraire les dates et les valeurs
    const dates = limitedData.map(d => new Date(d.date));
    const values = limitedData.map(d => d.value);
    
    // Calculer les changements pour la coloration
    const changeColors = [];
    for (let i = 1; i < values.length; i++) {
        changeColors.push(values[i] >= values[i-1] ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)');
    }
    // Ajouter une couleur pour le premier point
    changeColors.unshift(values[1] >= values[0] ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)');

    // Créer un dégradé pour le remplissage
    const fillColors = values.map((v, i) => {
        const color = changeColors[i];
        return color.replace('0.7', '0.1');
    });

    // Créer le graphique avec des options optimisées
    const traces = [
        {
            type: 'scatter',
            mode: 'lines',
            x: dates,
            y: values,
            name: 'Valeur',
            line: {
                color: '#3b82f6',
                width: 2,
                shape: 'spline',
                smoothing: 0.3
            },
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.1)'
        }
    ];

    // Ajouter un tracé pour les points importants (maxima, minima)
    const importantPoints = findImportantPoints(values, dates);
    if (importantPoints.length > 0) {
        traces.push({
            type: 'scatter',
            mode: 'markers',
            x: importantPoints.map(p => dates[p.index]),
            y: importantPoints.map(p => values[p.index]),
            marker: {
                size: 8,
                color: importantPoints.map(p => p.type === 'max' ? 'rgba(16, 185, 129, 1)' : 'rgba(239, 68, 68, 1)'),
                line: {
                    width: 1,
                    color: 'white'
                }
            },
            showlegend: false,
            hoverinfo: 'x+y',
            hovertemplate: '%{y:.2f}<extra></extra>'
        });
    }

    // Configuration spécifique du layout
    const customLayout = {
        ...defaultLayout,
        title: options.title || 'Aperçu du Marché',
        xaxis: {
            type: 'date',
            showgrid: false,
            tickfont: {
                size: 10
            }
        },
        yaxis: {
            showgrid: true,
            gridcolor: 'rgba(203, 213, 225, 0.3)',
            tickformat: ',.2f'
        }
    };

    // Rendu du graphique optimisé
    Plotly.newPlot(
        containerId, 
        traces, 
        customLayout, 
        optimizedConfig
    );
};

/**
 * Trouver les points importants dans les données (maximum, minimum)
 * @param {Array} values - Tableau de valeurs
 * @param {Array} dates - Tableau de dates correspondantes
 * @returns {Array} Points importants avec leur index et type
 */
function findImportantPoints(values, dates) {
    if (values.length < 5) return [];
    
    const points = [];
    const windowSize = Math.min(5, Math.floor(values.length / 10) || 2);
    
    for (let i = windowSize; i < values.length - windowSize; i++) {
        const window = values.slice(i - windowSize, i + windowSize + 1);
        const currentValue = values[i];
        
        // Vérifier si c'est un maximum local
        if (currentValue === Math.max(...window) && !isNearExistingPoint(points, i, windowSize)) {
            points.push({ index: i, type: 'max', value: currentValue });
        }
        // Vérifier si c'est un minimum local
        else if (currentValue === Math.min(...window) && !isNearExistingPoint(points, i, windowSize)) {
            points.push({ index: i, type: 'min', value: currentValue });
        }
    }
    
    // Limiter le nombre de points importants (pour éviter d'encombrer le graphique)
    if (points.length > 5) {
        // Trier par importance (différence avec la moyenne)
        const avg = values.reduce((sum, val) => sum + val, 0) / values.length;
        points.sort((a, b) => Math.abs(b.value - avg) - Math.abs(a.value - avg));
        return points.slice(0, 5);
    }
    
    return points;
}

/**
 * Vérifier si un point est proche d'un point existant
 * @param {Array} points - Points existants
 * @param {number} index - Index à vérifier
 * @param {number} windowSize - Taille de la fenêtre
 * @returns {boolean} True si un point est proche
 */
function isNearExistingPoint(points, index, windowSize) {
    return points.some(p => Math.abs(p.index - index) < windowSize);
}

/**
 * Création optimisée d'un graphique d'analyse de sentiment
 * @param {string} containerId - L'ID du conteneur HTML
 * @param {Array} data - Les données pour le graphique
 * @param {Object} options - Options de configuration supplémentaires
 */
window.createSentimentChart = function(containerId, data, options = {}) {
    if (!data || !Array.isArray(data) || data.length === 0) {
        console.warn('Données invalides pour le graphique de sentiment');
        return;
    }

    // Limiter les points de données pour améliorer les performances
    const maxPoints = options.maxPoints || 60;
    const step = data.length > maxPoints ? Math.floor(data.length / maxPoints) : 1;
    const limitedData = data.filter((_, i) => i % step === 0);

    // Extraire les dates et les scores de sentiment
    const dates = limitedData.map(d => new Date(d.date));
    const scores = limitedData.map(d => d.sentiment_score);
    const volumes = limitedData.map(d => d.volume || 1);
    
    // Normaliser les volumes pour le dimensionnement des points
    const minVolume = Math.min(...volumes);
    const maxVolume = Math.max(...volumes);
    const normalizedVolumes = volumes.map(v => 
        minVolume === maxVolume 
            ? 8 
            : 5 + (v - minVolume) / (maxVolume - minVolume) * 15
    );

    // Calculer les couleurs basées sur le score
    const colors = scores.map(score => {
        if (score > 0.2) return 'rgba(16, 185, 129, 0.7)'; // Positif
        if (score < -0.2) return 'rgba(239, 68, 68, 0.7)'; // Négatif
        return 'rgba(249, 115, 22, 0.7)'; // Neutre
    });

    // Créer le graphique avec des options optimisées
    const traces = [
        {
            type: 'scatter',
            mode: 'lines+markers',
            x: dates,
            y: scores,
            marker: {
                size: normalizedVolumes,
                color: colors,
                line: {
                    width: 1,
                    color: 'white'
                }
            },
            line: {
                color: 'rgba(59, 130, 246, 0.5)',
                width: 1,
                shape: 'spline',
                smoothing: 0.3
            },
            hoverinfo: 'x+y+text',
            hovertemplate: 'Sentiment: %{y:.2f}<br>Volume: %{text}<extra></extra>',
            text: volumes.map(v => v.toFixed(0))
        }
    ];

    // Ajouter des lignes de niveau pour référence
    traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [dates[0], dates[dates.length - 1]],
        y: [0.2, 0.2], // Seuil positif
        line: {
            color: 'rgba(16, 185, 129, 0.3)',
            width: 1,
            dash: 'dash'
        },
        showlegend: false,
        hoverinfo: 'none'
    });

    traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [dates[0], dates[dates.length - 1]],
        y: [-0.2, -0.2], // Seuil négatif
        line: {
            color: 'rgba(239, 68, 68, 0.3)',
            width: 1,
            dash: 'dash'
        },
        showlegend: false,
        hoverinfo: 'none'
    });

    // Ajouter une ligne de neutralité
    traces.push({
        type: 'scatter',
        mode: 'lines',
        x: [dates[0], dates[dates.length - 1]],
        y: [0, 0],
        line: {
            color: 'rgba(100, 116, 139, 0.2)',
            width: 1
        },
        showlegend: false,
        hoverinfo: 'none'
    });

    // Configuration spécifique du layout
    const customLayout = {
        ...defaultLayout,
        title: options.title || 'Analyse de Sentiment',
        xaxis: {
            type: 'date',
            showgrid: false,
            tickfont: {
                size: 10
            }
        },
        yaxis: {
            showgrid: true,
            gridcolor: 'rgba(203, 213, 225, 0.3)',
            zeroline: false,
            range: [-1, 1],
            tickformat: '.1f'
        },
        annotations: [
            {
                x: dates[0],
                y: 0.2,
                xref: 'x',
                yref: 'y',
                text: 'Positif',
                showarrow: false,
                font: {
                    size: 10,
                    color: 'rgba(16, 185, 129, 0.7)'
                },
                xanchor: 'left',
                yanchor: 'bottom'
            },
            {
                x: dates[0],
                y: -0.2,
                xref: 'x',
                yref: 'y',
                text: 'Négatif',
                showarrow: false,
                font: {
                    size: 10,
                    color: 'rgba(239, 68, 68, 0.7)'
                },
                xanchor: 'left',
                yanchor: 'top'
            }
        ]
    };

    // Rendu du graphique optimisé
    Plotly.newPlot(
        containerId,
        traces,
        customLayout,
        optimizedConfig
    );
};

/**
 * Crée un graphique de prédiction de prix optimisé
 * @param {string} containerId - L'ID du conteneur HTML
 * @param {Array} historicalData - Données historiques de prix
 * @param {Array} predictions - Prédictions futures
 * @param {Object} options - Options de configuration
 */
window.createPricePredictionChart = function(containerId, historicalData, predictions, options = {}) {
    if (!historicalData || !predictions) {
        console.warn('Données manquantes pour le graphique de prédiction');
        return;
    }

    // Configurer les données historiques
    const historicalDates = historicalData.map(d => new Date(d.date));
    const historicalPrices = historicalData.map(d => d.price);
    
    // Configurer les données de prédiction
    const predictionDates = predictions.map(d => new Date(d.date));
    const predictionPrices = predictions.map(d => d.predicted_price);
    const upperBounds = predictions.map(d => d.upper_bound || (d.predicted_price * 1.05));
    const lowerBounds = predictions.map(d => d.lower_bound || (d.predicted_price * 0.95));
    
    // Créer les traces pour le graphique
    const traces = [
        // Données historiques
        {
            type: 'scatter',
            mode: 'lines',
            name: 'Prix Historique',
            x: historicalDates,
            y: historicalPrices,
            line: {
                color: 'rgba(59, 130, 246, 0.8)',
                width: 2
            }
        },
        // Prédictions
        {
            type: 'scatter',
            mode: 'lines',
            name: 'Prédiction',
            x: predictionDates,
            y: predictionPrices,
            line: {
                color: 'rgba(14, 165, 233, 0.8)',
                width: 2,
                dash: 'dash'
            }
        },
        // Zone d'incertitude
        {
            type: 'scatter',
            name: 'Borne Supérieure',
            x: predictionDates,
            y: upperBounds,
            mode: 'lines',
            line: {
                width: 0,
                color: 'rgba(14, 165, 233, 0)'
            },
            showlegend: false
        },
        {
            type: 'scatter',
            name: 'Borne Inférieure',
            x: predictionDates,
            y: lowerBounds,
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(14, 165, 233, 0.2)',
            line: {
                width: 0,
                color: 'rgba(14, 165, 233, 0)'
            },
            showlegend: false
        }
    ];
    
    // Configuration du layout
    const customLayout = {
        ...defaultLayout,
        showlegend: true,
        legend: {
            x: 0,
            y: 1,
            orientation: 'h',
            yanchor: 'bottom',
            font: {
                size: 10
            }
        },
        title: options.title || 'Prédiction de Prix',
        xaxis: {
            type: 'date',
            showgrid: false
        },
        yaxis: {
            showgrid: true,
            gridcolor: 'rgba(203, 213, 225, 0.3)',
            tickformat: ',.2f',
            title: options.yaxisTitle || 'Prix'
        }
    };
    
    // Rendu du graphique optimisé
    Plotly.newPlot(
        containerId,
        traces,
        customLayout,
        optimizedConfig
    );
};

/**
 * Observer l'état du mode sombre et mettre à jour les graphiques
 */
function setupDarkModeObserver() {
    // Détecter les changements de mode sombre
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === 'class') {
                const darkModeEnabled = document.body.classList.contains('dark-mode');
                updateChartsTheme(darkModeEnabled);
            }
        });
    });
    
    // Observer les changements de classe sur le body
    observer.observe(document.body, { attributes: true });
    
    // Vérifier l'état initial
    const darkModeEnabled = document.body.classList.contains('dark-mode');
    updateChartsTheme(darkModeEnabled);
}

/**
 * Mettre à jour le thème des graphiques selon le mode sombre
 * @param {boolean} darkMode - Si le mode sombre est activé
 */
function updateChartsTheme(darkMode) {
    const charts = document.querySelectorAll('.chart-container');
    
    const newColors = {
        paperBgColor: darkMode ? 'rgba(0,0,0,0)' : 'rgba(0,0,0,0)',
        plotBgColor: darkMode ? 'rgba(0,0,0,0)' : 'rgba(0,0,0,0)',
        gridColor: darkMode ? 'rgba(71, 85, 105, 0.3)' : 'rgba(203, 213, 225, 0.3)',
        fontColor: darkMode ? '#e2e8f0' : '#334155',
        lineColor: darkMode ? '#60a5fa' : '#3b82f6'
    };
    
    charts.forEach(chart => {
        const plotlyChart = chart._fullLayout;
        if (!plotlyChart) return;
        
        Plotly.relayout(chart, {
            'paper_bgcolor': newColors.paperBgColor,
            'plot_bgcolor': newColors.plotBgColor,
            'font.color': newColors.fontColor,
            'xaxis.gridcolor': newColors.gridColor,
            'yaxis.gridcolor': newColors.gridColor
        });
    });
} 