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