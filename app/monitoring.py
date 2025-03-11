import time
import logging
import threading
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server

# Logger configuration
logger = logging.getLogger(__name__)

# Définition des métriques
TRADING_SIGNALS = Counter('trading_signals_total', 'Total number of trading signals generated', ['symbol', 'direction'])
EXECUTED_TRADES = Counter('executed_trades_total', 'Total number of executed trades', ['symbol', 'direction'])
POSITION_COUNT = Gauge('open_positions', 'Number of currently open positions', ['symbol'])
PORTFOLIO_VALUE = Gauge('portfolio_value', 'Current portfolio value in USD')
BALANCE = Gauge('balance', 'Current balance in USD')
API_REQUESTS = Counter('api_requests_total', 'Total number of API requests', ['endpoint', 'method', 'status'])
DATA_UPDATE_DURATION = Histogram('data_update_duration_seconds', 'Time spent updating market data', ['symbol'])
PREDICTION_DURATION = Summary('prediction_duration_seconds', 'Time spent generating predictions', ['model'])

# Metrics for financial performance
WIN_RATE = Gauge('win_rate', 'Percentage of winning trades')
PROFIT_FACTOR = Gauge('profit_factor', 'Ratio of gross profit to gross loss')
AVERAGE_WIN = Gauge('average_win', 'Average profit of winning trades in USD')
AVERAGE_LOSS = Gauge('average_loss', 'Average loss of losing trades in USD')
DRAWDOWN = Gauge('max_drawdown_percent', 'Maximum drawdown as percentage')

# Performance metrics
API_LATENCY = Histogram('api_request_latency_seconds', 'API request latency', ['endpoint'])
MODEL_PREDICTION_ERRORS = Counter('model_prediction_errors_total', 'Total number of model prediction errors', ['model'])
DB_QUERY_DURATION = Histogram('db_query_duration_seconds', 'Database query duration', ['query_type'])

class MonitoringService:
    """Service to monitor the trading system with Prometheus metrics."""
    
    def __init__(self, port=9090):
        self.port = port
        self.server_thread = None
        self.running = False
        
    def start(self):
        """Start the Prometheus metrics server."""
        if self.running:
            logger.warning("Prometheus server is already running")
            return
            
        def run_server():
            logger.info(f"Starting Prometheus metrics server on port {self.port}")
            start_http_server(self.port)
            self.running = True
            
            while self.running:
                time.sleep(1)
                
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        logger.info("Prometheus metrics server started")
        
    def stop(self):
        """Stop the Prometheus metrics server."""
        if not self.running:
            logger.warning("Prometheus server is not running")
            return
            
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
            logger.info("Prometheus metrics server stopped")
            
    def update_portfolio_metrics(self, portfolio_stats):
        """Update portfolio-related metrics."""
        if not portfolio_stats:
            return
            
        PORTFOLIO_VALUE.set(portfolio_stats.get('total_equity', 0))
        BALANCE.set(portfolio_stats.get('current_balance', 0))
        
        # Update performance metrics if available
        if 'win_rate' in portfolio_stats:
            WIN_RATE.set(portfolio_stats['win_rate'] * 100)  # Convert to percentage
        if 'profit_factor' in portfolio_stats:
            PROFIT_FACTOR.set(portfolio_stats['profit_factor'])
        if 'average_win' in portfolio_stats:
            AVERAGE_WIN.set(portfolio_stats['average_win'])
        if 'average_loss' in portfolio_stats:
            AVERAGE_LOSS.set(portfolio_stats['average_loss'])
        if 'max_drawdown' in portfolio_stats:
            DRAWDOWN.set(portfolio_stats['max_drawdown'] * 100)  # Convert to percentage
            
    def update_position_metrics(self, symbol, count):
        """Update position count metrics."""
        POSITION_COUNT.labels(symbol=symbol).set(count)
        
    def record_trading_signal(self, symbol, direction):
        """Record a generated trading signal."""
        TRADING_SIGNALS.labels(symbol=symbol, direction=direction).inc()
        
    def record_executed_trade(self, symbol, direction):
        """Record an executed trade."""
        EXECUTED_TRADES.labels(symbol=symbol, direction=direction).inc()
        
    def time_data_update(self, symbol):
        """Create a timer context for data updates."""
        return DATA_UPDATE_DURATION.labels(symbol=symbol).time()
        
    def time_prediction(self, model_name):
        """Create a timer context for predictions."""
        return PREDICTION_DURATION.labels(model=model_name).time()
        
    def record_api_request(self, endpoint, method, status):
        """Record an API request."""
        API_REQUESTS.labels(endpoint=endpoint, method=method, status=status).inc()
        
    def time_api_request(self, endpoint):
        """Create a timer context for API requests."""
        return API_LATENCY.labels(endpoint=endpoint).time()
        
    def record_model_error(self, model_name):
        """Record a model prediction error."""
        MODEL_PREDICTION_ERRORS.labels(model=model_name).inc()
        
    def time_db_query(self, query_type):
        """Create a timer context for database queries."""
        return DB_QUERY_DURATION.labels(query_type=query_type).time()

# Singleton instance
monitoring_service = MonitoringService()

def init_monitoring(app=None, port=9090):
    """Initialize the monitoring service."""
    global monitoring_service
    
    # Start the Prometheus metrics server
    monitoring_service = MonitoringService(port=port)
    monitoring_service.start()
    
    if app:
        # Add Flask middleware for tracking requests if app is provided
        @app.before_request
        def before_request():
            request = app.request_class.from_values()
            request.start_time = time.time()
            
        @app.after_request
        def after_request(response):
            request = app.request_class.from_values()
            if hasattr(request, 'start_time'):
                duration = time.time() - request.start_time
                endpoint = request.endpoint or 'unknown'
                API_LATENCY.labels(endpoint=endpoint).observe(duration)
                
            monitoring_service.record_api_request(
                endpoint=request.endpoint or 'unknown',
                method=request.method,
                status=response.status_code
            )
            return response
            
    return monitoring_service 