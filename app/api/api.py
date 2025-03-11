from flask import Blueprint, jsonify, request, current_app
from flask_cors import CORS
import logging
import threading
import json
from datetime import datetime, timedelta

# Import our trading components
from app.trading import TradingBot, DataManager
from app.models.position_manager import PositionManager
from app.monitoring import monitoring_service

# Configure logger
logger = logging.getLogger(__name__)

# Create Blueprint
api_blueprint = Blueprint('api', __name__)

# Enable CORS for all routes in this blueprint
CORS(api_blueprint)

@api_blueprint.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    with monitoring_service.time_api_request('/api/health'):
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })

@api_blueprint.route('/portfolio', methods=['GET'])
def get_portfolio():
    """Get current portfolio information."""
    with monitoring_service.time_api_request('/api/portfolio'):
        try:
            trading_bot = current_app.config.get('trading_bot')
            if not trading_bot:
                return jsonify({'error': 'Trading bot not initialized'}), 500
            
            # Get market prices for current positions
            data_manager = current_app.config.get('data_manager')
            market_prices = {}
            
            for position in trading_bot.position_manager.positions.values():
                if position.symbol not in market_prices:
                    try:
                        market_prices[position.symbol] = data_manager.get_latest_price(position.symbol)
                    except Exception as e:
                        logger.error(f"Error fetching price for {position.symbol}: {e}")
                        market_prices[position.symbol] = position.entry_price
            
            # Get portfolio stats
            portfolio_stats = trading_bot.position_manager.get_portfolio_stats(market_prices)
            
            # Update monitoring metrics
            monitoring_service.update_portfolio_metrics(portfolio_stats)
            
            # Return detailed portfolio information
            return jsonify({
                'portfolio_value': portfolio_stats.get('total_equity', 0),
                'cash_balance': portfolio_stats.get('current_balance', 0),
                'unrealized_pnl': portfolio_stats.get('unrealized_pnl', 0),
                'open_positions': [
                    {
                        'id': pos_id,
                        'symbol': pos.symbol,
                        'direction': pos.direction,
                        'entry_price': pos.entry_price,
                        'current_price': market_prices.get(pos.symbol, pos.entry_price),
                        'size': pos.size,
                        'unrealized_pnl': pos.calculate_unrealized_pnl(market_prices.get(pos.symbol, pos.entry_price)),
                        'unrealized_pnl_percent': pos.calculate_unrealized_pnl_percentage(market_prices.get(pos.symbol, pos.entry_price)),
                        'stop_loss': pos.stop_loss,
                        'take_profit': pos.take_profit,
                        'entry_time': pos.entry_time.isoformat() if pos.entry_time else None
                    }
                    for pos_id, pos in trading_bot.position_manager.positions.items()
                ],
                'closed_positions': [
                    {
                        'id': pos.id,
                        'symbol': pos.symbol,
                        'direction': pos.direction,
                        'entry_price': pos.entry_price,
                        'exit_price': pos.exit_price,
                        'size': pos.size,
                        'realized_pnl': pos.realized_pnl,
                        'realized_pnl_percent': pos.realized_pnl_percentage,
                        'entry_time': pos.entry_time.isoformat() if pos.entry_time else None,
                        'exit_time': pos.exit_time.isoformat() if pos.exit_time else None
                    }
                    for pos in trading_bot.position_manager.closed_positions[-10:]  # Last 10 closed positions
                ],
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error in portfolio API: {e}")
            return jsonify({'error': str(e)}), 500

@api_blueprint.route('/positions', methods=['GET'])
def get_positions():
    """Get all current positions."""
    with monitoring_service.time_api_request('/api/positions'):
        try:
            trading_bot = current_app.config.get('trading_bot')
            if not trading_bot:
                return jsonify({'error': 'Trading bot not initialized'}), 500
                
            # Get market prices for current positions
            data_manager = current_app.config.get('data_manager')
            market_prices = {}
            
            for position in trading_bot.position_manager.positions.values():
                if position.symbol not in market_prices:
                    try:
                        market_prices[position.symbol] = data_manager.get_latest_price(position.symbol)
                    except Exception as e:
                        logger.error(f"Error fetching price for {position.symbol}: {e}")
                        market_prices[position.symbol] = position.entry_price
            
            positions = [
                {
                    'id': pos_id,
                    'symbol': pos.symbol,
                    'direction': pos.direction,
                    'entry_price': pos.entry_price,
                    'current_price': market_prices.get(pos.symbol, pos.entry_price),
                    'size': pos.size,
                    'unrealized_pnl': pos.calculate_unrealized_pnl(market_prices.get(pos.symbol, pos.entry_price)),
                    'unrealized_pnl_percent': pos.calculate_unrealized_pnl_percentage(market_prices.get(pos.symbol, pos.entry_price)),
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'entry_time': pos.entry_time.isoformat() if pos.entry_time else None
                }
                for pos_id, pos in trading_bot.position_manager.positions.items()
            ]
            
            return jsonify({
                'positions': positions,
                'count': len(positions),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error in positions API: {e}")
            return jsonify({'error': str(e)}), 500

@api_blueprint.route('/positions/<position_id>', methods=['GET'])
def get_position(position_id):
    """Get details for a specific position."""
    with monitoring_service.time_api_request(f'/api/positions/{position_id}'):
        try:
            trading_bot = current_app.config.get('trading_bot')
            if not trading_bot:
                return jsonify({'error': 'Trading bot not initialized'}), 500
                
            # Check if position exists
            if position_id not in trading_bot.position_manager.positions:
                return jsonify({'error': 'Position not found'}), 404
                
            position = trading_bot.position_manager.positions[position_id]
            
            # Get current market price
            data_manager = current_app.config.get('data_manager')
            try:
                current_price = data_manager.get_latest_price(position.symbol)
            except Exception as e:
                logger.error(f"Error fetching price for {position.symbol}: {e}")
                current_price = position.entry_price
                
            return jsonify({
                'id': position.id,
                'symbol': position.symbol,
                'direction': position.direction,
                'entry_price': position.entry_price,
                'current_price': current_price,
                'size': position.size,
                'unrealized_pnl': position.calculate_unrealized_pnl(current_price),
                'unrealized_pnl_percent': position.calculate_unrealized_pnl_percentage(current_price),
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'entry_time': position.entry_time.isoformat() if position.entry_time else None,
                'metadata': position.metadata
            })
        except Exception as e:
            logger.error(f"Error in position detail API: {e}")
            return jsonify({'error': str(e)}), 500

@api_blueprint.route('/positions/<position_id>', methods=['DELETE'])
def close_position(position_id):
    """Close a specific position."""
    with monitoring_service.time_api_request(f'/api/positions/{position_id}/close'):
        try:
            trading_bot = current_app.config.get('trading_bot')
            if not trading_bot:
                return jsonify({'error': 'Trading bot not initialized'}), 500
                
            # Check if position exists
            if position_id not in trading_bot.position_manager.positions:
                return jsonify({'error': 'Position not found'}), 404
                
            position = trading_bot.position_manager.positions[position_id]
            
            # Get current market price
            data_manager = current_app.config.get('data_manager')
            try:
                current_price = data_manager.get_latest_price(position.symbol)
            except Exception as e:
                logger.error(f"Error fetching price for {position.symbol}: {e}")
                current_price = position.entry_price
                
            # Close the position
            realized_pnl = trading_bot.position_manager.close_position(position_id, current_price)
            
            # Record the trade exit in DB
            if hasattr(trading_bot, '_store_trade_exit'):
                trading_bot._store_trade_exit(
                    symbol=position.symbol,
                    entry_time=position.entry_time,
                    exit_time=datetime.now(),
                    exit_price=current_price,
                    pnl=realized_pnl,
                    reason='Manual Close'
                )
                
            # Record metrics
            monitoring_service.record_executed_trade(position.symbol, 'close')
            
            return jsonify({
                'success': True,
                'position_id': position_id,
                'symbol': position.symbol,
                'realized_pnl': realized_pnl,
                'exit_price': current_price,
                'close_time': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return jsonify({'error': str(e)}), 500

@api_blueprint.route('/positions/<position_id>/update', methods=['PATCH'])
def update_position(position_id):
    """Update stop-loss or take-profit levels for a position."""
    with monitoring_service.time_api_request(f'/api/positions/{position_id}/update'):
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            trading_bot = current_app.config.get('trading_bot')
            if not trading_bot:
                return jsonify({'error': 'Trading bot not initialized'}), 500
                
            # Check if position exists
            if position_id not in trading_bot.position_manager.positions:
                return jsonify({'error': 'Position not found'}), 404
                
            position = trading_bot.position_manager.positions[position_id]
            
            # Get stop loss and take profit values from request
            new_stop_loss = data.get('stop_loss')
            new_take_profit = data.get('take_profit')
            
            if new_stop_loss is None and new_take_profit is None:
                return jsonify({'error': 'No stop-loss or take-profit provided for update'}), 400
                
            # Update the position
            trading_bot.position_manager.update_position_stops(
                position_id=position_id,
                new_stop_loss=float(new_stop_loss) if new_stop_loss is not None else None,
                new_take_profit=float(new_take_profit) if new_take_profit is not None else None
            )
            
            return jsonify({
                'success': True,
                'position_id': position_id,
                'symbol': position.symbol,
                'new_stop_loss': new_stop_loss,
                'new_take_profit': new_take_profit,
                'update_time': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            return jsonify({'error': str(e)}), 500

@api_blueprint.route('/trade', methods=['POST'])
def execute_trade():
    """Execute a new trade."""
    with monitoring_service.time_api_request('/api/trade'):
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            # Validate required fields
            required_fields = ['symbol', 'direction', 'size']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
                
            symbol = data['symbol']
            direction = data['direction']
            size = float(data['size'])
            
            # Get additional parameters
            stop_loss = float(data['stop_loss']) if 'stop_loss' in data else None
            take_profit = float(data['take_profit']) if 'take_profit' in data else None
            
            trading_bot = current_app.config.get('trading_bot')
            if not trading_bot:
                return jsonify({'error': 'Trading bot not initialized'}), 500
                
            # Get current market price
            data_manager = current_app.config.get('data_manager')
            try:
                current_price = data_manager.get_latest_price(symbol)
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                return jsonify({'error': f'Could not get current price for {symbol}'}), 500
                
            # Auto-calculate stop loss and take profit if not provided
            if stop_loss is None or take_profit is None:
                risk_reward_ratio = 2.0  # Risk:Reward = 1:2
                price_range = current_price * 0.02  # 2% of current price
                
                if direction == 'buy' or direction == 'long':
                    stop_loss = stop_loss or current_price - price_range
                    take_profit = take_profit or current_price + (price_range * risk_reward_ratio)
                else:  # sell or short
                    stop_loss = stop_loss or current_price + price_range
                    take_profit = take_profit or current_price - (price_range * risk_reward_ratio)
            
            # Execute the trade
            if direction in ['buy', 'long']:
                trading_bot.execute_trade('buy', symbol, current_price, take_profit, stop_loss)
                monitoring_service.record_executed_trade(symbol, 'buy')
            elif direction in ['sell', 'short']:
                trading_bot.execute_trade('sell', symbol, current_price, take_profit, stop_loss)
                monitoring_service.record_executed_trade(symbol, 'sell')
            else:
                return jsonify({'error': f'Invalid direction: {direction}'}), 400
                
            # Get the newly created position (assumed to be the most recent)
            # Note: This is a simplification; in production, you'd want a more robust way to identify the new position
            new_position_id = None
            for pos_id, pos in trading_bot.position_manager.positions.items():
                if pos.symbol == symbol and abs(datetime.now().timestamp() - pos.entry_time.timestamp()) < 10:
                    new_position_id = pos_id
                    break
                    
            return jsonify({
                'success': True,
                'position_id': new_position_id,
                'symbol': symbol,
                'direction': direction,
                'entry_price': current_price,
                'size': size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return jsonify({'error': str(e)}), 500

@api_blueprint.route('/signals', methods=['GET'])
def get_signals():
    """Get the latest trading signals."""
    with monitoring_service.time_api_request('/api/signals'):
        try:
            trading_bot = current_app.config.get('trading_bot')
            if not trading_bot:
                return jsonify({'error': 'Trading bot not initialized'}), 500
                
            # Get latest signals from trading bot
            latest_signals = trading_bot.get_latest_signals() if hasattr(trading_bot, 'get_latest_signals') else []
            
            return jsonify({
                'signals': latest_signals,
                'count': len(latest_signals),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            return jsonify({'error': str(e)}), 500

@api_blueprint.route('/performance', methods=['GET'])
def get_performance():
    """Get performance metrics."""
    with monitoring_service.time_api_request('/api/performance'):
        try:
            trading_bot = current_app.config.get('trading_bot')
            if not trading_bot:
                return jsonify({'error': 'Trading bot not initialized'}), 500
                
            # Get time range from query parameters
            days = request.args.get('days', 30, type=int)
            
            # Fetch performance metrics from the database
            # This is a placeholder; implement the actual database query based on your schema
            conn = None
            try:
                from app.trading import get_db_connection
                conn = get_db_connection()
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT date, balance, equity, daily_pnl, total_trades, 
                               winning_trades, losing_trades, win_rate, 
                               average_win, average_loss, profit_factor
                        FROM performance_metrics
                        WHERE date >= %s
                        ORDER BY date ASC
                        """,
                        (datetime.now() - timedelta(days=days),)
                    )
                    metrics = cursor.fetchall()
            finally:
                if conn:
                    conn.close()
                    
            # Process metrics
            dates = []
            balance = []
            equity = []
            daily_pnl = []
            win_rate = []
            
            for metric in metrics:
                dates.append(metric[0].isoformat())
                balance.append(float(metric[1]))
                equity.append(float(metric[2]))
                daily_pnl.append(float(metric[3]))
                win_rate.append(float(metric[7]) * 100)  # Convert to percentage
                
            # Calculate cumulative metrics
            if metrics:
                last_metric = metrics[-1]
                cumulative = {
                    'current_balance': float(last_metric[1]),
                    'current_equity': float(last_metric[2]),
                    'total_trades': int(last_metric[4]),
                    'winning_trades': int(last_metric[5]),
                    'losing_trades': int(last_metric[6]),
                    'win_rate': float(last_metric[7]) * 100,  # Convert to percentage
                    'average_win': float(last_metric[8]),
                    'average_loss': float(last_metric[9]),
                    'profit_factor': float(last_metric[10])
                }
            else:
                cumulative = {
                    'current_balance': trading_bot.position_manager.current_balance,
                    'current_equity': trading_bot.position_manager.current_balance,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'average_win': 0,
                    'average_loss': 0,
                    'profit_factor': 0
                }
                
            return jsonify({
                'time_series': {
                    'dates': dates,
                    'balance': balance,
                    'equity': equity,
                    'daily_pnl': daily_pnl,
                    'win_rate': win_rate
                },
                'cumulative': cumulative,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return jsonify({'error': str(e)}), 500

@api_blueprint.route('/bot/control', methods=['POST'])
def control_bot():
    """Control the trading bot (start/stop)."""
    with monitoring_service.time_api_request('/api/bot/control'):
        try:
            data = request.get_json()
            if not data or 'action' not in data:
                return jsonify({'error': 'No action specified'}), 400
                
            action = data['action']
            
            trading_bot = current_app.config.get('trading_bot')
            if not trading_bot:
                return jsonify({'error': 'Trading bot not initialized'}), 500
                
            # Perform the requested action
            if action == 'start':
                if hasattr(trading_bot, 'set_status'):
                    trading_bot.set_status('running')
                
                # Start data updates and trading if not already running
                data_manager = current_app.config.get('data_manager')
                if data_manager and hasattr(data_manager, 'start_data_update'):
                    threading.Thread(target=data_manager.start_data_update, daemon=True).start()
                
                if hasattr(trading_bot, 'start_real_time_scanning'):
                    threading.Thread(target=lambda: trading_bot.start_real_time_scanning(data_manager), daemon=True).start()
                
                return jsonify({
                    'success': True,
                    'action': 'start',
                    'status': trading_bot.get_status() if hasattr(trading_bot, 'get_status') else 'running',
                    'timestamp': datetime.now().isoformat()
                })
                
            elif action == 'stop':
                if hasattr(trading_bot, 'set_status'):
                    trading_bot.set_status('stopped')
                
                return jsonify({
                    'success': True,
                    'action': 'stop',
                    'status': trading_bot.get_status() if hasattr(trading_bot, 'get_status') else 'stopped',
                    'timestamp': datetime.now().isoformat()
                })
                
            else:
                return jsonify({'error': f'Invalid action: {action}'}), 400
                
        except Exception as e:
            logger.error(f"Error controlling bot: {e}")
            return jsonify({'error': str(e)}), 500

@api_blueprint.route('/symbols', methods=['GET'])
def get_symbols():
    """Get list of available trading symbols."""
    with monitoring_service.time_api_request('/api/symbols'):
        try:
            data_manager = current_app.config.get('data_manager')
            if not data_manager:
                return jsonify({'error': 'Data manager not initialized'}), 500
                
            symbols = data_manager.symbols
            
            # Get prices for each symbol
            prices = {}
            for symbol in symbols:
                try:
                    prices[symbol] = data_manager.get_latest_price(symbol)
                except Exception as e:
                    logger.error(f"Error fetching price for {symbol}: {e}")
                    prices[symbol] = None
                    
            return jsonify({
                'symbols': [
                    {
                        'symbol': symbol,
                        'price': prices[symbol]
                    }
                    for symbol in symbols
                ],
                'count': len(symbols),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return jsonify({'error': str(e)}), 500

def register_api_routes(app):
    """Register API routes with the Flask app."""
    app.register_blueprint(api_blueprint, url_prefix='/api') 