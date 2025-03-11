from flask import Blueprint, render_template, jsonify, request, redirect, url_for, flash
import psutil
import plotly.graph_objs as go
import plotly.io as pio
from app import trading_bot, data_manager
from app.model_trainer import ModelTrainer
import json
from datetime import datetime, timedelta
import random  # Pour les données de démonstration
import os
from dotenv import load_dotenv
import psycopg2
import logging

main_blueprint = Blueprint('main', __name__)

# Initialize model trainer with global trading_bot
model_trainer = ModelTrainer(trading_bot)

@main_blueprint.route('/')
def dashboard():
    # CPU/GPU Performance Data
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    # Prepare CPU usage chart
    cpu_chart = go.Figure(data=[go.Bar(x=['CPU Usage'], y=[cpu_percent], marker=dict(color='rgb(55, 83, 109)'))])
    cpu_chart.update_layout(title='CPU Usage (%)')

    # Convert the chart to JSON
    cpu_chart_json = pio.to_json(cpu_chart)

    # Get trading statistics
    signals = trading_bot.get_latest_signals() if trading_bot else []

    return render_template('dashboard.html', 
                           cpu_percent=cpu_percent,
                           memory_info=memory_info,
                           cpu_chart=cpu_chart_json,
                           signals=signals)

@main_blueprint.route('/advanced')
def advanced_dashboard():
    """Advanced dashboard with interactive charts."""
    # Get bot status
    bot_status = trading_bot.get_status() if trading_bot else 'stopped'
    
    # Get trading signals
    signals = []
    if trading_bot:
        for signal_group in trading_bot.latest_signals:
            for signal in signal_group.get('signals', []):
                signals.append(signal)
    
    # Get portfolio statistics
    portfolio_stats = {}
    if trading_bot and hasattr(trading_bot, 'position_manager'):
        # Get market prices for all symbols
        market_prices = {}
        for symbol in trading_bot.market_data:
            if symbol in trading_bot.market_data and not trading_bot.market_data[symbol].empty:
                market_prices[symbol] = trading_bot.market_data[symbol]['Close'].iloc[-1]
        
        # Get portfolio statistics with current market prices
        portfolio_stats = trading_bot.position_manager.get_portfolio_stats(market_prices)
    
    # Get all symbols for the symbol selector
    symbols = []
    if data_manager:
        symbols = data_manager.symbols
    
    return render_template('advanced_dashboard.html',
                           bot_status=bot_status,
                           signals=signals,
                           portfolio_stats=portfolio_stats,
                           symbols=symbols)

@main_blueprint.route('/bot_status')
def bot_status():
    status = {
        'state': trading_bot.get_status() if trading_bot else 'stopped',
        'signals': trading_bot.get_latest_signals() if trading_bot else []
    }
    return jsonify(status)

@main_blueprint.route('/performance')
def performance():
    # Récupérer les données de performance depuis la base de données
    try:
        # Connexion à la base de données
        conn = psycopg2.connect(
            dbname=os.environ.get('DB_NAME', 'trading_db'),
            user=os.environ.get('DB_USER', 'trader'),
            password=os.environ.get('DB_PASSWORD', 'secure_password'),
            host=os.environ.get('DB_HOST', 'database'),
            port=os.environ.get('DB_PORT', '5432')
        )
        
        # Récupération des données de performance sur 30 jours
        with conn.cursor() as cur:
            # Récupérer l'historique du portefeuille
            cur.execute("""
                SELECT date, equity
                FROM performance_metrics
                WHERE date >= NOW() - INTERVAL '30 days'
                ORDER BY date ASC
            """)
            
            portfolio_history = cur.fetchall()
            
            if not portfolio_history:
                # En cas d'absence de données, rediriger vers le tableau de bord
                return redirect(url_for('main.dashboard'))
            
            # Récupérer les métriques de performance
            cur.execute("""
                SELECT 
                    AVG(daily_pnl) as avg_daily_return,
                    MAX(daily_pnl) as best_day,
                    MIN(daily_pnl) as worst_day,
                    (SELECT equity FROM performance_metrics ORDER BY date DESC LIMIT 1) as current_value,
                    (SELECT equity FROM performance_metrics ORDER BY date ASC LIMIT 1) as initial_value
                FROM performance_metrics
                WHERE date >= NOW() - INTERVAL '30 days'
            """)
            
            metrics_data = cur.fetchone()
        
        conn.close()
        
        # Convertir les données en format utilisable pour Plotly
        dates = [item[0].strftime('%Y-%m-%d') for item in portfolio_history]
        portfolio_values = [float(item[1]) for item in portfolio_history]
        
        # Créer un graphique de performance
        performance_chart = go.Figure()
        performance_chart.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='rgb(41, 128, 185)', width=2)
        ))
        
        performance_chart.update_layout(
            title='Portfolio Performance (Last 30 Days)',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_white'
        )
        
        # Calculer les métriques de performance
        if metrics_data:
            current_value = metrics_data[3]
            initial_value = metrics_data[4]
            total_return = ((current_value - initial_value) / initial_value * 100) if initial_value else 0
            
            performance_metrics = {
                'current_value': f"${current_value:.2f}",
                'initial_value': f"${initial_value:.2f}",
                'total_return': f"{total_return:.2f}%",
                'daily_avg_return': f"{metrics_data[0]:.2f}%" if metrics_data[0] else "0.00%",
                'best_day_return': f"{metrics_data[1]:.2f}%" if metrics_data[1] else "0.00%",
                'worst_day_return': f"{metrics_data[2]:.2f}%" if metrics_data[2] else "0.00%"
            }
        else:
            # Valeurs par défaut si aucune métrique n'est disponible
            performance_metrics = {
                'current_value': "$0.00",
                'initial_value': "$0.00",
                'total_return': "0.00%",
                'daily_avg_return': "0.00%",
                'best_day_return': "0.00%",
                'worst_day_return': "0.00%"
            }
        
        # Convertir le graphique en JSON pour le template
        performance_chart_json = pio.to_json(performance_chart)
        
        return render_template('performance.html',
                            performance_chart=performance_chart_json,
                            metrics=performance_metrics)
    except Exception as e:
        # Logger l'erreur
        logging.error(f"Error retrieving performance data: {str(e)}")
        
        # Message d'erreur pour l'utilisateur
        flash(f"Unable to retrieve performance data: {str(e)}", "error")
        return redirect(url_for('main.dashboard'))

@main_blueprint.route('/settings', methods=['GET', 'POST'])
def settings():
    # Charger les variables d'environnement actuelles
    load_dotenv()
    
    # Récupérer les paramètres actuels
    current_settings = {
        'initial_balance': os.environ.get('INITIAL_BALANCE', '100000'),
        'risk_per_trade': os.environ.get('RISK_PER_TRADE', '0.02'),
        'max_positions': os.environ.get('MAX_POSITIONS', '5'),
        'confidence_threshold': os.environ.get('CONFIDENCE_THRESHOLD', '0.65'),
        'enable_live_trading': os.environ.get('ENABLE_LIVE_TRADING', 'false') == 'true',
        'symbols': os.environ.get('SYMBOLS', 'AAPL,GOOGL,MSFT,AMZN,TSLA,BTC-USD,ETH-USD'),
        'data_update_interval': os.environ.get('DATA_UPDATE_INTERVAL', '5'),
        'scanning_interval': os.environ.get('SCANNING_INTERVAL', '60'),
        'telegram_token': os.environ.get('TELEGRAM_TOKEN', ''),
        'telegram_chat_id': os.environ.get('TELEGRAM_CHAT_ID', '')
    }
    
    if request.method == 'POST':
        try:
            # Récupérer les paramètres du formulaire
            updated_settings = {
                'INITIAL_BALANCE': request.form.get('initial_balance', '100000'),
                'RISK_PER_TRADE': request.form.get('risk_per_trade', '0.02'),
                'MAX_POSITIONS': request.form.get('max_positions', '5'),
                'CONFIDENCE_THRESHOLD': request.form.get('confidence_threshold', '0.65'),
                'ENABLE_LIVE_TRADING': 'true' if request.form.get('enable_live_trading') else 'false',
                'SYMBOLS': request.form.get('symbols', 'AAPL,GOOGL,MSFT,AMZN,TSLA,BTC-USD,ETH-USD'),
                'DATA_UPDATE_INTERVAL': request.form.get('data_update_interval', '5'),
                'SCANNING_INTERVAL': request.form.get('scanning_interval', '60'),
                'TELEGRAM_TOKEN': request.form.get('telegram_token', ''),
                'TELEGRAM_CHAT_ID': request.form.get('telegram_chat_id', '')
            }
            
            # Mettre à jour l'environnement actuel pour la session en cours
            for key, value in updated_settings.items():
                os.environ[key] = value
            
            # Mettre à jour le fichier .env
            env_path = os.path.join(os.getcwd(), '.env')
            try:
                # Vérifier si le fichier existe, sinon le créer
                if not os.path.exists(env_path):
                    logging.info(f"Création du fichier .env manquant à {env_path}")
                    with open(env_path, 'w') as file:
                        # Créer un fichier vide qui sera rempli plus tard
                        pass
                
                # Lire le fichier existant s'il n'est pas vide
                env_lines = []
                if os.path.getsize(env_path) > 0:
                    with open(env_path, 'r') as file:
                        env_lines = file.readlines()
                
                # Mettre à jour les lignes existantes ou en ajouter de nouvelles
                updated_lines = []
                updated_keys = set()
                
                for line in env_lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        updated_lines.append(line)
                        continue
                    
                    if '=' in line:
                        key = line.split('=')[0].strip()
                        if key in updated_settings:
                            updated_lines.append(f"{key}={updated_settings[key]}")
                            updated_keys.add(key)
                        else:
                            updated_lines.append(line)
                
                # Ajouter les clés qui n'existaient pas encore
                for key, value in updated_settings.items():
                    if key not in updated_keys:
                        updated_lines.append(f"{key}={value}")
                
                # Écrire les mises à jour dans le fichier .env
                with open(env_path, 'w') as file:
                    file.write('\n'.join(updated_lines) + '\n')
                
                flash('Paramètres mis à jour avec succès!', 'success')
            except Exception as e:
                logging.error(f"Erreur lors de la mise à jour des paramètres: {e}")
                flash(f'Erreur lors de la mise à jour des paramètres: {e}', 'danger')
            
            # Mettre à jour les paramètres actuels pour l'affichage
            current_settings = {
                'initial_balance': updated_settings['INITIAL_BALANCE'],
                'risk_per_trade': updated_settings['RISK_PER_TRADE'],
                'max_positions': updated_settings['MAX_POSITIONS'],
                'confidence_threshold': updated_settings['CONFIDENCE_THRESHOLD'],
                'enable_live_trading': updated_settings['ENABLE_LIVE_TRADING'] == 'true',
                'symbols': updated_settings['SYMBOLS'],
                'data_update_interval': updated_settings['DATA_UPDATE_INTERVAL'],
                'scanning_interval': updated_settings['SCANNING_INTERVAL'],
                'telegram_token': updated_settings['TELEGRAM_TOKEN'],
                'telegram_chat_id': updated_settings['TELEGRAM_CHAT_ID']
            }
            
            # Mettre à jour les paramètres dans la base de données
            try:
                conn = psycopg2.connect(
                    dbname=os.environ.get('DB_NAME', 'trading_db'),
                    user=os.environ.get('DB_USER', 'trader'),
                    password=os.environ.get('DB_PASSWORD', 'secure_password'),
                    host=os.environ.get('DB_HOST', 'database'),
                    port=os.environ.get('DB_PORT', '5432')
                )
                
                with conn.cursor() as cur:
                    for key, value in updated_settings.items():
                        cur.execute("""
                            INSERT INTO bot_settings (key, value)
                            VALUES (%s, %s)
                            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                        """, (key, value))
                
                conn.commit()
                conn.close()
            except Exception as db_err:
                logging.error(f"Error updating settings in database: {str(db_err)}")
                # Continue even if database update fails
            
            return render_template('settings.html', 
                                  settings=current_settings)
                                  
        except Exception as e:
            logging.error(f"Error updating settings: {str(e)}")
            return render_template('settings.html', 
                                  settings=current_settings,
                                  error_message=f"Error updating settings: {str(e)}")
    
    return render_template('settings.html', settings=current_settings)

@main_blueprint.route('/control_bot/<action>', methods=['POST'])
def control_bot(action):
    """Contrôle l'état du bot (démarrer, mettre en pause, arrêter)"""
    if not trading_bot:
        return jsonify({'success': False, 'message': 'Trading bot not initialized'}), 500
    
    try:
        if action == 'start':
            # Dans une implémentation réelle, nous démarrerions le bot dans un thread séparé
            trading_bot.set_status('running')
            message = 'Trading bot started successfully'
        elif action == 'pause':
            trading_bot.set_status('paused')
            message = 'Trading bot paused successfully'
        elif action == 'stop':
            trading_bot.set_status('stopped')
            message = 'Trading bot stopped successfully'
        else:
            return jsonify({'success': False, 'message': f'Invalid action: {action}'}), 400
        
        return jsonify({'success': True, 'message': message, 'state': trading_bot.get_status()})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

# API endpoints for the advanced dashboard
@main_blueprint.route('/api/performance/equity')
def api_performance_equity():
    """API endpoint for equity chart data."""
    # Get time range from query parameter
    time_range = request.args.get('range', '1m')
    
    # Calculate start date based on time range
    end_date = datetime.now()
    if time_range == '7d':
        start_date = end_date - timedelta(days=7)
    elif time_range == '1m':
        start_date = end_date - timedelta(days=30)
    elif time_range == '3m':
        start_date = end_date - timedelta(days=90)
    elif time_range == '1y':
        start_date = end_date - timedelta(days=365)
    else:
        # Default to all data
        start_date = datetime(2010, 1, 1)
    
    try:
        # Get equity data from database
        conn = psycopg2.connect(
            dbname=os.environ.get('DB_NAME', 'trading_db'),
            user=os.environ.get('DB_USER', 'trader'),
            password=os.environ.get('DB_PASSWORD', 'secure_password'),
            host=os.environ.get('DB_HOST', 'database'),
            port=os.environ.get('DB_PORT', '5432')
        )
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date, equity, balance
                FROM performance_metrics
                WHERE date >= %s
                ORDER BY date ASC
            """, (start_date,))
            
            rows = cur.fetchall()
            
        conn.close()
        
        if rows:
            dates = [row[0].strftime('%Y-%m-%d') for row in rows]
            equity = [float(row[1]) for row in rows]
            balance = [float(row[2]) for row in rows]
            
            # Calculate drawdown
            peak = equity[0]
            drawdown = [0] * len(equity)
            
            for i in range(len(equity)):
                if equity[i] > peak:
                    peak = equity[i]
                
                drawdown_pct = ((peak - equity[i]) / peak) * 100 if peak > 0 else 0
                drawdown[i] = drawdown_pct
        else:
            # Si aucune donnée n'est disponible, retourner des tableaux vides
            dates = []
            equity = []
            balance = []
            drawdown = []
            
            # Log warning that no data is available
            logging.warning("No equity data available in database for the requested time range.")
        
        return jsonify({
            'dates': dates,
            'equity': equity,
            'balance': balance,
            'drawdown': drawdown
        })
    
    except Exception as e:
        logging.error(f"Error getting equity data: {e}")
        return jsonify({
            'dates': [],
            'equity': [],
            'balance': [],
            'drawdown': [],
            'error': str(e)
        })

@main_blueprint.route('/api/performance/trade_distribution')
def api_trade_distribution():
    """API endpoint for trade distribution chart data."""
    try:
        # Get trade data from database or position manager
        if trading_bot and hasattr(trading_bot, 'position_manager'):
            closed_positions = trading_bot.position_manager.closed_positions
            
            winning_trades = sum(1 for p in closed_positions if p.pnl > 0)
            losing_trades = sum(1 for p in closed_positions if p.pnl < 0)
            breakeven_trades = sum(1 for p in closed_positions if p.pnl == 0)
            
            return jsonify({
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'breakeven_trades': breakeven_trades,
                'total_trades': winning_trades + losing_trades + breakeven_trades
            })
        else:
            # Si aucune donnée n'est disponible, essayer de récupérer depuis la base de données
            conn = psycopg2.connect(
                dbname=os.environ.get('DB_NAME', 'trading_db'),
                user=os.environ.get('DB_USER', 'trader'),
                password=os.environ.get('DB_PASSWORD', 'secure_password'),
                host=os.environ.get('DB_HOST', 'database'),
                port=os.environ.get('DB_PORT', '5432')
            )
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                        SUM(CASE WHEN pnl = 0 THEN 1 ELSE 0 END) as breakeven_trades
                    FROM trades
                    WHERE closed_at IS NOT NULL
                """)
                
                result = cur.fetchone()
                
            conn.close()
            
            if result and result[0] is not None:
                winning_trades = int(result[0])
                losing_trades = int(result[1])
                breakeven_trades = int(result[2])
                
                return jsonify({
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'breakeven_trades': breakeven_trades,
                    'total_trades': winning_trades + losing_trades + breakeven_trades
                })
            else:
                logging.warning("No trade distribution data available")
                return jsonify({
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'breakeven_trades': 0,
                    'total_trades': 0,
                    'data_available': False
                })
    
    except Exception as e:
        logging.error(f"Error getting trade distribution data: {e}")
        return jsonify({
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'total_trades': 0,
            'error': str(e)
        })

@main_blueprint.route('/api/performance/symbol_performance')
def api_symbol_performance():
    """API endpoint for symbol performance chart data."""
    try:
        symbols_data = {}
        
        # 1. D'abord, essayez de récupérer les données à partir du position_manager
        if trading_bot and hasattr(trading_bot, 'position_manager'):
            # Get all unique symbols from closed positions
            symbols = set()
            for position in trading_bot.position_manager.closed_positions:
                symbols.add(position.symbol)
            
            # Calculate performance metrics for each symbol
            for symbol in symbols:
                symbol_positions = [p for p in trading_bot.position_manager.closed_positions if p.symbol == symbol]
                
                winning_trades = sum(1 for p in symbol_positions if p.pnl > 0)
                total_trades = len(symbol_positions)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                total_pnl = sum(p.pnl for p in symbol_positions)
                
                symbols_data[symbol] = {
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'total_trades': total_trades
                }
        
        # 2. Si les données du position_manager sont insuffisantes, récupérer depuis la base de données
        if not symbols_data:
            conn = psycopg2.connect(
                dbname=os.environ.get('DB_NAME', 'trading_db'),
                user=os.environ.get('DB_USER', 'trader'),
                password=os.environ.get('DB_PASSWORD', 'secure_password'),
                host=os.environ.get('DB_HOST', 'database'),
                port=os.environ.get('DB_PORT', '5432')
            )
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        symbol,
                        SUM(pnl) as total_pnl,
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades
                    FROM trades
                    WHERE closed_at IS NOT NULL
                    GROUP BY symbol
                """)
                
                for row in cur.fetchall():
                    symbol = row[0]
                    total_pnl = float(row[1])
                    total_trades = int(row[2])
                    winning_trades = int(row[3])
                    
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0
                    
                    symbols_data[symbol] = {
                        'total_pnl': total_pnl,
                        'win_rate': win_rate,
                        'total_trades': total_trades
                    }
            
            conn.close()
            
        return jsonify({
            'symbols': symbols_data,
            'data_available': len(symbols_data) > 0
        })
    
    except Exception as e:
        logging.error(f"Error getting symbol performance data: {e}")
        return jsonify({
            'symbols': {},
            'data_available': False,
            'error': str(e)
        })

@main_blueprint.route('/api/predictions/<symbol>')
def api_price_predictions(symbol):
    """API endpoint for price prediction chart data."""
    try:
        # Try to get actual data for the symbol
        actual_data = None
        if data_manager and symbol in data_manager.data:
            actual_data = data_manager.data[symbol]
        
        if actual_data is not None and not actual_data.empty:
            # Get the last 30 days of data
            recent_data = actual_data.tail(30).copy()
            dates = recent_data.index.strftime('%Y-%m-%d').tolist()
            actual_prices = recent_data['Close'].tolist()
            
            # Variables pour stocker les prédictions réelles
            predicted_prices = None
            transformer_predictions = None
            
            # Récupérer les prédictions réelles des modèles
            if trading_bot:
                model_predictions = trading_bot.get_model_predictions(symbol)
                
                if model_predictions and 'price_predictions' in model_predictions:
                    predicted_prices = model_predictions['price_predictions']
                
                if model_predictions and 'transformer_predictions' in model_predictions:
                    transformer_predictions = model_predictions['transformer_predictions']
            
            # Si les prédictions ne sont pas disponibles, utiliser seulement les données réelles
            if predicted_prices is None or len(predicted_prices) == 0:
                logging.warning(f"No price predictions available for {symbol}")
                predicted_prices = actual_prices
            
            if transformer_predictions is None or len(transformer_predictions) == 0:
                logging.warning(f"No transformer predictions available for {symbol}")
                transformer_predictions = actual_prices
            
            # Assurer que toutes les listes ont la même longueur
            max_len = max(len(actual_prices), len(predicted_prices), len(transformer_predictions))
            
            # Ajouter des valeurs nulles pour les dates futures des prix réels
            actual_prices_with_nulls = actual_prices + [None] * (max_len - len(actual_prices))
            
            # Assurez-vous que les dates correspondent à la plus longue série
            while len(dates) < max_len:
                last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
                future_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                dates.append(future_date)
            
            return jsonify({
                'dates': dates,
                'actual_prices': actual_prices_with_nulls,
                'predicted_prices': predicted_prices,
                'transformer_predictions': transformer_predictions,
                'data_available': True
            })
        
        else:
            logging.warning(f"No market data available for {symbol}")
            return jsonify({
                'dates': [],
                'actual_prices': [],
                'predicted_prices': [],
                'transformer_predictions': [],
                'data_available': False,
                'message': f"No market data available for {symbol}"
            })
    
    except Exception as e:
        logging.error(f"Error getting price prediction data: {e}")
        return jsonify({
            'dates': [],
            'actual_prices': [],
            'predicted_prices': [],
            'transformer_predictions': [],
            'data_available': False,
            'error': str(e)
        })

@main_blueprint.route('/api/risk/heatmap')
def api_risk_heatmap():
    """API endpoint for risk heatmap data."""
    try:
        symbols_risk = {}
        
        # Essayer de récupérer les données de risque réelles depuis le modèle de risque
        if trading_bot and trading_bot.risk_model:
            if data_manager:
                for symbol in data_manager.symbols:
                    if symbol in data_manager.data and not data_manager.data[symbol].empty:
                        try:
                            risk_score = trading_bot.risk_model.predict_risk(data_manager.data[symbol])
                            symbols_risk[symbol] = risk_score
                        except Exception as e:
                            # En cas d'erreur, ne pas ajouter de données aléatoires mais logger l'erreur
                            logging.error(f"Error predicting risk for {symbol}: {e}")
        
        # Si aucune donnée n'est disponible, tenter de récupérer les évaluations de risque depuis la base de données
        if len(symbols_risk) == 0:
            try:
                conn = psycopg2.connect(
                    dbname=os.environ.get('DB_NAME', 'trading_db'),
                    user=os.environ.get('DB_USER', 'trader'),
                    password=os.environ.get('DB_PASSWORD', 'secure_password'),
                    host=os.environ.get('DB_HOST', 'database'),
                    port=os.environ.get('DB_PORT', '5432')
                )
                
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT symbol, risk_score 
                        FROM risk_assessments
                        WHERE created_at > NOW() - INTERVAL '1 day'
                        ORDER BY created_at DESC
                    """)
                    
                    rows = cur.fetchall()
                    for row in rows:
                        symbol = row[0]
                        risk_score = float(row[1])
                        
                        # Ne garder que la dernière évaluation par symbole
                        if symbol not in symbols_risk:
                            symbols_risk[symbol] = risk_score
                
                conn.close()
            except Exception as e:
                logging.error(f"Error fetching risk data from database: {e}")
        
        # Sort by risk score
        sorted_symbols = sorted(symbols_risk.items(), key=lambda x: x[1], reverse=True)
        symbols = [item[0] for item in sorted_symbols]
        risk_scores = [item[1] for item in sorted_symbols]
        
        return jsonify({
            'symbols': symbols,
            'risk_scores': risk_scores,
            'data_available': len(symbols_risk) > 0
        })
    
    except Exception as e:
        logging.error(f"Error getting risk heatmap data: {e}")
        return jsonify({
            'symbols': [],
            'risk_scores': [],
            'data_available': False,
            'error': str(e)
        })

@main_blueprint.route('/api/sentiment/timeline')
def api_sentiment_timeline():
    """API endpoint for sentiment timeline data."""
    try:
        # Générer dates pour les 30 derniers jours
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                for i in range((end_date - start_date).days + 1)]
        
        # Initialiser la structure pour récupérer les données de sentiment
        sentiment_data = {
            'dates': dates,
            'symbols': [],
            'sentiment_scores': [],
            'data_available': False
        }
        
        # Récupérer les données réelles de sentiment depuis la base de données
        try:
            conn = psycopg2.connect(
                dbname=os.environ.get('DB_NAME', 'trading_db'),
                user=os.environ.get('DB_USER', 'trader'),
                password=os.environ.get('DB_PASSWORD', 'secure_password'),
                host=os.environ.get('DB_HOST', 'database'),
                port=os.environ.get('DB_PORT', '5432')
            )
            
            with conn.cursor() as cur:
                # Récupérer les 5 symboles les plus fréquemment analysés
                cur.execute("""
                    SELECT symbol, COUNT(*) as count
                    FROM sentiment_analysis
                    WHERE created_at >= %s
                    GROUP BY symbol
                    ORDER BY count DESC
                    LIMIT 5
                """, (start_date,))
                
                top_symbols = [row[0] for row in cur.fetchall()]
                
                if not top_symbols and data_manager:
                    # Si aucun symbole n'a de données de sentiment, utiliser les premiers symboles du data_manager
                    top_symbols = data_manager.symbols[:5]
                
                # Si toujours pas de symboles, ne pas continuer
                if not top_symbols:
                    conn.close()
                    return jsonify(sentiment_data)
                
                sentiment_data['symbols'] = top_symbols
                
                # Pour chaque symbole, récupérer l'historique de sentiment
                for symbol in top_symbols:
                    # Récupérer les scores de sentiment pour ce symbole
                    cur.execute("""
                        SELECT date_trunc('day', created_at)::date as day, AVG(sentiment_score) as avg_score
                        FROM sentiment_analysis
                        WHERE symbol = %s AND created_at >= %s
                        GROUP BY day
                        ORDER BY day ASC
                    """, (symbol, start_date))
                    
                    # Initialiser un dictionnaire pour stocker les scores par date
                    daily_scores = {date: None for date in dates}
                    
                    # Remplir les dates pour lesquelles nous avons des données
                    for row in cur.fetchall():
                        date_str = row[0].strftime('%Y-%m-%d')
                        if date_str in daily_scores:
                            daily_scores[date_str] = float(row[1])
                    
                    # Convertir en liste et interpoler les valeurs manquantes
                    scores_list = list(daily_scores.values())
                    
                    # Interpolation linéaire pour les valeurs manquantes
                    last_valid = None
                    for i in range(len(scores_list)):
                        if scores_list[i] is not None:
                            last_valid = scores_list[i]
                        elif last_valid is not None:
                            scores_list[i] = last_valid
                    
                    # Si des valeurs sont toujours None au début, utiliser la première valeur valide
                    first_valid = next((x for x in scores_list if x is not None), 0)
                    scores_list = [first_valid if x is None else x for x in scores_list]
                    
                    sentiment_data['sentiment_scores'].append(scores_list)
            
            conn.close()
            
            # Si des données ont été trouvées, marquer comme disponible
            if sentiment_data['sentiment_scores']:
                sentiment_data['data_available'] = True
            
        except Exception as e:
            logging.error(f"Error getting sentiment timeline data from database: {e}")
            
            # Si l'accès à la base de données a échoué, essayer d'utiliser l'analyseur de sentiment en direct
            if trading_bot and trading_bot.sentiment_analyzer:
                symbols = []
                if data_manager:
                    symbols = data_manager.symbols[:5]  # Limiter aux 5 premiers symboles
                
                if symbols:
                    sentiment_data['symbols'] = symbols
                    sentiment_scores = []
                    
                    for symbol in symbols:
                        try:
                            # Analyser le sentiment actuel
                            sentiment = trading_bot.sentiment_analyzer.analyze_sentiment(symbol)
                            
                            if sentiment and 'sentiment_score' in sentiment:
                                # Utiliser le score actuel pour toutes les dates (pas idéal mais mieux que rien)
                                current_score = sentiment['sentiment_score']
                                sentiment_scores.append([current_score] * len(dates))
                                sentiment_data['data_available'] = True
                            
                        except Exception as e:
                            logging.error(f"Error analyzing sentiment for {symbol}: {e}")
                    
                    if sentiment_scores:
                        sentiment_data['sentiment_scores'] = sentiment_scores
        
        return jsonify(sentiment_data)
    
    except Exception as e:
        logging.error(f"Error in sentiment timeline API: {e}")
        return jsonify({
            'dates': [],
            'symbols': [],
            'sentiment_scores': [],
            'data_available': False,
            'error': str(e)
        })

@main_blueprint.route('/api/signals/<signal_id>')
def api_signal_details(signal_id):
    """API endpoint for signal details."""
    try:
        # Convert signal_id to int for indexing
        idx = int(signal_id) - 1
        
        # Get signals from trading bot
        all_signals = []
        if trading_bot:
            for signal_group in trading_bot.latest_signals:
                for signal in signal_group.get('signals', []):
                    all_signals.append(signal)
        
        # If we have the requested signal, return it
        if idx >= 0 and idx < len(all_signals):
            signal = all_signals[idx]
            
            # Get additional data for the signal
            symbol = signal.get('symbol')
            
            # Add predictions if we have market data
            if data_manager and symbol in data_manager.data and not data_manager.data[symbol].empty:
                current_price = data_manager.data[symbol]['Close'].iloc[-1]
                
                # Generate some prediction data
                signal['predictions'] = {
                    'current_price': current_price,
                    'standard_prediction': current_price * (1 + random.uniform(-0.05, 0.08)),
                    'transformer_prediction': current_price * (1 + random.uniform(-0.03, 0.07))
                }
                
                # Calculate changes
                signal['predictions']['standard_change'] = (
                    (signal['predictions']['standard_prediction'] / current_price) - 1
                ) * 100
                
                signal['predictions']['transformer_change'] = (
                    (signal['predictions']['transformer_prediction'] / current_price) - 1
                ) * 100
            
            return jsonify(signal)
        
        # If not, return sample data
        else:
            sample_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BTC-USD', 'ETH-USD']
            sample_signal = {
                'symbol': random.choice(sample_symbols),
                'direction': random.choice(['buy', 'sell']),
                'confidence': random.uniform(0.5, 0.9),
                'source': random.choice(['price_model', 'transformer', 'sentiment']),
                'timestamp': datetime.now().isoformat(),
                'risk_score': random.uniform(0.1, 0.5),
                'predicted_change': random.uniform(-5, 8)
            }
            
            # Add sentiment data
            if sample_signal['source'] == 'sentiment':
                sample_signal['sentiment'] = {
                    'sentiment_score': random.uniform(-0.8, 0.8),
                    'sources': random.sample(['Twitter', 'News', 'Reddit', 'Financial Reports'], k=2),
                    'keywords': random.sample(['earnings', 'growth', 'revenue', 'product', 'competition', 'market'], k=3)
                }
            
            # Add prediction data
            current_price = 100 + random.uniform(-20, 50)
            sample_signal['predictions'] = {
                'current_price': current_price,
                'standard_prediction': current_price * (1 + random.uniform(-0.05, 0.08)),
                'transformer_prediction': current_price * (1 + random.uniform(-0.03, 0.07))
            }
            
            # Calculate changes
            sample_signal['predictions']['standard_change'] = (
                (sample_signal['predictions']['standard_prediction'] / current_price) - 1
            ) * 100
            
            sample_signal['predictions']['transformer_change'] = (
                (sample_signal['predictions']['transformer_prediction'] / current_price) - 1
            ) * 100
            
            return jsonify(sample_signal)
    
    except Exception as e:
        logging.error(f"Error getting signal details: {e}")
        return jsonify({
            'error': f"Error getting signal details: {str(e)}"
        }), 500

@main_blueprint.route('/api/execute_trade', methods=['POST'])
def api_execute_trade():
    """API endpoint to execute a trade based on a signal."""
    try:
        data = request.json
        
        # Validate required parameters
        if not data or 'symbol' not in data or 'direction' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400
        
        # Get parameters
        symbol = data['symbol']
        direction = data['direction']
        
        # Execute the trade if trading bot is available
        if trading_bot and hasattr(trading_bot, 'position_manager'):
            # Get current price for the symbol
            current_price = None
            
            if data_manager and symbol in data_manager.data and not data_manager.data[symbol].empty:
                current_price = data_manager.data[symbol]['Close'].iloc[-1]
            else:
                return jsonify({
                    'success': False,
                    'error': 'Price data not available for the symbol'
                }), 400
            
            try:
                # Calculate position size (2% risk)
                position_size = trading_bot.calculate_position_size(
                    symbol=symbol,
                    price=current_price,
                    risk_percentage=0.02,
                    stop_loss_percentage=0.05
                )
                
                # Calculate stop loss and take profit levels
                stop_loss = None
                take_profit = None
                
                if direction == 'buy':
                    stop_loss = current_price * 0.95  # 5% below current price
                    take_profit = current_price * 1.15  # 15% above current price
                else:  # sell
                    stop_loss = current_price * 1.05  # 5% above current price
                    take_profit = current_price * 0.85  # 15% below current price
                
                # Open position
                position = trading_bot.position_manager.open_position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=current_price,
                    size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'signal_id': data.get('signal_id'),
                        'signal_source': 'API',
                        'execution_time': datetime.now().isoformat()
                    }
                )
                
                return jsonify({
                    'success': True,
                    'order_id': position.id,
                    'symbol': symbol,
                    'direction': direction,
                    'size': position_size,
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
                
            except Exception as e:
                logging.error(f"Error executing trade: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        else:
            return jsonify({
                'success': False,
                'error': 'Trading bot not available'
            }), 503
    
    except Exception as e:
        logging.error(f"Error in trade execution API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@main_blueprint.route('/api/system/cpu')
def api_system_cpu():
    """API endpoint for real CPU and memory usage data."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory_info = psutil.virtual_memory()
        
        return jsonify({
            'cpu_percent': cpu_percent,
            'memory_percent': memory_info.percent,
            'memory_used': memory_info.used,
            'memory_total': memory_info.total
        })
    
    except Exception as e:
        logging.error(f"Error getting system CPU data: {e}")
        return jsonify({
            'cpu_percent': 0,
            'memory_percent': 0,
            'memory_used': 0,
            'memory_total': 0
        })