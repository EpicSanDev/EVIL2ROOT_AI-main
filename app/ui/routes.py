from flask import Blueprint, render_template, jsonify, request, redirect, url_for, flash, session
import psutil
import plotly.graph_objs as go
import plotly.io as pio
# Configuration optimisée pour Plotly
pio.templates.default = "plotly_white"  # Définir un template par défaut
from app import trading_bot, data_manager
from app.model_trainer import ModelTrainer
import json
from datetime import datetime, timedelta
import random  # Pour les données de démonstration
import os
from dotenv import load_dotenv
import psycopg2
import logging
from app.plugins import plugin_manager
import tempfile
from flask_login import login_required, login_user, logout_user, current_user
from app import User
from werkzeug.security import check_password_hash
# Import pour les explications des modèles
from app.models.ensemble_integrator import EnsembleIntegrator

# Configuration des templates Plotly pour optimiser le temps de rendu
# Utiliser des templates minimalistes pour réduire le temps de génération
simple_layout = {
    'margin': {'l': 40, 'r': 20, 't': 40, 'b': 30},
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'showlegend': False,
    'hovermode': 'closest'
}

main_blueprint = Blueprint('main', __name__)

# Initialize model trainer with global trading_bot
model_trainer = ModelTrainer(trading_bot)

# Authentication routes
@main_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already authenticated, redirect to dashboard
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    # Handle login form submission
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Validation des entrées
        if not username or not password:
            flash('Veuillez fournir un nom d\'utilisateur et un mot de passe', 'error')
            return render_template('login.html')
        
        # Vérification des tentatives de connexion
        from app import check_login_attempts
        if not check_login_attempts(username):
            flash('Trop de tentatives de connexion échouées. Compte temporairement bloqué. Veuillez réessayer plus tard.', 'error')
            return render_template('login.html', is_locked=True)
        
        # Tentative d'authentification
        # Seul l'administrateur est autorisé pour le moment
        admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
        if username == admin_username:
            # Chargement de l'utilisateur
            user = User(1, admin_username)
            
            # Vérification du mot de passe
            # Nous appelons load_user pour obtenir l'objet utilisateur avec le hash et le sel
            from app import load_user
            admin_user = load_user('1')
            
            if admin_user and admin_user.verify_password(password):
                # Authentification réussie
                login_user(admin_user, remember=True)
                
                # Réinitialiser le compteur de tentatives de connexion
                check_login_attempts(username, success=True)
                
                # Redirection vers la page demandée ou le tableau de bord
                next_page = request.args.get('next')
                if next_page and next_page.startswith('/'):
                    return redirect(next_page)
                return redirect(url_for('main.dashboard'))
            else:
                # Authentification échouée
                flash('Nom d\'utilisateur ou mot de passe incorrect', 'error')
                # Nous ne réinitialisons pas le compteur de tentatives
        else:
            # Utilisateur non trouvé
            flash('Nom d\'utilisateur ou mot de passe incorrect', 'error')
            # Nous utilisons le même message pour ne pas révéler si l'utilisateur existe
    
    return render_template('login.html')

@main_blueprint.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Vous avez été déconnecté avec succès', 'success')
    return redirect(url_for('main.login'))

@main_blueprint.route('/')
def dashboard():
    # CPU/GPU Performance Data
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    # Prepare CPU usage chart - OPTIMISÉ
    cpu_chart = go.Figure(data=[go.Bar(x=['CPU Usage'], y=[cpu_percent], marker=dict(color='rgb(55, 83, 109)'))])
    cpu_chart.update_layout(
        title='CPU Usage (%)',
        **simple_layout  # Utiliser le layout simplifié
    )

    # Convert the chart to JSON - OPTIMISÉ
    cpu_chart_json = pio.to_json(cpu_chart, validate=False, pretty=False)  # Désactiver validation pour plus de vitesse

    # Get trading signals
    signals = trading_bot.get_latest_signals() if trading_bot else []

    return render_template('dashboard.html', 
                          cpu_percent=cpu_percent,
                          memory_info=memory_info,
                          cpu_chart=cpu_chart_json,
                          signals=signals)

@main_blueprint.route('/bot_status')
def get_bot_status():
    """Get the current status of the trading bot."""
    status = 'stopped'
    error = None
    
    try:
        if trading_bot:
            status = trading_bot.get_status()
        else:
            error = "Trading bot not initialized"
    except Exception as e:
        error = str(e)
        
    return jsonify({
        'success': not error,
        'state': status,
        'error': error
    })

@main_blueprint.route('/control_bot/<action>', methods=['POST'])
def control_bot(action):
    """
    Control the trading bot.
    
    Args:
        action: The action to take (start, pause, stop)
        
    Returns:
        JSON response with the result of the action
    """
    error = None
    message = ""
    
    if not trading_bot:
        return jsonify({
            'success': False,
            'message': "Trading bot not initialized",
            'state': 'stopped'
        })
    
    try:
        if action == 'start':
            message = "Bot started successfully"
            trading_bot.start()
        elif action == 'pause':
            message = "Bot paused successfully"
            trading_bot.pause()
        elif action == 'stop':
            message = "Bot stopped successfully"
            trading_bot.stop()
        else:
            error = f"Unknown action: {action}"
    except Exception as e:
        error = str(e)
    
    status = trading_bot.get_status()
    
    return jsonify({
        'success': not error,
        'message': error or message,
        'state': status
    })

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
        
        # Créer un graphique de performance - OPTIMISÉ
        performance_chart = go.Figure()
        # Utiliser une configuration plus légère pour le scatter plot
        performance_chart.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='rgb(41, 128, 185)', width=2)
        ))
        
        # Appliquer un layout simplifié
        performance_chart.update_layout(
            title='Portfolio Performance (Last 30 Days)',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            **simple_layout  # Utiliser le layout simplifié
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
@login_required
def settings():
    # Charger les variables d'environnement actuelles
    dotenv_path = os.path.join(os.getcwd(), '.env')
    secrets_path = os.path.join(os.getcwd(), 'config/secrets.env')
    
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
    
    # Charger les clés API depuis le fichier secrets.env
    api_keys = {}
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        if key in ['BINANCE_API_KEY', 'BINANCE_API_SECRET', 'NEWSAPI_KEY', 'FINNHUB_KEY',
                                 'TWITTER_CONSUMER_KEY', 'TWITTER_CONSUMER_SECRET', 
                                 'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_TOKEN_SECRET']:
                            api_keys[key] = value
        except Exception as e:
            logging.error(f"Erreur lors de la lecture du fichier secrets.env: {e}")
            flash(f'Erreur lors de la lecture des clés API: {e}', 'danger')
    
    # Ajouter les clés API aux paramètres actuels
    current_settings.update({
        'binance_api_key': api_keys.get('BINANCE_API_KEY', ''),
        'binance_api_secret': api_keys.get('BINANCE_API_SECRET', ''),
        'newsapi_key': api_keys.get('NEWSAPI_KEY', ''),
        'finnhub_key': api_keys.get('FINNHUB_KEY', ''),
        'twitter_consumer_key': api_keys.get('TWITTER_CONSUMER_KEY', ''),
        'twitter_consumer_secret': api_keys.get('TWITTER_CONSUMER_SECRET', ''),
        'twitter_access_token': api_keys.get('TWITTER_ACCESS_TOKEN', ''),
        'twitter_access_token_secret': api_keys.get('TWITTER_ACCESS_TOKEN_SECRET', '')
    })
    
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
            
            # Récupérer les clés API du formulaire
            api_settings = {
                'BINANCE_API_KEY': request.form.get('binance_api_key', ''),
                'BINANCE_API_SECRET': request.form.get('binance_api_secret', ''),
                'NEWSAPI_KEY': request.form.get('newsapi_key', ''),
                'FINNHUB_KEY': request.form.get('finnhub_key', ''),
                'TWITTER_CONSUMER_KEY': request.form.get('twitter_consumer_key', ''),
                'TWITTER_CONSUMER_SECRET': request.form.get('twitter_consumer_secret', ''),
                'TWITTER_ACCESS_TOKEN': request.form.get('twitter_access_token', ''),
                'TWITTER_ACCESS_TOKEN_SECRET': request.form.get('twitter_access_token_secret', '')
            }
            
            # Vérifier le mot de passe pour confirmer les modifications des clés API
            confirm_password = request.form.get('confirm_password', '')
            api_settings_changed = any(
                api_settings[key] != api_keys.get(key, '') 
                for key in api_settings.keys()
            )
            
            # Si des clés API ont été modifiées, vérifier le mot de passe
            save_api_settings = False
            if api_settings_changed:
                if not confirm_password:
                    flash('Mot de passe requis pour modifier les clés API', 'error')
                    return render_template('settings.html', settings=current_settings, 
                                          error_message="Veuillez entrer votre mot de passe pour confirmer les modifications des clés API")
                
                # Vérifier le mot de passe administrateur
                from app import load_user
                admin_user = load_user('1')
                if not admin_user or not admin_user.verify_password(confirm_password):
                    flash('Mot de passe incorrect. Les clés API n\'ont pas été modifiées.', 'error')
                    # Restaurer les anciennes valeurs des clés API
                    for key in api_settings.keys():
                        api_settings[key] = api_keys.get(key, '')
                else:
                    save_api_settings = True
            
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
                
                # Mettre à jour le fichier secrets.env pour les clés API
                updated_api_settings = False
                secrets_updated_lines = []
                updated_api_keys = set()
                
                # S'assurer que le répertoire config existe
                os.makedirs(os.path.dirname(secrets_path), exist_ok=True)
                
                # Lire le fichier secrets.env existant s'il existe
                if os.path.exists(secrets_path) and os.path.getsize(secrets_path) > 0:
                    with open(secrets_path, 'r') as file:
                        secrets_lines = file.readlines()
                    
                    for line in secrets_lines:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            secrets_updated_lines.append(line)
                            continue
                        
                        if '=' in line:
                            key = line.split('=')[0].strip()
                            if key in api_settings:
                                # Ne mettre à jour que si la clé n'est pas vide et si le mot de passe est validé
                                if api_settings[key] and save_api_settings:
                                    secrets_updated_lines.append(f"{key}={api_settings[key]}")
                                    updated_api_settings = True
                                else:
                                    secrets_updated_lines.append(line)
                                updated_api_keys.add(key)
                            else:
                                secrets_updated_lines.append(line)
                else:
                    # Créer l'en-tête du fichier secrets.env s'il n'existe pas
                    secrets_updated_lines.append("# Fichier de secrets pour le bot de trading")
                    secrets_updated_lines.append("# Remplacez les valeurs par vos véritables clés API")
                    secrets_updated_lines.append("")
                    secrets_updated_lines.append("# Clés API pour Binance")
                    
                # Ajouter les clés API manquantes dans le fichier
                if save_api_settings:  # Ne mettre à jour que si le mot de passe est validé
                    for key, value in api_settings.items():
                        # N'ajouter que si la valeur n'est pas vide et la clé n'existe pas encore
                        if value and key not in updated_api_keys:
                            if key in ['BINANCE_API_KEY', 'BINANCE_API_SECRET'] and not any(l.startswith('# Clés API pour Binance') for l in secrets_updated_lines):
                                secrets_updated_lines.append("# Clés API pour Binance")
                            elif key in ['NEWSAPI_KEY', 'FINNHUB_KEY'] and not any(l.startswith('# Clés API pour l\'analyse de sentiment') for l in secrets_updated_lines):
                                secrets_updated_lines.append("")
                                secrets_updated_lines.append("# Clés API pour l'analyse de sentiment")
                            elif key in ['TWITTER_CONSUMER_KEY', 'TWITTER_CONSUMER_SECRET', 'TWITTER_ACCESS_TOKEN', 'TWITTER_ACCESS_TOKEN_SECRET'] and not any(l.startswith('# Clés API pour Twitter') for l in secrets_updated_lines):
                                secrets_updated_lines.append("")
                                secrets_updated_lines.append("# Clés API pour Twitter (optionnelles)")
                            
                            secrets_updated_lines.append(f"{key}={value}")
                            updated_api_settings = True
                
                # Écrire les mises à jour dans le fichier secrets.env seulement si des changements ont été faits
                if updated_api_settings:
                    with open(secrets_path, 'w') as file:
                        file.write('\n'.join(secrets_updated_lines) + '\n')
                    if save_api_settings:
                        flash('Paramètres et clés API mis à jour avec succès!', 'success')
                    else:
                        flash('Paramètres mis à jour avec succès! (Clés API non modifiées)', 'success')
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
                'telegram_chat_id': updated_settings['TELEGRAM_CHAT_ID'],
                'binance_api_key': api_settings['BINANCE_API_KEY'],
                'binance_api_secret': api_settings['BINANCE_API_SECRET'],
                'newsapi_key': api_settings['NEWSAPI_KEY'],
                'finnhub_key': api_settings['FINNHUB_KEY'],
                'twitter_consumer_key': api_settings['TWITTER_CONSUMER_KEY'],
                'twitter_consumer_secret': api_settings['TWITTER_CONSUMER_SECRET'],
                'twitter_access_token': api_settings['TWITTER_ACCESS_TOKEN'],
                'twitter_access_token_secret': api_settings['TWITTER_ACCESS_TOKEN_SECRET']
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

@main_blueprint.route('/plugins')
@login_required
def plugins_page():
    """Page d'administration des plugins"""
    try:
        # Récupérer la liste des plugins installés
        installed_plugins = plugin_manager.get_installed_plugins()
        
        # Récupérer la liste des plugins disponibles
        available_plugins = plugin_manager.discover_plugins()
        
        # Fusionner les listes pour éviter les doublons
        available_ids = {plugin["id"] for plugin in available_plugins}
        installed_ids = {plugin["id"] for plugin in installed_plugins}
        
        # Liste des plugins découverts mais non installés
        discoverable_plugins = [p for p in available_plugins if p["id"] not in installed_ids]
        
        return render_template(
            'plugins.html',
            installed_plugins=installed_plugins,
            discoverable_plugins=discoverable_plugins,
            title="Gestion des Plugins"
        )
    except Exception as e:
        logging.error(f"Erreur lors du chargement de la page des plugins: {e}")
        flash(f"Erreur lors du chargement des plugins: {str(e)}", "danger")
        return redirect(url_for('main.index'))

@main_blueprint.route('/api/plugins/enable/<plugin_id>', methods=['POST'])
@login_required
def enable_plugin(plugin_id):
    """Active un plugin"""
    try:
        result = plugin_manager.enable_plugin(plugin_id)
        if result:
            flash(f"Plugin {plugin_id} activé avec succès", "success")
        else:
            flash(f"Erreur lors de l'activation du plugin {plugin_id}", "danger")
    except Exception as e:
        logging.error(f"Erreur lors de l'activation du plugin {plugin_id}: {e}")
        flash(f"Erreur lors de l'activation du plugin: {str(e)}", "danger")
    
    return redirect(url_for('main.plugins_page'))

@main_blueprint.route('/api/plugins/disable/<plugin_id>', methods=['POST'])
@login_required
def disable_plugin(plugin_id):
    """Désactive un plugin"""
    try:
        result = plugin_manager.disable_plugin(plugin_id)
        if result:
            flash(f"Plugin {plugin_id} désactivé avec succès", "success")
        else:
            flash(f"Erreur lors de la désactivation du plugin {plugin_id}", "danger")
    except Exception as e:
        logging.error(f"Erreur lors de la désactivation du plugin {plugin_id}: {e}")
        flash(f"Erreur lors de la désactivation du plugin: {str(e)}", "danger")
    
    return redirect(url_for('main.plugins_page'))

@main_blueprint.route('/api/plugins/uninstall/<plugin_id>', methods=['POST'])
@login_required
def uninstall_plugin(plugin_id):
    """Désinstalle un plugin"""
    try:
        result = plugin_manager.uninstall_plugin(plugin_id)
        if result:
            flash(f"Plugin {plugin_id} désinstallé avec succès", "success")
        else:
            flash(f"Erreur lors de la désinstallation du plugin {plugin_id}", "danger")
    except Exception as e:
        logging.error(f"Erreur lors de la désinstallation du plugin {plugin_id}: {e}")
        flash(f"Erreur lors de la désinstallation du plugin: {str(e)}", "danger")
    
    return redirect(url_for('main.plugins_page'))

@main_blueprint.route('/api/plugins/install', methods=['POST'])
@login_required
def install_plugin():
    """Installe un plugin à partir d'un fichier ZIP"""
    if 'plugin_file' not in request.files:
        flash("Aucun fichier fourni", "danger")
        return redirect(url_for('main.plugins_page'))
    
    plugin_file = request.files['plugin_file']
    
    if plugin_file.filename == '':
        flash("Aucun fichier sélectionné", "danger")
        return redirect(url_for('main.plugins_page'))
    
    if not plugin_file.filename.endswith('.zip'):
        flash("Le fichier doit être au format ZIP", "danger")
        return redirect(url_for('main.plugins_page'))
    
    try:
        # Sauvegarder le fichier temporairement
        temp_path = tempfile.mktemp(suffix='.zip')
        plugin_file.save(temp_path)
        
        # Installer le plugin
        plugin_id = plugin_manager.install_plugin(temp_path)
        
        # Supprimer le fichier temporaire
        os.unlink(temp_path)
        
        if plugin_id:
            flash(f"Plugin {plugin_id} installé avec succès", "success")
        else:
            flash("Erreur lors de l'installation du plugin", "danger")
    
    except Exception as e:
        logging.error(f"Erreur lors de l'installation du plugin: {e}")
        flash(f"Erreur lors de l'installation du plugin: {str(e)}", "danger")
    
    return redirect(url_for('main.plugins_page'))

@main_blueprint.route('/api/plugins/settings/<plugin_id>', methods=['GET', 'POST'])
@login_required
def plugin_settings(plugin_id):
    """Gère les paramètres d'un plugin"""
    # Vérifier si le plugin existe
    if plugin_id not in plugin_manager.plugins:
        flash(f"Plugin {plugin_id} non trouvé", "danger")
        return redirect(url_for('main.plugins_page'))
    
    plugin = plugin_manager.plugins[plugin_id]
    
    if request.method == 'POST':
        try:
            # Récupérer et mettre à jour les paramètres
            settings = {}
            for key, value in request.form.items():
                if key.startswith('setting_'):
                    setting_name = key[8:]  # Supprimer le préfixe 'setting_'
                    
                    # Convertir le type de valeur si nécessaire
                    if value.lower() == 'true':
                        settings[setting_name] = True
                    elif value.lower() == 'false':
                        settings[setting_name] = False
                    elif value.isdigit():
                        settings[setting_name] = int(value)
                    elif value.replace('.', '', 1).isdigit():
                        settings[setting_name] = float(value)
                    else:
                        settings[setting_name] = value
            
            # Mettre à jour les paramètres du plugin
            plugin.setup(settings)
            
            flash(f"Paramètres du plugin {plugin_id} mis à jour avec succès", "success")
            return redirect(url_for('main.plugins_page'))
            
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour des paramètres du plugin {plugin_id}: {e}")
            flash(f"Erreur lors de la mise à jour des paramètres: {str(e)}", "danger")
    
    # Récupérer les paramètres actuels et par défaut
    current_settings = plugin.get_settings()
    default_settings = plugin.get_default_settings()
    
    return render_template(
        'plugin_settings.html',
        plugin=plugin,
        current_settings=current_settings,
        default_settings=default_settings,
        title=f"Paramètres du plugin {plugin.plugin_name}"
    )

# Route API pour les explications des modèles
@main_blueprint.route('/api/model_explanations', methods=['POST'])
@login_required
def model_explanations_api():
    """API endpoint pour obtenir les explications des modèles d'IA"""
    try:
        # Récupérer les paramètres de la requête
        symbol = request.form.get('symbol', '')
        date_range = request.form.get('date_range', '1m')
        
        # Vérifier si le symbole est valide
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400
        
        # Obtenir les données du marché pour la période spécifiée
        market_data = None
        try:
            if hasattr(data_manager, 'get_historical_data'):
                market_data = data_manager.get_historical_data(symbol, period=date_range)
            elif trading_bot and hasattr(trading_bot, 'get_historical_data'):
                market_data = trading_bot.get_historical_data(symbol, period=date_range)
            else:
                # Fallback à yfinance directement
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                market_data = ticker.history(period=date_range)
                # Convertir les noms de colonnes pour correspondre à notre format standard
                market_data.columns = [col.title() for col in market_data.columns]
        except Exception as e:
            logging.error(f"Error fetching market data: {e}")
            return jsonify({'error': f'Could not fetch market data: {str(e)}'}), 500
        
        if market_data is None or market_data.empty:
            return jsonify({'error': 'No market data available for the specified symbol and time range'}), 404
        
        # Initialiser l'intégrateur d'ensemble
        ensemble_integrator = EnsembleIntegrator(use_explainable_ai=True)
        
        # Générer les prédictions et explications
        try:
            # Vérifier si les modèles existent, sinon utiliser une prédiction fictive pour la démo
            models_exist = False
            try:
                models_exist = ensemble_integrator.load_models(symbol)
            except:
                pass
            
            if models_exist:
                # Utiliser les modèles existants pour générer des prédictions réelles
                prediction_result = ensemble_integrator.predict(market_data, symbol)
                explanation_report = ensemble_integrator.generate_explanation_report(symbol, market_data)
            else:
                # Créer des données de démo pour la visualisation
                prediction_result, explanation_report = generate_demo_explanation(symbol, market_data)
                
            # Formater la réponse
            response = {
                'symbol': symbol,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'recommendation': prediction_result.get('recommendation', 'HOLD'),
                'confidence': prediction_result.get('confidence', 0.65),
                'risk_level': prediction_result.get('risk_level', 'medium'),
                'price': prediction_result.get('price'),
                'direction': prediction_result.get('direction'),
                'volatility': prediction_result.get('volatility'),
                'summary': f"Analysis completed for {symbol} based on {len(market_data)} data points.",
                'combined_explanation': explanation_report.get('combined_explanation'),
                'price_explanation': explanation_report.get('price_explanation'),
                'direction_explanation': explanation_report.get('direction_explanation'),
                'volatility_explanation': explanation_report.get('volatility_explanation'),
                'top_features': explanation_report.get('top_features', {}),
                'plots': explanation_report.get('plots', {})
            }
            
            return jsonify(response)
            
        except Exception as e:
            logging.error(f"Error generating model explanations: {e}")
            return jsonify({'error': f'Could not generate model explanations: {str(e)}'}), 500
            
    except Exception as e:
        logging.error(f"Error in model_explanations_api: {e}")
        return jsonify({'error': str(e)}), 500

def generate_demo_explanation(symbol, market_data):
    """
    Génère des données de démo pour l'explication des modèles lorsque
    les modèles réels ne sont pas disponibles.
    """
    # Calculer quelques indicateurs de base pour le réalisme
    latest_close = market_data['Close'].iloc[-1]
    prev_close = market_data['Close'].iloc[-2] if len(market_data) > 1 else latest_close
    price_change = (latest_close - prev_close) / prev_close
    
    # Décider de la direction basée sur les données réelles
    if price_change > 0:
        direction = 1
        recommendation = "BUY - moderate (0.87%)"
        confidence = 0.72
    else:
        direction = 0
        recommendation = "SELL - weak (-0.34%)"
        confidence = 0.68
    
    # Calculer la volatilité des 10 derniers jours
    volatility = market_data['Close'].pct_change().iloc[-10:].std() if len(market_data) > 10 else 0.015
    risk_level = 'high' if volatility > 0.02 else 'medium' if volatility > 0.01 else 'low'
    
    # Générer des features fictives mais réalistes
    price_features = {
        'SMA_20': 0.35,
        'Volatility_10': 0.28,
        'RSI_14': 0.22,
        'MACD': 0.18,
        'Volume_Change': 0.15,
        'OBV': 0.12,
        'ATR_14': 0.10,
        'BBW_20': 0.09,
        'Momentum_10': 0.08,
        'ROC_5': 0.07
    }
    
    direction_features = {
        'RSI_14': 0.41,
        'SMA_crossover_5_20': 0.33,
        'MACD': 0.25,
        'CMF_20': 0.19,
        'Momentum_10': 0.17,
        'Volume_Change': 0.14,
        'ADX': 0.12,
        'K_14': 0.10,
        'D_14': 0.09,
        'OBV': 0.08
    }
    
    volatility_features = {
        'ATR_14': 0.38,
        'BBW_20': 0.32,
        'Close_std_20': 0.26,
        'Volatility_10': 0.21,
        'Volume_Change': 0.17,
        'ADX': 0.14,
        'High_Low_Range': 0.12,
        'Close_kurt_50': 0.09,
        'Close_skew_50': 0.08,
        'MACD': 0.07
    }
    
    # Créer des explications textuelles
    trend_direction = "bullish" if direction == 1 else "bearish"
    price_explanation = f"Price prediction for {symbol} is influenced most by:\n"
    price_explanation += f"- SMA_20: The 20-day moving average shows a {trend_direction} trend, increasing the predicted price by 0.3542\n"
    price_explanation += f"- Volatility_10: Recent volatility is {'high' if volatility > 0.02 else 'moderate'}, affecting price prediction by 0.2814\n"
    price_explanation += f"- RSI_14: Current RSI is {random.randint(30, 70)}, suggesting {'overbought' if random.randint(0, 1) == 1 else 'neutral'} conditions\n"
    price_explanation += f"- MACD: The MACD line is {'above' if direction == 1 else 'below'} the signal line, indicating {trend_direction} momentum\n"
    price_explanation += f"- Volume: Recent volume is {'increasing' if random.randint(0, 1) == 1 else 'stable'}, supporting the {trend_direction} case"
    
    direction_explanation = f"Direction prediction ({'Up' if direction == 1 else 'Down'}) for {symbol} is influenced most by:\n"
    direction_explanation += f"- RSI_14: With a value of {random.randint(30, 70)}, RSI indicates {'potential reversal' if random.randint(0, 1) == 1 else 'trend continuation'}\n"
    direction_explanation += f"- SMA Crossover: The 5-day SMA is {'above' if direction == 1 else 'below'} the 20-day SMA, a {'bullish' if direction == 1 else 'bearish'} signal\n"
    direction_explanation += f"- MACD: Shows {'positive' if direction == 1 else 'negative'} momentum with the histogram {'increasing' if direction == 1 else 'decreasing'}\n"
    direction_explanation += f"- Money Flow: The Chaikin Money Flow indicates {'accumulation' if direction == 1 else 'distribution'} pattern"
    
    volatility_explanation = f"Volatility prediction for {symbol} is influenced most by:\n"
    volatility_explanation += f"- ATR_14: Average True Range is {volatility:.4f}, indicating {'high' if volatility > 0.02 else 'moderate' if volatility > 0.01 else 'low'} volatility\n"
    volatility_explanation += f"- Bollinger Band Width: Current width is {'expanding' if random.randint(0, 1) == 1 else 'contracting'}, suggesting {'increasing' if random.randint(0, 1) == 1 else 'decreasing'} volatility\n"
    volatility_explanation += f"- Standard Deviation: Price standard deviation over 20 days is {'high' if volatility > 0.02 else 'moderate'}\n"
    volatility_explanation += f"- Volume Changes: {'Irregular' if random.randint(0, 1) == 1 else 'Consistent'} volume patterns suggest {'unpredictable' if random.randint(0, 1) == 1 else 'stable'} price action"
    
    combined_explanation = f"# Trading Decision Explanation for {symbol}\n\n"
    combined_explanation += f"## Decision Summary\n"
    combined_explanation += f"Recommendation: {recommendation}\n"
    combined_explanation += f"Confidence: {confidence * 100:.1f}%\n"
    combined_explanation += f"Risk Level: {risk_level.upper()}\n\n"
    combined_explanation += f"## Price Prediction\n{price_explanation}\n\n"
    combined_explanation += f"## Direction Prediction\n{direction_explanation}\n\n"
    combined_explanation += f"## Volatility Prediction\n{volatility_explanation}\n\n"
    combined_explanation += f"## Most Important Features Overall\n"
    
    # Combiner toutes les features pour le résumé combiné
    combined_features = {}
    for feature, importance in price_features.items():
        combined_features[feature] = importance
    for feature, importance in direction_features.items():
        if feature in combined_features:
            combined_features[feature] += importance
        else:
            combined_features[feature] = importance
    for feature, importance in volatility_features.items():
        if feature in combined_features:
            combined_features[feature] += importance
        else:
            combined_features[feature] = importance
    
    # Trier et prendre les 10 meilleures features
    sorted_features = sorted(combined_features.items(), key=lambda x: x[1], reverse=True)[:10]
    for feature, importance in sorted_features:
        combined_explanation += f"- {feature}: {importance:.4f}\n"
    
    combined_explanation += "\n## Technical Interpretation\n"
    combined_explanation += "This combined analysis integrates price movement predictions with direction probabilities "
    combined_explanation += "and volatility forecasts to provide a holistic view of the expected market behavior. "
    combined_explanation += "The decision algorithm weighs these factors while considering current market conditions "
    combined_explanation += "to generate the most appropriate trading recommendation."
    
    # Construire les résultats
    prediction_result = {
        'price': latest_close * (1 + (0.01 if direction == 1 else -0.005)),
        'direction': direction,
        'volatility': volatility,
        'recommendation': recommendation,
        'confidence': confidence,
        'risk_level': risk_level
    }
    
    explanation_report = {
        'price_explanation': price_explanation,
        'direction_explanation': direction_explanation,
        'volatility_explanation': volatility_explanation,
        'combined_explanation': combined_explanation,
        'top_features': {
            'price': price_features,
            'direction': direction_features,
            'volatility': volatility_features
        },
        'plots': {}  # Pas de plots dans la version de démo
    }
    
    return prediction_result, explanation_report

# Route pour la page d'explications des modèles
@main_blueprint.route('/model_explanations')
@login_required
def model_explanations_page():
    """Page pour visualiser les explications des modèles d'IA"""
    # Récupérer la liste des symboles disponibles
    symbols = []
    try:
        if trading_bot and hasattr(trading_bot, 'get_available_symbols'):
            symbols = trading_bot.get_available_symbols()
        else:
            # Liste de symboles par défaut
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BTC-USD", "ETH-USD"]
    except Exception as e:
        logging.error(f"Error fetching symbols: {e}")
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    return render_template('model_explanations.html', symbols=symbols)