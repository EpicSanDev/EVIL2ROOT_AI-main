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