import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os
import sys
import json

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core import (
    BacktestEnvironment,
    SentimentBasedStrategy,
    TechnicalStrategy,
    RLBasedStrategy,
    HybridStrategy,
    plot_backtest_results,
    plot_monthly_heatmap,
    plot_trade_analysis,
    generate_performance_report
)
from src.models.sentiment import MarketSentimentAnalyzer
from src.services.market_data import BinanceConnector

# Configurer le logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_api_keys():
    """Charge les clés API depuis .env ou un fichier de configuration."""
    api_keys = {}
    
    try:
        # Essayez d'abord le fichier .env
        from dotenv import load_dotenv
        load_dotenv()
        
        # Clés pour les sources de données de sentiment
        api_keys['newsapi'] = os.getenv('NEWSAPI_KEY')
        api_keys['finnhub'] = os.getenv('FINNHUB_KEY')
        
        # Clés pour les connexions de marché
        api_keys['binance_api_key'] = os.getenv('BINANCE_API_KEY')
        api_keys['binance_api_secret'] = os.getenv('BINANCE_API_SECRET')
        
    except ImportError:
        logger.warning("dotenv non installé, vérification des variables d'environnement directes")
        
        # Clés pour les sources de données de sentiment
        api_keys['newsapi'] = os.environ.get('NEWSAPI_KEY')
        api_keys['finnhub'] = os.environ.get('FINNHUB_KEY')
        
        # Clés pour les connexions de marché
        api_keys['binance_api_key'] = os.environ.get('BINANCE_API_KEY')
        api_keys['binance_api_secret'] = os.environ.get('BINANCE_API_SECRET')
    
    # Supprimer les clés None
    api_keys = {k: v for k, v in api_keys.items() if v is not None}
    
    return api_keys

def fetch_historical_data(symbol: str = 'BTCUSDT', start_date=None, end_date=None, 
                        timeframes=None, use_saved=True, save=True):
    """
    Récupère les données historiques pour un symbole.
    
    Args:
        symbol: Symbole à récupérer
        start_date: Date de début
        end_date: Date de fin
        timeframes: Liste des timeframes à récupérer
        use_saved: Si True, utilise les données sauvegardées si disponibles
        save: Si True, sauvegarde les données récupérées
    
    Returns:
        Dict: Données par timeframe
    """
    if timeframes is None:
        timeframes = ['1h', '4h', '1d']
    
    # Définir les dates par défaut
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=365)  # 1 an de données par défaut
    
    # Vérifier si des données sauvegardées existent
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/market_data')
    os.makedirs(data_dir, exist_ok=True)
    
    data = {}
    
    try:
        # Essayer de charger des données sauvegardées
        if use_saved:
            for tf in timeframes:
                filename = f"{symbol}_{tf}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                filepath = os.path.join(data_dir, filename)
                
                if os.path.exists(filepath):
                    logger.info(f"Chargement des données depuis {filepath}")
                    df = pd.read_csv(filepath)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    data[tf] = df
        
        # Si des données manquent, les récupérer de l'API
        missing_timeframes = [tf for tf in timeframes if tf not in data]
        
        if missing_timeframes:
            logger.info(f"Récupération des données pour {symbol}, timeframes: {missing_timeframes}")
            
            api_keys = load_api_keys()
            connector = BinanceConnector(
                api_key=api_keys.get('binance_api_key'),
                api_secret=api_keys.get('binance_api_secret')
            )
            connector.connect()
            
            for tf in missing_timeframes:
                df = connector.get_historical_klines(
                    symbol=symbol,
                    interval=tf,
                    start_time=start_date,
                    end_time=end_date
                )
                
                data[tf] = df
                
                if save:
                    filename = f"{symbol}_{tf}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                    filepath = os.path.join(data_dir, filename)
                    df.to_csv(filepath)
                    logger.info(f"Données sauvegardées dans {filepath}")
            
            connector.disconnect()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données: {str(e)}")
        
        # Si pas de données, utiliser des données de test
        if not data:
            logger.info("Utilisation de données de test")
            data = generate_test_data(timeframes, start_date, end_date)
    
    return data

def generate_test_data(timeframes, start_date, end_date):
    """Génère des données de test en cas d'échec de la récupération."""
    data = {}
    
    for tf in timeframes:
        # Déterminer le nombre de périodes en fonction du timeframe
        if tf == '1h':
            freq = 'H'
        elif tf == '4h':
            freq = '4H'
        elif tf == '1d':
            freq = 'D'
        else:
            freq = 'D'
        
        # Générer des dates
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Générer des prix avec un processus de marche aléatoire
        np.random.seed(42)  # Pour la reproductibilité
        
        # Prix de départ
        price = 10000.0
        n_steps = len(dates)
        
        # Paramètres du processus
        mu = 0.0001  # Dérive
        sigma = 0.02  # Volatilité
        
        # Générer les rendements
        returns = np.random.normal(mu, sigma, n_steps)
        
        # Calculer les prix
        prices = price * np.cumprod(1 + returns)
        
        # Générer OHLCV
        high = prices * np.random.uniform(1.001, 1.02, n_steps)
        low = prices * np.random.uniform(0.98, 0.999, n_steps)
        close = prices
        open_prices = np.roll(close, 1)
        open_prices[0] = close[0] * np.random.uniform(0.99, 1.01)
        volume = np.random.uniform(100, 1000, n_steps) * prices
        
        # Créer le DataFrame
        df = pd.DataFrame({
            'Open': open_prices,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)
        
        data[tf] = df
    
    return data

def run_strategy_comparison(data):
    """
    Exécute un backtest pour comparer différentes stratégies.
    
    Args:
        data: Données par timeframe
    """
    # Créer l'environnement de backtesting
    backtest_env = BacktestEnvironment(
        data=data,
        initial_capital=100000,
        transaction_fee=0.001,
        slippage=0.0005,
        timeframes=list(data.keys())
    )
    
    # Charger les clés API
    api_keys = load_api_keys()
    
    # Créer les différentes stratégies
    strategies = {
        'technical': TechnicalStrategy(),
        'sentiment': SentimentBasedStrategy(api_keys=api_keys),
        'hybrid': HybridStrategy(
            tech_weight=0.5,
            sentiment_weight=0.5,
            rl_weight=0.0,  # Pas de RL pour cet exemple simple
            api_keys=api_keys
        )
    }
    
    # Exécuter le backtest pour chaque stratégie
    results = {}
    
    for name, strategy in strategies.items():
        logger.info(f"Exécution du backtest pour la stratégie: {name}")
        
        # Réinitialiser l'environnement
        backtest_env.reset()
        
        # Exécuter le backtest
        report = backtest_env.run_backtest(strategy)
        
        # Stocker les résultats
        results[name] = report
        
        # Générer les visualisations
        output_dir = os.path.join(os.path.dirname(__file__), '../../results/backtest')
        os.makedirs(output_dir, exist_ok=True)
        
        # Graphique des résultats
        plot_path = os.path.join(output_dir, f"{name}_results.png")
        backtest_env.plot_results(save_path=plot_path, plot_trades=True)
        
        # Analyse des transactions
        trades_path = os.path.join(output_dir, f"{name}_trades.png")
        plot_trade_analysis(backtest_env.trades_history, save_path=trades_path)
        
        # Heatmap des rendements mensuels
        main_tf = list(data.keys())[0]
        dates = data[main_tf].index[:len(backtest_env.equity_curve)]
        heatmap_path = os.path.join(output_dir, f"{name}_monthly.png")
        plot_monthly_heatmap(backtest_env.equity_curve, dates, save_path=heatmap_path)
        
        # Rapport HTML
        report_path = os.path.join(output_dir, f"{name}_report.html")
        generate_performance_report(
            metrics=report['metrics'],
            equity_curve=backtest_env.equity_curve,
            trades=backtest_env.trades_history,
            output_path=report_path
        )
    
    # Comparer les performances
    compare_strategies(results, output_dir=output_dir)

def compare_strategies(results, output_dir):
    """
    Compare les performances des différentes stratégies.
    
    Args:
        results: Résultats par stratégie
        output_dir: Répertoire de sortie
    """
    # Extraire les métriques clés
    comparison = {}
    for name, report in results.items():
        metrics = report['metrics']
        comparison[name] = {
            'total_return_pct': metrics['total_return_pct'],
            'annualized_return': metrics['annualized_return'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'volatility': metrics['volatility'],
            'total_trades': metrics['total_trades']
        }
    
    # Créer un DataFrame pour la comparaison
    df_comparison = pd.DataFrame(comparison).T
    
    # Sauvegarder les résultats
    comparison_path = os.path.join(output_dir, 'strategy_comparison.csv')
    df_comparison.to_csv(comparison_path)
    
    # Créer un graphique de comparaison
    plt.figure(figsize=(12, 8))
    
    # Barplot des rendements
    plt.subplot(2, 2, 1)
    df_comparison['total_return_pct'].plot(kind='bar', color='skyblue')
    plt.title('Rendement Total (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Barplot des ratios de Sharpe
    plt.subplot(2, 2, 2)
    df_comparison['sharpe_ratio'].plot(kind='bar', color='green')
    plt.title('Ratio de Sharpe')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Barplot des drawdowns
    plt.subplot(2, 2, 3)
    df_comparison['max_drawdown_pct'].plot(kind='bar', color='red')
    plt.title('Drawdown Maximum (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Barplot des volatilités
    plt.subplot(2, 2, 4)
    df_comparison['volatility'].plot(kind='bar', color='orange')
    plt.title('Volatilité (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    comparison_plot_path = os.path.join(output_dir, 'strategy_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Comparaison des stratégies sauvegardée dans {comparison_path} et {comparison_plot_path}")
    
    # Afficher un résumé
    print("\nComparaison des stratégies:")
    print(df_comparison)
    print("\nMeilleure stratégie par métrique:")
    print(f"Rendement total: {df_comparison['total_return_pct'].idxmax()} ({df_comparison['total_return_pct'].max():.2f}%)")
    print(f"Ratio de Sharpe: {df_comparison['sharpe_ratio'].idxmax()} ({df_comparison['sharpe_ratio'].max():.2f})")
    print(f"Drawdown minimal: {df_comparison['max_drawdown_pct'].idxmin()} ({df_comparison['max_drawdown_pct'].min():.2f}%)")

if __name__ == "__main__":
    # Récupérer les données historiques
    symbol = 'BTCUSDT'
    timeframes = ['1h', '4h', '1d']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    logger.info(f"Récupération des données historiques pour {symbol} du {start_date} au {end_date}")
    data = fetch_historical_data(symbol, start_date, end_date, timeframes)
    
    # Exécuter la comparaison des stratégies
    run_strategy_comparison(data) 