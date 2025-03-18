import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime
import os
import math

def plot_backtest_results(equity_curve: List[float], positions: List[float], 
                         trades: List[Dict], data: pd.DataFrame, metrics: Dict,
                         save_path: Optional[str] = None, plot_trades: bool = True):
    """
    Génère une visualisation complète des résultats du backtest.
    
    Args:
        equity_curve: Liste des valeurs de portefeuille
        positions: Liste des positions à chaque étape
        trades: Liste des transactions effectuées
        data: DataFrame des données de prix
        metrics: Métriques de performance calculées
        save_path: Chemin pour sauvegarder les graphiques (None pour afficher)
        plot_trades: Si True, affiche les transactions sur le graphique
    """
    # Configurer le style des graphiques
    plt.style.use('seaborn-darkgrid')
    sns.set(font_scale=1.2)
    
    # Créer une figure avec plusieurs sous-graphiques
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Graphique de la courbe d'équité
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)
    ax1.set_title('Performance du Portefeuille', fontsize=14, fontweight='bold')
    
    # Créer un DataFrame avec les dates
    df_equity = pd.DataFrame({
        'date': data.index[:len(equity_curve)],
        'equity': equity_curve,
        'position': positions if len(positions) == len(equity_curve) else positions[:len(equity_curve)]
    })
    
    # Tracer la courbe d'équité
    ax1.plot(df_equity['date'], df_equity['equity'], linewidth=2, label='Équité')
    
    # Ajouter une ligne horizontale pour le capital initial
    initial_capital = equity_curve[0]
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label='Capital Initial')
    
    # Formater l'axe des dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Valeur du Portefeuille')
    
    # Ajouter des annotations pour les rendements et drawdowns
    total_return = metrics['total_return_pct']
    max_drawdown = metrics['max_drawdown_pct']
    sharpe = metrics['sharpe_ratio']
    
    text_info = (
        f"Rendement total: {total_return:.2f}%\n"
        f"Drawdown max: {max_drawdown:.2f}%\n"
        f"Ratio de Sharpe: {sharpe:.2f}"
    )
    
    ax1.annotate(text_info, xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=12, backgroundcolor='white', alpha=0.8)
    
    # Ajouter les transactions si demandé
    if plot_trades and trades:
        # Convertir les transactions en DataFrame
        df_trades = pd.DataFrame(trades)
        df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
        
        # Tracer les points d'achat et de vente
        buys = df_trades[df_trades['type'] == 'BUY']
        sells = df_trades[df_trades['type'] == 'SELL']
        
        if not buys.empty:
            ax1.scatter(buys['timestamp'], [data.loc[date, 'Close'] for date in buys['timestamp']],
                       marker='^', color='green', s=100, alpha=0.7, label='Achat')
        
        if not sells.empty:
            ax1.scatter(sells['timestamp'], [data.loc[date, 'Close'] for date in sells['timestamp']],
                       marker='v', color='red', s=100, alpha=0.7, label='Vente')
    
    ax1.legend(loc='best')
    
    # 2. Graphique des drawdowns
    ax2 = plt.subplot2grid((4, 4), (2, 0), colspan=4)
    ax2.set_title('Drawdowns', fontsize=14, fontweight='bold')
    
    # Calculer les drawdowns
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100  # En pourcentage
    
    # Tracer les drawdowns
    ax2.fill_between(df_equity['date'], 0, drawdown, color='red', alpha=0.3)
    ax2.plot(df_equity['date'], drawdown, color='red', alpha=0.5)
    
    # Mettre en évidence le drawdown maximal
    max_dd_idx = np.argmax(drawdown)
    ax2.annotate(f"Max Drawdown: {drawdown[max_dd_idx]:.2f}%",
                xy=(df_equity['date'].iloc[max_dd_idx], drawdown[max_dd_idx]),
                xytext=(df_equity['date'].iloc[max_dd_idx], drawdown[max_dd_idx] + 5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_ylim(bottom=0)
    ax2.invert_yaxis()  # Inverser l'axe pour que les drawdowns pointent vers le bas
    
    # 3. Graphique des positions
    ax3 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
    ax3.set_title('Positions', fontsize=14, fontweight='bold')
    
    # Tracer les positions
    positions_array = np.array(positions[:len(df_equity)])
    ax3.fill_between(df_equity['date'], 0, positions_array, color='blue', alpha=0.3)
    ax3.plot(df_equity['date'], positions_array, color='blue', alpha=0.7)
    
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Position')
    ax3.set_ylim(-1.1, 1.1)  # Les positions sont entre -1 et 1
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # 4. Graphique de distribution des rendements
    ax4 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    ax4.set_title('Distribution des Rendements', fontsize=14, fontweight='bold')
    
    # Calculer les rendements quotidiens
    returns = np.diff(equity) / equity[:-1]
    
    # Tracer l'histogramme des rendements
    sns.histplot(returns * 100, bins=50, kde=True, ax=ax4)
    
    # Ajouter des statistiques
    skewness = pd.Series(returns).skew()
    kurtosis = pd.Series(returns).kurtosis()
    
    text_info = (
        f"Moyenne: {np.mean(returns) * 100:.2f}%\n"
        f"Écart-type: {np.std(returns) * 100:.2f}%\n"
        f"Asymétrie: {skewness:.2f}\n"
        f"Kurtosis: {kurtosis:.2f}"
    )
    
    ax4.annotate(text_info, xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, backgroundcolor='white', alpha=0.8)
    
    ax4.set_xlabel('Rendement quotidien (%)')
    ax4.set_ylabel('Fréquence')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder ou afficher
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_monthly_heatmap(equity_curve: List[float], dates: List[datetime],
                        save_path: Optional[str] = None):
    """
    Génère une heatmap des rendements mensuels.
    
    Args:
        equity_curve: Liste des valeurs de portefeuille
        dates: Liste des dates correspondantes
        save_path: Chemin pour sauvegarder le graphique (None pour afficher)
    """
    from .backtest_performance import calculate_monthly_returns
    
    # Calculer les rendements mensuels
    monthly_returns = calculate_monthly_returns(equity_curve, dates)
    
    if monthly_returns.empty:
        print("Données insuffisantes pour générer la heatmap mensuelle")
        return
    
    # Préparer les données pour la heatmap
    pivot_table = monthly_returns.pivot_table(
        index='year', columns='month', values='return'
    )
    
    # Configurer le style
    plt.figure(figsize=(12, 8))
    
    # Créer la heatmap
    ax = sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".2f",
        cmap=sns.diverging_palette(10, 220, as_cmap=True),
        center=0,
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": "Rendement mensuel (%)"}
    )
    
    # Configurer les étiquettes
    month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
    ax.set_xticklabels(month_names)
    
    # Titre et mise en page
    plt.title('Rendements Mensuels (%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder ou afficher
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_trade_analysis(trades: List[Dict], save_path: Optional[str] = None):
    """
    Génère des graphiques d'analyse des transactions.
    
    Args:
        trades: Liste des transactions effectuées
        save_path: Chemin pour sauvegarder le graphique (None pour afficher)
    """
    from .backtest_performance import analyze_trades
    
    # Analyser les transactions
    trade_stats = analyze_trades(trades)
    
    if trade_stats['total_trades'] == 0:
        print("Aucune transaction à analyser")
        return
    
    # Grouper les transactions par paires (entrée/sortie)
    trade_pairs = []
    open_positions = {}
    
    for trade in sorted(trades, key=lambda x: x['timestamp']):
        trade_type = trade['type']
        price = trade['price']
        size = trade['size']
        timestamp = trade['timestamp']
        
        if trade_type == 'BUY':
            # Ouvrir une nouvelle position longue
            position_id = len(open_positions)
            open_positions[position_id] = {
                'entry_time': timestamp,
                'entry_price': price,
                'size': size,
                'direction': 'LONG'
            }
        elif trade_type == 'SELL':
            # Fermer une position longue existante
            if open_positions:
                # Prendre la plus ancienne position ouverte
                position_id = min(open_positions.keys())
                position = open_positions.pop(position_id)
                
                # Calculer le P&L
                if position['direction'] == 'LONG':
                    profit = (price - position['entry_price']) * size
                else:
                    profit = (position['entry_price'] - price) * size
                
                # Enregistrer la paire complète
                trade_pairs.append({
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': price,
                    'size': size,
                    'profit': profit,
                    'profit_pct': (price / position['entry_price'] - 1) * 100 if position['direction'] == 'LONG' else (position['entry_price'] / price - 1) * 100,
                    'holding_period': (timestamp - position['entry_time']).total_seconds() / (60 * 60 * 24)  # En jours
                })
    
    # Configurer le style
    plt.style.use('seaborn-darkgrid')
    sns.set(font_scale=1.2)
    
    # Créer une figure avec plusieurs sous-graphiques
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Distribution des profits par transaction
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax1.set_title('Distribution des Profits par Transaction', fontsize=14, fontweight='bold')
    
    profits = [pair['profit'] for pair in trade_pairs]
    sns.histplot(profits, bins=30, kde=True, ax=ax1)
    
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Profit ($)')
    ax1.set_ylabel('Fréquence')
    
    # 2. Distribution des rendements en pourcentage par transaction
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    ax2.set_title('Distribution des Rendements (%)', fontsize=14, fontweight='bold')
    
    profit_pcts = [pair['profit_pct'] for pair in trade_pairs]
    sns.histplot(profit_pcts, bins=30, kde=True, ax=ax2)
    
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Rendement (%)')
    ax2.set_ylabel('Fréquence')
    
    # 3. Profit cumulatif au fil du temps
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3)
    ax3.set_title('Profit Cumulatif au Fil du Temps', fontsize=14, fontweight='bold')
    
    # Calculer le profit cumulatif
    cumulative_profit = np.cumsum(profits)
    
    # Créer un DataFrame pour le tracé
    df_cum_profit = pd.DataFrame({
        'trade': range(1, len(cumulative_profit) + 1),
        'cumulative_profit': cumulative_profit
    })
    
    # Tracer le profit cumulatif
    ax3.plot(df_cum_profit['trade'], df_cum_profit['cumulative_profit'], linewidth=2)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Numéro de Transaction')
    ax3.set_ylabel('Profit Cumulatif ($)')
    
    # 4. Durée de détention vs Profit
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax4.set_title('Durée de Détention vs Profit', fontsize=14, fontweight='bold')
    
    # Créer un DataFrame pour le nuage de points
    df_holding = pd.DataFrame({
        'holding_period': [pair['holding_period'] for pair in trade_pairs],
        'profit': profits
    })
    
    # Tracer le nuage de points
    sns.scatterplot(x='holding_period', y='profit', data=df_holding, ax=ax4)
    
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Durée de Détention (jours)')
    ax4.set_ylabel('Profit ($)')
    
    # 5. Statistiques des transactions
    ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=2)
    ax5.set_title('Statistiques des Transactions', fontsize=14, fontweight='bold')
    ax5.axis('off')
    
    # Créer un tableau de statistiques
    stats_text = (
        f"Nombre total de transactions: {trade_stats['total_trades']}\n"
        f"Taux de réussite: {trade_stats['win_rate']:.2f}%\n"
        f"Profit moyen par transaction: ${trade_stats['avg_profit_per_trade']:.2f}\n"
        f"Profit moyen des transactions gagnantes: ${trade_stats['avg_profit_winning']:.2f}\n"
        f"Perte moyenne des transactions perdantes: ${trade_stats['avg_loss_losing']:.2f}\n"
        f"Plus grand profit: ${trade_stats['largest_profit']:.2f}\n"
        f"Plus grande perte: ${trade_stats['largest_loss']:.2f}\n"
        f"Durée moyenne de détention: {trade_stats['avg_holding_period']:.2f} jours\n"
        f"Ratio profit/perte: {trade_stats['profit_loss_ratio']:.2f}\n"
        f"Facteur de profit: {trade_stats['profit_factor']:.2f}\n"
        f"Critère de Kelly: {trade_stats['kelly_criterion']:.2f}"
    )
    
    ax5.text(0.1, 0.5, stats_text, fontsize=12, va='center')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder ou afficher
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def generate_performance_report(metrics: Dict, equity_curve: List[float], 
                              trades: List[Dict], output_path: str):
    """
    Génère un rapport de performance complet au format HTML.
    
    Args:
        metrics: Métriques de performance calculées
        equity_curve: Liste des valeurs de portefeuille
        trades: Liste des transactions effectuées
        output_path: Chemin où sauvegarder le rapport HTML
    """
    import jinja2
    
    # Calculer des métriques supplémentaires
    from .backtest_performance import analyze_trades
    trade_analysis = analyze_trades(trades)
    
    # Créer le contenu du rapport
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rapport de Performance du Backtest</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #2c3e50; }
            .container { max-width: 1200px; margin: 0 auto; }
            .section { margin-bottom: 30px; }
            .metric-group { display: flex; flex-wrap: wrap; margin-bottom: 20px; }
            .metric { width: 250px; margin: 10px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            .metric h3 { margin-top: 0; color: #3498db; }
            .metric p { font-size: 18px; font-weight: bold; margin: 5px 0; }
            .metric small { color: #7f8c8d; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:hover { background-color: #f5f5f5; }
            .positive { color: green; }
            .negative { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="section">
                <h1>Rapport de Performance du Backtest</h1>
                <p>Date du rapport: {{ date_generated }}</p>
            </div>
            
            <div class="section">
                <h2>Métriques de Performance</h2>
                
                <div class="metric-group">
                    <div class="metric">
                        <h3>Rendement Total</h3>
                        <p class="{{ 'positive' if metrics.total_return_pct >= 0 else 'negative' }}">{{ "%.2f"|format(metrics.total_return_pct) }}%</p>
                        <small>{{ "%.2f"|format(metrics.total_return) }} unités</small>
                    </div>
                    
                    <div class="metric">
                        <h3>Rendement Annualisé</h3>
                        <p class="{{ 'positive' if metrics.annualized_return >= 0 else 'negative' }}">{{ "%.2f"|format(metrics.annualized_return) }}%</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Drawdown Maximum</h3>
                        <p class="negative">{{ "%.2f"|format(metrics.max_drawdown_pct) }}%</p>
                        <small>{{ "%.2f"|format(metrics.max_drawdown) }} unités</small>
                    </div>
                    
                    <div class="metric">
                        <h3>Volatilité</h3>
                        <p>{{ "%.2f"|format(metrics.volatility) }}%</p>
                    </div>
                </div>
                
                <div class="metric-group">
                    <div class="metric">
                        <h3>Ratio de Sharpe</h3>
                        <p>{{ "%.2f"|format(metrics.sharpe_ratio) }}</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Ratio de Sortino</h3>
                        <p>{{ "%.2f"|format(metrics.sortino_ratio) }}</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Ratio de Calmar</h3>
                        <p>{{ "%.2f"|format(metrics.calmar_ratio) }}</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Ratio Rendement/Drawdown</h3>
                        <p>{{ "%.2f"|format(metrics.return_drawdown_ratio) }}</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Analyse des Transactions</h2>
                
                <div class="metric-group">
                    <div class="metric">
                        <h3>Nombre de Transactions</h3>
                        <p>{{ trade_analysis.total_trades }}</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Taux de Réussite</h3>
                        <p>{{ "%.2f"|format(trade_analysis.win_rate) }}%</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Profit Moyen par Transaction</h3>
                        <p class="{{ 'positive' if trade_analysis.avg_profit_per_trade >= 0 else 'negative' }}">{{ "%.2f"|format(trade_analysis.avg_profit_per_trade) }}</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Facteur de Profit</h3>
                        <p>{{ "%.2f"|format(trade_analysis.profit_factor) }}</p>
                    </div>
                </div>
                
                <div class="metric-group">
                    <div class="metric">
                        <h3>Profit Moyen (Gagnants)</h3>
                        <p class="positive">{{ "%.2f"|format(trade_analysis.avg_profit_winning) }}</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Perte Moyenne (Perdants)</h3>
                        <p class="negative">{{ "%.2f"|format(trade_analysis.avg_loss_losing) }}</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Plus Grand Profit</h3>
                        <p class="positive">{{ "%.2f"|format(trade_analysis.largest_profit) }}</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Plus Grande Perte</h3>
                        <p class="negative">{{ "%.2f"|format(trade_analysis.largest_loss) }}</p>
                    </div>
                </div>
                
                <h3>10 Dernières Transactions</h3>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Type</th>
                        <th>Prix</th>
                        <th>Taille</th>
                        <th>Valeur</th>
                        <th>Frais</th>
                    </tr>
                    {% for trade in recent_trades %}
                    <tr>
                        <td>{{ trade.timestamp }}</td>
                        <td>{{ trade.type }}</td>
                        <td>{{ "%.4f"|format(trade.price) }}</td>
                        <td>{{ "%.4f"|format(trade.size) }}</td>
                        <td>{{ "%.2f"|format(trade.value) }}</td>
                        <td>{{ "%.2f"|format(trade.fee) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="section">
                <h2>Statistiques de Risque</h2>
                
                <div class="metric-group">
                    <div class="metric">
                        <h3>Value at Risk (95%)</h3>
                        <p>{{ "%.2f"|format(metrics.var_95) }}</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Expected Shortfall (95%)</h3>
                        <p>{{ "%.2f"|format(metrics.cvar_95) }}</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Ulcer Index</h3>
                        <p>{{ "%.4f"|format(metrics.ulcer_index) }}</p>
                    </div>
                    
                    <div class="metric">
                        <h3>Exposition Maximale</h3>
                        <p>{{ "%.2f"|format(metrics.max_exposure) }}</p>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Préparer les données pour le modèle
    template_data = {
        'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics,
        'trade_analysis': trade_analysis,
        'recent_trades': sorted(trades, key=lambda x: x['timestamp'], reverse=True)[:10]
    }
    
    # Générer le HTML
    template = jinja2.Template(template_str)
    html_content = template.render(**template_data)
    
    # Sauvegarder le fichier
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Rapport de performance sauvegardé à {output_path}")

def plot_regime_performance(equity_curves: Dict[str, List[float]], 
                          regimes: List[str], save_path: Optional[str] = None):
    """
    Compare la performance de stratégies dans différents régimes de marché.
    
    Args:
        equity_curves: Dictionnaire de courbes d'équité par régime
        regimes: Liste des régimes de marché
        save_path: Chemin pour sauvegarder le graphique (None pour afficher)
    """
    # Configurer le style
    plt.figure(figsize=(14, 8))
    
    # Tracer les courbes d'équité normalisées
    for regime, equity in equity_curves.items():
        if equity:
            # Normaliser à 100
            normalized_equity = np.array(equity) / equity[0] * 100
            plt.plot(normalized_equity, label=f"Régime: {regime}", linewidth=2)
    
    plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Capital Initial')
    
    # Embellir le graphique
    plt.title('Performance par Régime de Marché', fontsize=14, fontweight='bold')
    plt.xlabel('Périodes')
    plt.ylabel('Capital normalisé (base 100)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Sauvegarder ou afficher
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 