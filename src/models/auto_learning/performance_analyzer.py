import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import os
from scipy import stats

from .trade_journal import TradeJournal

class PerformanceAnalyzer:
    """
    Analyse les performances des transactions de trading pour identifier les modèles,
    les tendances et les domaines d'amélioration.
    """
    
    def __init__(self, trade_journal: TradeJournal):
        """
        Initialise l'analyseur de performance avec un journal de trading.
        
        Args:
            trade_journal: Instance de TradeJournal pour accéder aux données de transaction
        """
        self.trade_journal = trade_journal
        self.logger = logging.getLogger(__name__)
        
        # Créer le répertoire pour les graphiques s'il n'existe pas
        self.reports_dir = "data/performance_reports"
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
    
    def analyze_recent_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyse les performances des transactions récentes.
        
        Args:
            days: Nombre de jours à analyser
            
        Returns:
            Dictionnaire contenant les métriques d'analyse
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Récupérer les transactions de la période
        trades_df = self.trade_journal.get_trades(start_date=start_date, end_date=end_date)
        
        if trades_df.empty:
            self.logger.warning(f"Aucune transaction trouvée dans les {days} derniers jours")
            return {"error": "Aucune donnée disponible pour l'analyse"}
        
        # Calcul des métriques de base
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Analyse des performances par jour de la semaine
        trades_df['entry_day'] = pd.to_datetime(trades_df['entry_time']).dt.day_name()
        day_performance = trades_df.groupby('entry_day')['pnl'].agg(['sum', 'count', 'mean'])
        day_performance['win_rate'] = trades_df[trades_df['pnl'] > 0].groupby('entry_day').size() / day_performance['count']
        
        # Analyse des performances par symbole
        symbol_performance = trades_df.groupby('symbol')['pnl'].agg(['sum', 'count', 'mean'])
        symbol_performance['win_rate'] = trades_df[trades_df['pnl'] > 0].groupby('symbol').size() / symbol_performance['count']
        
        # Analyse des performances par stratégie
        strategy_performance = None
        if 'strategy_name' in trades_df.columns:
            strategy_performance = trades_df.groupby('strategy_name')['pnl'].agg(['sum', 'count', 'mean'])
            strategy_performance['win_rate'] = trades_df[trades_df['pnl'] > 0].groupby('strategy_name').size() / strategy_performance['count']
        
        # Calculer les statistiques sur les durées des transactions
        duration_stats = None
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            trades_df['duration'] = pd.to_datetime(trades_df['exit_time']) - pd.to_datetime(trades_df['entry_time'])
            trades_df['duration_hours'] = trades_df['duration'].dt.total_seconds() / 3600
            
            duration_stats = {
                'mean_duration': trades_df['duration_hours'].mean(),
                'median_duration': trades_df['duration_hours'].median(),
                'min_duration': trades_df['duration_hours'].min(),
                'max_duration': trades_df['duration_hours'].max(),
            }
            
            # Comparer les durées des transactions gagnantes vs perdantes
            winning_duration = trades_df[trades_df['pnl'] > 0]['duration_hours'].mean()
            losing_duration = trades_df[trades_df['pnl'] <= 0]['duration_hours'].mean()
            
            duration_stats['winning_avg_duration'] = winning_duration
            duration_stats['losing_avg_duration'] = losing_duration
            duration_stats['duration_ratio'] = winning_duration / losing_duration if losing_duration > 0 else float('inf')
        
        return {
            'period': f"{start_date.date()} - {end_date.date()}",
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': trades_df['pnl'].sum(),
            'avg_pnl': trades_df['pnl'].mean(),
            'max_profit': trades_df['pnl'].max(),
            'max_loss': trades_df['pnl'].min(),
            'day_performance': day_performance.to_dict() if not day_performance.empty else None,
            'symbol_performance': symbol_performance.to_dict() if not symbol_performance.empty else None,
            'strategy_performance': strategy_performance.to_dict() if strategy_performance is not None and not strategy_performance.empty else None,
            'duration_stats': duration_stats
        }
    
    def identify_market_condition_patterns(self) -> Dict[str, Any]:
        """
        Identifie les patterns de performance selon différentes conditions de marché.
        
        Returns:
            Dictionnaire contenant les patterns identifiés
        """
        # Récupérer toutes les transactions avec des conditions de marché
        trades_df = self.trade_journal.get_trades()
        
        if trades_df.empty or 'market_conditions' not in trades_df.columns:
            return {"error": "Données insuffisantes pour l'analyse des conditions de marché"}
        
        # Extraire et normaliser les conditions de marché
        market_conditions = []
        for _, row in trades_df.iterrows():
            if isinstance(row['market_conditions'], dict):
                market_condition = row['market_conditions']
                market_condition['trade_id'] = row['trade_id'] 
                market_condition['pnl'] = row['pnl']
                market_conditions.append(market_condition)
        
        if not market_conditions:
            return {"error": "Aucune condition de marché exploitable trouvée"}
        
        # Convertir en DataFrame pour l'analyse
        market_df = pd.DataFrame(market_conditions)
        
        # Identifier les conditions de marché qui ont tendance à être plus profitables
        condition_patterns = {}
        
        # Analyser les patterns pour chaque condition de marché
        for col in market_df.columns:
            if col not in ['trade_id', 'pnl'] and market_df[col].nunique() > 1:
                try:
                    # Grouper par condition et calculer les statistiques
                    grouped = market_df.groupby(col)['pnl'].agg(['mean', 'count', 'sum'])
                    grouped['success_rate'] = market_df[market_df['pnl'] > 0].groupby(col).size() / grouped['count']
                    
                    condition_patterns[col] = grouped.to_dict()
                    
                    # Si la condition est numérique, calculer la corrélation
                    if pd.api.types.is_numeric_dtype(market_df[col]):
                        correlation = market_df[col].corr(market_df['pnl'])
                        condition_patterns[f"{col}_correlation"] = correlation
                except Exception as e:
                    self.logger.error(f"Erreur lors de l'analyse de {col}: {e}")
        
        return {
            'condition_patterns': condition_patterns
        }
    
    def detect_trade_clusters(self, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Utilise des techniques de clustering pour regrouper les transactions similaires
        et identifier les caractéristiques communes.
        
        Args:
            n_clusters: Nombre de clusters à créer
            
        Returns:
            Dictionnaire contenant les résultats du clustering
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            self.logger.error("sklearn non disponible pour le clustering")
            return {"error": "sklearn non disponible pour le clustering"}
        
        # Récupérer les transactions
        trades_df = self.trade_journal.get_trades()
        
        if trades_df.empty:
            return {"error": "Aucune donnée disponible pour le clustering"}
        
        # Sélectionner et préparer les caractéristiques pour le clustering
        numeric_columns = []
        for col in trades_df.columns:
            if pd.api.types.is_numeric_dtype(trades_df[col]) and col not in ['id', 'trade_id']:
                numeric_columns.append(col)
        
        if len(numeric_columns) < 2:
            return {"error": "Données numériques insuffisantes pour le clustering"}
        
        # Préparation des données
        X = trades_df[numeric_columns].copy()
        X = X.fillna(0)  # Remplacer les valeurs manquantes
        
        # Standardisation des données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Ajuster le nombre de clusters au nombre de données
        n_clusters = min(n_clusters, len(X_scaled) - 1) if len(X_scaled) > 1 else 1
        
        if n_clusters < 1:
            return {"error": "Données insuffisantes pour le clustering"}
        
        # Appliquer K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        trades_df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyser les clusters
        cluster_analysis = {}
        for cluster in range(n_clusters):
            cluster_trades = trades_df[trades_df['cluster'] == cluster]
            
            cluster_analysis[f"cluster_{cluster}"] = {
                'count': len(cluster_trades),
                'avg_pnl': cluster_trades['pnl'].mean(),
                'win_rate': len(cluster_trades[cluster_trades['pnl'] > 0]) / len(cluster_trades),
                'common_features': {}
            }
            
            # Identifier les caractéristiques communes dans ce cluster
            for col in trades_df.columns:
                if col not in ['id', 'trade_id', 'cluster', 'pnl']:
                    try:
                        if pd.api.types.is_numeric_dtype(trades_df[col]):
                            cluster_analysis[f"cluster_{cluster}"]['common_features'][col] = cluster_trades[col].mean()
                        else:
                            # Pour les colonnes catégorielles, trouver la valeur la plus fréquente
                            if not cluster_trades[col].isnull().all():
                                most_common = cluster_trades[col].mode()[0]
                                cluster_analysis[f"cluster_{cluster}"]['common_features'][col] = most_common
                    except Exception as e:
                        self.logger.error(f"Erreur lors de l'analyse de {col} pour le cluster {cluster}: {e}")
        
        return {
            'n_clusters': n_clusters,
            'cluster_analysis': cluster_analysis
        }
    
    def generate_performance_visualizations(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Génère des visualisations des performances et les enregistre.
        
        Args:
            output_dir: Répertoire de sortie pour les visualisations
            
        Returns:
            Dictionnaire avec les chemins des visualisations
        """
        if output_dir is None:
            output_dir = self.reports_dir
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Récupérer les transactions
        trades_df = self.trade_journal.get_trades()
        
        if trades_df.empty:
            return {"error": "Aucune donnée disponible pour les visualisations"}
        
        # Convertir les timestamps en datetime
        if 'entry_time' in trades_df.columns:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        
        # Chemins des visualisations
        visualization_paths = {}
        
        # 1. Courbe de profit cumulatif
        try:
            plt.figure(figsize=(12, 6))
            
            # Trier par date d'entrée
            if 'entry_time' in trades_df.columns:
                trades_df = trades_df.sort_values('entry_time')
            
            # Calculer le profit cumulatif
            cumulative_pnl = trades_df['pnl'].cumsum()
            
            plt.plot(cumulative_pnl.values)
            plt.title('Profit cumulatif')
            plt.xlabel('Nombre de transactions')
            plt.ylabel('Profit cumulatif')
            plt.grid(True)
            
            cumulative_path = os.path.join(output_dir, 'cumulative_profit.png')
            plt.savefig(cumulative_path)
            plt.close()
            
            visualization_paths['cumulative_profit'] = cumulative_path
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de la courbe de profit cumulatif: {e}")
        
        # 2. Distribution des profits/pertes
        try:
            plt.figure(figsize=(12, 6))
            
            sns.histplot(trades_df['pnl'], kde=True)
            plt.title('Distribution des profits et pertes')
            plt.xlabel('Profit/Perte')
            plt.ylabel('Fréquence')
            plt.grid(True)
            
            distribution_path = os.path.join(output_dir, 'pnl_distribution.png')
            plt.savefig(distribution_path)
            plt.close()
            
            visualization_paths['pnl_distribution'] = distribution_path
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de la distribution des profits/pertes: {e}")
        
        # 3. Performance par symbole
        if 'symbol' in trades_df.columns:
            try:
                plt.figure(figsize=(14, 7))
                
                symbol_performance = trades_df.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
                symbol_performance.plot(kind='bar')
                plt.title('Performance totale par symbole')
                plt.xlabel('Symbole')
                plt.ylabel('Profit/Perte total')
                plt.xticks(rotation=45)
                plt.grid(True)
                
                symbol_path = os.path.join(output_dir, 'symbol_performance.png')
                plt.savefig(symbol_path, bbox_inches='tight')
                plt.close()
                
                visualization_paths['symbol_performance'] = symbol_path
            except Exception as e:
                self.logger.error(f"Erreur lors de la génération de la performance par symbole: {e}")
        
        # 4. Win rate par jour de la semaine
        if 'entry_time' in trades_df.columns:
            try:
                plt.figure(figsize=(12, 6))
                
                trades_df['day_of_week'] = trades_df['entry_time'].dt.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                day_win_rate = trades_df.groupby('day_of_week')['pnl'].apply(lambda x: (x > 0).mean())
                
                # Réordonner les jours de la semaine
                day_win_rate = day_win_rate.reindex(day_order)
                
                day_win_rate.plot(kind='bar')
                plt.title('Taux de réussite par jour de la semaine')
                plt.xlabel('Jour de la semaine')
                plt.ylabel('Taux de réussite')
                plt.ylim(0, 1)
                plt.grid(True)
                
                day_path = os.path.join(output_dir, 'day_win_rate.png')
                plt.savefig(day_path)
                plt.close()
                
                visualization_paths['day_win_rate'] = day_path
            except Exception as e:
                self.logger.error(f"Erreur lors de la génération du taux de réussite par jour: {e}")
        
        # Générer un rapport HTML simple
        try:
            html_content = """
            <html>
            <head>
                <title>Rapport de performance de trading</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    .image-container { margin: 20px 0; }
                    img { max-width: 100%; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                </style>
            </head>
            <body>
                <h1>Rapport de performance de trading</h1>
                <p>Généré le: {date}</p>
                
                <h2>Statistiques générales</h2>
                <table border="1" cellpadding="5">
                    <tr><th>Métrique</th><th>Valeur</th></tr>
                    <tr><td>Nombre total de transactions</td><td>{total_trades}</td></tr>
                    <tr><td>Transactions gagnantes</td><td>{winning_trades}</td></tr>
                    <tr><td>Transactions perdantes</td><td>{losing_trades}</td></tr>
                    <tr><td>Taux de réussite</td><td>{win_rate:.2%}</td></tr>
                    <tr><td>Profit/Perte total</td><td>{total_pnl:.2f}</td></tr>
                    <tr><td>Profit moyen</td><td>{avg_profit:.2f}</td></tr>
                    <tr><td>Perte moyenne</td><td>{avg_loss:.2f}</td></tr>
                </table>
                
                <h2>Visualisations</h2>
                
                <div class="image-container">
                    <h3>Profit cumulatif</h3>
                    <img src="{cumulative_profit}" alt="Profit cumulatif">
                </div>
                
                <div class="image-container">
                    <h3>Distribution des profits et pertes</h3>
                    <img src="{pnl_distribution}" alt="Distribution des profits et pertes">
                </div>
            """.format(
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                total_trades=len(trades_df),
                winning_trades=len(trades_df[trades_df['pnl'] > 0]),
                losing_trades=len(trades_df[trades_df['pnl'] <= 0]),
                win_rate=len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) if len(trades_df) > 0 else 0,
                total_pnl=trades_df['pnl'].sum(),
                avg_profit=trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0,
                avg_loss=trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] <= 0]) > 0 else 0,
                cumulative_profit=os.path.basename(visualization_paths.get('cumulative_profit', '')),
                pnl_distribution=os.path.basename(visualization_paths.get('pnl_distribution', ''))
            )
            
            # Ajouter des visualisations supplémentaires si elles existent
            if 'symbol_performance' in visualization_paths:
                html_content += """
                <div class="image-container">
                    <h3>Performance par symbole</h3>
                    <img src="{symbol_performance}" alt="Performance par symbole">
                </div>
                """.format(symbol_performance=os.path.basename(visualization_paths['symbol_performance']))
            
            if 'day_win_rate' in visualization_paths:
                html_content += """
                <div class="image-container">
                    <h3>Taux de réussite par jour</h3>
                    <img src="{day_win_rate}" alt="Taux de réussite par jour">
                </div>
                """.format(day_win_rate=os.path.basename(visualization_paths['day_win_rate']))
            
            html_content += """
            </body>
            </html>
            """
            
            html_path = os.path.join(output_dir, 'performance_report.html')
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            visualization_paths['html_report'] = html_path
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport HTML: {e}")
        
        return visualization_paths 