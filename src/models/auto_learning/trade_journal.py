import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import json
import sqlite3

class TradeJournal:
    """
    Système de journalisation des transactions qui enregistre et récupère l'historique des trades
    pour permettre l'analyse et l'apprentissage automatique des erreurs.
    """
    
    def __init__(self, db_path: str = "data/trade_journal.db"):
        """
        Initialise le journal de trading avec la connexion à la base de données.
        
        Args:
            db_path: Chemin vers la base de données SQLite
        """
        self.db_path = db_path
        self._ensure_db_dir_exists()
        self._init_database()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Journal de trading initialisé")
        
    def _ensure_db_dir_exists(self):
        """Assure que le répertoire pour la base de données existe"""
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
    def _init_database(self):
        """Initialise la structure de la base de données"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table des transactions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE,
            symbol TEXT,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            entry_price REAL,
            exit_price REAL,
            position_size REAL,
            direction TEXT,
            pnl REAL,
            pnl_percent REAL,
            fee REAL,
            strategy_name TEXT,
            model_version TEXT,
            entry_signals TEXT,
            exit_signals TEXT,
            market_conditions TEXT,
            trade_metadata TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Table d'analyse post-trade
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE,
            error_type TEXT,
            error_severity INTEGER,
            improvement_suggestions TEXT,
            ai_analysis TEXT,
            model_adjustments TEXT,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Enregistre une transaction dans le journal.
        
        Args:
            trade_data: Dictionnaire contenant les données de la transaction
            
        Returns:
            trade_id: Identifiant unique de la transaction
        """
        try:
            # Générer un identifiant unique si non fourni
            if 'trade_id' not in trade_data:
                trade_data['trade_id'] = f"{trade_data.get('symbol', 'UNKNOWN')}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
            
            # Convertir les objets complexes en JSON
            for field in ['entry_signals', 'exit_signals', 'market_conditions', 'trade_metadata']:
                if field in trade_data and isinstance(trade_data[field], (dict, list)):
                    trade_data[field] = json.dumps(trade_data[field])
            
            # Insérer dans la base de données
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            columns = ', '.join(trade_data.keys())
            placeholders = ', '.join(['?' for _ in trade_data])
            values = tuple(trade_data.values())
            
            cursor.execute(
                f"INSERT INTO trades ({columns}) VALUES ({placeholders})",
                values
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Transaction enregistrée avec ID: {trade_data['trade_id']}")
            return trade_data['trade_id']
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement de la transaction: {e}")
            raise
    
    def get_trade(self, trade_id: str) -> Dict[str, Any]:
        """
        Récupère les détails d'une transaction spécifique.
        
        Args:
            trade_id: Identifiant de la transaction
            
        Returns:
            Dictionnaire contenant les détails de la transaction
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trades WHERE trade_id = ?", (trade_id,))
        trade = cursor.fetchone()
        
        if not trade:
            conn.close()
            return None
        
        # Convertir en dictionnaire
        trade_dict = dict(trade)
        
        # Convertir les champs JSON en objets Python
        for field in ['entry_signals', 'exit_signals', 'market_conditions', 'trade_metadata']:
            if trade_dict.get(field) and isinstance(trade_dict[field], str):
                try:
                    trade_dict[field] = json.loads(trade_dict[field])
                except:
                    pass
        
        conn.close()
        return trade_dict
    
    def get_trades(self, 
                  start_date: Optional[datetime] = None, 
                  end_date: Optional[datetime] = None,
                  symbol: Optional[str] = None,
                  strategy: Optional[str] = None,
                  model_version: Optional[str] = None,
                  profitable_only: Optional[bool] = None) -> pd.DataFrame:
        """
        Récupère plusieurs transactions selon les critères spécifiés.
        
        Args:
            start_date: Date de début pour filtrer les transactions
            end_date: Date de fin pour filtrer les transactions
            symbol: Symbole de trading spécifique
            strategy: Nom de la stratégie
            model_version: Version du modèle
            profitable_only: Si True, retourne uniquement les transactions profitables
            
        Returns:
            DataFrame contenant les transactions
        """
        conn = sqlite3.connect(self.db_path)
        
        # Construire la requête SQL avec les filtres
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date.isoformat())
            
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date.isoformat())
            
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
            
        if strategy:
            query += " AND strategy_name = ?"
            params.append(strategy)
            
        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)
            
        if profitable_only is not None:
            query += " AND pnl > 0" if profitable_only else " AND pnl <= 0"
            
        # Exécuter la requête et convertir en DataFrame
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convertir les champs JSON en objets Python
        for field in ['entry_signals', 'exit_signals', 'market_conditions', 'trade_metadata']:
            if field in df.columns:
                df[field] = df[field].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        
        return df
    
    def log_trade_analysis(self, trade_id: str, analysis_data: Dict[str, Any]) -> bool:
        """
        Enregistre l'analyse post-trade d'une transaction.
        
        Args:
            trade_id: Identifiant de la transaction
            analysis_data: Données d'analyse
            
        Returns:
            True si l'opération a réussi, False sinon
        """
        try:
            # Préparer les données
            analysis_data['trade_id'] = trade_id
            
            # Convertir les objets complexes en JSON
            for field in ['improvement_suggestions', 'ai_analysis', 'model_adjustments']:
                if field in analysis_data and isinstance(analysis_data[field], (dict, list)):
                    analysis_data[field] = json.dumps(analysis_data[field])
            
            # Insérer dans la base de données
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            columns = ', '.join(analysis_data.keys())
            placeholders = ', '.join(['?' for _ in analysis_data])
            values = tuple(analysis_data.values())
            
            cursor.execute(
                f"INSERT OR REPLACE INTO trade_analysis ({columns}) VALUES ({placeholders})",
                values
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Analyse de transaction enregistrée pour ID: {trade_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement de l'analyse: {e}")
            return False
    
    def get_trade_with_analysis(self, trade_id: str) -> Dict[str, Any]:
        """
        Récupère une transaction avec son analyse associée.
        
        Args:
            trade_id: Identifiant de la transaction
            
        Returns:
            Dictionnaire combinant les données de transaction et d'analyse
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT t.*, a.error_type, a.error_severity, a.improvement_suggestions, 
                   a.ai_analysis, a.model_adjustments, a.analyzed_at
            FROM trades t
            LEFT JOIN trade_analysis a ON t.trade_id = a.trade_id
            WHERE t.trade_id = ?
        """, (trade_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
            
        # Convertir en dictionnaire
        result_dict = dict(result)
        
        # Convertir les champs JSON en objets Python
        json_fields = ['entry_signals', 'exit_signals', 'market_conditions', 'trade_metadata',
                      'improvement_suggestions', 'ai_analysis', 'model_adjustments']
        
        for field in json_fields:
            if result_dict.get(field) and isinstance(result_dict[field], str):
                try:
                    result_dict[field] = json.loads(result_dict[field])
                except:
                    pass
        
        return result_dict
    
    def export_to_csv(self, filepath: str, start_date: Optional[datetime] = None, 
                     end_date: Optional[datetime] = None) -> bool:
        """
        Exporte les données du journal vers un fichier CSV.
        
        Args:
            filepath: Chemin du fichier CSV à créer
            start_date: Date de début pour filtrer les transactions
            end_date: Date de fin pour filtrer les transactions
            
        Returns:
            True si l'opération a réussi, False sinon
        """
        try:
            # Récupérer les transactions
            trades_df = self.get_trades(start_date=start_date, end_date=end_date)
            
            # Convertir les champs JSON en chaînes pour l'export
            for col in trades_df.columns:
                if col in ['entry_signals', 'exit_signals', 'market_conditions', 'trade_metadata']:
                    trades_df[col] = trades_df[col].apply(lambda x: json.dumps(x) if x else None)
            
            # Exporter vers CSV
            trades_df.to_csv(filepath, index=False)
            self.logger.info(f"Journal de trading exporté vers {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exportation du journal: {e}")
            return False

    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calcule des statistiques globales sur les transactions enregistrées.
        
        Returns:
            Dictionnaire contenant les statistiques
        """
        conn = sqlite3.connect(self.db_path)
        
        # Récupérer toutes les transactions
        trades_df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()
        
        if trades_df.empty:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "total_pnl": 0
            }
        
        # Calculer les statistiques
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        total_pnl = trades_df['pnl'].sum()
        
        # Statistiques avancées par stratégie
        strategy_stats = {}
        for strategy in trades_df['strategy_name'].unique():
            strategy_df = trades_df[trades_df['strategy_name'] == strategy]
            strategy_win_rate = len(strategy_df[strategy_df['pnl'] > 0]) / len(strategy_df)
            strategy_pnl = strategy_df['pnl'].sum()
            
            strategy_stats[strategy] = {
                "trades": len(strategy_df),
                "win_rate": strategy_win_rate,
                "total_pnl": strategy_pnl
            }
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_pnl": total_pnl,
            "strategies": strategy_stats
        } 