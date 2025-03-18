import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Callable, Tuple, Any
from datetime import datetime, timedelta
import logging
import os
import json
import pickle
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

from ..models.sentiment import MarketSentimentAnalyzer
from ..models.rl.advanced_rl_agent import MarketRegimeRLTrainer

class BacktestEnvironment:
    """
    Environnement de backtesting avancé qui simule l'exécution de stratégies de trading
    sur des données historiques avec diverses fonctionnalités avancées.
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame], initial_capital: float = 100000,
                 transaction_fee: float = 0.001, slippage: float = 0.0005,
                 timeframes: List[str] = None, start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None):
        """
        Initialise l'environnement de backtesting.
        
        Args:
            data: Dictionnaire de DataFrames contenant les données OHLCV pour différents timeframes
            initial_capital: Capital initial pour le backtesting
            transaction_fee: Frais de transaction en pourcentage (0.001 = 0.1%)
            slippage: Glissement de prix en pourcentage (0.0005 = 0.05%)
            timeframes: Liste des timeframes à utiliser (ex: ['1h', '4h', '1d'])
            start_date: Date de début du backtesting
            end_date: Date de fin du backtesting
        """
        self.logger = logging.getLogger(__name__)
        
        # Préparation des données
        self.timeframes = timeframes or ['1d']
        self.data = {}
        
        for tf, df in data.items():
            if tf in self.timeframes:
                # S'assurer que les données ont le bon format
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'Date' in df.columns:
                        df = df.set_index('Date')
                    else:
                        df = df.set_index(pd.to_datetime(df.index))
                
                # Filtrer les données par période si spécifiée
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                # S'assurer que les colonnes OHLCV sont présentes
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in df.columns:
                        raise ValueError(f"La colonne {col} est manquante dans les données pour {tf}")
                
                self.data[tf] = df
        
        if not self.data:
            raise ValueError("Aucune donnée valide fournie pour le backtesting")
        
        # Paramètres de simulation
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        
        # État du backtest
        self.reset()
        
        # Historique des performances
        self.performance_metrics = {}
        
        # Enregistrement des opérations
        self.trades_history = []
    
    def reset(self):
        """Réinitialise l'environnement de backtesting."""
        self.current_index = 0
        self.current_position = 0.0
        self.current_capital = self.initial_capital
        self.equity_curve = []
        self.positions_history = []
        self.trades_history = []
        self.unrealized_pnl = 0.0
        self.last_trade_price = None
        self.cumulative_fee = 0.0
        self.is_finished = False
    
    def step(self, action: float) -> Dict:
        """
        Exécute une étape de simulation avec l'action fournie.
        
        Args:
            action: Valeur entre -1 et 1 représentant l'action à prendre
                   -1 = position courte complète
                    0 = pas de position
                    1 = position longue complète
        
        Returns:
            Dict avec les informations sur l'état après l'action
        """
        if self.is_finished:
            return self._get_final_state()
        
        # Obtenir les données actuelles
        current_data = self._get_current_data()
        if current_data is None:
            self.is_finished = True
            return self._get_final_state()
        
        # Calculer la position cible (en pourcentage du capital)
        target_position = action
        
        # Calculer le changement de position
        current_price = current_data['price']
        position_change = target_position - self.current_position
        
        # Calculer le coût de la transaction
        transaction_cost = 0.0
        transaction_amount = abs(position_change) * self.current_capital
        
        if transaction_amount > 0:
            # Appliquer le slippage (prix d'exécution moins favorable)
            execution_price = current_price * (1 + self.slippage * np.sign(position_change))
            
            # Calculer les frais
            transaction_cost = transaction_amount * self.transaction_fee
            self.cumulative_fee += transaction_cost
            
            # Mettre à jour le capital et la position
            self.current_capital -= transaction_cost
            
            # Enregistrer la transaction
            self._record_trade(position_change, execution_price, transaction_cost)
        
        # Mise à jour de la position
        self.current_position = target_position
        
        # Calculer le P&L non réalisé
        if self.current_position != 0 and self.last_trade_price is not None:
            self.unrealized_pnl = self.current_position * self.current_capital * (current_price / self.last_trade_price - 1)
        
        # Mettre à jour l'historique des positions
        self.positions_history.append(self.current_position)
        
        # Calculer la valeur actuelle du portefeuille
        portfolio_value = self.current_capital + self.unrealized_pnl
        self.equity_curve.append(portfolio_value)
        
        # Passer à l'étape suivante
        self.current_index += 1
        
        # Vérifier si le backtest est terminé
        if self.current_index >= len(list(self.data.values())[0]) - 1:
            self.is_finished = True
            return self._get_final_state()
        
        # Préparer les données d'observation pour la prochaine étape
        observation = self._get_observation()
        
        return {
            'observation': observation,
            'portfolio_value': portfolio_value,
            'position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'done': self.is_finished,
            'timestamp': current_data['timestamp']
        }
    
    def _get_current_data(self) -> Optional[Dict]:
        """Obtient les données actuelles pour tous les timeframes."""
        # Utiliser le premier timeframe comme référence
        main_tf = self.timeframes[0]
        main_data = self.data[main_tf]
        
        if self.current_index >= len(main_data):
            return None
        
        row = main_data.iloc[self.current_index]
        
        return {
            'price': row['Close'],
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'volume': row['Volume'],
            'timestamp': main_data.index[self.current_index]
        }
    
    def _get_observation(self) -> Dict:
        """Construit l'observation complète de l'état actuel pour tous les timeframes."""
        observation = {}
        
        for tf in self.timeframes:
            df = self.data[tf]
            if self.current_index < len(df):
                # Obtenir les données des dernières 30 périodes pour ce timeframe
                lookback = min(30, self.current_index + 1)
                history = df.iloc[self.current_index + 1 - lookback:self.current_index + 1]
                
                observation[tf] = {
                    'ohlcv': history[['Open', 'High', 'Low', 'Close', 'Volume']].values.tolist(),
                    'timestamp': history.index.tolist()
                }
        
        # Ajouter des informations sur l'état du portefeuille
        observation['portfolio'] = {
            'capital': self.current_capital,
            'position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'equity': self.current_capital + self.unrealized_pnl
        }
        
        return observation
    
    def _record_trade(self, position_change: float, price: float, fee: float):
        """Enregistre une transaction dans l'historique."""
        trade_type = "BUY" if position_change > 0 else "SELL"
        timestamp = self._get_current_data()['timestamp']
        
        trade = {
            'timestamp': timestamp,
            'type': trade_type,
            'price': price,
            'size': abs(position_change),
            'value': abs(position_change) * self.current_capital,
            'fee': fee,
            'position_after': self.current_position + position_change,
            'capital_after': self.current_capital - fee
        }
        
        self.trades_history.append(trade)
        self.last_trade_price = price
    
    def _get_final_state(self) -> Dict:
        """Retourne l'état final du backtest avec les métriques de performance."""
        # Calculer les métriques de performance
        from .backtest_performance import calculate_performance_metrics
        
        self.performance_metrics = calculate_performance_metrics(
            equity_curve=self.equity_curve,
            initial_capital=self.initial_capital,
            positions=self.positions_history,
            trades=self.trades_history
        )
        
        return {
            'observation': self._get_observation(),
            'portfolio_value': self.equity_curve[-1] if self.equity_curve else self.initial_capital,
            'position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'done': True,
            'metrics': self.performance_metrics,
            'trades': self.trades_history
        }
    
    def run_backtest(self, strategy) -> Dict:
        """
        Exécute un backtest complet avec la stratégie fournie.
        
        Args:
            strategy: Stratégie de trading à tester
        
        Returns:
            Dict avec les résultats du backtest
        """
        self.reset()
        
        while not self.is_finished:
            # Obtenir l'observation actuelle
            observation = self._get_observation()
            
            # Obtenir l'action de la stratégie
            action = strategy.get_action(observation)
            
            # Exécuter l'étape
            state = self.step(action)
        
        # Générer le rapport
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Génère un rapport complet du backtest."""
        if not self.performance_metrics:
            # Calculer les métriques de performance si ce n'est pas déjà fait
            from .backtest_performance import calculate_performance_metrics
            
            self.performance_metrics = calculate_performance_metrics(
                equity_curve=self.equity_curve,
                initial_capital=self.initial_capital,
                positions=self.positions_history,
                trades=self.trades_history
            )
        
        # Créer le rapport
        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': self.equity_curve[-1] if self.equity_curve else self.initial_capital,
                'total_return_pct': (self.equity_curve[-1] / self.initial_capital - 1) * 100 if self.equity_curve else 0,
                'total_trades': len(self.trades_history),
                'total_fees': self.cumulative_fee,
                'timeframe': self.timeframes,
                'start_date': self.data[self.timeframes[0]].index[0],
                'end_date': self.data[self.timeframes[0]].index[-1]
            },
            'metrics': self.performance_metrics,
            'trades': self.trades_history
        }
        
        return report
    
    def plot_results(self, save_path: Optional[str] = None, plot_trades: bool = True):
        """
        Génère des graphiques pour visualiser les résultats du backtest.
        
        Args:
            save_path: Chemin où sauvegarder les graphiques (None pour afficher)
            plot_trades: Si True, affiche les transactions sur le graphique
        """
        if not self.equity_curve:
            self.logger.warning("Aucune donnée à afficher. Exécutez d'abord un backtest.")
            return
        
        from .backtest_visualization import plot_backtest_results
        
        plot_backtest_results(
            equity_curve=self.equity_curve,
            positions=self.positions_history,
            trades=self.trades_history,
            data=self.data[self.timeframes[0]],
            metrics=self.performance_metrics,
            save_path=save_path,
            plot_trades=plot_trades
        )

# Interface de base pour les stratégies de trading
class TradingStrategy(ABC):
    """Interface abstraite pour les stratégies de trading."""
    
    @abstractmethod
    def get_action(self, observation: Dict) -> float:
        """
        Détermine l'action à prendre en fonction de l'observation.
        
        Args:
            observation: Observation actuelle du marché
            
        Returns:
            float: Valeur entre -1 et 1 représentant l'action à prendre
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Réinitialise l'état interne de la stratégie."""
        pass 