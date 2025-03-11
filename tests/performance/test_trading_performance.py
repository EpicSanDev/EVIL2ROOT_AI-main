import time
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.models.position_manager import PositionManager
from app.trading import TradingBot, DataManager

class TestTradingPerformance:
    
    @pytest.fixture
    def large_data_sample(self):
        """Generate a large dataset for performance testing"""
        # Créer 2 ans de données quotidiennes
        date_range = pd.date_range(start='2020-01-01', periods=500, freq='D')
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'PYPL', 'ADBE', 'NFLX']
        
        data_dict = {}
        for symbol in symbols:
            base_price = np.random.uniform(50, 500)
            noise = np.random.normal(0, 1, size=len(date_range))
            trend = np.linspace(0, 20, len(date_range))
            
            prices = base_price + trend + noise * 5
            daily_returns = np.random.normal(0.0005, 0.015, size=len(date_range))
            
            data = pd.DataFrame({
                'Open': prices * (1 - daily_returns/2),
                'High': prices * (1 + daily_returns),
                'Low': prices * (1 - daily_returns),
                'Close': prices,
                'Volume': np.random.uniform(1000000, 10000000, size=len(date_range))
            }, index=date_range)
            
            data_dict[symbol] = data
            
        return data_dict, symbols
    
    def test_batch_signal_generation_performance(self, large_data_sample):
        """Test performance of generating trading signals for multiple symbols"""
        data_dict, symbols = large_data_sample
        
        # Configurer les gestionnaires
        position_manager = PositionManager(initial_balance=1000000.0)
        trading_bot = TradingBot(initial_balance=1000000.0, position_manager=position_manager)
        
        # Mesurer le temps pour générer des signaux pour tous les symboles
        start_time = time.time()
        
        for symbol in symbols:
            signals = trading_bot.generate_trading_signals(symbol, data_dict[symbol])
            assert signals is not None
        
        elapsed_time = time.time() - start_time
        
        print(f"\nTemps pour générer les signaux pour {len(symbols)} symboles: {elapsed_time:.2f} secondes")
        # Le benchmark devrait être suffisamment rapide pour un trading en temps réel
        assert elapsed_time < 10.0, "La génération de signaux prend trop de temps"
    
    def test_portfolio_management_performance(self, large_data_sample):
        """Test performance de gestion d'un grand nombre de positions"""
        data_dict, symbols = large_data_sample
        
        # Configurer les gestionnaires
        position_manager = PositionManager(initial_balance=10000000.0)
        
        # Ouvrir 100 positions
        start_time = time.time()
        
        num_positions = 100
        positions = []
        
        # Créer des positions aléatoires
        for i in range(num_positions):
            symbol = symbols[i % len(symbols)]
            direction = 'long' if np.random.random() > 0.5 else 'short'
            entry_price = float(data_dict[symbol]['Close'].iloc[-1])
            size = np.random.uniform(10, 100)
            
            position = position_manager.open_position(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                size=size,
                stop_loss=entry_price * 0.95 if direction == 'long' else entry_price * 1.05,
                take_profit=entry_price * 1.1 if direction == 'long' else entry_price * 0.9
            )
            positions.append(position.id)
        
        open_time = time.time() - start_time
        print(f"\nTemps pour ouvrir {num_positions} positions: {open_time:.4f} secondes")
        
        # Vérifier les stops pour toutes les positions
        start_time = time.time()
        
        market_prices = {}
        for symbol in symbols:
            market_prices[symbol] = float(data_dict[symbol]['Close'].iloc[-1])
        
        for symbol in symbols:
            position_manager.check_stops(symbol, market_prices[symbol])
        
        check_time = time.time() - start_time
        print(f"Temps pour vérifier les stops sur tous les symboles: {check_time:.4f} secondes")
        
        # Fermer toutes les positions
        start_time = time.time()
        
        for pos_id in positions:
            if pos_id in position_manager.positions:
                symbol = position_manager.positions[pos_id].symbol
                position_manager.close_position(pos_id, market_prices[symbol])
        
        close_time = time.time() - start_time
        print(f"Temps pour fermer toutes les positions: {close_time:.4f} secondes")
        
        # Vérifier les performances
        assert open_time < 1.0, "L'ouverture des positions prend trop de temps"
        assert check_time < 0.5, "La vérification des stops prend trop de temps"
        assert close_time < 1.0, "La fermeture des positions prend trop de temps"
    
    def test_data_update_performance(self):
        """Test performance de mise à jour des données pour plusieurs symboles"""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'PYPL', 'ADBE', 'NFLX']
        
        # Créer un gestionnaire de données
        data_manager = DataManager(symbols=symbols)
        
        # Mesurer le temps pour initialiser les données
        start_time = time.time()
        data_manager.initialize_data()
        init_time = time.time() - start_time
        
        print(f"\nTemps pour initialiser les données de {len(symbols)} symboles: {init_time:.2f} secondes")
        
        # Mesurer le temps pour mettre à jour les données
        start_time = time.time()
        data_manager.update_data()
        update_time = time.time() - start_time
        
        print(f"Temps pour mettre à jour les données de {len(symbols)} symboles: {update_time:.2f} secondes")
        
        # Les benchmarks
        assert init_time < 20.0, "L'initialisation des données prend trop de temps"
        assert update_time < 10.0, "La mise à jour des données prend trop de temps"
    
    def test_backtesting_performance(self, large_data_sample):
        """Test performance du backtesting sur une grande quantité de données"""
        data_dict, symbols = large_data_sample
        
        # Créer un TradingBot pour le backtest
        position_manager = PositionManager(initial_balance=1000000.0)
        trading_bot = TradingBot(initial_balance=1000000.0, position_manager=position_manager)
        
        # Préparer les données pour un seul symbole
        symbol = symbols[0]
        data = data_dict[symbol]
        
        # Mesurer le temps pour exécuter un backtest
        start_time = time.time()
        
        # Simuler un backtest sur les 100 derniers jours
        for i in range(100):
            idx = len(data) - 100 + i
            current_data = data.iloc[:idx]
            
            # Générer des signaux basés sur les données disponibles jusqu'à maintenant
            signal = trading_bot.generate_trading_signals(symbol, current_data)
            
            # Simuler l'exécution de trades basée sur le signal
            current_price = float(current_data.iloc[-1]['Close'])
            
            if signal and 'decision' in signal:
                if signal['decision'] == 'buy':
                    trading_bot.execute_trade('buy', symbol, current_price, 
                                             current_price * 1.1, current_price * 0.95)
                elif signal['decision'] == 'sell':
                    trading_bot.execute_trade('sell', symbol, current_price, 
                                             current_price * 0.9, current_price * 1.05)
            
            # Gérer les positions ouvertes
            trading_bot.manage_open_positions(None)  # Normalement un data_manager serait passé ici
            
        elapsed_time = time.time() - start_time
        
        print(f"\nTemps pour exécuter un backtest de 100 jours: {elapsed_time:.2f} secondes")
        assert elapsed_time < 20.0, "Le backtesting prend trop de temps" 