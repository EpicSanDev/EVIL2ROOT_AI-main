import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime
import math

def calculate_performance_metrics(equity_curve: List[float], initial_capital: float,
                                 positions: List[float], trades: List[Dict]) -> Dict:
    """
    Calcule les métriques de performance pour un backtest.
    
    Args:
        equity_curve: Liste des valeurs de portefeuille à chaque étape
        initial_capital: Capital initial du backtest
        positions: Liste des positions à chaque étape
        trades: Liste des transactions effectuées
        
    Returns:
        Dict contenant toutes les métriques de performance calculées
    """
    if not equity_curve or len(equity_curve) < 2:
        return {
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'volatility': 0.0
        }
    
    # Convertir en arrays numpy
    equity = np.array(equity_curve)
    
    # Calculer les rendements journaliers
    returns = np.diff(equity) / equity[:-1]
    
    # Total return
    total_return = equity[-1] - initial_capital
    total_return_pct = (equity[-1] / initial_capital - 1) * 100
    
    # Annualized return (supposer 252 jours de trading par an)
    n_days = len(equity)
    annualized_return = ((1 + total_return_pct / 100) ** (252 / n_days) - 1) * 100
    
    # Volatilité annualisée
    volatility = np.std(returns) * np.sqrt(252) * 100
    
    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    drawdown_pct = drawdown / peak * 100
    max_drawdown = np.max(drawdown)
    max_drawdown_pct = np.max(drawdown_pct)
    
    # Ratios basés sur les transactions
    if trades:
        # Calculer les profits et pertes par transaction
        profits = [t['price'] * t['size'] - t.get('fee', 0) for t in trades if t['type'] == 'SELL']
        losses = [t['price'] * t['size'] + t.get('fee', 0) for t in trades if t['type'] == 'BUY']
        
        # Win rate
        winning_trades = sum(1 for p in profits if p > 0)
        losing_trades = sum(1 for p in profits if p <= 0)
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Profit factor
        total_profit = sum(p for p in profits if p > 0)
        total_loss = abs(sum(p for p in profits if p <= 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    else:
        win_rate = 0.0
        profit_factor = 0.0
    
    # Sharpe Ratio (supposer taux sans risque de 0%)
    sharpe_ratio = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
    
    # Sortino Ratio (se concentre sur la volatilité à la baisse)
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
    sortino_ratio = (np.mean(returns) * 252) / downside_deviation if downside_deviation > 0 else 0.0
    
    # Calmar Ratio (rendement annualisé / drawdown maximal)
    calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else 0.0
    
    # Maximum consecutive wins/losses
    if trades:
        results = []
        for i in range(1, len(trades)):
            # Une transaction profitable est quand on vend à un prix plus élevé qu'on a acheté
            if trades[i]['type'] == 'SELL' and trades[i-1]['type'] == 'BUY':
                profit = trades[i]['price'] - trades[i-1]['price']
                results.append(1 if profit > 0 else -1)
            elif trades[i]['type'] == 'BUY' and trades[i-1]['type'] == 'SELL':
                profit = trades[i-1]['price'] - trades[i]['price']
                results.append(1 if profit > 0 else -1)
        
        # Calculer les séquences de gains et pertes consécutifs
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for r in results:
            if r > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
    else:
        max_consecutive_wins = 0
        max_consecutive_losses = 0
    
    # Value at Risk (VaR) à 95%
    var_95 = np.percentile(returns, 5) * initial_capital
    
    # Expected Shortfall (ES) ou Conditional VaR à 95%
    cvar_95 = np.mean(returns[returns <= np.percentile(returns, 5)]) * initial_capital if len(returns[returns <= np.percentile(returns, 5)]) > 0 else 0.0
    
    # Ulcer Index (mesure du stress du portefeuille)
    # Calcule la racine carrée de la moyenne des carrés des drawdowns
    drawdown_percentage = drawdown_pct / 100  # Convertir en décimal
    ulcer_index = np.sqrt(np.mean(np.square(drawdown_percentage)))
    
    # Return / Drawdown (mesure de la récompense par unité de risque)
    return_drawdown_ratio = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else float('inf')
    
    # Kurtosis (mesure les événements extrêmes)
    kurtosis = pd.Series(returns).kurtosis()
    
    # Skewness (asymétrie des rendements)
    skewness = pd.Series(returns).skew()
    
    # Alpha et Beta (par rapport au marché) - non calculés ici car nécessitent un benchmark
    
    # Max Liquidité consommée (maximum de capital utilisé)
    if positions:
        max_exposure = max(abs(p) for p in positions)
    else:
        max_exposure = 0.0
    
    # Nombre de jours en position
    days_in_position = sum(1 for p in positions if p != 0)
    position_ratio = days_in_position / len(positions) if positions else 0.0
    
    # Créer le dictionnaire de métriques
    metrics = {
        # Métriques de base
        'total_return': float(total_return),
        'total_return_pct': float(total_return_pct),
        'annualized_return': float(annualized_return),
        'volatility': float(volatility),
        'max_drawdown': float(max_drawdown),
        'max_drawdown_pct': float(max_drawdown_pct),
        
        # Métriques de ratio
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'calmar_ratio': float(calmar_ratio),
        'return_drawdown_ratio': float(return_drawdown_ratio),
        
        # Métriques de risque
        'var_95': float(var_95),
        'cvar_95': float(cvar_95),
        'ulcer_index': float(ulcer_index),
        'max_exposure': float(max_exposure),
        
        # Statistiques des rendements
        'kurtosis': float(kurtosis),
        'skewness': float(skewness),
        
        # Métriques de transaction
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'max_consecutive_wins': int(max_consecutive_wins),
        'max_consecutive_losses': int(max_consecutive_losses),
        'total_trades': len(trades),
        'days_in_position': int(days_in_position),
        'position_ratio': float(position_ratio)
    }
    
    return metrics

def calculate_drawdowns(equity_curve: List[float]) -> List[Dict]:
    """
    Calcule tous les drawdowns significatifs dans une courbe d'équité.
    
    Args:
        equity_curve: Liste des valeurs de portefeuille
        
    Returns:
        Liste de dictionnaires contenant les informations sur chaque drawdown
    """
    if not equity_curve or len(equity_curve) < 2:
        return []
    
    equity = np.array(equity_curve)
    
    # Initialiser les structures de données
    drawdowns = []
    current_peak = equity[0]
    in_drawdown = False
    drawdown_start = 0
    
    for i in range(1, len(equity)):
        # Nouveau pic
        if equity[i] > current_peak:
            current_peak = equity[i]
            
            # Si nous étions en drawdown, c'est la fin du drawdown
            if in_drawdown:
                drawdown_end = i - 1
                recovery_days = i - drawdown_end
                
                # Calculer les métriques du drawdown
                drawdown_value = current_peak - np.min(equity[drawdown_start:drawdown_end+1])
                drawdown_pct = drawdown_value / current_peak * 100
                
                # Ajouter à la liste si le drawdown est significatif (> 1%)
                if drawdown_pct > 1.0:
                    drawdowns.append({
                        'start_idx': drawdown_start,
                        'end_idx': drawdown_end,
                        'recovery_idx': i,
                        'peak_value': float(current_peak),
                        'trough_value': float(np.min(equity[drawdown_start:drawdown_end+1])),
                        'drawdown_value': float(drawdown_value),
                        'drawdown_pct': float(drawdown_pct),
                        'duration_days': drawdown_end - drawdown_start + 1,
                        'recovery_days': recovery_days
                    })
                
                in_drawdown = False
        
        # Drawdown
        elif equity[i] < current_peak:
            # Si nous ne sommes pas déjà en drawdown, c'est le début d'un nouveau drawdown
            if not in_drawdown:
                drawdown_start = i - 1  # Le pic était à l'index précédent
                in_drawdown = True
    
    # Si nous sommes toujours en drawdown à la fin
    if in_drawdown:
        drawdown_end = len(equity) - 1
        drawdown_value = current_peak - np.min(equity[drawdown_start:drawdown_end+1])
        drawdown_pct = drawdown_value / current_peak * 100
        
        # Ajouter à la liste si le drawdown est significatif
        if drawdown_pct > 1.0:
            drawdowns.append({
                'start_idx': drawdown_start,
                'end_idx': drawdown_end,
                'recovery_idx': None,  # Pas encore récupéré
                'peak_value': float(current_peak),
                'trough_value': float(np.min(equity[drawdown_start:drawdown_end+1])),
                'drawdown_value': float(drawdown_value),
                'drawdown_pct': float(drawdown_pct),
                'duration_days': drawdown_end - drawdown_start + 1,
                'recovery_days': None  # Pas encore récupéré
            })
    
    # Trier les drawdowns par taille (du plus grand au plus petit)
    drawdowns.sort(key=lambda x: x['drawdown_pct'], reverse=True)
    
    return drawdowns

def calculate_monthly_returns(equity_curve: List[float], dates: List[datetime]) -> pd.DataFrame:
    """
    Calcule les rendements mensuels à partir d'une courbe d'équité.
    
    Args:
        equity_curve: Liste des valeurs de portefeuille
        dates: Liste des dates correspondantes
        
    Returns:
        DataFrame pandas avec les rendements mensuels
    """
    if not equity_curve or len(equity_curve) < 2 or len(equity_curve) != len(dates):
        return pd.DataFrame()
    
    # Créer un DataFrame avec les dates et valeurs
    df = pd.DataFrame({
        'date': dates,
        'equity': equity_curve
    })
    
    # Convertir la colonne date en datetime si ce n'est pas déjà fait
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Extraire la dernière valeur de chaque mois
    monthly_equity = df['equity'].resample('M').last()
    
    # Calculer les rendements mensuels
    monthly_returns = monthly_equity.pct_change().dropna()
    
    # Formater en DataFrame avec année et mois
    result = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values * 100  # En pourcentage
    })
    
    return result

def calculate_yearly_returns(equity_curve: List[float], dates: List[datetime]) -> pd.DataFrame:
    """
    Calcule les rendements annuels à partir d'une courbe d'équité.
    
    Args:
        equity_curve: Liste des valeurs de portefeuille
        dates: Liste des dates correspondantes
        
    Returns:
        DataFrame pandas avec les rendements annuels
    """
    if not equity_curve or len(equity_curve) < 2 or len(equity_curve) != len(dates):
        return pd.DataFrame()
    
    # Créer un DataFrame avec les dates et valeurs
    df = pd.DataFrame({
        'date': dates,
        'equity': equity_curve
    })
    
    # Convertir la colonne date en datetime si ce n'est pas déjà fait
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Extraire la dernière valeur de chaque année
    yearly_equity = df['equity'].resample('Y').last()
    
    # Calculer les rendements annuels
    yearly_returns = yearly_equity.pct_change().dropna()
    
    # Formater en DataFrame
    result = pd.DataFrame({
        'year': yearly_returns.index.year,
        'return': yearly_returns.values * 100  # En pourcentage
    })
    
    return result

def analyze_trades(trades: List[Dict]) -> Dict:
    """
    Analyse les transactions pour obtenir des statistiques avancées.
    
    Args:
        trades: Liste des transactions effectuées
        
    Returns:
        Dict contenant les statistiques des transactions
    """
    if not trades:
        return {
            'total_trades': 0,
            'avg_profit_per_trade': 0.0,
            'avg_profit_winning': 0.0,
            'avg_loss_losing': 0.0,
            'largest_profit': 0.0,
            'largest_loss': 0.0,
            'avg_holding_period': 0.0
        }
    
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
    
    # Statistiques des transactions
    profits = [pair['profit'] for pair in trade_pairs]
    winning_trades = [pair for pair in trade_pairs if pair['profit'] > 0]
    losing_trades = [pair for pair in trade_pairs if pair['profit'] <= 0]
    
    total_trades = len(trade_pairs)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
    
    avg_profit_per_trade = sum(profits) / total_trades if total_trades > 0 else 0.0
    avg_profit_winning = sum(pair['profit'] for pair in winning_trades) / len(winning_trades) if winning_trades else 0.0
    avg_loss_losing = sum(pair['profit'] for pair in losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    largest_profit = max(profits) if profits else 0.0
    largest_loss = min(profits) if profits else 0.0
    
    avg_holding_period = sum(pair['holding_period'] for pair in trade_pairs) / total_trades if total_trades > 0 else 0.0
    
    # Calculer l'expectative (profit moyen par trade)
    expectancy = win_rate * avg_profit_winning + (1 - win_rate) * avg_loss_losing
    
    # Calculer le ratio profit/perte moyen
    profit_loss_ratio = abs(avg_profit_winning / avg_loss_losing) if avg_loss_losing != 0 else float('inf')
    
    # Calculer le facteur de profit (somme des profits / somme des pertes)
    total_profit = sum(pair['profit'] for pair in winning_trades)
    total_loss = abs(sum(pair['profit'] for pair in losing_trades))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculer le facteur de Kelly (taux de profit optimal)
    if profit_loss_ratio == float('inf'):
        kelly_criterion = win_rate
    elif win_rate == 0 or profit_loss_ratio == 0:
        kelly_criterion = 0
    else:
        kelly_criterion = win_rate - ((1 - win_rate) / profit_loss_ratio)
        kelly_criterion = max(0, kelly_criterion)  # Ne pas permettre de valeurs négatives
    
    # Calculer la moyenne des rendements en pourcentage
    avg_profit_pct = sum(pair['profit_pct'] for pair in trade_pairs) / total_trades if total_trades > 0 else 0.0
    
    return {
        'total_trades': total_trades,
        'win_rate': float(win_rate * 100),  # En pourcentage
        'avg_profit_per_trade': float(avg_profit_per_trade),
        'avg_profit_winning': float(avg_profit_winning),
        'avg_loss_losing': float(avg_loss_losing),
        'avg_profit_pct': float(avg_profit_pct),
        'largest_profit': float(largest_profit),
        'largest_loss': float(largest_loss),
        'avg_holding_period': float(avg_holding_period),
        'expectancy': float(expectancy),
        'profit_loss_ratio': float(profit_loss_ratio),
        'profit_factor': float(profit_factor),
        'kelly_criterion': float(kelly_criterion),
        'avg_bars_winning': float(sum(pair['holding_period'] for pair in winning_trades) / len(winning_trades) if winning_trades else 0),
        'avg_bars_losing': float(sum(pair['holding_period'] for pair in losing_trades) / len(losing_trades) if losing_trades else 0)
    } 