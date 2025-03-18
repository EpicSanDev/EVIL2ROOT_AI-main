from .advanced_backtesting import BacktestEnvironment, TradingStrategy
from .backtest_strategies import (
    SentimentBasedStrategy,
    TechnicalStrategy,
    RLBasedStrategy,
    HybridStrategy
)
from .backtest_performance import (
    calculate_performance_metrics,
    calculate_drawdowns,
    calculate_monthly_returns,
    calculate_yearly_returns,
    analyze_trades
)
from .backtest_visualization import (
    plot_backtest_results,
    plot_monthly_heatmap,
    plot_trade_analysis,
    generate_performance_report,
    plot_regime_performance
)

__all__ = [
    # Environnement de backtesting
    'BacktestEnvironment',
    'TradingStrategy',
    
    # Stratégies
    'SentimentBasedStrategy',
    'TechnicalStrategy',
    'RLBasedStrategy',
    'HybridStrategy',
    
    # Métriques de performance
    'calculate_performance_metrics',
    'calculate_drawdowns',
    'calculate_monthly_returns',
    'calculate_yearly_returns',
    'analyze_trades',
    
    # Visualisation
    'plot_backtest_results',
    'plot_monthly_heatmap',
    'plot_trade_analysis',
    'generate_performance_report',
    'plot_regime_performance'
]
