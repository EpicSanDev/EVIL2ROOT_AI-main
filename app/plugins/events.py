"""
Module définissant les événements standard du système de plugins.
Les plugins peuvent s'abonner à ces événements pour étendre les fonctionnalités du trading bot.
"""

from typing import Dict, Any, List, Set, Optional
from enum import Enum, auto

class EventType(Enum):
    """Types d'événements standards du système de trading"""
    
    # Événements liés à l'analyse de marché
    BEFORE_MARKET_ANALYSIS = auto()    # Avant l'analyse du marché
    AFTER_MARKET_ANALYSIS = auto()     # Après l'analyse du marché
    SIGNAL_GENERATED = auto()          # Lorsqu'un signal de trading est généré
    
    # Événements liés aux opérations de trading
    BEFORE_TRADE = auto()              # Avant l'exécution d'un trade
    AFTER_TRADE = auto()               # Après l'exécution d'un trade
    TRADE_CLOSED = auto()              # Lorsqu'un trade est fermé
    PORTFOLIO_UPDATED = auto()         # Lorsque le portefeuille est mis à jour
    
    # Événements liés aux données
    MARKET_DATA_UPDATED = auto()       # Lorsque les données de marché sont mises à jour
    NEWS_RECEIVED = auto()             # Lorsque de nouvelles actualités sont reçues
    SENTIMENT_ANALYZED = auto()        # Après l'analyse de sentiment
    
    # Événements liés au système
    SYSTEM_STARTUP = auto()            # Au démarrage du système
    SYSTEM_SHUTDOWN = auto()           # À l'arrêt du système
    PLUGIN_LOADED = auto()             # Lorsqu'un plugin est chargé
    PLUGIN_UNLOADED = auto()           # Lorsqu'un plugin est déchargé
    
    # Événements liés à l'interface utilisateur
    UI_REFRESH = auto()                # Lorsque l'interface doit être rafraîchie
    SETTINGS_CHANGED = auto()          # Lorsque les paramètres sont modifiés
    USER_ACTION = auto()               # Lorsque l'utilisateur effectue une action

# Événements spécifiques à la récupération de news et à l'analyse de sentiment
class NewsEventType(Enum):
    """Types d'événements spécifiques aux actualités et à l'analyse de sentiment"""
    
    BEFORE_NEWS_FETCH = auto()         # Avant la récupération des actualités
    AFTER_NEWS_FETCH = auto()          # Après la récupération des actualités
    BEFORE_SENTIMENT_ANALYSIS = auto() # Avant l'analyse de sentiment
    AFTER_SENTIMENT_ANALYSIS = auto()  # Après l'analyse de sentiment
    HIGH_IMPACT_NEWS_DETECTED = auto() # Lorsqu'une actualité à fort impact est détectée

# Dictionnaire des paramètres standards pour chaque type d'événement
EVENT_PARAMETERS = {
    # Structure des paramètres pour les événements d'analyse de marché
    EventType.BEFORE_MARKET_ANALYSIS: {
        "symbol": str,                  # Symbole analysé
        "timeframe": str,               # Période temporelle (1h, 4h, 1d, etc.)
        "data": "DataFrame",            # Données de marché
    },
    EventType.AFTER_MARKET_ANALYSIS: {
        "symbol": str,                  # Symbole analysé
        "timeframe": str,               # Période temporelle
        "data": "DataFrame",            # Données de marché
        "analysis_results": Dict,       # Résultats de l'analyse
        "indicators": Dict,             # Valeurs des indicateurs
    },
    EventType.SIGNAL_GENERATED: {
        "symbol": str,                  # Symbole
        "signal_type": str,             # Type de signal (buy, sell, etc.)
        "confidence": float,            # Niveau de confiance
        "source": str,                  # Source du signal
        "indicators": Dict,             # Indicateurs qui ont généré le signal
        "timestamp": "datetime",        # Horodatage du signal
    },
    
    # Structure des paramètres pour les événements de trading
    EventType.BEFORE_TRADE: {
        "symbol": str,                  # Symbole
        "order_type": str,              # Type d'ordre (market, limit, etc.)
        "action": str,                  # Action (buy, sell)
        "quantity": float,              # Quantité
        "price": Optional[float],       # Prix (optionnel pour les ordres market)
        "reason": str,                  # Raison du trade
    },
    EventType.AFTER_TRADE: {
        "symbol": str,                  # Symbole
        "order_type": str,              # Type d'ordre
        "action": str,                  # Action
        "quantity": float,              # Quantité
        "price": float,                 # Prix d'exécution
        "order_id": str,                # ID de l'ordre
        "status": str,                  # Statut (filled, partial, etc.)
        "timestamp": "datetime",        # Horodatage
    },
    EventType.TRADE_CLOSED: {
        "symbol": str,                  # Symbole
        "profit_loss": float,           # Profit/Perte
        "entry_price": float,           # Prix d'entrée
        "exit_price": float,            # Prix de sortie
        "duration": "timedelta",        # Durée de la position
        "reason": str,                  # Raison de la clôture
    },
    EventType.PORTFOLIO_UPDATED: {
        "balance": float,               # Solde total
        "positions": Dict,              # Positions actuelles
        "pnl": float,                   # Profit/Perte net
        "drawdown": float,              # Drawdown actuel
    },
    
    # Structure des paramètres pour les événements de données
    EventType.MARKET_DATA_UPDATED: {
        "symbol": str,                  # Symbole
        "timeframe": str,               # Période temporelle
        "data": "DataFrame",            # Données mises à jour
        "is_new_candle": bool,          # S'il s'agit d'une nouvelle bougie
    },
    EventType.NEWS_RECEIVED: {
        "symbol": str,                  # Symbole concerné
        "news": List[Dict],             # Liste des actualités
        "source": str,                  # Source des actualités
        "timestamp": "datetime",        # Horodatage
    },
    EventType.SENTIMENT_ANALYZED: {
        "symbol": str,                  # Symbole concerné
        "sentiment_score": float,       # Score de sentiment
        "news": List[Dict],             # Actualités analysées
        "details": Dict,                # Détails de l'analyse
    },
    
    # Structure des paramètres pour les événements système
    EventType.SYSTEM_STARTUP: {
        "config": Dict,                 # Configuration du système
        "timestamp": "datetime",        # Horodatage du démarrage
    },
    EventType.SYSTEM_SHUTDOWN: {
        "reason": str,                  # Raison de l'arrêt
        "timestamp": "datetime",        # Horodatage de l'arrêt
    },
    EventType.PLUGIN_LOADED: {
        "plugin_id": str,               # ID du plugin
        "plugin_name": str,             # Nom du plugin
    },
    EventType.PLUGIN_UNLOADED: {
        "plugin_id": str,               # ID du plugin
        "plugin_name": str,             # Nom du plugin
    },
    
    # Structure des paramètres pour les événements UI
    EventType.UI_REFRESH: {
        "component": str,               # Composant à rafraîchir
        "data": Dict,                   # Données à mettre à jour
    },
    EventType.SETTINGS_CHANGED: {
        "settings": Dict,               # Paramètres modifiés
        "source": str,                  # Source de la modification
    },
    EventType.USER_ACTION: {
        "action": str,                  # Action effectuée
        "params": Dict,                 # Paramètres de l'action
        "user_id": Optional[str],       # ID de l'utilisateur (optionnel)
    },
}

# Ajouter les paramètres pour les événements spécifiques aux actualités
NEWS_EVENT_PARAMETERS = {
    NewsEventType.BEFORE_NEWS_FETCH: {
        "symbol": str,                  # Symbole
        "sources": List[str],           # Sources à consulter
        "max_results": int,             # Nombre maximum de résultats
    },
    NewsEventType.AFTER_NEWS_FETCH: {
        "symbol": str,                  # Symbole
        "news": List[Dict],             # Actualités récupérées
        "sources": List[str],           # Sources consultées
        "timestamp": "datetime",        # Horodatage
    },
    NewsEventType.BEFORE_SENTIMENT_ANALYSIS: {
        "symbol": str,                  # Symbole
        "news": List[Dict],             # Actualités à analyser
        "analysis_method": str,         # Méthode d'analyse
    },
    NewsEventType.AFTER_SENTIMENT_ANALYSIS: {
        "symbol": str,                  # Symbole
        "news": List[Dict],             # Actualités analysées
        "sentiment_results": Dict,      # Résultats de l'analyse
        "overall_score": float,         # Score global
    },
    NewsEventType.HIGH_IMPACT_NEWS_DETECTED: {
        "symbol": str,                  # Symbole
        "news_item": Dict,              # Actualité à fort impact
        "impact_score": float,          # Score d'impact
        "timestamp": "datetime",        # Horodatage
    },
} 