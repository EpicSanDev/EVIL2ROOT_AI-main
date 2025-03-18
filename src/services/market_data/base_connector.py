from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta

class BaseMarketConnector(ABC):
    """
    Classe de base abstraite pour tous les connecteurs de marché en temps réel.
    
    Cette classe définit l'interface commune à tous les connecteurs spécifiques
    à différentes plateformes d'échange (Binance, Coinbase, etc.).
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 testnet: bool = False, timeout: int = 10):
        """
        Initialise un connecteur de marché de base.
        
        Args:
            api_key (str, optional): La clé API pour l'authentification.
            api_secret (str, optional): Le secret API pour l'authentification.
            testnet (bool): Si True, utilise le réseau de test au lieu du réseau principal.
            timeout (int): Délai d'attente pour les requêtes en secondes.
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connected = False
        self.last_heartbeat = datetime.now()
        self.subscriptions = {}
        self.callbacks = {}
        
    @abstractmethod
    def connect(self) -> bool:
        """
        Établit une connexion avec l'API de l'échange.
        
        Returns:
            bool: True si la connexion est établie avec succès, False sinon.
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Ferme la connexion avec l'API de l'échange.
        
        Returns:
            bool: True si la déconnexion est réussie, False sinon.
        """
        pass
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict:
        """
        Récupère les informations de ticker pour un symbole donné.
        
        Args:
            symbol (str): Le symbole pour lequel récupérer le ticker.
            
        Returns:
            Dict: Informations de ticker incluant prix, volume, etc.
        """
        pass
    
    @abstractmethod
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """
        Récupère le carnet d'ordres pour un symbole donné.
        
        Args:
            symbol (str): Le symbole pour lequel récupérer le carnet d'ordres.
            limit (int): Nombre de niveaux de prix à récupérer.
            
        Returns:
            Dict: Carnet d'ordres avec offres d'achat et de vente.
        """
        pass
    
    @abstractmethod
    def get_historical_klines(self, symbol: str, interval: str, 
                              start_time: Optional[Union[int, datetime]] = None,
                              end_time: Optional[Union[int, datetime]] = None, 
                              limit: int = 500) -> pd.DataFrame:
        """
        Récupère les données historiques de bougies (OHLCV) pour un symbole.
        
        Args:
            symbol (str): Le symbole pour lequel récupérer les données.
            interval (str): L'intervalle de temps (1m, 5m, 15m, 1h, 4h, 1d, etc.).
            start_time: Heure de début pour les données.
            end_time: Heure de fin pour les données.
            limit (int): Nombre maximum d'enregistrements à récupérer.
            
        Returns:
            pd.DataFrame: DataFrame contenant les données OHLCV.
        """
        pass
    
    @abstractmethod
    def subscribe_to_ticker(self, symbol: str, callback: Callable) -> bool:
        """
        S'abonne aux mises à jour du ticker en temps réel.
        
        Args:
            symbol (str): Le symbole pour lequel s'abonner.
            callback (Callable): Fonction appelée à chaque mise à jour.
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon.
        """
        pass
    
    @abstractmethod
    def subscribe_to_klines(self, symbol: str, interval: str, callback: Callable) -> bool:
        """
        S'abonne aux mises à jour des bougies en temps réel.
        
        Args:
            symbol (str): Le symbole pour lequel s'abonner.
            interval (str): L'intervalle de temps (1m, 5m, 15m, 1h, 4h, 1d, etc.).
            callback (Callable): Fonction appelée à chaque mise à jour.
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon.
        """
        pass
    
    @abstractmethod
    def subscribe_to_orderbook(self, symbol: str, callback: Callable, depth: str = "10") -> bool:
        """
        S'abonne aux mises à jour du carnet d'ordres en temps réel.
        
        Args:
            symbol (str): Le symbole pour lequel s'abonner.
            callback (Callable): Fonction appelée à chaque mise à jour.
            depth (str): Profondeur du carnet d'ordres (5, 10, 20, etc.).
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon.
        """
        pass
    
    @abstractmethod
    def subscribe_to_trades(self, symbol: str, callback: Callable) -> bool:
        """
        S'abonne aux mises à jour des transactions en temps réel.
        
        Args:
            symbol (str): Le symbole pour lequel s'abonner.
            callback (Callable): Fonction appelée à chaque mise à jour.
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon.
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Se désabonne d'un flux de données.
        
        Args:
            subscription_id (str): L'identifiant de l'abonnement à annuler.
            
        Returns:
            bool: True si le désabonnement est réussi, False sinon.
        """
        pass
    
    def ping(self) -> bool:
        """
        Vérifie si la connexion est toujours active.
        
        Returns:
            bool: True si la connexion est active, False sinon.
        """
        # Implémentation par défaut - à surcharger si nécessaire
        if not self.connected:
            return False
        
        # Si le dernier battement de cœur date de plus de 30 secondes, considérez la connexion comme perdue
        if (datetime.now() - self.last_heartbeat).total_seconds() > 30:
            self.connected = False
            return False
        
        return True
    
    def update_heartbeat(self) -> None:
        """Met à jour le timestamp du dernier heartbeat"""
        self.last_heartbeat = datetime.now()
    
    def validate_symbol(self, symbol: str) -> str:
        """
        Valide et formate un symbole selon les conventions de l'échange.
        
        Args:
            symbol (str): Le symbole à valider.
            
        Returns:
            str: Le symbole formaté.
        """
        # Implémentation par défaut - à surcharger avec les règles spécifiques à chaque échange
        return symbol.upper()
    
    def handle_error(self, error: Exception, message: str = "") -> None:
        """
        Gère les erreurs de connexion et de requête.
        
        Args:
            error (Exception): L'erreur à gérer.
            message (str): Message supplémentaire à logger.
        """
        error_msg = f"{message}: {str(error)}" if message else str(error)
        self.logger.error(error_msg)
        
        # Si l'erreur est liée à la connexion, marquez-la comme perdue
        if "connect" in str(error).lower() or "timeout" in str(error).lower():
            self.connected = False
    
    def is_market_open(self, symbol: str) -> bool:
        """
        Vérifie si le marché est ouvert pour un symbole donné.
        
        Args:
            symbol (str): Le symbole à vérifier.
            
        Returns:
            bool: True si le marché est ouvert, False sinon.
        """
        # Pour les marchés crypto qui sont ouverts 24/7
        return True 