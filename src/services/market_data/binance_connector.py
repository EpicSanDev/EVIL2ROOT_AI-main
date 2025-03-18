import time
import hmac
import hashlib
import json
import logging
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Union, Callable
from datetime import datetime
import uuid
from .base_connector import BaseMarketConnector
from .websocket_handler import WebsocketHandler

class BinanceConnector(BaseMarketConnector):
    """
    Connecteur pour l'échange Binance, permettant l'accès aux données de marché en temps réel.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 testnet: bool = False, timeout: int = 10):
        """
        Initialise le connecteur Binance.
        
        Args:
            api_key (str, optional): La clé API pour l'authentification.
            api_secret (str, optional): Le secret API pour l'authentification.
            testnet (bool): Si True, utilise le réseau de test au lieu du réseau principal.
            timeout (int): Délai d'attente pour les requêtes en secondes.
        """
        super().__init__(api_key, api_secret, testnet, timeout)
        
        # URLs de base pour les APIs REST et WebSocket
        if testnet:
            self.base_url = "https://testnet.binance.vision/api"
            self.ws_base_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "https://api.binance.com/api"
            self.ws_base_url = "wss://stream.binance.com:9443/ws"
        
        self.ws_handler = None
        self.stream_id_counter = 1
        self.active_streams = {}
        
    def _generate_signature(self, data: Dict) -> str:
        """
        Génère une signature HMAC-SHA256 pour l'authentification.
        
        Args:
            data (Dict): Données à signer.
            
        Returns:
            str: Signature encodée en hexadécimal.
        """
        query_string = '&'.join([f"{k}={v}" for k, v in data.items()])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                      signed: bool = False, version: str = 'v3') -> Dict:
        """
        Effectue une requête API REST à Binance.
        
        Args:
            method (str): Méthode HTTP (GET, POST, etc.).
            endpoint (str): Point de terminaison API.
            params (Dict, optional): Paramètres de requête.
            signed (bool): Si True, la requête nécessite une signature.
            version (str): Version de l'API.
            
        Returns:
            Dict: Réponse de l'API.
            
        Raises:
            Exception: Si la requête échoue.
        """
        url = f"{self.base_url}/{version}/{endpoint}"
        headers = {}
        
        if params is None:
            params = {}
        
        if self.api_key and (signed or endpoint.startswith('user')):
            headers['X-MBX-APIKEY'] = self.api_key
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.handle_error(e, f"Erreur lors de la requête Binance ({endpoint})")
            raise
    
    def connect(self) -> bool:
        """
        Établit les connexions nécessaires et initialise le gestionnaire WebSocket.
        
        Returns:
            bool: True si la connexion est établie, False sinon.
        """
        try:
            # Vérification de la connectivité avec un ping simple
            self._make_request('GET', 'ping')
            
            # Initialisation du gestionnaire WebSocket
            if self.ws_handler is None:
                self.ws_handler = WebsocketHandler(
                    url=self.ws_base_url,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close,
                    on_open=self._on_ws_open
                )
                
                if not self.ws_handler.connect():
                    return False
            
            self.connected = True
            self.update_heartbeat()
            return True
        except Exception as e:
            self.handle_error(e, "Erreur lors de la connexion à Binance")
            return False
    
    def disconnect(self) -> bool:
        """
        Ferme toutes les connexions actives.
        
        Returns:
            bool: True si la déconnexion est réussie, False sinon.
        """
        try:
            if self.ws_handler:
                self.ws_handler.disconnect()
                self.ws_handler = None
            
            self.connected = False
            return True
        except Exception as e:
            self.handle_error(e, "Erreur lors de la déconnexion de Binance")
            return False
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Récupère les informations de ticker pour un symbole donné.
        
        Args:
            symbol (str): Le symbole pour lequel récupérer le ticker.
            
        Returns:
            Dict: Informations de ticker incluant prix, volume, etc.
        """
        symbol = self.validate_symbol(symbol)
        return self._make_request('GET', '24hr', {'symbol': symbol})
    
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """
        Récupère le carnet d'ordres pour un symbole donné.
        
        Args:
            symbol (str): Le symbole pour lequel récupérer le carnet d'ordres.
            limit (int): Nombre de niveaux de prix à récupérer.
            
        Returns:
            Dict: Carnet d'ordres avec offres d'achat et de vente.
        """
        symbol = self.validate_symbol(symbol)
        return self._make_request('GET', 'depth', {'symbol': symbol, 'limit': limit})
    
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
        symbol = self.validate_symbol(symbol)
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        # Convertir les datetimes en timestamps si nécessaire
        if start_time:
            if isinstance(start_time, datetime):
                start_time = int(start_time.timestamp() * 1000)
            params['startTime'] = start_time
            
        if end_time:
            if isinstance(end_time, datetime):
                end_time = int(end_time.timestamp() * 1000)
            params['endTime'] = end_time
        
        klines = self._make_request('GET', 'klines', params)
        
        # Convertir en DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convertir les types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                           'quote_asset_volume', 'taker_buy_base_asset_volume', 
                           'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Convertir les timestamps en datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Renommer les colonnes pour plus de clarté
        df = df.rename(columns={
            'timestamp': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        
        return df
    
    def _on_ws_message(self, ws, message):
        """
        Gère les messages reçus via WebSocket.
        
        Args:
            ws: Instance WebSocketApp.
            message: Le message reçu.
        """
        if isinstance(message, dict):
            data = message
        else:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                self.logger.warning(f"Message WebSocket non-JSON reçu: {message}")
                return
        
        # Mise à jour du heartbeat
        self.update_heartbeat()
        
        # Traitement des différents types de messages
        if 'stream' in data:
            stream_name = data['stream']
            if stream_name in self.active_streams:
                stream_info = self.active_streams[stream_name]
                if 'callback' in stream_info and stream_info['callback']:
                    stream_info['callback'](data['data'])
        elif 'e' in data:  # Message de type événement
            event_type = data['e']
            if event_type == 'error':
                self.logger.error(f"Erreur WebSocket Binance: {data}")
            elif event_type in ['trade', 'kline', 'ticker', 'depthUpdate']:
                symbol = data['s']
                stream_key = f"{event_type}_{symbol.lower()}"
                if stream_key in self.active_streams:
                    stream_info = self.active_streams[stream_key]
                    if 'callback' in stream_info and stream_info['callback']:
                        stream_info['callback'](data)
    
    def _on_ws_error(self, ws, error):
        """
        Gère les erreurs WebSocket.
        
        Args:
            ws: Instance WebSocketApp.
            error: L'erreur survenue.
        """
        self.handle_error(error, "Erreur WebSocket Binance")
    
    def _on_ws_close(self, ws, close_status_code, close_reason):
        """
        Gère la fermeture de la connexion WebSocket.
        
        Args:
            ws: Instance WebSocketApp.
            close_status_code: Code de statut de fermeture.
            close_reason: Raison de la fermeture.
        """
        self.connected = False
        self.logger.info(f"Connexion WebSocket Binance fermée: {close_status_code} - {close_reason}")
    
    def _on_ws_open(self, ws):
        """
        Gère l'ouverture de la connexion WebSocket.
        
        Args:
            ws: Instance WebSocketApp.
        """
        self.connected = True
        self.logger.info("Connexion WebSocket Binance établie")
        self.update_heartbeat()
        
        # Réabonnement aux flux actifs
        for stream_name, stream_info in self.active_streams.items():
            if 'subscribe_msg' in stream_info:
                self.ws_handler.send(stream_info['subscribe_msg'])
                self.logger.info(f"Réabonné au flux: {stream_name}")
    
    def _create_subscription(self, stream_name: str, params: Dict, callback: Callable) -> str:
        """
        Crée un abonnement à un flux de données.
        
        Args:
            stream_name (str): Nom du flux.
            params (Dict): Paramètres pour l'abonnement.
            callback (Callable): Fonction de rappel pour les données.
            
        Returns:
            str: Identifiant de l'abonnement.
        """
        # Générer un ID unique pour cet abonnement
        sub_id = str(self.stream_id_counter)
        self.stream_id_counter += 1
        
        # Créer les messages d'abonnement et de désabonnement
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": params,
            "id": sub_id
        }
        
        unsubscribe_msg = {
            "method": "UNSUBSCRIBE",
            "params": params,
            "id": sub_id
        }
        
        # Stocker les informations sur l'abonnement
        self.active_streams[stream_name] = {
            'sub_id': sub_id,
            'params': params,
            'callback': callback,
            'subscribe_msg': subscribe_msg,
            'unsubscribe_msg': unsubscribe_msg
        }
        
        # Ajouter l'abonnement au gestionnaire WebSocket
        if self.connected and self.ws_handler:
            self.ws_handler.add_subscription(
                sub_id=sub_id,
                subscribe_msg=subscribe_msg,
                unsubscribe_msg=unsubscribe_msg
            )
            self.ws_handler.send(subscribe_msg)
        
        return sub_id
    
    def subscribe_to_ticker(self, symbol: str, callback: Callable) -> bool:
        """
        S'abonne aux mises à jour du ticker en temps réel.
        
        Args:
            symbol (str): Le symbole pour lequel s'abonner.
            callback (Callable): Fonction appelée à chaque mise à jour.
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon.
        """
        if not self.connected and not self.connect():
            return False
        
        symbol = self.validate_symbol(symbol)
        stream_name = f"ticker_{symbol.lower()}"
        params = [f"{symbol.lower()}@ticker"]
        
        try:
            self._create_subscription(stream_name, params, callback)
            return True
        except Exception as e:
            self.handle_error(e, f"Erreur lors de l'abonnement au ticker {symbol}")
            return False
    
    def subscribe_to_klines(self, symbol: str, interval: str, callback: Callable) -> bool:
        """
        S'abonne aux mises à jour des bougies en temps réel.
        
        Args:
            symbol (str): Le symbole pour lequel s'abonner.
            interval (str): L'intervalle de temps (1m, 5m, 15m, 1h, etc.).
            callback (Callable): Fonction appelée à chaque mise à jour.
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon.
        """
        if not self.connected and not self.connect():
            return False
        
        symbol = self.validate_symbol(symbol)
        stream_name = f"kline_{symbol.lower()}_{interval}"
        params = [f"{symbol.lower()}@kline_{interval}"]
        
        try:
            self._create_subscription(stream_name, params, callback)
            return True
        except Exception as e:
            self.handle_error(e, f"Erreur lors de l'abonnement aux klines {symbol} {interval}")
            return False
    
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
        if not self.connected and not self.connect():
            return False
        
        symbol = self.validate_symbol(symbol)
        stream_name = f"depth_{symbol.lower()}_{depth}"
        params = [f"{symbol.lower()}@depth{depth}"]
        
        try:
            self._create_subscription(stream_name, params, callback)
            return True
        except Exception as e:
            self.handle_error(e, f"Erreur lors de l'abonnement au carnet d'ordres {symbol}")
            return False
    
    def subscribe_to_trades(self, symbol: str, callback: Callable) -> bool:
        """
        S'abonne aux mises à jour des transactions en temps réel.
        
        Args:
            symbol (str): Le symbole pour lequel s'abonner.
            callback (Callable): Fonction appelée à chaque mise à jour.
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon.
        """
        if not self.connected and not self.connect():
            return False
        
        symbol = self.validate_symbol(symbol)
        stream_name = f"trade_{symbol.lower()}"
        params = [f"{symbol.lower()}@trade"]
        
        try:
            self._create_subscription(stream_name, params, callback)
            return True
        except Exception as e:
            self.handle_error(e, f"Erreur lors de l'abonnement aux trades {symbol}")
            return False
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Se désabonne d'un flux de données.
        
        Args:
            subscription_id (str): L'identifiant de l'abonnement à annuler.
            
        Returns:
            bool: True si le désabonnement est réussi, False sinon.
        """
        # Trouver le flux correspondant à cet ID
        stream_to_remove = None
        for stream_name, stream_info in self.active_streams.items():
            if stream_info.get('sub_id') == subscription_id:
                stream_to_remove = stream_name
                break
        
        if not stream_to_remove:
            self.logger.warning(f"Aucun abonnement trouvé avec l'ID {subscription_id}")
            return False
        
        # Envoyer le message de désabonnement
        if self.connected and self.ws_handler:
            unsubscribe_msg = self.active_streams[stream_to_remove]['unsubscribe_msg']
            self.ws_handler.send(unsubscribe_msg)
        
        # Supprimer de la liste des flux actifs
        del self.active_streams[stream_to_remove]
        
        return True
    
    def validate_symbol(self, symbol: str) -> str:
        """
        Valide et formate un symbole selon les conventions de Binance.
        
        Args:
            symbol (str): Le symbole à valider.
            
        Returns:
            str: Le symbole formaté.
        """
        # Binance utilise des symboles en majuscules sans séparateur
        return symbol.upper().replace('/', '').replace('-', '').replace('_', '') 