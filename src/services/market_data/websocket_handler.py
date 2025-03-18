import json
import websocket
import threading
import logging
from typing import Dict, List, Callable, Optional, Any, Union
from datetime import datetime
import time
import ssl
import traceback

class WebsocketHandler:
    """
    Classe de gestion des connexions WebSocket pour les flux de données en temps réel.
    
    Cette classe fournit une interface commune pour gérer les connexions WebSocket
    à différents échanges de cryptomonnaies.
    """
    
    def __init__(self, url: str, on_message: Callable = None, on_error: Callable = None, 
                 on_close: Callable = None, on_open: Callable = None,
                 ping_interval: int = 30, ping_timeout: int = 10,
                 reconnect_delay: int = 5, max_reconnect_attempts: int = 10):
        """
        Initialise un gestionnaire de WebSocket.
        
        Args:
            url (str): L'URL du serveur WebSocket.
            on_message (Callable): Fonction appelée à la réception d'un message.
            on_error (Callable): Fonction appelée en cas d'erreur.
            on_close (Callable): Fonction appelée à la fermeture de la connexion.
            on_open (Callable): Fonction appelée à l'ouverture de la connexion.
            ping_interval (int): Intervalle en secondes entre les pings.
            ping_timeout (int): Délai d'attente en secondes pour un pong.
            reconnect_delay (int): Délai en secondes entre les tentatives de reconnexion.
            max_reconnect_attempts (int): Nombre maximum de tentatives de reconnexion.
        """
        self.url = url
        self.user_on_message = on_message
        self.user_on_error = on_error
        self.user_on_close = on_close
        self.user_on_open = on_open
        
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self.ws = None
        self.ws_thread = None
        self.is_connected = False
        self.reconnect_count = 0
        self.last_msg_time = None
        self.subscriptions = {}  # id -> subscription_info
        self.shutdown_requested = False
        
        self.logger = logging.getLogger(__name__)
    
    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        """
        Gestionnaire interne d'événements de message.
        
        Args:
            ws (websocket.WebSocketApp): Instance WebSocketApp.
            message (str): Message reçu.
        """
        self.last_msg_time = datetime.now()
        
        try:
            # Traitement de base - peut être personnalisé par chaque échange
            data = json.loads(message)
            
            # Si une fonction d'utilisateur est définie, la transmettre
            if self.user_on_message:
                self.user_on_message(ws, data)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Reçu un message non-JSON: {str(e)}")
            # Toujours transmettre le message brut à l'utilisateur
            if self.user_on_message:
                self.user_on_message(ws, message)
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du message WebSocket: {str(e)}")
            self.logger.debug(traceback.format_exc())
    
    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        """
        Gestionnaire interne d'événements d'erreur.
        
        Args:
            ws (websocket.WebSocketApp): Instance WebSocketApp.
            error (Exception): L'erreur survenue.
        """
        self.logger.error(f"Erreur WebSocket: {str(error)}")
        
        # Si une fonction d'utilisateur est définie, la transmettre
        if self.user_on_error:
            self.user_on_error(ws, error)
    
    def _on_close(self, ws: websocket.WebSocketApp, close_status_code: Optional[int], 
                  close_reason: Optional[str]) -> None:
        """
        Gestionnaire interne d'événements de fermeture.
        
        Args:
            ws (websocket.WebSocketApp): Instance WebSocketApp.
            close_status_code (int): Code de statut de fermeture.
            close_reason (str): Raison de la fermeture.
        """
        self.is_connected = False
        close_reason_str = close_reason if close_reason else "Raison inconnue"
        self.logger.info(f"Connexion WebSocket fermée: {close_status_code} - {close_reason_str}")
        
        # Si une fonction d'utilisateur est définie, la transmettre
        if self.user_on_close:
            self.user_on_close(ws, close_status_code, close_reason)
        
        # Tentative de reconnexion si ce n'est pas une fermeture intentionnelle
        if not self.shutdown_requested:
            self._attempt_reconnect()
    
    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        """
        Gestionnaire interne d'événements d'ouverture.
        
        Args:
            ws (websocket.WebSocketApp): Instance WebSocketApp.
        """
        self.is_connected = True
        self.reconnect_count = 0
        self.last_msg_time = datetime.now()
        self.logger.info("Connexion WebSocket établie")
        
        # Réabonnement aux flux précédents après reconnexion
        for sub_id, sub_info in self.subscriptions.items():
            self._resubscribe(sub_id, sub_info)
        
        # Si une fonction d'utilisateur est définie, la transmettre
        if self.user_on_open:
            self.user_on_open(ws)
    
    def _resubscribe(self, sub_id: str, sub_info: Dict) -> None:
        """
        Réabonne à un flux de données après reconnexion.
        
        Args:
            sub_id (str): Identifiant de l'abonnement.
            sub_info (Dict): Informations sur l'abonnement.
        """
        if "subscribe_msg" in sub_info:
            self.send(sub_info["subscribe_msg"])
            self.logger.info(f"Réabonné au flux: {sub_id}")
    
    def connect(self) -> bool:
        """
        Établit une connexion WebSocket.
        
        Returns:
            bool: True si la connexion est établie, False sinon.
        """
        if self.is_connected:
            return True
        
        self.shutdown_requested = False
        
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            self.ws_thread = threading.Thread(
                target=self.ws.run_forever,
                kwargs={
                    'ping_interval': self.ping_interval,
                    'ping_timeout': self.ping_timeout,
                    'sslopt': {"cert_reqs": ssl.CERT_NONE}
                }
            )
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Attendre que la connexion s'établisse
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < 10:
                time.sleep(0.1)
            
            return self.is_connected
        except Exception as e:
            self.logger.error(f"Erreur lors de la connexion WebSocket: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        Ferme la connexion WebSocket.
        
        Returns:
            bool: True si la déconnexion est réussie, False sinon.
        """
        self.shutdown_requested = True
        
        if self.ws:
            try:
                self.ws.close()
                self.is_connected = False
                
                # Attendre la fin du thread WebSocket
                if self.ws_thread and self.ws_thread.is_alive():
                    self.ws_thread.join(timeout=2)
                
                return True
            except Exception as e:
                self.logger.error(f"Erreur lors de la déconnexion WebSocket: {str(e)}")
                return False
        return True
    
    def send(self, message: Union[str, Dict]) -> bool:
        """
        Envoie un message via WebSocket.
        
        Args:
            message (Union[str, Dict]): Le message à envoyer.
            
        Returns:
            bool: True si l'envoi est réussi, False sinon.
        """
        if not self.is_connected:
            self.logger.warning("Tentative d'envoi de message sans connexion active")
            return False
        
        try:
            # Convertir en chaîne JSON si c'est un dictionnaire
            if isinstance(message, dict):
                message = json.dumps(message)
            
            self.ws.send(message)
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi de message WebSocket: {str(e)}")
            return False
    
    def _attempt_reconnect(self) -> None:
        """Tentative de reconnexion avec délai exponentiel."""
        if self.reconnect_count >= self.max_reconnect_attempts:
            self.logger.error(f"Échec après {self.reconnect_count} tentatives de reconnexion WebSocket")
            return
        
        # Délai exponentiel avec jitter
        delay = min(60, self.reconnect_delay * (2 ** self.reconnect_count))
        self.reconnect_count += 1
        
        self.logger.info(f"Tentative de reconnexion WebSocket dans {delay} secondes...")
        time.sleep(delay)
        
        if not self.shutdown_requested:
            self.connect()
    
    def add_subscription(self, sub_id: str, subscribe_msg: Union[str, Dict], 
                         unsubscribe_msg: Optional[Union[str, Dict]] = None,
                         callback: Optional[Callable] = None) -> bool:
        """
        Ajoute un abonnement à un flux de données.
        
        Args:
            sub_id (str): Identifiant unique pour cet abonnement.
            subscribe_msg (Union[str, Dict]): Message d'abonnement à envoyer.
            unsubscribe_msg (Union[str, Dict], optional): Message de désabonnement.
            callback (Callable, optional): Fonction de rappel spécifique à ce flux.
            
        Returns:
            bool: True si l'abonnement est réussi, False sinon.
        """
        if sub_id in self.subscriptions:
            self.logger.warning(f"L'abonnement {sub_id} existe déjà")
            return False
        
        subscription_info = {
            "subscribe_msg": subscribe_msg,
            "unsubscribe_msg": unsubscribe_msg,
            "callback": callback
        }
        
        self.subscriptions[sub_id] = subscription_info
        
        # Si connecté, envoyer immédiatement le message d'abonnement
        if self.is_connected:
            return self.send(subscribe_msg)
        
        return True
    
    def remove_subscription(self, sub_id: str) -> bool:
        """
        Supprime un abonnement à un flux de données.
        
        Args:
            sub_id (str): Identifiant de l'abonnement à supprimer.
            
        Returns:
            bool: True si le désabonnement est réussi, False sinon.
        """
        if sub_id not in self.subscriptions:
            self.logger.warning(f"L'abonnement {sub_id} n'existe pas")
            return False
        
        # Envoyer le message de désabonnement s'il existe
        if self.is_connected and "unsubscribe_msg" in self.subscriptions[sub_id]:
            self.send(self.subscriptions[sub_id]["unsubscribe_msg"])
        
        # Supprimer l'abonnement
        del self.subscriptions[sub_id]
        return True
    
    def check_connection_health(self) -> bool:
        """
        Vérifie l'état de santé de la connexion.
        
        Returns:
            bool: True si la connexion est saine, False sinon.
        """
        if not self.is_connected:
            return False
        
        # Vérifier si des messages ont été reçus récemment
        if self.last_msg_time:
            elapsed = (datetime.now() - self.last_msg_time).total_seconds()
            # Si aucun message depuis longtemps, la connexion est peut-être morte
            if elapsed > self.ping_interval * 2:
                self.logger.warning(f"Aucun message reçu depuis {elapsed} secondes, reconnexion...")
                self.disconnect()
                time.sleep(1)  # Petit délai pour laisser le temps à la déconnexion
                return self.connect()
        
        return True 