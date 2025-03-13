import redis
import time
import logging
import backoff
import os
from typing import Dict, Any, Optional
from functools import wraps

# Configure le logging
logger = logging.getLogger(__name__)

class RedisConnectionError(Exception):
    """Exception levée en cas d'erreur de connexion Redis."""
    pass

class RedisManager:
    """
    Gestionnaire de connexions Redis avec mécanismes de résilience.
    Fournit une gestion des erreurs améliorée et des tentatives de reconnexion.
    """
    
    def __init__(self, host=None, port=None, db=0, password=None, 
                 max_retries=5, retry_delay=1, socket_timeout=5):
        """
        Initialise le gestionnaire de connexions Redis.
        
        Args:
            host: Hôte Redis (défaut: variable d'environnement REDIS_HOST ou localhost)
            port: Port Redis (défaut: variable d'environnement REDIS_PORT ou 6379)
            db: Base de données Redis (défaut: 0)
            password: Mot de passe Redis (défaut: variable d'environnement REDIS_PASSWORD)
            max_retries: Nombre maximal de tentatives de connexion (défaut: 5)
            retry_delay: Délai entre les tentatives en secondes (défaut: 1)
            socket_timeout: Délai d'expiration de la connexion en secondes (défaut: 5)
        """
        self.host = host or os.environ.get('REDIS_HOST', 'localhost')
        self.port = int(port or os.environ.get('REDIS_PORT', 6379))
        self.db = db
        self.password = password or os.environ.get('REDIS_PASSWORD', None)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.socket_timeout = socket_timeout
        self._client = None
        self._pubsub = None
        self._connected = False
        
    @property
    def client(self) -> redis.Redis:
        """
        Fournit un client Redis connecté. Si la connexion n'existe pas, 
        elle est créée automatiquement.
        
        Returns:
            Client Redis
            
        Raises:
            RedisConnectionError: Si la connexion échoue après toutes les tentatives
        """
        if not self._client or not self._connected:
            self._connect()
        return self._client
        
    def _connect(self) -> None:
        """
        Établit une connexion au serveur Redis avec un mécanisme de retry.
        
        Raises:
            RedisConnectionError: Si la connexion échoue après toutes les tentatives
        """
        retry_count = 0
        last_exception = None
        
        while retry_count < self.max_retries:
            try:
                self._client = redis.Redis(
                    host=self.host, 
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_timeout=self.socket_timeout,
                    decode_responses=True
                )
                
                # Vérifie que la connexion fonctionne
                self._client.ping()
                self._connected = True
                logger.info(f"Connecté au serveur Redis à {self.host}:{self.port}")
                return
                
            except (redis.ConnectionError, redis.TimeoutError) as e:
                retry_count += 1
                last_exception = e
                wait_time = self.retry_delay * (2 ** (retry_count - 1))  # Backoff exponentiel
                logger.warning(f"Échec de la connexion Redis (tentative {retry_count}/{self.max_retries}): {e}. Nouvelle tentative dans {wait_time}s...")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Erreur inattendue lors de la connexion à Redis: {e}")
                raise RedisConnectionError(f"Erreur inattendue lors de la connexion à Redis: {e}")
        
        logger.error(f"Impossible de se connecter à Redis après {self.max_retries} tentatives. Dernière erreur: {last_exception}")
        raise RedisConnectionError(f"Échec de la connexion Redis après {self.max_retries} tentatives: {last_exception}")
    
    def check_connection(self) -> bool:
        """
        Vérifie si la connexion Redis est active.
        
        Returns:
            True si la connexion est active, False sinon
        """
        try:
            if self._client:
                self._client.ping()
                return True
            return False
        except Exception:
            self._connected = False
            return False
            
    def reconnect(self) -> None:
        """
        Force une reconnexion à Redis.
        
        Raises:
            RedisConnectionError: Si la reconnexion échoue
        """
        self._connected = False
        self._connect()
        
    def get_pubsub(self) -> redis.client.PubSub:
        """
        Récupère un objet PubSub pour les opérations de publication/abonnement.
        
        Returns:
            Objet PubSub Redis
        """
        if not self._pubsub:
            self._pubsub = self.client.pubsub()
        return self._pubsub
        
    def safe_publish(self, channel: str, message: str, max_retries: int = 3) -> bool:
        """
        Publie un message sur un canal avec gestion des erreurs et tentatives.
        
        Args:
            channel: Nom du canal
            message: Message à publier
            max_retries: Nombre maximal de tentatives (défaut: 3)
            
        Returns:
            True si le message a été publié avec succès, False sinon
        """
        for attempt in range(max_retries):
            try:
                if not self.check_connection():
                    self.reconnect()
                self.client.publish(channel, message)
                return True
            except Exception as e:
                logger.warning(f"Échec de publication sur Redis (tentative {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    self._connected = False  # Force une reconnexion
                    
        logger.error(f"Échec de publication du message sur le canal {channel} après {max_retries} tentatives")
        return False
        
    def safe_get(self, key: str, max_retries: int = 3) -> Optional[str]:
        """
        Récupère une valeur avec gestion des erreurs et tentatives.
        
        Args:
            key: Clé à récupérer
            max_retries: Nombre maximal de tentatives (défaut: 3)
            
        Returns:
            Valeur associée à la clé ou None si erreur/non existante
        """
        for attempt in range(max_retries):
            try:
                if not self.check_connection():
                    self.reconnect()
                return self.client.get(key)
            except Exception as e:
                logger.warning(f"Échec de récupération sur Redis (tentative {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    self._connected = False
                    
        logger.error(f"Échec de récupération de la clé {key} après {max_retries} tentatives")
        return None
        
    def safe_set(self, key: str, value: str, ex: Optional[int] = None, max_retries: int = 3) -> bool:
        """
        Définit une valeur avec gestion des erreurs et tentatives.
        
        Args:
            key: Clé à définir
            value: Valeur à associer
            ex: Délai d'expiration en secondes (optionnel)
            max_retries: Nombre maximal de tentatives (défaut: 3)
            
        Returns:
            True si la valeur a été définie avec succès, False sinon
        """
        for attempt in range(max_retries):
            try:
                if not self.check_connection():
                    self.reconnect()
                self.client.set(key, value, ex=ex)
                return True
            except Exception as e:
                logger.warning(f"Échec de définition sur Redis (tentative {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    self._connected = False
                    
        logger.error(f"Échec de définition de la clé {key} après {max_retries} tentatives")
        return False

# Créer une instance globale par défaut
redis_manager = RedisManager()

# Décorateur pour réessayer les opérations Redis en cas d'échec
def retry_redis_operation(max_retries=3, retry_delay=1):
    """
    Décorateur pour réessayer les opérations Redis qui échouent.
    
    Args:
        max_retries: Nombre maximal de tentatives
        retry_delay: Délai initial entre les tentatives (augmente exponentiellement)
        
    Returns:
        Décorateur pour fonction
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (redis.ConnectionError, redis.TimeoutError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        logger.warning(f"Échec de l'opération Redis (tentative {attempt+1}/{max_retries}): {e}. Nouvelle tentative dans {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Échec de l'opération Redis après {max_retries} tentatives: {e}")
            
            # Si toutes les tentatives ont échoué
            raise RedisConnectionError(f"L'opération a échoué après {max_retries} tentatives: {last_error}")
        return wrapper
    return decorator 