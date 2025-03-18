"""
Middleware de limitation de débit pour l'API du bot de trading.

Ce module implémente un système de limitation de requêtes basé sur Redis.
"""

import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis
import json
import logging
import os
from dotenv import load_dotenv
from typing import Dict, Tuple, Optional

# Charger les variables d'environnement
load_dotenv()

# Configuration Redis
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_RATE_LIMIT_DB", "1"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Configuration de limitation
DEFAULT_RATE_LIMIT = 100  # requêtes par minute
API_RATE_LIMIT = 1000  # requêtes par minute
PUBLIC_ENDPOINTS_RATE_LIMIT = 30  # requêtes par minute

# Configuration du logger
logger = logging.getLogger("api.rate_limit")

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware pour limiter le nombre de requêtes par utilisateur."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        try:
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            # Tester la connexion
            self.redis.ping()
            logger.info("Connexion Redis établie pour la limitation de débit")
        except redis.ConnectionError as e:
            logger.warning(f"Impossible de se connecter à Redis pour la limitation de débit: {e}")
            self.redis = None
            
        # Définir les limites par type d'endpoint
        self.endpoint_limits = {
            "/api/auth": PUBLIC_ENDPOINTS_RATE_LIMIT,
            "/api/trading": API_RATE_LIMIT,
            "/api/dashboard": DEFAULT_RATE_LIMIT,
            "/api/settings": DEFAULT_RATE_LIMIT,
            "/api/subscriptions": DEFAULT_RATE_LIMIT,
            "/api/backtest": DEFAULT_RATE_LIMIT,
        }
        
    async def dispatch(self, request: Request, call_next):
        """
        Traite la requête et vérifie les limites de débit.
        
        Args:
            request: La requête entrante
            call_next: Fonction pour passer la requête au middleware suivant
            
        Returns:
            La réponse HTTP
        """
        # Si Redis n'est pas disponible, on passe la requête sans limitation
        if not self.redis:
            return await call_next(request)
            
        # Obtenir une clé unique pour l'utilisateur ou l'IP
        client_key = self._get_client_key(request)
        
        # Déterminer la limite applicable
        rate_limit = self._get_rate_limit(request)
        
        # Vérifier et incrémenter le compteur
        current, ttl = self._increment_request_count(client_key, rate_limit)
        
        # Préparer les en-têtes pour les limites
        headers = {
            "X-RateLimit-Limit": str(rate_limit),
            "X-RateLimit-Remaining": str(max(0, rate_limit - current)),
            "X-RateLimit-Reset": str(int(time.time() + ttl)),
        }
        
        # Si la limite est dépassée, renvoyer une erreur 429
        if current > rate_limit:
            response = Response(
                content=json.dumps({
                    "detail": "Trop de requêtes. Veuillez réessayer plus tard."
                }),
                status_code=429,
                media_type="application/json",
                headers=headers
            )
            return response
            
        # Sinon, continuer le traitement
        response = await call_next(request)
        
        # Ajouter les en-têtes à la réponse
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value
            
        return response
        
    def _get_client_key(self, request: Request) -> str:
        """
        Obtient une clé unique pour identifier le client.
        
        Args:
            request: La requête entrante
            
        Returns:
            Une clé unique pour le client
        """
        # Si l'utilisateur est authentifié, utiliser son ID
        user = getattr(request.state, "user", None)
        if user:
            return f"ratelimit:user:{user.id}"
            
        # Sinon, utiliser l'adresse IP
        client_ip = request.client.host if request.client else "unknown"
        return f"ratelimit:ip:{client_ip}"
        
    def _get_rate_limit(self, request: Request) -> int:
        """
        Détermine la limite de débit applicable pour cette requête.
        
        Args:
            request: La requête entrante
            
        Returns:
            La limite de requêtes par minute
        """
        # Vérifier si l'utilisateur a une limite personnalisée
        user = getattr(request.state, "user", None)
        if user and hasattr(user, "rate_limit") and user.rate_limit:
            return user.rate_limit
            
        # Sinon, utiliser la limite basée sur le chemin
        path = request.url.path
        for prefix, limit in self.endpoint_limits.items():
            if path.startswith(prefix):
                return limit
                
        # Limite par défaut
        return DEFAULT_RATE_LIMIT
        
    def _increment_request_count(self, key: str, limit: int) -> Tuple[int, int]:
        """
        Incrémente et vérifie le compteur de requêtes pour la clé donnée.
        
        Args:
            key: La clé unique du client
            limit: La limite de requêtes applicable
            
        Returns:
            Un tuple (nombre actuel de requêtes, TTL en secondes)
        """
        pipe = self.redis.pipeline()
        
        # Compteur sur une fenêtre de 60 secondes
        current_time = int(time.time())
        window_key = f"{key}:{current_time // 60}"
        
        try:
            # Incrémenter le compteur
            pipe.incr(window_key)
            # Définir l'expiration si elle n'est pas déjà définie
            pipe.expire(window_key, 60)
            # Exécuter les commandes
            result = pipe.execute()
            
            # Récupérer le nombre actuel et le TTL
            current_count = result[0]
            ttl = self.redis.ttl(window_key)
            
            return current_count, ttl
            
        except redis.RedisError as e:
            logger.error(f"Erreur Redis lors de la limitation de débit: {e}")
            # En cas d'erreur, autoriser la requête
            return 0, 60 