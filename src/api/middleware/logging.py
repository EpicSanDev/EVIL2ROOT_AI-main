"""
Middleware de journalisation pour l'API du bot de trading.

Ce module enregistre les requêtes et réponses API à des fins de débogage et d'audit.
"""

import time
import logging
import json
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Dict, Any, Optional

# Configuration du logger
logger = logging.getLogger("api.requests")

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware pour journaliser les requêtes et réponses API."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next):
        """
        Journalise les requêtes et réponses API.
        
        Args:
            request: La requête entrante
            call_next: Fonction pour passer la requête au middleware suivant
            
        Returns:
            La réponse HTTP
        """
        # Générer un ID de requête unique
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Enregistrer le début de la requête
        start_time = time.time()
        
        # Préparer les informations de requête
        request_info = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
        }
        
        # Journaliser la requête entrante
        logger.info(f"Requête reçue: {json.dumps(request_info)}")
        
        try:
            # Traiter la requête
            response = await call_next(request)
            
            # Ajouter l'ID de requête aux en-têtes de réponse
            response.headers["X-Request-ID"] = request_id
            
            # Calculer la durée de traitement
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            # Préparer les informations de réponse
            response_info = {
                "request_id": request_id,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
            }
            
            # Journaliser la réponse
            log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
            logger.log(log_level, f"Réponse envoyée: {json.dumps(response_info)}")
            
            return response
            
        except Exception as e:
            # En cas d'erreur, journaliser l'exception
            error_info = {
                "request_id": request_id,
                "exception": str(e),
                "process_time_ms": round((time.time() - start_time) * 1000, 2),
            }
            
            logger.error(f"Erreur de traitement: {json.dumps(error_info)}", exc_info=True)
            
            # Propager l'exception
            raise 