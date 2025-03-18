"""
Middleware d'authentification pour l'API du bot de trading.

Ce module gère l'authentification des requêtes API via JWT.
"""

import jwt
import time
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
import logging

from src.api.models.user import User
from src.api.database.users import get_user_by_id

# Charger les variables d'environnement
load_dotenv()

# Configuration JWT
JWT_SECRET = os.getenv("JWT_SECRET", "my_secret_key")  # À configurer en production
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 86400  # 24 heures

# Configuration du logger
logger = logging.getLogger("api.auth")

# Schéma d'authentification Bearer
security = HTTPBearer()

async def authenticate_request(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Authentifie une requête à partir du token JWT.
    
    Args:
        request: La requête HTTP
        credentials: Les informations d'authentification extraites
        
    Returns:
        Dict contenant les informations utilisateur
        
    Raises:
        HTTPException: Si l'authentification échoue
    """
    token = credentials.credentials
    
    try:
        # Décoder le token JWT
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # Vérifier l'expiration
        if payload.get("exp") < time.time():
            raise HTTPException(
                status_code=401, 
                detail="Token expiré. Veuillez vous reconnecter."
            )
        
        # Récupérer l'utilisateur
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Token invalide. Identifiant utilisateur manquant."
            )
        
        # Charger les informations utilisateur depuis la base de données
        user = await get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Utilisateur non trouvé."
            )
        
        # Vérifier que l'utilisateur est actif
        if not user.is_active:
            raise HTTPException(
                status_code=403,
                detail="Compte utilisateur désactivé."
            )
        
        # Attacher l'utilisateur à la requête pour un accès facile
        request.state.user = user
        
        return payload
        
    except jwt.PyJWTError as e:
        logger.error(f"Erreur d'authentification JWT: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Token d'authentification invalide."
        )
    except Exception as e:
        logger.error(f"Erreur d'authentification: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erreur serveur lors de l'authentification."
        )

def create_access_token(user_id: str, additional_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Crée un token JWT d'accès pour un utilisateur.
    
    Args:
        user_id: L'identifiant unique de l'utilisateur
        additional_data: Données supplémentaires à inclure dans le token
        
    Returns:
        Token JWT encodé
    """
    payload = {
        "sub": user_id,
        "iat": time.time(),
        "exp": time.time() + JWT_EXPIRATION
    }
    
    # Ajouter des données supplémentaires si fournies
    if additional_data:
        payload.update(additional_data)
    
    # Encoder et signer le token
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return token

def get_current_user(request: Request) -> User:
    """
    Récupère l'utilisateur courant à partir de la requête.
    
    Args:
        request: La requête HTTP
        
    Returns:
        L'objet utilisateur courant
        
    Raises:
        HTTPException: Si l'utilisateur n'est pas authentifié
    """
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentification requise."
        )
    return user 