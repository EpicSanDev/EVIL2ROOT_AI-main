"""
Routes d'authentification pour l'API du bot de trading.

Ce module gère l'inscription, la connexion, et la gestion des utilisateurs.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional, Dict, Any
import logging
import secrets
import bcrypt
import time
from datetime import datetime, timedelta
import uuid

from src.api.models.user import (
    UserCreate, 
    UserLogin, 
    UserResponse, 
    UserUpdate, 
    PasswordChange, 
    PasswordReset
)
from src.api.database.users import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    update_user,
    update_user_password,
    store_reset_token
)
from src.api.middleware.authentication import (
    create_access_token,
    get_current_user
)
from src.api.utils.email import send_password_reset_email, send_welcome_email

# Configuration du logger
logger = logging.getLogger("api.auth")

# Configuration du router
router = APIRouter()

# OAuth2 pour la documentation Swagger
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    background_tasks: BackgroundTasks
):
    """
    Inscription d'un nouvel utilisateur.
    
    Args:
        user_data: Les informations d'inscription
        background_tasks: Tâches à exécuter en arrière-plan
        
    Returns:
        Les informations de l'utilisateur créé
        
    Raises:
        HTTPException: Si l'email est déjà utilisé
    """
    # Vérifier si l'email existe déjà
    existing_user = await get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Cet email est déjà utilisé."
        )
    
    # Hasher le mot de passe
    hashed_password = bcrypt.hashpw(
        user_data.password.encode('utf-8'), 
        bcrypt.gensalt()
    ).decode('utf-8')
    
    # Créer un nouvel utilisateur
    user_id = str(uuid.uuid4())
    new_user = await create_user(
        id=user_id,
        email=user_data.email,
        full_name=user_data.full_name,
        hashed_password=hashed_password
    )
    
    # Envoyer un email de bienvenue en arrière-plan
    background_tasks.add_task(
        send_welcome_email,
        email=user_data.email,
        name=user_data.full_name
    )
    
    # Journaliser la création de l'utilisateur
    logger.info(f"Nouvel utilisateur créé: {user_data.email}")
    
    return new_user.to_response()

@router.post("/login", response_model=Dict[str, Any])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Connexion d'un utilisateur.
    
    Args:
        form_data: Les informations de connexion
        
    Returns:
        Le token d'accès et les informations utilisateur
        
    Raises:
        HTTPException: Si les identifiants sont invalides
    """
    # Récupérer l'utilisateur
    user = await get_user_by_email(form_data.username)
    if not user:
        # Attendre un peu pour limiter les attaques par force brute
        time.sleep(1)
        raise HTTPException(
            status_code=401,
            detail="Email ou mot de passe incorrect."
        )
    
    # Vérifier le mot de passe
    is_password_valid = bcrypt.checkpw(
        form_data.password.encode('utf-8'),
        user.hashed_password.encode('utf-8')
    )
    
    if not is_password_valid:
        # Attendre un peu pour limiter les attaques par force brute
        time.sleep(1)
        raise HTTPException(
            status_code=401,
            detail="Email ou mot de passe incorrect."
        )
    
    # Vérifier que le compte est actif
    if not user.is_active:
        raise HTTPException(
            status_code=401,
            detail="Ce compte est désactivé."
        )
    
    # Mettre à jour la date de dernière connexion
    user.last_login = datetime.now()
    await update_user(user)
    
    # Créer un token JWT
    additional_data = {
        "email": user.email,
        "name": user.full_name,
        "tier": user.subscription_tier
    }
    access_token = create_access_token(user.id, additional_data)
    
    # Journaliser la connexion
    logger.info(f"Connexion de l'utilisateur: {user.email}")
    
    # Renvoyer le token et les informations utilisateur
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user.to_response().dict()
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(request: Request):
    """
    Récupère les informations de l'utilisateur actuellement connecté.
    
    Args:
        request: La requête HTTP
        
    Returns:
        Les informations de l'utilisateur
    """
    user = get_current_user(request)
    return user.to_response()

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    request: Request
):
    """
    Met à jour les informations de l'utilisateur actuellement connecté.
    
    Args:
        user_data: Les nouvelles informations
        request: La requête HTTP
        
    Returns:
        Les informations de l'utilisateur mises à jour
        
    Raises:
        HTTPException: Si l'email est déjà utilisé
    """
    current_user = get_current_user(request)
    
    # Vérifier si l'email est déjà utilisé
    if user_data.email and user_data.email != current_user.email:
        existing_user = await get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Cet email est déjà utilisé."
            )
    
    # Mettre à jour les champs non nuls
    if user_data.full_name:
        current_user.full_name = user_data.full_name
    if user_data.email:
        current_user.email = user_data.email
    if user_data.is_active is not None:
        current_user.is_active = user_data.is_active
    
    # Enregistrer les modifications
    updated_user = await update_user(current_user)
    
    return updated_user.to_response()

@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    request: Request
):
    """
    Change le mot de passe de l'utilisateur actuellement connecté.
    
    Args:
        password_data: Les données de changement de mot de passe
        request: La requête HTTP
        
    Returns:
        Un message de confirmation
        
    Raises:
        HTTPException: Si le mot de passe actuel est incorrect
    """
    current_user = get_current_user(request)
    
    # Vérifier le mot de passe actuel
    is_password_valid = bcrypt.checkpw(
        password_data.current_password.encode('utf-8'),
        current_user.hashed_password.encode('utf-8')
    )
    
    if not is_password_valid:
        raise HTTPException(
            status_code=401,
            detail="Le mot de passe actuel est incorrect."
        )
    
    # Hasher le nouveau mot de passe
    hashed_password = bcrypt.hashpw(
        password_data.new_password.encode('utf-8'), 
        bcrypt.gensalt()
    ).decode('utf-8')
    
    # Mettre à jour le mot de passe
    await update_user_password(current_user.id, hashed_password)
    
    # Journaliser le changement de mot de passe
    logger.info(f"Mot de passe changé pour l'utilisateur: {current_user.email}")
    
    return {"message": "Mot de passe modifié avec succès."}

@router.post("/forgot-password")
async def forgot_password(
    email: str,
    background_tasks: BackgroundTasks
):
    """
    Demande de réinitialisation de mot de passe.
    
    Args:
        email: L'email de l'utilisateur
        background_tasks: Tâches à exécuter en arrière-plan
        
    Returns:
        Un message de confirmation
    """
    # Récupérer l'utilisateur
    user = await get_user_by_email(email)
    
    # Même si l'utilisateur n'existe pas, renvoyer un message de confirmation
    # pour éviter de révéler l'existence de l'email
    if not user:
        return {"message": "Si cet email existe, un lien de réinitialisation a été envoyé."}
    
    # Générer un token de réinitialisation
    reset_token = secrets.token_urlsafe(32)
    expiration = datetime.now() + timedelta(hours=24)
    
    # Stocker le token
    await store_reset_token(user.id, reset_token, expiration)
    
    # Envoyer l'email de réinitialisation en arrière-plan
    background_tasks.add_task(
        send_password_reset_email,
        email=user.email,
        name=user.full_name,
        token=reset_token
    )
    
    # Journaliser la demande de réinitialisation
    logger.info(f"Demande de réinitialisation de mot de passe pour: {email}")
    
    return {"message": "Si cet email existe, un lien de réinitialisation a été envoyé."}

@router.post("/reset-password")
async def reset_password(
    reset_data: PasswordReset
):
    """
    Réinitialise le mot de passe avec un token.
    
    Args:
        reset_data: Les données de réinitialisation
        
    Returns:
        Un message de confirmation
        
    Raises:
        HTTPException: Si le token est invalide ou expiré
    """
    # Vérifier le token de réinitialisation
    user_id = await validate_reset_token(reset_data.token)
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="Token de réinitialisation invalide ou expiré."
        )
    
    # Récupérer l'utilisateur
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=400,
            detail="Utilisateur non trouvé."
        )
    
    # Hasher le nouveau mot de passe
    hashed_password = bcrypt.hashpw(
        reset_data.new_password.encode('utf-8'), 
        bcrypt.gensalt()
    ).decode('utf-8')
    
    # Mettre à jour le mot de passe
    await update_user_password(user_id, hashed_password)
    
    # Invalider le token de réinitialisation
    await invalidate_reset_token(reset_data.token)
    
    # Journaliser la réinitialisation du mot de passe
    logger.info(f"Mot de passe réinitialisé pour l'utilisateur: {user.email}")
    
    return {"message": "Mot de passe réinitialisé avec succès."}

@router.delete("/me")
async def delete_account(request: Request):
    """
    Supprime le compte de l'utilisateur actuellement connecté.
    
    Args:
        request: La requête HTTP
        
    Returns:
        Un message de confirmation
    """
    current_user = get_current_user(request)
    
    # Désactiver le compte plutôt que de le supprimer
    current_user.is_active = False
    await update_user(current_user)
    
    # Journaliser la suppression du compte
    logger.info(f"Compte désactivé pour l'utilisateur: {current_user.email}")
    
    return {"message": "Votre compte a été désactivé."} 