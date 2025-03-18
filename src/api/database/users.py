"""
Opérations CRUD pour les utilisateurs.

Ce module fournit les fonctions d'accès à la base de données
pour créer, lire, mettre à jour et supprimer des utilisateurs.
"""

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import uuid
import logging

from src.api.database.models.user import User, UserPreference, PasswordResetToken
from src.api.database.session import get_db

logger = logging.getLogger("api.database.users")


async def create_user(
    id: str = None,
    email: str = None,
    full_name: str = None,
    hashed_password: str = None,
    is_active: bool = True,
    subscription_tier: str = "free",
    db: AsyncSession = None
) -> User:
    """
    Crée un nouvel utilisateur dans la base de données.
    
    Args:
        id: ID de l'utilisateur (UUID)
        email: Adresse email
        full_name: Nom complet
        hashed_password: Mot de passe hashé
        is_active: Statut d'activation du compte
        subscription_tier: Niveau d'abonnement
        db: Session de base de données
        
    Returns:
        User: L'utilisateur créé
    """
    if not db:
        async for session in get_db():
            db = session
            break
            
    if not id:
        id = str(uuid.uuid4())
    
    # Créer l'utilisateur
    user = User(
        id=id,
        email=email,
        full_name=full_name,
        hashed_password=hashed_password,
        is_active=is_active,
        subscription_tier=subscription_tier,
        created_at=datetime.utcnow(),
        last_login=None
    )
    
    # Créer les préférences utilisateur par défaut
    preferences = UserPreference(
        user_id=id,
        theme="light",
        language="fr",
        timezone="Europe/Paris",
        notifications_enabled=True,
        email_notifications=True,
        telegram_notifications=False,
        display_currency="EUR",
        dashboard_widgets={}
    )
    
    # Associer les préférences à l'utilisateur
    user.preferences = preferences
    
    # Sauvegarder en base de données
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    logger.info(f"Utilisateur créé: {email}")
    
    return user


async def get_user_by_id(
    user_id: str,
    db: AsyncSession = None
) -> User:
    """
    Récupère un utilisateur par son ID.
    
    Args:
        user_id: ID de l'utilisateur
        db: Session de base de données
    
    Returns:
        User ou None: L'utilisateur trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(User).where(User.id == user_id, User.is_deleted == False)
    result = await db.execute(query)
    user = result.scalars().first()
    
    return user


async def get_user_by_email(
    email: str,
    db: AsyncSession = None
) -> User:
    """
    Récupère un utilisateur par son email.
    
    Args:
        email: Adresse email
        db: Session de base de données
    
    Returns:
        User ou None: L'utilisateur trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(User).where(User.email == email, User.is_deleted == False)
    result = await db.execute(query)
    user = result.scalars().first()
    
    return user


async def update_user(
    user: User,
    db: AsyncSession = None
) -> User:
    """
    Met à jour un utilisateur existant.
    
    Args:
        user: Utilisateur à mettre à jour
        db: Session de base de données
    
    Returns:
        User: L'utilisateur mis à jour
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    user.updated_at = datetime.utcnow()
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    logger.info(f"Utilisateur mis à jour: {user.email}")
    
    return user


async def update_user_password(
    user_id: str,
    hashed_password: str,
    db: AsyncSession = None
) -> bool:
    """
    Met à jour le mot de passe d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        hashed_password: Nouveau mot de passe hashé
        db: Session de base de données
    
    Returns:
        bool: True si la mise à jour a réussi, False sinon
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    stmt = (
        update(User)
        .where(User.id == user_id, User.is_deleted == False)
        .values(
            hashed_password=hashed_password,
            updated_at=datetime.utcnow()
        )
    )
    
    result = await db.execute(stmt)
    await db.commit()
    
    if result.rowcount > 0:
        logger.info(f"Mot de passe mis à jour pour l'utilisateur {user_id}")
        return True
    
    logger.warning(f"Échec de la mise à jour du mot de passe pour l'utilisateur {user_id}")
    return False


async def delete_user(
    user_id: str,
    hard_delete: bool = False,
    db: AsyncSession = None
) -> bool:
    """
    Supprime un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        hard_delete: Si True, effectue une suppression définitive,
                    sinon effectue une suppression logique
        db: Session de base de données
    
    Returns:
        bool: True si la suppression a réussi, False sinon
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    if hard_delete:
        # Suppression définitive
        stmt = delete(User).where(User.id == user_id)
    else:
        # Suppression logique
        stmt = (
            update(User)
            .where(User.id == user_id, User.is_deleted == False)
            .values(
                is_deleted=True,
                deleted_at=datetime.utcnow(),
                is_active=False,
                updated_at=datetime.utcnow()
            )
        )
    
    result = await db.execute(stmt)
    await db.commit()
    
    if result.rowcount > 0:
        logger.info(f"Utilisateur supprimé: {user_id} (hard_delete={hard_delete})")
        return True
    
    logger.warning(f"Échec de la suppression de l'utilisateur {user_id}")
    return False


async def update_user_preferences(
    user_id: str,
    preferences_data: dict,
    db: AsyncSession = None
) -> UserPreference:
    """
    Met à jour les préférences d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        preferences_data: Dictionnaire des préférences à mettre à jour
        db: Session de base de données
    
    Returns:
        UserPreference: Les préférences mises à jour
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    # Récupérer les préférences existantes
    query = select(UserPreference).where(UserPreference.user_id == user_id)
    result = await db.execute(query)
    preferences = result.scalars().first()
    
    if not preferences:
        # Créer les préférences si elles n'existent pas
        preferences = UserPreference(
            user_id=user_id,
            **preferences_data
        )
        db.add(preferences)
    else:
        # Mettre à jour les préférences existantes
        for key, value in preferences_data.items():
            if hasattr(preferences, key):
                setattr(preferences, key, value)
    
    preferences.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(preferences)
    
    logger.info(f"Préférences mises à jour pour l'utilisateur {user_id}")
    
    return preferences


async def store_reset_token(
    user_id: str,
    db: AsyncSession = None
) -> PasswordResetToken:
    """
    Génère et stocke un token de réinitialisation de mot de passe.
    
    Args:
        user_id: ID de l'utilisateur
        db: Session de base de données
    
    Returns:
        PasswordResetToken: Le token généré
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    # Invalider les anciens tokens
    query = select(PasswordResetToken).where(
        PasswordResetToken.user_id == user_id,
        PasswordResetToken.is_used == False
    )
    result = await db.execute(query)
    old_tokens = result.scalars().all()
    
    for token in old_tokens:
        token.is_used = True
    
    # Générer un nouveau token
    reset_token = PasswordResetToken.generate(user_id)
    db.add(reset_token)
    
    await db.commit()
    
    logger.info(f"Token de réinitialisation généré pour l'utilisateur {user_id}")
    
    return reset_token


async def get_reset_token(
    token: str,
    db: AsyncSession = None
) -> PasswordResetToken:
    """
    Récupère un token de réinitialisation par sa valeur.
    
    Args:
        token: Valeur du token
        db: Session de base de données
    
    Returns:
        PasswordResetToken ou None: Le token trouvé ou None
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    query = select(PasswordResetToken).where(PasswordResetToken.token == token)
    result = await db.execute(query)
    reset_token = result.scalars().first()
    
    return reset_token


async def invalidate_token(
    token_id: str,
    db: AsyncSession = None
) -> bool:
    """
    Invalide un token de réinitialisation.
    
    Args:
        token_id: ID du token
        db: Session de base de données
    
    Returns:
        bool: True si l'invalidation a réussi, False sinon
    """
    if not db:
        async for session in get_db():
            db = session
            break
    
    stmt = (
        update(PasswordResetToken)
        .where(PasswordResetToken.id == token_id)
        .values(is_used=True, updated_at=datetime.utcnow())
    )
    
    result = await db.execute(stmt)
    await db.commit()
    
    if result.rowcount > 0:
        logger.info(f"Token invalidé: {token_id}")
        return True
    
    return False