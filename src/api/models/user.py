"""
Modèle de données utilisateur pour l'API du bot de trading.

Ce module définit les modèles Pydantic pour les utilisateurs et leurs rôles.
"""

from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import re

class UserBase(BaseModel):
    """Modèle de base pour les utilisateurs."""
    
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=100)
    is_active: bool = True
    
    @validator('full_name')
    def full_name_must_contain_space(cls, v):
        if ' ' not in v:
            raise ValueError('Le nom complet doit contenir un prénom et un nom')
        return v

class UserCreate(UserBase):
    """Modèle pour la création d'un utilisateur."""
    
    password: str = Field(..., min_length=8)
    
    @validator('password')
    def password_strength(cls, v):
        """Vérifie la complexité du mot de passe."""
        if not re.search(r'[A-Z]', v):
            raise ValueError('Le mot de passe doit contenir au moins une lettre majuscule')
        if not re.search(r'[a-z]', v):
            raise ValueError('Le mot de passe doit contenir au moins une lettre minuscule')
        if not re.search(r'[0-9]', v):
            raise ValueError('Le mot de passe doit contenir au moins un chiffre')
        if not re.search(r'[^A-Za-z0-9]', v):
            raise ValueError('Le mot de passe doit contenir au moins un caractère spécial')
        return v

class UserLogin(BaseModel):
    """Modèle pour la connexion d'un utilisateur."""
    
    email: EmailStr
    password: str

class UserResponse(UserBase):
    """Modèle de réponse pour les données utilisateur."""
    
    id: str
    subscription_tier: str
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class UserUpdate(BaseModel):
    """Modèle pour la mise à jour des informations utilisateur."""
    
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = None
    
    @validator('full_name')
    def full_name_must_contain_space(cls, v):
        if v is not None and ' ' not in v:
            raise ValueError('Le nom complet doit contenir un prénom et un nom')
        return v

class PasswordChange(BaseModel):
    """Modèle pour le changement de mot de passe."""
    
    current_password: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def password_strength(cls, v):
        """Vérifie la complexité du mot de passe."""
        if not re.search(r'[A-Z]', v):
            raise ValueError('Le mot de passe doit contenir au moins une lettre majuscule')
        if not re.search(r'[a-z]', v):
            raise ValueError('Le mot de passe doit contenir au moins une lettre minuscule')
        if not re.search(r'[0-9]', v):
            raise ValueError('Le mot de passe doit contenir au moins un chiffre')
        if not re.search(r'[^A-Za-z0-9]', v):
            raise ValueError('Le mot de passe doit contenir au moins un caractère spécial')
        return v

class PasswordReset(BaseModel):
    """Modèle pour la réinitialisation du mot de passe."""
    
    token: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def password_strength(cls, v):
        """Vérifie la complexité du mot de passe."""
        if not re.search(r'[A-Z]', v):
            raise ValueError('Le mot de passe doit contenir au moins une lettre majuscule')
        if not re.search(r'[a-z]', v):
            raise ValueError('Le mot de passe doit contenir au moins une lettre minuscule')
        if not re.search(r'[0-9]', v):
            raise ValueError('Le mot de passe doit contenir au moins un chiffre')
        if not re.search(r'[^A-Za-z0-9]', v):
            raise ValueError('Le mot de passe doit contenir au moins un caractère spécial')
        return v

class UserPreferences(BaseModel):
    """Modèle pour les préférences utilisateur."""
    
    theme: Optional[str] = "light"
    notifications_enabled: bool = True
    email_notifications: bool = True
    telegram_notifications: bool = False
    language: str = "fr"
    timezone: str = "Europe/Paris"
    display_currency: str = "EUR"
    dashboard_widgets: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        orm_mode = True

class User:
    """Classe de modèle utilisateur pour l'application."""
    
    def __init__(
        self,
        id: str,
        email: str,
        full_name: str,
        hashed_password: str,
        is_active: bool = True,
        subscription_tier: str = "free",
        created_at: datetime = None,
        last_login: datetime = None,
        preferences: Dict[str, Any] = None
    ):
        self.id = id
        self.email = email
        self.full_name = full_name
        self.hashed_password = hashed_password
        self.is_active = is_active
        self.subscription_tier = subscription_tier
        self.created_at = created_at or datetime.now()
        self.last_login = last_login
        self.preferences = preferences or UserPreferences().dict()
        
    @classmethod
    def from_dict(cls, data):
        """Crée un utilisateur à partir d'un dictionnaire."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            email=data["email"],
            full_name=data["full_name"],
            hashed_password=data["hashed_password"],
            is_active=data.get("is_active", True),
            subscription_tier=data.get("subscription_tier", "free"),
            created_at=data.get("created_at"),
            last_login=data.get("last_login"),
            preferences=data.get("preferences", {})
        )
        
    def to_dict(self):
        """Convertit l'utilisateur en dictionnaire."""
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "hashed_password": self.hashed_password,
            "is_active": self.is_active,
            "subscription_tier": self.subscription_tier,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "preferences": self.preferences
        }
        
    def to_response(self):
        """Convertit l'utilisateur en modèle de réponse."""
        return UserResponse(
            id=self.id,
            email=self.email,
            full_name=self.full_name,
            is_active=self.is_active,
            subscription_tier=self.subscription_tier,
            created_at=self.created_at,
            last_login=self.last_login
        ) 