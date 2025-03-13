"""
Service de paiement en cryptomonnaie utilisant Coinbase Commerce.
Ce module permet de créer et gérer des paiements en crypto via l'API Coinbase Commerce.
"""

import os
import json
import logging
import requests
import time
import uuid
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration du logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration de l'API Coinbase Commerce
COINBASE_API_KEY = os.environ.get('COINBASE_API_KEY', '')
COINBASE_WEBHOOK_SECRET = os.environ.get('COINBASE_WEBHOOK_SECRET', '')
COINBASE_API_URL = "https://api.commerce.coinbase.com"

# Types d'abonnements et leurs prix
SUBSCRIPTION_PRICES = {
    'basic': 10.0,  # $10/mois
    'premium': 20.0,  # $20/mois
    'enterprise': 50.0  # $50/mois
}

# Durée des abonnements en jours
SUBSCRIPTION_DURATIONS = {
    'basic': 30,
    'premium': 30,
    'enterprise': 30
}

class CoinbasePaymentService:
    """
    Service de paiement en cryptomonnaie utilisant Coinbase Commerce.
    Permet de créer et gérer des paiements en crypto.
    """
    
    @staticmethod
    def get_subscription_price(subscription_type: str) -> float:
        """
        Récupère le prix d'un abonnement.
        
        Args:
            subscription_type: Type d'abonnement (basic, premium, enterprise)
            
        Returns:
            Prix de l'abonnement en USD
        """
        if subscription_type not in SUBSCRIPTION_PRICES:
            raise ValueError(f"Type d'abonnement invalide: {subscription_type}")
        
        return SUBSCRIPTION_PRICES[subscription_type]
    
    @staticmethod
    def get_subscription_duration(subscription_type: str) -> int:
        """
        Récupère la durée d'un abonnement en jours.
        
        Args:
            subscription_type: Type d'abonnement (basic, premium, enterprise)
            
        Returns:
            Durée de l'abonnement en jours
        """
        if subscription_type not in SUBSCRIPTION_DURATIONS:
            raise ValueError(f"Type d'abonnement invalide: {subscription_type}")
        
        return SUBSCRIPTION_DURATIONS[subscription_type]
    
    @staticmethod
    def create_charge(user_id: str, subscription_type: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Crée une demande de paiement (charge) via Coinbase Commerce.
        
        Args:
            user_id: ID de l'utilisateur
            subscription_type: Type d'abonnement
            metadata: Métadonnées supplémentaires à stocker avec la charge
            
        Returns:
            Tuple (succès, données de la charge)
        """
        if not COINBASE_API_KEY:
            logger.error("Clé API Coinbase Commerce non configurée")
            return False, {"error": "Clé API Coinbase Commerce non configurée"}
        
        # Vérifier le type d'abonnement
        if subscription_type not in SUBSCRIPTION_PRICES:
            logger.error(f"Type d'abonnement invalide: {subscription_type}")
            return False, {"error": f"Type d'abonnement invalide: {subscription_type}"}
        
        # Récupérer le prix et la durée
        price = SUBSCRIPTION_PRICES[subscription_type]
        duration = SUBSCRIPTION_DURATIONS[subscription_type]
        
        # Préparer les métadonnées
        meta = {
            "user_id": str(user_id),
            "subscription_type": subscription_type,
            "duration_days": duration,
            "created_at": datetime.now().isoformat()
        }
        
        if metadata:
            meta.update(metadata)
        
        # Préparer les données de la charge
        charge_data = {
            "name": f"Abonnement {subscription_type.capitalize()} EVIL2ROOT Trading Bot",
            "description": f"Abonnement {subscription_type.capitalize()} pour {duration} jours",
            "pricing_type": "fixed_price",
            "local_price": {
                "amount": str(price),
                "currency": "USD"
            },
            "metadata": meta,
            "redirect_url": os.environ.get('COINBASE_REDIRECT_URL', 'https://evil2root-ai.com/payment/success'),
            "cancel_url": os.environ.get('COINBASE_CANCEL_URL', 'https://evil2root-ai.com/payment/cancel')
        }
        
        try:
            # Créer la charge via l'API Coinbase Commerce
            response = requests.post(
                f"{COINBASE_API_URL}/charges",
                json=charge_data,
                headers={
                    "X-CC-Api-Key": COINBASE_API_KEY,
                    "X-CC-Version": "2018-03-22",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 201:
                charge = response.json()['data']
                logger.info(f"Charge créée avec succès: {charge['id']}")
                return True, charge
            else:
                logger.error(f"Erreur lors de la création de la charge: {response.text}")
                return False, {"error": f"Erreur {response.status_code}: {response.text}"}
        
        except Exception as e:
            logger.error(f"Exception lors de la création de la charge: {e}")
            return False, {"error": str(e)}
    
    @staticmethod
    def get_charge(charge_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Récupère les informations d'une charge.
        
        Args:
            charge_id: ID de la charge Coinbase Commerce
            
        Returns:
            Tuple (succès, données de la charge)
        """
        if not COINBASE_API_KEY:
            logger.error("Clé API Coinbase Commerce non configurée")
            return False, {"error": "Clé API Coinbase Commerce non configurée"}
        
        try:
            # Récupérer la charge via l'API Coinbase Commerce
            response = requests.get(
                f"{COINBASE_API_URL}/charges/{charge_id}",
                headers={
                    "X-CC-Api-Key": COINBASE_API_KEY,
                    "X-CC-Version": "2018-03-22"
                }
            )
            
            if response.status_code == 200:
                charge = response.json()['data']
                return True, charge
            else:
                logger.error(f"Erreur lors de la récupération de la charge: {response.text}")
                return False, {"error": f"Erreur {response.status_code}: {response.text}"}
        
        except Exception as e:
            logger.error(f"Exception lors de la récupération de la charge: {e}")
            return False, {"error": str(e)}
    
    @staticmethod
    def verify_charge_status(charge_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Vérifie le statut d'une charge.
        
        Args:
            charge_id: ID de la charge Coinbase Commerce
            
        Returns:
            Tuple (succès, statut, détails de la charge)
        """
        success, charge_data = CoinbasePaymentService.get_charge(charge_id)
        
        if not success:
            return False, "error", charge_data
        
        # Récupérer le statut de la timeline
        timeline = charge_data.get('timeline', [])
        status = "NEW"  # Statut par défaut
        
        for event in timeline:
            if event['status'] == 'COMPLETED':
                return True, "completed", charge_data
            elif event['status'] == 'CANCELED':
                return True, "canceled", charge_data
            elif event['status'] == 'EXPIRED':
                return True, "expired", charge_data
            elif event['status'] == 'PENDING':
                status = "pending"
        
        # Si aucun événement n'a renvoyé un statut final, utiliser le statut actuel
        return True, status, charge_data
    
    @staticmethod
    def get_supported_currencies() -> List[Dict[str, Any]]:
        """
        Récupère la liste des cryptomonnaies supportées par Coinbase Commerce.
        
        Returns:
            Liste des cryptomonnaies supportées
        """
        if not COINBASE_API_KEY:
            logger.error("Clé API Coinbase Commerce non configurée")
            return []
        
        try:
            response = requests.get(
                f"{COINBASE_API_URL}/currencies",
                headers={
                    "X-CC-Api-Key": COINBASE_API_KEY,
                    "X-CC-Version": "2018-03-22"
                }
            )
            
            if response.status_code == 200:
                currencies = response.json()['data']
                # Filtrer pour ne garder que les crypto
                crypto_currencies = [c for c in currencies if c.get('type') == 'crypto']
                return crypto_currencies
            else:
                logger.error(f"Erreur lors de la récupération des devises: {response.text}")
                return []
        
        except Exception as e:
            logger.error(f"Exception lors de la récupération des devises: {e}")
            return []
    
    @staticmethod
    def create_checkout(user_id: str, subscription_type: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Crée une page de paiement (checkout) via Coinbase Commerce.
        
        Args:
            user_id: ID de l'utilisateur
            subscription_type: Type d'abonnement
            metadata: Métadonnées supplémentaires à stocker avec le checkout
            
        Returns:
            Tuple (succès, données du checkout)
        """
        if not COINBASE_API_KEY:
            logger.error("Clé API Coinbase Commerce non configurée")
            return False, {"error": "Clé API Coinbase Commerce non configurée"}
        
        # Vérifier le type d'abonnement
        if subscription_type not in SUBSCRIPTION_PRICES:
            logger.error(f"Type d'abonnement invalide: {subscription_type}")
            return False, {"error": f"Type d'abonnement invalide: {subscription_type}"}
        
        # Récupérer le prix et la durée
        price = SUBSCRIPTION_PRICES[subscription_type]
        duration = SUBSCRIPTION_DURATIONS[subscription_type]
        
        # Préparer les métadonnées
        meta = {
            "user_id": str(user_id),
            "subscription_type": subscription_type,
            "duration_days": duration,
            "created_at": datetime.now().isoformat()
        }
        
        if metadata:
            meta.update(metadata)
        
        # Préparer les données du checkout
        checkout_data = {
            "name": f"Abonnement {subscription_type.capitalize()} EVIL2ROOT Trading Bot",
            "description": f"Abonnement {subscription_type.capitalize()} pour {duration} jours",
            "pricing_type": "fixed_price",
            "local_price": {
                "amount": str(price),
                "currency": "USD"
            },
            "requested_info": [],
            "metadata": meta
        }
        
        try:
            # Créer le checkout via l'API Coinbase Commerce
            response = requests.post(
                f"{COINBASE_API_URL}/checkouts",
                json=checkout_data,
                headers={
                    "X-CC-Api-Key": COINBASE_API_KEY,
                    "X-CC-Version": "2018-03-22",
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 201:
                checkout = response.json()['data']
                logger.info(f"Checkout créé avec succès: {checkout['id']}")
                return True, checkout
            else:
                logger.error(f"Erreur lors de la création du checkout: {response.text}")
                return False, {"error": f"Erreur {response.status_code}: {response.text}"}
        
        except Exception as e:
            logger.error(f"Exception lors de la création du checkout: {e}")
            return False, {"error": str(e)}
    
    @staticmethod
    def validate_webhook_signature(request_data: bytes, signature_header: str) -> bool:
        """
        Valide la signature d'un webhook Coinbase Commerce.
        
        Args:
            request_data: Données brutes du corps de la requête
            signature_header: Valeur de l'en-tête X-CC-Webhook-Signature
            
        Returns:
            True si la signature est valide, False sinon
        """
        if not COINBASE_WEBHOOK_SECRET:
            logger.error("Secret de webhook Coinbase Commerce non configuré")
            return False
        
        import hmac
        import hashlib
        
        expected_signature = hmac.new(
            COINBASE_WEBHOOK_SECRET.encode(), 
            msg=request_data, 
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Comparer les signatures
        return hmac.compare_digest(expected_signature, signature_header)

# Instance globale du service de paiement
coinbase_payment_service = CoinbasePaymentService() 