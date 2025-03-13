"""
Webhook de paiement pour Coinbase Commerce.
Ce module gère les notifications de paiement envoyées par Coinbase Commerce via webhooks.
"""

import os
import json
import logging
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify
from dotenv import load_dotenv

from app.models.user import User, UserManager
from app.services.coinbase_payment import coinbase_payment_service

# Charger les variables d'environnement
load_dotenv()

# Configuration du logger
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Créer le blueprint pour les webhooks
payment_webhook_bp = Blueprint('payment_webhook', __name__)

@payment_webhook_bp.route('/webhook/coinbase', methods=['POST'])
def coinbase_webhook():
    """
    Endpoint pour les webhooks de Coinbase Commerce.
    Traite les notifications de paiement.
    """
    logger.info("Réception d'un webhook Coinbase Commerce")
    
    # Vérifier que le corps de la requête n'est pas vide
    if not request.data:
        logger.error("Corps de la requête vide")
        return jsonify({"error": "Corps de la requête vide"}), 400
    
    # Vérifier la signature du webhook
    signature = request.headers.get('X-CC-Webhook-Signature', '')
    
    if not signature:
        logger.error("Signature du webhook manquante")
        return jsonify({"error": "Signature du webhook manquante"}), 400
    
    if not coinbase_payment_service.validate_webhook_signature(request.data, signature):
        logger.error("Signature du webhook invalide")
        return jsonify({"error": "Signature du webhook invalide"}), 401
    
    # Traiter le webhook
    try:
        webhook_data = json.loads(request.data.decode('utf-8'))
        
        # Vérifier que c'est bien un événement
        event_type = webhook_data.get('event', {}).get('type')
        
        if not event_type:
            logger.error("Type d'événement manquant dans le webhook")
            return jsonify({"error": "Type d'événement manquant"}), 400
        
        # Traiter uniquement les événements de paiement confirmé
        if event_type == 'charge:confirmed':
            logger.info(f"Événement de paiement confirmé reçu: {event_type}")
            
            # Récupérer les données de la charge
            charge_data = webhook_data.get('event', {}).get('data', {})
            
            # Récupérer les métadonnées
            metadata = charge_data.get('metadata', {})
            user_id = metadata.get('user_id')
            subscription_type = metadata.get('subscription_type')
            duration_days = int(metadata.get('duration_days', 30))
            
            if not user_id or not subscription_type:
                logger.error("Métadonnées incomplètes dans le webhook")
                return jsonify({"error": "Métadonnées incomplètes"}), 400
            
            # Mettre à jour l'abonnement de l'utilisateur
            success = update_user_subscription(user_id, subscription_type, duration_days)
            
            if success:
                logger.info(f"Abonnement mis à jour avec succès pour l'utilisateur {user_id}")
                return jsonify({"success": True}), 200
            else:
                logger.error(f"Erreur lors de la mise à jour de l'abonnement pour l'utilisateur {user_id}")
                return jsonify({"error": "Erreur lors de la mise à jour de l'abonnement"}), 500
        
        else:
            # Pour les autres types d'événements, simplement les logger
            logger.info(f"Événement reçu (non traité): {event_type}")
            return jsonify({"success": True, "message": "Événement reçu mais non traité"}), 200
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement du webhook: {e}")
        return jsonify({"error": f"Erreur lors du traitement du webhook: {str(e)}"}), 500

def update_user_subscription(user_id: str, subscription_type: str, duration_days: int) -> bool:
    """
    Met à jour l'abonnement d'un utilisateur.
    
    Args:
        user_id: ID de l'utilisateur
        subscription_type: Type d'abonnement
        duration_days: Durée de l'abonnement en jours
        
    Returns:
        True si la mise à jour a réussi, False sinon
    """
    conn = None
    
    try:
        conn = UserManager.get_db_connection()
        
        with conn.cursor() as cur:
            # Récupérer l'utilisateur
            cur.execute(
                "SELECT subscription_expiry FROM users WHERE id = %s",
                (user_id,)
            )
            
            user_data = cur.fetchone()
            if not user_data:
                logger.error(f"Utilisateur non trouvé: {user_id}")
                return False
            
            # Calculer la nouvelle date d'expiration
            current_expiry = user_data[0]
            
            # Si l'abonnement est expiré, partir de la date actuelle
            # Sinon, ajouter à la date d'expiration existante
            if current_expiry and current_expiry > datetime.now():
                new_expiry = current_expiry + timedelta(days=duration_days)
            else:
                new_expiry = datetime.now() + timedelta(days=duration_days)
            
            # Mettre à jour l'abonnement
            cur.execute(
                """
                UPDATE users 
                SET subscription_type = %s, subscription_expiry = %s 
                WHERE id = %s
                """,
                (subscription_type, new_expiry, user_id)
            )
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"Exception lors de la mise à jour de l'abonnement: {e}")
        if conn:
            conn.rollback()
        return False
        
    finally:
        if conn:
            conn.close() 