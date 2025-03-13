"""
Service de paiement en cryptomonnaie.
Ce module gère la génération d'adresses de paiement et la vérification des transactions.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import time
import hmac
import hashlib
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration des API de cryptomonnaie
CRYPTO_API_KEY = os.environ.get('CRYPTO_API_KEY', '')
CRYPTO_API_SECRET = os.environ.get('CRYPTO_API_SECRET', '')
CRYPTO_API_URL = os.environ.get('CRYPTO_API_URL', 'https://api.example.com')

# Adresses de réception des paiements
BTC_ADDRESS = os.environ.get('BTC_PAYMENT_ADDRESS', '')
ETH_ADDRESS = os.environ.get('ETH_PAYMENT_ADDRESS', '')
USDT_ADDRESS = os.environ.get('USDT_PAYMENT_ADDRESS', '')

# Configuration des prix des abonnements
SUBSCRIPTION_PRICES = {
    'basic': {
        'BTC': 0.0005,
        'ETH': 0.01,
        'USDT': 10
    },
    'premium': {
        'BTC': 0.001,
        'ETH': 0.02,
        'USDT': 20
    },
    'enterprise': {
        'BTC': 0.003,
        'ETH': 0.05,
        'USDT': 50
    }
}

# Durée des abonnements en jours
SUBSCRIPTION_DURATIONS = {
    'basic': 30,
    'premium': 30,
    'enterprise': 30
}

class CryptoPaymentService:
    """
    Service de paiement en cryptomonnaie.
    Gère la génération d'adresses de paiement et la vérification des transactions.
    """
    
    @staticmethod
    def get_subscription_price(subscription_type: str, currency: str) -> float:
        """
        Récupère le prix d'un abonnement dans la devise spécifiée.
        
        Args:
            subscription_type: Type d'abonnement (basic, premium, enterprise)
            currency: Devise (BTC, ETH, USDT)
            
        Returns:
            Prix de l'abonnement
        """
        if subscription_type not in SUBSCRIPTION_PRICES:
            raise ValueError(f"Type d'abonnement invalide: {subscription_type}")
        
        if currency not in SUBSCRIPTION_PRICES[subscription_type]:
            raise ValueError(f"Devise non prise en charge: {currency}")
        
        return SUBSCRIPTION_PRICES[subscription_type][currency]
    
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
    def get_payment_address(currency: str) -> str:
        """
        Récupère l'adresse de paiement pour la devise spécifiée.
        
        Args:
            currency: Devise (BTC, ETH, USDT)
            
        Returns:
            Adresse de paiement
        """
        if currency == 'BTC':
            return BTC_ADDRESS
        elif currency == 'ETH':
            return ETH_ADDRESS
        elif currency == 'USDT':
            return USDT_ADDRESS
        else:
            raise ValueError(f"Devise non prise en charge: {currency}")
    
    @staticmethod
    def generate_unique_payment_id(user_id: str, subscription_type: str, currency: str) -> str:
        """
        Génère un identifiant unique pour un paiement.
        
        Args:
            user_id: ID de l'utilisateur
            subscription_type: Type d'abonnement
            currency: Devise
            
        Returns:
            Identifiant unique pour le paiement
        """
        timestamp = int(time.time())
        data = f"{user_id}:{subscription_type}:{currency}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def verify_transaction(tx_hash: str, currency: str, expected_amount: float, 
                         payment_address: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Vérifie une transaction sur la blockchain.
        
        Args:
            tx_hash: Hash de la transaction
            currency: Devise (BTC, ETH, USDT)
            expected_amount: Montant attendu
            payment_address: Adresse de paiement
            
        Returns:
            Tuple (succès, détails de la transaction)
        """
        try:
            # Exemple d'implémentation pour vérifier une transaction Bitcoin
            if currency == 'BTC':
                url = f"https://blockchain.info/rawtx/{tx_hash}"
                response = requests.get(url)
                if response.status_code != 200:
                    return False, None
                
                tx_data = response.json()
                
                # Vérifier que la transaction est confirmée
                if tx_data.get('block_height') is None:
                    return False, {"error": "Transaction non confirmée"}
                
                # Vérifier le destinataire et le montant
                for output in tx_data.get('out', []):
                    if output.get('addr') == payment_address:
                        # Convertir les satoshis en BTC
                        amount = float(output.get('value', 0)) / 100000000
                        
                        # Vérifier que le montant est correct (avec une marge de tolérance)
                        if abs(amount - expected_amount) <= 0.00001:
                            return True, {
                                "hash": tx_hash,
                                "amount": amount,
                                "currency": currency,
                                "confirmations": tx_data.get('block_height'),
                                "timestamp": tx_data.get('time')
                            }
                
                return False, {"error": "Montant ou destinataire incorrect"}
            
            # Exemple d'implémentation pour vérifier une transaction Ethereum
            elif currency == 'ETH' or currency == 'USDT':
                # Pour une implémentation réelle, utilisez une API comme Etherscan
                # Ceci est un exemple simplifié
                url = f"https://api.etherscan.io/api?module=proxy&action=eth_getTransactionByHash&txhash={tx_hash}&apikey={CRYPTO_API_KEY}"
                response = requests.get(url)
                if response.status_code != 200:
                    return False, None
                
                tx_data = response.json().get('result', {})
                
                # Vérifier que la transaction est confirmée
                if not tx_data or tx_data.get('blockNumber') is None:
                    return False, {"error": "Transaction non confirmée"}
                
                # Vérifier le destinataire
                if tx_data.get('to', '').lower() != payment_address.lower():
                    return False, {"error": "Destinataire incorrect"}
                
                # Convertir le montant de Wei à ETH
                amount = int(tx_data.get('value', '0'), 16) / 1e18
                
                # Vérifier que le montant est correct (avec une marge de tolérance)
                if abs(amount - expected_amount) <= 0.0001:
                    return True, {
                        "hash": tx_hash,
                        "amount": amount,
                        "currency": currency,
                        "confirmations": int(tx_data.get('blockNumber', '0'), 16),
                        "timestamp": int(time.time())
                    }
                
                return False, {"error": "Montant incorrect"}
            
            else:
                return False, {"error": f"Devise non prise en charge: {currency}"}
        
        except Exception as e:
            logging.error(f"Erreur lors de la vérification de la transaction: {e}")
            return False, {"error": str(e)}
    
    @staticmethod
    def get_current_prices() -> Dict[str, float]:
        """
        Récupère les prix actuels des cryptomonnaies en USD.
        
        Returns:
            Dictionnaire des prix actuels
        """
        try:
            # Utiliser CoinGecko API pour récupérer les prix
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,tether&vs_currencies=usd"
            response = requests.get(url)
            
            if response.status_code != 200:
                # Valeurs par défaut en cas d'erreur
                return {
                    "BTC": 50000.0,
                    "ETH": 3000.0,
                    "USDT": 1.0
                }
            
            data = response.json()
            
            return {
                "BTC": data.get("bitcoin", {}).get("usd", 50000.0),
                "ETH": data.get("ethereum", {}).get("usd", 3000.0),
                "USDT": data.get("tether", {}).get("usd", 1.0)
            }
        
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des prix: {e}")
            # Valeurs par défaut en cas d'erreur
            return {
                "BTC": 50000.0,
                "ETH": 3000.0,
                "USDT": 1.0
            }
    
    @staticmethod
    def get_subscription_prices_usd() -> Dict[str, float]:
        """
        Récupère les prix des abonnements en USD.
        
        Returns:
            Dictionnaire des prix des abonnements en USD
        """
        prices = CryptoPaymentService.get_current_prices()
        
        result = {}
        for sub_type, sub_prices in SUBSCRIPTION_PRICES.items():
            # Utiliser le prix en USDT comme référence
            result[sub_type] = sub_prices.get("USDT", 0)
        
        return result
    
    @staticmethod
    def get_qr_code_url(payment_address: str, amount: float, currency: str) -> str:
        """
        Génère une URL pour un QR code de paiement.
        
        Args:
            payment_address: Adresse de paiement
            amount: Montant
            currency: Devise
            
        Returns:
            URL du QR code
        """
        if currency == 'BTC':
            return f"bitcoin:{payment_address}?amount={amount}"
        elif currency == 'ETH':
            return f"ethereum:{payment_address}?value={amount}"
        elif currency == 'USDT':
            return f"ethereum:{payment_address}?value={amount}&token=usdt"
        else:
            raise ValueError(f"Devise non prise en charge: {currency}") 