"""
Module d'initialisation pour les routes API.
Ce module enregistre les blueprints pour les routes API.
"""

def register_blueprints(app):
    """
    Enregistre tous les blueprints pour les routes API.
    
    Args:
        app: Application Flask
    """
    from app.api.payment_webhooks import payment_webhook_bp
    
    # Enregistrer les routes
    app.register_blueprint(payment_webhook_bp, url_prefix='/api')
    
    # Logging des routes enregistrées
    app.logger.info("Blueprints API enregistrés")

from app.api.api import register_api_routes, api_blueprint

__all__ = ['register_api_routes', 'api_blueprint'] 