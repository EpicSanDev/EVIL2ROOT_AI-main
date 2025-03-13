"""
Initialisation des routes de l'application
"""

from app.routes.auth import auth
from app.routes.main import main

def register_routes(app):
    """
    Enregistre tous les blueprints dans l'application Flask
    """
    app.register_blueprint(auth, url_prefix='/auth')
    app.register_blueprint(main) 