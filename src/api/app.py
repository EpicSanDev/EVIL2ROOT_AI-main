"""
Point d'entrée principal de l'API FastAPI pour le bot de trading EVIL2ROOT.
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import time
import logging
from pathlib import Path
import uvicorn

from src.api.routes import (
    auth,
    trading,
    settings,
    dashboard,
    subscriptions,
    backtest,
)
from src.api.middleware.authentication import authenticate_request
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.logging import LoggingMiddleware
from src.utils.log_config import setup_logging

logger = setup_logging("api", "api.log")

def create_app(debug=False):
    """Crée et configure l'application FastAPI"""
    
    app = FastAPI(
        title="EVIL2ROOT Trading Bot API",
        description="API pour interagir avec le bot de trading EVIL2ROOT",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En production, spécifier les domaines autorisés
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Middlewares personnalisés
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RateLimitMiddleware)
    
    # Routes API
    app.include_router(auth.router, prefix="/api/auth", tags=["Authentification"])
    app.include_router(trading.router, prefix="/api/trading", tags=["Trading"])
    app.include_router(settings.router, prefix="/api/settings", tags=["Configuration"])
    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Tableau de bord"])
    app.include_router(subscriptions.router, prefix="/api/subscriptions", tags=["Abonnements"])
    app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtesting"])
    
    # Documentation personnalisée
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="EVIL2ROOT Trading Bot API",
            swagger_js_url="/static/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui.css",
        )
    
    @app.get("/openapi.json", include_in_schema=False)
    async def get_open_api_endpoint():
        return get_openapi(
            title="EVIL2ROOT Trading Bot API",
            version="1.0.0",
            description="API pour interagir avec le bot de trading EVIL2ROOT",
            routes=app.routes,
        )
    
    # Gestion des erreurs globale
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Erreur non gérée: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Une erreur interne s'est produite. Veuillez réessayer plus tard."},
        )
    
    # Route de health check
    @app.get("/health", tags=["Statut"])
    async def health_check():
        return {
            "status": "ok",
            "timestamp": time.time(),
            "version": "1.0.0",
        }
    
    # Configuration des fichiers statiques
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    
    return app

def start_api_server(host="0.0.0.0", port=8000, debug=False):
    """Lance le serveur API"""
    app = create_app(debug=debug)
    uvicorn.run(app, host=host, port=port, log_level="debug" if debug else "info")

if __name__ == "__main__":
    start_api_server(debug=True) 