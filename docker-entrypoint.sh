#!/bin/bash
set -e

# Création des répertoires nécessaires
mkdir -p data logs saved_models

# Affichage des informations
echo "=== Evil2Root Trading Bot - Démarrage Docker ==="
echo "Date de démarrage: $(date)"
echo "Environnement: $FLASK_ENV"
echo "Port utilisé: ${PORT:-5000}"

# Fonction pour charger les secrets depuis les fichiers Docker Secrets
load_secrets() {
    echo "Chargement des secrets Docker..."
    
    # Fonction pour charger un secret de manière sécurisée
    load_secret() {
        local file_var="$1_FILE"
        local secret_file="${!file_var}"
        
        if [ -n "$secret_file" ] && [ -f "$secret_file" ]; then
            # Utiliser read pour éviter les problèmes de fin de ligne
            local secret_value
            read -r secret_value < "$secret_file"
            export "$1"="$secret_value"
            echo "Secret $1 chargé à partir de $secret_file"
        else
            echo "Attention: Secret $1 non trouvé, vérifiez votre configuration"
            # Ne pas poursuivre si des secrets essentiels sont manquants
            if [[ "$1" == "DB_USER" || "$1" == "DB_PASSWORD" || "$1" == "SECRET_KEY" ]]; then
                echo "ERREUR: Secret essentiel $1 manquant. Impossible de démarrer l'application en toute sécurité."
                exit 1
            fi
        fi
    }
    
    # Chargement des secrets spécifiques
    load_secret "DB_USER"
    load_secret "DB_PASSWORD"
    load_secret "SECRET_KEY"
    load_secret "ADMIN_PASSWORD"
    load_secret "TELEGRAM_TOKEN"
    load_secret "FINNHUB_API_KEY"
    load_secret "OPENROUTER_API_KEY"
    load_secret "COINBASE_API_KEY"
    load_secret "COINBASE_WEBHOOK_SECRET"
    
    # Construction dynamique de DATABASE_URI avec les identifiants chargés des secrets
    if [ -n "$DB_USER" ] && [ -n "$DB_PASSWORD" ]; then
        export DATABASE_URI="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST:-db}:${DB_PORT:-5432}/${DB_NAME:-trading_db}"
        echo "DATABASE_URI mise à jour avec les secrets"
    fi
}

# Charger les secrets au démarrage
load_secrets

# Vérification de sécurité : s'assurer que les répertoires critiques sont accessibles
if [ ! -w "data" ] || [ ! -w "logs" ] || [ ! -w "saved_models" ]; then
    echo "ERREUR: Les répertoires requis ne sont pas accessibles en écriture. Vérifiez les permissions."
    exit 1
fi

# Fonction pour démarrer le scheduler d'analyse en arrière-plan
start_market_scheduler() {
    echo "Démarrage du scheduler d'analyse de marché..."
    python -c "from app.market_analysis_scheduler import run_market_analysis_scheduler; run_market_analysis_scheduler()" &
    SCHEDULER_PID=$!
    echo "Scheduler démarré avec PID: $SCHEDULER_PID"
    echo $SCHEDULER_PID > /app/.scheduler_pid
    
    # Configurer un trap pour arrêter proprement le scheduler lors de l'arrêt
    trap "echo 'Arrêt du scheduler'; kill -TERM $SCHEDULER_PID 2>/dev/null || true" EXIT
}

# Détection du mode de fonctionnement
if [ "$1" = "scheduler" ]; then
    echo "Mode scheduler activé"
    exec python -c "from app.market_analysis_scheduler import run_market_analysis_scheduler; run_market_analysis_scheduler()"
elif [ "$1" = "web-with-scheduler" ]; then
    echo "Mode web avec scheduler activé"
    # Démarrage du scheduler en arrière-plan
    start_market_scheduler
    # Démarrage de l'application web avec PORT
    exec gunicorn run:app --bind=0.0.0.0:${PORT:-5000}
elif [ "$1" = "analysis-bot" ]; then
    echo "Mode analysis-bot activé"
    exec python start_daily_analysis.py
elif [ "$1" = "train-and-analyze" ]; then
    echo "Mode entraînement forcé et analyse activé"
    # Lancer le script avec l'option --force-train
    exec python start_daily_analysis.py --force-train
elif [ "$1" = "scheduler-force-train" ]; then
    echo "Mode scheduler avec entraînement forcé activé"
    # Définir la variable d'environnement pour forcer l'entraînement
    export FORCE_MODEL_TRAINING=true
    exec python -c "from app.market_analysis_scheduler import run_market_analysis_scheduler; run_market_analysis_scheduler()"
elif [ "$1" = "gunicorn" ]; then
    # Mode spécifique pour DigitalOcean - exécuter gunicorn avec PORT
    echo "Mode gunicorn activé pour DigitalOcean"
    shift
    exec gunicorn run:app --bind=0.0.0.0:${PORT:-5000} "$@"
else
    # Si la commande est gunicorn avec des arguments, utiliser PORT
    if [[ "$1" == "gunicorn" ]]; then
        echo "Ajustement du port pour gunicorn: ${PORT:-5000}"
        # Remplacer le paramètre --bind s'il existe
        ARGS=()
        BIND_SET=false
        while [[ $# -gt 0 ]]; do
            if [[ "$1" == "--bind" ]]; then
                ARGS+=("$1")
                shift
                ARGS+=("0.0.0.0:${PORT:-5000}")
                BIND_SET=true
            else
                ARGS+=("$1")
            fi
            shift
        done
        
        # Si --bind n'était pas dans les arguments, l'ajouter
        if [[ "$BIND_SET" == "false" ]]; then
            exec gunicorn run:app --bind=0.0.0.0:${PORT:-5000}
        else
            exec gunicorn "${ARGS[@]}"
        fi
    else
        # Mode par défaut: application web uniquement
        echo "Mode par défaut (web) activé"
        exec "$@"
    fi
fi 