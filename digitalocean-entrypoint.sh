#!/bin/bash
set -e

# Script d'entrée pour le conteneur DigitalOcean App Platform
# Ce script gère les différents modes de démarrage du conteneur

MODE="$1"

case "$MODE" in
  gunicorn)
    # Mode serveur Web (Gunicorn)
    echo "Démarrage du serveur Web avec Gunicorn..."
    exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --threads 4 "app:create_app()"
    ;;
    
  scheduler)
    # Mode scheduler pour les tâches planifiées
    echo "Démarrage du scheduler..."
    exec python -m app.scheduler
    ;;
    
  analysis-bot)
    # Mode bot d'analyse
    echo "Exécution du bot d'analyse..."
    exec python -m app.analysis_bot
    ;;
    
  cron-worker)
    # Mode worker cron
    echo "Démarrage du worker cron..."
    
    # Créer le fichier crontab
    CRON_SCHEDULE=${CRON_SCHEDULE:-"0 0 * * *"}
    CRON_COMMAND=${CRON_COMMAND:-"/app/digitalocean-entrypoint.sh analysis-bot"}
    
    echo "Configuration de la tâche cron: $CRON_SCHEDULE $CRON_COMMAND"
    
    # Installer cron si nécessaire
    apt-get update && apt-get install -y cron
    
    # Créer un fichier crontab temporaire
    echo "$CRON_SCHEDULE $CRON_COMMAND >> /var/log/cron.log 2>&1" > /tmp/crontab
    
    # Installer le fichier crontab
    crontab /tmp/crontab
    
    # Créer le fichier de log
    touch /var/log/cron.log
    
    # Démarrer cron en premier plan
    echo "Démarrage de cron..."
    cron
    
    # Afficher les logs en continu
    exec tail -f /var/log/cron.log
    ;;
    
  *)
    echo "Mode non reconnu: $MODE"
    echo "Modes disponibles: gunicorn, scheduler, analysis-bot, cron-worker"
    exit 1
    ;;
esac 