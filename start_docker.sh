#!/bin/bash

# Script pour démarrer tous les services Docker
# Usage: ./start_docker.sh [--force-train]

echo "=== Evil2Root Trading Bot - Démarrage avec Docker ==="
echo ""

# Vérifier les arguments
FORCE_TRAIN=0
for arg in "$@"; do
    case $arg in
        --force-train)
            FORCE_TRAIN=1
            shift
            ;;
    esac
done

# Vérifier si Docker et Docker Compose sont installés
if ! command -v docker &> /dev/null; then
    echo "Erreur: Docker n'est pas installé."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Erreur: Docker Compose n'est pas installé."
    exit 1
fi

# Vérifier si le fichier .env existe
if [ ! -f .env ]; then
    echo "Attention: Fichier .env non trouvé. Création d'un exemple à partir de .env.example"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Fichier .env créé. Veuillez l'éditer avec vos propres valeurs."
        echo "Appuyez sur une touche pour continuer..."
        read -n 1
    else
        echo "Erreur: Ni .env ni .env.example n'ont été trouvés."
        exit 1
    fi
fi

# Vérifier les variables d'environnement requises
if ! grep -q "TELEGRAM_TOKEN" .env || ! grep -q "TELEGRAM_CHAT_ID" .env; then
    echo "Attention: Configuration Telegram manquante dans .env"
    echo "Veuillez vous assurer que TELEGRAM_TOKEN et TELEGRAM_CHAT_ID sont configurés."
    echo "Le bot ne pourra pas envoyer de signaux sans ces configurations."
    echo ""
    read -p "Continuer quand même? (o/n): " continue_answer
    if [ "$continue_answer" != "o" ]; then
        exit 1
    fi
fi

echo "Démarrage des services Docker..."
echo ""

# Option pour construire les images
read -p "Reconstruire les images Docker ? (o/n): " rebuild_answer
if [ "$rebuild_answer" = "o" ]; then
    docker-compose build
fi

# Si force-train est activé, démarrer les services avec entraînement forcé
if [ "$FORCE_TRAIN" -eq 1 ]; then
    echo "Mode d'entraînement forcé activé. Les modèles existants seront réentraînés."
    
    # Arrêter d'abord les services réguliers s'ils sont en cours d'exécution
    docker-compose down
    
    # Démarrer les services avec entraînement forcé
    docker-compose up -d db redis
    echo "Attente du démarrage de la base de données et Redis..."
    sleep 5
    
    echo "Lancement de l'entraînement forcé..."
    # Démarrer le service d'entraînement et attendre sa fin
    docker-compose up train-and-analyze
    
    # Une fois l'entraînement terminé, démarrer les services normaux
    docker-compose up -d web market-scheduler-force-train
else
    # Démarrer tous les services normaux
    docker-compose up -d
fi

echo ""
echo "Services démarrés avec succès!"
echo ""
echo "Pour voir les logs:"
echo "- Application web: docker logs -f trading-bot-web"
echo "- Bot d'analyse: docker logs -f trading-bot-analysis"
echo "- Scheduler de marché: docker logs -f trading-bot-market-scheduler"
echo ""
echo "Pour arrêter les services: ./stop_docker.sh ou docker-compose down" 