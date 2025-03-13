#!/bin/bash

# Script pour lancer le scheduler d'analyse de marché
# Usage: ./start_market_scheduler.sh

echo "=== Evil2Root Trading Bot - Scheduler d'analyse de marché ==="
echo "Ce script lance le bot qui analyse les marchés toutes les 30 minutes et met à jour les modèles avec des news toutes les 4 heures"
echo ""

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "Erreur: Python 3 n'est pas installé."
    exit 1
fi

# Vérifier que les répertoires nécessaires existent
mkdir -p logs
mkdir -p saved_models
mkdir -p data

# Vérifier si le fichier .env existe
if [ ! -f .env ]; then
    echo "Attention: Fichier .env non trouvé. Création d'un exemple à partir de .env.example"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Fichier .env créé. Veuillez l'éditer avec vos propres valeurs."
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
    read -p "Continuer quand même? (o/n): " continue_answer
    if [ "$continue_answer" != "o" ]; then
        exit 1
    fi
fi

# Lancer le scheduler en arrière-plan avec nohup pour qu'il continue à s'exécuter même si le terminal est fermé
echo ""
echo "Lancement du scheduler d'analyse de marché..."
echo "Le processus va s'exécuter en arrière-plan. Les logs seront écrits dans logs/market_analysis_scheduler.log"
echo ""

# Lancer le script Python
nohup python3 -c "from app.market_analysis_scheduler import run_market_analysis_scheduler; run_market_analysis_scheduler()" > logs/scheduler_output.log 2>&1 &

# Récupérer le PID du processus
PID=$!
echo "Le scheduler est démarré avec le PID: $PID"
echo "Pour arrêter le scheduler, utilisez la commande: kill $PID"

# Écrire le PID dans un fichier pour pouvoir l'arrêter facilement plus tard
echo $PID > .scheduler_pid

echo ""
echo "Vous pouvez suivre les logs avec la commande:"
echo "tail -f logs/market_analysis_scheduler.log"
echo "" 