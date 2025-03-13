#!/bin/bash

# Script de démarrage optimisé pour RTX 2070 SUPER + i5 + 64GB RAM
# Usage: ./start_rtx_train.sh

echo "=== Evil2Root AI Trading Bot - Démarrage optimisé RTX 2070 SUPER ==="
echo "Configuration: RTX 2070 SUPER + i5 + 64GB RAM"
echo ""

# Vérifier si Docker est installé
if ! command -v docker &> /dev/null; then
    echo "ERREUR: Docker n'est pas installé."
    exit 1
fi

# Vérifier si NVIDIA Docker est installé
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    echo "ERREUR: NVIDIA Docker n'est pas configuré correctement."
    echo "Veuillez exécuter le script d'installation: sudo ./install_nvidia_docker.sh"
    exit 1
fi

# Vérifier les informations GPU
echo "=== GPU détecté ==="
nvidia-smi
echo ""

# Vérifier la mémoire disponible
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
FREE_MEM=$(free -m | awk '/^Mem:/{print $4}')
echo "Mémoire système: ${TOTAL_MEM}MB total, ${FREE_MEM}MB libre"
echo ""

# Configuration optimale pour RTX 2070 SUPER
cat > .env.gpu << EOL
# Configuration optimisée pour RTX 2070 SUPER + i5 + 64GB RAM
USE_GPU=true
ENABLE_PARALLEL_TRAINING=true
MAX_PARALLEL_MODELS=3
MAX_OPTIMIZATION_TRIALS=15
OPTIMIZATION_TIMEOUT=3600
BATCH_SIZE=64
MODEL_COMPLEXITY=high
SEQUENCE_LENGTH=60
FORECAST_HORIZON=5
TRANSFORMER_LAYERS=3
LSTM_UNITS=128
TRANSFORMER_MODEL_ENABLED=true
# Conservez tous les symboles pour l'entraînement
TRAIN_ONLY_ESSENTIAL_SYMBOLS=false
EOL

# Fusionner avec le fichier .env existant
if [ -f .env ]; then
    echo "Fusion de la configuration GPU avec .env existant..."
    cat .env.gpu >> .env
    echo "Configuration GPU ajoutée à .env"
else
    echo "ERREUR: Fichier .env non trouvé. Veuillez créer un fichier .env de base."
    exit 1
fi

# Nettoyer les logs Docker et les conteneurs arrêtés
echo "Nettoyage des logs et conteneurs arrêtés..."
docker system prune -f

# Activer la surveillance de la mémoire Docker pendant l'entraînement
echo "Démarrage de la surveillance GPU et mémoire..."
./monitor_docker_memory.sh 10 &
MONITOR_PID=$!

# Dans un autre terminal, afficher les stats GPU
gnome-terminal -- bash -c "watch -n 5 nvidia-smi" || 
konsole -e "watch -n 5 nvidia-smi" || 
xterm -e "watch -n 5 nvidia-smi" || 
echo "Impossible de démarrer un terminal pour surveiller le GPU. Continuez manuellement."

# Démarrer l'entraînement avec configuration GPU
echo "Démarrage de l'entraînement avec support GPU RTX 2070 SUPER..."
docker-compose down
docker-compose up -d db redis
echo "Attente du démarrage de la base de données et Redis (10 secondes)..."
sleep 10

echo "Lancement de l'entraînement des modèles avec GPU..."
docker-compose up train-and-analyze

# Après la fin de l'entraînement, démarrer les services normaux
echo "Entraînement terminé, démarrage des services normaux..."
docker-compose up -d web market-scheduler-force-train analysis-bot

# Arrêter la surveillance de la mémoire
kill $MONITOR_PID 2>/dev/null

echo ""
echo "=== Services démarrés avec succès! ==="
echo "Modèles entraînés avec accélération GPU RTX 2070 SUPER"
echo ""
echo "Pour voir les logs:"
echo "- Application web: docker logs -f trading-bot-web"
echo "- Bot d'analyse: docker logs -f trading-bot-analysis"
echo "- Scheduler de marché: docker logs -f trading-bot-market-scheduler-force-train"
echo ""
echo "Pour arrêter les services: ./stop_docker.sh ou docker-compose down" 