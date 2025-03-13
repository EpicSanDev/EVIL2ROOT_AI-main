#!/bin/bash

# Script pour démarrer les services Docker avec entraînement forcé des modèles
# Usage: ./start_docker_force_train.sh

echo "=== Evil2Root Trading Bot - Démarrage avec entraînement forcé ==="
echo "Ce script lance le bot avec un entraînement forcé des modèles"
echo ""

# Vérifier les ressources disponibles
TOTAL_MEM=$(free -m | awk '/^Mem:/{print $2}')
echo "Mémoire totale disponible: ${TOTAL_MEM}MB"

# Demander à l'utilisateur s'il souhaite activer la surveillance de la mémoire
read -p "Activer la surveillance de la mémoire pendant l'entraînement ? (o/n): " monitor_memory

# Demander s'il faut limiter les symboles à entraîner
read -p "Limiter l'entraînement aux symboles essentiels (AAPL, GOOGL, BTC-USD) ? (o/n): " limit_symbols
if [ "$limit_symbols" = "o" ]; then
    export TRAIN_ONLY_ESSENTIAL_SYMBOLS=true
    echo "Entraînement limité aux symboles essentiels"
else
    export TRAIN_ONLY_ESSENTIAL_SYMBOLS=false
    echo "Tous les symboles seront entraînés"
fi

# Demander si on doit désactiver les modèles Transformer
read -p "Désactiver les modèles Transformer (recommandé pour économiser de la mémoire) ? (o/n): " disable_transformer
if [ "$disable_transformer" = "o" ]; then
    export USE_TRANSFORMER_MODEL=false
    echo "Modèles Transformer désactivés"
else
    export USE_TRANSFORMER_MODEL=true
    echo "Modèles Transformer activés"
fi

# Demander la complexité des modèles
echo "Niveau de complexité des modèles :"
echo "1) Faible (économise de la mémoire mais moins précis)"
echo "2) Moyen (équilibré)"
echo "3) Élevé (précis mais consomme beaucoup de mémoire)"
read -p "Choisir un niveau (1-3): " complexity_level

case $complexity_level in
    1)
        export MODEL_COMPLEXITY=low
        echo "Complexité faible sélectionnée"
        ;;
    3)
        export MODEL_COMPLEXITY=high
        echo "Complexité élevée sélectionnée"
        ;;
    *)
        export MODEL_COMPLEXITY=medium
        echo "Complexité moyenne sélectionnée"
        ;;
esac

# Exporter les variables pour limiter l'utilisation de la mémoire
export MAX_OPTIMIZATION_TRIALS=10
export OPTIMIZATION_TIMEOUT=1800
export USE_GPU=false
export BATCH_SIZE=32
export ENABLE_PARALLEL_TRAINING=false

# Démarrer la surveillance de la mémoire en arrière-plan si demandé
if [ "$monitor_memory" = "o" ]; then
    echo "Démarrage de la surveillance de la mémoire..."
    ./monitor_docker_memory.sh 10 &
    MONITOR_PID=$!
    echo "Surveillance démarrée avec PID: $MONITOR_PID"
fi

# Exécuter le script de démarrage avec l'option force-train
./start_docker.sh --force-train

# Arrêter la surveillance de la mémoire si elle est active
if [ "$monitor_memory" = "o" ] && [ -n "$MONITOR_PID" ]; then
    echo "Arrêt de la surveillance de la mémoire..."
    kill $MONITOR_PID
    echo "Vous pouvez consulter les logs de mémoire dans le dossier logs/"
fi 