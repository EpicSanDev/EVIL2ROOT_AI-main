#!/bin/bash

# Script pour lancer le bot d'analyse avec entraînement forcé des modèles
# Usage: ./start_train_and_analyze.sh

echo "=== Evil2Root Trading Bot - Analyse avec entraînement préalable ==="
echo "Ce script lance le bot d'analyse en forçant l'entraînement des modèles"
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

# Vérifier si des modèles existants sont présents
model_files=$(find saved_models -name "*.h5" -o -name "*.pkl" -o -name "*.model" 2>/dev/null | wc -l)
if [ "$model_files" -gt 0 ]; then
    echo "$model_files modèle(s) existant(s) trouvé(s). Ils seront réentraînés."
else
    echo "Aucun modèle existant trouvé. Les modèles seront entraînés à partir de zéro."
fi

echo ""
echo "Lancement du bot d'analyse avec entraînement forcé des modèles..."
echo "Appuyez sur Ctrl+C pour arrêter à tout moment."
echo ""

# Lancer le script Python avec l'option force-train
python3 start_daily_analysis.py --force-train 