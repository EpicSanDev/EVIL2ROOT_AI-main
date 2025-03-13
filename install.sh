#!/bin/bash

# Script d'installation pour EVIL2ROOT Trading Bot
echo "=============================================="
echo "   Installation de EVIL2ROOT Trading Bot      "
echo "=============================================="

# Vérifier si Python 3.8+ est installé
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    echo "Erreur: Python 3.8 ou supérieur est requis, version actuelle: $python_version"
    echo "Veuillez installer une version compatible de Python et réessayer."
    exit 1
fi

echo "Python $python_version détecté."

# Créer un environnement virtuel
echo "Création de l'environnement virtuel..."
python3 -m venv venv
source venv/bin/activate

# Mettre à jour pip
echo "Mise à jour de pip..."
pip install --upgrade pip wheel setuptools

# Installer les dépendances
echo "Installation des dépendances..."
pip install -r requirements.txt

# Installer le package en mode développement
echo "Installation du package en mode développement..."
pip install -e .

# Créer les dossiers de logs et data s'ils n'existent pas
echo "Création des dossiers nécessaires..."
mkdir -p logs data data/historical data/processed data/models

# Copier le fichier .env.example vers .env si .env n'existe pas
if [ ! -f .env ]; then
    echo "Création du fichier .env à partir de .env.example..."
    cp config/environments/.env.example .env
    echo "ATTENTION: Veuillez modifier le fichier .env avec vos propres paramètres."
fi

# Rendre les scripts exécutables
echo "Configuration des permissions pour les scripts..."
chmod +x scripts/shell/*.sh

echo "=============================================="
echo "L'installation est terminée !"
echo "Pour démarrer le bot, utilisez : ./scripts/shell/start_docker.sh"
echo "ou en mode développement : python src/main.py"
echo "==============================================" 