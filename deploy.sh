#!/bin/bash
set -e

# Afficher la date et l'heure de début du déploiement
echo "===== Déploiement démarré le $(date) ====="

# Variables
REPO_URL="https://github.com/yourusername/Evil2Root_TRADING.git"
DEPLOY_DIR="/opt/trading-bot"
BACKUP_DIR="/opt/backups/trading-bot/$(date +%Y%m%d_%H%M%S)"

# Créer un répertoire de sauvegarde
echo "Création d'une sauvegarde..."
mkdir -p $BACKUP_DIR
if [ -d "$DEPLOY_DIR" ]; then
    cp -r $DEPLOY_DIR/data $BACKUP_DIR/ 2>/dev/null || true
    cp -r $DEPLOY_DIR/logs $BACKUP_DIR/ 2>/dev/null || true
    cp -r $DEPLOY_DIR/saved_models $BACKUP_DIR/ 2>/dev/null || true
    cp $DEPLOY_DIR/.env $BACKUP_DIR/ 2>/dev/null || true
fi

# Récupérer la dernière version depuis le registre Docker
echo "Téléchargement de la dernière image Docker..."
docker pull evil2root/trading-bot:latest

# Préserver les données importantes
echo "Préservation des données importantes..."
mkdir -p $DEPLOY_DIR/data
mkdir -p $DEPLOY_DIR/logs
mkdir -p $DEPLOY_DIR/saved_models

# Arrêter et supprimer les conteneurs existants
echo "Arrêt des services existants..."
cd $DEPLOY_DIR
docker-compose down || true

# Récupérer la dernière version du code
echo "Récupération du code source..."
if [ ! -d "$DEPLOY_DIR/.git" ]; then
    # Premier déploiement
    git clone $REPO_URL $DEPLOY_DIR
    cd $DEPLOY_DIR
else
    # Mise à jour
    cd $DEPLOY_DIR
    git pull
fi

# Vérifier et mettre à jour le .env si nécessaire
if [ ! -f "$DEPLOY_DIR/.env" ]; then
    echo "Création du fichier .env..."
    cp .env.example .env
    echo "ATTENTION: Veuillez éditer le fichier .env avec vos variables d'environnement"
fi

# Redémarrer avec la nouvelle image
echo "Démarrage des services..."
docker-compose up -d

# Vérifier l'état des conteneurs
echo "Vérification de l'état des conteneurs..."
docker-compose ps

# Nettoyer les images anciennes pour économiser de l'espace disque
echo "Nettoyage des anciennes images..."
docker image prune -af --filter "until=24h"

# Rotation des sauvegardes (conserver les 5 dernières)
echo "Rotation des sauvegardes..."
ls -dt /opt/backups/trading-bot/* | tail -n +6 | xargs rm -rf

echo "===== Déploiement terminé le $(date) =====" 