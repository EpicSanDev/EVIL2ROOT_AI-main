#!/bin/bash
# Script pour compiler une image Docker sur un Droplet DigitalOcean
# et la pousser vers DigitalOcean Container Registry

# Configuration
DROPLET_IP="votre_ip_droplet"
REGISTRY_NAME="your-registry-name"
IMAGE_NAME="evil2root-ai"
SSH_KEY="~/.ssh/id_rsa"  # Chemin vers votre clé SSH

echo "🚀 Copie des fichiers vers le Droplet..."
rsync -avz --exclude 'node_modules' --exclude '.git' \
  --exclude 'data' --exclude 'logs' --exclude 'saved_models' \
  -e "ssh -i $SSH_KEY" ./ root@${DROPLET_IP}:/root/app/

echo "🔨 Compilation de l'image Docker sur le Droplet..."
ssh -i $SSH_KEY root@${DROPLET_IP} "cd /root/app && \
  docker build -t registry.digitalocean.com/${REGISTRY_NAME}/${IMAGE_NAME}:latest . && \
  doctl auth init && \
  doctl registry login && \
  docker push registry.digitalocean.com/${REGISTRY_NAME}/${IMAGE_NAME}:latest"

echo "✅ Image compilée et poussée avec succès!" 