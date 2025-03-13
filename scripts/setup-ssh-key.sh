#!/bin/bash
# Script pour configurer la clé SSH pour le déploiement DigitalOcean
# Usage: ./setup-ssh-key.sh [DROPLET_IP] [DIGITALOCEAN_TOKEN]

set -e

# Vérifier les arguments
if [ -z "$1" ]; then
  echo "Erreur: IP de la Droplet non fournie"
  echo "Usage: $0 [DROPLET_IP] [DIGITALOCEAN_TOKEN]"
  exit 1
fi

DROPLET_IP=$1
DIGITALOCEAN_TOKEN=$2
KEY_PATH=~/.ssh/evil2root_builder_key

# Vérifier si la clé SSH existe déjà
if [ ! -f "$KEY_PATH" ]; then
  echo "Génération d'une nouvelle paire de clés SSH..."
  ssh-keygen -t ed25519 -N "" -f $KEY_PATH
  echo "Clé SSH générée: $KEY_PATH"
else
  echo "Utilisation de la clé SSH existante: $KEY_PATH"
fi

# Obtenir l'empreinte SSH du serveur (host key)
echo "Récupération de l'empreinte SSH de la Droplet..."
BUILDER_HOST_KEY=$(ssh-keyscan $DROPLET_IP 2>/dev/null)

if [ -z "$BUILDER_HOST_KEY" ]; then
  echo "⚠️ Impossible de récupérer l'empreinte SSH. Vérifiez que la Droplet est accessible."
  exit 1
fi

# Ajouter la clé SSH à known_hosts
echo "Ajout de l'empreinte à ~/.ssh/known_hosts..."
mkdir -p ~/.ssh
echo "$BUILDER_HOST_KEY" >> ~/.ssh/known_hosts

# Copier la clé publique vers la Droplet (si token DO fourni)
if [ ! -z "$DIGITALOCEAN_TOKEN" ]; then
  echo "Copie de la clé publique vers la Droplet..."
  
  # Installer doctl si nécessaire
  if ! command -v doctl &> /dev/null; then
    echo "Installation de doctl..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS
      brew install doctl
    else
      # Ubuntu/Debian
      curl -sL https://github.com/digitalocean/doctl/releases/download/v1.92.1/doctl-1.92.1-linux-amd64.tar.gz | tar -xzv
      sudo mv doctl /usr/local/bin
    fi
  fi
  
  # Authentification doctl
  doctl auth init -t $DIGITALOCEAN_TOKEN
  
  # Copier la clé SSH via l'API DigitalOcean
  PUBLIC_KEY=$(cat $KEY_PATH.pub)
  
  echo "Ajout de la clé SSH à la Droplet via l'API DigitalOcean..."
  doctl compute ssh-key import evil2root-builder-key --public-key-file $KEY_PATH.pub
  
  # Obtenir l'ID de la clé
  KEY_ID=$(doctl compute ssh-key list --format ID,Name --no-header | grep evil2root-builder-key | awk '{print $1}')
  
  if [ ! -z "$KEY_ID" ]; then
    echo "Ajout de la clé SSH à la Droplet..."
    doctl compute droplet-action add-droplet-key $DROPLET_IP --key-id $KEY_ID
  fi
else
  echo "Token DigitalOcean non fourni. Ajoutez manuellement la clé SSH à la Droplet avec:"
  echo "cat $KEY_PATH.pub | ssh root@$DROPLET_IP 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'"
fi

# Sortie pour configurer GitHub Actions
echo ""
echo "======================================================"
echo "Configuration pour GitHub Actions"
echo "======================================================"
echo ""
echo "1. Dans votre repository GitHub, allez dans Settings > Secrets and variables > Actions"
echo "2. Ajoutez les secrets suivants:"
echo ""
echo "BUILDER_SSH_KEY:"
echo "--------------------------"
cat $KEY_PATH
echo "--------------------------"
echo ""
echo "BUILDER_HOST_KEY:"
echo "--------------------------"
echo "$BUILDER_HOST_KEY"
echo "--------------------------"
echo ""
echo "BUILDER_IP:"
echo "--------------------------"
echo "$DROPLET_IP"
echo "--------------------------"
echo ""
echo "3. Vérifiez que vous avez également configuré DIGITALOCEAN_ACCESS_TOKEN"
echo ""
echo "Pour tester la connexion SSH à votre Droplet builder:"
echo "ssh -i $KEY_PATH root@$DROPLET_IP 'echo Connexion réussie!'" 