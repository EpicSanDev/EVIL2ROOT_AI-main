#!/bin/bash
# Script pour authentifier la Droplet builder au registre DigitalOcean Container Registry
# Usage: ./authenticate-registry.sh [DIGITALOCEAN_TOKEN] [BUILDER_IP]

set -e

# Vérifier si les arguments sont fournis
if [ -z "$1" ]; then
  echo "Erreur: Token DigitalOcean non fourni"
  echo "Usage: $0 [DIGITALOCEAN_TOKEN] [BUILDER_IP]"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Erreur: IP de la Droplet builder non fournie"
  echo "Usage: $0 [DIGITALOCEAN_TOKEN] [BUILDER_IP]"
  exit 1
fi

DIGITALOCEAN_TOKEN=$1
BUILDER_IP=$2

echo "Configuration de l'authentification au registre pour la Droplet $BUILDER_IP..."

# Méthode 1: Utiliser doctl pour l'authentification
echo "Tentative d'authentification avec doctl..."
ssh root@$BUILDER_IP "echo \"$DIGITALOCEAN_TOKEN\" | doctl auth init --access-token-stdin && doctl registry login"

# Vérifier si l'authentification a réussi
if ssh root@$BUILDER_IP "docker pull registry.digitalocean.com/evil2root-registry/hello-world:latest" 2>/dev/null; then
  echo "✅ Authentification réussie avec doctl!"
  exit 0
fi

# Méthode 2: Authentification directe via Docker si doctl échoue
echo "Tentative d'authentification directe avec Docker..."
ssh root@$BUILDER_IP "echo \"$DIGITALOCEAN_TOKEN\" | docker login registry.digitalocean.com -u token --password-stdin"

# Vérifier à nouveau si l'authentification a réussi
if ssh root@$BUILDER_IP "docker pull registry.digitalocean.com/evil2root-registry/hello-world:latest" 2>/dev/null; then
  echo "✅ Authentification réussie avec Docker login!"
  exit 0
else
  echo "❌ Échec d'authentification au registre DigitalOcean Container Registry."
  echo "Vérifiez les éléments suivants:"
  echo "  1. Le token DigitalOcean est valide"
  echo "  2. Le token possède les permissions nécessaires pour accéder au registre"
  echo "  3. Le registre 'evil2root-registry' existe bien"
  exit 1
fi 