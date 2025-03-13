#!/bin/bash
# Script pour vérifier si une Droplet builder existe déjà
# Utilisage: ./check-builder-exists.sh [DIGITALOCEAN_TOKEN]

set -e

# Vérifier si le token est fourni
if [ -z "$1" ]; then
  echo "Erreur: Token DigitalOcean non fourni"
  echo "Usage: $0 [DIGITALOCEAN_TOKEN]"
  exit 1
fi

DIGITALOCEAN_TOKEN=$1
DROPLET_NAME="evil2root-builder"
REGISTRY_NAME="evil2root-registry"

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

# Configurer l'authentification doctl
echo "Configuration de doctl avec votre token..."
doctl auth init -t $DIGITALOCEAN_TOKEN

# Vérifier si la Droplet existe déjà
echo "Vérification de l'existence de la Droplet $DROPLET_NAME..."
DROPLET_EXISTS=$(doctl compute droplet list --format Name --no-header | grep -w "$DROPLET_NAME" || echo "not_found")

if [[ "$DROPLET_EXISTS" == "$DROPLET_NAME" ]]; then
  DROPLET_IP=$(doctl compute droplet get $DROPLET_NAME --format PublicIPv4 --no-header)
  echo "✅ La Droplet $DROPLET_NAME existe déjà avec l'IP: $DROPLET_IP"
  echo "DROPLET_IP=$DROPLET_IP" > builder_info.env
  
  # Vérifier la santé de la Droplet
  echo "Vérification de la santé de la Droplet..."
  if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$DROPLET_IP 'uptime' 2>/dev/null; then
    echo "✅ La Droplet est accessible et fonctionne normalement"
    echo "DROPLET_STATUS=healthy" >> builder_info.env
  else
    echo "⚠️ La Droplet existe mais semble inaccessible"
    echo "DROPLET_STATUS=unreachable" >> builder_info.env
  fi
else
  echo "❌ La Droplet $DROPLET_NAME n'existe pas encore"
  echo "DROPLET_STATUS=not_found" > builder_info.env
  echo ""
  echo "Pour créer la Droplet builder, exécutez:"
  echo "./scripts/setup-builder-droplet.sh $DIGITALOCEAN_TOKEN"
fi

# Vérifier si le Container Registry existe
echo "Vérification de l'existence du Container Registry $REGISTRY_NAME..."
REGISTRY_EXISTS=$(doctl registry get 2>/dev/null | grep -w "$REGISTRY_NAME" || echo "not_found")

if [[ "$REGISTRY_EXISTS" != *"not_found"* ]]; then
  echo "✅ Le Container Registry $REGISTRY_NAME existe déjà"
  echo "REGISTRY_STATUS=exists" >> builder_info.env
else
  echo "❌ Le Container Registry $REGISTRY_NAME n'existe pas encore"
  echo "REGISTRY_STATUS=not_found" >> builder_info.env
fi

echo ""
echo "======================================================"
echo "Résumé:"
echo "Droplet builder: $(grep DROPLET_STATUS builder_info.env | cut -d= -f2)"
if [[ -f builder_info.env ]] && grep -q "DROPLET_IP" builder_info.env; then
  echo "IP de la Droplet: $(grep DROPLET_IP builder_info.env | cut -d= -f2)"
fi
echo "Container Registry: $(grep REGISTRY_STATUS builder_info.env | cut -d= -f2)"
echo "======================================================"
echo ""

if [[ -f builder_info.env ]] && grep -q "DROPLET_IP" builder_info.env && grep -q "DROPLET_STATUS=healthy" builder_info.env; then
  echo "Pour obtenir l'empreinte SSH de la Droplet pour GitHub Actions:"
  echo "ssh-keyscan $(grep DROPLET_IP builder_info.env | cut -d= -f2)"
  echo ""
  echo "Pour tester le build manuellement:"
  echo "ssh root@$(grep DROPLET_IP builder_info.env | cut -d= -f2) '/opt/builder/build.sh'"
fi 