#!/bin/bash
# Script complet pour configurer l'environnement de build sur DigitalOcean
# Ce script:
# 1. Vérifie si une Droplet builder existe déjà
# 2. Crée une nouvelle Droplet si elle n'existe pas
# 3. Configure les clés SSH pour GitHub Actions
#
# Usage: ./setup-builder-complete.sh [DIGITALOCEAN_TOKEN]

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
KEY_PATH=~/.ssh/evil2root_builder_key

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

# Vérifier si la clé SSH existe déjà, sinon la créer
if [ ! -f "$KEY_PATH" ]; then
  echo "Génération d'une nouvelle paire de clés SSH..."
  ssh-keygen -t ed25519 -N "" -f $KEY_PATH
  echo "Clé SSH générée: $KEY_PATH"
else
  echo "Utilisation de la clé SSH existante: $KEY_PATH"
fi

# Importer la clé SSH dans DigitalOcean
echo "Import de la clé SSH dans DigitalOcean..."
SSH_KEY_EXISTS=$(doctl compute ssh-key list --format ID,Name --no-header | grep -w "evil2root-builder-key" || echo "not_found")

if [[ "$SSH_KEY_EXISTS" == *"not_found"* ]]; then
  echo "Ajout de la clé SSH..."
  doctl compute ssh-key import evil2root-builder-key --public-key-file $KEY_PATH.pub
fi

# Récupérer l'ID de la clé SSH
SSH_KEY_ID=$(doctl compute ssh-key list --format ID,Name --no-header | grep evil2root-builder-key | awk '{print $1}')
if [ -z "$SSH_KEY_ID" ]; then
  echo "Erreur: Impossible de récupérer l'ID de la clé SSH"
  exit 1
fi

# Vérifier si la Droplet existe déjà
echo "Vérification de l'existence de la Droplet $DROPLET_NAME..."
DROPLET_EXISTS=$(doctl compute droplet list --format Name --no-header | grep -w "$DROPLET_NAME" || echo "not_found")

if [[ "$DROPLET_EXISTS" == "$DROPLET_NAME" ]]; then
  echo "✅ La Droplet $DROPLET_NAME existe déjà"
  # Récupérer l'adresse IP de la Droplet
  DROPLET_IP=$(doctl compute droplet get $DROPLET_NAME --format PublicIPv4 --no-header)
  echo "IP de la Droplet: $DROPLET_IP"
  
  # Vérifier que la clé SSH est bien configurée sur la Droplet
  echo "Ajout de la clé SSH à la Droplet..."
  doctl compute droplet-action add-droplet-key $DROPLET_NAME --key-id $SSH_KEY_ID --wait
else
  echo "❌ La Droplet $DROPLET_NAME n'existe pas encore, création en cours..."
  
  # Créer la Droplet
  doctl compute droplet create $DROPLET_NAME \
    --image docker-20-04 \
    --size s-2vcpu-4gb \
    --region fra1 \
    --ssh-keys $SSH_KEY_ID \
    --wait
  
  # Récupérer l'adresse IP de la Droplet
  DROPLET_IP=$(doctl compute droplet get $DROPLET_NAME --format PublicIPv4 --no-header)
  echo "Droplet créée avec l'adresse IP: $DROPLET_IP"
  
  # Initialisation de la Droplet
  echo "Attente de l'initialisation complète de la Droplet..."
  
  # Fonction pour vérifier si SSH est prêt
  wait_for_ssh() {
    local ip=$1
    local max_attempts=20
    local attempt=1
    local sleep_time=15
    
    echo "Vérification de la disponibilité du service SSH..."
    while [ $attempt -le $max_attempts ]; do
      echo "Tentative $attempt/$max_attempts..."
      
      if nc -z -w 5 $ip 22 2>/dev/null; then
        echo "✅ Service SSH accessible!"
        return 0
      else
        echo "Service SSH pas encore disponible, nouvelle tentative dans ${sleep_time} secondes..."
        sleep $sleep_time
        attempt=$((attempt+1))
      fi
    done
    
    echo "❌ Échec de connexion au service SSH après $max_attempts tentatives."
    return 1
  }
  
  # Attendre que SSH soit disponible
  if ! wait_for_ssh $DROPLET_IP; then
    echo "⚠️ Impossible de se connecter au service SSH. Vérifiez l'état de la Droplet manuellement."
    echo "doctl compute droplet get $DROPLET_NAME"
    exit 1
  fi
  
  # Générer un script d'initialisation pour la Droplet
  cat > init-droplet.sh << 'EOF'
#!/bin/bash
set -e

# Mise à jour du système
apt-get update
apt-get upgrade -y

# Installation des dépendances
apt-get install -y git rsync jq

# Vérifier que Docker est installé
if ! command -v docker &> /dev/null; then
  echo "Docker n'est pas installé. Installation..."
  apt-get install -y docker.io
  systemctl enable docker
  systemctl start docker
fi

# Installer doctl
if ! command -v doctl &> /dev/null; then
  echo "Installation de doctl..."
  curl -sL https://github.com/digitalocean/doctl/releases/download/v1.92.1/doctl-1.92.1-linux-amd64.tar.gz | tar -xzv
  mv doctl /usr/local/bin
fi

# Créer répertoire de travail
mkdir -p /opt/builder
chmod 755 /opt/builder

# Créer un script de build qui sera exécuté par le webhook
cat > /opt/builder/build.sh << 'EOFINNER'
#!/bin/bash
set -e

echo "$(date) - Démarrage du build"

# Chemin du repository
REPO_PATH="/opt/builder/repo"

# Vider ou créer le répertoire du repo
rm -rf $REPO_PATH
mkdir -p $REPO_PATH

# Cloner le repository
git clone https://github.com/votre-compte/Evil2Root_TRADING $REPO_PATH
cd $REPO_PATH

# Se connecter au registry
doctl auth init -t $DIGITALOCEAN_TOKEN
doctl registry login

# Construire l'image
echo "Construction de l'image Docker..."
docker build -t registry.digitalocean.com/evil2root-registry/evil2root-ai:latest .

# Pousser l'image vers le registry
echo "Push de l'image vers DigitalOcean Container Registry..."
docker push registry.digitalocean.com/evil2root-registry/evil2root-ai:latest

# Nettoyer pour économiser de l'espace
docker system prune -af

echo "$(date) - Build terminé avec succès"
EOFINNER

chmod +x /opt/builder/build.sh

# Créer un service systemd pour exécuter un webhook simple
cat > /etc/systemd/system/webhook.service << 'EOFSERVICE'
[Unit]
Description=GitHub Webhook
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/builder
ExecStart=/usr/bin/python3 -m http.server 8080
Restart=always

[Install]
WantedBy=multi-user.target
EOFSERVICE

# Activer et démarrer le service
systemctl enable webhook
systemctl start webhook

echo "Configuration terminée!"
EOF

  # Rendre le script exécutable
  chmod +x init-droplet.sh
  
  # Pour éviter les problèmes de first-time connection avec SSH
  echo "Récupération de l'empreinte SSH de la Droplet..."
  max_attempts=5
  attempt=1
  
  while [ $attempt -le $max_attempts ]; do
    BUILDER_HOST_KEY=$(ssh-keyscan $DROPLET_IP 2>/dev/null)
    if [ ! -z "$BUILDER_HOST_KEY" ]; then
      echo "✅ Empreinte SSH récupérée avec succès!"
      break
    fi
    echo "Tentative $attempt/$max_attempts d'obtenir l'empreinte SSH..."
    sleep 10
    attempt=$((attempt+1))
  done
  
  if [ -z "$BUILDER_HOST_KEY" ]; then
    echo "⚠️ Impossible de récupérer l'empreinte SSH après $max_attempts tentatives."
    exit 1
  fi
  
  # Ajouter la clé SSH à known_hosts
  echo "Ajout de l'empreinte à ~/.ssh/known_hosts..."
  mkdir -p ~/.ssh
  echo "$BUILDER_HOST_KEY" >> ~/.ssh/known_hosts
  
  # Fonctions pour gérer les tentatives de connexion SSH
  try_ssh_command() {
    local ip=$1
    local key=$2
    local cmd=$3
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
      echo "Tentative $attempt/$max_attempts d'exécution de la commande SSH..."
      
      if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $key root@$ip "$cmd" 2>/dev/null; then
        return 0
      else
        echo "Échec de la commande SSH, nouvelle tentative dans 10 secondes..."
        sleep 10
        attempt=$((attempt+1))
      fi
    done
    
    return 1
  }
  
  try_scp_command() {
    local ip=$1
    local key=$2
    local src=$3
    local dest=$4
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
      echo "Tentative $attempt/$max_attempts de copie SCP..."
      
      if scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $key $src root@$ip:$dest 2>/dev/null; then
        return 0
      else
        echo "Échec de la copie SCP, nouvelle tentative dans 10 secondes..."
        sleep 10
        attempt=$((attempt+1))
      fi
    done
    
    return 1
  }
  
  # Copier le script vers la Droplet
  echo "Copie du script d'initialisation vers la Droplet..."
  if ! try_scp_command $DROPLET_IP $KEY_PATH init-droplet.sh /root/; then
    echo "⚠️ Échec de la copie du script d'initialisation après plusieurs tentatives."
    echo "Vérifiez manuellement l'état de la Droplet et réessayez plus tard."
    exit 1
  fi
  
  # Exécuter le script d'initialisation
  echo "Exécution du script d'initialisation sur la Droplet..."
  if ! try_ssh_command $DROPLET_IP $KEY_PATH 'bash /root/init-droplet.sh'; then
    echo "⚠️ Échec de l'exécution du script d'initialisation après plusieurs tentatives."
    echo "Connectez-vous manuellement à la Droplet et exécutez: bash /root/init-droplet.sh"
    exit 1
  fi
  
  # Nettoyer le script local
  rm init-droplet.sh
  
  echo "✅ Initialisation de la Droplet terminée avec succès!"
fi

# Obtenir l'empreinte SSH de la Droplet
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

# Vérifier si le Container Registry existe
echo "Vérification de l'existence du Container Registry $REGISTRY_NAME..."
REGISTRY_EXISTS=$(doctl registry get 2>/dev/null || echo "not_found")

if [[ "$REGISTRY_EXISTS" == *"not_found"* ]]; then
  echo "Création du Container Registry $REGISTRY_NAME..."
  doctl registry create $REGISTRY_NAME
else
  echo "Le Container Registry existe déjà."
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
echo "DIGITALOCEAN_ACCESS_TOKEN:"
echo "(Votre token DigitalOcean)"
echo ""
echo "Pour tester la connexion SSH à votre Droplet builder:"
echo "ssh -i $KEY_PATH root@$DROPLET_IP 'echo Connexion réussie!'"
echo ""
echo "Pour tester manuellement le build:"
echo "ssh -i $KEY_PATH root@$DROPLET_IP '/opt/builder/build.sh'" 