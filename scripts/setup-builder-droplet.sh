#!/bin/bash
# =====================================================
# ATTENTION : CE SCRIPT DOIT ÊTRE EXÉCUTÉ UNE SEULE FOIS
# =====================================================
# Ce script crée une Droplet DigitalOcean qui servira de serveur de build.
# Il ne doit PAS être exécuté à chaque push, mais uniquement lors de la
# configuration initiale de votre environnement.
#
# Après l'exécution de ce script, les builds se feront automatiquement
# via GitHub Actions qui utilisera la Droplet existante.
# =====================================================
# Utilisage: ./setup-builder-droplet.sh [DIGITALOCEAN_TOKEN]

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

# Récupérer l'ID de la clé SSH
echo "Récupération de votre clé SSH..."
SSH_KEYS=$(doctl compute ssh-key list --format ID,Name --no-header)
if [ -z "$SSH_KEYS" ]; then
  echo "Aucune clé SSH trouvée. Veuillez ajouter une clé SSH à votre compte DigitalOcean."
  exit 1
fi
echo "Clés SSH disponibles:"
echo "$SSH_KEYS"
echo "Entrez l'ID de la clé SSH à utiliser:"
read SSH_KEY_ID

# Créer la Droplet
echo "Création de la Droplet $DROPLET_NAME..."
doctl compute droplet create $DROPLET_NAME \
  --image docker-20-04 \
  --size s-2vcpu-4gb \
  --region fra1 \
  --ssh-keys $SSH_KEY_ID \
  --wait

# Attendre que la Droplet soit prête
echo "Attente de l'initialisation de la Droplet..."
sleep 30

# Récupérer l'adresse IP de la Droplet
DROPLET_IP=$(doctl compute droplet get $DROPLET_NAME --format PublicIPv4 --no-header)
echo "Droplet créée avec l'adresse IP: $DROPLET_IP"

# Créer le Container Registry s'il n'existe pas déjà
REGISTRY_EXISTS=$(doctl registry get || echo "not_found")
if [[ "$REGISTRY_EXISTS" == *"not_found"* ]]; then
  echo "Création du Container Registry $REGISTRY_NAME..."
  doctl registry create $REGISTRY_NAME
else
  echo "Le Container Registry existe déjà."
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

# Copier le script vers la Droplet
echo "Copie du script d'initialisation vers la Droplet..."
scp -o StrictHostKeyChecking=no init-droplet.sh root@$DROPLET_IP:/root/

# Exécuter le script d'initialisation
echo "Exécution du script d'initialisation sur la Droplet..."
ssh -o StrictHostKeyChecking=no root@$DROPLET_IP 'bash /root/init-droplet.sh'

# Nettoyer le script local
rm init-droplet.sh

echo "======================================================"
echo "Configuration terminée!"
echo "Voici les informations pour les secrets GitHub:"
echo "DIGITALOCEAN_ACCESS_TOKEN: Votre token DigitalOcean"
echo "BUILDER_IP: $DROPLET_IP"
echo "BUILDER_SSH_KEY: Votre clé SSH privée"
echo "BUILDER_HOST_KEY: Exécutez 'ssh-keyscan $DROPLET_IP' pour l'obtenir"
echo "======================================================"
echo "Pour tester manuellement le build:"
echo "ssh root@$DROPLET_IP '/opt/builder/build.sh'"
echo "======================================================" 