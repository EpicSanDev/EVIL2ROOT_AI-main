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
