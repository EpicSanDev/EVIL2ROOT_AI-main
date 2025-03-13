#!/bin/bash

# Script de déploiement pour Evil2Root Trading Bot sur Digital Ocean (Ubuntu)
# Ce script installe toutes les dépendances nécessaires et configure l'environnement

set -e  # Arrêter l'exécution en cas d'erreur

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Déploiement de Evil2Root Trading Bot sur Digital Ocean (Ubuntu) ===${NC}"

# Mettre à jour le système
echo -e "${YELLOW}[1/10] Mise à jour du système...${NC}"
sudo apt-get update
sudo apt-get upgrade -y

# Installer les dépendances
echo -e "${YELLOW}[2/10] Installation des dépendances...${NC}"
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    make \
    python3-pip \
    ufw

# Configurer le pare-feu
echo -e "${YELLOW}[3/10] Configuration du pare-feu...${NC}"
sudo ufw allow ssh
sudo ufw allow 5001/tcp  # Port pour l'interface web
sudo ufw --force enable

# Installer Docker
echo -e "${YELLOW}[4/10] Installation de Docker...${NC}"
if ! command -v docker &> /dev/null; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    sudo systemctl enable docker
    sudo systemctl start docker
    
    # Ajouter l'utilisateur actuel au groupe docker
    sudo usermod -aG docker $USER
    echo -e "${GREEN}Docker installé. Vous devrez vous reconnecter pour utiliser Docker sans sudo.${NC}"
else
    echo -e "${GREEN}Docker est déjà installé.${NC}"
fi

# Installer Docker Compose
echo -e "${YELLOW}[5/10] Installation de Docker Compose...${NC}"
if ! command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
    sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}Docker Compose installé.${NC}"
else
    echo -e "${GREEN}Docker Compose est déjà installé.${NC}"
fi

# Cloner le dépôt (si non existant)
echo -e "${YELLOW}[6/10] Clonage/Mise à jour du dépôt...${NC}"
APP_DIR="$HOME/EVIL2ROOT_AI"

if [ -d "$APP_DIR" ]; then
    echo "Le répertoire existe déjà, mise à jour..."
    cd "$APP_DIR"
    git pull
else
    echo "Clonage du dépôt..."
    git clone https://github.com/Evil2Root/EVIL2ROOT_AI.git "$APP_DIR"
    cd "$APP_DIR"
fi

# Configuration de l'environnement
echo -e "${YELLOW}[7/10] Configuration de l'environnement...${NC}"
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "Création du fichier .env à partir de .env.example..."
        cp .env.example .env
        
        # Demander à l'utilisateur de configurer certaines variables clés
        read -p "Entrez votre clé API OpenRouter pour Claude (OPENROUTER_API_KEY): " openrouter_key
        if [ ! -z "$openrouter_key" ]; then
            sed -i "s/OPENROUTER_API_KEY=your_key_here/OPENROUTER_API_KEY=$openrouter_key/g" .env
        fi
        
        # Pour le trading en direct, demander confirmation
        read -p "Activer le trading en direct? (oui/non - par défaut: non): " enable_live
        if [ "$enable_live" = "oui" ]; then
            sed -i 's/ENABLE_LIVE_TRADING=false/ENABLE_LIVE_TRADING=true/g' .env
            echo -e "${RED}ATTENTION: Trading en direct activé!${NC}"
        fi
        
        echo -e "${YELLOW}Veuillez vérifier et compléter les autres variables dans le fichier .env${NC}"
    else
        echo -e "${RED}Fichier .env.example introuvable. Veuillez créer manuellement un fichier .env${NC}"
        exit 1
    fi
fi

# Création des dossiers nécessaires
echo -e "${YELLOW}[8/10] Création des dossiers nécessaires...${NC}"
mkdir -p data logs saved_models
chmod -R 777 logs data saved_models

# Préparation des scripts d'entrée
echo -e "${YELLOW}[9/10] Préparation des scripts...${NC}"
chmod +x docker/services/entrypoint-*.sh
chmod +x start_production.sh

# Démarrage des services
echo -e "${YELLOW}[10/10] Démarrage des services...${NC}"
docker-compose down || true  # Arrêter les conteneurs existants si nécessaire
docker-compose up -d --build

# Afficher le statut des conteneurs
echo -e "${GREEN}=== Déploiement terminé! ===${NC}"
echo -e "${YELLOW}Vérification de l'état des conteneurs:${NC}"
sleep 10
docker-compose ps

# Configuration des tâches CRON pour les redémarrages périodiques (optionnel)
read -p "Voulez-vous configurer un redémarrage automatique quotidien? (oui/non): " setup_cron
if [ "$setup_cron" = "oui" ]; then
    # Ajouter une tâche cron pour redémarrer les services chaque jour à 4h du matin
    (crontab -l 2>/dev/null; echo "0 4 * * * cd $APP_DIR && docker-compose restart") | crontab -
    echo -e "${GREEN}Tâche CRON configurée pour redémarrer les services chaque jour à 4h du matin.${NC}"
fi

# Informations finales
echo -e "${GREEN}=== Informations importantes ===${NC}"
echo -e "IP du serveur: $(curl -s ifconfig.me)"
echo -e "Interface web disponible à l'adresse: http://$(curl -s ifconfig.me):5001"
echo -e "Dossier de l'application: $APP_DIR"
echo -e ""
echo -e "Commandes utiles:"
echo -e "- Voir les logs: cd $APP_DIR && docker-compose logs -f"
echo -e "- Redémarrer les services: cd $APP_DIR && docker-compose restart"
echo -e "- Arrêter les services: cd $APP_DIR && docker-compose down"
echo -e "- Mettre à jour l'application: cd $APP_DIR && git pull && docker-compose up -d --build" 