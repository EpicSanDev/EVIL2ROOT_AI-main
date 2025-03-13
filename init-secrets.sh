#!/bin/bash

# Script d'initialisation des secrets pour Docker
# Ce script lit le fichier .env et génère des fichiers de secrets pour Docker

# Vérification que le fichier .env existe
if [ ! -f .env ]; then
    echo "Erreur : Fichier .env non trouvé"
    exit 1
fi

# Création du répertoire secrets s'il n'existe pas
mkdir -p secrets

# Extraction et création des fichiers de secrets
extract_secret() {
    local var_name=$1
    local value=$(grep "^$var_name=" .env | cut -d= -f2)
    
    if [ -n "$value" ]; then
        echo "$value" > "./secrets/${var_name,,}.txt"
        echo "Secret $var_name créé"
    else
        echo "Attention : $var_name non trouvé dans .env"
    fi
}

# Extraction des secrets depuis .env
extract_secret "DB_USER"
extract_secret "DB_PASSWORD"
extract_secret "SECRET_KEY"
extract_secret "ADMIN_PASSWORD"
extract_secret "TELEGRAM_TOKEN"
extract_secret "FINNHUB_API_KEY"
extract_secret "OPENROUTER_API_KEY"
extract_secret "COINBASE_API_KEY"
extract_secret "COINBASE_WEBHOOK_SECRET"

# Protection du répertoire secrets
chmod 700 secrets
chmod 600 secrets/*.txt

# Création du fichier .env.non-sensitive avec les variables non sensibles
grep -v "^\(DB_USER\|DB_PASSWORD\|SECRET_KEY\|ADMIN_PASSWORD\|TELEGRAM_TOKEN\|FINNHUB_API_KEY\|OPENROUTER_API_KEY\|COINBASE_API_KEY\|COINBASE_WEBHOOK_SECRET\)=" .env > .env.non-sensitive

echo "Configuration des secrets terminée"
echo "Utilisez 'docker-compose up' pour démarrer les services" 