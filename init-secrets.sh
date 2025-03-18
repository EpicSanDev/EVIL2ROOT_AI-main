#!/bin/bash

# Script d'initialisation des secrets pour Docker
# Ce script lit le fichier .env.local et génère des fichiers de secrets pour Docker

set -e

# Vérifier si .env.local existe
if [ ! -f .env.local ]; then
    echo "Erreur: Fichier .env.local non trouvé."
    echo "Veuillez créer un fichier .env.local basé sur .env avec vos secrets."
    echo "Exemple: cp .env .env.local puis modifiez .env.local avec vos secrets."
    exit 1
fi

# Création du répertoire secrets s'il n'existe pas
mkdir -p secrets

# Extraction et création des fichiers de secrets
extract_secret() {
    local var_name=$1
    local value
    
    # Extraire la valeur de la variable depuis .env.local
    value=$(grep "^$var_name=" .env.local | cut -d '=' -f2-)
    
    if [ -z "$value" ]; then
        echo "⚠️ Attention: Secret $var_name non trouvé dans .env.local"
        return 1
    fi
    
    # Écrire le secret dans un fichier
    echo "$value" > "./secrets/${var_name,,}.txt"
    echo "✅ Secret $var_name créé"
    return 0
}

# Extraction des secrets depuis .env.local
echo "🔒 Création des secrets Docker depuis .env.local..."
extract_secret "DB_USER" || echo "⚠️ Secret DB_USER requis pour le fonctionnement de l'application"
extract_secret "DB_PASSWORD" || echo "⚠️ Secret DB_PASSWORD requis pour le fonctionnement de l'application"
extract_secret "SECRET_KEY" || echo "⚠️ Secret SECRET_KEY requis pour le fonctionnement de l'application"
extract_secret "ADMIN_PASSWORD" || echo "⚠️ Secret ADMIN_PASSWORD requis pour le fonctionnement de l'application"
extract_secret "TELEGRAM_TOKEN" || echo "⚠️ Sans TELEGRAM_TOKEN, les notifications Telegram ne fonctionneront pas"
extract_secret "FINNHUB_API_KEY" || echo "⚠️ Sans FINNHUB_API_KEY, certaines données de marché ne seront pas disponibles"
extract_secret "OPENROUTER_API_KEY" || echo "⚠️ Sans OPENROUTER_API_KEY, la validation avancée par IA ne fonctionnera pas"
extract_secret "COINBASE_API_KEY" || echo "⚠️ Sans COINBASE_API_KEY, les paiements Coinbase ne fonctionneront pas"
extract_secret "COINBASE_WEBHOOK_SECRET" || echo "⚠️ Sans COINBASE_WEBHOOK_SECRET, les webhooks Coinbase ne fonctionneront pas"

# Protection du répertoire secrets
chmod 700 secrets
chmod 600 secrets/*.txt

# Création du fichier .env.non-sensitive sans les secrets
echo "📝 Création du fichier .env.non-sensitive..."
grep -v "^\(DB_USER\|DB_PASSWORD\|SECRET_KEY\|ADMIN_PASSWORD\|TELEGRAM_TOKEN\|FINNHUB_API_KEY\|OPENROUTER_API_KEY\|COINBASE_API_KEY\|COINBASE_WEBHOOK_SECRET\)=" .env.local > .env.non-sensitive

echo "✅ Configuration des secrets terminée"
echo "🔒 Les secrets sont stockés dans le répertoire 'secrets/' avec des permissions restrictives"
echo "⚠️ IMPORTANT: Ne jamais committer ce répertoire dans Git!"

# Vérification finale
if [ ! -f "./secrets/db_user.txt" ] || [ ! -f "./secrets/db_password.txt" ] || [ ! -f "./secrets/secret_key.txt" ]; then
    echo "❌ ERREUR: Certains secrets essentiels n'ont pas pu être créés."
    echo "Assurez-vous que DB_USER, DB_PASSWORD et SECRET_KEY sont définis dans .env.local"
    exit 1
fi

echo "🚀 Tout est prêt pour le déploiement!" 