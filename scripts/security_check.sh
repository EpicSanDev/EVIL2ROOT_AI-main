#!/bin/bash

# Script de vérification et d'amélioration de la sécurité du projet Evil2Root Trading Bot
# Ce script doit être exécuté régulièrement pour s'assurer que le projet suit les bonnes pratiques de sécurité.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

cd "$PROJECT_ROOT"

echo -e "${GREEN}= = = = = = = = = = = = = = = = = = = = = = =${NC}"
echo -e "${GREEN}=   Vérification de sécurité Evil2Root AI   =${NC}"
echo -e "${GREEN}= = = = = = = = = = = = = = = = = = = = = = =${NC}"
echo ""

# Vérifier la présence de secrets dans le code source
check_secrets() {
    echo -e "${YELLOW}Vérification des secrets dans le code source...${NC}"
    
    # Modèles de recherche pour les secrets
    PATTERNS=(
        'api_key'
        'apikey'
        'secret'
        'password'
        'token'
        'BEGIN PRIVATE KEY'
        'BEGIN RSA PRIVATE KEY'
    )
    
    # Fichiers à exclure (résultats normaux/attendus)
    EXCLUDE=(
        '.git/'
        'docs/SECRETS_MANAGEMENT.md'
        'scripts/security_check.sh'
        '.env.example'
        '.gitignore'
    )
    
    # Construire la commande grep avec les exclusions
    EXCLUDE_ARGS=""
    for pattern in "${EXCLUDE[@]}"; do
        EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude-dir=$pattern"
    done
    
    # Rechercher les secrets
    FOUND_SECRETS=false
    for pattern in "${PATTERNS[@]}"; do
        RESULTS=$(grep -r -i -l $EXCLUDE_ARGS "$pattern" --include="*.py" --include="*.sh" --include="*.yml" --include="*.yaml" --include="*.json" --include="*.md" --include="*.env" . || true)
        
        if [ -n "$RESULTS" ]; then
            echo -e "${RED}⚠️ Secrets potentiels trouvés pour le modèle: $pattern${NC}"
            echo "$RESULTS"
            FOUND_SECRETS=true
        fi
    done
    
    if [ "$FOUND_SECRETS" = false ]; then
        echo -e "${GREEN}✅ Aucun secret suspect trouvé dans le code source.${NC}"
    else
        echo -e "${RED}⚠️ Des secrets potentiels ont été trouvés. Veuillez les vérifier et les déplacer vers des variables d'environnement sécurisées.${NC}"
    fi
    echo ""
}

# Vérifier les permissions de fichiers sensibles
check_file_permissions() {
    echo -e "${YELLOW}Vérification des permissions de fichiers...${NC}"
    
    # Vérifier si le répertoire secrets existe
    if [ -d "./secrets" ]; then
        PERM=$(stat -c "%a" ./secrets)
        if [ "$PERM" != "700" ]; then
            echo -e "${RED}⚠️ Le répertoire ./secrets devrait avoir les permissions 700, actuellement: $PERM${NC}"
            echo "Corriger avec: chmod 700 ./secrets"
        else
            echo -e "${GREEN}✅ Permissions correctes pour ./secrets${NC}"
        fi
        
        # Vérifier les fichiers dans secrets
        find ./secrets -type f -name "*.txt" | while read -r file; do
            PERM=$(stat -c "%a" "$file")
            if [ "$PERM" != "600" ]; then
                echo -e "${RED}⚠️ Le fichier $file devrait avoir les permissions 600, actuellement: $PERM${NC}"
                echo "Corriger avec: chmod 600 $file"
            fi
        done
    else
        echo -e "${YELLOW}ℹ️ Aucun répertoire ./secrets trouvé${NC}"
    fi
    
    # Vérifier .env
    if [ -f ".env" ]; then
        PERM=$(stat -c "%a" .env)
        if [ "$PERM" != "600" ] && [ "$PERM" != "640" ]; then
            echo -e "${RED}⚠️ Le fichier .env devrait avoir les permissions 600 ou 640, actuellement: $PERM${NC}"
            echo "Corriger avec: chmod 600 .env"
        else
            echo -e "${GREEN}✅ Permissions correctes pour .env${NC}"
        fi
    fi
    echo ""
}

# Vérifier si des secrets sont exposés dans .env
check_env_file() {
    echo -e "${YELLOW}Vérification du fichier .env...${NC}"
    
    if [ -f ".env" ]; then
        SENSITIVE_VALUES=$(grep -E "(PASSWORD|TOKEN|SECRET|API_KEY).*=.+" .env | grep -v "=$" || true)
        
        if [ -n "$SENSITIVE_VALUES" ]; then
            echo -e "${RED}⚠️ Valeurs sensibles trouvées dans .env:${NC}"
            echo "$SENSITIVE_VALUES"
            echo -e "${RED}Ces valeurs doivent être stockées dans des secrets Kubernetes ou Docker, pas dans .env.${NC}"
        else
            echo -e "${GREEN}✅ Aucune valeur sensible trouvée dans .env${NC}"
        fi
    else
        echo -e "${YELLOW}ℹ️ Fichier .env non trouvé${NC}"
    fi
    echo ""
}

# Vérifier les fichiers YAML Kubernetes pour les secrets en clair
check_kubernetes_files() {
    echo -e "${YELLOW}Vérification des fichiers Kubernetes...${NC}"
    
    FOUND_SECRETS=false
    YAML_FILES=$(find ./kubernetes -type f -name "*.yaml" -o -name "*.yml")
    
    for file in $YAML_FILES; do
        # Rechercher des secrets en clair (non encodés en base64 ou autres valeurs sensibles)
        SECRETS=$(grep -E "(value|stringData).*:.*(password|token|secret|apikey|api_key)" "$file" || true)
        
        if [ -n "$SECRETS" ]; then
            echo -e "${RED}⚠️ Valeurs sensibles potentielles trouvées dans $file:${NC}"
            echo "$SECRETS"
            FOUND_SECRETS=true
        fi
    done
    
    if [ "$FOUND_SECRETS" = false ]; then
        echo -e "${GREEN}✅ Aucun secret en clair trouvé dans les fichiers Kubernetes${NC}"
    fi
    echo ""
}

# Vérifier si Docker utilise un utilisateur non-root
check_docker_user() {
    echo -e "${YELLOW}Vérification de l'utilisateur Docker...${NC}"
    
    if grep -q "USER " Dockerfile; then
        if grep -q "USER root" Dockerfile; then
            echo -e "${RED}⚠️ Le Dockerfile utilise explicitement l'utilisateur root${NC}"
        else
            echo -e "${GREEN}✅ Le Dockerfile utilise un utilisateur non-root${NC}"
        fi
    else
        echo -e "${RED}⚠️ Le Dockerfile ne spécifie pas d'utilisateur non-root explicitement${NC}"
    fi
    echo ""
}

# Vérifier les pratiques recommandées pour Kubernetes
check_kubernetes_best_practices() {
    echo -e "${YELLOW}Vérification des bonnes pratiques Kubernetes...${NC}"
    
    # Vérifier la présence de limites de ressources
    YAML_FILES=$(find ./kubernetes -type f -name "*.yaml" -o -name "*.yml")
    
    MISSING_LIMITS=false
    for file in $YAML_FILES; do
        if grep -q "kind: Deployment\|kind: StatefulSet\|kind: DaemonSet" "$file"; then
            if ! grep -q "resources:" "$file" || ! grep -q "limits:" "$file"; then
                echo -e "${RED}⚠️ Limites de ressources manquantes dans $file${NC}"
                MISSING_LIMITS=true
            fi
        fi
    done
    
    if [ "$MISSING_LIMITS" = false ]; then
        echo -e "${GREEN}✅ Toutes les charges de travail Kubernetes ont des limites de ressources${NC}"
    fi
    
    # Vérifier la présence de readinessProbe et livenessProbe
    MISSING_PROBES=false
    for file in $YAML_FILES; do
        if grep -q "kind: Deployment\|kind: StatefulSet" "$file"; then
            if ! grep -q "livenessProbe:" "$file" || ! grep -q "readinessProbe:" "$file"; then
                echo -e "${RED}⚠️ Les sondes liveness/readiness sont manquantes dans $file${NC}"
                MISSING_PROBES=true
            fi
        fi
    done
    
    if [ "$MISSING_PROBES" = false ]; then
        echo -e "${GREEN}✅ Toutes les charges de travail Kubernetes ont des sondes liveness/readiness${NC}"
    fi
    echo ""
}

# Exécuter toutes les vérifications
check_secrets
check_file_permissions
check_env_file
check_kubernetes_files
check_docker_user
check_kubernetes_best_practices

echo -e "${GREEN}= = = = = = = = = = = = = = = = = = = = = = =${NC}"
echo -e "${GREEN}=   Vérification de sécurité terminée        =${NC}"
echo -e "${GREEN}= = = = = = = = = = = = = = = = = = = = = = =${NC}"
echo ""
echo "Pour obtenir des recommandations détaillées, consultez docs/SECRETS_MANAGEMENT.md" 