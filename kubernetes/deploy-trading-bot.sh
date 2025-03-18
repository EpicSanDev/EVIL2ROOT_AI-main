#!/bin/bash
set -e

# Variables par défaut
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"localhost:5000"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
NAMESPACE=${NAMESPACE:-"default"}
CONFIG_FILE=${CONFIG_FILE:-"trading-bot-deployment.yaml"}
SECRETS_FILE=${SECRETS_FILE:-"../config/secrets.env"}

# Afficher l'aide
show_help() {
    echo "Déploiement du bot de trading sur Kubernetes"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -r, --registry REGISTRY     Spécifie le registre Docker (défaut: $DOCKER_REGISTRY)"
    echo "  -t, --tag TAG               Spécifie le tag de l'image (défaut: $IMAGE_TAG)"
    echo "  -n, --namespace NAMESPACE   Spécifie le namespace Kubernetes (défaut: $NAMESPACE)"
    echo "  -c, --config FILE           Spécifie le fichier de config Kubernetes (défaut: $CONFIG_FILE)"
    echo "  -s, --secrets FILE          Spécifie le fichier de secrets (défaut: $SECRETS_FILE)"
    echo "  -h, --help                  Affiche ce message d'aide"
    echo ""
    echo "Exemple:"
    echo "  $0 --registry my-registry.com --tag v1.0.0 --namespace trading"
}

# Analyser les arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--registry) DOCKER_REGISTRY="$2"; shift ;;
        -t|--tag) IMAGE_TAG="$2"; shift ;;
        -n|--namespace) NAMESPACE="$2"; shift ;;
        -c|--config) CONFIG_FILE="$2"; shift ;;
        -s|--secrets) SECRETS_FILE="$2"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Option inconnue: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Vérifier si kubectl est disponible
if ! command -v kubectl &> /dev/null; then
    echo "kubectl n'est pas disponible. Veuillez l'installer d'abord."
    exit 1
fi

# Vérifier si le namespace existe, sinon le créer
kubectl get namespace ${NAMESPACE} > /dev/null 2>&1 || kubectl create namespace ${NAMESPACE}

# Vérifier si le fichier de secrets existe
if [ -f "$SECRETS_FILE" ]; then
    echo "Création des secrets depuis $SECRETS_FILE..."
    
    # Créer un fichier temporaire pour stocker les secrets
    TEMP_SECRETS=$(mktemp)
    
    # En-tête du fichier YAML
    cat > ${TEMP_SECRETS} << EOF
apiVersion: v1
kind: Secret
metadata:
  name: trading-bot-secrets
  namespace: ${NAMESPACE}
type: Opaque
data:
EOF
    
    # Lire les secrets du fichier et les ajouter encodés en base64
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Ignorer les commentaires et les lignes vides
        [[ $key =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Convertir la clé au format attendu par Kubernetes
        k8s_key=$(echo $key | tr '[:upper:]' '[:lower:]' | tr '_' '-')
        
        # Encoder la valeur en base64 et l'ajouter au fichier
        echo "  $k8s_key: $(echo -n $value | base64)" >> ${TEMP_SECRETS}
    done < "$SECRETS_FILE"
    
    # Appliquer les secrets
    kubectl apply -f ${TEMP_SECRETS} -n ${NAMESPACE}
    
    # Supprimer le fichier temporaire
    rm ${TEMP_SECRETS}
else
    echo "Attention: Fichier de secrets $SECRETS_FILE non trouvé. Les secrets existants seront utilisés."
fi

# Remplacer les variables dans le fichier de déploiement
echo "Préparation du déploiement..."
TEMP_CONFIG=$(mktemp)
sed "s|\${DOCKER_REGISTRY}|${DOCKER_REGISTRY}|g" ${CONFIG_FILE} > ${TEMP_CONFIG}
sed -i "s|\${IMAGE_TAG}|${IMAGE_TAG}|g" ${TEMP_CONFIG}

# Appliquer la configuration
echo "Déploiement du bot de trading dans le namespace ${NAMESPACE}..."
kubectl apply -f ${TEMP_CONFIG} -n ${NAMESPACE}

# Supprimer le fichier temporaire
rm ${TEMP_CONFIG}

echo "Déploiement terminé."
echo "Pour voir les pods en cours d'exécution:"
echo "kubectl get pods -n ${NAMESPACE} -l app=trading-bot"

# Attendre que les pods soient prêts
echo "Attente du démarrage des pods..."
kubectl wait --for=condition=ready pods -l app=trading-bot -n ${NAMESPACE} --timeout=180s

echo "Bot de trading déployé avec succès!"
echo "Pour voir les logs:"
echo "kubectl logs -f -l app=trading-bot -n ${NAMESPACE}" 