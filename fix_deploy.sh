#!/bin/bash
set -e

echo "=== Correction et déploiement de l'application EVIL2ROOT Trading ==="

# Vérification des outils requis
for cmd in python3 kubectl; do
    if ! command -v $cmd &> /dev/null; then
        echo "Erreur: $cmd n'est pas installé ou n'est pas dans le PATH"
        exit 1
    fi
done

# Vérification de l'installation du module PyYAML
python3 -c "import yaml" 2>/dev/null || {
    echo "Installation du module PyYAML..."
    pip install pyyaml
}

# Exécuter le script Python pour corriger les fichiers YAML
echo "Correction des fichiers YAML de déploiement..."
python3 fix_yaml.py

# Vérifier si la correction a réussi
if [ $? -ne 0 ]; then
    echo "Échec de la correction des fichiers YAML. Abandon du déploiement."
    exit 1
fi

# Supprimer les déploiements existants s'ils existent
echo "Suppression des déploiements existants..."
kubectl delete deployment trading-bot-web --namespace=evil2root-trading --ignore-not-found=true
kubectl delete deployment analysis-bot --namespace=evil2root-trading --ignore-not-found=true
kubectl delete deployment market-scheduler --namespace=evil2root-trading --ignore-not-found=true

# Attendre que les pods soient complètement supprimés
echo "Attente de la suppression complète des pods..."
kubectl wait --for=delete pod --selector=app=trading-bot-web --namespace=evil2root-trading --timeout=60s || true
kubectl wait --for=delete pod --selector=app=analysis-bot --namespace=evil2root-trading --timeout=60s || true
kubectl wait --for=delete pod --selector=app=market-scheduler --namespace=evil2root-trading --timeout=60s || true

# Déploiement des composants de l'application
echo "Déploiement des composants de l'application..."
kubectl apply -f kubernetes/deployments/trading-bot-web.yaml || {
    echo "Erreur lors du déploiement de trading-bot-web. Vérification du fichier YAML..."
    kubectl apply -f kubernetes/deployments/trading-bot-web.yaml --validate=true --dry-run=client
    exit 1
}

kubectl apply -f kubernetes/deployments/analysis-bot.yaml || {
    echo "Erreur lors du déploiement de analysis-bot. Vérification du fichier YAML..."
    kubectl apply -f kubernetes/deployments/analysis-bot.yaml --validate=true --dry-run=client
    exit 1
}

kubectl apply -f kubernetes/deployments/market-scheduler.yaml || {
    echo "Erreur lors du déploiement de market-scheduler. Vérification du fichier YAML..."
    kubectl apply -f kubernetes/deployments/market-scheduler.yaml --validate=true --dry-run=client
    exit 1
}

# Déploiement des services
echo "Déploiement des services..."
kubectl apply -f kubernetes/services/trading-bot-web.yaml
kubectl apply -f kubernetes/services/web-service.yaml

# Déploiement de l'ingress (avec une pause pour s'assurer que le webhook est prêt)
echo "Attente avant le déploiement de l'ingress..."
sleep 30
kubectl apply -f kubernetes/ingress/trading-bot-web-ingress.yaml || {
    echo "Première tentative de déploiement de l'ingress échouée. Attente supplémentaire..."
    sleep 60
    kubectl apply -f kubernetes/ingress/trading-bot-web-ingress.yaml
}

# Déploiement des autoscalers
echo "Déploiement des autoscalers..."
kubectl apply -f kubernetes/hpa/trading-bot-web-hpa.yaml
kubectl apply -f kubernetes/hpa/analysis-bot-hpa.yaml

echo "Application déployée avec succès !"
echo "Pour vérifier le statut des pods:"
echo "kubectl get pods -n evil2root-trading"
echo "Pour voir les logs de l'analysis-bot:"
echo "kubectl logs -f -l app=trading-bot,component=analysis-bot -n evil2root-trading" 