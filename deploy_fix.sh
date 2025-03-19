#!/bin/bash
set -e

echo "=== Correction et redéploiement de l'application EVIL2ROOT Trading ==="

# Reconstruire l'image Docker avec la correction de tweepy
echo "Reconstruction de l'image Docker..."
docker build -t evil2root-ai:latest .
docker tag evil2root-ai:latest registry.digitalocean.com/evil2root-registry/evil2root-ai:latest
docker tag evil2root-ai:latest registry.digitalocean.com/evil2root-registry/evil2root-ai:$(git rev-parse HEAD)

# Pousser l'image vers le registre
echo "Envoi de l'image vers le registre DigitalOcean..."
docker push registry.digitalocean.com/evil2root-registry/evil2root-ai:latest
docker push registry.digitalocean.com/evil2root-registry/evil2root-ai:$(git rev-parse HEAD)

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
IMAGE_TAG="registry.digitalocean.com/evil2root-registry/evil2root-ai:$(git rev-parse HEAD)"
echo "Déploiement des nouveaux composants avec l'image: $IMAGE_TAG"

# Mise à jour des fichiers de déploiement avec la dernière image
for file in kubernetes/deployments/trading-bot-web.yaml kubernetes/deployments/analysis-bot.yaml kubernetes/deployments/market-scheduler.yaml; do
  # Sauvegarde du fichier original
  cp "$file" "${file}.bak"
  
  # Mise à jour de l'image
  sed -i "s|image:.*|image: $IMAGE_TAG|g" "$file"
  
  # S'assurer que REDIS_PORT est correctement configuré (sans utiliser de substitution de variables)
  # Nous remplaçons simplement la valeur existante par "6379"
  sed -i '/REDIS_PORT/,/value:/s|value:.*|value: "6379"|' "$file"
  
  # Ajouter la variable d'environnement INSTALL_MISSING_DEPS pour analysis-bot
  if [[ "$file" == *"analysis-bot.yaml" ]]; then
    if ! grep -q "INSTALL_MISSING_DEPS" "$file"; then
      sed -i '/LOG_LEVEL/a\        - name: INSTALL_MISSING_DEPS\n          value: "true"' "$file"
    fi
  fi
done

# Déploiement des composants de l'application
kubectl apply -f kubernetes/deployments/trading-bot-web.yaml
kubectl apply -f kubernetes/deployments/analysis-bot.yaml
kubectl apply -f kubernetes/deployments/market-scheduler.yaml

# Déploiement des services
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
kubectl apply -f kubernetes/hpa/trading-bot-web-hpa.yaml
kubectl apply -f kubernetes/hpa/analysis-bot-hpa.yaml

echo "Application déployée avec succès !"
echo "Pour vérifier le statut des pods:"
echo "kubectl get pods -n evil2root-trading"
echo "Pour voir les logs de l'analysis-bot:"
echo "kubectl logs -f -l app=trading-bot,component=analysis-bot -n evil2root-trading" 