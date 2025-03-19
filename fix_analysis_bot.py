#!/usr/bin/env python3

# Script pour générer un fichier YAML propre pour analysis-bot

content = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: analysis-bot
  namespace: evil2root-trading
spec:
  replicas: 2
  selector:
    matchLabels:
      app: trading-bot
      component: analysis-bot
  template:
    metadata:
      labels:
        app: trading-bot
        component: analysis-bot
    spec:
      containers:
      - name: analysis-bot
        image: registry.digitalocean.com/evil2root-registry/evil2root-ai:latest
        imagePullPolicy: Always
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: APP_MODE
          value: "analysis"
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
        - name: POSTGRES_HOST
          value: "postgres"
        - name: LOG_LEVEL
          value: "info"
      imagePullSecrets:
      - name: registry-evil2root-registry
"""

with open("kubernetes/deployments/analysis-bot.yaml", "w") as f:
    f.write(content)

print("Fichier analysis-bot.yaml généré avec succès.") 