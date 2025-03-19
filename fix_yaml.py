#!/usr/bin/env python3

# Script pour générer un fichier YAML propre

content = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-bot-web
  namespace: evil2root-trading
spec:
  replicas: 2
  selector:
    matchLabels:
      app: trading-bot
      component: web
  template:
    metadata:
      labels:
        app: trading-bot
        component: web
    spec:
      containers:
      - name: web
        image: registry.digitalocean.com/evil2root-registry/evil2root-ai:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 5000
          name: web
        - containerPort: 9090
          name: metrics
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: APP_MODE
          value: "web"
        - name: REDIS_HOST
          value: "redis"
        - name: REDIS_PORT
          value: "6379"
        - name: POSTGRES_HOST
          value: "postgres"
        - name: PORT
          value: "5000"
        - name: METRICS_PORT
          value: "9090"
        - name: LOG_LEVEL
          value: "info"
      imagePullSecrets:
      - name: registry-evil2root-registry
"""

with open("kubernetes/deployments/trading-bot-web.yaml", "w") as f:
    f.write(content)

print("Fichier trading-bot-web.yaml généré avec succès.") 