# Correctif pour la gestion des connexions Redis

## Problème

L'application affiche l'erreur suivante lors du démarrage :

```
Traceback (most recent call last):
  File "/app/app/trading.py", line 48, in <module>
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
ValueError: invalid literal for int() with base 10: 'tcp://10.245.177.46:6379'
```

Cette erreur se produit car Kubernetes fournit la variable d'environnement `REDIS_PORT` au format `tcp://10.245.177.46:6379` au lieu d'un simple numéro de port (`6379`).

## Solution

Nous avons mis en place deux approches pour résoudre ce problème :

### 1. Correction dans le code source

Le script `fix-redis-connection.sh` modifie le fichier `app/trading.py` pour gérer correctement les URL Redis. Le code modifié extrait le numéro de port de l'URL complète si nécessaire.

```python
# Parse Redis connection details
redis_port_raw = os.environ.get('REDIS_PORT', '6379')

# Handle case when REDIS_PORT is a full URL (tcp://host:port)
if '://' in redis_port_raw:
    import re
    # Extract port from URL using regex
    port_match = re.search(r':(\d+)$', redis_port_raw)
    redis_port = int(port_match.group(1)) if port_match else 6379
else:
    redis_port = int(redis_port_raw)
```

Ce correctif est intégré au Dockerfile pour être appliqué lors de la construction de l'image.

### 2. Modification des déploiements Kubernetes

Le workflow GitHub Actions a été modifié pour ajuster les variables d'environnement dans les déploiements Kubernetes. Cette approche :

1. Conserve l'URL Redis originale dans `REDIS_PORT_ORIGINAL`
2. Extrait uniquement le numéro de port et le met dans `REDIS_PORT`

Cette solution fonctionne également pour les déploiements où l'on ne peut pas modifier le code source.

## Utilisation

Ce correctif est automatiquement appliqué lors du processus de CI/CD. Aucune action manuelle n'est nécessaire.

## Recommandation pour l'avenir

Pour éviter ce problème à l'avenir, nous recommandons de :

1. Concevoir le code pour gérer les URL complètes dans les variables d'environnement
2. Utiliser des variables distinctes pour l'hôte et le port (`REDIS_HOST`, `REDIS_PORT`)
3. Utiliser des bibliothèques qui peuvent analyser les URL de connexion (comme `urllib.parse`) 