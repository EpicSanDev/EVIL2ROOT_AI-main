# Sécurité du Système EVIL2ROOT Trading Bot

Ce document détaille les mesures de sécurité mises en place pour protéger le système EVIL2ROOT Trading Bot, les données qu'il traite et les aspects à considérer lors de son déploiement.

## Table des matières

1. [Vue d'ensemble de la sécurité](#vue-densemble-de-la-sécurité)
2. [Gestion des secrets et des clés API](#gestion-des-secrets-et-des-clés-api)
3. [Sécurité des communications](#sécurité-des-communications)
4. [Sécurité de la base de données](#sécurité-de-la-base-de-données)
5. [Authentification et autorisation](#authentification-et-autorisation)
6. [Contrôles d'accès](#contrôles-daccès)
7. [Journalisation et surveillance](#journalisation-et-surveillance)
8. [Protection des données](#protection-des-données)
9. [Sécurité du déploiement](#sécurité-du-déploiement)
10. [Atténuation des risques](#atténuation-des-risques)
11. [Politique de divulgation des vulnérabilités](#politique-de-divulgation-des-vulnérabilités)

## Vue d'ensemble de la sécurité

Le système EVIL2ROOT Trading Bot manipule des données financières sensibles et peut effectuer des transactions réelles. La sécurité est donc une priorité absolue dans sa conception et son exploitation. Les principes suivants guident notre approche de la sécurité :

- **Défense en profondeur** : Mise en place de multiples couches de sécurité
- **Principe du moindre privilège** : Attribution des droits minimaux nécessaires
- **Sécurité par conception** : Intégration de la sécurité dès la conception
- **Mise à jour régulière** : Maintien des dépendances à jour
- **Surveillance continue** : Détection rapide des incidents de sécurité potentiels
- **Audit et traçabilité** : Journalisation complète des actions sensibles

## Gestion des secrets et des clés API

### Stockage des secrets

Les secrets (clés API, identifiants, etc.) sont gérés comme suit :

1. **Variables d'environnement** : Les secrets sont stockés dans des variables d'environnement, et non en dur dans le code
   ```bash
   # Exemple de configuration dans .env (NE JAMAIS committer ce fichier)
   OPENROUTER_API_KEY=abc123...
   TELEGRAM_TOKEN=12345:abc...
   DB_PASSWORD=secure_password
   ```

2. **Fichier .env** : Un fichier `.env` est utilisé pour le développement local, explicitement exclu du contrôle de version
   ```bash
   # Dans .gitignore
   .env
   ```

3. **Gestionnaire de secrets** pour les déploiements en production :
   - Pour AWS : AWS Secrets Manager
   - Pour GCP : Google Secret Manager
   - Pour les déploiements simples : Sécurisation du fichier `.env` avec des permissions restrictives

### Rotation des clés

Les pratiques de rotation des clés incluent :

- Rotation régulière des clés API (tous les 90 jours)
- Rotation immédiate en cas de suspicion de compromission
- Surveillance des accès aux clés API

### Accès aux API externes

Pour les API externes (OpenRouter, Telegram, etc.) :

- Utilisation de clés API avec privilèges minimaux
- Restriction des adresses IP autorisées lorsque c'est possible
- Surveillance de l'utilisation anormale des API

## Sécurité des communications

### Chiffrement en transit

Toutes les communications sont chiffrées en transit :

- **HTTPS** pour les communications web (TLS 1.3)
- **TLS** pour les connexions à la base de données PostgreSQL
- **SSH** pour les accès aux serveurs
- **Certificats SSL** générés via Let's Encrypt et automatiquement renouvelés

### Configuration NGINX pour HTTPS

```nginx
server {
    listen 443 ssl http2;
    server_name votre-domaine.com;
    
    ssl_certificate /etc/letsencrypt/live/votre-domaine.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/votre-domaine.com/privkey.pem;
    
    # Configurations optimales pour la sécurité
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers 'ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305';
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;
    
    # HSTS (31536000 secondes = 1 an)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Autres en-têtes de sécurité
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Rediriger HTTP vers HTTPS
server {
    listen 80;
    server_name votre-domaine.com;
    
    location / {
        return 301 https://$host$request_uri;
    }
}
```

### Environnement local sécurisé

Même pour les déploiements locaux, les communications sont sécurisées :

- Redis configuré pour n'accepter que les connexions locales
- API Web accessible uniquement sur localhost par défaut
- Variables d'environnement correctement sécurisées

## Sécurité de la base de données

### Configuration de PostgreSQL

1. **Authentification** :
   - Utilisation de mots de passe forts ou d'authentification par certificat
   - Pas d'accès anonyme
   - Rotation périodique des mots de passe

2. **Contrôle d'accès** :
   ```sql
   -- Créer un utilisateur avec privilèges limités
   CREATE USER evil2root_app WITH PASSWORD 'strong_password';
   
   -- Accorder uniquement les privilèges nécessaires
   GRANT CONNECT ON DATABASE evil2root_trading TO evil2root_app;
   GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO evil2root_app;
   GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO evil2root_app;
   
   -- Limiter les connexions à certaines adresses IP
   host    evil2root_trading    evil2root_app    192.168.1.0/24    md5
   ```

3. **Chiffrement** :
   - Connexions SSL/TLS requises
   - Chiffrement des données sensibles au repos (transactions, clés API stockées)

4. **Journalisation** :
   - Journalisation des accès
   - Journalisation des modifications de schéma
   - Journalisation des requêtes lentes

### Sauvegardes sécurisées

- Chiffrement des sauvegardes de base de données
- Stockage des sauvegardes dans un emplacement sécurisé
- Tests réguliers de restauration
- Conservation des sauvegardes selon une politique de rétention (30 jours, 90 jours, 1 an)

### Protection contre les injections SQL

- Utilisation de requêtes paramétrées et d'ORM (SQLAlchemy)
- Validation des entrées
- Principe du moindre privilège pour les utilisateurs de la base de données
- Séparation des rôles entre lecture et écriture

Exemple de code sécurisé utilisant des requêtes paramétrées :
```python
# Bon exemple avec requêtes paramétrées
def get_trades_by_symbol(symbol):
    query = "SELECT * FROM trade_history WHERE symbol = %s"
    cursor.execute(query, (symbol,))
    return cursor.fetchall()

# À ÉVITER : Exemple vulnérable aux injections SQL
def get_trades_by_symbol_insecure(symbol):
    query = f"SELECT * FROM trade_history WHERE symbol = '{symbol}'"  # Vulnérable !
    cursor.execute(query)
    return cursor.fetchall()
```

## Authentification et autorisation

### Système d'authentification

L'authentification utilise les mécanismes suivants :

1. **Interface Web** :
   - Authentification par JWT (JSON Web Tokens)
   - Support 2FA optionnel (TOTP)
   - Verrouillage de compte après plusieurs tentatives échouées
   - Sessions avec expiration automatique (15 minutes d'inactivité)

2. **API REST** :
   - Authentification par token JWT
   - Règles strictes pour la génération et la validation des tokens
   - Expiration courte des tokens (1 heure)
   - Refresh tokens avec révocation possible

3. **Telegram** :
   - Authentification par code secret
   - Liste blanche d'utilisateurs Telegram autorisés

Code d'implémentation du middleware d'authentification JWT :
```python
@app.middleware("http")
async def jwt_authentication(request, call_next):
    # Exceptions pour les routes publiques
    if request.url.path in ["/api/v1/auth/token", "/health"]:
        return await call_next(request)
        
    # Vérifier le token JWT
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=401,
            content={"error": "Token d'authentification manquant ou invalide"}
        )
        
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(
            token, 
            config.JWT_SECRET_KEY, 
            algorithms=[config.JWT_ALGORITHM]
        )
        # Ajouter les informations d'utilisateur à la requête
        request.state.user = payload
    except jwt.PyJWTError:
        return JSONResponse(
            status_code=401,
            content={"error": "Token d'authentification invalide ou expiré"}
        )
        
    # Continuer avec la requête
    return await call_next(request)
```

### Système d'autorisation

Le système d'autorisation est basé sur les rôles avec les principes suivants :

1. **Rôles définis** :
   - `admin` : Accès complet
   - `trader` : Exécution de transactions, modifications des paramètres
   - `analyst` : Lecture seule et exécution d'analyses
   - `viewer` : Lecture seule
   - `system` : Utilisé pour les intégrations programmatiques

2. **Contrôle d'accès basé sur les rôles (RBAC)** :
   ```python
   def require_roles(roles):
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               current_user = get_current_user()
               if not current_user:
                   abort(401, description="Authentification requise")
               
               if current_user.role not in roles:
                   abort(403, description="Accès interdit")
                   
               return func(*args, **kwargs)
           return wrapper
       return decorator
   
   # Utilisation
   @app.route('/api/v1/config', methods=['PUT'])
   @require_roles(['admin', 'trader'])
   def update_config():
       # Code pour mettre à jour la configuration
   ```

3. **Vérification granulaire des permissions** :
   - Vérifications au niveau des objets (ex: un utilisateur ne peut modifier que ses propres transactions)
   - Autorisations personnalisables par utilisateur

## Contrôles d'accès

### Sécurité réseau

1. **Pare-feu** :
   - Pare-feu au niveau du système (UFW, iptables)
   - Pare-feu au niveau du cloud (Groupes de sécurité AWS, règles de pare-feu GCP)
   - Ouverture minimale des ports (SSH, HTTP, HTTPS uniquement)

2. **Configuration UFW** :
   ```bash
   # Activer UFW
   sudo ufw enable
   
   # Règles de base
   sudo ufw default deny incoming
   sudo ufw default allow outgoing
   
   # Autoriser SSH, HTTP, HTTPS
   sudo ufw allow ssh
   sudo ufw allow http
   sudo ufw allow https
   
   # Pour le développement seulement
   sudo ufw allow 5000/tcp
   
   # Limiter les tentatives de connexion SSH
   sudo ufw limit ssh
   ```

3. **Segmentation réseau avec Docker** :
   ```yaml
   # docker-compose.yml
   networks:
     frontend:
       # Réseau pour les services exposés
     backend:
       # Réseau interne pour les services de base de données
     redis_network:
       # Réseau dédié pour les communications Redis
   
   services:
     web:
       networks:
         - frontend
         - redis_network
     
     postgres:
       networks:
         - backend
       # Pas d'exposition directe à l'extérieur
   ```

### Isolation et conteneurisation

1. **Sécurité des conteneurs Docker** :
   - Images de base minimales (Alpine Linux)
   - Exécution en tant qu'utilisateur non privilégié
   - Lecture seule du système de fichiers lorsque possible
   - Scan des vulnérabilités des images

2. **Configuration Docker sécurisée** :
   ```dockerfile
   # Dockerfile
   FROM python:3.9-alpine
   
   # Créer un utilisateur non privilégié
   RUN addgroup -S appgroup && adduser -S appuser -G appgroup
   
   # Copier les fichiers d'application
   WORKDIR /app
   COPY --chown=appuser:appgroup . .
   
   # Installer les dépendances
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Utiliser l'utilisateur non privilégié
   USER appuser
   
   # Montages en lecture seule
   VOLUME ["/app/config:ro"]
   
   # Exécuter l'application
   CMD ["python", "run.py"]
   ```

## Journalisation et surveillance

### Journalisation de sécurité

1. **Éléments journalisés** :
   - Connexions réussies et échouées
   - Modifications de configuration
   - Transactions exécutées
   - Accès aux API sensibles
   - Erreurs système

2. **Format des journaux** :
   ```json
   {
     "timestamp": "2023-04-15T10:23:45.123Z",
     "level": "INFO",
     "event": "LOGIN_SUCCESS",
     "user_id": "user123",
     "ip_address": "192.168.1.100",
     "user_agent": "Mozilla/5.0...",
     "details": {
       "method": "password"
     }
   }
   ```

3. **Stockage sécurisé des journaux** :
   - Rotation des journaux
   - Archivage chiffré
   - Conservation conforme aux exigences réglementaires

### Surveillance de sécurité

1. **Détection d'anomalies** :
   - Surveillance des connexions depuis des emplacements inhabituels
   - Détection des comportements anormaux (transactions multiples rapides)
   - Alertes en cas d'activité suspecte

2. **Alertes de sécurité** :
   - Notification immédiate pour les événements critiques
   - Escalade automatique pour les incidents graves
   - Corrélation d'événements pour réduire les faux positifs

3. **Audits périodiques** :
   - Revue régulière des journaux
   - Vérification des autorisations
   - Tests d'intrusion

## Protection des données

### Classification des données

Les données sont classifiées selon leur sensibilité :

1. **Publiques** : Informations générales, documentation
2. **Internes** : Configurations non sensibles, métriques agrégées
3. **Confidentielles** : Transactions, stratégies de trading
4. **Hautement sensibles** : Clés API, identifiants, mots de passe

### Chiffrement des données sensibles

1. **Données en transit** : TLS/SSL pour toutes les communications
2. **Données au repos** :
   - Chiffrement transparent de la base de données
   - Chiffrement au niveau applicatif pour les données les plus sensibles

3. **Implémentation du chiffrement au niveau applicatif** :
   ```python
   from cryptography.fernet import Fernet
   
   # Générer une clé (à stocker de manière sécurisée)
   def generate_key():
       return Fernet.generate_key()
   
   # Chiffrer des données
   def encrypt_data(data, key):
       f = Fernet(key)
       return f.encrypt(data.encode())
   
   # Déchiffrer des données
   def decrypt_data(encrypted_data, key):
       f = Fernet(key)
       return f.decrypt(encrypted_data).decode()
   ```

### Anonymisation et minimisation

1. **Minimisation des données** :
   - Collecte uniquement des données nécessaires
   - Suppression des données qui ne sont plus nécessaires

2. **Anonymisation pour l'analyse** :
   - Suppression des identifiants directs
   - Agrégation des données sensibles
   - Séparation des données d'identification et des données de transaction

### Conformité réglementaire

1. **RGPD** (si applicable) :
   - Consentement explicite
   - Droit à l'effacement
   - Portabilité des données

2. **Conformité financière** :
   - Conservation des enregistrements de transactions
   - Piste d'audit complète
   - Rapports réglementaires si nécessaire

## Sécurité du déploiement

### Pipeline CI/CD sécurisé

1. **Vérifications automatisées** :
   - Analyse statique du code (SonarQube, Bandit)
   - Analyse des dépendances (OWASP Dependency Check)
   - Tests de sécurité automatisés

2. **Gestion des secrets dans CI/CD** :
   - Utilisation de coffres-forts pour les secrets (HashiCorp Vault, AWS Secrets Manager)
   - Variables d'environnement protégées
   - Pas de secrets en clair dans les logs de CI/CD

3. **Approbation des déploiements** :
   - Processus de revue pour les déploiements de production
   - Principe des quatre yeux pour les changements critiques

### Hardening des serveurs

1. **Configuration de base** :
   - Désactivation des services inutiles
   - Mise à jour régulière du système
   - Utilisateurs avec privilèges minimaux

2. **SSH sécurisé** :
   ```bash
   # /etc/ssh/sshd_config
   PermitRootLogin no
   PasswordAuthentication no
   PubkeyAuthentication yes
   AllowUsers deploy_user
   Protocol 2
   ```

3. **Protection contre les attaques** :
   - Protection contre les attaques par force brute (fail2ban)
   - Protection contre les attaques DDoS (rate limiting)
   - Surveillance des activités suspectes

## Atténuation des risques

### Plan de réponse aux incidents

1. **Procédure en cas de compromission** :
   - Isolation du système compromis
   - Révocation immédiate des clés et tokens
   - Investigation de l'étendue de la compromission
   - Restauration à partir de sauvegardes propres

2. **Communication en cas d'incident** :
   - Notification aux utilisateurs affectés
   - Canaux de communication d'urgence
   - Transparence concernant l'incident

3. **Procédure de récupération** :
   - Restauration des systèmes à partir de sauvegardes
   - Vérification de l'intégrité des données
   - Renforcement des mesures de sécurité

### Tests de sécurité

1. **Tests d'intrusion** :
   - Tests réguliers par une équipe externe
   - Tests de pénétration sur l'infrastructure
   - Tests sur l'interface web et l'API

2. **Tests internes** :
   - Revue de code axée sur la sécurité
   - Tests de fuzzing
   - Simulation d'attaques internes

3. **Scans de vulnérabilités** :
   - Scans réguliers de l'infrastructure
   - Vérification des configurations
   - Analyses de dépendances

## Politique de divulgation des vulnérabilités

### Signalement des vulnérabilités

1. **Contact** :
   - Email dédié : security@evil2root.com
   - Programme de divulgation coordonnée
   - Protection des informateurs

2. **Récompenses et reconnaissance** :
   - Remerciements publics aux découvreurs de vulnérabilités
   - Programme de primes aux bugs (selon les ressources disponibles)

### Processus de correction

1. **Délais** :
   - Accusé de réception sous 48 heures
   - Évaluation de la vulnérabilité sous 7 jours
   - Correctifs selon la gravité (critique : 24h, haute : 7j, moyenne : 30j)

2. **Communication** :
   - Information aux utilisateurs sur les vulnérabilités corrigées
   - Notes de version détaillant les correctifs de sécurité
   - Transparence sur les processus de remédiation

### Divulgation responsable

Les directives pour une divulgation responsable incluent :

1. Signaler les vulnérabilités directement à l'équipe de sécurité
2. Fournir suffisamment de détails pour reproduire le problème
3. Donner un délai raisonnable pour corriger avant toute divulgation publique
4. Ne pas exploiter la vulnérabilité au-delà de la preuve de concept
5. Ne pas accéder ou modifier les données d'autres utilisateurs 