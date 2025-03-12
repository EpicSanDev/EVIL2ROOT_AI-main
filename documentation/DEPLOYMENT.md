# Guide de Déploiement de EVIL2ROOT Trading Bot

Ce document détaille les différentes méthodes de déploiement du système EVIL2ROOT Trading Bot, incluant les prérequis, les étapes d'installation et les configurations post-déploiement.

## Table des matières

1. [Prérequis système](#prérequis-système)
2. [Déploiement avec Docker](#déploiement-avec-docker)
3. [Installation manuelle](#installation-manuelle)
4. [Déploiement sur le cloud](#déploiement-sur-le-cloud)
5. [Configuration post-déploiement](#configuration-post-déploiement)
6. [Mise à jour du système](#mise-à-jour-du-système)
7. [Surveillance et maintenance](#surveillance-et-maintenance)
8. [Résolution des problèmes](#résolution-des-problèmes)

## Prérequis système

### Configuration matérielle recommandée

| Composant | Minimum | Recommandé | Haute performance |
|-----------|---------|------------|-------------------|
| CPU | 4 cœurs | 8 cœurs | 16+ cœurs |
| RAM | 8 GB | 16 GB | 32+ GB |
| Stockage | SSD 100 GB | SSD 250 GB | SSD 500+ GB |
| GPU | Non requis | NVIDIA GTX 1060+ | NVIDIA RTX 2070+ |
| Réseau | 10 Mbps | 100 Mbps | 1 Gbps |

### Logiciels requis

- **Docker et Docker Compose** (pour déploiement conteneurisé)
- **Python 3.8+** (pour installation manuelle)
- **PostgreSQL 12+**
- **Redis 6+**
- **NGINX** (pour production)
- **CUDA Toolkit 11.0+** (si GPU utilisé)

### API et clés externes

- **Compte OpenRouter** avec crédit pour l'API Claude 3.7
- **Bot Telegram** créé via BotFather pour les notifications
- **Compte yfinance** ou autres API financières (selon la configuration)

## Déploiement avec Docker

Le déploiement avec Docker est la méthode recommandée pour la plupart des utilisateurs, offrant isolation, facilité de configuration et portabilité.

### Installation de Docker

#### Sur Linux (Ubuntu/Debian)

```bash
# Mise à jour du système
sudo apt update && sudo apt upgrade -y

# Installation de Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Installation de Docker Compose
sudo apt install docker-compose-plugin

# Ajouter l'utilisateur au groupe docker
sudo usermod -aG docker $USER
```

#### Sur macOS

1. Télécharger et installer [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Ouvrir l'application et autoriser les privilèges demandés

#### Sur Windows

1. Activer WSL2 (Windows Subsystem for Linux 2)
2. Télécharger et installer [Docker Desktop](https://www.docker.com/products/docker-desktop/)
3. Configurer Docker pour utiliser WSL2

### Déploiement du système

1. **Cloner le dépôt**

```bash
git clone https://github.com/Evil2Root/EVIL2ROOT_AI.git
cd EVIL2ROOT_AI
```

2. **Configurer les variables d'environnement**

```bash
cp .env.example .env
```

Ouvrir le fichier `.env` et configurer les paramètres selon vos besoins, notamment :
- `OPENROUTER_API_KEY` : Votre clé API OpenRouter
- `TELEGRAM_TOKEN` : Token de votre bot Telegram
- `ENABLE_LIVE_TRADING` : Définir à `false` pour commencer en mode simulation
- `SYMBOLS` : Liste des symboles à trader
- `DATABASE_URL` : URL de connexion à la base de données PostgreSQL
- `REDIS_URL` : URL de connexion à Redis

3. **Définir les permissions des scripts**

```bash
chmod +x docker-entrypoint.sh
chmod +x start_docker.sh
chmod +x stop_docker.sh
```

4. **Construire et démarrer les conteneurs**

```bash
# Démarrage standard
./start_docker.sh

# OU avec entraînement forcé des modèles
./start_docker_force_train.sh

# OU avec analyse de marché planifiée
./start_market_scheduler.sh
```

5. **Vérifier le déploiement**

```bash
docker compose ps
```

Tous les services devraient être en état "Up". Vous pouvez accéder à l'interface web à http://localhost:5000.

### Configuration avec GPU

Pour utiliser un GPU pour l'entraînement des modèles :

1. **Installer NVIDIA Docker**

```bash
./install_nvidia_docker.sh
```

2. **Modifier docker-compose.yml**

```yaml
services:
  trading-bot:
    # ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

3. **Démarrer avec support GPU**

```bash
./start_rtx_train.sh
```

## Installation manuelle

L'installation manuelle est recommandée uniquement pour le développement ou les déploiements hautement personnalisés.

### Prérequis

- Python 3.8+
- Virtualenv
- PostgreSQL 12+
- Redis 6+

### Étapes d'installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/Evil2Root/EVIL2ROOT_AI.git
cd EVIL2ROOT_AI
```

2. **Créer et activer un environnement virtuel**

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Configurer la base de données PostgreSQL**

```bash
sudo -u postgres psql
CREATE DATABASE evil2root_trading;
CREATE USER evil2root WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE evil2root_trading TO evil2root;
\q
```

5. **Configurer Redis**

Assurez-vous que Redis est installé et en cours d'exécution :

```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

6. **Configurer les variables d'environnement**

```bash
cp .env.example .env
```

Modifier le fichier `.env` avec les paramètres appropriés.

7. **Initialiser la base de données**

```bash
python -c "from app import init_db; init_db()"
```

8. **Démarrer les composants**

Pour un déploiement complet, vous devrez démarrer plusieurs processus :

```bash
# Terminal 1 : Interface web
python run.py

# Terminal 2 : Bot de trading
python -c "from app.trading import TradingBot; bot = TradingBot(); bot.run()"

# Terminal 3 : Validateur IA
python -c "from app.ai_trade_validator import AITradeValidator; validator = AITradeValidator(); validator.run()"

# Terminal 4 : Analyses planifiées
python -c "from app.market_analysis_scheduler import MarketAnalysisScheduler; scheduler = MarketAnalysisScheduler(); scheduler.start()"
```

## Déploiement sur le cloud

### AWS (Amazon Web Services)

1. **Créer une instance EC2**
   - Type recommandé : t3.large pour débutant ou c5.2xlarge pour production
   - Système d'exploitation : Ubuntu Server 20.04 LTS
   - Stockage : 100 GB SSD minimum
   - Groupe de sécurité : Ouvrir les ports 22 (SSH), 80/443 (HTTP/HTTPS), 5000 (interface web)

2. **Se connecter à l'instance**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Installer Docker**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   logout
   # Reconnectez-vous pour appliquer les changements
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

4. **Déployer avec Docker**
   - Suivre les étapes du déploiement Docker ci-dessus

5. **Configurer un nom de domaine et HTTPS**
   - Associer un nom de domaine à votre instance
   - Installer Certbot pour un certificat SSL gratuit :
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d votre-domaine.com
   ```

### Digital Ocean

Le script `deploy_to_digitalocean.sh` permet un déploiement automatisé sur Digital Ocean.

1. **Créer un token d'API Digital Ocean**
   - Aller sur https://cloud.digitalocean.com/account/api/tokens
   - Générer un nouveau token avec accès en écriture

2. **Configurer et exécuter le script**
   ```bash
   export DO_API_TOKEN="your_digital_ocean_token"
   ./deploy_to_digitalocean.sh
   ```

Le script crée automatiquement un droplet, installe toutes les dépendances et démarre le système.

### Google Cloud Platform

1. **Créer une VM Compute Engine**
   - Type de machine : e2-standard-4 (4 vCPU, 16 GB RAM)
   - Système d'exploitation : Ubuntu 20.04 LTS
   - Ajouter GPU si nécessaire (ex: NVIDIA T4)
   - Ouvrir les ports HTTP/HTTPS et créer une règle pour le port 5000

2. **Déployer avec le script**
   ```bash
   # Se connecter à la VM
   gcloud compute ssh instance-name
   
   # Cloner le repo et déployer
   git clone https://github.com/Evil2Root/EVIL2ROOT_AI.git
   cd EVIL2ROOT_AI
   ./deploy.sh
   ```

## Configuration post-déploiement

### Sécurisation du système

1. **Configurer un pare-feu**
   ```bash
   sudo ufw allow ssh
   sudo ufw allow http
   sudo ufw allow https
   sudo ufw allow 5000  # Interface web
   sudo ufw enable
   ```

2. **Mettre en place un proxy inverse avec NGINX**
   ```bash
   sudo apt install nginx
   ```

   Créer un fichier de configuration :
   ```bash
   sudo nano /etc/nginx/sites-available/evil2root
   ```

   Ajouter la configuration suivante :
   ```nginx
   server {
       listen 80;
       server_name votre-domaine.com;
       
       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

   Activer le site :
   ```bash
   sudo ln -s /etc/nginx/sites-available/evil2root /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

3. **Sécuriser avec SSL**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d votre-domaine.com
   ```

### Configuration des sauvegardes

1. **Sauvegarde de la base de données**
   
   Créer un script de sauvegarde :
   ```bash
   mkdir -p /backups
   
   cat > /usr/local/bin/backup_db.sh << 'EOL'
   #!/bin/bash
   DATE=$(date +%Y%m%d_%H%M%S)
   docker exec evil2root-postgres pg_dump -U postgres -d evil2root_trading > /backups/db_backup_$DATE.sql
   find /backups -name "db_backup_*" -type f -mtime +30 -delete
   EOL
   
   chmod +x /usr/local/bin/backup_db.sh
   ```

   Ajouter au cron pour des sauvegardes quotidiennes :
   ```bash
   (crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup_db.sh") | crontab -
   ```

2. **Sauvegarde des modèles et configurations**
   ```bash
   mkdir -p /backups/models
   
   cat > /usr/local/bin/backup_models.sh << 'EOL'
   #!/bin/bash
   DATE=$(date +%Y%m%d)
   tar -czf /backups/models/models_$DATE.tar.gz /path/to/EVIL2ROOT_AI/saved_models
   find /backups/models -name "models_*" -type f -mtime +90 -delete
   EOL
   
   chmod +x /usr/local/bin/backup_models.sh
   
   (crontab -l 2>/dev/null; echo "0 3 * * 0 /usr/local/bin/backup_models.sh") | crontab -
   ```

## Mise à jour du système

### Mise à jour via Docker

1. **Arrêter les services**
   ```bash
   ./stop_docker.sh
   ```

2. **Mettre à jour le code source**
   ```bash
   git pull origin main
   ```

3. **Reconstruire et redémarrer les services**
   ```bash
   docker compose build
   ./start_docker.sh
   ```

### Mise à jour manuelle

1. **Arrêter les services en cours d'exécution**
2. **Mettre à jour le code source**
   ```bash
   git pull origin main
   ```
3. **Mettre à jour les dépendances**
   ```bash
   pip install -r requirements.txt
   ```
4. **Appliquer les migrations de base de données si nécessaire**
   ```bash
   python -c "from app import migrate_db; migrate_db()"
   ```
5. **Redémarrer les services**

## Surveillance et maintenance

### Surveillance du système

1. **Utilisation des ressources**
   ```bash
   # Utilisation CPU/RAM
   ./monitor_docker_memory.sh
   
   # Utilisation GPU
   ./monitor_gpu.sh
   ```

2. **Journaux Docker**
   ```bash
   # Tous les services
   docker compose logs -f
   
   # Service spécifique
   docker compose logs -f trading-bot
   ```

3. **Métriques de base de données**
   ```bash
   docker exec -it evil2root-postgres psql -U postgres -d evil2root_trading -c "SELECT count(*) FROM trade_history;"
   ```

### Maintenance régulière

1. **Nettoyage des journaux**
   ```bash
   find /var/log -name "*.log" -type f -size +100M -exec truncate -s 0 {} \;
   ```

2. **Optimisation de la base de données**
   ```bash
   docker exec -it evil2root-postgres psql -U postgres -d evil2root_trading -c "VACUUM ANALYZE;"
   ```

3. **Vérification des sauvegardes**
   ```bash
   # Tester la restauration d'une sauvegarde
   docker exec -it evil2root-postgres psql -U postgres -d evil2root_trading_test -c "DROP DATABASE IF EXISTS evil2root_trading_test;"
   docker exec -it evil2root-postgres psql -U postgres -c "CREATE DATABASE evil2root_trading_test;"
   cat /backups/db_backup_latest.sql | docker exec -i evil2root-postgres psql -U postgres -d evil2root_trading_test
   ```

4. **Mise à jour du système d'exploitation**
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

## Résolution des problèmes

### Problèmes courants

#### Conteneurs Docker qui ne démarrent pas

1. **Vérifier les journaux**
   ```bash
   docker compose logs trading-bot
   ```

2. **Vérifier la configuration**
   ```bash
   docker compose config
   ```

3. **Reconstruire les images**
   ```bash
   docker compose build --no-cache
   ```

#### Problèmes de connectivité à la base de données

1. **Vérifier l'état du conteneur PostgreSQL**
   ```bash
   docker ps | grep postgres
   ```

2. **Vérifier les journaux PostgreSQL**
   ```bash
   docker compose logs postgres
   ```

3. **Vérifier la connectivité**
   ```bash
   docker exec -it evil2root-postgres pg_isready
   ```

4. **Recréer la base de données si nécessaire**
   ```bash
   docker exec -it evil2root-postgres psql -U postgres -c "CREATE DATABASE evil2root_trading;"
   ```

#### Problèmes avec l'API Claude

1. **Vérifier la clé API**
   ```bash
   grep OPENROUTER_API_KEY .env
   ```

2. **Tester l'API manuellement**
   ```bash
   curl -X POST https://openrouter.ai/api/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     -d '{
       "model": "anthropic/claude-3.7-sonnet",
       "messages": [{"role": "user", "content": "Hello, Claude!"}]
     }'
   ```

### Script de diagnostic

Un script de diagnostic est disponible pour identifier les problèmes courants :

```bash
./check_code.py --diagnose
```

Ce script vérifie :
- La validité des fichiers de configuration
- La connectivité à la base de données et à Redis
- L'accès aux API externes
- Les chemins d'accès aux modèles et aux fichiers de données
- La validité du code Python
- Les dépendances manquantes

### Support et aide

En cas de problèmes persistants :

1. Consulter la documentation dans le dossier `documentation/`
2. Vérifier les issues connues sur GitHub
3. Ouvrir une nouvelle issue avec les détails du problème
4. Contacter l'équipe de support via Telegram ou email 