# 🚀 EVIL2ROOT Trading Bot - Guide de Démarrage Rapide

## ⚡ Démarrage en 3 étapes

### 1. 🔧 Configuration Initiale (une seule fois)
```bash
# Vérifier que tout est prêt
python3 check_system.py

# Configuration initiale
./start_trading_bot.sh setup
```

### 2. 🔑 Configurer vos Clés API (Optionnel)
```bash
# Éditer le fichier de configuration
nano .env

# Remplacer les placeholders par vos vraies clés :
# OPENROUTER_API_KEY=sk-or-v1-votre-cle-ici
# TELEGRAM_TOKEN=votre-token-telegram
# FINNHUB_API_KEY=votre-cle-finnhub
```

### 3. 🎯 Démarrer le Système
```bash
# Démarrage en mode sécurisé (paper trading)
./start_trading_bot.sh paper

# OU démarrage complet
./start_trading_bot.sh start
```

## 🌐 Accès aux Interfaces

Une fois démarré, votre système est accessible sur :

| 🖥️ Interface | 🔗 URL | 📝 Description |
|-------------|---------|---------------|
| **Frontend React** | http://localhost:3000 | Interface moderne |
| **Web Admin** | http://localhost:5000 | Administration Flask |
| **API** | http://localhost:8000 | API FastAPI + Docs |
| **Database** | http://localhost:5050 | PgAdmin |
| **Proxy** | http://localhost:80 | Nginx |

## 🔐 Identifiants par Défaut

- **Web Interface** : `admin` / `Evil2Root_Admin_2024!`
- **PgAdmin** : `admin@evil2root.local` / `pgadmin2024secure`

## 📊 Commandes Utiles

```bash
# Voir les logs en temps réel
./start_trading_bot.sh logs

# Vérifier le statut des services
./start_trading_bot.sh status

# Arrêter le système
./start_trading_bot.sh stop

# Redémarrer
./start_trading_bot.sh restart

# Aide complète
./start_trading_bot.sh help
```

## 🛡️ Modes de Fonctionnement

| Mode | Commande | Description |
|------|----------|-------------|
| **Paper** | `./start_trading_bot.sh paper` | 🛡️ Mode sécurisé (simulation) |
| **Full** | `./start_trading_bot.sh start` | 🚀 Système complet |
| **Web** | `./start_trading_bot.sh web` | 🌐 Interface web seulement |
| **API** | `./start_trading_bot.sh api` | 🔌 API backend seulement |
| **Analysis** | `./start_trading_bot.sh analysis` | 📈 Bot d'analyse seulement |

## ⚠️ Important : Sécurité

- **Par défaut, le trading réel est DÉSACTIVÉ**
- Testez d'abord en mode `paper` (simulation)
- Configurez vos vraies clés API avant le trading réel
- Ne changez `ENABLE_LIVE_TRADING=true` qu'après validation complète

## 🆘 Dépannage Rapide

### Problème : Services qui ne démarrent pas
```bash
# Vérifier Docker
docker info

# Reconstruire les images
./start_trading_bot.sh build

# Nettoyer et redémarrer
./start_trading_bot.sh clean
./start_trading_bot.sh start
```

### Problème : Erreurs de permissions
```bash
# Réparer les permissions
chmod +x *.sh *.py
sudo chown -R $USER:$USER .
```

### Problème : Ports occupés
```bash
# Voir les ports utilisés
netstat -tulpn | grep :5000
netstat -tulpn | grep :8000

# Arrêter les processus conflictuels
./start_trading_bot.sh stop
```

## 📚 Documentation Complète

- **Guide Complet** : `PROJET_COMPLETE.md`
- **Architecture** : `ARCHITECTURE.md`
- **Fonctionnalités** : `ADVANCED_FEATURES.md`
- **API** : http://localhost:8000/docs (une fois démarré)

## 🎉 C'est Tout !

Votre bot de trading IA est maintenant opérationnel !

**Mode recommandé pour commencer :** `./start_trading_bot.sh paper`

---

💡 **Astuce** : Gardez un terminal ouvert avec `./start_trading_bot.sh logs` pour surveiller l'activité en temps réel.
