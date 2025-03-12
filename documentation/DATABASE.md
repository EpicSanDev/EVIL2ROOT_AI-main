# Base de Données EVIL2ROOT Trading Bot

Ce document détaille la structure de la base de données PostgreSQL utilisée par le système EVIL2ROOT Trading Bot, ses tables, relations et optimisations.

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Schéma de la base de données](#schéma-de-la-base-de-données)
3. [Tables principales](#tables-principales)
4. [Relations entre tables](#relations-entre-tables)
5. [Optimisations](#optimisations)
6. [Gestion des migrations](#gestion-des-migrations)
7. [Sauvegarde et restauration](#sauvegarde-et-restauration)
8. [Monitoring et maintenance](#monitoring-et-maintenance)

## Vue d'ensemble

Le système EVIL2ROOT Trading Bot utilise PostgreSQL comme système de gestion de base de données relationnelle pour stocker toutes les données persistantes, notamment :

- Données de marché historiques
- Signaux de trading générés par les modèles
- Transactions exécutées
- Positions ouvertes
- Configurations du système
- Résultats d'analyses
- Performances des modèles
- Métriques du système

La base de données est conçue pour être performante avec de grands volumes de données temporelles, avec des optimisations spécifiques pour les requêtes fréquentes et les séries temporelles.

## Schéma de la base de données

Le diagramme suivant présente une vue simplifiée des principales tables et leurs relations :

```
+----------------+       +----------------+       +----------------+
| trade_history  |       | open_positions |       | trading_signals|
+----------------+       +----------------+       +----------------+
| id             |       | id             |       | id             |
| symbol         |<----->| symbol         |<----->| symbol         |
| action         |       | entry_price    |       | signal_type    |
| entry_price    |       | entry_time     |       | signal_source  |
| exit_price     |       | quantity       |       | direction      |
| entry_time     |       | stop_loss      |       | strength       |
| exit_time      |       | take_profit    |       | timestamp      |
| quantity       |       | current_price  |       | parameters     |
| pnl            |       | unrealized_pnl |       | confidence     |
| pnl_percent    |       | last_update    |       +----------------+
| stop_loss      |       +----------------+
| take_profit    |
| validator_score|
| status         |
+-------^--------+
        |
        |
+-------v--------+       +----------------+       +----------------+
| market_data    |       | bot_settings   |       | analysis_results|
+----------------+       +----------------+       +----------------+
| id             |       | id             |       | id             |
| symbol         |       | parameter_name |       | symbol         |
| timestamp      |       | parameter_value|       | analysis_type  |
| open           |       | updated_at     |       | timeframe      |
| high           |       | description    |       | result_summary |
| low            |       +----------------+       | result_detail  |
| close          |                                | opportunities  |
| volume         |                                | timestamp      |
| indicators     |                                | score          |
+----------------+                                +----------------+

+----------------+       +----------------+       +----------------+
|model_performance|      |performance_    |       | system_logs    |
+----------------+       |metrics         |       +----------------+
| id             |       +----------------+       | id             |
| model_name     |       | id             |       | timestamp      |
| symbol         |       | date           |       | component      |
| parameters     |       | total_trades   |       | level          |
| metrics        |       | win_rate       |       | message        |
| timestamp      |       | profit_factor  |       | details        |
| version        |       | sharpe_ratio   |       +----------------+
+----------------+       | max_drawdown   |
                         | net_profit     |
                         | symbol         |
                         +----------------+
```

## Tables principales

### trade_history

Stocke l'historique complet des transactions exécutées par le système.

| Colonne          | Type          | Description                                   |
|------------------|---------------|-----------------------------------------------|
| id               | SERIAL        | Identifiant unique de la transaction          |
| symbol           | VARCHAR(20)   | Symbole/instrument tradé                      |
| action           | VARCHAR(10)   | BUY ou SELL                                   |
| entry_price      | DECIMAL(18,8) | Prix d'entrée                                 |
| exit_price       | DECIMAL(18,8) | Prix de sortie (NULL si position ouverte)     |
| entry_time       | TIMESTAMP     | Date et heure d'entrée                        |
| exit_time        | TIMESTAMP     | Date et heure de sortie (NULL si ouverte)     |
| quantity         | DECIMAL(18,8) | Quantité/volume de la transaction             |
| pnl              | DECIMAL(18,8) | Profit ou perte (en devise)                   |
| pnl_percent      | DECIMAL(8,4)  | Profit ou perte (en pourcentage)              |
| stop_loss        | DECIMAL(18,8) | Niveau de stop-loss initial                   |
| take_profit      | DECIMAL(18,8) | Niveau de take-profit initial                 |
| validator_score  | DECIMAL(5,4)  | Score de confiance du validateur IA           |
| status           | VARCHAR(20)   | OPEN, CLOSED, CANCELLED                       |

```sql
CREATE TABLE trade_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL CHECK (action IN ('BUY', 'SELL')),
    entry_price DECIMAL(18,8) NOT NULL,
    exit_price DECIMAL(18,8),
    entry_time TIMESTAMP NOT NULL,
    exit_time TIMESTAMP,
    quantity DECIMAL(18,8) NOT NULL,
    pnl DECIMAL(18,8),
    pnl_percent DECIMAL(8,4),
    stop_loss DECIMAL(18,8),
    take_profit DECIMAL(18,8),
    validator_score DECIMAL(5,4),
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN',
    CONSTRAINT valid_status CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED'))
);

CREATE INDEX idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX idx_trade_history_entry_time ON trade_history(entry_time);
CREATE INDEX idx_trade_history_status ON trade_history(status);
```

### open_positions

Stocke les positions actuellement ouvertes par le système.

| Colonne         | Type          | Description                               |
|-----------------|---------------|-------------------------------------------|
| id              | SERIAL        | Identifiant unique de la position         |
| symbol          | VARCHAR(20)   | Symbole/instrument                        |
| entry_price     | DECIMAL(18,8) | Prix d'entrée moyen                       |
| entry_time      | TIMESTAMP     | Date et heure d'entrée                    |
| quantity        | DECIMAL(18,8) | Quantité/volume total                     |
| stop_loss       | DECIMAL(18,8) | Niveau de stop-loss actuel                |
| take_profit     | DECIMAL(18,8) | Niveau de take-profit actuel              |
| current_price   | DECIMAL(18,8) | Prix actuel du marché                     |
| unrealized_pnl  | DECIMAL(18,8) | Profit ou perte non réalisé (en devise)   |
| last_update     | TIMESTAMP     | Dernière mise à jour des données          |

### trading_signals

Stocke tous les signaux générés par les différents modèles d'analyse.

| Colonne        | Type          | Description                               |
|----------------|---------------|-------------------------------------------|
| id             | SERIAL        | Identifiant unique du signal              |
| symbol         | VARCHAR(20)   | Symbole/instrument                        |
| signal_type    | VARCHAR(50)   | Type de signal (technique, ML, RL, etc.)  |
| signal_source  | VARCHAR(50)   | Source du signal (nom du modèle)          |
| direction      | VARCHAR(10)   | BUY, SELL ou NEUTRAL                      |
| strength       | DECIMAL(5,4)  | Force du signal (0-1)                     |
| timestamp      | TIMESTAMP     | Date et heure de génération du signal     |
| parameters     | JSONB         | Paramètres additionnels spécifiques       |
| confidence     | DECIMAL(5,4)  | Score de confiance (0-1)                  |

### market_data

Stocke les données de marché historiques pour chaque symbole.

| Colonne       | Type          | Description                             |
|---------------|---------------|-----------------------------------------|
| id            | SERIAL        | Identifiant unique                      |
| symbol        | VARCHAR(20)   | Symbole/instrument                      |
| timestamp     | TIMESTAMP     | Date et heure de la bougie              |
| open          | DECIMAL(18,8) | Prix d'ouverture                        |
| high          | DECIMAL(18,8) | Prix le plus haut                       |
| low           | DECIMAL(18,8) | Prix le plus bas                        |
| close         | DECIMAL(18,8) | Prix de clôture                         |
| volume        | DECIMAL(18,8) | Volume                                  |
| indicators    | JSONB         | Indicateurs techniques précalculés      |

### performance_metrics

Stocke les métriques de performance quotidiennes du système.

| Colonne        | Type          | Description                            |
|----------------|---------------|----------------------------------------|
| id             | SERIAL        | Identifiant unique                     |
| date           | DATE          | Date des métriques                     |
| total_trades   | INTEGER       | Nombre total de transactions           |
| win_rate       | DECIMAL(5,4)  | Taux de réussite (0-1)                 |
| profit_factor  | DECIMAL(8,4)  | Facteur de profit                      |
| sharpe_ratio   | DECIMAL(8,4)  | Ratio de Sharpe                        |
| max_drawdown   | DECIMAL(8,4)  | Drawdown maximum (en pourcentage)      |
| net_profit     | DECIMAL(18,8) | Profit net (en devise)                 |
| symbol         | VARCHAR(20)   | Symbole (NULL pour global)             |

### bot_settings

Stocke la configuration du système, permettant une configuration dynamique sans redémarrage.

| Colonne         | Type        | Description                            |
|-----------------|-------------|----------------------------------------|
| id              | SERIAL      | Identifiant unique                     |
| parameter_name  | VARCHAR(50) | Nom du paramètre                       |
| parameter_value | TEXT        | Valeur du paramètre                    |
| updated_at      | TIMESTAMP   | Date et heure de dernière mise à jour  |
| description     | TEXT        | Description du paramètre               |

### analysis_results

Stocke les résultats des analyses quotidiennes et planifiées.

| Colonne         | Type        | Description                             |
|-----------------|-------------|-----------------------------------------|
| id              | SERIAL      | Identifiant unique                      |
| symbol          | VARCHAR(20) | Symbole/instrument analysé              |
| analysis_type   | VARCHAR(50) | Type d'analyse effectuée                |
| timeframe       | VARCHAR(10) | Timeframe de l'analyse                  |
| result_summary  | TEXT        | Résumé des résultats d'analyse          |
| result_detail   | JSONB       | Détails complets en format JSON         |
| opportunities   | JSONB       | Opportunités identifiées                |
| timestamp       | TIMESTAMP   | Date et heure de l'analyse              |
| score           | DECIMAL(5,4)| Score d'opportunité global (0-1)        |

### model_performance

Stocke les métriques de performance des différents modèles d'IA.

| Colonne        | Type        | Description                              |
|----------------|-------------|------------------------------------------|
| id             | SERIAL      | Identifiant unique                       |
| model_name     | VARCHAR(50) | Nom du modèle                            |
| symbol         | VARCHAR(20) | Symbole (NULL pour global)               |
| parameters     | JSONB       | Paramètres du modèle                     |
| metrics        | JSONB       | Métriques de performance (JSON)          |
| timestamp      | TIMESTAMP   | Date et heure de l'évaluation            |
| version        | VARCHAR(20) | Version du modèle                        |

### system_logs

Stocke les journaux système pour le débogage et la surveillance.

| Colonne    | Type         | Description                              |
|------------|--------------|------------------------------------------|
| id         | SERIAL       | Identifiant unique                       |
| timestamp  | TIMESTAMP    | Date et heure du log                     |
| component  | VARCHAR(50)  | Composant source du log                  |
| level      | VARCHAR(10)  | Niveau (INFO, WARNING, ERROR, etc.)      |
| message    | TEXT         | Message de log                           |
| details    | JSONB        | Détails additionnels (JSON)              |

## Relations entre tables

Les relations clés entre les tables incluent :

1. **trade_history** et **open_positions**
   - Une position fermée dans trade_history n'apparaît plus dans open_positions

2. **trade_history** et **trading_signals**
   - Une transaction dans trade_history est généralement liée à un ou plusieurs signaux dans trading_signals

3. **market_data** et **trading_signals**
   - Les signaux sont basés sur les données de marché

4. **open_positions** et **market_data**
   - Les données de prix actuelles dans open_positions sont mises à jour à partir de market_data

## Optimisations

### Indexes

Des index sont créés pour optimiser les requêtes fréquentes :

```sql
-- Indexes sur trade_history
CREATE INDEX idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX idx_trade_history_entry_time ON trade_history(entry_time);
CREATE INDEX idx_trade_history_status ON trade_history(status);

-- Indexes sur market_data (optimisés pour les séries temporelles)
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp DESC);

-- Indexes sur trading_signals
CREATE INDEX idx_trading_signals_symbol_timestamp ON trading_signals(symbol, timestamp DESC);
CREATE INDEX idx_trading_signals_source_timestamp ON trading_signals(signal_source, timestamp DESC);
```

### Partitionnement

Le partitionnement est utilisé pour les tables avec de grands volumes de données :

```sql
-- Exemple de partitionnement de market_data par mois
CREATE TABLE market_data (
    id SERIAL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(18,8) NOT NULL,
    high DECIMAL(18,8) NOT NULL,
    low DECIMAL(18,8) NOT NULL,
    close DECIMAL(18,8) NOT NULL,
    volume DECIMAL(18,8) NOT NULL,
    indicators JSONB,
    PRIMARY KEY (symbol, timestamp)
) PARTITION BY RANGE (timestamp);

-- Création de partitions mensuelles
CREATE TABLE market_data_y2023m01 PARTITION OF market_data
    FOR VALUES FROM ('2023-01-01') TO ('2023-02-01');
    
CREATE TABLE market_data_y2023m02 PARTITION OF market_data
    FOR VALUES FROM ('2023-02-01') TO ('2023-03-01');
-- etc.
```

### Rétention des données

Stratégie de rétention pour les données historiques :

- Données de marché détaillées : 2 ans
- Données agrégées (journalières) : conservation permanente
- Journaux système : 3 mois
- Signaux de trading : 1 an
- Transactions : conservation permanente

## Gestion des migrations

Les migrations de base de données sont gérées avec Alembic, intégré avec SQLAlchemy :

```python
# Exemple de migration avec Alembic
"""Add validator_score to trade_history

Revision ID: a1b2c3d4e5f6
Revises: g7h8i9j0k1l2
Create Date: 2023-04-15 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'a1b2c3d4e5f6'
down_revision = 'g7h8i9j0k1l2'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('trade_history', sa.Column('validator_score', sa.DECIMAL(5, 4)))

def downgrade():
    op.drop_column('trade_history', 'validator_score')
```

## Sauvegarde et restauration

Stratégie de sauvegarde :

1. **Sauvegardes complètes quotidiennes**
   ```bash
   pg_dump -U postgres -d evil2root_trading > /backups/evil2root_trading_$(date +%Y%m%d).sql
   ```

2. **Sauvegardes incrémentielles toutes les 6 heures**
   - Utilisation de WAL (Write-Ahead Logging)

3. **Rétention des sauvegardes**
   - Quotidiennes : 30 jours
   - Hebdomadaires : 6 mois
   - Mensuelles : 2 ans

4. **Procédure de restauration**
   ```bash
   # Restauration complète
   psql -U postgres -d evil2root_trading < /backups/evil2root_trading_20230415.sql
   
   # Restauration point-in-time
   pg_restore -U postgres -d evil2root_trading /backups/evil2root_trading_base.backup
   # Puis application des WAL jusqu'au point désiré
   ```

## Monitoring et maintenance

Procédures de maintenance régulières :

1. **VACUUM ANALYZE quotidien**
   ```sql
   VACUUM ANALYZE;
   ```

2. **Réindexation hebdomadaire**
   ```sql
   REINDEX DATABASE evil2root_trading;
   ```

3. **Vérification de l'intégrité mensuelle**
   ```bash
   pg_dump -U postgres -d evil2root_trading -F c -f /tmp/test_dump
   # Vérifier que le dump peut être restauré dans une base test
   ```

4. **Monitoring des performances**
   - Temps de réponse des requêtes fréquentes
   - Taille des tables et index
   - Utilisation du cache
   - Verrous et connexions
   
Les outils utilisés pour le monitoring incluent :
- pg_stat_statements pour l'analyse des requêtes
- pgAdmin pour le monitoring général
- Des requêtes personnalisées pour surveiller la croissance des tables 