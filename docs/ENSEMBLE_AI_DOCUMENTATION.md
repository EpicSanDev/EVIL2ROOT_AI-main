# Documentation: Ensemble Learning & Explainable AI

## Table des matières

1. [Introduction](#introduction)
2. [Architecture des Modèles d'Ensemble](#architecture-des-modèles-densemble)
3. [Feature Engineering Avancé](#feature-engineering-avancé)
4. [Explainable AI (XAI)](#explainable-ai-xai)
5. [Interface Utilisateur](#interface-utilisateur)
6. [API Reference](#api-reference)
7. [Guide d'Interprétation](#guide-dinterprétation)
8. [Dépendances](#dépendances)
9. [Troubleshooting](#troubleshooting)
10. [Ressources Additionnelles](#ressources-additionnelles)

---

## Introduction

Le module d'Ensemble Learning et d'Explainable AI implémenté dans EVIL2ROOT Trading Bot améliore significativement la précision et la transparence des décisions de trading. Cette documentation couvre l'architecture, l'utilisation et l'interprétation des résultats de ces nouvelles fonctionnalités.

### Objectifs

- **Améliorer la précision des prédictions** en combinant plusieurs modèles complémentaires
- **Rendre les décisions de trading explicables** pour l'utilisateur final
- **Fournir des insights détaillés** sur les facteurs qui influencent les décisions
- **Augmenter la confiance** dans les décisions du système de trading

---

## Architecture des Modèles d'Ensemble

### Vue d'ensemble

Le système utilise une approche d'ensemble à multiple niveaux qui combine différents algorithmes de machine learning pour trois tâches prédictives distinctes:

1. **Prédiction de Prix** (Régression)
2. **Prédiction de Direction** (Classification)
3. **Prédiction de Volatilité** (Régression)

Chaque ensemble est ensuite combiné pour produire une décision de trading finale avec un niveau de confiance et une évaluation du risque.

### Types d'Ensembles Implémentés

#### Stacking Ensemble (Prédiction de Prix et Volatilité)

![Stacking Ensemble](https://i.imgur.com/vGJhZ4m.png)

L'approche de stacking utilise un méta-modèle qui apprend à combiner les prédictions des modèles de base. Cette approche est particulièrement efficace pour les tâches de régression comme la prédiction de prix et de volatilité.

**Modèles de base utilisés**:
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

**Méta-modèle**: Ridge Regression

#### Voting Ensemble (Prédiction de Direction)

![Voting Ensemble](https://i.imgur.com/K7dBYt2.png)

L'approche de voting combine les prédictions des modèles de base par vote pondéré. Cette approche est particulièrement adaptée pour la classification de direction (hausse/baisse).

**Modèles de base utilisés**:
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier

### Validation Croisée Temporelle

Pour éviter le data leakage, tous les modèles sont entraînés et validés avec une approche de validation croisée temporelle (`TimeSeriesSplit`), ce qui respecte l'ordre chronologique des données financières.

### Code d'Exemple

Pour initialiser les modèles d'ensemble:

```python
from app.models.ensemble_integrator import EnsembleIntegrator

# Initialisation
ensemble_integrator = EnsembleIntegrator(use_explainable_ai=True)

# Entraînement
symbols = ["AAPL", "MSFT", "GOOGL"]
results = ensemble_integrator.train_models(data, symbols, existing_models)

# Prédiction
prediction_result = ensemble_integrator.predict(market_data, "AAPL")
```

---

## Feature Engineering Avancé

Le système implémente plus de 40 indicateurs techniques et transformations avancées pour capturer différents aspects du marché.

### Catégories de Features

#### 1. Indicateurs de Tendance
- Moving Average Crossovers (5/20, 10/50, 20/100)
- Average Directional Index (ADX)
- MACD

#### 2. Indicateurs de Momentum
- Rate of Change (ROC) sur différentes périodes
- Momentum
- RSI
- Stochastic Oscillator (K et D)

#### 3. Indicateurs de Volatilité
- Average True Range (ATR)
- Bollinger Bands (normalisées et width)

#### 4. Indicateurs basés sur le Volume
- On-Balance Volume (OBV)
- Chaikin Money Flow
- Accumulation/Distribution Line

#### 5. Reconnaissance de Patterns
- Doji, Hammer
- Reconnaissance de patterns simplifiée

#### 6. Features Transformées
- Lags de prix et de rendements
- Statistiques mobiles (écart-type, kurtosis, skewness)
- Croisements de features (ex: RSI × BBW)

#### 7. Features basées sur Transformation de Fourier
- Composantes fréquentielles dominantes

### Exemple de Code pour les Features

```python
# Ajout de features avancées
def add_feature_engineering(data):
    df = data.copy()
    
    # Tendance
    for fast, slow in [(5, 20), (10, 50), (20, 100)]:
        df[f'SMA_{fast}'] = df['Close'].rolling(window=fast).mean()
        df[f'SMA_{slow}'] = df['Close'].rolling(window=slow).mean()
        df[f'SMA_crossover_{fast}_{slow}'] = np.where(df[f'SMA_{fast}'] > df[f'SMA_{slow}'], 1, -1)
    
    # Momentum
    for period in [5, 10, 20]:
        df[f'ROC_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
        df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
    
    # Et bien plus...
    
    return df
```

---

## Explainable AI (XAI)

L'Explainable AI est implémentée à travers plusieurs méthodes complémentaires pour offrir une vue complète et compréhensible des décisions de trading.

### Méthodes XAI Implémentées

#### 1. SHAP (SHapley Additive exPlanations)

![SHAP Valeurs](https://i.imgur.com/XfRnJdp.png)

SHAP utilise la théorie des jeux pour attribuer à chaque feature une valeur indiquant son importance dans la prédiction. L'implémentation utilise:

- **Waterfall Plots**: Montre comment chaque feature contribue à la prédiction finale
- **Beeswarm Plots**: Vue d'ensemble de l'impact des features sur toutes les prédictions

#### 2. Feature Importance

Deux types d'importances de features sont calculés:

- **Model-based Importance**: Dérivée directement des modèles (e.g., feature_importances_ pour les forêts aléatoires)
- **Permutation Importance**: Calculée en mélangeant aléatoirement les valeurs d'une feature et en mesurant la dégradation des performances

#### 3. Rapports d'Explication Textuels

Des explications en langage naturel sont générées pour chaque décision, expliquant:

- Quelles features ont le plus contribué à la décision
- Si ces contributions étaient positives ou négatives
- L'amplitude de ces contributions

### Intégration avec l'Interface Utilisateur

Les explications SHAP sont rendues visuellement dans l'interface web pour une interprétation intuitive des décisions.

### Exemple de Code SHAP

```python
# Initialisation de l'explainer SHAP
explainer = shap.Explainer(model, X_train)

# Calcul des valeurs SHAP pour les dernières données
shap_values = explainer(X_recent)

# Génération de graphiques
shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)
```

---

## Interface Utilisateur

Une interface utilisateur dédiée à l'explainable AI est disponible à l'adresse `/model_explanations`. Cette interface permet:

### Fonctionnalités

1. **Sélection du Symbole**: Choisir parmi les symboles disponibles
2. **Sélection de la Période**: 1 jour à 1 an
3. **Résumé de Décision**: Vue synthétique avec recommandation, confiance et niveau de risque
4. **Analyse Combinée**: Vue d'ensemble des prédictions et explications
5. **Analyses Spécifiques**: Onglets dédiés pour prix, direction et volatilité
6. **Visualisations SHAP**: Graphiques interactifs pour l'interprétation des décisions

### Visualisations Disponibles

- **Barres d'Importance des Features**: Représentation visuelle de l'importance relative
- **Graphiques Waterfall**: Impact individuel des features sur une prédiction
- **Graphiques Beeswarm**: Distribution des impacts des features sur toutes les prédictions

### Screenshot

![Interface Explainable AI](https://i.imgur.com/vBwPGVm.png)

---

## API Reference

### Endpoints

#### `POST /api/model_explanations`

Génère des explications détaillées pour un symbole spécifique.

**Paramètres**:
- `symbol` (requis): Symbole boursier (ex: AAPL)
- `date_range` (optionnel): Période d'analyse (1d, 1w, 1m, 3m, 6m, 1y)

**Réponse**:
```json
{
  "symbol": "AAPL",
  "timestamp": "2023-06-10 15:30:45",
  "recommendation": "BUY - moderate (0.87%)",
  "confidence": 0.72,
  "risk_level": "medium",
  "price": 178.25,
  "direction": 1,
  "volatility": 0.015,
  "summary": "Analysis completed for AAPL based on 30 data points.",
  "combined_explanation": "# Trading Decision Explanation for AAPL...",
  "price_explanation": "Price prediction for AAPL is influenced most by...",
  "direction_explanation": "Direction prediction (Up) for AAPL is influenced most by...",
  "volatility_explanation": "Volatility prediction for AAPL is influenced most by...",
  "top_features": {
    "price": {
      "SMA_20": 0.35,
      "Volatility_10": 0.28,
      // ...
    },
    "direction": {
      // ...
    },
    "volatility": {
      // ...
    }
  },
  "plots": {
    "price": {
      "waterfall_path": "/static/plots/AAPL_price_waterfall.png",
      "beeswarm_path": "/static/plots/AAPL_price_beeswarm.png"
    },
    // ...
  }
}
```

#### `GET /model_explanations`

Renvoie la page d'interface utilisateur des explications de modèles.

### Classes Principales

#### `EnsembleModel`

Classe principale pour l'implémentation des modèles d'ensemble avec explainable AI.

**Méthodes principales**:
- `add_feature_engineering(data)`: Ajoute des features avancées
- `train(data, symbol, target_col, classification)`: Entraîne le modèle d'ensemble
- `predict(data, symbol)`: Génère des prédictions avec explications
- `generate_explanation_report(data, symbol)`: Génère un rapport d'explication complet

#### `EnsembleIntegrator`

Intègre plusieurs modèles d'ensemble pour différentes tâches prédictives.

**Méthodes principales**:
- `train_models(data, symbols, existing_models)`: Entraîne tous les modèles d'ensemble
- `predict(data, symbol)`: Génère des prédictions combinées
- `generate_explanation_report(symbol, data)`: Génère un rapport d'explication combiné

---

## Guide d'Interprétation

### Lire un Graphique Waterfall

![Waterfall Plot Guide](https://i.imgur.com/RG1XJCk.png)

Le graphique waterfall montre:
- La valeur de base (prédiction moyenne)
- La contribution positive (rouge) ou négative (bleue) de chaque feature
- La valeur finale prédite

### Interpréter les Scores d'Importance

Les scores d'importance sont normalisés. Pour les interpréter:
- **Valeur positive**: La feature augmente la prédiction
- **Valeur négative**: La feature diminue la prédiction
- **Magnitude**: Plus la valeur absolue est grande, plus l'impact est important

### Comprendre la Confiance et le Risque

- **Score de confiance**: Combinaison de la confiance directionnelle et de l'amplitude du mouvement prédit
- **Niveau de risque**: Basé principalement sur la volatilité prédite
  - Faible: < 1% de volatilité attendue
  - Moyen: 1-3% de volatilité attendue
  - Élevé: > 3% de volatilité attendue

### Exemple d'Interprétation

Pour une recommandation "BUY - moderate (0.87%)":
1. L'action est attendue à la hausse (BUY)
2. La conviction est modérée (moderate)
3. L'amplitude anticipée est +0.87%

---

## Dépendances

Les nouvelles fonctionnalités requièrent l'installation des packages suivants:

```
# Explainable AI / Ensemble Learning
shap>=0.43.0
lime>=0.2.0
interpret>=0.4.2
interpret-core>=0.4.2
alibi>=0.9.5
dalex>=1.5.0
econml>=0.14.1
causalml>=0.13.0
bayesian-optimization>=1.4.3
mlflow>=2.11.0
wandb>=0.16.5
mlxtend>=0.23.0
catboost>=1.2.3
eli5>=0.13.0
graphviz>=0.20.1
dtreeviz>=2.2.2
yellowbrick>=1.5
plotly_express>=0.4.1
```

Installation des dépendances:

```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Problèmes Courants

#### 1. Pas de Visualisations SHAP

**Problème**: Les graphiques SHAP ne s'affichent pas dans l'interface.

**Solution**:
- Vérifier que les dépendances sont correctement installées
- S'assurer que le dossier `static/plots` existe et est accessible en écriture
- Vérifier les logs pour des erreurs liées à matplotlib

#### 2. Erreurs de Mémoire

**Problème**: Out of Memory lors de l'entraînement ou de la génération d'explications.

**Solution**:
- Réduire le nombre de features utilisées
- Limiter la période d'analyse
- Augmenter la mémoire disponible au conteneur
- Utiliser l'option `use_shap=False` pour désactiver SHAP si nécessaire

#### 3. Modèles Divergents

**Problème**: Les différents modèles d'ensemble donnent des prédictions contradictoires.

**Solution**:
- Réviser les hyperparamètres des modèles
- Ajouter plus de données d'entraînement
- Vérifier la qualité des données d'entrée
- Ajuster les poids des modèles dans l'ensemble

### Logs de Débogage

Pour activer les logs détaillés, définir la variable d'environnement:

```bash
export LOG_LEVEL=DEBUG
```

---

## Ressources Additionnelles

### Articles Techniques

- [SHAP: A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)
- [A Structured Approach to Designing Ensemble Learning Algorithms](https://doi.org/10.1016/j.knosys.2020.106619)
- [XAI for Finance: Beyond Feature Attribution](https://arxiv.org/abs/2306.11724)

### Vidéos Explicatives

- [Understanding SHAP Values in Machine Learning](https://www.youtube.com/watch?v=VB9uVfPR0dw)
- [How to Build Ensemble Models for Time Series Forecasting](https://www.youtube.com/watch?v=Ku0Jz3UpA0s)

### Documentation Officielle

- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/) 