# Gestion des dépendances et conflits

Ce document explique comment nous gérons les conflits de dépendances dans le projet Evil2Root AI, particulièrement pour les bibliothèques Python qui présentent des incompatibilités.

## Problème de dépendances conflictuelles avec Torch

Notre projet utilise plusieurs bibliothèques qui dépendent de `torch` avec des exigences de version incompatibles :

- `optimum 1.9.1` requiert torch>=1.9
- `stable-baselines3 2.1.0` requiert torch>=1.13  
- `fastai 2.7.12` requiert torch<2.1 et >=1.7
- `finbert-embedding 0.1.3` requiert torch==1.1.0 (incompatible avec les autres)

## Solution mise en place

1. **Alignement des versions compatibles**  
   - Dans `requirements.txt`, nous avons défini `torch>=1.13.0,<2.1.0` pour satisfaire `optimum`, `stable-baselines3` et `fastai`.
   - `finbert-embedding` a été commenté car sa dépendance sur torch==1.1.0 est incompatible avec le reste du projet.

2. **Installation stratégique dans Docker**  
   - Le Dockerfile a été modifié pour installer les dépendances par groupes logiques
   - Nous installons d'abord les dépendances de base, puis web, puis ML (en séparant torch et les bibliothèques qui en dépendent)
   - Les flags `--no-deps` et `--use-pep517` sont utilisés stratégiquement pour éviter les conflits

3. **Environnement virtuel séparé pour finbert-embedding**  
   - Script `fix-finbert-install.sh` qui crée un environnement virtuel dédié avec torch 1.1.0
   - Wrapper `run-finbert` pour exécuter du code nécessitant finbert-embedding

## Comment utiliser finbert-embedding

Si vous avez besoin d'utiliser finbert-embedding, deux options s'offrent à vous :

1. **Dans Docker** : Exécuter le script `/tmp/fix-finbert-install.sh`, puis utiliser la commande `run-finbert` pour exécuter vos scripts qui nécessitent finbert-embedding.

2. **En développement local** : Créer un environnement virtuel séparé avec torch 1.1.0 :

```bash
python -m venv finbert-venv
source finbert-venv/bin/activate  # ou finbert-venv\Scripts\activate sous Windows
pip install torch==1.1.0
pip install finbert-embedding==0.1.3
```

## Autres problèmes connus et solutions

- **causalml et econml** : Gérés par le script `fix-causalml-install.sh` qui installe des versions compatibles
- **Plotly et Dash** : Installés séparément sans dépendances pour éviter les conflits

## Dépannage

Si vous rencontrez des erreurs d'installation liées aux dépendances :

1. Vérifiez la section "ResolutionImpossible" des logs pour identifier les conflits de version
2. Essayez d'installer les packages conflictuels séparément ou avec des flags comme `--no-deps`
3. Créez des environnements virtuels séparés pour les combinaisons incompatibles
4. Pour les packages non essentiels, considérez de les commenter dans requirements.txt

Pour toute question ou problème persistant, ouvrez une issue dans le dépôt du projet. 