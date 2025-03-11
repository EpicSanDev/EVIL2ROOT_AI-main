# Contribuer au projet EVIL2ROOT Trading Bot

Nous sommes ravis que vous souhaitiez contribuer au projet EVIL2ROOT Trading Bot ! Ce document fournit des lignes directrices pour contribuer au projet.

## Table des matières

- [Code de conduite](#code-de-conduite)
- [Comment puis-je contribuer?](#comment-puis-je-contribuer)
  - [Signaler des bugs](#signaler-des-bugs)
  - [Suggérer des améliorations](#suggérer-des-améliorations)
  - [Contribuer du code](#contribuer-du-code)
  - [Documentation](#documentation)
- [Style de codage](#style-de-codage)
- [Pull Requests](#pull-requests)
- [Processus de développement](#processus-de-développement)
- [Communication](#communication)

## Code de conduite

Ce projet et tous ses participants sont régis par notre [Code de Conduite](CODE_OF_CONDUCT.md). En participant, vous êtes censé respecter ce code. Veuillez signaler tout comportement inacceptable à evil2root@protonmail.com.

## Comment puis-je contribuer?

### Signaler des bugs

Les bugs sont suivis comme des [issues GitHub](https://github.com/Evil2Root/EVIL2ROOT_AI/issues). Créez une issue et fournissez les informations suivantes:

- Utilisez un titre clair et descriptif
- Décrivez les étapes exactes pour reproduire le problème
- Décrivez le comportement observé et ce à quoi vous vous attendiez
- Incluez des captures d'écran si possible
- Précisez votre environnement (OS, version Python, etc.)

### Suggérer des améliorations

Les suggestions d'amélioration sont également suivies via les [issues GitHub](https://github.com/Evil2Root/EVIL2ROOT_AI/issues):

- Utilisez un titre et une description clairs
- Fournissez une explication détaillée de la fonctionnalité proposée
- Expliquez pourquoi cette fonctionnalité serait utile
- Si possible, décrivez comment vous envisagez l'implémentation

### Contribuer du code

Si vous souhaitez corriger un bug ou implémenter une fonctionnalité:

1. Forker le dépôt
2. Créer une branche à partir de `main`:
   ```bash
   git checkout -b feature/ma-fonctionnalite
   # ou
   git checkout -b fix/mon-correctif
   ```
3. Apporter vos modifications
4. S'assurer que les tests passent:
   ```bash
   make test
   ```
5. Faire un commit de vos changements avec des messages explicites:
   ```bash
   git commit -m "feat: Ajouter une nouvelle fonctionnalité"
   # ou
   git commit -m "fix: Corriger un problème avec X"
   ```
   Nous suivons [Conventional Commits](https://www.conventionalcommits.org/) pour les messages de commit.
6. Pousser vers votre fork:
   ```bash
   git push origin feature/ma-fonctionnalite
   ```
7. Ouvrir une Pull Request contre la branche `main` du dépôt original

### Documentation

Les améliorations de la documentation sont toujours les bienvenues. Vous pouvez:

- Améliorer le README.md
- Ajouter des commentaires au code
- Créer ou améliorer des guides dans le wiki
- Corriger des fautes de frappe ou clarifier des explications existantes

## Style de codage

- Nous suivons [PEP 8](https://pep8.org/) pour le code Python
- Utilisez 4 espaces pour l'indentation (pas de tabulations)
- Lignes limitées à 100 caractères maximum
- Utilisez des docstrings pour documenter les fonctions et les classes
- Exécutez `flake8` avant de soumettre votre code

## Pull Requests

- Mettez à jour la documentation si nécessaire
- Mettez à jour le CHANGELOG.md avec vos changements
- Les PR doivent passer tous les tests CI
- Les PR nécessitent l'approbation d'au moins un maintainer
- Nous pouvons demander des modifications avant de merger votre PR

## Processus de développement

1. **Issues**: Toutes les tâches commencent par une issue
2. **Branches**: Créez une branche à partir de `main` pour travailler
3. **Pull Requests**: Soumettez votre travail via une PR
4. **Revue**: Un maintainer examinera votre PR
5. **Merge**: Après approbation, votre PR sera mergée dans `main`
6. **Release**: Les versions sont tagguées selon [Semantic Versioning](https://semver.org/)

## Communication

- Utilisez les issues GitHub pour les discussions liées au code
- Pour les questions générales, utilisez les [discussions GitHub](https://github.com/Evil2Root/EVIL2ROOT_AI/discussions)
- Pour les communications privées, contactez evil2root@protonmail.com

Nous apprécions tous les contributeurs et nous nous efforçons de traiter chaque contribution avec respect. Merci de consacrer du temps à notre projet! 