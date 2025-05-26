# Guide d'installation ARM64 (Apple Silicon)

Ce document fournit des instructions et des solutions pour compiler EVIL2ROOT AI sur des machines ARM64, en particulier les Mac équipés d'Apple Silicon (M1/M2/M3).

## Défis spécifiques à ARM64

L'architecture ARM64 (Apple Silicon) présente des défis particuliers lors de la compilation:

1. **Détection d'architecture**: Les scripts `config.guess` et `config.sub` des packages plus anciens comme TA-Lib ne reconnaissent pas toujours correctement l'architecture ARM64.

2. **Compatibilité des bibliothèques**: Certains packages C/C++ n'ont pas été conçus pour ARM64 et nécessitent des modifications pour compiler correctement.

3. **Émulation Rosetta**: Parfois, l'émulation x86_64 via Rosetta est nécessaire pour les dépendances qui n'ont pas de support ARM64 natif.

4. **Timeouts de build**: Les compilations peuvent prendre plus de temps, surtout pour les packages complexes comme TA-Lib, ce qui peut entraîner des timeouts.

## Préparation de l'environnement

### 1. Configuration Docker optimale

Dans Docker Desktop:

1. Allouez suffisamment de ressources:
   - Au moins 4GB de RAM
   - Au moins 2 CPUs
   - 20GB+ d'espace disque

2. Activez les fonctionnalités expérimentales:
   - `Use Rosetta for x86/amd64 emulation`
   - `VirtioFS` pour de meilleures performances de système de fichiers

### 2. Outils de diagnostic

Nous fournissons un script de diagnostic ARM64:

```bash
# Diagnostiquer votre environnement
./docker/arm64-troubleshoot.sh --diagnose

# Optimiser votre environnement
./docker/arm64-troubleshoot.sh --prepare-env

# Vérifier la configuration Rosetta
./docker/arm64-troubleshoot.sh --check-rosetta
```

## Méthodes de build pour ARM64

### Option 1: Build optimisé ARM64 natif (recommandé)

Cette méthode utilise des optimisations spécifiques à ARM64:

```bash
# Via le script dédié
./docker/build-arm64.sh

# Ou via make
make build-arm64
```

### Option 2: Build avec Mock TA-Lib (plus rapide)

Cette méthode utilise une implémentation simplifiée de TA-Lib optimisée pour ARM64:

```bash
# Via le script dédié
./docker/build-arm64.sh --use-mock-talib

# Ou via make
make build-arm64-mock
```

### Option 3: Build minimal (le plus rapide)

Cette option n'installe que les dépendances essentielles:

```bash
# Via le script dédié
./docker/build-arm64.sh --essential-only

# Ou via make
make build-arm64-minimal
```

### Option 4: Forcer l'utilisation du Mock TA-Lib

Si vous rencontrez des problèmes persistants avec TA-Lib:

```bash
# Créer une configuration optimisée avec mock TA-Lib
./docker/arm64-troubleshoot.sh --force-mock

# Puis construire avec le Dockerfile optimisé
./quick-build-arm64.sh
```

## Résolution des problèmes courants

### Timeouts lors de la compilation de TA-Lib

Si vous rencontrez des timeouts lors de la compilation de TA-Lib:

```bash
# Réparer spécifiquement TA-Lib
./docker/arm64-troubleshoot.sh --fix-talib
```

### Erreurs de compilation TA-Lib

Les erreurs courantes incluent:

1. **Erreur de détection d'architecture**:
   ```
   configure: error: cannot guess build type; you must specify one
   ```
   Solution: Utilisez le script `--fix-talib` ou `build-arm64.sh` qui corrige automatiquement ce problème.

2. **Erreurs de compilation de code source**:
   ```
   ta_abstract.c:27:10: fatal error: 'stddef.h' file not found
   ```
   Solution: Utilisez l'option `--use-mock-talib` qui évite complètement la compilation.

3. **Problèmes avec les en-têtes Python**:
   Solution: Le script `talib-fallback-install.sh` amélioré ajoute automatiquement les chemins corrects.

### Erreurs Docker

1. **Manque de mémoire**:
   ```
   Killed
   ```
   Solution: Augmentez la mémoire allouée à Docker Desktop.

2. **Problèmes avec Rosetta**:
   ```
   qemu: uncaught target signal 11 (Segmentation fault) - core dumped
   ```
   Solution: Exécutez `./docker/arm64-troubleshoot.sh --check-rosetta` pour vérifier et configurer Rosetta.

## Fonctionnalités du Mock TA-Lib pour ARM64

Notre implémentation mockup de TA-Lib offre:

- Implémentations natives des indicateurs populaires (SMA, EMA, RSI, MACD, BBANDS)
- Optimisations spécifiques à ARM64
- Rapidité d'installation sans compilation
- Compatible avec l'API TA-Lib standard pour assurer le fonctionnement du bot

## Notes sur les performances

- Les builds natifs ARM64 sont généralement 30-40% plus rapides que les versions émulées
- Le mock TA-Lib peut être 5-10% moins précis mais offre d'excellentes performances
- La version minimale est suffisante pour le trading de base mais manque de fonctionnalités d'analyse avancées

## Benchmarks

| Configuration | Temps de build | Taille d'image | Performances |
|---------------|----------------|----------------|--------------|
| ARM64 Complet | ~45 min        | ~2.5GB         | 100%         |
| ARM64 + Mock  | ~15 min        | ~2.3GB         | 95%          |
| ARM64 Minimal | ~5 min         | ~800MB         | 70%          |

## Support et contact

Si vous rencontrez des problèmes persistants avec les builds ARM64, contactez-nous sur le Discord de EVIL2ROOT ou ouvrez un ticket sur GitHub.
