#!/bin/bash
set -e

echo "Installation de finbert-embedding dans un environnement virtuel séparé..."

# Créer un environnement virtuel dédié pour finbert-embedding
python -m venv /opt/finbert-venv

# Activer l'environnement virtuel
source /opt/finbert-venv/bin/activate

# Installer la version spécifique de torch requise par finbert-embedding
echo "Installation de torch 1.1.0 requise par finbert-embedding..."
pip install torch==1.1.0

# Installer finbert-embedding
echo "Installation de finbert-embedding..."
pip install finbert-embedding==0.1.3

# Créer un wrapper script pour utiliser finbert dans l'environnement spécifique
cat > /usr/local/bin/run-finbert << 'EOF'
#!/bin/bash
# Wrapper pour exécuter du code Python avec finbert-embedding
source /opt/finbert-venv/bin/activate
python "$@"
EOF

chmod +x /usr/local/bin/run-finbert

echo "Installation de finbert-embedding terminée."
echo "Pour utiliser finbert-embedding, exécutez vos scripts Python avec la commande 'run-finbert' au lieu de 'python'."
echo "Exemple: run-finbert mon_script_finbert.py"

# Retourner à l'environnement principal
deactivate

exit 0 