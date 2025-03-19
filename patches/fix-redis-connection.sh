#!/bin/bash
set -e

# Vérifier si le fichier trading.py existe
if [ ! -f "app/trading.py" ]; then
    echo "Le fichier trading.py n'existe pas dans le chemin attendu."
    exit 1
fi

echo "Création d'une sauvegarde du fichier original..."
cp app/trading.py app/trading.py.bak

echo "Application du correctif pour la gestion de l'URL Redis..."
sed -i '
/redis_port = int(os.environ.get(.REDIS_PORT., 6379))/c\
# Parse Redis connection details\
redis_port_raw = os.environ.get("REDIS_PORT", "6379")\
# Handle case when REDIS_PORT is a full URL (tcp://host:port)\
if "://" in redis_port_raw:\
    import re\
    # Extract port from URL using regex\
    port_match = re.search(r":(\\d+)$", redis_port_raw)\
    redis_port = int(port_match.group(1)) if port_match else 6379\
else:\
    redis_port = int(redis_port_raw)
' app/trading.py

echo "Correctif appliqué avec succès!" 