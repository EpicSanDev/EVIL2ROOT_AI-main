#!/bin/bash
echo "Début de l'installation alternative de la bibliothèque C TA-Lib..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

apt-get update
apt-get install -y --no-install-recommends wget build-essential gcc g++ make pkg-config

# Utiliser une méthode alternative pour installer TA-Lib
echo "Téléchargement de TA-Lib 0.4.0..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/

# Appliquer un patch pour corriger les problèmes de symboles manquants
# Patch pour ajouter TA_AVGDEV_Lookback et autres symboles manquants
cat > avgdev_patch.h << 'EOL'
/* TA_AVGDEV - Average Deviation */
int TA_AVGDEV_Lookback(int optInTimePeriod);

TA_RetCode TA_AVGDEV(int startIdx, int endIdx, const double inReal[],
                    int optInTimePeriod, 
                    int *outBegIdx, int *outNbElement, double outReal[]);
EOL

# Ajouter le patch au fichier d'en-tête approprié
cp avgdev_patch.h include/ta_func/

echo "Configuration et compilation de TA-Lib..."
./configure --prefix=/usr
make -j$(nproc)
make install

ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so
ln -sf /usr/lib/libta_lib.so.0 /usr/lib/libta-lib.so
echo "/usr/lib" > /etc/ld.so.conf.d/talib.conf
ldconfig
echo "Installation alternative de TA-Lib terminée avec succès!"
