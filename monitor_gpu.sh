#!/bin/bash

# Script de surveillance GPU pour RTX 2070 SUPER
# Usage: ./monitor_gpu.sh [intervalle en secondes]

# Définir l'intervalle en secondes (par défaut 5 secondes)
INTERVAL=${1:-5}

# Vérifier que nvidia-smi est disponible
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERREUR: nvidia-smi n'est pas disponible."
    echo "Vérifiez que les pilotes NVIDIA sont correctement installés."
    exit 1
fi

# Créer un fichier de log
LOG_FILE="logs/gpu_monitoring_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "=== Surveillance GPU RTX 2070 SUPER ===" | tee -a "$LOG_FILE"
echo "Date de début: $(date)" | tee -a "$LOG_FILE"
echo "Intervalle: ${INTERVAL} secondes" | tee -a "$LOG_FILE"
echo "Log enregistré dans: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Appuyez sur Ctrl+C pour arrêter..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Récupérer les infos de base du GPU
GPU_INFO=$(nvidia-smi -L)
echo "GPU détecté: $GPU_INFO" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# En-tête du tableau
printf "%-25s | %-8s | %-12s | %-12s | %-15s | %-10s\n" "TIMESTAMP" "TEMP(°C)" "UTIL GPU(%)" "MEM GPU(%)" "MEM USAGE(MB)" "POWER(W)" | tee -a "$LOG_FILE"
printf "%-25s-+-%-8s-+-%-12s-+-%-12s-+-%-15s-+-%-10s\n" "-------------------------" "--------" "------------" "------------" "---------------" "----------" | tee -a "$LOG_FILE"

# Surveiller en continu
while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    
    # Obtenir les statistiques GPU
    GPU_STATS=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits)
    
    # Traiter les statistiques
    while IFS="," read -r TEMP GPU_UTIL MEM_UTIL MEM_USED MEM_TOTAL POWER; do
        # Calculer le pourcentage d'utilisation mémoire
        MEM_PERCENT=$(echo "scale=2; ($MEM_USED / $MEM_TOTAL) * 100" | bc)
        
        # Nettoyer les valeurs (supprimer les espaces)
        TEMP=$(echo $TEMP | xargs)
        GPU_UTIL=$(echo $GPU_UTIL | xargs)
        MEM_UTIL=$(echo $MEM_UTIL | xargs)
        MEM_USED=$(echo $MEM_USED | xargs)
        MEM_TOTAL=$(echo $MEM_TOTAL | xargs)
        POWER=$(echo $POWER | xargs)
        
        # Formater l'affichage
        MEM_DISPLAY="${MEM_USED}/${MEM_TOTAL} (${MEM_PERCENT}%)"
        
        # Afficher les résultats
        printf "%-25s | %-8s | %-12s | %-12s | %-15s | %-10s\n" "$TIMESTAMP" "$TEMP" "$GPU_UTIL" "$MEM_UTIL" "$MEM_DISPLAY" "$POWER" | tee -a "$LOG_FILE"
    done <<< "$GPU_STATS"
    
    # Attendre l'intervalle spécifié
    sleep $INTERVAL
done 