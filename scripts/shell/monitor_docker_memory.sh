#!/bin/bash

# Script pour surveiller la consommation de mémoire des conteneurs Docker
# Usage: ./monitor_docker_memory.sh [intervalle en secondes]

# Définir l'intervalle en secondes (par défaut 5 secondes)
INTERVAL=${1:-5}

# Fonction pour convertir les unités en Mo
convert_to_mb() {
    local value=$1
    local unit=$2
    
    case $unit in
        B)
            echo "scale=2; $value / 1048576" | bc
            ;;
        KB)
            echo "scale=2; $value / 1024" | bc
            ;;
        MB)
            echo $value
            ;;
        GB)
            echo "scale=2; $value * 1024" | bc
            ;;
        *)
            echo "0"
            ;;
    esac
}

# Créer un fichier de log
LOG_FILE="logs/docker_memory_usage_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "=== Surveillance de la mémoire Docker ===" | tee -a "$LOG_FILE"
echo "Date de début: $(date)" | tee -a "$LOG_FILE"
echo "Intervalle: ${INTERVAL} secondes" | tee -a "$LOG_FILE"
echo "Log enregistré dans: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Appuyez sur Ctrl+C pour arrêter..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# En-tête du tableau
printf "%-25s | %-10s | %-20s | %-25s | %-15s\n" "TIMESTAMP" "CONTAINER" "NAME" "MEMORY USAGE" "CPU %" | tee -a "$LOG_FILE"
printf "%-25s-+-%-10s-+-%-20s-+-%-25s-+-%-15s\n" "-------------------------" "----------" "--------------------" "-------------------------" "---------------" | tee -a "$LOG_FILE"

# Surveiller en continu
while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    
    # Récupérer les statistiques Docker
    docker stats --no-stream --format "{{.Container}};{{.Name}};{{.MemUsage}};{{.CPUPerc}}" | while IFS=';' read -r container name mem_usage cpu_perc; do
        # Extraire les valeurs de mémoire
        mem_used=$(echo $mem_usage | awk '{print $1}')
        mem_used_unit=$(echo $mem_usage | awk '{print $2}' | sed 's/[^A-Z]//g')
        mem_limit=$(echo $mem_usage | awk '{print $3}')
        mem_limit_unit=$(echo $mem_usage | awk '{print $4}' | sed 's/[^A-Z]//g')
        
        # Convertir en Mo pour faciliter la comparaison
        mem_used_mb=$(convert_to_mb $mem_used $mem_used_unit)
        mem_limit_mb=$(convert_to_mb $mem_limit $mem_limit_unit)
        mem_percent=$(echo "scale=2; ($mem_used_mb / $mem_limit_mb) * 100" | bc)
        
        # Formater l'affichage
        formatted_mem="${mem_used_mb}MB / ${mem_limit_mb}MB (${mem_percent}%)"
        
        # Afficher le résultat
        printf "%-25s | %-10s | %-20s | %-25s | %-15s\n" "$TIMESTAMP" "${container:0:10}" "$name" "$formatted_mem" "$cpu_perc" | tee -a "$LOG_FILE"
    done
    
    # Ajouter une ligne pour séparer les intervalles
    echo "" | tee -a "$LOG_FILE"
    
    # Attendre l'intervalle spécifié
    sleep $INTERVAL
done 