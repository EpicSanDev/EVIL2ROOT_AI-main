#!/bin/bash

# Script pour arrêter le scheduler d'analyse de marché
# Usage: ./stop_market_scheduler.sh

echo "=== Evil2Root Trading Bot - Arrêt du scheduler d'analyse de marché ==="
echo ""

# Vérifier si le fichier PID existe
if [ -f .scheduler_pid ]; then
    PID=$(cat .scheduler_pid)
    
    # Vérifier si le processus existe toujours
    if ps -p $PID > /dev/null; then
        echo "Arrêt du scheduler avec PID: $PID..."
        kill $PID
        
        # Attendre que le processus se termine
        echo "Attente de la fin du processus..."
        for i in {1..5}; do
            if ! ps -p $PID > /dev/null; then
                echo "Processus terminé avec succès"
                break
            fi
            sleep 1
        done
        
        # Si le processus est toujours en vie après 5 secondes, le tuer avec force
        if ps -p $PID > /dev/null; then
            echo "Le processus ne répond pas. Arrêt forcé..."
            kill -9 $PID
            sleep 1
            
            if ! ps -p $PID > /dev/null; then
                echo "Processus terminé avec force"
            else
                echo "Impossible d'arrêter le processus. Veuillez le terminer manuellement."
            fi
        fi
        
        # Supprimer le fichier PID
        rm .scheduler_pid
        
    else
        echo "Aucun processus en cours d'exécution avec le PID: $PID"
        echo "Le scheduler a peut-être déjà été arrêté."
        rm .scheduler_pid
    fi
else
    echo "Fichier PID non trouvé. Le scheduler n'est peut-être pas en cours d'exécution."
    
    # Tentative de trouver le processus manuellement
    PID=$(ps aux | grep "market_analysis_scheduler" | grep -v grep | awk '{print $2}')
    
    if [ -n "$PID" ]; then
        echo "Processus scheduler trouvé avec PID: $PID"
        echo "Arrêt en cours..."
        kill $PID
        echo "Processus arrêté"
    else
        echo "Aucun processus scheduler trouvé en cours d'exécution."
    fi
fi

echo ""
echo "Opération terminée." 