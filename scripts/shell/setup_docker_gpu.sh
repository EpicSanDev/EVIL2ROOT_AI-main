#!/bin/bash

# Script pour configurer l'environnement Docker avec support GPU
# Ce script détecte la présence d'un GPU NVIDIA et génère un docker-compose.gpu.yml approprié

# Fonction pour afficher un message en couleur
print_color() {
    COLOR=$1
    MESSAGE=$2
    case $COLOR in
        "red") echo -e "\033[0;31m$MESSAGE\033[0m" ;;
        "green") echo -e "\033[0;32m$MESSAGE\033[0m" ;;
        "yellow") echo -e "\033[0;33m$MESSAGE\033[0m" ;;
        "blue") echo -e "\033[0;34m$MESSAGE\033[0m" ;;
        *) echo "$MESSAGE" ;;
    esac
}

# Fonction pour détecter la présence d'un GPU NVIDIA
detect_nvidia_gpu() {
    print_color blue "Détection des GPUs NVIDIA..."
    
    # Méthode 1: Utilisation de nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        print_color blue "nvidia-smi trouvé, vérification des GPUs..."
        if nvidia-smi -L | grep -q "GPU"; then
            print_color green "GPU NVIDIA détecté via nvidia-smi:"
            nvidia-smi -L
            echo ""
            return 0
        else
            print_color yellow "nvidia-smi est installé mais aucun GPU détecté."
        fi
    else
        print_color yellow "nvidia-smi n'est pas installé sur ce système."
    fi
    
    # Méthode 2: Vérification des périphériques PCI (Linux uniquement)
    if [ -f /proc/driver/nvidia/version ]; then
        print_color green "Driver NVIDIA détecté via /proc/driver/nvidia/version"
        cat /proc/driver/nvidia/version
        echo ""
        return 0
    fi
    
    if command -v lspci &> /dev/null; then
        if lspci | grep -i nvidia | grep -i vga; then
            print_color green "GPU NVIDIA détecté via lspci"
            return 0
        fi
    fi
    
    print_color red "Aucun GPU NVIDIA détecté sur ce système."
    return 1
}

# Fonction pour vérifier si Docker est configuré pour NVIDIA
check_docker_nvidia() {
    print_color blue "Vérification de la configuration Docker pour NVIDIA..."
    
    # Vérifier la présence de nvidia-docker2 ou nvidia-container-toolkit
    if docker info 2>/dev/null | grep -i "runtimes" | grep -q "nvidia"; then
        print_color green "Docker est configuré avec le runtime NVIDIA."
        return 0
    else
        print_color yellow "Le runtime NVIDIA n'est pas configuré dans Docker."
        
        # Vérifier si le plugin est installé mais pas configuré
        if [ -f /usr/bin/nvidia-container-toolkit ] || [ -f /usr/bin/nvidia-docker ]; then
            print_color yellow "Les outils NVIDIA Docker sont installés mais pas configurés correctement."
            print_color yellow "Exécutez 'sudo systemctl restart docker' pour appliquer les modifications."
        else
            print_color yellow "Les outils NVIDIA Docker ne sont pas installés."
            print_color yellow "Pour les installer, exécutez 'sudo ./scripts/shell/install_nvidia_docker.sh'"
        fi
        
        return 1
    fi
}

# Fonction pour générer le fichier docker-compose.gpu.yml
generate_docker_compose_gpu() {
    print_color blue "Génération du fichier docker-compose.gpu.yml avec support GPU..."
    
    # Créer une copie du fichier docker-compose.yml d'origine
    cp docker-compose.yml docker-compose.gpu.yml
    
    # Ajouter la configuration runtime: nvidia pour les services qui utilisent GPU
    sed -i.bak 's/    environment:/    runtime: nvidia\n    environment:/g' docker-compose.gpu.yml
    
    # Ajouter les variables d'environnement NVIDIA aux services qui utilisent GPU
    sed -i.bak '/USE_GPU=\${USE_GPU:-true}/a\      - NVIDIA_VISIBLE_DEVICES=all\n      - TF_FORCE_GPU_ALLOW_GROWTH=true' docker-compose.gpu.yml
    
    # Supprimer les fichiers de backup
    rm -f docker-compose.gpu.yml.bak
    
    print_color green "Fichier docker-compose.gpu.yml généré avec succès."
}

# Fonction pour créer un script de lancement
create_launch_script() {
    print_color blue "Création du script de lancement avec détection GPU automatique..."
    
    cat > start_trading_bot.sh << 'EOF'
#!/bin/bash

# Script de lancement du trading bot avec détection automatique de GPU

# Fonction pour afficher un message en couleur
print_color() {
    COLOR=$1
    MESSAGE=$2
    case $COLOR in
        "red") echo -e "\033[0;31m$MESSAGE\033[0m" ;;
        "green") echo -e "\033[0;32m$MESSAGE\033[0m" ;;
        "yellow") echo -e "\033[0;33m$MESSAGE\033[0m" ;;
        "blue") echo -e "\033[0;34m$MESSAGE\033[0m" ;;
        *) echo "$MESSAGE" ;;
    esac
}

# Vérifier si le GPU est disponible
if command -v nvidia-smi &> /dev/null && nvidia-smi -L | grep -q "GPU"; then
    print_color green "GPU NVIDIA détecté. Utilisation de docker-compose.gpu.yml"
    export USE_GPU=true
    docker-compose -f docker-compose.gpu.yml up -d
else
    print_color yellow "Aucun GPU NVIDIA détecté. Utilisation de docker-compose.yml standard (CPU)"
    export USE_GPU=false
    docker-compose up -d
fi

print_color blue "Services démarrés. Utilisez 'docker-compose logs -f' pour voir les journaux."
EOF
    
    chmod +x start_trading_bot.sh
    print_color green "Script de lancement 'start_trading_bot.sh' créé avec succès."
}

# Fonction principale
main() {
    print_color blue "=== Configuration Docker pour GPU EVIL2ROOT Trading Bot ==="
    echo ""
    
    # Vérifier si on est à la racine du projet (présence de docker-compose.yml)
    if [ ! -f docker-compose.yml ]; then
        print_color red "Erreur: docker-compose.yml non trouvé."
        print_color red "Ce script doit être exécuté depuis la racine du projet."
        print_color yellow "Utilisation: ./scripts/shell/setup_docker_gpu.sh"
        exit 1
    fi
    
    # Vérifier si un GPU NVIDIA est disponible
    detect_nvidia_gpu
    GPU_DETECTED=$?
    
    # Vérifier si Docker est configuré pour NVIDIA
    check_docker_nvidia
    DOCKER_NVIDIA_CONFIGURED=$?
    
    echo ""
    
    # Générer les fichiers appropriés
    if [ $GPU_DETECTED -eq 0 ] && [ $DOCKER_NVIDIA_CONFIGURED -eq 0 ]; then
        print_color green "GPU détecté et Docker correctement configuré."
        generate_docker_compose_gpu
        create_launch_script
        
        print_color green "=== Configuration terminée avec succès ==="
        print_color green "Pour démarrer les services avec GPU, exécutez:"
        print_color blue "  ./start_trading_bot.sh"
        print_color green "Ou directement:"
        print_color blue "  docker-compose -f docker-compose.gpu.yml up -d"
    else
        if [ $GPU_DETECTED -eq 0 ]; then
            print_color yellow "GPU détecté mais Docker n'est pas correctement configuré pour NVIDIA."
            print_color yellow "Configurez Docker avec le runtime NVIDIA puis réessayez."
            print_color yellow "Vous pouvez exécuter: ./scripts/shell/install_nvidia_docker.sh"
        else
            print_color yellow "Aucun GPU NVIDIA détecté. Configuration pour mode CPU uniquement."
            print_color yellow "Pour démarrer les services en mode CPU, exécutez:"
            print_color blue "  docker-compose up -d"
            
            # Créer quand même le script de lancement qui fera la détection automatiquement
            create_launch_script
        fi
    fi
}

# Lancer la fonction principale
main 