#!/bin/bash

# Script d'installation de NVIDIA Container Toolkit pour utiliser le GPU avec Docker
# Pour RTX 2070 SUPER

# Vérifier si l'utilisateur est root ou a des droits sudo
if [ "$EUID" -ne 0 ]; then
  echo "Ce script nécessite des droits d'administrateur."
  echo "Veuillez l'exécuter avec sudo: sudo $0"
  exit 1
fi

echo "=== Installation de NVIDIA Container Toolkit ==="
echo "Ce script va configurer Docker pour utiliser votre RTX 2070 SUPER"
echo ""

# Vérifier que NVIDIA driver est installé
if ! command -v nvidia-smi &> /dev/null; then
  echo "ERREUR: Les pilotes NVIDIA ne semblent pas être installés."
  echo "Veuillez installer les pilotes NVIDIA avant de continuer."
  exit 1
fi

# Afficher les informations GPU
echo "=== Informations GPU détectées ==="
nvidia-smi -L
echo ""
nvidia-smi
echo ""

# Détection du système d'exploitation
if [ -f /etc/os-release ]; then
  . /etc/os-release
  OS=$NAME
  VER=$VERSION_ID
else
  echo "Impossible de détecter le système d'exploitation."
  exit 1
fi

echo "Système détecté: $OS $VER"

# Installation selon la distribution
if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
  # Pour Ubuntu/Debian
  echo "Installation pour Ubuntu/Debian..."
  
  # Ajout des dépôts
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list
  
  # Installation
  apt-get update
  apt-get install -y nvidia-docker2
  
elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"RHEL"* ]]; then
  # Pour CentOS/RHEL
  echo "Installation pour CentOS/RHEL..."
  
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
    tee /etc/yum.repos.d/nvidia-docker.repo
    
  yum install -y nvidia-docker2
  
else
  echo "Distribution non supportée: $OS"
  echo "Veuillez installer NVIDIA Container Toolkit manuellement."
  exit 1
fi

# Redémarrage du service Docker
systemctl restart docker

# Vérification de l'installation
echo ""
echo "=== Vérification de l'installation ==="
if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi; then
  echo ""
  echo "=== INSTALLATION RÉUSSIE ==="
  echo "Votre RTX 2070 SUPER est maintenant configurée pour être utilisée avec Docker."
  echo "Vous pouvez maintenant exécuter vos conteneurs Docker avec l'option --gpus all"
  echo "Par exemple: docker run --gpus all evil2root/trading-bot:latest"
else
  echo ""
  echo "=== ERREUR ==="
  echo "L'installation a échoué. Veuillez vérifier les messages d'erreur ci-dessus."
fi 