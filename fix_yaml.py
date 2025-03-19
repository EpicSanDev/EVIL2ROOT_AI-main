#!/usr/bin/env python3

import yaml
import sys
import os

def fix_yaml_file(file_path):
    """Corrige un fichier YAML en le parsant et en l'écrivant correctement."""
    try:
        # Lire le contenu du fichier
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Parser le YAML
        data = yaml.safe_load(content)
        
        # Écrire le YAML corrigé
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
            
        print(f"Fichier {file_path} corrigé avec succès.")
        return True
    except Exception as e:
        print(f"Erreur lors de la correction du fichier {file_path}: {str(e)}")
        return False

def fix_deployment_files():
    """Corrige les fichiers de déploiement Kubernetes."""
    deployment_dir = "kubernetes/deployments"
    files_to_fix = [
        f"{deployment_dir}/trading-bot-web.yaml",
        f"{deployment_dir}/analysis-bot.yaml",
        f"{deployment_dir}/market-scheduler.yaml"
    ]
    
    success = True
    for file_path in files_to_fix:
        if not fix_yaml_file(file_path):
            success = False
            
    return success

def update_images_and_resources():
    """Met à jour les images et les ressources dans les fichiers corrigés."""
    # Configuration commune
    image_tag = "registry.digitalocean.com/evil2root-registry/evil2root-ai:e2ed59c21fb15657139e5135f96cfe695b82b53e"
    deployment_dir = "kubernetes/deployments"
    files = [
        f"{deployment_dir}/trading-bot-web.yaml",
        f"{deployment_dir}/analysis-bot.yaml",
        f"{deployment_dir}/market-scheduler.yaml"
    ]
    
    for file_path in files:
        try:
            # Lire le YAML
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
                
            # Mise à jour de l'image
            data['spec']['template']['spec']['containers'][0]['image'] = image_tag
            
            # Mise à jour des ressources mémoire
            resources = data['spec']['template']['spec']['containers'][0]['resources']
            resources['requests']['memory'] = "512Mi"
            resources['limits']['memory'] = "1Gi"
            
            # Mise à jour des réplicas
            if "trading-bot-web.yaml" in file_path or "analysis-bot.yaml" in file_path:
                data['spec']['replicas'] = 2
            elif "market-scheduler.yaml" in file_path:
                data['spec']['replicas'] = 1
                
            # Mise à jour de REDIS_PORT
            env_vars = data['spec']['template']['spec']['containers'][0]['env']
            redis_port_found = False
            
            for var in env_vars:
                if var['name'] == 'REDIS_PORT':
                    var['value'] = "6379"
                    redis_port_found = True
                    break
                    
            if not redis_port_found and "REDIS_HOST" in [v['name'] for v in env_vars]:
                # Ajouter REDIS_PORT s'il n'existe pas
                env_vars.append({
                    'name': 'REDIS_PORT',
                    'value': "6379"
                })
            
            # Écrire le YAML mis à jour
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
                
            print(f"Fichier {file_path} mis à jour avec succès.")
        except Exception as e:
            print(f"Erreur lors de la mise à jour du fichier {file_path}: {str(e)}")
            return False
            
    return True

if __name__ == "__main__":
    print("Correction des fichiers YAML de déploiement...")
    
    # Sauvegarder les fichiers originaux
    os.system("mkdir -p tmp_yaml")
    os.system("cp kubernetes/deployments/*.yaml tmp_yaml/")
    
    # Corriger la structure YAML
    if fix_deployment_files():
        # Mettre à jour les images et ressources
        if update_images_and_resources():
            print("Tous les fichiers ont été corrigés et mis à jour avec succès.")
            sys.exit(0)
    
    print("Restauration des fichiers originaux depuis les sauvegardes...")
    os.system("cp tmp_yaml/*.yaml kubernetes/deployments/")
    sys.exit(1) 