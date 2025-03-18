#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de diagnostic pour cluster Kubernetes DigitalOcean
Ce script vérifie l'état de votre cluster Kubernetes sur DigitalOcean
et affiche des informations détaillées sur sa configuration.
"""

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime

# Couleurs pour le formatage
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def print_header(message):
    print(f"\n{Colors.GREEN}==== {message} ===={Colors.NC}\n")

def print_info(message):
    print(f"{Colors.BLUE}INFO: {message}{Colors.NC}")

def print_warning(message):
    print(f"{Colors.YELLOW}AVERTISSEMENT: {message}{Colors.NC}")

def print_error(message):
    print(f"{Colors.RED}ERREUR: {message}{Colors.NC}")

def check_token(token):
    """Vérifie si le token DigitalOcean est valide."""
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get('https://api.digitalocean.com/v2/account', headers=headers)
    return response.status_code == 200

def get_clusters(token):
    """Récupère la liste des clusters Kubernetes sur DigitalOcean."""
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get('https://api.digitalocean.com/v2/kubernetes/clusters', headers=headers)
    
    if response.status_code != 200:
        print_error(f"Impossible de récupérer les clusters. Code: {response.status_code}")
        print_error(f"Réponse: {response.text}")
        return []
    
    return response.json().get('kubernetes_clusters', [])

def get_cluster_details(token, cluster_id):
    """Récupère les détails d'un cluster spécifique."""
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(f'https://api.digitalocean.com/v2/kubernetes/clusters/{cluster_id}', headers=headers)
    
    if response.status_code != 200:
        print_error(f"Impossible de récupérer les détails du cluster. Code: {response.status_code}")
        print_error(f"Réponse: {response.text}")
        return None
    
    return response.json().get('kubernetes_cluster')

def get_node_pools(token, cluster_id):
    """Récupère les pools de nœuds d'un cluster."""
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(
        f'https://api.digitalocean.com/v2/kubernetes/clusters/{cluster_id}/node_pools', 
        headers=headers
    )
    
    if response.status_code != 200:
        print_error(f"Impossible de récupérer les pools de nœuds. Code: {response.status_code}")
        print_error(f"Réponse: {response.text}")
        return []
    
    return response.json().get('node_pools', [])

def get_all_droplets(token):
    """Récupère la liste de tous les droplets pour référence."""
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get('https://api.digitalocean.com/v2/droplets', headers=headers)
    
    if response.status_code != 200:
        print_error(f"Impossible de récupérer les droplets. Code: {response.status_code}")
        return []
    
    return response.json().get('droplets', [])

def get_load_balancers(token):
    """Récupère la liste des load balancers."""
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get('https://api.digitalocean.com/v2/load_balancers', headers=headers)
    
    if response.status_code != 200:
        print_error(f"Impossible de récupérer les load balancers. Code: {response.status_code}")
        return []
    
    return response.json().get('load_balancers', [])

def format_timestamp(timestamp):
    """Formate un timestamp ISO en format lisible."""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime('%d/%m/%Y %H:%M:%S')
    except:
        return timestamp

def display_cluster_info(cluster):
    """Affiche les informations détaillées d'un cluster."""
    print(f"Nom: {cluster['name']}")
    print(f"ID: {cluster['id']}")
    print(f"Région: {cluster['region']}")
    print(f"Version: {cluster['version']}")
    print(f"État: {cluster['status']['state']}")
    
    if cluster['status'].get('message'):
        print(f"Message d'état: {cluster['status']['message']}")
    
    print(f"Créé le: {format_timestamp(cluster['created_at'])}")
    print(f"Mis à jour le: {format_timestamp(cluster['updated_at'])}")
    
    # Afficher le point de terminaison de l'API
    endpoint = cluster.get('endpoint')
    if endpoint:
        print(f"Point de terminaison API: {endpoint}")
    else:
        print_warning("Point de terminaison API: Non disponible")
    
    # Afficher les détails du cluster autoscaler
    auto_upgrade = cluster.get('auto_upgrade', False)
    print(f"Mise à niveau automatique: {'Activée' if auto_upgrade else 'Désactivée'}")
    
    # Afficher les fonctionnalités activées
    print("\nFonctionnalités activées:")
    for feature in cluster.get('features', []):
        print(f"- {feature}")
    
    if not cluster.get('features'):
        print("- Aucune fonctionnalité spéciale activée")

def display_node_pools(node_pools):
    """Affiche les informations sur les pools de nœuds."""
    print_header("Pools de nœuds")
    
    if not node_pools:
        print_warning("Aucun pool de nœuds trouvé.")
        return
    
    for pool in node_pools:
        print(f"Nom du pool: {pool['name']}")
        print(f"Taille: {pool['size']}")
        print(f"Nombre: {pool['count']}")
        print(f"Labels: {json.dumps(pool.get('labels', {}), indent=2)}")
        
        print("\nNœuds:")
        for node in pool.get('nodes', []):
            print(f"  - ID: {node['id']}")
            print(f"    Nom: {node['name']}")
            print(f"    État: {node['status']['state']}")
            if node['status'].get('message'):
                print(f"    Message: {node['status']['message']}")
            print(f"    Créé le: {format_timestamp(node['created_at'])}")
            print("")
        
        print("-----")

def check_load_balancers(load_balancers, cluster_id):
    """Vérifie les load balancers associés au cluster."""
    print_header("Load Balancers")
    
    cluster_lbs = []
    for lb in load_balancers:
        # Vérifier si le LB appartient au cluster (via tag kubernetes:cluster-id ou nom)
        tags = lb.get('tags', [])
        cluster_tag = f"k8s:{cluster_id}"
        
        if cluster_tag in tags or any(tag.startswith("k8s-") for tag in tags):
            cluster_lbs.append(lb)
    
    if not cluster_lbs:
        print_warning("Aucun Load Balancer associé au cluster n'a été trouvé.")
        print_info("Cela peut indiquer que les services de type LoadBalancer ne sont pas correctement déployés.")
        return
    
    for lb in cluster_lbs:
        print(f"Nom: {lb['name']}")
        print(f"ID: {lb['id']}")
        print(f"IP: {lb.get('ip')}")
        print(f"État: {lb.get('status')}")
        
        print("\nRègles de transfert:")
        for rule in lb.get('forwarding_rules', []):
            print(f"  - {rule.get('entry_protocol', 'N/A')}:{rule.get('entry_port', 'N/A')} → "
                  f"{rule.get('target_protocol', 'N/A')}:{rule.get('target_port', 'N/A')}")
        
        print("-----")

def suggest_solutions(cluster):
    """Suggère des solutions en fonction de l'état du cluster."""
    print_header("Diagnostic et suggestions")
    
    state = cluster['status']['state']
    message = cluster['status'].get('message', '')
    
    if state == 'running':
        if not cluster.get('endpoint'):
            print_warning("Le cluster est en cours d'exécution mais aucun point de terminaison API n'est disponible.")
            print_info("Suggestions:")
            print("1. Attendez quelques minutes, le point de terminaison peut prendre du temps à être provisionné.")
            print("2. Vérifiez que le contrôleur de plan est en bon état.")
        else:
            print_info("Le cluster semble être en bon état de fonctionnement.")
            print_info("Si votre application ne fonctionne pas correctement, vérifiez:")
            print("1. Les pods Kubernetes: kubectl get pods --all-namespaces")
            print("2. Les services: kubectl get services --all-namespaces")
            print("3. Les ingress: kubectl get ingress --all-namespaces")
            print("4. Les événements: kubectl get events")
    elif state == 'provisioning':
        print_warning("Le cluster est en cours de provisionnement.")
        print_info("Suggestions:")
        print("1. Le provisionnement peut prendre 5 à 10 minutes, veuillez patienter.")
        print("2. Vérifiez l'état dans quelques minutes en réexécutant ce script.")
    elif state == 'degraded':
        print_error("Le cluster est dans un état dégradé.")
        print_info("Suggestions:")
        print("1. Vérifiez le message d'erreur spécifique:", message)
        print("2. Vérifiez l'état des nœuds avec kubectl get nodes")
        print("3. Contactez le support DigitalOcean avec l'ID du cluster et le message d'erreur.")
    elif state == 'error':
        print_error("Le cluster est en état d'erreur.")
        print_info("Suggestions:")
        print("1. Message d'erreur spécifique:", message)
        print("2. Il peut être nécessaire de recréer le cluster.")
        print("3. Contactez le support DigitalOcean avec l'ID du cluster et le message d'erreur.")
    else:
        print_warning(f"État du cluster: {state}")
        print_info("Suggestions:")
        print("1. Vérifiez la documentation DigitalOcean pour plus d'informations sur cet état.")
        print("2. Si le problème persiste, contactez le support DigitalOcean.")

def main():
    parser = argparse.ArgumentParser(description='Diagnostic pour cluster Kubernetes DigitalOcean')
    parser.add_argument('-t', '--token', help='Token API DigitalOcean')
    parser.add_argument('-n', '--name', default='evil2root-trading', help='Nom du cluster (défaut: evil2root-trading)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Afficher des informations détaillées')
    args = parser.parse_args()
    
    # Récupérer le token
    token = args.token
    if not token:
        token = os.environ.get('DIGITALOCEAN_TOKEN')
    
    if not token:
        print_error("Token API DigitalOcean non spécifié.")
        print_info("Utilisez -t/--token ou définissez la variable d'environnement DIGITALOCEAN_TOKEN")
        return 1
    
    # Vérifier si le token est valide
    print_info("Vérification du token API...")
    if not check_token(token):
        print_error("Token API DigitalOcean invalide.")
        return 1
    
    # Récupérer la liste des clusters
    print_info("Récupération des clusters Kubernetes...")
    clusters = get_clusters(token)
    
    if not clusters:
        print_error("Aucun cluster Kubernetes trouvé.")
        print_info("Assurez-vous que vous avez créé un cluster et que votre token a les droits nécessaires.")
        return 1
    
    # Chercher le cluster spécifique
    target_cluster = None
    for cluster in clusters:
        if cluster['name'] == args.name:
            target_cluster = cluster
            break
    
    if not target_cluster:
        print_error(f"Cluster '{args.name}' non trouvé.")
        print_info("Clusters disponibles:")
        for cluster in clusters:
            print(f"- {cluster['name']} (ID: {cluster['id']})")
        return 1
    
    # Récupérer les détails complets du cluster
    print_info(f"Récupération des détails du cluster '{args.name}'...")
    cluster_details = get_cluster_details(token, target_cluster['id'])
    
    if not cluster_details:
        print_error("Impossible de récupérer les détails du cluster.")
        return 1
    
    # Afficher les informations du cluster
    print_header("Informations du cluster")
    display_cluster_info(cluster_details)
    
    # Récupérer et afficher les pools de nœuds
    node_pools = get_node_pools(token, target_cluster['id'])
    display_node_pools(node_pools)
    
    # Récupérer les load balancers associés
    if args.verbose:
        print_info("Récupération des load balancers...")
        load_balancers = get_load_balancers(token)
        check_load_balancers(load_balancers, target_cluster['id'])
    
    # Suggérer des solutions
    suggest_solutions(cluster_details)
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 