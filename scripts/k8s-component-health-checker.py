#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de vérification de santé des composants dans Kubernetes
Ce script vérifie l'état des différents composants de l'application,
identifie les erreurs potentielles et les enregistre dans un système de logging.
"""

import os
import sys
import json
import logging
import datetime
import time
import socket
import requests
import subprocess
from pathlib import Path
from kubernetes import client, config
from urllib3.exceptions import InsecureRequestWarning
import traceback

# Suppression des avertissements SSL pour les environnements de développement
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# Configuration du système de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/health-checks/component-health.log', mode='a')
    ]
)
logger = logging.getLogger('component-health-checker')

class ComponentHealthChecker:
    """Classe principale pour vérifier la santé des composants de l'application"""
    
    def __init__(self, namespace="evil2root-trading"):
        """Initialisation du vérificateur de santé
        
        Args:
            namespace (str): Namespace Kubernetes à vérifier
        """
        self.namespace = namespace
        self.hostname = socket.gethostname()
        self.start_time = datetime.datetime.now()
        self.issues_found = 0
        self.components_checked = 0
        self.error_log = []
        
        # Initialisation du client Kubernetes
        try:
            # En production dans K8s
            if os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount/token'):
                config.load_incluster_config()
                logger.info("Configuration Kubernetes chargée depuis le cluster")
            else:
                # En local/dev
                config.load_kube_config()
                logger.info("Configuration Kubernetes chargée depuis kubeconfig local")
                
            self.k8s_core_api = client.CoreV1Api()
            self.k8s_apps_api = client.AppsV1Api()
            self.k8s_batch_api = client.BatchV1Api()
            
        except Exception as e:
            logger.critical(f"Erreur lors de l'initialisation du client Kubernetes: {e}")
            logger.error(traceback.format_exc())
            raise
            
        # Création du répertoire de logs si nécessaire
        Path('/var/log/health-checks').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ComponentHealthChecker initialisé pour le namespace '{namespace}'")
    
    def run_all_checks(self):
        """Lance toutes les vérifications de santé"""
        try:
            logger.info("=== DÉBUT DES VÉRIFICATIONS DE SANTÉ ===")
            
            # Vérification des pods
            self.check_pods()
            
            # Vérification des déploiements
            self.check_deployments()
            
            # Vérification des services
            self.check_services()
            
            # Vérification des endpoints
            self.check_endpoints()
            
            # Vérification des persistent volumes
            self.check_persistent_volumes()
            
            # Vérification des secrets
            self.check_secrets()
            
            # Vérification des config maps
            self.check_configmaps()
            
            # Vérification des composants spécifiques
            self.check_trading_bot()
            self.check_api_service()
            self.check_frontend()
            self.check_database()
            self.check_monitoring()
            
            # Vérification des ressources du système
            self.check_resource_usage()
            
            # Génération du rapport final
            self.generate_report()
            
            return self.issues_found == 0
            
        except Exception as e:
            logger.critical(f"Erreur lors de l'exécution des vérifications: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "global", "error": str(e), "traceback": traceback.format_exc()})
            return False
            
        finally:
            # Toujours générer un rapport, même en cas d'erreur
            self.generate_report()
    
    def check_pods(self):
        """Vérifie l'état des pods dans le namespace"""
        logger.info("Vérification des pods...")
        
        try:
            pods = self.k8s_core_api.list_namespaced_pod(namespace=self.namespace)
            self.components_checked += len(pods.items)
            
            for pod in pods.items:
                pod_name = pod.metadata.name
                pod_status = pod.status.phase
                
                if pod_status != 'Running' and pod_status != 'Succeeded':
                    self.issues_found += 1
                    error_msg = f"Pod '{pod_name}' en état '{pod_status}'"
                    logger.warning(error_msg)
                    
                    # Récupération des événements liés à ce pod
                    field_selector = f"involvedObject.name={pod_name}"
                    events = self.k8s_core_api.list_namespaced_event(
                        namespace=self.namespace,
                        field_selector=field_selector
                    )
                    
                    # Extraction des messages d'erreur des événements
                    event_messages = []
                    for event in events.items:
                        if event.type == "Warning" or event.type == "Error":
                            event_messages.append(f"{event.reason}: {event.message}")
                    
                    # Récupération des logs du pod si possible
                    pod_logs = ""
                    try:
                        if pod.status.container_statuses:
                            container_name = pod.status.container_statuses[0].name
                            pod_logs = self.k8s_core_api.read_namespaced_pod_log(
                                name=pod_name,
                                namespace=self.namespace,
                                container=container_name,
                                tail_lines=50
                            )
                    except Exception as log_error:
                        pod_logs = f"Erreur lors de la récupération des logs: {str(log_error)}"
                    
                    # Enregistrement de l'erreur
                    self.error_log.append({
                        "component": f"pod/{pod_name}",
                        "status": pod_status,
                        "error": error_msg,
                        "events": event_messages,
                        "logs": pod_logs
                    })
                
                # Vérification supplémentaire des états des conteneurs
                if pod.status.container_statuses:
                    for container in pod.status.container_statuses:
                        if not container.ready and pod_status == 'Running':
                            self.issues_found += 1
                            error_msg = f"Conteneur '{container.name}' dans le pod '{pod_name}' n'est pas prêt"
                            logger.warning(error_msg)
                            
                            # Récupération de l'état du conteneur
                            container_state = ""
                            if container.state.waiting:
                                container_state = f"Waiting: {container.state.waiting.reason} - {container.state.waiting.message}"
                            elif container.state.terminated:
                                container_state = f"Terminated: {container.state.terminated.reason} - Exit code: {container.state.terminated.exit_code}"
                                
                            # Enregistrement de l'erreur
                            self.error_log.append({
                                "component": f"container/{pod_name}/{container.name}",
                                "status": "NotReady",
                                "error": error_msg,
                                "state": container_state
                            })
            
            logger.info(f"Vérification de {len(pods.items)} pods terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des pods: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "pods", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_deployments(self):
        """Vérifie l'état des déploiements dans le namespace"""
        logger.info("Vérification des déploiements...")
        
        try:
            deployments = self.k8s_apps_api.list_namespaced_deployment(namespace=self.namespace)
            self.components_checked += len(deployments.items)
            
            for deployment in deployments.items:
                deploy_name = deployment.metadata.name
                desired_replicas = deployment.spec.replicas
                available_replicas = deployment.status.available_replicas or 0
                
                if available_replicas < desired_replicas:
                    self.issues_found += 1
                    error_msg = f"Déploiement '{deploy_name}' a {available_replicas}/{desired_replicas} replicas disponibles"
                    logger.warning(error_msg)
                    
                    # Vérification des conditions de déploiement
                    conditions = []
                    if deployment.status.conditions:
                        for condition in deployment.status.conditions:
                            if condition.status != "True":
                                conditions.append(f"{condition.type}: {condition.reason} - {condition.message}")
                    
                    # Enregistrement de l'erreur
                    self.error_log.append({
                        "component": f"deployment/{deploy_name}",
                        "status": f"{available_replicas}/{desired_replicas}",
                        "error": error_msg,
                        "conditions": conditions
                    })
            
            logger.info(f"Vérification de {len(deployments.items)} déploiements terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des déploiements: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "deployments", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_services(self):
        """Vérifie l'état des services dans le namespace"""
        logger.info("Vérification des services...")
        
        try:
            services = self.k8s_core_api.list_namespaced_service(namespace=self.namespace)
            self.components_checked += len(services.items)
            
            for service in services.items:
                service_name = service.metadata.name
                service_type = service.spec.type
                
                # Vérification des sélecteurs
                if not service.spec.selector:
                    continue  # Certains services comme headless n'ont pas de sélecteurs
                
                # Vérification que le service a des endpoints
                endpoints = self.k8s_core_api.list_namespaced_endpoints(
                    namespace=self.namespace,
                    field_selector=f"metadata.name={service_name}"
                )
                
                has_endpoints = False
                if endpoints.items and endpoints.items[0].subsets:
                    for subset in endpoints.items[0].subsets:
                        if subset.addresses:
                            has_endpoints = True
                            break
                
                if not has_endpoints:
                    self.issues_found += 1
                    error_msg = f"Service '{service_name}' n'a pas d'endpoints"
                    logger.warning(error_msg)
                    
                    # Enregistrement de l'erreur
                    self.error_log.append({
                        "component": f"service/{service_name}",
                        "type": service_type,
                        "error": error_msg,
                        "selector": service.spec.selector
                    })
            
            logger.info(f"Vérification de {len(services.items)} services terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des services: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "services", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_endpoints(self):
        """Vérifie l'état des endpoints dans le namespace"""
        logger.info("Vérification des endpoints...")
        
        try:
            endpoints = self.k8s_core_api.list_namespaced_endpoints(namespace=self.namespace)
            self.components_checked += len(endpoints.items)
            
            for endpoint in endpoints.items:
                endpoint_name = endpoint.metadata.name
                
                # Vérifier s'il y a des subsets
                if not endpoint.subsets:
                    self.issues_found += 1
                    error_msg = f"Endpoint '{endpoint_name}' n'a pas de subsets"
                    logger.warning(error_msg)
                    
                    # Enregistrement de l'erreur
                    self.error_log.append({
                        "component": f"endpoint/{endpoint_name}",
                        "error": error_msg
                    })
                    continue
                
                # Vérifier s'il y a des adresses
                has_addresses = False
                for subset in endpoint.subsets:
                    if subset.addresses:
                        has_addresses = True
                        break
                
                if not has_addresses:
                    self.issues_found += 1
                    error_msg = f"Endpoint '{endpoint_name}' n'a pas d'adresses"
                    logger.warning(error_msg)
                    
                    # Enregistrement de l'erreur
                    self.error_log.append({
                        "component": f"endpoint/{endpoint_name}",
                        "error": error_msg
                    })
            
            logger.info(f"Vérification de {len(endpoints.items)} endpoints terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des endpoints: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "endpoints", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_persistent_volumes(self):
        """Vérifie l'état des volumes persistants dans le namespace"""
        logger.info("Vérification des volumes persistants...")
        
        try:
            pvcs = self.k8s_core_api.list_namespaced_persistent_volume_claim(namespace=self.namespace)
            self.components_checked += len(pvcs.items)
            
            for pvc in pvcs.items:
                pvc_name = pvc.metadata.name
                pvc_status = pvc.status.phase
                
                if pvc_status != 'Bound':
                    self.issues_found += 1
                    error_msg = f"PVC '{pvc_name}' en état '{pvc_status}', devrait être 'Bound'"
                    logger.warning(error_msg)
                    
                    # Enregistrement de l'erreur
                    self.error_log.append({
                        "component": f"pvc/{pvc_name}",
                        "status": pvc_status,
                        "error": error_msg
                    })
            
            logger.info(f"Vérification de {len(pvcs.items)} PVCs terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des volumes persistants: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "pvcs", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_secrets(self):
        """Vérifie la présence des secrets nécessaires"""
        logger.info("Vérification des secrets...")
        
        try:
            secrets = self.k8s_core_api.list_namespaced_secret(namespace=self.namespace)
            self.components_checked += 1  # On compte comme un composant global
            
            # Liste des secrets requis (à adapter selon votre application)
            required_secrets = [
                "db-credentials",
                "api-keys",
                "trading-credentials"
            ]
            
            # Vérification des secrets requis
            found_secrets = [s.metadata.name for s in secrets.items]
            missing_secrets = []
            
            for required in required_secrets:
                found = False
                for secret_name in found_secrets:
                    if required in secret_name:
                        found = True
                        break
                
                if not found:
                    missing_secrets.append(required)
            
            if missing_secrets:
                self.issues_found += 1
                error_msg = f"Secrets manquants: {', '.join(missing_secrets)}"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "secrets",
                    "error": error_msg,
                    "missing": missing_secrets
                })
            
            logger.info("Vérification des secrets terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des secrets: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "secrets", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_configmaps(self):
        """Vérifie la présence des configmaps nécessaires"""
        logger.info("Vérification des configmaps...")
        
        try:
            configmaps = self.k8s_core_api.list_namespaced_config_map(namespace=self.namespace)
            self.components_checked += 1  # On compte comme un composant global
            
            # Liste des configmaps requis (à adapter selon votre application)
            required_configmaps = [
                "app-config",
                "trading-parameters"
            ]
            
            # Vérification des configmaps requis
            found_configmaps = [cm.metadata.name for cm in configmaps.items]
            missing_configmaps = []
            
            for required in required_configmaps:
                found = False
                for configmap_name in found_configmaps:
                    if required in configmap_name:
                        found = True
                        break
                
                if not found:
                    missing_configmaps.append(required)
            
            if missing_configmaps:
                self.issues_found += 1
                error_msg = f"ConfigMaps manquants: {', '.join(missing_configmaps)}"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "configmaps",
                    "error": error_msg,
                    "missing": missing_configmaps
                })
            
            logger.info("Vérification des configmaps terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des configmaps: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "configmaps", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_trading_bot(self):
        """Vérifie l'état spécifique du bot de trading"""
        logger.info("Vérification du bot de trading...")
        
        try:
            self.components_checked += 1
            
            # Vérification du déploiement du bot de trading
            bot_deployments = self.k8s_apps_api.list_namespaced_deployment(
                namespace=self.namespace,
                label_selector="app=trading-bot"
            )
            
            if not bot_deployments.items:
                self.issues_found += 1
                error_msg = "Aucun déploiement trouvé pour le bot de trading"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "trading-bot",
                    "error": error_msg
                })
                return
            
            # Vérification des pods du bot de trading
            bot_pods = self.k8s_core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector="app=trading-bot"
            )
            
            if not bot_pods.items:
                self.issues_found += 1
                error_msg = "Aucun pod trouvé pour le bot de trading"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "trading-bot",
                    "error": error_msg
                })
                return
            
            # Vérification des logs des pods du bot de trading pour détecter des erreurs
            for pod in bot_pods.items:
                pod_name = pod.metadata.name
                
                try:
                    # Récupération des logs récents
                    pod_logs = self.k8s_core_api.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=self.namespace,
                        tail_lines=100
                    )
                    
                    # Recherche d'erreurs dans les logs
                    error_keywords = ["ERROR", "Exception", "Traceback", "Failed", "Timeout"]
                    errors_found = []
                    
                    for line in pod_logs.splitlines():
                        for keyword in error_keywords:
                            if keyword in line:
                                errors_found.append(line)
                                break
                    
                    if errors_found:
                        self.issues_found += 1
                        error_msg = f"Erreurs détectées dans les logs du pod '{pod_name}'"
                        logger.warning(error_msg)
                        
                        # Enregistrement de l'erreur
                        self.error_log.append({
                            "component": f"trading-bot/pod/{pod_name}",
                            "error": error_msg,
                            "log_errors": errors_found[:10]  # Limiter à 10 erreurs
                        })
                
                except Exception as log_error:
                    logger.error(f"Erreur lors de la récupération des logs du pod '{pod_name}': {log_error}")
            
            logger.info("Vérification du bot de trading terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du bot de trading: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "trading-bot", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_api_service(self):
        """Vérifie l'état de l'API service"""
        logger.info("Vérification de l'API service...")
        
        try:
            self.components_checked += 1
            
            # Vérification du déploiement de l'API
            api_deployments = self.k8s_apps_api.list_namespaced_deployment(
                namespace=self.namespace,
                label_selector="app=api"
            )
            
            if not api_deployments.items:
                self.issues_found += 1
                error_msg = "Aucun déploiement trouvé pour l'API"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "api",
                    "error": error_msg
                })
                return
            
            # Vérification du service de l'API
            api_services = self.k8s_core_api.list_namespaced_service(
                namespace=self.namespace,
                label_selector="app=api"
            )
            
            if not api_services.items:
                self.issues_found += 1
                error_msg = "Aucun service trouvé pour l'API"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "api",
                    "error": error_msg
                })
                return
            
            # Essayer de se connecter à l'API
            api_service = api_services.items[0]
            api_port = None
            
            for port in api_service.spec.ports:
                if port.name == "http" or port.name == "api":
                    api_port = port.port
                    break
            
            if api_port:
                # Essai de connexion à l'API via le service DNS Kubernetes interne
                service_url = f"http://{api_service.metadata.name}.{self.namespace}.svc.cluster.local:{api_port}/health"
                
                try:
                    response = requests.get(service_url, timeout=5, verify=False)
                    
                    if response.status_code != 200:
                        self.issues_found += 1
                        error_msg = f"L'API a répondu avec le code {response.status_code} au lieu de 200"
                        logger.warning(error_msg)
                        
                        # Enregistrement de l'erreur
                        self.error_log.append({
                            "component": "api",
                            "error": error_msg,
                            "response": response.text[:200]  # Limiter la taille
                        })
                
                except requests.exceptions.RequestException as req_error:
                    self.issues_found += 1
                    error_msg = f"Impossible de se connecter à l'API: {str(req_error)}"
                    logger.warning(error_msg)
                    
                    # Enregistrement de l'erreur
                    self.error_log.append({
                        "component": "api",
                        "error": error_msg
                    })
            
            logger.info("Vérification de l'API service terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'API service: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "api", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_frontend(self):
        """Vérifie l'état du frontend"""
        logger.info("Vérification du frontend...")
        
        try:
            self.components_checked += 1
            
            # Vérification du déploiement du frontend
            frontend_deployments = self.k8s_apps_api.list_namespaced_deployment(
                namespace=self.namespace,
                label_selector="app=frontend"
            )
            
            if not frontend_deployments.items:
                self.issues_found += 1
                error_msg = "Aucun déploiement trouvé pour le frontend"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "frontend",
                    "error": error_msg
                })
                return
            
            # Vérification du service du frontend
            frontend_services = self.k8s_core_api.list_namespaced_service(
                namespace=self.namespace,
                label_selector="app=frontend"
            )
            
            if not frontend_services.items:
                self.issues_found += 1
                error_msg = "Aucun service trouvé pour le frontend"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "frontend",
                    "error": error_msg
                })
            
            logger.info("Vérification du frontend terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du frontend: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "frontend", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_database(self):
        """Vérifie l'état de la base de données"""
        logger.info("Vérification de la base de données...")
        
        try:
            self.components_checked += 1
            
            # Vérification du déploiement de la base de données
            db_deployments = self.k8s_apps_api.list_namespaced_deployment(
                namespace=self.namespace,
                label_selector="app=db"
            )
            
            # Vérification du statefulset de la base de données (si applicable)
            db_statefulsets = self.k8s_apps_api.list_namespaced_stateful_set(
                namespace=self.namespace,
                label_selector="app=db"
            )
            
            if not db_deployments.items and not db_statefulsets.items:
                self.issues_found += 1
                error_msg = "Aucun déploiement ou statefulset trouvé pour la base de données"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "database",
                    "error": error_msg
                })
                return
            
            # Vérification du service de la base de données
            db_services = self.k8s_core_api.list_namespaced_service(
                namespace=self.namespace,
                label_selector="app=db"
            )
            
            if not db_services.items:
                self.issues_found += 1
                error_msg = "Aucun service trouvé pour la base de données"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "database",
                    "error": error_msg
                })
            
            logger.info("Vérification de la base de données terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de la base de données: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "database", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_monitoring(self):
        """Vérifie l'état du système de monitoring"""
        logger.info("Vérification du système de monitoring...")
        
        try:
            self.components_checked += 1
            
            # Vérification du déploiement du système de monitoring
            monitoring_deployments = self.k8s_apps_api.list_namespaced_deployment(
                namespace=self.namespace,
                label_selector="app=monitoring"
            )
            
            if not monitoring_deployments.items:
                self.issues_found += 1
                error_msg = "Aucun déploiement trouvé pour le système de monitoring"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "monitoring",
                    "error": error_msg
                })
                return
            
            # Vérification du service du système de monitoring
            monitoring_services = self.k8s_core_api.list_namespaced_service(
                namespace=self.namespace,
                label_selector="app=monitoring"
            )
            
            if not monitoring_services.items:
                self.issues_found += 1
                error_msg = "Aucun service trouvé pour le système de monitoring"
                logger.warning(error_msg)
                
                # Enregistrement de l'erreur
                self.error_log.append({
                    "component": "monitoring",
                    "error": error_msg
                })
                return
            
            # Essayer de se connecter au service de monitoring
            monitoring_service = monitoring_services.items[0]
            monitoring_port = None
            
            for port in monitoring_service.spec.ports:
                if port.name == "http" or port.name == "web" or port.name == "prometheus":
                    monitoring_port = port.port
                    break
            
            if monitoring_port:
                # Essai de connexion au monitoring via le service DNS Kubernetes interne
                service_url = f"http://{monitoring_service.metadata.name}.{self.namespace}.svc.cluster.local:{monitoring_port}/-/healthy"
                
                try:
                    response = requests.get(service_url, timeout=5, verify=False)
                    
                    if response.status_code != 200:
                        self.issues_found += 1
                        error_msg = f"Le service de monitoring a répondu avec le code {response.status_code} au lieu de 200"
                        logger.warning(error_msg)
                        
                        # Enregistrement de l'erreur
                        self.error_log.append({
                            "component": "monitoring",
                            "error": error_msg,
                            "response": response.text[:200]  # Limiter la taille
                        })
                
                except requests.exceptions.RequestException as req_error:
                    self.issues_found += 1
                    error_msg = f"Impossible de se connecter au service de monitoring: {str(req_error)}"
                    logger.warning(error_msg)
                    
                    # Enregistrement de l'erreur
                    self.error_log.append({
                        "component": "monitoring",
                        "error": error_msg
                    })
            
            logger.info("Vérification du système de monitoring terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du système de monitoring: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "monitoring", "error": str(e), "traceback": traceback.format_exc()})
    
    def check_resource_usage(self):
        """Vérifie l'utilisation des ressources des pods"""
        logger.info("Vérification de l'utilisation des ressources...")
        
        try:
            self.components_checked += 1
            
            # Cette vérification nécessite metrics-server
            # Exécution de kubectl top pods via subprocess
            try:
                cmd = ["kubectl", "top", "pods", "-n", self.namespace]
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                
                high_resource_pods = []
                lines = output.strip().split('\n')
                
                # Ignorer la première ligne (en-tête)
                if len(lines) > 1:
                    for line in lines[1:]:
                        parts = line.split()
                        if len(parts) >= 3:
                            pod_name = parts[0]
                            cpu_usage = parts[1]
                            memory_usage = parts[2]
                            
                            # Vérifier si l'utilisation est élevée
                            # Exemple simple: si CPU > 80% ou mémoire > 80%
                            is_high_cpu = False
                            is_high_memory = False
                            
                            # Analyse de l'utilisation CPU
                            if cpu_usage.endswith('m'):
                                cpu_millicores = int(cpu_usage[:-1])
                                # Considérer > 800m comme élevé (exemple)
                                if cpu_millicores > 800:
                                    is_high_cpu = True
                            
                            # Analyse de l'utilisation mémoire
                            if memory_usage.endswith('Mi'):
                                memory_mb = int(memory_usage[:-2])
                                # Considérer > 1024Mi comme élevé (exemple)
                                if memory_mb > 1024:
                                    is_high_memory = True
                            elif memory_usage.endswith('Gi'):
                                memory_gb = float(memory_usage[:-2])
                                # Considérer > 1Gi comme élevé (exemple)
                                if memory_gb > 1:
                                    is_high_memory = True
                            
                            if is_high_cpu or is_high_memory:
                                high_resource_pods.append({
                                    "pod": pod_name,
                                    "cpu": cpu_usage,
                                    "memory": memory_usage,
                                    "high_cpu": is_high_cpu,
                                    "high_memory": is_high_memory
                                })
                
                if high_resource_pods:
                    self.issues_found += 1
                    error_msg = f"{len(high_resource_pods)} pods utilisent beaucoup de ressources"
                    logger.warning(error_msg)
                    
                    # Enregistrement de l'erreur
                    self.error_log.append({
                        "component": "resources",
                        "error": error_msg,
                        "high_resource_pods": high_resource_pods
                    })
            
            except subprocess.CalledProcessError as cmd_error:
                # Le metrics-server n'est peut-être pas disponible
                logger.warning(f"Impossible d'obtenir les métriques d'utilisation des ressources: {cmd_error}")
                
                # On ne compte pas cela comme une erreur critique
                self.error_log.append({
                    "component": "resources",
                    "warning": "Metrics server not available",
                    "details": str(cmd_error)
                })
            
            logger.info("Vérification de l'utilisation des ressources terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'utilisation des ressources: {e}")
            logger.error(traceback.format_exc())
            self.issues_found += 1
            self.error_log.append({"component": "resources", "error": str(e), "traceback": traceback.format_exc()})
    
    def generate_report(self):
        """Génère un rapport de santé complet"""
        logger.info("Génération du rapport de santé...")
        
        end_time = datetime.datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report = {
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration,
            "hostname": self.hostname,
            "namespace": self.namespace,
            "components_checked": self.components_checked,
            "issues_found": self.issues_found,
            "status": "healthy" if self.issues_found == 0 else "unhealthy",
            "issues": self.error_log
        }
        
        # Enregistrement du rapport dans un fichier JSON
        report_file = f"/var/log/health-checks/health-report-{end_time.strftime('%Y%m%d-%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Affichage du résumé
        if self.issues_found == 0:
            logger.info("=== RÉSUMÉ DU RAPPORT DE SANTÉ ===")
            logger.info(f"Statut: {report['status'].upper()}")
            logger.info(f"Composants vérifiés: {report['components_checked']}")
            logger.info(f"Problèmes détectés: {report['issues_found']}")
            logger.info(f"Durée: {report['duration_seconds']:.2f} secondes")
            logger.info(f"Rapport complet enregistré dans: {report_file}")
        else:
            logger.warning("=== RÉSUMÉ DU RAPPORT DE SANTÉ ===")
            logger.warning(f"Statut: {report['status'].upper()}")
            logger.warning(f"Composants vérifiés: {report['components_checked']}")
            logger.warning(f"Problèmes détectés: {report['issues_found']}")
            logger.warning(f"Durée: {report['duration_seconds']:.2f} secondes")
            logger.warning(f"Rapport complet enregistré dans: {report_file}")
            
            # Affichage des problèmes critiques
            logger.warning("\nProblèmes critiques détectés:")
            for i, issue in enumerate(self.error_log, 1):
                component = issue.get("component", "unknown")
                error = issue.get("error", "unknown error")
                logger.warning(f"{i}. {component}: {error}")
            
            logger.warning(f"\nConsultez le rapport complet pour plus de détails: {report_file}")
        
        return report

def main():
    """Fonction principale"""
    try:
        # Récupération du namespace depuis les variables d'environnement ou utilisation de la valeur par défaut
        namespace = os.environ.get("KUBERNETES_NAMESPACE", "evil2root-trading")
        
        logger.info(f"Démarrage de la vérification de santé des composants dans le namespace '{namespace}'")
        
        # Création et exécution du vérificateur de santé
        checker = ComponentHealthChecker(namespace=namespace)
        result = checker.run_all_checks()
        
        # Sortie avec le code approprié
        sys.exit(0 if result else 1)
        
    except Exception as e:
        logger.critical(f"Erreur critique lors de l'exécution du script: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(2)

if __name__ == "__main__":
    main() 