"""
Gestionnaire de plugins pour EVIL2ROOT Trading Bot.
Permet de charger, activer et désactiver des plugins dynamiquement.
"""

import os
import sys
import json
import logging
import importlib
import inspect
import pkgutil
import shutil
import zipfile
import tempfile
from typing import Dict, List, Any, Optional, Type, Callable, Set
from pathlib import Path

from app.plugins.plugin_base import PluginBase

# Configuration du logger
logger = logging.getLogger("plugins")

class PluginManager:
    """
    Gestionnaire de plugins pour EVIL2ROOT Trading Bot.
    Permet le chargement dynamique, l'activation et la désactivation des plugins.
    """
    
    def __init__(self):
        """Initialise le gestionnaire de plugins"""
        # Chemin du dossier des plugins installés
        self.plugins_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "installed")
        self.plugins_temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        self.plugins_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins.json")
        
        # Créer les dossiers s'ils n'existent pas
        os.makedirs(self.plugins_dir, exist_ok=True)
        os.makedirs(self.plugins_temp_dir, exist_ok=True)
        
        # Dictionnaire des plugins chargés
        self.plugins: Dict[str, PluginBase] = {}
        
        # Dictionnaire des callbacks par type
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Chargement des métadonnées des plugins
        self.plugins_metadata: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.plugins_db_path):
            try:
                with open(self.plugins_db_path, 'r') as f:
                    self.plugins_metadata = json.load(f)
            except Exception as e:
                logger.error(f"Erreur lors du chargement des métadonnées des plugins: {e}")
                self.plugins_metadata = {}
        
        # Ajouter les répertoires de plugins au chemin de recherche Python
        if self.plugins_dir not in sys.path:
            sys.path.append(self.plugins_dir)
    
    def discover_plugins(self) -> List[Dict[str, Any]]:
        """
        Découvre et retourne la liste de tous les plugins disponibles.
        
        Returns:
            Liste de dictionnaires contenant les informations des plugins
        """
        plugins_info = []
        
        # Lister les modules dans le dossier des plugins
        for finder, name, ispkg in pkgutil.iter_modules([self.plugins_dir]):
            if ispkg:  # Ne considérer que les packages
                try:
                    # Charger le module
                    module = importlib.import_module(f"{name}")
                    
                    # Trouver toutes les classes qui héritent de PluginBase
                    for attribute_name in dir(module):
                        attribute = getattr(module, attribute_name)
                        if (inspect.isclass(attribute) and 
                            issubclass(attribute, PluginBase) and 
                            attribute is not PluginBase):
                            
                            # Extraire les métadonnées du plugin
                            plugin_info = {
                                "id": attribute.plugin_id,
                                "name": attribute.plugin_name,
                                "description": attribute.plugin_description,
                                "version": attribute.plugin_version,
                                "author": attribute.plugin_author,
                                "module": name,
                                "class": attribute_name,
                                "enabled": self.is_plugin_enabled(attribute.plugin_id)
                            }
                            plugins_info.append(plugin_info)
                
                except Exception as e:
                    logger.error(f"Erreur lors de la découverte du plugin {name}: {e}")
        
        return plugins_info
    
    def load_plugin(self, plugin_id: str) -> bool:
        """
        Charge un plugin spécifique par son ID.
        
        Args:
            plugin_id: L'identifiant unique du plugin
            
        Returns:
            True si le chargement a réussi, False sinon
        """
        # Vérifier si le plugin est déjà chargé
        if plugin_id in self.plugins:
            logger.info(f"Le plugin {plugin_id} est déjà chargé")
            return True
        
        # Trouver les informations du plugin
        plugin_info = None
        for plugin in self.discover_plugins():
            if plugin["id"] == plugin_id:
                plugin_info = plugin
                break
        
        if not plugin_info:
            logger.error(f"Plugin {plugin_id} non trouvé")
            return False
        
        try:
            # Charger le module
            module = importlib.import_module(f"{plugin_info['module']}")
            
            # Obtenir la classe du plugin
            plugin_class = getattr(module, plugin_info["class"])
            
            # Instancier le plugin
            plugin_instance = plugin_class()
            
            # Ajouter le plugin à la liste des plugins chargés
            self.plugins[plugin_id] = plugin_instance
            
            # Marquer le plugin comme chargé dans les métadonnées
            if plugin_id not in self.plugins_metadata:
                self.plugins_metadata[plugin_id] = {}
            
            self.plugins_metadata[plugin_id].update({
                "loaded": True,
                "info": plugin_info
            })
            self._save_metadata()
            
            logger.info(f"Plugin {plugin_id} chargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du plugin {plugin_id}: {e}")
            return False
    
    def unload_plugin(self, plugin_id: str) -> bool:
        """
        Décharge un plugin spécifique.
        
        Args:
            plugin_id: L'identifiant unique du plugin
            
        Returns:
            True si le déchargement a réussi, False sinon
        """
        if plugin_id not in self.plugins:
            logger.warning(f"Le plugin {plugin_id} n'est pas chargé")
            return False
        
        try:
            # Obtenir l'instance du plugin
            plugin = self.plugins[plugin_id]
            
            # Appeler la méthode de déchargement du plugin
            plugin.on_unload()
            
            # Retirer le plugin de la liste des plugins chargés
            del self.plugins[plugin_id]
            
            # Mise à jour des métadonnées
            if plugin_id in self.plugins_metadata:
                self.plugins_metadata[plugin_id]["loaded"] = False
                self._save_metadata()
            
            logger.info(f"Plugin {plugin_id} déchargé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du déchargement du plugin {plugin_id}: {e}")
            return False
    
    def enable_plugin(self, plugin_id: str) -> bool:
        """
        Active un plugin spécifique.
        
        Args:
            plugin_id: L'identifiant unique du plugin
            
        Returns:
            True si l'activation a réussi, False sinon
        """
        # Charger le plugin s'il n'est pas déjà chargé
        if plugin_id not in self.plugins:
            if not self.load_plugin(plugin_id):
                return False
        
        try:
            plugin = self.plugins[plugin_id]
            
            # Initialiser le plugin
            plugin.on_enable()
            
            # Mise à jour des métadonnées
            if plugin_id not in self.plugins_metadata:
                self.plugins_metadata[plugin_id] = {}
                
            self.plugins_metadata[plugin_id]["enabled"] = True
            self._save_metadata()
            
            logger.info(f"Plugin {plugin_id} activé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'activation du plugin {plugin_id}: {e}")
            return False
    
    def disable_plugin(self, plugin_id: str) -> bool:
        """
        Désactive un plugin spécifique.
        
        Args:
            plugin_id: L'identifiant unique du plugin
            
        Returns:
            True si la désactivation a réussi, False sinon
        """
        if plugin_id not in self.plugins:
            logger.warning(f"Le plugin {plugin_id} n'est pas chargé")
            return False
        
        try:
            plugin = self.plugins[plugin_id]
            
            # Désactiver le plugin
            plugin.on_disable()
            
            # Mise à jour des métadonnées
            if plugin_id in self.plugins_metadata:
                self.plugins_metadata[plugin_id]["enabled"] = False
                self._save_metadata()
            
            logger.info(f"Plugin {plugin_id} désactivé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la désactivation du plugin {plugin_id}: {e}")
            return False
    
    def is_plugin_enabled(self, plugin_id: str) -> bool:
        """
        Vérifie si un plugin est activé.
        
        Args:
            plugin_id: L'identifiant unique du plugin
            
        Returns:
            True si le plugin est activé, False sinon
        """
        if plugin_id in self.plugins_metadata:
            return self.plugins_metadata[plugin_id].get("enabled", False)
        return False
    
    def install_plugin(self, zip_path: str) -> Optional[str]:
        """
        Installe un plugin à partir d'un fichier ZIP.
        
        Args:
            zip_path: Chemin vers le fichier ZIP du plugin
            
        Returns:
            L'ID du plugin installé en cas de succès, None sinon
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extraire le ZIP dans un dossier temporaire
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Vérifier si le fichier manifest.json existe
                manifest_path = os.path.join(temp_dir, "manifest.json")
                if not os.path.exists(manifest_path):
                    logger.error("Le fichier manifest.json est manquant dans le plugin")
                    return None
                
                # Charger le manifest
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                # Vérifier les champs obligatoires
                required_fields = ["id", "name", "version", "module"]
                for field in required_fields:
                    if field not in manifest:
                        logger.error(f"Champ obligatoire manquant dans le manifest: {field}")
                        return None
                
                plugin_id = manifest["id"]
                module_name = manifest["module"]
                
                # Vérifier si le module existe dans le dossier extrait
                module_path = os.path.join(temp_dir, module_name)
                if not os.path.exists(module_path) or not os.path.isdir(module_path):
                    logger.error(f"Le module {module_name} n'existe pas dans le plugin")
                    return None
                
                # Créer le dossier de destination
                plugin_dest = os.path.join(self.plugins_dir, module_name)
                
                # Supprimer l'ancienne version si elle existe
                if os.path.exists(plugin_dest):
                    shutil.rmtree(plugin_dest)
                
                # Copier le module
                shutil.copytree(module_path, plugin_dest)
                
                # Enregistrer les métadonnées
                if plugin_id not in self.plugins_metadata:
                    self.plugins_metadata[plugin_id] = {}
                
                self.plugins_metadata[plugin_id].update({
                    "installed": True,
                    "version": manifest["version"],
                    "name": manifest["name"],
                    "description": manifest.get("description", ""),
                    "author": manifest.get("author", ""),
                    "module": module_name,
                    "install_date": str(import_datetime().now())
                })
                self._save_metadata()
                
                logger.info(f"Plugin {plugin_id} installé avec succès")
                return plugin_id
                
        except Exception as e:
            logger.error(f"Erreur lors de l'installation du plugin: {e}")
            return None
    
    def uninstall_plugin(self, plugin_id: str) -> bool:
        """
        Désinstalle un plugin.
        
        Args:
            plugin_id: L'identifiant unique du plugin
            
        Returns:
            True si la désinstallation a réussi, False sinon
        """
        if plugin_id not in self.plugins_metadata:
            logger.warning(f"Le plugin {plugin_id} n'est pas installé")
            return False
        
        try:
            # Décharger le plugin s'il est chargé
            if plugin_id in self.plugins:
                self.unload_plugin(plugin_id)
            
            # Obtenir le nom du module
            module_name = self.plugins_metadata[plugin_id].get("module")
            if not module_name:
                logger.error(f"Nom de module manquant pour le plugin {plugin_id}")
                return False
            
            # Chemin du module
            module_path = os.path.join(self.plugins_dir, module_name)
            
            # Supprimer le dossier du module
            if os.path.exists(module_path):
                shutil.rmtree(module_path)
            
            # Supprimer les métadonnées
            if plugin_id in self.plugins_metadata:
                del self.plugins_metadata[plugin_id]
                self._save_metadata()
            
            logger.info(f"Plugin {plugin_id} désinstallé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la désinstallation du plugin {plugin_id}: {e}")
            return False
    
    def get_installed_plugins(self) -> List[Dict[str, Any]]:
        """
        Récupère la liste des plugins installés avec leurs métadonnées.
        
        Returns:
            Liste des plugins installés
        """
        installed_plugins = []
        
        for plugin_id, metadata in self.plugins_metadata.items():
            if metadata.get("installed", False):
                # Compléter avec les informations dynamiques (chargé, activé)
                metadata.update({
                    "id": plugin_id,
                    "loaded": plugin_id in self.plugins,
                    "enabled": self.is_plugin_enabled(plugin_id)
                })
                installed_plugins.append(metadata)
        
        return installed_plugins
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Enregistre une fonction de callback pour un type d'événement donné.
        
        Args:
            event_type: Le type d'événement (ex: 'before_trade', 'after_analysis')
            callback: La fonction à appeler lors de l'événement
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback)
        logger.debug(f"Callback enregistré pour l'événement {event_type}")
    
    def unregister_callback(self, event_type: str, callback: Callable) -> bool:
        """
        Supprime une fonction de callback pour un type d'événement donné.
        
        Args:
            event_type: Le type d'événement
            callback: La fonction callback à supprimer
            
        Returns:
            True si le callback a été supprimé, False sinon
        """
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
            logger.debug(f"Callback supprimé pour l'événement {event_type}")
            return True
        return False
    
    def trigger_event(self, event_type: str, **kwargs) -> List[Any]:
        """
        Déclenche un événement et appelle tous les callbacks associés.
        
        Args:
            event_type: Le type d'événement à déclencher
            **kwargs: Les paramètres à passer aux callbacks
            
        Returns:
            Liste des résultats des callbacks
        """
        results = []
        
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    result = callback(**kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Erreur lors de l'exécution du callback pour {event_type}: {e}")
        
        return results
    
    def load_all_enabled_plugins(self) -> None:
        """Charge et active tous les plugins marqués comme activés"""
        for plugin_id, metadata in self.plugins_metadata.items():
            if metadata.get("enabled", False):
                try:
                    self.load_plugin(plugin_id)
                    self.enable_plugin(plugin_id)
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du plugin {plugin_id}: {e}")
    
    def _save_metadata(self) -> None:
        """Sauvegarde les métadonnées des plugins dans le fichier JSON"""
        try:
            with open(self.plugins_db_path, 'w') as f:
                json.dump(self.plugins_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métadonnées des plugins: {e}")

# Fonction utilitaire pour importer datetime uniquement lors de son utilisation
def import_datetime():
    """Importe datetime uniquement lors de son utilisation"""
    from datetime import datetime
    return datetime 