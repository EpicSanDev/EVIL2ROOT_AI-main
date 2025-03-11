"""
Classe de base pour les plugins EVIL2ROOT Trading Bot.
Tous les plugins doivent hériter de cette classe.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class PluginBase(ABC):
    """
    Classe de base pour tous les plugins.
    Les plugins doivent implémenter les méthodes requises et définir les attributs de classe.
    """
    
    # Attributs de classe obligatoires à définir dans les plugins
    plugin_id = "base_plugin"  # Identifiant unique du plugin
    plugin_name = "Plugin de base"  # Nom lisible du plugin
    plugin_description = "Plugin de base pour EVIL2ROOT Trading Bot"  # Description
    plugin_version = "0.1.0"  # Version du plugin (format semver)
    plugin_author = "Inconnu"  # Auteur du plugin
    
    def __init__(self):
        """Initialise le plugin"""
        self.settings = {}  # Paramètres du plugin
        self.enabled = False  # État d'activation
        self.dependencies = {}  # Dépendances du plugin
    
    @abstractmethod
    def on_enable(self) -> None:
        """
        Appelé lorsque le plugin est activé.
        C'est ici que le plugin doit s'enregistrer pour les événements et initialiser ses ressources.
        """
        pass
    
    @abstractmethod
    def on_disable(self) -> None:
        """
        Appelé lorsque le plugin est désactivé.
        C'est ici que le plugin doit se désinscrire des événements et libérer ses ressources.
        """
        pass
    
    def on_unload(self) -> None:
        """
        Appelé lorsque le plugin est déchargé.
        Par défaut, appelle on_disable() puis effectue des nettoyages supplémentaires.
        """
        self.on_disable()
    
    def setup(self, settings: Dict[str, Any]) -> None:
        """
        Configure le plugin avec les paramètres spécifiés.
        
        Args:
            settings: Dictionnaire des paramètres du plugin
        """
        self.settings.update(settings)
    
    def get_settings(self) -> Dict[str, Any]:
        """
        Récupère les paramètres actuels du plugin.
        
        Returns:
            Dictionnaire des paramètres actuels
        """
        return self.settings
    
    def get_default_settings(self) -> Dict[str, Any]:
        """
        Récupère les paramètres par défaut du plugin.
        Les plugins doivent surcharger cette méthode pour définir leurs paramètres par défaut.
        
        Returns:
            Dictionnaire des paramètres par défaut
        """
        return {}
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Récupère un paramètre spécifique du plugin.
        
        Args:
            key: Clé du paramètre
            default: Valeur par défaut si le paramètre n'existe pas
            
        Returns:
            Valeur du paramètre ou valeur par défaut
        """
        return self.settings.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> None:
        """
        Définit un paramètre spécifique du plugin.
        
        Args:
            key: Clé du paramètre
            value: Nouvelle valeur du paramètre
        """
        self.settings[key] = value
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Récupère les métadonnées du plugin.
        
        Returns:
            Dictionnaire des métadonnées
        """
        return {
            "id": self.plugin_id,
            "name": self.plugin_name,
            "description": self.plugin_description,
            "version": self.plugin_version,
            "author": self.plugin_author,
            "settings": self.get_settings(),
            "default_settings": self.get_default_settings(),
            "enabled": self.enabled
        }
    
    def get_ui_components(self) -> List[Dict[str, Any]]:
        """
        Récupère les composants d'interface utilisateur fournis par le plugin.
        Les plugins peuvent surcharger cette méthode pour définir des composants UI.
        
        Returns:
            Liste de composants UI (widgets, pages, etc.)
        """
        return []
    
    def validate_dependencies(self) -> Dict[str, bool]:
        """
        Vérifie si toutes les dépendances du plugin sont satisfaites.
        
        Returns:
            Dictionnaire des dépendances avec leur état (satisfaite ou non)
        """
        result = {}
        for dep_name, dep_version in self.dependencies.items():
            # TODO: Implémentation réelle de la vérification des dépendances
            result[dep_name] = True
        return result
    
    def register_event_handlers(self, plugin_manager) -> None:
        """
        Enregistre les gestionnaires d'événements du plugin.
        Cette méthode est appelée automatiquement lors de l'activation.
        Les plugins doivent surcharger cette méthode pour s'enregistrer aux événements.
        
        Args:
            plugin_manager: Le gestionnaire de plugins
        """
        pass 