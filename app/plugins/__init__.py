"""
Module de gestion des plugins pour EVIL2ROOT Trading Bot.
Ce module permet de charger, gérer et utiliser des plugins communautaires.
"""

from app.plugins.plugin_manager import PluginManager

# Initialiser le gestionnaire de plugins
plugin_manager = PluginManager()

__all__ = ['plugin_manager'] 