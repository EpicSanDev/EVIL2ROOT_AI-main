import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import copy
import pickle
import warnings
from pathlib import Path

from .trade_journal import TradeJournal
from .error_detector import ErrorDetector
from .performance_analyzer import PerformanceAnalyzer

class ModelAdjuster:
    """
    Ajuste automatiquement les paramètres des modèles de trading
    en fonction des erreurs détectées et des analyses de performance.
    Cette classe est responsable de l'auto-apprentissage des modèles.
    """
    
    def __init__(self, 
                trade_journal: TradeJournal,
                error_detector: ErrorDetector,
                performance_analyzer: PerformanceAnalyzer,
                models_dir: str = "saved_models",
                adjustment_history_path: str = "data/adjustment_history.json"):
        """
        Initialise l'ajusteur de modèles.
        
        Args:
            trade_journal: Instance de TradeJournal pour accéder aux données
            error_detector: Instance d'ErrorDetector pour analyser les erreurs
            performance_analyzer: Instance de PerformanceAnalyzer pour analyser les performances
            models_dir: Répertoire où sont stockés les modèles
            adjustment_history_path: Chemin vers le fichier d'historique des ajustements
        """
        self.trade_journal = trade_journal
        self.error_detector = error_detector
        self.performance_analyzer = performance_analyzer
        self.models_dir = models_dir
        self.adjustment_history_path = adjustment_history_path
        self.logger = logging.getLogger(__name__)
        
        # Initialiser l'historique des ajustements
        self._init_adjustment_history()
        
        # Dictionnaire pour stocker les modèles chargés
        self.loaded_models = {}
        
        # Créer le répertoire des modèles s'il n'existe pas
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
    
    def _init_adjustment_history(self):
        """Initialise ou charge l'historique des ajustements des modèles"""
        try:
            if os.path.exists(self.adjustment_history_path):
                with open(self.adjustment_history_path, 'r') as f:
                    self.adjustment_history = json.load(f)
            else:
                # Créer le répertoire parent si nécessaire
                parent_dir = os.path.dirname(self.adjustment_history_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)
                
                # Initialiser un historique vide
                self.adjustment_history = {
                    "models": {},
                    "global_adjustments": []
                }
                # Sauvegarder l'historique initial
                self._save_adjustment_history()
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation de l'historique des ajustements: {e}")
            # Créer un historique vide en cas d'erreur
            self.adjustment_history = {
                "models": {},
                "global_adjustments": []
            }
    
    def _save_adjustment_history(self):
        """Sauvegarde l'historique des ajustements dans un fichier JSON"""
        try:
            with open(self.adjustment_history_path, 'w') as f:
                json.dump(self.adjustment_history, f, indent=4)
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de l'historique des ajustements: {e}")
            
    def analyze_and_adjust_models(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyse les performances récentes et ajuste les modèles en conséquence.
        
        Args:
            days: Nombre de jours d'historique à analyser
            
        Returns:
            Dictionnaire contenant les résultats des ajustements
        """
        self.logger.info(f"Démarrage de l'analyse et de l'ajustement des modèles sur {days} jours")
        
        # 1. Analyser les erreurs récentes
        error_analysis = self.error_detector.analyze_losing_trades(days=days)
        
        if 'error' in error_analysis:
            self.logger.warning(f"Analyse des erreurs impossible: {error_analysis['error']}")
            error_improvements = None
        else:
            # Analyser la fréquence des erreurs et obtenir des suggestions d'amélioration
            error_improvements = self.error_detector.analyze_error_frequency(error_analysis)
            self.logger.info(f"Analyse des erreurs terminée: {len(error_analysis.get('detected_errors', []))} erreurs détectées")
        
        # 2. Analyser les performances récentes
        performance_data = self.performance_analyzer.analyze_recent_performance(days=days)
        
        if 'error' in performance_data:
            self.logger.warning(f"Analyse des performances impossible: {performance_data['error']}")
        else:
            self.logger.info(f"Analyse des performances terminée: taux de réussite = {performance_data.get('win_rate', 0):.2%}")
        
        # 3. Identifier les modèles qui nécessitent des ajustements
        models_to_adjust = self._identify_models_to_adjust(performance_data, error_analysis)
        
        if not models_to_adjust:
            self.logger.info("Aucun modèle nécessitant des ajustements n'a été identifié")
            return {
                "status": "Aucun ajustement nécessaire",
                "error_analysis": error_analysis,
                "performance_data": performance_data,
                "adjustments_made": []
            }
        
        # 4. Effectuer les ajustements sur chaque modèle identifié
        adjustments_made = []
        
        for model_info in models_to_adjust:
            model_name = model_info['model_name']
            model_version = model_info['model_version']
            model_type = model_info['model_type']
            
            self.logger.info(f"Ajustement du modèle: {model_name} (version {model_version}, type {model_type})")
            
            try:
                # Charger le modèle
                model = self._load_model(model_name, model_version)
                
                if model is None:
                    self.logger.error(f"Impossible de charger le modèle {model_name} v{model_version}")
                    continue
                
                # Appliquer les ajustements spécifiques au type de modèle
                adjusted_model, adjustments = self._adjust_model(
                    model, 
                    model_type,
                    model_info.get('error_types', []),
                    model_info.get('performance_metrics', {}),
                    error_improvements
                )
                
                if adjusted_model is None:
                    self.logger.warning(f"Aucun ajustement effectué sur le modèle {model_name}")
                    continue
                
                # Enregistrer le modèle ajusté avec une nouvelle version
                new_version = self._increment_version(model_version)
                
                if self._save_model(adjusted_model, model_name, new_version):
                    # Enregistrer les ajustements dans l'historique
                    adjustment_record = {
                        "model_name": model_name,
                        "from_version": model_version,
                        "to_version": new_version,
                        "timestamp": datetime.now().isoformat(),
                        "adjustments": adjustments,
                        "based_on_errors": model_info.get('error_types', []),
                        "performance_before": model_info.get('performance_metrics', {})
                    }
                    
                    if model_name not in self.adjustment_history["models"]:
                        self.adjustment_history["models"][model_name] = []
                    
                    self.adjustment_history["models"][model_name].append(adjustment_record)
                    self._save_adjustment_history()
                    
                    # Ajouter l'ajustement à la liste des ajustements effectués
                    adjustments_made.append(adjustment_record)
                    
                    self.logger.info(f"Modèle {model_name} ajusté et sauvegardé en version {new_version}")
                else:
                    self.logger.error(f"Échec de la sauvegarde du modèle ajusté {model_name}")
            
            except Exception as e:
                self.logger.error(f"Erreur lors de l'ajustement du modèle {model_name}: {e}")
        
        return {
            "status": "success" if adjustments_made else "no_adjustments",
            "error_analysis": error_analysis,
            "performance_data": performance_data,
            "adjustments_made": adjustments_made
        }
    
    def _identify_models_to_adjust(self, 
                                 performance_data: Dict[str, Any], 
                                 error_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identifie les modèles qui nécessitent des ajustements en fonction des analyses.
        
        Args:
            performance_data: Données de performance
            error_analysis: Analyse des erreurs
            
        Returns:
            Liste des modèles à ajuster avec leurs informations
        """
        models_to_adjust = []
        
        # Récupérer les transactions récentes pour identifier les modèles actifs
        try:
            # Récupérer les dernières transactions (30 jours par défaut)
            recent_trades = self.trade_journal.get_trades(
                start_date=datetime.now() - timedelta(days=30)
            )
            
            if recent_trades.empty:
                self.logger.warning("Aucune transaction récente trouvée pour identifier les modèles à ajuster")
                return []
            
            # Identifier les modèles uniques utilisés
            models_used = {}
            
            for _, trade in recent_trades.iterrows():
                model_name = trade.get('strategy_name')
                model_version = trade.get('model_version')
                
                if not model_name or not model_version:
                    continue
                
                key = f"{model_name}_{model_version}"
                
                if key not in models_used:
                    models_used[key] = {
                        'model_name': model_name,
                        'model_version': model_version,
                        'trades': [],
                        'error_types': set(),
                        'performance_metrics': {
                            'win_count': 0,
                            'loss_count': 0,
                            'total_pnl': 0
                        }
                    }
                
                # Ajouter cette transaction à la liste des transactions du modèle
                models_used[key]['trades'].append(trade['trade_id'])
                
                # Mettre à jour les métriques de performance
                if trade['pnl'] > 0:
                    models_used[key]['performance_metrics']['win_count'] += 1
                else:
                    models_used[key]['performance_metrics']['loss_count'] += 1
                
                models_used[key]['performance_metrics']['total_pnl'] += trade['pnl']
            
            # Analyser les erreurs par modèle
            if 'detected_errors' in error_analysis:
                for trade_error in error_analysis['detected_errors']:
                    trade_id = trade_error['trade_id']
                    
                    # Trouver à quel modèle appartient cette transaction
                    for model_key, model_info in models_used.items():
                        if trade_id in model_info['trades']:
                            # Ajouter les types d'erreurs détectés
                            for error in trade_error.get('detected_errors', []):
                                model_info['error_types'].add(error['error_type'])
            
            # Déterminer quels modèles nécessitent des ajustements
            for model_key, model_info in models_used.items():
                # Calculer le taux de réussite
                total_trades = model_info['performance_metrics']['win_count'] + model_info['performance_metrics']['loss_count']
                win_rate = model_info['performance_metrics']['win_count'] / total_trades if total_trades > 0 else 0
                
                model_info['performance_metrics']['win_rate'] = win_rate
                model_info['performance_metrics']['total_trades'] = total_trades
                
                # Déterminer le type de modèle
                model_info['model_type'] = self._determine_model_type(model_info['model_name'])
                
                # Convertir error_types de set à liste pour la sérialisation JSON
                model_info['error_types'] = list(model_info['error_types'])
                
                # Critères pour ajuster un modèle
                needs_adjustment = False
                
                # 1. Le modèle a un taux de réussite inférieur à 50%
                if win_rate < 0.5 and total_trades >= 10:
                    needs_adjustment = True
                
                # 2. Le modèle a des erreurs spécifiques identifiées
                if model_info['error_types']:
                    needs_adjustment = True
                
                # 3. Le modèle a un PnL négatif
                if model_info['performance_metrics']['total_pnl'] < 0 and total_trades >= 5:
                    needs_adjustment = True
                
                if needs_adjustment:
                    models_to_adjust.append(model_info)
            
            return models_to_adjust
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'identification des modèles à ajuster: {e}")
            return []
    
    def _determine_model_type(self, model_name: str) -> str:
        """
        Détermine le type de modèle en fonction de son nom.
        
        Args:
            model_name: Nom du modèle
            
        Returns:
            Type du modèle (rl, ensemble, price, sentiment, etc.)
        """
        # Règles simples basées sur le nom du modèle
        model_name_lower = model_name.lower()
        
        if 'rl' in model_name_lower or 'reinforcement' in model_name_lower:
            return 'rl'
        elif 'ensemble' in model_name_lower:
            return 'ensemble'
        elif 'price' in model_name_lower or 'technical' in model_name_lower:
            return 'price'
        elif 'sentiment' in model_name_lower or 'nlp' in model_name_lower:
            return 'sentiment'
        else:
            # Par défaut, on considère que c'est un modèle de prix
            return 'price'
    
    def _increment_version(self, version: str) -> str:
        """
        Incrémente la version du modèle.
        
        Args:
            version: Version actuelle
            
        Returns:
            Nouvelle version
        """
        try:
            # Parse la version (format attendu: x.y.z)
            parts = version.split('.')
            
            if len(parts) >= 3:
                major, minor, patch = map(int, parts[:3])
                # Incrémenter le numéro de patch
                patch += 1
                return f"{major}.{minor}.{patch}"
            elif len(parts) == 2:
                major, minor = map(int, parts)
                # Ajouter un numéro de patch à 1
                return f"{major}.{minor}.1"
            elif len(parts) == 1:
                major = int(parts[0])
                # Ajouter un numéro de minor et patch
                return f"{major}.0.1"
            else:
                # Version invalide, renvoyer une nouvelle version par défaut
                return "1.0.0"
        except ValueError:
            # En cas d'erreur de parsing, renvoyer une nouvelle version
            return "1.0.0"
    
    def _load_model(self, model_name: str, model_version: str) -> Optional[Any]:
        """
        Charge un modèle depuis le disque.
        
        Args:
            model_name: Nom du modèle
            model_version: Version du modèle
            
        Returns:
            Le modèle chargé ou None en cas d'erreur
        """
        try:
            # Clé du modèle pour le cache
            model_key = f"{model_name}_{model_version}"
            
            # Vérifier si le modèle est déjà chargé
            if model_key in self.loaded_models:
                return self.loaded_models[model_key]
            
            # Construire le chemin du fichier du modèle
            model_path = os.path.join(self.models_dir, f"{model_name}_{model_version}.pkl")
            
            if not os.path.exists(model_path):
                self.logger.error(f"Le modèle {model_name} v{model_version} n'existe pas à l'emplacement {model_path}")
                return None
            
            # Charger le modèle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Mettre en cache le modèle
            self.loaded_models[model_key] = model
            
            self.logger.info(f"Modèle {model_name} v{model_version} chargé avec succès")
            return model
        
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle {model_name} v{model_version}: {e}")
            return None
    
    def _save_model(self, model: Any, model_name: str, model_version: str) -> bool:
        """
        Sauvegarde un modèle sur le disque.
        
        Args:
            model: Le modèle à sauvegarder
            model_name: Nom du modèle
            model_version: Version du modèle
            
        Returns:
            True si sauvegarde réussie, False sinon
        """
        try:
            # Construire le chemin du fichier du modèle
            model_path = os.path.join(self.models_dir, f"{model_name}_{model_version}.pkl")
            
            # Sauvegarder le modèle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Mettre à jour le cache
            model_key = f"{model_name}_{model_version}"
            self.loaded_models[model_key] = model
            
            self.logger.info(f"Modèle {model_name} v{model_version} sauvegardé avec succès à {model_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du modèle {model_name} v{model_version}: {e}")
            return False
    
    def _adjust_model(self, 
                     model: Any, 
                     model_type: str,
                     error_types: List[str],
                     performance_metrics: Dict[str, Any],
                     error_improvements: Optional[Dict[str, Any]]) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
        """
        Ajuste un modèle en fonction des erreurs détectées et des métriques de performance.
        
        Args:
            model: Le modèle à ajuster
            model_type: Type du modèle (rl, ensemble, price, sentiment)
            error_types: Types d'erreurs détectées pour ce modèle
            performance_metrics: Métriques de performance du modèle
            error_improvements: Suggestions d'amélioration basées sur l'analyse d'erreurs
            
        Returns:
            Tuple contenant le modèle ajusté et la liste des ajustements effectués
        """
        if not error_types and performance_metrics.get('win_rate', 0) >= 0.5:
            self.logger.info("Aucun ajustement nécessaire pour ce modèle")
            return None, []
        
        # Sélectionner la méthode d'ajustement en fonction du type de modèle
        if model_type == 'rl':
            return self._adjust_rl_model(model, error_types, performance_metrics, error_improvements)
        elif model_type == 'ensemble':
            return self._adjust_ensemble_model(model, error_types, performance_metrics, error_improvements)
        elif model_type == 'price':
            return self._adjust_price_model(model, error_types, performance_metrics, error_improvements)
        elif model_type == 'sentiment':
            return self._adjust_sentiment_model(model, error_types, performance_metrics, error_improvements)
        else:
            self.logger.warning(f"Type de modèle inconnu: {model_type}")
            return None, []
    
    def _adjust_rl_model(self, 
                        model: Any, 
                        error_types: List[str],
                        performance_metrics: Dict[str, Any],
                        error_improvements: Optional[Dict[str, Any]]) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Ajuste un modèle de reinforcement learning.
        
        Args:
            model: Le modèle RL à ajuster
            error_types: Types d'erreurs détectées
            performance_metrics: Métriques de performance
            error_improvements: Suggestions d'amélioration
            
        Returns:
            Tuple contenant le modèle ajusté et la liste des ajustements effectués
        """
        adjustments = []
        adjusted_model = copy.deepcopy(model)  # Copie profonde pour ne pas modifier l'original
        
        try:
            # Extraire les hyperparamètres courants du modèle
            current_params = getattr(adjusted_model, 'hyperparameters', {})
            if not current_params:
                # Certains modèles peuvent stocker les hyperparamètres différemment
                if hasattr(adjusted_model, 'get_hyperparameters'):
                    current_params = adjusted_model.get_hyperparameters()
                else:
                    current_params = {}
            
            # Préparer les nouveaux paramètres
            new_params = current_params.copy()
            
            # Ajustements spécifiques basés sur les types d'erreurs
            for error_type in error_types:
                if error_type == 'TIMING_ERROR':
                    # Ajuster la fréquence d'action pour réduire la latence
                    if 'action_frequency' in new_params:
                        new_params['action_frequency'] = max(1, int(new_params['action_frequency'] * 0.8))
                        adjustments.append({
                            'param': 'action_frequency',
                            'old_value': current_params['action_frequency'],
                            'new_value': new_params['action_frequency'],
                            'reason': 'Réduction de la latence pour les entrées/sorties'
                        })
                
                elif error_type == 'SIZE_ERROR':
                    # Réduire la taille maximale des positions
                    if 'max_position_size' in new_params:
                        new_params['max_position_size'] = new_params['max_position_size'] * 0.8
                        adjustments.append({
                            'param': 'max_position_size',
                            'old_value': current_params['max_position_size'],
                            'new_value': new_params['max_position_size'],
                            'reason': 'Réduction de la taille max des positions pour limiter le risque'
                        })
                
                elif error_type == 'TREND_MISREAD':
                    # Augmenter l'importance des indicateurs de tendance
                    if 'trend_weight' in new_params:
                        new_params['trend_weight'] = min(1.0, new_params['trend_weight'] * 1.2)
                        adjustments.append({
                            'param': 'trend_weight',
                            'old_value': current_params['trend_weight'],
                            'new_value': new_params['trend_weight'],
                            'reason': 'Augmentation de l\'importance des indicateurs de tendance'
                        })
                
                elif error_type == 'DELAYED_EXIT':
                    # Augmenter la sensibilité des signaux de sortie
                    if 'exit_threshold' in new_params:
                        new_params['exit_threshold'] = new_params['exit_threshold'] * 0.9
                        adjustments.append({
                            'param': 'exit_threshold',
                            'old_value': current_params['exit_threshold'],
                            'new_value': new_params['exit_threshold'],
                            'reason': 'Augmentation de la sensibilité des signaux de sortie'
                        })
            
            # Ajustements basés sur les performances générales
            win_rate = performance_metrics.get('win_rate', 0)
            
            if win_rate < 0.4:
                # Augmenter le seuil de confiance pour les entrées
                if 'entry_threshold' in new_params:
                    new_params['entry_threshold'] = min(0.9, new_params['entry_threshold'] * 1.1)
                    adjustments.append({
                        'param': 'entry_threshold',
                        'old_value': current_params['entry_threshold'],
                        'new_value': new_params['entry_threshold'],
                        'reason': 'Augmentation du seuil de confiance pour les entrées en raison d\'un faible taux de réussite'
                    })
                
                # Ajuster le gamma (facteur d'actualisation) pour privilégier les récompenses à court terme
                if 'gamma' in new_params:
                    new_params['gamma'] = max(0.1, new_params['gamma'] * 0.95)
                    adjustments.append({
                        'param': 'gamma',
                        'old_value': current_params['gamma'],
                        'new_value': new_params['gamma'],
                        'reason': 'Réduction du facteur d\'actualisation pour privilégier les récompenses à court terme'
                    })
            
            # Appliquer les nouveaux paramètres au modèle
            if hasattr(adjusted_model, 'hyperparameters'):
                adjusted_model.hyperparameters = new_params
            
            if hasattr(adjusted_model, 'set_hyperparameters'):
                adjusted_model.set_hyperparameters(new_params)
            
            return adjusted_model, adjustments
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement du modèle RL: {e}")
            return model, []  # Retourner le modèle original en cas d'erreur
    
    def _adjust_ensemble_model(self, 
                             model: Any, 
                             error_types: List[str],
                             performance_metrics: Dict[str, Any],
                             error_improvements: Optional[Dict[str, Any]]) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Ajuste un modèle d'ensemble (combinaison de plusieurs modèles).
        
        Args:
            model: Le modèle d'ensemble à ajuster
            error_types: Types d'erreurs détectées
            performance_metrics: Métriques de performance
            error_improvements: Suggestions d'amélioration
            
        Returns:
            Tuple contenant le modèle ajusté et la liste des ajustements effectués
        """
        adjustments = []
        adjusted_model = copy.deepcopy(model)  # Copie profonde pour ne pas modifier l'original
        
        try:
            # Pour les modèles d'ensemble, on ajuste les poids des sous-modèles
            sub_models = getattr(adjusted_model, 'models', [])
            weights = getattr(adjusted_model, 'weights', None)
            
            if not sub_models or weights is None:
                self.logger.warning("Le modèle d'ensemble n'a pas d'attributs 'models' ou 'weights'")
                return model, []
            
            # Évaluer la performance relative des sous-modèles (simulation)
            # Dans un cas réel, cette évaluation serait basée sur les données historiques
            model_performances = [0.5] * len(sub_models)  # Performances par défaut
            
            # Ajuster les poids en fonction des types d'erreurs
            new_weights = weights.copy()
            
            for error_type in error_types:
                if error_type == 'TREND_MISREAD':
                    # Augmenter le poids des modèles de tendance
                    for i, sub_model in enumerate(sub_models):
                        if hasattr(sub_model, 'name') and 'trend' in sub_model.name.lower():
                            new_weights[i] *= 1.2
                
                elif error_type == 'SIGNAL_CONFLICT':
                    # Ajuster l'arbitrage entre modèles contradictoires
                    if hasattr(adjusted_model, 'arbitration_method'):
                        old_method = adjusted_model.arbitration_method
                        adjusted_model.arbitration_method = 'weighted_consensus'
                        adjustments.append({
                            'param': 'arbitration_method',
                            'old_value': old_method,
                            'new_value': 'weighted_consensus',
                            'reason': 'Amélioration de la résolution des conflits entre signaux'
                        })
            
            # Normaliser les poids
            total_weight = sum(new_weights)
            if total_weight > 0:
                new_weights = [w / total_weight for w in new_weights]
            
            # Enregistrer les ajustements de poids si changés
            if new_weights != weights:
                adjustments.append({
                    'param': 'weights',
                    'old_value': weights,
                    'new_value': new_weights,
                    'reason': 'Ajustement des poids des sous-modèles en fonction des erreurs détectées'
                })
                
                # Appliquer les nouveaux poids
                adjusted_model.weights = new_weights
            
            return adjusted_model, adjustments
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement du modèle d'ensemble: {e}")
            return model, []
    
    def _adjust_price_model(self, 
                          model: Any, 
                          error_types: List[str],
                          performance_metrics: Dict[str, Any],
                          error_improvements: Optional[Dict[str, Any]]) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Ajuste un modèle basé sur le prix et les indicateurs techniques.
        
        Args:
            model: Le modèle de prix à ajuster
            error_types: Types d'erreurs détectées
            performance_metrics: Métriques de performance
            error_improvements: Suggestions d'amélioration
            
        Returns:
            Tuple contenant le modèle ajusté et la liste des ajustements effectués
        """
        adjustments = []
        adjusted_model = copy.deepcopy(model)  # Copie profonde pour ne pas modifier l'original
        
        try:
            # Extraire les paramètres courants
            params = getattr(adjusted_model, 'parameters', {})
            if not params:
                if hasattr(adjusted_model, 'get_parameters'):
                    params = adjusted_model.get_parameters()
                else:
                    params = {}
            
            # Copier les paramètres pour les modifier
            new_params = params.copy()
            
            # Ajustements basés sur les types d'erreurs
            for error_type in error_types:
                if error_type == 'TIMING_ERROR':
                    # Ajuster les périodes des moyennes mobiles
                    for param_name in params:
                        if 'period' in param_name.lower() and isinstance(params[param_name], int):
                            # Réduire les périodes pour être plus réactif
                            new_params[param_name] = max(3, int(params[param_name] * 0.9))
                            adjustments.append({
                                'param': param_name,
                                'old_value': params[param_name],
                                'new_value': new_params[param_name],
                                'reason': 'Réduction de la période pour améliorer la réactivité'
                            })
                
                elif error_type == 'TREND_MISREAD':
                    # Ajuster les paramètres liés à la détection de tendance
                    if 'trend_threshold' in params:
                        # Augmenter le seuil pour une détection plus stricte
                        new_params['trend_threshold'] = params['trend_threshold'] * 1.1
                        adjustments.append({
                            'param': 'trend_threshold',
                            'old_value': params['trend_threshold'],
                            'new_value': new_params['trend_threshold'],
                            'reason': 'Augmentation du seuil de détection de tendance'
                        })
                    
                    # Ajouter ou renforcer l'utilisation d'ADX
                    if 'use_adx' in params:
                        new_params['use_adx'] = True
                        if not params['use_adx']:
                            adjustments.append({
                                'param': 'use_adx',
                                'old_value': False,
                                'new_value': True,
                                'reason': 'Activation de l\'indicateur ADX pour améliorer la détection de tendance'
                            })
                
                elif error_type == 'PREMATURE_EXIT':
                    # Ajuster les paramètres de sortie
                    if 'take_profit' in params:
                        # Augmenter le niveau de prise de profit
                        new_params['take_profit'] = params['take_profit'] * 1.1
                        adjustments.append({
                            'param': 'take_profit',
                            'old_value': params['take_profit'],
                            'new_value': new_params['take_profit'],
                            'reason': 'Augmentation du niveau de prise de profit'
                        })
                
                elif error_type == 'DELAYED_EXIT':
                    # Ajuster les stops de sortie
                    if 'stop_loss' in params:
                        # Resserrer les stops pour sortir plus rapidement
                        new_params['stop_loss'] = params['stop_loss'] * 0.9
                        adjustments.append({
                            'param': 'stop_loss',
                            'old_value': params['stop_loss'],
                            'new_value': new_params['stop_loss'],
                            'reason': 'Resserrement des stops pour réagir plus rapidement'
                        })
            
            # Appliquer les nouveaux paramètres au modèle
            if hasattr(adjusted_model, 'parameters'):
                adjusted_model.parameters = new_params
            
            if hasattr(adjusted_model, 'set_parameters'):
                adjusted_model.set_parameters(new_params)
            
            return adjusted_model, adjustments
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement du modèle de prix: {e}")
            return model, []
    
    def _adjust_sentiment_model(self, 
                              model: Any, 
                              error_types: List[str],
                              performance_metrics: Dict[str, Any],
                              error_improvements: Optional[Dict[str, Any]]) -> Tuple[Any, List[Dict[str, Any]]]:
        """
        Ajuste un modèle basé sur l'analyse de sentiment.
        
        Args:
            model: Le modèle de sentiment à ajuster
            error_types: Types d'erreurs détectées
            performance_metrics: Métriques de performance
            error_improvements: Suggestions d'amélioration
            
        Returns:
            Tuple contenant le modèle ajusté et la liste des ajustements effectués
        """
        adjustments = []
        adjusted_model = copy.deepcopy(model)  # Copie profonde pour ne pas modifier l'original
        
        try:
            # Extraire les paramètres courants
            params = getattr(adjusted_model, 'parameters', {})
            if not params:
                if hasattr(adjusted_model, 'get_parameters'):
                    params = adjusted_model.get_parameters()
                else:
                    params = {}
            
            # Copier les paramètres pour les modifier
            new_params = params.copy()
            
            # Ajustements basés sur les types d'erreurs
            for error_type in error_types:
                if error_type == 'NEWS_IMPACT':
                    # Augmenter l'importance des actualités
                    if 'news_weight' in params:
                        new_params['news_weight'] = min(1.0, params['news_weight'] * 1.2)
                        adjustments.append({
                            'param': 'news_weight',
                            'old_value': params['news_weight'],
                            'new_value': new_params['news_weight'],
                            'reason': 'Augmentation de l\'importance des actualités dans l\'analyse'
                        })
                    
                    # Augmenter la fréquence de mise à jour des données d'actualités
                    if 'news_update_frequency' in params:
                        new_params['news_update_frequency'] = max(1, int(params['news_update_frequency'] * 0.8))
                        adjustments.append({
                            'param': 'news_update_frequency',
                            'old_value': params['news_update_frequency'],
                            'new_value': new_params['news_update_frequency'],
                            'reason': 'Augmentation de la fréquence de mise à jour des actualités'
                        })
                
                elif error_type == 'SIGNAL_CONFLICT':
                    # Ajuster le seuil de sentiment pour les signaux plus clairs
                    if 'sentiment_threshold' in params:
                        new_params['sentiment_threshold'] = params['sentiment_threshold'] * 1.1
                        adjustments.append({
                            'param': 'sentiment_threshold',
                            'old_value': params['sentiment_threshold'],
                            'new_value': new_params['sentiment_threshold'],
                            'reason': 'Augmentation du seuil de sentiment pour des signaux plus clairs'
                        })
            
            # Ajustements basés sur les performances générales
            win_rate = performance_metrics.get('win_rate', 0)
            
            if win_rate < 0.4:
                # Augmenter la fenêtre temporelle d'analyse
                if 'analysis_window' in params:
                    new_params['analysis_window'] = int(params['analysis_window'] * 1.2)
                    adjustments.append({
                        'param': 'analysis_window',
                        'old_value': params['analysis_window'],
                        'new_value': new_params['analysis_window'],
                        'reason': 'Augmentation de la fenêtre d\'analyse en raison d\'un faible taux de réussite'
                    })
            
            # Appliquer les nouveaux paramètres au modèle
            if hasattr(adjusted_model, 'parameters'):
                adjusted_model.parameters = new_params
            
            if hasattr(adjusted_model, 'set_parameters'):
                adjusted_model.set_parameters(new_params)
            
            return adjusted_model, adjustments
        
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement du modèle de sentiment: {e}")
            return model, [] 