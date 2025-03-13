import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
import logging
from datetime import datetime, timedelta
import json
import os

from .trade_journal import TradeJournal

class ErrorDetector:
    """
    Analyse les transactions échouées pour identifier les patterns d'erreur
    et les opportunités d'amélioration des modèles.
    """
    
    # Types d'erreurs connus
    ERROR_TYPES = {
        'TIMING_ERROR': 'Entrée ou sortie mal chronométrée',
        'SIZE_ERROR': 'Taille de position inadaptée au risque',
        'TREND_MISREAD': 'Mauvaise lecture de la tendance',
        'SIGNAL_CONFLICT': 'Signaux contradictoires ignorés',
        'OVERTRADING': 'Surtrading (trop de transactions)',
        'FOMO_ENTRY': 'Entrée basée sur la peur de rater (FOMO)',
        'PREMATURE_EXIT': 'Sortie prématurée avant objectif',
        'DELAYED_EXIT': 'Sortie retardée après signal',
        'IGNORED_STOP': 'Stop-loss ignoré',
        'CHASING_LOSS': 'Poursuite des pertes',
        'MARKET_CONDITION': 'Inadaptation aux conditions de marché',
        'HIGH_VOLATILITY': 'Sous-estimation de la volatilité',
        'CORRELATION_EFFECT': 'Effet de corrélation entre actifs',
        'NEWS_IMPACT': 'Impact non anticipé des actualités',
        'TECHNICAL_ISSUE': 'Problème technique lors de l\'exécution'
    }
    
    def __init__(self, trade_journal: TradeJournal):
        """
        Initialise le détecteur d'erreurs.
        
        Args:
            trade_journal: Instance de TradeJournal pour accéder aux données
        """
        self.trade_journal = trade_journal
        self.logger = logging.getLogger(__name__)
    
    def analyze_losing_trades(self, days: int = 90) -> Dict[str, Any]:
        """
        Analyse les transactions perdantes pour identifier les modèles d'erreur.
        
        Args:
            days: Nombre de jours à analyser
            
        Returns:
            Dictionnaire contenant les erreurs identifiées et leur fréquence
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Récupérer les transactions perdantes
        trades_df = self.trade_journal.get_trades(
            start_date=start_date, 
            end_date=end_date,
            profitable_only=False
        )
        
        # Filtrer pour ne garder que les transactions perdantes
        losing_trades_df = trades_df[trades_df['pnl'] <= 0]
        
        if losing_trades_df.empty:
            self.logger.warning(f"Aucune transaction perdante trouvée dans les {days} derniers jours")
            return {"error": "Aucune transaction perdante disponible pour l'analyse"}
        
        # Initialiser les résultats
        results = {
            'period': f"{start_date.date()} - {end_date.date()}",
            'total_losing_trades': len(losing_trades_df),
            'average_loss': losing_trades_df['pnl'].mean(),
            'max_loss': losing_trades_df['pnl'].min(),
            'total_loss': losing_trades_df['pnl'].sum(),
            'error_patterns': {},
            'symbol_errors': {},
            'time_patterns': {},
            'detected_errors': []
        }
        
        # Détecter les erreurs courantes dans chaque transaction
        for _, trade in losing_trades_df.iterrows():
            trade_errors = self._detect_errors_in_trade(trade)
            
            trade_analysis = {
                'trade_id': trade['trade_id'],
                'symbol': trade['symbol'],
                'entry_time': trade['entry_time'],
                'exit_time': trade['exit_time'],
                'pnl': trade['pnl'],
                'detected_errors': trade_errors
            }
            
            results['detected_errors'].append(trade_analysis)
            
            # Compter les types d'erreurs
            for error in trade_errors:
                error_type = error['error_type']
                if error_type not in results['error_patterns']:
                    results['error_patterns'][error_type] = {
                        'count': 0,
                        'description': self.ERROR_TYPES.get(error_type, "Erreur inconnue"),
                        'avg_loss': 0,
                        'total_loss': 0
                    }
                
                results['error_patterns'][error_type]['count'] += 1
                results['error_patterns'][error_type]['total_loss'] += trade['pnl']
            
            # Analyser les erreurs par symbole
            symbol = trade['symbol']
            if symbol not in results['symbol_errors']:
                results['symbol_errors'][symbol] = {
                    'count': 0,
                    'total_loss': 0,
                    'error_types': {}
                }
            
            results['symbol_errors'][symbol]['count'] += 1
            results['symbol_errors'][symbol]['total_loss'] += trade['pnl']
            
            for error in trade_errors:
                error_type = error['error_type']
                if error_type not in results['symbol_errors'][symbol]['error_types']:
                    results['symbol_errors'][symbol]['error_types'][error_type] = 0
                results['symbol_errors'][symbol]['error_types'][error_type] += 1
            
            # Analyser les patterns temporels
            if 'entry_time' in trade and trade['entry_time']:
                entry_time = pd.to_datetime(trade['entry_time'])
                day_of_week = entry_time.day_name()
                hour_of_day = entry_time.hour
                
                # Erreurs par jour de la semaine
                if day_of_week not in results['time_patterns']:
                    results['time_patterns'][day_of_week] = {
                        'count': 0,
                        'total_loss': 0,
                        'error_types': {}
                    }
                
                results['time_patterns'][day_of_week]['count'] += 1
                results['time_patterns'][day_of_week]['total_loss'] += trade['pnl']
                
                for error in trade_errors:
                    error_type = error['error_type']
                    if error_type not in results['time_patterns'][day_of_week]['error_types']:
                        results['time_patterns'][day_of_week]['error_types'][error_type] = 0
                    results['time_patterns'][day_of_week]['error_types'][error_type] += 1
        
        # Calculer les moyennes
        for error_type in results['error_patterns']:
            pattern = results['error_patterns'][error_type]
            pattern['avg_loss'] = pattern['total_loss'] / pattern['count'] if pattern['count'] > 0 else 0
        
        # Trier les erreurs par fréquence
        results['error_patterns'] = dict(sorted(
            results['error_patterns'].items(),
            key=lambda item: item[1]['count'],
            reverse=True
        ))
        
        return results
    
    def _detect_errors_in_trade(self, trade: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Détecte les erreurs potentielles dans une transaction en analysant ses caractéristiques.
        
        Args:
            trade: Dictionnaire contenant les données de la transaction
            
        Returns:
            Liste des erreurs détectées avec leur type et description
        """
        errors = []
        
        # Données de base
        entry_signals = trade.get('entry_signals', {})
        exit_signals = trade.get('exit_signals', {})
        market_conditions = trade.get('market_conditions', {})
        
        # 1. Vérifier les erreurs de timing
        if self._check_timing_error(trade, entry_signals, exit_signals):
            errors.append({
                'error_type': 'TIMING_ERROR',
                'description': "Entrée ou sortie mal synchronisée avec les signaux techniques",
                'severity': 3
            })
        
        # 2. Vérifier les erreurs de taille de position
        if self._check_size_error(trade):
            errors.append({
                'error_type': 'SIZE_ERROR',
                'description': "Taille de position inadaptée au niveau de risque",
                'severity': 2
            })
        
        # 3. Vérifier les erreurs de lecture de tendance
        if self._check_trend_misread(trade, entry_signals, market_conditions):
            errors.append({
                'error_type': 'TREND_MISREAD',
                'description': "Position prise à contre-tendance sans confirmation suffisante",
                'severity': 4
            })
        
        # 4. Vérifier les conflits de signaux
        if self._check_signal_conflict(entry_signals):
            errors.append({
                'error_type': 'SIGNAL_CONFLICT',
                'description': "Signaux contradictoires ignorés lors de l'entrée en position",
                'severity': 3
            })
        
        # 5. Vérifier le surtrading
        if self._check_overtrading(trade):
            errors.append({
                'error_type': 'OVERTRADING',
                'description': "Position ouverte après une série de transactions fréquentes",
                'severity': 2
            })
        
        # 6. Vérifier les sorties prématurées
        if self._check_premature_exit(trade, exit_signals):
            errors.append({
                'error_type': 'PREMATURE_EXIT',
                'description': "Position fermée avant d'atteindre l'objectif de profit",
                'severity': 2
            })
        
        # 7. Vérifier les sorties tardives
        if self._check_delayed_exit(trade, exit_signals):
            errors.append({
                'error_type': 'DELAYED_EXIT',
                'description': "Position maintenue après des signaux de sortie clairs",
                'severity': 4
            })
        
        # 8. Vérifier l'ignorance des stops
        if self._check_ignored_stop(trade, exit_signals):
            errors.append({
                'error_type': 'IGNORED_STOP',
                'description': "Stop-loss ignoré ou déplacé à la baisse",
                'severity': 5
            })
        
        # 9. Vérifier les conditions de marché inadaptées
        if self._check_market_condition_mismatch(trade, market_conditions):
            errors.append({
                'error_type': 'MARKET_CONDITION',
                'description': "Stratégie inadaptée aux conditions actuelles du marché",
                'severity': 3
            })
            
        # 10. Vérifier l'impact de la volatilité
        if self._check_volatility_impact(trade, market_conditions):
            errors.append({
                'error_type': 'HIGH_VOLATILITY',
                'description': "Impact négatif d'une volatilité élevée sur la transaction",
                'severity': 3
            })
        
        return errors
    
    def _check_timing_error(self, trade: Dict[str, Any], entry_signals: Dict, exit_signals: Dict) -> bool:
        """Vérifie les erreurs de timing dans les entrées et sorties"""
        # Si nous avons des données temporelles dans les signaux
        if isinstance(entry_signals, dict) and 'signal_time' in entry_signals:
            entry_time = pd.to_datetime(trade.get('entry_time'))
            signal_time = pd.to_datetime(entry_signals.get('signal_time'))
            
            # Si l'entrée est trop tardive (plus de 30 minutes après le signal)
            if signal_time and entry_time and (entry_time - signal_time).total_seconds() > 1800:
                return True
                
        # Vérifier si la sortie a été retardée après un signal de sortie
        if isinstance(exit_signals, dict) and 'signal_time' in exit_signals:
            exit_time = pd.to_datetime(trade.get('exit_time'))
            signal_time = pd.to_datetime(exit_signals.get('signal_time'))
            
            if signal_time and exit_time and (exit_time - signal_time).total_seconds() > 1800:
                return True
                
        return False
    
    def _check_size_error(self, trade: Dict[str, Any]) -> bool:
        """Vérifie si la taille de la position était inappropriée"""
        # Vérifier si la perte dépasse un seuil qui suggère une taille de position trop importante
        if 'pnl_percent' in trade and trade['pnl_percent'] < -5:
            return True
            
        return False
    
    def _check_trend_misread(self, trade: Dict[str, Any], entry_signals: Dict, market_conditions: Dict) -> bool:
        """Vérifie si la transaction a été prise contre la tendance dominante"""
        if isinstance(market_conditions, dict):
            # Si le trade est long mais la tendance est baissière
            if trade.get('direction') == 'BUY' and market_conditions.get('trend') == 'DOWNTREND':
                return True
                
            # Si le trade est court mais la tendance est haussière
            if trade.get('direction') == 'SELL' and market_conditions.get('trend') == 'UPTREND':
                return True
                
        return False
    
    def _check_signal_conflict(self, entry_signals: Dict) -> bool:
        """Vérifie s'il y avait des signaux contradictoires qui ont été ignorés"""
        if isinstance(entry_signals, dict):
            # Vérifier les signaux contradictoires (exemple: signaux techniques vs fondamentaux)
            buy_signals = 0
            sell_signals = 0
            
            # Compter les signaux d'achat et de vente
            for key, value in entry_signals.items():
                if 'signal' in key.lower():
                    if isinstance(value, str):
                        if 'buy' in value.lower() or 'long' in value.lower():
                            buy_signals += 1
                        elif 'sell' in value.lower() or 'short' in value.lower():
                            sell_signals += 1
            
            # S'il y a à la fois des signaux d'achat et de vente significatifs
            if buy_signals > 0 and sell_signals > 0:
                return True
                
        return False
    
    def _check_overtrading(self, trade: Dict[str, Any]) -> bool:
        """Détecte le surtrading en cherchant les périodes de haute fréquence de transactions"""
        # Cette fonction nécessiterait d'analyser l'historique des transactions récentes
        # Pour simplifier, nous pouvons utiliser des métadonnées si elles sont disponibles
        metadata = trade.get('trade_metadata', {})
        if isinstance(metadata, dict):
            return metadata.get('is_overtrading', False)
            
        return False
    
    def _check_premature_exit(self, trade: Dict[str, Any], exit_signals: Dict) -> bool:
        """Vérifie si la sortie était prématurée par rapport aux objectifs"""
        if isinstance(exit_signals, dict):
            # Si la sortie a eu lieu avant d'atteindre l'objectif de profit
            target = exit_signals.get('profit_target')
            exit_price = trade.get('exit_price')
            entry_price = trade.get('entry_price')
            
            if target and exit_price and entry_price:
                direction = trade.get('direction')
                
                if direction == 'BUY' and exit_price < (entry_price * (1 + target/100)):
                    return True
                    
                if direction == 'SELL' and exit_price > (entry_price * (1 - target/100)):
                    return True
                    
        return False
    
    def _check_delayed_exit(self, trade: Dict[str, Any], exit_signals: Dict) -> bool:
        """Vérifie si la sortie a été retardée après l'apparition de signaux clairs"""
        # Similaire à _check_timing_error mais spécifique aux retards significatifs
        if isinstance(exit_signals, dict) and 'signal_time' in exit_signals:
            exit_time = pd.to_datetime(trade.get('exit_time'))
            signal_time = pd.to_datetime(exit_signals.get('signal_time'))
            
            # Si la sortie est retardée de plus d'une heure
            if signal_time and exit_time and (exit_time - signal_time).total_seconds() > 3600:
                return True
                
        return False
    
    def _check_ignored_stop(self, trade: Dict[str, Any], exit_signals: Dict) -> bool:
        """Vérifie si un stop-loss a été ignoré"""
        if isinstance(exit_signals, dict):
            stop_loss = exit_signals.get('stop_loss')
            exit_price = trade.get('exit_price')
            direction = trade.get('direction')
            
            if stop_loss is not None and exit_price:
                # Pour un trade long, vérifier si le prix de sortie est bien en-dessous du stop-loss
                if direction == 'BUY' and exit_price < stop_loss * 0.98:
                    return True
                    
                # Pour un trade court, vérifier si le prix de sortie est bien au-dessus du stop-loss
                if direction == 'SELL' and exit_price > stop_loss * 1.02:
                    return True
                    
        return False
    
    def _check_market_condition_mismatch(self, trade: Dict[str, Any], market_conditions: Dict) -> bool:
        """Vérifie si la stratégie était adaptée aux conditions de marché"""
        if isinstance(market_conditions, dict):
            # Vérifier l'adéquation de la stratégie aux conditions de marché
            strategy = trade.get('strategy_name')
            market_condition = market_conditions.get('market_type')
            
            # Exemples de cas de non-adéquation (à personnaliser selon les stratégies)
            if strategy == 'trend_following' and market_condition == 'ranging':
                return True
                
            if strategy == 'mean_reversion' and market_condition == 'trending':
                return True
                
            if strategy == 'breakout' and market_condition == 'low_volatility':
                return True
                
        return False
    
    def _check_volatility_impact(self, trade: Dict[str, Any], market_conditions: Dict) -> bool:
        """Vérifie si la volatilité a eu un impact négatif sur la transaction"""
        if isinstance(market_conditions, dict):
            volatility = market_conditions.get('volatility')
            
            # Si la volatilité est marquée comme haute
            if volatility and isinstance(volatility, (int, float)) and volatility > 0.8:
                return True
                
            if isinstance(volatility, str) and volatility.lower() == 'high':
                return True
                
        return False
    
    def analyze_error_frequency(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse la fréquence des erreurs et leur impact pour identifier
        les domaines prioritaires d'amélioration.
        
        Args:
            error_data: Données d'analyse d'erreur du journal
            
        Returns:
            Dictionnaire contenant les priorités d'amélioration
        """
        if 'error' in error_data:
            return error_data
            
        # Récupérer les patterns d'erreur
        error_patterns = error_data.get('error_patterns', {})
        
        # Si aucune erreur n'a été détectée
        if not error_patterns:
            return {
                'status': 'No errors detected',
                'improvement_areas': []
            }
            
        # Calculer le score d'impact pour chaque type d'erreur
        impact_scores = {}
        for error_type, data in error_patterns.items():
            frequency = data['count']
            avg_loss = abs(data.get('avg_loss', 0))
            
            # Formule de score: fréquence * perte moyenne
            impact_score = frequency * avg_loss
            impact_scores[error_type] = impact_score
            
        # Trier les erreurs par score d'impact
        sorted_errors = sorted(
            [(error_type, score) for error_type, score in impact_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Créer des suggestions d'amélioration pour les types d'erreurs les plus impactants
        improvement_areas = []
        
        for error_type, score in sorted_errors:
            if error_type in error_patterns:
                error_data = error_patterns[error_type]
                
                improvement = {
                    'error_type': error_type,
                    'description': self.ERROR_TYPES.get(error_type, "Type d'erreur inconnu"),
                    'frequency': error_data['count'],
                    'avg_loss': error_data['avg_loss'],
                    'impact_score': score,
                    'suggested_actions': self._get_suggested_actions(error_type)
                }
                
                improvement_areas.append(improvement)
        
        return {
            'status': 'Success',
            'improvement_areas': improvement_areas
        }
    
    def _get_suggested_actions(self, error_type: str) -> List[str]:
        """
        Retourne des suggestions d'actions pour corriger un type d'erreur spécifique.
        
        Args:
            error_type: Type d'erreur à corriger
            
        Returns:
            Liste de suggestions d'actions
        """
        # Dictionnaire des suggestions par type d'erreur
        suggestions = {
            'TIMING_ERROR': [
                "Optimiser les algorithmes de détection de signaux pour réduire la latence",
                "Mettre en place des ordres conditionnels préprogrammés",
                "Ajouter une analyse de confirmation avant exécution"
            ],
            'SIZE_ERROR': [
                "Implémenter un gestionnaire de risque plus strict pour calculer la taille des positions",
                "Limiter la taille des positions à un pourcentage maximum du capital",
                "Ajuster dynamiquement la taille en fonction de la volatilité du marché"
            ],
            'TREND_MISREAD': [
                "Incorporer des indicateurs multi-temporels pour mieux évaluer la tendance",
                "Ajouter des filtres de confirmation de tendance avant entrée",
                "Renforcer l'analyse directionnelle avec des indicateurs complémentaires"
            ],
            'SIGNAL_CONFLICT': [
                "Mettre en œuvre un système de pondération des signaux contradictoires",
                "Ajouter une couche d'arbitrage pour résoudre les conflits de signaux",
                "Exiger un consensus minimum entre les indicateurs avant d'entrer en position"
            ],
            'OVERTRADING': [
                "Implémenter une limite quotidienne de transactions",
                "Augmenter le seuil de confiance requis après une série de transactions",
                "Ajouter des périodes de refroidissement obligatoires entre les transactions"
            ],
            'FOMO_ENTRY': [
                "Mettre en place des règles strictes pour les entrées après un mouvement important",
                "Ajouter une vérification de la distance par rapport aux moyennes mobiles",
                "Exiger des signaux de confirmation supplémentaires pour les entrées tardives"
            ],
            'PREMATURE_EXIT': [
                "Ajuster les paramètres des prises de profit pour permettre plus d'extension",
                "Mettre en œuvre des sorties partielles et des trailing stops",
                "Incorporer l'analyse de la dynamique du marché dans les décisions de sortie"
            ],
            'DELAYED_EXIT': [
                "Automatiser les sorties sur signaux critiques",
                "Renforcer les règles de sortie en cas de retournement de tendance",
                "Implémenter des stops adaptatifs basés sur la volatilité"
            ],
            'IGNORED_STOP': [
                "Rendre les stops non modifiables une fois définis",
                "Ajouter un mécanisme d'exécution forcée des stops",
                "Intégrer des stops cachés ou des ordres stop-market pour garantir l'exécution"
            ],
            'CHASING_LOSS': [
                "Imposer des limites de pertes quotidiennes et des périodes de pause",
                "Mettre en place une réduction progressive de la taille des positions après des pertes",
                "Exiger une analyse post-mortem avant de reprendre le trading après des pertes importantes"
            ],
            'MARKET_CONDITION': [
                "Développer des détecteurs de régime de marché plus précis",
                "Créer des modes de trading spécifiques pour différentes conditions de marché",
                "Adapter automatiquement les paramètres du modèle aux conditions de marché actuelles"
            ],
            'HIGH_VOLATILITY': [
                "Réduire automatiquement la taille des positions en période de forte volatilité",
                "Ajuster les stops et objectifs en fonction de la volatilité actuelle",
                "Mettre en place des filtres spécifiques pour les périodes de forte volatilité"
            ],
            'CORRELATION_EFFECT': [
                "Analyser la corrélation entre les actifs du portefeuille avant prise de position",
                "Développer un système de gestion de risque tenant compte de la corrélation",
                "Diversifier les stratégies et les actifs pour réduire l'exposition corrélée"
            ],
            'NEWS_IMPACT': [
                "Intégrer un calendrier économique pour éviter les positions avant les événements majeurs",
                "Développer un mécanisme de détection d'actualités et de réaction rapide",
                "Ajouter une analyse de sentiment des actualités dans le processus de décision"
            ],
            'TECHNICAL_ISSUE': [
                "Mettre en place des systèmes de surveillance et d'alerte pour les défaillances techniques",
                "Développer des mécanismes de failover et de redondance",
                "Automatiser les tests de connectivité et d'intégrité des données"
            ]
        }
        
        # Retourner les suggestions pour ce type d'erreur, ou une liste vide si non trouvé
        return suggestions.get(error_type, ["Analyser plus en détail ce type d'erreur", 
                                           "Consulter les logs détaillés pour identifier les causes",
                                           "Développer des règles spécifiques pour prévenir cette erreur"]) 