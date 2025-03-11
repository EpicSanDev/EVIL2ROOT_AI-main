"""
Module principal du plugin d'analyse de news.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from app.plugins.plugin_base import PluginBase
from app.plugins.events import EventType, NewsEventType

logger = logging.getLogger("plugins.news_analyzer")

class NewsAnalyzerPlugin(PluginBase):
    """
    Plugin d'analyse avancée de news et de sentiment.
    Ajoute des fonctionnalités supplémentaires pour l'analyse des actualités financières.
    """
    
    # Métadonnées du plugin
    plugin_id = "news_analyzer_plugin"
    plugin_name = "Analyseur Avancé de News"
    plugin_description = "Analyse avancée des actualités financières avec détection d'impact et de tendances."
    plugin_version = "1.0.0"
    plugin_author = "EVIL2ROOT Team"
    
    def __init__(self):
        """Initialise le plugin"""
        super().__init__()
        
        # Paramètres par défaut
        self.settings = self.get_default_settings()
        
        # Historique des nouvelles analysées
        self.news_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Statistiques d'analyse
        self.analysis_stats = {
            "total_news_analyzed": 0,
            "high_impact_news_detected": 0,
            "symbols_analyzed": set(),
            "last_update": None
        }
        
        # Chemin de stockage des données du plugin
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Fichier de stockage de l'historique
        self.history_file = os.path.join(self.data_dir, "news_history.json")
        
        # Charger l'historique s'il existe
        self._load_history()
    
    def get_default_settings(self) -> Dict[str, Any]:
        """Retourne les paramètres par défaut du plugin"""
        return {
            "enable_topic_detection": True,
            "enable_trend_analysis": True,
            "high_impact_threshold": 0.75,
            "save_history": True,
            "history_days": 30,
            "max_news_per_symbol": 100,
            "additional_keywords": {
                "positive": ["breakthrough", "surpass", "exceed", "growth"],
                "negative": ["downgrade", "lawsuit", "decline", "miss"]
            }
        }
    
    def on_enable(self) -> None:
        """Appelé lorsque le plugin est activé"""
        logger.info(f"Plugin {self.plugin_name} activé")
        
        # S'enregistrer pour les événements
        if hasattr(self, "plugin_manager"):
            # Événements de news
            self.plugin_manager.register_callback(str(NewsEventType.AFTER_NEWS_FETCH), self.on_news_fetched)
            self.plugin_manager.register_callback(str(NewsEventType.BEFORE_SENTIMENT_ANALYSIS), self.on_before_sentiment_analysis)
            self.plugin_manager.register_callback(str(NewsEventType.AFTER_SENTIMENT_ANALYSIS), self.on_after_sentiment_analysis)
            
            # Événements standard
            self.plugin_manager.register_callback(str(EventType.SYSTEM_SHUTDOWN), self.on_system_shutdown)
    
    def on_disable(self) -> None:
        """Appelé lorsque le plugin est désactivé"""
        logger.info(f"Plugin {self.plugin_name} désactivé")
        
        # Se désinscrire des événements
        if hasattr(self, "plugin_manager"):
            # Événements de news
            self.plugin_manager.unregister_callback(str(NewsEventType.AFTER_NEWS_FETCH), self.on_news_fetched)
            self.plugin_manager.unregister_callback(str(NewsEventType.BEFORE_SENTIMENT_ANALYSIS), self.on_before_sentiment_analysis)
            self.plugin_manager.unregister_callback(str(NewsEventType.AFTER_SENTIMENT_ANALYSIS), self.on_after_sentiment_analysis)
            
            # Événements standard
            self.plugin_manager.unregister_callback(str(EventType.SYSTEM_SHUTDOWN), self.on_system_shutdown)
        
        # Sauvegarder l'historique
        if self.settings["save_history"]:
            self._save_history()
    
    def register_event_handlers(self, plugin_manager) -> None:
        """Enregistre les gestionnaires d'événements"""
        self.plugin_manager = plugin_manager
        
        # Événements de news
        plugin_manager.register_callback(str(NewsEventType.AFTER_NEWS_FETCH), self.on_news_fetched)
        plugin_manager.register_callback(str(NewsEventType.BEFORE_SENTIMENT_ANALYSIS), self.on_before_sentiment_analysis)
        plugin_manager.register_callback(str(NewsEventType.AFTER_SENTIMENT_ANALYSIS), self.on_after_sentiment_analysis)
        
        # Événements standard
        plugin_manager.register_callback(str(EventType.SYSTEM_SHUTDOWN), self.on_system_shutdown)
    
    def on_news_fetched(self, symbol: str, news: List[Dict], sources: List[str], timestamp: datetime) -> List[Dict]:
        """
        Appelé après la récupération des nouvelles.
        Enrichit les nouvelles avec des métadonnées supplémentaires.
        
        Args:
            symbol: Symbole concerné
            news: Liste des actualités récupérées
            sources: Sources consultées
            timestamp: Horodatage
            
        Returns:
            Liste des actualités enrichies
        """
        if not news:
            return news
        
        logger.info(f"Analyse des actualités pour {symbol}: {len(news)} articles")
        
        # Enrichir les actualités avec des informations supplémentaires
        enriched_news = []
        for item in news:
            # Copier l'article original
            enriched_item = item.copy()
            
            # Ajouter des métadonnées
            enriched_item.update({
                "analyzed_by": self.plugin_name,
                "plugin_version": self.plugin_version,
                "analysis_timestamp": timestamp.isoformat(),
            })
            
            # Analyser les mots-clés additionnels
            if "title" in item:
                title_lower = item["title"].lower()
                summary_lower = item.get("summary", "").lower()
                full_text = f"{title_lower} {summary_lower}"
                
                # Compter les mots-clés positifs et négatifs
                positive_keywords = self.settings["additional_keywords"]["positive"]
                negative_keywords = self.settings["additional_keywords"]["negative"]
                
                positive_count = sum(1 for keyword in positive_keywords if keyword.lower() in full_text)
                negative_count = sum(1 for keyword in negative_keywords if keyword.lower() in full_text)
                
                # Calculer un score d'impact basé sur les mots-clés
                keyword_score = (positive_count - negative_count) / (positive_count + negative_count + 1)
                
                # Analyser les catégories thématiques
                if self.settings["enable_topic_detection"]:
                    topics = self._detect_topics(full_text)
                    enriched_item["topics"] = topics
                
                # Calculer le score d'impact global
                base_impact = item.get("relevance_score", 0.5)
                keyword_weight = 0.4
                impact_score = base_impact * (1 - keyword_weight) + keyword_score * keyword_weight
                
                # Mettre à jour les données d'impact
                enriched_item["impact_details"] = {
                    "base_score": base_impact,
                    "keyword_score": keyword_score,
                    "positive_keywords": positive_count,
                    "negative_keywords": negative_count,
                    "final_impact_score": impact_score
                }
                
                # Détecter les actualités à fort impact
                if impact_score > self.settings["high_impact_threshold"]:
                    enriched_item["high_impact"] = True
                    self.analysis_stats["high_impact_news_detected"] += 1
                    
                    # Déclencher un événement pour les actualités à fort impact
                    if hasattr(self, "plugin_manager"):
                        self.plugin_manager.trigger_event(
                            str(NewsEventType.HIGH_IMPACT_NEWS_DETECTED),
                            symbol=symbol,
                            news_item=enriched_item,
                            impact_score=impact_score,
                            timestamp=timestamp
                        )
                else:
                    enriched_item["high_impact"] = False
            
            enriched_news.append(enriched_item)
        
        # Mettre à jour les statistiques
        self.analysis_stats["total_news_analyzed"] += len(news)
        self.analysis_stats["symbols_analyzed"].add(symbol)
        self.analysis_stats["last_update"] = timestamp
        
        # Mettre à jour l'historique
        if symbol not in self.news_history:
            self.news_history[symbol] = []
        
        # Ajouter les nouvelles actualités à l'historique
        self.news_history[symbol].extend(enriched_news)
        
        # Limiter la taille de l'historique
        max_news = self.settings["max_news_per_symbol"]
        if len(self.news_history[symbol]) > max_news:
            self.news_history[symbol] = self.news_history[symbol][-max_news:]
        
        # Sauvegarder l'historique
        if self.settings["save_history"]:
            self._save_history()
        
        return enriched_news
    
    def on_before_sentiment_analysis(self, symbol: str, news: List[Dict], analysis_method: str) -> Dict[str, Any]:
        """
        Appelé avant l'analyse de sentiment.
        Peut modifier les paramètres d'analyse.
        
        Args:
            symbol: Symbole concerné
            news: Actualités à analyser
            analysis_method: Méthode d'analyse
            
        Returns:
            Dictionnaire avec les paramètres modifiés
        """
        if not news:
            return {"news": news, "analysis_method": analysis_method}
        
        # Récupérer l'historique des actualités pour ce symbole
        symbol_history = self.news_history.get(symbol, [])
        
        # Si l'historique est vide ou désactivé, retourner les actualités telles quelles
        if not symbol_history or not self.settings["enable_trend_analysis"]:
            return {"news": news, "analysis_method": analysis_method}
        
        # Analyser les tendances précédentes
        trend_info = self._analyze_news_trend(symbol)
        
        # Enrichir les actualités avec le contexte des tendances
        enriched_news = []
        for item in news:
            enriched_item = item.copy()
            
            # Ajouter le contexte de tendance
            enriched_item["trend_context"] = {
                "recent_sentiment": trend_info["recent_sentiment"],
                "sentiment_change": trend_info["sentiment_change"],
                "volatility": trend_info["volatility"]
            }
            
            enriched_news.append(enriched_item)
        
        # Suggérer une méthode d'analyse basée sur la volatilité des actualités
        suggested_method = analysis_method
        if trend_info["volatility"] > 0.5:
            suggested_method = "ensemble"  # Méthode plus robuste pour les actualités volatiles
        
        return {
            "news": enriched_news,
            "analysis_method": suggested_method,
            "trend_info": trend_info
        }
    
    def on_after_sentiment_analysis(self, symbol: str, news: List[Dict], sentiment_results: Dict, overall_score: float) -> Dict[str, Any]:
        """
        Appelé après l'analyse de sentiment.
        Peut enrichir ou modifier les résultats de l'analyse.
        
        Args:
            symbol: Symbole concerné
            news: Actualités analysées
            sentiment_results: Résultats de l'analyse
            overall_score: Score global
            
        Returns:
            Dictionnaire avec les résultats enrichis
        """
        if not news or not sentiment_results:
            return sentiment_results
        
        # Récupérer l'historique des actualités pour ce symbole
        symbol_history = self.news_history.get(symbol, [])
        
        # Si l'historique est vide ou désactivé, retourner les résultats tels quels
        if not symbol_history or not self.settings["enable_trend_analysis"]:
            return sentiment_results
        
        # Analyser les tendances précédentes
        trend_info = self._analyze_news_trend(symbol)
        
        # Calculer les métriques de qualité
        quality_metrics = self._calculate_quality_metrics(news, sentiment_results)
        
        # Créer une analyse enrichie
        enriched_results = sentiment_results.copy()
        
        # Ajouter les métriques de qualité et les informations de tendance
        enriched_results.update({
            "quality_metrics": quality_metrics,
            "trend_analysis": trend_info,
            "enhanced_by": self.plugin_name
        })
        
        # Ajuster le score global en fonction des tendances historiques
        if trend_info["sentiment_change"] != 0:
            # Facteur de momentum: donner plus de poids aux changements récents
            momentum_factor = 0.2
            adjusted_score = overall_score * (1 + trend_info["sentiment_change"] * momentum_factor)
            
            # S'assurer que le score reste dans les limites [-1, 1]
            adjusted_score = max(-1.0, min(1.0, adjusted_score))
            
            enriched_results["adjusted_score"] = adjusted_score
            enriched_results["adjustment_reason"] = "Ajustement basé sur les tendances récentes"
        else:
            enriched_results["adjusted_score"] = overall_score
        
        return enriched_results
    
    def on_system_shutdown(self, reason: str, timestamp: datetime) -> None:
        """
        Appelé lors de l'arrêt du système.
        Sauvegarde les données du plugin.
        
        Args:
            reason: Raison de l'arrêt
            timestamp: Horodatage
        """
        logger.info(f"Sauvegarde des données du plugin {self.plugin_name} avant l'arrêt du système")
        
        # Sauvegarder l'historique
        if self.settings["save_history"]:
            self._save_history()
    
    def _detect_topics(self, text: str) -> List[str]:
        """
        Détecte les thèmes principaux dans un texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Liste des thèmes détectés
        """
        topics = []
        
        # Liste de thèmes financiers courants avec leurs mots-clés associés
        financial_topics = {
            "earnings": ["earnings", "revenue", "profit", "eps", "income", "quarter", "financial results"],
            "merger_acquisition": ["merger", "acquisition", "takeover", "buy", "purchase", "deal"],
            "product_launch": ["launch", "release", "introduce", "unveil", "new product"],
            "management": ["ceo", "executive", "management", "appointed", "resigned", "board"],
            "regulation": ["regulation", "compliance", "law", "legal", "sec", "regulator"],
            "market_trend": ["market", "trend", "sector", "industry", "performance"],
            "technology": ["technology", "tech", "innovation", "patent", "digital"],
            "economy": ["economy", "economic", "gdp", "inflation", "rate", "fed", "central bank"]
        }
        
        # Détecter les thèmes en comptant les mots-clés
        text_lower = text.lower()
        for topic, keywords in financial_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _analyze_news_trend(self, symbol: str) -> Dict[str, float]:
        """
        Analyse les tendances des actualités pour un symbole donné.
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Dictionnaire contenant les informations de tendance
        """
        # Récupérer l'historique des actualités pour ce symbole
        symbol_history = self.news_history.get(symbol, [])
        
        # Si l'historique est vide, retourner des valeurs par défaut
        if not symbol_history:
            return {
                "recent_sentiment": 0.0,
                "sentiment_change": 0.0,
                "volatility": 0.0,
                "news_count": 0
            }
        
        # Trier l'historique par date
        try:
            sorted_history = sorted(
                symbol_history, 
                key=lambda x: datetime.fromisoformat(x.get("analysis_timestamp", "2000-01-01T00:00:00")),
                reverse=True
            )
        except Exception as e:
            logger.error(f"Erreur lors du tri de l'historique: {e}")
            sorted_history = symbol_history
        
        # Diviser l'historique en périodes récentes et anciennes
        recent_period = sorted_history[:min(10, len(sorted_history))]
        older_period = sorted_history[min(10, len(sorted_history)):min(20, len(sorted_history))]
        
        # Calculer le sentiment moyen pour chaque période
        recent_sentiment = self._calculate_average_sentiment(recent_period)
        older_sentiment = self._calculate_average_sentiment(older_period) if older_period else recent_sentiment
        
        # Calculer le changement de sentiment
        sentiment_change = recent_sentiment - older_sentiment
        
        # Calculer la volatilité (écart-type des scores de sentiment)
        sentiment_scores = [
            item.get("impact_details", {}).get("final_impact_score", 0)
            for item in recent_period
            if "impact_details" in item
        ]
        
        volatility = 0.0
        if sentiment_scores:
            import statistics
            try:
                volatility = statistics.stdev(sentiment_scores)
            except Exception as e:
                logger.error(f"Erreur lors du calcul de la volatilité: {e}")
        
        return {
            "recent_sentiment": recent_sentiment,
            "sentiment_change": sentiment_change,
            "volatility": volatility,
            "news_count": len(recent_period)
        }
    
    def _calculate_average_sentiment(self, news_items: List[Dict]) -> float:
        """
        Calcule le sentiment moyen pour une liste d'actualités.
        
        Args:
            news_items: Liste d'actualités
            
        Returns:
            Score de sentiment moyen
        """
        if not news_items:
            return 0.0
        
        # Extraire les scores d'impact
        impact_scores = []
        for item in news_items:
            if "impact_details" in item and "final_impact_score" in item["impact_details"]:
                impact_scores.append(item["impact_details"]["final_impact_score"])
            elif "impact" in item and item["impact"] in ["positive", "negative", "neutral"]:
                # Convertir l'impact en valeur numérique
                impact_map = {"positive": 0.7, "negative": -0.7, "neutral": 0.0}
                impact_scores.append(impact_map[item["impact"]])
        
        # Calculer la moyenne
        return sum(impact_scores) / len(impact_scores) if impact_scores else 0.0
    
    def _calculate_quality_metrics(self, news: List[Dict], sentiment_results: Dict) -> Dict[str, Any]:
        """
        Calcule des métriques de qualité pour l'analyse de sentiment.
        
        Args:
            news: Actualités analysées
            sentiment_results: Résultats de l'analyse
            
        Returns:
            Dictionnaire des métriques de qualité
        """
        # Extraire les scores d'impact
        impact_scores = []
        for item in news:
            if "impact_details" in item and "final_impact_score" in item["impact_details"]:
                impact_scores.append(item["impact_details"]["final_impact_score"])
        
        # Compter les actualités à haut impact
        high_impact_count = sum(1 for item in news if item.get("high_impact", False))
        
        # Calculer les métriques
        metrics = {
            "news_count": len(news),
            "high_impact_count": high_impact_count,
            "high_impact_ratio": high_impact_count / len(news) if news else 0,
            "coverage": min(1.0, len(news) / 10)  # Ratio de couverture (10 news = 100%)
        }
        
        # Ajouter des mesures de dispersion si nous avons des scores
        if impact_scores:
            import statistics
            try:
                metrics.update({
                    "mean_impact": statistics.mean(impact_scores),
                    "std_dev": statistics.stdev(impact_scores) if len(impact_scores) > 1 else 0,
                    "min_impact": min(impact_scores),
                    "max_impact": max(impact_scores)
                })
            except Exception as e:
                logger.error(f"Erreur lors du calcul des statistiques d'impact: {e}")
        
        return metrics
    
    def _save_history(self) -> None:
        """Sauvegarde l'historique des actualités dans un fichier JSON"""
        try:
            # Limiter l'historique par date
            max_days = self.settings.get("history_days", 30)
            cutoff_date = (datetime.now() - timedelta(days=max_days)).isoformat()
            
            # Filtrer l'historique par date
            filtered_history = {}
            for symbol, news_list in self.news_history.items():
                filtered_news = [
                    item for item in news_list
                    if item.get("analysis_timestamp", "9999-12-31") > cutoff_date
                ]
                
                if filtered_news:
                    filtered_history[symbol] = filtered_news
            
            # Sauvegarder dans un fichier JSON
            with open(self.history_file, 'w') as f:
                # Convertir les ensembles en listes pour la sérialisation JSON
                stats_copy = self.analysis_stats.copy()
                stats_copy["symbols_analyzed"] = list(stats_copy["symbols_analyzed"])
                
                json.dump({
                    "history": filtered_history,
                    "stats": stats_copy,
                    "last_saved": datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Historique sauvegardé pour {len(filtered_history)} symboles")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'historique: {e}")
    
    def _load_history(self) -> None:
        """Charge l'historique des actualités depuis un fichier JSON"""
        if not os.path.exists(self.history_file):
            logger.info("Aucun historique trouvé, création d'un nouvel historique")
            return
        
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                
                # Charger l'historique
                self.news_history = data.get("history", {})
                
                # Charger les statistiques
                stats = data.get("stats", {})
                if stats:
                    # Convertir les listes en ensembles
                    if "symbols_analyzed" in stats:
                        stats["symbols_analyzed"] = set(stats["symbols_analyzed"])
                    
                    self.analysis_stats = stats
                
                logger.info(f"Historique chargé pour {len(self.news_history)} symboles")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'historique: {e}")
            self.news_history = {} 