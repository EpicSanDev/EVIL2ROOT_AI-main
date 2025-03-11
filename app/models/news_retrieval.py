import logging
import requests
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time

class NewsRetriever:
    """
    Classe pour récupérer des nouvelles financières à l'aide d'OpenRouter (incluant l'accès à Perplexity Sonar)
    """
    
    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        
        if not self.openrouter_api_key:
            logging.warning("OPENROUTER_API_KEY non configurée. La récupération des news via OpenRouter et Perplexity Sonar sera désactivée.")
        
        # Cache pour éviter de multiples requêtes pour les mêmes informations
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 3600  # 1 heure
        
    def get_news_openrouter(self, symbol: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les news financières pour un symbole donné via OpenRouter (Claude)
        
        Args:
            symbol: Symbole boursier (ex: AAPL, BTC-USD)
            max_results: Nombre maximum de résultats à retourner
            
        Returns:
            Liste de dictionnaires contenant les news (titre, source, date, url, résumé)
        """
        cache_key = f"openrouter_claude_{symbol}_{max_results}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            logging.info(f"Utilisation du cache pour les news OpenRouter Claude de {symbol}")
            return self.cache[cache_key]
        
        if not self.openrouter_api_key:
            logging.error("Clé API OpenRouter non configurée")
            return []
        
        try:
            # Construction de la requête pour OpenRouter
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            # Construction du prompt pour l'API OpenRouter (Claude)
            current_date = datetime.now().strftime("%Y-%m-%d")
            prompt = f"""Récupère les {max_results} dernières nouvelles financières importantes sur {symbol} (au {current_date}).
            
            Format de réponse attendu:
            [
              {{
                "title": "Titre de la nouvelle",
                "date": "YYYY-MM-DD",
                "source": "Nom de la source",
                "url": "URL de la source (si disponible)",
                "summary": "Bref résumé de la nouvelle"
              }},
              ...
            ]
            
            Retourne uniquement le JSON sans aucun texte supplémentaire.
            """
            
            data = {
                "model": "anthropic/claude-3-opus-20240229",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                logging.error(f"Erreur OpenRouter API: {response.status_code} - {response.text}")
                return []
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Parse le JSON retourné (peut être sous forme de chaîne)
            if isinstance(content, str):
                try:
                    news_data = json.loads(content)
                    if isinstance(news_data, dict) and "news" in news_data:
                        news_items = news_data["news"]
                    else:
                        news_items = news_data
                except json.JSONDecodeError:
                    logging.error(f"Erreur de parsing JSON: {content}")
                    return []
            else:
                news_items = content
            
            # S'assurer que nous avons une liste de news
            if not isinstance(news_items, list):
                logging.error(f"Format de réponse OpenRouter incorrect: {news_items}")
                return []
            
            # Mise en cache des résultats
            self.cache[cache_key] = news_items
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return news_items
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des news via OpenRouter: {e}")
            return []
    
    def get_news_perplexity_sonar(self, symbol: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les news financières pour un symbole donné via OpenRouter (accès à Perplexity Sonar)
        
        Args:
            symbol: Symbole boursier (ex: AAPL, BTC-USD)
            max_results: Nombre maximum de résultats à retourner
            
        Returns:
            Liste de dictionnaires contenant les news (titre, source, date, url, résumé)
        """
        cache_key = f"openrouter_sonar_{symbol}_{max_results}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            logging.info(f"Utilisation du cache pour les news Perplexity Sonar de {symbol}")
            return self.cache[cache_key]
        
        if not self.openrouter_api_key:
            logging.error("Clé API OpenRouter non configurée")
            return []
        
        try:
            # Construction de la requête pour OpenRouter (avec accès à Perplexity Sonar)
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            # Construction du prompt pour obtenir des informations web actuelles via Sonar
            prompt = f"""Utilise Perplexity Sonar pour rechercher les dernières nouvelles financières importantes concernant {symbol}.
            
            Retourne les informations sous ce format JSON:
            [
              {{
                "title": "Titre de la nouvelle",
                "date": "YYYY-MM-DD",
                "source": "Nom de la source",
                "url": "URL de la source",
                "summary": "Bref résumé de la nouvelle"
              }},
              ...
            ]
            
            Limite-toi à {max_results} résultats maximum, focus sur les sources financières fiables.
            Recherche spécifiquement les nouvelles les plus récentes qui pourraient affecter le prix/valeur de {symbol}.
            """
            
            data = {
                "model": "perplexity/sonar-medium-online",  # Utilisation du modèle Sonar via OpenRouter
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                logging.error(f"Erreur OpenRouter API (Sonar): {response.status_code} - {response.text}")
                return []
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Parse le JSON retourné
            if isinstance(content, str):
                try:
                    news_data = json.loads(content)
                    if isinstance(news_data, dict) and "news" in news_data:
                        news_items = news_data["news"]
                    elif isinstance(news_data, list):
                        news_items = news_data
                    else:
                        news_items = []
                except json.JSONDecodeError:
                    logging.error(f"Erreur de parsing JSON (Sonar): {content}")
                    return []
            else:
                news_items = content
            
            # S'assurer que nous avons une liste de news
            if not isinstance(news_items, list):
                logging.error(f"Format de réponse Sonar incorrect: {news_items}")
                return []
            
            # Mise en cache des résultats
            self.cache[cache_key] = news_items
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return news_items
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des news via Perplexity Sonar: {e}")
            return []
    
    def get_combined_news(self, symbol: str, max_results: int = 15) -> List[Dict[str, Any]]:
        """
        Combine les résultats d'OpenRouter (Claude) et OpenRouter (Perplexity Sonar) pour obtenir des news plus complètes
        
        Args:
            symbol: Symbole boursier
            max_results: Nombre maximum de résultats au total
            
        Returns:
            Liste combinée de news provenant des deux sources
        """
        # Limiter le nombre de résultats par source
        per_source = max(5, max_results // 2)
        
        # Récupérer les news des deux sources
        claude_news = self.get_news_openrouter(symbol, per_source)
        sonar_news = self.get_news_perplexity_sonar(symbol, per_source)
        
        # Combiner les résultats en évitant les doublons (basé sur les titres)
        combined_news = claude_news.copy()
        titles = {news["title"].lower() for news in combined_news}
        
        for news in sonar_news:
            if news["title"].lower() not in titles:
                combined_news.append(news)
                titles.add(news["title"].lower())
                
                # Limiter au nombre maximum demandé
                if len(combined_news) >= max_results:
                    break
        
        return combined_news[:max_results]

    def get_news_headlines(self, symbol: str, max_results: int = 10) -> List[str]:
        """
        Récupère uniquement les titres des news pour un symbole donné
        
        Args:
            symbol: Symbole boursier
            max_results: Nombre maximum de résultats
            
        Returns:
            Liste de titres de news
        """
        news = self.get_combined_news(symbol, max_results)
        return [item["title"] for item in news]

# Fonction utilitaire pour l'utilisation directe
def get_news_for_symbol(symbol: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Récupère les news pour un symbole donné en utilisant toutes les sources disponibles
    
    Args:
        symbol: Symbole boursier (ex: AAPL, BTC-USD)
        max_results: Nombre maximum de résultats
        
    Returns:
        Liste de dictionnaires contenant les news
    """
    retriever = NewsRetriever()
    return retriever.get_combined_news(symbol, max_results) 