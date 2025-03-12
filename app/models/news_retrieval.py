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
        # Use os.environ.get instead of os.getenv for consistency with other files
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        
        if not self.openrouter_api_key:
            logging.warning("OPENROUTER_API_KEY non configurée. La récupération des news via OpenRouter et Perplexity Sonar sera désactivée.")
        else:
            # Log a portion of the key for debugging (only first 8 chars)
            key_preview = self.openrouter_api_key[:8] + "..." if len(self.openrouter_api_key) > 8 else "invalid"
            logging.info(f"OpenRouter API key configured: {key_preview}...")
        
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
                "Content-Type": "application/json",
                "HTTP-Referer": "https://evil2root.ai/",  # Required by OpenRouter
                "X-Title": "Evil2Root Trading AI"  # Helps OpenRouter identify your app
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
                "summary": "Bref résumé de la nouvelle",
                "impact": "Potentiel impact sur le prix (positive, negative, neutral)",
                "relevance_score": 0.0-1.0
              }},
              ...
            ]
            
            Pour chaque nouvelle, analyse son impact potentiel sur le prix du titre et attribue un score de pertinence de 0 à 1.
            Assure-toi de trouver les nouvelles les plus pertinentes qui pourraient directement affecter la valeur de {symbol}.
            Concentre-toi particulièrement sur: résultats financiers, fusions/acquisitions, changements réglementaires, lancements de produits, et analyses d'analystes.
            
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
                "Content-Type": "application/json",
                "HTTP-Referer": "https://evil2root.ai/",  # Required by OpenRouter
                "X-Title": "Evil2Root Trading AI"  # Helps OpenRouter identify your app
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
                "summary": "Bref résumé de la nouvelle",
                "impact": "Potentiel impact sur le prix (positive, negative, neutral)",
                "relevance_score": 0.0-1.0
              }},
              ...
            ]
            
            Pour chaque nouvelle, analyse son impact potentiel sur le prix du titre et attribue un score de pertinence de 0 à 1.
            Assure-toi de trouver les nouvelles les plus pertinentes qui pourraient directement affecter la valeur de {symbol}.
            Concentre-toi particulièrement sur: résultats financiers, fusions/acquisitions, changements réglementaires, lancements de produits, et analyses d'analystes.
            
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

    def fetch_financial_news_api(self, symbol: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les news depuis des API financières spécialisées (alphavantage, marketaux, etc)
        
        Args:
            symbol: Symbole boursier
            max_results: Nombre maximum de résultats
            
        Returns:
            Liste de dictionnaires contenant les news
        """
        cache_key = f"financial_api_{symbol}_{max_results}"
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > time.time():
            logging.info(f"Utilisation du cache pour les news API financières de {symbol}")
            return self.cache[cache_key]
            
        # Clés API pour différents services
        alphavantage_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
        marketaux_key = os.getenv("MARKETAUX_API_KEY", "")
        finnhub_key = os.getenv("FINNHUB_API_KEY", "")
        
        news_items = []
        
        # 1. Essayer Alphavantage si disponible
        if alphavantage_key:
            try:
                url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={alphavantage_key}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if "feed" in data:
                        for item in data["feed"][:max_results]:
                            news_items.append({
                                "title": item.get("title", ""),
                                "date": item.get("time_published", "")[:10],
                                "source": item.get("source", ""),
                                "url": item.get("url", ""),
                                "summary": item.get("summary", ""),
                                "impact": "positive" if item.get("overall_sentiment_score", 0) > 0 else 
                                          ("negative" if item.get("overall_sentiment_score", 0) < 0 else "neutral"),
                                "relevance_score": min(1.0, max(0.0, item.get("relevance_score", 0.5)))
                            })
            except Exception as e:
                logging.error(f"Erreur lors de la récupération des news via Alphavantage: {e}")
                
        # 2. Essayer Marketaux si disponible
        if marketaux_key and len(news_items) < max_results:
            try:
                url = f"https://api.marketaux.com/v1/news/all?symbols={symbol}&language=en&api_token={marketaux_key}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data:
                        for item in data["data"][:max_results - len(news_items)]:
                            news_items.append({
                                "title": item.get("title", ""),
                                "date": item.get("published_at", "")[:10],
                                "source": item.get("source", ""),
                                "url": item.get("url", ""),
                                "summary": item.get("description", ""),
                                "impact": "positive" if item.get("sentiment", "") == "positive" else 
                                          ("negative" if item.get("sentiment", "") == "negative" else "neutral"),
                                "relevance_score": 0.8  # Valeur par défaut
                            })
            except Exception as e:
                logging.error(f"Erreur lors de la récupération des news via Marketaux: {e}")
                
        # 3. Essayer Finnhub si disponible
        if finnhub_key and len(news_items) < max_results:
            try:
                url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={datetime.now().date() - timedelta(days=7)}&to={datetime.now().date()}&token={finnhub_key}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, list):
                        for item in data[:max_results - len(news_items)]:
                            news_items.append({
                                "title": item.get("headline", ""),
                                "date": datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d"),
                                "source": item.get("source", ""),
                                "url": item.get("url", ""),
                                "summary": item.get("summary", ""),
                                "impact": "neutral",  # Finnhub ne fournit pas de sentiment par défaut
                                "relevance_score": 0.7  # Valeur par défaut
                            })
            except Exception as e:
                logging.error(f"Erreur lors de la récupération des news via Finnhub: {e}")
        
        # Mise en cache des résultats
        if news_items:
            self.cache[cache_key] = news_items
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
        return news_items
    
    def get_combined_news(self, symbol: str, max_results: int = 15) -> List[Dict[str, Any]]:
        """
        Combine les résultats de toutes les sources de news pour obtenir une vue complète
        
        Args:
            symbol: Symbole boursier
            max_results: Nombre maximum de résultats au total
            
        Returns:
            Liste combinée de news provenant de toutes les sources
        """
        # Limiter le nombre de résultats par source
        per_source = max(5, max_results // 3)
        
        # Récupérer les news des différentes sources
        claude_news = self.get_news_openrouter(symbol, per_source)
        sonar_news = self.get_news_perplexity_sonar(symbol, per_source)
        api_news = self.fetch_financial_news_api(symbol, per_source)
        
        # Combiner les résultats en évitant les doublons (basé sur les titres)
        combined_news = claude_news.copy()
        titles = {news["title"].lower() for news in combined_news}
        
        # Ajouter les news de Sonar
        for news in sonar_news:
            if news["title"].lower() not in titles:
                combined_news.append(news)
                titles.add(news["title"].lower())
                
                # Limiter au nombre maximum demandé
                if len(combined_news) >= max_results:
                    break
        
        # Ajouter les news des API financières
        for news in api_news:
            if news["title"].lower() not in titles:
                combined_news.append(news)
                titles.add(news["title"].lower())
                
                # Limiter au nombre maximum demandé
                if len(combined_news) >= max_results:
                    break
        
        # Trier les nouvelles par pertinence_score (de la plus pertinente à la moins pertinente)
        combined_news.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
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
        
    def get_news_with_sentiment_data(self, symbol: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Récupère les news avec des données de sentiment pré-analysées
        
        Args:
            symbol: Symbole boursier
            max_results: Nombre maximum de résultats
            
        Returns:
            Liste de dictionnaires contenant les news avec des infos de sentiment
        """
        news = self.get_combined_news(symbol, max_results)
        
        # Enrichir avec des métadonnées supplémentaires pour l'analyse de sentiment
        for item in news:
            # Convertir l'impact en score numérique pour le sentiment analyzer
            impact = item.get("impact", "neutral").lower()
            if impact == "positive":
                item["sentiment_score"] = 0.7
            elif impact == "negative":
                item["sentiment_score"] = -0.7
            else:
                item["sentiment_score"] = 0.0
                
            # Ajouter le texte complet pour l'analyse de sentiment
            item["full_text"] = f"{item.get('title', '')}. {item.get('summary', '')}"
        
        return news

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