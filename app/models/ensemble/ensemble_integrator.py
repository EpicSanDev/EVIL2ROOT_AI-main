import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from app.models.ensemble_model import EnsembleModel
from app.models.price_prediction import PricePredictionModel
from app.models.transformer_model import TransformerModel
from app.models.indicator_management import IndicatorManagementModel
from app.models.risk_management import RiskManagementModel
from app.models.tp_sl_management import TpSlManagementModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class EnsembleIntegrator:
    """
    Integrates ensemble models with the existing trading system.
    Acts as a bridge between ensemble models and the trading bot.
    """
    
    def __init__(self, model_dir='saved_models', use_explainable_ai=True):
        """
        Initialize the ensemble integrator.
        
        Args:
            model_dir: Directory for models
            use_explainable_ai: Whether to use explainable AI features
        """
        self.model_dir = model_dir
        self.use_explainable_ai = use_explainable_ai
        
        # Ensemble models for different prediction tasks
        self.price_ensemble = None
        self.direction_ensemble = None
        self.volatility_ensemble = None
        
        # Explanations cache
        self.explanations = {}
        
        # Results cache
        self.last_predictions = {}
        self.last_explanations = {}
        
        # Initialize ensemble models
        self._initialize_models()
        
        logger.info(f"Initialized EnsembleIntegrator with explainable AI: {use_explainable_ai}")
    
    def _initialize_models(self):
        """Initialize different ensemble models for various trading tasks"""
        try:
            # Price prediction ensemble (regression)
            self.price_ensemble = EnsembleModel(
                model_dir=os.path.join(self.model_dir, 'price_ensemble'),
                ensemble_type='stacking',
                use_shap=self.use_explainable_ai,
                use_lime=False
            )
            
            # Direction prediction ensemble (classification)
            self.direction_ensemble = EnsembleModel(
                model_dir=os.path.join(self.model_dir, 'direction_ensemble'),
                ensemble_type='voting',
                use_shap=self.use_explainable_ai,
                use_lime=False
            )
            
            # Volatility prediction ensemble (regression)
            self.volatility_ensemble = EnsembleModel(
                model_dir=os.path.join(self.model_dir, 'volatility_ensemble'),
                ensemble_type='stacking',
                use_shap=self.use_explainable_ai,
                use_lime=False
            )
            
            logger.info("Successfully initialized all ensemble models")
            
        except Exception as e:
            logger.error(f"Error initializing ensemble models: {e}")
    
    def train_models(self, 
                    data: Dict[str, pd.DataFrame], 
                    symbols: List[str],
                    existing_models: Dict[str, Any]):
        """
        Train ensemble models using data and existing models as a foundation.
        
        Args:
            data: Dictionary of dataframes by symbol
            symbols: List of symbols to train for
            existing_models: Dictionary of existing models (price, indicator, etc.)
            
        Returns:
            Dictionary of training results
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Training ensemble models for {symbol}")
            symbol_results = {}
            
            try:
                if symbol not in data:
                    logger.warning(f"No data found for {symbol}, skipping training")
                    continue
                
                # Get data for this symbol
                symbol_data = data[symbol].copy()
                
                # 1. Prepare data for price prediction (target = next day close)
                price_data = symbol_data.copy()
                price_data['NextClose'] = price_data['Close'].shift(-1)
                price_data.dropna(inplace=True)
                
                # Train price prediction ensemble
                price_result = self.price_ensemble.train(
                    data=price_data,
                    symbol=symbol,
                    target_col='NextClose',
                    classification=False
                )
                symbol_results['price'] = price_result
                
                # 2. Prepare data for direction prediction (target = price direction)
                direction_data = symbol_data.copy()
                direction_data['Direction'] = np.where(
                    direction_data['Close'].shift(-1) > direction_data['Close'], 1, 0
                )
                direction_data.dropna(inplace=True)
                
                # Train direction prediction ensemble
                direction_result = self.direction_ensemble.train(
                    data=direction_data,
                    symbol=symbol,
                    target_col='Direction',
                    classification=True
                )
                symbol_results['direction'] = direction_result
                
                # 3. Prepare data for volatility prediction (target = next day range/close)
                volatility_data = symbol_data.copy()
                volatility_data['NextVolatility'] = (symbol_data['High'].shift(-1) - symbol_data['Low'].shift(-1)) / symbol_data['Close'].shift(-1)
                volatility_data.dropna(inplace=True)
                
                # Train volatility prediction ensemble
                volatility_result = self.volatility_ensemble.train(
                    data=volatility_data,
                    symbol=symbol,
                    target_col='NextVolatility',
                    classification=False
                )
                symbol_results['volatility'] = volatility_result
                
                # Store results for this symbol
                results[symbol] = symbol_results
                logger.info(f"Completed ensemble training for {symbol}")
                
            except Exception as e:
                logger.error(f"Error training ensemble models for {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def predict(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Generate comprehensive predictions using ensemble models.
        
        Args:
            data: Market data for prediction
            symbol: Trading symbol
            
        Returns:
            Dictionary with predictions and explanations
        """
        logger.info(f"Generating ensemble predictions for {symbol}")
        result = {
            'price': None,
            'direction': None,
            'volatility': None,
            'explanations': {},
            'confidence': 0.0,
            'recommendation': None,
            'risk_level': 'medium',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            # 1. Generate price predictions
            price_predictions, price_explanations = self.price_ensemble.predict(data, symbol)
            
            # 2. Generate direction predictions
            direction_predictions, direction_explanations = self.direction_ensemble.predict(data, symbol)
            
            # 3. Generate volatility predictions
            volatility_predictions, volatility_explanations = self.volatility_ensemble.predict(data, symbol)
            
            # Store predictions
            if price_predictions is not None:
                result['price'] = float(price_predictions[-1])
                
            if direction_predictions is not None:
                result['direction'] = int(direction_predictions[-1])
                
            if volatility_predictions is not None:
                result['volatility'] = float(volatility_predictions[-1])
            
            # Store explanations
            if price_explanations is not None:
                result['explanations']['price'] = price_explanations
                
            if direction_explanations is not None:
                result['explanations']['direction'] = direction_explanations
                
            if volatility_explanations is not None:
                result['explanations']['volatility'] = volatility_explanations
            
            # Generate trading recommendation
            result['recommendation'] = self._generate_recommendation(
                price=result['price'],
                direction=result['direction'],
                volatility=result['volatility'],
                current_price=data['Close'].iloc[-1]
            )
            
            # Calculate confidence score (simplified)
            if direction_predictions is not None and price_predictions is not None:
                # Calculate confidence based on direction consistency and volatility
                direction_confidence = 0.5 + 0.5 * abs(result['direction'] - 0.5) * 2  # Scale 0.5-1.0
                
                # Price movement magnitude (normalized)
                if result['price'] is not None and 'Close' in data.columns:
                    price_change_pct = abs((result['price'] - data['Close'].iloc[-1]) / data['Close'].iloc[-1])
                    price_confidence = min(price_change_pct * 10, 1.0)  # Scale 0-1.0 based on predicted move
                else:
                    price_confidence = 0.5
                
                # Combined confidence score
                result['confidence'] = 0.6 * direction_confidence + 0.4 * price_confidence
            
            # Risk level assessment
            if result['volatility'] is not None:
                if result['volatility'] < 0.01:  # Low volatility
                    result['risk_level'] = 'low'
                elif result['volatility'] > 0.03:  # High volatility
                    result['risk_level'] = 'high'
                else:
                    result['risk_level'] = 'medium'
            
            # Cache results
            self.last_predictions[symbol] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating ensemble predictions for {symbol}: {e}")
            return {
                'error': str(e),
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def _generate_recommendation(self, price, direction, volatility, current_price):
        """
        Generate a trading recommendation based on model predictions.
        
        Args:
            price: Predicted price
            direction: Predicted direction (1 = up, 0 = down)
            volatility: Predicted volatility
            current_price: Current price
            
        Returns:
            Trading recommendation string
        """
        if price is None or direction is None:
            return "HOLD - Insufficient prediction data"
        
        # Default recommendation
        recommendation = "HOLD"
        
        # Direction-based recommendation
        if direction == 1:  # Predicted up
            recommendation = "BUY"
        else:  # Predicted down
            recommendation = "SELL"
        
        # Add strength based on predicted move size
        if price is not None and current_price is not None:
            predicted_change_pct = (price - current_price) / current_price * 100
            
            # Adjust recommendation strength based on predicted change
            if abs(predicted_change_pct) < 0.5:
                strength = "weak"
            elif abs(predicted_change_pct) < 1.5:
                strength = "moderate"
            else:
                strength = "strong"
            
            recommendation = f"{recommendation} - {strength} ({predicted_change_pct:.2f}%)"
        
        # If volatility is very high, adjust recommendation to be more cautious
        if volatility is not None and volatility > 0.05:  # 5% volatility
            recommendation += " [HIGH VOLATILITY - Use Caution]"
        
        return recommendation
    
    def generate_explanation_report(self, symbol: str, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation report for a symbol.
        
        Args:
            symbol: Trading symbol
            data: Optional recent data for new predictions
            
        Returns:
            Dictionary with explanation report data
        """
        report = {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'price_explanation': None,
            'direction_explanation': None,
            'volatility_explanation': None,
            'combined_explanation': None,
            'top_features': {},
            'plots': {}
        }
        
        try:
            # Generate new predictions if data is provided
            if data is not None:
                self.predict(data, symbol)
            
            # Get explanation reports from each model
            price_report = self.price_ensemble.generate_explanation_report(data, symbol)
            direction_report = self.direction_ensemble.generate_explanation_report(data, symbol)
            volatility_report = self.volatility_ensemble.generate_explanation_report(data, symbol)
            
            # Combine top features
            if 'top_features' in price_report:
                report['top_features']['price'] = price_report['top_features']
            
            if 'top_features' in direction_report:
                report['top_features']['direction'] = direction_report['top_features']
            
            if 'top_features' in volatility_report:
                report['top_features']['volatility'] = volatility_report['top_features']
            
            # Add explanation texts
            if 'explanation_text' in price_report:
                report['price_explanation'] = price_report['explanation_text']
            
            if 'explanation_text' in direction_report:
                report['direction_explanation'] = direction_report['explanation_text']
            
            if 'explanation_text' in volatility_report:
                report['volatility_explanation'] = volatility_report['explanation_text']
            
            # Add plots
            if 'explanation_plots' in price_report:
                report['plots']['price'] = price_report['explanation_plots']
            
            if 'explanation_plots' in direction_report:
                report['plots']['direction'] = direction_report['explanation_plots']
            
            if 'explanation_plots' in volatility_report:
                report['plots']['volatility'] = volatility_report['explanation_plots']
            
            # Generate combined explanation
            report['combined_explanation'] = self._generate_combined_explanation(
                symbol, report['price_explanation'], report['direction_explanation'], 
                report['volatility_explanation'], report['top_features']
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating explanation report for {symbol}: {e}")
            return {'error': str(e)}
    
    def _generate_combined_explanation(self, symbol, price_expl, direction_expl, volatility_expl, top_features):
        """
        Generate a combined natural language explanation from individual model explanations.
        
        Args:
            symbol: Trading symbol
            price_expl: Price model explanation
            direction_expl: Direction model explanation
            volatility_expl: Volatility model explanation
            top_features: Dictionary of top features
            
        Returns:
            Combined explanation text
        """
        explanation = f"# Trading Decision Explanation for {symbol}\n\n"
        
        # Get last prediction if available
        if symbol in self.last_predictions:
            pred = self.last_predictions[symbol]
            explanation += f"## Decision Summary\n"
            explanation += f"Recommendation: {pred.get('recommendation', 'Unknown')}\n"
            explanation += f"Confidence: {pred.get('confidence', 0.0) * 100:.1f}%\n"
            explanation += f"Risk Level: {pred.get('risk_level', 'Unknown').upper()}\n\n"
        
        # Add price prediction explanation
        explanation += f"## Price Prediction\n"
        if price_expl:
            explanation += f"{price_expl}\n\n"
        else:
            explanation += "No detailed explanation available for price prediction.\n\n"
        
        # Add direction prediction explanation
        explanation += f"## Direction Prediction\n"
        if direction_expl:
            explanation += f"{direction_expl}\n\n"
        else:
            explanation += "No detailed explanation available for direction prediction.\n\n"
        
        # Add volatility prediction explanation
        explanation += f"## Volatility Prediction\n"
        if volatility_expl:
            explanation += f"{volatility_expl}\n\n"
        else:
            explanation += "No detailed explanation available for volatility prediction.\n\n"
        
        # Add top features section
        explanation += f"## Most Important Features Overall\n"
        
        # Combine and rank all features
        all_features = {}
        
        # Add price features
        if 'price' in top_features:
            for feature, importance in top_features['price'].items():
                if feature in all_features:
                    all_features[feature] += importance
                else:
                    all_features[feature] = importance
        
        # Add direction features
        if 'direction' in top_features:
            for feature, importance in top_features['direction'].items():
                if feature in all_features:
                    all_features[feature] += importance
                else:
                    all_features[feature] = importance
        
        # Add volatility features
        if 'volatility' in top_features:
            for feature, importance in top_features['volatility'].items():
                if feature in all_features:
                    all_features[feature] += importance
                else:
                    all_features[feature] = importance
        
        # Sort and take top 10
        top_10_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for feature, importance in top_10_features:
            explanation += f"- {feature}: {importance:.4f}\n"
        
        explanation += "\n## Technical Interpretation\n"
        explanation += "This combined analysis integrates price movement predictions with direction probabilities "
        explanation += "and volatility forecasts to provide a holistic view of the expected market behavior. "
        explanation += "The decision algorithm weighs these factors while considering current market conditions "
        explanation += "to generate the most appropriate trading recommendation."
        
        return explanation
    
    def update_trading_bot(self, trading_bot):
        """
        Met à jour le bot de trading avec les modèles d'ensemble prédictifs.
        
        Args:
            trading_bot: Instance du bot de trading à mettre à jour
            
        Returns:
            Bot de trading mis à jour
        """
        logger.info("Intégration des modèles d'ensemble au bot de trading")
        
        # Ajouter l'instance ensemble_model au trading_bot
        if not hasattr(trading_bot, 'ensemble_model'):
            setattr(trading_bot, 'ensemble_model', self)
            logger.info("Attribut ensemble_model ajouté au trading_bot")
        else:
            trading_bot.ensemble_model = self
            logger.info("Attribut ensemble_model mis à jour sur le trading_bot")
        
        # Étendre la méthode get_model_predictions pour inclure les prédictions de l'ensemble
        original_get_model_predictions = trading_bot.get_model_predictions
        
        def enhanced_get_model_predictions(symbol):
            """Version améliorée de get_model_predictions avec les prédictions d'ensemble"""
            # Obtenir les prédictions des modèles standards
            predictions = original_get_model_predictions(symbol)
            
            if predictions is None:
                predictions = {}
                
            try:
                # Ajouter les prédictions d'ensemble si les données de marché existent
                if symbol in trading_bot.market_data and not trading_bot.market_data[symbol].empty:
                    data = trading_bot.market_data[symbol].tail(30).copy()
                    
                    # Générer les prédictions d'ensemble
                    ensemble_result = self.predict(data, symbol)
                    
                    if ensemble_result:
                        # Ajouter les prédictions d'ensemble au dictionnaire de résultats
                        predictions['ensemble_price'] = ensemble_result.get('price')
                        predictions['ensemble_direction'] = ensemble_result.get('direction')
                        predictions['ensemble_volatility'] = ensemble_result.get('volatility')
                        predictions['ensemble_confidence'] = ensemble_result.get('confidence')
                        predictions['ensemble_recommendation'] = ensemble_result.get('recommendation')
                        predictions['ensemble_risk_level'] = ensemble_result.get('risk_level')
                        
                        # Générer un rapport d'explication si l'IA explicable est activée
                        if self.use_explainable_ai:
                            explanation = self.generate_explanation_report(symbol, data)
                            predictions['ensemble_explanation'] = explanation
                        
                        logger.info(f"Prédictions d'ensemble ajoutées pour {symbol}")
            except Exception as e:
                logger.error(f"Erreur lors de l'ajout des prédictions d'ensemble pour {symbol}: {e}")
                
            return predictions
        
        # Remplacer la méthode par la version améliorée
        trading_bot.get_model_predictions = enhanced_get_model_predictions
        
        # Étendre la méthode combine_signals pour utiliser les prédictions d'ensemble
        original_combine_signals = trading_bot.combine_signals
        
        def enhanced_combine_signals(predicted_price, indicator_signal, risk_score, tp, sl, rl_signal, sentiment_score):
            """Version améliorée de combine_signals qui intègre les prédictions d'ensemble"""
            # Appeler d'abord la méthode originale
            original_decision = original_combine_signals(predicted_price, indicator_signal, risk_score, tp, sl, rl_signal, sentiment_score)
            
            # Récupérer le symbole actuel (contexte de l'appel)
            # Nous devons extraire le symbole du contexte actuel de trading_bot
            current_symbol = None
            if hasattr(trading_bot, 'current_symbol'):
                current_symbol = trading_bot.current_symbol
                
            # Si nous n'avons pas de symbole ou pas de prédictions d'ensemble,
            # retourner simplement la décision originale
            if not current_symbol:
                return original_decision
                
            try:
                # Récupérer les prédictions d'ensemble pour ce symbole
                predictions = trading_bot.get_model_predictions(current_symbol)
                
                if predictions and 'ensemble_recommendation' in predictions:
                    ensemble_recommendation = predictions['ensemble_recommendation']
                    ensemble_confidence = predictions.get('ensemble_confidence', 0.5)
                    
                    # Si la confiance de l'ensemble est élevée (>0.7), utiliser sa recommandation
                    if ensemble_confidence > 0.7:
                        logger.info(f"Utilisation de la recommandation d'ensemble avec confiance élevée: {ensemble_recommendation}")
                        return ensemble_recommendation
                    
                    # Sinon, combiner les deux décisions
                    # Si les deux sont d'accord, utiliser cette décision
                    if ensemble_recommendation == original_decision:
                        logger.info(f"Décision unanime entre ensemble et modèles standards: {original_decision}")
                        return original_decision
                    
                    # Si désaccord, favoriser la décision originale mais tenir compte de l'ensemble
                    # avec un poids proportionnel à sa confiance
                    if ensemble_confidence > 0.5:
                        logger.info(f"Désaccord - préférence pour ensemble ({ensemble_recommendation}) vs original ({original_decision})")
                        return ensemble_recommendation
                    else:
                        logger.info(f"Désaccord - préférence pour original ({original_decision}) vs ensemble ({ensemble_recommendation})")
                        return original_decision
            except Exception as e:
                logger.error(f"Erreur lors de l'intégration des signaux d'ensemble: {e}")
            
            # Par défaut, retourner la décision originale
            return original_decision
        
        # Remplacer la méthode par la version améliorée
        trading_bot.combine_signals = enhanced_combine_signals
        
        # Définir un attribut pour suivre le symbole actuel en cours de traitement
        if not hasattr(trading_bot, 'current_symbol'):
            trading_bot.current_symbol = None
        
        # Étendre la méthode execute_trades pour suivre le symbole actuel
        original_execute_trades = trading_bot.execute_trades
        
        def enhanced_execute_trades(data_manager):
            result = None
            try:
                # Obtenir les symboles disponibles
                symbols = list(data_manager.data.keys())
                
                # Traiter chaque symbole
                for symbol in symbols:
                    # Mettre à jour le symbole actuel
                    trading_bot.current_symbol = symbol
                    
                # Exécuter la méthode originale
                result = original_execute_trades(data_manager)
                
                # Réinitialiser le symbole actuel
                trading_bot.current_symbol = None
                
            except Exception as e:
                logger.error(f"Erreur dans enhanced_execute_trades: {e}")
                # Réinitialiser le symbole actuel en cas d'erreur
                trading_bot.current_symbol = None
                
            return result
        
        # Remplacer la méthode par la version améliorée
        trading_bot.execute_trades = enhanced_execute_trades
        
        logger.info("Bot de trading mis à jour avec intégration complète de l'ensemble")
        return trading_bot 