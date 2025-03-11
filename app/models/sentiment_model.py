# app/models/sentiment_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

class SentimentModel:
    """
    A class for sentiment analysis using a pre-trained model.
    """
    def __init__(self, model_path=None, tokenizer=None):
        """
        Initialize the sentiment model.
        
        :param model_path: Path to the model file.
        :param tokenizer: Tokenizer for text preprocessing.
        """
        self.model = None
        self.tokenizer = tokenizer
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a pre-trained sentiment model.
        
        :param model_path: Path to the model file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found.")
        self.model = tf.keras.models.load_model(model_path)
        return self.model
    
    def predict(self, text):
        """
        Predict the sentiment score of a given text using the loaded model.
        
        :param text: The text to analyze.
        :return: Sentiment score.
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model or tokenizer not initialized.")
        
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=100)
        sentiment_score = self.model.predict(padded_sequence)[0][0]
        return sentiment_score

# Keep the original functions for backward compatibility
def load_sentiment_model(model_path):
    """
    Charge un modèle de sentiment pré-entraîné.
    
    :param model_path: Chemin vers le fichier du modèle.
    :return: Modèle TensorFlow/Keras chargé.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier du modèle {model_path} est introuvable.")
    model = tf.keras.models.load_model(model_path)
    return model

def predict_sentiment(text, model, tokenizer):
    """
    Prédit le score de sentiment d'un texte donné en utilisant le modèle chargé.
    
    :param text: Le texte à analyser.
    :param model: Le modèle de sentiment.
    :param tokenizer: Le tokenizer pour prétraiter le texte.
    :return: Score de sentiment.
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    sentiment_score = model.predict(padded_sequence)[0][0]
    return sentiment_score