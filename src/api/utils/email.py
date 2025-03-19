"""
Utilitaires d'envoi d'email pour l'API.

Ce module contient les fonctions pour envoyer des emails aux utilisateurs
pour différents cas d'usage (bienvenue, réinitialisation de mot de passe, etc.).
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

logger = logging.getLogger("api.email")

# Configuration des paramètres d'email à partir des variables d'environnement
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@tradingbot.com")
EMAIL_SENDER_NAME = os.getenv("EMAIL_SENDER_NAME", "Trading Bot")

def send_email(to_email: str, subject: str, html_content: str, text_content: Optional[str] = None) -> bool:
    """
    Envoie un email avec contenu HTML et texte brut.
    
    Args:
        to_email: Adresse email du destinataire
        subject: Sujet de l'email
        html_content: Contenu HTML de l'email
        text_content: Contenu texte de l'email (facultatif)
        
    Returns:
        True si l'email a été envoyé correctement, False sinon
    """
    if not all([SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD]):
        logger.warning("Configuration SMTP incomplète. Email non envoyé.")
        return False
    
    # Créer un message multipart
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{EMAIL_SENDER_NAME} <{EMAIL_FROM}>"
    msg["To"] = to_email
    
    # Ajouter la version texte et HTML
    if text_content:
        part1 = MIMEText(text_content, "plain")
        msg.attach(part1)
        
    part2 = MIMEText(html_content, "html")
    msg.attach(part2)
    
    try:
        # Se connecter au serveur SMTP
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        
        # Envoyer l'email
        server.sendmail(EMAIL_FROM, to_email, msg.as_string())
        server.quit()
        
        logger.info(f"Email envoyé à {to_email}: {subject}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'email à {to_email}: {e}")
        return False

def send_welcome_email(email: str, name: str) -> bool:
    """
    Envoie un email de bienvenue à un nouvel utilisateur.
    
    Args:
        email: Adresse email de l'utilisateur
        name: Nom complet de l'utilisateur
        
    Returns:
        True si l'email a été envoyé correctement, False sinon
    """
    subject = "Bienvenue sur notre plateforme de trading"
    
    html_content = f"""
    <html>
    <body>
        <h1>Bienvenue, {name}!</h1>
        <p>Nous sommes ravis de vous accueillir sur notre plateforme de trading.</p>
        <p>Vous pouvez maintenant configurer vos préférences et commencer à utiliser nos outils d'analyse et de trading.</p>
        <p>N'hésitez pas à nous contacter si vous avez des questions.</p>
        <p>Cordialement,<br>L'équipe Trading Bot</p>
    </body>
    </html>
    """
    
    text_content = f"""
    Bienvenue, {name}!
    
    Nous sommes ravis de vous accueillir sur notre plateforme de trading.
    
    Vous pouvez maintenant configurer vos préférences et commencer à utiliser nos outils d'analyse et de trading.
    
    N'hésitez pas à nous contacter si vous avez des questions.
    
    Cordialement,
    L'équipe Trading Bot
    """
    
    return send_email(email, subject, html_content, text_content)

def send_password_reset_email(email: str, reset_token: str, expiration_minutes: int = 30) -> bool:
    """
    Envoie un email de réinitialisation de mot de passe.
    
    Args:
        email: Adresse email de l'utilisateur
        reset_token: Token de réinitialisation de mot de passe
        expiration_minutes: Durée de validité du token en minutes
        
    Returns:
        True si l'email a été envoyé correctement, False sinon
    """
    subject = "Réinitialisation de votre mot de passe"
    
    # Construire l'URL de réinitialisation (à adapter selon la configuration frontend)
    reset_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/reset-password?token={reset_token}"
    
    html_content = f"""
    <html>
    <body>
        <h1>Réinitialisation de votre mot de passe</h1>
        <p>Vous avez demandé une réinitialisation de mot de passe pour votre compte.</p>
        <p>Veuillez cliquer sur le lien ci-dessous pour définir un nouveau mot de passe :</p>
        <p><a href="{reset_url}">Réinitialiser mon mot de passe</a></p>
        <p>Ce lien est valable pendant {expiration_minutes} minutes.</p>
        <p>Si vous n'avez pas demandé cette réinitialisation, veuillez ignorer cet email.</p>
        <p>Cordialement,<br>L'équipe Trading Bot</p>
    </body>
    </html>
    """
    
    text_content = f"""
    Réinitialisation de votre mot de passe
    
    Vous avez demandé une réinitialisation de mot de passe pour votre compte.
    
    Veuillez cliquer sur le lien ci-dessous pour définir un nouveau mot de passe :
    {reset_url}
    
    Ce lien est valable pendant {expiration_minutes} minutes.
    
    Si vous n'avez pas demandé cette réinitialisation, veuillez ignorer cet email.
    
    Cordialement,
    L'équipe Trading Bot
    """
    
    return send_email(email, subject, html_content, text_content) 