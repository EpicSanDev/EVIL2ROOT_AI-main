"""
Formulaires pour l'authentification et la gestion des utilisateurs
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, EmailField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError
from app.models.db_user import User

class LoginForm(FlaskForm):
    """Formulaire de connexion"""
    username = StringField('Nom d\'utilisateur', validators=[DataRequired()])
    password = PasswordField('Mot de passe', validators=[DataRequired()])
    remember_me = BooleanField('Se souvenir de moi')
    submit = SubmitField('Se connecter')

class RegistrationForm(FlaskForm):
    """Formulaire d'inscription"""
    username = StringField('Nom d\'utilisateur', validators=[
        DataRequired(),
        Length(min=3, max=64, message='Le nom d\'utilisateur doit contenir entre 3 et 64 caractères.')
    ])
    email = EmailField('Email', validators=[
        DataRequired(),
        Email(message='Veuillez entrer une adresse email valide.')
    ])
    password = PasswordField('Mot de passe', validators=[
        DataRequired(),
        Length(min=8, message='Le mot de passe doit contenir au moins 8 caractères.')
    ])
    password2 = PasswordField('Confirmer le mot de passe', validators=[
        DataRequired(),
        EqualTo('password', message='Les mots de passe doivent correspondre.')
    ])
    telegram_id = StringField('ID Telegram (optionnel)')
    submit = SubmitField('S\'inscrire')

    def validate_username(self, username):
        """Valide que le nom d'utilisateur n'est pas déjà utilisé"""
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Ce nom d\'utilisateur est déjà utilisé. Veuillez en choisir un autre.')

    def validate_email(self, email):
        """Valide que l'email n'est pas déjà utilisé"""
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Cette adresse email est déjà utilisée. Veuillez en choisir une autre.')

class ResetPasswordRequestForm(FlaskForm):
    """Formulaire de demande de réinitialisation de mot de passe"""
    email = EmailField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Réinitialiser le mot de passe')

class ResetPasswordForm(FlaskForm):
    """Formulaire de réinitialisation de mot de passe"""
    password = PasswordField('Nouveau mot de passe', validators=[
        DataRequired(),
        Length(min=8, message='Le mot de passe doit contenir au moins 8 caractères.')
    ])
    password2 = PasswordField('Confirmer le mot de passe', validators=[
        DataRequired(),
        EqualTo('password', message='Les mots de passe doivent correspondre.')
    ])
    submit = SubmitField('Réinitialiser le mot de passe') 