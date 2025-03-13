"""
Routes pour l'authentification des utilisateurs
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime
from werkzeug.urls import url_parse

from app.forms import LoginForm, RegistrationForm, ResetPasswordRequestForm, ResetPasswordForm
from app.models.db_user import db, User

# Création du Blueprint pour l'authentification
auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    """Route de connexion des utilisateurs"""
    # Si l'utilisateur est déjà connecté, rediriger vers la page d'accueil
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    # Créer une instance du formulaire de connexion
    form = LoginForm()
    
    # Si le formulaire est soumis et valide
    if form.validate_on_submit():
        # Rechercher l'utilisateur par son nom d'utilisateur
        user = User.query.filter_by(username=form.username.data).first()
        
        # Vérifier si l'utilisateur existe et si le mot de passe est correct
        if user is None or not user.check_password(form.password.data):
            flash('Nom d\'utilisateur ou mot de passe incorrect', 'danger')
            return render_template('login.html', form=form)
        
        # Mettre à jour la date de dernière connexion
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Connecter l'utilisateur
        login_user(user, remember=form.remember_me.data)
        
        # Rediriger vers la page demandée ou la page d'accueil
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('main.dashboard')
        
        flash('Connexion réussie', 'success')
        return redirect(next_page)
    
    # Afficher le formulaire de connexion
    return render_template('login.html', form=form)

@auth.route('/register', methods=['GET', 'POST'])
def register():
    """Route d'inscription des utilisateurs"""
    # Si l'utilisateur est déjà connecté, rediriger vers la page d'accueil
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    # Créer une instance du formulaire d'inscription
    form = RegistrationForm()
    
    # Si le formulaire est soumis et valide
    if form.validate_on_submit():
        # Créer un nouvel utilisateur
        user = User(
            username=form.username.data,
            email=form.email.data,
            telegram_id=form.telegram_id.data if form.telegram_id.data else None,
            subscription_type='free'  # Par défaut, les nouveaux utilisateurs ont un abonnement gratuit
        )
        
        # Définir le mot de passe de l'utilisateur
        user.set_password(form.password.data)
        
        # Ajouter l'utilisateur à la base de données
        db.session.add(user)
        db.session.commit()
        
        # Notifier l'utilisateur que son compte a été créé
        flash('Votre compte a été créé avec succès! Vous pouvez maintenant vous connecter.', 'success')
        
        # Rediriger vers la page de connexion
        return redirect(url_for('auth.login'))
    
    # Afficher le formulaire d'inscription
    return render_template('register.html', form=form)

@auth.route('/logout')
@login_required
def logout():
    """Route de déconnexion des utilisateurs"""
    logout_user()
    flash('Vous avez été déconnecté avec succès', 'success')
    return redirect(url_for('auth.login'))

@auth.route('/change_password', methods=['POST'])
@login_required
def change_password():
    """Route pour changer le mot de passe d'un utilisateur connecté"""
    # Récupérer les données du formulaire
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    # Vérifier que tous les champs sont remplis
    if not current_password or not new_password or not confirm_password:
        flash('Tous les champs sont obligatoires', 'danger')
        return redirect(url_for('main.profile'))
    
    # Vérifier que le mot de passe actuel est correct
    if not current_user.check_password(current_password):
        flash('Le mot de passe actuel est incorrect', 'danger')
        return redirect(url_for('main.profile'))
    
    # Vérifier que les deux nouveaux mots de passe correspondent
    if new_password != confirm_password:
        flash('Les nouveaux mots de passe ne correspondent pas', 'danger')
        return redirect(url_for('main.profile'))
    
    # Vérifier que le nouveau mot de passe est différent de l'ancien
    if new_password == current_password:
        flash('Le nouveau mot de passe doit être différent de l\'ancien', 'danger')
        return redirect(url_for('main.profile'))
    
    # Vérifier que le nouveau mot de passe a au moins 8 caractères
    if len(new_password) < 8:
        flash('Le nouveau mot de passe doit contenir au moins 8 caractères', 'danger')
        return redirect(url_for('main.profile'))
    
    try:
        # Mettre à jour le mot de passe
        current_user.set_password(new_password)
        db.session.commit()
        flash('Votre mot de passe a été mis à jour avec succès', 'success')
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Erreur lors du changement de mot de passe: {str(e)}")
        flash('Une erreur est survenue lors de la mise à jour du mot de passe', 'danger')
    
    return redirect(url_for('main.profile'))

# Réinitialisation de mot de passe - à implémenter ultérieurement
@auth.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    """Route pour demander la réinitialisation du mot de passe"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = ResetPasswordRequestForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            # Envoyer un email avec un lien pour réinitialiser le mot de passe
            # Cette fonctionnalité serait à implémenter plus tard
            pass
        
        # Toujours afficher ce message même si l'email n'existe pas (sécurité)
        flash('Un email avec les instructions pour réinitialiser votre mot de passe a été envoyé', 'info')
        return redirect(url_for('auth.login'))
    
    return render_template('reset_password_request.html', form=form)

@auth.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Route pour réinitialiser le mot de passe avec un token"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    # Vérifier le token et obtenir l'utilisateur correspondant
    # Cette fonctionnalité serait à implémenter plus tard
    user = None
    
    if user is None:
        flash('Le lien de réinitialisation est invalide ou a expiré', 'danger')
        return redirect(url_for('auth.login'))
    
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash('Votre mot de passe a été réinitialisé', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('reset_password.html', form=form) 