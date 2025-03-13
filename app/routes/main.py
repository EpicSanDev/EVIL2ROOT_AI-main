"""
Routes principales de l'application
"""

import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, jsonify
from flask_login import login_required, current_user

from app.models.db_user import db, User

main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Page d'accueil, redirige vers le tableau de bord si connecté, sinon vers la page de connexion"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('auth.login'))

@main.route('/dashboard')
@login_required
def dashboard():
    """Page principale du tableau de bord"""
    return render_template('dashboard.html')

@main.route('/performance')
@login_required
def performance():
    """Page d'analyse des performances"""
    return render_template('performance.html')

@main.route('/settings')
@login_required
def settings():
    """Page des paramètres utilisateur"""
    return render_template('settings.html')

@main.route('/models/explanation')
@login_required
def model_explanations():
    """Page des explications sur les modèles d'IA"""
    return render_template('model_explanations.html')

@main.route('/advanced-dashboard')
@login_required
def advanced_dashboard():
    """Tableau de bord avancé"""
    # Vérifier si l'utilisateur a un abonnement premium
    if current_user.subscription_type not in ['premium', 'enterprise']:
        flash('Cette fonctionnalité nécessite un abonnement Premium ou Entreprise', 'warning')
        return redirect(url_for('main.dashboard'))
    
    return render_template('advanced_dashboard.html')

@main.route('/plugins')
@login_required
def plugins():
    """Page de gestion des plugins"""
    # Vérifier si l'utilisateur a un abonnement qui permet d'accéder aux plugins
    if current_user.subscription_type not in ['basic', 'premium', 'enterprise']:
        flash('Cette fonctionnalité nécessite au minimum un abonnement Basic', 'warning')
        return redirect(url_for('main.dashboard'))
    
    return render_template('plugins.html')

@main.route('/plugins/settings')
@login_required
def plugin_settings():
    """Page de paramètres des plugins"""
    # Vérifier si l'utilisateur a un abonnement qui permet d'accéder aux plugins
    if current_user.subscription_type not in ['basic', 'premium', 'enterprise']:
        flash('Cette fonctionnalité nécessite au minimum un abonnement Basic', 'warning')
        return redirect(url_for('main.dashboard'))
    
    return render_template('plugin_settings.html')

@main.route('/profile')
@login_required
def profile():
    """Page de profil utilisateur"""
    return render_template('profile.html')

@main.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    """Mettre à jour les informations de profil de l'utilisateur"""
    # Vérifier si les données sont au format JSON
    if not request.is_json:
        return jsonify(success=False, message="Format de requête invalide"), 400
    
    # Récupérer les données du formulaire
    data = request.get_json()
    first_name = data.get('first_name', '').strip()
    last_name = data.get('last_name', '').strip()
    telegram_id = data.get('telegram_id', '').strip()
    
    try:
        # Mettre à jour les informations de l'utilisateur
        current_user.first_name = first_name
        current_user.last_name = last_name
        
        # Vérifier si l'ID Telegram est différent
        if telegram_id != current_user.telegram_id:
            # Vérifier si l'ID Telegram est déjà utilisé
            existing_user = User.query.filter_by(telegram_id=telegram_id).first()
            if existing_user and existing_user.id != current_user.id:
                return jsonify(success=False, message="Cet ID Telegram est déjà utilisé par un autre compte"), 400
            
            current_user.telegram_id = telegram_id if telegram_id else None
        
        # Enregistrer les modifications
        db.session.commit()
        return jsonify(success=True, message="Profil mis à jour avec succès")
    
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Erreur lors de la mise à jour du profil: {str(e)}")
        return jsonify(success=False, message="Une erreur est survenue lors de la mise à jour du profil"), 500

@main.route('/upload_profile_image', methods=['POST'])
@login_required
def upload_profile_image():
    """Télécharger une image de profil"""
    if 'profile_image' not in request.files:
        flash('Aucun fichier sélectionné', 'danger')
        return redirect(url_for('main.profile'))
    
    file = request.files['profile_image']
    
    if file.filename == '':
        flash('Aucun fichier sélectionné', 'danger')
        return redirect(url_for('main.profile'))
    
    # Vérifier l'extension du fichier
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        flash('Format de fichier non autorisé. Utilisez PNG, JPG, JPEG ou GIF', 'danger')
        return redirect(url_for('main.profile'))
    
    try:
        # Sécuriser le nom du fichier
        filename = secure_filename(f"{current_user.id}_{int(datetime.utcnow().timestamp())}.{file.filename.rsplit('.', 1)[1].lower()}")
        
        # Créer le répertoire de téléchargement s'il n'existe pas
        upload_folder = os.path.join(current_app.static_folder, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        # Enregistrer le fichier
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        
        # Supprimer l'ancienne image de profil si elle existe
        if current_user.profile_image:
            old_file_path = os.path.join(upload_folder, current_user.profile_image)
            if os.path.exists(old_file_path):
                os.remove(old_file_path)
        
        # Mettre à jour le profil utilisateur
        current_user.profile_image = filename
        db.session.commit()
        
        flash('Image de profil mise à jour avec succès', 'success')
    except Exception as e:
        current_app.logger.error(f"Erreur lors du téléchargement de l'image: {str(e)}")
        flash('Une erreur est survenue lors du téléchargement de l\'image', 'danger')
    
    return redirect(url_for('main.profile')) 