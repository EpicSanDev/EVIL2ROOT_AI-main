"""
Module pour les routes de santé de l'application
"""
from flask import Blueprint, jsonify
import psutil
import os

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de santé pour vérifier que l'application est opérationnelle
    Utilisé par Docker et DigitalOcean pour les health checks
    """
    # Vérification de l'utilisation des ressources
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Statut de l'application
    status = {
        'status': 'healthy',
        'memory': {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent
        },
        'disk': {
            'total': disk.total,
            'free': disk.free,
            'percent': disk.percent
        },
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'version': os.environ.get('APP_VERSION', 'dev')
    }
    
    return jsonify(status) 