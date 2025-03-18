#!/usr/bin/env python
"""
Script de validation des modèles SQLAlchemy

Ce script parcourt automatiquement tous les modèles SQLAlchemy du projet
et vérifie les erreurs courantes sans avoir besoin de créer les tables.
"""

import os
import sys
import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import traceback

# Ajoutez le répertoire du projet au path Python
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

from sqlalchemy import inspect as sa_inspect, Column
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import RelationshipProperty

# Liste des mots réservés SQLAlchemy à éviter comme noms d'attributs
SQLALCHEMY_RESERVED_NAMES = {
    'metadata', 'query', 'query_class', '__tablename__', '__table__', '__mapper__',
    'registry', '_decl_class_registry', '_sa_class_manager', '__dict__', 
    '__weakref__', '__init__', '__new__', '__repr__', '__str__'
}

# Répertoires à analyser (relatifs à src/)
MODEL_DIRECTORIES = [
    'api/database/models',
    'models',
    'core/models'
]

def is_sqlalchemy_model(cls: Any) -> bool:
    """Vérifie si une classe est un modèle SQLAlchemy."""
    return isinstance(cls, DeclarativeMeta)

def get_all_python_files(base_dir: str) -> List[str]:
    """Récupère tous les fichiers Python dans le répertoire spécifié de manière récursive."""
    python_files = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def import_module_from_path(file_path: str) -> Any:
    """Importe un module Python à partir de son chemin de fichier."""
    module_path = file_path.replace(project_root, '').replace('/', '.').replace('\\', '.').lstrip('.')
    
    if module_path.endswith('.py'):
        module_path = module_path[:-3]
    
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        print(f"Erreur lors de l'importation de {module_path}: {e}")
        return None

def find_sqlalchemy_models() -> Dict[str, Dict[str, Any]]:
    """Trouve tous les modèles SQLAlchemy dans les répertoires spécifiés."""
    models = {}
    
    for dir_name in MODEL_DIRECTORIES:
        base_dir = os.path.join(project_root, 'src', dir_name)
        
        if not os.path.exists(base_dir):
            continue
        
        python_files = get_all_python_files(base_dir)
        
        for file_path in python_files:
            module = import_module_from_path(file_path)
            
            if not module:
                continue
            
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and is_sqlalchemy_model(obj) and obj.__module__ == module.__name__:
                    models[obj.__name__] = {
                        'class': obj,
                        'file_path': file_path,
                        'module_name': module.__name__
                    }
    
    return models

def check_reserved_names(model_cls: Any) -> List[str]:
    """Vérifie si le modèle utilise des noms réservés par SQLAlchemy."""
    errors = []
    
    for name in dir(model_cls):
        if name in SQLALCHEMY_RESERVED_NAMES and not name.startswith('__'):
            if hasattr(model_cls, name) and isinstance(getattr(model_cls, name), Column):
                errors.append(f"Le modèle '{model_cls.__name__}' utilise le nom réservé '{name}' comme nom de colonne")
    
    return errors

def check_relationship_integrity(models: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Vérifie l'intégrité des relations entre les modèles."""
    relation_errors = {}
    
    for model_name, model_info in models.items():
        model_cls = model_info['class']
        errors = []
        
        mapper = sa_inspect(model_cls)
        
        for rel in mapper.relationships:
            target_model = rel.mapper.class_
            target_model_name = target_model.__name__
            
            if target_model_name not in models and target_model_name != model_name:
                errors.append(f"Relation '{rel.key}' fait référence au modèle '{target_model_name}' qui n'a pas été trouvé")
            
            # Vérifier la cohérence des relations back_populates/backref
            if rel.back_populates:
                if not hasattr(target_model, rel.back_populates):
                    errors.append(f"La relation '{rel.key}' spécifie back_populates='{rel.back_populates}', mais '{target_model_name}' n'a pas cet attribut")
                else:
                    target_rel = getattr(sa_inspect(target_model).all_orm_descriptors, rel.back_populates, None)
                    if not isinstance(target_rel, RelationshipProperty):
                        errors.append(f"L'attribut back_populates='{rel.back_populates}' sur '{model_name}.{rel.key}' n'est pas une relation")
        
        if errors:
            relation_errors[model_name] = errors
    
    return relation_errors

def check_table_names(models: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Vérifie les noms de tables pour s'assurer qu'ils sont cohérents."""
    table_errors = {}
    
    for model_name, model_info in models.items():
        model_cls = model_info['class']
        errors = []
        
        # Vérifier si __tablename__ est défini
        if not hasattr(model_cls, '__tablename__') and not hasattr(model_cls, '__table__'):
            # SQLAlchemy génère automatiquement le nom, vérifier s'il est correct
            expected_tablename = model_name.lower()
            actual_tablename = model_cls.__tablename__ if hasattr(model_cls, '__tablename__') else None
            
            if actual_tablename and actual_tablename != expected_tablename:
                errors.append(f"Le nom de table '{actual_tablename}' est différent de la convention (le nom devrait être '{expected_tablename}')")
        
        if errors:
            table_errors[model_name] = errors
    
    return table_errors

def validate_class_instantiation(models: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Tente d'instancier chaque modèle pour détecter les erreurs de configuration."""
    instantiation_errors = {}
    
    for model_name, model_info in models.items():
        model_cls = model_info['class']
        
        try:
            # Tente de valider le modèle via sa métadonnée sans créer de tables
            model_cls.__table__.tometadata(model_cls.metadata, schema=model_cls.__table__.schema)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            instantiation_errors[model_name] = error_msg
    
    return instantiation_errors

def validate_sqlalchemy_models() -> Dict[str, Any]:
    """Valide tous les modèles SQLAlchemy et retourne un rapport des erreurs."""
    models = find_sqlalchemy_models()
    
    if not models:
        return {"status": "error", "message": "Aucun modèle SQLAlchemy trouvé."}
    
    report = {
        "status": "success",
        "total_models": len(models),
        "models_with_errors": 0,
        "errors": {}
    }
    
    # Vérifier les noms réservés
    for model_name, model_info in models.items():
        model_errors = []
        
        reserved_name_errors = check_reserved_names(model_info['class'])
        if reserved_name_errors:
            model_errors.extend(reserved_name_errors)
        
        if model_errors:
            report["errors"][model_name] = {
                "file_path": model_info['file_path'],
                "reserved_names": reserved_name_errors
            }
    
    # Vérifier l'intégrité des relations
    relation_errors = check_relationship_integrity(models)
    for model_name, errors in relation_errors.items():
        if model_name not in report["errors"]:
            report["errors"][model_name] = {
                "file_path": models[model_name]['file_path'],
                "relation_errors": []
            }
        report["errors"][model_name]["relation_errors"] = errors
    
    # Vérifier les noms de tables
    table_name_errors = check_table_names(models)
    for model_name, errors in table_name_errors.items():
        if model_name not in report["errors"]:
            report["errors"][model_name] = {
                "file_path": models[model_name]['file_path'],
                "table_name_errors": []
            }
        report["errors"][model_name]["table_name_errors"] = errors
    
    # Valider l'instanciation des modèles
    instantiation_errors = validate_class_instantiation(models)
    for model_name, error in instantiation_errors.items():
        if model_name not in report["errors"]:
            report["errors"][model_name] = {
                "file_path": models[model_name]['file_path'],
                "instantiation_error": ""
            }
        report["errors"][model_name]["instantiation_error"] = error
    
    # Mettre à jour le nombre de modèles avec des erreurs
    report["models_with_errors"] = len(report["errors"])
    
    return report

def print_report(report: Dict[str, Any]) -> None:
    """Affiche le rapport de validation des modèles de manière lisible."""
    print("=" * 80)
    print(f"RAPPORT DE VALIDATION DES MODÈLES SQLALCHEMY")
    print("=" * 80)
    print(f"Modèles analysés: {report['total_models']}")
    print(f"Modèles avec erreurs: {report['models_with_errors']}")
    print("=" * 80)
    
    if report["models_with_errors"] == 0:
        print("✅ Aucune erreur détectée dans les modèles !")
        return
    
    for model_name, errors in report["errors"].items():
        print(f"\n📁 MODÈLE: {model_name}")
        print(f"   Fichier: {errors['file_path']}")
        
        if "reserved_names" in errors and errors["reserved_names"]:
            print("\n   ❌ NOMS RÉSERVÉS:")
            for error in errors["reserved_names"]:
                print(f"      - {error}")
        
        if "relation_errors" in errors and errors["relation_errors"]:
            print("\n   ❌ ERREURS DE RELATION:")
            for error in errors["relation_errors"]:
                print(f"      - {error}")
        
        if "table_name_errors" in errors and errors["table_name_errors"]:
            print("\n   ❌ ERREURS DE NOM DE TABLE:")
            for error in errors["table_name_errors"]:
                print(f"      - {error}")
        
        if "instantiation_error" in errors and errors["instantiation_error"]:
            print("\n   ❌ ERREUR D'INSTANCIATION:")
            error_lines = errors["instantiation_error"].split("\n")
            for line in error_lines[:10]:  # Limiter à 10 lignes pour lisibilité
                print(f"      {line}")
            if len(error_lines) > 10:
                print(f"      ... ({len(error_lines) - 10} lignes supplémentaires)")
        
        print("-" * 80)

def main():
    """Fonction principale qui exécute la validation des modèles."""
    print("Validation des modèles SQLAlchemy en cours...")
    report = validate_sqlalchemy_models()
    print_report(report)
    
    # Retourne un code d'erreur si des problèmes ont été détectés
    return 1 if report["models_with_errors"] > 0 else 0

if __name__ == "__main__":
    sys.exit(main()) 