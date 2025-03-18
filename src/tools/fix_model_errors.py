#!/usr/bin/env python
"""
Script de correction automatique des erreurs courantes dans les modèles SQLAlchemy.

Ce script détecte et corrige automatiquement certaines erreurs courantes dans les modèles :
- Renommage des attributs qui utilisent des noms réservés
- Correction des contraintes d'unicité mal formées
- Autres problèmes de syntaxe courants
"""

import os
import sys
import re
from pathlib import Path
import ast
from typing import List, Dict, Tuple, Optional, Set

# Ajoutez le répertoire du projet au path Python
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

# Liste des mots réservés SQLAlchemy à remplacer automatiquement
SQLALCHEMY_RESERVED_NAMES = {
    'metadata': 'model_metadata',
    'query': 'db_query',
    'query_class': 'db_query_class',
    '__table__': None,  # Ne pas remplacer automatiquement
    '__mapper__': None  # Ne pas remplacer automatiquement
}

# Répertoires à analyser (relatifs à src/)
MODEL_DIRECTORIES = [
    'api/database/models',
    'models',
    'core/models'
]

def get_python_files(base_dir: str) -> List[str]:
    """Récupère tous les fichiers Python dans le répertoire spécifié de manière récursive."""
    python_files = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def fix_reserved_column_names(file_path: str) -> List[str]:
    """Détecte et corrige les noms de colonnes qui sont des mots réservés par SQLAlchemy."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    fixes = []
    
    # Rechercher les définitions de colonnes avec des noms réservés
    for reserved_name, replacement in SQLALCHEMY_RESERVED_NAMES.items():
        if replacement is None:
            continue  # Ignorer les noms sans remplacement automatique
            
        # Rechercher les modèles de définition de colonne : "metadata = Column(...)"
        pattern = rf'(\s+){reserved_name}\s*=\s*Column\('
        matches = re.finditer(pattern, content)
        
        for match in matches:
            # Remplacer le nom réservé par son alternative
            line_start = content[:match.start()].rfind('\n') + 1
            line_end = content.find('\n', match.start())
            line = content[line_start:line_end]
            
            fixed_line = line.replace(f'{reserved_name} =', f'{replacement} =')
            content = content.replace(line, fixed_line)
            
            # Mettre à jour les références dans to_dict
            to_dict_pattern = rf'["\']?{reserved_name}["\']?\s*:\s*self\.{reserved_name}'
            to_dict_replacement = f'"{reserved_name}": self.{replacement}'
            content = re.sub(to_dict_pattern, to_dict_replacement, content)
            
            fixes.append(f"Remplacé '{reserved_name}' par '{replacement}' dans {os.path.basename(file_path)}")
    
    # Si des modifications ont été apportées, écrire le contenu mis à jour
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return fixes

def fix_table_args_syntax(file_path: str) -> List[str]:
    """Corrige la syntaxe incorrecte dans __table_args__."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    fixes = []
    
    # Rechercher le motif problématique: __table_args__ = ({"UniqueConstraint": ...})
    pattern = r'__table_args__\s*=\s*\(\s*{["\']?UniqueConstraint["\']?\s*:\s*\(([^)]+)\)\s*}\s*\)'
    matches = re.finditer(pattern, content)
    
    for match in matches:
        # Extraire les colonnes mentionnées dans la contrainte
        columns_text = match.group(1)
        columns = [col.strip(' "\'') for col in columns_text.split(',')]
        fixed_constraint = f'__table_args__ = (\n        # Contrainte d\'unicité\n        UniqueConstraint({", ".join([f"\"{col}\"" for col in columns])}),\n    )'
        
        # Vérifier si l'import UniqueConstraint existe, sinon l'ajouter
        if 'UniqueConstraint' not in content[:match.start()]:
            import_pattern = r'from sqlalchemy import\s+\(([^)]+)\)'
            import_match = re.search(import_pattern, content)
            if import_match:
                imports = import_match.group(1)
                if 'UniqueConstraint' not in imports:
                    new_imports = imports.rstrip() + ', UniqueConstraint\n'
                    content = content.replace(imports, new_imports)
                    fixes.append(f"Ajouté l'import de UniqueConstraint dans {os.path.basename(file_path)}")
        
        # Remplacer la contrainte mal formée
        content = content.replace(match.group(0), fixed_constraint)
        fixes.append(f"Corrigé la syntaxe de __table_args__ dans {os.path.basename(file_path)}")
    
    # Si des modifications ont été apportées, écrire le contenu mis à jour
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return fixes

def fix_models_in_directory(directory: str) -> Dict[str, List[str]]:
    """Parcourt tous les fichiers Python dans le répertoire et corrige les erreurs courantes."""
    fixes = {}
    
    python_files = get_python_files(directory)
    for file_path in python_files:
        file_fixes = []
        
        # Appliquer les corrections
        file_fixes.extend(fix_reserved_column_names(file_path))
        file_fixes.extend(fix_table_args_syntax(file_path))
        
        if file_fixes:
            fixes[file_path] = file_fixes
    
    return fixes

def main():
    """Fonction principale qui exécute la correction des modèles."""
    print("Recherche et correction automatique des erreurs dans les modèles SQLAlchemy...")
    all_fixes = {}
    
    for dir_name in MODEL_DIRECTORIES:
        base_dir = os.path.join(project_root, 'src', dir_name)
        
        if not os.path.exists(base_dir):
            continue
        
        directory_fixes = fix_models_in_directory(base_dir)
        all_fixes.update(directory_fixes)
    
    # Afficher un rapport des corrections
    if all_fixes:
        print("\n" + "=" * 80)
        print(f"RAPPORT DE CORRECTION DES MODÈLES SQLALCHEMY")
        print("=" * 80)
        print(f"Fichiers modifiés: {len(all_fixes)}")
        
        for file_path, fixes in all_fixes.items():
            print(f"\n📄 {os.path.relpath(file_path, project_root)}")
            for fix in fixes:
                print(f"   ✓ {fix}")
        
        print("\n✅ Corrections terminées. Exécutez validate_models.py pour vérifier s'il reste des problèmes.")
    else:
        print("\n✅ Aucune erreur courante détectée dans les modèles.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 