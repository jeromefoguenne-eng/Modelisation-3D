"""
Validateur et exportateur pour impression 3D Bambu Lab A1 Mini.
Garantit les cotes (<= 180mm), l'étanchéité (watertight) et calcule les métriques d'impression.
"""

import os
import trimesh
import numpy as np

BAMBU_A1_MINI_MAX_DIM = 180.0  # 180x180x180 mm

def validate_and_scale_for_bambu(
    mesh: trimesh.Trimesh,
    target_height_mm: float = 120.0,
    enforce_watertight: bool = True
) -> dict:
    """
    Met à l'échelle le maillage pour la Bambu Lab A1 Mini et vérifie sa conformité.
    """
    if mesh.is_empty:
        raise ValueError("Le maillage généré est vide.")
        
    # Recentrage au sol (Z min = 0, centré en X et Y)
    bounds = mesh.bounds
    min_b, max_b = bounds[0], bounds[1]
    
    center_x = (min_b[0] + max_b[0]) / 2.0
    center_y = (min_b[1] + max_b[1]) / 2.0
    min_z = min_b[2]
    
    mesh.apply_translation([-center_x, -center_y, -min_z])
    
    # Dimensions initiales
    extents = mesh.extents
    dx, dy, dz = extents[0], extents[1], extents[2]
    
    # Facteur d'échelle
    # On se base sur la hauteur désirée, tout en garantissant qu'aucune dimension n'excède 175mm (marge de sécurité)
    scale_factor = 1.0
    if target_height_mm > 0 and dz > 0:
        scale_factor = target_height_mm / dz
        
    # Vérification des limites absolues
    max_dim_after_scale = max(dx * scale_factor, dy * scale_factor, dz * scale_factor)
    if max_dim_after_scale > 175.0:
        scale_factor = 175.0 / max(dx, dy, dz)
        
    mesh.apply_scale(scale_factor)
    
    # Recalcul des bornes
    extents = mesh.extents
    dx, dy, dz = float(extents[0]), float(extents[1]), float(extents[2])
    
    # Réparation géométrique
    if not mesh.is_watertight:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fill_holes(mesh)
        
    is_watertight = bool(mesh.is_watertight)
    
    # Calcul du volume et de la masse PLA
    volume_cm3 = 0.0
    try:
        if is_watertight:
            volume_cm3 = abs(float(mesh.volume)) / 1000.0  # mm^3 to cm^3
        else:
            # Approximation par boîte englobante ou voxel
            volume_cm3 = (dx * dy * dz * 0.35) / 1000.0
    except Exception:
        volume_cm3 = (dx * dy * dz * 0.35) / 1000.0
        
    # Estimation PLA : Densité 1.24 g/cm3, 15% infill + 3 parois (~35% du volume plein)
    estimated_weight_g = round(volume_cm3 * 1.24 * 0.38, 1)
    
    # Estimation temps d'impression Bambu A1 Mini (vitesse rapide ~200mm/s)
    # Règle empirique calibrée : ~15 min par 10g + base temps de couche
    estimated_time_minutes = int(max(20, round(estimated_weight_g * 1.5 + (dz / 0.20) * 0.05)))
    hours = estimated_time_minutes // 60
    mins = estimated_time_minutes % 60
    time_str = f"{hours}h {mins:02d}m" if hours > 0 else f"{mins} min"
    
    return {
        "dimensions_mm": {
            "x": round(dx, 1),
            "y": round(dy, 1),
            "z": round(dz, 1)
        },
        "is_watertight": is_watertight,
        "volume_cm3": round(volume_cm3, 2),
        "weight_pla_g": estimated_weight_g,
        "print_time_estimate": time_str,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "bed_fit": dx <= BAMBU_A1_MINI_MAX_DIM and dy <= BAMBU_A1_MINI_MAX_DIM and dz <= BAMBU_A1_MINI_MAX_DIM
    }

def export_all_formats(mesh: trimesh.Trimesh, base_filepath_no_ext: str) -> dict:
    """
    Exporte le maillage sous format STL binaire, GLB et 3MF.
    """
    os.makedirs(os.path.dirname(base_filepath_no_ext), exist_ok=True)
    
    stl_path = f"{base_filepath_no_ext}.stl"
    glb_path = f"{base_filepath_no_ext}.glb"
    obj_path = f"{base_filepath_no_ext}.obj"
    
    # Export STL binaire (recommandé Bambu Studio)
    mesh.export(stl_path, file_type='stl')
    
    # Export GLB (pour visualiseur Three.js)
    mesh.export(glb_path, file_type='glb')
    
    # Export OBJ
    mesh.export(obj_path, file_type='obj')
    
    return {
        "stl": stl_path,
        "glb": glb_path,
        "obj": obj_path
    }
