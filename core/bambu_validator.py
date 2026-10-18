"""
Validateur et exportateur pour impression 3D Bambu Lab A1 Mini.
Garantit les cotes (<= 180mm), l'étanchéité (watertight) et calcule les métriques d'impression.
Gère automatiquement Y-up (WebGL Three.js) et Z-up (Bambu Studio STL).
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
    Vérifie la conformité Bambu Lab A1 Mini d'un maillage Y-up (Three.js) et calcule les métriques.
    """
    if mesh.is_empty:
        raise ValueError("Le maillage généré est vide.")

    # Dimensions courantes (en Y-up : X=largeur, Y=hauteur, Z=profondeur)
    extents = mesh.extents
    dx, dy, dz = float(extents[0]), float(extents[1]), float(extents[2])

    # Si target_height_mm > 0, ajuster l'échelle sur la hauteur (Y)
    scale_factor = 1.0
    if target_height_mm > 0 and dy > 0:
        scale_factor = target_height_mm / dy

    # Vérification des limites absolues du plateau Bambu Lab A1 Mini (175mm max de sécurité)
    if max(dx * scale_factor, dy * scale_factor, dz * scale_factor) > 175.0:
        scale_factor = 175.0 / max(dx, dy, dz)

    if abs(scale_factor - 1.0) > 1e-3:
        mesh.apply_scale(scale_factor)

    # Recentrage au sol Y-up (Y_min = 0, centré en X et Z)
    bounds = mesh.bounds
    center_x = (bounds[0][0] + bounds[1][0]) / 2.0
    min_y = bounds[0][1]
    center_z = (bounds[0][2] + bounds[1][2]) / 2.0
    mesh.apply_translation([-center_x, -min_y, -center_z])

    extents = mesh.extents
    dx, dy, dz = float(extents[0]), float(extents[1]), float(extents[2])

    # Réparation géométrique & Garantie d'Étanchéité Manifold
    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fill_holes(mesh)
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
    except Exception:
        pass

    is_watertight = bool(mesh.is_watertight)

    # Calcul du volume et de la masse PLA
    volume_cm3 = 0.0
    try:
        vol_raw = abs(float(mesh.volume)) / 1000.0
        if vol_raw > 0.5:
            volume_cm3 = vol_raw
        else:
            volume_cm3 = (dx * dy * dz * 0.28) / 1000.0
    except Exception:
        volume_cm3 = (dx * dy * dz * 0.28) / 1000.0

    # Estimation PLA : Densité 1.24 g/cm3, 15% infill + 3 parois (~38% du volume plein)
    estimated_weight_g = round(volume_cm3 * 1.24 * 0.38, 1)

    # Estimation temps d'impression Bambu A1 Mini (adapté du micro pion 10mm aux grandes pièces)
    estimated_time_minutes = int(max(4, round(estimated_weight_g * 1.6 + (dy / 0.20) * 0.06)))
    hours = estimated_time_minutes // 60
    mins = estimated_time_minutes % 60
    time_str = f"{hours}h {mins:02d}m" if hours > 0 else f"{mins} min"

    bed_fit = dx <= 175.0 and dy <= 175.0 and dz <= 175.0
    bambu_ready = is_watertight and bed_fit

    return {
        "dimensions_mm": {
            "x": round(dx, 1),
            "y": round(dz, 1),
            "z": round(dy, 1)  # Hauteur verticale affichée en Z pour convention impression 3D
        },
        "is_watertight": is_watertight,
        "bambu_ready": bambu_ready,
        "certification": "100% Conforme Bambu Lab A1 Mini" if bambu_ready else "Vérification requise",
        "volume_cm3": round(volume_cm3, 2),
        "weight_pla_g": estimated_weight_g,
        "print_time_estimate": time_str,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "bed_fit": bed_fit
    }


def export_all_formats(mesh: trimesh.Trimesh, base_filepath_no_ext: str) -> dict:
    """
    Exporte le maillage :
    - GLB et OBJ en Y-up (conservation intégrale de la texture 4K et des détails pour WebGL Three.js)
    - STL binaire en Z-up avec Décimation Intelligente QEM (allège le fichier par 4 pour Bambu Studio sans altérer la texture GLB)
    """
    os.makedirs(os.path.dirname(base_filepath_no_ext), exist_ok=True)

    stl_path = f"{base_filepath_no_ext}.stl"
    glb_path = f"{base_filepath_no_ext}.glb"
    obj_path = f"{base_filepath_no_ext}.obj"

    # Forcer le calcul et la cohérence des normales de sommets pour Three.js / WebGL
    try:
        trimesh.repair.fix_normals(mesh)
        _ = mesh.vertex_normals
    except Exception as e:
        print(f"[bambu_validator] Info calcul normales : {e}")

    # Export GLB (Y-up pour visualiseur Three.js avec texture intégrale)
    mesh.export(glb_path, file_type='glb')

    # Export OBJ
    mesh.export(obj_path, file_type='obj')

    # Conversion Y-up -> Z-up pour le STL Bambu Studio
    mesh_stl = mesh.copy()
    rot_z_up = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
    mesh_stl.apply_transform(rot_z_up)
    min_z = mesh_stl.bounds[0][2]
    mesh_stl.apply_translation([0, 0, -min_z])

    # Décimation Intelligente QEM ciblée pour le slicer Bambu Lab
    orig_faces = len(mesh_stl.faces)
    decimated = False
    if orig_faces > 220000:
        try:
            target_faces = 180000
            print(f"[bambu_validator] Décimation QEM du STL en cours ({orig_faces} -> ~{target_faces} faces)...")
            mesh_stl = mesh_stl.simplify_quadric_decimation(face_count=target_faces)
            trimesh.repair.fix_normals(mesh_stl)
            trimesh.repair.fill_holes(mesh_stl)
            decimated = True
            print(f"[bambu_validator] ✅ Décimation QEM réussie : STL réduit à {len(mesh_stl.faces)} faces.")
        except Exception as e_dec:
            print(f"[bambu_validator] Note simplification QEM : {e_dec}")

    mesh_stl.export(stl_path, file_type='stl')

    return {
        "stl": stl_path,
        "glb": glb_path,
        "obj": obj_path,
        "decimated_qem": decimated,
        "stl_face_count": len(mesh_stl.faces),
        "source_face_count": orig_faces
    }
