"""
Moteur de reconstruction 3D Volumétrique Haute Définition (GPU RTX 4070 Ti).
Génère une figurine 3D monobloc étanche avec micro-relief photonique et socle Bambu.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, gaussian_filter
from skimage import measure
import trimesh
import os

def generate_manifold_hd_mesh(
    mask_np: np.ndarray,
    luminance_np: np.ndarray,
    target_thickness_mm: float = 24.0,
    relief_depth_mm: float = 4.5,
    min_feature_mm: float = 1.2,
    include_base: bool = True,
    base_height_mm: float = 4.0,
    grid_resolution: int = 400
) -> trimesh.Trimesh:
    """
    Génère un maillage 3D volumétrique dense et étanche à partir d'un masque et de la luminance.
    """
    # 1. Normalisation et redimensionnement sur la grille de calcul
    from PIL import Image
    mask_img = Image.fromarray((mask_np > 128).astype(np.uint8) * 255)
    lum_img = Image.fromarray((np.clip(luminance_np, 0, 1) * 255).astype(np.uint8))
    
    # Redimensionnement sur grille carrée
    mask_res = np.array(mask_img.resize((grid_resolution, grid_resolution), Image.Resampling.NEAREST)) > 128
    lum_res = np.array(lum_img.resize((grid_resolution, grid_resolution), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    
    # 2. Renforcement morphologique des éléments ultra-fins (rayons de vélo, câbles, bâtons)
    # Pour garantir qu'ils ne disparaissent pas au tranchage Bambu Studio
    dilated_mask = binary_dilation(mask_res, iterations=2)
    
    # 3. Calcul de la carte de distance euclidienne (EDT)
    dist_map = distance_transform_edt(dilated_mask)
    max_dist = np.max(dist_map)
    if max_dist < 1e-4:
        raise ValueError("Le masque d'entrée est vide ou non détecté.")
        
    norm_dist = dist_map / max_dist
    
    # Profil de gonflement volumétrique parabolique
    # z_thick(x, y) définit le demi-profil d'épaisseur
    half_thick_voxel = (target_thickness_mm / 2.0)
    # Courbure anatomique douce
    volume_profile = np.power(norm_dist, 0.55) * half_thick_voxel
    
    # 4. Micro-sculpture photonique (détails du visage, du vélo, du maillot)
    # Filtre passe-haut sur la luminance pour extraire les arêtes et textures fines
    lum_smooth = gaussian_filter(lum_res, sigma=1.5)
    lum_detail = lum_res - lum_smooth
    lum_enhanced = lum_smooth + lum_detail * 1.5
    lum_enhanced = np.clip(lum_enhanced, 0, 1)
    
    relief_map = (lum_enhanced - 0.5) * relief_depth_mm
    # Le relief ne s'applique que là où le sujet est présent
    relief_map = relief_map * (dilated_mask.astype(np.float32))
    
    # 5. Construction de la grille 3D scalaire (Voxel Grid)
    # Profondeur Z discrétisée
    depth_slices = int(target_thickness_mm * 3.5)  # Echantillonnage fin en Z
    if depth_slices % 2 == 1:
        depth_slices += 1
        
    z_coords = np.linspace(-half_thick_voxel - 2.0, half_thick_voxel + 2.0, depth_slices)
    
    # Grille 3D : shape (H, W, Z)
    volume_3d = np.zeros((grid_resolution, grid_resolution, depth_slices), dtype=np.float32)
    
    # Remplissage par distance signée
    for k, z_val in enumerate(z_coords):
        # En face avant (z > 0), on ajoute le micro-relief photonique
        if z_val > 0:
            effective_profile = volume_profile + relief_map
        else:
            # En face arrière, profil galbé doux
            effective_profile = volume_profile * 0.95
            
        # Distance signée : positive à l'intérieur, négative à l'extérieur
        slice_dist = effective_profile - np.abs(z_val)
        # Rendre strictement négatif hors du masque
        slice_dist[~dilated_mask] = -1.0
        volume_3d[:, :, k] = slice_dist
        
    # 6. Ajout optionnel du socle d'adhérence Bambu Lab A1 Mini
    if include_base:
        # Trouver le bas du sujet (Y max dans l'image)
        y_indices, x_indices = np.where(dilated_mask)
        if len(y_indices) > 0:
            y_bottom = np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            
            # Socle : ellipse ou rectangle arrondi sous les pieds/roues
            base_thickness_voxels = int(grid_resolution * 0.04)
            y_start = max(0, y_bottom - base_thickness_voxels)
            y_end = min(grid_resolution - 1, y_bottom + base_thickness_voxels)
            
            x_center = (x_min + x_max) // 2
            x_radius = int((x_max - x_min) * 0.58)
            
            y_grid, x_grid = np.ogrid[:grid_resolution, :grid_resolution]
            base_mask = (((x_grid - x_center)**2) / (x_radius**2 + 1e-5) + ((y_grid - y_bottom)**2) / (base_thickness_voxels**2 + 1e-5)) <= 1.0
            
            # Application du socle sur toute l'épaisseur Z
            for k, z_val in enumerate(z_coords):
                if np.abs(z_val) <= (half_thick_voxel * 0.85):
                    volume_3d[base_mask, k] = np.maximum(volume_3d[base_mask, k], 1.5)
                    
    # Fermeture frontière étanche garantie : forcer les bords de la boîte à -1.0
    volume_3d[0, :, :] = -1.0
    volume_3d[-1, :, :] = -1.0
    volume_3d[:, 0, :] = -1.0
    volume_3d[:, -1, :] = -1.0
    volume_3d[:, :, 0] = -1.0
    volume_3d[:, :, -1] = -1.0
                    
    # 7. Extraction de l'iso-surface étanche via Marching Cubes
    # Niveau 0.0 sépare l'intérieur (>0) de l'extérieur (<0)
    print("[ManifoldHD] Extraction Marching Cubes haute précision...")
    verts, faces, normals, _ = measure.marching_cubes(
        volume_3d,
        level=0.0,
        spacing=(1.0, 1.0, (target_thickness_mm / depth_slices))
    )
    
    # Inversion de l'axe Y pour correspondre à l'orientation 3D standard
    verts[:, 0] = grid_resolution - verts[:, 0]
    
    # Création du maillage Trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    
    # Lissage Taubin pour éliminer les escaliers de voxels sans perdre de volume
    print("[ManifoldHD] Lissage Taubin de surface...")
    try:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=5)
    except Exception as e:
        print(f"[ManifoldHD] Lissage ignoré ({e})")
        
    return mesh
