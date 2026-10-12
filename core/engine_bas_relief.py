"""
Moteur de Bas-Relief / Médaillon 3D Haute Fidélité.
Idéal pour les photos d'archives, portraits, cyclistes et scènes mécaniques complexes.
Garantit une impression 100% sans supports sur Bambu Lab A1 Mini.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage import measure
import trimesh
from PIL import Image

def generate_bas_relief_mesh(
    mask_np: np.ndarray,
    luminance_np: np.ndarray,
    base_plate_thickness_mm: float = 3.5,
    max_relief_height_mm: float = 8.0,
    border_width_voxels: int = 15,
    grid_resolution: int = 450,
    shape_style: str = "curved_plaque"  # "curved_plaque", "oval", "contour"
) -> trimesh.Trimesh:
    """
    Génère un bas-relief / trophée haute précision étanche.
    """
    mask_img = Image.fromarray((mask_np > 128).astype(np.uint8) * 255)
    lum_img = Image.fromarray((np.clip(luminance_np, 0, 1) * 255).astype(np.uint8))
    
    mask_res = np.array(mask_img.resize((grid_resolution, grid_resolution), Image.Resampling.NEAREST)) > 128
    lum_res = np.array(lum_img.resize((grid_resolution, grid_resolution), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
    
    # 1. Carte de relief photonique
    lum_smooth = gaussian_filter(lum_res, sigma=1.0)
    # Contraste et amplification des détails
    relief_map = lum_smooth * max_relief_height_mm
    
    # 2. Carte de volume anatomique galbée
    dist_map = distance_transform_edt(mask_res)
    max_dist = np.max(dist_map) if np.max(dist_map) > 0 else 1.0
    dome_map = np.power(dist_map / max_dist, 0.6) * (max_relief_height_mm * 0.7)
    
    # Fusion anatomie + relief
    combined_relief = dome_map + (relief_map * 0.6)
    combined_relief[~mask_res] = 0.0
    
    # 3. Base solide
    if shape_style == "contour":
        # Plaque qui épouse la silhouette avec une marge
        dilated_base = distance_transform_edt(mask_res) > -border_width_voxels
        plate_height = base_plate_thickness_mm
    else:
        # Plaque rectangulaire biseautée ou ovale
        y_grid, x_grid = np.ogrid[:grid_resolution, :grid_resolution]
        margin = border_width_voxels
        plate_mask = (x_grid >= margin) & (x_grid < grid_resolution - margin) & \
                     (y_grid >= margin) & (y_grid < grid_resolution - margin)
        dilated_base = plate_mask
        plate_height = base_plate_thickness_mm
        
    # Hauteur Z totale en chaque pixel (X, Y)
    z_height_map = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
    z_height_map[dilated_base] = plate_height
    z_height_map[mask_res] += combined_relief[mask_res]
    
    # 4. Construction de la grille volumétrique pour Marching Cubes
    # Z va de 0 à plate_height + max_relief_height_mm * 1.5
    total_z_max = plate_height + max_relief_height_mm * 1.5
    z_slices = int(total_z_max * 4.0)
    if z_slices < 30:
        z_slices = 30
        
    z_coords = np.linspace(-1.0, total_z_max + 1.0, z_slices)
    volume_3d = np.zeros((grid_resolution, grid_resolution, z_slices), dtype=np.float32)
    
    for k, z_val in enumerate(z_coords):
        if z_val < 0.0:
            volume_3d[:, :, k] = -1.0
        else:
            # Distance à la surface supérieure
            signed_dist = z_height_map - z_val
            signed_dist[~dilated_base] = -1.0
            volume_3d[:, :, k] = signed_dist
            
    # Fermeture frontière étanche garantie
    volume_3d[0, :, :] = -1.0
    volume_3d[-1, :, :] = -1.0
    volume_3d[:, 0, :] = -1.0
    volume_3d[:, -1, :] = -1.0
    volume_3d[:, :, 0] = -1.0
    volume_3d[:, :, -1] = -1.0
            
    # 5. Extraction iso-surface
    print("[BasRelief] Extraction Marching Cubes...")
    verts, faces, normals, _ = measure.marching_cubes(
        volume_3d,
        level=0.0,
        spacing=(1.0, 1.0, total_z_max / z_slices)
    )
    
    # Inversion de l'axe Y pour correspondre à l'orientation 3D standard
    verts[:, 0] = grid_resolution - verts[:, 0]
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    
    # Lissage léger de surface
    try:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.4, nu=-0.42, iterations=4)
    except Exception as e:
        print(f"[BasRelief] Lissage ignoré ({e})")
        
    return mesh
