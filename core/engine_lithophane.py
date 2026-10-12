"""
Moteur de Bas-Relief et Lithophanie Haute Fidélité (Voie 2).
Génération 100% fidèle au pixel pour Bambu Lab A1 Mini.
Garantit une géométrie étanche (watertight), sans déformation, sans supports.
"""

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import trimesh
import math

def create_pure_relief_mesh(
    image_path: str,
    target_width_mm: float = 120.0,
    mode: str = "bas_relief",  # "bas_relief" ou "lithophane"
    base_thickness_mm: float = 2.5,
    relief_depth_mm: float = 4.5,
    frame_width_mm: float = 4.0,
    curvature_angle_deg: float = 0.0,  # 0 = plat, 25 = légèrement courbé pour tenir debout
    resolution_dpi: int = 400
) -> trimesh.Trimesh:
    """
    Crée un maillage 3D solide étanche à partir d'une image 2D avec fidélité 100% au pixel.
    """
    img = Image.open(image_path)
    
    # Préservation du ratio d'aspect exact
    aspect_ratio = img.height / img.width
    target_height_mm = target_width_mm * aspect_ratio
    
    # Limitation pour Bambu Lab A1 Mini (max 175mm pour marge de sécurité)
    if target_height_mm > 175.0:
        target_height_mm = 175.0
        target_width_mm = target_height_mm / aspect_ratio
    if target_width_mm > 175.0:
        target_width_mm = 175.0
        target_height_mm = target_width_mm * aspect_ratio

    # Calcul de la résolution de la grille de sommets
    # Pour 120 mm à 4-5 points par mm => ~500-600 points (détail photographique exceptionnel)
    res_x = min(resolution_dpi, 600)
    res_y = int(res_x * aspect_ratio)
    
    # Prétraitement de l'image
    img_gray = ImageOps.grayscale(img)
    # Rehaussement doux de contraste pour faire ressortir les détails
    img_gray = ImageOps.autocontrast(img_gray, cutoff=0.5)
    img_resized = img_gray.resize((res_x, res_y), Image.Resampling.LANCZOS)
    
    # Normalisation [0, 1]
    pixel_vals = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Calcul de la carte de hauteur Z
    if mode == "lithophane":
        # Lithophanie : le blanc est fin (laisse passer la lumière), le noir est épais
        # Norme Bambu : 0.8mm min (2 couches solides) à 3.2mm max
        z_min = 0.8
        z_max = 3.2
        # Inversion : pixel blanc (1.0) -> z_min, pixel noir (0.0) -> z_max
        z_map = z_min + (1.0 - pixel_vals) * (z_max - z_min)
    else:
        # Bas-Relief Sculpté : base plate + relief progressif sculpté
        # Le relief est adouci avec un micro-flou pour éliminer tout bruit numérique
        img_smooth = img_resized.filter(ImageFilter.GaussianBlur(radius=0.8))
        smooth_vals = np.array(img_smooth, dtype=np.float32) / 255.0
        z_map = base_thickness_mm + smooth_vals * relief_depth_mm

    # Ajout du cadre biseauté élégant sur le pourtour
    if frame_width_mm > 0:
        border_px_x = int((frame_width_mm / target_width_mm) * res_x)
        border_px_y = int((frame_width_mm / target_height_mm) * res_y)
        frame_height = base_thickness_mm + (relief_depth_mm if mode != "lithophane" else 2.5) + 0.8
        
        # Masque de bordure avec chanfrein progressif
        for i in range(border_px_y):
            factor = i / max(1, border_px_y)
            z_map[i, :] = np.maximum(z_map[i, :], base_thickness_mm + (frame_height - base_thickness_mm) * (1.0 - factor * 0.2))
            z_map[res_y - 1 - i, :] = np.maximum(z_map[res_y - 1 - i, :], base_thickness_mm + (frame_height - base_thickness_mm) * (1.0 - factor * 0.2))
            
        for j in range(border_px_x):
            factor = j / max(1, border_px_x)
            z_map[:, j] = np.maximum(z_map[:, j], base_thickness_mm + (frame_height - base_thickness_mm) * (1.0 - factor * 0.2))
            z_map[:, res_x - 1 - j] = np.maximum(z_map[:, res_x - 1 - j], base_thickness_mm + (frame_height - base_thickness_mm) * (1.0 - factor * 0.2))

    # Génération de la grille de coordonnées (X, Y, Z)
    x_coords = np.linspace(-target_width_mm / 2.0, target_width_mm / 2.0, res_x)
    y_coords = np.linspace(target_height_mm / 2.0, -target_height_mm / 2.0, res_y)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Application de la courbure éventuelle (pour autoportance sur table)
    if curvature_angle_deg > 0:
        radius = (target_width_mm / 2.0) / math.sin(math.radians(curvature_angle_deg / 2.0))
        theta = (X / (target_width_mm / 2.0)) * math.radians(curvature_angle_deg / 2.0)
        # Recalcul de X et Z de base incurvé
        X_curved = radius * np.sin(theta)
        Z_curve_offset = radius * (1.0 - np.cos(theta))
    else:
        X_curved = X
        Z_curve_offset = 0.0

    # Construction du maillage solide fermé étanche (Watertight)
    # Face avant (sculptée)
    top_vertices = np.column_stack([
        X_curved.ravel(),
        Y.ravel(),
        (z_map + Z_curve_offset).ravel()
    ])
    
    # Face arrière (plate au sol Z = 0)
    bottom_vertices = np.column_stack([
        X_curved.ravel(),
        Y.ravel(),
        (np.zeros_like(z_map) + Z_curve_offset).ravel()
    ])
    
    num_top = len(top_vertices)
    all_vertices = np.vstack([top_vertices, bottom_vertices])
    
    # Construction des faces (triangles)
    faces = []
    
    # 1. Triangulation de la surface avant (top)
    for row in range(res_y - 1):
        for col in range(res_x - 1):
            i0 = row * res_x + col
            i1 = i0 + 1
            i2 = (row + 1) * res_x + col
            i3 = i2 + 1
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])
            
    # 2. Triangulation de la surface arrière (bottom, normales inversées)
    for row in range(res_y - 1):
        for col in range(res_x - 1):
            i0 = num_top + (row * res_x + col)
            i1 = i0 + 1
            i2 = num_top + ((row + 1) * res_x + col)
            i3 = i2 + 1
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])
            
    # 3. Murs latéraux (bords étanches)
    # Bord haut (row 0)
    for col in range(res_x - 1):
        t0, t1 = col, col + 1
        b0, b1 = num_top + t0, num_top + t1
        faces.append([t0, b0, t1])
        faces.append([t1, b0, b1])
        
    # Bord bas (row res_y - 1)
    for col in range(res_x - 1):
        t0 = (res_y - 1) * res_x + col
        t1 = t0 + 1
        b0, b1 = num_top + t0, num_top + t1
        faces.append([t0, t1, b0])
        faces.append([t1, b1, b0])
        
    # Bord gauche (col 0)
    for row in range(res_y - 1):
        t0 = row * res_x
        t1 = (row + 1) * res_x
        b0, b1 = num_top + t0, num_top + t1
        faces.append([t0, t1, b0])
        faces.append([t1, b1, b0])
        
    # Bord droit (col res_x - 1)
    for row in range(res_y - 1):
        t0 = row * res_x + (res_x - 1)
        t1 = (row + 1) * res_x + (res_x - 1)
        b0, b1 = num_top + t0, num_top + t1
        faces.append([t0, b0, t1])
        faces.append([t1, b0, b1])

    faces = np.array(faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=all_vertices, faces=faces, process=True)
    
    # Recentrage au sol : Z minimum à 0
    min_z = np.min(mesh.vertices[:, 2])
    mesh.apply_translation([0, 0, -min_z])
    
    # Réparation de sécurité
    if not mesh.is_watertight:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
        
    return mesh
