"""
Script de Génération Ultra-HD 2.7M Polygones (3x la densité 896k)
Sculpture de haute précision sub-millimétrique des traits du visage, lunettes, casque,
musculature et vélo, avec projection couleur 360° Voronoi sans couture.
Calibré pour Bambu Lab A1 Mini (<= 180mm, Watertight).
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import trimesh
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import (
    gaussian_filter,
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    map_coordinates
)
from transformers import pipeline

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
BRAIN_DIR = r"C:\Users\TrendingPC\.gemini\antigravity\brain\6f356426-1b48-4c89-8e8b-a62d57a95f48"

def build_ultra_hd_2_7m():
    print("=" * 70)
    print("DEMARRAGE DU PIPELINE ULTRA-HD (2.7M POLYGONES / 3X SOTA)")
    print("=" * 70)
    t_start = time.time()

    # 1. Image source 1414 x 2000
    img_path = os.path.join(BASE_DIR, "uploads", "upload_019cc412_van_aert_original.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image introuvable: {img_path}")
    
    print(f"[1/7] Chargement et analyse de l'image source : {img_path}")
    img = Image.open(img_path).convert("RGBA")
    arr = np.array(img)
    ys, xs = np.where(arr[:, :, 3] > 30)
    crop = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
    H, W = crop.shape[:2]
    print(f"      Dimensions du sujet détouré : {W} x {H} pixels")

    mask = (crop[:, :, 3] > 30).astype(np.float32)
    dist_px = distance_transform_edt(mask > 0.5)
    boundary_fade = np.clip((dist_px - 4.0) / 16.0, 0.0, 1.0)

    rgb = crop[:, :, :3].astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Voronoi nearest-color extrapolation pour une couleur 360° sans aucune zone noire
    print("      Calcul de l'extrapolation Voronoi 2D pour les tranches et le dos...")
    dist_extrap, indices_extrap = distance_transform_edt(~(mask > 0.5), return_indices=True)
    extrap_rgb = crop[:, :, :3][indices_extrap[0], indices_extrap[1]].astype(np.float32) / 255.0

    # 2. Inpainting de l'albédo (textes et logos pour ne pas creuser le maillot)
    print("[2/7] Filtrage de l'albédo (protection des logos Jumbo-Visma et cuissard)...")
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]

    in_torso_box = (rows >= 435) & (rows <= 795) & (cols >= 220) & (cols <= 570) & (mask > 0.5)
    is_yellow = in_torso_box & (r > 0.44) & (g > 0.36) & ((r - b) > 0.22)
    yellow_region = binary_fill_holes(binary_dilation(is_yellow, iterations=10)) & in_torso_box
    ink_yellow = binary_dilation(yellow_region & (~is_yellow), iterations=4) & yellow_region

    clean_lum = lum.copy()
    wy = gaussian_filter(is_yellow.astype(np.float32), sigma=10.0)
    vy = gaussian_filter(np.where(is_yellow, lum, 0.0), sigma=10.0)
    clean_lum[ink_yellow] = (vy / (wy + 1e-5))[ink_yellow]

    in_shorts_box = (rows >= 795) & (rows <= 1100) & (mask > 0.5)
    is_black_fabric = in_shorts_box & (lum < 0.26)
    shorts_region = binary_fill_holes(binary_dilation(is_black_fabric, iterations=8)) & in_shorts_box
    ink_shorts = binary_dilation(shorts_region & (lum >= 0.26), iterations=3) & shorts_region
    ws = gaussian_filter(is_black_fabric.astype(np.float32), sigma=8.0)
    vs = gaussian_filter(np.where(is_black_fabric, lum, 0.0), sigma=8.0)
    clean_lum[ink_shorts] = (vs / (ws + 1e-5))[ink_shorts]

    # 3. Carte de relief multi-échelle avec ultra-finesse des traits
    print("[3/7] Génération du relief multi-échelle (Depth Anything V2 + micro-traits de 2.5px)...")
    comp_clean = np.stack([clean_lum, clean_lum, clean_lum], axis=-1)
    comp_clean = comp_clean * mask[:, :, None] + 0.5 * (1.0 - mask[:, :, None])
    
    depth_pipe = pipeline(
        task='depth-estimation',
        model='depth-anything/Depth-Anything-V2-Small-hf',
        device=0 if torch.cuda.is_available() else -1
    )
    d_map = np.array(depth_pipe(Image.fromarray((comp_clean * 255).astype(np.uint8)))['depth'], dtype=np.float32)
    d_map = (d_map - d_map.min()) / (d_map.max() - d_map.min() + 1e-6)

    # 1) Méso-relief volumétrique (ondes musculaires et morphologie globale)
    meso_depth = d_map - gaussian_filter(d_map, sigma=40.0)
    # 2) Relief anatomique et vêtements (fermeture éclair, coutures, muscles)
    mid_shading = clean_lum - gaussian_filter(clean_lum, sigma=16.0)
    # 3) MICRO-TRAITS HAUTE FREQUENCE (lunettes, narines, dents/bouche, ouïes du casque, phalanges)
    micro_traits = clean_lum - gaussian_filter(clean_lum, sigma=2.8)

    # Boost ciblé sur le visage, le casque et les lunettes (lignes < 440)
    head_mask = np.where(rows < 440, 1.65, 0.85)

    norm_meso = meso_depth / (np.std(meso_depth[mask > 0.5]) + 1e-5)
    norm_mid = mid_shading / (np.std(mid_shading[mask > 0.5]) + 1e-5)
    norm_micro = micro_traits / (np.std(micro_traits[mask > 0.5]) + 1e-5)

    total_2d_relief = (0.35 * norm_meso + 0.40 * norm_mid + 0.25 * norm_micro) * head_mask * boundary_fade
    total_2d_relief = np.clip(total_2d_relief / 2.3, -1.0, 1.0)
    print("      Carte de relief multi-échelle calculée avec succès.")

    # 4. Chargement du maillage Hunyuan 512 et Subdivision x3 (~2.7M polygones)
    raw_mesh_path = os.path.join(OUTPUTS_DIR, "sota_test", "vanaert_hunyuan21_512.glb")
    print(f"[4/7] Chargement du maillage de base 512 ({raw_mesh_path})...")
    m_base = trimesh.load(raw_mesh_path, force="mesh")
    print(f"      Maillage d'origine : {len(m_base.vertices):,} sommets, {len(m_base.faces):,} faces")

    # Lissage Taubin de base pour éliminer les micro-facettes de l'octree
    trimesh.smoothing.filter_taubin(m_base, lamb=0.5, nu=-0.53, iterations=10)

    # Calibrage à 120 mm
    m_base.apply_scale(120.0 / m_base.extents[1])
    b = m_base.bounds
    m_base.apply_translation([-(b[0][0]+b[1][0])/2.0, -b[0][1], -(b[0][2]+b[1][2])/2.0])

    print("[5/7] SUBDIVISION HAUTE DENSITE VERS 2.7M+ POLYGONES...")
    # Midpoint subdivision : divise chaque triangle en 4
    v_sub, f_sub = trimesh.remesh.subdivide(m_base.vertices, m_base.faces)
    print(f"      Après subdivision 4x : {len(v_sub):,} sommets, {len(f_sub):,} faces")

    # Simplification ciblée vers 2,688,000 faces (exactement 3x 896,000 faces)
    target_faces = 2688000
    import fast_simplification
    print(f"      Décimation quadrique de haute fidélité vers exactement {target_faces:,} faces (3x densité)...")
    t_dec = time.time()
    v_simp, f_simp = fast_simplification.simplify(
        v_sub.astype(np.float32),
        f_sub.astype(np.int64),
        target_reduction=1.0 - (target_faces / len(f_sub))
    )
    m_high = trimesh.Trimesh(vertices=v_simp, faces=f_simp, process=False)
    print(f"      Simplification terminée en {time.time()-t_dec:.1f}s : {len(m_high.vertices):,} sommets, {len(m_high.faces):,} faces")

    # Lissage Taubin après subdivision pour garantir une courbure continue sans facettes
    trimesh.smoothing.filter_taubin(m_high, lamb=0.45, nu=-0.47, iterations=3)

    # 5. Déplacement sub-millimétrique des traits
    print("[6/7] Sculpture géométrique des micro-traits sur les 2,688,000 polygones...")
    verts = m_high.vertices.copy()
    norms = m_high.vertex_normals.copy()
    xmin, ymin, zmin = m_high.bounds[0]
    xmax, ymax, zmax = m_high.bounds[1]

    u_coords = np.clip((verts[:, 0] - xmin) / (xmax - xmin + 1e-6) * (W - 1), 0, W - 1)
    v_coords = np.clip((1.0 - (verts[:, 1] - ymin) / (ymax - ymin + 1e-6)) * (H - 1), 0, H - 1)

    vis_n = np.clip((norms[:, 2] - 0.15) / 0.55, 0.0, 1.0) ** 1.3
    z_front_fade = np.where(
        verts[:, 1] >= 56.0,
        np.clip((verts[:, 2] + 15.0) / 10.0, 0.0, 1.0),
        np.clip((verts[:, 2] + 5.0) / 10.0, 0.0, 1.0)
    )
    above_wheels = np.clip((verts[:, 1] - 25.0) / 8.0, 0.0, 1.0)
    front_weight = vis_n * z_front_fade * above_wheels

    sampled_relief = map_coordinates(total_2d_relief, [v_coords, u_coords], order=1, mode='nearest')
    # Amplitude de déplacement chirurgicale : +/- 0.65 mm (parfaite pour les traits fins)
    disp_mm = sampled_relief * front_weight * 0.65
    m_high.vertices = verts + norms * disp_mm[:, None]

    # Micro-lissage local pour intégrer harmonieusement les reliefs sculptés
    trimesh.smoothing.filter_taubin(m_high, lamb=0.28, nu=-0.29, iterations=2)

    # 6. Texturation 360° Voronoi Seamless
    print("[7/7] Texturation haute résolution 360° Voronoi (sans couture)...")
    v_colors = np.zeros((len(m_high.vertices), 4), dtype=np.uint8)
    norms_final = m_high.vertex_normals
    ao_light = np.clip(0.75 + 0.25 * np.maximum(0.0, norms_final[:, 2]), 0.68, 1.0)

    for ch in range(3):
        sampled_c = map_coordinates(extrap_rgb[:, :, ch], [v_coords, u_coords], order=1, mode='nearest')
        v_colors[:, ch] = np.clip(sampled_c * ao_light * 255.0, 0, 255).astype(np.uint8)
    v_colors[:, 3] = 255
    m_high.visual.vertex_colors = v_colors

    # 7. Création et export des versions "Sans Socle" et "Avec Socle"
    from core.bambu_validator import validate_and_scale_for_bambu, export_all_formats
    from core.engine_360 import postprocess_360_figurine
    from core.studio_renderer import render_mesh_preview

    # A) Version SANS SOCLE (2.69M faces)
    dir_nobase = os.path.join(OUTPUTS_DIR, "vanaert_2.7m_no_base")
    os.makedirs(dir_nobase, exist_ok=True)
    m_nobase = postprocess_360_figurine(
        mesh=m_high.copy(),
        target_height_mm=120.0,
        include_base=False,
        smooth_iterations=0
    )
    print(f"\n>>> EXPORT SANS SOCLE : {len(m_nobase.vertices):,} sommets, {len(m_nobase.faces):,} faces")
    validate_and_scale_for_bambu(m_nobase, target_height_mm=0, enforce_watertight=True)
    export_all_formats(m_nobase, os.path.join(dir_nobase, "model_bambu"))

    # B) Version AVEC SOCLE (2.69M faces)
    dir_withbase = os.path.join(OUTPUTS_DIR, "vanaert_2.7m_with_base")
    os.makedirs(dir_withbase, exist_ok=True)
    m_withbase = postprocess_360_figurine(
        mesh=m_high.copy(),
        target_height_mm=120.0,
        include_base=True,
        base_height_mm=5.0,
        smooth_iterations=0
    )
    print(f">>> EXPORT AVEC SOCLE : {len(m_withbase.vertices):,} sommets, {len(m_withbase.faces):,} faces")
    validate_and_scale_for_bambu(m_withbase, target_height_mm=0, enforce_watertight=True)
    export_all_formats(m_withbase, os.path.join(dir_withbase, "model_bambu"))

    # 8. Rendu des prévisualisations Studio Ultra-HD (Full et Macro Zoom Visage)
    print("\n[8/8] Génération des rendus Studio Haute Résolution et Zoom Macro...")
    
    # 3/4 avec socle
    p_base_34 = os.path.join(BRAIN_DIR, "sota_2.7m_with_base_34.png")
    render_mesh_preview(m_withbase, p_base_34, img_size=1024, angle_deg=28.0)
    print(f"      Rendu 3/4 avec socle : {p_base_34}")

    # Face avec socle
    p_base_front = os.path.join(BRAIN_DIR, "sota_2.7m_with_base_front.png")
    render_mesh_preview(m_withbase, p_base_front, img_size=1024, angle_deg=0.0)
    print(f"      Rendu Face avec socle : {p_base_front}")

    # 3/4 sans socle
    p_nobase_34 = os.path.join(BRAIN_DIR, "sota_2.7m_no_base_34.png")
    render_mesh_preview(m_nobase, p_nobase_34, img_size=1024, angle_deg=28.0)
    print(f"      Rendu 3/4 sans socle : {p_nobase_34}")

    # Profil sans socle
    p_nobase_side = os.path.join(BRAIN_DIR, "sota_2.7m_no_base_side.png")
    render_mesh_preview(m_nobase, p_nobase_side, img_size=1024, angle_deg=75.0)
    print(f"      Rendu Profil sans socle : {p_nobase_side}")

    # ZOOM MACRO SUR LA TETE (Casque, lunettes, nez, bouche, dents sculptées)
    print("      Génération du Zoom Macro Visage & Casque...")
    head_sub = m_nobase.copy()
    head_mask = head_sub.vertices[:, 1] >= 82.0  # Zone tête + cou + lunettes + casque
    # Conserver uniquement les faces du haut du corps / tête
    face_in_head = np.all(head_mask[head_sub.faces], axis=1)
    if np.sum(face_in_head) > 500:
        head_mesh = trimesh.Trimesh(
            vertices=head_sub.vertices,
            faces=head_sub.faces[face_in_head],
            vertex_colors=head_sub.visual.vertex_colors,
            process=False
        )
        head_mesh.remove_unreferenced_vertices()
        p_head_zoom = os.path.join(BRAIN_DIR, "sota_2.7m_macro_head_zoom.png")
        render_mesh_preview(head_mesh, p_head_zoom, img_size=1024, angle_deg=18.0)
        print(f"      Rendu Zoom Macro Tête : {p_head_zoom}")

    print("=" * 70)
    print(f"PIPELINE ULTRA-HD 2.7M TERMINE AVEC SUCCES EN {time.time()-t_start:.1f}s !")
    print("=" * 70)

if __name__ == "__main__":
    build_ultra_hd_2_7m()
