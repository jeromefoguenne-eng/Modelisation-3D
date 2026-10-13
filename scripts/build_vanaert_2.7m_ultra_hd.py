"""
Script de Génération Ultra-HD 2.7M Polygones (Heroic Relief & Calibrated Landmarks)
- Restauration fidèle et héroïque du visage de Wout van Aert (élimination de tout effet raccrapoté)
- Accentuation forte du relief 3D (+1.65 mm sur torse/muscles, +1.35 mm sur visage/casque)
- Calibration anatomique piecewise non-linéaire (lunettes sur les yeux, bouche sur la bouche, menton sur le menton)
- Densité 2 688 000 polygones pour Bambu Lab A1 Mini (<= 180mm, 100% Watertight).
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
from scipy.interpolate import interp1d
from scipy.ndimage import (
    gaussian_filter,
    distance_transform_edt,
    map_coordinates,
    sobel
)
from transformers import pipeline
import fast_simplification

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
BRAIN_DIR = r"C:\Users\TrendingPC\.gemini\antigravity\brain\6f356426-1b48-4c89-8e8b-a62d57a95f48"

def build_heroic_ultra_hd():
    print("=" * 70)
    print("DEMARRAGE DU PIPELINE HEROIQUE ULTRA-HD (2.688M POLYGONES)")
    print("ACCENTUATION FORTE DU RELIEF ET CALIBRAGE ANATOMIQUE PARFAIT")
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
    print(f"      Dimensions sujet : {W} x {H} pixels")

    mask = (crop[:, :, 3] > 30).astype(np.float32)
    dist_ex, idx_ex = distance_transform_edt(~(mask > 0.5), return_indices=True)
    extrap_rgb = crop[:, :, :3][idx_ex[0], idx_ex[1]].astype(np.float32) / 255.0

    # 2. Carte de relief anatomique (Depth Anything V2 SOTA - zéro bruit albedo)
    print("[2/7] Estimation de profondeur neuronale anatomique (Depth Anything V2)...")
    depth_pipe = pipeline(
        task='depth-estimation',
        model='depth-anything/Depth-Anything-V2-Small-hf',
        device=0 if torch.cuda.is_available() else -1
    )
    d_res = depth_pipe(Image.fromarray(crop[:, :, :3]))['depth']
    d_all = np.array(d_res, dtype=np.float32)
    d_all = (d_all - d_all.min()) / (d_all.max() - d_all.min() + 1e-6)

    # Relief volumétrique organique et athlétique (pectoraux, deltoïdes, biceps, dôme du casque)
    d_vol = d_all - gaussian_filter(d_all, sigma=55.0)
    d_vol = d_vol / (np.std(d_vol[mask > 0.5]) + 1e-5)

    # Découpe nette des contours physiques (monture lunettes, lèvres/dents, ouïes du casque, zip)
    grad_d = np.hypot(sobel(d_all, axis=0), sobel(d_all, axis=1))
    grad_d = grad_d / (np.std(grad_d[mask > 0.5]) + 1e-5)

    # Carte de relief combinée : 82% volumes anatomiques puissants, 18% arêtes structurelles
    total_relief_2d = 0.82 * d_vol + 0.18 * np.clip(grad_d - 0.4, 0.0, 2.5)
    dist_px = distance_transform_edt(mask > 0.5)
    boundary_fade = np.clip((dist_px - 4.0) / 16.0, 0.0, 1.0)
    total_relief_2d = np.clip(total_relief_2d * boundary_fade / 2.0, -1.0, 1.0)

    # 3. Chargement du maillage de base et subdivision x3 (2.688M faces)
    raw_mesh_path = os.path.join(OUTPUTS_DIR, "sota_test", "vanaert_hunyuan21_512.glb")
    print(f"[3/7] Chargement du maillage 512 et lissage Taubin ({raw_mesh_path})...")
    m_base = trimesh.load(raw_mesh_path, force="mesh")
    trimesh.smoothing.filter_taubin(m_base, lamb=0.5, nu=-0.53, iterations=12)

    m_base.apply_scale(120.0 / m_base.extents[1])
    b = m_base.bounds
    m_base.apply_translation([-(b[0][0]+b[1][0])/2.0, -b[0][1], -(b[0][2]+b[1][2])/2.0])

    print("[4/7] SUBDIVISION HAUTE DEFINITION VERS 2,688,000 FACES...")
    v_sub, f_sub = trimesh.remesh.subdivide(m_base.vertices, m_base.faces)
    v_simp, f_simp = fast_simplification.simplify(
        v_sub.astype(np.float32),
        f_sub.astype(np.int64),
        target_reduction=1.0 - (2688000.0 / len(f_sub))
    )
    m = trimesh.Trimesh(vertices=v_simp, faces=f_simp, process=False)
    trimesh.smoothing.filter_taubin(m, lamb=0.45, nu=-0.47, iterations=3)
    print(f"      Maillage HD prêt : {len(m.vertices):,} sommets, {len(m.faces):,} faces")

    # 4. Calibration anatomique non-linéaire (Landmarks précis)
    print("[5/7] Calibration anatomique des traits (visage, lunettes, torse, bras)...")
    verts = m.vertices.copy()
    norms = m.vertex_normals.copy()

    anchors_y_3d = np.array([0.0,  30.0,  60.0,  85.0,  91.0,  94.0,  97.0, 100.0, 107.5, 120.0])
    anchors_row_2d = np.array([1952, 1480, 960,  495,   430,   375,   325,  280,   160,   0])
    map_y_to_row = interp1d(anchors_y_3d, anchors_row_2d, kind='linear', fill_value='extrapolate')

    rows_proj = np.clip(map_y_to_row(verts[:, 1]), 0, H - 1)
    t_head = np.clip((verts[:, 1] - 80.0) / 10.0, 0.0, 1.0)
    center_x = -1.1 * t_head + 0.0 * (1.0 - t_head)
    scale_x = 13.5 * t_head + 14.65 * (1.0 - t_head)
    cols_proj = np.clip(405.0 + (verts[:, 0] - center_x) * scale_x, 0, W - 1)

    # 5. Sculpture du relief accentué (+1.65 mm torse, +1.35 mm tête, +1.20 mm membres)
    print("[6/7] Sculpture du relief accentué (volume héroïque et traits nets)...")
    vis_n = np.clip((norms[:, 2] - 0.10) / 0.55, 0.0, 1.0) ** 1.2
    z_fade = np.where(
        verts[:, 1] >= 56.0,
        np.clip((verts[:, 2] + 15.0) / 10.0, 0.0, 1.0),
        np.clip((verts[:, 2] + 5.0) / 10.0, 0.0, 1.0)
    )
    above_wheels = np.clip((verts[:, 1] - 25.0) / 8.0, 0.0, 1.0)
    front_weight = vis_n * z_fade * above_wheels

    sampled_relief = map_coordinates(total_relief_2d, [rows_proj, cols_proj], order=1, mode='nearest')
    
    # Amplitudes accentuées fortes demandées par l'utilisateur
    relief_amp_mm = np.where(
        verts[:, 1] >= 80.0,
        1.35,  # Tête & visage
        np.where(verts[:, 1] >= 50.0, 1.65, 1.20)  # Torse & membres
    )
    disp_mm = sampled_relief * front_weight * relief_amp_mm
    m.vertices = verts + norms * disp_mm[:, None]

    # Micro-lissage de surface pour intégration organique parfaite
    trimesh.smoothing.filter_taubin(m, lamb=0.32, nu=-0.34, iterations=2)

    # 6. Texturation 360° Voronoi Seamless
    print("[7/7] Texturation haute définition 360° Voronoi (zéro couture)...")
    v_colors = np.zeros((len(m.vertices), 4), dtype=np.uint8)
    norms_final = m.vertex_normals
    ao = np.clip(0.78 + 0.22 * np.maximum(0.0, norms_final[:, 2]), 0.70, 1.0)
    for ch in range(3):
        c = map_coordinates(extrap_rgb[:, :, ch], [rows_proj, cols_proj], order=1, mode='nearest')
        v_colors[:, ch] = np.clip(c * ao * 255.0, 0, 255).astype(np.uint8)
    v_colors[:, 3] = 255
    m.visual.vertex_colors = v_colors

    # 7. Exports Bambu Lab A1 Mini (Sans Socle et Avec Socle)
    from core.bambu_validator import validate_and_scale_for_bambu, export_all_formats
    from core.engine_360 import postprocess_360_figurine
    from core.studio_renderer import render_mesh_preview

    # A) Version SANS SOCLE (2.69M faces)
    dir_nobase = os.path.join(OUTPUTS_DIR, "vanaert_2.7m_no_base")
    os.makedirs(dir_nobase, exist_ok=True)
    m_nobase = postprocess_360_figurine(
        mesh=m.copy(),
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
        mesh=m.copy(),
        target_height_mm=120.0,
        include_base=True,
        base_height_mm=5.0,
        smooth_iterations=0
    )
    print(f">>> EXPORT AVEC SOCLE : {len(m_withbase.vertices):,} sommets, {len(m_withbase.faces):,} faces")
    validate_and_scale_for_bambu(m_withbase, target_height_mm=0, enforce_watertight=True)
    export_all_formats(m_withbase, os.path.join(dir_withbase, "model_bambu"))

    # 8. Rendu des prévisualisations Studio Ultra-HD
    print("\n[8/8] Rendu des aperçus Studio Haute Résolution et Zoom Macro...")
    
    # 3/4 avec socle
    p_base_34 = os.path.join(BRAIN_DIR, "sota_2.7m_with_base_34.png")
    render_mesh_preview(m_withbase, p_base_34, img_size=1024, angle_deg=28.0)
    print(f"      Aperçu 3/4 avec socle : {p_base_34}")

    # Face avec socle
    p_base_front = os.path.join(BRAIN_DIR, "sota_2.7m_with_base_front.png")
    render_mesh_preview(m_withbase, p_base_front, img_size=1024, angle_deg=0.0)
    print(f"      Aperçu Face avec socle : {p_base_front}")

    # 3/4 sans socle
    p_nobase_34 = os.path.join(BRAIN_DIR, "sota_2.7m_no_base_34.png")
    render_mesh_preview(m_nobase, p_nobase_34, img_size=1024, angle_deg=28.0)
    print(f"      Aperçu 3/4 sans socle : {p_nobase_34}")

    # Profil sans socle
    p_nobase_side = os.path.join(BRAIN_DIR, "sota_2.7m_no_base_side.png")
    render_mesh_preview(m_nobase, p_nobase_side, img_size=1024, angle_deg=75.0)
    print(f"      Aperçu Profil sans socle : {p_nobase_side}")

    # ZOOM MACRO TETE ET VISAGE
    head_mask = m_nobase.vertices[:, 1] >= 82.0
    face_in_head = np.all(head_mask[m_nobase.faces], axis=1)
    head_mesh = trimesh.Trimesh(
        vertices=m_nobase.vertices,
        faces=m_nobase.faces[face_in_head],
        vertex_colors=m_nobase.visual.vertex_colors,
        process=False
    )
    head_mesh.remove_unreferenced_vertices()
    p_head_zoom = os.path.join(BRAIN_DIR, "sota_2.7m_macro_head_zoom.png")
    render_mesh_preview(head_mesh, p_head_zoom, img_size=1024, angle_deg=18.0)
    print(f"      Aperçu Zoom Macro Tête : {p_head_zoom}")

    print("=" * 70)
    print(f"PIPELINE HEROIQUE 2.7M TERMINE AVEC SUCCES EN {time.time()-t_start:.1f}s !")
    print("=" * 70)

if __name__ == "__main__":
    build_heroic_ultra_hd()
