"""
Pipeline Masterpiece V3 : Modélisation 3D Ultra-Haute Définition (5 376 000 Polygones)
Sculpture Faciale & Corporelle Haute Précision Anatomique (Relief Neuronal Pur du Visage)
Calibrage Isométrique et Texture PBR 2K pour Wout van Aert.
"""

import os
import sys
import time
import shutil
import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel, map_coordinates
import fast_simplification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.studio_renderer import render_mesh_preview

BRAIN_DIR = r"C:\Users\TrendingPC\.gemini\antigravity\brain\6f356426-1b48-4c89-8e8b-a62d57a95f48"

def build_vanaert_5_4m_masterpiece():
    print("=" * 80)
    print("DEMARRAGE DU PIPELINE MASTERPIECE V3 FINAL (5 376 000 POLYGONES)")
    print("SCULPTURE FACIALE PURE NEURONALE & CALIBRAGE ISOMETRIQUE SANS DEFORMATION")
    print("=" * 80)
    t0 = time.time()

    # 1. Image source 1414 x 2000
    img_path = os.path.join(BASE_DIR, "uploads", "upload_019cc412_van_aert_original.png")
    src_img = Image.open(img_path).convert("RGBA")
    arr_src = np.array(src_img)

    alpha = arr_src[:, :, 3]
    ys, xs = np.where(alpha > 30)
    crop_src = arr_src[ys.min():ys.max()+1, xs.min():xs.max()+1]
    H_crop, W_crop = crop_src.shape[:2]
    print(f"[1/7] Sujet source recadré : {W_crop} x {H_crop} pixels")

    # Inpainting Voronoi des bordures pour éliminer les franges sombres
    mask = (crop_src[:, :, 3] > 30).astype(np.float32)
    dist_ex, idx_ex = distance_transform_edt(~(mask > 0.5), return_indices=True)
    dilated_crop_rgb = crop_src[:, :, :3][idx_ex[0], idx_ex[1]]
    dilated_crop_rgba = np.dstack([dilated_crop_rgb, np.full((H_crop, W_crop), 255, dtype=np.uint8)])

    # 2. Carte de relief neuronale du corps et de la tête
    print("[2/7] Extraction du relief neuronal haute définition (Corps & Tête)...")
    import torch
    from transformers import pipeline
    depth_pipe = pipeline(
        task='depth-estimation',
        model='depth-anything/Depth-Anything-V2-Small-hf',
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Relief corps
    d_res_body = depth_pipe(Image.fromarray(crop_src[:, :, :3]))['depth']
    d_body = np.array(d_res_body, dtype=np.float32)
    d_body = (d_body - d_body.min()) / (d_body.max() - d_body.min() + 1e-6)
    d_body_macro = d_body - gaussian_filter(d_body, sigma=45.0)
    d_body_macro = d_body_macro / (np.std(d_body_macro[mask > 0.5]) + 1e-5)
    grad_body = np.hypot(sobel(d_body, axis=0), sobel(d_body, axis=1))
    grad_body = grad_body / (np.std(grad_body[mask > 0.5]) + 1e-5)
    total_relief_body = 0.82 * d_body_macro + 0.18 * np.clip(grad_body - 0.35, 0.0, 2.5)
    dist_px = distance_transform_edt(mask > 0.5)
    boundary_fade = np.clip((dist_px - 4.0) / 16.0, 0.0, 1.0)
    total_relief_body = np.clip(total_relief_body * boundary_fade / 2.0, -1.0, 1.0)

    # Relief tête haute résolution (sans aucune approximation géométrique)
    head_crop_rgb = crop_src[250:520, 300:550, :3]
    d_res_head = depth_pipe(Image.fromarray(head_crop_rgb))['depth']
    d_head = np.array(d_res_head, dtype=np.float32)
    d_head = (d_head - d_head.min()) / (d_head.max() - d_head.min() + 1e-6)
    d_head_smooth = gaussian_filter(d_head, sigma=2.0)
    d_head_relief = (d_head_smooth - gaussian_filter(d_head_smooth, sigma=25.0)) * 3.5

    # 3. Chargement et Double Subdivision vers 5 376 000 faces
    base_glb = os.path.join(BASE_DIR, "outputs", "sota_test", "vanaert_hunyuan21_512.glb")
    print(f"[3/7] Chargement du maillage de base ({base_glb})...")
    m_base = trimesh.load(base_glb, force="mesh")
    trimesh.smoothing.filter_taubin(m_base, lamb=0.5, nu=-0.53, iterations=12)

    m_base.apply_scale(120.0 / m_base.extents[1])
    b = m_base.bounds
    m_base.apply_translation([-(b[0][0]+b[1][0])/2.0, -b[0][1], -(b[0][2]+b[1][2])/2.0])

    print("[4/7] DOUBLE SUBDIVISION VERS 5 376 000 FACES...")
    v_sub1, f_sub1 = trimesh.remesh.subdivide(m_base.vertices, m_base.faces)
    v_sub2, f_sub2 = trimesh.remesh.subdivide(v_sub1, f_sub1)

    target_count = 5376000
    target_reduction = 1.0 - (float(target_count) / len(f_sub2))
    v_simp, f_simp = fast_simplification.simplify(
        v_sub2.astype(np.float32),
        f_sub2.astype(np.int64),
        target_reduction=target_reduction
    )
    m = trimesh.Trimesh(vertices=v_simp, faces=f_simp, process=False)
    trimesh.smoothing.filter_taubin(m, lamb=0.45, nu=-0.48, iterations=3)
    print(f"      Maillage Ultra-HD : {len(m.vertices):,} sommets, {len(m.faces):,} faces")

    # 4. Calibration Isométrique Parfaite et Sculpture 3D
    print("[5/7] Sculpture anatomique (visage pur neuronal, torse héroïque, vélo)...")
    verts = m.vertices.copy()
    norms = m.vertex_normals.copy()

    # Repères anatomiques isométriques calqués sur la photo
    anchors_y_3d = np.array([0.0, 30.0, 60.0, 82.0, 90.0, 93.5, 97.0, 100.5, 104.0, 107.5, 120.0])
    anchors_row_2d = np.array([1952.0, 1480.0, 960.0, 620.0, 485.0, 430.0, 390.0, 350.0, 310.0, 250.0, 0.0])
    map_y_to_row = interp1d(anchors_y_3d, anchors_row_2d, kind='linear', fill_value='extrapolate')

    rows_proj = np.clip(map_y_to_row(verts[:, 1]), 0, H_crop - 1)
    t_head = np.clip((verts[:, 1] - 80.0) / 12.0, 0.0, 1.0)
    scale_x = 12.0 * t_head + 14.0 * (1.0 - t_head)
    center_col = 432.0 * t_head + 405.0 * (1.0 - t_head)
    center_3d_x = 0.8 * t_head + 0.0 * (1.0 - t_head)

    cols_proj = np.clip(center_col + (verts[:, 0] - center_3d_x) * scale_x, 0, W_crop - 1)

    # Relief volumétrique du corps
    vis_n = np.clip((norms[:, 2] - 0.05) / 0.60, 0.0, 1.0) ** 1.15
    z_fade = np.clip((verts[:, 2] + 16.0) / 12.0, 0.0, 1.0)
    above_wheels = np.clip((verts[:, 1] - 22.0) / 8.0, 0.0, 1.0)
    front_weight = vis_n * z_fade * above_wheels

    sampled_relief_body = map_coordinates(total_relief_body, [rows_proj, cols_proj], order=1, mode='nearest')
    relief_body_mm = np.where(verts[:, 1] >= 50.0, 1.75, 1.25)
    disp_body = sampled_relief_body * front_weight * relief_body_mm

    # Sculpture pure neuronale de la tête
    r_in_head = np.clip(rows_proj - 250.0, 0, 269)
    c_in_head = np.clip(cols_proj - 300.0, 0, 249)
    sampled_head_d = map_coordinates(d_head_relief, [r_in_head, c_in_head], order=1, mode='nearest')

    is_head_front = (verts[:, 1] >= 89.0) & (verts[:, 1] <= 107.5) & (verts[:, 2] >= -12.0)
    fade_head = np.clip((norms[:, 2] - 0.1) / 0.5, 0.0, 1.0) * is_head_front
    disp_head_mm = np.clip(sampled_head_d * fade_head * 1.5, -2.0, 2.5)

    # Fusion des reliefs corps et tête
    total_disp_normals = np.where(is_head_front, disp_head_mm, disp_body)
    m.vertices = verts + norms * total_disp_normals[:, None]
    trimesh.smoothing.filter_taubin(m, lamb=0.35, nu=-0.38, iterations=1)
    print("      Sculpture anatomique terminée avec succès !")

    # 5. Paramétrisation UV et Atlas de Texture PBR 2K
    print("[6/7] Paramétrisation UV isométrique et texture PBR 2K...")
    TEX_SIZE = 2048
    HALF_W = TEX_SIZE // 2
    atlas = np.zeros((TEX_SIZE, TEX_SIZE, 4), dtype=np.uint8)
    atlas[:, :, 3] = 255

    front_img = Image.fromarray(dilated_crop_rgba).resize((HALF_W, TEX_SIZE), Image.Resampling.LANCZOS)
    atlas[:, :HALF_W] = np.array(front_img)

    back_arr = np.zeros((TEX_SIZE, HALF_W, 4), dtype=np.uint8)
    back_arr[:, :, 3] = 255
    for y in range(TEX_SIZE):
        norm_y = 1.0 - (y / TEX_SIZE)
        if norm_y > 0.88:
            back_arr[y, :, :3] = [32, 40, 58]  # Casque Red Bull dos
        elif norm_y > 0.80:
            back_arr[y, :, :3] = [190, 145, 120]  # Peau / nuque
        elif norm_y > 0.45:
            for x in range(HALF_W):
                edge_dist = min(x, HALF_W - 1 - x) / float(HALF_W / 2)
                if edge_dist < 0.22:
                    back_arr[y, x, :3] = [28, 28, 30]  # Panneaux noirs
                else:
                    back_arr[y, x, :3] = [245, 215, 20]  # Jaune officiel
        elif norm_y > 0.28:
            back_arr[y, :, :3] = [22, 22, 25]  # Cuissard noir
        else:
            back_arr[y, :, :3] = [35, 36, 38]  # Vélo carbone Cervélo

    atlas[:, HALF_W:] = back_arr
    atlas_img = Image.fromarray(atlas)

    # Calcul des UVs sur le maillage final sculpté
    m_verts = m.vertices
    m_norms = m.vertex_normals
    rows_m = np.clip(map_y_to_row(m_verts[:, 1]), 0, H_crop - 1)
    v_coords = 1.0 - (rows_m / float(H_crop))

    t_head_m = np.clip((m_verts[:, 1] - 80.0) / 12.0, 0.0, 1.0)
    scale_x_m = 12.0 * t_head_m + 14.0 * (1.0 - t_head_m)
    center_col_m = 432.0 * t_head_m + 405.0 * (1.0 - t_head_m)
    center_3d_x_m = 0.8 * t_head_m + 0.0 * (1.0 - t_head_m)

    cols_m = np.clip(center_col_m + (m_verts[:, 0] - center_3d_x_m) * scale_x_m, 0, W_crop - 1)
    cols_norm_m = cols_m / float(W_crop)

    is_front = (m_norms[:, 2] >= -0.15) | (m_verts[:, 2] >= -10.0)
    uvs = np.zeros((len(m_verts), 2), dtype=np.float32)
    uvs[is_front, 0] = cols_norm_m[is_front] * 0.49
    uvs[is_front, 1] = v_coords[is_front]
    uvs[~is_front, 0] = 0.51 + (1.0 - cols_norm_m[~is_front]) * 0.48
    uvs[~is_front, 1] = v_coords[~is_front]
    uvs = np.clip(uvs, 0.0, 1.0)

    pbr_material = trimesh.visual.material.PBRMaterial(
        name="WoutVanAert_Masterpiece_5M",
        baseColorTexture=atlas_img,
        roughnessFactor=0.38,
        metallicFactor=0.06
    )
    m.visual = trimesh.visual.TextureVisuals(uv=uvs, material=pbr_material)

    # 6. Exportations GLB & STL (Sans Socle & Avec Socle)
    print("[7/7] Exportation des modèles GLB & STL (5 376 000 polygones)...")
    out_dir_no_base = os.path.join(BASE_DIR, "outputs", "vanaert_5.4m_no_base")
    out_dir_with_base = os.path.join(BASE_DIR, "outputs", "vanaert_5.4m_with_base")
    demo_dir = os.path.join(BASE_DIR, "static", "demo")
    os.makedirs(out_dir_no_base, exist_ok=True)
    os.makedirs(out_dir_with_base, exist_ok=True)

    glb_nobase = os.path.join(out_dir_no_base, "model_bambu.glb")
    stl_nobase = os.path.join(out_dir_no_base, "model_bambu.stl")
    glb_demo = os.path.join(demo_dir, "Van_Aert_Figurine_A1_Mini.glb")

    m.export(glb_nobase)
    m.export(glb_demo)
    m.export(stl_nobase)
    print(f"      Sans Socle prêt : {glb_nobase} ({os.path.getsize(glb_nobase)/(1024*1024):.1f} Mo)")

    # Version avec socle
    base_file = os.path.join(BASE_DIR, "outputs", "vanaert_896k_with_base", "model_bambu.glb")
    the_base = None
    if os.path.exists(base_file):
        m_base_old = trimesh.load(base_file, force="mesh")
        comps = m_base_old.split(only_watertight=False)
        if len(comps) > 1:
            the_base = sorted(comps, key=lambda c: len(c.faces), reverse=True)[1]
            cyl_uv = np.full((len(the_base.vertices), 2), [0.85, 0.15], dtype=np.float32)
            the_base.visual = trimesh.visual.TextureVisuals(uv=cyl_uv, material=pbr_material)
            m_combined = trimesh.util.concatenate([m, the_base])
            glb_withbase = os.path.join(out_dir_with_base, "model_bambu.glb")
            stl_withbase = os.path.join(out_dir_with_base, "model_bambu.stl")
            m_combined.export(glb_withbase)
            m_combined.export(stl_withbase)
            print(f"      Avec Socle prêt : {glb_withbase} ({os.path.getsize(glb_withbase)/(1024*1024):.1f} Mo)")

    # 7. Rendus Studio Professionnels
    print("      Génération des rendus studio haute résolution...")
    p_front = os.path.join(out_dir_no_base, "preview_front.png")
    render_mesh_preview(m, p_front, img_size=1200, angle_deg=0.0)
    shutil.copy2(p_front, os.path.join(BRAIN_DIR, "masterpiece_5m_front.png"))

    p_34 = os.path.join(out_dir_no_base, "preview_34.png")
    render_mesh_preview(m, p_34, img_size=1200, angle_deg=25.0)
    shutil.copy2(p_34, os.path.join(BRAIN_DIR, "masterpiece_5m_34.png"))

    p_side = os.path.join(out_dir_no_base, "preview_side.png")
    render_mesh_preview(m, p_side, img_size=1200, angle_deg=75.0)
    shutil.copy2(p_side, os.path.join(BRAIN_DIR, "masterpiece_5m_side.png"))

    # Macro Tête
    m_head = m.copy()
    head_mask = (m_head.vertices[:, 1] >= 80.0)
    m_head.update_faces(np.isin(m_head.faces, np.where(head_mask)[0]).all(axis=1))
    m_head.remove_unreferenced_vertices()

    p_macro = os.path.join(out_dir_no_base, "preview_macro_head.png")
    render_mesh_preview(m_head, p_macro, img_size=1200, angle_deg=0.0)
    shutil.copy2(p_macro, os.path.join(BRAIN_DIR, "masterpiece_5m_macro_head.png"))

    if the_base is not None:
        p_base_front = os.path.join(out_dir_with_base, "preview_front.png")
        render_mesh_preview(m_combined, p_base_front, img_size=1200, angle_deg=0.0)
        shutil.copy2(p_base_front, os.path.join(BRAIN_DIR, "masterpiece_5m_with_base_front.png"))

        p_base_34 = os.path.join(out_dir_with_base, "preview_34.png")
        render_mesh_preview(m_combined, p_base_34, img_size=1200, angle_deg=25.0)
        shutil.copy2(p_base_34, os.path.join(BRAIN_DIR, "masterpiece_5m_with_base_34.png"))

    dt = time.time() - t0
    print("=" * 80)
    print(f"MASTERPIECE V3 COMPLETEE EN {dt:.1f}s ! (5 376 000 POLYGONES)")
    print("=" * 80)

if __name__ == "__main__":
    build_vanaert_5_4m_masterpiece()
