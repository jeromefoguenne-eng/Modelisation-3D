"""
Script de Création de Texture PBR Ultra-Haute Définition (2048x2048)
et Paramétrisation UV Dual-Hémisphère pour Wout van Aert.
Génère un GLB avec texture diffuse 2K, normales lisses et matériaux PBR professionnels.
"""

import os
import sys
import shutil
import time
import numpy as np
import trimesh
from PIL import Image, ImageFilter
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.studio_renderer import render_mesh_preview

def build_hd_textured_masterpiece():
    print("=" * 75)
    print("CREATION DE LA TEXTURE HD PBR 2K & PARAMETRISATION UV MATHEMATIQUE")
    print("=" * 75)

    # 1. Image source 1414 x 2000
    img_path = os.path.join(BASE_DIR, "uploads", "upload_019cc412_van_aert_original.png")
    src_img = Image.open(img_path).convert("RGBA")
    arr_src = np.array(src_img)
    
    alpha = arr_src[:, :, 3]
    ys, xs = np.where(alpha > 30)
    crop_src = arr_src[ys.min():ys.max()+1, xs.min():xs.max()+1]
    H_crop, W_crop = crop_src.shape[:2]
    print(f"[1/6] Sujet source recadré : {W_crop} x {H_crop} pixels")

    # Dilation Voronoi des bordures pour éviter tout artefact de frange sombre
    mask = (crop_src[:, :, 3] > 30).astype(np.float32)
    dist_ex, idx_ex = distance_transform_edt(~(mask > 0.5), return_indices=True)
    dilated_crop_rgb = crop_src[:, :, :3][idx_ex[0], idx_ex[1]]
    dilated_crop_rgba = np.dstack([dilated_crop_rgb, np.full((H_crop, W_crop), 255, dtype=np.uint8)])

    # 2. Création de l'Atlas de Texture 2048 x 2048
    # Moitié Gauche [0, 1024[ : Face Avant HD Photographique
    # Moitié Droite [1024, 2048[ : Face Arrière PBR Studio
    TEX_SIZE = 2048
    HALF_W = TEX_SIZE // 2
    atlas = np.zeros((TEX_SIZE, TEX_SIZE, 4), dtype=np.uint8)
    atlas[:, :, 3] = 255

    # A) Face Avant : Redimensionnement haute fidélité (Lanczos) vers [TEX_SIZE, HALF_W]
    front_img = Image.fromarray(dilated_crop_rgba).resize((HALF_W, TEX_SIZE), Image.Resampling.LANCZOS)
    front_arr = np.array(front_img)
    atlas[:, :HALF_W] = front_arr
    print(f"[2/6] Atlas Face Avant inséré : {HALF_W} x {TEX_SIZE} pixels")

    # B) Face Arrière : Maillot Jaune Visma | Lease a Bike, cuissard noir, casque Red Bull dos
    back_arr = np.zeros((TEX_SIZE, HALF_W, 4), dtype=np.uint8)
    back_arr[:, :, 3] = 255

    for y in range(TEX_SIZE):
        norm_y = 1.0 - (y / TEX_SIZE)  # 0 en bas (sol), 1 en haut (casque)
        if norm_y > 0.88:
            # Casque Red Bull dos (Bleu nuit / Blanc / Carbone)
            back_arr[y, :, :3] = [32, 40, 58]
        elif norm_y > 0.80:
            # Nuque & cheveux châtains
            back_arr[y, :, :3] = [190, 145, 120]
        elif norm_y > 0.45:
            # Maillot Jaune Visma | Lease a Bike
            # Dégradé jaune avec bandes latérales sombres
            for x in range(HALF_W):
                edge_dist = min(x, HALF_W - 1 - x) / float(HALF_W / 2)
                if edge_dist < 0.25:
                    back_arr[y, x, :3] = [28, 28, 30]  # Flancs noirs du maillot
                else:
                    back_arr[y, x, :3] = [245, 215, 20]  # Jaune abeille officiel
        elif norm_y > 0.28:
            # Cuissard de cyclisme noir mat
            back_arr[y, :, :3] = [22, 22, 25]
        else:
            # Vélo Carbone Cervélo & Roues
            back_arr[y, :, :3] = [35, 36, 38]

    atlas[:, HALF_W:] = back_arr
    atlas_img = Image.fromarray(atlas)

    tex_out_path = os.path.join(BASE_DIR, "outputs", "sota_test", "vanaert_texture_2k.png")
    atlas_img.save(tex_out_path, format="PNG")
    print(f"[3/6] Atlas de texture 2K sauvegardé : {tex_out_path}")

    # 3. Chargement du Maillage 2.688M
    glb_in = os.path.join(BASE_DIR, "outputs", "vanaert_2.7m_no_base", "model_bambu.glb")
    m = trimesh.load(glb_in, force="mesh")
    print(f"[4/6] Maillage chargé : {len(m.vertices):,} sommets, {len(m.faces):,} faces")

    verts = m.vertices.copy()
    norms = m.vertex_normals.copy()

    # 4. Paramétrisation Mathématique des Coordonnées UV
    # Anchors anatomiques identiques à build_vanaert_2.7m_ultra_hd.py
    anchors_y_3d = np.array([0.0,  30.0,  60.0,  85.0,  91.0,  94.0,  97.0, 100.0, 107.5, 120.0])
    anchors_row_norm = np.array([1952.0, 1480.0, 960.0, 495.0, 430.0, 375.0, 325.0, 280.0, 160.0, 0.0]) / float(H_crop)
    map_y_to_v = interp1d(anchors_y_3d, anchors_row_norm, kind='linear', fill_value='extrapolate')

    rows_norm = np.clip(map_y_to_v(verts[:, 1]), 0.0, 1.0)
    # Dans les textures UV OpenGL/glTF : V = 0 en bas, V = 1 en haut
    v_coords = 1.0 - rows_norm

    # Projection horizontale X
    t_head = np.clip((verts[:, 1] - 80.0) / 10.0, 0.0, 1.0)
    center_x = -1.1 * t_head + 0.0 * (1.0 - t_head)
    scale_x = 13.5 * t_head + 14.65 * (1.0 - t_head)
    cols_proj = 405.0 + (verts[:, 0] - center_x) * scale_x
    cols_norm = np.clip(cols_proj / float(W_crop), 0.0, 1.0)

    # Face avant vs arrière : normale Z et position Z
    is_front = (norms[:, 2] >= -0.15) | (verts[:, 2] >= -10.0)

    uvs = np.zeros((len(verts), 2), dtype=np.float32)

    # Sommets Face Avant : U dans [0.005, 0.495]
    uvs[is_front, 0] = cols_norm[is_front] * 0.49
    uvs[is_front, 1] = v_coords[is_front]

    # Sommets Face Arrière : U dans [0.51, 0.99]
    # Inversion symétrique X pour la vue arrière
    uvs[~is_front, 0] = 0.51 + (1.0 - cols_norm[~is_front]) * 0.48
    uvs[~is_front, 1] = v_coords[~is_front]

    uvs = np.clip(uvs, 0.0, 1.0)

    # Matériau PBR haute fidélité
    pbr_material = trimesh.visual.material.PBRMaterial(
        name="WoutVanAert_Masterpiece_2K",
        baseColorTexture=atlas_img,
        roughnessFactor=0.38,
        metallicFactor=0.06
    )

    m.visual = trimesh.visual.TextureVisuals(
        uv=uvs,
        material=pbr_material
    )

    # 5. Export des Fichiers GLB & STL (Sans Socle & Avec Socle)
    print("[5/6] Export des modèles GLB texturés 2K...")
    out_dir_no_base = os.path.join(BASE_DIR, "outputs", "vanaert_2.7m_no_base")
    out_dir_with_base = os.path.join(BASE_DIR, "outputs", "vanaert_2.7m_with_base")
    demo_dir = os.path.join(BASE_DIR, "static", "demo")

    out_glb_nobase = os.path.join(out_dir_no_base, "model_bambu.glb")
    out_glb_demo = os.path.join(demo_dir, "Van_Aert_Figurine_A1_Mini.glb")

    m.export(out_glb_nobase)
    m.export(out_glb_demo)
    print(f"      Modèle Sans Socle exporté : {out_glb_nobase} ({os.path.getsize(out_glb_nobase)/(1024*1024):.1f} Mo)")
    print(f"      Modèle Démo Web Viewer exporté : {out_glb_demo} ({os.path.getsize(out_glb_demo)/(1024*1024):.1f} Mo)")

    # Modèle avec socle
    base_file = os.path.join(BASE_DIR, "outputs", "vanaert_896k_with_base", "model_bambu.glb")
    if os.path.exists(base_file):
        m_base_old = trimesh.load(base_file, force="mesh")
        comps = m_base_old.split(only_watertight=False)
        if len(comps) > 1:
            the_base = sorted(comps, key=lambda c: len(c.faces), reverse=True)[1]
            cyl_uv = np.full((len(the_base.vertices), 2), [0.85, 0.15], dtype=np.float32)
            the_base.visual = trimesh.visual.TextureVisuals(uv=cyl_uv, material=pbr_material)
            m_combined = trimesh.util.concatenate([m, the_base])
            out_glb_withbase = os.path.join(out_dir_with_base, "model_bambu.glb")
            m_combined.export(out_glb_withbase)
            print(f"      Modèle Avec Socle exporté : {out_glb_withbase} ({os.path.getsize(out_glb_withbase)/(1024*1024):.1f} Mo)")

    # 6. Rendu Studio Professionnel des Vues Clés (Haute Précision)
    print("[6/6] Génération des rendus Studio Haute Définition...")
    brain_dir = r"C:\Users\TrendingPC\.gemini\antigravity\brain\6f356426-1b48-4c89-8e8b-a62d57a95f48"

    # Vue Face
    preview_front = os.path.join(out_dir_no_base, "preview_front.png")
    render_mesh_preview(m, preview_front, img_size=1200, angle_deg=0.0)
    shutil.copy2(preview_front, os.path.join(brain_dir, "sota_2.7m_texture_front.png"))

    # Vue 3/4
    preview_34 = os.path.join(out_dir_no_base, "preview_34.png")
    render_mesh_preview(m, preview_34, img_size=1200, angle_deg=25.0)
    shutil.copy2(preview_34, os.path.join(brain_dir, "sota_2.7m_texture_34.png"))

    # Vue Côté
    preview_side = os.path.join(out_dir_no_base, "preview_side.png")
    render_mesh_preview(m, preview_side, img_size=1200, angle_deg=75.0)
    shutil.copy2(preview_side, os.path.join(brain_dir, "sota_2.7m_texture_side.png"))

    # Vue Macro Tête / Visage (zoom sur Y >= 80)
    m_head = m.copy()
    head_mask = m_head.vertices[:, 1] >= 78.0
    head_v_idx = np.where(head_mask)[0]
    face_mask = np.isin(m_head.faces, head_v_idx).all(axis=1)
    m_head.update_faces(face_mask)
    m_head.remove_unreferenced_vertices()

    preview_macro = os.path.join(out_dir_no_base, "preview_macro_head.png")
    render_mesh_preview(m_head, preview_macro, img_size=1200, angle_deg=0.0)
    shutil.copy2(preview_macro, os.path.join(brain_dir, "sota_2.7m_macro_head_texture.png"))

    print("=" * 75)
    print("TERMINE AVEC SUCCES : FIGURINE 3D WOUT VAN AERT 2K PBR TEXTUREE !")
    print("=" * 75)

if __name__ == "__main__":
    build_hd_textured_masterpiece()
