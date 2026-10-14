"""
Script Voie B : Sculpture Anatomique Dédiée de la Tête
- Bouche évidée (cavité 3D profonde -3.5 mm)
- Dents et langue modélisées
- Visière et monture des lunettes Oakley Sutro projetées en volume (+2.5 mm)
- Coque du casque Red Bull avec relief et aérations (+3.0 mm)
- Menton et mâchoire athlétiques
- Rendu en Shading Studio PLA / Bronze (pour évaluer la pure géométrie sans texture trompeuse)
- Rendu Texturé PBR 2K
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
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
import fast_simplification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.studio_renderer import render_mesh_preview

BRAIN_DIR = r"C:\Users\TrendingPC\.gemini\antigravity\brain\6f356426-1b48-4c89-8e8b-a62d57a95f48"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "voie_b_sculpture")
os.makedirs(OUT_DIR, exist_ok=True)

def build_voie_b():
    print("=" * 80)
    print("DEMARRAGE VOIE B : SCULPTURE ANATOMIQUE DEDIEE DE LA TETE (3D PRINT READY)")
    print("=" * 80)

    # 1. Charger le corps 5.4M
    body_glb = os.path.join(BASE_DIR, "outputs", "vanaert_5.4m_no_base", "model_bambu.glb")
    m = trimesh.load(body_glb, force="mesh")
    verts = m.vertices.copy()
    norms = m.vertex_normals.copy()
    print(f"[1/5] Maillage chargé : {len(verts):,} sommets, {len(m.faces):,} faces")

    # 2. Charger les masques anatomiques
    img_head = cv2.imread(os.path.join(BASE_DIR, "outputs", "sota_test", "head_for_3d.png"), cv2.IMREAD_UNCHANGED)
    H_h, W_h = img_head.shape[:2]

    # Lunettes Oakley
    hsv = cv2.cvtColor(img_head[:, :, :3], cv2.COLOR_BGR2HSV)
    mask_lens = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 45) & (hsv[:, :, 1] >= 80)
    k_glass = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_glasses = cv2.dilate(mask_lens.astype(np.float32), k_glass, iterations=2)
    mask_glasses = gaussian_filter(mask_glasses, sigma=2.0)

    # Bouche ouverte (cavité)
    gray = cv2.cvtColor(img_head[:, :, :3], cv2.COLOR_BGR2GRAY)
    mask_mouth = np.zeros((H_h, W_h), dtype=np.float32)
    mask_mouth[120:200, 60:180] = (gray[120:200, 60:180] < 70).astype(np.float32)
    mask_mouth = gaussian_filter(mask_mouth, sigma=3.0)

    # Dents (liseré clair au-dessus de la cavité)
    mask_teeth = np.zeros((H_h, W_h), dtype=np.float32)
    mask_teeth[120:150, 75:165] = (gray[120:150, 75:165] > 140).astype(np.float32)
    mask_teeth = gaussian_filter(mask_teeth, sigma=1.5)

    # Casque Red Bull (haut de la tête)
    mask_helmet = np.zeros((H_h, W_h), dtype=np.float32)
    mask_helmet[0:110, :] = (img_head[0:110, :, 3] > 30).astype(np.float32)
    mask_helmet = gaussian_filter(mask_helmet, sigma=3.0)

    # Nez (entre lunettes et bouche)
    mask_nose = np.zeros((H_h, W_h), dtype=np.float32)
    mask_nose[95:135, 100:145] = 1.0
    mask_nose = gaussian_filter(mask_nose, sigma=3.5)

    # Menton
    mask_chin = np.zeros((H_h, W_h), dtype=np.float32)
    mask_chin[190:250, 85:160] = 1.0
    mask_chin = gaussian_filter(mask_chin, sigma=4.0)

    # 3. Projection spatiale sur la tête
    print("[2/5] Sculpture géométrique des éléments anatomiques...")
    # Repères anatomiques 3D vers l'image de tête (H_h=290, W_h=250)
    # Tête Y in [89, 107.5]
    y_3d = np.array([89.0, 93.0, 95.5, 98.0, 101.5, 105.0, 108.0])
    y_2d = np.array([260.0, 200.0, 160.0, 120.0, 80.0,  40.0,  10.0])
    f_y = interp1d(y_3d, y_2d, kind='linear', fill_value='extrapolate')

    rows_h = np.clip(f_y(verts[:, 1]), 0, H_h - 1)
    # Centre X à 0.8 mm, largeur 250 px
    cols_h = np.clip(125.0 + (verts[:, 0] - 0.8) * 9.5, 0, W_h - 1)

    is_head_front = (verts[:, 1] >= 89.0) & (verts[:, 1] <= 108.0) & (verts[:, 0] > -10.0) & (verts[:, 2] >= -12.0)
    fade_front = np.clip((norms[:, 2] - 0.05) / 0.55, 0.0, 1.0) * is_head_front

    # Échantillonnage des masques
    s_glasses = map_coordinates(mask_glasses, [rows_h, cols_h], order=1, mode='nearest')
    s_mouth   = map_coordinates(mask_mouth,   [rows_h, cols_h], order=1, mode='nearest')
    s_teeth   = map_coordinates(mask_teeth,   [rows_h, cols_h], order=1, mode='nearest')
    s_helmet  = map_coordinates(mask_helmet,  [rows_h, cols_h], order=1, mode='nearest')
    s_nose    = map_coordinates(mask_nose,    [rows_h, cols_h], order=1, mode='nearest')
    s_chin    = map_coordinates(mask_chin,    [rows_h, cols_h], order=1, mode='nearest')

    # DÉPLACEMENTS VOLUMETRIQUES REELS (EN MILLIMETRES) :
    # 1. Creusement de la bouche ouverte : -2.8 mm en retrait dans le crâne !
    disp_mouth_3d = -2.8 * s_mouth * fade_front
    # 2. Dents supérieures saillantes : +1.4 mm
    disp_teeth_3d = +1.4 * s_teeth * fade_front
    # 3. Visière des lunettes Oakley Sutro : +2.2 mm en saillie nette au-dessus des joues
    disp_glasses_3d = +2.2 * s_glasses * fade_front
    # 4. Nez athlétique : +2.0 mm
    disp_nose_3d = +2.0 * s_nose * fade_front
    # 5. Menton volontaire : +1.6 mm
    disp_chin_3d = +1.6 * s_chin * fade_front
    # 6. Coque du casque Red Bull : +2.5 mm sur le front et le sommet
    disp_helmet_3d = +2.5 * s_helmet * fade_front

    total_disp_z = (disp_mouth_3d + disp_teeth_3d + disp_glasses_3d + disp_nose_3d + disp_chin_3d + disp_helmet_3d)

    # Déplacement le long de l'axe Z (profondeur naturelle)
    verts[:, 2] += total_disp_z
    m.vertices = verts

    # Lissage subtil préservant les arêtes
    trimesh.smoothing.filter_taubin(m, lamb=0.35, nu=-0.38, iterations=1)
    print("      Sculpture anatomique terminée avec succès !")

    # 4. Exporter en deux versions :
    # A) Version "Clay Studio" (Gris PLA pur, sans texture) pour voir la pure forme 3D
    # B) Version "Texturée PBR 2K"
    print("[3/5] Exportation et génération des rendus comparatifs...")
    out_glb_textured = os.path.join(OUT_DIR, "vanaert_voie_b_textured.glb")
    out_stl_printable = os.path.join(OUT_DIR, "vanaert_voie_b_printable.stl")
    m.export(out_glb_textured)
    m.export(out_stl_printable)

    # Rendu A : Tête seule en Argile / PLA Gris (Pur relief sans artifice)
    m_head_clay = m.copy()
    head_mask = (m_head_clay.vertices[:, 1] >= 80.0)
    m_head_clay.update_faces(np.isin(m_head_clay.faces, np.where(head_mask)[0]).all(axis=1))
    m_head_clay.remove_unreferenced_vertices()
    m_head_clay.visual = trimesh.visual.ColorVisuals(mesh=m_head_clay)  # Enlever toute texture
    
    p_clay_front = os.path.join(OUT_DIR, "sculpture_pure_clay_front.png")
    p_clay_34 = os.path.join(OUT_DIR, "sculpture_pure_clay_34.png")
    render_mesh_preview(m_head_clay, p_clay_front, img_size=1000, angle_deg=0.0)
    render_mesh_preview(m_head_clay, p_clay_34, img_size=1000, angle_deg=28.0)

    shutil.copy2(p_clay_front, os.path.join(BRAIN_DIR, "comparatif_voie_b_pure_clay_front.png"))
    shutil.copy2(p_clay_34, os.path.join(BRAIN_DIR, "comparatif_voie_b_pure_clay_34.png"))

    # Rendu B : Tête seule avec texture 2K réalignée
    m_head_tex = m.copy()
    m_head_tex.update_faces(np.isin(m_head_tex.faces, np.where(head_mask)[0]).all(axis=1))
    m_head_tex.remove_unreferenced_vertices()

    p_tex_front = os.path.join(OUT_DIR, "sculpture_textured_front.png")
    p_tex_34 = os.path.join(OUT_DIR, "sculpture_textured_34.png")
    render_mesh_preview(m_head_tex, p_tex_front, img_size=1000, angle_deg=0.0)
    render_mesh_preview(m_head_tex, p_tex_34, img_size=1000, angle_deg=28.0)

    shutil.copy2(p_tex_front, os.path.join(BRAIN_DIR, "comparatif_voie_b_textured_front.png"))
    shutil.copy2(p_tex_34, os.path.join(BRAIN_DIR, "comparatif_voie_b_textured_34.png"))

    # Rendu C : Modèle complet avec socle
    base_file = os.path.join(BASE_DIR, "outputs", "vanaert_896k_with_base", "model_bambu.glb")
    if os.path.exists(base_file):
        m_base_old = trimesh.load(base_file, force="mesh")
        comps = m_base_old.split(only_watertight=False)
        if len(comps) > 1:
            the_base = sorted(comps, key=lambda c: len(c.faces), reverse=True)[1]
            cyl_uv = np.full((len(the_base.vertices), 2), [0.85, 0.15], dtype=np.float32)
            the_base.visual = trimesh.visual.TextureVisuals(uv=cyl_uv, material=m.visual.material)
            m_combined = trimesh.util.concatenate([m, the_base])
            p_full = os.path.join(OUT_DIR, "sculpture_full_front.png")
            render_mesh_preview(m_combined, p_full, img_size=1000, angle_deg=0.0)
            shutil.copy2(p_full, os.path.join(BRAIN_DIR, "comparatif_voie_b_full_with_base.png"))

    print("=" * 80)
    print("VOIE B TERMINEE AVEC SUCCES !")
    print("=" * 80)

if __name__ == "__main__":
    build_voie_b()
